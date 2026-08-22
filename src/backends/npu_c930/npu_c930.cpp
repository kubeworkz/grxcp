// npu_c930.cpp — GRX930 NPU backend implementation.
//
// Drives the C930's INT8 systolic-array GEMM accelerator through its
// AXI4-Lite MMIO register interface.
//
// This module has NO dependency on vortex2.h and works on:
//   - Bare-metal (direct register access via fixed addresses)
//   - Linux user-space (mmap /dev/mem or UIO device)
//   - Simulation (DPI or backdoor access)

#include "npu_c930.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

// ---- Platform-specific MMIO access ----

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

static int map_mmio(npu_c930_device_t* dev, uint32_t phys_addr) {
    dev->fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (dev->fd < 0) {
        // Try UIO device
        dev->fd = open("/dev/uio0", O_RDWR | O_SYNC);
        if (dev->fd < 0) return -1;
    }
    dev->mmio_base = (uint8_t*)mmap(nullptr, 0x1000,
                                     PROT_READ | PROT_WRITE,
                                     MAP_SHARED, dev->fd, phys_addr);
    if (dev->mmio_base == MAP_FAILED) {
        close(dev->fd);
        dev->fd = -1;
        dev->mmio_base = nullptr;
        return -1;
    }
    return 0;
}

static void unmap_mmio(npu_c930_device_t* dev) {
    if (dev->mmio_base) {
        munmap(dev->mmio_base, 0x1000);
        dev->mmio_base = nullptr;
    }
    if (dev->fd >= 0) {
        close(dev->fd);
        dev->fd = -1;
    }
}

#else
// Bare-metal / simulation: fixed MMIO address, no mmap needed.
static int map_mmio(npu_c930_device_t* dev, uint32_t phys_addr) {
    (void)phys_addr;
    // On bare-metal, the MMIO address is directly accessible.
    // The caller must set dev->mmio_base before calling detect.
    // For simulation, this can be overridden with a DPI backdoor.
    if (!dev->mmio_base) {
        // Default: cast the physical address to a pointer (bare-metal only)
        dev->mmio_base = reinterpret_cast<uint8_t*>(
            static_cast<uintptr_t>(NPU_C930_MMIO_BASE));
    }
    dev->fd = 0;
    return 0;
}

static void unmap_mmio(npu_c930_device_t* dev) {
    (void)dev;
    // Nothing to unmap on bare-metal.
}
#endif

// ---- MMIO read/write helpers ----

static inline uint32_t mmio_read32(const volatile uint8_t* base, uint32_t offset) {
    const volatile uint32_t* addr =
        reinterpret_cast<const volatile uint32_t*>(base + offset);
    return *addr;
}

static inline void mmio_write32(volatile uint8_t* base, uint32_t offset,
                                uint32_t value) {
    volatile uint32_t* addr =
        reinterpret_cast<volatile uint32_t*>(base + offset);
    *addr = value;
}

// ---- Public API ----

extern "C" {

int npu_c930_detect(npu_c930_device_t* dev) {
    if (!dev) return 0;
    memset(dev, 0, sizeof(*dev));
    dev->fd = -1;

    if (map_mmio(dev, NPU_C930_MMIO_BASE) != 0) {
        dev->present = 0;
        return 0;
    }

    // Read STATUS register.  A present NPU returns a valid value
    // (bits 0-2 may be set, bits 3-31 are 0).
    // An absent NPU returns 0x0 (all bits clear) or 0xFFFFFFFF (bus error).
    uint32_t status = mmio_read32(dev->mmio_base, 0x04);
    if (status == 0xFFFFFFFF || status > 0x7) {
        // Bus error or unexpected value — NPU not present
        unmap_mmio(dev);
        dev->present = 0;
        return 0;
    }

    dev->present = 1;
    dev->busy = (status & NPU_C930_STATUS_BUSY) != 0;
    dev->error = (status & NPU_C930_STATUS_ERROR) != 0;
    return 1;
}

uint32_t npu_c930_read_status(npu_c930_device_t* dev) {
    if (!dev || !dev->mmio_base) return 0;
    uint32_t status = mmio_read32(dev->mmio_base, 0x04);
    dev->busy = (status & NPU_C930_STATUS_BUSY) != 0;
    dev->error = (status & NPU_C930_STATUS_ERROR) != 0;
    return status;
}

int npu_c930_wait_idle(npu_c930_device_t* dev, int timeout_us) {
    if (!dev || !dev->mmio_base) return -1;

    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);

    while (1) {
        uint32_t status = mmio_read32(dev->mmio_base, 0x04);
        if (!(status & NPU_C930_STATUS_BUSY)) {
            dev->busy = 0;
            dev->error = (status & NPU_C930_STATUS_ERROR) != 0;
            return 0;
        }

        clock_gettime(CLOCK_MONOTONIC, &now);
        int elapsed_us = (int)((now.tv_sec - start.tv_sec) * 1000000 +
                               (now.tv_nsec - start.tv_nsec) / 1000);
        if (elapsed_us >= timeout_us) {
            fprintf(stderr, "npu_c930: timeout waiting for idle "
                    "(status=0x%08x, elapsed=%d us)\n", status, elapsed_us);
            return -1;
        }

        // Busy-wait with a small yield to avoid hammering the bus
        struct timespec ts = {0, 1000};  // 1 us
        nanosleep(&ts, nullptr);
    }
}

int npu_c930_gemm(npu_c930_device_t* dev,
                   int m, int n, int k,
                   uint32_t a_addr, uint32_t b_addr, uint32_t c_addr) {
    if (!dev || !dev->mmio_base) return -1;

    // Validate dimensions
    if (m <= 0 || n <= 0 || k <= 0) {
        fprintf(stderr, "npu_c930: invalid dimensions M=%d N=%d K=%d\n", m, n, k);
        return -1;
    }
    if (m > NPU_C930_MAX_M || n > NPU_C930_MAX_N || k > NPU_C930_MAX_K) {
        fprintf(stderr, "npu_c930: dimensions exceed hardware limits "
                "(M=%d>%d, N=%d>%d, K=%d>%d)\n",
                m, NPU_C930_MAX_M, n, NPU_C930_MAX_N, k, NPU_C930_MAX_K);
        return -1;
    }

    // Check alignment (4-byte aligned for AXI)
    if ((a_addr & 0x3) || (b_addr & 0x3) || (c_addr & 0x3)) {
        fprintf(stderr, "npu_c930: base addresses must be 4-byte aligned "
                "(A=0x%08x B=0x%08x C=0x%08x)\n", a_addr, b_addr, c_addr);
        return -1;
    }

    // Wait for NPU to be idle
    if (npu_c930_wait_idle(dev, 1000000) != 0) {
        fprintf(stderr, "npu_c930: NPU not idle before launch\n");
        return -1;
    }

    // Program dimensions and base addresses
    mmio_write32(dev->mmio_base, 0x08, (uint32_t)m);     // DIM_M
    mmio_write32(dev->mmio_base, 0x0C, (uint32_t)n);     // DIM_N
    mmio_write32(dev->mmio_base, 0x10, (uint32_t)k);     // DIM_K
    mmio_write32(dev->mmio_base, 0x14, a_addr);           // A_BASE
    mmio_write32(dev->mmio_base, 0x18, b_addr);           // B_BASE
    mmio_write32(dev->mmio_base, 0x1C, c_addr);           // C_BASE

    // Launch (write CTRL.START — also clears DONE and ERROR)
    mmio_write32(dev->mmio_base, 0x00, NPU_C930_CTRL_START);

    // Wait for completion
    if (npu_c930_wait_idle(dev, 10000000) != 0) {
        fprintf(stderr, "npu_c930: GEMM execution timed out\n");
        return -1;
    }

    // Check for errors
    uint32_t status = mmio_read32(dev->mmio_base, 0x04);
    if (status & NPU_C930_STATUS_ERROR) {
        fprintf(stderr, "npu_c930: NPU reported error (status=0x%08x)\n", status);
        return -1;
    }

    return 0;
}

}  // extern "C"
