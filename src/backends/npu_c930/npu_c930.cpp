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

// Every register access in this file goes through these two, so the hardware
// path and an injected model differ in exactly one place.
static inline uint32_t reg_read(const npu_c930_device_t* dev, uint32_t offset) {
    if (dev->read32) return dev->read32(dev->io_ctx, offset);
    return mmio_read32(dev->mmio_base, offset);
}

static inline void reg_write(npu_c930_device_t* dev, uint32_t offset,
                             uint32_t value) {
    if (dev->write32) { dev->write32(dev->io_ctx, offset, value); return; }
    mmio_write32(dev->mmio_base, offset, value);
}

// True when this device has somewhere to send a register access at all.
static inline int reg_ready(const npu_c930_device_t* dev) {
    return dev && (dev->read32 != nullptr || dev->mmio_base != nullptr);
}

// ---- Public API ----

extern "C" {

void npu_c930_attach_model(npu_c930_device_t* dev,
                           npu_c930_read_fn read32,
                           npu_c930_write_fn write32,
                           void* ctx) {
    if (!dev) return;
    memset(dev, 0, sizeof(*dev));
    dev->fd      = -1;
    dev->read32  = read32;
    dev->write32 = write32;
    dev->io_ctx  = ctx;
}

int npu_c930_detect(npu_c930_device_t* dev) {
    if (!dev) return 0;

    // Clear the RESULT fields only. The whole-struct memset that used to be
    // here also wiped an injected register model -- and wiped an mmio_base the
    // caller had set, which the bare-metal path's own comment tells callers to
    // do ("the caller must set dev->mmio_base before calling detect"). Both
    // documented ways of pointing this at something were erased on entry.
    dev->present = 0;
    dev->busy    = 0;
    dev->error   = 0;

    if (!dev->read32 && !dev->write32) {
        dev->fd = -1;
        if (map_mmio(dev, NPU_C930_MMIO_BASE) != 0) {
            dev->present = 0;
            return 0;
        }
    }

    // DETECTION NEEDS POSITIVE EVIDENCE, NOT THE ABSENCE OF A BUS ERROR.
    //
    // The first version of this read STATUS and accepted anything that was not
    // 0xFFFFFFFF and not greater than 0x7. Its own comment said an absent NPU
    // reads 0x0 -- and 0x0 passes that test, so it reported PRESENT. That is not
    // a corner case: mmap over unbacked physical address space reads as zeros on
    // an ordinary machine, so any host where /dev/mem opens would have grown a
    // GRX930 NPU it does not have. Measured, with open()/mmap() shimmed to
    // return a zeroed page:
    //
    //   npu_c930_detect() -> 1   dev.present=1
    //   npu_c930_gemm()   -> 0   (SUCCESS)  STATUS 0x00000000, DONE never set
    //
    // A phantom device that reports successful GEMMs and never writes C is the
    // exact failure AGENTS.md section 1 exists to prevent, and it is worse than
    // a crash because the caller gets whatever was already in the output buffer.
    //
    // So the probe is a WRITE-READBACK on a register the map documents as R/W.
    // A real register block returns what was stored; unbacked memory returns
    // zeros, and a dead bus returns all ones. Neither can fake a readback of a
    // value it was never given. DIM_M is safe to use for this: the device is
    // idle at probe time and every dimension is reprogrammed before each launch
    // anyway. The original value is put back regardless.
    const uint32_t status = reg_read(dev, 0x04);
    if (status == 0xFFFFFFFF || status > 0x7) {
        unmap_mmio(dev);
        dev->present = 0;
        return 0;
    }

    // Two patterns, not one: a single pattern that happens to equal what the
    // location already holds proves nothing, and 0x00000000/0xFFFFFFFF are
    // exactly the two values a missing device produces.
    const uint32_t saved = reg_read(dev, 0x08);
    static const uint32_t kProbes[2] = {0x00000005u, 0x00000002u};
    int alive = 1;
    for (int i = 0; i < 2 && alive; ++i) {
        reg_write(dev, 0x08, kProbes[i]);
        if (reg_read(dev, 0x08) != kProbes[i]) alive = 0;
    }
    reg_write(dev, 0x08, saved);

    if (!alive) {
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
    if (!reg_ready(dev)) return 0;
    uint32_t status = reg_read(dev, 0x04);
    dev->busy = (status & NPU_C930_STATUS_BUSY) != 0;
    dev->error = (status & NPU_C930_STATUS_ERROR) != 0;
    return status;
}

int npu_c930_wait_idle(npu_c930_device_t* dev, int timeout_us) {
    if (!reg_ready(dev)) return -1;

    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);

    while (1) {
        uint32_t status = reg_read(dev, 0x04);
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
    if (!reg_ready(dev)) return -1;

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
    reg_write(dev, 0x08, (uint32_t)m);     // DIM_M
    reg_write(dev, 0x0C, (uint32_t)n);     // DIM_N
    reg_write(dev, 0x10, (uint32_t)k);     // DIM_K
    reg_write(dev, 0x14, a_addr);           // A_BASE
    reg_write(dev, 0x18, b_addr);           // B_BASE
    reg_write(dev, 0x1C, c_addr);           // C_BASE

    // Launch (write CTRL.START — also clears DONE and ERROR)
    reg_write(dev, 0x00, NPU_C930_CTRL_START);

    // Wait for completion
    if (npu_c930_wait_idle(dev, 10000000) != 0) {
        fprintf(stderr, "npu_c930: GEMM execution timed out\n");
        return -1;
    }

    // Check for errors
    uint32_t status = reg_read(dev, 0x04);
    if (status & NPU_C930_STATUS_ERROR) {
        fprintf(stderr, "npu_c930: NPU reported error (status=0x%08x)\n", status);
        return -1;
    }

    // AND CHECK THAT IT ACTUALLY RAN.
    //
    // Waiting for !BUSY and no ERROR is not the same as completion. A device
    // that ignored every write is never busy and never errors, so those two
    // checks pass instantly and this function used to return success over a
    // GEMM that never happened -- leaving C holding whatever it held before.
    // DONE is latched by the hardware precisely so the host can tell "finished"
    // from "never started"; the register map documents it and nothing was
    // reading it.
    if (!(status & NPU_C930_STATUS_DONE)) {
        fprintf(stderr, "npu_c930: no DONE after CTRL.START (status=0x%08x). "
                "The NPU did not run this GEMM; C is unchanged.\n", status);
        return -1;
    }

    return 0;
}

int npu_c930_set_precision(npu_c930_device_t* dev, uint32_t mode) {
    if (!reg_ready(dev)) return -1;
    if (mode > 3) {
        fprintf(stderr, "npu_c930: invalid precision mode %u\n", mode);
        return -1;
    }
    reg_write(dev, 0x20, mode);
    return 0;
}

int npu_c930_get_precision(npu_c930_device_t* dev) {
    if (!reg_ready(dev)) return -1;
    return (int)(reg_read(dev, 0x20) & 0x3);
}

}  // extern "C"
