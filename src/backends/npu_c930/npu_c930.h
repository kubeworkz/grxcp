// npu_c930.h — GRX930 NPU backend for grxcp.
//
// This module drives the C930's INT8 systolic-array GEMM accelerator through
// its AXI4-Lite MMIO register interface.  It is a library-level backend: it
// does NOT use vortex2.h and has no dependency on the Vortex driver.
//
// The NPU is a memory-mapped accelerator on the GRX930 SoC's AXI fabric.
// The CPU programs it over MMIO; the NPU autonomously fetches A/B from DDR,
// runs the GEMM, and writes C back — no CPU data staging required.
//
// Register map (from c930/doc/c930_architecture.md §5):
//   0x00 CTRL    (W)  bit0 = START
//   0x04 STATUS  (R)  bit0 = BUSY, bit1 = DONE (latched), bit2 = ERROR
//   0x08 DIM_M   (R/W) output rows
//   0x0C DIM_N   (R/W) output cols
//   0x10 DIM_K   (R/W) reduction length
//   0x14 A_BASE  (R/W) A matrix base address (DMA read)
//   0x18 B_BASE  (R/W) B matrix base address (DMA read)
//   0x1C C_BASE  (R/W) C result base address (DMA write)
//
// Integration with grxcp:
//   1. npu_c930_detect() probes the NPU at runtime.
//   2. npu_c930_gemm() dispatches an INT8 GEMM through the NPU.
//   3. grxblasGemmEx routes to the NPU path when the current device is an NPU.

#ifndef NPU_C930_H
#define NPU_C930_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// ---- MMIO base address (from c930_soc_top.sv) ----
#define NPU_C930_MMIO_BASE   0x40000000u

// ---- Register offsets (word-aligned, from c930_npu_csr.sv) ----
#define NPU_C930_REG_CTRL    (NPU_C930_MMIO_BASE + 0x00)
#define NPU_C930_REG_STATUS  (NPU_C930_MMIO_BASE + 0x04)
#define NPU_C930_REG_DIM_M   (NPU_C930_MMIO_BASE + 0x08)
#define NPU_C930_REG_DIM_N   (NPU_C930_MMIO_BASE + 0x0C)
#define NPU_C930_REG_DIM_K   (NPU_C930_MMIO_BASE + 0x10)
#define NPU_C930_REG_A_BASE  (NPU_C930_MMIO_BASE + 0x14)
#define NPU_C930_REG_B_BASE  (NPU_C930_MMIO_BASE + 0x18)
#define NPU_C930_REG_C_BASE  (NPU_C930_MMIO_BASE + 0x1C)
#define NPU_C930_REG_PREC    (NPU_C930_MMIO_BASE + 0x20)

// ---- CTRL bits ----
#define NPU_C930_CTRL_START  0x1u

// ---- STATUS bits ----
#define NPU_C930_STATUS_BUSY  0x1u
#define NPU_C930_STATUS_DONE  0x2u
#define NPU_C930_STATUS_ERROR 0x4u

// ---- Precision modes ----
#define NPU_C930_PREC_INT8   0u
#define NPU_C930_PREC_INT16  1u
#define NPU_C930_PREC_FP16   2u
#define NPU_C930_PREC_BF16   3u

// ---- NPU hardware limits (from c930_soc_top.sv defaults) ----
#define NPU_C930_MAX_M  8
#define NPU_C930_MAX_K  16
#define NPU_C930_MAX_N  12
#define NPU_C930_NUM_ROWS 4   // systolic rows (reduction per pass)
#define NPU_C930_NUM_COLS 4   // systolic cols (output width)

// ---- NPU capability flags (for grxDeviceProp_t.capabilities) ----
#define NPU_C930_CAP_STREAMS          0x02u  // GRX_CAP_STREAMS
#define NPU_C930_CAP_EVENTS           0x04u  // GRX_CAP_EVENTS
#define NPU_C930_CAP_MEMCPY           0x08u  // GRX_CAP_MEMCPY
#define NPU_C930_CAP_GEMM             0x20u  // GRX_CAP_GEMM

// ---- Data types ----
typedef enum {
    NPU_C930_INT8  = 0,
    NPU_C930_INT16 = 1,
    NPU_C930_FP16  = 2,
    NPU_C930_FP32  = 3
} npu_c930_dtype_t;

// ---- Register access, and why it is indirect ----
//
// The register block is normally an mmap of physical memory. It does not have
// to be: this header already contemplates simulation ("DPI or backdoor
// access"), and there is a more immediate reason. Every interesting behaviour
// of this backend is a response to what the registers do -- a device that is
// absent, a device that is present, a device that accepts a START and never
// finishes -- and none of those can be produced on a machine with no c930.
//
// So reads and writes go through function pointers. Leave them null and the
// mmap'd base is used, which is the hardware path and the default. Point them
// at a register model and the detection, launch and completion logic can be
// driven through states real hardware would take days to reproduce on purpose.
//
// A model is NOT hardware and no result obtained through one may be reported as
// the NPU working -- the same rule tests/mock lives under. What it checks is
// this file's logic, which is the half that was wrong.
typedef uint32_t (*npu_c930_read_fn)(void* ctx, uint32_t offset);
typedef void     (*npu_c930_write_fn)(void* ctx, uint32_t offset, uint32_t v);

// ---- NPU device handle ----
typedef struct npu_c930_device {
    int      fd;            // /dev/mem file descriptor (Linux) or 0 (bare-metal)
    uint8_t* mmio_base;    // mmap'd MMIO base
    int      present;      // 1 if NPU detected
    int      busy;         // last known BUSY state
    int      error;        // last known ERROR state

    // Injected register model. Null on the hardware path.
    npu_c930_read_fn  read32;
    npu_c930_write_fn write32;
    void*             io_ctx;
} npu_c930_device_t;

// Point a device at a register model instead of at MMIO. Call BEFORE
// npu_c930_detect; detect then skips the mmap and probes the model.
void npu_c930_attach_model(npu_c930_device_t* dev,
                           npu_c930_read_fn read32,
                           npu_c930_write_fn write32,
                           void* ctx);

// ---- Detection ----

// Probe for the NPU at the given MMIO base address.
// Returns 1 if the NPU is present (STATUS register readable), 0 otherwise.
// On bare-metal, pass fd=0 and mmio_base=NULL to use fixed addresses.
int npu_c930_detect(npu_c930_device_t* dev);

// ---- GEMM dispatch ----

// INT8 GEMM: C[M x N] = A[M x K] * B[K x N], INT8 in, INT32 out.
//
// A and B must be in DDR memory reachable by the NPU's AXI4 DMA.
// C is written back to DDR by the NPU.
//
// All pointers are physical DDR addresses (byte-aligned).
// The function blocks until the NPU completes (polls STATUS.DONE).
//
// Returns 0 on success, -1 on error (check dev->error).
int npu_c930_gemm(npu_c930_device_t* dev,
                   int m, int n, int k,
                   uint32_t a_addr,   // byte address of A in DDR
                   uint32_t b_addr,   // byte address of B in DDR
                   uint32_t c_addr);  // byte address of C in DDR

// ---- Status ----

// Read the STATUS register.  Updates dev->busy and dev->error.
uint32_t npu_c930_read_status(npu_c930_device_t* dev);

// Wait for the NPU to become idle (polls STATUS.BUSY).
// Returns 0 on success, -1 on timeout.
int npu_c930_wait_idle(npu_c930_device_t* dev, int timeout_us);

// Set the precision mode (must be called before npu_c930_gemm).
// mode: NPU_C930_PREC_INT8, NPU_C930_PREC_INT16, etc.
// Returns 0 on success, -1 on error.
int npu_c930_set_precision(npu_c930_device_t* dev, uint32_t mode);

// Read the current precision mode.
// Returns the mode value (0-3), or -1 on error.
int npu_c930_get_precision(npu_c930_device_t* dev);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NPU_C930_H
