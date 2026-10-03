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

// ---- The PTA register block (c930_npu_csr.sv, offsets from the NPU's base) ----
//
// A c930 can be built with a photonic tensor tile in place of the systolic
// array. The tile is an ERROR MODEL: a GEMM that runs on it with any impairment
// enabled is not the GEMM the caller asked for, it is a noisy approximation of
// it, and AGENTS.md section 3 says that has to be reported. These are the
// registers the report is read from (docs/designs/pta_cpu_integration.md
// section 7.1, docs/designs/pta_chiplet_regmap.md section 2).
#define NPU_C930_PTA_ID       0x100u   // R:  [31:8] "PTA", [7:0] the map's version
#define NPU_C930_PTA_CAPS0    0x104u   // R:  [9:0] rows [19:10] cols [25:20] DIN_W [31:26] ACC_W
#define NPU_C930_PTA_CAPS1    0x108u   // R:  [6:0] impairments this build implements
#define NPU_C930_PTA_CAPS2    0x10Cu   // R:  [19:18] tile kind, [31] it is an emulation
#define NPU_C930_PTA_CTRL     0x140u   // RW: [0] EN -- the CALIBRATION ENGINE's, see below
#define NPU_C930_PTA_IMPAIR   0x148u   // RW: [6:0] one bit per impairment
#define NPU_C930_PTA_BITS     0x14Cu   // RW: [3:0] B_a [7:4] B_w [11:8] B_adc [17:12] S
#define NPU_C930_PTA_SEED     0x150u   // RW: every per-GEMM noise stream

#define NPU_C930_PTA_MAGIC    0x50544100u   // "PTA" in PTA_ID[31:8]

// PTA_IMPAIR, and PTA_CAPS1[6:0] in the same order.
#define NPU_C930_PTA_QUANT     0x01u
#define NPU_C930_PTA_THERMAL   0x02u
#define NPU_C930_PTA_SHOT      0x04u
#define NPU_C930_PTA_DRIFT     0x08u
#define NPU_C930_PTA_XTALK     0x10u
#define NPU_C930_PTA_MZM_NL    0x20u
#define NPU_C930_PTA_PROG_ERR  0x40u
#define NPU_C930_PTA_DEFINED   0x7Fu   // every bit the register defines

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

// ---- And the same for DDR, for the same reason ----
//
// The register hooks above cover the CONTROL path. They say nothing about the
// DATA path, and the data path has exactly the same problem: the NPU's DDR is
// a 64 KB window the engine's DMA reads and writes, and on a machine with no
// c930 there is no way to put anything in it.
//
// There is also no way on a machine WITH one, yet: the hardware path would be
// an mmap of the DDR aperture or a bounce through the AXI DMA, and neither is
// written. So these default to null and a copy through a device with no memory
// hooks is REFUSED rather than silently doing nothing -- an accepted memcpy
// that moves no bytes leaves the caller reading whatever was in the buffer,
// which is the failure class this project bans.
//
// Return 0 on success, non-zero on failure (an address outside the window).
typedef int (*npu_c930_mem_read_fn)(void* ctx, uint32_t addr, void* dst,
                                    uint32_t bytes);
typedef int (*npu_c930_mem_write_fn)(void* ctx, uint32_t addr, const void* src,
                                     uint32_t bytes);

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

    // Injected DDR model. Null everywhere today -- see the note above.
    npu_c930_mem_read_fn  mem_read;
    npu_c930_mem_write_fn mem_write;
    void*                 mem_ctx;
} npu_c930_device_t;

// Point a device at a register model instead of at MMIO. Call BEFORE
// npu_c930_detect; detect then skips the mmap and probes the model.
void npu_c930_attach_model(npu_c930_device_t* dev,
                           npu_c930_read_fn read32,
                           npu_c930_write_fn write32,
                           void* ctx);

// Point a device's DDR at a model. Call AFTER npu_c930_attach_model, which
// zeroes the whole struct -- doing it the other way round silently discards
// these three fields, which is the kind of ordering hazard worth stating in
// the header rather than discovering in a debugger.
void npu_c930_attach_memory(npu_c930_device_t* dev,
                            npu_c930_mem_read_fn mem_read,
                            npu_c930_mem_write_fn mem_write,
                            void* ctx);

// Copy into / out of the device's DDR window. Returns 0 on success, -1 when
// the device has no memory path at all -- which is every device today except
// one with a model attached.
int npu_c930_mem_write(npu_c930_device_t* dev, uint32_t addr, const void* src,
                       uint32_t bytes);
int npu_c930_mem_read(npu_c930_device_t* dev, uint32_t addr, void* dst,
                      uint32_t bytes);

// True when this device can move bytes into and out of DDR.
int npu_c930_mem_ready(const npu_c930_device_t* dev);

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

// ---- What a GEMM on this device IS ----
//
// Three things about the hardware decide this, and each was found by reading
// the RTL rather than the specification, which had two of them wrong.
//
// 1. PTA_CTRL.EN IS NOT WHAT MAKES A GEMM ANALOG. It enables the calibration
//    engine and nothing else. A build with a tile runs every GEMM on the tile,
//    and the result is inexact exactly when PTA_IMPAIR is non-zero -- with EN
//    set or clear. A report keyed on EN would call a noisy GEMM native, which
//    is the one direction this report must never be wrong in.
//
// 2. THE BLOCK IS IN EVERY BUILD. PTA_IMPAIR and PTA_BITS write and read back
//    on a c930 with no tile exactly as on one with a tile, so reading them
//    says nothing about which you have. Only PTA_CAPS1 does: the mask of
//    impairments the build implements, zero on the digital array.
//
// 3. AN OLDER REGISTER FILE HAS NO PTA_ID, and one older still has no block at
//    all: it decodes sixteen words and aliases everything above them, so
//    PTA_IMPAIR's address reads back DIM_M. A driver that trusted that would
//    report an analog GEMM with "impairments" equal to whatever M was last
//    programmed. So the magic is checked first, and without it every field is
//    UNKNOWN (-1) -- not "no tile", which this driver has no way to know.
//
// Every field is -1 when it does not apply or cannot be sourced, never zero:
// zero activation bits is a legitimate value (unquantised) and a plausible
// wrong one.
typedef struct {
    int     identified;              // 1: PTA_ID carried the magic. 0: nothing below is known
    int     map_version;             // PTA_ID[7:0], or -1
    int     analog;                  // 1: GEMMs here are impaired. 0: exact. -1: unknown
    int     tile_present;            // 1 / 0 / -1 unknown
    int     activation_bits;         // PTA_BITS[3:0], 0 = unquantised; -1 unless analog
    int     weight_bits;             // PTA_BITS[7:4]
    int     adc_bits;                // PTA_BITS[11:8], 0 = no ADC quantisation
    int     adc_shift;               // PTA_BITS[17:12]; LSB_adc = 2^adc_shift
    int64_t seed;                    // PTA_SEED; -1 unless analog
    int64_t impairments;             // PTA_IMPAIR[6:0], the enables in force; -1 unless analog
    int64_t impairments_implemented; // PTA_CAPS1[6:0]; -1 only when unidentified
    // What PTA_IMPAIR holds whatever the build is. Not part of the report --
    // on a build with no tile it describes nothing that runs -- but it is what
    // explains a refused START, and that is the only thing it is used for.
    int64_t impairments_requested;   // -1 when unidentified
} npu_c930_analog_t;

// Read the block and decide. Returns 0 when a determination was made -- and
// "unidentified, everything unknown" IS a determination -- or -1 when the
// device has no register path at all, in which case *out is all unknown.
//
// Not atomic: each field is its own register read, so a writer racing this
// can produce a mixture. That is the registers' nature, not a bug to paper
// over; a caller that needs a consistent view quiesces the device first.
int npu_c930_read_analog(npu_c930_device_t* dev, npu_c930_analog_t* out);

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
