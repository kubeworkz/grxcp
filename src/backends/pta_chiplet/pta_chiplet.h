// pta_chiplet.h -- the host's driver for the PTA chiplet.
//
// The chiplet is a photonic tensor tile and its interface chip, in the GPU's
// package. Its register block is one page of the GPU's BAR, reached over
// CXL.io, and docs/designs/pta_chiplet_regmap.md is the map. Its work arrives
// over a die-to-die link from the GPU's copy engine.
//
// THERE IS NO CHIPLET, AND NO WAY TO REACH ONE FROM HERE. Two things a host
// needs do not exist yet, and this driver says so rather than standing
// something in for them:
//
//   - the window. The GPU's driver owns the BAR, and it has no call that reads
//     or writes the PTA's page. Until it does, the only way in is the pair of
//     hooks below, and the only thing behind them is a model.
//   - the link. Nothing specifies what a command on it looks like (the
//     interface document's section 7, item 7). The submit hook is a function
//     call standing in for that protocol, and only a model implements it.
//
// So on hardware this driver detects nothing and runs nothing, and that is the
// honest result. With the digital twin attached (pta_chiplet_twin.h) it drives
// the map exactly as it would a chiplet's, which is what the twin is for.
//
// It shares its block's layout with the c930's tile -- the same registers, 0x100
// lower -- and tests hold this reader and the c930 backend's to one answer on
// the same device.

#ifndef PTA_CHIPLET_H
#define PTA_CHIPLET_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ---- The map, offsets from the window's base ----
#define PTA_CHIPLET_ID           0x000u   // R:    [31:8] "PTA", [7:0] the map's version
#define PTA_CHIPLET_CAPS0        0x004u   // R:    [9:0] rows [19:10] cols [25:20] DIN_W [31:26] ACC_W
#define PTA_CHIPLET_CAPS1        0x008u   // R:    [6:0] impairments built
#define PTA_CHIPLET_CAPS2        0x00Cu   // R:    [16] cal engine [17] act stage [19:18] kind [31] a model
#define PTA_CHIPLET_IRQ_STATUS   0x010u   // RW1C: [0] CAL_DONE [1] ERR [2] DRIFT_ALARM [3] SAT_THRESHOLD
#define PTA_CHIPLET_GEMM_CT      0x018u   // R:    GEMMs started since the last PTA_SEED write
#define PTA_CHIPLET_CTRL         0x040u
#define PTA_CHIPLET_STATUS       0x044u   // R:    [0] CAL_BUSY [1] CAL_VALID [4] BUSY
#define PTA_CHIPLET_IMPAIR       0x048u
#define PTA_CHIPLET_BITS         0x04Cu
#define PTA_CHIPLET_SEED         0x050u
#define PTA_CHIPLET_SIGMA_TH     0x054u
#define PTA_CHIPLET_SIGMA_SH     0x058u
#define PTA_CHIPLET_SIGMA_PR     0x05Cu
#define PTA_CHIPLET_DRIFT        0x060u
#define PTA_CHIPLET_XTALK        0x064u
#define PTA_CHIPLET_DRIFT_MAX    0x0D0u

#define PTA_CHIPLET_MAGIC        0x50544100u   // "PTA" in PTA_ID[31:8]
#define PTA_CHIPLET_MAP_VERSION  1u            // the version of the map this driver was written to
#define PTA_CHIPLET_DEFINED      0x7Fu         // every impairment bit the register defines
#define PTA_CHIPLET_STATUS_CAL_BUSY  0x01u
#define PTA_CHIPLET_STATUS_CAL_VALID 0x02u
#define PTA_CHIPLET_STATUS_BUSY      0x10u
#define PTA_CHIPLET_CAPS2_MODEL      0x80000000u

// ---- The window ----
typedef uint32_t (*pta_chiplet_read_fn)(void* ctx, uint32_t offset);
typedef void     (*pta_chiplet_write_fn)(void* ctx, uint32_t offset, uint32_t value);

// ---- The link, as a function call ----
//
// One command: C = A . B on weight bank `bank`, A being M x K and B K x N, row
// major, in the tile's own integers. Returns 0 if the chiplet took the command
// and non-zero if it did not. *status is PENDING from acceptance until the
// command ends, and then says how.
//
// This signature is the twin's, and it is not a proposal for the link. When the
// link has a format this hook is what gets replaced.
#define PTA_CHIPLET_CMD_PENDING  0
#define PTA_CHIPLET_CMD_DONE     1   // C holds the result
#define PTA_CHIPLET_CMD_REFUSED  2   // the tile cannot do this; C is untouched
#define PTA_CHIPLET_CMD_LOST     3   // a reset discarded it; C is untouched
typedef int (*pta_chiplet_submit_fn)(void* ctx, int bank, int M, int N, int K,
                                     const int32_t* A, const int32_t* B,
                                     int64_t* C, int* status);

// ---- Why a call failed ----
#define PTA_CHIPLET_OK                0
#define PTA_CHIPLET_ERR_NO_WINDOW     1   // no path to the register block
#define PTA_CHIPLET_ERR_NO_LINK       2   // no path to issue work
#define PTA_CHIPLET_ERR_NOT_PRESENT   3   // the block did not identify itself as a tile this driver knows
#define PTA_CHIPLET_ERR_NOT_ACCEPTED  4   // the chiplet's queue did not take the command
#define PTA_CHIPLET_ERR_REFUSED       5   // the tile refused it: see refused_impairments
#define PTA_CHIPLET_ERR_WEDGED        6   // it never stopped being busy
#define PTA_CHIPLET_ERR_LOST          7   // it ended with no result

typedef struct pta_chiplet_device {
    int      present;               // 1 once pta_chiplet_detect has identified a tile
    int      error;                 // PTA_CHIPLET_ERR_* of the last call that failed
    // After PTA_CHIPLET_ERR_REFUSED: the enabled impairments this tile does not
    // build, PTA_IMPAIR & ~PTA_CAPS1. Zero when the refusal was for something
    // else -- an ADC shift past 40, or a bank the tile does not have.
    uint32_t refused_impairments;

    pta_chiplet_read_fn   read32;
    pta_chiplet_write_fn  write32;
    void*                 io_ctx;
    pta_chiplet_submit_fn submit;
    void*                 link_ctx;
} pta_chiplet_device_t;

// Point a device at a window. Zeroes the whole struct first, so call it BEFORE
// pta_chiplet_attach_link -- the other order silently discards the link.
void pta_chiplet_attach_window(pta_chiplet_device_t* dev,
                               pta_chiplet_read_fn read32,
                               pta_chiplet_write_fn write32, void* ctx);
void pta_chiplet_attach_link(pta_chiplet_device_t* dev,
                             pta_chiplet_submit_fn submit, void* ctx);

// Is there a tile behind the window that this driver knows how to drive?
// Returns 1 and sets dev->present if PTA_ID carries the magic and map version
// 1 and PTA_CAPS1 says a tile is built. Returns 0 otherwise -- including for a
// device with no window, which is every device today that is not a model.
int pta_chiplet_detect(pta_chiplet_device_t* dev);

// What a GEMM on this device is. The c930 backend's npu_c930_analog_t, field
// for field, and what the chiplet adds to it. Every field is -1 when it does
// not apply or cannot be sourced, never zero.
typedef struct {
    int     identified;              // 1: PTA_ID carried the magic
    int     map_version;
    int     analog;                  // 1: GEMMs here are impaired. 0: exact. -1: unknown
    int     tile_present;
    int     activation_bits, weight_bits, adc_bits, adc_shift;
    int64_t seed;                    // PTA_SEED: the RUN's seed, not a GEMM's
    int64_t impairments;
    int64_t impairments_implemented;
    int64_t impairments_requested;
    int     sigma_thermal_q8, shot_k_q8, sigma_prog_q8;
    int     drift_sigma_q8, drift_log2_shots, drift_clamp_q8, crosstalk_q8;
    int     loop_modes, calibration_valid;
    int     tile_rows, tile_cols, operand_bits, accumulator_bits;

    // THE CHIPLET'S OWN. Each GEMM runs on a seed derived from PTA_SEED and the
    // GEMM's index, so the index is as much a part of the answer as the seed.
    // PTA_GEMM_CT: the index the next GEMM will take. Reported wherever a tile
    // is present, impaired or not, because an exact GEMM takes one too.
    int64_t gemm_index;
    int     is_model;                // PTA_CAPS2[31]: 1 a model, 0 not, -1 unknown
} pta_chiplet_analog_t;

// Read the block and decide. Returns 0 when a determination was made --
// "unidentified, everything unknown" is one -- or -1 when the device has no
// window at all, in which case *out is all unknown.
int pta_chiplet_read_analog(pta_chiplet_device_t* dev, pta_chiplet_analog_t* out);

// The map's completion test: ONE read of PTA_STATUS showing neither BUSY nor
// CAL_BUSY. Returns 1 idle, 0 not, -1 with no window.
int pta_chiplet_idle(pta_chiplet_device_t* dev);

// One GEMM, start to end. Waits for the tile to be idle, hands the command to
// the link, and waits for it to end. Returns 0 with C written, or -1 with C
// untouched and dev->error saying why.
int pta_chiplet_gemm(pta_chiplet_device_t* dev, int bank, int M, int N, int K,
                     const int32_t* A, const int32_t* B, int64_t* C);

// GEMM i's seed under PTA_SEED `seed`: the map's section 4. Here so that what
// reads the property can reproduce a result without a twin.
uint32_t pta_chiplet_gemm_seed(uint32_t seed, uint32_t index);

const char* pta_chiplet_error_string(int error);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // PTA_CHIPLET_H
