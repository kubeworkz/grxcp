// The NPU backend's DECISIONS, driven through a register model.
//
// test_npu_c930.cc covers the offline half well -- register offsets, argument
// validation, INT8 packing, a GEMM reference. What it cannot reach is the half
// that talks to the device, and it says so: its mock mode prints
// "[SKIP] Mock mode requires mmap infrastructure". That skipped half is where
// both bugs were.
//
// A register model is that infrastructure. Four models, each one state real
// hardware can be in, and each producing a decision this backend has to get
// right:
//
//   ABSENT   reads 0, writes go nowhere   -- must NOT be detected
//   DEAD     reads all-ones               -- must NOT be detected
//   LIVE     stores writes, START->DONE   -- detected, and a GEMM succeeds
//   WEDGED   stores writes, START->nothing-- detected, and a GEMM must FAIL
//
// ABSENT and WEDGED are the two that were wrong, and they are wrong in the
// direction that does not announce itself:
//
//   - detect() read STATUS and accepted anything that was not 0xFFFFFFFF and
//     not above 0x7. Its own comment said an absent NPU reads 0x0. 0x0 passes
//     that test. Any host where /dev/mem opens grew an NPU it did not have.
//   - gemm() waited for !BUSY and no ERROR, which a device that ignored every
//     write satisfies instantly, and never read the DONE bit the register map
//     latches for exactly this purpose. It returned success over a GEMM that
//     never ran, leaving C holding whatever it held before.
//
// Together: a device that is not there, reporting successful GEMMs, producing
// stale output. Every gate in this project is supposed to be watched failing --
// these two were watched passing when they should have failed, which is the
// same evidence read the other way round.
//
// A MODEL IS NOT HARDWARE. Nothing here says the c930 works, and a green run
// must never be reported as the NPU working. It says this file's logic is
// right, which is the half that was not.
//
// THE SECOND HALF OF THIS FILE is one more decision: whether a GEMM on the
// device is the GEMM that was asked for. A c930 can be built with a photonic
// tile, which is an error model, and npu_c930_read_analog is what says so. Six
// more register files, each a state the RTL can be in:
//
//   ALIASING  sixteen words, no PTA block  -- UNKNOWN, and not "impaired by
//                                             whatever DIM_M happens to hold"
//   NO-ID     the block, without its magic -- UNKNOWN, not "no tile"
//   ARRAY     the block, nothing built     -- exact, whatever PTA_IMPAIR says
//   TILE/OFF  a tile, PTA_IMPAIR clear     -- exact, and stale config unreported
//   TILE/ON   a tile, PTA_IMPAIR set       -- ANALOG, with PTA_CTRL.EN CLEAR
//   REFUSING  PTA_IMPAIR asks for a bit the build lacks -- the GEMM fails
//
// TILE/ON with EN clear is the case the specification had wrong. EN is the
// calibration engine's enable; a report keyed on it calls that GEMM native.

#include "npu_c930.h"

#include <cstdio>
#include <cstring>

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
  std::printf("  %s  %s\n", cond ? "ok  " : "FAIL", what);
  if (!cond) ++g_failures;
}

// ---------------------------------------------------------------------------
// The models
// ---------------------------------------------------------------------------

// A device that is not on the bus. Reads return zero -- which is what an mmap
// over unbacked physical address space gives you -- and writes are discarded,
// because there is nothing there to store them.
uint32_t absent_read(void*, uint32_t)          { return 0u; }
void     absent_write(void*, uint32_t, uint32_t) {}

// A bus that faults or floats high.
uint32_t dead_read(void*, uint32_t)            { return 0xFFFFFFFFu; }
void     dead_write(void*, uint32_t, uint32_t)   {}

// A register file. `completes` decides whether writing CTRL.START latches DONE:
// with it clear this is a device that accepts a launch and does nothing, which
// is what a wedged DMA or an unclocked accelerator looks like from the host.
struct Regs {
  uint32_t r[16] = {0};
  bool     completes = true;
  int      starts = 0;
};

uint32_t regs_read(void* ctx, uint32_t off) {
  return static_cast<Regs*>(ctx)->r[(off >> 2) & 0xF];
}

void regs_write(void* ctx, uint32_t off, uint32_t v) {
  Regs* g = static_cast<Regs*>(ctx);
  const uint32_t idx = (off >> 2) & 0xF;
  if (off == 0x00 && (v & NPU_C930_CTRL_START)) {
    ++g->starts;
    // Real hardware clears DONE on START and latches it again at completion.
    // The whole point of the wedged model is that the second half never
    // happens, so BUSY drops but DONE stays clear.
    g->r[1] = g->completes ? NPU_C930_STATUS_DONE : 0u;
    return;
  }
  g->r[idx] = v;
}

// A register file with the PTA block: 256 words, as c930_npu_csr.sv decodes
// since the block was added. `id` and `built` are what PTA_ID and PTA_CAPS1
// read -- the two things that differ between builds. The core's refusal is
// modelled because the backend's handling of it is one of the decisions: a
// START whose PTA_IMPAIR asks for a bit outside `built` sets ERROR and runs
// nothing.
struct PtaRegs {
  uint32_t r[256] = {0};
  uint32_t id     = 0x50544101u;
  uint32_t built  = 0;
  int      starts = 0;
  int      refusals = 0;
};

uint32_t pta_read(void* ctx, uint32_t off) {
  PtaRegs* g = static_cast<PtaRegs*>(ctx);
  switch (off) {
    case NPU_C930_PTA_ID:    return g->id;
    case NPU_C930_PTA_CAPS1: return g->built;
    default:                 return g->r[(off >> 2) & 0xFF];
  }
}

void pta_write(void* ctx, uint32_t off, uint32_t v) {
  PtaRegs* g = static_cast<PtaRegs*>(ctx);
  if (off == NPU_C930_PTA_ID || off == NPU_C930_PTA_CAPS1) return;  // read-only
  if (off == 0x00 && (v & NPU_C930_CTRL_START)) {
    ++g->starts;
    const uint32_t impair = g->r[NPU_C930_PTA_IMPAIR >> 2] & NPU_C930_PTA_DEFINED;
    if (impair & ~g->built) {
      ++g->refusals;
      g->r[1] = NPU_C930_STATUS_ERROR;
    } else {
      g->r[1] = NPU_C930_STATUS_DONE;
    }
    return;
  }
  g->r[(off >> 2) & 0xFF] = v;
}

const uint32_t kTileBuilt = NPU_C930_PTA_DEFINED & ~NPU_C930_PTA_MZM_NL;  // 0x5F

bool all_unknown(const npu_c930_analog_t& a) {
  return a.identified == 0 && a.map_version == -1 && a.analog == -1 &&
         a.tile_present == -1 && a.activation_bits == -1 &&
         a.weight_bits == -1 && a.adc_bits == -1 && a.adc_shift == -1 &&
         a.seed == -1 && a.impairments == -1 &&
         a.impairments_implemented == -1 && a.impairments_requested == -1;
}

bool config_unreported(const npu_c930_analog_t& a) {
  return a.activation_bits == -1 && a.weight_bits == -1 && a.adc_bits == -1 &&
         a.adc_shift == -1 && a.seed == -1 && a.impairments == -1;
}

// ---------------------------------------------------------------------------

void case_absent() {
  std::printf("a device that is not on the bus (reads 0, writes discarded):\n");
  npu_c930_device_t dev;
  npu_c930_attach_model(&dev, absent_read, absent_write, nullptr);
  const int found = npu_c930_detect(&dev);
  check(found == 0, "npu_c930_detect reports it absent");
  check(dev.present == 0, "and dev.present stays 0");
  if (found) {
    std::printf("        A GRX930 NPU has been enumerated on a machine that\n"
                "        has none. grx-smi will list it and grxSetDevice will\n"
                "        select it.\n");
  }
}

void case_dead() {
  std::printf("a bus that floats high (reads all-ones):\n");
  npu_c930_device_t dev;
  npu_c930_attach_model(&dev, dead_read, dead_write, nullptr);
  check(npu_c930_detect(&dev) == 0, "npu_c930_detect reports it absent");
}

void case_live() {
  std::printf("a device that is really there:\n");
  Regs regs;
  regs.completes = true;
  npu_c930_device_t dev;
  npu_c930_attach_model(&dev, regs_read, regs_write, &regs);

  check(npu_c930_detect(&dev) == 1, "npu_c930_detect finds it");
  check(regs.r[2] == 0u,
        "the probe put DIM_M back the way it found it");

  const int rc = npu_c930_gemm(&dev, 4, 4, 4, 0x1000, 0x2000, 0x3000);
  check(rc == 0, "a GEMM reports success");
  check(regs.starts == 1, "CTRL.START was pulsed exactly once");
  check(regs.r[2] == 4u && regs.r[3] == 4u && regs.r[4] == 4u,
        "M, N and K reached DIM_M/DIM_N/DIM_K");
  check(regs.r[5] == 0x1000u && regs.r[6] == 0x2000u && regs.r[7] == 0x3000u,
        "A, B and C addresses reached A_BASE/B_BASE/C_BASE");
}

void case_wedged() {
  std::printf("a device that accepts a launch and never finishes:\n");
  Regs regs;
  regs.completes = false;
  npu_c930_device_t dev;
  npu_c930_attach_model(&dev, regs_read, regs_write, &regs);

  check(npu_c930_detect(&dev) == 1,
        "it is detected -- the registers are real");
  const int rc = npu_c930_gemm(&dev, 4, 4, 4, 0x1000, 0x2000, 0x3000);
  check(rc != 0, "but the GEMM is reported as FAILED, not as success");
  if (rc == 0) {
    std::printf("        C was never written and the caller was told the GEMM\n"
                "        succeeded. Whatever was in that buffer is now the\n"
                "        answer.\n");
  }
}

// The two models that must be REJECTED are only evidence if the same predicate
// accepts something. Without this, a detect() that returned 0 unconditionally
// would pass the first half of this file perfectly.
void case_discrimination() {
  std::printf("the detector actually discriminates:\n");
  Regs regs;
  npu_c930_device_t live, absent;
  npu_c930_attach_model(&live, regs_read, regs_write, &regs);
  npu_c930_attach_model(&absent, absent_read, absent_write, nullptr);
  const int a = npu_c930_detect(&live);
  const int b = npu_c930_detect(&absent);
  check(a == 1 && b == 0,
        "the same predicate says yes to hardware and no to nothing");
}

// ---------------------------------------------------------------------------
// Whether a GEMM here is the GEMM that was asked for
// ---------------------------------------------------------------------------

void case_analog_aliasing() {
  std::printf("a register file with no PTA block, which aliases above 0x3C:\n");
  Regs regs;
  npu_c930_device_t dev;
  npu_c930_attach_model(&dev, regs_read, regs_write, &regs);
  npu_c930_detect(&dev);
  // The trap, set deliberately: PTA_IMPAIR is 0x148 and this file decodes
  // four address bits, so that address IS DIM_M. Program M = 4 and a driver
  // that read the block without checking the magic sees SHOT enabled.
  npu_c930_gemm(&dev, 4, 4, 4, 0x1000, 0x2000, 0x3000);
  check(regs_read(&regs, NPU_C930_PTA_IMPAIR) == 4u,
        "PTA_IMPAIR's address really does read back DIM_M here (4)");
  npu_c930_analog_t a;
  check(npu_c930_read_analog(&dev, &a) == 0, "a determination is made");
  check(all_unknown(a), "and it is UNKNOWN in every field");
  check(a.analog != 1 && a.impairments != 4,
        "in particular not 'analog, impairments 0x04', which is what M = 4 looks like");
}

void case_analog_no_id() {
  std::printf("a register file with the block but no identity word:\n");
  PtaRegs regs;
  regs.id = 0;                     // what main reads today, before PTA_ID
  regs.built = kTileBuilt;         // even with a tile behind it
  npu_c930_device_t dev;
  npu_c930_attach_model(&dev, pta_read, pta_write, &regs);
  npu_c930_detect(&dev);
  pta_write(&regs, NPU_C930_PTA_IMPAIR, NPU_C930_PTA_SHOT);
  npu_c930_analog_t a;
  npu_c930_read_analog(&dev, &a);
  check(all_unknown(a),
        "UNKNOWN -- not 'no tile', which nothing here could have established");
  regs.id = 0x50544200u;           // a near miss: "PTB"
  npu_c930_read_analog(&dev, &a);
  check(all_unknown(a), "and a magic that is one letter off is not the magic");
}

void case_analog_array() {
  std::printf("the block, on a build with no tile:\n");
  PtaRegs regs;
  regs.built = 0;
  npu_c930_device_t dev;
  npu_c930_attach_model(&dev, pta_read, pta_write, &regs);
  npu_c930_detect(&dev);
  npu_c930_analog_t a;
  npu_c930_read_analog(&dev, &a);
  check(a.identified == 1 && a.map_version == 1, "identified, map version 1");
  check(a.tile_present == 0 && a.analog == 0, "no tile, and GEMMs are exact");
  check(a.impairments_implemented == 0,
        "nothing implemented is reported as 0 -- it is known, so not -1");
  check(config_unreported(a), "and no model configuration is reported");

  // The registers still take writes on this build. That changes nothing about
  // what a GEMM is, because there is nothing here to impair it.
  pta_write(&regs, NPU_C930_PTA_IMPAIR, NPU_C930_PTA_THERMAL);
  pta_write(&regs, NPU_C930_PTA_BITS, 0x00003688u);
  npu_c930_read_analog(&dev, &a);
  check(a.tile_present == 0 && a.analog == 0 && config_unreported(a),
        "PTA_IMPAIR set on an array is still not an analog GEMM");
  check(a.impairments_requested == NPU_C930_PTA_THERMAL,
        "though what was asked for is kept, to explain the refusal");
}

void case_analog_tile_off() {
  std::printf("a tile, with PTA_IMPAIR clear:\n");
  PtaRegs regs;
  regs.built = kTileBuilt;
  npu_c930_device_t dev;
  npu_c930_attach_model(&dev, pta_read, pta_write, &regs);
  npu_c930_detect(&dev);
  // Stale configuration, as a previous user would leave it -- and EN set,
  // which is the calibration engine's and makes nothing analog.
  pta_write(&regs, NPU_C930_PTA_BITS, 0x00003688u);
  pta_write(&regs, NPU_C930_PTA_SEED, 0x2au);
  pta_write(&regs, NPU_C930_PTA_CTRL, 0x1u);
  npu_c930_analog_t a;
  npu_c930_read_analog(&dev, &a);
  check(a.tile_present == 1, "the tile is reported present");
  check(a.analog == 0, "GEMMs are exact -- EN set does not make them analog");
  check(a.impairments_implemented == (int64_t)kTileBuilt,
        "what the tile could model is reported (0x5f)");
  check(config_unreported(a),
        "the stale BITS and SEED are NOT reported: they describe a model that "
        "is not running");
}

void case_analog_tile_on() {
  std::printf("a tile, with PTA_IMPAIR set and PTA_CTRL.EN CLEAR:\n");
  PtaRegs regs;
  regs.built = kTileBuilt;
  npu_c930_device_t dev;
  npu_c930_attach_model(&dev, pta_read, pta_write, &regs);
  npu_c930_detect(&dev);
  pta_write(&regs, NPU_C930_PTA_CTRL, 0x0u);                     // EN clear
  pta_write(&regs, NPU_C930_PTA_IMPAIR, kTileBuilt);
  pta_write(&regs, NPU_C930_PTA_BITS, 8u | (8u << 4) | (6u << 8) | (3u << 12));
  pta_write(&regs, NPU_C930_PTA_SEED, 0x2au);
  npu_c930_analog_t a;
  npu_c930_read_analog(&dev, &a);
  check(a.analog == 1,
        "ANALOG -- with EN clear, which is the case a report keyed on EN gets wrong");
  if (a.analog != 1) {
    std::printf("        This GEMM is noisy and was reported as native. That is\n"
                "        the one direction this report must never be wrong in.\n");
  }
  check(a.tile_present == 1, "tile present");
  check(a.activation_bits == 8 && a.weight_bits == 8 && a.adc_bits == 6 &&
        a.adc_shift == 3, "PTA_BITS decodes to a8, w8, ADC 6 bits << 3");
  check(a.seed == 0x2a, "the seed is PTA_SEED");
  check(a.impairments == (int64_t)kTileBuilt &&
        a.impairments_implemented == (int64_t)kTileBuilt,
        "the enables in force, and the mask of what is implemented");

  // The reasons the fields are -1-or-value and 64-bit, each exercised.
  pta_write(&regs, NPU_C930_PTA_BITS, 0u);
  pta_write(&regs, NPU_C930_PTA_SEED, 0xFFFFFFFFu);
  npu_c930_read_analog(&dev, &a);
  check(a.activation_bits == 0 && a.weight_bits == 0 && a.adc_bits == 0,
        "unquantised is reported as 0, which is why 'not applicable' is -1");
  check(a.seed == 4294967295LL && a.seed != -1,
        "a seed of 0xffffffff is 4294967295, not the unknown sentinel");

  // One bit is enough.
  pta_write(&regs, NPU_C930_PTA_IMPAIR, NPU_C930_PTA_QUANT);
  npu_c930_read_analog(&dev, &a);
  check(a.analog == 1 && a.impairments == NPU_C930_PTA_QUANT,
        "a single impairment is an analog GEMM");

  // And it is a live read, not a latched one.
  pta_write(&regs, NPU_C930_PTA_IMPAIR, 0u);
  npu_c930_read_analog(&dev, &a);
  check(a.analog == 0 && config_unreported(a),
        "clearing PTA_IMPAIR is seen by the next read");
}

void case_analog_refusing() {
  std::printf("PTA_IMPAIR asks for an impairment the build does not have:\n");
  PtaRegs regs;
  regs.built = kTileBuilt;
  npu_c930_device_t dev;
  npu_c930_attach_model(&dev, pta_read, pta_write, &regs);
  npu_c930_detect(&dev);
  pta_write(&regs, NPU_C930_PTA_IMPAIR, NPU_C930_PTA_QUANT | NPU_C930_PTA_MZM_NL);
  npu_c930_analog_t a;
  npu_c930_read_analog(&dev, &a);
  check((a.impairments & ~a.impairments_implemented) == NPU_C930_PTA_MZM_NL,
        "impairments & ~implemented is exactly the bit that will refuse");
  std::printf("  note  the driver explains each refusal on stderr:\n");
  const int rc = npu_c930_gemm(&dev, 4, 4, 4, 0x1000, 0x2000, 0x3000);
  check(rc != 0, "the GEMM is reported as FAILED");
  check(regs.refusals == 1 && regs.starts == 1,
        "by the device's refusal, once -- the driver did not pre-empt the START");

  // The same on an array, where every bit refuses.
  PtaRegs arr;
  arr.built = 0;
  npu_c930_device_t adev;
  npu_c930_attach_model(&adev, pta_read, pta_write, &arr);
  npu_c930_detect(&adev);
  pta_write(&arr, NPU_C930_PTA_IMPAIR, NPU_C930_PTA_QUANT);
  check(npu_c930_gemm(&adev, 4, 4, 4, 0x1000, 0x2000, 0x3000) != 0,
        "an array asked for QUANT fails too -- no exact result under an analog label");
  pta_write(&arr, NPU_C930_PTA_IMPAIR, 0u);
  check(npu_c930_gemm(&adev, 4, 4, 4, 0x1000, 0x2000, 0x3000) == 0,
        "and runs again once PTA_IMPAIR is cleared");
}

void case_analog_no_path() {
  std::printf("a device with no register path at all:\n");
  npu_c930_device_t dev;
  std::memset(&dev, 0, sizeof(dev));
  npu_c930_analog_t a;
  check(npu_c930_read_analog(&dev, &a) != 0, "the read reports that it could not");
  check(all_unknown(a), "and leaves every field unknown rather than zero");
  check(npu_c930_read_analog(&dev, nullptr) != 0, "a null result pointer is refused");
}

}  // namespace

int main() {
  std::printf("=== GRX930 NPU backend: decisions, against a register model ===\n");
  std::printf("NOTE: a model is not hardware. Passing here says this file's\n"
              "      logic is right. It says nothing about the c930.\n\n");
  case_absent();
  case_dead();
  case_live();
  case_wedged();
  case_discrimination();
  std::printf("\n--- whether a GEMM here is the GEMM that was asked for ---\n");
  case_analog_aliasing();
  case_analog_no_id();
  case_analog_array();
  case_analog_tile_off();
  case_analog_tile_on();
  case_analog_refusing();
  case_analog_no_path();
  std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "PASSED",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures ? 1 : 0;
}
