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
  std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "PASSED",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures ? 1 : 0;
}
