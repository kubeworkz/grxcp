// Cycle-probe validation gate.
//
// Runs the same kernel at 1x, 2x and 4x the work and checks that the measured
// cycle count follows. What this rules out, in order of how badly each would
// mislead:
//
//   * a probe that returns a constant (ratios would be 1)
//   * a probe reading a clock unrelated to the work (ratios would be noise)
//   * a probe measuring only its own overhead (the intercept would dominate)
//   * a counter that does not advance at all (everything zero)
//
// The tolerance is deliberately loose. The claim being tested is "this counts
// work", not "this counts work to three decimal places"; a tight bound here
// would fail on scheduling jitter and teach everyone to ignore the gate.

#include <grx/grx.h>
#include <grx/grx_cycles.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common.h"

#define CHECK(call)                                                      \
  do {                                                                   \
    grxError_t e_ = (call);                                              \
    if (e_ != grxSuccess) {                                              \
      std::fprintf(stderr, "%s -> %s (%s)\n", #call,                     \
                   grxGetErrorString(e_), grxGetErrorName(e_));          \
      return 1;                                                          \
    }                                                                    \
  } while (0)

namespace {
int failures = 0;
void expect(bool ok, const char* what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++failures;
}
}  // namespace

int main(int argc, char** argv) {
  const char* image = (argc > 1) ? argv[1] : "cycles.vxbin";

  int count = 0;
  CHECK(grxGetDeviceCount(&count));
  if (count == 0) { std::fprintf(stderr, "no devices\n"); return 77; }
  CHECK(grxSetDevice(0));

  grxDeviceProp_t prop{};
  CHECK(grxGetDeviceProperties(&prop, 0));

  grxModule_t   mod = nullptr;
  grxFunction_t fn  = nullptr;
  CHECK(grxModuleLoad(&mod, image));
  CHECK(grxModuleGetFunction(&fn, mod, "spin"));

  const unsigned warps_per_block = 2;
  const unsigned block  = (unsigned)prop.warpSize * warps_per_block;
  const unsigned blocks = 2;
  const int      nslots = (int)(blocks * warps_per_block);

  void *dSlots = nullptr, *dSink = nullptr;
  CHECK(grxMalloc(&dSlots, (size_t)nslots * sizeof(grxCycleSlot)));
  CHECK(grxMalloc(&dSink, sizeof(uint64_t)));

  auto measure = [&](uint32_t iters, bool probe,
                     grxCycleSummary* out) -> grxError_t {
    grxError_t e = grxMemset(dSlots, 0, (size_t)nslots * sizeof(grxCycleSlot));
    if (e != grxSuccess) return e;

    cycles_args args{};
    args.slots = probe ? (uint64_t)(uintptr_t)dSlots : 0;
    args.sink  = (uint64_t)(uintptr_t)dSink;
    args.iters = iters;
    args.seed  = 7;

    e = grxLaunchFunction(fn, dim3_t{blocks, 1, 1}, dim3_t{block, 1, 1},
                          &args, sizeof(args), 0, nullptr);
    if (e != grxSuccess) return e;
    e = grxDeviceSynchronize();
    if (e != grxSuccess) return e;

    std::vector<grxCycleSlot> slots(nslots);
    e = grxMemcpy(slots.data(), dSlots, slots.size() * sizeof(grxCycleSlot),
                  grxMemcpyDefault);
    if (e != grxSuccess) return e;
    grxCycleSummarize(slots.data(), nslots, out);
    return grxSuccess;
  };

  std::printf("%s: %u blocks x %u warps, cycle counter is per core\n",
              prop.name, blocks, warps_per_block);

  std::printf("does the probe respond to work:\n");
  const uint32_t base_iters = 200;
  grxCycleSummary s1{}, s2{}, s4{};
  CHECK(measure(base_iters, true, &s1));
  CHECK(measure(base_iters * 2, true, &s2));
  CHECK(measure(base_iters * 4, true, &s4));

  std::printf("        %6u iters: median %llu cycles per warp, span %llu\n",
              base_iters, (unsigned long long)s1.busyMedian,
              (unsigned long long)s1.span);
  std::printf("        %6u iters: median %llu cycles per warp, span %llu\n",
              base_iters * 2, (unsigned long long)s2.busyMedian,
              (unsigned long long)s2.span);
  std::printf("        %6u iters: median %llu cycles per warp, span %llu\n",
              base_iters * 4, (unsigned long long)s4.busyMedian,
              (unsigned long long)s4.span);

  expect(s1.warps == nslots && s2.warps == nslots && s4.warps == nslots,
         "every warp wrote its slot");
  expect(s1.busyMedian > 0, "the counter advances during a kernel");
  expect(s2.busyMedian > s1.busyMedian && s4.busyMedian > s2.busyMedian,
         "more work costs more cycles");

  // Slope, not ratio: the fixed cost of entering the kernel and of the probe
  // itself sits in every measurement, and comparing increments cancels it.
  const double d1 = (double)s2.busyMedian - (double)s1.busyMedian;
  const double d2 = (double)s4.busyMedian - (double)s2.busyMedian;
  const double slope_ratio = (d1 > 0.0) ? d2 / d1 : 0.0;
  std::printf("        doubling the work adds %.0f then %.0f cycles "
              "(ratio %.2f, want ~2)\n", d1, d2, slope_ratio);
  expect(slope_ratio > 1.8 && slope_ratio < 2.2,
         "cycles scale linearly with the work");

  const double per_iter = d2 / (double)(base_iters * 2);
  std::printf("        marginal cost %.2f cycles per iteration\n", per_iter);
  expect(per_iter > 0.5 && per_iter < 200.0,
         "the per-iteration cost is a plausible number of cycles");

  std::printf("the probe reports honestly:\n");
  expect(s1.spanIsValid == (prop.multiProcessorCount >= 1 ? s1.cores == 1 : 0),
         "a span is offered only when every warp shared a core");
  expect(s1.span >= s1.busyMax,
         "the span covers the longest warp");

  {
    // A disabled probe must leave the array alone: the null-pointer path is
    // what production kernels take, and it has to cost nothing and touch
    // nothing.
    grxCycleSummary off{};
    CHECK(measure(base_iters, false, &off));
    expect(off.warps == 0, "a null probe pointer writes no slots");
  }

  CHECK(grxFree(dSlots));
  CHECK(grxFree(dSink));
  CHECK(grxModuleUnload(mod));

  if (failures) { std::printf("FAILED (%d)\n", failures); return 1; }
  std::printf("PASSED\n");
  return 0;
}
