// grxCycleSummarize: does the summary say when its own span is meaningless?
//
// The span is "last end minus first start". That is a duration only if every
// slot came from ONE launch on ONE core. The cross-core case has been refused
// since the header was written. The cross-LAUNCH case was not, and it is the
// harder of the two to notice because nothing about the numbers looks wrong:
// MCYCLE restarts at zero at every launch (SimX's ProcessorImpl::run() opens
// with reset(), and MCYCLE reads PerfStats::cycles), so two launches produce
// two sets of small overlapping timestamps and the span across them is a
// maximum over strangers.
//
// It cost something real. tests/bench/block_cycles.cpp gave attention's four
// launches four regions of one probe buffer and summarised the buffer, and
// grxBLAS's sgemm kernel-selection rule was reverted on a 27.6% "regression"
// read off the result. Measured per launch the sign flips.
//
// What catches it is `maxLive`: the most slots live at once. A device holds
// maxWarpsPerMultiProcessor * multiProcessorCount warps and not one more, so a
// buffer reporting more than that did not come from one launch, whatever its
// timestamps say. This test is the arithmetic of that number -- no device
// needed, which is the point: it runs in tier 1 where the defect it describes
// could not be reproduced at all.

#include <grx/grx.h>
#include <grx/grx_cycles.h>

#include <vector>

#include "grx_test.h"

namespace {

grxCycleSlot slot(uint64_t start, uint64_t end, uint32_t core = 0,
                  uint32_t warp = 0) {
  grxCycleSlot s{};
  s.start = start; s.end = end; s.core = core; s.warp = warp;
  return s;
}

int live_of(const std::vector<grxCycleSlot>& v) {
  grxCycleSummary s{};
  grxCycleSummarize(v.data(), (int)v.size(), &s);
  return s.maxLive;
}

}  // namespace

int main() {
  grxtest::section("one launch");
  {
    // Four warps of one launch, started together and finishing apart. All four
    // are live at once, which is what a launch on a four-slot machine looks
    // like -- and is not evidence of anything wrong.
    const std::vector<grxCycleSlot> v = {
        slot(10, 90), slot(11, 80), slot(12, 95), slot(13, 70)};
    grxtest::check(live_of(v) == 4, "four overlapping warps report four live");

    grxCycleSummary s{};
    grxCycleSummarize(v.data(), (int)v.size(), &s);
    grxtest::check(s.span == 95 - 10, "and the span is last end minus first start");
    grxtest::check(s.spanIsValid == 1, "and it is valid: one core, one launch");
  }

  grxtest::section("warps that do not overlap");
  {
    // Two waves through the same slot: at most one live at a time. A machine
    // with one warp slot running two warps in sequence, which is legal and
    // which must NOT read like two launches.
    const std::vector<grxCycleSlot> v = {slot(10, 20), slot(30, 40)};
    grxtest::check(live_of(v) == 1, "sequential warps report one live");

    // And the boundary case, because it decides which side of the comparison
    // the sweep uses: a warp ending exactly as the next begins is one live,
    // not two. Ends are ordered before starts at equal cycles.
    const std::vector<grxCycleSlot> touch = {slot(10, 20), slot(20, 30)};
    grxtest::check(live_of(touch) == 1,
                   "a warp ending exactly as another begins is still one live");
  }

  grxtest::section("slots no warp reached");
  {
    // A zeroed slot is not a warp that ran from cycle 0 to cycle 0. It is
    // skipped, and it must not count toward maxLive either -- otherwise every
    // over-sized probe buffer would report itself as a multi-launch buffer.
    std::vector<grxCycleSlot> v = {slot(10, 90), slot(0, 0), slot(0, 0)};
    grxtest::check(live_of(v) == 1, "unwritten slots are not live warps");
  }

  grxtest::section("two launches in one buffer -- the case this exists for");
  {
    // Attention in miniature, and these are the shapes of the real numbers:
    // the scores GEMM ran 8 warps for ~10000 cycles, and the mask ran 16 warps
    // that reported starting at ~3000 -- on its OWN clock, which also began at
    // zero. Spanning the two gives a plausible-looking 10000 that is a maximum
    // over two unrelated counters.
    std::vector<grxCycleSlot> v;
    for (int i = 0; i < 8; ++i)  v.push_back(slot(0 + i, 10000 + i));
    for (int i = 0; i < 16; ++i) v.push_back(slot(3000 + i, 6000 + i));

    grxCycleSummary s{};
    grxCycleSummarize(v.data(), (int)v.size(), &s);
    grxtest::check(s.maxLive == 24,
                   "a two-launch buffer reports every warp live at once");
    grxtest::check(s.spanIsValid == 1,
                   "and the span still LOOKS valid -- one core, no refusal, "
                   "which is exactly why maxLive has to be checked");

    // The refusal a caller makes with it. Neither launch alone could put 24
    // warps on a 16-warp machine, so the buffer is not one launch.
    const int occupancy = 16;
    grxtest::check(s.maxLive > occupancy,
                   "and it exceeds a 16-warp machine, which proves it");

    // The control: each launch summarised alone is under the limit and gives a
    // real duration. Same slots, split the way the fix splits them.
    grxCycleSummary a{}, b{};
    grxCycleSummarize(v.data(), 8, &a);
    grxCycleSummarize(v.data() + 8, 16, &b);
    grxtest::check(a.maxLive == 8 && b.maxLive == 16,
                   "each launch alone is within what the machine holds");
    grxtest::check(a.span == 10007 && b.span == 3015,
                   "and each alone gives a duration rather than a maximum");
    grxtest::check(a.span + b.span > s.span,
                   "the spanned reading was SMALLER than the two real ones -- "
                   "it understated the stage rather than merely blurring it");
  }

  grxtest::section("what a device says it holds");
  {
    // The comparison is against a device property, not a constant, so this
    // checks the property exists and is positive wherever a device does. No
    // device is a skip, not a failure.
    int count = 0;
    if (grxGetDeviceCount(&count) == grxSuccess && count > 0) {
      grxDeviceProp_t p{};
      if (grxGetDeviceProperties(&p, 0) == grxSuccess) {
        grxtest::check(p.maxWarpsPerMultiProcessor > 0 &&
                       p.multiProcessorCount > 0,
                       "the device reports an occupancy to compare against");
      }
    } else {
      std::printf("  (no device: the occupancy check needs one)\n");
    }
  }

  return grxtest::report();
}
