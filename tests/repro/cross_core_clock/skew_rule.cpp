// What grxCycleSummarize reports across cores, checked against hand-built
// slots -- no device, no simulator.
//
// It does NOT decide whether a cross-core span is meaningful. An earlier
// version of this file tested a rule that accepted one when coreSkew was a
// small fraction of the span, and that rule was wrong: skew cannot separate
// disagreeing clocks from cores that honestly started at different times. See
// grx_cycles.h. What is checked here is that the EVIDENCE is right -- the core
// count, the skew, the cross-core extent -- and that `span` still refuses to
// cross cores so no existing caller changes meaning underneath it.

#include <cstdio>
#include <cstring>
#include <vector>

#include "grx/grx_cycles.h"

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("  %-5s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}

// n warps per core, each core's warps starting at `origin[c]` and running
// `busy` cycles, laid out so the span is dominated by the work rather than by
// the skew unless the skew is made large.
static std::vector<grxCycleSlot> build(const std::vector<uint64_t>& origin,
                                       int per_core, uint64_t busy,
                                       uint64_t stagger) {
  std::vector<grxCycleSlot> v;
  for (size_t c = 0; c < origin.size(); ++c) {
    for (int w = 0; w < per_core; ++w) {
      grxCycleSlot s{};
      s.start = origin[c] + (uint64_t)w * stagger;
      s.end   = s.start + busy;
      s.core  = (uint32_t)c;
      s.warp  = (uint32_t)w;
      v.push_back(s);
    }
  }
  return v;
}

int main() {
  std::printf("cross-core evidence reported by grxCycleSummarize\n\n");

  // 1. One core. Unchanged behaviour, and the case every existing caller relies
  //    on -- if this moved, the rule broke something that was already right.
  {
    grxCycleSummary s{};
    std::vector<grxCycleSlot> v = build({1000}, 8, 500, 10);
    grxCycleSummarize(v.data(), (int)v.size(), &s);
    std::printf("one core\n");
    check(s.cores == 1, "reports 1 core");
    check(s.spanCrossesCores == 0, "does not claim to cross cores");
    check(s.spanIsValid == 1, "span is valid");
    check(s.span == 570, "span is last end minus first start");
    check(s.coreSkew == 0, "skew is zero");
  }

  // 2. Four cores that agree on the origin. This is what simx does: 158 cycles
  //    of spread against a 37731-cycle span. Scaled down, same shape.
  {
    grxCycleSummary s{};
    std::vector<grxCycleSlot> v =
        build({1000, 1050, 1100, 1158}, 8, 5000, 10);
    grxCycleSummarize(v.data(), (int)v.size(), &s);
    std::printf("\nfour cores, 158 apart (what simx does)\n");
    check(s.cores == 4, "counts four distinct cores");
    check(s.spanCrossesCores == 1, "says the span crosses cores");
    check(s.coreSkew == 158, "reports the skew it measured");
    check(s.crossCoreSpan == 5228, "crossCoreSpan is the full extent");
    check(s.spanIsValid == 0 && s.span == 0,
          "and `span` still refuses to cross cores");
  }

  // 3. Four cores far apart. Whether this is disagreeing clocks or a very
  //    staggered dispatch is EXACTLY what this function cannot tell, so it
  //    reports both numbers and asserts nothing about which.
  {
    grxCycleSummary s{};
    std::vector<grxCycleSlot> v =
        build({1000, 3000, 5000, 7000}, 8, 500, 10);
    grxCycleSummarize(v.data(), (int)v.size(), &s);
    std::printf("\nfour cores, 6000 apart\n");
    check(s.cores == 4, "counts four distinct cores");
    check(s.coreSkew == 6000, "reports the large skew");
    check(s.crossCoreSpan == 6570, "still reports the extent");
    check(s.spanIsValid == 0 && s.span == 0, "`span` refuses, as always");
  }

  // 5. Unwritten slots stay zero and must not drag the origin to cycle 0 --
  //    the failure that would make every span look enormous and valid.
  {
    grxCycleSummary s{};
    std::vector<grxCycleSlot> v = build({1000, 1050}, 4, 5000, 10);
    v.resize(v.size() + 16);            // 16 all-zero slots
    grxCycleSummarize(v.data(), (int)v.size(), &s);
    std::printf("\nunwritten slots\n");
    check(s.warps == 8, "counts only the slots a warp wrote");
    check(s.cores == 2, "and only the cores that wrote one");
    check(s.crossCoreSpan > 5000 && s.crossCoreSpan < 6000,
          "extent is not dragged back to cycle zero");
  }

  std::printf("\n%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
