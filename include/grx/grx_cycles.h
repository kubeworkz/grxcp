// GRXCP — device cycle measurement.
//
// WHY THIS EXISTS AT ALL
//
// grxEventElapsedTime cannot answer "how fast is this kernel". The driver's
// timestamps are host clocks taken around the command, so on a simulator they
// measure the SIMULATOR (cuda_mapping.md section 7.4). The device's own cycle
// counter is the only thing that measures the device, and reading it means the
// kernel has to cooperate -- hence an ABI rather than an API.
//
// A kernel constructs a grx::cycle_probe (include/grx/device/grx_cycles.h) over
// an array of these slots, one per warp, and calls finish() before it returns.
// The host allocates the array, passes it in the kernel's argument block, and
// summarises it afterwards. A null pointer disables the whole thing, so one
// kernel serves both the measured and the unmeasured case.
//
// WHAT THE NUMBERS ARE, PRECISELY
//
// The counter is VX_CSR_MCYCLE: the core's cycle count, PER CORE, incremented
// once per cycle by hardware and once per simulated tick by SimX. So:
//
//   * Within one core AND ONE LAUNCH, differences are real cycle counts.
//   * ACROSS cores -- this said, for as long as the file existed, that "the
//     counters are independent and a span computed from two different cores
//     means nothing", and grxCycleSummarize refused to produce one. THAT WAS
//     NEVER MEASURED. It was inferred from MCYCLE being per-core storage, and
//     storage being per core does not make TIME per core.
//
//     Measured (tests/repro/cross_core_clock/): on simx at 4 SMs, one launch,
//     64 warps evenly spread, the four cores' first reads are 32434, 32488,
//     32592, 32523 -- a spread of 158 cycles against a 37731-cycle span, 0.42%.
//     They share an origin. In simx every Core is a SimObject ticked once per
//     simulated cycle and Core::reset() zeroes perf_stats_ for all of them at
//     the launch, so there is one clock wearing four hats.
//
//     The cost of the wrong claim was the whole instrument: the transformer
//     block bench reported 0 of 12 stages on a 4-SM device, because at 4 SMs
//     every stage spreads across cores. A refusal is not free just because it
//     is conservative -- this one made the machine unmeasurable exactly where
//     measurement was about to matter.
//
//     So `crossCoreSpan` is now produced unconditionally and `coreSkew` ships
//     beside it. Whether the span MEANS anything is a fact about a BACKEND, and
//     this header cannot ask the device, so it reports and the caller decides.
//     Same division of labour as `maxLive`.
//
//     BOTH SIMULATOR BACKENDS HAVE NOW BEEN MEASURED, same shape (16x16
//     layernorm), one-core run first to establish what the launch preamble
//     costs:
//
//       backend  cores  per-core first reads          baseline  skew
//       simx       1     9477                            --       0
//       simx       4     8554 8563 8580 8919            9477     365
//       rtlsim     1     8633                            --       0
//       rtlsim     4     8372 8456 8456 8842            8633     470
//
//     On both, the four-core readings sit ON the one-core preamble rather than
//     collapsing toward zero -- rtlsim's 8372 against a baseline of 8633 is
//     within 3%. Aligned on both. SILICON IS STILL UNMEASURED and gets no
//     assumption either way.
//
//     THE TEST ITSELF WAS WRONG FIRST, and the error is the interesting part.
//     The probe originally concluded "small skew => aligned". That is backwards.
//     If each core's counter were reset when THAT CORE got work, every core
//     would read a small number from its own reset and the cores would AGREE --
//     independent counters produce a TINY skew. Aligned counters carry the whole
//     preamble, so their readings are large and any spread between them is
//     genuine dispatch stagger. The discriminator is the MAGNITUDE of the first
//     reads against a one-core baseline, not the spread between them, and one
//     run cannot settle it.
//
//     A DESIGN THAT WAS TRIED AND WAS WRONG, kept because the reasoning is the
//     trap: the first version accepted a cross-core span when coreSkew was a
//     small fraction of it, as if skew measured clock error. It does not.
//     Once the counters are known aligned, a core whose first warp starts 3014
//     cycles after another's REALLY DID START 3014 CYCLES LATER -- that is
//     dispatch ramp, it is part of what the stage costs, and the span
//     including it is the right number. Start skew cannot separate "the clocks
//     disagree" from "the cores began at different times", so it cannot be the
//     test for the first. It threw away five of twelve stages for being
//     honestly staggered. Alignment is established ONCE per backend by
//     tests/repro/cross_core_clock/align_probe, not re-derived per stage from
//     a quantity that cannot see it.
//   * ACROSS LAUNCHES they are independent too, and this one is easy to get
//     wrong because nothing about the numbers looks unusual. On SimX the
//     counter RESTARTS AT ZERO for every launch: ProcessorImpl::run() opens
//     with reset(), which assigns a fresh PerfStats, and MCYCLE reads
//     PerfStats::cycles (sim/simx/processor.cpp, sim/simx/csr_unit.cpp). Two
//     launches therefore report overlapping small numbers, and a span taken
//     across them is not a duration of anything -- it is a maximum over
//     several unrelated clocks.
//
//     tests/bench/block_cycles.cpp did exactly that. It gave each of
//     attention's four launches its own region of one probe buffer and
//     summarised the whole buffer, believing the result was the cost of
//     attention. It was not, and the error was not small: the kernel-selection
//     rule in src/libs/grxblas/grxblas.cpp was reverted on the strength of a
//     27.6% "regression" read off that number.
//
//     `maxLive` below is what catches it. See the note on that field.
//   * A cycle count from a simulator is the cycle count of the MODEL, which is
//     the point: it is what the design does, not what the simulator's host did.

#ifndef GRX_CYCLES_H
#define GRX_CYCLES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One per warp. The kernel writes it; nothing else does.
typedef struct {
  uint64_t start;   // cycle counter when the probe was constructed
  uint64_t end;     // ... when finish() was called
  uint32_t core;    // which SM produced them; counters are per core
  uint32_t warp;    // warp index within its block, for attribution
} grxCycleSlot;

typedef struct {
  int      warps;        // slots actually written
  int      cores;        // distinct cores that wrote one
  int      spanIsValid;  // 0 when the warps did not share a core
  uint64_t span;         // last end minus first start, in cycles; 0 if invalid
  uint64_t busyMin;      // shortest per-warp end - start
  uint64_t busyMedian;
  uint64_t busyMax;      // longest
  // The most slots whose [start, end) intervals overlap at any one instant.
  //
  // THIS IS THE CROSS-LAUNCH DETECTOR, and it is here because there is no
  // other one. A slot carries no launch identity -- the kernel writes start,
  // end, core and warp, and none of those distinguishes two launches whose
  // counters both began at zero. What DOES distinguish them is a number the
  // machine cannot produce: a device holds
  //
  //     maxWarpsPerMultiProcessor * multiProcessorCount
  //
  // warps at once, and no more. A buffer reporting more live than that did not
  // come from one launch, whatever its timestamps suggest. On the configuration
  // this was found on -- 1 SM, 16 warp slots -- attention's four-launch buffer
  // reported 64.
  //
  // Compared here rather than acted on here: this header knows nothing about
  // the device, and reaching for grxGetDeviceProperties from an inline in a
  // header the DEVICE also parses is a worse trade than making the caller ask.
  // grxCycleSummarize therefore reports maxLive and leaves spanIsValid alone;
  // the caller that owns a grxDeviceProp_t is the one that can refuse.
  int      maxLive;

  // THE CROSS-CORE EVIDENCE. See the "ACROSS cores" note at the top of this
  // file, which used to say a cross-core span "means nothing" and was wrong.
  //
  // coreSkew is the spread of per-core FIRST STARTS: the largest minus the
  // smallest, over the cores that wrote a slot. It is the whole question. If
  // each core's counter began when that core got work, a core that started
  // later reads a smaller number and the skew is on the order of the span. If
  // the cores were reset together and tick together, they agree on the origin
  // and the skew is small.
  //
  // Measured aligned on simx and on rtlsim (see the table at the top of this
  // file). That is a fact about a backend, not about the architecture, so it is
  // reported rather than assumed and silicon has to answer for itself.
  //
  // Note what skew is NOT: it is not clock error. On a backend whose counters
  // share an origin it is how far apart the cores actually started, which is
  // dispatch ramp and is part of what the stage costs.
  uint64_t coreSkew;
  int      spanCrossesCores;   // 1 when more than one core wrote a slot

  // Last end minus first start over ALL cores, always computed. On a backend
  // whose per-core counters share an origin this is the stage's wall clock,
  // dispatch ramp included. On one whose counters do not, it is meaningless --
  // which is why it is a separate field from `span` rather than a widening of
  // it: nothing that reads `span` today starts believing a cross-core number
  // by accident.
  uint64_t crossCoreSpan;

  // The earliest MCYCLE any warp of this launch read. Since the counter is
  // zeroed at the launch, this IS the launch preamble measured on the device:
  // reset to the first warp reaching its probe.
  //
  // It is reported because every span in this file EXCLUDES it. A span runs
  // from the first warp starting to the last warp finishing, so a caller that
  // adds up stage spans and calls the total "what the workload costs" has
  // silently dropped one preamble per launch. On a workload of many small
  // launches that is not a rounding error.
  uint64_t firstStart;
} grxCycleSummary;



#ifdef __cplusplus
}  // extern "C"
#endif

// Host-side summary. Header-only and host-only: device code has no allocator
// and no business sorting anything.
#if defined(__cplusplus) && !defined(__VORTEX__)

#include <algorithm>
#include <utility>
#include <vector>

inline void grxCycleSummarize(const grxCycleSlot* slots, int n,
                              grxCycleSummary* out) {
  if (!out) return;
  *out = grxCycleSummary{};
  if (!slots || n <= 0) return;

  std::vector<uint64_t> busy;
  // (cycle, +1 on a start, -1 on an end), swept below for maxLive. Ends sort
  // before starts at equal cycles so that a warp finishing exactly as another
  // begins is not counted as two live at once.
  std::vector<std::pair<uint64_t, int> > edges;
  // Per core: its own earliest start. The spread of these is the evidence for
  // or against the cores sharing a clock, and it is the only thing here that
  // cannot be recovered from the aggregate.
  std::vector<std::pair<uint32_t, uint64_t> > core_first;
  uint64_t first_start = UINT64_MAX, last_end = 0;

  for (int i = 0; i < n; ++i) {
    // A slot no warp reached stays zero. Counting it would drag the span back
    // to cycle 0 and report a kernel that ran since the machine booted.
    if (slots[i].end == 0 && slots[i].start == 0) continue;

    bool found = false;
    for (size_t c = 0; c < core_first.size(); ++c) {
      if (core_first[c].first == slots[i].core) {
        core_first[c].second = std::min(core_first[c].second, slots[i].start);
        found = true;
        break;
      }
    }
    if (!found)
      core_first.push_back(std::make_pair(slots[i].core, slots[i].start));

    first_start = std::min(first_start, slots[i].start);
    last_end    = std::max(last_end, slots[i].end);
    busy.push_back(slots[i].end - slots[i].start);
    edges.push_back(std::make_pair(slots[i].start, 1));
    edges.push_back(std::make_pair(slots[i].end, -1));
  }
  if (busy.empty()) return;

  uint64_t skew_lo = UINT64_MAX, skew_hi = 0;
  for (size_t c = 0; c < core_first.size(); ++c) {
    skew_lo = std::min(skew_lo, core_first[c].second);
    skew_hi = std::max(skew_hi, core_first[c].second);
  }
  const uint64_t skew = skew_hi - skew_lo;
  const uint64_t combined = last_end - first_start;
  const bool one_core = (core_first.size() == 1);

  std::sort(edges.begin(), edges.end());
  int live = 0, max_live = 0;
  for (size_t e = 0; e < edges.size(); ++e) {
    live += edges[e].second;
    if (live > max_live) max_live = live;
  }

  std::sort(busy.begin(), busy.end());
  out->warps            = (int)busy.size();
  out->cores            = (int)core_first.size();
  out->spanIsValid      = one_core ? 1 : 0;
  out->span             = one_core ? combined : 0;
  out->busyMin          = busy.front();
  out->busyMedian       = busy[busy.size() / 2];
  out->busyMax          = busy.back();
  out->maxLive          = max_live;
  out->coreSkew         = skew;
  out->spanCrossesCores = one_core ? 0 : 1;
  out->crossCoreSpan    = combined;
  out->firstStart       = first_start;
}

#endif  // __cplusplus && !__VORTEX__

#endif  // GRX_CYCLES_H
