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
//   * ACROSS cores the counters are independent and a span computed from two
//     different cores means nothing. grxCycleSummarize refuses to produce one
//     and says why -- it does not quietly return a plausible number.
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
  uint64_t span;         // last end minus first start, in cycles
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
  uint64_t first_start = UINT64_MAX, last_end = 0;
  uint32_t core = 0;
  bool     one_core = true, seen = false;

  for (int i = 0; i < n; ++i) {
    // A slot no warp reached stays zero. Counting it would drag the span back
    // to cycle 0 and report a kernel that ran since the machine booted.
    if (slots[i].end == 0 && slots[i].start == 0) continue;
    if (!seen) { core = slots[i].core; seen = true; }
    else if (slots[i].core != core) one_core = false;

    first_start = std::min(first_start, slots[i].start);
    last_end    = std::max(last_end, slots[i].end);
    busy.push_back(slots[i].end - slots[i].start);
    edges.push_back(std::make_pair(slots[i].start, 1));
    edges.push_back(std::make_pair(slots[i].end, -1));
  }
  if (busy.empty()) return;

  std::sort(edges.begin(), edges.end());
  int live = 0, max_live = 0;
  for (size_t e = 0; e < edges.size(); ++e) {
    live += edges[e].second;
    if (live > max_live) max_live = live;
  }

  std::sort(busy.begin(), busy.end());
  out->warps       = (int)busy.size();
  out->cores       = one_core ? 1 : 2;   // "more than one" is all that matters
  out->spanIsValid = one_core ? 1 : 0;
  out->span        = one_core ? (last_end - first_start) : 0;
  out->busyMin     = busy.front();
  out->busyMedian  = busy[busy.size() / 2];
  out->busyMax     = busy.back();
  out->maxLive     = max_live;
}

#endif  // __cplusplus && !__VORTEX__

#endif  // GRX_CYCLES_H
