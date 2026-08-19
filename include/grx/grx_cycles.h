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
//   * Within one core, differences are real cycle counts.
//   * ACROSS cores the counters are independent and a span computed from two
//     different cores means nothing. grxCycleSummarize refuses to produce one
//     and says why -- it does not quietly return a plausible number.
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
} grxCycleSummary;

#ifdef __cplusplus
}  // extern "C"
#endif

// Host-side summary. Header-only and host-only: device code has no allocator
// and no business sorting anything.
#if defined(__cplusplus) && !defined(__VORTEX__)

#include <algorithm>
#include <vector>

inline void grxCycleSummarize(const grxCycleSlot* slots, int n,
                              grxCycleSummary* out) {
  if (!out) return;
  *out = grxCycleSummary{};
  if (!slots || n <= 0) return;

  std::vector<uint64_t> busy;
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
  }
  if (busy.empty()) return;

  std::sort(busy.begin(), busy.end());
  out->warps       = (int)busy.size();
  out->cores       = one_core ? 1 : 2;   // "more than one" is all that matters
  out->spanIsValid = one_core ? 1 : 0;
  out->span        = one_core ? (last_end - first_start) : 0;
  out->busyMin     = busy.front();
  out->busyMedian  = busy[busy.size() / 2];
  out->busyMax     = busy.back();
}

#endif  // __cplusplus && !__VORTEX__

#endif  // GRX_CYCLES_H
