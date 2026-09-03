// Do the per-core MCYCLE counters share a time origin?
//
// grx_cycles.h used to say "across cores the counters are independent and a
// span computed from two different cores means nothing", and refused such a
// span. That was never measured -- it was inferred from MCYCLE being per-core
// STORAGE, and storage being per core does not make TIME per core.
//
// THE TEST, AND WHY THE OBVIOUS ONE IS BACKWARDS. The first version of this
// file concluded "small skew between per-core first reads => aligned". That is
// the wrong way round and it reached the right answer by luck:
//
//   * If each core's counter were reset when THAT CORE got work, every core's
//     first warp would read a SMALL number -- the few cycles between its own
//     reset and its own probe. All cores would agree, and the skew would be
//     tiny. Small skew is what INDEPENDENT counters look like.
//   * If the counters are reset together at the launch and tick together, each
//     core's first read carries the whole launch preamble, and the readings are
//     LARGE. Any spread between them is genuine dispatch stagger, not clock
//     error.
//
// So the discriminator is the MAGNITUDE of the first reads, against the same
// figure from a ONE-CORE run of the same shape. One core establishes what the
// preamble costs. If the N-core first reads are of that order, the counters
// have been running since a common origin. If they collapse toward zero, each
// core started its own clock.
//
// One run cannot settle it. Pass the one-core baseline as argv[3] to get a
// verdict; without it this prints the readings and says what to compare.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <vector>

#include "grx/grx_runtime.h"
#include "grx/grxdnn.h"
#include "grx/grx_cycles.h"

int main(int argc, char** argv) {
  const int rows = (argc > 1) ? std::atoi(argv[1]) : 64;
  const int cols = (argc > 2) ? std::atoi(argv[2]) : 16;
  // The one-core first read for this same shape and backend. 0 = not supplied.
  const uint64_t baseline = (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 0;

  grxDeviceProp_t p{};
  if (grxGetDeviceProperties(&p, 0) != grxSuccess) {
    std::printf("no device\n");
    return 1;
  }
  std::printf("%s: %d SMs, %d warps/SM, warp %d\n", p.name,
              p.multiProcessorCount, p.maxWarpsPerMultiProcessor, p.warpSize);

  grxdnnHandle_t dh = nullptr;
  if (grxdnnCreate(&dh) != GRXDNN_STATUS_SUCCESS) { std::printf("no grxdnn\n"); return 1; }

  const int cap = grxdnnCycleSlotsNeeded(dh, rows);
  if (cap <= 0) { std::printf("no slots\n"); return 1; }

  void* dslots = nullptr;
  if (grxMalloc(&dslots, (size_t)cap * sizeof(grxCycleSlot)) != grxSuccess) return 1;
  grxMemset(dslots, 0, (size_t)cap * sizeof(grxCycleSlot));
  if (grxdnnSetCycleProbe(dh, (grxCycleSlot*)dslots, cap) != GRXDNN_STATUS_SUCCESS) return 1;

  // A layernorm is one warp per row, so `rows` sets the warp count directly.
  void *x = nullptr, *y = nullptr, *g = nullptr, *b = nullptr;
  const size_t n = (size_t)rows * cols;
  if (grxMalloc(&x, n * 4) != grxSuccess) return 1;
  if (grxMalloc(&y, n * 4) != grxSuccess) return 1;
  if (grxMalloc(&g, (size_t)cols * 4) != grxSuccess) return 1;
  if (grxMalloc(&b, (size_t)cols * 4) != grxSuccess) return 1;
  {
    std::vector<float> h(n, 1.0f), hg((size_t)cols, 1.0f), hb((size_t)cols, 0.0f);
    for (size_t i = 0; i < n; ++i) h[i] = (float)((i % 13) + 1);
    grxMemcpy(x, h.data(), n * 4, grxMemcpyDefault);
    grxMemcpy(g, hg.data(), (size_t)cols * 4, grxMemcpyDefault);
    grxMemcpy(b, hb.data(), (size_t)cols * 4, grxMemcpyDefault);
  }

  if (grxdnnLayerNormForward(dh, rows, cols, (const float*)x, cols,
                             (const float*)g, (const float*)b, 1e-5f,
                             (float*)y, cols) != GRXDNN_STATUS_SUCCESS) {
    std::printf("layernorm failed\n");
    return 1;
  }
  grxDeviceSynchronize();

  std::vector<grxCycleSlot> h((size_t)cap);
  grxMemcpy(h.data(), dslots, (size_t)cap * sizeof(grxCycleSlot), grxMemcpyDefault);

  // Group by core. Unwritten slots are all-zero and are skipped.
  struct Ext { uint64_t lo = ~0ull, hi = 0; int warps = 0; };
  std::map<uint32_t, Ext> per_core;
  for (const grxCycleSlot& s : h) {
    if (s.start == 0 && s.end == 0) continue;
    Ext& e = per_core[s.core];
    e.lo = std::min(e.lo, s.start);
    e.hi = std::max(e.hi, s.end);
    ++e.warps;
  }
  if (per_core.empty()) { std::printf("no slots written\n"); return 1; }

  std::printf("\n%-6s %8s %14s %14s %14s\n", "core", "warps", "first start",
              "last end", "span");
  uint64_t gmin_start = ~0ull, gmax_start = 0, gmax_end = 0;
  for (const auto& kv : per_core) {
    std::printf("%-6u %8d %14llu %14llu %14llu\n", kv.first, kv.second.warps,
                (unsigned long long)kv.second.lo,
                (unsigned long long)kv.second.hi,
                (unsigned long long)(kv.second.hi - kv.second.lo));
    gmin_start = std::min(gmin_start, kv.second.lo);
    gmax_start = std::max(gmax_start, kv.second.lo);
    gmax_end   = std::max(gmax_end, kv.second.hi);
  }

  const uint64_t skew = gmax_start - gmin_start;
  const uint64_t combined = gmax_end - gmin_start;
  std::printf("\ncores            %zu\n", per_core.size());
  std::printf("entry skew       %llu cycles  (spread of per-core first start)\n",
              (unsigned long long)skew);
  std::printf("combined span    %llu cycles  (max end - min start)\n",
              (unsigned long long)combined);
  if (combined)
    std::printf("skew / span      %.4f\n", (double)skew / (double)combined);

  if (per_core.size() < 2) {
    std::printf("\nONE CORE. This run cannot say anything about alignment, but\n"
                "it IS the baseline the multi-core run needs. Pass %llu as the\n"
                "third argument to a run with more cores.\n",
                (unsigned long long)gmin_start);
    grxdnnSetCycleProbe(dh, nullptr, 0);
    grxdnnDestroy(dh);
    return 0;
  }

  if (baseline == 0) {
    std::printf("\nNO BASELINE GIVEN, so no verdict. Run this same shape on the\n"
                "same backend at NUM_CORES=1 and pass its first read as argv[3].\n"
                "Aligned counters put these readings at that order of magnitude;\n"
                "independent ones collapse them toward zero.\n");
  } else {
    // Aligned: every core carries the preamble, so the smallest first read is
    // of the baseline's order. Independent: each core reads from its own reset,
    // so the readings fall far below it. A half-baseline floor separates the
    // two by a wide margin -- this is not a tuned threshold, the predictions
    // differ by orders of magnitude.
    std::printf("\none-core baseline %llu\n", (unsigned long long)baseline);
    if (gmin_start * 2 >= baseline) {
      std::printf("\nCOUNTERS SHARE AN ORIGIN. The smallest first read (%llu) is\n"
                  "of the one-core preamble's order (%llu), so every core's\n"
                  "counter has been running since the launch. The %llu-cycle\n"
                  "spread is dispatch stagger, and it is part of what a stage\n"
                  "costs -- a cross-core span is a real duration on this backend.\n",
                  (unsigned long long)gmin_start, (unsigned long long)baseline,
                  (unsigned long long)skew);
    } else {
      std::printf("\nCOUNTERS ARE INDEPENDENT. The smallest first read (%llu) has\n"
                  "collapsed far below the one-core preamble (%llu): each core is\n"
                  "counting from its own reset. Cross-core spans mean nothing on\n"
                  "this backend and grxCycleSummarize's `span` must stay refused.\n",
                  (unsigned long long)gmin_start, (unsigned long long)baseline);
    }
  }

  grxdnnSetCycleProbe(dh, nullptr, 0);
  grxdnnDestroy(dh);
  return 0;
}
