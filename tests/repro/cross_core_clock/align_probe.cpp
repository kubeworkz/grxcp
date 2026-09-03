// Do the per-core MCYCLE counters share a time origin?
//
// grx_cycles.h states that "across cores the counters are independent and a
// span computed from two different cores means nothing", and grxCycleSummarize
// refuses such a span. That claim was never measured -- it was inferred from
// MCYCLE being per-core storage. Storage being per core does not make TIME per
// core.
//
// This asks the device. One instrumented grxDNN launch wide enough to spread
// warps over every SM, then the RAW slots: which core each warp ran on, and
// what it read at entry and exit. If the cores were started together and tick
// together, their entry readings cluster; if each counter began whenever its
// core got work, they do not.
//
// Prints the per-core extent so the answer is visible rather than asserted.

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

  // The question this program exists to answer, stated as a verdict rather
  // than left to the reader. A skew that is a small fraction of the span is
  // consistent with one shared origin; a skew comparable to the span is what
  // independent per-core clocks would produce.
  if (per_core.size() < 2) {
    std::printf("\nONE CORE -- says nothing about alignment. Raise NUM_CORES.\n");
  } else if (skew * 20 < combined) {
    std::printf("\nCOUNTERS LOOK ALIGNED: entry skew is under 5%% of the span.\n"
                "A combined cross-core span is meaningful on THIS backend.\n");
  } else {
    std::printf("\nCOUNTERS LOOK INDEPENDENT: entry skew is not small against\n"
                "the span. Cross-core spans must stay refused.\n");
  }

  grxdnnSetCycleProbe(dh, nullptr, 0);
  grxdnnDestroy(dh);
  return 0;
}
