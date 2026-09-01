// grxBLAS: what a matrix-vector product costs, in device cycles.
//
// WHY THIS EXISTS. The hot-loop census (ci/check_kernel_loops.py) ranked sgemv
// as the most expensive SHIPPING kernel per unit of float work in the image:
// fifteen instructions and two loads to produce one multiply-add. That is an
// instruction count, and an instruction count is a PREDICTION. This is the
// measurement -- the same distinction the rest of the tuning work has kept, and
// the reason none of it is quoted from a disassembly alone.
//
// WHAT IS MEASURED. The span: first warp starting to last warp finishing, from
// the device's own cycle counter through grxblasSetCycleProbe. Not events,
// which on a simulator measure the simulator (cuda_mapping.md 7.4).
//
// BOTH TRAVERSALS, because sgemv is two kernels wearing one name. Untransposed
// is one thread per output row; transposed is one WARP per output column with a
// reduction across lanes. They have different shapes, different costs, and a
// change can help one and hurt the other -- so neither is reported alone.
//
// THE SPAN IS REFUSED WHEN IT WOULD BE A LIE. MCYCLE restarts at zero at every
// launch on SimX, so a buffer holding slots from two launches yields a span
// that is a maximum over unrelated clocks. maxLive catches exactly that: a
// device holds maxWarpsPerMultiProcessor * multiProcessorCount warps and no
// more, and a buffer claiming more live than that came from more than one
// launch. Every measurement here is one launch, and the check says so rather
// than assuming it.
//
// WHAT THE NUMBERS ARE NOT. This is a one-SM, four-lane SimX configuration.
// These cycles are what the MODEL does. They are the right thing to compare two
// kernels against and the wrong thing to quote as hardware performance.

#include <grx/grx.h>
#include <grx/grxblas.h>
#include <grx/grx_cycles.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++failures;
}

struct Point {
  const char* trans;
  int      m, n;
  uint64_t span;
  uint64_t busy_median;
  int      warps;
};

// One launch with the probe attached, summarised, with the cross-launch check
// applied by the caller that owns the device properties.
bool measure(grxblasHandle_t h, void* dSlots, int nslots,
             const grxDeviceProp_t& prop, const char* what,
             const std::function<grxblasStatus_t()>& call,
             grxCycleSummary* out) {
  grxCycleSummary sum{};
  if (grxMemset(dSlots, 0, (size_t)nslots * sizeof(grxCycleSlot)) != grxSuccess)
    return false;
  grxblasSetCycleProbe(h, (grxCycleSlot*)dSlots, nslots);
  const grxblasStatus_t st = call();
  grxDeviceSynchronize();
  grxblasSetCycleProbe(h, nullptr, 0);
  if (st != GRXBLAS_STATUS_SUCCESS) {
    std::printf("  %s failed: %s\n", what, grxblasGetStatusString(st));
    return false;
  }
  std::vector<grxCycleSlot> slots(nslots);
  if (grxMemcpy(slots.data(), dSlots, slots.size() * sizeof(grxCycleSlot),
                grxMemcpyDefault) != grxSuccess)
    return false;
  grxCycleSummarize(slots.data(), nslots, &sum);

  const int occupancy = prop.maxWarpsPerMultiProcessor * prop.multiProcessorCount;
  if (sum.maxLive > occupancy) {
    std::printf("  %s: %d warps live at once on a device that holds %d --\n"
                "        this buffer spans more than one launch and its span is"
                " not a duration.\n", what, sum.maxLive, occupancy);
    return false;
  }
  if (!sum.spanIsValid) {
    std::printf("  %s: the warps did not share a core; the span is not a"
                " duration.\n", what);
    return false;
  }
  *out = sum;
  return true;
}

void fill(std::vector<float>& v, unsigned seed) {
  for (size_t i = 0; i < v.size(); ++i) {
    seed = seed * 1664525u + 1013904223u;
    v[i] = (float)((int)(seed >> 16) % 17 - 8) * 0.25f;
  }
}

}  // namespace

int main(int argc, char** argv) {
  // --out, spelled the same way tests/bench/gemm_cycles.cpp spells it, so the
  // two bench steps in ci/run_real.sh read alike.
  const char* out_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--out" && i + 1 < argc) out_path = argv[++i];
  }

  grxDeviceProp_t prop{};
  if (grxGetDeviceProperties(&prop, 0) != grxSuccess) {
    std::printf("  no device\n");
    return 77;
  }
  grxblasHandle_t h = nullptr;
  if (grxblasCreate(&h) != GRXBLAS_STATUS_SUCCESS) {
    std::printf("  grxblasCreate failed\n");
    return 77;
  }

  std::printf("device: %s, %d SM x %d warps x %d lanes\n", prop.name,
              prop.multiProcessorCount, prop.maxWarpsPerMultiProcessor,
              prop.warpSize);

  // Enough slots for the widest launch below. One slot per BLOCK.
  const int nslots = 256;
  void* dSlots = nullptr;
  if (grxMalloc(&dSlots, (size_t)nslots * sizeof(grxCycleSlot)) != grxSuccess) {
    std::printf("  could not allocate the probe buffer\n");
    grxblasDestroy(h);
    return 77;
  }

  struct Shape { int m, n; };
  // Square-ish and both lopsided ways, because the two traversals divide the
  // work along different axes: a tall A gives OP_N many threads and OP_T a long
  // reduction, and a wide A does the reverse.
  const Shape shapes[] = {{64, 64}, {256, 16}, {16, 256}};

  std::vector<Point> points;
  const float alpha = 2.0f, beta = 3.0f;

  for (int t = 0; t < 2; ++t) {
    const bool tr = (t == 1);
    const grxblasOperation_t op = tr ? GRXBLAS_OP_T : GRXBLAS_OP_N;
    for (const Shape& s : shapes) {
      const int rows  = tr ? s.n : s.m;   // outputs
      const int depth = tr ? s.m : s.n;   // reduction length

      std::vector<float> A((size_t)s.m * (size_t)s.n);
      std::vector<float> x((size_t)depth), y((size_t)rows);
      fill(A, 11u); fill(x, 29u); fill(y, 47u);

      void *dA = nullptr, *dx = nullptr, *dy = nullptr;
      bool alloc_ok =
          grxMalloc(&dA, A.size() * sizeof(float)) == grxSuccess &&
          grxMalloc(&dx, x.size() * sizeof(float)) == grxSuccess &&
          grxMalloc(&dy, y.size() * sizeof(float)) == grxSuccess;
      if (alloc_ok) {
        alloc_ok =
            grxMemcpy(dA, A.data(), A.size() * sizeof(float), grxMemcpyDefault) == grxSuccess &&
            grxMemcpy(dx, x.data(), x.size() * sizeof(float), grxMemcpyDefault) == grxSuccess &&
            grxMemcpy(dy, y.data(), y.size() * sizeof(float), grxMemcpyDefault) == grxSuccess;
      }
      char what[96];
      std::snprintf(what, sizeof(what), "sgemv %s %dx%d", tr ? "T" : "N", s.m, s.n);
      if (!alloc_ok) { expect(false, what); continue; }

      grxCycleSummary sum{};
      const bool ok = measure(h, dSlots, nslots, prop, what,
                              [&] {
                                return grxblasSgemv(h, op, s.m, s.n, &alpha, dA,
                                                    s.m, dx, 1, &beta, dy, 1);
                              },
                              &sum);
      if (ok) {
        points.push_back({tr ? "T" : "N", s.m, s.n, sum.span, sum.busyMedian,
                          sum.warps});
        std::printf("  %-16s span %8llu  busy median %8llu  warps %3d  "
                    "%.1f cycles/output\n",
                    what, (unsigned long long)sum.span,
                    (unsigned long long)sum.busyMedian, sum.warps,
                    (double)sum.span / (double)rows);
      }
      expect(ok, what);
      grxFree(dA); grxFree(dx); grxFree(dy);
    }
  }

  // Raw integers only, like the other benches: cycles per output derives from
  // the span and the shape, so ci/check_perf.py computes it rather than storing
  // a quotient that would need a tolerance to survive its own rounding.
  if (out_path && !points.empty()) {
    std::FILE* f = std::fopen(out_path, "w");
    if (!f) {
      std::printf("  could not write %s\n", out_path);
      ++failures;
    } else {
      std::fprintf(f, "{\n  \"bench\": \"gemv_cycles\",\n");
      std::fprintf(f, "  \"device\": {\n    \"name\": \"%s\",\n"
                      "    \"sms\": %d,\n    \"warp\": %d,\n"
                      "    \"warp_slots\": %d\n  },\n",
                   prop.name, prop.multiProcessorCount, prop.warpSize,
                   prop.maxWarpsPerMultiProcessor);
      std::fprintf(f, "  \"points\": [\n");
      for (size_t i = 0; i < points.size(); ++i) {
        const Point& p = points[i];
        std::fprintf(f, "    {\"trans\": \"%s\", \"m\": %d, \"n\": %d,"
                        " \"span\": %llu, \"busy_median\": %llu,"
                        " \"warps\": %d}%s\n",
                     p.trans, p.m, p.n, (unsigned long long)p.span,
                     (unsigned long long)p.busy_median, p.warps,
                     (i + 1 == points.size()) ? "" : ",");
      }
      std::fprintf(f, "  ]\n}\n");
      std::fclose(f);
      std::printf("  wrote %s\n", out_path);
    }
  }

  grxFree(dSlots);
  grxblasDestroy(h);
  std::printf("\n%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
