// grxBLAS: what a GEMM costs, in device cycles, scalar against tensor.
//
// This is the baseline the tuned tensor-core kernel has to beat, and the
// number the phase 3 exit gate is stated against. It is measured with the
// device's own cycle counter through grxblasSetCycleProbe -- not with events,
// which on a simulator measure the simulator (cuda_mapping.md section 7.4).
//
// WHAT THE NUMBERS MEAN, AND WHAT THEY DO NOT
//
// The device here is a one-SM, four-lane SimX configuration. A cycle count
// from it is what the MODEL does, which is the right thing to compare kernels
// against and the wrong thing to quote as hardware performance. Scaling these
// figures to a 128-SM flagship by multiplying is exactly the arithmetic this
// file exists to make unnecessary.
//
// The headline is the SPAN -- first warp starting to last warp finishing --
// divided by output elements, because that is what "faster" means. The
// per-warp median sits beside it: when the two diverge, warps are waiting on
// each other, and a kernel can be improved by fixing that alone.
//
// The two kernels do not compute the same thing: sgemm takes fp32 inputs and
// GemmEx takes fp16 with an fp32 accumulator. The comparison is of COST per
// output element at the same shape, which is what the exit gate is stated
// against -- not of numerical equivalence, which they do not have.
//
// The gate is checked here: GemmEx must cost no more than a fifth of sgemm per
// output element. It is a real threshold now that there are two kernels.

#include <grx/grx.h>
#include <grx/grxblas.h>
#include <grx/grx_cycles.h>

#include "../common/fp16.h"

#include <cstdio>
#include <functional>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++failures;
}

struct Point {
  int      m, n, k;
  uint64_t span;
  double   per_element;      // headline: whole-kernel cycles / output element
  double   per_mac;
  double   warp_per_element; // secondary: summed warp windows / output element
  double   tensor_per_element;   // the same, for the tensor kernel
  // And the same for sgemm forced to the REFERENCE kernel. Measured because
  // the phase 3 gate is a RATIO, and a ratio moves when either side does: this
  // is the fixed point against which "the tensor path did not regress, its
  // denominator improved" is a statement of fact rather than a claim.
  double   ref_per_element;
  uint64_t tensor_span;
  // Kept so the baseline can hold the raw pair warp_per_element derives from
  // rather than the quotient.
  int      nslots = 0;
  uint64_t busy_median = 0;
};

// Runs a launch with the probe attached and summarises what came back.
grxCycleSummary measure(grxblasHandle_t h, void* dSlots, int nslots,
                        bool* ok, const char* what,
                        const std::function<grxblasStatus_t()>& call) {
  grxCycleSummary sum{};
  *ok = false;
  if (grxMemset(dSlots, 0, (size_t)nslots * sizeof(grxCycleSlot)) != grxSuccess)
    return sum;
  grxblasSetCycleProbe(h, (grxCycleSlot*)dSlots, nslots);
  const grxblasStatus_t st = call();
  grxDeviceSynchronize();
  grxblasSetCycleProbe(h, nullptr, 0);
  if (st != GRXBLAS_STATUS_SUCCESS) {
    std::printf("  %s failed: %s\n", what, grxblasGetStatusString(st));
    return sum;
  }
  std::vector<grxCycleSlot> slots(nslots);
  if (grxMemcpy(slots.data(), dSlots, slots.size() * sizeof(grxCycleSlot),
                grxMemcpyDefault) != grxSuccess)
    return sum;
  grxCycleSummarize(slots.data(), nslots, &sum);
  *ok = true;
  return sum;
}

// The machine-readable half, for ci/perf/baselines/. Raw integers only --
// spans and the median busy window. Cycles per element, per MAC and the
// GemmEx speedup all derive from these and from the shape, so ci/check_perf.py
// computes them rather than storing them; a stored 104.6 would need a
// tolerance to survive its own rounding, and a stored 33512 does not.
void write_json(const char* path, const grxDeviceProp_t& prop, bool have_tensor,
                int tm, int tn, int tk, const std::vector<Point>& points) {
  std::FILE* f = std::fopen(path, "w");
  if (!f) {
    std::printf("  could not write %s\n", path);
    ++failures;
    return;
  }
  std::fprintf(f, "{\n  \"bench\": \"gemm_cycles\",\n");
  std::fprintf(f,
               "  \"device\": {\"name\": \"%s\", \"sms\": %d, \"warp\": %d, "
               "\"mhz\": %d, \"warp_slots\": %d},\n",
               prop.name, prop.multiProcessorCount, prop.warpSize,
               prop.clockRateMHz,
               prop.warpSize ? prop.maxThreadsPerBlock / prop.warpSize : 0);
  if (have_tensor)
    std::fprintf(f, "  \"tensor_tile\": {\"m\": %d, \"n\": %d, \"k\": %d},\n",
                 tm, tn, tk);
  else
    std::fprintf(f, "  \"tensor_tile\": null,\n");
  std::fprintf(f, "  \"shapes\": [\n");
  for (size_t i = 0; i < points.size(); ++i) {
    const Point& p = points[i];
    std::fprintf(f,
                 "    {\"m\": %d, \"n\": %d, \"k\": %d, \"sgemm_span\": %llu, "
                 "\"tensor_span\": %llu, \"nslots\": %d, \"busy_median\": %llu}%s\n",
                 p.m, p.n, p.k, (unsigned long long)p.span,
                 (unsigned long long)p.tensor_span, p.nslots,
                 (unsigned long long)p.busy_median,
                 i + 1 == points.size() ? "" : ",");
  }
  std::fprintf(f, "  ]\n}\n");
  std::fclose(f);
  std::printf("  wrote %s\n", path);
}

}  // namespace

int main(int argc, char** argv) {
  const char* out_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    } else {
      std::printf("usage: gemm_cycles [--out <results.json>]\n");
      return 2;
    }
  }

  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count <= 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }
  if (grxSetDevice(0) != grxSuccess) return 1;

  grxDeviceProp_t prop{};
  grxGetDeviceProperties(&prop, 0);

  grxblasHandle_t h = nullptr;
  if (grxblasCreate(&h) != GRXBLAS_STATUS_SUCCESS) return 1;

  const float one = 1.0f, zero = 0.0f;
  std::printf("%s: %d SMs, warp %d, %d MHz\n", prop.name,
              prop.multiProcessorCount, prop.warpSize, prop.clockRateMHz);

  int tm = 0, tn = 0, tk = 0;
  const bool have_tensor =
      (grxblasGetTensorTile(h, &tm, &tn, &tk) == GRXBLAS_STATUS_SUCCESS);
  if (have_tensor)
    std::printf("sgemm v0: one thread per output element, no blocking.  "
                "GemmEx: %dx%dx%d tensor tiles, DMA-staged, single buffered\n",
                tm, tn, tk);
  else
    std::printf("sgemm v0: one thread per output element, no blocking.  "
                "GemmEx: unavailable on this device\n");

  const struct { int m, n, k; } shapes[] = {
    { 16, 16,  16},
    { 16, 16,  32},
    { 16, 16,  64},
    { 32, 32,  32},
    // A second shape that gives the tensor path more tiles than the core has
    // warp slots, so the exit-gate threshold does not stand on a single point.
    // Kept to two: the scalar baseline is one thread per output element and
    // simx charges for every one of them, so a 64x32 shape alone takes longer
    // than the rest of the suite put together. See the tile-starvation note at
    // the gate itself.
    { 32, 32,  64},
  };

  std::printf("%-14s %14s %14s %10s\n", "shape",
              "sgemm cyc/elem", "GemmEx cyc/elem", "speedup");

  std::vector<Point> points;
  for (const auto& s : shapes) {
    const double elems = (double)s.m * s.n;

    // fp32 operands for sgemm, the same numbers as fp16 for GemmEx. Values are
    // irrelevant to the timing; ones keep the comparison free of denormals.
    std::vector<float>    A32((size_t)s.m * s.k, 1.0f), B32((size_t)s.k * s.n, 1.0f);
    std::vector<uint16_t> A16(A32.size(), grxtest::float_to_half(1.0f));
    std::vector<uint16_t> B16(B32.size(), grxtest::float_to_half(1.0f));

    void *dA32 = nullptr, *dB32 = nullptr, *dA16 = nullptr, *dB16 = nullptr,
         *dC = nullptr, *dSlots = nullptr;
    const int nslots = grxblasCycleSlotsNeeded(h, s.m, s.n);
    if (nslots <= 0 ||
        grxMalloc(&dA32, A32.size() * 4) != grxSuccess ||
        grxMalloc(&dB32, B32.size() * 4) != grxSuccess ||
        grxMalloc(&dA16, A16.size() * 2) != grxSuccess ||
        grxMalloc(&dB16, B16.size() * 2) != grxSuccess ||
        grxMalloc(&dC, (size_t)s.m * s.n * 4) != grxSuccess ||
        grxMalloc(&dSlots, (size_t)nslots * sizeof(grxCycleSlot)) != grxSuccess) {
      std::printf("  allocation failed at %dx%dx%d\n", s.m, s.n, s.k);
      ++failures;
      break;
    }
    grxMemcpy(dA32, A32.data(), A32.size() * 4, grxMemcpyDefault);
    grxMemcpy(dB32, B32.data(), B32.size() * 4, grxMemcpyDefault);
    grxMemcpy(dA16, A16.data(), A16.size() * 2, grxMemcpyDefault);
    grxMemcpy(dB16, B16.data(), B16.size() * 2, grxMemcpyDefault);

    bool ok = false;
    const grxCycleSummary scalar = measure(h, dSlots, nslots, &ok, "sgemm",
        [&] { return grxblasSgemm(h, GRXBLAS_OP_N, GRXBLAS_OP_N, s.m, s.n, s.k,
                                  &one, dA32, s.m, dB32, s.k, &zero, dC, s.m); });
    if (!ok) {
      std::printf("grxblas device kernels not found; skipping\n");
      grxblasDestroy(h);
      return 77;
    }

    Point p{};
    p.m = s.m; p.n = s.n; p.k = s.k;
    p.span = scalar.spanIsValid ? scalar.span : 0;
    // The headline is the SPAN: first warp starting to last warp finishing,
    // over the elements produced. That is what "faster" means -- it charges a
    // kernel for poor parallel efficiency, which a per-warp average hides.
    p.per_element      = p.span ? (double)p.span / elems : 0.0;
    p.per_mac          = p.per_element / (double)(s.k ? s.k : 1);
    p.warp_per_element = scalar.busyMedian
                             ? (double)scalar.busyMedian * (double)nslots / elems
                             : 0.0;
    p.nslots      = nslots;
    p.busy_median = scalar.busyMedian;

    // sgemm again, forced to the reference kernel. Same operands, same probe,
    // same code path -- only the kernel selection differs, through the hook
    // the sweeps use. Not gated; it is the fixed denominator the gate's report
    // reads against.
    {
      setenv("GRXBLAS_SGEMM_NAIVE", "1", 1);
      bool rok = false;
      const grxCycleSummary ref = measure(h, dSlots, nslots, &rok, "sgemm(ref)",
          [&] { return grxblasSgemm(h, GRXBLAS_OP_N, GRXBLAS_OP_N, s.m, s.n,
                                    s.k, &one, dA32, s.m, dB32, s.k, &zero,
                                    dC, s.m); });
      unsetenv("GRXBLAS_SGEMM_NAIVE");
      if (rok && ref.spanIsValid)
        p.ref_per_element = (double)ref.span / elems;
    }

    if (have_tensor) {
      bool tok = false;
      const grxCycleSummary tensor = measure(h, dSlots, nslots, &tok, "GemmEx",
          [&] { return grxblasGemmEx(h, GRXBLAS_OP_N, GRXBLAS_OP_N, s.m, s.n,
                                     s.k, &one, dA16, GRX_R_16F, s.m,
                                     dB16, GRX_R_16F, s.k, &zero,
                                     dC, GRX_R_32F, s.m); });
      if (tok) {
        p.tensor_span = tensor.spanIsValid ? tensor.span : 0;
        p.tensor_per_element =
            p.tensor_span ? (double)p.tensor_span / elems : 0.0;
      } else {
        ++failures;
      }
    }

    points.push_back(p);
    if (p.tensor_per_element > 0.0)
      std::printf("%3dx%3dx%3d %14.1f %14.1f %9.2fx\n", s.m, s.n, s.k,
                  p.per_element, p.tensor_per_element,
                  p.per_element / p.tensor_per_element);
    else
      std::printf("%3dx%3dx%3d %14.1f %14s %10s\n", s.m, s.n, s.k,
                  p.per_element, "-", "-");

    grxFree(dA32); grxFree(dB32); grxFree(dA16); grxFree(dB16);
    grxFree(dC); grxFree(dSlots);
  }

  std::printf("is the measurement sound:\n");
  expect(points.size() == sizeof(shapes) / sizeof(shapes[0]),
         "every shape produced a measurement");
  if (points.size() >= 3) {
    // Same m and n, k doubling: cost per element must follow k, because the
    // inner loop IS k. If it does not, the probe is not measuring the loop.
    expect(points[1].per_element > points[0].per_element * 1.5 &&
           points[2].per_element > points[1].per_element * 1.5,
           "cost per element grows with k");
    const double per_mac_spread =
        points[2].per_mac / (points[0].per_mac > 0 ? points[0].per_mac : 1.0);
    std::printf("        cycles per MAC across the k sweep: %.2f -> %.2f\n",
                points[0].per_mac, points[2].per_mac);
    expect(per_mac_spread > 0.5 && per_mac_spread < 1.5,
           "cost per multiply-add is roughly constant, as an unblocked loop "
           "should be");
  }

  std::printf("\n  Baseline: %.1f cycles per output element at %dx%dx%d, on %s "
              "with %d SM\n  and a %d-lane warp. A cycle count from a simulator "
              "is what the MODEL does: the right\n  thing to compare kernels "
              "with, the wrong thing to quote as hardware.\n",
              points.empty() ? 0.0 : points[0].per_element,
              points.empty() ? 0 : points[0].m, points.empty() ? 0 : points[0].n,
              points.empty() ? 0 : points[0].k,
              prop.name, prop.multiProcessorCount, prop.warpSize);

  if (have_tensor) {
    std::printf("\nPHASE 3 EXIT GATE: GemmEx at no more than a fifth of "
                "sgemm's cycles per element\n");

    // TILE STARVATION, and why the threshold is not applied to every shape.
    //
    // The tensor kernel's parallelism is bounded by its OUTPUT TILE COUNT: it
    // runs one CTA and hands one tile to each warp, so a GEMM with fewer tiles
    // than the core has warp slots leaves slots empty no matter how wide the
    // core is. sgemm has no such bound -- one thread per output element -- so
    // it keeps absorbing warps.
    //
    // On a 4-warp core that difference is invisible: 16x16x16 is 8 tiles and 4
    // slots, so both paths saturate, and the worst speedup read 7.38x. Widening
    // the core to 16 warps doubled sgemm's throughput (304.9 -> 165.9 cycles
    // per element) and moved the tensor path far less (41.3 -> 34.6), because 8
    // tiles cannot fill 16 slots. The worst speedup fell to 4.80x with nothing
    // having got slower.
    //
    // So the threshold is applied where the comparison is between two
    // saturated kernels, and the starved shapes are REPORTED rather than
    // silently dropped -- a gate that quietly skipped its hardest case would be
    // worse than one that fails.
    //
    // The tile bound itself is a consequence of the single-CTA workaround for
    // the tensor unit's multi-CTA deadlock (tests/repro/tcu_multi_cta/). When
    // that is fixed, tiles map to CTAs and the starvation goes with it.
    const int warp_slots = prop.maxThreadsPerBlock / prop.warpSize;
    bool measured = false, met = true;
    double worst_ratio = 0.0;
    int starved = 0;
    for (const Point& p : points) {
      if (p.tensor_per_element <= 0.0 || p.per_element <= 0.0) continue;
      measured = true;
      const double speedup = p.per_element / p.tensor_per_element;

      // Tiles, from the shape the device itself reported for this build.
      const int tiles = (p.m / tm) * (p.n / tn);
      if (tiles > 0 && warp_slots > 0 && tiles < warp_slots) {
        ++starved;
        std::printf("        %3dx%3dx%3d  %.2fx  (not gated: %d tiles for %d "
                    "warp slots -- the tensor path cannot fill this core)\n",
                    p.m, p.n, p.k, speedup, tiles, warp_slots);
        continue;
      }
      if (speedup < 5.0) met = false;
      if (worst_ratio == 0.0 || speedup < worst_ratio) worst_ratio = speedup;
    }
    expect(measured, "the tensor path was measured at every shape");
    if (worst_ratio == 0.0) {
      std::printf("        every shape was tile-starved on this device; the "
                  "threshold has nothing to apply to.\n");
      expect(false, "at least one shape gives the tensor path enough tiles to "
                    "fill the core");
    } else {
      std::printf("        worst speedup among the %d shape(s) that fill the "
                  "core: %.2fx", (int)points.size() - starved, worst_ratio);
      if (starved) std::printf("   (%d starved, listed above)", starved);
      std::printf("\n");

      // The same ratio against the REFERENCE sgemm, which is the fixed point.
      // Printed always, because it is what tells a reader whether a change in
      // the gated number came from the tensor path or from its denominator.
      double worst_ref = 0.0;
      for (const Point& p : points) {
        if (p.tensor_per_element <= 0.0 || p.ref_per_element <= 0.0) continue;
        const int tiles = (p.m / tm) * (p.n / tn);
        if (tiles > 0 && warp_slots > 0 && tiles < warp_slots) continue;
        const double r = p.ref_per_element / p.tensor_per_element;
        if (worst_ref == 0.0 || r < worst_ref) worst_ref = r;
      }
      if (worst_ref > 0.0)
        std::printf("        against the REFERENCE sgemm, the same shapes: "
                    "%.2fx\n", worst_ref);

      // NOT MET, AND LEFT THAT WAY ON PURPOSE.
      //
      // This threshold was set when sgemm meant the one-thread-per-output
      // reference. It has since been beaten twice -- register blocking, then
      // the 2D micro-tile -- and the tuned SIMT kernel is now 2.32x faster than
      // the reference at these shapes. The tensor path did not move: 29.4 and
      // 44.2 cycles per element in both configurations, which is why the line
      // above is printed. The ratio fell from 5.62x to 4.30x with nothing
      // having got slower.
      //
      // Three things could have been done about that and only one of them is
      // honest without a decision on the record. Moving the threshold to 4x
      // would read as a relaxation forever. Restating the gate against the
      // reference kernel would freeze its denominator and make it unfalsifiable
      // by SIMT work. Leaving it is what AGENTS.md section 4 requires: an
      // assertion is not relaxed as a side effect of unrelated progress.
      //
      // So it FAILS, deliberately, and ci/run_real.sh defers the failure to the
      // end of the run rather than stopping there -- the same treatment the
      // PERF BASELINE GATE gets, and for the same reason: a red gate that
      // hides the twenty after it is worse than a red gate.
      //
      // What would make it pass again: the tensor unit's multi-CTA deadlock
      // (cuda_mapping.md 7.12). Two of five shapes are already excluded here
      // for tile starvation, which is a consequence of the single-CTA
      // workaround; lifting it gives the tensor path the parallelism the ratio
      // was originally measured with.
      expect(met, "GemmEx costs at most a fifth of sgemm per output element");
      if (!met)
        std::printf("        NOT MET at %.2fx, and recorded rather than "
                    "adjusted: the tensor path is unchanged and the SIMT "
                    "kernel it is\n        measured against got %.2fx faster. "
                    "See the note in this file.\n",
                    worst_ratio,
                    worst_ref > 0.0 ? worst_ref / worst_ratio : 0.0);
    }
  }

  if (out_path) write_json(out_path, prop, have_tensor, tm, tn, tk, points);

  grxblasDestroy(h);
  if (failures) { std::printf("FAILED (%d)\n", failures); return 1; }
  std::printf("PASSED\n");
  return 0;
}
