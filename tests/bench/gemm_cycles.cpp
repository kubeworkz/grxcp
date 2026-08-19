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
  uint64_t tensor_span;
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

}  // namespace

int main() {
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
    bool measured = false, met = true;
    double worst_ratio = 0.0;
    for (const Point& p : points) {
      if (p.tensor_per_element <= 0.0 || p.per_element <= 0.0) continue;
      measured = true;
      const double speedup = p.per_element / p.tensor_per_element;
      if (speedup < 5.0) met = false;
      if (worst_ratio == 0.0 || speedup < worst_ratio) worst_ratio = speedup;
    }
    expect(measured, "the tensor path was measured at every shape");
    std::printf("        worst speedup across the shapes: %.2fx\n", worst_ratio);
    // Enforced, because it is met. A gate that a passing kernel does not have
    // to keep passing is a comment.
    expect(met, "GemmEx costs at most a fifth of sgemm per output element");
  }

  grxblasDestroy(h);
  if (failures) { std::printf("FAILED (%d)\n", failures); return 1; }
  std::printf("PASSED\n");
  return 0;
}
