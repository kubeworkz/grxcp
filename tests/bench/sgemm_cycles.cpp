// grxBLAS sgemm: what it costs, in device cycles.
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
// It is a report, not a gate: it fails only if the measurement itself is
// broken (nothing recorded, or cost that does not grow with the work). A
// threshold belongs here once there are two kernels to compare.

#include <grx/grx.h>
#include <grx/grxblas.h>
#include <grx/grx_cycles.h>

#include <cstdio>
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
};

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
  std::printf("sgemm v0 (one thread per output element, no blocking, no "
              "tensor cores)\n");

  const struct { int m, n, k; } shapes[] = {
    { 16, 16,  16},
    { 16, 16,  32},
    { 16, 16,  64},
    { 32, 32,  32},
    { 64, 64,  32},
  };

  std::vector<Point> points;
  for (const auto& s : shapes) {
    std::vector<float> A((size_t)s.m * s.k, 1.0f);
    std::vector<float> B((size_t)s.k * s.n, 1.0f);

    void *dA = nullptr, *dB = nullptr, *dC = nullptr;
    if (grxMalloc(&dA, A.size() * sizeof(float)) != grxSuccess ||
        grxMalloc(&dB, B.size() * sizeof(float)) != grxSuccess ||
        grxMalloc(&dC, (size_t)s.m * s.n * sizeof(float)) != grxSuccess) {
      std::printf("  allocation failed at %dx%dx%d\n", s.m, s.n, s.k);
      ++failures;
      break;
    }
    grxMemcpy(dA, A.data(), A.size() * sizeof(float), grxMemcpyDefault);
    grxMemcpy(dB, B.data(), B.size() * sizeof(float), grxMemcpyDefault);

    const int nslots = grxblasCycleSlotsNeeded(h, s.m, s.n);
    void* dSlots = nullptr;
    if (nslots <= 0 ||
        grxMalloc(&dSlots, (size_t)nslots * sizeof(grxCycleSlot)) != grxSuccess) {
      std::printf("  probe allocation failed at %dx%dx%d\n", s.m, s.n, s.k);
      ++failures;
      break;
    }
    grxMemset(dSlots, 0, (size_t)nslots * sizeof(grxCycleSlot));
    grxblasSetCycleProbe(h, (grxCycleSlot*)dSlots, nslots);

    const grxblasStatus_t st =
        grxblasSgemm(h, GRXBLAS_OP_N, GRXBLAS_OP_N, s.m, s.n, s.k, &one,
                     dA, s.m, dB, s.k, &zero, dC, s.m);
    grxDeviceSynchronize();
    grxblasSetCycleProbe(h, nullptr, 0);

    if (st != GRXBLAS_STATUS_SUCCESS) {
      if (st == GRXBLAS_STATUS_NOT_INITIALIZED) {
        std::printf("grxblas device kernels not found; skipping\n");
        grxblasDestroy(h);
        return 77;
      }
      std::printf("  %dx%dx%d failed: %s\n", s.m, s.n, s.k,
                  grxblasGetStatusString(st));
      ++failures;
      break;
    }

    std::vector<grxCycleSlot> slots(nslots);
    grxMemcpy(slots.data(), dSlots, slots.size() * sizeof(grxCycleSlot),
              grxMemcpyDefault);
    grxCycleSummary sum{};
    grxCycleSummarize(slots.data(), nslots, &sum);

    Point p{};
    p.m = s.m; p.n = s.n; p.k = s.k;
    p.span = sum.spanIsValid ? sum.span : 0;
    const double elems = (double)s.m * s.n;
    // The headline is the SPAN: first warp starting to last warp finishing,
    // over the elements produced. That is what "faster" means -- it charges a
    // kernel for poor parallel efficiency, which a per-warp average hides. The
    // per-warp figure stays beside it because the two diverging is itself
    // information: it means warps are waiting on each other.
    p.per_element      = p.span ? (double)p.span / elems : 0.0;
    p.per_mac          = p.per_element / (double)s.k;
    p.warp_per_element = sum.busyMedian
                             ? (double)sum.busyMedian * (double)nslots / elems
                             : 0.0;
    points.push_back(p);

    std::printf("  %3dx%3dx%3d  %4d warps  span %8llu  %8.1f cyc/element  "
                "%6.2f cyc/MAC   (per-warp median %6llu, %7.1f warp-cyc/elem)\n",
                s.m, s.n, s.k, nslots, (unsigned long long)p.span,
                p.per_element, p.per_mac,
                (unsigned long long)sum.busyMedian, p.warp_per_element);

    grxFree(dA); grxFree(dB); grxFree(dC); grxFree(dSlots);
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
    std::printf("\n  BASELINE for the phase 3 gate: %.1f cycles per output "
                "element at %dx%dx%d,\n  on %s with %d SM and a %d-lane warp. "
                "A cycle count from a simulator is what the\n  MODEL does: the "
                "right thing to compare kernels with, the wrong thing to quote "
                "as hardware.\n",
                points[0].per_element, points[0].m, points[0].n, points[0].k,
                prop.name, prop.multiProcessorCount, prop.warpSize);
  }

  grxblasDestroy(h);
  if (failures) { std::printf("FAILED (%d)\n", failures); return 1; }
  std::printf("PASSED (a report, not a threshold)\n");
  return 0;
}
