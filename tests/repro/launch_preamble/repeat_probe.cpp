// Does the launch preamble grow with the grid, or with the launch COUNT?
//
// The grid sweep in ph2 ran seven launches back to back in one process, in
// ascending block order. Every "earliest block entry" it reported is therefore
// confounded with sweep position. This probe removes the grid from the
// experiment entirely: it launches the SAME shape N times and prints what each
// launch reports.
//
//   flat        -> MCYCLE restarts per launch; the sweep measured the grid
//   monotonic   -> MCYCLE is cumulative; the sweep measured its own position
//
// argv: [blocks] [threads-per-block] [reps]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include "grx/grx_runtime.h"

struct pargs { uint32_t abi; uint32_t slots; uint64_t out; };

int main(int argc, char** argv) {
  const int nb   = (argc > 1) ? std::atoi(argv[1]) : 4;
  const int tpb  = (argc > 2) ? std::atoi(argv[2]) : 16;
  const int reps = (argc > 3) ? std::atoi(argv[3]) : 6;

  grxDeviceProp_t p{};
  grxGetDeviceProperties(&p, 0);
  std::printf("%s: %d SMs   grid=%d blocks x %d threads, %d identical launches\n\n",
              p.name, p.multiProcessorCount, nb, tpb, reps);

  grxModule_t m = nullptr;
  if (grxModuleLoad(&m, "build-real/preamble.vxbin") != grxSuccess) return 1;
  grxFunction_t f = nullptr;
  if (grxModuleGetFunction(&f, m, "preamble_probe") != grxSuccess) return 1;

  const int MAXB = 64;
  const size_t bytes = (size_t)MAXB * 4 * 8;
  void* d = nullptr;
  if (grxMalloc(&d, bytes) != grxSuccess) return 1;

  std::printf("%-6s %14s %14s %10s %12s\n",
              "launch", "first entry", "last entry", "spread", "delta first");

  uint64_t prev = 0;
  for (int r = 0; r < reps; ++r) {
    grxMemset(d, 0, bytes);
    pargs a{};
    a.abi = 3; a.slots = MAXB; a.out = (uint64_t)(uintptr_t)d;
    grxError_t e = grxLaunchFunction(f, dim3_t{(unsigned)nb, 1, 1},
                                     dim3_t{(unsigned)tpb, 1, 1},
                                     &a, sizeof(a), 0, nullptr);
    if (e == grxSuccess) e = grxDeviceSynchronize();
    std::vector<uint64_t> o((size_t)MAXB * 4, 0);
    if (e == grxSuccess) grxMemcpy(o.data(), d, bytes, grxMemcpyDefault);

    uint64_t lo = ~0ull, hi = 0; int seen = 0;
    for (int b = 0; b < nb; ++b) {
      if (o[(size_t)b * 4 + 3] == 0) continue;
      ++seen;
      lo = std::min(lo, o[(size_t)b * 4]);
      hi = std::max(hi, o[(size_t)b * 4]);
    }
    if (!seen) { std::printf("%-6d %14s\n", r, "(none ran)"); continue; }
    std::printf("%-6d %14llu %14llu %10llu %12lld\n", r,
                (unsigned long long)lo, (unsigned long long)hi,
                (unsigned long long)(hi - lo),
                (long long)(r ? (int64_t)lo - (int64_t)prev : 0));
    prev = lo;
  }

  std::printf("\nSame grid every time. If 'first entry' climbs, the counter is\n"
              "cumulative across launches and the grid sweep measured itself.\n");
  return 0;
}
