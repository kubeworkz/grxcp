// Phase 1 exit gate: run a kernel on the device and check the arithmetic.
//
// Everything before this proved the runtime could describe a device and move
// bytes. This proves it can make the device compute, which is the only claim
// that matters for a compute platform.
//
// The kernel is built separately into a .vxbin by ci/build_kernel.sh; the host
// side below is ordinary GRXCP.

#include <grx/grx.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "common.h"

#define CHECK(call)                                                      \
  do {                                                                   \
    grxError_t e_ = (call);                                              \
    if (e_ != grxSuccess) {                                              \
      std::fprintf(stderr, "%s -> %s (%s)\n", #call,                     \
                   grxGetErrorString(e_), grxGetErrorName(e_));          \
      return 1;                                                          \
    }                                                                    \
  } while (0)

int main(int argc, char** argv) {
  const char* image = (argc > 1) ? argv[1] : "vecadd.vxbin";
  const int   N     = (argc > 2) ? std::atoi(argv[2]) : 64;

  int count = 0;
  CHECK(grxGetDeviceCount(&count));
  if (count == 0) { std::fprintf(stderr, "no devices\n"); return 77; }
  CHECK(grxSetDevice(0));

  grxDeviceProp_t prop{};
  CHECK(grxGetDeviceProperties(&prop, 0));

  grxModule_t   mod = nullptr;
  grxFunction_t fn  = nullptr;
  CHECK(grxModuleLoad(&mod, image));
  CHECK(grxModuleGetFunction(&fn, mod, "vecadd"));

  const size_t bytes = (size_t)N * sizeof(float);
  std::vector<float> ha(N), hb(N), hc(N, 0.0f);
  for (int i = 0; i < N; ++i) {
    ha[i] = (float)i * 1.5f;
    hb[i] = (float)i * 0.25f + 1.0f;
  }

  void *da = nullptr, *db = nullptr, *dc = nullptr;
  CHECK(grxMalloc(&da, bytes));
  CHECK(grxMalloc(&db, bytes));
  CHECK(grxMalloc(&dc, bytes));
  CHECK(grxMemcpy(da, ha.data(), bytes, grxMemcpyDefault));
  CHECK(grxMemcpy(db, hb.data(), bytes, grxMemcpyDefault));
  CHECK(grxMemset(dc, 0, bytes));

  vecadd_args args{};
  args.n = (uint32_t)N;
  args.a = (uint64_t)(uintptr_t)da;
  args.b = (uint64_t)(uintptr_t)db;
  args.c = (uint64_t)(uintptr_t)dc;

  // One warp per block keeps this inside the small default configuration; the
  // grid covers N with the last block partially masked when N is not a
  // multiple, which is exactly the case the kernel's bounds check exists for.
  const unsigned block = (unsigned)prop.warpSize;
  const unsigned grid  = (unsigned)((N + block - 1) / block);
  std::printf("launching vecadd: grid=%u block=%u n=%d on %s\n",
              grid, block, N, prop.name);

  CHECK(grxLaunchFunction(fn, dim3_t{grid, 1, 1}, dim3_t{block, 1, 1},
                          &args, sizeof(args), /*sharedMem=*/0,
                          /*stream=*/nullptr));
  CHECK(grxDeviceSynchronize());
  CHECK(grxMemcpy(hc.data(), dc, bytes, grxMemcpyDefault));

  int bad = 0;
  for (int i = 0; i < N; ++i) {
    const float want = ha[i] + hb[i];
    if (std::fabs(hc[i] - want) > 1e-5f) {
      if (bad < 5)
        std::fprintf(stderr, "  mismatch at %d: got %f want %f\n", i, hc[i], want);
      ++bad;
    }
  }

  CHECK(grxFree(da));
  CHECK(grxFree(db));
  CHECK(grxFree(dc));
  CHECK(grxModuleUnload(mod));

  if (bad) { std::fprintf(stderr, "FAILED: %d/%d elements wrong\n", bad, N); return 1; }
  std::printf("PASSED: %d/%d elements correct\n", N, N);
  return 0;
}
