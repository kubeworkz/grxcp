// GRXCP sample: vector addition.
//
// This is the target-state source form -- what compiles once grxcc exists
// (roadmap phase 4). Before that, the same program is written with an explicit
// grxLaunchKernel call and a module load; see tests/samples/vecadd/README.md.
//
//   grxcc -grx-arch=g100 vecadd.grx.cpp -o vecadd

#include <grx/grx.h>
#include <grx/device/grx_device.h>
#include <cstdio>

__global__ void vecadd(const float* a, const float* b, float* c, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

#define GRX_CHECK(call)                                                   \
  do {                                                                    \
    grxError_t err_ = (call);                                             \
    if (err_ != grxSuccess) {                                             \
      std::fprintf(stderr, "%s failed: %s\n", #call,                      \
                   grxGetErrorString(err_));                              \
      return 1;                                                           \
    }                                                                     \
  } while (0)

int main() {
  constexpr int N = 1 << 16;
  constexpr size_t bytes = N * sizeof(float);

  grxDeviceProp_t prop{};
  GRX_CHECK(grxGetDeviceProperties(&prop, 0));
  std::printf("device: %s  SMs=%d  warp=%d  smem/SM=%zu  backend=%d\n",
              prop.name, prop.multiProcessorCount, prop.warpSize,
              prop.sharedMemPerMultiprocessor, (int)prop.backend);

  float *ha = new float[N], *hb = new float[N], *hc = new float[N];
  for (int i = 0; i < N; ++i) { ha[i] = float(i); hb[i] = float(2 * i); }

  float *da = nullptr, *db = nullptr, *dc = nullptr;
  GRX_CHECK(grxMalloc((void**)&da, bytes));
  GRX_CHECK(grxMalloc((void**)&db, bytes));
  GRX_CHECK(grxMalloc((void**)&dc, bytes));

  GRX_CHECK(grxMemcpy(da, ha, bytes, grxMemcpyDefault));
  GRX_CHECK(grxMemcpy(db, hb, bytes, grxMemcpyDefault));

  const int block = prop.warpSize * 4;
  const int grid  = (N + block - 1) / block;
  vecadd<<<grid, block>>>(da, db, dc, N);
  GRX_CHECK(grxGetLastError());
  GRX_CHECK(grxDeviceSynchronize());

  GRX_CHECK(grxMemcpy(hc, dc, bytes, grxMemcpyDefault));

  for (int i = 0; i < N; ++i) {
    if (hc[i] != ha[i] + hb[i]) {
      std::fprintf(stderr, "mismatch at %d: %f != %f\n", i, hc[i],
                   ha[i] + hb[i]);
      return 1;
    }
  }
  std::printf("PASSED\n");

  GRX_CHECK(grxFree(da)); GRX_CHECK(grxFree(db)); GRX_CHECK(grxFree(dc));
  delete[] ha; delete[] hb; delete[] hc;
  return 0;
}
