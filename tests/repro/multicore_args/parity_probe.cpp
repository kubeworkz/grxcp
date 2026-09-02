// Same kernel, same argsSize, same buffer, ten times. Nothing varies but the
// launch index. If the result alternates, args size was never the variable.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "grx/grx_runtime.h"
struct shape_args { uint32_t abi_version; uint32_t pad; uint64_t out; };
int main(int argc, char** argv) {
  const int n = (argc > 1) ? atoi(argv[1]) : 10;
  grxDeviceProp_t p{}; grxGetDeviceProperties(&p, 0);
  std::printf("device %s backend=%d SMs=%d warps/SM=%d\n", p.name, (int)p.backend,
              p.multiProcessorCount, p.maxWarpsPerMultiProcessor);
  grxModule_t mod = nullptr;
  if (grxModuleLoad(&mod, "build-real/grxblas_kernels.vxbin") != grxSuccess) return 1;
  grxFunction_t fn = nullptr;
  if (grxModuleGetFunction(&fn, mod, "sgemm_shape") != grxSuccess) return 1;
  void* dshape = nullptr; const size_t bytes = 28;
  if (grxMalloc(&dshape, bytes) != grxSuccess) return 1;
  shape_args sargs{}; sargs.abi_version = 3; sargs.out = (uint64_t)(uintptr_t)dshape;
  int ok = 0;
  for (int i = 0; i < n; ++i) {
    grxMemset(dshape, 0, bytes);
    grxError_t e = grxLaunchFunction(fn, dim3_t{1,1,1}, dim3_t{(unsigned)p.warpSize,1,1},
                                     &sargs, sizeof(sargs), 0, nullptr);
    if (e == grxSuccess) e = grxDeviceSynchronize();
    uint32_t s[7] = {0};
    if (e == grxSuccess) grxMemcpy(s, dshape, bytes, grxMemcpyDefault);
    const bool good = (s[0] == 4);
    ok += good;
    std::printf("launch %2d: %s\n", i, good ? "wrote" : "*** silent ***");
    std::fflush(stdout);
  }
  std::printf("%d/%d launches wrote\n", ok, n);
  return 0;
}
