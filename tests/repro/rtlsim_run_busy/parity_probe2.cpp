// One kernel, N launches, and a DIFFERENT output buffer for each.
//
// WHY THE FIRST VERSION OF THIS PROBE WAS NOT ENOUGH, which is the whole point
// of the second.
//
// parity_probe.cpp launches `sgemm_shape` N times against ONE buffer with an
// identical argument blob, and asks how many launches wrote. That detects a
// launch that does not run. It CANNOT detect a launch whose work happens one
// call late, because launch N-1's output lands in the same buffer launch N was
// going to write, and the host reads a correct-looking value either way.
//
// That blind spot was not hypothetical. grxgpu's first fix for gap 7.37 drained
// residual busy BEFORE the start pulse instead of after it, which leaves every
// frame's work to be carried out during the NEXT call. Against this probe it
// scored 7/8 -- one failure, from launch 0, which has no predecessor -- and read
// as very nearly fixed. It was not fixed at all; `test_grxblas` still aborted.
// A regression suite that reuses one buffer, which is most of them, would have
// passed it.
//
// So: buffer per launch. Launch i writes buffer i, and the host checks buffer i
// immediately after synchronising on launch i. Work that lands one call late
// leaves buffer i empty at the moment it is read, and the probe says so.
//
// Detects, distinctly:
//   * a launch that never ran           -- its own buffer stays empty
//   * work that lands one call late     -- EVERY buffer empty at read time
//   * work that lands in the wrong slot -- a buffer written that should not be

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "grx/grx_runtime.h"

struct shape_args { uint32_t abi_version; uint32_t pad; uint64_t out; };

int main(int argc, char** argv) {
  const int n = (argc > 1) ? std::atoi(argv[1]) : 8;

  grxDeviceProp_t p{};
  grxGetDeviceProperties(&p, 0);
  std::printf("device %s backend=%d SMs=%d warps/SM=%d\n", p.name, (int)p.backend,
              p.multiProcessorCount, p.maxWarpsPerMultiProcessor);

  grxModule_t mod = nullptr;
  if (grxModuleLoad(&mod, "build-real/grxblas_kernels.vxbin") != grxSuccess) return 1;
  grxFunction_t fn = nullptr;
  if (grxModuleGetFunction(&fn, mod, "sgemm_shape") != grxSuccess) return 1;

  const size_t bytes = 28;
  std::vector<void*> buf((size_t)n, nullptr);
  for (int i = 0; i < n; ++i) {
    if (grxMalloc(&buf[i], bytes) != grxSuccess) return 1;
    grxMemset(buf[i], 0, bytes);
  }

  // Read at the point the host would: right after synchronising on launch i.
  std::vector<int> at_read((size_t)n, 0);
  for (int i = 0; i < n; ++i) {
    shape_args a{};
    a.abi_version = 3;
    a.out = (uint64_t)(uintptr_t)buf[i];
    grxError_t e = grxLaunchFunction(fn, dim3_t{1,1,1},
                                     dim3_t{(unsigned)p.warpSize,1,1},
                                     &a, sizeof(a), 0, nullptr);
    if (e == grxSuccess) e = grxDeviceSynchronize();
    uint32_t s[7] = {0};
    if (e == grxSuccess) grxMemcpy(s, buf[i], bytes, grxMemcpyDefault);
    at_read[i] = (s[0] == 4);
    std::printf("launch %2d: %s\n", i, at_read[i] ? "wrote" : "*** silent ***");
    std::fflush(stdout);
  }

  // Read every buffer again at the END. A buffer that was empty when the host
  // read it but is written now was filled by a LATER launch -- which is the
  // shifted case, and is invisible to a probe that reuses one buffer.
  int ok_at_read = 0, ok_at_end = 0, late = 0;
  for (int i = 0; i < n; ++i) {
    uint32_t s[7] = {0};
    grxMemcpy(s, buf[i], bytes, grxMemcpyDefault);
    const int now = (s[0] == 4);
    ok_at_read += at_read[i];
    ok_at_end  += now;
    if (!at_read[i] && now) ++late;
  }

  std::printf("\n%d/%d written by the time the host read them\n", ok_at_read, n);
  std::printf("%d/%d written by the end of the run\n", ok_at_end, n);

  if (late > 0) {
    std::printf("\n*** %d BUFFER%s FILLED AFTER THE HOST READ %s ***\n"
                "Work is landing in a later call than the one that launched it.\n"
                "A probe reusing ONE buffer cannot see this: it reads the\n"
                "PREVIOUS launch's output and calls it correct.\n",
                late, late == 1 ? "" : "S", late == 1 ? "IT" : "THEM");
    return 1;
  }
  if (ok_at_read != n) {
    std::printf("\n%d launch%s did not run at all.\n", n - ok_at_read,
                (n - ok_at_read) == 1 ? "" : "es");
    return 1;
  }
  std::printf("\nevery launch wrote its own buffer, in its own call.\n");
  return 0;
}
