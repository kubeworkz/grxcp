// The phase 4 exit-gate sample: ONE file, __global__ kernels and <<<>>>
// launches, compiled by grxcc and run.
//
// Nothing here calls grxLaunchKernel, loads a module, or names a .vxbin. That
// is the whole point -- the same source shape a CUDA programmer writes, with
// the device half and the host half in one file, and a driver that splits it.
//
// It is deliberately not minimal. A single kernel with one pointer argument
// would not exercise the argument packing at all, so this has a kernel with
// mixed scalar and pointer parameters, a second kernel to prove the registry
// holds more than one, a launch with a shared-memory size and a stream, and a
// launch inside a loop.

#include <grx/grx.h>

#include <cstdio>
#include <vector>

// Mixed parameters: two pointers, a scalar, and a float. The argument blob has
// to place each one where the device expects it, and a wrong offset here is a
// wrong answer rather than a crash -- which is why the host checks values and
// not just the return code.
__global__ void axpy(const float* x, float* y, unsigned int n, float a) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

// A second kernel, so the registration constructor has more than one entry to
// key by stub address.
__global__ void fill_iota(float* out, unsigned int n, float base) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = base + (float)i;
}

// Shared memory, taken through the launch's third <<<>>> argument. A kernel
// that reads past what was asked for is what the sanitizer's shared-memory
// check exists for; this one stays inside.
__global__ void reverse_block(const float* in, float* out, unsigned int n) {
  float* s = grx::shared_memory<float>();
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int t = threadIdx.x;
  if (i < n) s[t] = in[i];
  __syncthreads();
  if (i < n) out[i] = s[blockDim.x - 1u - t];
}

#define CHECK(call)                                                        \
  do {                                                                     \
    grxError_t e_ = (call);                                                \
    if (e_ != grxSuccess) {                                                \
      std::printf("%s -> %s\n", #call, grxGetErrorString(e_));             \
      return 1;                                                            \
    }                                                                      \
  } while (0)

int main() {
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count == 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }
  CHECK(grxSetDevice(0));

  grxDeviceProp_t prop{};
  CHECK(grxGetDeviceProperties(&prop, 0));

  const unsigned int block = (unsigned int)prop.warpSize * 2u;
  const unsigned int n     = block * 5u + 3u;      // deliberately ragged
  const unsigned int grid  = (n + block - 1u) / block;

  float *dx = nullptr, *dy = nullptr, *dz = nullptr;
  CHECK(grxMalloc((void**)&dx, n * sizeof(float)));
  CHECK(grxMalloc((void**)&dy, n * sizeof(float)));
  CHECK(grxMalloc((void**)&dz, n * sizeof(float)));

  // Two launches with different shapes, one of them in a loop.
  fill_iota<<<grid, block>>>(dx, n, 1.0f);
  fill_iota<<<grid, block>>>(dy, n, 100.0f);
  CHECK(grxDeviceSynchronize());

  const float alpha = 3.0f;
  for (int rep = 0; rep < 2; ++rep) axpy<<<grid, block>>>(dx, dy, n, alpha);
  CHECK(grxDeviceSynchronize());

  std::vector<float> got(n, -1.0f);
  CHECK(grxMemcpy(got.data(), dy, n * sizeof(float), grxMemcpyDefault));

  int bad = 0;
  for (unsigned int i = 0; i < n; ++i) {
    // y starts at 100+i, x is 1+i, and axpy ran twice.
    const float want = 100.0f + (float)i + 2.0f * alpha * (1.0f + (float)i);
    if (got[i] != want) {
      if (bad < 4) std::printf("  axpy[%u] got %g want %g\n", i, (double)got[i],
                               (double)want);
      ++bad;
    }
  }
  std::printf("%s  axpy over %u elements, two launches deep\n",
              bad ? "FAIL" : "ok  ", n);

  // A launch with a shared-memory size and an explicit stream.
  grxStream_t stream = nullptr;
  CHECK(grxStreamCreate(&stream));
  reverse_block<<<grid, block, block * sizeof(float), stream>>>(dx, dz, n);
  CHECK(grxStreamSynchronize(stream));
  CHECK(grxStreamDestroy(stream));

  std::vector<float> rev(n, -1.0f);
  CHECK(grxMemcpy(rev.data(), dz, n * sizeof(float), grxMemcpyDefault));
  int bad_rev = 0;
  for (unsigned int i = 0; i < n; ++i) {
    const unsigned int base = (i / block) * block;
    const unsigned int mirror = base + block - 1u - (i - base);
    // The tail block is partial, so its mirror may fall outside n. Those
    // elements were never written; the kernel's guard is what makes that safe.
    if (mirror >= n) continue;
    const float want = 1.0f + (float)mirror;
    if (rev[i] != want) {
      if (bad_rev < 4) std::printf("  reverse[%u] got %g want %g\n", i,
                                   (double)rev[i], (double)want);
      ++bad_rev;
    }
  }
  std::printf("%s  reverse through shared memory, %u elements\n",
              bad_rev ? "FAIL" : "ok  ", n);

  CHECK(grxFree(dx));
  CHECK(grxFree(dy));
  CHECK(grxFree(dz));

  if (bad || bad_rev) { std::printf("FAILED\n"); return 1; }
  std::printf("PASSED\n");
  return 0;
}
