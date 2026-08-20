// Warp-level reduction with __shfl_down_sync -- the idiom from NVIDIA's
// "Faster Parallel Reductions on Kepler" post, which every modern CUDA
// reduction is built on.
//
// warpSize is a CUDA built-in, and this uses it rather than the literal 32,
// which is the portable way to write it and the only way that works on a
// device whose warp is not 32 lanes.

#include <grx/grx_cuda_compat.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #call, \
                   cudaGetErrorString(err));                               \
      return EXIT_FAILURE;                                                 \
    }                                                                      \
  } while (0)

__inline__ __device__ float warpReduceSum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__global__ void warpSumKernel(const float* in, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = (i < n) ? in[i] : 0.0f;
  v = warpReduceSum(v);
  // Lane 0 of each warp writes the warp's total.
  if ((threadIdx.x % warpSize) == 0)
    out[i / warpSize] = v;
}

int main() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  const int threads = prop.warpSize;          // one warp per block
  const int n = threads * 16;
  const int nwarps = n / prop.warpSize;

  std::vector<float> h_in(n);
  for (int i = 0; i < n; ++i) h_in[i] = (float)(i % 9) - 4.0f;

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_out, nwarps * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  warpSumKernel<<<n / threads, threads>>>(d_in, d_out, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(nwarps);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, nwarps * sizeof(float),
                        cudaMemcpyDeviceToHost));

  int bad = 0;
  for (int w = 0; w < nwarps; ++w) {
    float want = 0.0f;
    for (int l = 0; l < prop.warpSize; ++l) want += h_in[w * prop.warpSize + l];
    if (fabsf(h_out[w] - want) > 1e-4f) {
      if (bad < 4)
        std::printf("  warp %d got %g want %g\n", w, (double)h_out[w],
                    (double)want);
      ++bad;
    }
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  if (bad) { std::printf("FAILED: %d of %d warps wrong\n", bad, nwarps); return 1; }
  std::printf("PASSED  warp shuffle reduction, %d warps of %d lanes\n",
              nwarps, prop.warpSize);
  return 0;
}
