// Hillis-Steele inclusive scan within a block, double-buffered in shared
// memory -- the scan from GPU Gems 3 chapter 39, in its naive form.
//
// The double buffer is the interesting part for a port: two halves of one
// dynamic shared allocation, with the roles swapping every step, and a
// __syncthreads() between the read and the write.

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

__global__ void scan(const float* g_idata, float* g_odata, int n) {
  extern __shared__ float temp[];

  const int thid = threadIdx.x;
  int pout = 0, pin = 1;

  temp[pout * n + thid] = (thid < n) ? g_idata[thid] : 0.0f;
  __syncthreads();

  for (int offset = 1; offset < n; offset *= 2) {
    pout = 1 - pout;
    pin = 1 - pout;
    if (thid >= offset)
      temp[pout * n + thid] = temp[pin * n + thid] + temp[pin * n + thid - offset];
    else
      temp[pout * n + thid] = temp[pin * n + thid];
    __syncthreads();
  }

  g_odata[thid] = temp[pout * n + thid];
}

int main() {
  const int n = 32;                       // one block

  std::vector<float> h_in(n), h_out(n, -1.0f);
  for (int i = 0; i < n; ++i) h_in[i] = (float)(i % 5) + 1.0f;

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  scan<<<1, n, 2 * n * sizeof(float)>>>(d_in, d_out, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  int bad = 0;
  float running = 0.0f;
  for (int i = 0; i < n; ++i) {
    running += h_in[i];
    if (fabsf(h_out[i] - running) > 1e-4f) {
      if (bad < 4)
        std::printf("  [%d] got %g want %g\n", i, (double)h_out[i],
                    (double)running);
      ++bad;
    }
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  if (bad) { std::printf("FAILED: %d of %d wrong\n", bad, n); return 1; }
  std::printf("PASSED  inclusive scan over %d elements\n", n);
  return 0;
}
