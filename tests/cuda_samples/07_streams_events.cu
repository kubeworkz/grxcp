// Streams and events: two streams each doing their own work, an event
// recorded on one and waited on by the other, and elapsed time reported.
//
// This is the shape of every CUDA overlap example. What it actually proves on
// a given platform depends on whether the streams run concurrently -- so it
// checks ORDERING and RESULTS, which are defined, and prints the time without
// claiming anything about it.

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

__global__ void scale(float* data, float k, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] *= k;
}

__global__ void shift(float* data, float k, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] += k;
}

int main() {
  const int n = 512;
  const size_t bytes = n * sizeof(float);
  const int threads = 32;
  const int blocks = (n + threads - 1) / threads;

  std::vector<float> h(n, 1.0f), out(n, -1.0f);

  float* d = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d, bytes));

  cudaStream_t s1, s2;
  CUDA_CHECK(cudaStreamCreate(&s1));
  CUDA_CHECK(cudaStreamCreate(&s2));

  cudaEvent_t start, mid, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&mid));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, s1));
  CUDA_CHECK(cudaMemcpyAsync(d, h.data(), bytes, cudaMemcpyHostToDevice, s1));
  scale<<<blocks, threads, 0, s1>>>(d, 3.0f, n);
  CUDA_CHECK(cudaEventRecord(mid, s1));

  // s2 must not start until s1's scale has finished, or the result is a race.
  CUDA_CHECK(cudaStreamWaitEvent(s2, mid, 0));
  shift<<<blocks, threads, 0, s2>>>(d, 2.0f, n);
  CUDA_CHECK(cudaMemcpyAsync(out.data(), d, bytes, cudaMemcpyDeviceToHost, s2));
  CUDA_CHECK(cudaEventRecord(stop, s2));

  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  int bad = 0;
  for (int i = 0; i < n; ++i)
    if (fabsf(out[i] - 5.0f) > 1e-5f) {           // 1*3 + 2
      if (bad < 4) std::printf("  [%d] got %g want 5\n", i, (double)out[i]);
      ++bad;
    }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(mid));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(s1));
  CUDA_CHECK(cudaStreamDestroy(s2));
  CUDA_CHECK(cudaFree(d));

  if (bad) { std::printf("FAILED: %d of %d wrong\n", bad, n); return 1; }
  std::printf("PASSED  two streams ordered by an event (%.3f ms elapsed)\n",
              (double)ms);
  return 0;
}
