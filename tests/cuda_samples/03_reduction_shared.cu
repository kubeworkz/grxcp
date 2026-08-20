// reduction -- the shared-memory tree reduction from the CUDA samples, with
// the sequential-addressing addressing scheme (reduce3 in NVIDIA's kernel set).
//
// Note the extern __shared__ declaration: the block's scratch comes from the
// launch's third <<<>>> argument rather than from a fixed-size array.

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

__global__ void reduce(const float* g_idata, float* g_odata, unsigned int n) {
  extern __shared__ float sdata[];

  const unsigned int tid = threadIdx.x;
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
  const unsigned int n = 1024;
  const unsigned int threads = 32;
  const unsigned int blocks = (n + threads - 1) / threads;

  std::vector<float> h_in(n);
  for (unsigned int i = 0; i < n; ++i) h_in[i] = (float)(i % 13) - 6.0f;

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_out, blocks * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  reduce<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_partial(blocks);
  CUDA_CHECK(cudaMemcpy(h_partial.data(), d_out, blocks * sizeof(float),
                        cudaMemcpyDeviceToHost));

  double got = 0.0, want = 0.0;
  for (unsigned int b = 0; b < blocks; ++b) got += h_partial[b];
  for (unsigned int i = 0; i < n; ++i) want += h_in[i];

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  if (fabs(got - want) > 1e-3) {
    std::printf("FAILED: sum %g, want %g\n", got, want);
    return 1;
  }
  std::printf("PASSED  reduction over %u elements = %g\n", n, want);
  return 0;
}
