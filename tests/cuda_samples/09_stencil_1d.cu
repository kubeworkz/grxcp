// 1D stencil with a halo staged in shared memory -- the example from
// NVIDIA's "An Even Easier Introduction to CUDA" follow-ups, and the smallest
// program where getting __syncthreads() wrong gives a wrong answer instead of
// a hang.

#include <grx/grx_cuda_compat.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define RADIUS 3
#define BLOCK_SIZE 32

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #call, \
                   cudaGetErrorString(err));                               \
      return EXIT_FAILURE;                                                 \
    }                                                                      \
  } while (0)

__global__ void stencil_1d(const float* in, float* out) {
  __shared__ float temp[BLOCK_SIZE + 2 * RADIUS];

  const int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  const int lindex = threadIdx.x + RADIUS;

  temp[lindex] = in[gindex];
  if (threadIdx.x < RADIUS) {
    temp[lindex - RADIUS] = in[gindex - RADIUS];
    temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
  }
  __syncthreads();

  float result = 0.0f;
  for (int offset = -RADIUS; offset <= RADIUS; ++offset)
    result += temp[lindex + offset];

  out[gindex] = result;
}

int main() {
  const int n = BLOCK_SIZE * 8;
  const int padded = n + 2 * RADIUS;

  // The classic layout: the array is padded by RADIUS on both ends, and the
  // kernel is launched over the interior.
  std::vector<float> h_in(padded), h_out(padded, 0.0f);
  for (int i = 0; i < padded; ++i) h_in[i] = (float)(i % 11);

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_in, padded * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_out, padded * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), padded * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, padded * sizeof(float)));

  stencil_1d<<<n / BLOCK_SIZE, BLOCK_SIZE>>>(d_in + RADIUS, d_out + RADIUS);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, padded * sizeof(float),
                        cudaMemcpyDeviceToHost));

  int bad = 0;
  for (int i = RADIUS; i < n + RADIUS; ++i) {
    float want = 0.0f;
    for (int o = -RADIUS; o <= RADIUS; ++o) want += h_in[i + o];
    if (fabsf(h_out[i] - want) > 1e-4f) {
      if (bad < 4)
        std::printf("  [%d] got %g want %g\n", i, (double)h_out[i],
                    (double)want);
      ++bad;
    }
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  if (bad) { std::printf("FAILED: %d of %d wrong\n", bad, n); return 1; }
  std::printf("PASSED  1D stencil radius %d over %d elements\n", RADIUS, n);
  return 0;
}
