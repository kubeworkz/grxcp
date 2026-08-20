// vectorAdd -- the first CUDA program anybody writes.
//
// Everything about this file is ordinary CUDA except the include on the next
// line. See README.md for the rule this directory is held to.

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

__global__ void vectorAdd(const float* A, const float* B, float* C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  const int numElements = 5000;
  const size_t size = numElements * sizeof(float);

  std::vector<float> h_A(numElements), h_B(numElements), h_C(numElements);
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = (float)i / (float)numElements;
    h_B[i] = (float)(numElements - i) / (float)numElements;
  }

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_A, size));
  CUDA_CHECK(cudaMalloc((void**)&d_B, size));
  CUDA_CHECK(cudaMalloc((void**)&d_C, size));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

  const int threadsPerBlock = 32;
  const int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

  int bad = 0;
  for (int i = 0; i < numElements; ++i) {
    const float want = h_A[i] + h_B[i];
    if (fabsf(h_C[i] - want) > 1e-5f) {
      if (bad < 4)
        std::printf("  [%d] got %g want %g\n", i, (double)h_C[i], (double)want);
      ++bad;
    }
  }

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  if (bad) { std::printf("FAILED: %d of %d wrong\n", bad, numElements); return 1; }
  std::printf("PASSED  vectorAdd, %d elements\n", numElements);
  return 0;
}
