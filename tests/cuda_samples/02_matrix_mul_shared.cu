// matrixMul -- tiled multiply through static __shared__ arrays.
//
// The shape every CUDA tutorial reaches for second: a BLOCK_SIZE x BLOCK_SIZE
// tile staged in shared memory, __syncthreads() on both sides of the inner
// loop, and the accumulator in a register.

#include <grx/grx_cuda_compat.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define BLOCK_SIZE 4

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #call, \
                   cudaGetErrorString(err));                               \
      return EXIT_FAILURE;                                                 \
    }                                                                      \
  } while (0)

__global__ void matrixMulCUDA(float* C, const float* A, const float* B,
                              int wA, int wB) {
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  const int bx = blockIdx.x, by = blockIdx.y;
  const int tx = threadIdx.x, ty = threadIdx.y;

  const int row = by * BLOCK_SIZE + ty;
  const int col = bx * BLOCK_SIZE + tx;

  float Csub = 0.0f;
  for (int t = 0; t < wA / BLOCK_SIZE; ++t) {
    As[ty][tx] = A[row * wA + (t * BLOCK_SIZE + tx)];
    Bs[ty][tx] = B[(t * BLOCK_SIZE + ty) * wB + col];
    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; ++k) Csub += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  C[row * wB + col] = Csub;
}

int main() {
  const int N = 16;                        // N x N, a multiple of BLOCK_SIZE
  const size_t bytes = (size_t)N * N * sizeof(float);

  std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N, -1.0f);
  for (int i = 0; i < N * N; ++i) {
    h_A[i] = (float)((i % 7) - 3);
    h_B[i] = (float)((i % 5) - 2);
  }

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE, N / BLOCK_SIZE);
  matrixMulCUDA<<<grid, threads>>>(d_C, d_A, d_B, N, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

  int bad = 0;
  for (int r = 0; r < N; ++r)
    for (int c = 0; c < N; ++c) {
      float want = 0.0f;
      for (int k = 0; k < N; ++k) want += h_A[r * N + k] * h_B[k * N + c];
      const float got = h_C[r * N + c];
      if (fabsf(got - want) > 1e-3f) {
        if (bad < 4)
          std::printf("  [%d,%d] got %g want %g\n", r, c, (double)got,
                      (double)want);
        ++bad;
      }
    }

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  if (bad) { std::printf("FAILED: %d of %d wrong\n", bad, N * N); return 1; }
  std::printf("PASSED  matrixMul %dx%d through shared memory\n", N, N);
  return 0;
}
