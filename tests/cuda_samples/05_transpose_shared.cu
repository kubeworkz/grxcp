// transpose -- the coalesced shared-memory transpose from NVIDIA's
// "Efficient Matrix Transpose" post, padding included.
//
// The +1 on the tile's second dimension is the bank-conflict padding every
// version of this kernel carries. It is kept because a port that quietly drops
// it is a port that changed the program.

#include <grx/grx_cuda_compat.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define TILE_DIM 4
#define BLOCK_ROWS 4

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #call, \
                   cudaGetErrorString(err));                               \
      return EXIT_FAILURE;                                                 \
    }                                                                      \
  } while (0)

__global__ void transposeCoalesced(float* odata, const float* idata, int width) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main() {
  const int N = 16;
  const size_t bytes = (size_t)N * N * sizeof(float);

  std::vector<float> h_in(N * N), h_out(N * N, -1.0f);
  for (int i = 0; i < N * N; ++i) h_in[i] = (float)i;

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  dim3 threads(TILE_DIM, BLOCK_ROWS);
  dim3 grid(N / TILE_DIM, N / TILE_DIM);
  transposeCoalesced<<<grid, threads>>>(d_out, d_in, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  int bad = 0;
  for (int r = 0; r < N; ++r)
    for (int c = 0; c < N; ++c) {
      const float want = h_in[c * N + r];
      const float got = h_out[r * N + c];
      if (got != want) {
        if (bad < 4)
          std::printf("  [%d,%d] got %g want %g\n", r, c, (double)got,
                      (double)want);
        ++bad;
      }
    }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  if (bad) { std::printf("FAILED: %d of %d wrong\n", bad, N * N); return 1; }
  std::printf("PASSED  transpose %dx%d through a padded shared tile\n", N, N);
  return 0;
}
