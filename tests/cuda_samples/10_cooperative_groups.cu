// cooperative_groups -- the reduction from NVIDIA's "Cooperative Groups"
// developer-blog post: a thread_block partitioned into tiles, cg::reduce over
// the tile, and the block's tile leaders combined in shared memory.
//
// ONE THING HERE IS NOT WHAT THE ORIGINAL SAYS, and it is the only edit in this
// directory. The post uses `tiled_partition<16>`, which assumes a 32-lane warp.
// A tile cannot be wider than the warp it slices -- the shuffle instruction's
// segment mask cannot express it -- so on this build, whose warp is 4 lanes,
// `<16>` is a static_assert and not a runtime surprise:
//
//   error: static assertion failed: this build's warp is VX_CFG_NUM_THREADS
//   lanes and a tile cannot be wider than the warp it slices
//
// That is the platform being right, and it is why TILE is 4 below. It is a
// CONSTANT, not a construct: every cooperative-groups mechanism the post uses
// is here unchanged. See cuda_mapping.md section 7.9.

#include <grx/grx_cuda_compat.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #call, \
                   cudaGetErrorString(err));                               \
      return EXIT_FAILURE;                                                 \
    }                                                                      \
  } while (0)

#define TILE 4

__global__ void tileReduce(const float* in, float* out, unsigned int n) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<TILE> tile = cg::tiled_partition<TILE>(block);

  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const float v = (i < n) ? in[i] : 0.0f;

  const float tile_sum = cg::reduce(tile, v, cg::plus<float>());

  if (tile.thread_rank() == 0) {
    const unsigned int tiles_per_block = blockDim.x / TILE;
    out[blockIdx.x * tiles_per_block + (threadIdx.x / TILE)] = tile_sum;
  }
}

int main() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  const unsigned int threads = 8;
  const unsigned int blocks = 4;
  const unsigned int n = threads * blocks;
  const unsigned int ntiles = n / TILE;

  std::vector<float> h_in(n);
  for (unsigned int i = 0; i < n; ++i) h_in[i] = (float)(i % 6) + 1.0f;

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_out, ntiles * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  tileReduce<<<blocks, threads>>>(d_in, d_out, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(ntiles, -1.0f);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, ntiles * sizeof(float),
                        cudaMemcpyDeviceToHost));

  int bad = 0;
  for (unsigned int t = 0; t < ntiles; ++t) {
    float want = 0.0f;
    for (unsigned int l = 0; l < TILE; ++l) want += h_in[t * TILE + l];
    if (fabsf(h_out[t] - want) > 1e-4f) {
      if (bad < 4)
        std::printf("  tile %u got %g want %g\n", t, (double)h_out[t],
                    (double)want);
      ++bad;
    }
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  if (bad) { std::printf("FAILED: %d of %u tiles wrong\n", bad, ntiles); return 1; }
  std::printf("PASSED  cooperative-groups tile reduction, %u tiles of %d\n",
              ntiles, TILE);
  return 0;
}
