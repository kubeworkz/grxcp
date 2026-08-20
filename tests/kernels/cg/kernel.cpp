// Cooperative groups on the device: blocks, tiles, coalesced groups, the
// cluster, and the grid-wide barrier.
//
// Everything is integer, so every comparison the host makes is exact. Each
// collective writes its own output band, so a wrong one is identifiable rather
// than showing up as a wrong total three collectives later.

#include <grx/device/grx_cg.h>

#include "common.h"

namespace cg = grx::cg;

namespace {

__forceinline__ uint32_t* band(cg_args* arg, uint32_t b) {
  return reinterpret_cast<uint32_t*>(arg->out) + (size_t)b * arg->threads;
}

}  // namespace

// thread_block: ranks, size, and a sync that actually has to work -- every
// thread reads the shared cell its neighbour wrote, which is only defined
// after the barrier and spans warps.
__global__ void cg_block(cg_args* __UNIFORM__ arg) {
  const uint32_t* in = reinterpret_cast<const uint32_t*>(arg->in);
  cg::thread_block block = cg::this_thread_block();

  const uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= arg->threads) return;

  const uint32_t rank = block.thread_rank();
  const uint32_t n    = block.num_threads();

  uint32_t* smem = grx::shared_memory<uint32_t>();
  smem[rank] = in[tid];
  block.sync();
  const uint32_t neighbour = smem[(rank + 1u) % n];

  band(arg, CG_BAND_RANK)[tid]   = rank;
  band(arg, CG_BAND_SIZE)[tid]   = n;
  band(arg, CG_BAND_REDUCE)[tid] = neighbour;
}

// The tile collectives. Two entry points because the width is a template
// parameter -- which is the point: a tile wider than the warp does not compile.
template <unsigned Size>
__forceinline__ void tile_body(cg_args* arg) {
  const uint32_t* in = reinterpret_cast<const uint32_t*>(arg->in);
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= arg->threads) return;

  auto tile = cg::tiled_partition<Size>(cg::this_thread_block());
  const uint32_t v = in[tid];

  band(arg, CG_BAND_RANK)[tid]      = tile.thread_rank();
  band(arg, CG_BAND_SIZE)[tid]      = tile.num_threads();
  band(arg, CG_BAND_REDUCE)[tid]    = tile.reduce(v, cg::plus<uint32_t>());
  band(arg, CG_BAND_INCLUSIVE)[tid] = tile.inclusive_scan(v);
  band(arg, CG_BAND_EXCLUSIVE)[tid] = tile.exclusive_scan(v);
  band(arg, CG_BAND_SHFL)[tid]      = tile.shfl(v, 0);
  band(arg, CG_BAND_BALLOT)[tid]    = tile.ballot((int)(tile.thread_rank() & 1u));
  band(arg, CG_BAND_META)[tid]      = tile.meta_group_rank();
}

__global__ void cg_tile_full(cg_args* __UNIFORM__ arg) {
  tile_body<VX_CFG_NUM_THREADS>(arg);
}

#if VX_CFG_NUM_THREADS > 1
__global__ void cg_tile_half(cg_args* __UNIFORM__ arg) {
  tile_body<(VX_CFG_NUM_THREADS / 2)>(arg);
}
#endif

// coalesced_group, taken inside a divergent branch. Only the odd lanes are
// converged here, so a group that reports the whole warp is wrong -- and so is
// one whose ranks are lane ids rather than dense.
__global__ void cg_coalesced(cg_args* __UNIFORM__ arg) {
  const uint32_t* in = reinterpret_cast<const uint32_t*>(arg->in);
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= arg->threads) return;

  band(arg, CG_BAND_RANK)[tid] = 0xffffffffu;
  band(arg, CG_BAND_SIZE)[tid] = 0xffffffffu;
  band(arg, CG_BAND_SHFL)[tid] = 0xffffffffu;

  if (grx::lane_id() & 1u) {
    cg::coalesced_group g = cg::coalesced_threads();
    band(arg, CG_BAND_RANK)[tid] = g.thread_rank();
    band(arg, CG_BAND_SIZE)[tid] = g.num_threads();
    // Broadcast from the group's own rank 0, which is lane 1, not lane 0.
    band(arg, CG_BAND_SHFL)[tid] = g.shfl(in[tid], 0);
  }
}

// cluster_group. With a cluster of one CTA this checks the arithmetic and,
// more usefully, that the cluster barrier's slot does not collide with the one
// __syncthreads() uses -- a collision there hangs, and looks like hardware.
__global__ void cg_cluster(cg_args* __UNIFORM__ arg) {
  const uint32_t* in = reinterpret_cast<const uint32_t*>(arg->in);
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= arg->threads) return;

  cg::cluster_group cluster = cg::this_cluster();
  uint32_t* smem = grx::shared_memory<uint32_t>();

  smem[threadIdx.x] = in[tid];
  cluster.sync();
  cg::this_thread_block().sync();

  band(arg, CG_BAND_RANK)[tid] = cluster.block_rank();
  band(arg, CG_BAND_SIZE)[tid] = cluster.num_blocks();
  band(arg, CG_BAND_SHFL)[tid] = smem[threadIdx.x];
}

// The grid-wide barrier. Every thread publishes, the whole grid rendezvouses,
// then every thread reads a value another BLOCK published.
//
// Block 0 stalls first, and that is the whole point. Without the delay both
// blocks would publish long before either read, and the test would pass with
// or without a barrier -- a green light that proves nothing. With it, a block
// that does not wait reads the sentinel the host pre-filled.
//
// cg_grid_nosync is the same kernel with the barrier removed. The gate runs it
// and requires it to FAIL, because a control that cannot fail does not
// validate the test above it.

namespace {

__forceinline__ uint32_t stall(uint32_t n) {
  // vx_rdcycle is a CSR read: not foldable, not hoistable out of the loop.
  uint32_t acc = 0;
  for (uint32_t i = 0; i < n; ++i) acc += (uint32_t)vx_rdcycle();
  return acc;
}

template <bool Sync>
__forceinline__ void grid_body(cg_args* arg) {
  const uint32_t* in = reinterpret_cast<const uint32_t*>(arg->in);
  cg::grid_group grid = cg::this_grid();

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= arg->threads) return;

  if (blockIdx.x == 0)
    band(arg, CG_BAND_META)[tid] = stall(2000);

  band(arg, CG_BAND_REDUCE)[tid] = in[tid] + 1u;
  if (Sync) grid.sync();

  const uint32_t peer = (tid + blockDim.x) % arg->threads;
  band(arg, CG_BAND_SHFL)[tid] = band(arg, CG_BAND_REDUCE)[peer];
  band(arg, CG_BAND_RANK)[tid] = grid.thread_rank();
  band(arg, CG_BAND_SIZE)[tid] = grid.num_blocks();
}

}  // namespace

__global__ void cg_grid(cg_args* __UNIFORM__ arg)        { grid_body<true>(arg); }
__global__ void cg_grid_nosync(cg_args* __UNIFORM__ arg) { grid_body<false>(arg); }
