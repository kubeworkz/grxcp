// Warp primitives on the device: shuffle, vote, and the two of them sharing a
// CTA with the caller's shared memory.
//
// The shuffle is a hardware instruction here -- SHFL.UP / DOWN / BFLY / IDX,
// with NVIDIA's segmented semantics. The gate is not about whether the
// instruction works; it is about whether GRXCP's `width` argument lands on the
// right segment mask and clamp. Get that wrong and a reduction written for a
// half-warp silently sums the wrong lanes, which is the kind of bug that
// survives a lot of testing.

#include <grx/device/grx_warp.h>

#include "common.h"

// The phase 2 exit-gate kernel: a warp sums its lanes with __shfl_down_sync.
__global__ void warp_reduce(warp_args* __UNIFORM__ arg) {
  const uint32_t* in  = reinterpret_cast<const uint32_t*>(arg->in);
  uint32_t*       out = reinterpret_cast<uint32_t*>(arg->out);

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= arg->threads) return;

  uint32_t v = in[tid];
  for (uint32_t d = grx::warp_size() / 2; d > 0; d >>= 1)
    v += __shfl_down_sync(GRX_FULL_MASK, v, d);

  // One result per warp, from the lane that holds the total.
  if (grx::lane_id() == 0) out[tid / grx::warp_size()] = v;
}

// Every shuffle form, at whatever segment width the host asks for. Written as
// four separate output arrays so a wrong one is identifiable rather than just
// a wrong sum.
__global__ void warp_modes(warp_args* __UNIFORM__ arg) {
  const uint32_t* in  = reinterpret_cast<const uint32_t*>(arg->in);
  uint32_t*       out = reinterpret_cast<uint32_t*>(arg->out);

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= arg->threads) return;
  const uint32_t n = arg->threads;
  const int      w = (int)arg->width;

  const uint32_t v = in[tid];
  out[0 * n + tid] = __shfl_sync(GRX_FULL_MASK, v, 0, w);
  out[1 * n + tid] = __shfl_up_sync(GRX_FULL_MASK, v, 1, w);
  out[2 * n + tid] = __shfl_down_sync(GRX_FULL_MASK, v, 1, w);
  out[3 * n + tid] = __shfl_xor_sync(GRX_FULL_MASK, v, 1, w);
}

// The vote family, which is native: one CSR read.
__global__ void warp_vote(warp_args* __UNIFORM__ arg) {
  uint32_t* out = reinterpret_cast<uint32_t*>(arg->out);

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= arg->threads) return;

  const uint32_t lane = grx::lane_id();
  const unsigned active = __activemask();
  const unsigned odd    = __ballot_sync(GRX_FULL_MASK, (int)(lane & 1u));
  const int      any0   = __any_sync(GRX_FULL_MASK, (int)(lane == 0));
  const int      all    = __all_sync(GRX_FULL_MASK, 1);

  if (lane == 0) {
    uint32_t* w = out + 4u * (tid / grx::warp_size());
    w[0] = active;
    w[1] = odd;
    w[2] = (uint32_t)any0;
    w[3] = (uint32_t)all;
  }
}

// Shuffle and shared memory in one kernel. The emulated shuffle this replaced
// staged through a scratch region that started at the same address
// grx::shared_memory() returns, so a kernel doing both corrupted its own data
// -- silently, and only in kernels that did both. The instruction touches no
// memory at all now, and this stays as the check that says so.
__global__ void warp_shared_coexist(warp_args* __UNIFORM__ arg) {
  uint32_t* out = reinterpret_cast<uint32_t*>(arg->out);
  uint32_t* aux = reinterpret_cast<uint32_t*>(arg->aux);

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= arg->threads) return;

  // Each thread owns one shared cell and fills it with something recognisable.
  uint32_t* smem = grx::shared_memory<uint32_t>();
  smem[threadIdx.x] = 0xc0de0000u + tid;
  __syncthreads();

  // Now shuffle, which writes the scratch region. If the two overlap, the
  // shared cells below come back as shuffle scratch instead of the pattern.
  const uint32_t shuffled =
      __shfl_sync(GRX_FULL_MASK, 0xa000u + grx::lane_id(), 0);
  __syncthreads();

  out[tid] = shuffled;
  aux[tid] = smem[threadIdx.x];
}
