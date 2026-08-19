// GRXCP device kernel for the DXA gate: stage one tile of a 2D array into
// shared memory with the DMA engine, then write it back to global memory so
// the host can check what actually landed there.
//
// Copying it straight back out is the point. The gate is not asking whether
// the kernel can compute something; it is asking whether the bytes the engine
// deposited in shared memory are the bytes the descriptor described.
//
// The staging buffer is the CTA's shared-memory slot, sized by the launch's
// sharedMem argument -- there is no static __shared__ on this platform, see
// include/grx/device/grx_device.h.

#include <grx/device/grx_pipeline.h>

#include "common.h"

__global__ void dxa_copy_tile(dxa_args* __UNIFORM__ arg) {
  uint32_t*      out       = reinterpret_cast<uint32_t*>(arg->out);
  uint32_t*      smem_tile = grx::shared_memory<uint32_t>();
  const uint32_t n         = arg->tile0 * arg->tile1;

  // Poison first: a transfer that never lands would otherwise be indis-
  // tinguishable from one that landed correctly on a previous launch, since
  // shared memory keeps whatever the last CTA in this slot left behind.
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x)
    smem_tile[i] = 0xdeadbeefu;
  __syncthreads();

  grx::barrier    bar(arg->barrier);
  grx::tensor_map map(arg->slot);

  // One warp issues; every warp waits. expect_tx has to precede the issue --
  // a transfer that completes before its count is registered leaves the
  // barrier waiting for something that already happened.
  if (grx::warp_id() == 0) {
    bar.expect_tx(1);
    grx::memcpy_async(smem_tile, map, arg->coord0, arg->coord1, bar);
  }
  bar.arrive_and_wait();

  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) out[i] = smem_tile[i];
}
