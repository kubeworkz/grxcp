// GRXCP — asynchronous tile copy and transaction barriers.
//
// Backed by DXA (the async data-transfer accelerator) and the barrier
// transaction-count support in the GRX-G100 kernel API:
//
//   vx_dxa_issue_{1..5}d_wg           global -> shared async tile copy
//   vx_barrier_expect_tx(bar, count)  pre-register pending transactions
//   vx_barrier_arrive / vx_barrier_wait
//
// This is the cp.async.bulk.tensor / TMA family, and the shape follows from
// what the engine actually consumes.
//
// WHY THERE IS NO POINTER-TO-POINTER memcpy_async
//
// `cuda::memcpy_async(dst, src, bytes, barrier)` copies an arbitrary byte
// range. DXA does not do that: it copies a TILE of an array whose layout was
// described in advance by a descriptor the HOST programmed into a device slot
// (see grx_tensormap.h). A kernel names the slot and the tile coordinates; it
// cannot describe a new region on the fly.
//
// So the pointer-based form is absent rather than emulated. Emulating it means
// a plain load/store loop, which is exactly what the caller was trying not to
// write -- it would occupy the very warps the async copy exists to free, while
// carrying a name that promises otherwise. Code that wants a byte-range copy
// should write the loop and see it.
//
// SEQUENCE, which the hardware does not enforce for you
//
//   1. bar.expect_tx(n)      -- BEFORE the issue, from the issuing warp
//   2. memcpy_async(...)     -- one warp issues
//   3. bar.arrive_and_wait() -- EVERY warp of the CTA, including the issuer
//
// expect_tx after the issue is a race: a fast transfer can complete before the
// count is registered, and the barrier then waits for a completion that has
// already happened.

#ifndef GRX_PIPELINE_H
#define GRX_PIPELINE_H

#include "grx_device.h"
#include <vx_dxa.h>
#include <vx_barrier.h>

// Same backstop as grx_wmma.h: including this header is a statement that the
// kernel uses the DMA engine, so a configuration without one is a build error
// rather than a silently different kernel. See ci/README.md, "configuration
// provenance".
#if !defined(VX_CFG_EXT_DXA_ENABLED)
#error "grx_pipeline.h: no device configuration. Compile with \
ci/build_kernel.sh, which resolves the configuration from the installed sysroot."
#elif !VX_CFG_EXT_DXA_ENABLED
#error "grx_pipeline.h: this device configuration has no DMA engine. Rebuild \
the sysroot with ci/build_sysroot.sh --configs \"-DVX_CFG_EXT_DXA_ENABLE\", or \
do not include this header."
#endif

namespace grx {

// A barrier that counts arrivals AND asynchronous transactions -- the mbarrier
// shape. `id` selects one of the CTA's hardware barrier slots (the device
// reports how many as grxDeviceProp_t::numBarriers); picking them is the
// kernel's job, the same way shared-memory offsets are.
class barrier {
 public:
  __forceinline__ explicit barrier(uint32_t id,
                                   uint32_t num_warps = get_num_sub_groups())
      : bar_(id, num_warps) {}

  // Register `count` transactions that have not been issued yet. Cumulative.
  __forceinline__ void expect_tx(uint32_t count = 1) { bar_.expect_tx(count); }

  // Arrive without blocking; the returned phase is the token wait() consumes.
  __forceinline__ uint32_t arrive() { return bar_.arrive(); }
  __forceinline__ void wait(uint32_t phase) { bar_.wait(phase); }

  // Blocks until every warp has arrived AND every expected transaction has
  // completed. This is the call that makes an async copy visible.
  __forceinline__ void arrive_and_wait() { bar_.arrive_and_wait(); }

  // Packed id, as the DXA issue instructions want it.
  __forceinline__ uint32_t id() const { return bar_.id(); }

 private:
  vortex::barrier bar_;
};

// A descriptor slot programmed by the host. This carries no layout information
// on purpose: the device cannot see what is in the slot, and a kernel-side copy
// of the geometry would be a second source of truth that nothing checks.
class tensor_map {
 public:
  __forceinline__ explicit tensor_map(uint32_t slot) : slot_(slot) {}
  __forceinline__ uint32_t slot() const { return slot_; }

 private:
  uint32_t slot_;
};

// Fetch the tile at `coord` into this CTA's shared memory. One warp issues;
// completion arrives as a transaction on `bar`. See the sequence note above.
//
// smem_dst is an address in this CTA's shared-memory slot -- typically
// grx::shared_memory<T>(), or an offset into it. It must be a REAL local
// memory address: the engine writes straight to the local-memory port, so a
// pointer into global memory is accepted and silently lands somewhere else.
// (That is not hypothetical -- it is how this gate found that GRXCP's old
// static __shared__ was placing arrays in the ELF image.)
__forceinline__ void memcpy_async(void* smem_dst, const tensor_map& map,
                                  uint32_t coord0, barrier& bar) {
  vx_dxa_issue_1d_wg(map.slot(), bar.id(), smem_dst, coord0);
}

__forceinline__ void memcpy_async(void* smem_dst, const tensor_map& map,
                                  uint32_t coord0, uint32_t coord1,
                                  barrier& bar) {
  vx_dxa_issue_2d_wg(map.slot(), bar.id(), smem_dst, coord0, coord1);
}

// ---------------------------------------------------------------------------
// Not here yet
// ---------------------------------------------------------------------------
//
//   rank 3-5 tiles      vx_dxa_issue_{3,4,5}d_wg exist and are one-line
//                       forwards, but grxTensorMapProgramAsync only programs
//                       rank 1 and 2, so a device call for them could not be
//                       pointed at anything. The two sides move together.
//   multicast           vx_dxa_issue_Nd_multicast_wg fills every CTA of a
//                       cluster from one issue. It needs a cluster launch, a
//                       second (group) barrier for the rendezvous, and the
//                       dispatcher's co-residency contract; vortex::dxa_multicast_2d
//                       in vx_dxa.h already wraps the idiom correctly. Wrapping
//                       it here without a cluster-launch gate would be shipping
//                       an untested concurrency protocol.
//   grid_barrier        the global barrier unit, for cooperative launches.
//   pipeline<Stages>    the multi-stage prologue/steady-state/epilogue
//                       structure. Its shape should be decided by the blocked
//                       GEMM that will be its first user, not guessed at first.

}  // namespace grx

#endif  // GRX_PIPELINE_H
