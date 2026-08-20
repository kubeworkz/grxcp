// GRXCP — device-side programming model (layer L2).
//
// Compiled by the device toolchain only. Every construct here is a thin
// binding over a GRX-G100 mechanism that already exists; where one does not,
// the header says so in place rather than inventing an abstraction.
//
// Backing headers from the GRX-G100 kernel sysroot (vortex-kernel.pc):
//   vx_spawn2.h     threadIdx/blockIdx/blockDim/gridDim over VX_CSR_CTA_*
//   vx_intrinsics.h thread mask, split/join/pred, fence, rdcycle
//   vx_barrier.h    barrier / gbarrier, arrive / wait / expect_tx
//   vx_print.h      device-side printf

#ifndef GRX_DEVICE_H
#define GRX_DEVICE_H

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include <vx_print.h>
#include <stdint.h>

// PULLED IN BEFORE THE MACROS BELOW, ON PURPOSE.
//
// This header defines `printf` and `assert` as macros -- which is what lets a
// kernel call them. A macro named `printf` poisons <cstdio> if the standard
// header is parsed AFTERWARDS:
//
//   cstdio:127:11: error: no member named 'printf' in the global namespace
//
// grxcc used to dodge that by choosing where to insert this header, but the
// dodge only works while grxcc controls the order. It does not: a CUDA file
// writing `#include <cooperative_groups.h>` above `#include <cstdio>` pulls
// this header in through the first and hits the poison on the second.
//
// Including them here instead makes the ordering irrelevant. Their include
// guards mean a later `#include <cstdio>` is a no-op, so there is nothing left
// to poison. They cost nothing at -nostdlib: neither declares anything the
// device links against.
#include <cstdio>
#include <cassert>

// ---------------------------------------------------------------------------
// Function qualifiers
// ---------------------------------------------------------------------------
//
// __global__ maps to the `vortex.kernel` annotation that drives the kernel
// calling convention and makes the backend emit the __vx_kentry_<name> alias
// the launch path resolves through the .vxbin VXSYMTAB footer. `used`/`retain`
// keep the body alive: the device dispatches by address, so nothing references
// the kernel statically and --gc-sections would otherwise drop it.

#define __global__   extern "C" __attribute__((annotate("vortex.kernel"), used, retain))
#define __device__   /* ordinary device-target function */
#define __host__     /* host-target function; grxcc dual-emits __host__ __device__ */
#define __forceinline__ __attribute__((always_inline)) inline
#define __restrict__ __restrict

// Shared memory is the CTA's fixed-stride local-memory slot: the base is in
// VX_CSR_CTA_LMEM_ADDR and the size is the launch's sharedMem argument. That
// is CUDA's DYNAMIC shared memory, and grx::shared_memory<T>() below returns
// it.
//
// CUDA's STATIC __shared__ has no equivalent, and the definition that used to
// sit here was worse than not having one: it put the variable in a `.shared`
// section that the device link script does not mention, so the array was
// placed in the ELF image -- in GLOBAL memory. It compiled and it ran, and it
// was not shared memory. That went unnoticed until the first kernel that
// actually used it (the DXA gate) had the DMA engine write into local memory
// while the kernel read the symbol from global memory.
//
// Making it a compile error is the honest replacement: carving a per-CTA
// local-memory slot at link time needs toolchain support that does not exist,
// and there is no way to fake it that a caller would thank us for. See
// docs/designs/cuda_mapping.md.
#define __shared__                                                          \
  __attribute__((unavailable(                                               \
      "GRXCP has no static __shared__: nothing carves a per-CTA local "     \
      "memory slot at link time, so the variable would land in global "     \
      "memory. Use grx::shared_memory<T>() and pass the byte size as the "  \
      "launch's sharedMem argument.")))

// __constant__ currently lowers to read-only global memory: GRX-G100 has no
// exposed broadcast constant path. Semantics are correct, the broadcast
// bandwidth advantage is not there, and grxDeviceProp_t.constantMemoryIsGlobal
// reports it. See docs/designs/cuda_mapping.md section 7.2.
#define __constant__ __attribute__((section(".rodata.grxconst")))

// ---------------------------------------------------------------------------
// Indices and geometry
// ---------------------------------------------------------------------------
// threadIdx / blockIdx / blockDim / gridDim come straight from vx_spawn2.h and
// read the CTA CSRs directly. They are intentionally not redefined here.

// CUDA's built-in `warpSize`, which is a VARIABLE in CUDA and not a constant --
// `for (int o = warpSize / 2; o > 0; o /= 2)` is the canonical shuffle
// reduction, and it has to keep working on a device whose warp is not 32 lanes.
//
// An object with a conversion operator rather than a macro, deliberately.
// `warpSize` is also a member of grxDeviceProp_t, and a macro would rewrite
// `prop.warpSize` in any translation unit that saw both -- a global name does
// not touch member access.
namespace grx { namespace detail {
struct warp_size_t {
  // EXACTLY ONE conversion operator. Adding an `operator unsigned()` as well
  // makes `threadIdx.x % warpSize` ambiguous: threadIdx.x is itself a struct
  // with a user-defined conversion, and with two candidates on this side the
  // compiler cannot choose. CUDA's warpSize is an int, so int it is.
  __attribute__((always_inline)) inline operator int() const {
    return (int)csr_read_nv(VX_CSR_NUM_THREADS);
  }
};
}}  // namespace grx::detail
static const grx::detail::warp_size_t warpSize;

namespace grx {

// lane_id is the hardware lane index within the warp (VX_CSR_THREAD_ID), not
// threadIdx.x % warpSize -- those differ for multi-dimensional blocks.
__forceinline__ uint32_t warp_size()     { return (uint32_t)csr_read_nv(VX_CSR_NUM_THREADS); }
__forceinline__ uint32_t lane_id()       { return (uint32_t)vx_thread_id(); }
__forceinline__ uint32_t warp_id()       { return get_sub_group_id(); }   // VX_CSR_CTA_RANK
__forceinline__ uint32_t warps_per_cta() { return get_num_sub_groups(); } // VX_CSR_CTA_SIZE

// Thread-block clusters (the Hopper-style feature the KMU already implements:
// a cluster's CTAs are emitted contiguously and land in consecutive
// fixed-stride LMEM slots, so a peer's shared base is base + rank*stride).
__forceinline__ uint32_t cluster_size() { return get_cluster_size(); }
__forceinline__ uint32_t cluster_rank() { return get_cluster_rank(); }

__forceinline__ uint64_t clock64() { return (uint64_t)vx_rdcycle(); }

// ---------------------------------------------------------------------------
// The hardware barrier-slot map
// ---------------------------------------------------------------------------
//
// The barrier unit holds VX_CFG_NUM_WARPS * VX_CFG_NUM_BARRIERS slots, and the
// id a kernel passes is a PACKED PAIR, not a slot index:
//
//     raw id = cta_id | (bar_no << 8)      flat slot = cta_id * NUM_BARRIERS + bar_no
//
// Two consequences that are easy to get wrong, and both produce a hang rather
// than an error:
//
//   __syncthreads() uses bar_no 0, so slot cta_id*NUM_BARRIERS is spoken for in
//   every CTA. A kernel picking its own barrier numbers starts at 1.
//
//   The cross-CTA forms -- vortex::group_barrier and vortex::gbarrier -- force
//   cta_id to 0 so every CTA names the same slot. That is what makes them
//   shared, and it also means they land inside CTA 0's range. A cluster
//   barrier on bar_no 1 collides with CTA 0's own grx::barrier(1).
//
// So cooperative groups take the TOP two numbers and kernels count up from 1.

constexpr uint32_t kBarrierCount     = VX_CFG_NUM_BARRIERS;
constexpr uint32_t kClusterBarrierNo = VX_CFG_NUM_BARRIERS - 1;
constexpr uint32_t kGridBarrierNo    = VX_CFG_NUM_BARRIERS - 2;
// The highest number a kernel may use for its own barriers.
constexpr uint32_t kMaxUserBarrierNo = VX_CFG_NUM_BARRIERS - 3;

static_assert(VX_CFG_NUM_BARRIERS >= 3,
              "this configuration has too few barrier slots to reserve one "
              "for the cluster barrier and one for the grid barrier");

// Base of this CTA's shared-memory slot -- CUDA's dynamic shared memory. The
// slot is `sharedMem` bytes, from the launch; nothing on the device knows that
// number, so reading past it silently reads the next CTA's slot. The runtime
// checks the request against grxDeviceProp_t::sharedMemPerBlock at launch,
// which is the only place the check can be made.
__forceinline__ void* shared_memory() { return __local_mem(); }

template <typename T>
__forceinline__ T* shared_memory() {
  return reinterpret_cast<T*>(shared_memory());
}

}  // namespace grx

// ---------------------------------------------------------------------------
// Synchronization
// ---------------------------------------------------------------------------

// __syncthreads() -- and why GRXCP does not use the one from vx_spawn2.h.
//
// Upstream's is `vx_barrier(get_local_group_id(), get_num_sub_groups())`, and
// vx_barrier is a `volatile` inline asm. Volatile stops the optimizer from
// DELETING or REORDERING the barrier. It does not stop it from DUPLICATING it,
// and on a machine with a split/join reconvergence stack that difference is a
// deadlock.
//
// Observed, in the first program grxcc ever compiled:
//
//     if (i < n) s[t] = in[i];
//     __syncthreads();
//     if (i < n) out[i] = s[blockDim.x - 1 - t];
//
// LLVM tail-duplicated the code after the first branch, so the barrier landed
// on BOTH sides of it:
//
//     vx_split_n a0, a7
//     beqz  a7, .else
//       ...            vx_bar a5, a7      <-- copy 1
//       j .join
//     .else:
//       vx_bar a1, a2                     <-- copy 2
//     .join:
//     vx_join a0
//
// A warp whose lanes all agree executes one of the two and arrives once. A
// DIVERGED warp runs both sides of the split and arrives TWICE, so a CTA of two
// warps posts three arrivals against a barrier expecting two: the first two
// release, the third opens a new generation nobody else will ever join, and the
// kernel hangs. It only shows up when a warp actually diverges, which is why a
// grid that divides evenly passes and a ragged one hangs -- the failure is
// data-dependent and silent.
//
// The fix is to tell the optimizer what the asm cannot: `convergent` forbids
// making the call more or less conditional than the source made it, and
// `noduplicate` forbids copying it.
//
// Measured, by counting vx_bar in the kernel above at each setting:
//
//     (no attribute)                        2   <-- the bug
//     convergent                            1
//     noduplicate                           1
//     noinline, convergent, noduplicate     1
//
// Either attribute alone is sufficient on this compiler; both are kept because
// which one an optimizer respects is not a promise anyone has made. `noinline`
// is carried for the same reason and is NOT what makes this work -- the wrapper
// is inlined anyway at -O3, and the emitted barrier is correct regardless.
//
// This is a workaround for a toolchain gap, not a fix: every other caller of
// vx_barrier in the grxgpu tree has the same exposure. See cuda_mapping.md
// section 7.20 and tests/repro/barrier_duplication/.

__attribute__((noinline, convergent, noduplicate))
inline void __grx_cta_barrier(int bar_id, int num_warps) {
  vx_barrier(bar_id, num_warps);
}

#undef __syncthreads
#define __syncthreads()                                     \
  __grx_cta_barrier((int)get_local_group_id(), (int)get_num_sub_groups())

// Within a warp the SIMT pipeline is lockstep, so __syncwarp is a compiler
// scheduling fence, not a hardware barrier. It still matters: it prevents the
// compiler from sinking a shared-memory write past a dependent read.
#define __syncwarp(...)  __asm__ __volatile__("" ::: "memory")

#define __threadfence_block()  vx_fence()
#define __threadfence()        vx_fence()
#define __threadfence_system() vx_fence()

// ---------------------------------------------------------------------------
// Device-side I/O and assertions
// ---------------------------------------------------------------------------

#define printf vx_printf

// <cassert> was pulled in above, so its macro is already here; replacing it is
// the intent, and #undef says so instead of leaving a redefinition warning in
// every kernel compile.
#undef assert
#ifdef NDEBUG
#define assert(x) ((void)0)
#else
#define assert(x)                                                        \
  do {                                                                   \
    if (!(x)) {                                                          \
      vx_printf("assert failed: %s at %s:%d (block %u thread %u)\n",      \
                #x, __FILE__, __LINE__,                                   \
                (unsigned)blockIdx.x, (unsigned)threadIdx.x);             \
      __builtin_trap();                                                   \
    }                                                                    \
  } while (0)
#endif

#endif  // GRX_DEVICE_H
