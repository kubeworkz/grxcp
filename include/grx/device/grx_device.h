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

// Static __shared__ carves from this CTA's fixed-stride local-memory slot.
// The base address comes from VX_CSR_CTA_LMEM_ADDR; the slot's size is the
// per-kernel `lmem_size` programmed at launch.
#define __shared__   __attribute__((section(".shared")))

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

}  // namespace grx

// ---------------------------------------------------------------------------
// Synchronization
// ---------------------------------------------------------------------------

// __syncthreads() comes from vx_spawn2.h as vx_barrier(cta_id, num_warps).

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
