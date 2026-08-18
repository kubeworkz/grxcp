// GRXCP — asynchronous copy and transaction barriers.
//
// Backed by DXA (the async data-transfer accelerator) and the barrier
// transaction-count support that already exists in the GRX-G100 kernel API:
//
//   vx_dxa_issue_{1..5}d_wg           global -> shared async copy, up to 5D
//   vx_dxa_issue_{1..5}d_multicast_wg one issuer fills K cluster peers' shared
//   vx_barrier_expect_tx(bar, count)  pre-register pending transactions
//   vx_barrier_arrive / vx_barrier_wait
//
// This is the cuda::pipeline / cp.async / TMA family, and the GRX-G100 version
// is in places broader: descriptors go to 5 dimensions, and multicast is exact
// because a cluster's CTAs occupy consecutive fixed-stride local-memory slots
// (peer base = base + rank * stride), so no per-slot base table is needed.

#ifndef GRX_PIPELINE_H
#define GRX_PIPELINE_H

#include "grx_device.h"
#include <vx_dxa.h>

namespace grx {

// A barrier that counts both arrivals and asynchronous transactions -- the
// mbarrier shape. Producers call expect_tx() BEFORE issuing the copy so that
// non-issuing warps know how many completions to wait for.
class barrier {
 public:
  __device__ explicit barrier(uint32_t id, uint32_t num_warps);
  __device__ void expect_tx(uint32_t count = 1);
  __device__ void arrive();
  __device__ void wait();
  __device__ void arrive_and_wait();
  __device__ uint32_t id() const;
};

// Grid-wide barrier for cooperative launches, over the global barrier unit.
class grid_barrier {
 public:
  __device__ void sync();
};

// --- async copy -----------------------------------------------------------

// Copy `bytes` from global to this CTA's shared memory. Completion is signaled
// as a transaction on `bar`; the caller must have called bar.expect_tx() first.
__device__ void memcpy_async(void* smem_dst, const void* gmem_src,
                             size_t bytes, barrier& bar);

// Strided forms, mapping onto the DXA 2D-5D descriptors.
__device__ void memcpy_async_2d(void* smem_dst, const void* gmem_src,
                                size_t row_bytes, size_t rows,
                                size_t src_pitch, barrier& bar);
__device__ void memcpy_async_3d(void* smem_dst, const void* gmem_src,
                                const size_t extent[3], const size_t pitch[2],
                                barrier& bar);

// Multicast: one issuer fills the shared memory of every CTA in its cluster.
// Requires a cluster launch (grxLaunchKernelEx with a cluster dimension).
__device__ void memcpy_async_multicast(void* smem_dst, const void* gmem_src,
                                       size_t bytes, barrier& bar);

// --- pipeline -------------------------------------------------------------
// Multi-stage software pipeline over the barriers above: the standard
// prologue/steady-state/epilogue GEMM structure.

template <int Stages>
class pipeline {
 public:
  __device__ void producer_acquire();
  __device__ void producer_commit();
  __device__ void consumer_wait();
  __device__ void consumer_release();
};

}  // namespace grx

#endif  // GRX_PIPELINE_H
