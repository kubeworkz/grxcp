// GRXCP — cooperative groups (layer L2).
//
// CUDA's `cooperative_groups` namespace, over the mechanisms GRX-G100 actually
// has: the CTA barrier unit, the cluster-wide global barrier, the native warp
// shuffle and vote instructions, and the thread mask.
//
// Three things about this header differ from CUDA, and they differ because the
// hardware differs. Each is a compile error rather than a silent
// approximation:
//
//   Tile width. `thread_block_tile<32>` is universal in CUDA because every
//   CUDA warp is 32 lanes. This build's warp is VX_CFG_NUM_THREADS lanes, so a
//   tile wider than that cannot exist -- the shuffle segment it would need is
//   not a thing the instruction can express. The template checks, the way
//   grx::wmma::fragment checks its shape (cuda_mapping.md section 7.9).
//
//   Grid barrier scope. The hardware's "global" barrier releases per CLUSTER:
//   the cluster aggregates one arrival per core and compares against the
//   participant count, and there is no cross-cluster stage above it. On a
//   single-cluster device that is exactly a grid barrier. On a multi-cluster
//   device it is not, and grid_group::sync() is unavailable rather than
//   deadlocking (cuda_mapping.md section 7.17).
//
//   map_shared_rank. A cluster's CTAs land in consecutive fixed-stride
//   local-memory slots, so a peer's shared base is base + delta*stride -- but
//   nothing exposes the stride to the kernel. The two-argument CUDA form is
//   unavailable; the three-argument form takes the stride the caller already
//   knows from its launch.
//
// Backing headers: vx_spawn2.h (indices, __syncthreads), vx_barrier.h
// (barrier / gbarrier / group_barrier), grx_warp.h (shuffle, vote).

#ifndef GRX_CG_H
#define GRX_CG_H

#include <grx/device/grx_device.h>
#include <grx/device/grx_warp.h>

#include <vx_barrier.h>

namespace grx {
namespace cg {

// The barrier numbers these groups use come from grx::kClusterBarrierNo and
// grx::kGridBarrierNo in grx_device.h, which is also where the slot encoding
// is explained. The short version: the cross-CTA barrier forms force the CTA
// field of the id to zero so every CTA names the same slot, which puts them
// inside CTA 0's range -- so cooperative groups take the top two numbers and a
// kernel picking its own counts up from 1.

namespace detail {

// The cross-CTA barriers go through the same convergent wrapper as
// __syncthreads(), and for the same reason: vortex::group_barrier and
// vortex::gbarrier bottom out in the same unmarked `volatile` inline asm, so
// an optimizer is free to duplicate them across a divergent branch and hang
// the kernel. grx_device.h has the disassembly of the failure.
//
// This is the only place these two forms are called from, so wrapping here
// covers cooperative groups entirely.
__attribute__((noinline, convergent, noduplicate))
inline void group_barrier_wait(uint32_t bar_no, uint32_t participants) {
  vortex::group_barrier bar(bar_no, participants);
  bar.arrive_and_wait();
}

__attribute__((noinline, convergent, noduplicate))
inline void grid_barrier_wait(uint32_t bar_no, uint32_t participants) {
  vortex::gbarrier bar(bar_no, participants);
  bar.arrive_and_wait();
}

}  // namespace detail

// ---------------------------------------------------------------------------
// thread_block
// ---------------------------------------------------------------------------

struct dim3_index {
  uint32_t x, y, z;
};

class thread_block {
public:
  __forceinline__ void sync() const { __syncthreads(); }

  __forceinline__ uint32_t num_threads() const {
    return (uint32_t)blockDim.x * (uint32_t)blockDim.y * (uint32_t)blockDim.z;
  }
  __forceinline__ uint32_t size() const { return num_threads(); }

  __forceinline__ uint32_t thread_rank() const {
    return (uint32_t)threadIdx.x +
           (uint32_t)blockDim.x * ((uint32_t)threadIdx.y +
                                   (uint32_t)blockDim.y * (uint32_t)threadIdx.z);
  }

  __forceinline__ dim3_index group_index() const {
    return dim3_index{(uint32_t)blockIdx.x, (uint32_t)blockIdx.y,
                      (uint32_t)blockIdx.z};
  }
  __forceinline__ dim3_index thread_index() const {
    return dim3_index{(uint32_t)threadIdx.x, (uint32_t)threadIdx.y,
                      (uint32_t)threadIdx.z};
  }
  __forceinline__ dim3_index dim_threads() const {
    return dim3_index{(uint32_t)blockDim.x, (uint32_t)blockDim.y,
                      (uint32_t)blockDim.z};
  }
};

__forceinline__ thread_block this_thread_block() { return thread_block(); }

// ---------------------------------------------------------------------------
// Reduction operators
// ---------------------------------------------------------------------------
//
// Deliberately not std::plus and friends: the device build is -nostdlib and
// these have to be usable without any of it.

template <typename T> struct plus    { __forceinline__ T operator()(T a, T b) const { return a + b; } };
template <typename T> struct multiply{ __forceinline__ T operator()(T a, T b) const { return a * b; } };
template <typename T> struct less    { __forceinline__ T operator()(T a, T b) const { return (b < a) ? b : a; } };
template <typename T> struct greater { __forceinline__ T operator()(T a, T b) const { return (a < b) ? b : a; } };
template <typename T> struct bit_and { __forceinline__ T operator()(T a, T b) const { return a & b; } };
template <typename T> struct bit_or  { __forceinline__ T operator()(T a, T b) const { return a | b; } };
template <typename T> struct bit_xor { __forceinline__ T operator()(T a, T b) const { return a ^ b; } };

// ---------------------------------------------------------------------------
// thread_block_tile<Size>
// ---------------------------------------------------------------------------
//
// A tile is a power-of-two slice of one warp, which is what makes its sync a
// compiler fence rather than a hardware barrier: the lanes of a warp advance in
// lockstep, so they are already synchronized and the only hazard is the
// compiler reordering a shared-memory write past a dependent read.
//
// Ranks are hardware lane ranks within the segment, not threadIdx arithmetic.
// Those differ the moment a block is multi-dimensional, and a reduction that
// mixes them sums the wrong lanes.

template <unsigned Size>
class thread_block_tile {
  static_assert(Size > 0 && (Size & (Size - 1)) == 0,
                "a cooperative-groups tile must be a power of two");
  static_assert(Size <= VX_CFG_NUM_THREADS,
                "this build's warp is VX_CFG_NUM_THREADS lanes and a tile "
                "cannot be wider than the warp it slices: the shuffle "
                "instruction's segment mask cannot express it. Query "
                "grxDeviceProp_t::warpSize on the host and size the tile from "
                "the device, the way tests/kernels/warp does.");

public:
  static constexpr unsigned tile_size = Size;

  __forceinline__ void sync() const { __syncwarp(); }

  __forceinline__ uint32_t num_threads() const { return Size; }
  __forceinline__ uint32_t size() const { return Size; }

  __forceinline__ uint32_t thread_rank() const {
    return grx::lane_id() & (Size - 1u);
  }
  // Which tile this is, among the tiles of the warp it was partitioned from.
  __forceinline__ uint32_t meta_group_rank() const {
    return grx::lane_id() / Size;
  }
  __forceinline__ uint32_t meta_group_size() const {
    return grx::warp_size() / Size;
  }

  template <typename T>
  __forceinline__ T shfl(T v, uint32_t src_rank) const {
    return __shfl_sync(GRX_FULL_MASK, v, (int)src_rank, (int)Size);
  }
  template <typename T>
  __forceinline__ T shfl_up(T v, uint32_t delta) const {
    return __shfl_up_sync(GRX_FULL_MASK, v, delta, (int)Size);
  }
  template <typename T>
  __forceinline__ T shfl_down(T v, uint32_t delta) const {
    return __shfl_down_sync(GRX_FULL_MASK, v, delta, (int)Size);
  }
  template <typename T>
  __forceinline__ T shfl_xor(T v, uint32_t lane_mask) const {
    return __shfl_xor_sync(GRX_FULL_MASK, v, (int)lane_mask, (int)Size);
  }

  __forceinline__ int any(int predicate) const {
    return (ballot(predicate) != 0) ? 1 : 0;
  }
  __forceinline__ int all(int predicate) const {
    const uint32_t full = (Size >= 32) ? 0xffffffffu : ((1u << Size) - 1u);
    return (ballot(predicate) == full) ? 1 : 0;
  }
  // Ballot over the TILE, not the warp: the warp-wide bits are shifted down so
  // bit i is tile rank i, which is what a caller who asked for a tile means.
  __forceinline__ uint32_t ballot(int predicate) const {
    const uint32_t warp_bits = __ballot_sync(GRX_FULL_MASK, predicate);
    const uint32_t base      = grx::lane_id() & ~(Size - 1u);
    const uint32_t mask      = (Size >= 32) ? 0xffffffffu : ((1u << Size) - 1u);
    return (warp_bits >> base) & mask;
  }

  // Butterfly reduction: every lane ends with the result, which is CUDA's
  // contract for cg::reduce and is why it is an xor shuffle rather than a
  // down shuffle.
  template <typename T, typename Op>
  __forceinline__ T reduce(T value, Op op) const {
    for (unsigned d = Size >> 1; d > 0; d >>= 1)
      value = op(value, shfl_xor(value, d));
    return value;
  }

  // Hillis-Steele. The rank guard is not optional: a shuffle whose source
  // falls outside the segment returns the caller's OWN value, so an unguarded
  // add would fold a lane's value into itself at the segment edge.
  template <typename T, typename Op>
  __forceinline__ T inclusive_scan(T value, Op op) const {
    const uint32_t rank = thread_rank();
    for (unsigned d = 1; d < Size; d <<= 1) {
      const T other = shfl_up(value, d);
      if (rank >= d) value = op(other, value);
    }
    return value;
  }

  template <typename T, typename Op>
  __forceinline__ T exclusive_scan(T value, Op op, T identity) const {
    const T inc = inclusive_scan(value, op);
    const T sh  = shfl_up(inc, 1);
    return (thread_rank() == 0) ? identity : sh;
  }

  template <typename T>
  __forceinline__ T reduce(T value) const { return reduce(value, plus<T>()); }
  template <typename T>
  __forceinline__ T inclusive_scan(T value) const {
    return inclusive_scan(value, plus<T>());
  }
  template <typename T>
  __forceinline__ T exclusive_scan(T value) const {
    return exclusive_scan(value, plus<T>(), T(0));
  }
};

template <unsigned Size>
__forceinline__ thread_block_tile<Size> tiled_partition(const thread_block&) {
  return thread_block_tile<Size>();
}

// ---------------------------------------------------------------------------
// coalesced_group
// ---------------------------------------------------------------------------
//
// The lanes of this warp that are converged here, ranked among themselves.
// CUDA's use for it is inside divergent code: the lanes that took this branch
// can still cooperate, and their ranks are dense whatever the mask looks like.
//
// The mask is read once at construction. That is deliberate and it matters: if
// it were re-read inside each collective, a divergence between two calls would
// silently change the group's membership underneath the caller.

class coalesced_group {
public:
  __forceinline__ coalesced_group() : mask_(__activemask()) {}

  __forceinline__ uint32_t num_threads() const {
    return (uint32_t)__popc(mask_);
  }
  __forceinline__ uint32_t size() const { return num_threads(); }

  // Rank among the active lanes: how many active lanes are below this one.
  __forceinline__ uint32_t thread_rank() const {
    const uint32_t below = mask_ & ((1u << grx::lane_id()) - 1u);
    return (uint32_t)__popc(below);
  }

  __forceinline__ uint32_t mask() const { return mask_; }

  __forceinline__ void sync() const { __syncwarp(); }

  // Rank -> lane. The n-th set bit of the mask, found by clearing the lowest
  // set bit n times. Bounded by the warp width, and the compiler unrolls it
  // for the widths this build uses.
  __forceinline__ uint32_t lane_of_rank(uint32_t rank) const {
    uint32_t m = mask_;
    for (uint32_t i = 0; i < rank && m; ++i) m &= (m - 1u);
    return m ? (uint32_t)(__ffs((int)m) - 1) : grx::lane_id();
  }

  template <typename T>
  __forceinline__ T shfl(T v, uint32_t src_rank) const {
    return __shfl_sync(GRX_FULL_MASK, v, (int)lane_of_rank(src_rank),
                       (int)grx::warp_size());
  }

  __forceinline__ uint32_t ballot(int predicate) const {
    // Only the group's own lanes, compacted so bit i is rank i.
    const uint32_t bits = __ballot_sync(GRX_FULL_MASK, predicate) & mask_;
    uint32_t out = 0, m = mask_;
    for (uint32_t r = 0; m; ++r) {
      const uint32_t lane = (uint32_t)(__ffs((int)m) - 1);
      if (bits & (1u << lane)) out |= (1u << r);
      m &= (m - 1u);
    }
    return out;
  }
  __forceinline__ int any(int predicate) const {
    return ((__ballot_sync(GRX_FULL_MASK, predicate) & mask_) != 0) ? 1 : 0;
  }
  __forceinline__ int all(int predicate) const {
    const uint32_t bits = __ballot_sync(GRX_FULL_MASK, predicate) & mask_;
    return (bits == mask_) ? 1 : 0;
  }

private:
  uint32_t mask_;
};

__forceinline__ coalesced_group coalesced_threads() { return coalesced_group(); }

// ---------------------------------------------------------------------------
// cluster_group
// ---------------------------------------------------------------------------
//
// A CTA cluster: consecutive CTAs emitted together by the KMU, landing in
// consecutive fixed-stride local-memory slots on ONE core. That last part is
// what makes the cluster barrier a plain CTA-barrier slot rather than anything
// global -- the barrier unit is per core, and a cluster never spans one.

class cluster_group {
public:
  __forceinline__ uint32_t num_blocks() const { return grx::cluster_size(); }
  __forceinline__ uint32_t block_rank() const { return grx::cluster_rank(); }

  __forceinline__ uint32_t num_threads() const {
    return num_blocks() * (uint32_t)blockDim.x * (uint32_t)blockDim.y *
           (uint32_t)blockDim.z;
  }
  __forceinline__ uint32_t size() const { return num_threads(); }

  __forceinline__ uint32_t thread_rank() const {
    const uint32_t per_block =
        (uint32_t)blockDim.x * (uint32_t)blockDim.y * (uint32_t)blockDim.z;
    return block_rank() * per_block + this_thread_block().thread_rank();
  }

  // A cluster of one CTA is that CTA, and __syncthreads() is both correct and
  // cheaper. It is also the only correct choice: the shared barrier slot is
  // named identically by every CTA on the core, so with one-CTA clusters two
  // co-resident CTAs would rendezvous with each other instead of within
  // themselves. The wider form is right only when a core holds at most one
  // cluster, which is what a cluster launch asks the dispatcher for.
  //
  // The branch is uniform across the CTA -- cluster size is a launch property
  // -- so it does not diverge.
  __forceinline__ void sync() const {
    const uint32_t n = num_blocks();
    if (n <= 1) { __syncthreads(); return; }
    detail::group_barrier_wait(grx::kClusterBarrierNo,
                               grx::warps_per_cta() * n);
  }

  // A peer CTA's shared-memory base.
  //
  // `stride` is the per-CTA local-memory stride: the launch's sharedMem
  // request plus any per-kernel need, rounded up to the cache-line granule.
  // The caller has to supply it because nothing on the device does:
  // VX_CSR_CTA_LMEM_ADDR gives this CTA's base and there is no companion
  // register for the stride. Passing the wrong number reads another CTA's
  // memory silently, so the argument is required rather than defaulted.
  __forceinline__ void* map_shared_rank(void* addr, uint32_t rank,
                                        uint32_t stride) const {
    const int64_t delta = (int64_t)rank - (int64_t)block_rank();
    return (void*)((int8_t*)addr + delta * (int64_t)stride);
  }

  // The CUDA spelling, which cannot work here.
  __attribute__((unavailable(
      "GRXCP cannot infer the per-CTA local-memory stride: "
      "VX_CSR_CTA_LMEM_ADDR gives this CTA's base and nothing gives the "
      "stride. Use map_shared_rank(addr, rank, stride) with the launch's "
      "sharedMem rounded up to the cache-line granule.")))
  void* map_shared_rank(void* addr, uint32_t rank) const;
};

__forceinline__ cluster_group this_cluster() { return cluster_group(); }

// ---------------------------------------------------------------------------
// grid_group
// ---------------------------------------------------------------------------

class grid_group {
public:
  __forceinline__ uint32_t num_blocks() const {
    return (uint32_t)gridDim.x * (uint32_t)gridDim.y * (uint32_t)gridDim.z;
  }
  __forceinline__ uint32_t block_rank() const {
    return (uint32_t)blockIdx.x +
           (uint32_t)gridDim.x * ((uint32_t)blockIdx.y +
                                  (uint32_t)gridDim.y * (uint32_t)blockIdx.z);
  }
  __forceinline__ uint32_t num_threads() const {
    return num_blocks() * this_thread_block().num_threads();
  }
  __forceinline__ uint32_t size() const { return num_threads(); }
  __forceinline__ uint32_t thread_rank() const {
    return block_rank() * this_thread_block().num_threads() +
           this_thread_block().thread_rank();
  }

  // Whether sync() will terminate.
  //
  // CUDA answers "was this a cooperative launch". This answers the structural
  // question as well, because the hardware barrier is cluster-scoped: a
  // multi-cluster device has no barrier that covers a grid. The launch half of
  // the promise is the host's -- grxLaunchCooperativeKernel refuses a grid
  // that does not fit, and one that does not cover every core.
  __forceinline__ bool is_valid() const { return VX_CFG_NUM_CLUSTERS == 1; }

#if VX_CFG_NUM_CLUSTERS == 1
  // Every warp of every resident CTA must arrive. The barrier unit counts the
  // core's active warps itself and forwards one arrival per core to the
  // cluster, which releases when every core has arrived -- so the launch must
  // put at least one CTA on every core, and the whole grid must be resident.
  //
  // VX_CFG_NUM_CORES is cores per CLUSTER, which is the right participant
  // count precisely because the release stage is the cluster's. The total
  // across the device is VX_CSR_NUM_CORES, and using it here would wait for
  // arrivals the cluster's own mask cannot even represent.
  __forceinline__ void sync() const {
    detail::grid_barrier_wait(grx::kGridBarrierNo, VX_CFG_NUM_CORES);
  }
#else
  __attribute__((unavailable(
      "GRX-G100's global barrier releases per CLUSTER: the cluster aggregates "
      "one arrival per core and there is no stage above it, so on a "
      "multi-cluster device a grid-wide barrier has nothing to implement it. "
      "Restructure as separate kernel launches, or keep the cooperating work "
      "inside one cluster and use grx::cg::this_cluster(). See "
      "docs/designs/cuda_mapping.md section 7.17.")))
  void sync() const;
#endif
};

__forceinline__ grid_group this_grid() { return grid_group(); }

// ---------------------------------------------------------------------------
// Free-function collectives
// ---------------------------------------------------------------------------

template <typename Group, typename T, typename Op>
__forceinline__ T reduce(const Group& g, T value, Op op) {
  return g.reduce(value, op);
}
template <typename Group, typename T, typename Op>
__forceinline__ T inclusive_scan(const Group& g, T value, Op op) {
  return g.inclusive_scan(value, op);
}
template <typename Group, typename T, typename Op>
__forceinline__ T exclusive_scan(const Group& g, T value, Op op, T identity) {
  return g.exclusive_scan(value, op, identity);
}

}  // namespace cg
}  // namespace grx

// CUDA source says `cooperative_groups::` or `namespace cg = cooperative_groups`.
// The alias is here rather than in grx_cuda_compat.h because it is a device-side
// name and that header is host-side.
namespace cooperative_groups = grx::cg;

#endif  // GRX_CG_H
