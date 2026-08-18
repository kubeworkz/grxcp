// GRXCP — warp-level primitives.
//
// READ THIS BEFORE USING SHUFFLE.
//
// Ballot-class primitives are native: the active thread mask is a CSR read
// (vx_active_threads), so __activemask/__ballot_sync/__any_sync/__all_sync are
// one instruction and cost nothing.
//
// Shuffle is NOT native. vx_wgather gathers within a 4-lane GROUP -- its
// documented purpose is a 4x4 transpose (vx_transpose4) -- not from an
// arbitrary lane of a 32-lane warp. So __shfl_sync and friends are implemented
// here by staging through the CTA's local memory: write your value to a
// per-warp scratch slot, fence, read the source lane's slot. That is correct
// and portable and roughly an order of magnitude slower than a register
// shuffle would be.
//
// This is the platform's single highest-impact hardware gap, because warp
// reductions and scans are the backbone of ported CUDA code (reduction,
// softmax, layernorm, prefix sum, sort). It is tracked in
// docs/designs/cuda_mapping.md section 7.1, the fix is the proposed WSHFL ISA
// extension, and grxDeviceProp_t.warpShuffleIsEmulated reports the state at
// runtime so a benchmark can say which it measured.

#ifndef GRX_WARP_H
#define GRX_WARP_H

#include "grx_device.h"

#define GRX_FULL_MASK 0xffffffffu

// --- native: active mask and ballot ---------------------------------------

__forceinline__ unsigned __activemask() {
  return (unsigned)vx_active_threads();
}

__forceinline__ unsigned __ballot_sync(unsigned mask, int predicate) {
  // Under SIMT predication the inactive lanes drop out of the thread mask, so
  // the ballot is the active mask observed inside the predicated region.
  unsigned r = 0;
  if (predicate) r = (unsigned)vx_active_threads();
  return r & mask;
}

__forceinline__ int __any_sync(unsigned mask, int predicate) {
  return __ballot_sync(mask, predicate) != 0u;
}

__forceinline__ int __all_sync(unsigned mask, int predicate) {
  return __ballot_sync(mask, predicate) == (mask & (unsigned)vx_active_threads());
}

// --- native: bit manipulation (RISC-V Zb*) --------------------------------

__forceinline__ int __popc(unsigned x)  { return __builtin_popcount(x); }
__forceinline__ int __clz (unsigned x)  { return x ? __builtin_clz(x) : 32; }
__forceinline__ int __ffs (unsigned x)  { return __builtin_ffs((int)x); }
__forceinline__ unsigned __brev(unsigned x) {
  x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
  x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
  x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
  x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
  return (x >> 16) | (x << 16);
}

// --- emulated: shuffle ----------------------------------------------------
//
// Scratch layout: each resident warp owns warpSize 8-byte slots at the top of
// its CTA's local-memory allocation. grxcc reserves this region; a kernel
// compiled without grxcc must reserve it via the launch's dynamic shared size.

namespace grx { namespace detail {

__forceinline__ volatile uint64_t* shfl_scratch() {
  uint8_t* lmem = (uint8_t*)__local_mem();
  const uint32_t slot = grx::warp_id() * grx::warp_size();
  return (volatile uint64_t*)(lmem) + slot;
}

template <typename T>
__forceinline__ T shfl_impl(T value, uint32_t src_lane) {
  volatile uint64_t* s = shfl_scratch();
  uint64_t packed = 0;
  __builtin_memcpy((void*)&packed, &value, sizeof(T));
  s[grx::lane_id()] = packed;
  __syncwarp();
  uint64_t got = s[src_lane % grx::warp_size()];
  T out;
  __builtin_memcpy(&out, (const void*)&got, sizeof(T));
  return out;
}

}}  // namespace grx::detail

template <typename T>
__forceinline__ T __shfl_sync(unsigned, T v, int srcLane, int = 32) {
  return grx::detail::shfl_impl(v, (uint32_t)srcLane);
}

template <typename T>
__forceinline__ T __shfl_up_sync(unsigned, T v, unsigned delta, int width = 32) {
  const uint32_t lane = grx::lane_id();
  return grx::detail::shfl_impl(v, (lane >= delta) ? (lane - delta) : lane);
}

template <typename T>
__forceinline__ T __shfl_down_sync(unsigned, T v, unsigned delta, int width = 32) {
  const uint32_t lane = grx::lane_id();
  const uint32_t src  = lane + delta;
  return grx::detail::shfl_impl(v, (src < (uint32_t)width) ? src : lane);
}

template <typename T>
__forceinline__ T __shfl_xor_sync(unsigned, T v, int laneMask, int = 32) {
  return grx::detail::shfl_impl(v, grx::lane_id() ^ (uint32_t)laneMask);
}

#endif  // GRX_WARP_H
