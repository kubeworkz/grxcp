// GRXCP — warp-level primitives.
//
// ALL OF THESE ARE NATIVE. Vote and shuffle are single instructions:
//
//   vx_vote_all / any / uni / ballot   VOTE.*  in the ALU
//   vx_shfl_up / down / bfly / idx     SHFL.*  in the ALU
//
// This header used to say the opposite -- that shuffle was "the platform's
// single highest-impact hardware gap", emulated by staging values through the
// CTA's local memory at roughly an order of magnitude the cost, with a
// proposed WSHFL ISA extension as the fix. That was true when it was written
// and is not true now: the instructions are in vx_intrinsics.h, unconditional
// (no VX_CFG gate), and the SimX ALU implements them. The emulation was
// removed rather than kept as a fallback, because an unexercised fallback is a
// liability and there is no configuration here that needs one.
//
// The one thing worth reading before use: the SHFL instructions implement
// NVIDIA's segmented semantics exactly -- a `c` operand carrying a clamp and a
// segment mask, with out-of-segment lanes keeping their own value -- so CUDA's
// `width` argument maps onto them directly rather than being approximated.
// The mapping is in shfl_control() below.

#ifndef GRX_WARP_H
#define GRX_WARP_H

#include "grx_device.h"

#define GRX_FULL_MASK 0xffffffffu

// --- vote ------------------------------------------------------------------
//
// The `mask` argument is CUDA's participation mask. It is applied to the
// RESULT rather than to the vote, because the hardware votes over the lanes
// that are actually active and there is no way to make a lane that is not
// executing participate. For the usual GRX_FULL_MASK call that is the same
// thing; for a partial mask it means the answer describes the lanes that both
// were asked for and were there.

__forceinline__ unsigned __activemask() {
  return (unsigned)vx_active_threads();
}

__forceinline__ unsigned __ballot_sync(unsigned mask, int predicate) {
  return (unsigned)vx_vote_ballot(predicate) & mask;
}

__forceinline__ int __any_sync(unsigned mask, int predicate) {
  return (mask == 0xffffffffu) ? (int)vx_vote_any(predicate)
                               : (__ballot_sync(mask, predicate) != 0u);
}

__forceinline__ int __all_sync(unsigned mask, int predicate) {
  if (mask == 0xffffffffu) return (int)vx_vote_all(predicate);
  const unsigned participating = mask & (unsigned)vx_active_threads();
  return __ballot_sync(mask, predicate) == participating;
}

// Is the predicate the same on every active lane? No CUDA spelling; the
// hardware has it and a divergence check is worth having.
__forceinline__ int __uni_sync(int predicate) {
  return (int)vx_vote_uni(predicate);
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

// --- shuffle ----------------------------------------------------------------
//
// The instructions take a packed control word: bval (the delta, or the source
// lane for IDX), cval (the clamp) and mask (the segment mask), and compute
//
//   minLane = lane & mask
//   maxLane = (lane & mask) | (cval & ~mask)
//
// keeping a lane's own value when the computed source falls outside
// [minLane, maxLane]. CUDA's `width` splits the warp into segments of that
// size, which is minLane = lane & ~(width-1) and maxLane = minLane + width - 1
// -- so mask = ~(width-1) and cval = width-1, and the rest falls out.

namespace grx { namespace detail {

__forceinline__ int shfl_control(uint32_t bval, int width) {
  const uint32_t w = (width > 0) ? (uint32_t)width : grx::warp_size();
  const uint32_t seg_mask = 0x3fu & ~(w - 1u);   // minLane = lane & seg_mask
  const uint32_t clamp    = (w - 1u) & 0x3fu;    // maxLane = minLane + w - 1
  return (int)((seg_mask << 12) | (clamp << 6) | (bval & 0x3fu));
}

// The instructions move a machine word. Anything smaller rides inside one;
// anything larger has to be split by the caller, which is the same rule CUDA
// has.
template <typename T, typename Op>
__forceinline__ T shfl_bits(T value, Op op) {
  static_assert(sizeof(T) <= sizeof(size_t),
                "grx: shuffle moves one machine word; split a larger value");
  size_t packed = 0;
  __builtin_memcpy(&packed, &value, sizeof(T));
  const size_t got = op(packed);
  T out;
  __builtin_memcpy(&out, &got, sizeof(T));
  return out;
}

}}  // namespace grx::detail

template <typename T>
__forceinline__ T __shfl_sync(unsigned, T v, int srcLane,
                              int width = (int)grx::warp_size()) {
  const int c = grx::detail::shfl_control((uint32_t)srcLane, width);
  return grx::detail::shfl_bits(v, [c](size_t x) { return vx_shfl_idx(x, c & 0x3f, (c >> 6) & 0x3f, (c >> 12) & 0x3f); });
}

template <typename T>
__forceinline__ T __shfl_up_sync(unsigned, T v, unsigned delta,
                                 int width = (int)grx::warp_size()) {
  const int c = grx::detail::shfl_control(delta, width);
  return grx::detail::shfl_bits(v, [c](size_t x) { return vx_shfl_up(x, c & 0x3f, (c >> 6) & 0x3f, (c >> 12) & 0x3f); });
}

template <typename T>
__forceinline__ T __shfl_down_sync(unsigned, T v, unsigned delta,
                                   int width = (int)grx::warp_size()) {
  const int c = grx::detail::shfl_control(delta, width);
  return grx::detail::shfl_bits(v, [c](size_t x) { return vx_shfl_down(x, c & 0x3f, (c >> 6) & 0x3f, (c >> 12) & 0x3f); });
}

template <typename T>
__forceinline__ T __shfl_xor_sync(unsigned, T v, int laneMask,
                                  int width = (int)grx::warp_size()) {
  const int c = grx::detail::shfl_control((uint32_t)laneMask, width);
  return grx::detail::shfl_bits(v, [c](size_t x) { return vx_shfl_bfly(x, c & 0x3f, (c >> 6) & 0x3f, (c >> 12) & 0x3f); });
}

#endif  // GRX_WARP_H
