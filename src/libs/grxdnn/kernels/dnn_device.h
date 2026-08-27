// Device-side pieces shared by grxDNN's kernels.
//
// all.cpp puts every kernel file into ONE translation unit, so anything defined
// in two of them is a redefinition error — and, worse, anything defined in one
// is silently visible to the others through the shared anonymous namespace,
// which makes a file compile only because of the order all.cpp happens to
// include it in. Everything more than one kernel needs is therefore here, named
// and included on purpose.

#ifndef GRXDNN_KERNELS_DNN_DEVICE_H
#define GRXDNN_KERNELS_DNN_DEVICE_H

#include <grx/device/grx_cg.h>

namespace grxdnn_dev {

// The largest finite float, negated.
//
// It is the identity for a max reduction — a lane with no element must
// contribute something that cannot win, and 0 would win over a row of
// negatives — and it is also what a masked attention score is set to. Not an
// actual -inf: dev_exp clamps below -88 and returns exactly zero, so a masked
// entry contributes nothing to the sum, whereas a true infinity would give
// inf - inf = NaN in the max subtraction the moment a whole row was masked.
constexpr float kNegInf = -3.402823466e+38f;

// FLOAT SELECTS COMPILE TO BRANCHES HERE, AND BRANCHES DIVERGE WARPS.
//
// That is the finding these three exist for. The integer ternaries in
// grxBLAS's kernels become czero/or -- conditional moves, no control flow --
// so it was reasonable to assume the float ones did too. They do not: rewriting
// dev_exp's early returns as ternaries left `dnn_gelu`'s vx_split and vx_join
// counts at exactly 18 and 18, because the compiler emits a branch either way.
// Its column loop was 86 instructions for 21 float operations, six of those
// instructions vx_split and six vx_join.
//
// RISC-V F has single instructions for all three of these -- fmin.s, fmax.s and
// fsgnj.s -- and they are the ones the compiler will not reach for from a
// comparison. Asked for by name, it emits them, and the branch goes away.
//
// NaN: fmin/fmax here follow the IEEE-754 minNum/maxNum the hardware
// implements, which returns the non-NaN operand. Every use below passes a
// constant as one operand, so a NaN input propagates as the constant rather
// than as a NaN -- which matches what the early returns it replaced did, since
// a NaN failed their comparisons and fell through to the clamped path.
__forceinline__ float dev_fmin(float a, float b) { return __builtin_fminf(a, b); }
__forceinline__ float dev_fmax(float a, float b) { return __builtin_fmaxf(a, b); }
__forceinline__ float dev_copysign(float m, float s) {
  return __builtin_copysignf(m, s);
}
__forceinline__ float dev_fabs(float a) { return __builtin_fabsf(a); }

// exp() for the device. The device build is -nostdlib, so the libm one is not
// available; this is the standard range reduction.
//
//   e^x = 2^k * e^r,  k = round(x / ln2),  r = x - k*ln2,  |r| <= ln2/2
//
// e^r over that range is a degree-5 Taylor series. Softmax, tanh and erf all
// go through here, so its error is the floor under all three, and it is
// measured against PyTorch by tests/libs/test_grxdnn_gelu.cpp rather than
// quoted.
__forceinline__ float dev_exp(float x) {
  // SELECTS RATHER THAN EARLY RETURNS -- and the reason it was done is not the
  // reason it is kept, which is worth writing down.
  //
  // The guards here and in dev_tanh were early returns, and `dnn_gelu` carries
  // eighteen vx_split and eighteen vx_join instructions: warp divergence
  // machinery, on a kernel whose actual work is a fifteen-operation polynomial.
  // The obvious inference was that the guards were causing it. They were not.
  // Rewritten as selects, the split and join counts are IDENTICAL -- 18 and 18
  // before and after -- because the compiler was already emitting conditional
  // moves for them. The whole change is worth 2.5% of the GELU stage, about
  // half a percent of the block.
  //
  // It is kept for the two smaller reasons that survive. An early return in
  // device code is a hazard this project has been bitten by twice already
  // (cuda_mapping.md 7.24, and the texture border case), so having fewer of
  // them is worth half a percent on its own. And the arithmetic is UNCHANGED
  // for every in-range input -- xc == x, underflow is false, the same
  // operations in the same order -- so the measured accuracy figures at the top
  // of kernels/elementwise.cpp do not move.
  //
  // Where gelu's cycles actually go, ablated rather than argued: of the 35573
  // the stage cost at S=8, removing the whole tanh saves 24012 and removing
  // only the divide saves 2243. It is the exponential's polynomial, and nothing
  // structural will touch it.
  //
  // The clamp itself is still needed for the reason it always was: a large
  // negative x gives a k below -127, and the exponent-field construction at the
  // bottom of this function would build a denormal or wrap rather than
  // underflow to zero.
  x = dev_fmin(dev_fmax(x, -88.0f), 88.0f);

  const float kInvLn2 = 1.44269504088896340736f;
  const float kLn2Hi  = 0.693359375f;
  const float kLn2Lo  = -2.12194440e-4f;

  const int   k = (int)(x * kInvLn2 + dev_copysign(0.5f, x));
  const float r = (x - (float)k * kLn2Hi) - (float)k * kLn2Lo;

  const float r2 = r * r;
  const float p  = 1.0f + r +
                   r2 * (0.5f + r * (0.16666666666f +
                         r * (0.04166666666f + r * 0.00833333333f)));

  // 2^k by building the exponent field directly. k is within +-127 here
  // because x was clamped to +-88 and 88/ln2 < 127.
  union { float f; uint32_t u; } scale;
  scale.u = (uint32_t)((k + 127) & 0xFF) << 23;
  // No underflow select. Clamping at -88 gives k = -127 exactly, so
  // (k + 127) & 0xFF is zero, scale.f is +0.0f, and the product is exactly the
  // zero the early return used to hand back. The guard was doing work the
  // arithmetic already did.
  return p * scale.f;
}

// Which row this warp owns, and how far to jump for the next one.
//
// ONE WARP PER ROW is grxDNN's whole device-side layout, and every kernel uses
// this so they agree about which lane touches which element. The norms need it
// because a warp reduce is one butterfly of xor shuffles with no shared memory
// and no barrier; the elementwise kernels do not need it at all, and use it
// anyway so that a fused version of any two of them is a rewrite of the body
// rather than of the indexing.
struct RowMap {
  uint32_t row;      // the row this warp starts on
  uint32_t stride;   // how far to jump for the next row
  uint32_t lane;
  uint32_t width;
};

__forceinline__ RowMap row_map() {
  const uint32_t w = grx::warp_size();
  RowMap m;
  m.lane   = grx::lane_id();
  m.width  = w;
  const uint32_t warps_per_block = (blockDim.x + w - 1u) / w;
  m.row    = blockIdx.x * warps_per_block + (threadIdx.x / w);
  m.stride = gridDim.x * warps_per_block;
  return m;
}

}  // namespace grxdnn_dev

#endif  // GRXDNN_KERNELS_DNN_DEVICE_H
