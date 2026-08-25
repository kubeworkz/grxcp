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
  // Clamp before scaling: without this, a large negative x produces a huge
  // negative k and 2^k underflows to a denormal or a NaN rather than to zero.
  if (x < -88.0f) return 0.0f;
  if (x >  88.0f) x = 88.0f;

  const float kInvLn2 = 1.44269504088896340736f;
  const float kLn2Hi  = 0.693359375f;
  const float kLn2Lo  = -2.12194440e-4f;

  const int   k = (int)(x * kInvLn2 + (x >= 0.0f ? 0.5f : -0.5f));
  const float r = (x - (float)k * kLn2Hi) - (float)k * kLn2Lo;

  const float r2 = r * r;
  const float p  = 1.0f + r +
                   r2 * (0.5f + r * (0.16666666666f +
                         r * (0.04166666666f + r * 0.00833333333f)));

  // 2^k by building the exponent field directly. k is within +-127 here
  // because x was clamped to +-88 and 88/ln2 < 127.
  union { float f; uint32_t u; } scale;
  scale.u = (uint32_t)((k + 127) & 0xFF) << 23;
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
