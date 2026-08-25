// GELU and the bias broadcast: the two ops a transformer's MLP needs that are
// neither a GEMM nor a reduction.
//
// Both are elementwise and both are memory bound. One warp per row, matching
// the norms, so every grxDNN kernel agrees about which lane touches which
// element and a row's worth of work never crosses a warp boundary.
//
// THE TRANSCENDENTALS ARE THE INTERESTING PART. The device build is -nostdlib,
// so there is no erf and no tanh, and GELU needs one or the other depending on
// which form the caller asked for. Both are built here on the dev_exp that
// softmax already uses.
//
// MEASURED, swept over [-1000, 1000] against PyTorch by
// tests/libs/test_grxdnn_gelu.cpp:
//
//   gelu, exact (erf) form    worst 5.36e-07 absolute, at x = 0.8158
//   gelu, tanh form           worst 4.77e-07 absolute, at x = 4.0401
//
// Those are end-to-end figures for GELU, which is what a caller sees. They are
// larger than the erf polynomial's own 1.5e-7 would suggest, so dev_exp and
// fp32 rounding contribute rather than the polynomial dominating -- which is
// the sort of thing only a measurement tells you.
//
// Every accuracy figure in this file is measured, not quoted from a textbook.
// That distinction has already cost this library once: dev_rsqrt shipped with a
// comment claiming "about 2e-6 relative" for a bit-hack that is actually
// 1.7e-3, and every layer-norm case failed while every softmax case passed.

#include <grx/device/grx_cg.h>
#include <grx/device/grx_cycles.h>

#include "../dnn_abi.h"
#include "dnn_device.h"

namespace {

using grxdnn_dev::dev_exp;
using grxdnn_dev::RowMap;
using grxdnn_dev::row_map;

// tanh(x) = 1 - 2 / (e^{2x} + 1)
//
// Written in the form that stays accurate for POSITIVE x and mirrored for
// negative, because e^{2x} overflows long before tanh stops being interesting.
// Beyond |x| = 9 the result is 1 to within fp32's last bit, so it saturates
// rather than computing an exponential that has already gone to infinity.
__forceinline__ float dev_tanh(float x) {
  const float ax = (x < 0.0f) ? -x : x;
  if (ax > 9.0f) return (x < 0.0f) ? -1.0f : 1.0f;
  const float e = dev_exp(2.0f * ax);
  const float t = 1.0f - 2.0f / (e + 1.0f);
  return (x < 0.0f) ? -t : t;
}

// erf(x), Abramowitz & Stegun 7.1.26.
//
//   erf(x) = 1 - (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5) e^{-x^2},
//   t = 1 / (1 + p x),   x >= 0
//
// and odd symmetry below zero. A&S quotes 1.5e-7 absolute for the polynomial
// itself. What GELU delivers on top of it -- polynomial, dev_exp and fp32
// rounding together -- is 5.36e-07, measured, and recorded at the top of this
// file. The book's number is the floor, not the answer.
//
// Beyond |x| = 4 the true value is within 1.5e-8 of 1, which is below fp32
// resolution, so it saturates -- and that also keeps e^{-x^2} away from the
// underflow region where the polynomial would be multiplied by a denormal.
__forceinline__ float dev_erf(float x) {
  const float ax = (x < 0.0f) ? -x : x;
  if (ax > 4.0f) return (x < 0.0f) ? -1.0f : 1.0f;

  const float p  = 0.3275911f;
  const float a1 = 0.254829592f;
  const float a2 = -0.284496736f;
  const float a3 = 1.421413741f;
  const float a4 = -1.453152027f;
  const float a5 = 1.061405429f;

  const float t = 1.0f / (1.0f + p * ax);
  // Horner, so the fifth-degree term does not need t^5 formed separately.
  const float poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
  const float y = 1.0f - poly * dev_exp(-ax * ax);
  return (x < 0.0f) ? -y : y;
}

__forceinline__ float dev_gelu_exact(float x) {
  // 1/sqrt(2), so the divide is a multiply.
  const float kInvSqrt2 = 0.70710678118654752440f;
  return 0.5f * x * (1.0f + dev_erf(x * kInvSqrt2));
}

__forceinline__ float dev_gelu_tanh(float x) {
  const float kSqrt2OverPi = 0.79788456080286535588f;
  const float kCoef = 0.044715f;
  const float inner = kSqrt2OverPi * (x + kCoef * x * x * x);
  return 0.5f * x * (1.0f + dev_tanh(inner));
}

}  // namespace

// y[i][j] = x[i][j] + bias[j]
__global__ void dnn_add_bias(grxdnn_bias_args* __UNIFORM__ arg) {
  if (arg->abi_version != GRXDNN_ABI_VERSION) return;
  grx::cycle_probe probe(reinterpret_cast<grxCycleSlot*>(arg->cycles));

  const float* x = reinterpret_cast<const float*>(arg->x);
  const float* b = reinterpret_cast<const float*>(arg->bias);
  float*       y = reinterpret_cast<float*>(arg->y);
  const uint32_t rows = arg->rows, cols = arg->cols;
  const int32_t  ldx = arg->ldx, ldy = arg->ldy;

  const RowMap m = row_map();
  for (uint32_t r = m.row; r < rows; r += m.stride) {
    const float* xr = x + (size_t)r * (size_t)ldx;
    float*       yr = y + (size_t)r * (size_t)ldy;
    // The bias index is the COLUMN, so every row reads the same vector. On a
    // real cache that is a broadcast; here it is simply the same addresses
    // again, which is why this is not worth a shared-memory stage.
    for (uint32_t j = m.lane; j < cols; j += m.width)
      yr[j] = xr[j] + b[j];
  }
}

// y = gelu(x), in whichever of the two forms the caller asked for.
__global__ void dnn_gelu(grxdnn_gelu_args* __UNIFORM__ arg) {
  if (arg->abi_version != GRXDNN_ABI_VERSION) return;
  grx::cycle_probe probe(reinterpret_cast<grxCycleSlot*>(arg->cycles));

  const float* x = reinterpret_cast<const float*>(arg->x);
  float*       y = reinterpret_cast<float*>(arg->y);
  const uint32_t rows = arg->rows, cols = arg->cols;
  const int32_t  ldx = arg->ldx, ldy = arg->ldy;
  const uint32_t tanh_mode = arg->mode;

  const RowMap m = row_map();
  // The branch is on a UNIFORM value -- every lane in every warp takes the same
  // side -- so it is hoisted out of the inner loop rather than diverging the
  // warp on every element.
  for (uint32_t r = m.row; r < rows; r += m.stride) {
    const float* xr = x + (size_t)r * (size_t)ldx;
    float*       yr = y + (size_t)r * (size_t)ldy;
    if (tanh_mode) {
      for (uint32_t j = m.lane; j < cols; j += m.width)
        yr[j] = dev_gelu_tanh(xr[j]);
    } else {
      for (uint32_t j = m.lane; j < cols; j += m.width)
        yr[j] = dev_gelu_exact(xr[j]);
    }
  }
}
