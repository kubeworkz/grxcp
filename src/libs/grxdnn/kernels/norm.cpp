// grxDNN row-wise reductions: softmax and layer normalisation.
//
// ONE WARP PER ROW, and that is the whole design.
//
// Both ops reduce along a row and then rescale it, so the reduction's width is
// the natural unit of work. Giving a row to a whole BLOCK would need a
// block-wide reduction -- shared memory, two stages, and a __syncthreads()
// between them -- and grx_cg.h deliberately has no block reduce, because the
// barrier is the part of this machine that is easiest to get wrong (see
// cuda_mapping.md 7.20). A warp reduce is one butterfly of xor shuffles, needs
// no shared memory and no barrier at all, and every lane ends up holding the
// result, so there is nothing to broadcast afterwards.
//
// The cost is that a row is walked by warp_size lanes rather than blockDim.
// For a transformer that is the right trade anyway: rows are tokens x heads
// and there are thousands of them, so the machine fills across rows.
//
// BOTH OPS ARE MULTI-PASS OVER THE ROW, deliberately.
//
//   softmax    max, then sum of exp, then write        3 passes
//   layernorm  mean, then variance, then write         3 passes
//
// A single-pass softmax (the "online" form) and a single-pass variance (sum
// and sum-of-squares together) both exist and both trade accuracy for
// bandwidth. v0 takes the accurate form: the numbers are checked against a CPU
// reference, and a library whose first version is fast and slightly wrong is
// harder to fix than one that is correct and slow.

#include <grx/device/grx_cg.h>
#include <grx/device/grx_cycles.h>

#include "../dnn_abi.h"
#include "dnn_device.h"

namespace {

namespace cg = grx::cg;

using grxdnn_dev::kNegInf;    // see dnn_device.h for why it is not -inf
using grxdnn_dev::dev_exp;
using grxdnn_dev::dev_exp_nonpos;
using grxdnn_dev::RowMap;
using grxdnn_dev::row_map;

__forceinline__ float dev_rsqrt(float x) {
  // The HARDWARE square root. The device is -march=rv64imafd, so fsqrt.s is a
  // real instruction and __builtin_sqrtf lowers to it, correctly rounded.
  //
  // The first version of this was the famous 0x5f3759df estimate with one
  // Newton step, and the comment above it claimed "about 2e-6 relative". That
  // was wrong -- one Newton step off that estimate is about 1.7e-3 -- and the
  // gate caught it: every layer-norm case failed by 1e-4 to 2e-3 while every
  // softmax case passed, which is the signature of the only thing the two do
  // not share. A reciprocal square root is not worth approximating on a
  // machine that has one.
  return 1.0f / __builtin_sqrtf(x);
}

}  // namespace

// y[i][j] = exp(x[i][j] - max_i) / sum_i exp(x[i][j] - max_i)
__global__ void dnn_softmax(grxdnn_softmax_args* __UNIFORM__ arg) {
  if (arg->abi_version != GRXDNN_ABI_VERSION) return;
  grx::cycle_probe probe(reinterpret_cast<grxCycleSlot*>(arg->cycles));

  const float* x = reinterpret_cast<const float*>(arg->x);
  float*       y = reinterpret_cast<float*>(arg->y);
  const uint32_t rows = arg->rows, cols = arg->cols;
  const int32_t  ldx = arg->ldx, ldy = arg->ldy;

  const RowMap m = row_map();
  auto tile = cg::tiled_partition<VX_CFG_NUM_THREADS>(cg::this_thread_block());

  for (uint32_t r = m.row; r < rows; r += m.stride) {
    const float* xr = x + (size_t)r * (size_t)ldx;
    float*       yr = y + (size_t)r * (size_t)ldy;

    // Pass 1: the row maximum. Lanes past the end contribute -FLT_MAX, which
    // cannot win; contributing 0 would, on a row of negatives.
    //
    // dev_fmax rather than `if (xr[j] > part)`, and the difference is not
    // stylistic: a comparison between floats compiles to a BRANCH on this
    // toolchain, and a branch inside a per-element loop diverges the warp on
    // every element. See the note on dev_fmin in dnn_device.h -- the same
    // rewrite took `dnn_gelu` from twelve vx_split instructions to none and
    // made it 1.34x faster with the arithmetic untouched.
    float part = kNegInf;
    for (uint32_t j = m.lane; j < cols; j += m.width)
      part = grxdnn_dev::dev_fmax(part, xr[j]);
    const float row_max = tile.reduce(part, cg::greater<float>());

    // Pass 2: the shifted exponentials, SUMMED AND KEPT.
    //
    // This used to sum them and throw them away, and pass 3 computed every one
    // a second time. The comment there said recomputing was cheaper than
    // "staging the row anywhere a third pass could read it back from", and the
    // premise was the mistake: there is nowhere to stage it only if you are
    // looking for scratch memory. The output row is already allocated, already
    // the right size, and already about to be written.
    //
    // So the exponential is computed once per element instead of twice, and
    // pass 3 becomes a load, a multiply and a store. dev_exp is the expensive
    // thing in this kernel by a wide margin -- the census prices this loop at
    // 17 float operations, nearly all of them its polynomial -- and softmax is
    // HALF of attention at every sequence length measured (49.4% at S=8 rising
    // to 53.0% at S=64), which is why the second copy was worth finding.
    //
    // IN PLACE STAYS CORRECT, and it is the case to check. With y == x the
    // write below lands on xr[j] -- but only after this lane has read it, and
    // no other lane touches this j. Pass 3 then reads what pass 2 wrote rather
    // than the input, which is exactly what it needs. row_max was reduced
    // across the warp before any store happened.
    float sum = 0.0f;
    for (uint32_t j = m.lane; j < cols; j += m.width) {
      // xr[j] - row_max is <= 0 by construction; see dev_exp_nonpos.
      const float e = dev_exp_nonpos(xr[j] - row_max);
      yr[j] = e;
      sum += e;
    }
    const float row_sum = tile.reduce(sum, cg::plus<float>());

    // A row of zero width would divide by zero. The host refuses cols == 0, so
    // this is belt and braces -- but a NaN written here would propagate through
    // every downstream layer and be blamed on something else.
    const float inv = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    // Pass 3: normalise what pass 2 left behind. No second exponential.
    for (uint32_t j = m.lane; j < cols; j += m.width)
      yr[j] *= inv;
  }
}

// y[i][j] = gamma[j] * (x[i][j] - mean_i) * rsqrt(var_i + eps) + beta[j]
__global__ void dnn_layernorm(grxdnn_layernorm_args* __UNIFORM__ arg) {
  if (arg->abi_version != GRXDNN_ABI_VERSION) return;
  grx::cycle_probe probe(reinterpret_cast<grxCycleSlot*>(arg->cycles));

  const float* x     = reinterpret_cast<const float*>(arg->x);
  float*       y     = reinterpret_cast<float*>(arg->y);
  const float* gamma = reinterpret_cast<const float*>(arg->gamma);
  const float* beta  = reinterpret_cast<const float*>(arg->beta);
  const uint32_t rows = arg->rows, cols = arg->cols;
  const int32_t  ldx = arg->ldx, ldy = arg->ldy;
  const float eps = arg->eps;

  const RowMap m = row_map();
  auto tile = cg::tiled_partition<VX_CFG_NUM_THREADS>(cg::this_thread_block());
  const float inv_n = 1.0f / (float)cols;

  for (uint32_t r = m.row; r < rows; r += m.stride) {
    const float* xr = x + (size_t)r * (size_t)ldx;
    float*       yr = y + (size_t)r * (size_t)ldy;

    // Pass 1: the mean.
    float s = 0.0f;
    for (uint32_t j = m.lane; j < cols; j += m.width) s += xr[j];
    const float mean = tile.reduce(s, cg::plus<float>()) * inv_n;

    // Pass 2: the variance, from deviations rather than from E[x^2] - E[x]^2.
    // The one-pass identity cancels two large numbers when the mean is large
    // relative to the spread, and can even come out negative; this form
    // cannot. It costs one extra read of the row.
    float v = 0.0f;
    for (uint32_t j = m.lane; j < cols; j += m.width) {
      const float d = xr[j] - mean;
      v += d * d;
    }
    // BIASED: divided by cols, not cols-1. PyTorch's LayerNorm, cuDNN and
    // every transformer implementation do the same, and the unbiased form
    // would make a ported model's outputs quietly differ.
    const float var = tile.reduce(v, cg::plus<float>()) * inv_n;
    const float scale = dev_rsqrt(var + eps);

    // Pass 3: write, with the two null tests HOISTED OUT of the element loop.
    //
    // They used to be inside it -- `if (gamma) t *= gamma[j];` -- and gamma and
    // beta are kernel ARGUMENTS: the same for every element, every lane and
    // every row. The compiler kept them anyway, and a branch inside a
    // per-element loop is warp divergence on this machine. ci/check_kernel_loops.py
    // found it: this kernel's hot loop carried four vx_split and seven stack
    // accesses, the worst of any kernel in the image.
    //
    // Four copies of a two-operation loop is the price, and dnn_gelu already
    // pays it for its mode argument for the same reason.
    if (gamma && beta) {
      for (uint32_t j = m.lane; j < cols; j += m.width)
        yr[j] = (xr[j] - mean) * scale * gamma[j] + beta[j];
    } else if (gamma) {
      for (uint32_t j = m.lane; j < cols; j += m.width)
        yr[j] = (xr[j] - mean) * scale * gamma[j];
    } else if (beta) {
      for (uint32_t j = m.lane; j < cols; j += m.width)
        yr[j] = (xr[j] - mean) * scale + beta[j];
    } else {
      for (uint32_t j = m.lane; j < cols; j += m.width)
        yr[j] = (xr[j] - mean) * scale;
    }
  }
}
