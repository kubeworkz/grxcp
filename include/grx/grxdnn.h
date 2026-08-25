// grxDNN — neural-network primitives for GRX-G100.
//
// Shaped after cuDNN's *operations*, not its descriptor machinery. cuDNN's API
// is built around handle/descriptor objects because it has to describe 4D and
// 5D tensors with arbitrary strides for convolution. v0 has no convolution, and
// inventing that apparatus before there is anything to describe would be
// scaffolding around an empty room. The ops here take shapes directly, and the
// descriptor form arrives with conv2d, when it earns its keep.
//
// ROW-MAJOR, AND THAT DIFFERS FROM grxBLAS ON PURPOSE.
//
//   grxBLAS is COLUMN-major, because cuBLAS is, because Fortran was.
//   grxDNN is ROW-major, because cuDNN is, and because a transformer's
//   activations are stored [tokens][features] everywhere they come from.
//
// Two libraries in one platform with opposite conventions is a trap, so it is
// stated here in capitals rather than left to be discovered. The alternative --
// making one of them match the other -- breaks the promise the whole platform
// rests on: that ported code keeps working. A cuDNN port that had its tensors
// silently transposed would produce plausible numbers and wrong answers.
//
//   x is rows x cols, ldx >= cols      element (i, j) at x[i * ldx + j]
//
// Every op reduces along a ROW: softmax normalises each row independently,
// layer norm takes each row's mean and variance. That is the axis a
// transformer normalises over, and it is the one that makes `cols` the
// feature dimension.
//
// STATUS: v0 is softmax, layer norm and scaled dot-product attention, fp32,
// forward only. No backward pass, no conv, no attention FUSION -- attention
// here is two grxBLAS GEMMs, a mask and a softmax, which is correct and
// unfused. What is here is checked against an outside reference to a stated
// tolerance (a CPU reference for the norms, PyTorch itself for attention);
// what is not here is absent rather than stubbed, so a port that needs it
// fails to compile.

#ifndef GRXDNN_H
#define GRXDNN_H

#include "grx_runtime.h"
#include "grx_types.h"
#include "grxblas.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct grxdnnContext* grxdnnHandle_t;

typedef enum {
  GRXDNN_STATUS_SUCCESS          = 0,
  GRXDNN_STATUS_NOT_INITIALIZED  = 1,
  GRXDNN_STATUS_ALLOC_FAILED     = 2,
  GRXDNN_STATUS_INVALID_VALUE    = 3,
  GRXDNN_STATUS_ARCH_MISMATCH    = 4,
  GRXDNN_STATUS_EXECUTION_FAILED = 5,
  GRXDNN_STATUS_NOT_SUPPORTED    = 6,
  GRXDNN_STATUS_KERNEL_NOT_FOUND = 7
} grxdnnStatus_t;

const char* grxdnnGetStatusString(grxdnnStatus_t s);

grxdnnStatus_t grxdnnCreate    (grxdnnHandle_t* handle);
grxdnnStatus_t grxdnnDestroy   (grxdnnHandle_t handle);
grxdnnStatus_t grxdnnSetStream (grxdnnHandle_t handle, grxStream_t stream);
grxdnnStatus_t grxdnnGetStream (grxdnnHandle_t handle, grxStream_t* stream);

// Where the library looks for its device kernels. Same mechanism as grxBLAS,
// and they share one module: only one can be resident at a time, so a program
// that uses both libraries would otherwise fail to load the second.
grxdnnStatus_t grxdnnSetKernelPath(const char* dir);

// The file the kernels were actually loaded from, or NULL if none have been
// loaded yet (loading is lazy, so this is NULL until the first real call).
// Valid until the handle is destroyed.
//
// grxBLAS has the same call, and comparing the two is how
// tests/libs/test_libs_together.cpp checks the claim that both libraries share
// ONE image rather than each quietly finding its own -- a claim that is
// otherwise invisible from the outside, because two separate images produce the
// same numbers right up until the second one fails to load.
grxdnnStatus_t grxdnnGetLoadedKernelPath(grxdnnHandle_t handle,
                                         const char** path);

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------
//
//   y[i][j] = exp(x[i][j] - max_j x[i][j]) / sum_j exp(x[i][j] - max_j ...)
//
// The max subtraction is not an optimisation, it is the difference between a
// result and an inf. exp(89.f) already overflows fp32, and attention logits
// reach that routinely. `tests/libs/test_grxdnn.cpp` checks a row that
// overflows the naive form, so the stable one cannot quietly be replaced.
//
// In-place is allowed: y may equal x.
grxdnnStatus_t grxdnnSoftmaxForward(grxdnnHandle_t handle,
                                    int rows, int cols,
                                    const float* x, int ldx,
                                    float* y, int ldy);

// ---------------------------------------------------------------------------
// Layer normalisation
// ---------------------------------------------------------------------------
//
//   mean_i = (1/cols) * sum_j x[i][j]
//   var_i  = (1/cols) * sum_j (x[i][j] - mean_i)^2        <- biased, as in
//                                                            PyTorch's LayerNorm
//   y[i][j] = gamma[j] * (x[i][j] - mean_i) / sqrt(var_i + eps) + beta[j]
//
// `gamma` and `beta` are per-FEATURE vectors of length `cols`, and either may
// be null for an unscaled or unshifted norm.
//
// The variance is the BIASED estimator -- divided by cols, not cols-1 -- which
// is what PyTorch, cuDNN and every transformer implementation use. The
// unbiased one differs by a factor of cols/(cols-1) and would make every
// ported model's outputs subtly wrong.
//
// In-place is allowed: y may equal x.
grxdnnStatus_t grxdnnLayerNormForward(grxdnnHandle_t handle,
                                      int rows, int cols,
                                      const float* x, int ldx,
                                      const float* gamma, const float* beta,
                                      float eps,
                                      float* y, int ldy);

// ---------------------------------------------------------------------------
// Scaled dot-product attention
// ---------------------------------------------------------------------------
//
//   scores = Q Kᵀ / sqrt(headDim)          [seqLen x seqLen] per head
//   P      = softmax(scores, along keys)
//   out    = P V                           [seqLen x headDim] per head
//
// Q, K, V and out are ROW-major [batch][heads][seqLen][headDim], contiguous —
// the layout a transformer's activations already have, and the layout the rest
// of grxDNN uses. `headDim` is the innermost dimension.
//
// SELF-ATTENTION ONLY in v0: Q, K and V share one seqLen. Cross-attention needs
// a second length and is absent rather than stubbed, so a port that needs it
// fails to compile instead of silently attending over the wrong span.
//
// THIS OP CALLS grxBLAS, and that is worth knowing rather than hiding. The two
// GEMMs are `grxblasSgemmStridedBatched`, which means grxDNN's row-major
// tensors have to be presented to a COLUMN-major library — twice, once
// transposed. No data is moved: a row-major (r, c) matrix with leading
// dimension ld *is* the column-major (c, r) matrix with the same bytes, so the
// transposes live in the arguments. The bookkeeping is checked against
// PyTorch's own `scaled_dot_product_attention` in
// `tests/libs/test_grxdnn_attn.cpp`, because a reference derived from the same
// reasoning as the implementation would agree with it whether or not either
// was right.
//
// The handle carries its own grxBLAS handle and shares this one's stream.
//
// NOT HERE: dropout, arbitrary attention masks, cross-attention, multi-query
// or grouped-query attention, a backward pass, and any fusion at all. This is
// three library calls and a mask, which is correct and unfused; the flash-style
// single-pass form is a later increment with its own numerical gate.

typedef enum {
  GRXDNN_ATTN_MASK_NONE   = 0,   // every key visible to every query
  GRXDNN_ATTN_MASK_CAUSAL = 1    // query i may not see key j > i
} grxdnnAttnMask_t;

// Bytes of scratch `grxdnnAttentionForward` needs for the given shape, or an
// error. The scores matrix is [batch][heads][seqLen][seqLen] and is the whole
// of it — it grows as the SQUARE of the sequence, which is the reason flash
// attention exists and the reason this is the caller's allocation rather than
// a hidden one inside a "forward" call.
grxdnnStatus_t grxdnnAttentionWorkspaceSize(int batch, int heads, int seqLen,
                                            int headDim, size_t* bytes);

// `workspace` must be a device allocation of at least the size above.
// In-place is NOT allowed: out must not alias Q, K or V.
grxdnnStatus_t grxdnnAttentionForward(grxdnnHandle_t handle,
                                      int batch, int heads,
                                      int seqLen, int headDim,
                                      const float* Q, const float* K,
                                      const float* V,
                                      grxdnnAttnMask_t mask,
                                      void* workspace, size_t workspaceBytes,
                                      float* out);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GRXDNN_H
