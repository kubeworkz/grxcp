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

#include "grx_cycles.h"
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

// --- instrumentation -------------------------------------------------------
//
// Attach an array of cycle slots and the NEXT grxDNN call records how long each
// warp took, measured by the DEVICE's own cycle counter — the only clock that
// measures the device rather than the simulator (see grx_cycles.h). NULL turns
// it off, and off is the default: the same kernel runs either way, so the
// number describes the kernel that ships rather than an instrumented variant.
//
// Every grxDNN kernel has carried a `cycle_probe` since it was written and no
// host call could reach it, so the path was dead code until a whole transformer
// block existed to point it at. `tests/bench/block_cycles.cpp` is what it is
// for: where a real workload's cycles actually go, which is a different
// question from whether any one kernel is correct.
//
// `slots` IS A DEVICE ALLOCATION — grxMalloc, not a host array. The kernel
// writes it, so a host pointer here is an address the device cannot reach and
// the result is a silent record of nothing: every stage reports zero cycles and
// the summary looks like a kernel that never ran. Zero it with grxMemset before
// the call and grxMemcpy it back afterwards; tests/bench/block_cycles.cpp is
// the worked example, and it got this wrong first.
//
// The probe stays attached across calls until it is cleared. One slot per warp;
// grxdnnCycleSlotsNeeded says how many a given shape will use. A capacity too
// small is an error rather than a partial record.
grxdnnStatus_t grxdnnSetCycleProbe(grxdnnHandle_t handle,
                                   grxCycleSlot* slots, int capacity);

// How many slots a call over `rows` rows will write. Every grxDNN kernel is one
// warp per row and strides over them, so this is the launch's warp count and
// not the row count — a 4096-row softmax on a narrow device uses a handful of
// warps, each going round many times.
int grxdnnCycleSlotsNeeded(grxdnnHandle_t handle, int rows);

// ---------------------------------------------------------------------------
// Bias broadcast
// ---------------------------------------------------------------------------
//
//   y[i][j] = x[i][j] + bias[j]
//
// `bias` has length `cols` and is added down every row — the epilogue of every
// Linear layer in a transformer, which grxBLAS does not do because BLAS has no
// concept of it.
//
// A separate pass, not fused into the GEMM. Fusing it is the obvious
// optimisation and it is a later increment with its own gate; a bias welded
// into a GEMM epilogue that nobody has checked separately is two things failing
// as one. In-place is allowed: y may equal x.
grxdnnStatus_t grxdnnAddBiasForward(grxdnnHandle_t handle,
                                    int rows, int cols,
                                    const float* x, int ldx,
                                    const float* bias,
                                    float* y, int ldy);

// ---------------------------------------------------------------------------
// GELU
// ---------------------------------------------------------------------------
//
// TWO FORMS, AND THE CALLER MUST PICK. They are different functions:
//
//   EXACT  y = x * 0.5 * (1 + erf(x / sqrt(2)))          torch.nn.GELU()
//   TANH   y = x * 0.5 * (1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))
//                                                        torch.nn.GELU('tanh')
//
// They differ by up to ~1e-3 in the middle of their range, which is far above
// fp32 resolution and far above this library's tolerances. GPT-2 and BERT were
// trained with the tanh form and their published weights expect it; most
// modern PyTorch code uses the exact one. Substituting either for the other
// gives a ported model plausible outputs and wrong ones, which is the same
// reason layer norm above uses the biased variance.
//
// There is no default. A caller that does not know which form its weights
// expect has a question to answer, not a parameter to leave out.
//
// In-place is allowed: y may equal x.
typedef enum {
  GRXDNN_GELU_EXACT = 0,   // erf form
  GRXDNN_GELU_TANH  = 1    // tanh approximation
} grxdnnGeluMode_t;

grxdnnStatus_t grxdnnGeluForward(grxdnnHandle_t handle,
                                 int rows, int cols,
                                 const float* x, int ldx,
                                 grxdnnGeluMode_t mode,
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
