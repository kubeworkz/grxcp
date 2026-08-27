// Argument block for the sgemm kernel, shared by the host library and the
// device kernel. One definition, included by both -- two definitions that look
// alike is how a GEMM silently transposes itself.

#ifndef GRXBLAS_SGEMM_ABI_H
#define GRXBLAS_SGEMM_ABI_H

#include <stdint.h>

// Transpose flags, matching grxblasOperation_t. Duplicated as plain integers
// because the device side must not include the host header.
#define GRXBLAS_ABI_OP_N 0
#define GRXBLAS_ABI_OP_T 1

// Bumped whenever this struct changes. The kernel checks it and refuses to run
// on a mismatch, which turns "the .vxbin and the library were built at
// different times" from silent memory corruption into an obvious wrong answer.
// That is not hypothetical: adding the cycles field below, with a stale test
// binary still passing the old (shorter) blob, had the kernel read a pointer
// from past the end of the argument staging area and store through it.
#define GRXBLAS_SGEMM_ABI_VERSION 3u

struct grxblas_sgemm_args {
  uint32_t abi_version;   // GRXBLAS_SGEMM_ABI_VERSION -- first field, never moves
  uint32_t m, n, k;
  uint32_t lda, ldb, ldc;
  uint32_t transa, transb;
  float    alpha, beta;
  uint64_t a, b, c;
  // Optional cycle probe (grxCycleSlot[], one per warp). Zero disables it, and
  // zero is what every call that is not being measured passes -- the measured
  // and unmeasured cases must be the SAME kernel, or the number describes a
  // kernel nobody ships.
  uint64_t cycles;
  // Strided batching. `batch` is 1 for an unbatched call, and the strides are
  // in ELEMENTS, matching cuBLAS. Signed, because a caller is allowed to walk a
  // batch backwards and clamping that to unsigned would run off the front of
  // the allocation instead of the back.
  uint32_t batch;
  uint32_t pad;
  int64_t  stride_a, stride_b, stride_c;
};

// ---------------------------------------------------------------------------
// What one thread of each blocked kernel produces, ASKED OF THE MODULE.
// ---------------------------------------------------------------------------
//
// The host sizes the grid from these, so a host that guesses wrong launches a
// grid covering the wrong number of outputs -- silently, in whichever direction
// the two drifted. That was a live hazard: `kSgemmRowsPerThread` in grxblas.cpp
// carried a comment saying it MUST match RM in kernels/sgemm.cpp, and nothing
// checked it. A stale .vxbin is exactly the case where it would not.
//
// So the module reports its own geometry, the way the tensor path already does
// (hgemm_abi.h). A module with no sgemm_shape entry point gets the REFERENCE
// kernel and nothing else: not knowing the tile means not being able to size a
// launch for it, and a guess there is a wrong answer rather than a slow one.
//
// No warp-width field, unlike the tensor shape. These kernels take their width
// from blockDim and bake nothing in, so a field for it would be one the host
// could not act on.
enum {
  GRXBLAS_SGEMM_SHAPE_RB_RM = 0,  // sgemm_rb: outputs per thread, down a column
  GRXBLAS_SGEMM_SHAPE_2D_RM,      // sgemm_2d: rows of C per thread
  GRXBLAS_SGEMM_SHAPE_2D_RN,      // sgemm_2d: columns of C per thread
  GRXBLAS_SGEMM_SHAPE_COUNT
};

struct grxblas_sgemm_shape_args {
  uint32_t abi_version;   // GRXBLAS_SGEMM_ABI_VERSION
  uint32_t pad;
  uint64_t out;           // uint32_t[GRXBLAS_SGEMM_SHAPE_COUNT]
};

#endif  // GRXBLAS_SGEMM_ABI_H
