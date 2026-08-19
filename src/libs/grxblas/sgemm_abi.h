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
#define GRXBLAS_SGEMM_ABI_VERSION 2u

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
};

#endif  // GRXBLAS_SGEMM_ABI_H
