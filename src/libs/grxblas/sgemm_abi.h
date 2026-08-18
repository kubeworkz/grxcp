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

struct grxblas_sgemm_args {
  uint32_t m, n, k;
  uint32_t lda, ldb, ldc;
  uint32_t transa, transb;
  float    alpha, beta;
  uint64_t a, b, c;
};

#endif  // GRXBLAS_SGEMM_ABI_H
