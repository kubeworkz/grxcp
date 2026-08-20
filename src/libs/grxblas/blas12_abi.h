// Argument blocks for the level-1 and level-2 grxBLAS kernels, shared by the
// host library and the device kernels. One definition, included by both.
//
// Increments are SIGNED, because BLAS defines a negative increment as
// traversing the vector backwards -- element i of a length-n vector lives at
// index (inc > 0) ? i*inc : (n-1-i)*(-inc). Storing them unsigned and clamping
// would silently turn a reverse traversal into a forward one, which is a wrong
// answer rather than an error.

#ifndef GRXBLAS_BLAS12_ABI_H
#define GRXBLAS_BLAS12_ABI_H

#include <stdint.h>

#include "sgemm_abi.h"   // GRXBLAS_ABI_OP_N / _T

// Bumped whenever any struct here changes; every kernel checks it and refuses
// to run on a mismatch. See sgemm_abi.h for the incident that made this a rule.
#define GRXBLAS_BLAS12_ABI_VERSION 1u

// y = alpha * x + y
struct grxblas_axpy_args {
  uint32_t abi_version;
  uint32_t n;
  int32_t  incx, incy;
  float    alpha;
  uint32_t pad;
  uint64_t x, y;
  uint64_t cycles;   // optional grxCycleSlot[], 0 to disable
};

// x = alpha * x
struct grxblas_scal_args {
  uint32_t abi_version;
  uint32_t n;
  int32_t  incx;
  float    alpha;
  uint64_t x;
  uint64_t cycles;
};

// y = alpha * op(A) * x + beta * y, A stored column major, m x n, ld >= m.
//
// `rows` is the length of the OUTPUT vector -- m for OP_N, n for OP_T -- and
// `depth` the length of the reduction. Both are computed on the host so the
// kernel never has to re-derive which of m and n it is looking at; getting that
// backwards produces a plausible-looking result of the wrong shape.
struct grxblas_gemv_args {
  uint32_t abi_version;
  uint32_t m, n;
  uint32_t lda;
  uint32_t trans;      // GRXBLAS_ABI_OP_N / _T
  uint32_t rows;       // outputs
  uint32_t depth;      // reduction length
  int32_t  incx, incy;
  float    alpha, beta;
  uint64_t a, x, y;
  uint64_t cycles;
};

#endif  // GRXBLAS_BLAS12_ABI_H
