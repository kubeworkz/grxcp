// grxBLAS — dense linear algebra for GRX-G100.
//
// Shaped after cuBLAS, including its column-major convention. That convention
// surprises everyone who meets it, and it would be tempting to "fix" it to
// row-major -- but the platform's whole argument is that ported code keeps
// working, and silently transposing someone's matrices is the single most
// destructive way to break that promise. Column-major it is, and it is stated
// here rather than buried.
//
//   C = alpha * op(A) * op(B) + beta * C
//
//   A is m x k, lda >= m      element (i,j) at A[i + j*lda]
//   B is k x n, ldb >= k
//   C is m x n, ldc >= m
//
// STATUS: v0 is CORRECT, NOT FAST. The kernel behind grxblasSgemm is a
// straightforward one-thread-per-output-element implementation with no
// blocking, no shared-memory staging and no tensor-core use. It exists to
// establish the API, the kernel packaging and the numerical gate. The tuned
// tensor-core path is the actual phase 3 exit gate and is not here yet, so do
// not benchmark against this and conclude anything about the hardware.

#ifndef GRXBLAS_H
#define GRXBLAS_H

#include "grx_runtime.h"
#include "grx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct grxblasContext* grxblasHandle_t;

typedef enum {
  GRXBLAS_STATUS_SUCCESS          = 0,
  GRXBLAS_STATUS_NOT_INITIALIZED  = 1,
  GRXBLAS_STATUS_ALLOC_FAILED     = 2,
  GRXBLAS_STATUS_INVALID_VALUE    = 3,
  GRXBLAS_STATUS_ARCH_MISMATCH    = 4,
  GRXBLAS_STATUS_EXECUTION_FAILED = 5,
  GRXBLAS_STATUS_NOT_SUPPORTED    = 6,
  GRXBLAS_STATUS_INTERNAL_ERROR   = 7
} grxblasStatus_t;

typedef enum {
  GRXBLAS_OP_N = 0,   // use A as given
  GRXBLAS_OP_T = 1    // use A transposed
} grxblasOperation_t;

const char* grxblasGetStatusString(grxblasStatus_t s);

grxblasStatus_t grxblasCreate   (grxblasHandle_t* handle);
grxblasStatus_t grxblasDestroy  (grxblasHandle_t handle);
grxblasStatus_t grxblasSetStream(grxblasHandle_t handle, grxStream_t stream);
grxblasStatus_t grxblasGetStream(grxblasHandle_t handle, grxStream_t* stream);

// Single-precision general matrix multiply. All pointers are device addresses.
// alpha and beta are read from host memory (cuBLAS host pointer mode).
grxblasStatus_t grxblasSgemm(grxblasHandle_t handle,
                             grxblasOperation_t transa,
                             grxblasOperation_t transb,
                             int m, int n, int k,
                             const float* alpha,
                             const void* A, int lda,
                             const void* B, int ldb,
                             const float* beta,
                             void* C, int ldc);

// Where the library looks for its device kernels. grxBLAS ships precompiled
// .vxbin modules rather than compiling at runtime, the same way a vendor BLAS
// ships tuned binaries.
//
// The first CONFIGURED source wins outright, and there is no fallback past it:
//
//   1. the path set here, if any                          -- else
//   2. $GRXBLAS_KERNEL_PATH, if set                       -- else
//   3. the directory holding the running executable, then the bare filename
//
// If a path is configured and the kernel is not in it, the call fails. It does
// not quietly load some other build: knowing which binary ran matters more than
// the call succeeding, particularly when the answer decides a benchmark.
grxblasStatus_t grxblasSetKernelPath(const char* path);

// The file the kernels were actually loaded from, or NULL if none have been
// loaded yet (loading is lazy, so this is NULL until the first real call).
// Valid until the handle is destroyed.
grxblasStatus_t grxblasGetLoadedKernelPath(grxblasHandle_t handle,
                                           const char** path);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GRXBLAS_H
