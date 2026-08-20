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

#include "grx_cycles.h"
#include "grx_runtime.h"
#include "grx_tensormap.h"
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

// Mixed-precision GEMM on the tensor unit: fp16 in, fp32 accumulate, fp32 out.
// Shaped after cublasGemmEx, and like it, the types are arguments rather than
// part of the name.
//
//   C = alpha * A * B + beta * C,  A m x k, B k x n, C m x n, all column major
//
// WHAT v0 ACCEPTS. Atype and Btype must be GRX_R_16F, Ctype GRX_R_32F, and
// both operations GRXBLAS_OP_N. Anything else returns
// GRXBLAS_STATUS_NOT_SUPPORTED rather than falling back to the scalar kernel:
// a silent fallback would turn "the tensor path does not handle this" into "the
// tensor path is slow", and the caller would have no way to tell.
//
// The device needs a tensor unit AND a DMA engine (grxDeviceProp_t
// capabilities); without either this returns GRXBLAS_STATUS_ARCH_MISMATCH.
//
// A and B are staged through DXA descriptors, so they must live in memory the
// engine can reach -- see grxMallocPhysical in grx_tensormap.h. On a device
// without virtual memory that is any device allocation.
//
// SLOTS. This uses two descriptor slots, 0 and 1 by default. They are device
// state shared with everything else on the device, so a program that programs
// its own tensor maps must either avoid those two or move grxBLAS elsewhere
// with grxblasSetTensorMapSlots. Nothing detects a collision.
grxblasStatus_t grxblasGemmEx(grxblasHandle_t handle,
                              grxblasOperation_t transa,
                              grxblasOperation_t transb,
                              int m, int n, int k,
                              const float* alpha,
                              const void* A, grxDataType_t Atype, int lda,
                              const void* B, grxDataType_t Btype, int ldb,
                              const float* beta,
                              void* C, grxDataType_t Ctype, int ldc);

// Which input types this device's tensor unit accepts.
//
// A bitmask, because the answer is a set and because it is a property of the
// DEVICE BUILD rather than of the architecture: the tensor unit's type support
// is compile-time configuration on GRX-G100, so two sysroots of the same
// hardware can differ. The value comes from the loaded kernel module, which is
// the only thing that knows -- see src/libs/grxblas/hgemm_abi.h.
//
// bf16 is not in this enum. It is not a type this tensor unit has, in any
// configuration, so a bit for it would be a bit that is always zero and reads
// like a build option somebody forgot to turn on. See
// docs/designs/cuda_mapping.md section 7.19.
typedef enum {
  GRXBLAS_TENSOR_FP16 = 0x01,
  GRXBLAS_TENSOR_TF32 = 0x02,
  GRXBLAS_TENSOR_FP8  = 0x04,
  GRXBLAS_TENSOR_FP4  = 0x08,
  GRXBLAS_TENSOR_INT8 = 0x10,
  GRXBLAS_TENSOR_INT4 = 0x20
} grxblasTensorType_t;

// The set of input types the tensor unit accepts, as GRXBLAS_TENSOR_* bits.
// Returns ARCH_MISMATCH on a device with no tensor unit, and like
// grxblasGetTensorTile it needs the kernels to be loadable to answer.
grxblasStatus_t grxblasGetTensorTypes(grxblasHandle_t handle,
                                      unsigned* typeMask);

// Which descriptor slots grxblasGemmEx programs. Defaults to 0 and 1.
grxblasStatus_t grxblasSetTensorMapSlots(grxblasHandle_t handle,
                                         int slotA, int slotB);

// The tile shape the device's tensor unit actually provides, which is what
// grxblasGemmEx blocks the problem into. Derived from the hardware
// configuration rather than fixed at 16x16x16 -- see
// include/grx/device/grx_wmma.h. Returns NOT_SUPPORTED on a device with no
// tensor unit, and the kernels have to be loadable for this to answer.
grxblasStatus_t grxblasGetTensorTile(grxblasHandle_t handle,
                                     int* m, int* n, int* k);

// ---------------------------------------------------------------------------
// Level 1 and level 2
// ---------------------------------------------------------------------------
//
// All pointers are device addresses; alpha and beta are read from host memory
// (cuBLAS host pointer mode). These are memory-bound and use no tensor unit, so
// they work on any device the runtime can open.
//
// INCREMENTS follow BLAS, including the negative case: element i of a length-n
// vector is at index (inc > 0) ? i*inc : (n-1-i)*(-inc), so a negative
// increment walks the vector backwards. `sscal` is the exception BLAS itself
// makes -- it requires incx > 0, and so does this.
//
// A zero increment is REFUSED rather than accepted. BLAS leaves it undefined
// and implementations differ; here every element would alias one address, and
// the result would depend on the order threads happened to run in.

// y = alpha * x + y
grxblasStatus_t grxblasSaxpy(grxblasHandle_t handle, int n, const float* alpha,
                             const float* x, int incx, float* y, int incy);

// x = alpha * x
grxblasStatus_t grxblasSscal(grxblasHandle_t handle, int n, const float* alpha,
                             float* x, int incx);

// y = alpha * op(A) * x + beta * y, with A stored column major, m rows by n
// columns, leading dimension lda >= m.
//
// op(A) is m x n for GRXBLAS_OP_N and n x m for GRXBLAS_OP_T, so x has length
// n and y length m in the first case and the reverse in the second. The
// dimensions m and n always describe A AS STORED, which is BLAS's convention
// and the opposite of what most people guess for the transposed case.
grxblasStatus_t grxblasSgemv(grxblasHandle_t handle, grxblasOperation_t trans,
                             int m, int n, const float* alpha,
                             const void* A, int lda,
                             const void* x, int incx,
                             const float* beta,
                             void* y, int incy);

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

// --- instrumentation -------------------------------------------------------
//
// Attach an array of cycle slots and the next sgemm records how long each warp
// took, measured by the DEVICE's own cycle counter -- the only clock that
// measures the device (see grx_cycles.h). NULL turns it off, and off is the
// default: the same kernel runs either way, so the number describes the kernel
// that ships rather than an instrumented variant of it.
//
// `capacity` is checked against what the call needs; too small is an error
// rather than a partial record. grxblasCycleSlotsNeeded says how many an m x n
// call will use.
grxblasStatus_t grxblasSetCycleProbe(grxblasHandle_t handle,
                                     grxCycleSlot* slots, int capacity);
int             grxblasCycleSlotsNeeded(grxblasHandle_t handle, int m, int n);

// The file the kernels were actually loaded from, or NULL if none have been
// loaded yet (loading is lazy, so this is NULL until the first real call).
// Valid until the handle is destroyed.
grxblasStatus_t grxblasGetLoadedKernelPath(grxblasHandle_t handle,
                                           const char** path);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GRXBLAS_H
