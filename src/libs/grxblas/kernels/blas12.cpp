// grxBLAS level-1 and level-2 kernels: saxpy, sscal, sgemv.
//
// These are memory-bound, so the only design question that matters is which
// thread reads which address. The tensor unit is irrelevant here and is not
// used.
//
// sgemv has two shapes for one reason, and it is the reason:
//
//   OP_N   y[row] = sum_j A[row + j*lda] * x[j]. One thread per row means
//          consecutive lanes read consecutive rows of the same column --
//          adjacent addresses, one coalesced access per column.
//
//   OP_T   y[col] = sum_i A[i + col*lda] * x[i]. One thread per column would
//          have consecutive lanes strided lda apart, which is a separate
//          access per lane for every step of the reduction. So instead a whole
//          WARP takes one column and walks it together -- adjacent addresses
//          again -- and finishes with a tile reduction over the lanes.
//
// The transposed case is not a variant of the untransposed one; it is a
// different traversal that happens to compute a transposed product.

#include <grx/device/grx_cg.h>
#include <grx/device/grx_cycles.h>

#include "../blas12_abi.h"

namespace {

namespace cg = grx::cg;

// BLAS increment addressing. A negative increment traverses backwards from the
// far end, which is why this is a function and not a multiply.
__forceinline__ uint32_t vec_index(uint32_t i, uint32_t n, int32_t inc) {
  return (inc > 0) ? (i * (uint32_t)inc)
                   : ((n - 1u - i) * (uint32_t)(-inc));
}

}  // namespace

// y = alpha * x + y, one element per thread.
__global__ void saxpy(grxblas_axpy_args* __UNIFORM__ arg) {
  if (arg->abi_version != GRXBLAS_BLAS12_ABI_VERSION) return;
  grx::cycle_probe probe(reinterpret_cast<grxCycleSlot*>(arg->cycles));

  const float* x = reinterpret_cast<const float*>(arg->x);
  float*       y = reinterpret_cast<float*>(arg->y);
  const uint32_t n = arg->n;
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    const uint32_t xi = vec_index(i, n, arg->incx);
    const uint32_t yi = vec_index(i, n, arg->incy);
    y[yi] = arg->alpha * x[xi] + y[yi];
  }

  probe.finish();
}

// x = alpha * x.
__global__ void sscal(grxblas_scal_args* __UNIFORM__ arg) {
  if (arg->abi_version != GRXBLAS_BLAS12_ABI_VERSION) return;
  grx::cycle_probe probe(reinterpret_cast<grxCycleSlot*>(arg->cycles));

  float* x = reinterpret_cast<float*>(arg->x);
  const uint32_t n = arg->n;
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    const uint32_t xi = vec_index(i, n, arg->incx);
    x[xi] = arg->alpha * x[xi];
  }

  probe.finish();
}

// y = alpha * op(A) * x + beta * y.
//
// The launch geometry differs between the two cases and the host picks it:
// OP_N is one thread per output, OP_T is one warp per output. The kernel reads
// arg->trans to know which it is in, rather than being two entry points,
// because the alternative is two names that must be kept in step with two
// launch shapes on the host.
__global__ void sgemv(grxblas_gemv_args* __UNIFORM__ arg) {
  if (arg->abi_version != GRXBLAS_BLAS12_ABI_VERSION) return;
  grx::cycle_probe probe(reinterpret_cast<grxCycleSlot*>(arg->cycles));

  const float* A = reinterpret_cast<const float*>(arg->a);
  const float* x = reinterpret_cast<const float*>(arg->x);
  float*       y = reinterpret_cast<float*>(arg->y);

  const uint32_t lda   = arg->lda;
  const uint32_t rows  = arg->rows;
  const uint32_t depth = arg->depth;
  const float    alpha = arg->alpha;
  const float    beta  = arg->beta;

  if (arg->trans == GRXBLAS_ABI_OP_N) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
      float acc = 0.0f;
      for (uint32_t j = 0; j < depth; ++j)
        acc += A[row + j * lda] * x[vec_index(j, depth, arg->incx)];

      const uint32_t yi = vec_index(row, rows, arg->incy);
      // Reading y when beta is zero would be wrong as well as wasteful: the
      // caller may pass uninitialised memory, and 0 * NaN is NaN.
      y[yi] = (beta == 0.0f) ? (alpha * acc) : (alpha * acc + beta * y[yi]);
    }
  } else {
    // One warp per output column. Lane l takes elements l, l+W, l+2W, ... of
    // the column, so the lanes read adjacent addresses on every step.
    const uint32_t col = blockIdx.x;
    if (col < rows) {
      auto tile = cg::tiled_partition<VX_CFG_NUM_THREADS>(cg::this_thread_block());
      const uint32_t lane = tile.thread_rank();
      const uint32_t W    = tile.num_threads();

      float part = 0.0f;
      for (uint32_t i = lane; i < depth; i += W)
        part += A[i + col * lda] * x[vec_index(i, depth, arg->incx)];

      // Every lane ends with the total; only one of them stores.
      const float acc = tile.reduce(part, cg::plus<float>());
      if (lane == 0) {
        const uint32_t yi = vec_index(col, rows, arg->incy);
        y[yi] = (beta == 0.0f) ? (alpha * acc) : (alpha * acc + beta * y[yi]);
      }
    }
  }

  probe.finish();
}
