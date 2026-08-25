// grxBLAS sgemm kernel -- correct, deliberately simple.
//
// One thread per output element, an inner loop over k, no blocking and no
// shared-memory staging. That is not how a fast GEMM is written and this file
// makes no claim to be one: it exists so the library has a numerically
// verified reference while the tensor-core path is built, and so the packaging
// (a precompiled .vxbin resolved by name) is exercised end to end.
//
// The tuned path will be a separate kernel selected by the host, not an
// evolution of this one -- keeping a correct reference around is worth more
// than editing it into something unrecognisable.

#include <grx/device/grx_cycles.h>

#include "../sgemm_abi.h"

// Named sgemm, not sgemm_nn: one entry point handles all four transpose
// combinations by branching on the argument block. A tuned build will ship
// specialised entries and the host will select between them, at which point the
// names will say which is which -- until then a name that promises NN only
// would be a lie the loader cannot catch.
__global__ void sgemm(grxblas_sgemm_args* __UNIFORM__ arg) {
  // A host that was built against a different version of this struct is
  // passing a blob whose fields are not where this kernel thinks they are.
  // Doing nothing produces an obviously wrong result; carrying on produces a
  // wild pointer store.
  if (arg->abi_version != GRXBLAS_SGEMM_ABI_VERSION) return;

  const float* A = reinterpret_cast<const float*>(arg->a);
  const float* B = reinterpret_cast<const float*>(arg->b);
  float*       C = reinterpret_cast<float*>(arg->c);

  // The probe brackets the whole kernel, and the tail check below is a branch
  // around the body rather than an early return: a warp that returned early
  // would never write its slot, so a measurement would quietly describe only
  // the warps that had work to do. With a null pointer this compiles to
  // nothing, which is what every unmeasured call passes.
  grx::cycle_probe probe(reinterpret_cast<grxCycleSlot*>(arg->cycles));

  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t m = arg->m, n = arg->n, k = arg->k;

  // The batch is the grid's SECOND dimension rather than a division of a
  // linearised index: one integer divide per thread by a runtime value is a
  // real cost in a kernel whose inner loop is two loads and a multiply-add,
  // and the dispatcher walks a 2D grid for free.
  const uint32_t batch = blockIdx.y;
  A += batch * arg->stride_a;
  B += batch * arg->stride_b;
  C += batch * arg->stride_c;

  if (idx < m * n) {
    // Column-major: consecutive idx walks down a column, which is also the
    // direction consecutive lanes should read C for a coalesced store.
    const uint32_t row = idx % m;
    const uint32_t col = idx / m;

    const uint32_t lda = arg->lda, ldb = arg->ldb, ldc = arg->ldc;
    const bool ta = (arg->transa == GRXBLAS_ABI_OP_T);
    const bool tb = (arg->transb == GRXBLAS_ABI_OP_T);

    // Everything is stored column-major, so a stored element (r, c) of a
    // matrix with leading dimension ld is at [r + c*ld] -- always, for A, B
    // and C. The transpose flags do not change the storage, they change which
    // stored element a given position of op() names:
    //
    //   op(A)(row, l) = ta ? A_stored(l, row) : A_stored(row, l)
    //   op(B)(l, col) = tb ? B_stored(col, l) : B_stored(l, col)
    //
    // Written as the swap of a single subscript pair, because writing the two
    // cases as independent index expressions is how a transposed GEMM ends up
    // reading a correct-looking address in the wrong matrix.
    float acc = 0.0f;
    for (uint32_t l = 0; l < k; ++l) {
      const float a = ta ? A[l + row * lda] : A[row + l * lda];
      const float b = tb ? B[col + l * ldb] : B[l + col * ldb];
      acc += a * b;
    }

    const float beta = arg->beta;
    // Reading C when beta is zero would be wrong as well as wasteful: the
    // caller is allowed to pass uninitialised memory, and 0 * NaN is NaN.
    C[row + col * ldc] = (beta == 0.0f)
                             ? (arg->alpha * acc)
                             : (arg->alpha * acc + beta * C[row + col * ldc]);
  }

  probe.finish();
}

// ---------------------------------------------------------------------------
// sgemm_rb -- the same GEMM, register blocked.
// ---------------------------------------------------------------------------
//
// A SEPARATE ENTRY POINT, not an edit of the one above, which is what that
// file header said the tuned path would be. Two reasons, and the second is the
// one that matters:
//
//   1. A correct reference is worth keeping.
//   2. It makes the reference an ORACLE. tests/libs/test_grxblas_rb.cpp runs
//      both kernels over the same operands on the same device and compares
//      them, so the fast one is checked against the slow one directly rather
//      than only against a host reference at whatever tolerance that allows.
//      A tuned kernel that agrees with the naive one to the last bit on every
//      shape is a much stronger statement than one that lands inside 1e-4.
//
// WHAT IT CHANGES. The kernel above does, per output element, k iterations of
// two loads and one multiply-add. The B element it loads is the same for every
// output in that COLUMN, and it reloads it for each of them.
//
// So one thread now owns RM outputs in one column, and loads B once for all of
// them:
//
//   naive     per k-step, per output:   2 loads, 1 FMA
//   blocked   per k-step, per RM:       RM + 1 loads, RM FMAs
//
// At RM = 4 that is 5 loads per 4 FMAs instead of 8 -- and the loop overhead,
// the bounds arithmetic and the index math are amortised over four outputs
// rather than paid per output.
//
// THE ROWS ARE STRIDED, NOT ADJACENT. Thread `sub` owns rows
// sub, sub + row_blocks, sub + 2*row_blocks, ... rather than 4*sub .. 4*sub+3.
// Adjacent rows would be the obvious choice and would break coalescing: C is
// column-major, consecutive lanes must write consecutive rows, and giving each
// lane a contiguous run of four puts a stride of 4 between neighbouring lanes.
// The strided form keeps every one of the RM stores coalesced across the warp.
//
// The tail is handled by CLAMPING the row index rather than branching on it.
// A bounds test inside the k loop diverges the warp on every iteration for the
// one block that straddles the edge; clamping reads a row that is in range,
// computes a value nobody wants, and discards it at the store. Wasted
// arithmetic on at most RM-1 lanes of one block, against a branch in the hot
// loop for everybody.
__global__ void sgemm_rb(grxblas_sgemm_args* __UNIFORM__ arg) {
  if (arg->abi_version != GRXBLAS_SGEMM_ABI_VERSION) return;

  const float* A = reinterpret_cast<const float*>(arg->a);
  const float* B = reinterpret_cast<const float*>(arg->b);
  float*       C = reinterpret_cast<float*>(arg->c);

  grx::cycle_probe probe(reinterpret_cast<grxCycleSlot*>(arg->cycles));

  const uint32_t m = arg->m, n = arg->n, k = arg->k;
  const uint32_t batch = blockIdx.y;
  A += batch * arg->stride_a;
  B += batch * arg->stride_b;
  C += batch * arg->stride_c;

  constexpr uint32_t RM = 4;
  const uint32_t row_blocks = (m + RM - 1u) / RM;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (m != 0u && idx < row_blocks * n) {
    const uint32_t sub = idx % row_blocks;
    const uint32_t col = idx / row_blocks;

    const uint32_t lda = arg->lda, ldb = arg->ldb, ldc = arg->ldc;
    const bool ta = (arg->transa == GRXBLAS_ABI_OP_T);
    const bool tb = (arg->transb == GRXBLAS_ABI_OP_T);

    // Row for each of the RM outputs, clamped into range, and whether it is
    // real. Both computed ONCE, outside the k loop, which is the whole point.
    uint32_t row[RM];
    bool     live[RM];
    #pragma unroll
    for (uint32_t i = 0; i < RM; ++i) {
      const uint32_t r = sub + i * row_blocks;
      live[i] = (r < m);
      row[i]  = live[i] ? r : (m - 1u);
    }

    float acc[RM];
    #pragma unroll
    for (uint32_t i = 0; i < RM; ++i) acc[i] = 0.0f;

    // The same index algebra as the reference above, and deliberately written
    // the same way: op(A)(row, l) is a swap of one subscript pair, not two
    // independent expressions.
    for (uint32_t l = 0; l < k; ++l) {
      const float b = tb ? B[col + l * ldb] : B[l + col * ldb];
      #pragma unroll
      for (uint32_t i = 0; i < RM; ++i) {
        const float a = ta ? A[l + row[i] * lda] : A[row[i] + l * lda];
        acc[i] += a * b;
      }
    }

    const float alpha = arg->alpha, beta = arg->beta;
    #pragma unroll
    for (uint32_t i = 0; i < RM; ++i) {
      if (!live[i]) continue;
      const size_t at = (size_t)row[i] + (size_t)col * ldc;
      // Same rule as the reference: C is not read when beta is zero, because
      // the caller may pass uninitialised memory and 0 * NaN is NaN.
      C[at] = (beta == 0.0f) ? (alpha * acc[i])
                             : (alpha * acc[i] + beta * C[at]);
    }
  }

  probe.finish();
}
