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
