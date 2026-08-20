// grxBLAS numerical gate.
//
// Every library kernel needs a CPU reference and a bounded comparison
// (AGENTS.md section 4). This is that reference for sgemm. It runs on a real
// device -- there is no way to fake it, since the arithmetic happens on the
// simulator.
//
// The tolerance is a relative bound scaled by k, because the device
// accumulates in a different order than the reference and float addition is
// not associative. A fixed absolute epsilon would either pass wrong results at
// large k or fail correct ones.
//
// This gate has been checked against a deliberately broken kernel: reverting
// the transposed-A load in kernels/sgemm.cpp to A[l*lda + row] (which is the
// same address as the non-transposed load, so it transposes nothing) makes the
// TN and TT cases fail and leaves NN and NT passing. Worth redoing after any
// change to the reference -- a gate nobody has watched fail is a gate nobody
// knows works.

#include <grx/grx.h>
#include <grx/grxblas.h>

#include "../unit/grx_test.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using grxtest::check;
using grxtest::section;

namespace {

// Unpack a column-major buffer into a dense row-major matrix of the given
// logical shape, applying the transpose while doing so.
//
// The reference deliberately does this in two steps -- unpack, then multiply --
// where the kernel does it in one fused index expression. The first version of
// this file computed its indices the same way the kernel did, and the result
// was that the transpose-A case passed while transposing nothing: two copies of
// one misconception agree with each other perfectly. A reference is only worth
// having if it can disagree.
std::vector<float> unpack(const std::vector<float>& src, int ld, bool trans,
                          int rows, int cols) {
  std::vector<float> out((size_t)rows * (size_t)cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      // Storage holds a (trans ? cols x rows : rows x cols) matrix, column
      // major, leading dimension ld. Transposing swaps which subscript indexes
      // the stored rows.
      const int sr = trans ? c : r;
      const int sc = trans ? r : c;
      out[(size_t)r * (size_t)cols + (size_t)c] =
          src[(size_t)sr + (size_t)sc * (size_t)ld];
    }
  }
  return out;
}

// Column-major reference: C = alpha * op(A) * op(B) + beta * C.
void reference_sgemm(bool ta, bool tb, int m, int n, int k, float alpha,
                     const std::vector<float>& A, int lda,
                     const std::vector<float>& B, int ldb, float beta,
                     std::vector<float>& C, int ldc) {
  const std::vector<float> a = unpack(A, lda, ta, m, k);   // m x k, row major
  const std::vector<float> b = unpack(B, ldb, tb, k, n);   // k x n, row major
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      double acc = 0.0;
      for (int l = 0; l < k; ++l)
        acc += (double)a[(size_t)i * (size_t)k + (size_t)l] *
               (double)b[(size_t)l * (size_t)n + (size_t)j];
      const size_t ci = (size_t)i + (size_t)j * (size_t)ldc;
      C[ci] = (beta == 0.0f) ? (float)(alpha * acc)
                             : (float)(alpha * acc + (double)beta * C[ci]);
    }
  }
}

// Strided batching, checked as a whole buffer rather than matrix by matrix.
//
// `gap` puts slack between batch members so a kernel that walks its own stride
// instead of the caller's writes into it and is caught. Passing strideB = 0
// broadcasts one B across the batch, which cuBLAS allows and which a kernel
// that multiplies the batch index into every pointer gets wrong.
bool run_batched(grxblasHandle_t h, bool ta, bool tb, int m, int n, int k,
                 float alpha, float beta, int batch, int gap, bool broadcast_b,
                 const char* label) {
  const int lda = std::max(1, ta ? k : m);
  const int ldb = std::max(1, tb ? n : k);
  const int ldc = std::max(1, m);

  const size_t a_elems = (size_t)lda * (size_t)(ta ? m : k);
  const size_t b_elems = (size_t)ldb * (size_t)(tb ? k : n);
  const size_t c_elems = (size_t)ldc * (size_t)n;

  const long long sa = (long long)(a_elems + gap);
  const long long sb = broadcast_b ? 0 : (long long)(b_elems + gap);
  const long long sc = (long long)(c_elems + gap);

  std::vector<float> A((size_t)(sa * (batch - 1)) + a_elems + gap);
  std::vector<float> B(broadcast_b ? b_elems + gap
                                   : (size_t)(sb * (batch - 1)) + b_elems + gap);
  std::vector<float> C((size_t)(sc * (batch - 1)) + c_elems + gap);

  auto fill = [](std::vector<float>& v, unsigned seed) {
    for (size_t i = 0; i < v.size(); ++i) {
      seed = seed * 1664525u + 1013904223u;
      v[i] = (float)((int)(seed >> 16) % 17 - 8) * 0.25f;
    }
  };
  fill(A, 3u); fill(B, 11u); fill(C, 17u);

  // Reference: one GEMM per batch member, into a copy of the whole C buffer,
  // so anything outside the m x n windows has to come back untouched.
  std::vector<float> expected = C;
  for (int bi = 0; bi < batch; ++bi) {
    const std::vector<float> Ab(A.begin() + (size_t)(sa * bi), A.end());
    const std::vector<float> Bb(B.begin() + (size_t)(sb * bi), B.end());
    std::vector<float> Cb(expected.begin() + (size_t)(sc * bi), expected.end());
    reference_sgemm(ta, tb, m, n, k, alpha, Ab, lda, Bb, ldb, beta, Cb, ldc);
    std::copy(Cb.begin(), Cb.begin() + (long)c_elems,
              expected.begin() + (size_t)(sc * bi));
  }

  void *dA = nullptr, *dB = nullptr, *dC = nullptr;
  if (grxMalloc(&dA, A.size() * sizeof(float)) != grxSuccess ||
      grxMalloc(&dB, B.size() * sizeof(float)) != grxSuccess ||
      grxMalloc(&dC, C.size() * sizeof(float)) != grxSuccess) {
    std::printf("  FAIL  %s (allocation)\n", label);
    return false;
  }
  grxMemcpy(dA, A.data(), A.size() * sizeof(float), grxMemcpyDefault);
  grxMemcpy(dB, B.data(), B.size() * sizeof(float), grxMemcpyDefault);
  grxMemcpy(dC, C.data(), C.size() * sizeof(float), grxMemcpyDefault);

  const grxblasStatus_t st = grxblasSgemmStridedBatched(
      h, ta ? GRXBLAS_OP_T : GRXBLAS_OP_N, tb ? GRXBLAS_OP_T : GRXBLAS_OP_N,
      m, n, k, &alpha, dA, lda, sa, dB, ldb, sb, &beta, dC, ldc, sc, batch);
  if (st != GRXBLAS_STATUS_SUCCESS) {
    std::printf("  FAIL  %s (%s)\n", label, grxblasGetStatusString(st));
    grxFree(dA); grxFree(dB); grxFree(dC);
    return false;
  }
  grxDeviceSynchronize();

  std::vector<float> got(C.size(), 0.0f);
  grxMemcpy(got.data(), dC, got.size() * sizeof(float), grxMemcpyDefault);
  grxFree(dA); grxFree(dB); grxFree(dC);

  int bad = 0;
  float worst = 0.0f;
  for (size_t i = 0; i < got.size(); ++i) {
    const float d = std::fabs(got[i] - expected[i]);
    if (d > worst) worst = d;
    const float scale = std::fabs(expected[i]) + 1.0f;
    if (d > 1e-4f * scale * (float)(k + 1)) {
      if (bad < 3)
        std::printf("        [%zu] got %g want %g\n", i, (double)got[i],
                    (double)expected[i]);
      ++bad;
    }
  }
  if (bad) {
    std::printf("  FAIL  %s (%d wrong, worst %g)\n", label, bad, (double)worst);
    return false;
  }
  std::printf("  ok    %s (worst %g)\n", label, (double)worst);
  return true;
}

// pad grows every leading dimension past its minimum. A GEMM that ignores ld
// and assumes tightly packed columns passes every unpadded test and corrupts
// the first real sub-matrix it is handed, so at least one case must pad.
bool run_case(grxblasHandle_t h, bool ta, bool tb, int m, int n, int k,
              float alpha, float beta, const char* label, int pad = 0) {
  // BLAS requires ld >= max(1, minimum); the max matters only when k == 0.
  const int lda = std::max(1, (ta ? k : m) + pad);
  const int ldb = std::max(1, (tb ? n : k) + pad);
  const int ldc = std::max(1, m + pad);

  std::vector<float> A((size_t)lda * (ta ? m : k));
  std::vector<float> B((size_t)ldb * (tb ? k : n));
  std::vector<float> C((size_t)ldc * n);

  // Deterministic values with a mix of signs and magnitudes; a matrix of ones
  // would hide index errors because every wrong element still sums to k.
  auto fill = [](std::vector<float>& v, unsigned seed) {
    for (size_t i = 0; i < v.size(); ++i) {
      seed = seed * 1664525u + 1013904223u;
      v[i] = (float)((int)(seed >> 16) % 17 - 8) * 0.25f;
    }
  };
  fill(A, 1u); fill(B, 7u); fill(C, 13u);

  std::vector<float> expected = C;
  reference_sgemm(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, expected, ldc);

  // k == 0 leaves A and B empty. Allocating a byte anyway keeps the device
  // pointers valid and non-null, which is what a caller scaling C by beta with
  // no multiply to do would realistically pass.
  auto bytes = [](const std::vector<float>& v) {
    return v.empty() ? sizeof(float) : v.size() * sizeof(float);
  };
  void *dA = nullptr, *dB = nullptr, *dC = nullptr;
  if (grxMalloc(&dA, bytes(A)) != grxSuccess ||
      grxMalloc(&dB, bytes(B)) != grxSuccess ||
      grxMalloc(&dC, bytes(C)) != grxSuccess) {
    std::printf("  FAIL  %s (allocation)\n", label);
    return false;
  }
  if (!A.empty()) grxMemcpy(dA, A.data(), A.size() * sizeof(float), grxMemcpyDefault);
  if (!B.empty()) grxMemcpy(dB, B.data(), B.size() * sizeof(float), grxMemcpyDefault);
  grxMemcpy(dC, C.data(), C.size() * sizeof(float), grxMemcpyDefault);

  const grxblasStatus_t s =
      grxblasSgemm(h, ta ? GRXBLAS_OP_T : GRXBLAS_OP_N,
                   tb ? GRXBLAS_OP_T : GRXBLAS_OP_N, m, n, k, &alpha, dA, lda,
                   dB, ldb, &beta, dC, ldc);
  if (s != GRXBLAS_STATUS_SUCCESS) {
    std::printf("  FAIL  %s (%s)\n", label, grxblasGetStatusString(s));
    return false;
  }
  grxDeviceSynchronize();

  std::vector<float> got(C.size(), 0.0f);
  grxMemcpy(got.data(), dC, got.size() * sizeof(float), grxMemcpyDefault);

  // Sweep the whole buffer, not just the m x n window. Padding rows sit between
  // the columns of C; anything the kernel writes there is a stray write into
  // the caller's data, and it has to be exactly as uploaded, not merely close.
  const float tol = 1e-5f * (float)(k + 1);
  int bad = 0, stray = 0;
  float worst = 0.0f;
  for (size_t idx = 0; idx < C.size(); ++idx) {
    const int i = (int)(idx % (size_t)ldc);
    const int j = (int)(idx / (size_t)ldc);
    const bool inside = (i < m && j < n);
    const float d = std::fabs(got[idx] - expected[idx]);
    if (inside && d > worst) worst = d;
    if (inside ? (d > tol) : (got[idx] != expected[idx])) {
      if (bad + stray < 3)
        std::printf("        %s(%d,%d) got %g want %g\n",
                    inside ? "" : "padding ", i, j, got[idx], expected[idx]);
      if (inside) ++bad; else ++stray;
    }
  }

  grxFree(dA); grxFree(dB); grxFree(dC);

  if (bad || stray) {
    std::printf("  FAIL  %s (%d/%zu elements wrong", label, bad, C.size());
    if (stray) std::printf(", %d stray writes outside the m x n window", stray);
    std::printf(", worst %g)\n", worst);
    return false;
  }
  std::printf("  ok    %s (worst error %g)\n", label, worst);
  return true;
}

}  // namespace

int main() {
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count <= 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }
  if (grxSetDevice(0) != grxSuccess) return 1;

  grxblasHandle_t h = nullptr;
  const grxblasStatus_t cs = grxblasCreate(&h);
  check(cs == GRXBLAS_STATUS_SUCCESS, "grxblasCreate");
  if (cs != GRXBLAS_STATUS_SUCCESS) return grxtest::report();

  // Probe once before claiming to test anything. Without the device toolchain
  // the .vxbin does not exist, and every case below would fail identically --
  // which reads as "the GEMM is broken" when the truth is "nobody compiled it".
  {
    void* d = nullptr;
    const float one = 1.0f, zero = 0.0f;
    if (grxMalloc(&d, sizeof(float)) == grxSuccess) {
      const grxblasStatus_t probe =
          grxblasSgemm(h, GRXBLAS_OP_N, GRXBLAS_OP_N, 1, 1, 1, &one, d, 1, d, 1,
                       &zero, d, 1);
      grxFree(d);
      if (probe == GRXBLAS_STATUS_NOT_INITIALIZED) {
        std::printf("grxblas device kernels not found; skipping\n"
                    "  build them with: ci/build_kernel.sh --grxgpu <path> "
                    "src/libs/grxblas/kernels/sgemm.cpp\n");
        grxblasDestroy(h);
        return 77;
      }
    }
  }

  // A numerical pass is a statement about one specific binary. Say which.
  const char* kpath = nullptr;
  if (grxblasGetLoadedKernelPath(h, &kpath) == GRXBLAS_STATUS_SUCCESS && kpath)
    std::printf("kernels: %s\n", kpath);

  section("sgemm against a CPU reference");
  bool all = true;
  all &= run_case(h, false, false,  4,  4,  4, 1.0f, 0.0f, "4x4x4 NN alpha=1 beta=0");
  all &= run_case(h, false, false,  8,  5,  3, 1.0f, 0.0f, "8x5x3 NN rectangular");
  all &= run_case(h, false, false,  7,  6,  5, 2.5f, -1.5f, "7x6x5 NN alpha/beta");
  all &= run_case(h, true,  false,  6,  4,  5, 1.0f, 0.0f, "6x4x5 TN transpose A");
  all &= run_case(h, false, true,   5,  7,  4, 1.0f, 0.0f, "5x7x4 NT transpose B");
  all &= run_case(h, true,  true,   4,  5,  6, 1.0f, 0.5f, "4x5x6 TT both");
  all &= run_case(h, false, false, 16, 16, 16, 1.0f, 0.0f, "16x16x16 NN");
  all &= run_case(h, true,  true,  16, 16, 16, 1.0f, 0.0f, "16x16x16 TT");
  all &= run_case(h, false, false,  1,  1,  9, 1.0f, 0.0f, "1x1x9 degenerate");
  all &= run_case(h, false, false,  5,  4,  6, 1.0f, 0.0f, "5x4x6 NN padded ld", 3);
  all &= run_case(h, true,  true,   6,  5,  4, 2.0f, -1.0f, "6x5x4 TT padded ld", 2);
  all &= run_case(h, false, false,  4,  3,  0, 1.0f, 2.0f, "4x3x0 k=0 scales C by beta");
  check(all, "every sgemm case matches the reference");

  section("sgemm, strided batched");
  {
    bool all = true;
    all &= run_batched(h, false, false, 6, 5, 4, 1.0f, 0.0f, 3, 0, false,
                       "3 matrices, tight strides");
    all &= run_batched(h, false, false, 6, 5, 4, 2.0f, -1.0f, 4, 7, false,
                       "4 matrices, slack between them, alpha and beta");
    all &= run_batched(h, false, false, 6, 5, 4, 1.0f, 0.0f, 3, 5, true,
                       "strideB = 0 broadcasts one B across the batch");
    all &= run_batched(h, true, true, 5, 4, 6, 1.0f, 0.5f, 3, 3, false,
                       "transposed operands, batched");
    all &= run_batched(h, false, false, 7, 3, 5, 1.0f, 0.0f, 1, 4, false,
                       "batchCount = 1 is the unbatched path");
    check(all, "every batched case matches a per-matrix reference");

    const float one = 1.0f, zero = 0.0f;
    void* d = nullptr;
    grxMalloc(&d, 1024);
    check(grxblasSgemmStridedBatched(h, GRXBLAS_OP_N, GRXBLAS_OP_N, 4, 4, 4,
                                     &one, d, 4, 16, d, 4, 16, &zero, d, 4, 16,
                                     0) == GRXBLAS_STATUS_SUCCESS,
          "batchCount = 0 succeeds and does nothing");
    check(grxblasSgemmStridedBatched(h, GRXBLAS_OP_N, GRXBLAS_OP_N, 4, 4, 4,
                                     &one, d, 4, 16, d, 4, 16, &zero, d, 4, 16,
                                     -1) == GRXBLAS_STATUS_INVALID_VALUE,
          "a negative batchCount is refused");
    grxFree(d);
  }

  section("argument validation");
  {
    const float one = 1.0f, zero = 0.0f;
    void* d = nullptr;
    grxMalloc(&d, 1024);
    check(grxblasSgemm(h, GRXBLAS_OP_N, GRXBLAS_OP_N, 4, 4, 4, &one, d, /*lda=*/2,
                       d, 4, &zero, d, 4) == GRXBLAS_STATUS_INVALID_VALUE,
          "a too-small lda is rejected rather than reading out of bounds");
    check(grxblasSgemm(h, GRXBLAS_OP_N, GRXBLAS_OP_N, 4, 4, 4, &one, d, 4,
                       d, 4, &zero, d, /*ldc=*/1) == GRXBLAS_STATUS_INVALID_VALUE,
          "a too-small ldc is rejected");
    check(grxblasSgemm(h, GRXBLAS_OP_N, GRXBLAS_OP_N, 0, 4, 4, &one, d, 4,
                       d, 4, &zero, d, 4) == GRXBLAS_STATUS_SUCCESS,
          "an empty problem succeeds without launching");
    check(grxblasSgemm(nullptr, GRXBLAS_OP_N, GRXBLAS_OP_N, 4, 4, 4, &one, d, 4,
                       d, 4, &zero, d, 4) == GRXBLAS_STATUS_NOT_INITIALIZED,
          "a null handle is rejected");
    grxFree(d);
  }

  check(grxblasDestroy(h) == GRXBLAS_STATUS_SUCCESS, "grxblasDestroy");
  return grxtest::report();
}
