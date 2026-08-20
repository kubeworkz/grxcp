// grxBLAS GemmEx numerical gate: the tensor-core path, checked exactly.
//
// Inputs are multiples of 0.5 in [-4, 4]. Every one is exact in binary16,
// every product is exact in binary32, and the accumulation stays inside the
// exactly-representable integers-over-four range at these sizes -- so a
// correct tensor unit reproduces the reference bit for bit and no tolerance is
// available to hide a wrong answer behind. The inputs are checked for that
// property rather than assumed to have it.
//
// The shapes matter more than usual here. The tile is 8 x 4 x 8 on this
// configuration, and a GEMM whose dimensions are all multiples of the tile
// exercises none of the interesting paths: the descriptor's zero padding on
// partial input tiles, the output masking on partial output tiles, or a k that
// does not fill its last step. Most of the cases below are deliberately ragged.

#include <grx/grx.h>
#include <grx/grxblas.h>

#include "../common/fp16.h"
#include "../unit/grx_test.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using grxtest::check;
using grxtest::section;
using grxtest::float_to_half;
using grxtest::exact_in_half;

namespace {

float sample(unsigned& seed) {
  seed = seed * 1664525u + 1013904223u;
  return (float)((int)((seed >> 16) % 17u) - 8) * 0.5f;
}

// C = alpha * op(A) * op(B) + beta * C, column major, computed from the float
// values the halves were made from.
//
// The transpose is applied here as a swap of a single subscript pair, and
// deliberately not as a second index expression: writing the two cases
// independently is how a reference ends up agreeing with a kernel that
// transposes nothing. The sgemm gate learned that the hard way.
void reference(bool ta, bool tb, int m, int n, int k, float alpha,
               const std::vector<float>& A, int lda,
               const std::vector<float>& B, int ldb, float beta,
               std::vector<float>& C, int ldc) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      float acc = 0.0f;
      for (int l = 0; l < k; ++l) {
        const float a = ta ? A[(size_t)l + (size_t)i * lda]
                           : A[(size_t)i + (size_t)l * lda];
        const float b = tb ? B[(size_t)j + (size_t)l * ldb]
                           : B[(size_t)l + (size_t)j * ldb];
        acc += a * b;
      }
      const size_t ci = (size_t)i + (size_t)j * ldc;
      C[ci] = (beta == 0.0f) ? (alpha * acc) : (alpha * acc + beta * C[ci]);
    }
  }
}

bool run_case(grxblasHandle_t h, bool ta, bool tb, int m, int n, int k,
              float alpha, float beta, int pad, const char* label) {
  // Storage shapes follow op(), not the logical ones: a transposed A is stored
  // k x m and its leading dimension bounds k.
  const int kk     = k ? k : 1;
  const int a_rows = ta ? kk : m,  a_cols = ta ? m : kk;
  const int b_rows = tb ? n : kk,  b_cols = tb ? kk : n;
  const int lda = a_rows + pad, ldb = b_rows + pad, ldc = m + pad;

  std::vector<float> A((size_t)lda * (size_t)a_cols),
      B((size_t)ldb * (size_t)b_cols), C((size_t)ldc * n);
  unsigned seed = 99u;
  for (auto& v : A) v = sample(seed);
  for (auto& v : B) v = sample(seed);
  for (auto& v : C) v = sample(seed) * 2.0f;

  for (float v : A) if (!exact_in_half(v)) { std::printf("  FAIL  %s (A not exact in fp16)\n", label); return false; }
  for (float v : B) if (!exact_in_half(v)) { std::printf("  FAIL  %s (B not exact in fp16)\n", label); return false; }

  std::vector<uint16_t> hA(A.size()), hB(B.size());
  for (size_t i = 0; i < A.size(); ++i) hA[i] = float_to_half(A[i]);
  for (size_t i = 0; i < B.size(); ++i) hB[i] = float_to_half(B[i]);

  std::vector<float> expected = C;
  reference(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, expected, ldc);

  void *dA = nullptr, *dB = nullptr, *dC = nullptr;
  auto bytes16 = [](const std::vector<uint16_t>& v) {
    return v.empty() ? sizeof(uint16_t) : v.size() * sizeof(uint16_t);
  };
  if (grxMalloc(&dA, bytes16(hA)) != grxSuccess ||
      grxMalloc(&dB, bytes16(hB)) != grxSuccess ||
      grxMalloc(&dC, C.size() * sizeof(float)) != grxSuccess) {
    std::printf("  FAIL  %s (allocation)\n", label);
    return false;
  }
  if (!hA.empty()) grxMemcpy(dA, hA.data(), hA.size() * sizeof(uint16_t), grxMemcpyDefault);
  if (!hB.empty()) grxMemcpy(dB, hB.data(), hB.size() * sizeof(uint16_t), grxMemcpyDefault);
  grxMemcpy(dC, C.data(), C.size() * sizeof(float), grxMemcpyDefault);

  const grxblasStatus_t s =
      grxblasGemmEx(h, ta ? GRXBLAS_OP_T : GRXBLAS_OP_N,
                    tb ? GRXBLAS_OP_T : GRXBLAS_OP_N, m, n, k, &alpha,
                    dA, GRX_R_16F, lda, dB, GRX_R_16F, ldb, &beta,
                    dC, GRX_R_32F, ldc);
  if (s != GRXBLAS_STATUS_SUCCESS) {
    std::printf("  FAIL  %s (%s)\n", label, grxblasGetStatusString(s));
    grxFree(dA); grxFree(dB); grxFree(dC);
    return false;
  }
  grxDeviceSynchronize();

  std::vector<float> got(C.size(), 0.0f);
  grxMemcpy(got.data(), dC, got.size() * sizeof(float), grxMemcpyDefault);

  // Sweep the whole buffer: the padding rows between columns of C must be
  // exactly as uploaded, or an edge tile wrote outside the caller's matrix.
  int bad = 0, stray = 0;
  float worst = 0.0f;
  for (size_t idx = 0; idx < C.size(); ++idx) {
    const int i = (int)(idx % (size_t)ldc);
    const int j = (int)(idx / (size_t)ldc);
    const bool inside = (i < m && j < n);
    const float d = got[idx] - expected[idx];
    const float ad = d < 0 ? -d : d;
    if (inside && ad > worst) worst = ad;
    if (got[idx] != expected[idx]) {
      if (bad + stray < 3)
        std::printf("        %s(%d,%d) got %g want %g\n",
                    inside ? "" : "padding ", i, j, got[idx], expected[idx]);
      if (inside) ++bad; else ++stray;
    }
  }

  grxFree(dA); grxFree(dB); grxFree(dC);

  if (bad || stray) {
    std::printf("  FAIL  %s (%d wrong", label, bad);
    if (stray) std::printf(", %d stray writes outside the m x n window", stray);
    std::printf(", worst %g)\n", worst);
    return false;
  }
  std::printf("  ok    %s\n", label);
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

  grxDeviceProp_t prop{};
  grxGetDeviceProperties(&prop, 0);

  grxblasHandle_t h = nullptr;
  if (grxblasCreate(&h) != GRXBLAS_STATUS_SUCCESS) return 1;

  int tm = 0, tn = 0, tk = 0;
  const grxblasStatus_t ts = grxblasGetTensorTile(h, &tm, &tn, &tk);
  if (ts == GRXBLAS_STATUS_ARCH_MISMATCH) {
    std::printf("SKIPPED: %s has no tensor unit or no DMA engine\n", prop.name);
    grxblasDestroy(h);
    return 77;
  }
  if (ts == GRXBLAS_STATUS_NOT_INITIALIZED) {
    std::printf("grxblas tensor kernels not found; skipping\n");
    grxblasDestroy(h);
    return 77;
  }
  check(ts == GRXBLAS_STATUS_SUCCESS, "the tensor tile shape can be queried");
  if (ts != GRXBLAS_STATUS_SUCCESS) return grxtest::report();
  std::printf("tensor tile %dx%dx%d (fp16 in, fp32 out) on %s\n", tm, tn, tk,
              prop.name);

  section("what the tensor unit accepts, asked rather than assumed");
  {
    unsigned types = 0;
    const grxblasStatus_t qs = grxblasGetTensorTypes(h, &types);
    check(qs == GRXBLAS_STATUS_SUCCESS, "the input types can be queried");

    static const struct { unsigned bit; const char* name; } kNames[] = {
      {GRXBLAS_TENSOR_FP16, "fp16"}, {GRXBLAS_TENSOR_TF32, "tf32"},
      {GRXBLAS_TENSOR_FP8,  "fp8"},  {GRXBLAS_TENSOR_FP4,  "fp4"},
      {GRXBLAS_TENSOR_INT8, "int8"}, {GRXBLAS_TENSOR_INT4, "int4"},
    };
    std::printf("  device accepts:");
    for (const auto& n : kNames)
      if (types & n.bit) std::printf(" %s", n.name);
    std::printf("%s\n", types ? "" : " nothing");

    // GemmEx ran fp16 above, so fp16 must be in the set. A mask that says
    // otherwise is not a device limitation, it is a broken report -- and a
    // broken capability report is worse than none, because callers act on it.
    check((types & GRXBLAS_TENSOR_FP16) != 0,
          "fp16 is reported, and the GEMM above proves it is really there");
  }

  section("GemmEx against a CPU reference");
  bool all = true;
  all &= run_case(h, false, false, tm, tn, tk, 1.0f, 0.0f, 0, "exactly one tile, one k step");
  all &= run_case(h, false, false, tm * 2, tn * 2, tk * 2, 1.0f, 0.0f, 0, "four tiles, two k steps");
  all &= run_case(h, false, false, 16, 16, 16, 1.0f, 0.0f, 0, "16x16x16");
  all &= run_case(h, false, false, 5, 3, 7, 1.0f, 0.0f, 0, "5x3x7 -- every dimension ragged");
  all &= run_case(h, false, false, 17, 9, 13, 1.0f, 0.0f, 0, "17x9x13 -- ragged and larger");
  all &= run_case(h, false, false, 1, 1, 1, 1.0f, 0.0f, 0, "1x1x1 -- one element, one product");
  all &= run_case(h, false, false, 12, 6, 10, 2.5f, -1.5f, 0, "alpha and beta");
  all &= run_case(h, false, false, 12, 6, 10, 1.0f, 0.0f, 3, "padded leading dimensions");
  all &= run_case(h, false, false, 6, 5, 0, 1.0f, 2.0f, 0, "k = 0 scales C by beta");

  section("GemmEx with transposed operands");
  // Every combination, and the ragged-k cases are the point. Transposing an
  // operand moves k between the descriptor's dimension 0 and its outer
  // dimension, and dimension 0 is the one the DXA engine does not pad
  // (cuda_mapping.md 7.14). TN puts k in dimension 0 for BOTH operands, which
  // is the combination that used to be arithmetically impossible here.
  for (int which = 0; which < 3; ++which) {
    const bool ta = (which != 1), tb = (which != 0);
    const char* name = ta && tb ? "TT" : (ta ? "TN" : "NT");
    char label[96];

    std::snprintf(label, sizeof(label), "%s exactly one tile", name);
    all &= run_case(h, ta, tb, tm, tn, tk, 1.0f, 0.0f, 0, label);
    std::snprintf(label, sizeof(label), "%s four tiles, two k steps", name);
    all &= run_case(h, ta, tb, tm * 2, tn * 2, tk * 2, 1.0f, 0.0f, 0, label);
    std::snprintf(label, sizeof(label), "%s 5x3x7 -- every dimension ragged", name);
    all &= run_case(h, ta, tb, 5, 3, 7, 1.0f, 0.0f, 0, label);
    std::snprintf(label, sizeof(label), "%s 17x9x13 -- ragged and larger", name);
    all &= run_case(h, ta, tb, 17, 9, 13, 1.0f, 0.0f, 0, label);
    std::snprintf(label, sizeof(label), "%s alpha, beta and padded lds", name);
    all &= run_case(h, ta, tb, 12, 6, 10, 2.5f, -1.5f, 3, label);
  }
  check(all, "every transposed GemmEx case matches the reference exactly");
  check(all, "every GemmEx case matches the reference exactly");

  section("GemmEx, int8 in and int32 out");
  {
    unsigned types = 0;
    grxblasGetTensorTypes(h, &types);
    if (!(types & GRXBLAS_TENSOR_INT8)) {
      std::printf("  SKIPPED: this device's tensor unit has no int8 "
                  "(rebuild with -DVX_CFG_TCU_INT8_ENABLE)\n");
    } else {
      // Integers, so every comparison is EXACT -- no tolerance, not even the
      // "values chosen so it comes out exact" kind the fp16 cases need.
      auto int8_case = [&](bool ta, bool tb, int m, int n, int k, float alpha,
                           float beta, int pad, const char* label) -> bool {
        const int kk = k ? k : 1;
        const int a_rows = ta ? kk : m, a_cols = ta ? m : kk;
        const int b_rows = tb ? n : kk, b_cols = tb ? kk : n;
        const int lda = a_rows + pad, ldb = b_rows + pad, ldc = m + pad;

        std::vector<int8_t>  A((size_t)lda * a_cols), B((size_t)ldb * b_cols);
        std::vector<int32_t> C((size_t)ldc * n);
        for (size_t i = 0; i < A.size(); ++i) A[i] = (int8_t)((int)(i * 7 % 15) - 7);
        for (size_t i = 0; i < B.size(); ++i) B[i] = (int8_t)((int)(i * 5 % 13) - 6);
        for (size_t i = 0; i < C.size(); ++i) C[i] = (int32_t)(i * 3 % 41) - 20;

        std::vector<int32_t> want = C;
        const int32_t ai = (int32_t)alpha, bi = (int32_t)beta;
        for (int j = 0; j < n; ++j)
          for (int i = 0; i < m; ++i) {
            int32_t acc = 0;
            for (int l = 0; l < k; ++l) {
              const int32_t av = ta ? A[(size_t)l + (size_t)i * lda]
                                    : A[(size_t)i + (size_t)l * lda];
              const int32_t bv = tb ? B[(size_t)j + (size_t)l * ldb]
                                    : B[(size_t)l + (size_t)j * ldb];
              acc += av * bv;
            }
            const size_t ci = (size_t)i + (size_t)j * ldc;
            want[ci] = (bi == 0) ? (ai * acc) : (ai * acc + bi * want[ci]);
          }

        void *dA = nullptr, *dB = nullptr, *dC = nullptr;
        if (grxMalloc(&dA, A.empty() ? 1 : A.size()) != grxSuccess ||
            grxMalloc(&dB, B.empty() ? 1 : B.size()) != grxSuccess ||
            grxMalloc(&dC, C.size() * sizeof(int32_t)) != grxSuccess) {
          std::printf("  FAIL  %s (allocation)\n", label); return false;
        }
        if (!A.empty()) grxMemcpy(dA, A.data(), A.size(), grxMemcpyDefault);
        if (!B.empty()) grxMemcpy(dB, B.data(), B.size(), grxMemcpyDefault);
        grxMemcpy(dC, C.data(), C.size() * sizeof(int32_t), grxMemcpyDefault);

        const grxblasStatus_t st = grxblasGemmEx(
            h, ta ? GRXBLAS_OP_T : GRXBLAS_OP_N, tb ? GRXBLAS_OP_T : GRXBLAS_OP_N,
            m, n, k, &alpha, dA, GRX_R_8I, lda, dB, GRX_R_8I, ldb, &beta,
            dC, GRX_R_32I, ldc);
        if (st != GRXBLAS_STATUS_SUCCESS) {
          std::printf("  FAIL  %s (%s)\n", label, grxblasGetStatusString(st));
          grxFree(dA); grxFree(dB); grxFree(dC); return false;
        }
        grxDeviceSynchronize();
        std::vector<int32_t> got(C.size(), 0);
        grxMemcpy(got.data(), dC, got.size() * sizeof(int32_t), grxMemcpyDefault);
        grxFree(dA); grxFree(dB); grxFree(dC);

        int bad = 0, stray = 0;
        for (size_t idx = 0; idx < C.size(); ++idx) {
          const int i = (int)(idx % (size_t)ldc), j = (int)(idx / (size_t)ldc);
          if (got[idx] == want[idx]) continue;
          if (bad + stray < 3)
            std::printf("        %s(%d,%d) got %d want %d\n",
                        (i < m && j < n) ? "" : "padding ", i, j, got[idx],
                        want[idx]);
          if (i < m && j < n) ++bad; else ++stray;
        }
        if (bad || stray) {
          std::printf("  FAIL  %s (%d wrong, %d stray)\n", label, bad, stray);
          return false;
        }
        std::printf("  ok    %s\n", label);
        return true;
      };

      bool ok = true;
      ok &= int8_case(false, false, tm, tn, 16, 1.0f, 0.0f, 0, "one tile, one k step");
      ok &= int8_case(false, false, 16, 16, 32, 1.0f, 0.0f, 0, "16x16x32");
      ok &= int8_case(false, false, 5, 3, 7, 1.0f, 0.0f, 0, "5x3x7 -- ragged, and k < the tile depth");
      ok &= int8_case(false, false, 17, 9, 29, 2.0f, -1.0f, 3, "17x9x29 ragged, alpha/beta, padded lds");
      ok &= int8_case(true, false, 13, 7, 19, 1.0f, 0.0f, 0, "TN");
      ok &= int8_case(false, true, 13, 7, 19, 1.0f, 0.0f, 0, "NT");
      ok &= int8_case(true, true, 13, 7, 19, 1.0f, 1.0f, 0, "TT with beta");
      check(ok, "every int8 case matches an integer reference exactly");

      const float half_alpha = 2.5f, zero2 = 0.0f;
      void* d = nullptr;
      grxMalloc(&d, 4096);
      check(grxblasGemmEx(h, GRXBLAS_OP_N, GRXBLAS_OP_N, 8, 4, 16, &half_alpha,
                          d, GRX_R_8I, 8, d, GRX_R_8I, 16, &zero2, d,
                          GRX_R_32I, 8) == GRXBLAS_STATUS_INVALID_VALUE,
            "a non-integral alpha is refused rather than rounded");
      grxFree(d);
    }
  }

  section("what the tensor path refuses");
  {
    const float one = 1.0f, zero = 0.0f;
    void* d = nullptr;
    grxMalloc(&d, 4096);
    // A transposed operand used to be refused here. It is implemented now, so
    // what this checks is the leading-dimension rule that comes with it: a
    // transposed A is stored k x m, so lda bounds k and an lda of 4 with k = 8
    // is too small even though it is big enough for m.
    check(grxblasGemmEx(h, GRXBLAS_OP_T, GRXBLAS_OP_N, 4, 4, 8, &one, d,
                        GRX_R_16F, 4, d, GRX_R_16F, 8, &zero, d, GRX_R_32F, 4)
              == GRXBLAS_STATUS_INVALID_VALUE,
          "lda bounds k when A is transposed, not m");
    check(grxblasGemmEx(h, GRXBLAS_OP_N, GRXBLAS_OP_N, 8, 4, 8, &one, d,
                        GRX_R_32F, 8, d, GRX_R_16F, 8, &zero, d, GRX_R_32F, 8)
              == GRXBLAS_STATUS_NOT_SUPPORTED,
          "an fp32 input is refused");
    check(grxblasGemmEx(h, GRXBLAS_OP_N, GRXBLAS_OP_N, 8, 4, 8, &one, d,
                        GRX_R_16F, 2, d, GRX_R_16F, 8, &zero, d, GRX_R_32F, 8)
              == GRXBLAS_STATUS_INVALID_VALUE,
          "a too-small lda is rejected");
    check(grxblasGemmEx(nullptr, GRXBLAS_OP_N, GRXBLAS_OP_N, 8, 4, 8, &one, d,
                        GRX_R_16F, 8, d, GRX_R_16F, 8, &zero, d, GRX_R_32F, 8)
              == GRXBLAS_STATUS_NOT_INITIALIZED,
          "a null handle is rejected");
    grxFree(d);
  }

  check(grxblasDestroy(h) == GRXBLAS_STATUS_SUCCESS, "grxblasDestroy");
  return grxtest::report();
}
