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

// C = alpha * A * B + beta * C, column major, computed from the float values
// the halves were made from.
void reference(int m, int n, int k, float alpha, const std::vector<float>& A,
               int lda, const std::vector<float>& B, int ldb, float beta,
               std::vector<float>& C, int ldc) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      float acc = 0.0f;
      for (int l = 0; l < k; ++l)
        acc += A[(size_t)i + (size_t)l * lda] * B[(size_t)l + (size_t)j * ldb];
      const size_t ci = (size_t)i + (size_t)j * ldc;
      C[ci] = (beta == 0.0f) ? (alpha * acc) : (alpha * acc + beta * C[ci]);
    }
  }
}

bool run_case(grxblasHandle_t h, int m, int n, int k, float alpha, float beta,
              int pad, const char* label) {
  const int lda = m + pad, ldb = (k ? k : 1) + pad, ldc = m + pad;

  std::vector<float> A((size_t)lda * (k ? k : 1)), B((size_t)ldb * n),
      C((size_t)ldc * n);
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
  reference(m, n, k, alpha, A, lda, B, ldb, beta, expected, ldc);

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
      grxblasGemmEx(h, GRXBLAS_OP_N, GRXBLAS_OP_N, m, n, k, &alpha,
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
  all &= run_case(h, tm, tn, tk, 1.0f, 0.0f, 0, "exactly one tile, one k step");
  all &= run_case(h, tm * 2, tn * 2, tk * 2, 1.0f, 0.0f, 0, "four tiles, two k steps");
  all &= run_case(h, 16, 16, 16, 1.0f, 0.0f, 0, "16x16x16");
  all &= run_case(h, 5, 3, 7, 1.0f, 0.0f, 0, "5x3x7 -- every dimension ragged");
  all &= run_case(h, 17, 9, 13, 1.0f, 0.0f, 0, "17x9x13 -- ragged and larger");
  all &= run_case(h, 1, 1, 1, 1.0f, 0.0f, 0, "1x1x1 -- one element, one product");
  all &= run_case(h, 12, 6, 10, 2.5f, -1.5f, 0, "alpha and beta");
  all &= run_case(h, 12, 6, 10, 1.0f, 0.0f, 3, "padded leading dimensions");
  all &= run_case(h, 6, 5, 0, 1.0f, 2.0f, 0, "k = 0 scales C by beta");
  check(all, "every GemmEx case matches the reference exactly");

  section("what the tensor path refuses");
  {
    const float one = 1.0f, zero = 0.0f;
    void* d = nullptr;
    grxMalloc(&d, 4096);
    check(grxblasGemmEx(h, GRXBLAS_OP_T, GRXBLAS_OP_N, 8, 4, 8, &one, d,
                        GRX_R_16F, 8, d, GRX_R_16F, 8, &zero, d, GRX_R_32F, 8)
              == GRXBLAS_STATUS_NOT_SUPPORTED,
          "a transposed operand is refused, not silently handled elsewhere");
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
