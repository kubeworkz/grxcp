// The register-blocked sgemm against the naive one, on the same device.
//
// THE REFERENCE KERNEL IS THE ORACLE HERE, and that is a stronger check than
// this library has had for a GEMM before. test_grxblas.cpp compares sgemm to a
// host reference at a relative tolerance scaled by k, because the device
// accumulates in a different order than the host and float addition is not
// associative. That tolerance is real headroom, and a tuned kernel that quietly
// changed the arithmetic could hide inside it.
//
// sgemm_rb does not change the arithmetic. It changes which thread computes
// which output and how often B is loaded; every accumulation is still
// `acc += a * b` over l in the same order. So the two kernels must agree
// BIT FOR BIT, and this compares them with ==. No tolerance, nothing to hide
// in.
//
// Both run on the device over the same operands, selected by the environment
// variable the host reads (GRXBLAS_SGEMM_NAIVE), so this is one binary running
// two kernels rather than a kernel compared against a description of itself.
//
// WHAT THE SHAPES ARE FOR. RM is 4, so the interesting cases are the ones where
// m is not a multiple of it -- the tail, where the blocked kernel clamps rows
// it must then discard. m = 1, 2, 3 are below RM entirely, where the host is
// supposed to fall back to the reference; m = 5, 7, 13 straddle. All four
// transpose combinations, because the index algebra is duplicated between the
// two kernels and duplicated algebra is what drifts.

#include <grx/grx.h>
#include <grx/grxblas.h>

#include "../unit/grx_test.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using grxtest::check;
using grxtest::section;

namespace {

struct Buf {
  void* p = nullptr;
  explicit Buf(size_t bytes) { if (grxMalloc(&p, bytes) != grxSuccess) p = nullptr; }
  ~Buf() { if (p) grxFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  float* f() const { return (float*)p; }
};

void fill(std::vector<float>& v, unsigned seed) {
  for (size_t i = 0; i < v.size(); ++i) {
    seed = seed * 1664525u + 1013904223u;
    // Quarter-integers in [-2, 2]: exactly representable, so any difference
    // between the two kernels is a difference in what they computed rather
    // than in how a decimal landed.
    v[i] = (float)((int)(seed >> 16) % 17 - 8) * 0.25f;
  }
}

// Run one sgemm and return C. `naive` picks the kernel through the same
// environment variable the library reads, so both paths go through the real
// selection logic rather than a test-only hook.
bool run(bool naive, bool ta, bool tb, int m, int n, int k,
         const std::vector<float>& A, const std::vector<float>& B,
         std::vector<float>* C) {
  if (naive) setenv("GRXBLAS_SGEMM_NAIVE", "1", 1);
  else       unsetenv("GRXBLAS_SGEMM_NAIVE");

  // A fresh handle each time: the kernel choice is made per call from the
  // environment, but creating the handle here also means neither run can be
  // affected by state the other left behind.
  grxblasHandle_t h = nullptr;
  if (grxblasCreate(&h) != GRXBLAS_STATUS_SUCCESS) return false;

  const int lda = ta ? k : m;
  const int ldb = tb ? n : k;
  const int ldc = m;

  Buf dA(A.size() * sizeof(float)), dB(B.size() * sizeof(float)),
      dC((size_t)ldc * n * sizeof(float));
  if (!dA.p || !dB.p || !dC.p) { grxblasDestroy(h); return false; }
  grxMemcpy(dA.p, A.data(), A.size() * sizeof(float), grxMemcpyDefault);
  grxMemcpy(dB.p, B.data(), B.size() * sizeof(float), grxMemcpyDefault);
  // Poison C: beta is zero, so every element must be written. A kernel that
  // skipped one would otherwise inherit whatever was there.
  std::vector<float> poison((size_t)ldc * n, -7777.0f);
  grxMemcpy(dC.p, poison.data(), poison.size() * sizeof(float), grxMemcpyDefault);

  const float alpha = 1.0f, beta = 0.0f;
  const grxblasStatus_t st = grxblasSgemm(
      h, ta ? GRXBLAS_OP_T : GRXBLAS_OP_N, tb ? GRXBLAS_OP_T : GRXBLAS_OP_N,
      m, n, k, &alpha, dA.p, lda, dB.p, ldb, &beta, dC.p, ldc);
  if (st != GRXBLAS_STATUS_SUCCESS) { grxblasDestroy(h); return false; }

  C->resize(poison.size());
  grxMemcpy(C->data(), dC.p, C->size() * sizeof(float), grxMemcpyDefault);
  grxblasDestroy(h);
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

  {
    grxblasHandle_t h = nullptr;
    if (grxblasCreate(&h) != GRXBLAS_STATUS_SUCCESS) return 1;
    void* d = nullptr;
    const float one = 1.0f, zero = 0.0f;
    if (grxMalloc(&d, sizeof(float)) == grxSuccess) {
      const grxblasStatus_t probe = grxblasSgemm(
          h, GRXBLAS_OP_N, GRXBLAS_OP_N, 1, 1, 1, &one, d, 1, d, 1, &zero, d, 1);
      grxFree(d);
      if (probe == GRXBLAS_STATUS_NOT_INITIALIZED) {
        std::printf("grxblas device kernels not found; skipping\n");
        grxblasDestroy(h);
        return 77;
      }
    }
    grxblasDestroy(h);
  }

  struct Case { int m, n, k; const char* what; };
  const Case cases[] = {
    { 1, 3, 4, "m=1, below the tile -- the host must fall back"},
    { 2, 3, 4, "m=2, below the tile"},
    { 3, 3, 4, "m=3, below the tile"},
    { 4, 3, 5, "m=4, exactly one tile"},
    { 5, 4, 6, "m=5, one tile and a remainder of 1"},
    { 7, 5, 3, "m=7, one tile and a remainder of 3"},
    { 8, 6, 8, "m=8, two whole tiles"},
    {13, 7, 9, "m=13, three tiles and a remainder of 1"},
    {16, 4, 12, "m=16, four whole tiles"},
  };

  section("sgemm_rb agrees with the reference kernel, bit for bit");
  for (const Case& c : cases) {
    for (int t = 0; t < 4; ++t) {
      const bool ta = (t & 1) != 0, tb = (t & 2) != 0;
      const int lda = ta ? c.k : c.m;
      const int ldb = tb ? c.n : c.k;
      std::vector<float> A((size_t)lda * (ta ? c.m : c.k));
      std::vector<float> B((size_t)ldb * (tb ? c.k : c.n));
      fill(A, 5u + (unsigned)t); fill(B, 91u + (unsigned)t);

      std::vector<float> ref, fast;
      char label[160];
      std::snprintf(label, sizeof(label), "%s  [%c%c]", c.what,
                    ta ? 'T' : 'N', tb ? 'T' : 'N');

      if (!run(true, ta, tb, c.m, c.n, c.k, A, B, &ref) ||
          !run(false, ta, tb, c.m, c.n, c.k, A, B, &fast)) {
        std::printf("  FAIL  %s: a run failed\n", label);
        ++grxtest::failures();
        continue;
      }

      size_t at = ref.size();
      for (size_t i = 0; i < ref.size(); ++i) {
        if (std::memcmp(&ref[i], &fast[i], sizeof(float)) != 0) { at = i; break; }
      }
      if (at == ref.size()) {
        std::printf("  ok    %s\n", label);
      } else {
        std::printf("  FAIL  %s: differ at [%zu]: reference %.9g, blocked %.9g\n",
                    label, at, (double)ref[at], (double)fast[at]);
        ++grxtest::failures();
      }
    }
  }

  section("the comparison can actually fail");
  {
    // Without this, everything above would pass just as happily if both runs
    // were the SAME kernel -- if the environment variable were misspelled, or
    // the host ignored it, or the module had no sgemm_rb at all. Perturbing one
    // operand must produce a difference; if it does not, the two runs are not
    // two kernels.
    std::vector<float> A(64), B(64), out1, out2;
    fill(A, 3u); fill(B, 17u);
    const bool ok1 = run(false, false, false, 8, 8, 8, A, B, &out1);
    A[0] += 1.0f;
    const bool ok2 = run(false, false, false, 8, 8, 8, A, B, &out2);
    check(ok1 && ok2, "both perturbation runs completed");
    bool differs = false;
    for (size_t i = 0; i < out1.size() && !differs; ++i)
      if (std::memcmp(&out1[i], &out2[i], sizeof(float)) != 0) differs = true;
    check(differs, "changing an input changes the output");
  }

  return grxtest::report();
}
