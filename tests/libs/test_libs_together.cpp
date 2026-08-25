// TWO LIBRARIES, ONE PROCESS. That is the whole gate.
//
// grxBLAS and grxDNN each ship precompiled device kernels in a .vxbin, and
// every .vxbin links at the same fixed load address (STARTUP_ADDR), so exactly
// ONE module can be resident on the device at a time -- cuda_mapping.md 7.13.
// Within a library that is a packaging detail: grxBLAS puts sgemm and GemmEx in
// one image and nobody outside notices. Across libraries it is a program that
// does not run. A transformer layer is attention and GEMM from grxBLAS and
// layer norm and softmax from grxDNN, in one process, and the second library to
// initialise would fail to load its kernels.
//
// src/libs/kernels_all.cpp is the answer: one image holding both libraries'
// entry points, which both libraries prefer. This file is what checks that the
// answer works, and it checks it three ways, because the interesting failures
// are not "it did not run".
//
//   1. INTERLEAVED, not sequential. blas, dnn, blas, dnn. A fix that unloads
//      the first library's module to make room for the second passes a
//      sequential test and fails here on the third call -- and unloading is the
//      obvious thing to reach for, so it is the thing worth catching.
//
//   2. THE NUMBERS, every time. A module that is resident but wrong produces
//      arithmetic, not an error. Each call is checked against a host reference,
//      including the calls after the other library has been through.
//
//   3. THE SAME FILE. Both libraries report where they loaded from and the two
//      paths must be equal. Two separate images that both happened to load
//      would give identical numbers right up until the day they did not fit,
//      so the shared image is asserted rather than inferred.
//
// The negative control is in ci/run_real.sh: the same binary is run against a
// directory holding two SEPARATE per-library images and no combined one, where
// it must fail. Without that run, this file passing would only show that
// something works, not that the thing it names is what makes it work.

#include <grx/grx.h>
#include <grx/grxblas.h>
#include <grx/grxdnn.h>

#include "../unit/grx_test.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using grxtest::check;
using grxtest::section;

namespace {

// A device buffer that frees itself, because this test has enough early exits
// that hand-written cleanup would leak on most of them.
struct Buf {
  void* p = nullptr;
  explicit Buf(size_t bytes) {
    if (grxMalloc(&p, bytes) != grxSuccess) p = nullptr;
  }
  ~Buf() { if (p) grxFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
};

void fill(std::vector<float>& v, unsigned seed) {
  for (size_t i = 0; i < v.size(); ++i) {
    seed = seed * 1664525u + 1013904223u;
    v[i] = (float)((int)(seed >> 16) % 17 - 8) * 0.25f;
  }
}

// C = A * B, column major, no transposes. Small and square: this gate is about
// which library is resident, and test_grxblas.cpp already covers the shapes.
void reference_sgemm(int n, const std::vector<float>& A,
                     const std::vector<float>& B, std::vector<float>& C) {
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      double acc = 0.0;
      for (int l = 0; l < n; ++l)
        acc += (double)A[(size_t)i + (size_t)l * n] *
               (double)B[(size_t)l + (size_t)j * n];
      C[(size_t)i + (size_t)j * n] = (float)acc;
    }
}

void reference_layernorm(int rows, int cols, const std::vector<float>& x,
                         float eps, std::vector<float>& y) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x.data() + (size_t)r * cols;
    float* yr = y.data() + (size_t)r * cols;
    double mean = 0.0;
    for (int j = 0; j < cols; ++j) mean += xr[j];
    mean /= cols;
    double var = 0.0;
    for (int j = 0; j < cols; ++j) var += (xr[j] - mean) * (xr[j] - mean);
    var /= cols;                       // biased, as grxdnn.h states
    const double scale = 1.0 / std::sqrt(var + eps);
    for (int j = 0; j < cols; ++j) yr[j] = (float)((xr[j] - mean) * scale);
  }
}

void reference_softmax(int rows, int cols, const std::vector<float>& x,
                       std::vector<float>& y) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x.data() + (size_t)r * cols;
    float* yr = y.data() + (size_t)r * cols;
    float m = xr[0];
    for (int j = 1; j < cols; ++j) if (xr[j] > m) m = xr[j];
    double sum = 0.0;
    for (int j = 0; j < cols; ++j) sum += std::exp((double)xr[j] - m);
    for (int j = 0; j < cols; ++j)
      yr[j] = (float)(std::exp((double)xr[j] - m) / sum);
  }
}

bool close(const std::vector<float>& got, const std::vector<float>& want,
           float tol, const char* label) {
  float worst = 0.0f;
  size_t at = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    const float d = std::fabs(got[i] - want[i]);
    if (d > worst) { worst = d; at = i; }
  }
  if (worst <= tol) {
    std::printf("  ok    %s (worst |diff| %.3g)\n", label, (double)worst);
    return true;
  }
  std::printf("  FAIL  %s: worst |diff| %.3g at [%zu] (got %.9g, want %.9g)\n",
              label, (double)worst, at, (double)got[at], (double)want[at]);
  ++grxtest::failures();
  return false;
}

constexpr int kN    = 8;    // sgemm is kN x kN x kN
constexpr int kRows = 4;    // the norms are kRows x kCols, row major
constexpr int kCols = 12;

// One sgemm, checked. `tag` distinguishes the call before grxDNN has loaded
// anything from the call after.
bool blas_round(grxblasHandle_t h, unsigned seed, const char* tag) {
  std::vector<float> A((size_t)kN * kN), B((size_t)kN * kN);
  fill(A, seed); fill(B, seed + 7u);
  std::vector<float> want((size_t)kN * kN);
  reference_sgemm(kN, A, B, want);

  const size_t bytes = A.size() * sizeof(float);
  Buf dA(bytes), dB(bytes), dC(bytes);
  if (!dA.p || !dB.p || !dC.p) {
    std::printf("  FAIL  %s (allocation)\n", tag);
    ++grxtest::failures();
    return false;
  }
  grxMemcpy(dA.p, A.data(), bytes, grxMemcpyDefault);
  grxMemcpy(dB.p, B.data(), bytes, grxMemcpyDefault);

  const float one = 1.0f, zero = 0.0f;
  const grxblasStatus_t st =
      grxblasSgemm(h, GRXBLAS_OP_N, GRXBLAS_OP_N, kN, kN, kN, &one,
                   dA.p, kN, dB.p, kN, &zero, dC.p, kN);
  if (st != GRXBLAS_STATUS_SUCCESS) {
    std::printf("  FAIL  %s: %s\n", tag, grxblasGetStatusString(st));
    ++grxtest::failures();
    return false;
  }
  std::vector<float> got(A.size());
  grxMemcpy(got.data(), dC.p, bytes, grxMemcpyDefault);
  // kN terms of products of quarter-integers in [-2, 2]: the products are
  // exact, so only the summation order differs from the reference.
  return close(got, want, 1e-4f, tag);
}

bool dnn_round(grxdnnHandle_t h, bool softmax, unsigned seed, const char* tag) {
  std::vector<float> x((size_t)kRows * kCols);
  fill(x, seed);
  std::vector<float> want(x.size());
  if (softmax) reference_softmax(kRows, kCols, x, want);
  else         reference_layernorm(kRows, kCols, x, 1e-5f, want);

  const size_t bytes = x.size() * sizeof(float);
  Buf dX(bytes), dY(bytes);
  if (!dX.p || !dY.p) {
    std::printf("  FAIL  %s (allocation)\n", tag);
    ++grxtest::failures();
    return false;
  }
  grxMemcpy(dX.p, x.data(), bytes, grxMemcpyDefault);

  const grxdnnStatus_t st =
      softmax ? grxdnnSoftmaxForward(h, kRows, kCols, (const float*)dX.p, kCols,
                                     (float*)dY.p, kCols)
              : grxdnnLayerNormForward(h, kRows, kCols, (const float*)dX.p,
                                       kCols, nullptr, nullptr, 1e-5f,
                                       (float*)dY.p, kCols);
  if (st != GRXDNN_STATUS_SUCCESS) {
    std::printf("  FAIL  %s: %s\n", tag, grxdnnGetStatusString(st));
    ++grxtest::failures();
    return false;
  }
  std::vector<float> got(x.size());
  grxMemcpy(got.data(), dY.p, bytes, grxMemcpyDefault);
  return close(got, want, 2e-5f, tag);
}

}  // namespace

int main() {
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count <= 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }
  if (grxSetDevice(0) != grxSuccess) return 1;

  grxblasHandle_t bh = nullptr;
  grxdnnHandle_t  dh = nullptr;
  if (grxblasCreate(&bh) != GRXBLAS_STATUS_SUCCESS) {
    std::printf("grxblasCreate failed\n");
    return 1;
  }
  if (grxdnnCreate(&dh) != GRXDNN_STATUS_SUCCESS) {
    std::printf("grxdnnCreate failed\n");
    grxblasDestroy(bh);
    return 1;
  }

  // Probe grxBLAS before claiming anything. Without the device toolchain there
  // is no .vxbin at all, and every case below would fail identically -- which
  // reads as "the libraries collide" when the truth is "nobody compiled them".
  //
  // grxDNN is deliberately NOT probed the same way. A missing grxDNN module is
  // exactly the failure this gate exists to catch, so treating it as a skip
  // would turn the gate off precisely when it fires.
  {
    void* d = nullptr;
    const float one = 1.0f, zero = 0.0f;
    if (grxMalloc(&d, sizeof(float)) == grxSuccess) {
      const grxblasStatus_t probe = grxblasSgemm(
          bh, GRXBLAS_OP_N, GRXBLAS_OP_N, 1, 1, 1, &one, d, 1, d, 1, &zero, d, 1);
      grxFree(d);
      if (probe == GRXBLAS_STATUS_NOT_INITIALIZED) {
        std::printf("no device kernels; skipping\n"
                    "  build them with: ci/build_kernel.sh --grxgpu <path> "
                    "src/libs/kernels_all.cpp -o grxlibs_kernels.vxbin\n");
        grxdnnDestroy(dh);
        grxblasDestroy(bh);
        return 77;
      }
    }
  }

  section("grxBLAS and grxDNN interleaved in one process");
  blas_round(bh, 3u,  "sgemm, before grxDNN has loaded anything");
  dnn_round (dh, false, 11u, "layernorm, with grxBLAS already resident");
  blas_round(bh, 23u, "sgemm again, with grxDNN now resident too");
  dnn_round (dh, true,  31u, "softmax, after grxBLAS has run again");

  section("both libraries resolved their kernels from the same image");
  const char* bpath = nullptr;
  const char* dpath = nullptr;
  const bool bok =
      grxblasGetLoadedKernelPath(bh, &bpath) == GRXBLAS_STATUS_SUCCESS && bpath;
  const bool dok =
      grxdnnGetLoadedKernelPath(dh, &dpath) == GRXDNN_STATUS_SUCCESS && dpath;
  check(bok, "grxBLAS reports which file it loaded");
  check(dok, "grxDNN reports which file it loaded");
  if (bok && dok) {
    std::printf("        grxblas: %s\n        grxdnn:  %s\n", bpath, dpath);
    check(std::strcmp(bpath, dpath) == 0,
          "the two paths are the same file, not two images that both fit");
  }

  // ONE LIBRARY GOES AWAY AND THE OTHER KEEPS WORKING.
  //
  // Sharing a module means sharing its lifetime, and the counting is the part
  // that is easy to get wrong in the direction that looks fine: grxBLAS's
  // handle is destroyed here, which releases its reference, and if that release
  // took the image off the device with it then grxDNN is left holding kernels
  // in memory nobody owns. A launch after that does not report an error -- the
  // function handle is still valid, the address is still an address -- so this
  // is checked by the NUMBERS coming back right, not by a status code.
  section("grxDNN outlives grxBLAS");
  grxblasDestroy(bh);
  bh = nullptr;
  dnn_round(dh, false, 47u, "layernorm after grxBLAS was destroyed");

  grxdnnDestroy(dh);
  if (bh) grxblasDestroy(bh);
  return grxtest::report();
}
