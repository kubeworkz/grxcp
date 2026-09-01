// grxBLAS level-1 / level-2 numerical gate: saxpy, sscal, sgemv.
//
// Every value here is a small integer held in a float, so every comparison is
// EXACT. There is no tolerance to hide a wrong answer behind -- unlike the
// sgemm gate, which needs one because the device accumulates in a different
// order and float addition is not associative. At these magnitudes every
// summation order gives the same bits, so any disagreement is a real one.
//
// The references are written from the BLAS definitions rather than from the
// kernels' index expressions. That distinction is the reason the sgemm gate
// once passed a transpose that transposed nothing: two copies of one
// misconception agree with each other perfectly.

#include <grx/grx.h>
#include <grx/grxblas.h>

#include "../unit/grx_test.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using grxtest::check;
using grxtest::section;

namespace {

constexpr float kPoison = -12345.0f;

// BLAS increment addressing, written once here and once on the device. A
// negative increment walks the vector backwards.
size_t vec_index(int i, int n, int inc) {
  return (inc > 0) ? (size_t)i * (size_t)inc
                   : (size_t)(n - 1 - i) * (size_t)(-inc);
}

// How many elements a length-n vector with this increment spans.
size_t span(int n, int inc) {
  const int a = (inc < 0) ? -inc : inc;
  return (n <= 0) ? 0 : (size_t)(n - 1) * (size_t)a + 1;
}

struct DeviceVec {
  void*  ptr = nullptr;
  size_t n   = 0;

  bool alloc(size_t elems) {
    n = elems;
    return grxMalloc(&ptr, elems * sizeof(float)) == grxSuccess;
  }
  bool put(const std::vector<float>& host) {
    return grxMemcpy(ptr, host.data(), host.size() * sizeof(float),
                     grxMemcpyDefault) == grxSuccess;
  }
  std::vector<float> get() const {
    std::vector<float> out(n, 0.0f);
    grxMemcpy(out.data(), ptr, out.size() * sizeof(float), grxMemcpyDefault);
    return out;
  }
  ~DeviceVec() { if (ptr) grxFree(ptr); }
};

// Count elements that differ from the reference, and report the first few.
int diff(const std::vector<float>& got, const std::vector<float>& want,
         const char* label) {
  int bad = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    if (got[i] != want[i]) {
      if (bad < 4)
        std::printf("        %s[%zu] got %g want %g\n", label, i, (double)got[i],
                    (double)want[i]);
      ++bad;
    }
  }
  return bad;
}

// --- saxpy ------------------------------------------------------------------

void test_axpy(grxblasHandle_t h, int n, int incx, int incy, const char* what) {
  const float alpha = 3.0f;

  std::vector<float> x(span(n, incx), kPoison);
  std::vector<float> y(span(n, incy) + 8, kPoison);   // 8 elements of overhang
  for (int i = 0; i < n; ++i) x[vec_index(i, n, incx)] = (float)(i + 1);
  for (int i = 0; i < n; ++i) y[vec_index(i, n, incy)] = (float)(100 + i);

  std::vector<float> want = y;
  for (int i = 0; i < n; ++i)
    want[vec_index(i, n, incy)] =
        alpha * x[vec_index(i, n, incx)] + y[vec_index(i, n, incy)];

  DeviceVec dx, dy;
  if (!dx.alloc(x.size()) || !dy.alloc(y.size())) { check(false, what); return; }
  if (!dx.put(x) || !dy.put(y))                   { check(false, what); return; }

  const grxblasStatus_t s =
      grxblasSaxpy(h, n, &alpha, (const float*)dx.ptr, incx, (float*)dy.ptr, incy);
  if (s != GRXBLAS_STATUS_SUCCESS) {
    std::printf("        status %s\n", grxblasGetStatusString(s));
    check(false, what);
    return;
  }
  grxDeviceSynchronize();
  check(diff(dy.get(), want, "y") == 0, what);
}

// --- sscal ------------------------------------------------------------------

void test_scal(grxblasHandle_t h, int n, int incx, float alpha,
               const char* what) {
  std::vector<float> x(span(n, incx) + 8, kPoison);
  for (int i = 0; i < n; ++i) x[vec_index(i, n, incx)] = (float)(i + 1);

  std::vector<float> want = x;
  for (int i = 0; i < n; ++i)
    want[vec_index(i, n, incx)] = alpha * x[vec_index(i, n, incx)];

  DeviceVec dx;
  if (!dx.alloc(x.size()) || !dx.put(x)) { check(false, what); return; }

  const grxblasStatus_t s = grxblasSscal(h, n, &alpha, (float*)dx.ptr, incx);
  if (s != GRXBLAS_STATUS_SUCCESS) {
    std::printf("        status %s\n", grxblasGetStatusString(s));
    check(false, what);
    return;
  }
  grxDeviceSynchronize();
  check(diff(dx.get(), want, "x") == 0, what);
}

// --- sgemv ------------------------------------------------------------------

void test_gemv(grxblasHandle_t h, grxblasOperation_t trans, int m, int n,
               int lda, int incx, int incy, float alpha, float beta,
               const char* what) {
  const bool tr    = (trans == GRXBLAS_OP_T);
  const int  rows  = tr ? n : m;    // outputs
  const int  depth = tr ? m : n;    // reduction length

  // A as stored: m x n, column major, leading dimension lda. The padding
  // between columns is poisoned so a kernel that walks lda where it should
  // walk m produces a visibly wrong number rather than a nearly right one.
  std::vector<float> A((size_t)lda * (size_t)n, kPoison);
  for (int c = 0; c < n; ++c)
    for (int r = 0; r < m; ++r)
      A[(size_t)r + (size_t)c * (size_t)lda] = (float)(((r * 7 + c * 3) % 17) - 8);

  std::vector<float> x(span(depth, incx), kPoison);
  for (int i = 0; i < depth; ++i)
    x[vec_index(i, depth, incx)] = (float)((i % 9) - 4);

  std::vector<float> y(span(rows, incy) + 8, kPoison);
  for (int i = 0; i < rows; ++i) y[vec_index(i, rows, incy)] = (float)(i + 1);

  // The reference, straight from the definition:
  //   OP_N  y[i] = alpha * sum_j A(i,j) * x[j] + beta * y[i]
  //   OP_T  y[j] = alpha * sum_i A(i,j) * x[i] + beta * y[j]
  std::vector<float> want = y;
  for (int o = 0; o < rows; ++o) {
    float acc = 0.0f;
    for (int d = 0; d < depth; ++d) {
      const int ar = tr ? d : o;
      const int ac = tr ? o : d;
      acc += A[(size_t)ar + (size_t)ac * (size_t)lda] *
             x[vec_index(d, depth, incx)];
    }
    const size_t yi = vec_index(o, rows, incy);
    want[yi] = (beta == 0.0f) ? (alpha * acc) : (alpha * acc + beta * y[yi]);
  }

  // With beta == 0 the kernel must not READ y. Filling the output positions
  // with NaN proves it: 0 * NaN is NaN, so a kernel that multiplies through
  // produces NaN everywhere and the comparison catches it.
  if (beta == 0.0f) {
    const float nan = std::nanf("");
    for (int i = 0; i < rows; ++i) y[vec_index(i, rows, incy)] = nan;
  }

  DeviceVec dA, dx, dy;
  if (!dA.alloc(A.size()) || !dx.alloc(x.size()) || !dy.alloc(y.size())) {
    check(false, what); return;
  }
  if (!dA.put(A) || !dx.put(x) || !dy.put(y)) { check(false, what); return; }

  const grxblasStatus_t s =
      grxblasSgemv(h, trans, m, n, &alpha, dA.ptr, lda, dx.ptr, incx, &beta,
                   dy.ptr, incy);
  if (s != GRXBLAS_STATUS_SUCCESS) {
    std::printf("        status %s\n", grxblasGetStatusString(s));
    check(false, what);
    return;
  }
  grxDeviceSynchronize();
  check(diff(dy.get(), want, "y") == 0, what);

  // A must not have been written to.
  check(diff(dA.get(), A, "A") == 0, "sgemv leaves A alone");
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
  if (grxblasCreate(&h) != GRXBLAS_STATUS_SUCCESS) return 1;

  // Probe before claiming to test anything: without the device toolchain the
  // .vxbin does not exist and every case below fails identically, which reads
  // as "the library is broken" when the truth is "nobody compiled it".
  {
    void* d = nullptr;
    const float one = 1.0f;
    if (grxMalloc(&d, 4 * sizeof(float)) == grxSuccess) {
      const grxblasStatus_t probe = grxblasSscal(h, 1, &one, (float*)d, 1);
      grxFree(d);
      if (probe == GRXBLAS_STATUS_NOT_INITIALIZED) {
        std::printf("grxblas device kernels not found; skipping\n");
        grxblasDestroy(h);
        return 77;
      }
      if (probe == GRXBLAS_STATUS_NOT_SUPPORTED) {
        std::printf("the loaded module has no level-1 entry points; skipping\n"
                    "  rebuild it from src/libs/grxblas/kernels/all.cpp\n");
        grxblasDestroy(h);
        return 77;
      }
    }
  }

  section("saxpy");
  test_axpy(h, 37, 1, 1,   "unit increments");
  test_axpy(h, 21, 2, 3,   "strided, and the gaps are untouched");
  test_axpy(h, 19, -1, 1,  "negative incx walks x backwards");
  test_axpy(h, 19, 1, -2,  "negative incy walks y backwards");
  test_axpy(h, 1,  1, 1,   "a single element");
  {
    const float a = 1.0f;
    void* d = nullptr;
    grxMalloc(&d, 4 * sizeof(float));
    check(grxblasSaxpy(h, 0, &a, (float*)d, 1, (float*)d, 1) ==
              GRXBLAS_STATUS_SUCCESS,
          "n = 0 succeeds and does nothing");
    check(grxblasSaxpy(h, 4, &a, (float*)d, 0, (float*)d, 1) ==
              GRXBLAS_STATUS_INVALID_VALUE,
          "a zero increment is refused, not silently aliased");
    grxFree(d);
  }

  section("sscal");
  test_scal(h, 33, 1, 0.5f,  "unit increment, and 0.5 is exact in binary");
  test_scal(h, 17, 3, 2.0f,  "strided, and the gaps are untouched");
  test_scal(h, 25, 1, 0.0f,  "alpha = 0 zeroes the vector");
  {
    const float a = 1.0f;
    void* d = nullptr;
    grxMalloc(&d, 4 * sizeof(float));
    check(grxblasSscal(h, 4, &a, (float*)d, -1) == GRXBLAS_STATUS_INVALID_VALUE,
          "a negative increment is refused, as BLAS requires");
    grxFree(d);
  }

  section("sgemv, untransposed");
  test_gemv(h, GRXBLAS_OP_N, 13, 7, 13, 1, 1, 2.0f, 3.0f, "square-ish, ld = m");
  test_gemv(h, GRXBLAS_OP_N, 13, 7, 19, 1, 1, 2.0f, 3.0f, "padded leading dimension");
  test_gemv(h, GRXBLAS_OP_N, 13, 7, 19, 2, 3, 2.0f, 3.0f, "strided x and y");
  test_gemv(h, GRXBLAS_OP_N, 13, 7, 19, 1, 1, 1.0f, 0.0f, "beta = 0 must not read y");
  test_gemv(h, GRXBLAS_OP_N,  1, 9, 1,  1, 1, 2.0f, 3.0f, "one row");
  test_gemv(h, GRXBLAS_OP_N,  9, 1, 9,  1, 1, 2.0f, 3.0f, "one column");
  // A NEGATIVE incx WAS NOT COVERED HERE, and it is the case the kernel treats
  // least like the others: BLAS reads such a vector backwards from its far end,
  // so the walk starts at (depth-1)*|incx| and steps DOWN. saxpy has had the
  // equivalent case since it was written; sgemv had incx of 1 and 2 only, and
  // a reversed y but never a reversed x. The gaps between used elements are
  // poisoned, so a walk that starts at the wrong end reads -12345 and says so.
  test_gemv(h, GRXBLAS_OP_N, 13, 7, 19, -1, 1, 2.0f, 3.0f, "reversed x");
  test_gemv(h, GRXBLAS_OP_N, 13, 7, 19, -3, -2, 2.0f, 3.0f, "reversed x and y, both strided");

  section("sgemv, transposed -- a different traversal, not a variant");
  test_gemv(h, GRXBLAS_OP_T, 13, 7, 13, 1, 1, 2.0f, 3.0f, "square-ish, ld = m");
  test_gemv(h, GRXBLAS_OP_T, 13, 7, 19, 1, 1, 2.0f, 3.0f, "padded leading dimension");
  test_gemv(h, GRXBLAS_OP_T, 13, 7, 19, 2, -3, 2.0f, 3.0f, "strided x, reversed y");
  test_gemv(h, GRXBLAS_OP_T, 13, 7, 19, 1, 1, 1.0f, 0.0f, "beta = 0 must not read y");
  // The reduction is longer than the warp here, so the warp-per-column path
  // has to loop and then reduce across lanes -- which is the case a
  // single-pass reduction gets wrong.
  test_gemv(h, GRXBLAS_OP_T, 37, 5, 37, 1, 1, 2.0f, 3.0f, "reduction longer than a warp");
  // The transposed path is where a reversed x is hardest: each LANE starts at
  // its own offset into x and steps by incx * warp_width, so the sign decides
  // both where the lane begins and which way it travels. Two shapes, because
  // they fail differently -- the long one gets the stride wrong, the short one
  // gets the lanes that have no element at all wrong.
  test_gemv(h, GRXBLAS_OP_T, 37, 5, 37, -1, 1, 2.0f, 3.0f, "reversed x, reduction longer than a warp");
  test_gemv(h, GRXBLAS_OP_T, 37, 5, 37, -2, 3, 2.0f, 3.0f, "reversed x, strided both ways");
  // depth = 3 is shorter than the warp, so lane 3 has no element. Its start
  // offset is the one that underflows if it is computed before the bound is
  // tested -- which is exactly what hoisting the index out of the loop moved.
  test_gemv(h, GRXBLAS_OP_T,  3, 5,  3, -1, 1, 2.0f, 3.0f, "reversed x, reduction shorter than a warp");

  section("sgemv argument checking");
  {
    const float a = 1.0f, b = 0.0f;
    void* d = nullptr;
    grxMalloc(&d, 64 * sizeof(float));
    check(grxblasSgemv(h, GRXBLAS_OP_N, 4, 4, &a, d, 3, d, 1, &b, d, 1) ==
              GRXBLAS_STATUS_INVALID_VALUE,
          "lda < m is refused");
    check(grxblasSgemv(h, GRXBLAS_OP_N, 0, 4, &a, d, 4, d, 1, &b, d, 1) ==
              GRXBLAS_STATUS_SUCCESS,
          "m = 0 succeeds and does nothing");
    grxFree(d);
  }

  grxblasDestroy(h);
  return grxtest::report();
}
