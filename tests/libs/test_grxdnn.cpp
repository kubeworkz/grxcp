// grxDNN v0: softmax and layer norm against a CPU reference.
//
// Both ops are checked on shapes that are awkward on purpose -- a row shorter
// than a warp, a row much longer, a ragged row count, a padded leading
// dimension -- because a row-wise reduction that only ever sees rows which
// divide evenly by the warp width is a reduction nobody has tested.
//
// THE TWO CONTROLS ARE THE POINT OF THIS FILE.
//
//   Softmax has a row of large logits. The naive exp(x)/sum(exp(x)) overflows
//   fp32 there and produces inf/inf = NaN. The stable form -- subtract the row
//   max first -- gets it right. If someone ever "simplifies" the kernel back,
//   this row is what notices.
//
//   Layer norm has a row with a large mean and a small spread. The one-pass
//   variance identity E[x^2] - E[x]^2 cancels two nearly equal large numbers
//   there and loses most of its significant digits; it can even come out
//   negative, which makes rsqrt produce a NaN. The two-pass form cannot.
//
// Both controls are computed on the HOST as well, in the wrong form, and the
// test asserts the wrong form actually fails -- so the row is known to be
// discriminating rather than assumed to be.

#include <grx/grx.h>
#include <grx/grxdnn.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include <vector>

#include "../unit/grx_test.h"

namespace {

// The shared harness takes a plain string; these checks want to report the
// shape and the worst difference, so this formats and forwards.
void checkf(bool ok, const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  grxtest::check(ok, buf);
}

// ---- CPU references -------------------------------------------------------

void ref_softmax(const std::vector<float>& x, int rows, int cols, int ld,
                 std::vector<float>* y) {
  y->assign(x.size(), 0.0f);
  for (int r = 0; r < rows; ++r) {
    const float* xr = x.data() + (size_t)r * ld;
    float* yr = y->data() + (size_t)r * ld;
    double m = -1e300;
    for (int j = 0; j < cols; ++j) if (xr[j] > m) m = xr[j];
    double s = 0.0;
    for (int j = 0; j < cols; ++j) s += std::exp((double)xr[j] - m);
    for (int j = 0; j < cols; ++j)
      yr[j] = (float)(std::exp((double)xr[j] - m) / s);
  }
}

void ref_layernorm(const std::vector<float>& x, int rows, int cols, int ld,
                   const std::vector<float>* gamma,
                   const std::vector<float>* beta, float eps,
                   std::vector<float>* y) {
  y->assign(x.size(), 0.0f);
  for (int r = 0; r < rows; ++r) {
    const float* xr = x.data() + (size_t)r * ld;
    float* yr = y->data() + (size_t)r * ld;
    double mean = 0.0;
    for (int j = 0; j < cols; ++j) mean += xr[j];
    mean /= cols;
    double var = 0.0;
    for (int j = 0; j < cols; ++j) {
      const double d = (double)xr[j] - mean;
      var += d * d;
    }
    var /= cols;                       // BIASED, as PyTorch's LayerNorm is
    const double scale = 1.0 / std::sqrt(var + (double)eps);
    for (int j = 0; j < cols; ++j) {
      double t = ((double)xr[j] - mean) * scale;
      if (gamma) t *= (*gamma)[j];
      if (beta)  t += (*beta)[j];
      yr[j] = (float)t;
    }
  }
}

// The naive forms, on the host, so the controls can be shown to discriminate.
bool naive_softmax_survives(const float* row, int cols) {
  double s = 0.0;
  for (int j = 0; j < cols; ++j) s += std::exp((double)(float)std::exp(row[j]));
  (void)s;
  // In fp32: sum of exp(x) with no max subtraction.
  float sum = 0.0f;
  for (int j = 0; j < cols; ++j) sum += std::exp(row[j]);
  if (!std::isfinite(sum)) return false;
  for (int j = 0; j < cols; ++j)
    if (!std::isfinite(std::exp(row[j]) / sum)) return false;
  return true;
}

bool naive_variance_survives(const float* row, int cols) {
  // E[x^2] - E[x]^2 in fp32.
  float s = 0.0f, s2 = 0.0f;
  for (int j = 0; j < cols; ++j) { s += row[j]; s2 += row[j] * row[j]; }
  const float mean = s / (float)cols;
  const float var  = s2 / (float)cols - mean * mean;
  // "Survives" means it is usable: non-negative and within 1% of the truth.
  double m = 0.0;
  for (int j = 0; j < cols; ++j) m += row[j];
  m /= cols;
  double v = 0.0;
  for (int j = 0; j < cols; ++j) { const double d = row[j] - m; v += d * d; }
  v /= cols;
  if (!(var > 0.0f)) return false;
  return std::fabs((double)var - v) <= 0.01 * v;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b,
                   int rows, int cols, int ld, int* worst_r, int* worst_c) {
  float worst = 0.0f;
  for (int r = 0; r < rows; ++r)
    for (int j = 0; j < cols; ++j) {
      const size_t k = (size_t)r * ld + j;
      const float d = std::fabs(a[k] - b[k]);
      if (d > worst) { worst = d; *worst_r = r; *worst_c = j; }
    }
  return worst;
}

}  // namespace

int main() {
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count == 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }
  if (grxSetDevice(0) != grxSuccess) return 1;

  grxtest::section("grxdnn");

  grxdnnHandle_t h = nullptr;
  if (grxdnnCreate(&h) != GRXDNN_STATUS_SUCCESS) {
    std::printf("grxdnnCreate failed\n");
    return 1;
  }

  // Probe once before claiming to test anything. Without the device toolchain
  // there is no .vxbin, every case below fails identically, and the run reads
  // as "grxDNN is broken" when the truth is "nobody compiled the kernels".
  // test_grxblas.cpp has done this from the start; this file went without it
  // and duly reported fourteen numerical failures the first time it was run
  // under cmake against the mock driver.
  {
    void* d = nullptr;
    if (grxMalloc(&d, sizeof(float)) == grxSuccess) {
      const grxdnnStatus_t probe =
          grxdnnSoftmaxForward(h, 1, 1, (const float*)d, 1, (float*)d, 1);
      grxFree(d);
      if (probe == GRXDNN_STATUS_NOT_INITIALIZED) {
        std::printf("grxdnn device kernels not found; skipping\n"
                    "  build them with: ci/build_kernel.sh --grxgpu <path> "
                    "src/libs/kernels_all.cpp -o grxlibs_kernels.vxbin\n");
        grxdnnDestroy(h);
        return 77;
      }
    }
  }


  // Shapes: a row shorter than the warp, one longer, a ragged count, and a
  // padded leading dimension so ldx > cols is exercised rather than assumed.
  struct Case { int rows, cols, pad; const char* what; };
  const Case cases[] = {
    {  1,   1, 0, "1x1"},
    {  3,   2, 0, "3x2, rows shorter than a warp"},
    {  5,  17, 3, "5x17 ragged, padded ld"},
    { 33,  64, 0, "33x64"},
    {  7, 129, 5, "7x129, long rows and padded ld"},
  };

  for (const Case& c : cases) {
    const int ld = c.cols + c.pad;
    const size_t n = (size_t)c.rows * ld;
    std::vector<float> hx(n), want, got(n, -1.0f);
    for (size_t i = 0; i < n; ++i)
      hx[i] = std::sin((float)i * 0.7f) * 3.0f + 0.25f * (float)(i % 5);

    float *dx = nullptr, *dy = nullptr;
    if (grxMalloc((void**)&dx, n * sizeof(float)) != grxSuccess) return 1;
    if (grxMalloc((void**)&dy, n * sizeof(float)) != grxSuccess) return 1;
    grxMemcpy(dx, hx.data(), n * sizeof(float), grxMemcpyHostToDevice);
    grxMemset(dy, 0, n * sizeof(float));

    // --- softmax ---
    grxdnnStatus_t st = grxdnnSoftmaxForward(h, c.rows, c.cols, dx, ld, dy, ld);
    if (st != GRXDNN_STATUS_SUCCESS) {
      checkf(false, "softmax %s -> %s", c.what, grxdnnGetStatusString(st));
    } else {
      grxDeviceSynchronize();
      grxMemcpy(got.data(), dy, n * sizeof(float), grxMemcpyDeviceToHost);
      ref_softmax(hx, c.rows, c.cols, ld, &want);
      int wr = 0, wc = 0;
      const float d = max_abs_diff(got, want, c.rows, c.cols, ld, &wr, &wc);
      checkf(d <= 1e-5f, "softmax %s (worst |diff| %.3g at [%d][%d])",
              c.what, (double)d, wr, wc);

      // Every row must sum to 1. A kernel that got the max right and the sum
      // wrong can still match elementwise to 1e-5 on a peaked row.
      float worst_sum = 0.0f;
      for (int r = 0; r < c.rows; ++r) {
        double s = 0.0;
        for (int j = 0; j < c.cols; ++j) s += got[(size_t)r * ld + j];
        const float e = (float)std::fabs(s - 1.0);
        if (e > worst_sum) worst_sum = e;
      }
      checkf(worst_sum <= 1e-5f, "softmax %s rows sum to 1 (worst %.3g)",
              c.what, (double)worst_sum);
    }

    // --- layer norm ---
    std::vector<float> hgamma(c.cols), hbeta(c.cols);
    for (int j = 0; j < c.cols; ++j) {
      hgamma[j] = 1.0f + 0.1f * (float)(j % 3);
      hbeta[j]  = -0.05f * (float)(j % 7);
    }
    float *dg = nullptr, *db = nullptr;
    grxMalloc((void**)&dg, (size_t)c.cols * sizeof(float));
    grxMalloc((void**)&db, (size_t)c.cols * sizeof(float));
    grxMemcpy(dg, hgamma.data(), (size_t)c.cols * sizeof(float), grxMemcpyHostToDevice);
    grxMemcpy(db, hbeta.data(),  (size_t)c.cols * sizeof(float), grxMemcpyHostToDevice);

    grxMemset(dy, 0, n * sizeof(float));
    st = grxdnnLayerNormForward(h, c.rows, c.cols, dx, ld, dg, db, 1e-5f, dy, ld);
    if (st != GRXDNN_STATUS_SUCCESS) {
      checkf(false, "layernorm %s -> %s", c.what, grxdnnGetStatusString(st));
    } else {
      grxDeviceSynchronize();
      grxMemcpy(got.data(), dy, n * sizeof(float), grxMemcpyDeviceToHost);
      ref_layernorm(hx, c.rows, c.cols, ld, &hgamma, &hbeta, 1e-5f, &want);
      int wr = 0, wc = 0;
      const float d = max_abs_diff(got, want, c.rows, c.cols, ld, &wr, &wc);
      checkf(d <= 1e-4f, "layernorm %s (worst |diff| %.3g at [%d][%d])",
              c.what, (double)d, wr, wc);
    }

    grxFree(dg); grxFree(db); grxFree(dx); grxFree(dy);
  }

  // --- gamma and beta are optional --------------------------------------
  {
    const int rows = 4, cols = 9, ld = 9;
    const size_t n = (size_t)rows * ld;
    std::vector<float> hx(n), want, got(n, -1.0f);
    for (size_t i = 0; i < n; ++i) hx[i] = 0.5f * (float)i - 3.0f;

    float *dx = nullptr, *dy = nullptr;
    grxMalloc((void**)&dx, n * sizeof(float));
    grxMalloc((void**)&dy, n * sizeof(float));
    grxMemcpy(dx, hx.data(), n * sizeof(float), grxMemcpyHostToDevice);

    const grxdnnStatus_t st =
        grxdnnLayerNormForward(h, rows, cols, dx, ld, nullptr, nullptr,
                               1e-5f, dy, ld);
    grxDeviceSynchronize();
    grxMemcpy(got.data(), dy, n * sizeof(float), grxMemcpyDeviceToHost);
    ref_layernorm(hx, rows, cols, ld, nullptr, nullptr, 1e-5f, &want);
    int wr = 0, wc = 0;
    const float d = max_abs_diff(got, want, rows, cols, ld, &wr, &wc);
    checkf(st == GRXDNN_STATUS_SUCCESS && d <= 1e-4f,
            "layernorm with no gamma or beta (worst %.3g)", (double)d);
    grxFree(dx); grxFree(dy);
  }

  // --- CONTROL 1: a row the naive softmax cannot survive ------------------
  {
    const int rows = 2, cols = 8, ld = 8;
    const size_t n = (size_t)rows * ld;
    std::vector<float> hx(n), want, got(n, -1.0f);
    // Logits an attention layer really produces. exp(120) is inf in fp32.
    for (int j = 0; j < cols; ++j) {
      hx[j]           = 100.0f + 4.0f * (float)j;   // row 0: overflows
      hx[ld + j]      = -100.0f - 4.0f * (float)j;  // row 1: underflows
    }

    // The control has to be shown to discriminate, not assumed to.
    checkf(!naive_softmax_survives(hx.data(), cols),
            "control: the naive softmax really does fail on this row");

    float *dx = nullptr, *dy = nullptr;
    grxMalloc((void**)&dx, n * sizeof(float));
    grxMalloc((void**)&dy, n * sizeof(float));
    grxMemcpy(dx, hx.data(), n * sizeof(float), grxMemcpyHostToDevice);

    const grxdnnStatus_t st =
        grxdnnSoftmaxForward(h, rows, cols, dx, ld, dy, ld);
    grxDeviceSynchronize();
    grxMemcpy(got.data(), dy, n * sizeof(float), grxMemcpyDeviceToHost);
    ref_softmax(hx, rows, cols, ld, &want);

    bool finite = true;
    for (size_t i = 0; i < n; ++i) if (!std::isfinite(got[i])) finite = false;
    int wr = 0, wc = 0;
    const float d = max_abs_diff(got, want, rows, cols, ld, &wr, &wc);
    checkf(st == GRXDNN_STATUS_SUCCESS && finite && d <= 1e-5f,
            "softmax stays finite and correct on extreme logits (worst %.3g)",
            (double)d);
    grxFree(dx); grxFree(dy);
  }

  // --- CONTROL 2: a row the one-pass variance cannot survive --------------
  {
    const int rows = 1, cols = 16, ld = 16;
    const size_t n = (size_t)rows * ld;
    std::vector<float> hx(n), want, got(n, -1.0f);
    // A large mean with a tiny spread: E[x^2] - E[x]^2 subtracts two numbers
    // that agree to seven digits, and fp32 has about seven to give.
    for (int j = 0; j < cols; ++j)
      hx[j] = 100000.0f + 0.01f * (float)(j % 4);

    checkf(!naive_variance_survives(hx.data(), cols),
            "control: the one-pass variance really does fail on this row");

    float *dx = nullptr, *dy = nullptr;
    grxMalloc((void**)&dx, n * sizeof(float));
    grxMalloc((void**)&dy, n * sizeof(float));
    grxMemcpy(dx, hx.data(), n * sizeof(float), grxMemcpyHostToDevice);

    const grxdnnStatus_t st =
        grxdnnLayerNormForward(h, rows, cols, dx, ld, nullptr, nullptr,
                               1e-5f, dy, ld);
    grxDeviceSynchronize();
    grxMemcpy(got.data(), dy, n * sizeof(float), grxMemcpyDeviceToHost);
    ref_layernorm(hx, rows, cols, ld, nullptr, nullptr, 1e-5f, &want);

    bool finite = true;
    for (size_t i = 0; i < n; ++i) if (!std::isfinite(got[i])) finite = false;
    int wr = 0, wc = 0;
    const float d = max_abs_diff(got, want, rows, cols, ld, &wr, &wc);
    checkf(st == GRXDNN_STATUS_SUCCESS && finite && d <= 2e-2f,
            "layernorm survives a large mean with a small spread (worst %.3g)",
            (double)d);
    grxFree(dx); grxFree(dy);
  }

  // --- in-place, and the argument checks ---------------------------------
  {
    const int rows = 3, cols = 5, ld = 5;
    const size_t n = (size_t)rows * ld;
    std::vector<float> hx(n), want, got(n, -1.0f);
    for (size_t i = 0; i < n; ++i) hx[i] = 0.3f * (float)i - 1.0f;

    float* dx = nullptr;
    grxMalloc((void**)&dx, n * sizeof(float));
    grxMemcpy(dx, hx.data(), n * sizeof(float), grxMemcpyHostToDevice);
    const grxdnnStatus_t st =
        grxdnnSoftmaxForward(h, rows, cols, dx, ld, dx, ld);   // y == x
    grxDeviceSynchronize();
    grxMemcpy(got.data(), dx, n * sizeof(float), grxMemcpyDeviceToHost);
    ref_softmax(hx, rows, cols, ld, &want);
    int wr = 0, wc = 0;
    const float d = max_abs_diff(got, want, rows, cols, ld, &wr, &wc);
    checkf(st == GRXDNN_STATUS_SUCCESS && d <= 1e-5f,
            "softmax in place, y == x (worst %.3g)", (double)d);

    checkf(grxdnnSoftmaxForward(h, 0, cols, dx, ld, dx, ld) ==
            GRXDNN_STATUS_INVALID_VALUE, "zero rows is refused");
    checkf(grxdnnSoftmaxForward(h, rows, 0, dx, ld, dx, ld) ==
            GRXDNN_STATUS_INVALID_VALUE, "zero cols is refused");
    checkf(grxdnnSoftmaxForward(h, rows, cols, dx, cols - 1, dx, ld) ==
            GRXDNN_STATUS_INVALID_VALUE, "ld smaller than cols is refused");
    checkf(grxdnnLayerNormForward(h, rows, cols, dx, ld, nullptr, nullptr,
                                   -1.0f, dx, ld) ==
            GRXDNN_STATUS_INVALID_VALUE, "a negative epsilon is refused");
    grxFree(dx);
  }

  grxdnnDestroy(h);
  return grxtest::report();
}
