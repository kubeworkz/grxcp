// GELU and the bias broadcast, against PyTorch — and the accuracy of the
// device's own transcendentals, MEASURED.
//
// The device build is -nostdlib, so grxDNN carries its own exp, and erf and
// tanh are built on top of it. GELU is the only op that uses them, which makes
// this gate the only place their error is visible. It therefore prints the
// worst deviation it observes rather than only passing or failing, so the
// numbers in src/libs/grxdnn/kernels/elementwise.cpp are a record of a
// measurement and not a quotation from a table.
//
// That distinction is not academic here. dev_rsqrt shipped with a comment
// claiming "about 2e-6 relative" for the 0x5f3759df estimate with one Newton
// step, which is really 1.7e-3, and it took every layer-norm case failing at
// once to catch it. A&S 7.1.26 quotes 1.5e-7 for erf's polynomial; what this
// implementation delivers is that plus whatever dev_exp contributes plus fp32
// rounding, and only a measurement knows which of those dominates.
//
// WATCHED FAILING, two ways:
//
//   bias indexed by ROW instead of column   worst |diff| 3.76, and the
//                                           row-invariance control fails too
//   the two GELU modes swapped              both form checks fail by exactly
//                                           4.73e-04 -- the spread between the
//                                           forms, so the gate is sensitive
//                                           precisely where they differ
//
// The second ablation leaves "the two forms differ" PASSING, which is correct
// and worth saying: that check catches a mode argument being ignored, not one
// being inverted. The per-form comparisons catch inversion. Neither check
// covers the other, which is why both are here.
//
// TWO FORMS, CHECKED SEPARATELY, AND CHECKED TO DIFFER. The exact and tanh
// GELUs are different functions — the reference script measures their spread at
// 4.73e-04, three orders of magnitude above this gate's tolerance — so a run
// that passed both against the same expected values would mean the mode
// argument was being ignored. The last section checks the device produces
// genuinely different answers for the two modes, which is what makes the other
// two sections evidence.

#include <grx/grx.h>
#include <grx/grxdnn.h>

#include "../unit/grx_test.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using grxtest::check;
using grxtest::section;

namespace {

constexpr uint32_t kMagic = 0x554C4547u;   // 'GELU'
constexpr uint32_t kVersion = 1u;

struct Ref {
  uint32_t rows = 0, cols = 0;
  std::vector<float> x, exact, tanh_, bias, biased;
};

bool load_ref(const char* path, Ref* r) {
  std::FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  uint32_t magic = 0, version = 0;
  auto u32 = [&](uint32_t* v) { return std::fread(v, sizeof(*v), 1, f) == 1; };
  if (!u32(&magic) || !u32(&version) || !u32(&r->rows) || !u32(&r->cols) ||
      magic != kMagic || version != kVersion || r->rows == 0 || r->cols == 0) {
    std::fclose(f);
    return false;
  }
  const size_t n = (size_t)r->rows * r->cols;
  auto vec = [&](std::vector<float>* v, size_t count) {
    v->resize(count);
    return std::fread(v->data(), sizeof(float), count, f) == count;
  };
  const bool ok = vec(&r->x, n) && vec(&r->exact, n) && vec(&r->tanh_, n) &&
                  vec(&r->bias, r->cols) && vec(&r->biased, n);
  std::fclose(f);
  return ok;
}

struct Buf {
  void* p = nullptr;
  explicit Buf(size_t bytes) { if (grxMalloc(&p, bytes) != grxSuccess) p = nullptr; }
  ~Buf() { if (p) grxFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
};

// Worst absolute deviation, and where. Absolute rather than relative because
// GELU's output passes through zero: a relative bound there divides by nothing
// and reports enormous errors for values that are correct to the last bit.
double worst_abs(const std::vector<float>& got, const std::vector<float>& want,
                 size_t* at) {
  double worst = 0.0;
  *at = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double d = std::fabs((double)got[i] - (double)want[i]);
    if (d > worst) { worst = d; *at = i; }
  }
  return worst;
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = (argc > 1) ? argv[1] : std::getenv("GRXDNN_GELU_REF");
  if (!path) path = "gelu_ref.bin";

  Ref ref;
  if (!load_ref(path, &ref)) {
    std::printf("cannot read reference vectors from %s\n"
                "  regenerate with: python3 tests/libs/gelu_ref.py "
                "--write tests/libs/gelu_ref.bin\n", path);
    return 1;
  }

  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count <= 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }
  if (grxSetDevice(0) != grxSuccess) return 1;

  grxdnnHandle_t h = nullptr;
  if (grxdnnCreate(&h) != GRXDNN_STATUS_SUCCESS) return 1;

  const int rows = (int)ref.rows, cols = (int)ref.cols;
  const size_t n = ref.x.size();
  const size_t bytes = n * sizeof(float);

  Buf dx(bytes), dy(bytes), db(ref.cols * sizeof(float));
  if (!dx.p || !dy.p || !db.p) {
    std::printf("allocation failed\n");
    grxdnnDestroy(h);
    return 1;
  }
  grxMemcpy(dx.p, ref.x.data(), bytes, grxMemcpyDefault);
  grxMemcpy(db.p, ref.bias.data(), ref.cols * sizeof(float), grxMemcpyDefault);

  // Probe before claiming anything: no toolchain means no .vxbin, and every
  // case below would fail identically.
  {
    const grxdnnStatus_t probe = grxdnnGeluForward(
        h, 1, 1, (const float*)dx.p, 1, GRXDNN_GELU_EXACT, (float*)dy.p, 1);
    if (probe == GRXDNN_STATUS_NOT_INITIALIZED ||
        probe == GRXDNN_STATUS_KERNEL_NOT_FOUND) {
      std::printf("grxdnn gelu kernel not found; skipping\n");
      grxdnnDestroy(h);
      return 77;
    }
  }

  std::vector<float> got_exact(n), got_tanh(n), got_bias(n);

  section("GELU against PyTorch, swept over [-1000, 1000]");
  // 2e-6 absolute. Chosen from the measurement below, with room: the observed
  // worst is well under it, and the bound is stated in absolute terms because
  // GELU crosses zero.
  const double tol = 2e-6;
  {
    check(grxdnnGeluForward(h, rows, cols, (const float*)dx.p, cols,
                            GRXDNN_GELU_EXACT, (float*)dy.p, cols) ==
              GRXDNN_STATUS_SUCCESS,
          "the exact (erf) form runs");
    grxMemcpy(got_exact.data(), dy.p, bytes, grxMemcpyDefault);
    size_t at = 0;
    const double w = worst_abs(got_exact, ref.exact, &at);
    std::printf("  %s  exact form: worst |diff| %.3g at x = %.6g\n",
                w <= tol ? "ok  " : "FAIL", w, (double)ref.x[at]);
    if (w > tol) ++grxtest::failures();
  }
  {
    check(grxdnnGeluForward(h, rows, cols, (const float*)dx.p, cols,
                            GRXDNN_GELU_TANH, (float*)dy.p, cols) ==
              GRXDNN_STATUS_SUCCESS,
          "the tanh form runs");
    grxMemcpy(got_tanh.data(), dy.p, bytes, grxMemcpyDefault);
    size_t at = 0;
    const double w = worst_abs(got_tanh, ref.tanh_, &at);
    std::printf("  %s  tanh form:  worst |diff| %.3g at x = %.6g\n",
                w <= tol ? "ok  " : "FAIL", w, (double)ref.x[at]);
    if (w > tol) ++grxtest::failures();
  }

  section("the mode argument is not being ignored");
  {
    // The reference script measured the two forms 4.73e-04 apart. If the device
    // produced the same numbers for both modes, everything above would still
    // pass while the mode argument did nothing.
    size_t at = 0;
    const double spread = worst_abs(got_exact, got_tanh, &at);
    std::printf("        the device's two forms differ by up to %.3g "
                "(at x = %.6g)\n", spread, (double)ref.x[at]);
    check(spread > 1e-4,
          "the two GELU forms produce measurably different answers");
  }

  section("bias broadcast");
  {
    check(grxdnnAddBiasForward(h, rows, cols, (const float*)dx.p, cols,
                               (const float*)db.p, (float*)dy.p, cols) ==
              GRXDNN_STATUS_SUCCESS,
          "the bias add runs");
    grxMemcpy(got_bias.data(), dy.p, bytes, grxMemcpyDefault);
    size_t at = 0;
    const double w = worst_abs(got_bias, ref.biased, &at);
    // A sum of two floats, so any deviation beyond rounding on the larger
    // operand is a wrong index rather than an accuracy question. The sweep
    // reaches 1000, whose ulp is about 6e-5.
    const double btol = 1e-4;
    std::printf("  %s  x + bias: worst |diff| %.3g\n",
                w <= btol ? "ok  " : "FAIL", w);
    if (w > btol) ++grxtest::failures();

    // The bias must vary along the COLUMN and be constant down rows. Adding a
    // bias indexed by row instead would still produce plausible numbers, so it
    // is checked directly: two different rows must have received the same
    // bias vector.
    bool same = true;
    for (int j = 0; j < cols && same; ++j) {
      const double d0 = (double)got_bias[j] - (double)ref.x[j];
      const double d1 = (double)got_bias[(size_t)(rows - 1) * cols + j] -
                        (double)ref.x[(size_t)(rows - 1) * cols + j];
      if (std::fabs(d0 - d1) > 1e-4) same = false;
    }
    check(same, "the same bias vector reached the first and last rows");
  }

  section("what it refuses");
  {
    float* f = (float*)dx.p;
    check(grxdnnGeluForward(h, rows, cols, f, cols, (grxdnnGeluMode_t)7,
                            (float*)dy.p, cols) == GRXDNN_STATUS_INVALID_VALUE,
          "an unknown GELU mode is refused rather than defaulted");
    check(grxdnnGeluForward(h, 0, cols, f, cols, GRXDNN_GELU_EXACT,
                            (float*)dy.p, cols) == GRXDNN_STATUS_INVALID_VALUE,
          "zero rows are refused");
    check(grxdnnAddBiasForward(h, rows, cols, f, cols - 1, (const float*)db.p,
                               (float*)dy.p, cols) ==
              GRXDNN_STATUS_INVALID_VALUE,
          "a leading dimension smaller than the row is refused");
    check(grxdnnAddBiasForward(h, rows, cols, f, cols, nullptr,
                               (float*)dy.p, cols) ==
              GRXDNN_STATUS_INVALID_VALUE,
          "a null bias is refused");
  }

  grxdnnDestroy(h);
  return grxtest::report();
}
