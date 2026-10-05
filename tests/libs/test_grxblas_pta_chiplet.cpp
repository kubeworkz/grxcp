// grxblasGemmEx on the PTA chiplet, bitwise against the model.
//
// The board plan's X5 gate reads "grxcp's backend gates pass against it,
// bitwise against the model", and until the chiplet was a device the runtime
// could see, only the second half of that could be met. This is the first
// half: the same call a program makes -- grxSetDevice, grxblasGemmEx -- on the
// chiplet's digital twin, and every element of every result held to the error
// model.
//
// THE REFERENCE IS BUILT FROM THE DEVICE PROPERTY AND NOTHING ELSE. The
// configuration, the tile, the run's seed and the GEMM's index are all read
// out of grxDeviceProp_t.analogGemm before the GEMM they describe, and the
// model is run on those. The last of them is what the chiplet adds: each of its
// GEMMs runs on a seed derived from PTA_SEED and the GEMM's index
// (docs/designs/pta_chiplet_regmap.md section 4), so a property that carried
// the seed alone would reproduce nothing here. This file is what makes
// gemmIndex a tested claim.
//
// WHOSE POINTERS. The chiplet has no memory. A, B and C are allocated on its
// parent GPU -- grxDeviceProp_t.parentDevice -- and the GEMM is issued with
// the chiplet as the current device. That is the one place a pointer is
// resolved against a device other than the current one, and the refusals at
// the end of this file are its edges.
//
// WHICH OPERAND IS THE WEIGHT. As on the c930: A is programmed into the tile
// and each column of B is a shot. One library, two tiles, one answer, and the
// "weights" case below is what would notice the other one.
//
// ONE RUN IS ONE TILE, because enumeration happens once in a process. The tile
// is the command line's (tests/common/pta_twin_adapter.h): the chiplet's
// working geometry, 128 x 64, with no argument, and the 256 x 64 it was first
// settled at with "256x64". ci/build_mock.sh runs both, and the cases below are
// derived for the tile the run was given.
//
// NOTHING HERE IS HARDWARE. The device is the twin, it says it is a model, and
// a pass is a statement about the error model's arithmetic and the runtime's
// path to it.

#include <grx/grx.h>
#include <grx/grxblas.h>

#include "grx_test.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef GRXCP_ENABLE_PTA
#include "pta_twin_adapter.h"
#endif

using grxtest::check;
using grxtest::section;

#ifdef GRXCP_ENABLE_PTA
namespace {

#define PTA_BITS_OF(a, w, adc, s) \
  ((uint32_t)(a) | ((uint32_t)(w) << 4) | ((uint32_t)(adc) << 8) | ((uint32_t)(s) << 12))

struct Case {
  const char* name;
  const char* why;
  int         m, n, k;      // the CALLER's shape, column-major: C(m x n) = A(m x k) . B(k x n)
  uint32_t    op_seed;
  int         gemms;
  uint32_t    impair, bits, seed;
  uint32_t    sigma_th, sigma_sh, sigma_pr, drift, xtalk, drift_max;
};

// The tile's word is eight bits, so a setting of 6 is a six-bit quantiser. The
// noise figures are in LSB of the ADC each case configures: 64 is a quarter of
// the 7-bit ADC's LSB in "v1", and 256 is one LSB of the 6-bit ADC's in "v0".
//
// TWO THINGS FOLLOW THE TILE'S ROWS, and are derived here from the build the
// run was given rather than written down for one tile.
//
//   The ADC shift of the four cases that fill a K tile. Its LSB puts a K tile
//   of int8 products inside the ADC (pta_twin_adc_shift): 2^16 for the 7-bit
//   ADC and 2^17 for the 6-bit one on 256 rows, 2^15 and 2^16 on 128.
//   The same four cases' K: one K tile and 44 rows of the next (4, in "drift"),
//   which is two K tiles on either build. 300 and 260 on 256 rows, 172 and 132
//   on 128. Their 100 or 70 outputs are two tiles of the 64 columns already.
//
// The other four cases are smaller than either tile, and their shifts are sized
// to their own K.
std::vector<Case> cases_for(const pta_twin_build& b) {
  const int k_two_tiles = b.rows + 44, k_drift = b.rows + 4;
  const uint32_t s7 = grxtest::pta_twin_adc_shift(b, 7), s6 = grxtest::pta_twin_adc_shift(b, 6);
  return {
  {"v1", "the board plan's version 1 allowances, every impairment but drift, "
         "on a shape of two tiles each way",
   100, 6, k_two_tiles, 1, 1,
   0x57, PTA_BITS_OF(6, 6, 7, s7), 0x2a, 64, 47, 256, 0, 5, 0},
  {"v0", "version 0's: a coarser DAC and ADC and four times the noise",
   100, 6, k_two_tiles, 2, 1,
   0x57, PTA_BITS_OF(5, 6, 6, s6), 0x2b, 256, 148, 1024, 0, 26, 0},
  {"quant", "quantisation alone, on a shape smaller than the tile every way",
   5, 3, 7, 3, 1,
   0x01, PTA_BITS_OF(5, 6, 6, 10), 0x2c, 0, 0, 0, 0, 0, 0},
  {"weights", "only the weight side is impaired: four-bit weights and "
              "programming error. Which operand that lands on is what the "
              "roles check below turns on",
   4, 4, 8, 4, 1,
   0x41, PTA_BITS_OF(0, 4, 0, 0), 0x2d, 0, 0, 512, 0, 0, 0},
  {"xtalk-short-k", "crosstalk with K far shorter than the tile",
   6, 2, 3, 5, 1,
   0x10, 0, 0x2e, 0, 0, 0, 0, 64, 0},
  {"unit", "one element, one shot, one weight",
   1, 1, 1, 6, 1,
   0x03, PTA_BITS_OF(0, 0, 7, 4), 0x2f, 512, 0, 0, 0, 0, 0},
  {"drift", "drift is device state: the second GEMM is the one a tile that "
            "forgot its history gets wrong",
   70, 8, k_drift, 7, 2,
   0x49, PTA_BITS_OF(6, 6, 0, 0), 0x30, 0, 0, 256, 256, 0, 0x0C00},
  {"v1-drift", "everything at once, three times",
   100, 6, k_two_tiles, 8, 3,
   0x5F, PTA_BITS_OF(6, 6, 7, s7), 0x31, 64, 47, 256, 128 | (1u << 16), 5, 0x0C00},
  };
}

// Full-range int8 from the model's own xorshift, so a reader needs no data.
void operands(const Case& c, std::vector<int8_t>* A, std::vector<int8_t>* B) {
  uint32_t s = c.op_seed ^ 0x2545f491u;
  A->resize((size_t)c.m * c.k);
  B->resize((size_t)c.k * c.n);
  for (int8_t& v : *A) { s = pta_xorshift32(s); v = (int8_t)(s & 0xFFu); }
  for (int8_t& v : *B) { s = pta_xorshift32(s); v = (int8_t)(s & 0xFFu); }
}

// The exact product, in the caller's column-major terms.
std::vector<int32_t> exact_gemm(const Case& c, const std::vector<int8_t>& A,
                                const std::vector<int8_t>& B) {
  std::vector<int32_t> C((size_t)c.m * c.n);
  for (int j = 0; j < c.n; ++j)
    for (int i = 0; i < c.m; ++i) {
      int32_t s = 0;
      for (int p = 0; p < c.k; ++p)
        s += (int32_t)A[i + (size_t)p * c.m] * (int32_t)B[p + (size_t)j * c.k];
      C[i + (size_t)j * c.m] = s;
    }
  return C;
}

// The model's answer, in the caller's terms. weights_are_A is the mapping as
// built; false is the reading a reader might assume from "C = A . B".
bool reference_gemm(const pta_cfg& cfg, const pta_tile& tile, pta_device* dev,
                    const Case& c, const std::vector<int8_t>& A,
                    const std::vector<int8_t>& B, bool weights_are_A,
                    std::vector<int32_t>* C) {
  const int m = c.m, n = c.n, k = c.k;
  C->assign((size_t)m * n, 0);
  if (weights_are_A) {
    // The tile computes C^T (n x m) = B^T (n x k) . A^T (k x m), row-major.
    std::vector<int32_t> act((size_t)n * k), wgt((size_t)k * m);
    std::vector<int64_t> out((size_t)n * m);
    for (int j = 0; j < n; ++j)
      for (int p = 0; p < k; ++p)
        act[(size_t)j * k + p] = B[p + (size_t)j * k];          // shot j = column j of B
    for (int p = 0; p < k; ++p)
      for (int i = 0; i < m; ++i)
        wgt[(size_t)p * m + i] = A[i + (size_t)p * m];          // weight (p, i) = A(i, p)
    if (pta_gemm(&cfg, &tile, dev, 0, n, m, k, act.data(), wgt.data(), out.data()) < 0)
      return false;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i)
        (*C)[i + (size_t)j * m] = (int32_t)(uint32_t)(uint64_t)out[(size_t)j * m + i];
  } else {
    std::vector<int32_t> act((size_t)m * k), wgt((size_t)k * n);
    std::vector<int64_t> out((size_t)m * n);
    for (int i = 0; i < m; ++i)
      for (int p = 0; p < k; ++p)
        act[(size_t)i * k + p] = A[i + (size_t)p * m];
    for (int p = 0; p < k; ++p)
      for (int j = 0; j < n; ++j)
        wgt[(size_t)p * n + j] = B[p + (size_t)j * k];
    if (pta_gemm(&cfg, &tile, dev, 0, m, n, k, act.data(), wgt.data(), out.data()) < 0)
      return false;
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        (*C)[i + (size_t)j * m] = (int32_t)(uint32_t)(uint64_t)out[(size_t)i * n + j];
  }
  return true;
}

// The model's configuration for ONE GEMM, from what the DEVICE says about
// itself and nothing else. The seed is the chiplet's own: derived from the
// run's seed and the index the property reported before this GEMM.
pta_cfg cfg_of_property(const grxAnalogGemm_t& p) {
  pta_cfg cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.impair      = (uint32_t)p.impairments;
  cfg.act_bits    = (uint32_t)p.activationBits;
  cfg.w_bits      = (uint32_t)p.weightBits;
  cfg.adc_bits    = (uint32_t)p.adcBits;
  cfg.adc_shift   = (uint32_t)p.adcShift;
  cfg.seed        = pta_chiplet_gemm_seed((uint32_t)p.seed, (uint32_t)p.gemmIndex);
  cfg.sigma_th    = (uint32_t)p.thermalSigmaQ8;
  cfg.k_shot      = (uint32_t)p.shotCoefficientQ8;
  cfg.sigma_pr    = (uint32_t)p.programmingSigmaQ8;
  cfg.drift_sigma = (uint32_t)p.driftSigmaQ8;
  cfg.drift_log2  = (uint32_t)p.driftLog2Shots;
  cfg.drift_max   = (uint32_t)p.driftClampQ8;
  cfg.xtalk       = (uint32_t)p.crosstalkQ8;
  return cfg;
}

bool property_is_the_case(const grxAnalogGemm_t& p, const Case& c, int gemm) {
  return p.gemmIsAnalogEmulated == 1 && p.tileIsPresent == 1 &&
         p.impairments == (int64_t)c.impair &&
         p.activationBits == (int)(c.bits & 0xF) &&
         p.weightBits == (int)((c.bits >> 4) & 0xF) &&
         p.adcBits == (int)((c.bits >> 8) & 0xF) &&
         p.adcShift == (int)((c.bits >> 12) & 0x3F) &&
         p.seed == (int64_t)c.seed && p.gemmIndex == (int64_t)gemm &&
         p.thermalSigmaQ8 == (int)c.sigma_th &&
         p.shotCoefficientQ8 == (int)c.sigma_sh &&
         p.programmingSigmaQ8 == (int)c.sigma_pr &&
         p.driftSigmaQ8 == (int)(c.drift & 0xFFFF) &&
         p.driftLog2Shots == (int)((c.drift >> 16) & 0x1F) &&
         p.driftClampQ8 == (int)c.drift_max && p.crosstalkQ8 == (int)c.xtalk &&
         p.loopModes == 0 && p.calibrationValid == 0;
}

// Program the chiplet for a case, the way another holder of it would: through
// its registers, behind the runtime's back, because no grx call configures a
// tile (cuda_mapping.md 7.40). The seed once, which restarts the GEMM count;
// then a model reset, which reloads the drift generator from it.
void program(pta_twin* t, const Case& c) {
  pta_twin_write32(t, PTA_TWIN_IMPAIR, c.impair);
  pta_twin_write32(t, PTA_TWIN_BITS, c.bits);
  pta_twin_write32(t, PTA_TWIN_SIGMA_TH, c.sigma_th);
  pta_twin_write32(t, PTA_TWIN_SIGMA_SH, c.sigma_sh);
  pta_twin_write32(t, PTA_TWIN_SIGMA_PR, c.sigma_pr);
  pta_twin_write32(t, PTA_TWIN_DRIFT, c.drift);
  pta_twin_write32(t, PTA_TWIN_DRIFT_MAX, c.drift_max);
  pta_twin_write32(t, PTA_TWIN_XTALK, c.xtalk);
  pta_twin_write32(t, PTA_TWIN_SEED, c.seed);
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_MODEL_RST);
}

// The devices, and the three buffers of one GEMM on the PARENT.
struct Rig {
  int pta = -1, parent = -1;
  grxblasHandle_t h = nullptr;
};

const int32_t kSentinel = (int32_t)0x5A5A5A5A;

// One GEMM through the front door. Operands are allocated and filled on the
// parent, the GEMM is issued with the chiplet current, and C is read back from
// the parent. *C is whatever the parent's buffer holds afterwards.
grxblasStatus_t device_gemm(const Rig& rig, const Case& c, const std::vector<int8_t>& A,
                            const std::vector<int8_t>& B, std::vector<int32_t>* C,
                            const float* alpha = nullptr,
                            grxblasOperation_t transa = GRXBLAS_OP_N) {
  const size_t cn = (size_t)c.m * c.n;
  void *dA = nullptr, *dB = nullptr, *dC = nullptr;
  grxSetDevice(rig.parent);
  if (grxMalloc(&dA, A.size()) != grxSuccess || grxMalloc(&dB, B.size()) != grxSuccess ||
      grxMalloc(&dC, cn * sizeof(int32_t)) != grxSuccess)
    return GRXBLAS_STATUS_ALLOC_FAILED;
  // A sentinel no result here can equal by accident, so a GEMM that reported
  // success and wrote nothing is caught rather than read back as an answer.
  std::vector<int32_t> sentinel(cn, kSentinel);
  grxMemcpy(dA, A.data(), A.size(), grxMemcpyDefault);
  grxMemcpy(dB, B.data(), B.size(), grxMemcpyDefault);
  grxMemcpy(dC, sentinel.data(), cn * sizeof(int32_t), grxMemcpyDefault);

  grxSetDevice(rig.pta);
  const float one = 1.0f, zero = 0.0f;
  const grxblasStatus_t s = grxblasGemmEx(
      rig.h, transa, GRXBLAS_OP_N, c.m, c.n, c.k, alpha ? alpha : &one,
      dA, GRX_R_8I, c.m, dB, GRX_R_8I, c.k, &zero, dC, GRX_R_32I, c.m);
  grxDeviceSynchronize();

  grxSetDevice(rig.parent);
  C->assign(cn, 0);
  grxMemcpy(C->data(), dC, cn * sizeof(int32_t), grxMemcpyDefault);
  grxFree(dA); grxFree(dB); grxFree(dC);
  grxSetDevice(rig.pta);
  return s;
}

int mismatches(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  if (a.size() != b.size()) return (int)(a.size() > b.size() ? a.size() : b.size());
  int n = 0;
  for (size_t i = 0; i < a.size(); ++i) n += (a[i] != b[i]);
  return n;
}

bool untouched(const std::vector<int32_t>& c) {
  for (int32_t v : c) if (v != kSentinel) return false;
  return true;
}

grxAnalogGemm_t read_property(int device) {
  grxDeviceProp_t p{};
  grxGetDeviceProperties(&p, device);
  return p.analogGemm;
}

}  // namespace
#endif  // GRXCP_ENABLE_PTA

int main(int argc, char** argv) {
#ifndef GRXCP_ENABLE_PTA
  (void)argc; (void)argv;
  std::printf("built without GRXCP_ENABLE_PTA; no PTA chiplet backend here. skipping\n");
  return 77;
#else
  pta_twin_build build;
  if (!grxtest::pta_twin_build_from_args(argc, argv, &build)) return 2;
  const std::vector<Case> cases = cases_for(build);
  const int num_cases = (int)cases.size();

  // BEFORE THE FIRST grx CALL. Enumeration runs once and the seam refuses after.
  const bool installed = grxtest::pta_twin_install(build);
  section("the twin is behind the runtime's PTA device");
  check(installed, "the twin's window and link are installed");
  if (!installed) return grxtest::report();
  pta_twin* twin = grxtest::pta_twin_installed();

  int count = 0;
  GRX_REQUIRE(grxGetDeviceCount(&count), "grxGetDeviceCount");
  Rig rig;
  for (int i = 0; i < count; ++i) {
    grxDeviceProp_t p{};
    if (grxGetDeviceProperties(&p, i) == grxSuccess && p.deviceType == GRX_DEVICE_TYPE_PTA) {
      rig.pta = i;
      rig.parent = p.parentDevice;
      std::printf("  note  %s\n", p.name);
      check(p.backend == GRX_BACKEND_MODEL, "it says it is a model, not hardware");
      break;
    }
  }
  check(rig.pta >= 0 && rig.parent >= 0, "a PTA chiplet is enumerated, with a parent");
  if (rig.pta < 0 || rig.parent < 0) return grxtest::report();
  const pta_tile tile = {build.rows, build.cols, build.din_w, build.acc_w};
  {
    const grxAnalogGemm_t a = read_property(rig.pta);
    check(a.tileRows == tile.rows && a.tileCols == tile.cols && a.operandBits == tile.din_w &&
          a.accumulatorBits == tile.acc_w, "its tile is the one this run's cases are derived for");
    std::printf("  note  tile %dx%d, as the device reports it\n", a.tileRows, a.tileCols);
  }
  GRX_REQUIRE(grxSetDevice(rig.pta), "grxSetDevice");
  if (grxblasCreate(&rig.h) != GRXBLAS_STATUS_SUCCESS) {
    check(false, "grxblasCreate");
    return grxtest::report();
  }
  {
    grxblasEngine_t engine = GRXBLAS_ENGINE_NONE;
    int device = -1;
    grxblasGetGemmEngine(rig.h, 4, 4, 4, GRX_R_8I, GRX_R_8I, GRX_R_32I, &engine, &device);
    std::printf("  note  engine: %s\n", grxblasGetEngineString(engine));
    check(engine == GRXBLAS_ENGINE_PTA_CHIPLET && device == rig.pta,
          "grxBLAS routes an int8 GEMM here to the PTA chiplet");
  }

  section("unimpaired, the tile is the product");
  {
    std::vector<int8_t> A, B;
    std::vector<int32_t> C;
    const Case& c = cases[0];
    operands(c, &A, &B);
    // What the tile itself counts, read behind the runtime as its registers
    // are written: a K that was another build's would be a different walk.
    const uint32_t loads = pta_twin_read32(twin, PTA_TWIN_WLOAD_CT);
    const uint32_t shots = pta_twin_read32(twin, PTA_TWIN_SHOT_CT);
    const grxblasStatus_t s = device_gemm(rig, c, A, B, &C);
    char what[128];
    std::snprintf(what, sizeof(what), "%d x %d x %d int8: every element exact", c.m, c.n, c.k);
    check(s == GRXBLAS_STATUS_SUCCESS && mismatches(C, exact_gemm(c, A, B)) == 0, what);
    check(pta_twin_read32(twin, PTA_TWIN_WLOAD_CT) - loads == 4u &&
          pta_twin_read32(twin, PTA_TWIN_SHOT_CT) - shots == 4u * (uint32_t)c.n,
          "on two tiles each way: the tile counts four programmings, and a shot a column of B on each");
    check(read_property(rig.pta).gemmIndex == 1, "and it took an index: the property reads 1");
  }

  // ---- the gate -----------------------------------------------------------
  section("bitwise: the device, and the model from the property alone");
  int total_results = 0;
  for (int ci = 0; ci < num_cases; ++ci) {
    const Case& c = cases[ci];
    std::vector<int8_t> A, B;
    operands(c, &A, &B);
    const std::vector<int32_t> exact = exact_gemm(c, A, B);

    program(twin, c);

    // The reference device: reset to the seed the property reports, as the
    // chiplet was by the MODEL_RST above.
    pta_device ref;
    bool ref_ok = pta_device_init(&ref, &tile) == 0;
    bool prop_ok = true;
    int vs_model = 0, moved = 0, carried = 0, failed = 0;
    std::vector<int32_t> first;
    for (int g = 0; g < c.gemms; ++g) {
      // Read BEFORE the GEMM it describes: the index is the one this GEMM takes.
      const grxAnalogGemm_t p = read_property(rig.pta);
      prop_ok = prop_ok && property_is_the_case(p, c, g);
      if (g == 0 && ref_ok) pta_model_reset(&ref, (uint32_t)p.seed);
      const pta_tile ptile = {p.tileRows, p.tileCols, p.operandBits, p.accumulatorBits};
      std::vector<int32_t> want, C;
      ref_ok = ref_ok && prop_ok &&
               reference_gemm(cfg_of_property(p), ptile, &ref, c, A, B, true, &want);
      if (device_gemm(rig, c, A, B, &C) != GRXBLAS_STATUS_SUCCESS) ++failed;
      if (ref_ok) vs_model += mismatches(C, want);
      moved += mismatches(C, exact);
      if (g == 0) first = C; else carried += mismatches(C, first);
      total_results += (int)C.size();
    }
    pta_device_free(&ref);

    char what[256];
    std::snprintf(what, sizeof(what),
                  "%-14s the property reports what was programmed, and each GEMM's index",
                  c.name);
    check(prop_ok, what);
    std::snprintf(what, sizeof(what),
                  "%-14s %d x %d x %d%s: the device's C is the model's from the "
                  "property alone, bit for bit", c.name, c.m, c.n, c.k,
                  c.gemms == 2 ? ", twice" : c.gemms == 3 ? ", three times" : "");
    check(failed == 0 && ref_ok && vs_model == 0, what);
    if (failed) std::printf("        %d GEMM(s) did not report success\n", failed);
    if (vs_model) std::printf("        %d element(s) differ\n", vs_model);
    std::snprintf(what, sizeof(what),
                  "%-14s which is not the exact product (%d of %d moved)", c.name,
                  moved, c.m * c.n * c.gemms);
    check(moved > 0, what);
    if (c.gemms > 1) {
      std::snprintf(what, sizeof(what),
                    "%-14s a later GEMM is not the first: its own seed, and the "
                    "drift carried (%d moved)", c.name, carried);
      check(carried > 0, what);
    }
  }
  std::printf("  note  %d results compared\n", total_results);

  // ---- the index is not decoration ---------------------------------------
  // The same case, reproduced as a c930's property would have it: on PTA_SEED
  // itself. If that gave the device's answer, gemmIndex would be carrying
  // nothing.
  section("the seed alone does not reproduce a chiplet's GEMM");
  {
    const Case& c = cases[0];
    std::vector<int8_t> A, B;
    std::vector<int32_t> C, on_seed;
    operands(c, &A, &B);
    program(twin, c);
    const grxAnalogGemm_t p = read_property(rig.pta);
    pta_cfg raw = cfg_of_property(p);
    raw.seed = (uint32_t)p.seed;
    pta_device ref;
    pta_device_init(&ref, &tile);
    pta_model_reset(&ref, (uint32_t)p.seed);
    const bool ran = reference_gemm(raw, tile, &ref, c, A, B, true, &on_seed);
    pta_device_free(&ref);
    const grxblasStatus_t s = device_gemm(rig, c, A, B, &C);
    const int diff = mismatches(C, on_seed);
    std::printf("  note  %d of %d elements differ from the model run on PTA_SEED itself\n",
                diff, c.m * c.n);
    check(ran && s == GRXBLAS_STATUS_SUCCESS && diff > 0,
          "a reference built without the index is a different answer");
  }

  // ---- whose operand is the weight ----------------------------------------
  section("which operand is the weight");
  {
    const Case& c = cases[3];
    std::vector<int8_t> A, B;
    std::vector<int32_t> C, as_built, other;
    operands(c, &A, &B);
    program(twin, c);
    const grxAnalogGemm_t p = read_property(rig.pta);
    pta_device r1, r2;
    pta_device_init(&r1, &tile);
    pta_device_init(&r2, &tile);
    pta_model_reset(&r1, (uint32_t)p.seed);
    pta_model_reset(&r2, (uint32_t)p.seed);
    const bool ok1 = reference_gemm(cfg_of_property(p), tile, &r1, c, A, B, true, &as_built);
    const bool ok2 = reference_gemm(cfg_of_property(p), tile, &r2, c, A, B, false, &other);
    pta_device_free(&r1);
    pta_device_free(&r2);
    const grxblasStatus_t s = device_gemm(rig, c, A, B, &C);
    check(ok1 && ok2 && s == GRXBLAS_STATUS_SUCCESS && mismatches(C, as_built) == 0,
          "A as the weight set and B's columns as the shots is what the device does");
    const int diff = mismatches(as_built, other);
    std::printf("  note  %d of %d elements differ under the other reading\n", diff, c.m * c.n);
    check(diff > 0, "the other reading is a different answer, so this is pinned");
  }

  // ---- what it refuses ------------------------------------------------------
  section("what it refuses, with C untouched");
  {
    const Case& c = cases[2];
    std::vector<int8_t> A, B;
    std::vector<int32_t> C;
    operands(c, &A, &B);
    program(twin, c);

    // An impairment the tile does not build.
    pta_twin_write32(twin, PTA_TWIN_IMPAIR, PTA_QUANT | PTA_MZM_NL);
    grxblasStatus_t s = device_gemm(rig, c, A, B, &C);
    check(s == GRXBLAS_STATUS_EXECUTION_FAILED && untouched(C),
          "PTA_IMPAIR asking for MZM_NL: the GEMM fails, and is not the exact product instead");
    check(read_property(rig.pta).gemmIndex == 0, "a refused GEMM takes no index");
    pta_twin_write32(twin, PTA_TWIN_IMPAIR, c.impair);

    const float two = 2.0f;
    s = device_gemm(rig, c, A, B, &C, &two);
    check(s == GRXBLAS_STATUS_NOT_SUPPORTED && untouched(C), "alpha = 2: the tile computes A . B and nothing else");
    s = device_gemm(rig, c, A, B, &C, nullptr, GRXBLAS_OP_T);
    check(s != GRXBLAS_STATUS_SUCCESS && untouched(C), "a transposed A: no untransposed answer in its place");

    // Pointers that are not the parent's.
    std::vector<int32_t> hostC((size_t)c.m * c.n, kSentinel);
    const float one = 1.0f, zero = 0.0f;
    s = grxblasGemmEx(rig.h, GRXBLAS_OP_N, GRXBLAS_OP_N, c.m, c.n, c.k, &one,
                      A.data(), GRX_R_8I, c.m, B.data(), GRX_R_8I, c.k, &zero,
                      hostC.data(), GRX_R_32I, c.m);
    check(s == GRXBLAS_STATUS_INVALID_VALUE && untouched(hostC),
          "host pointers: refused, the operands have to be on the parent");
    check(read_property(rig.pta).gemmIndex == 0, "and none of those started a GEMM");

    // And with the register put right, the same call runs.
    s = device_gemm(rig, c, A, B, &C);
    check(s == GRXBLAS_STATUS_SUCCESS && !untouched(C) && read_property(rig.pta).gemmIndex == 1,
          "with PTA_IMPAIR put right the same call runs, as GEMM 0");
  }

  // ---- and back -----------------------------------------------------------
  section("PTA_IMPAIR cleared");
  pta_twin_write32(twin, PTA_TWIN_IMPAIR, 0);
  {
    const grxAnalogGemm_t a = read_property(rig.pta);
    check(a.gemmIsAnalogEmulated == 0 && a.thermalSigmaQ8 == -1 && a.gemmIndex >= 0,
          "the property says native again, withdraws the model's numbers, and keeps the index");
    std::vector<int8_t> A, B;
    std::vector<int32_t> C;
    operands(cases[1], &A, &B);
    const grxblasStatus_t s = device_gemm(rig, cases[1], A, B, &C);
    check(s == GRXBLAS_STATUS_SUCCESS && mismatches(C, exact_gemm(cases[1], A, B)) == 0,
          "and a GEMM is the exact product again");
  }

  check(grxblasDestroy(rig.h) == GRXBLAS_STATUS_SUCCESS, "grxblasDestroy");
  return grxtest::report();
#endif
}
