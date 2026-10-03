// grxblasGemmEx on a photonic tile, bitwise against the model.
//
// AGENTS.md section 4 says every library kernel has a CPU reference and a
// numerical gate that is bitwise or ULP-bounded, and that "close enough" is not
// a gate. An analog device breaks that as written: a GEMM on the tile is a
// noisy approximation of the product, and the temptation is to loosen the rule
// into a tolerance, which destroys it. docs/designs/pta_cpu_integration.md
// section 7 keeps the rule by splitting it in two. This file is the first half.
//
// THE ERROR MODEL IS DETERMINISTIC. Given its configuration, the tile's
// geometry and the operands, pta_gemm() has exactly one answer. So a GEMM on
// the tile has an exact expected result after all -- not the product, but the
// model's -- and it can be held to that bit for bit. That is the gate that
// catches bugs. (How far the model's answer is from the product is the second
// half, the distributional report, and is deliberately not this file.)
//
// THREE THINGS ARE COMPARED, AND THEY ARE THREE BECAUSE EACH PINS SOMETHING
// THE OTHER TWO DO NOT.
//
//   1. The device's C, through grxblasGemmEx, the runtime, the driver and the
//      register map.
//   2. The model's C, recomputed here from grxDeviceProp_t.analogGemm AND
//      NOTHING ELSE. If the property is missing something the answer depends
//      on, this is where it shows -- and it did: the property as first built
//      carried a seed and a mask and no sigmas, which reproduces nothing.
//   3. tests/libs/pta_gemm_golden.txt, the committed numbers. 1 and 2 are both
//      computed by today's vendored model, so they would move together if it
//      changed. The golden file is what makes a model change VISIBLE: it goes
//      red, and stays red until the file is regenerated as a reviewed step and
//      the diff is in the same change as the model. ci/check_perf.py treats
//      its baselines exactly that way, and for the same reason.
//
// WHOSE OPERAND IS THE WEIGHT. grxBLAS is column-major and the c930 is
// row-major, so the NPU path hands the engine B where it expects A and A where
// it expects B (grxblas.cpp, npu_gemm_path). On a digital array that is
// invisible: the product is the product. On a tile it decides which matrix is
// PROGRAMMED and which is SHONE THROUGH, and the error model treats the two
// quite differently. As built:
//
//     C(m x n) = A(m x k) . B(k x n)
//       A      is the weight set: k x m cells, quantised to weightBits,
//              subject to programming error and drift
//       B      is the activations: each COLUMN of B is one optical shot
//
// which is the natural reading of Y = W . X with a batch of column vectors.
// reference_gemm() below says so in code, and one case checks that the other
// reading gives a different answer, so the statement is not decoration.
//
// A MODEL IS NOT HARDWARE. The device here is the GRX930 team's register model
// with their reference error model behind it. Green says our host reaches the
// tile correctly and reports it truthfully. It says nothing about a c930.
//
//   test_grxblas_pta                       run the gate (from tests/libs)
//   test_grxblas_pta --regenerate [FILE]   rewrite the golden data from the
//                                          vendored model; review the diff

#include <grx/grx.h>
#include <grx/grxblas.h>

#include "grx_test.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef GRXCP_ENABLE_NPU
#include "npu_tile_adapter.h"
#endif

using grxtest::check;
using grxtest::section;

#ifdef GRXCP_ENABLE_NPU
namespace {

const char* const kGoldenDefault = "pta_gemm_golden.txt";
const int         kGoldenVersion = 1;

// The tile the golden data is generated for: the register model's own. Written
// out here, and compared with what the device reports, so data generated for
// one tile cannot be silently checked against another.
const pta_tile kTile = {4, 4, 16, 48};

#define PTA_BITS_OF(a, w, adc, s) \
  ((uint32_t)(a) | ((uint32_t)(w) << 4) | ((uint32_t)(adc) << 8) | ((uint32_t)(s) << 12))

struct Case {
  const char* name;
  const char* why;
  int         m, n, k;      // the CALLER's shape, column-major: C(m x n) = A(m x k) . B(k x n)
  uint32_t    op_seed;
  int         gemms;        // 2 where drift has to carry from one to the next
  uint32_t    impair, bits, seed;
  uint32_t    sigma_th, sigma_sh, sigma_pr, drift, xtalk, drift_max;
};

// The operands are int8 and the tile's word is sixteen bits, so the quantiser
// settings are counts of sixteen: 13 and 14 here are five and six bits of an
// int8 operand, the board plan's version 0 and version 1 DACs. An ADC LSB of
// 2^9 or 2^10 puts a K tile of four int8 products inside a 7- or 6-bit ADC.
const Case kCases[] = {
  {"v1", "the board plan's version 1 allowances, every impairment but drift, "
         "at the largest shape the engine takes",
   12, 8, 16, 1, 1,
   0x57, PTA_BITS_OF(14, 14, 7, 9), 0x2a, 64, 47, 256, 0, 5, 0},
  {"v0", "version 0's: a coarser DAC and ADC and four times the noise",
   12, 8, 16, 2, 1,
   0x57, PTA_BITS_OF(13, 14, 6, 10), 0x2b, 256, 148, 1024, 0, 26, 0},
  {"quant", "quantisation alone, on a shape that leaves a ragged tile in "
            "every direction",
   5, 3, 7, 3, 1,
   0x01, PTA_BITS_OF(13, 14, 6, 10), 0x2c, 0, 0, 0, 0, 0, 0},
  {"weights", "only the weight side is impaired: four-bit weights and "
              "programming error, activations untouched. Which operand that "
              "lands on is what the roles check below turns on",
   4, 4, 8, 4, 1,
   0x41, PTA_BITS_OF(0, 12, 0, 0), 0x2d, 0, 0, 512, 0, 0, 0},
  {"xtalk-short-k", "crosstalk with K shorter than the tile, where a row "
                    "holds no weight to couple from",
   6, 2, 3, 5, 1,
   0x10, 0, 0x2e, 0, 0, 0, 0, 64, 0},
  {"unit", "one element, one shot, one weight",
   1, 1, 1, 6, 1,
   0x03, PTA_BITS_OF(0, 0, 7, 4), 0x2f, 512, 0, 0, 0, 0, 0},
  {"drift", "drift is device state: the second GEMM is the one a tile that "
            "forgot its history gets wrong",
   8, 8, 16, 7, 2,
   0x49, PTA_BITS_OF(14, 14, 0, 0), 0x30, 0, 0, 256, 256, 0, 0x0C00},
  {"v1-drift", "everything at once, twice",
   12, 8, 16, 8, 2,
   0x5F, PTA_BITS_OF(14, 14, 7, 9), 0x31, 64, 47, 256, 128 | (1u << 16), 5, 0x0C00},
};
const int kNumCases = (int)(sizeof(kCases) / sizeof(kCases[0]));

// ---------------------------------------------------------------------------
// Operands, and the reference
// ---------------------------------------------------------------------------

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

// THE MODEL'S ANSWER, in the caller's terms.
//
// weights_are_A is the mapping as built: A is programmed into the tile and each
// column of B is a shot. With it false the roles are the other way round -- A's
// rows are the shots and B is the weight set -- which is what a reader might
// assume from "C = A . B" and is NOT what the device does.
bool reference_gemm(const pta_cfg& cfg, const pta_tile& tile, pta_device* dev,
                    const Case& c, const std::vector<int8_t>& A,
                    const std::vector<int8_t>& B, bool weights_are_A,
                    std::vector<int32_t>* C) {
  const int m = c.m, n = c.n, k = c.k;
  C->assign((size_t)m * n, 0);
  if (weights_are_A) {
    // The engine computes C^T (n x m) = B^T (n x k) . A^T (k x m), row-major.
    std::vector<int32_t> act((size_t)n * k), wgt((size_t)k * m);
    std::vector<int64_t> out((size_t)n * m);
    for (int j = 0; j < n; ++j)
      for (int p = 0; p < k; ++p)
        act[(size_t)j * k + p] = B[p + (size_t)j * k];          // shot j = column j of B
    for (int p = 0; p < k; ++p)
      for (int i = 0; i < m; ++i)
        wgt[(size_t)p * m + i] = A[i + (size_t)p * m];          // weight (p, i) = A(i, p)
    if (pta_gemm(&cfg, &tile, dev, 0, n, m, k, act.data(), wgt.data(),
                 out.data()) < 0)
      return false;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i)
        (*C)[i + (size_t)j * m] = (int32_t)(uint32_t)(uint64_t)out[(size_t)j * m + i];
  } else {
    std::vector<int32_t> act((size_t)m * k), wgt((size_t)k * n);
    std::vector<int64_t> out((size_t)m * n);
    for (int i = 0; i < m; ++i)
      for (int p = 0; p < k; ++p)
        act[(size_t)i * k + p] = A[i + (size_t)p * m];          // shot i = row i of A
    for (int p = 0; p < k; ++p)
      for (int j = 0; j < n; ++j)
        wgt[(size_t)p * n + j] = B[p + (size_t)j * k];
    if (pta_gemm(&cfg, &tile, dev, 0, m, n, k, act.data(), wgt.data(),
                 out.data()) < 0)
      return false;
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        (*C)[i + (size_t)j * m] = (int32_t)(uint32_t)(uint64_t)out[(size_t)i * n + j];
  }
  return true;
}

// A case's registers as a model configuration. Used to WRITE the golden data,
// and for nothing else: the gate itself builds its configuration from the
// device property, below.
pta_cfg cfg_of_case(const Case& c) {
  pta_cfg cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.impair      = c.impair;
  cfg.act_bits    = c.bits & 0xFu;
  cfg.w_bits      = (c.bits >> 4) & 0xFu;
  cfg.adc_bits    = (c.bits >> 8) & 0xFu;
  cfg.adc_shift   = (c.bits >> 12) & 0x3Fu;
  cfg.seed        = c.seed;
  cfg.sigma_th    = c.sigma_th;
  cfg.k_shot      = c.sigma_sh;
  cfg.sigma_pr    = c.sigma_pr;
  cfg.drift_sigma = c.drift & 0xFFFFu;
  cfg.drift_log2  = (c.drift >> 16) & 0x1Fu;
  cfg.drift_max   = c.drift_max;
  cfg.xtalk       = c.xtalk;
  return cfg;
}

// The same, from what the DEVICE says about itself, and nothing else.
pta_cfg cfg_of_property(const grxAnalogGemm_t& p) {
  pta_cfg cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.impair      = (uint32_t)p.impairments;
  cfg.act_bits    = (uint32_t)p.activationBits;
  cfg.w_bits      = (uint32_t)p.weightBits;
  cfg.adc_bits    = (uint32_t)p.adcBits;
  cfg.adc_shift   = (uint32_t)p.adcShift;
  cfg.seed        = (uint32_t)p.seed;
  cfg.sigma_th    = (uint32_t)p.thermalSigmaQ8;
  cfg.k_shot      = (uint32_t)p.shotCoefficientQ8;
  cfg.sigma_pr    = (uint32_t)p.programmingSigmaQ8;
  cfg.drift_sigma = (uint32_t)p.driftSigmaQ8;
  cfg.drift_log2  = (uint32_t)p.driftLog2Shots;
  cfg.drift_max   = (uint32_t)p.driftClampQ8;
  cfg.xtalk       = (uint32_t)p.crosstalkQ8;
  return cfg;
}

// Every GEMM of a case, straight from the model: the tile reset to the case's
// seed, then `gemms` GEMMs on it with its drift carried.
bool model_case(const pta_cfg& cfg, const pta_tile& tile, const Case& c,
                bool weights_are_A, std::vector<std::vector<int32_t>>* out) {
  std::vector<int8_t> A, B;
  operands(c, &A, &B);
  pta_device dev;
  if (pta_device_init(&dev, &tile) != 0) return false;
  pta_model_reset(&dev, cfg.seed);
  out->clear();
  bool ok = true;
  for (int g = 0; g < c.gemms && ok; ++g) {
    std::vector<int32_t> C;
    ok = reference_gemm(cfg, tile, &dev, c, A, B, weights_are_A, &C);
    out->push_back(C);
  }
  pta_device_free(&dev);
  return ok;
}

// ---------------------------------------------------------------------------
// The golden file
// ---------------------------------------------------------------------------

struct GoldenCase {
  std::string name;
  int m = 0, n = 0, k = 0, gemms = 0;
  uint32_t op_seed = 0;
  uint32_t reg[9] = {0};
  std::vector<std::vector<int32_t>> c;
};

struct Golden {
  int version = 0;
  pta_tile tile = {0, 0, 0, 0};
  std::vector<GoldenCase> cases;
};

bool write_golden(const char* path) {
  FILE* f = std::fopen(path, "wb");
  if (!f) return false;
  std::fprintf(f,
      "# grxblasGemmEx on the photonic tile: golden results.\n"
      "#\n"
      "# Emitted by tests/libs/test_grxblas_pta --regenerate from\n"
      "# third_party/grx930/pta_tile_model.c. tests/libs/test_grxblas_pta holds\n"
      "# grxblasGemmEx on the tile to every number below, bit for bit.\n"
      "#\n"
      "# GOLDEN DATA. Do not edit by hand. A number here moves only when the\n"
      "# vendored error model moves, and then this file is regenerated and its\n"
      "# diff goes in the same change, so the move is seen (AGENTS.md section 4).\n"
      "#\n"
      "# Operands are int8, from the model's xorshift32 seeded with\n"
      "# op_seed ^ 0x2545f491: A first (m*k values, column-major), then B\n"
      "# (k*n), each the low byte of the state. The tile is reset to the\n"
      "# case's seed before its first GEMM. C is column-major, m*n int32.\n"
      "# A is the weight set; each column of B is one shot.\n"
      "version %d\n"
      "tile rows=%d cols=%d operand_bits=%d accumulator_bits=%d\n"
      "cases %d\n",
      kGoldenVersion, kTile.rows, kTile.cols, kTile.din_w, kTile.acc_w, kNumCases);
  bool ok = true;
  for (int ci = 0; ci < kNumCases && ok; ++ci) {
    const Case& c = kCases[ci];
    std::vector<std::vector<int32_t>> res;
    ok = model_case(cfg_of_case(c), kTile, c, true, &res);
    std::fprintf(f, "\ncase %s\n", c.name);
    std::fprintf(f, "why %s\n", c.why);
    std::fprintf(f, "shape m=%d n=%d k=%d op_seed=%u gemms=%d\n", c.m, c.n, c.k,
                 c.op_seed, c.gemms);
    std::fprintf(f, "reg impair=0x%02x bits=0x%05x seed=0x%08x sigma_th=0x%04x "
                    "sigma_sh=0x%04x sigma_pr=0x%04x drift=0x%06x xtalk=0x%02x "
                    "drift_max=0x%04x\n",
                 c.impair, c.bits, c.seed, c.sigma_th, c.sigma_sh, c.sigma_pr,
                 c.drift, c.xtalk, c.drift_max);
    for (int g = 0; g < c.gemms && ok; ++g) {
      std::fprintf(f, "c %d", g);
      for (size_t i = 0; i < res[g].size(); ++i)
        std::fprintf(f, "%s%08x", (i % 8 == 0) ? "\n " : " ", (uint32_t)res[g][i]);
      std::fprintf(f, "\n");
    }
  }
  std::fclose(f);
  return ok;
}

bool read_golden(const char* path, Golden* g) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  char line[512];
  GoldenCase* cur = nullptr;
  int expect_cases = -1;
  std::vector<int32_t>* cvec = nullptr;
  bool ok = true;
  while (ok && std::fgets(line, sizeof(line), f)) {
    size_t len = std::strlen(line);
    while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
    if (!len || line[0] == '#') continue;
    if (line[0] == ' ') {                       // a row of C
      if (!cvec) { ok = false; break; }
      char* p = line;
      for (;;) {
        while (*p == ' ') ++p;
        if (!*p) break;
        char* end = nullptr;
        const unsigned long v = std::strtoul(p, &end, 16);
        if (end == p) { ok = false; break; }
        cvec->push_back((int32_t)(uint32_t)v);
        p = end;
      }
      continue;
    }
    char name[128];
    if (std::sscanf(line, "version %d", &g->version) == 1) continue;
    if (std::sscanf(line, "tile rows=%d cols=%d operand_bits=%d accumulator_bits=%d",
                    &g->tile.rows, &g->tile.cols, &g->tile.din_w,
                    &g->tile.acc_w) == 4) continue;
    if (std::sscanf(line, "cases %d", &expect_cases) == 1) continue;
    if (std::sscanf(line, "case %127s", name) == 1) {
      g->cases.emplace_back();
      cur = &g->cases.back();
      cur->name = name;
      cvec = nullptr;
      continue;
    }
    if (!std::strncmp(line, "why ", 4)) continue;
    if (!cur) { ok = false; break; }
    if (std::sscanf(line, "shape m=%d n=%d k=%d op_seed=%u gemms=%d", &cur->m,
                    &cur->n, &cur->k, &cur->op_seed, &cur->gemms) == 5) continue;
    if (std::sscanf(line, "reg impair=%x bits=%x seed=%x sigma_th=%x sigma_sh=%x "
                          "sigma_pr=%x drift=%x xtalk=%x drift_max=%x",
                    &cur->reg[0], &cur->reg[1], &cur->reg[2], &cur->reg[3],
                    &cur->reg[4], &cur->reg[5], &cur->reg[6], &cur->reg[7],
                    &cur->reg[8]) == 9) continue;
    int gi = -1;
    if (std::sscanf(line, "c %d", &gi) == 1 && gi == (int)cur->c.size()) {
      cur->c.emplace_back();
      cvec = &cur->c.back();
      continue;
    }
    ok = false;                                  // a line this reader does not know
  }
  std::fclose(f);
  return ok && expect_cases == (int)g->cases.size();
}

bool golden_matches_table(const GoldenCase& gc, const Case& c) {
  const uint32_t reg[9] = {c.impair, c.bits, c.seed, c.sigma_th, c.sigma_sh,
                           c.sigma_pr, c.drift, c.xtalk, c.drift_max};
  if (gc.name != c.name || gc.m != c.m || gc.n != c.n || gc.k != c.k ||
      gc.op_seed != c.op_seed || gc.gemms != c.gemms ||
      (int)gc.c.size() != c.gemms)
    return false;
  for (int i = 0; i < 9; ++i)
    if (gc.reg[i] != reg[i]) return false;
  for (const std::vector<int32_t>& v : gc.c)
    if (v.size() != (size_t)c.m * c.n) return false;
  return true;
}

// ---------------------------------------------------------------------------
// The device
// ---------------------------------------------------------------------------

// A case's registers, written the way another user of the device would write
// them: straight into the block. grxcp has no call that configures the tile,
// and this gate is not where one gets invented.
void program(const Case& c) {
  npu_dpi_csr_write(NPU_CSR_PTA_IMPAIR, c.impair);
  npu_dpi_csr_write(NPU_CSR_PTA_BITS, c.bits);
  npu_dpi_csr_write(NPU_CSR_PTA_SEED, c.seed);
  npu_dpi_csr_write(NPU_CSR_PTA_SIGMA_TH, c.sigma_th);
  npu_dpi_csr_write(NPU_CSR_PTA_SIGMA_SH, c.sigma_sh);
  npu_dpi_csr_write(NPU_CSR_PTA_SIGMA_PR, c.sigma_pr);
  npu_dpi_csr_write(NPU_CSR_PTA_DRIFT, c.drift);
  npu_dpi_csr_write(NPU_CSR_PTA_XTALK, c.xtalk);
  npu_dpi_csr_write(NPU_CSR_PTA_DRIFT_MAX, c.drift_max);
  // After the seed: the reset reloads the drift generator from it.
  npu_dpi_csr_write(NPU_CSR_PTA_CTRL, NPU_PTA_CTRL_MODEL_RST);
}

bool property_is_the_case(const grxAnalogGemm_t& p, const Case& c) {
  return p.gemmIsAnalogEmulated == 1 && p.tileIsPresent == 1 &&
         p.impairments == (int64_t)c.impair &&
         p.activationBits == (int)(c.bits & 0xF) &&
         p.weightBits == (int)((c.bits >> 4) & 0xF) &&
         p.adcBits == (int)((c.bits >> 8) & 0xF) &&
         p.adcShift == (int)((c.bits >> 12) & 0x3F) &&
         p.seed == (int64_t)c.seed && p.thermalSigmaQ8 == (int)c.sigma_th &&
         p.shotCoefficientQ8 == (int)c.sigma_sh &&
         p.programmingSigmaQ8 == (int)c.sigma_pr &&
         p.driftSigmaQ8 == (int)(c.drift & 0xFFFF) &&
         p.driftLog2Shots == (int)((c.drift >> 16) & 0x1F) &&
         p.driftClampQ8 == (int)c.drift_max && p.crosstalkQ8 == (int)c.xtalk &&
         p.loopModes == 0 && p.calibrationValid == 0;
}

// One GEMM through the front door. Returns the status; *C is the device's.
grxblasStatus_t device_gemm(grxblasHandle_t h, const Case& c,
                            const std::vector<int8_t>& A,
                            const std::vector<int8_t>& B,
                            std::vector<int32_t>* C) {
  const size_t cn = (size_t)c.m * c.n;
  void *dA = nullptr, *dB = nullptr, *dC = nullptr;
  if (grxMalloc(&dA, A.size()) != grxSuccess ||
      grxMalloc(&dB, B.size()) != grxSuccess ||
      grxMalloc(&dC, cn * sizeof(int32_t)) != grxSuccess)
    return GRXBLAS_STATUS_ALLOC_FAILED;
  // A sentinel no result here can equal by accident, so a GEMM that reported
  // success and wrote nothing is caught rather than read back as an answer.
  std::vector<int32_t> sentinel(cn, (int32_t)0x5A5A5A5A);
  grxMemcpy(dA, A.data(), A.size(), grxMemcpyDefault);
  grxMemcpy(dB, B.data(), B.size(), grxMemcpyDefault);
  grxMemcpy(dC, sentinel.data(), cn * sizeof(int32_t), grxMemcpyDefault);
  const float one = 1.0f, zero = 0.0f;
  const grxblasStatus_t s = grxblasGemmEx(
      h, GRXBLAS_OP_N, GRXBLAS_OP_N, c.m, c.n, c.k, &one,
      dA, GRX_R_8I, c.m, dB, GRX_R_8I, c.k, &zero, dC, GRX_R_32I, c.m);
  grxDeviceSynchronize();
  C->assign(cn, 0);
  grxMemcpy(C->data(), dC, cn * sizeof(int32_t), grxMemcpyDefault);
  grxFree(dA); grxFree(dB); grxFree(dC);
  return s;
}

int mismatches(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  if (a.size() != b.size()) return (int)(a.size() > b.size() ? a.size() : b.size());
  int n = 0;
  for (size_t i = 0; i < a.size(); ++i) n += (a[i] != b[i]);
  return n;
}

}  // namespace
#endif  // GRXCP_ENABLE_NPU

int main(int argc, char** argv) {
#ifndef GRXCP_ENABLE_NPU
  (void)argc; (void)argv;
  std::printf("built without GRXCP_ENABLE_NPU; no NPU backend here. skipping\n");
  return 77;
#else
  if (argc > 1 && !std::strcmp(argv[1], "--regenerate")) {
    const char* path = (argc > 2) ? argv[2] : kGoldenDefault;
    if (!write_golden(path)) {
      std::printf("could not write %s\n", path);
      return 1;
    }
    std::printf("wrote %d cases to %s from the vendored model.\n"
                "This is golden data: review the diff, and commit it in the same "
                "change as whatever moved the model.\n", kNumCases, path);
    return 0;
  }
  const char* path = (argc > 1) ? argv[1] : kGoldenDefault;

  // BEFORE THE FIRST grx CALL. Enumeration runs once and the seam refuses
  // afterwards.
  const bool installed = grxtest::npu_tile_install();

  section("the vendored model is behind the register map");
  check(installed, "the shim was built with the tile model and took it");
  if (!installed) {
    std::printf("        This binary's shim was compiled without\n"
                "        NPU_DPI_WITH_PTA, so there is no tile to gate. That is\n"
                "        a build fault here, not a skip.\n");
    return grxtest::report();
  }

  // ---- the golden file ----------------------------------------------------
  section("the golden data");
  Golden golden;
  const bool have_golden = read_golden(path, &golden);
  check(have_golden, "tests/libs/pta_gemm_golden.txt reads, every line of it");
  if (!have_golden) {
    std::printf("        Looked for '%s' from the working directory. Run this\n"
                "        from tests/libs, or pass the path; to create the file,\n"
                "        --regenerate.\n", path);
    return grxtest::report();
  }
  check(golden.version == kGoldenVersion, "format version 1");
  check(golden.tile.rows == kTile.rows && golden.tile.cols == kTile.cols &&
        golden.tile.din_w == kTile.din_w && golden.tile.acc_w == kTile.acc_w,
        "generated for a 4 x 4 tile of 16-bit operands and 48-bit sums");
  bool table_ok = (int)golden.cases.size() == kNumCases;
  for (int ci = 0; ci < kNumCases && table_ok; ++ci)
    table_ok = golden_matches_table(golden.cases[ci], kCases[ci]);
  check(table_ok, "its cases are this file's: names, shapes and registers");
  if (!table_ok) {
    std::printf("        The golden file describes a different suite. If the\n"
                "        cases were changed on purpose, --regenerate and review.\n");
    return grxtest::report();
  }

  // The file against the model that is vendored TODAY. This is the check that
  // goes red when the model changes under the data, before any device is
  // involved -- and the one a hand-edited number fails.
  {
    int bad = 0, total = 0;
    for (int ci = 0; ci < kNumCases; ++ci) {
      std::vector<std::vector<int32_t>> res;
      if (!model_case(cfg_of_case(kCases[ci]), kTile, kCases[ci], true, &res)) {
        bad += 1;
        continue;
      }
      for (int g = 0; g < kCases[ci].gemms; ++g) {
        bad += mismatches(res[g], golden.cases[ci].c[g]);
        total += (int)res[g].size();
      }
    }
    std::printf("  note  %d results in %d cases\n", total, kNumCases);
    check(bad == 0, "every number in it is what the vendored model computes");
    if (bad) {
      std::printf("        %d differ. Either the model moved or the file was\n"
                  "        edited. If the model moved on purpose: --regenerate,\n"
                  "        review the diff, commit both together.\n", bad);
      return grxtest::report();
    }
  }

  // ---- the device ---------------------------------------------------------
  int count = 0;
  GRX_REQUIRE(grxGetDeviceCount(&count), "grxGetDeviceCount");
  int npu = -1;
  for (int i = 0; i < count; ++i) {
    grxDeviceProp_t p{};
    if (grxGetDeviceProperties(&p, i) == grxSuccess &&
        p.deviceType == GRX_DEVICE_TYPE_NPU) { npu = i; break; }
  }
  section("the device");
  check(npu >= 0, "an NPU is enumerated");
  if (npu < 0) return grxtest::report();
  GRX_REQUIRE(grxSetDevice(npu), "grxSetDevice");
  {
    grxDeviceProp_t p{};
    grxGetDeviceProperties(&p, npu);
    std::printf("  note  %s\n", p.name);
    check(p.backend == GRX_BACKEND_MODEL,
          "it says it is a register model, not hardware");
    const grxAnalogGemm_t& a = p.analogGemm;
    check(a.tileIsPresent == 1 && a.gemmIsAnalogEmulated == 0,
          "a tile is present, and with PTA_IMPAIR clear its GEMMs are exact");
    check(a.impairmentsImplemented == 0x5F, "it implements all but MZM_NL");
    check(a.tileRows == kTile.rows && a.tileCols == kTile.cols &&
          a.operandBits == kTile.din_w && a.accumulatorBits == kTile.acc_w,
          "and it is the tile the golden data was generated for");
  }

  grxblasHandle_t h = nullptr;
  if (grxblasCreate(&h) != GRXBLAS_STATUS_SUCCESS) {
    check(false, "grxblasCreate");
    return grxtest::report();
  }

  section("unimpaired, the tile is the product");
  {
    std::vector<int8_t> A, B;
    std::vector<int32_t> C;
    operands(kCases[0], &A, &B);
    const grxblasStatus_t s = device_gemm(h, kCases[0], A, B, &C);
    check(s == GRXBLAS_STATUS_SUCCESS &&
          mismatches(C, exact_gemm(kCases[0], A, B)) == 0,
          "12 x 8 x 16 int8: every element exact");
  }

  // ---- the gate -----------------------------------------------------------
  section("bitwise: the device, the property, and the golden data");
  int total_results = 0;
  for (int ci = 0; ci < kNumCases; ++ci) {
    const Case& c = kCases[ci];
    std::vector<int8_t> A, B;
    operands(c, &A, &B);
    const std::vector<int32_t> exact = exact_gemm(c, A, B);

    program(c);

    grxDeviceProp_t prop{};
    grxGetDeviceProperties(&prop, npu);
    const grxAnalogGemm_t p = prop.analogGemm;
    const bool prop_ok = property_is_the_case(p, c);

    // The reference, built from the property alone -- the configuration AND the
    // tile -- on a device reset to the seed the property reports.
    const pta_cfg  pcfg  = cfg_of_property(p);
    const pta_tile ptile = {p.tileRows, p.tileCols, p.operandBits, p.accumulatorBits};
    std::vector<std::vector<int32_t>> ref;
    const bool ref_ok = prop_ok && model_case(pcfg, ptile, c, true, &ref);

    int vs_golden = 0, vs_property = 0, moved = 0, carried = 0, failed = 0;
    std::vector<int32_t> first;
    for (int g = 0; g < c.gemms; ++g) {
      std::vector<int32_t> C;
      if (device_gemm(h, c, A, B, &C) != GRXBLAS_STATUS_SUCCESS) ++failed;
      vs_golden += mismatches(C, golden.cases[ci].c[g]);
      if (ref_ok) vs_property += mismatches(C, ref[g]);
      moved += mismatches(C, exact);
      if (g == 0) first = C; else carried += mismatches(C, first);
      total_results += (int)C.size();
    }

    char what[256];
    std::snprintf(what, sizeof(what),
                  "%-14s the property reports what was programmed", c.name);
    check(prop_ok, what);
    std::snprintf(what, sizeof(what),
                  "%-14s %d x %d x %d%s: the device's C is the golden data's, "
                  "bit for bit", c.name, c.m, c.n, c.k,
                  c.gemms > 1 ? ", twice" : "");
    check(failed == 0 && vs_golden == 0, what);
    if (failed) std::printf("        %d GEMM(s) did not report success\n", failed);
    if (vs_golden) std::printf("        %d element(s) differ\n", vs_golden);
    std::snprintf(what, sizeof(what),
                  "%-14s and it is what the model gives from the property alone",
                  c.name);
    check(ref_ok && vs_property == 0, what);
    std::snprintf(what, sizeof(what),
                  "%-14s which is not the exact product (%d of %d moved)", c.name,
                  moved, c.m * c.n * c.gemms);
    check(moved > 0, what);
    if (c.gemms > 1) {
      std::snprintf(what, sizeof(what),
                    "%-14s the second GEMM is not the first: drift carried "
                    "(%d of %d moved)", c.name, carried, c.m * c.n);
      check(carried > 0, what);
    }
  }
  std::printf("  note  %d results compared three ways\n", total_results);

  // ---- whose operand is the weight ----------------------------------------
  // The "weights" case impairs only the weight side. Reproduced with the roles
  // the other way round -- A's rows as the shots, B as the weight set -- the
  // model gives a different C, so the statement at the top of this file is
  // something the gate would notice being wrong.
  section("which operand is the weight");
  {
    const Case& c = kCases[3];
    std::vector<std::vector<int32_t>> as_built, other;
    const bool ok1 = model_case(cfg_of_case(c), kTile, c, true, &as_built);
    const bool ok2 = model_case(cfg_of_case(c), kTile, c, false, &other);
    check(ok1 && ok2 && mismatches(as_built[0], golden.cases[3].c[0]) == 0,
          "A as the weight set and B's columns as the shots is the golden data");
    const int diff = (ok1 && ok2) ? mismatches(as_built[0], other[0]) : 0;
    std::printf("  note  %d of %d elements differ under the other reading\n", diff,
                c.m * c.n);
    check(diff > 0, "the other reading is a different answer, so this is pinned");
  }

  // ---- and back -----------------------------------------------------------
  section("PTA_IMPAIR cleared");
  npu_dpi_csr_write(NPU_CSR_PTA_IMPAIR, 0);
  {
    grxDeviceProp_t prop{};
    grxGetDeviceProperties(&prop, npu);
    check(prop.analogGemm.gemmIsAnalogEmulated == 0 &&
          prop.analogGemm.thermalSigmaQ8 == -1,
          "the property says native again, and withdraws the model's numbers");
    std::vector<int8_t> A, B;
    std::vector<int32_t> C;
    operands(kCases[1], &A, &B);
    const grxblasStatus_t s = device_gemm(h, kCases[1], A, B, &C);
    check(s == GRXBLAS_STATUS_SUCCESS &&
          mismatches(C, exact_gemm(kCases[1], A, B)) == 0,
          "and a GEMM is the exact product again");
  }

  check(grxblasDestroy(h) == GRXBLAS_STATUS_SUCCESS, "grxblasDestroy");
  return grxtest::report();
#endif
}
