// grxblasGemmEx on a photonic tile: how far from the product, and why.
//
// docs/designs/pta_cpu_integration.md section 7 splits the numerical gate for
// an analog GEMM in two. The first half (test_grxblas_pta.cpp) holds the device
// to the error model bit for bit, and catches bugs. This is the second half,
// and it answers the other question: how far is the model's answer from the
// product the caller asked for? That is "reported rather than gated at zero
// tolerance ... with a stated bound that moves as the error model is tuned".
//
// THE STATED BOUND IS A PREDICTION, NOT A FITTED NUMBER.
//
// The obvious way to bound an error is to measure it once and write down
// something a little larger. That bound knows nothing: it moves when somebody
// re-measures, and it passes a tile whose noise has silently been switched off,
// because less error is under the bound too.
//
// So each impairment's contribution is worked out here from first principles --
// the configuration the device reports, the tile's geometry, and four moments
// of the operands -- and the measured error has to land within a stated band of
// it, ABOVE AND BELOW. The bound then moves by formula when the error model is
// tuned, which is what "moves as the error model is tuned" should mean, and a
// measured error well under the prediction fails as surely as one well over:
// that is an impairment not being applied.
//
// Per output element, in units of the product, with K_t the number of K tiles
// an output sums over, S the ADC shift and g^2 the variance of the model's
// Gaussian generator (four bytes summed, times 443: 0.998):
//
//   operand quantisation   k (E[a^2] E[dw^2] + E[da^2] E[w^2] + E[da^2] E[dw^2]
//                             + 2 E[a da] E[w dw])
//   ADC quantisation       K_t 4^S / 12
//   thermal                K_t (sigma_th/256)^2 4^S g^2
//   shot                   K_t (k_shot/256)^2 2^S E|y_tile| g^2
//   programming            k E[a^2] (sigma_pr/256)^2 g^2
//   crosstalk              (chi/256)^2 E[a^2] E[w^2] * (neighbour pairs)
//
// where a is an activation, w a weight, da and dw their quantisation errors,
// and y_tile one K tile's partial sum. The operand moments and E|y_tile| are
// measured on a CALIBRATION set of operands, exactly, by this file's own
// arithmetic -- not by the model -- and the ADC shift is set on the same set by
// the rule grx930's accuracy harness uses: the smallest shift that clips no
// more than one K-tile sum in ten thousand.
//
// THE OPERATING POINTS are the board plan's (board_program_plan.md section
// 4.3): version 0, measured one impairment at a time, and version 1, what
// survived running them together. Each is run here one impairment at a time
// and then all at once, so the report also says whether GEMM-level error
// ADDS -- in quadrature, as independent errors do -- or compounds.
//
// WHAT THIS IS A REPORT ABOUT. The register model's tile: 4 x 4, sixteen-bit
// operands. One shape, 12 x 8 x 16, the largest the engine takes. Uniform int8
// operands. A different tile sums over a different number of K tiles and a
// different workload has different moments, and both move every number here;
// the formulas take them as inputs, the measurements do not transfer. And it
// is the error MODEL: nothing here is a statement about a photonic device.
//
//   test_grxblas_pta_dist                       run the report (from tests/libs)
//   test_grxblas_pta_dist --regenerate [FILE]   re-record what was measured

#include <grx/grx.h>
#include <grx/grxblas.h>

#include "grx_test.h"

#include <cmath>
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

const char* const kRecordDefault = "pta_gemm_distribution.txt";
const int         kRecordVersion = 1;

// The shape, and how much of it. 128 GEMMs of 96 elements is 12,288 samples a
// point, which puts the sampling error of an RMS under one percent.
const int kM = 12, kN = 8, kK = 16;
const int kGemms    = 128;
const int kCalGemms = 64;

// How far a measurement may sit from its prediction, either way. The formulas
// assume independent, zero-mean operands and exact square roots; the model's
// root is a four-segment approximation and its Gaussian is four bytes. Fifteen
// percent of an RMS is room for those and for nothing structural: a doubled or
// a missing impairment is outside it by a wide margin.
const double kBand = 0.15;

// The fraction of K-tile sums the ADC may clip, as grx930's sim/pta_mnist.c
// sets each layer's shift (its CLIP_FRAC).
const double kClipFrac = 1e-4;

// The variance of the model's Gaussian generator, as a fraction of the unit
// one it stands for: the four bytes of an xorshift state summed (variance
// 4 * (256^2 - 1) / 12), times 443, against 2^16.
const double kGaussVar = (443.0 * 443.0 * 4.0 * (65536.0 - 1.0) / 12.0) /
                         (65536.0 * 65536.0);

// The board plan's two versions, as grx930's sim/pta_mnist.sh passes them
// (its v0_bits/v0_noise and tight_bits/tighter_noise), in the model's units:
// thermal and programming sigma in Q8.8, the shot coefficient 256/sqrt(photons
// per ADC LSB), crosstalk in Q0.8. The bit counts are EFFECTIVE bits of an
// int8 operand; what is written to PTA_BITS depends on the tile's word.
struct Version {
  const char* name;
  int      act_bits, w_bits, adc_bits;
  uint32_t sigma_th, k_shot, sigma_pr, xtalk;
};
const Version kVersions[2] = {
  {"v0", 5, 6, 6, 256, 148, 1024, 26},   // thermal 1, 3 photons, prog 4, 10%
  {"v1", 6, 6, 7, 64, 47, 256, 5},       // thermal 0.25, 30 photons, prog 1, 2%
};

enum Part { P_QUANT, P_THERMAL, P_SHOT, P_PROG, P_XTALK, P_ALL, P_COUNT };
const char* const kPartName[P_COUNT] = {"quantisers", "thermal", "shot",
                                        "programming", "crosstalk", "all"};
const uint32_t kPartImpair[P_COUNT] = {0x01, 0x02, 0x04, 0x40, 0x10, 0x57};

// ---------------------------------------------------------------------------
// Operands
// ---------------------------------------------------------------------------

// Uniform int8 from the model's xorshift, the stream gate 1 uses.
void operands(uint32_t seed, std::vector<int8_t>* A, std::vector<int8_t>* B) {
  uint32_t s = seed ^ 0x2545f491u;
  A->resize((size_t)kM * kK);
  B->resize((size_t)kK * kN);
  for (int8_t& v : *A) { s = pta_xorshift32(s); v = (int8_t)(s & 0xFFu); }
  for (int8_t& v : *B) { s = pta_xorshift32(s); v = (int8_t)(s & 0xFFu); }
}

// THE REFERENCE IS fp32, as the design asks, and it is exact. Every partial sum
// of a 12 x 8 x 16 int8 GEMM is below 2^24, so float accumulation loses
// nothing; the integer product is computed beside it and the two must agree,
// which is what entitles the rest of this file to call the error "the error".
bool reference(const std::vector<int8_t>& A, const std::vector<int8_t>& B,
               std::vector<int32_t>* C) {
  C->assign((size_t)kM * kN, 0);
  bool same = true;
  for (int j = 0; j < kN; ++j)
    for (int i = 0; i < kM; ++i) {
      float f = 0.0f;
      int32_t s = 0;
      for (int p = 0; p < kK; ++p) {
        const int a = A[i + (size_t)p * kM], b = B[p + (size_t)j * kK];
        f += (float)a * (float)b;
        s += a * b;
      }
      if ((float)s != f) same = false;
      (*C)[i + (size_t)j * kM] = (int32_t)f;
    }
  return same;
}

// ---------------------------------------------------------------------------
// Calibration: what the predictions are made from
// ---------------------------------------------------------------------------

// The tile's quantiser, written from its definition and not by calling the
// model: keep `bits` of an int8, round to nearest with halves up, clamp to the
// signed range. bits == 0 is unquantised.
int quantise(int x, int bits) {
  if (bits <= 0 || bits >= 8) return x;
  const int h = 8 - bits;
  int v = (x + (1 << (h - 1))) >> h;            // arithmetic shift: floor
  const int lim = 1 << (bits - 1);
  if (v > lim - 1) v = lim - 1;
  if (v < -lim)    v = -lim;
  return v * (1 << h);
}

struct Moments {
  double a2 = 0, w2 = 0;          // E[a^2], E[w^2]
  double da2 = 0, dw2 = 0;        // E[da^2], E[dw^2]
  double ada = 0, wdw = 0;        // E[a da], E[w dw]
  double abs_tile = 0;            // E|y_tile|, unquantised operands
  int    shift = 0;               // the ADC shift the clip rule gives
  double clipped = 0;             // the fraction it clips on this set
};

// For C = A . B through grxblasGemmEx, A is the weight set and each column of
// B is a shot (test_grxblas_pta.cpp pins that). A K tile is `rows` consecutive
// values of p.
Moments calibrate(const Version& v, int rows) {
  Moments m;
  double na = 0, nw = 0, nt = 0;
  std::vector<long> hist(64, 0);
  const int64_t lim = ((int64_t)1 << (v.adc_bits - 1)) - 1;
  for (int g = 0; g < kCalGemms; ++g) {
    std::vector<int8_t> A, B;
    operands(100000u + (uint32_t)g, &A, &B);
    for (int8_t x : B) {                         // activations
      const int d = quantise(x, v.act_bits) - x;
      m.a2 += (double)x * x; m.da2 += (double)d * d; m.ada += (double)x * d;
      na += 1;
    }
    for (int8_t x : A) {                         // weights
      const int d = quantise(x, v.w_bits) - x;
      m.w2 += (double)x * x; m.dw2 += (double)d * d; m.wdw += (double)x * d;
      nw += 1;
    }
    for (int j = 0; j < kN; ++j)
      for (int i = 0; i < kM; ++i)
        for (int base = 0; base < kK; base += rows) {
          int64_t raw = 0, wq = 0;
          for (int p = base; p < base + rows && p < kK; ++p) {
            const int a = B[p + (size_t)j * kK], w = A[i + (size_t)p * kM];
            raw += (int64_t)a * w;
            wq  += (int64_t)a * quantise(w, v.w_bits);   // weights as programmed
          }
          m.abs_tile += (double)(raw < 0 ? -raw : raw);
          nt += 1;
          // The smallest shift at which this sum fits the ADC: round(v / 2^s)
          // no greater than the largest code.
          int64_t mag = wq < 0 ? -wq : wq;
          int s = 0;
          while ((s == 0 ? mag : (mag + ((int64_t)1 << (s - 1))) >> s) > lim) ++s;
          ++hist[s];
        }
  }
  m.a2 /= na; m.da2 /= na; m.ada /= na;
  m.w2 /= nw; m.dw2 /= nw; m.wdw /= nw;
  m.abs_tile /= nt;
  // The smallest shift that clips no more than kClipFrac of the sums.
  long over = (long)nt;
  for (int s = 0; s < 64; ++s) {
    over -= hist[s];
    if (over <= (long)(kClipFrac * nt)) { m.shift = s; m.clipped = over / nt; break; }
  }
  return m;
}

// The predicted error VARIANCE of one output element for each part, in product
// units squared. The sum of the first five is the prediction for "all": that
// is what independence means, and whether it holds is one of the findings.
void predict(const Version& v, const Moments& m, int rows, double out[P_COUNT]) {
  const double S4 = std::ldexp(1.0, 2 * m.shift), S2 = std::ldexp(1.0, m.shift);
  int kt = 0, pairs = 0;
  for (int base = 0; base < kK; base += rows) {
    const int kr = (kK - base < rows) ? kK - base : rows;
    ++kt;
    pairs += 2 * (kr - 1);           // each row couples to its neighbours in the tile
  }
  const double th = v.sigma_th / 256.0, sh = v.k_shot / 256.0;
  const double pr = v.sigma_pr / 256.0, xt = v.xtalk / 256.0;
  const double operand_q = kK * (m.a2 * m.dw2 + m.da2 * m.w2 + m.da2 * m.dw2 +
                                 2.0 * m.ada * m.wdw);
  const double adc_q = kt * S4 / 12.0;
  out[P_QUANT]   = operand_q + adc_q;
  out[P_THERMAL] = kt * th * th * S4 * kGaussVar;
  out[P_SHOT]    = kt * sh * sh * S2 * m.abs_tile * kGaussVar;
  out[P_PROG]    = kK * m.a2 * pr * pr * kGaussVar;
  out[P_XTALK]   = xt * xt * m.a2 * m.w2 * pairs;
  out[P_ALL]     = out[P_QUANT] + out[P_THERMAL] + out[P_SHOT] + out[P_PROG] +
                   out[P_XTALK];
}

// ---------------------------------------------------------------------------
// Measurement, through the device
// ---------------------------------------------------------------------------

struct Measured {
  uint64_t sum_e2 = 0, sum_c2 = 0;
  int64_t  sum_e = 0;
  uint32_t max_abs_e = 0;
  uint32_t moved = 0, sats = 0, count = 0;
  bool     ok = true;
  double rel() const { return sum_c2 ? std::sqrt((double)sum_e2 / (double)sum_c2) : 0.0; }
};

uint32_t bits_setting(const Version& v, Part part, int operand_bits, int shift) {
  // Quantisers only where the part includes them. Five bits of an int8 on a
  // sixteen-bit word is a setting of thirteen: the quantiser keeps the top.
  const bool q = (part == P_QUANT || part == P_ALL);
  const uint32_t a = q ? (uint32_t)(operand_bits - 8 + v.act_bits) : 0u;
  const uint32_t w = q ? (uint32_t)(operand_bits - 8 + v.w_bits) : 0u;
  const uint32_t adc = q ? (uint32_t)v.adc_bits : 0u;
  // The shift is set for every part: the noise sigmas are in ADC LSB, so the
  // LSB has to be the same one whether or not the ADC is quantising.
  return a | (w << 4) | (adc << 8) | ((uint32_t)shift << 12);
}

void program(const Version& v, Part part, int operand_bits, int shift) {
  npu_dpi_csr_write(NPU_CSR_PTA_IMPAIR, kPartImpair[part]);
  npu_dpi_csr_write(NPU_CSR_PTA_BITS, bits_setting(v, part, operand_bits, shift));
  npu_dpi_csr_write(NPU_CSR_PTA_SIGMA_TH, v.sigma_th);
  npu_dpi_csr_write(NPU_CSR_PTA_SIGMA_SH, v.k_shot);
  npu_dpi_csr_write(NPU_CSR_PTA_SIGMA_PR, v.sigma_pr);
  npu_dpi_csr_write(NPU_CSR_PTA_DRIFT, 0);
  npu_dpi_csr_write(NPU_CSR_PTA_XTALK, v.xtalk);
  npu_dpi_csr_write(NPU_CSR_PTA_DRIFT_MAX, 0);
}

Measured measure(grxblasHandle_t h, int point_index) {
  Measured r;
  const size_t cn = (size_t)kM * kN;
  void *dA = nullptr, *dB = nullptr, *dC = nullptr;
  if (grxMalloc(&dA, (size_t)kM * kK) != grxSuccess ||
      grxMalloc(&dB, (size_t)kK * kN) != grxSuccess ||
      grxMalloc(&dC, cn * sizeof(int32_t)) != grxSuccess) {
    r.ok = false;
    return r;
  }
  const float one = 1.0f, zero = 0.0f;
  std::vector<int8_t> A, B;
  std::vector<int32_t> C(cn), exact;
  for (int g = 0; g < kGemms; ++g) {
    // A seed per GEMM, as a host that wants independent noise has to write one:
    // every per-GEMM stream reloads from PTA_SEED at a start.
    npu_dpi_csr_write(NPU_CSR_PTA_SEED, 0x51000000u + (uint32_t)point_index * 4096u + (uint32_t)g);
    operands((uint32_t)g + 1u, &A, &B);
    if (!reference(A, B, &exact)) r.ok = false;
    grxMemcpy(dA, A.data(), A.size(), grxMemcpyDefault);
    grxMemcpy(dB, B.data(), B.size(), grxMemcpyDefault);
    if (grxblasGemmEx(h, GRXBLAS_OP_N, GRXBLAS_OP_N, kM, kN, kK, &one,
                      dA, GRX_R_8I, kM, dB, GRX_R_8I, kK, &zero,
                      dC, GRX_R_32I, kM) != GRXBLAS_STATUS_SUCCESS)
      r.ok = false;
    grxDeviceSynchronize();
    grxMemcpy(C.data(), dC, cn * sizeof(int32_t), grxMemcpyDefault);
    r.sats += npu_dpi_csr_read(NPU_CSR_PTA_SAT_CT);
    for (size_t i = 0; i < cn; ++i) {
      const int64_t e = (int64_t)C[i] - (int64_t)exact[i];
      const uint64_t ae = (uint64_t)(e < 0 ? -e : e);
      r.sum_e2 += ae * ae;
      r.sum_c2 += (uint64_t)((int64_t)exact[i] * (int64_t)exact[i]);
      r.sum_e  += e;
      if (ae > r.max_abs_e) r.max_abs_e = (uint32_t)ae;
      r.moved  += (e != 0);
      ++r.count;
    }
  }
  grxFree(dA); grxFree(dB); grxFree(dC);
  return r;
}

// ---------------------------------------------------------------------------
// The record of what was measured last time
// ---------------------------------------------------------------------------

struct Recorded {
  std::string name;
  unsigned long long sum_e2 = 0, sum_c2 = 0;
  long long sum_e = 0;
  unsigned max_abs_e = 0, moved = 0, sats = 0, count = 0;
};

std::string point_name(const Version& v, Part p) {
  return std::string(v.name) + "-" + kPartName[p];
}

bool write_record(const char* path, const pta_tile& tile, const int shift[2],
                  const std::vector<Measured>& all) {
  FILE* f = std::fopen(path, "wb");
  if (!f) return false;
  std::fprintf(f,
      "# grxblasGemmEx on the photonic tile: the error against the product.\n"
      "#\n"
      "# Written by tests/libs/test_grxblas_pta_dist --regenerate. This is a\n"
      "# RECORD of what was measured, not a reference: the reference for these\n"
      "# numbers is the prediction the test computes, and the test fails when a\n"
      "# measurement leaves its band whatever this file says. What the file is\n"
      "# for is making a move VISIBLE -- when the error model is tuned, the\n"
      "# test prints every point that differs from the line below, and the\n"
      "# regenerated diff goes in the same change.\n"
      "#\n"
      "# Each point is %d GEMMs of %d x %d x %d, uniform int8 operands, a fresh\n"
      "# PTA_SEED per GEMM. sum_e2 and sum_c2 are the sums of the squared error\n"
      "# and of the squared exact product over every element, in product units;\n"
      "# the relative RMS error is sqrt(sum_e2 / sum_c2).\n"
      "version %d\n"
      "tile rows=%d cols=%d operand_bits=%d accumulator_bits=%d\n"
      "shape m=%d n=%d k=%d gemms=%d\n"
      "shift v0=%d v1=%d\n"
      "points %d\n",
      kGemms, kM, kN, kK, kRecordVersion, tile.rows, tile.cols, tile.din_w,
      tile.acc_w, kM, kN, kK, kGemms, shift[0], shift[1], (int)all.size());
  for (size_t i = 0; i < all.size(); ++i) {
    const Measured& r = all[i];
    std::fprintf(f, "point %s sum_e2=%llu sum_c2=%llu sum_e=%lld max_abs_e=%u "
                    "moved=%u sats=%u count=%u\n",
                 point_name(kVersions[i / P_COUNT], (Part)(i % P_COUNT)).c_str(),
                 (unsigned long long)r.sum_e2, (unsigned long long)r.sum_c2,
                 (long long)r.sum_e, r.max_abs_e, r.moved, r.sats, r.count);
  }
  std::fclose(f);
  return true;
}

bool read_record(const char* path, pta_tile* tile, int shift[2],
                 std::vector<Recorded>* out) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  char line[512];
  int version = 0, points = -1, m = 0, n = 0, k = 0, gemms = 0;
  bool ok = true;
  while (ok && std::fgets(line, sizeof(line), f)) {
    size_t len = std::strlen(line);
    while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
    if (!len || line[0] == '#') continue;
    char name[64];
    Recorded r;
    if (std::sscanf(line, "version %d", &version) == 1) continue;
    if (std::sscanf(line, "tile rows=%d cols=%d operand_bits=%d accumulator_bits=%d",
                    &tile->rows, &tile->cols, &tile->din_w, &tile->acc_w) == 4) continue;
    if (std::sscanf(line, "shape m=%d n=%d k=%d gemms=%d", &m, &n, &k, &gemms) == 4) continue;
    if (std::sscanf(line, "shift v0=%d v1=%d", &shift[0], &shift[1]) == 2) continue;
    if (std::sscanf(line, "points %d", &points) == 1) continue;
    if (std::sscanf(line, "point %63s sum_e2=%llu sum_c2=%llu sum_e=%lld max_abs_e=%u "
                          "moved=%u sats=%u count=%u",
                    name, &r.sum_e2, &r.sum_c2, &r.sum_e, &r.max_abs_e, &r.moved,
                    &r.sats, &r.count) == 8) {
      r.name = name;
      out->push_back(r);
      continue;
    }
    ok = false;
  }
  std::fclose(f);
  return ok && version == kRecordVersion && points == (int)out->size() &&
         m == kM && n == kN && k == kK && gemms == kGemms;
}

double snr_db(double rel) { return rel > 0 ? -20.0 * std::log10(rel) : 0.0; }
double eff_bits(double rel) { return (snr_db(rel) - 1.76) / 6.02; }

}  // namespace
#endif  // GRXCP_ENABLE_NPU

int main(int argc, char** argv) {
#ifndef GRXCP_ENABLE_NPU
  (void)argc; (void)argv;
  std::printf("built without GRXCP_ENABLE_NPU; no NPU backend here. skipping\n");
  return 77;
#else
  const bool regenerate = argc > 1 && !std::strcmp(argv[1], "--regenerate");
  const char* path = regenerate ? (argc > 2 ? argv[2] : kRecordDefault)
                                : (argc > 1 ? argv[1] : kRecordDefault);

  // BEFORE THE FIRST grx CALL.
  const bool installed = grxtest::npu_tile_install();
  section("the device");
  check(installed, "the register model was built with the tile and took it");
  if (!installed) return grxtest::report();

  int count = 0;
  GRX_REQUIRE(grxGetDeviceCount(&count), "grxGetDeviceCount");
  int npu = -1;
  for (int i = 0; i < count; ++i) {
    grxDeviceProp_t p{};
    if (grxGetDeviceProperties(&p, i) == grxSuccess &&
        p.deviceType == GRX_DEVICE_TYPE_NPU) { npu = i; break; }
  }
  check(npu >= 0, "an NPU is enumerated");
  if (npu < 0) return grxtest::report();
  GRX_REQUIRE(grxSetDevice(npu), "grxSetDevice");

  grxDeviceProp_t prop{};
  grxGetDeviceProperties(&prop, npu);
  const pta_tile tile = {prop.analogGemm.tileRows, prop.analogGemm.tileCols,
                         prop.analogGemm.operandBits,
                         prop.analogGemm.accumulatorBits};
  std::printf("  note  %s\n", prop.name);
  std::printf("  note  tile %d x %d, %d-bit operands, as the device reports it\n",
              tile.rows, tile.cols, tile.din_w);
  check(prop.analogGemm.tileIsPresent == 1 && tile.rows > 0 && tile.din_w >= 8,
        "it has a tile, and says what shape");
  if (prop.analogGemm.tileIsPresent != 1 || tile.rows <= 0 || tile.din_w < 8)
    return grxtest::report();

  grxblasHandle_t h = nullptr;
  if (grxblasCreate(&h) != GRXBLAS_STATUS_SUCCESS) {
    check(false, "grxblasCreate");
    return grxtest::report();
  }

  // ---- calibration --------------------------------------------------------
  section("calibration: the operands, and the ADC's shift");
  Moments mom[2];
  int shift[2] = {0, 0};
  for (int vi = 0; vi < 2; ++vi) {
    mom[vi] = calibrate(kVersions[vi], tile.rows);
    shift[vi] = mom[vi].shift;
    std::printf("  note  %s: E[a^2] %.1f  E[w^2] %.1f  E[da^2] %.3f  E[dw^2] %.3f  "
                "E|y_tile| %.1f\n", kVersions[vi].name, mom[vi].a2, mom[vi].w2,
                mom[vi].da2, mom[vi].dw2, mom[vi].abs_tile);
    std::printf("  note  %s: a %d-bit ADC needs a shift of %d to clip no more than "
                "1 sum in 10,000 (it clips %.5f%%)\n", kVersions[vi].name,
                kVersions[vi].adc_bits, shift[vi], 100.0 * mom[vi].clipped);
  }
  check(shift[0] > 0 && shift[1] > 0 && shift[0] >= shift[1],
        "a coarser ADC needs the larger shift");

  // ---- measurement --------------------------------------------------------
  std::vector<Measured> all;
  double predicted[2][P_COUNT];
  for (int vi = 0; vi < 2; ++vi) {
    predict(kVersions[vi], mom[vi], tile.rows, predicted[vi]);
    for (int p = 0; p < P_COUNT; ++p) {
      program(kVersions[vi], (Part)p, tile.din_w, shift[vi]);
      all.push_back(measure(h, vi * P_COUNT + p));
    }
  }
  npu_dpi_csr_write(NPU_CSR_PTA_IMPAIR, 0);

  if (regenerate) {
    if (!write_record(path, tile, shift, all)) {
      std::printf("could not write %s\n", path);
      return 1;
    }
    std::printf("\nrecorded %d points to %s.\nThis is a record of a measurement: "
                "review the diff, and commit it in the same change as whatever "
                "moved the model.\n", (int)all.size(), path);
    return 0;
  }

  // ---- the report ---------------------------------------------------------
  section("the report: error against the product, as a fraction of its RMS");
  bool every_ok = true, fp32_exact = true;
  for (const Measured& r : all) { every_ok = every_ok && r.count == (uint32_t)(kGemms * kM * kN); fp32_exact = fp32_exact && r.ok; }
  check(every_ok, "every point measured 12,288 elements");
  check(fp32_exact, "the fp32 reference is exact here: it equals the integer "
                    "product in every element, and every GEMM reported success");

  // "bias" is the mean error, as a fraction of the product's RMS. It is the
  // part of the error that averaging over shots would NOT remove, and it is
  // small everywhere here: these are noise and rounding, not offsets.
  std::printf("\n  %-16s %10s %10s %7s %8s %6s %8s %9s\n", "point", "measured",
              "predicted", "ratio", "SNR dB", "bits", "bias", "clipped");
  int out_of_band = 0;
  for (int vi = 0; vi < 2; ++vi) {
    const double rms_c = std::sqrt((double)all[vi * P_COUNT].sum_c2 /
                                   (double)all[vi * P_COUNT].count);
    for (int p = 0; p < P_COUNT; ++p) {
      const Measured& r = all[vi * P_COUNT + p];
      const double rel  = r.rel();
      const double pred = std::sqrt(predicted[vi][p]) / rms_c;
      const double ratio = pred > 0 ? rel / pred : 0.0;
      const bool in_band = ratio >= 1.0 - kBand && ratio <= 1.0 + kBand;
      if (!in_band) ++out_of_band;
      std::printf("  %-16s %9.3f%% %9.3f%% %7.3f %8.2f %6.2f %+7.3f%% %8.4f%%%s\n",
                  point_name(kVersions[vi], (Part)p).c_str(), 100.0 * rel,
                  100.0 * pred, ratio, snr_db(rel), eff_bits(rel),
                  100.0 * ((double)r.sum_e / (double)r.count) / rms_c,
                  100.0 * r.sats / ((double)r.count * (kK / tile.rows)),
                  in_band ? "" : "   OUT OF BAND");
    }
    std::printf("\n");
  }
  char what[200];
  std::snprintf(what, sizeof(what),
                "all twelve measurements are within %.0f%% of their predictions, "
                "above and below", 100.0 * kBand);
  check(out_of_band == 0, what);

  // ---- does it add? -------------------------------------------------------
  // Independent errors add in quadrature. If the five parts measured alone
  // account for the error measured with all five at once, GEMM-level error
  // adds; if "all" is well above their root-sum-square, it compounds.
  section("do the parts add?");
  for (int vi = 0; vi < 2; ++vi) {
    double ss = 0;
    for (int p = 0; p < P_ALL; ++p) ss += (double)all[vi * P_COUNT + p].sum_e2;
    const double joint = (double)all[vi * P_COUNT + P_ALL].sum_e2;
    const double ratio = std::sqrt(joint / ss);
    std::printf("  note  %s: all at once is %.3f of the root-sum-square of the "
                "five alone\n", kVersions[vi].name, ratio);
    std::snprintf(what, sizeof(what),
                  "%s: the parts account for the whole, to within the band",
                  kVersions[vi].name);
    check(ratio >= 1.0 - kBand && ratio <= 1.0 + kBand, what);
  }

  // ---- what moved ---------------------------------------------------------
  section("against the record");
  pta_tile rec_tile = {0, 0, 0, 0};
  int rec_shift[2] = {0, 0};
  std::vector<Recorded> rec;
  const bool have = read_record(path, &rec_tile, rec_shift, &rec);
  check(have && rec.size() == all.size(),
        "tests/libs/pta_gemm_distribution.txt reads, and is this suite's");
  if (!have || rec.size() != all.size()) {
    std::printf("        Looked for '%s'. Run from tests/libs, or pass the\n"
                "        path; --regenerate writes it.\n", path);
    return grxtest::report();
  }
  check(rec_tile.rows == tile.rows && rec_tile.cols == tile.cols &&
        rec_tile.din_w == tile.din_w && rec_tile.acc_w == tile.acc_w,
        "it was recorded on this tile");
  int moved_points = 0;
  for (size_t i = 0; i < all.size(); ++i) {
    const Measured& r = all[i];
    const Recorded& o = rec[i];
    const bool same = o.name == point_name(kVersions[i / P_COUNT], (Part)(i % P_COUNT)) &&
                      o.sum_e2 == r.sum_e2 && o.sum_c2 == r.sum_c2 &&
                      o.sum_e == r.sum_e && o.max_abs_e == r.max_abs_e &&
                      o.moved == r.moved && o.sats == r.sats && o.count == r.count;
    if (!same) {
      ++moved_points;
      const double was = o.sum_c2 ? std::sqrt((double)o.sum_e2 / (double)o.sum_c2) : 0.0;
      std::printf("  MOVED %-16s %.4f%% -> %.4f%%\n", o.name.c_str(), 100.0 * was,
                  100.0 * r.rel());
    }
  }
  if (moved_points || rec_shift[0] != shift[0] || rec_shift[1] != shift[1]) {
    // NOT a failure, and that is the design: this half is reported, and the
    // band above is what fails. But a move is never silent.
    std::printf("  note  %d of %d points differ from the record. That is what a "
                "tuned error model looks like;\n"
                "        --regenerate, review the diff, and commit it with the "
                "change that moved them.\n", moved_points, (int)all.size());
  } else {
    std::printf("  note  every point is exactly as recorded\n");
  }

  check(grxblasDestroy(h) == GRXBLAS_STATUS_SUCCESS, "grxblasDestroy");
  return grxtest::report();
#endif
}
