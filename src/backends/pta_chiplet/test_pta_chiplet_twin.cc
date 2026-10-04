// test_pta_chiplet_twin.cc -- the gate for the PTA chiplet's digital twin (X5).
//
// The board plan's gate for X5 is "grxcp's backend gates pass against it,
// bitwise against the model".  This is that, in three parts:
//
//   1. THE MAP.  Every behaviour docs/designs/pta_chiplet_regmap.md gives the
//      chiplet -- identity, the per-GEMM seed, 64-bit counters, the completion
//      contract behind a link, the calibration engine's words -- driven through
//      the twin's window and checked.
//   2. THE MODEL.  A GEMM through the twin is pta_gemm() called directly, bit
//      for bit, and a calibration through it is pta_cal_bank().  The reference
//      here is a second device this file owns and never shows the twin.
//   3. THE DRIVER.  npu_c930_read_analog(), the c930 backend's reader, pointed
//      at the twin through nothing but a change of base -- which is the map's
//      claim that one driver addresses both -- and then a GEMM reproduced from
//      what that driver read, and PTA_GEMM_CT, alone.
//
// NOTHING HERE IS HARDWARE.  The twin is a register file in front of an error
// model, and a pass says the map can be implemented as written and that this
// implementation is the model.  It says nothing about a chiplet, which does
// not exist, or about a photonic device.
//
// Built three more times with one of the twin's ablation switches each, this
// file has to FAIL: ci/build_mock.sh requires it.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "npu_c930.h"

extern "C" {
#include "pta_chiplet_twin.h"
#include "pta_tile_model.h"
}

namespace {

int g_fail = 0;
int g_checks = 0;

void check(const std::string& name, bool ok) {
  ++g_checks;
  if (!ok) ++g_fail;
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", name.c_str());
}

void section(const char* title) { std::printf("\n%s\n", title); }

// ---- grx930's gemm_seed(), written out a second time ------------------------
// So that a twin with the wrong function is caught by something that is not
// the twin.  t_seed_function() pins this copy to three values computed from
// grx930's own source.
uint64_t ref_splitmix64(uint64_t* s) {
  uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

uint32_t ref_gemm_seed(uint32_t base, uint32_t index) {
  uint64_t s = (static_cast<uint64_t>(base) << 32) | index;
  return static_cast<uint32_t>(ref_splitmix64(&s) >> 32);
}

// ---- a configuration, as the registers hold it ---------------------------------
struct Cfg {
  uint32_t impair = 0, act = 0, w = 0, adc = 0, shift = 0;
  uint32_t sth = 0, ksh = 0, spr = 0, dsig = 0, dlog2 = 0, dmax = 0, xt = 0;
  uint32_t trim_log2 = 0, trim_max = 0;
};

uint32_t bits_word(const Cfg& c) { return c.act | (c.w << 4) | (c.adc << 8) | (c.shift << 12); }

void program(pta_twin* t, const Cfg& c) {
  pta_twin_write32(t, PTA_TWIN_IMPAIR, c.impair);
  pta_twin_write32(t, PTA_TWIN_BITS, bits_word(c));
  pta_twin_write32(t, PTA_TWIN_SIGMA_TH, c.sth);
  pta_twin_write32(t, PTA_TWIN_SIGMA_SH, c.ksh);
  pta_twin_write32(t, PTA_TWIN_SIGMA_PR, c.spr);
  pta_twin_write32(t, PTA_TWIN_DRIFT, c.dsig | (c.dlog2 << 16));
  pta_twin_write32(t, PTA_TWIN_DRIFT_MAX, c.dmax);
  pta_twin_write32(t, PTA_TWIN_XTALK, c.xt);
  pta_twin_write32(t, PTA_TWIN_TRIM, c.trim_log2 | (c.trim_max << 16));
}

pta_cfg model_cfg(const Cfg& c, uint32_t seed) {
  pta_cfg m;
  std::memset(&m, 0, sizeof m);
  m.impair = c.impair;
  m.act_bits = c.act;
  m.w_bits = c.w;
  m.adc_bits = c.adc;
  m.adc_shift = c.shift;
  m.seed = seed;
  m.sigma_th = c.sth;
  m.k_shot = c.ksh;
  m.sigma_pr = c.spr;
  m.drift_sigma = c.dsig;
  m.drift_log2 = c.dlog2;
  m.drift_max = c.dmax;
  m.xtalk = c.xt;
  m.trim_step = 1u << c.trim_log2;
  m.trim_max = c.trim_max;
  return m;
}

// ---- the reference: a device the twin never sees ----------------------------------
struct Ref {
  pta_tile tile;
  pta_device dev;
  explicit Ref(const pta_twin_build& b) {
    tile.rows = b.rows;
    tile.cols = b.cols;
    tile.din_w = b.din_w;
    tile.acc_w = b.acc_w;
    pta_device_init(&dev, &tile);
    pta_model_reset(&dev, 0);
  }
  ~Ref() { pta_device_free(&dev); }
  Ref(const Ref&) = delete;
  Ref& operator=(const Ref&) = delete;

  long gemm(const Cfg& c, uint32_t base_seed, uint32_t index, int bank, int M, int N, int K,
            const std::vector<int32_t>& A, const std::vector<int32_t>& B, std::vector<int64_t>* C) {
    const pta_cfg m = model_cfg(c, ref_gemm_seed(base_seed, index));
    C->assign(static_cast<size_t>(M) * N, 0);
    return pta_gemm(&m, &tile, &dev, bank, M, N, K, A.data(), B.data(), C->data());
  }
};

// ---- operands ---------------------------------------------------------------------
std::vector<int32_t> operands(uint32_t seed, size_t n, int amax) {
  std::vector<int32_t> v(n);
  uint32_t s = seed * 2654435761u + 12345u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = static_cast<int32_t>((s >> 8) % static_cast<uint32_t>(2 * amax + 1)) - amax;
  }
  return v;
}

// ---- a GEMM through the twin, start to end -------------------------------------------
struct Run {
  int status = -99;
  long sats = -1;
  std::vector<int64_t> C;
};

const int64_t POISON = 0x5A5A5A5A5A5A5A5All;

int submit(pta_twin* t, int bank, int M, int N, int K, const std::vector<int32_t>& A,
           const std::vector<int32_t>& B, Run* r) {
  r->C.assign(static_cast<size_t>(M) * N, POISON);
  pta_twin_cmd c;
  c.bank = bank;
  c.M = M;
  c.N = N;
  c.K = K;
  c.A = A.data();
  c.B = B.data();
  c.C = r->C.data();
  c.status = &r->status;
  c.sats = &r->sats;
  return pta_twin_submit(t, &c);
}

bool busy(pta_twin* t) {
  return (pta_twin_read32(t, PTA_TWIN_STATUS) &
          (PTA_TWIN_STATUS_BUSY | PTA_TWIN_STATUS_CAL_BUSY)) != 0;
}

// Run until one read of PTA_STATUS shows neither BUSY nor CAL_BUSY: the map's
// own completion test.  Returns false if that never came.
bool drain(pta_twin* t) {
  for (int i = 0; i < 100000 && busy(t); ++i) pta_twin_run(t, 1u << 20);
  return !busy(t);
}

bool run_gemm(pta_twin* t, int bank, int M, int N, int K, const std::vector<int32_t>& A,
              const std::vector<int32_t>& B, Run* r) {
  if (submit(t, bank, M, N, K, A, B, r) != 0) return false;
  return drain(t);
}

bool all_poison(const std::vector<int64_t>& c) {
  for (int64_t v : c)
    if (v != POISON) return false;
  return true;
}

const pta_twin_build B4x4 = {4, 4, 16, 48, 4};       // the c930 register model's tile
const pta_twin_build B8x8 = {8, 8, 8, 48, 4};        // grx930's bench tile
const pta_twin_build B256x64 = {256, 64, 8, 48, 4};  // one of X2's candidates for the chiplet

int amax_of(const pta_twin_build& b) { return b.din_w >= 16 ? 127 : (1 << (b.din_w - 1)) - 1; }

// A quantised configuration that fits this tile: the quantiser keeps the top of
// the operand word, so six bits of an int8 on a sixteen-bit tile are fourteen.
Cfg base_cfg(const pta_twin_build& b) {
  Cfg c;
  c.impair = PTA_QUANT;
  c.act = static_cast<uint32_t>(b.din_w - 2);
  c.w = static_cast<uint32_t>(b.din_w - 2);
  c.adc = 7;
  // A K tile's sum is rows * amax^2 wide; leave the ADC's seven bits over it.
  int64_t span = static_cast<int64_t>(b.rows) * amax_of(b) * amax_of(b);
  uint32_t s = 0;
  while ((static_cast<int64_t>(63) << s) < span) ++s;
  c.shift = s;
  return c;
}

// ---------------------------------------------------------------------------------------
void t_seed_function() {
  section("the seed function is grx930's (pta_mnist.c, gemm_seed)");
  // Computed by compiling grx930's own splitmix64() and gemm_seed(), lifted
  // from c930/sim/pta_mnist.c as it stands on its main, with these arguments.
  check("gemm_seed(1, 0), (0xC0FFEE, 7) and (0, 0xFFFFFFFF) are the harness's",
        ref_gemm_seed(1u, 0u) == 3291240986u && ref_gemm_seed(0xC0FFEEu, 7u) == 1993064625u &&
            ref_gemm_seed(0u, 0xFFFFFFFFu) == 1940994978u);
  bool same = true;
  for (uint32_t i = 0; i < 64; ++i)
    for (uint32_t s : {0u, 1u, 0xC0FFEEu, 0xFFFFFFFFu})
      same = same && pta_twin_gemm_seed(s, i) == ref_gemm_seed(s, i);
  check("and the twin's is the same function", same);
  check("a calibration's seed is PTA_CAL_SEED ^ (index * 0x9E3779B1)",
        pta_twin_cal_seed(0xCA1u, 0) == 0xCA1u &&
            pta_twin_cal_seed(0xCA1u, 3) == (0xCA1u ^ (3u * 0x9E3779B1u)));
}

void t_identity() {
  section("identity and capability (the map's section 2)");
  for (const pta_twin_build* b : {&B4x4, &B8x8, &B256x64}) {
    pta_twin* t = pta_twin_new(b);
    const uint32_t c0 = pta_twin_read32(t, PTA_TWIN_CAPS0);
    const uint32_t c1 = pta_twin_read32(t, PTA_TWIN_CAPS1);
    const uint32_t c2 = pta_twin_read32(t, PTA_TWIN_CAPS2);
    char name[160];
    std::snprintf(name, sizeof name, "%3d x %-3d the magic, map version 1, and the build's own geometry",
                  b->rows, b->cols);
    check(name, pta_twin_read32(t, PTA_TWIN_ID) == 0x50544101u &&
                    (c0 & 0x3FFu) == static_cast<uint32_t>(b->rows) &&
                    ((c0 >> 10) & 0x3FFu) == static_cast<uint32_t>(b->cols) &&
                    ((c0 >> 20) & 0x3Fu) == static_cast<uint32_t>(b->din_w) &&
                    ((c0 >> 26) & 0x3Fu) == static_cast<uint32_t>(b->acc_w));
    const uint32_t qmax = b->din_w - 1 > 15 ? 15u : static_cast<uint32_t>(b->din_w - 1);
    std::snprintf(name, sizeof name, "%3d x %-3d every impairment but MZM_NL, two banks, widest bits %u",
                  b->rows, b->cols, qmax);
    check(name, (c1 & 0x7Fu) == 0x5Fu && ((c1 >> 8) & 0xFFu) == 2u &&
                    ((c1 >> 16) & 0xFu) == qmax && ((c1 >> 20) & 0xFu) == qmax &&
                    ((c1 >> 24) & 0xFu) == 15u);
    std::snprintf(name, sizeof name,
                  "%3d x %-3d a twin, of kind 3, with a calibration engine, no activation stage, no rate",
                  b->rows, b->cols);
    check(name, (c2 & 0x80000000u) != 0 && ((c2 >> 18) & 3u) == 3u && (c2 & 0x10000u) != 0 &&
                    (c2 & 0x20000u) == 0 && (c2 & 0xFFFFu) == 0);
    // Read-only means a write changes nothing.
    for (uint32_t off : {PTA_TWIN_ID, PTA_TWIN_CAPS0, PTA_TWIN_CAPS1, PTA_TWIN_CAPS2})
      pta_twin_write32(t, off, 0xFFFFFFFFu);
    std::snprintf(name, sizeof name, "%3d x %-3d and none of the four words can be written", b->rows, b->cols);
    check(name, pta_twin_read32(t, PTA_TWIN_ID) == 0x50544101u && pta_twin_read32(t, PTA_TWIN_CAPS0) == c0 &&
                    pta_twin_read32(t, PTA_TWIN_CAPS1) == c1 && pta_twin_read32(t, PTA_TWIN_CAPS2) == c2);
    pta_twin_free(t);
  }
  const pta_twin_build bad[] = {{0, 4, 8, 48, 4}, {4, 1024, 8, 48, 4}, {4, 4, 1, 48, 4},
                                {4, 4, 8, 64, 4}, {4, 4, 8, 48, 0}};
  bool refused = pta_twin_new(nullptr) == nullptr;
  for (const pta_twin_build& b : bad) refused = refused && pta_twin_new(&b) == nullptr;
  check("a build the capability words could not report is not built", refused);
}

// The c930 backend's reader, with the twin where the c930's block would be.
uint32_t c930_read(void* ctx, uint32_t off) {
  return off >= 0x100u ? pta_twin_read32(static_cast<pta_twin*>(ctx), off - 0x100u) : 0u;
}
void c930_write(void* ctx, uint32_t off, uint32_t v) {
  if (off >= 0x100u) pta_twin_write32(static_cast<pta_twin*>(ctx), off - 0x100u, v);
}

void t_one_driver() {
  section("one driver, two bases (the map's section 1)");
  pta_twin* t = pta_twin_new(&B256x64);
  npu_c930_device_t dev;
  npu_c930_attach_model(&dev, c930_read, c930_write, t);
  npu_c930_analog_t an;
  check("the c930 backend's reader identifies the twin through its own offsets",
        npu_c930_read_analog(&dev, &an) == 0 && an.identified == 1 && an.map_version == 1);
  check("and reports the tile it has: 256 x 64, 8-bit operands, 48-bit sums, implementing 0x5f",
        an.tile_present == 1 && an.tile_rows == 256 && an.tile_cols == 64 && an.operand_bits == 8 &&
            an.accumulator_bits == 48 && an.impairments_implemented == 0x5F);
  check("exact, with PTA_IMPAIR clear, and no configuration reported", an.analog == 0 && an.seed == -1);

  // Written through the c930's offsets too: 0x148 is the chiplet's 0x048.
  c930_write(t, NPU_C930_PTA_IMPAIR, PTA_QUANT | PTA_THERMAL | PTA_XTALK);
  c930_write(t, NPU_C930_PTA_BITS, 6u | (5u << 4) | (7u << 8) | (16u << 12));
  c930_write(t, NPU_C930_PTA_SEED, 0xD00Du);
  c930_write(t, NPU_C930_PTA_SIGMA_TH, 0x0123u);
  c930_write(t, NPU_C930_PTA_SIGMA_SH, 0x0045u);
  c930_write(t, NPU_C930_PTA_SIGMA_PR, 0x0067u);
  c930_write(t, NPU_C930_PTA_DRIFT_CFG, 55u | (9u << 16));
  c930_write(t, NPU_C930_PTA_DRIFT_MAX, 8643u);
  c930_write(t, NPU_C930_PTA_XTALK_CHI, 26u);
  npu_c930_read_analog(&dev, &an);
  check("impaired, and every field is what was written",
        an.analog == 1 && an.impairments == (PTA_QUANT | PTA_THERMAL | PTA_XTALK) &&
            an.activation_bits == 6 && an.weight_bits == 5 && an.adc_bits == 7 && an.adc_shift == 16 &&
            an.seed == 0xD00D && an.sigma_thermal_q8 == 0x123 && an.shot_k_q8 == 0x45 &&
            an.sigma_prog_q8 == 0x67 && an.drift_sigma_q8 == 55 && an.drift_log2_shots == 9 &&
            an.drift_clamp_q8 == 8643 && an.crosstalk_q8 == 26);
  check("the shipped loop order, and no calibration in force", an.loop_modes == 0 && an.calibration_valid == 0);

  // And the claim that matters: what the driver read, with PTA_GEMM_CT, gives
  // the device's answer.  The c930's property has no field for the count, and
  // on the chiplet a GEMM cannot be reproduced without it.
  const int M = 3, N = 70, K = 300;
  const auto A = operands(11, static_cast<size_t>(M) * K, 127);
  const auto B = operands(12, static_cast<size_t>(K) * N, 127);
  Run r0, r1;
  const uint32_t ct0 = pta_twin_read32(t, PTA_TWIN_GEMM_CT);
  bool ran = run_gemm(t, 0, M, N, K, A, B, &r0);
  const uint32_t ct1 = pta_twin_read32(t, PTA_TWIN_GEMM_CT);
  ran = ran && run_gemm(t, 0, M, N, K, A, B, &r1);

  pta_cfg m;
  std::memset(&m, 0, sizeof m);
  m.impair = static_cast<uint32_t>(an.impairments);
  m.act_bits = static_cast<uint32_t>(an.activation_bits);
  m.w_bits = static_cast<uint32_t>(an.weight_bits);
  m.adc_bits = static_cast<uint32_t>(an.adc_bits);
  m.adc_shift = static_cast<uint32_t>(an.adc_shift);
  m.sigma_th = static_cast<uint32_t>(an.sigma_thermal_q8);
  m.k_shot = static_cast<uint32_t>(an.shot_k_q8);
  m.sigma_pr = static_cast<uint32_t>(an.sigma_prog_q8);
  m.drift_sigma = static_cast<uint32_t>(an.drift_sigma_q8);
  m.drift_log2 = static_cast<uint32_t>(an.drift_log2_shots);
  m.drift_max = static_cast<uint32_t>(an.drift_clamp_q8);
  m.xtalk = static_cast<uint32_t>(an.crosstalk_q8);
  Ref ref(B256x64);
  ref.tile.rows = an.tile_rows;
  ref.tile.cols = an.tile_cols;
  ref.tile.din_w = an.operand_bits;
  ref.tile.acc_w = an.accumulator_bits;
  std::vector<int64_t> e0(static_cast<size_t>(M) * N), e1(e0.size()), raw(e0.size());
  m.seed = ref_gemm_seed(static_cast<uint32_t>(an.seed), ct0);
  pta_gemm(&m, &ref.tile, &ref.dev, 0, M, N, K, A.data(), B.data(), e0.data());
  m.seed = ref_gemm_seed(static_cast<uint32_t>(an.seed), ct1);
  pta_gemm(&m, &ref.tile, &ref.dev, 0, M, N, K, A.data(), B.data(), e1.data());
  check("two GEMMs reproduced from what the driver read and PTA_GEMM_CT, bit for bit",
        ran && r0.status == PTA_TWIN_DONE && r1.status == PTA_TWIN_DONE && ct0 == 0 && ct1 == 1 &&
            r0.C == e0 && r1.C == e1);
  Ref ref2(B256x64);
  m.seed = static_cast<uint32_t>(an.seed);
  pta_gemm(&m, &ref2.tile, &ref2.dev, 0, M, N, K, A.data(), B.data(), raw.data());
  check("and NOT from PTA_SEED alone: the seed a GEMM runs on is derived", r0.C != raw && r0.C != r1.C);
  pta_twin_free(t);
}

struct Case {
  const char* name;
  uint32_t impair;
};

const Case CASES[] = {
    {"QUANT", PTA_QUANT},
    {"THERMAL", PTA_QUANT | PTA_THERMAL},
    {"SHOT", PTA_QUANT | PTA_SHOT},
    {"PROG_ERR", PTA_QUANT | PTA_PROG_ERR},
    {"XTALK", PTA_QUANT | PTA_XTALK},
    {"DRIFT", PTA_QUANT | PTA_DRIFT},
    {"all six", PTA_TWIN_BUILT},
};

void t_bitwise() {
  section("a GEMM through the twin is the model, bit for bit");
  for (const pta_twin_build* b : {&B4x4, &B8x8, &B256x64}) {
    // Two N tiles and three K tiles, the last of each partial.
    const int M = 5, N = b->cols + b->cols / 2 + 1, K = 2 * b->rows + 1;
    const auto A = operands(static_cast<uint32_t>(b->rows), static_cast<size_t>(M) * K, amax_of(*b));
    const auto B = operands(static_cast<uint32_t>(b->cols) + 99, static_cast<size_t>(K) * N, amax_of(*b));
    const uint64_t tiles = 2 * 3, shots = tiles * M;
    char name[200];

    // Nothing enabled: the exact product.
    {
      pta_twin* t = pta_twin_new(b);
      Run r;
      bool ran = run_gemm(t, 0, M, N, K, A, B, &r);
      std::vector<int64_t> exact(static_cast<size_t>(M) * N, 0);
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
          for (int k = 0; k < K; ++k)
            exact[static_cast<size_t>(m) * N + n] += static_cast<int64_t>(A[static_cast<size_t>(m) * K + k]) *
                                                       B[static_cast<size_t>(k) * N + n];
      std::snprintf(name, sizeof name, "%3d x %-3d %-9s the exact product, computed here without the model",
                    b->rows, b->cols, "clear");
      check(name, ran && r.status == PTA_TWIN_DONE && r.C == exact && r.sats == 0);
      pta_twin_free(t);
    }

    for (const Case& cs : CASES) {
      pta_twin* t = pta_twin_new(b);
      Ref ref(*b);
      Cfg c = base_cfg(*b);
      c.impair = cs.impair;
      c.sth = 0x0180;  // 1.5 ADC LSB
      c.ksh = 0x0060;
      c.spr = 0x0200;  // 2 weight LSB
      c.dsig = 0x0300;
      c.dlog2 = 1;     // a step every other shot, so the drift moves inside one GEMM
      c.dmax = 8643;
      c.xt = 26;       // 10%
      const uint32_t seed = 0xA5A50000u + static_cast<uint32_t>(cs.impair);
      pta_twin_write32(t, PTA_TWIN_SEED, seed);
      program(t, c);

      // Twice on bank 0 and once on bank 1: drift is device state, and a
      // second GEMM is where a twin that reloaded it would show.
      bool ok = true, moved = false, counted = true;
      const bool draws = (cs.impair & (PTA_THERMAL | PTA_SHOT | PTA_PROG_ERR | PTA_DRIFT)) != 0;
      std::vector<int64_t> first;
      long total_sats = 0;
      for (uint32_t i = 0; i < 3; ++i) {
        const int bank = i == 2 ? 1 : 0;
        Run r;
        std::vector<int64_t> want;
        ok = ok && run_gemm(t, bank, M, N, K, A, B, &r);
        const long sats = ref.gemm(c, seed, i, bank, M, N, K, A, B, &want);
        ok = ok && r.status == PTA_TWIN_DONE && r.C == want && r.sats == sats;
        counted = counted && pta_twin_read32(t, PTA_TWIN_SAT_CT) == static_cast<uint32_t>(sats) &&
                  ((pta_twin_read32(t, PTA_TWIN_STATUS) & PTA_TWIN_STATUS_SAT) != 0) == (sats != 0) &&
                  pta_twin_read32(t, PTA_TWIN_GEMM_CT) == i + 1 &&
                  pta_twin_read32(t, PTA_TWIN_SHOT_CT) == (i + 1) * shots &&
                  pta_twin_read32(t, PTA_TWIN_WLOAD_CT) == (i + 1) * tiles;
        if (i == 0) first = r.C;
        if (i == 1) moved = r.C != first;
        total_sats += sats;
      }
      std::snprintf(name, sizeof name, "%3d x %-3d %-9s three GEMMs, %d x %d x %d, each equal to pta_gemm()",
                    b->rows, b->cols, cs.name, M, N, K);
      check(name, ok);
      std::snprintf(name, sizeof name, "%3d x %-3d %-9s and the counters are the model's: GEMMs, shots, programmings, saturations",
                    b->rows, b->cols, cs.name);
      check(name, counted);
      // Quantisation and crosstalk draw nothing, so there the same operands
      // give the same answer.  Everything else draws, and each GEMM draws its own.
      std::snprintf(name, sizeof name, draws
                        ? "%3d x %-3d %-9s and the second GEMM is not the first: each has a seed of its own"
                        : "%3d x %-3d %-9s and the second GEMM is the first again: nothing here draws noise",
                    b->rows, b->cols, cs.name);
      check(name, moved == draws);
      pta_twin_free(t);
    }
  }

  // A count that is never zero proves nothing about the counter.
  {
    pta_twin* t = pta_twin_new(&B8x8);
    Ref ref(B8x8);
    Cfg c = base_cfg(B8x8);
    c.shift = 4;  // far too small a range: the ADC clips
    program(t, c);
    const int M = 4, N = 8, K = 8;
    const auto A = operands(5, static_cast<size_t>(M) * K, 127);
    const auto B = operands(6, static_cast<size_t>(K) * N, 127);
    Run r;
    std::vector<int64_t> want;
    bool ran = run_gemm(t, 0, M, N, K, A, B, &r);
    const long sats = ref.gemm(c, 0, 0, 0, M, N, K, A, B, &want);
    check("a range too small saturates, and PTA_SAT_CT and STATUS.SAT say how often",
          ran && sats > 0 && r.sats == sats && r.C == want &&
              pta_twin_read32(t, PTA_TWIN_SAT_CT) == static_cast<uint32_t>(sats) &&
              (pta_twin_read32(t, PTA_TWIN_STATUS) & PTA_TWIN_STATUS_SAT) != 0);
    c.shift = base_cfg(B8x8).shift;
    program(t, c);
    ran = run_gemm(t, 0, M, N, K, A, B, &r);
    check("and the count restarts at the next GEMM",
          ran && pta_twin_read32(t, PTA_TWIN_SAT_CT) == 0 &&
              (pta_twin_read32(t, PTA_TWIN_STATUS) & PTA_TWIN_STATUS_SAT) == 0);
    pta_twin_free(t);
  }
}

// grx930's harness walks a network a GEMM at a time: one seed for the run, the
// GEMM's index mixed in, layer l on bank l & 1, a range of its own a layer, and
// one device from start to end.  Its walk, in miniature.
void t_harness_walk() {
  section("the accuracy harness's walk: one PTA_SEED, a GEMM counter, alternating banks");
  pta_twin* t = pta_twin_new(&B256x64);
  Ref ref(B256x64);
  const uint32_t seed = 20261004u;
  Cfg c = base_cfg(B256x64);
  c.impair = PTA_TWIN_BUILT;
  c.sth = 0x0100;
  c.ksh = 0x0030;
  c.spr = 0x0100;
  c.dsig = 55;
  c.dlog2 = 3;
  c.dmax = 8643;
  c.xt = 5;
  pta_twin_write32(t, PTA_TWIN_SEED, seed);
  // The harness resets the model on its run's seed before the first GEMM.
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_MODEL_RST);
  pta_model_reset(&ref.dev, seed);
  pta_cal_reset(&ref.dev);

  bool ok = true;
  uint32_t index = 0;
  for (int batch = 0; batch < 3 && ok; ++batch) {
    // Two layers: 784 -> 100 in tiles of at most 256 inputs and 8 outputs, as
    // the harness cuts them, then 100 -> 10.
    struct Layer { int in, out; uint32_t shift; };
    const Layer layers[] = {{784, 100, c.shift}, {100, 10, c.shift - 2}};
    const int M = 4;
    for (int l = 0; l < 2 && ok; ++l) {
      Cfg cl = c;
      cl.shift = layers[l].shift;
      for (int n0 = 0; n0 < layers[l].out && ok; n0 += 8) {
        const int N = layers[l].out - n0 < 8 ? layers[l].out - n0 : 8;
        for (int k0 = 0; k0 < layers[l].in && ok; k0 += 256) {
          const int K = layers[l].in - k0 < 256 ? layers[l].in - k0 : 256;
          const auto A = operands(index * 2 + 1, static_cast<size_t>(M) * K, 127);
          const auto B = operands(index * 2 + 2, static_cast<size_t>(K) * N, 127);
          program(t, cl);  // a layer's range is written before its GEMMs, as a driver would
          Run r;
          std::vector<int64_t> want;
          ok = run_gemm(t, l & 1, M, N, K, A, B, &r);
          const long sats = ref.gemm(cl, seed, index, l & 1, M, N, K, A, B, &want);
          ok = ok && r.status == PTA_TWIN_DONE && r.C == want && r.sats == sats;
          ++index;
        }
      }
    }
  }
  char name[160];
  std::snprintf(name, sizeof name, "%u GEMMs over three batches of a 784-100-10 walk, each equal to the model's", index);
  check(name, ok && index == 3 * (13 * 4 + 2) && pta_twin_read32(t, PTA_TWIN_GEMM_CT) == index);
  pta_twin_free(t);
}

void t_seed_contract() {
  section("PTA_SEED is written once, and PTA_GEMM_CT says which GEMM this is (section 4)");
  pta_twin* t = pta_twin_new(&B8x8);
  Cfg c = base_cfg(B8x8);
  c.impair = PTA_QUANT | PTA_THERMAL;
  c.sth = 0x0200;
  program(t, c);
  const int M = 4, N = 8, K = 8;
  const auto A = operands(1, static_cast<size_t>(M) * K, 127);
  const auto B = operands(2, static_cast<size_t>(K) * N, 127);
  Run a, b, again, other;
  pta_twin_write32(t, PTA_TWIN_SEED, 77);
  bool ran = run_gemm(t, 0, M, N, K, A, B, &a) && run_gemm(t, 0, M, N, K, A, B, &b);
  check("two GEMMs under one seed differ, and the count reads 2",
        ran && a.C != b.C && pta_twin_read32(t, PTA_TWIN_GEMM_CT) == 2);
  pta_twin_write32(t, PTA_TWIN_SEED, 77);
  check("writing PTA_SEED restarts the count", pta_twin_read32(t, PTA_TWIN_GEMM_CT) == 0);
  ran = run_gemm(t, 0, M, N, K, A, B, &again);
  check("and the run is reproduced: GEMM 0 again", ran && again.C == a.C);
  pta_twin_write32(t, PTA_TWIN_SEED, 78);
  ran = run_gemm(t, 0, M, N, K, A, B, &other);
  check("another seed is another run", ran && other.C != a.C);
  pta_twin_free(t);
}

void t_completion() {
  section("completion, behind a link (section 5)");
  pta_twin* t = pta_twin_new(&B8x8);
  Ref ref(B8x8);
  Cfg c = base_cfg(B8x8);
  program(t, c);
  pta_twin_write32(t, PTA_TWIN_TW, 3);
  pta_twin_write32(t, PTA_TWIN_TS, 5);
  const int M = 4, N = 12, K = 20;  // two N tiles, three K tiles
  const uint64_t tiles = 6, shots = 24, cycles = tiles * 3 + shots * 5;
  const auto A = operands(3, static_cast<size_t>(M) * K, 127);
  const auto B = operands(4, static_cast<size_t>(K) * N, 127);
  Run r;
  check("idle: one read of PTA_STATUS shows neither BUSY nor CAL_BUSY", !busy(t) && pta_twin_pending(t) == 0);
  const int rc = submit(t, 0, M, N, K, A, B, &r);
  const uint32_t st = pta_twin_read32(t, PTA_TWIN_STATUS);
  check("a command is BUSY from the moment it is accepted, in the same word as CAL_BUSY",
        rc == 0 && (st & PTA_TWIN_STATUS_BUSY) != 0 && (st & PTA_TWIN_STATUS_CAL_BUSY) == 0 &&
            r.status == PTA_TWIN_PENDING && pta_twin_pending(t) == 1);
  pta_twin_run(t, cycles - 1);
  check("it holds the tile for programmings * PTA_TW + shots * PTA_TS cycles, and delivers nothing early",
        busy(t) && r.status == PTA_TWIN_PENDING && all_poison(r.C));
  pta_twin_run(t, 1);
  std::vector<int64_t> want;
  ref.gemm(c, 0, 0, 0, M, N, K, A, B, &want);
  check("and at the last of them BUSY clears and the result is there",
        !busy(t) && r.status == PTA_TWIN_DONE && r.C == want && pta_twin_now(t) == cycles &&
            pta_twin_pending(t) == 0);

  // Configuration is sampled when a GEMM starts, so a write behind a running
  // command reaches the next one and not it.
  Cfg c2 = c;
  c2.impair = PTA_QUANT | PTA_THERMAL;
  c2.sth = 0x0400;
  Run r1, r2;
  submit(t, 0, M, N, K, A, B, &r1);
  program(t, c2);
  submit(t, 0, M, N, K, A, B, &r2);
  const bool drained = drain(t);
  std::vector<int64_t> w1, w2;
  ref.gemm(c, 0, 1, 0, M, N, K, A, B, &w1);
  ref.gemm(c2, 0, 2, 0, M, N, K, A, B, &w2);
  check("a register written behind a running command takes effect at the next GEMM start",
        drained && r1.C == w1 && r2.C == w2 && w1 != w2);

  // The queue: depth 4 behind the running command, and the fifth is not taken.
  program(t, c);
  std::vector<Run> runs(6);
  int taken = 0;
  for (int i = 0; i < 6; ++i)
    if (submit(t, 0, M, N, K, A, B, &runs[static_cast<size_t>(i)]) == 0) ++taken;
  check("one running and four queued are accepted; the sixth is not, and its status is not written",
        taken == 5 && pta_twin_pending(t) == 5 && runs[5].status == -99 && all_poison(runs[5].C));
  const bool d2 = drain(t);
  bool in_order = d2;
  for (uint32_t i = 0; i < 5; ++i) {
    std::vector<int64_t> w;
    ref.gemm(c, 0, 3 + i, 0, M, N, K, A, B, &w);
    in_order = in_order && runs[i].status == PTA_TWIN_DONE && runs[i].C == w;
  }
  check("and the five run in the order they were accepted", in_order);
  pta_twin_free(t);
}

// The calibration the map describes, on the reference device.
struct RefCal {
  int64_t left = -1, found = 0;
  long sats = 0;
  int clamped = 0;
};

RefCal ref_calibrate(Ref* ref, const Cfg& c, uint32_t cal_seed, uint32_t index, uint32_t amp,
                     uint32_t reps_log2, uint32_t passes, int bank) {
  RefCal out;
  const pta_cfg m = model_cfg(c, 0);
  const pta_cal_cfg cc = {amp, reps_log2, passes};
  pta_streams st;
  pta_start(&st, cal_seed ^ (index * 0x9E3779B1u));
  out.left = pta_cal_bank(&ref->dev, &m, &ref->tile, bank, &st, &cc, &out.sats, &out.clamped, &out.found);
  return out;
}

uint32_t cal_cfg_word(uint32_t amp, uint32_t reps_log2, uint32_t passes, int bank) {
  return amp | (reps_log2 << 4) | (passes << 8) | (static_cast<uint32_t>(bank) << 10);
}

// A tile that has drifted and been programmed with error: something to correct.
Cfg drifting_cfg() {
  Cfg c = base_cfg(B8x8);
  c.impair = PTA_QUANT | PTA_PROG_ERR | PTA_DRIFT;
  c.spr = 0x0100;
  c.dsig = 0x0040;
  c.dlog2 = 0;  // a step every shot
  c.dmax = 8643;
  c.trim_log2 = 2;
  c.trim_max = 0x4000;
  return c;
}

void t_calibration() {
  section("calibration is the model's engine, behind PTA_CTRL.CAL_NOW (sections 3 and 4)");
  pta_twin* t = pta_twin_new(&B8x8);
  Ref ref(B8x8);
  const Cfg c = drifting_cfg();
  const uint32_t seed = 0xBEEF, cal_seed = 0xCA1B;
  const int M = 32, N = 8, K = 8;
  const auto A = operands(21, static_cast<size_t>(M) * K, 127);
  const auto B = operands(22, static_cast<size_t>(K) * N, 127);
  pta_twin_write32(t, PTA_TWIN_SEED, seed);
  pta_twin_write32(t, PTA_TWIN_CAL_SEED, cal_seed);
  program(t, c);
  pta_twin_write32(t, PTA_TWIN_TW, 2);
  pta_twin_write32(t, PTA_TWIN_TS, 1);

  Run r;
  std::vector<int64_t> want;
  bool ran = run_gemm(t, 0, M, N, K, A, B, &r);   // 32 shots of drift
  ref.gemm(c, seed, 0, 0, M, N, K, A, B, &want);
  const uint32_t shots0 = pta_twin_read32(t, PTA_TWIN_SHOT_CT);
  const uint32_t wload0 = pta_twin_read32(t, PTA_TWIN_WLOAD_CT);

  // Without EN the engine is not listening.
  pta_twin_write32(t, PTA_TWIN_CAL_CFG, cal_cfg_word(6, 2, 3, 0));
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_CAL_NOW);
  pta_twin_run(t, 1000);
  check("CAL_NOW with EN clear does nothing",
        ran && r.C == want && !busy(t) && pta_twin_read32(t, PTA_TWIN_CAL_CT) == 0 &&
            pta_twin_read32(t, PTA_TWIN_IRQ_STATUS) == 0);

  // Amplitude 6 = DIN_W - 2, four repeats, three auto-ranging passes, bank 0.
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_EN | PTA_TWIN_CTRL_CAL_NOW);
  const uint32_t during = pta_twin_read32(t, PTA_TWIN_STATUS);
  check("with EN set it takes the tile: CAL_BUSY, and not BUSY -- a calibration is not a command",
        (during & PTA_TWIN_STATUS_CAL_BUSY) != 0 && (during & PTA_TWIN_STATUS_BUSY) == 0 &&
            (during & PTA_TWIN_STATUS_CAL_VALID) == 0 && pta_twin_read32(t, PTA_TWIN_CTRL) == PTA_TWIN_CTRL_EN);
  const uint64_t cycles = 3ull * 4 * (2 + 8 * 1);   // passes * repeats * (TW + rows * TS)
  pta_twin_run(t, cycles - 1);
  const bool still = (pta_twin_read32(t, PTA_TWIN_STATUS) & PTA_TWIN_STATUS_CAL_BUSY) != 0 &&
                     pta_twin_read32(t, PTA_TWIN_CAL_CT) == 0;
  pta_twin_run(t, 1);
  const RefCal rc = ref_calibrate(&ref, c, cal_seed, 0, 6, 2, 3, 0);
  const uint32_t after = pta_twin_read32(t, PTA_TWIN_STATUS);
  check("for passes * repeats * (PTA_TW + rows * PTA_TS) cycles, which PTA_CAL_CYC then reads",
        still && (after & PTA_TWIN_STATUS_CAL_BUSY) == 0 && pta_twin_read32(t, PTA_TWIN_CAL_CYC) == cycles);
  check("it found what pta_cal_bank() finds, and left what it leaves: PTA_ERR_FOUND, PTA_ERR_MAX, the residual",
        rc.left >= 0 && rc.found > 0 && rc.clamped == 0 &&
            pta_twin_read32(t, PTA_TWIN_ERR_FOUND) == static_cast<uint32_t>(rc.found) &&
            pta_twin_read32(t, PTA_TWIN_ERR_MAX) == static_cast<uint32_t>(rc.left) &&
            ((after >> 8) & 0xFFFFu) == (static_cast<uint32_t>(rc.left) & 0xFFFFu));
  check("CAL_VALID, PTA_CAL_CT = 1, and PTA_IRQ_STATUS.CAL_DONE",
        (after & PTA_TWIN_STATUS_CAL_VALID) != 0 && pta_twin_read32(t, PTA_TWIN_CAL_CT) == 1 &&
            (pta_twin_read32(t, PTA_TWIN_IRQ_STATUS) & PTA_TWIN_IRQ_CAL_DONE) != 0);
  check("its probe shots are shots, and its weight writes are not a GEMM's programmings",
        pta_twin_read32(t, PTA_TWIN_SHOT_CT) == shots0 + 3 * 4 * 8 &&
            pta_twin_read32(t, PTA_TWIN_WLOAD_CT) == wload0);

  Run g1;
  std::vector<int64_t> w1, w1_untrimmed;
  ran = run_gemm(t, 0, M, N, K, A, B, &g1);
  ref.gemm(c, seed, 1, 0, M, N, K, A, B, &w1);
  {
    // The same GEMM on a device that drifted and was never calibrated.
    Ref plain(B8x8);
    std::vector<int64_t> scratch;
    plain.gemm(c, seed, 0, 0, M, N, K, A, B, &scratch);
    plain.gemm(c, seed, 1, 0, M, N, K, A, B, &w1_untrimmed);
  }
  check("the GEMM after it runs on the trims the reference wrote, bit for bit",
        ran && g1.C == w1 && w1 != w1_untrimmed);

  // The second calibration draws from PTA_CAL_SEED ^ (1 * 0x9E3779B1).
  pta_twin_write32(t, PTA_TWIN_IRQ_STATUS, PTA_TWIN_IRQ_CAL_DONE);
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_EN | PTA_TWIN_CTRL_CAL_NOW);
  const bool d = drain(t);
  Ref wrong(B8x8);   // a device that reused the first calibration's seed
  {
    std::vector<int64_t> scratch;
    wrong.gemm(c, seed, 0, 0, M, N, K, A, B, &scratch);
    ref_calibrate(&wrong, c, cal_seed, 0, 6, 2, 3, 0);
    wrong.gemm(c, seed, 1, 0, M, N, K, A, B, &scratch);
    ref_calibrate(&wrong, c, cal_seed, 0, 6, 2, 3, 0);
  }
  const RefCal rc2 = ref_calibrate(&ref, c, cal_seed, 1, 6, 2, 3, 0);
  Run g2;
  std::vector<int64_t> w2, w2_wrong;
  ran = run_gemm(t, 0, M, N, K, A, B, &g2);
  ref.gemm(c, seed, 2, 0, M, N, K, A, B, &w2);
  wrong.gemm(c, seed, 2, 0, M, N, K, A, B, &w2_wrong);
  check("the second calibration draws its own noise: its seed carries PTA_CAL_CT",
        d && ran && pta_twin_read32(t, PTA_TWIN_CAL_CT) == 2 &&
            pta_twin_read32(t, PTA_TWIN_ERR_FOUND) == static_cast<uint32_t>(rc2.found) && g2.C == w2 &&
            w2 != w2_wrong);

  // The other bank is a different store.
  pta_twin_write32(t, PTA_TWIN_CAL_CFG, cal_cfg_word(6, 2, 3, 1));
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_EN | PTA_TWIN_CTRL_CAL_NOW);
  const bool d3 = drain(t);
  ref_calibrate(&ref, c, cal_seed, 2, 6, 2, 3, 1);
  Run g3;
  std::vector<int64_t> w3;
  ran = run_gemm(t, 1, M, N, K, A, B, &g3);
  ref.gemm(c, seed, 3, 1, M, N, K, A, B, &w3);
  check("PTA_CAL_CFG[10] picks the bank", d3 && ran && g3.C == w3);

  // A probe the estimator cannot read back: amplitude 7 is past DIN_W - 2.
  pta_twin_write32(t, PTA_TWIN_IRQ_STATUS, 0xF);
  pta_twin_write32(t, PTA_TWIN_CAL_CFG, cal_cfg_word(7, 2, 3, 0));
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_EN | PTA_TWIN_CTRL_CAL_NOW);
  const bool d4 = drain(t);
  const RefCal bad = ref_calibrate(&ref, c, cal_seed, 3, 7, 2, 3, 0);
  const uint32_t irq = pta_twin_read32(t, PTA_TWIN_IRQ_STATUS);
  check("a probe amplitude past DIN_W - 2 is refused: IRQ ERR, no CAL_VALID, and PTA_CAL_CT unmoved",
        d4 && bad.left < 0 && (irq & PTA_TWIN_IRQ_ERR) != 0 && (irq & PTA_TWIN_IRQ_CAL_DONE) != 0 &&
            (pta_twin_read32(t, PTA_TWIN_STATUS) & PTA_TWIN_STATUS_CAL_VALID) == 0 &&
            pta_twin_read32(t, PTA_TWIN_CAL_CT) == 3);
  Run g4;
  std::vector<int64_t> w4;
  ran = run_gemm(t, 0, M, N, K, A, B, &g4);
  ref.gemm(c, seed, 4, 0, M, N, K, A, B, &w4);
  check("and it wrote no trim: the next GEMM is the reference's", ran && g4.C == w4);

  // A DAC that cannot hold what the estimator asks for.
  Cfg tight = c;
  tight.trim_max = 4;
  program(t, tight);
  pta_twin_write32(t, PTA_TWIN_IRQ_STATUS, 0xF);
  pta_twin_write32(t, PTA_TWIN_CAL_CFG, cal_cfg_word(6, 2, 3, 0));
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_EN | PTA_TWIN_CTRL_CAL_NOW);
  const bool d5 = drain(t);
  const RefCal clamp = ref_calibrate(&ref, tight, cal_seed, 3, 6, 2, 3, 0);
  check("a trim that clamps is DRIFT_ALARM, in PTA_STATUS and as an interrupt",
        d5 && clamp.clamped == 1 && (pta_twin_read32(t, PTA_TWIN_STATUS) & PTA_TWIN_STATUS_DRIFT_ALARM) != 0 &&
            (pta_twin_read32(t, PTA_TWIN_IRQ_STATUS) & PTA_TWIN_IRQ_DRIFT_ALARM) != 0);
  pta_twin_free(t);
}

void t_cal_queue() {
  section("commands during a calibration queue; they are not lost and not taken (section 5)");
  pta_twin* t = pta_twin_new(&B8x8);
  Ref ref(B8x8);
  const Cfg c = drifting_cfg();
  const uint32_t seed = 0x51DE, cal_seed = 0xCA1C;
  const int M = 6, N = 8, K = 8;
  const auto A = operands(31, static_cast<size_t>(M) * K, 127);
  const auto B = operands(32, static_cast<size_t>(K) * N, 127);
  pta_twin_write32(t, PTA_TWIN_SEED, seed);
  pta_twin_write32(t, PTA_TWIN_CAL_SEED, cal_seed);
  program(t, c);
  pta_twin_write32(t, PTA_TWIN_TS, 1);
  pta_twin_write32(t, PTA_TWIN_CAL_CFG, cal_cfg_word(6, 1, 2, 0));
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_EN | PTA_TWIN_CTRL_CAL_NOW);
  const uint64_t cal_cycles = 2ull * 2 * 8;

  // Three, back to back, with the occupancy checked at each step: the CPU
  // document's regression, moved to the chiplet's own queue.
  Run r[3];
  bool stepwise = true;
  for (int i = 0; i < 3; ++i) {
    const int rc = submit(t, 0, M, N, K, A, B, &r[i]);
    const uint32_t st = pta_twin_read32(t, PTA_TWIN_STATUS);
    stepwise = stepwise && rc == 0 && pta_twin_pending(t) == i + 1 && r[i].status == PTA_TWIN_PENDING &&
               (st & PTA_TWIN_STATUS_CAL_BUSY) != 0 && (st & PTA_TWIN_STATUS_BUSY) != 0;
  }
  check("three commands during CAL_BUSY: occupancy 1, 2, 3, each PENDING, and BUSY set for them", stepwise);
  pta_twin_run(t, cal_cycles - 1);
  check("nothing is dispatched into the calibrating tile, and nothing is delivered",
        (pta_twin_read32(t, PTA_TWIN_STATUS) & PTA_TWIN_STATUS_CAL_BUSY) != 0 && pta_twin_pending(t) == 3 &&
            all_poison(r[0].C) && all_poison(r[1].C) && all_poison(r[2].C) &&
            pta_twin_read32(t, PTA_TWIN_GEMM_CT) == 0);
  pta_twin_run(t, 1);
  const uint32_t st = pta_twin_read32(t, PTA_TWIN_STATUS);
  check("when it ends the first of them takes the tile: CAL_BUSY clear, BUSY set, never \"finished\"",
        (st & PTA_TWIN_STATUS_CAL_BUSY) == 0 && (st & PTA_TWIN_STATUS_BUSY) != 0 && pta_twin_pending(t) == 3);
  const bool d = drain(t);
  ref_calibrate(&ref, c, cal_seed, 0, 6, 1, 2, 0);
  bool ok = d;
  for (uint32_t i = 0; i < 3; ++i) {
    std::vector<int64_t> w;
    ref.gemm(c, seed, i, 0, M, N, K, A, B, &w);
    ok = ok && r[i].status == PTA_TWIN_DONE && r[i].C == w;
  }
  check("and all three run after it, in order, on the tile it left", ok);

  // A calibration asked for behind a running command waits for it, and goes
  // ahead of whatever is queued.
  Run a, b;
  submit(t, 0, M, N, K, A, B, &a);
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_EN | PTA_TWIN_CTRL_CAL_NOW);
  submit(t, 0, M, N, K, A, B, &b);
  const uint32_t mid = pta_twin_read32(t, PTA_TWIN_STATUS);
  const bool d2 = drain(t);
  std::vector<int64_t> wa, wb;
  ref.gemm(c, seed, 3, 0, M, N, K, A, B, &wa);
  ref_calibrate(&ref, c, cal_seed, 1, 6, 1, 2, 0);
  ref.gemm(c, seed, 4, 0, M, N, K, A, B, &wb);
  check("a CAL_NOW behind a running command waits for it, then runs before the queue",
        (mid & PTA_TWIN_STATUS_CAL_BUSY) == 0 && (mid & PTA_TWIN_STATUS_BUSY) != 0 && d2 && a.C == wa &&
            b.C == wb && pta_twin_read32(t, PTA_TWIN_CAL_CT) == 2);
  pta_twin_free(t);
}

void t_model_rst() {
  section("MODEL_RST is refused while anything holds the tile, and says so (section 5)");
  pta_twin* t = pta_twin_new(&B8x8);
  Ref ref(B8x8);
  Cfg c = drifting_cfg();
  const uint32_t seed = 0x7E57;
  const int M = 16, N = 8, K = 8;
  const auto A = operands(41, static_cast<size_t>(M) * K, 127);
  const auto B = operands(42, static_cast<size_t>(K) * N, 127);
  pta_twin_write32(t, PTA_TWIN_SEED, seed);
  program(t, c);
  pta_twin_write32(t, PTA_TWIN_TS, 1);
  pta_twin_write32(t, PTA_TWIN_GAIN0 + 4, 300);

  Run r0, r1, r2;
  std::vector<int64_t> w;
  pta_column_cal(&ref.dev, 1, 300, 0);
  run_gemm(t, 0, M, N, K, A, B, &r0);
  ref.gemm(c, seed, 0, 0, M, N, K, A, B, &w);

  submit(t, 0, M, N, K, A, B, &r1);
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_MODEL_RST);
  check("written while BUSY, it raises PTA_IRQ_STATUS.ERR",
        (pta_twin_read32(t, PTA_TWIN_IRQ_STATUS) & PTA_TWIN_IRQ_ERR) != 0);
  drain(t);
  ref.gemm(c, seed, 1, 0, M, N, K, A, B, &w);
  bool same = r1.C == w;
  run_gemm(t, 0, M, N, K, A, B, &r2);
  ref.gemm(c, seed, 2, 0, M, N, K, A, B, &w);
  check("and leaves the device as it was: the drift and the correction are still there",
        same && r2.C == w && pta_twin_read32(t, PTA_TWIN_GAIN0 + 4) == 300);

  pta_twin_write32(t, PTA_TWIN_IRQ_STATUS, 0xF);
  pta_twin_write32(t, PTA_TWIN_CAL_CFG, cal_cfg_word(6, 1, 1, 0));
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_EN | PTA_TWIN_CTRL_CAL_NOW);
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_EN | PTA_TWIN_CTRL_MODEL_RST);
  check("written during a calibration, the same",
        (pta_twin_read32(t, PTA_TWIN_IRQ_STATUS) & PTA_TWIN_IRQ_ERR) != 0);
  drain(t);
  ref_calibrate(&ref, c, 0, 0, 6, 1, 1, 0);
  Run r3;
  run_gemm(t, 0, M, N, K, A, B, &r3);
  ref.gemm(c, seed, 3, 0, M, N, K, A, B, &w);
  check("and the calibration it was written under still holds", r3.C == w &&
        (pta_twin_read32(t, PTA_TWIN_STATUS) & PTA_TWIN_STATUS_CAL_VALID) != 0);

  pta_twin_write32(t, PTA_TWIN_IRQ_STATUS, 0xF);
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_MODEL_RST);
  pta_model_reset(&ref.dev, seed);
  pta_cal_reset(&ref.dev);
  Run r4;
  run_gemm(t, 0, M, N, K, A, B, &r4);
  ref.gemm(c, seed, 4, 0, M, N, K, A, B, &w);
  check("idle, it is honoured: drift to zero on PTA_SEED, trims and gains cleared, CAL_VALID clear, no ERR",
        r4.C == w && pta_twin_read32(t, PTA_TWIN_IRQ_STATUS) == 0 &&
            pta_twin_read32(t, PTA_TWIN_GAIN0 + 4) == 256 &&
            (pta_twin_read32(t, PTA_TWIN_STATUS) & PTA_TWIN_STATUS_CAL_VALID) == 0);
  check("and it does not restart PTA_GEMM_CT, which only a PTA_SEED write does",
        pta_twin_read32(t, PTA_TWIN_GEMM_CT) == 5);
  pta_twin_free(t);
}

void t_refusals() {
  section("a command the tile cannot honour is refused, not dropped (section 5)");
  pta_twin* t = pta_twin_new(&B8x8);
  Ref ref(B8x8);
  Cfg c = base_cfg(B8x8);
  const int M = 4, N = 8, K = 8;
  const auto A = operands(51, static_cast<size_t>(M) * K, 127);
  const auto B = operands(52, static_cast<size_t>(K) * N, 127);
  program(t, c);
  Run good;
  run_gemm(t, 0, M, N, K, A, B, &good);
  const uint32_t shots = pta_twin_read32(t, PTA_TWIN_SHOT_CT);

  struct Bad { const char* what; uint32_t impair; uint32_t shift; int bank; };
  const Bad bads[] = {
      {"MZM_NL, which no tile builds", PTA_QUANT | PTA_MZM_NL, c.shift, 0},
      {"an ADC shift of 41", PTA_QUANT, 41, 0},
      {"bank 2", PTA_QUANT, c.shift, 2},
  };
  for (const Bad& b : bads) {
    Cfg bc = c;
    bc.impair = b.impair;
    bc.shift = b.shift;
    program(t, bc);
    pta_twin_write32(t, PTA_TWIN_IRQ_STATUS, 0xF);
    Run r;
    const int rc = submit(t, b.bank, M, N, K, A, B, &r);
    drain(t);
    char name[160];
    std::snprintf(name, sizeof name, "%-28s REFUSED, IRQ ERR, C untouched, nothing counted", b.what);
    check(name, rc == 0 && r.status == PTA_TWIN_REFUSED && all_poison(r.C) &&
                    (pta_twin_read32(t, PTA_TWIN_IRQ_STATUS) & PTA_TWIN_IRQ_ERR) != 0 &&
                    pta_twin_read32(t, PTA_TWIN_GEMM_CT) == 1 && pta_twin_read32(t, PTA_TWIN_SHOT_CT) == shots &&
                    !busy(t));
  }
  // The register that asked for it reads back what was asked.
  Cfg bc = c;
  bc.impair = PTA_QUANT | PTA_MZM_NL;
  program(t, bc);
  check("PTA_IMPAIR still reads what was written: the refusal is the tile's, not the register's",
        pta_twin_read32(t, PTA_TWIN_IMPAIR) == (PTA_QUANT | PTA_MZM_NL));

  // A refused command does not hold the tile, so the one behind it runs.
  Run bad, ok;
  submit(t, 0, M, N, K, A, B, &bad);
  program(t, c);
  submit(t, 0, M, N, K, A, B, &ok);
  drain(t);
  std::vector<int64_t> w;
  ref.gemm(c, 0, 0, 0, M, N, K, A, B, &w);   // the reference's GEMM 0, to keep it in step
  ref.gemm(c, 0, 1, 0, M, N, K, A, B, &w);
  check("and the command behind a refused one runs, as GEMM 1",
        bad.status == PTA_TWIN_REFUSED && ok.status == PTA_TWIN_DONE && ok.C == w);

  // Not accepted is a different answer from refused.
  Run r;
  pta_twin_cmd cmd;
  r.C.assign(static_cast<size_t>(M) * N, POISON);
  cmd.bank = 0;
  cmd.M = 0;
  cmd.N = N;
  cmd.K = K;
  cmd.A = A.data();
  cmd.B = B.data();
  cmd.C = r.C.data();
  cmd.status = &r.status;
  cmd.sats = nullptr;
  const int rc0 = pta_twin_submit(t, &cmd);
  cmd.M = M;
  cmd.C = nullptr;
  const int rc1 = pta_twin_submit(t, &cmd);
  check("a command with no rows, or nowhere to put C, is not accepted and its status is not written",
        rc0 == -1 && rc1 == -1 && r.status == -99 && pta_twin_pending(t) == 0);
  pta_twin_free(t);
}

void t_counters() {
  section("counters are 64 bits, and the upper half is latched by reading the lower (section 4)");
  pta_twin* t = pta_twin_new(&B8x8);
  Cfg c;           // nothing enabled: one pass, exact
  c.trim_max = 0x100;
  program(t, c);
  pta_twin_write32(t, PTA_TWIN_CAL_CFG, cal_cfg_word(6, 0, 1, 0));
  pta_twin_write32(t, PTA_TWIN_TS, 0x40000000u);   // 8 rows * 2^30 = 2^33 cycles a calibration
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_EN | PTA_TWIN_CTRL_CAL_NOW);
  pta_twin_run(t, (1ull << 33) - 1);
  const bool still = (pta_twin_read32(t, PTA_TWIN_STATUS) & PTA_TWIN_STATUS_CAL_BUSY) != 0;
  pta_twin_run(t, 1);
  const uint32_t lo = pta_twin_read32(t, PTA_TWIN_CAL_CYC);
  const uint32_t hi = pta_twin_read32(t, PTA_TWIN_CAL_CYC_HI);
  check("a calibration of 2^33 cycles reads 0 and then 2", still && !busy(t) && lo == 0 && hi == 2);

  pta_twin_write32(t, PTA_TWIN_TS, 0x40000001u);   // and one of 2^33 + 8
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_EN | PTA_TWIN_CTRL_CAL_NOW);
  drain(t);
  const uint32_t stale = pta_twin_read32(t, PTA_TWIN_CAL_CYC_HI);
  const uint32_t lo2 = pta_twin_read32(t, PTA_TWIN_CAL_CYC);
  const uint32_t hi2 = pta_twin_read32(t, PTA_TWIN_CAL_CYC_HI);
  check("the upper half holds until the lower is read again: 2, then 8 and 4",
        stale == 2 && lo2 == 8 && hi2 == 4);
  check("the clock agrees: 2^34 + 8 cycles", pta_twin_now(t) >= (1ull << 34) + 8);
  check("and the three counters that did not pass 2^32 read zero above it",
        pta_twin_read32(t, PTA_TWIN_SHOT_CT) == 16 && pta_twin_read32(t, PTA_TWIN_SHOT_CT_HI) == 0 &&
            pta_twin_read32(t, PTA_TWIN_WLOAD_CT) == 0 && pta_twin_read32(t, PTA_TWIN_WLOAD_CT_HI) == 0 &&
            pta_twin_read32(t, PTA_TWIN_SAT_CT) == 0 && pta_twin_read32(t, PTA_TWIN_SAT_CT_HI) == 0);
  pta_twin_free(t);
}

void t_irq() {
  section("interrupts: write 1 to clear, and a mask a bit (section 3)");
  pta_twin* t = pta_twin_new(&B8x8);
  pta_twin_write32(t, PTA_TWIN_CAL_CFG, cal_cfg_word(7, 0, 1, 0));   // refused: CAL_DONE and ERR
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_EN | PTA_TWIN_CTRL_CAL_NOW);
  drain(t);
  const uint32_t both = PTA_TWIN_IRQ_CAL_DONE | PTA_TWIN_IRQ_ERR;
  check("polling works with every interrupt masked: the status is set and the line is not",
        pta_twin_read32(t, PTA_TWIN_IRQ_STATUS) == both && pta_twin_irq(t) == 0);
  pta_twin_write32(t, PTA_TWIN_IRQ_MASK, PTA_TWIN_IRQ_ERR);
  check("unmasked, the line is up", pta_twin_irq(t) == 1 && pta_twin_read32(t, PTA_TWIN_IRQ_MASK) == PTA_TWIN_IRQ_ERR);
  pta_twin_write32(t, PTA_TWIN_IRQ_STATUS, PTA_TWIN_IRQ_CAL_DONE);
  check("clearing another bit leaves it up", pta_twin_irq(t) == 1 &&
        pta_twin_read32(t, PTA_TWIN_IRQ_STATUS) == PTA_TWIN_IRQ_ERR);
  pta_twin_write32(t, PTA_TWIN_IRQ_STATUS, PTA_TWIN_IRQ_ERR);
  check("and clearing its own brings it down", pta_twin_irq(t) == 0 && pta_twin_read32(t, PTA_TWIN_IRQ_STATUS) == 0);
  pta_twin_write32(t, PTA_TWIN_IRQ_STATUS, 0);
  pta_twin_write32(t, PTA_TWIN_IRQ_MASK, 0xFFFFFFFFu);
  check("the mask has four bits", pta_twin_read32(t, PTA_TWIN_IRQ_MASK) == 0xF);
  pta_twin_free(t);
}

void t_affine() {
  section("PTA_GAIN[j] and PTA_OFFS[j]: eight words, whatever the tile's width");
  const pta_twin_build b12 = {4, 12, 8, 48, 4};
  pta_twin* t = pta_twin_new(&b12);
  Ref ref(b12);
  Cfg c = base_cfg(b12);
  program(t, c);
  const int M = 3, N = 12, K = 4;
  const auto A = operands(61, static_cast<size_t>(M) * K, 127);
  const auto B = operands(62, static_cast<size_t>(K) * N, 127);
  pta_twin_write32(t, PTA_TWIN_GAIN0 + 4 * 3, 300);
  pta_twin_write32(t, PTA_TWIN_OFFS0 + 4 * 3, static_cast<uint32_t>(-5));
  pta_twin_write32(t, PTA_TWIN_GAIN0 + 4 * 7, 0x3FF00u);   // -256: eighteen bits, signed
  pta_column_cal(&ref.dev, 3, 300, -5);
  pta_column_cal(&ref.dev, 7, -256, 0);
  Run r;
  std::vector<int64_t> w;
  const bool ran = run_gemm(t, 0, M, N, K, A, B, &r);
  ref.gemm(c, 0, 0, 0, M, N, K, A, B, &w);
  check("a column's pair reaches the tile, and reads back: 300 and -5 on column 3, -256 on column 7",
        ran && r.C == w && pta_twin_read32(t, PTA_TWIN_GAIN0 + 12) == 300 &&
            pta_twin_read32(t, PTA_TWIN_OFFS0 + 12) == static_cast<uint32_t>(-5) &&
            pta_twin_read32(t, PTA_TWIN_GAIN0 + 28) == 0x3FF00u);
  // Every word of the window, written: columns 8 to 11 must not move.
  for (uint32_t off = 0; off < PTA_TWIN_WINDOW; off += 4) {
    if (off == PTA_TWIN_CTRL || off == PTA_TWIN_IMPAIR || off == PTA_TWIN_BITS || off == PTA_TWIN_SEED) continue;
    if (off >= PTA_TWIN_SIGMA_TH && off <= PTA_TWIN_TS) continue;
    if (off >= PTA_TWIN_DRIFT_MAX && off <= PTA_TWIN_CAL_SEED) continue;
    pta_twin_write32(t, off, 512);
  }
  for (int j = 0; j < 8; ++j) pta_column_cal(&ref.dev, j, 512, 512);
  Run r2;
  std::vector<int64_t> w2;
  const bool ran2 = run_gemm(t, 0, M, N, K, A, B, &r2);
  ref.gemm(c, 0, 1, 0, M, N, K, A, B, &w2);
  bool wide_untouched = ran2 && r2.C == w2;
  for (int m = 0; m < M && wide_untouched; ++m)
    for (int n = 8; n < 12; ++n)
      wide_untouched = wide_untouched && r2.C[static_cast<size_t>(m) * N + n] == w[static_cast<size_t>(m) * N + n];
  check("no word in the window reaches a column past the eighth: 8 to 11 stay at unity and zero", wide_untouched);
  pta_twin_free(t);

  pta_twin* n4 = pta_twin_new(&B4x4);
  for (int j = 0; j < 8; ++j) pta_twin_write32(n4, PTA_TWIN_GAIN0 + 4u * static_cast<uint32_t>(j), 400);
  check("a four-column tile has four of the eight; the rest read zero and hold nothing",
        pta_twin_read32(n4, PTA_TWIN_GAIN0 + 12) == 400 && pta_twin_read32(n4, PTA_TWIN_GAIN0 + 16) == 0 &&
            pta_twin_read32(n4, PTA_TWIN_GAIN0 + 28) == 0);
  pta_twin_free(n4);
}

void t_unbuilt() {
  section("what the twin does not build reads zero, and says so");
  pta_twin* t = pta_twin_new(&B8x8);
  pta_twin_write32(t, PTA_TWIN_CTRL, 0xFFFFFFF5u);   // every bit but CAL_NOW and MODEL_RST
  check("PTA_CTRL: the scheduler, the loop orders and the c930's feed options do not stick",
        pta_twin_read32(t, PTA_TWIN_CTRL) == PTA_TWIN_CTRL_EN);
  pta_twin_write32(t, PTA_TWIN_CAL_PER, 1000);
  pta_twin_write32(t, PTA_TWIN_CAL_THR, 0xFFFFFFFFu);
  pta_twin_run(t, 1u << 20);
  check("PTA_CAL_PER and PTA_CAL_THR store and read back, and schedule nothing",
        pta_twin_read32(t, PTA_TWIN_CAL_PER) == 1000 && pta_twin_read32(t, PTA_TWIN_CAL_THR) == 0xFFFFFFu &&
            pta_twin_read32(t, PTA_TWIN_CAL_CT) == 0 && !busy(t));
  // Field widths, as the c930's register file has them.
  pta_twin_write32(t, PTA_TWIN_IMPAIR, 0xFFFFFFFFu);
  pta_twin_write32(t, PTA_TWIN_BITS, 0xFFFFFFFFu);
  pta_twin_write32(t, PTA_TWIN_SIGMA_TH, 0xFFFFFFFFu);
  pta_twin_write32(t, PTA_TWIN_DRIFT, 0xFFFFFFFFu);
  pta_twin_write32(t, PTA_TWIN_XTALK, 0xFFFFFFFFu);
  pta_twin_write32(t, PTA_TWIN_DRIFT_MAX, 0xFFFFFFFFu);
  pta_twin_write32(t, PTA_TWIN_CAL_CFG, 0xFFFFFFFFu);
  pta_twin_write32(t, PTA_TWIN_TRIM, 0xFFFFFFFFu);
  check("the configuration words keep the widths the c930's register file gives them",
        pta_twin_read32(t, PTA_TWIN_IMPAIR) == 0x7Fu && pta_twin_read32(t, PTA_TWIN_BITS) == 0x3FFFFu &&
            pta_twin_read32(t, PTA_TWIN_SIGMA_TH) == 0xFFFFu && pta_twin_read32(t, PTA_TWIN_DRIFT) == 0x1FFFFFu &&
            pta_twin_read32(t, PTA_TWIN_XTALK) == 0xFFu && pta_twin_read32(t, PTA_TWIN_DRIFT_MAX) == 0xFFFFu &&
            pta_twin_read32(t, PTA_TWIN_CAL_CFG) == 0x7FFu && pta_twin_read32(t, PTA_TWIN_TRIM) == 0xFFFF000Fu);
  bool zero = true;
  for (uint32_t off : {0x01Cu, 0x020u, 0x03Cu, 0x0F4u, 0x0FCu, 0x100u, 0x800u, 0xFFCu, 0x041u, 0x1000u, 0x4000u}) {
    pta_twin_write32(t, off, 0xFFFFFFFFu);
    zero = zero && pta_twin_read32(t, off) == 0;
  }
  check("every other word of the page, an unaligned address and one past the page read zero", zero);
  pta_twin_free(t);

  // SAT_THRESHOLD has no threshold in the map, so nothing can raise it -- not
  // even a GEMM that saturates on every element.
  pta_twin* s = pta_twin_new(&B8x8);
  Cfg c = base_cfg(B8x8);
  c.shift = 0;
  program(s, c);
  const auto A = operands(71, 64, 127);
  const auto B = operands(72, 64, 127);
  Run r;
  run_gemm(s, 0, 8, 8, 8, A, B, &r);
  check("a GEMM that saturates throughout raises no interrupt: SAT_THRESHOLD has nothing to compare with",
        r.sats > 32 && pta_twin_read32(s, PTA_TWIN_IRQ_STATUS) == 0);
  pta_twin_free(s);
}

void t_age() {
  section("ageing: the model's fast-forward, which is the twin's and not the map's");
  pta_twin* t = pta_twin_new(&B8x8);
  Ref ref(B8x8);
  Cfg c = base_cfg(B8x8);
  c.impair = PTA_QUANT | PTA_DRIFT;
  c.dsig = 55;     // TFLT's fit
  c.dlog2 = 31;    // a step every 2^31 shots: no GEMM here reaches one
  c.dmax = 8643;
  const uint32_t seed = 0xA6E;
  const int M = 4, N = 8, K = 8;
  const auto A = operands(91, static_cast<size_t>(M) * K, 127);
  const auto B = operands(92, static_cast<size_t>(K) * N, 127);
  pta_twin_write32(t, PTA_TWIN_SEED, seed);
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_MODEL_RST);
  pta_model_reset(&ref.dev, seed);
  program(t, c);
  Run before, after;
  std::vector<int64_t> w0, w1;
  run_gemm(t, 0, M, N, K, A, B, &before);
  ref.gemm(c, seed, 0, 0, M, N, K, A, B, &w0);
  const uint32_t shots = pta_twin_read32(t, PTA_TWIN_SHOT_CT);
  const uint64_t now = pta_twin_now(t);

  const int rc = pta_twin_age(t, 1676);   // steps no GEMM here would reach by firing shots
  const bool still = pta_twin_read32(t, PTA_TWIN_SHOT_CT) == shots && pta_twin_now(t) == now && !busy(t);
  const pta_cfg m = model_cfg(c, 0);
  pta_drift_age(&ref.dev, &m, 1676);
  run_gemm(t, 0, M, N, K, A, B, &after);
  ref.gemm(c, seed, 1, 0, M, N, K, A, B, &w1);
  check("1,676 steps of drift leave the state pta_drift_age() leaves: the next GEMM is the reference's",
        rc == 0 && before.C == w0 && after.C == w1 && w1 != w0);
  check("no shot was issued for it and no time passed", still);

  // Under a running command it is refused, and the command's successor shows
  // that nothing moved.
  pta_twin_write32(t, PTA_TWIN_TS, 10);
  Run held, next;
  std::vector<int64_t> w2, w3;
  submit(t, 0, M, N, K, A, B, &held);
  const int busy_rc = pta_twin_age(t, 1000);
  drain(t);
  run_gemm(t, 0, M, N, K, A, B, &next);
  ref.gemm(c, seed, 2, 0, M, N, K, A, B, &w2);
  ref.gemm(c, seed, 3, 0, M, N, K, A, B, &w3);
  check("it is refused while a command holds the tile, and ages nothing",
        busy_rc == -1 && held.C == w2 && next.C == w3);

  // With DRIFT clear a shot steps nothing, so neither does this.
  Cfg plain = c;
  plain.impair = PTA_QUANT;
  program(t, plain);
  const int clear_rc = pta_twin_age(t, 1000);
  program(t, c);
  Run last;
  std::vector<int64_t> w4;
  run_gemm(t, 0, M, N, K, A, B, &last);
  ref.gemm(c, seed, 4, 0, M, N, K, A, B, &w4);
  check("with DRIFT clear in PTA_IMPAIR it does nothing, as a shot would step nothing",
        clear_rc == 0 && last.C == w4);
  pta_twin_free(t);
}

void t_reset() {
  section("the chiplet's reset");
  pta_twin* t = pta_twin_new(&B8x8);
  Ref ref(B8x8);
  Cfg c = drifting_cfg();
  program(t, c);
  pta_twin_write32(t, PTA_TWIN_SEED, 9);
  pta_twin_write32(t, PTA_TWIN_TS, 1000);
  pta_twin_write32(t, PTA_TWIN_IRQ_MASK, 0xF);
  const int M = 4, N = 8, K = 8;
  const auto A = operands(81, static_cast<size_t>(M) * K, 127);
  const auto B = operands(82, static_cast<size_t>(K) * N, 127);
  Run done, running, queued;
  run_gemm(t, 0, M, N, K, A, B, &done);
  submit(t, 0, M, N, K, A, B, &running);
  submit(t, 0, M, N, K, A, B, &queued);
  pta_twin_run(t, 10);
  pta_twin_reset(t);
  check("what was running and what was queued end LOST, with C untouched",
        running.status == PTA_TWIN_LOST && queued.status == PTA_TWIN_LOST && all_poison(running.C) &&
            all_poison(queued.C) && done.status == PTA_TWIN_DONE);
  check("and the registers, the counters and the clock are as a new twin's",
        !busy(t) && pta_twin_pending(t) == 0 && pta_twin_now(t) == 0 && pta_twin_read32(t, PTA_TWIN_IMPAIR) == 0 &&
            pta_twin_read32(t, PTA_TWIN_SEED) == 0 && pta_twin_read32(t, PTA_TWIN_GEMM_CT) == 0 &&
            pta_twin_read32(t, PTA_TWIN_SHOT_CT) == 0 && pta_twin_read32(t, PTA_TWIN_IRQ_MASK) == 0 &&
            pta_twin_read32(t, PTA_TWIN_TS) == 0 && pta_twin_read32(t, PTA_TWIN_GAIN0) == 256);
  program(t, c);
  pta_twin_write32(t, PTA_TWIN_SEED, 9);
  Run fresh;
  std::vector<int64_t> w;
  run_gemm(t, 0, M, N, K, A, B, &fresh);
  ref.gemm(c, 9, 0, 0, M, N, K, A, B, &w);
  check("the tile too: no drift, so the first GEMM after it is a new device's", fresh.C == w && fresh.C == done.C);
  pta_twin_free(t);
}

}  // namespace

int main() {
  std::printf("PTA chiplet twin: the map of docs/designs/pta_chiplet_regmap.md, with the\n"
              "error model of third_party/grx930 behind it.  A model, not a chiplet.\n");
  t_seed_function();
  t_identity();
  t_one_driver();
  t_bitwise();
  t_harness_walk();
  t_seed_contract();
  t_completion();
  t_calibration();
  t_cal_queue();
  t_model_rst();
  t_refusals();
  t_counters();
  t_irq();
  t_affine();
  t_unbuilt();
  t_age();
  t_reset();
  std::printf("\n%d checks, %d failed\n", g_checks, g_fail);
  if (g_fail) {
    std::printf("FAILED\n");
    return 1;
  }
  std::printf("PASSED\n");
  return 0;
}
