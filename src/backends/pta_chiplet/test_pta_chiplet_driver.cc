// test_pta_chiplet_driver.cc -- the chiplet's host driver, against the twin and
// against windows and links that misbehave.
//
// pta_chiplet.cpp is the half of this that would meet hardware, and there is no
// hardware, so everything it decides is a response to what a window and a link
// do. Those are hooks precisely so that they can be made to do it here: a
// window with nothing behind it, one that never stops being busy, a link that
// will not take a command and one that takes it and loses it.
//
// The twin is the well-behaved case, and the one that says the driver reads the
// map: its answer through the driver is the model's, reproduced from what the
// driver read and nothing else.
//
// NOTHING HERE IS HARDWARE. A pass says this driver's logic is right. It says
// nothing about a chiplet.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "npu_c930.h"
#include "pta_chiplet.h"

extern "C" {
#include "pta_chiplet_twin.h"
#include "pta_tile_model.h"
}

// Two transcriptions of one map: the driver's and the twin's. If they part, one
// of them is not the map.
static_assert(PTA_CHIPLET_ID == PTA_TWIN_ID && PTA_CHIPLET_CAPS0 == PTA_TWIN_CAPS0 &&
              PTA_CHIPLET_CAPS1 == PTA_TWIN_CAPS1 && PTA_CHIPLET_CAPS2 == PTA_TWIN_CAPS2 &&
              PTA_CHIPLET_IRQ_STATUS == PTA_TWIN_IRQ_STATUS &&
              PTA_CHIPLET_GEMM_CT == PTA_TWIN_GEMM_CT && PTA_CHIPLET_CTRL == PTA_TWIN_CTRL &&
              PTA_CHIPLET_STATUS == PTA_TWIN_STATUS && PTA_CHIPLET_IMPAIR == PTA_TWIN_IMPAIR &&
              PTA_CHIPLET_BITS == PTA_TWIN_BITS && PTA_CHIPLET_SEED == PTA_TWIN_SEED &&
              PTA_CHIPLET_SIGMA_TH == PTA_TWIN_SIGMA_TH &&
              PTA_CHIPLET_SIGMA_SH == PTA_TWIN_SIGMA_SH &&
              PTA_CHIPLET_SIGMA_PR == PTA_TWIN_SIGMA_PR && PTA_CHIPLET_DRIFT == PTA_TWIN_DRIFT &&
              PTA_CHIPLET_XTALK == PTA_TWIN_XTALK && PTA_CHIPLET_DRIFT_MAX == PTA_TWIN_DRIFT_MAX,
              "the driver's offsets are the twin's");
static_assert(PTA_CHIPLET_STATUS_BUSY == PTA_TWIN_STATUS_BUSY &&
              PTA_CHIPLET_STATUS_CAL_BUSY == PTA_TWIN_STATUS_CAL_BUSY &&
              PTA_CHIPLET_STATUS_CAL_VALID == PTA_TWIN_STATUS_CAL_VALID &&
              PTA_CHIPLET_CAPS2_MODEL == PTA_TWIN_CAPS2_TWIN &&
              (PTA_CHIPLET_MAGIC | PTA_CHIPLET_MAP_VERSION) == PTA_TWIN_ID_VALUE,
              "and so are its bits");
static_assert(PTA_CHIPLET_CMD_PENDING == PTA_TWIN_PENDING && PTA_CHIPLET_CMD_DONE == PTA_TWIN_DONE &&
              PTA_CHIPLET_CMD_REFUSED == PTA_TWIN_REFUSED && PTA_CHIPLET_CMD_LOST == PTA_TWIN_LOST,
              "and its link results");

namespace {

int g_fail = 0, g_checks = 0;

void check(const char* name, bool ok) {
  ++g_checks;
  if (!ok) ++g_fail;
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", name);
}
void section(const char* title) { std::printf("\n%s\n", title); }

// ---- the twin, behind the driver's hooks ----------------------------------
uint32_t twin_read(void* ctx, uint32_t off) {
  pta_twin* t = static_cast<pta_twin*>(ctx);
  if (off == PTA_CHIPLET_STATUS) pta_twin_run(t, 1u << 16);   // a poll is where the time goes
  return pta_twin_read32(t, off);
}
void twin_write(void* ctx, uint32_t off, uint32_t v) {
  pta_twin_write32(static_cast<pta_twin*>(ctx), off, v);
}
int twin_submit(void* ctx, int bank, int M, int N, int K, const int32_t* A, const int32_t* B,
                int64_t* C, int* status) {
  pta_twin_cmd c;
  std::memset(&c, 0, sizeof c);
  c.bank = bank; c.M = M; c.N = N; c.K = K; c.A = A; c.B = B; c.C = C; c.status = status;
  return pta_twin_submit(static_cast<pta_twin*>(ctx), &c);
}

// ---- windows with no chiplet behind them ------------------------------------
uint32_t read_zero(void*, uint32_t) { return 0u; }
uint32_t read_ones(void*, uint32_t) { return 0xFFFFFFFFu; }
// A later version of the map, with a tile built: everything a version-1 driver
// would want to see, and a version number it was not written to.
uint32_t read_v2(void*, uint32_t off) {
  if (off == PTA_CHIPLET_ID) return PTA_CHIPLET_MAGIC | 2u;
  if (off == PTA_CHIPLET_CAPS1) return 0x5Fu;
  return 0u;
}
// The magic and the version, and no impairment built: a block with no tile.
uint32_t read_no_tile(void*, uint32_t off) {
  return off == PTA_CHIPLET_ID ? (PTA_CHIPLET_MAGIC | 1u) : 0u;
}
void write_nowhere(void*, uint32_t, uint32_t) {}

// A tile that identifies itself and never stops being busy.
uint32_t read_wedged(void*, uint32_t off) {
  if (off == PTA_CHIPLET_ID) return PTA_CHIPLET_MAGIC | 1u;
  if (off == PTA_CHIPLET_CAPS1) return 0x5Fu;
  if (off == PTA_CHIPLET_STATUS) return PTA_CHIPLET_STATUS_BUSY;
  return 0u;
}

// ---- links that misbehave -----------------------------------------------------
int submit_full(void*, int, int, int, int, const int32_t*, const int32_t*, int64_t*, int*) {
  return -1;                      // the queue did not take it
}
int submit_and_lose(void*, int, int, int, int, const int32_t*, const int32_t*, int64_t*, int* status) {
  *status = PTA_CHIPLET_CMD_PENDING;   // taken, and never heard of again
  return 0;
}
int submit_then_reset(void*, int, int, int, int, const int32_t*, const int32_t*, int64_t*, int* status) {
  *status = PTA_CHIPLET_CMD_LOST;
  return 0;
}

std::vector<int32_t> operands(uint32_t seed, size_t n) {
  std::vector<int32_t> v(n);
  uint32_t s = seed * 2654435761u + 12345u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = static_cast<int32_t>((s >> 8) % 255u) - 127;
  }
  return v;
}

const int64_t POISON = 0x5A5A5A5A5A5A5A5All;
bool all_poison(const std::vector<int64_t>& c) {
  for (int64_t v : c) if (v != POISON) return false;
  return true;
}

// The model's configuration, from what the DRIVER read and nothing else.
pta_cfg cfg_of(const pta_chiplet_analog_t& a) {
  pta_cfg m;
  std::memset(&m, 0, sizeof m);
  m.impair = static_cast<uint32_t>(a.impairments);
  m.act_bits = static_cast<uint32_t>(a.activation_bits);
  m.w_bits = static_cast<uint32_t>(a.weight_bits);
  m.adc_bits = static_cast<uint32_t>(a.adc_bits);
  m.adc_shift = static_cast<uint32_t>(a.adc_shift);
  m.seed = pta_chiplet_gemm_seed(static_cast<uint32_t>(a.seed), static_cast<uint32_t>(a.gemm_index));
  m.sigma_th = static_cast<uint32_t>(a.sigma_thermal_q8);
  m.k_shot = static_cast<uint32_t>(a.shot_k_q8);
  m.sigma_pr = static_cast<uint32_t>(a.sigma_prog_q8);
  m.drift_sigma = static_cast<uint32_t>(a.drift_sigma_q8);
  m.drift_log2 = static_cast<uint32_t>(a.drift_log2_shots);
  m.drift_max = static_cast<uint32_t>(a.drift_clamp_q8);
  m.xtalk = static_cast<uint32_t>(a.crosstalk_q8);
  return m;
}

// The c930 backend's reader, with the twin where the c930's block would be.
uint32_t c930_read(void* ctx, uint32_t off) {
  return off >= 0x100u ? pta_twin_read32(static_cast<pta_twin*>(ctx), off - 0x100u) : 0u;
}
void c930_write(void* ctx, uint32_t off, uint32_t v) {
  if (off >= 0x100u) pta_twin_write32(static_cast<pta_twin*>(ctx), off - 0x100u, v);
}

bool same_block(const pta_chiplet_analog_t& a, const npu_c930_analog_t& n) {
  return a.identified == n.identified && a.map_version == n.map_version && a.analog == n.analog &&
         a.tile_present == n.tile_present && a.activation_bits == n.activation_bits &&
         a.weight_bits == n.weight_bits && a.adc_bits == n.adc_bits && a.adc_shift == n.adc_shift &&
         a.seed == n.seed && a.impairments == n.impairments &&
         a.impairments_implemented == n.impairments_implemented &&
         a.impairments_requested == n.impairments_requested &&
         a.sigma_thermal_q8 == n.sigma_thermal_q8 && a.shot_k_q8 == n.shot_k_q8 &&
         a.sigma_prog_q8 == n.sigma_prog_q8 && a.drift_sigma_q8 == n.drift_sigma_q8 &&
         a.drift_log2_shots == n.drift_log2_shots && a.drift_clamp_q8 == n.drift_clamp_q8 &&
         a.crosstalk_q8 == n.crosstalk_q8 && a.loop_modes == n.loop_modes &&
         a.calibration_valid == n.calibration_valid && a.tile_rows == n.tile_rows &&
         a.tile_cols == n.tile_cols && a.operand_bits == n.operand_bits &&
         a.accumulator_bits == n.accumulator_bits;
}

const pta_twin_build BUILD = {256, 64, 8, 48, 4};

void program_v1(pta_twin* t) {
  pta_twin_write32(t, PTA_TWIN_IMPAIR, PTA_QUANT | PTA_THERMAL | PTA_SHOT | PTA_PROG_ERR | PTA_XTALK);
  pta_twin_write32(t, PTA_TWIN_BITS, 6u | (6u << 4) | (7u << 8) | (16u << 12));
  pta_twin_write32(t, PTA_TWIN_SIGMA_TH, 0x0080u);   // half an LSB of that 7-bit ADC
  pta_twin_write32(t, PTA_TWIN_SIGMA_SH, 0x0030u);
  pta_twin_write32(t, PTA_TWIN_SIGMA_PR, 0x0100u);
  pta_twin_write32(t, PTA_TWIN_XTALK, 5u);
}

// ---------------------------------------------------------------------------------
void t_detect() {
  section("detection: a tile this driver knows, or nothing");
  pta_chiplet_device_t dev;
  std::memset(&dev, 0, sizeof dev);
  check("no window at all: not present, and the error says there is no path to the block",
        pta_chiplet_detect(&dev) == 0 && dev.present == 0 && dev.error == PTA_CHIPLET_ERR_NO_WINDOW);

  struct W { const char* name; pta_chiplet_read_fn rd; };
  const W dead[] = {
      {"a window that reads zero", read_zero},
      {"a window that floats high", read_ones},
      {"a later version of the map, tile and all", read_v2},
      {"the magic and no tile behind it", read_no_tile},
  };
  for (const W& w : dead) {
    pta_chiplet_attach_window(&dev, w.rd, write_nowhere, nullptr);
    char name[160];
    std::snprintf(name, sizeof name, "%-40s is not a chiplet", w.name);
    check(name, pta_chiplet_detect(&dev) == 0 && dev.present == 0 &&
                    dev.error == PTA_CHIPLET_ERR_NOT_PRESENT);
  }

  pta_twin* t = pta_twin_new(&BUILD);
  pta_chiplet_attach_window(&dev, twin_read, twin_write, t);
  check("the twin is: the magic, map version 1, and a tile", pta_chiplet_detect(&dev) == 1 &&
        dev.present == 1 && dev.error == PTA_CHIPLET_OK);
  // attach_window zeroes the struct, so a link attached before it is gone.
  pta_chiplet_attach_link(&dev, twin_submit, t);
  pta_chiplet_attach_window(&dev, twin_read, twin_write, t);
  check("attaching the window discards a link attached before it, as the header warns",
        dev.submit == nullptr);
  pta_twin_free(t);
}

void t_reader() {
  section("the reader: the c930 backend's, and what the chiplet adds");
  pta_twin* t = pta_twin_new(&BUILD);
  pta_chiplet_device_t dev;
  pta_chiplet_attach_window(&dev, twin_read, twin_write, t);
  npu_c930_device_t c930;
  npu_c930_attach_model(&c930, c930_read, c930_write, t);

  pta_chiplet_analog_t a;
  npu_c930_analog_t n;
  pta_chiplet_device_t none;
  std::memset(&none, 0, sizeof none);
  check("with no window every field is unknown, and the call says it decided nothing",
        pta_chiplet_read_analog(&none, &a) == -1 && a.identified == 0 && a.analog == -1 &&
            a.tile_present == -1 && a.gemm_index == -1 && a.is_model == -1);

  pta_chiplet_read_analog(&dev, &a);
  npu_c930_read_analog(&c930, &n);
  check("clear: field for field what the c930's reader gives through a change of base",
        same_block(a, n) && a.analog == 0 && a.tile_present == 1);
  check("and the tile's own: 256 x 64, 8-bit operands, 48-bit sums, a model, GEMM 0 next",
        a.tile_rows == 256 && a.tile_cols == 64 && a.operand_bits == 8 && a.accumulator_bits == 48 &&
            a.is_model == 1 && a.gemm_index == 0);

  pta_twin_write32(t, PTA_TWIN_SEED, 0xD00Du);
  program_v1(t);
  pta_twin_write32(t, PTA_TWIN_DRIFT, 55u | (9u << 16));
  pta_twin_write32(t, PTA_TWIN_DRIFT_MAX, 8643u);
  pta_chiplet_read_analog(&dev, &a);
  npu_c930_read_analog(&c930, &n);
  check("impaired: the same again, every sigma and the drift walk",
        same_block(a, n) && a.analog == 1 && a.seed == 0xD00D && a.sigma_thermal_q8 == 0x80 &&
            a.drift_sigma_q8 == 55 && a.drift_log2_shots == 9 && a.drift_clamp_q8 == 8643);

  // An unbuilt bit is requested and reported, not hidden: it is what explains
  // the refusal a GEMM would get.
  pta_twin_write32(t, PTA_TWIN_IMPAIR, PTA_QUANT | PTA_MZM_NL);
  pta_chiplet_read_analog(&dev, &a);
  check("a requested impairment the tile does not build is reported as requested",
        a.impairments_requested == (PTA_QUANT | PTA_MZM_NL) && a.impairments_implemented == 0x5F);
  pta_twin_free(t);
}

void t_gemm() {
  section("a GEMM through the driver is the model, from what the driver read");
  pta_twin* t = pta_twin_new(&BUILD);
  pta_chiplet_device_t dev;
  pta_chiplet_attach_window(&dev, twin_read, twin_write, t);
  pta_chiplet_attach_link(&dev, twin_submit, t);
  pta_chiplet_detect(&dev);
  pta_twin_write32(t, PTA_TWIN_SEED, 0x51Du);
  program_v1(t);

  const int M = 4, N = 70, K = 300;
  const auto A = operands(1, static_cast<size_t>(M) * K);
  const auto B = operands(2, static_cast<size_t>(K) * N);
  pta_tile tile = {0, 0, 0, 0};
  pta_device ref;
  bool ok = true, moved = true, counted = true;
  std::vector<int64_t> first;
  for (int g = 0; g < 3; ++g) {
    // Everything the reference needs is read BEFORE the GEMM it describes.
    pta_chiplet_analog_t a;
    pta_chiplet_read_analog(&dev, &a);
    if (g == 0) {
      tile.rows = a.tile_rows; tile.cols = a.tile_cols;
      tile.din_w = a.operand_bits; tile.acc_w = a.accumulator_bits;
      pta_device_init(&ref, &tile);
      pta_model_reset(&ref, 0);
    }
    counted = counted && a.gemm_index == g;
    const pta_cfg m = cfg_of(a);
    std::vector<int64_t> C(static_cast<size_t>(M) * N, POISON), want(C.size());
    const int rc = pta_chiplet_gemm(&dev, 0, M, N, K, A.data(), B.data(), C.data());
    pta_gemm(&m, &tile, &ref, 0, M, N, K, A.data(), B.data(), want.data());
    ok = ok && rc == 0 && dev.error == PTA_CHIPLET_OK && C == want;
    if (g == 0) first = C; else moved = moved && C != first;
  }
  pta_device_free(&ref);
  check("three GEMMs, 4 x 70 x 300, each the model's from the reader's fields and its index", ok);
  check("PTA_GEMM_CT read 0, 1 and 2 before them, and each drew its own noise", counted && moved);
  check("the seed function is the twin's",
        pta_chiplet_gemm_seed(0x51Du, 2) == pta_twin_gemm_seed(0x51Du, 2) &&
            pta_chiplet_gemm_seed(1u, 0u) == 3291240986u);

  // It waits for whatever holds the tile: here, a calibration.
  pta_twin_write32(t, PTA_TWIN_TS, 1000);   // long enough that the first poll finds it calibrating
  pta_twin_write32(t, PTA_TWIN_TRIM, 2u | (0x4000u << 16));
  pta_twin_write32(t, PTA_TWIN_CAL_CFG, 6u | (1u << 4) | (2u << 8));
  pta_twin_write32(t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_EN | PTA_TWIN_CTRL_CAL_NOW);
  const bool calibrating = pta_chiplet_idle(&dev) == 0;
  std::vector<int64_t> C(static_cast<size_t>(M) * N, POISON);
  const int rc = pta_chiplet_gemm(&dev, 0, M, N, K, A.data(), B.data(), C.data());
  check("asked for during a calibration, it waits it out and then runs",
        calibrating && rc == 0 && !all_poison(C) && pta_twin_read32(t, PTA_TWIN_CAL_CT) == 1 &&
            pta_twin_read32(t, PTA_TWIN_GEMM_CT) == 4);
  pta_twin_free(t);
}

void t_failures() {
  section("when it cannot run, C is untouched and the error says why");
  const int M = 2, N = 8, K = 8;
  const auto A = operands(3, static_cast<size_t>(M) * K);
  const auto B = operands(4, static_cast<size_t>(K) * N);
  std::vector<int64_t> C(static_cast<size_t>(M) * N, POISON);
  pta_twin* t = pta_twin_new(&BUILD);
  pta_chiplet_device_t dev;

  pta_chiplet_attach_window(&dev, twin_read, twin_write, t);
  pta_chiplet_detect(&dev);
  int rc = pta_chiplet_gemm(&dev, 0, M, N, K, A.data(), B.data(), C.data());
  check("no link: refused as having no path to issue work, which is every chiplet off a model",
        rc == -1 && dev.error == PTA_CHIPLET_ERR_NO_LINK && all_poison(C) &&
            pta_twin_read32(t, PTA_TWIN_GEMM_CT) == 0);

  pta_chiplet_attach_link(&dev, twin_submit, t);
  pta_twin_write32(t, PTA_TWIN_IMPAIR, PTA_QUANT | PTA_MZM_NL | PTA_THERMAL);
  rc = pta_chiplet_gemm(&dev, 0, M, N, K, A.data(), B.data(), C.data());
  check("an impairment the tile does not build: REFUSED, and the driver names the bit, 0x20",
        rc == -1 && dev.error == PTA_CHIPLET_ERR_REFUSED && dev.refused_impairments == 0x20u &&
            all_poison(C));
  pta_twin_write32(t, PTA_TWIN_IMPAIR, PTA_QUANT);
  pta_twin_write32(t, PTA_TWIN_BITS, 6u | (6u << 4) | (7u << 8) | (41u << 12));
  rc = pta_chiplet_gemm(&dev, 0, M, N, K, A.data(), B.data(), C.data());
  check("an ADC shift of 41: REFUSED, and no impairment is blamed for it",
        rc == -1 && dev.error == PTA_CHIPLET_ERR_REFUSED && dev.refused_impairments == 0u && all_poison(C));
  pta_twin_write32(t, PTA_TWIN_BITS, 6u | (6u << 4) | (7u << 8) | (16u << 12));
  rc = pta_chiplet_gemm(&dev, 0, M, N, K, A.data(), B.data(), C.data());
  check("and with the register put right the same command runs",
        rc == 0 && dev.error == PTA_CHIPLET_OK && dev.refused_impairments == 0u && !all_poison(C));

  std::fill(C.begin(), C.end(), POISON);
  pta_chiplet_attach_link(&dev, submit_full, nullptr);
  rc = pta_chiplet_gemm(&dev, 0, M, N, K, A.data(), B.data(), C.data());
  check("a link that will not take the command: NOT ACCEPTED",
        rc == -1 && dev.error == PTA_CHIPLET_ERR_NOT_ACCEPTED && all_poison(C));
  pta_chiplet_attach_link(&dev, submit_and_lose, nullptr);
  rc = pta_chiplet_gemm(&dev, 0, M, N, K, A.data(), B.data(), C.data());
  check("a command taken and never ended, on a tile that reads idle: LOST, not success",
        rc == -1 && dev.error == PTA_CHIPLET_ERR_LOST && all_poison(C));
  pta_chiplet_attach_link(&dev, submit_then_reset, nullptr);
  rc = pta_chiplet_gemm(&dev, 0, M, N, K, A.data(), B.data(), C.data());
  check("a command a reset discarded: LOST", rc == -1 && dev.error == PTA_CHIPLET_ERR_LOST && all_poison(C));

  pta_chiplet_device_t stuck;
  pta_chiplet_attach_window(&stuck, read_wedged, write_nowhere, nullptr);
  pta_chiplet_attach_link(&stuck, twin_submit, t);
  const bool found = pta_chiplet_detect(&stuck) == 1;
  rc = pta_chiplet_gemm(&stuck, 0, M, N, K, A.data(), B.data(), C.data());
  check("a tile that never stops being busy: WEDGED, and the command was never sent",
        found && rc == -1 && stuck.error == PTA_CHIPLET_ERR_WEDGED && all_poison(C) &&
            pta_twin_pending(t) == 0);

  pta_chiplet_device_t undetected;
  pta_chiplet_attach_window(&undetected, twin_read, twin_write, t);
  pta_chiplet_attach_link(&undetected, twin_submit, t);
  rc = pta_chiplet_gemm(&undetected, 0, M, N, K, A.data(), B.data(), C.data());
  check("a device nobody detected is not driven", rc == -1 &&
        undetected.error == PTA_CHIPLET_ERR_NOT_PRESENT && all_poison(C));
  bool named = true;
  for (int e = PTA_CHIPLET_OK; e <= PTA_CHIPLET_ERR_LOST; ++e)
    named = named && std::strcmp(pta_chiplet_error_string(e), "unknown error") != 0;
  check("every error has a sentence, and a number that is not one says so",
        named && std::strcmp(pta_chiplet_error_string(99), "unknown error") == 0);
  pta_twin_free(t);
}

}  // namespace

int main() {
  std::printf("PTA chiplet driver, against the digital twin. A model, not a chiplet.\n");
  t_detect();
  t_reader();
  t_gemm();
  t_failures();
  std::printf("\n%d checks, %d failed\n", g_checks, g_fail);
  std::printf("%s\n", g_fail ? "FAILED" : "PASSED");
  return g_fail ? 1 : 0;
}
