// pta_chiplet.cpp -- the host's driver for the PTA chiplet. See pta_chiplet.h
// for what exists behind it today, which is a model and nothing else.

#include "pta_chiplet.h"

#include <cstring>

namespace {

// How many times the completion test is asked before a command is called
// wedged. A poll is one read of PTA_STATUS across a link, so this is a bound on
// reads and not on time; a model advances its own clock as it is read.
const int kMaxPolls = 1 << 20;

inline bool has_window(const pta_chiplet_device_t* dev) {
  return dev && dev->read32;
}

inline uint32_t rd(const pta_chiplet_device_t* dev, uint32_t off) {
  return dev->read32(dev->io_ctx, off);
}

uint64_t splitmix64(uint64_t* s) {
  uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

void all_unknown(pta_chiplet_analog_t* out) {
  out->identified              = 0;
  out->map_version             = -1;
  out->analog                  = -1;
  out->tile_present            = -1;
  out->activation_bits         = -1;
  out->weight_bits             = -1;
  out->adc_bits                = -1;
  out->adc_shift               = -1;
  out->seed                    = -1;
  out->impairments             = -1;
  out->impairments_implemented = -1;
  out->impairments_requested   = -1;
  out->sigma_thermal_q8        = -1;
  out->shot_k_q8               = -1;
  out->sigma_prog_q8           = -1;
  out->drift_sigma_q8          = -1;
  out->drift_log2_shots        = -1;
  out->drift_clamp_q8          = -1;
  out->crosstalk_q8            = -1;
  out->loop_modes              = -1;
  out->calibration_valid       = -1;
  out->tile_rows               = -1;
  out->tile_cols               = -1;
  out->operand_bits            = -1;
  out->accumulator_bits        = -1;
  out->gemm_index              = -1;
  out->is_model                = -1;
}

}  // namespace

extern "C" {

void pta_chiplet_attach_window(pta_chiplet_device_t* dev,
                               pta_chiplet_read_fn read32,
                               pta_chiplet_write_fn write32, void* ctx) {
  if (!dev) return;
  std::memset(dev, 0, sizeof(*dev));
  dev->read32  = read32;
  dev->write32 = write32;
  dev->io_ctx  = ctx;
}

void pta_chiplet_attach_link(pta_chiplet_device_t* dev,
                             pta_chiplet_submit_fn submit, void* ctx) {
  if (!dev) return;
  dev->submit   = submit;
  dev->link_ctx = ctx;
}

int pta_chiplet_detect(pta_chiplet_device_t* dev) {
  if (!dev) return 0;
  dev->present = 0;
  if (!has_window(dev)) {
    dev->error = PTA_CHIPLET_ERR_NO_WINDOW;
    return 0;
  }
  // The magic, the version, and a tile: in that order, and nothing is believed
  // until all three hold. A window with nothing behind it reads zero or all
  // ones and is neither. A later version of the map may have moved anything,
  // so a driver written to version 1 drives version 1. And a block with no
  // impairment built is a block with no tile.
  const uint32_t id = rd(dev, PTA_CHIPLET_ID);
  if ((id & 0xFFFFFF00u) != PTA_CHIPLET_MAGIC ||
      (id & 0xFFu) != PTA_CHIPLET_MAP_VERSION ||
      (rd(dev, PTA_CHIPLET_CAPS1) & PTA_CHIPLET_DEFINED) == 0) {
    dev->error = PTA_CHIPLET_ERR_NOT_PRESENT;
    return 0;
  }
  dev->present = 1;
  dev->error   = PTA_CHIPLET_OK;
  return 1;
}

int pta_chiplet_read_analog(pta_chiplet_device_t* dev, pta_chiplet_analog_t* out) {
  if (!out) return -1;
  all_unknown(out);
  if (!has_window(dev)) return -1;

  // The magic first, and nothing else until it has matched.
  const uint32_t id = rd(dev, PTA_CHIPLET_ID);
  if ((id & 0xFFFFFF00u) != PTA_CHIPLET_MAGIC) return 0;
  out->identified  = 1;
  out->map_version = (int)(id & 0xFFu);

  const uint32_t built  = rd(dev, PTA_CHIPLET_CAPS1) & PTA_CHIPLET_DEFINED;
  const uint32_t impair = rd(dev, PTA_CHIPLET_IMPAIR) & PTA_CHIPLET_DEFINED;
  out->impairments_implemented = (int64_t)built;
  out->impairments_requested   = (int64_t)impair;
  out->is_model = (rd(dev, PTA_CHIPLET_CAPS2) & PTA_CHIPLET_CAPS2_MODEL) ? 1 : 0;

  if (built == 0) {
    out->tile_present = 0;
    out->analog       = 0;
    return 0;
  }
  out->tile_present = 1;

  const uint32_t caps0 = rd(dev, PTA_CHIPLET_CAPS0);
  out->tile_rows        = (int)(caps0 & 0x3FFu);
  out->tile_cols        = (int)((caps0 >> 10) & 0x3FFu);
  out->operand_bits     = (int)((caps0 >> 20) & 0x3Fu);
  out->accumulator_bits = (int)((caps0 >> 26) & 0x3Fu);

  // Which GEMM comes next. Device state rather than configuration, so it is
  // read whether or not anything is impaired: an exact GEMM takes an index too,
  // and a reader that learned it only once the model was switched on would
  // have no way to say which GEMM the first impaired one was.
  out->gemm_index = (int64_t)rd(dev, PTA_CHIPLET_GEMM_CT);

  // A tile with nothing enabled computes the exact product. Its configuration
  // registers still hold whatever was last written, and those values describe
  // a model that is not running, so they are not reported.
  if (impair == 0) {
    out->analog = 0;
    return 0;
  }

  const uint32_t bits  = rd(dev, PTA_CHIPLET_BITS);
  const uint32_t drift = rd(dev, PTA_CHIPLET_DRIFT);
  out->analog           = 1;
  out->activation_bits  = (int)(bits & 0xFu);
  out->weight_bits      = (int)((bits >> 4) & 0xFu);
  out->adc_bits         = (int)((bits >> 8) & 0xFu);
  out->adc_shift        = (int)((bits >> 12) & 0x3Fu);
  out->seed             = (int64_t)rd(dev, PTA_CHIPLET_SEED);
  out->impairments      = (int64_t)impair;
  out->sigma_thermal_q8 = (int)(rd(dev, PTA_CHIPLET_SIGMA_TH) & 0xFFFFu);
  out->shot_k_q8        = (int)(rd(dev, PTA_CHIPLET_SIGMA_SH) & 0xFFFFu);
  out->sigma_prog_q8    = (int)(rd(dev, PTA_CHIPLET_SIGMA_PR) & 0xFFFFu);
  out->drift_sigma_q8   = (int)(drift & 0xFFFFu);
  out->drift_log2_shots = (int)((drift >> 16) & 0x1Fu);
  out->drift_clamp_q8   = (int)(rd(dev, PTA_CHIPLET_DRIFT_MAX) & 0xFFFFu);
  out->crosstalk_q8     = (int)(rd(dev, PTA_CHIPLET_XTALK) & 0xFFu);
  out->loop_modes       = (int)((rd(dev, PTA_CHIPLET_CTRL) >> 7) & 0x7u);
  out->calibration_valid =
      (rd(dev, PTA_CHIPLET_STATUS) & PTA_CHIPLET_STATUS_CAL_VALID) ? 1 : 0;
  return 0;
}

int pta_chiplet_idle(pta_chiplet_device_t* dev) {
  if (!has_window(dev)) return -1;
  // One read. BUSY and CAL_BUSY are in the same word so that this cannot
  // straddle a change, and it is the whole of the device's side of the test.
  return (rd(dev, PTA_CHIPLET_STATUS) &
          (PTA_CHIPLET_STATUS_BUSY | PTA_CHIPLET_STATUS_CAL_BUSY)) ? 0 : 1;
}

static int wait_idle(pta_chiplet_device_t* dev) {
  for (int i = 0; i < kMaxPolls; ++i)
    if (pta_chiplet_idle(dev) == 1) return 0;
  return -1;
}

int pta_chiplet_gemm(pta_chiplet_device_t* dev, int bank, int M, int N, int K,
                     const int32_t* A, const int32_t* B, int64_t* C) {
  if (!dev) return -1;
  dev->refused_impairments = 0;
  if (!has_window(dev)) { dev->error = PTA_CHIPLET_ERR_NO_WINDOW; return -1; }
  if (!dev->present)    { dev->error = PTA_CHIPLET_ERR_NOT_PRESENT; return -1; }
  // No link is not "no result": a GEMM that returned without running would
  // leave the caller reading whatever was in C.
  if (!dev->submit)     { dev->error = PTA_CHIPLET_ERR_NO_LINK; return -1; }

  // Whatever holds the tile -- a calibration, or another holder's command --
  // goes first. The command would queue behind it on the chiplet anyway; this
  // is so that the wait below is for THIS command and no other.
  if (wait_idle(dev) != 0) { dev->error = PTA_CHIPLET_ERR_WEDGED; return -1; }

  int status = PTA_CHIPLET_CMD_PENDING;
  if (dev->submit(dev->link_ctx, bank, M, N, K, A, B, C, &status) != 0) {
    dev->error = PTA_CHIPLET_ERR_NOT_ACCEPTED;
    return -1;
  }
  if (wait_idle(dev) != 0) { dev->error = PTA_CHIPLET_ERR_WEDGED; return -1; }

  if (status == PTA_CHIPLET_CMD_DONE) {
    dev->error = PTA_CHIPLET_OK;
    return 0;
  }
  if (status == PTA_CHIPLET_CMD_REFUSED) {
    // The reason, because "refused" alone reads as a driver fault and the
    // cause is a register somebody else may have written.
    dev->refused_impairments =
        rd(dev, PTA_CHIPLET_IMPAIR) & PTA_CHIPLET_DEFINED &
        ~rd(dev, PTA_CHIPLET_CAPS1);
    dev->error = PTA_CHIPLET_ERR_REFUSED;
    return -1;
  }
  // Idle, and the command neither done nor refused: it ended without a result.
  dev->error = PTA_CHIPLET_ERR_LOST;
  return -1;
}

uint32_t pta_chiplet_gemm_seed(uint32_t seed, uint32_t index) {
  uint64_t s = ((uint64_t)seed << 32) | index;
  return (uint32_t)(splitmix64(&s) >> 32);
}

const char* pta_chiplet_error_string(int error) {
  switch (error) {
    case PTA_CHIPLET_OK:               return "no error";
    case PTA_CHIPLET_ERR_NO_WINDOW:
      return "no path to the chiplet's register block: the GPU's driver has no call that reaches it";
    case PTA_CHIPLET_ERR_NO_LINK:
      return "no path to issue work to the chiplet: the link's command format is not specified";
    case PTA_CHIPLET_ERR_NOT_PRESENT:
      return "no PTA tile this driver knows behind the window";
    case PTA_CHIPLET_ERR_NOT_ACCEPTED: return "the chiplet did not accept the command";
    case PTA_CHIPLET_ERR_REFUSED:      return "the tile refused the command";
    case PTA_CHIPLET_ERR_WEDGED:       return "the chiplet never stopped being busy";
    case PTA_CHIPLET_ERR_LOST:         return "the command ended with no result";
  }
  return "unknown error";
}

}  // extern "C"
