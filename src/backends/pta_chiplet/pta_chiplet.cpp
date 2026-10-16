// pta_chiplet.cpp -- the host's driver for the PTA chiplet. See pta_chiplet.h
// for what exists behind it today, which is a model and nothing else.

#include "pta_chiplet.h"

#include <atomic>
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

inline void wr(const pta_chiplet_device_t* dev, uint32_t off, uint32_t v) {
  if (dev->write32) dev->write32(dev->io_ctx, off, v);
}

// How many times a queued GEMM's record is looked at before the GEMM is called
// lost. Each look is one read of host memory and, while the slot is pending,
// one read of PTQ_STATUS: that read is how a halted queue is told from a slow
// one. On a board the host would spin on its own cache between such reads; on a
// model the read is also where the model's time passes, as PTA_STATUS's is.
const int kQueuePolls = 1 << 20;
// How many reads of PTQ_STATUS a teardown waits for ENABLED to clear.
const int kQuiescePolls = 1 << 12;

// Little-endian, whatever the host is, and through volatile: the module writes
// these lines, and nothing the compiler can see does.
inline uint32_t ld32(const volatile uint8_t* p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
inline void put32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
inline void put64(uint8_t* p, uint64_t v) {
  put32(p, (uint32_t)v);
  put32(p + 4, (uint32_t)(v >> 32));
}
inline uint64_t round_line(uint64_t n) {
  return (n + PTA_CHIPLET_LINE - 1) / PTA_CHIPLET_LINE * PTA_CHIPLET_LINE;
}
inline volatile uint8_t* slot_of(const pta_chiplet_device_t* dev, uint32_t idx) {
  return static_cast<volatile uint8_t*>(dev->ring.cpu) +
         (size_t)(idx & ((1u << dev->ring_log2) - 1u)) * PTA_CHIPLET_SLOT;
}
inline uint32_t slot_status(const pta_chiplet_device_t* dev, uint32_t idx) {
  return ld32(slot_of(dev, idx) + PTA_CHIPLET_D_STATUS);
}
inline void store_status(const pta_chiplet_device_t* dev, uint32_t idx, uint32_t st) {
  volatile uint8_t* s = slot_of(dev, idx) + PTA_CHIPLET_D_STATUS;
  s[0] = (uint8_t)st; s[1] = (uint8_t)(st >> 8); s[2] = (uint8_t)(st >> 16); s[3] = (uint8_t)(st >> 24);
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

int pta_chiplet_attach_queue(pta_chiplet_device_t* dev, pta_chiplet_dma_alloc_fn alloc,
                             pta_chiplet_dma_free_fn free_fn, void* ctx, int ring_log2) {
  if (!dev) return 0;
  if (!has_window(dev)) { dev->error = PTA_CHIPLET_ERR_NO_WINDOW; return 0; }
  if (dev->queue) pta_chiplet_detach_queue(dev);
  // The magic, the version, version 1 descriptors, and room for the ring, in
  // that order. A window with no second page reads zero there, which is every
  // chiplet behind a GPU.
  const uint32_t id = rd(dev, PTA_CHIPLET_PTQ_ID);
  const uint32_t caps = rd(dev, PTA_CHIPLET_PTQ_CAPS);
  if ((id & 0xFFFFFF00u) != PTA_CHIPLET_PTQ_MAGIC || (id & 0xFFu) != PTA_CHIPLET_PTQ_VERSION ||
      !(caps & PTA_CHIPLET_PTQ_CAPS_V1) || ring_log2 < 0 || (uint32_t)ring_log2 > (caps & 0x1Fu) ||
      !alloc || !free_fn) {
    dev->error = PTA_CHIPLET_ERR_NO_LINK;
    return 0;
  }
  pta_chiplet_dma_t ring;
  std::memset(&ring, 0, sizeof ring);
  const size_t bytes = (size_t)PTA_CHIPLET_SLOT << ring_log2;
  if (alloc(ctx, bytes, &ring) != 0 || !ring.cpu || ring.bytes < bytes || (ring.addr & 0xFFFu)) {
    if (ring.cpu) free_fn(ctx, &ring);
    dev->error = PTA_CHIPLET_ERR_NO_LINK;
    return 0;
  }
  std::memset(ring.cpu, 0, ring.bytes);
  std::atomic_thread_fence(std::memory_order_release);
  wr(dev, PTA_CHIPLET_PTQ_CTRL, 0);
  wr(dev, PTA_CHIPLET_PTQ_BASE_LO, (uint32_t)ring.addr);
  wr(dev, PTA_CHIPLET_PTQ_BASE_HI, (uint32_t)(ring.addr >> 32));
  wr(dev, PTA_CHIPLET_PTQ_SIZE, (uint32_t)ring_log2);
  wr(dev, PTA_CHIPLET_PTQ_TAIL, 0);
  wr(dev, PTA_CHIPLET_PTQ_CTRL, PTA_CHIPLET_PTQ_ENABLE);
  // And read back: a queue that did not take the ring is no queue.
  if (!(rd(dev, PTA_CHIPLET_PTQ_STATUS) & PTA_CHIPLET_PTQ_ENABLED) ||
      rd(dev, PTA_CHIPLET_PTQ_BASE_LO) != (uint32_t)ring.addr ||
      rd(dev, PTA_CHIPLET_PTQ_BASE_HI) != (uint32_t)(ring.addr >> 32) ||
      rd(dev, PTA_CHIPLET_PTQ_SIZE) != (uint32_t)ring_log2) {
    wr(dev, PTA_CHIPLET_PTQ_CTRL, 0);
    free_fn(ctx, &ring);
    dev->error = PTA_CHIPLET_ERR_NO_LINK;
    return 0;
  }
  dev->queue = 1;
  dev->queue_dead = 0;
  dev->ring_log2 = (uint32_t)ring_log2;
  dev->next = dev->ended = 0;
  dev->ring = ring;
  std::memset(&dev->stage, 0, sizeof dev->stage);
  dev->dma_alloc = alloc;
  dev->dma_free = free_fn;
  dev->dma_ctx = ctx;
  dev->last_index = -1;
  dev->error = PTA_CHIPLET_OK;
  return 1;
}

void pta_chiplet_detach_queue(pta_chiplet_device_t* dev) {
  if (!dev || !dev->queue) return;
  wr(dev, PTA_CHIPLET_PTQ_CTRL, 0);
  if (dev->stage.cpu) dev->dma_free(dev->dma_ctx, &dev->stage);
  dev->dma_free(dev->dma_ctx, &dev->ring);
  std::memset(&dev->stage, 0, sizeof dev->stage);
  std::memset(&dev->ring, 0, sizeof dev->ring);
  dev->queue = 0;
}

int pta_chiplet_queue_teardown(pta_chiplet_device_t* dev) {
  if (!dev || !dev->queue || dev->queue_dead) return -1;
  // 1. Stop it, and wait until it says it writes nothing more.
  wr(dev, PTA_CHIPLET_PTQ_CTRL, 0);
  bool stopped = false;
  for (int i = 0; i < kQuiescePolls && !stopped; ++i)
    stopped = !(rd(dev, PTA_CHIPLET_PTQ_STATUS) & PTA_CHIPLET_PTQ_ENABLED);
  if (!stopped) {
    dev->queue_dead = 1;
    return -1;
  }
  // 2. Behind PTQ_HEAD the module has ended its slots. From there to our own
  //    next index they are ours again, and each ends lost. A head that is not
  //    among our slots is not believed: then every slot still pending ends.
  uint32_t from = rd(dev, PTA_CHIPLET_PTQ_HEAD);
  if (from - dev->ended > dev->next - dev->ended) from = dev->ended;
  std::atomic_thread_fence(std::memory_order_acquire);
  for (uint32_t i = from; i != dev->next; ++i)
    if (slot_status(dev, i) == 0) store_status(dev, i, PTA_CHIPLET_REC_LOST);
  // 3. Start it again, empty: ENABLE sets PTQ_HEAD to PTQ_TAIL.
  std::atomic_thread_fence(std::memory_order_release);
  wr(dev, PTA_CHIPLET_PTQ_TAIL, dev->next);
  wr(dev, PTA_CHIPLET_PTQ_CTRL, PTA_CHIPLET_PTQ_ENABLE);
  return 0;
}

// Publish one descriptor, ring the doorbell, and wait for its record: the
// design's section 4, the host's half, whoever's memory the three addresses
// are in. Returns 0 when the GEMM ended done, with dev->last_index its index,
// or -1 with dev->error saying why.
static int queue_run(pta_chiplet_device_t* dev, int bank, int M, int N, int K,
                     uint64_t act, uint64_t wgt, uint64_t out) {
  // The descriptor, version 1, status zero: the host path's, field for field.
  uint8_t d[PTA_CHIPLET_LINE];
  std::memset(d, 0, sizeof d);
  put32(d + 0x00, (1u << 16) | 64u);
  put32(d + 0x04, (uint32_t)bank << 8);
  put32(d + 0x08, 0);
  put32(d + 0x0C, (uint32_t)M);
  put32(d + 0x10, (uint32_t)K);
  put32(d + 0x14, (uint32_t)N);
  put64(d + 0x18, act);
  put64(d + 0x20, wgt);
  put64(d + 0x28, out);
  const uint32_t idx = dev->next;
  volatile uint8_t* slot = slot_of(dev, idx);
  for (unsigned i = 0; i < sizeof d; ++i) slot[i] = d[i];

  // The fence, then the doorbell: the module must read what was written here.
  std::atomic_thread_fence(std::memory_order_release);
  ++dev->next;
  wr(dev, PTA_CHIPLET_PTQ_TAIL, dev->next);

  uint32_t st = 0;
  for (int i = 0; i < kQueuePolls; ++i) {
    st = ld32(slot + PTA_CHIPLET_D_STATUS);
    if (st != 0) break;
    if (rd(dev, PTA_CHIPLET_PTQ_STATUS) & PTA_CHIPLET_PTQ_HALTED) break;
  }
  if (st == 0) {
    // Halted, or it never ended: tear the queue down, which ends the slot.
    if (pta_chiplet_queue_teardown(dev) != 0) {
      dev->ended = dev->next;
      dev->error = PTA_CHIPLET_ERR_WEDGED;
      return -1;
    }
    st = ld32(slot + PTA_CHIPLET_D_STATUS);
  }
  dev->ended = dev->next;
  std::atomic_thread_fence(std::memory_order_acquire);

  switch (st) {
    case PTA_CHIPLET_REC_DONE:
      dev->last_index = (int64_t)ld32(slot + PTA_CHIPLET_D_INDEX);
      dev->error = PTA_CHIPLET_OK;
      return 0;
    case PTA_CHIPLET_REC_REFUSED:
      dev->refused_impairments =
          rd(dev, PTA_CHIPLET_IMPAIR) & PTA_CHIPLET_DEFINED & ~rd(dev, PTA_CHIPLET_CAPS1);
      dev->error = PTA_CHIPLET_ERR_REFUSED;
      return -1;
    case PTA_CHIPLET_REC_NOT_ISSUED:
    case PTA_CHIPLET_REC_NOT_RUN:
      dev->error = PTA_CHIPLET_ERR_NOT_ACCEPTED;
      return -1;
  }
  dev->error = PTA_CHIPLET_ERR_LOST;   // lost, or a status no version of the queue writes
  return -1;
}

// One GEMM through the queue, from the driver's int32 interface: the operands
// are copied into memory the driver owns, and the results out of it.
static int queue_gemm(pta_chiplet_device_t* dev, int bank, int M, int N, int K,
                      const int32_t* A, const int32_t* B, int64_t* C) {
  if (dev->queue_dead) { dev->error = PTA_CHIPLET_ERR_WEDGED; return -1; }
  if (M <= 0 || N <= 0 || K <= 0 || bank < 0 || bank > 0xFF || !A || !B || !C) {
    dev->error = PTA_CHIPLET_ERR_NOT_ACCEPTED;
    return -1;
  }
  const uint64_t na = (uint64_t)M * K, nb = (uint64_t)K * N, nc = (uint64_t)M * N;
  for (uint64_t i = 0; i < na; ++i)
    if (A[i] < -128 || A[i] > 127) { dev->error = PTA_CHIPLET_ERR_FORMAT; return -1; }
  for (uint64_t i = 0; i < nb; ++i)
    if (B[i] < -128 || B[i] > 127) { dev->error = PTA_CHIPLET_ERR_FORMAT; return -1; }
  if (dev->next - dev->ended >= (1u << dev->ring_log2)) {
    dev->error = PTA_CHIPLET_ERR_NOT_ACCEPTED;
    return -1;
  }

  // The staging buffer: activations, weights and results, each from a line.
  // It is the copy this driver's int32 interface makes necessary: until a
  // caller's own buffers can be handed to the module (the transport's open
  // item 11), the operands go through memory the driver owns.
  const uint64_t off_w = round_line(na), off_c = off_w + round_line(nb);
  const uint64_t bytes = off_c + round_line(nc * 4u);
  if (dev->stage.bytes < bytes) {
    if (dev->stage.cpu) dev->dma_free(dev->dma_ctx, &dev->stage);
    std::memset(&dev->stage, 0, sizeof dev->stage);
    if (dev->dma_alloc(dev->dma_ctx, (size_t)bytes, &dev->stage) != 0 || !dev->stage.cpu ||
        dev->stage.bytes < bytes || (dev->stage.addr % PTA_CHIPLET_LINE)) {
      if (dev->stage.cpu) dev->dma_free(dev->dma_ctx, &dev->stage);
      std::memset(&dev->stage, 0, sizeof dev->stage);
      dev->error = PTA_CHIPLET_ERR_NOT_ACCEPTED;
      return -1;
    }
  }
  uint8_t* stage = static_cast<uint8_t*>(dev->stage.cpu);
  for (uint64_t i = 0; i < na; ++i) stage[i] = (uint8_t)(int8_t)A[i];
  for (uint64_t i = 0; i < nb; ++i) stage[off_w + i] = (uint8_t)(int8_t)B[i];

  if (queue_run(dev, bank, M, N, K, dev->stage.addr, dev->stage.addr + off_w,
                dev->stage.addr + off_c) != 0)
    return -1;
  const volatile uint8_t* out = static_cast<volatile uint8_t*>(dev->stage.cpu) + off_c;
  for (uint64_t i = 0; i < nc; ++i) C[i] = (int64_t)(int32_t)ld32(out + 4u * i);
  return 0;
}

int pta_chiplet_gemm_at(pta_chiplet_device_t* dev, int bank, int M, int N, int K,
                        uint64_t act, uint64_t wgt, uint64_t out) {
  if (!dev) return -1;
  dev->refused_impairments = 0;
  dev->last_index = -1;
  if (!has_window(dev)) { dev->error = PTA_CHIPLET_ERR_NO_WINDOW; return -1; }
  if (!dev->present)    { dev->error = PTA_CHIPLET_ERR_NOT_PRESENT; return -1; }
  if (!dev->queue)      { dev->error = PTA_CHIPLET_ERR_NO_LINK; return -1; }
  if (dev->queue_dead)  { dev->error = PTA_CHIPLET_ERR_WEDGED; return -1; }
  // What the module would end NOT ISSUED is refused here, before a doorbell.
  if (M <= 0 || N <= 0 || K <= 0 || bank < 0 || bank > 0xFF ||
      (act | wgt | out) % PTA_CHIPLET_LINE != 0 ||
      dev->next - dev->ended >= (1u << dev->ring_log2)) {
    dev->error = PTA_CHIPLET_ERR_NOT_ACCEPTED;
    return -1;
  }
  return queue_run(dev, bank, M, N, K, act, wgt, out);
}

int pta_chiplet_gemm(pta_chiplet_device_t* dev, int bank, int M, int N, int K,
                     const int32_t* A, const int32_t* B, int64_t* C) {
  if (!dev) return -1;
  dev->refused_impairments = 0;
  dev->last_index = -1;
  if (!has_window(dev)) { dev->error = PTA_CHIPLET_ERR_NO_WINDOW; return -1; }
  if (!dev->present)    { dev->error = PTA_CHIPLET_ERR_NOT_PRESENT; return -1; }
  if (dev->queue) return queue_gemm(dev, bank, M, N, K, A, B, C);
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
    case PTA_CHIPLET_ERR_FORMAT:
      return "an operand does not fit the queue's format 0, whose operands are int8";
  }
  return "unknown error";
}

}  // extern "C"
