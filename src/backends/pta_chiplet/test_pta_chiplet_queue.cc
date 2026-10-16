// test_pta_chiplet_queue.cc -- the first board's queue: the module in front of
// the twin, and the driver's transport through it.
//
// docs/designs/pta_board_transport.md is the design. pta_chiplet_module.c is the
// module's half of it, a model in front of the twin; pta_chiplet.cpp's queue is
// the host's half. This holds the first to the design with descriptors written
// here by hand, and the second to the twin's own function-call path: one GEMM,
// through the queue or through the call, is one answer, bit for bit.
//
// Host memory is this process's. A line's address is its pointer, and every
// line the module writes is logged, which is how the order the design asks for
// -- results, then the record -- is seen.
//
// NOTHING HERE IS HARDWARE. A pass says the queue can be built as designed and
// that this driver uses it correctly. It says nothing about a module.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "pta_chiplet.h"

extern "C" {
#include "pta_chiplet_module.h"
#include "pta_chiplet_twin.h"
#include "pta_tile_model.h"
}

// Two transcriptions of one queue: the driver's and the module's.
static_assert(PTA_CHIPLET_PTQ_ID == PTA_MODULE_PTQ_ID && PTA_CHIPLET_PTQ_CAPS == PTA_MODULE_PTQ_CAPS &&
              PTA_CHIPLET_PTQ_CTRL == PTA_MODULE_PTQ_CTRL &&
              PTA_CHIPLET_PTQ_STATUS == PTA_MODULE_PTQ_STATUS &&
              PTA_CHIPLET_PTQ_BASE_LO == PTA_MODULE_PTQ_BASE_LO &&
              PTA_CHIPLET_PTQ_BASE_HI == PTA_MODULE_PTQ_BASE_HI &&
              PTA_CHIPLET_PTQ_SIZE == PTA_MODULE_PTQ_SIZE && PTA_CHIPLET_PTQ_TAIL == PTA_MODULE_PTQ_TAIL &&
              PTA_CHIPLET_PTQ_HEAD == PTA_MODULE_PTQ_HEAD,
              "the driver's queue registers are the module's");
static_assert((PTA_CHIPLET_PTQ_MAGIC | PTA_CHIPLET_PTQ_VERSION) == PTA_MODULE_PTQ_ID_VALUE &&
              PTA_CHIPLET_PTQ_ENABLE == PTA_MODULE_CTRL_ENABLE &&
              PTA_CHIPLET_PTQ_ENABLED == PTA_MODULE_STATUS_ENABLED &&
              PTA_CHIPLET_PTQ_HALTED == PTA_MODULE_STATUS_HALTED && PTA_CHIPLET_SLOT == PTA_MODULE_SLOT &&
              PTA_CHIPLET_LINE == PTA_MODULE_LINE && PTA_CHIPLET_D_STATUS == PTA_MODULE_D_STATUS &&
              PTA_CHIPLET_D_INDEX == PTA_MODULE_D_INDEX && PTA_CHIPLET_D_SATS == PTA_MODULE_D_SATS,
              "and so are its bits and its slot");
static_assert(PTA_CHIPLET_REC_DONE == PTA_MODULE_DONE && PTA_CHIPLET_REC_REFUSED == PTA_MODULE_REFUSED &&
              PTA_CHIPLET_REC_LOST == PTA_MODULE_LOST &&
              PTA_CHIPLET_REC_NOT_ISSUED == PTA_MODULE_NOT_ISSUED &&
              PTA_CHIPLET_REC_NOT_RUN == PTA_MODULE_NOT_RUN,
              "and how a command ends");

namespace {

int g_fail = 0, g_checks = 0;
char g_tile[16] = "";

void check(const char* name, bool ok) {
  ++g_checks;
  if (!ok) ++g_fail;
  std::printf("  %s  %s%s\n", ok ? "ok  " : "FAIL", g_tile, name);
}
void section(const char* title) { std::printf("\n%s\n", title); }

uint32_t get32(const uint8_t* p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
void put32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
void put64(uint8_t* p, uint64_t v) { put32(p, (uint32_t)v); put32(p + 4, (uint32_t)(v >> 32)); }
uint64_t addr_of(const void* p) { return (uint64_t)(uintptr_t)p; }

// ---- host memory, as the module reaches it ---------------------------------
struct Host {
  std::vector<uint64_t> writes;      // every line the module wrote, in order
  std::vector<uint64_t> reads;       // and every line it read
  uint64_t poison_lo = 0, poison_hi = 0;   // a range whose reads fail
  int msi[2] = {0, 0};
  pta_module* m = nullptr;
  uint8_t* ring = nullptr;           // for what an interrupt finds
  uint32_t ring_slots = 0;
  bool msi_found_record = true;
};

int host_read(void* ctx, uint64_t addr, uint8_t line[64]) {
  Host* h = static_cast<Host*>(ctx);
  if (addr >= h->poison_lo && addr < h->poison_hi) return -1;
  h->reads.push_back(addr);
  std::memcpy(line, reinterpret_cast<const void*>((uintptr_t)addr), 64);
  return 0;
}
int host_write(void* ctx, uint64_t addr, const uint8_t line[64]) {
  Host* h = static_cast<Host*>(ctx);
  h->writes.push_back(addr);
  std::memcpy(reinterpret_cast<void*>((uintptr_t)addr), line, 64);
  return 0;
}
void host_msi(void* ctx, unsigned v) {
  Host* h = static_cast<Host*>(ctx);
  if (v < 2) ++h->msi[v];
  if (v == 0 && h->m && h->ring) {
    // Raised when PTQ_HEAD reached PTQ_IRQ_AT: the slot before it has its record.
    const uint32_t at = pta_module_read32(h->m, PTA_MODULE_PTQ_IRQ_AT) - 1u;
    h->msi_found_record = h->msi_found_record &&
        get32(h->ring + (at % h->ring_slots) * PTA_MODULE_SLOT + PTA_MODULE_D_STATUS) != 0;
  }
}

void* aligned(size_t align, size_t bytes) {
  const size_t n = (bytes + align - 1) / align * align;
  void* p = std::aligned_alloc(align, n ? n : align);
  if (p) std::memset(p, 0, n ? n : align);
  return p;
}

// ---- the tile's configuration, as the driver gate has it ---------------------
constexpr pta_twin_build B128x64 = {128, 64, 8, 48, 4, 0};
constexpr pta_twin_build B256x64 = {256, 64, 8, 48, 4, 0};

constexpr uint32_t adc_shift_v1(const pta_twin_build& b) {
  const int64_t span = static_cast<int64_t>(b.rows) << (2 * (b.din_w - 1));
  uint32_t s = 0;
  while ((static_cast<int64_t>(64) << s) < span) ++s;
  return s;
}
uint32_t bits_v1(const pta_twin_build& b) { return 6u | (6u << 4) | (7u << 8) | (adc_shift_v1(b) << 12); }

void program_v1(pta_twin* t, const pta_twin_build& b, uint32_t seed) {
  pta_twin_write32(t, PTA_TWIN_SEED, seed);
  pta_twin_write32(t, PTA_TWIN_IMPAIR, PTA_QUANT | PTA_THERMAL | PTA_SHOT | PTA_PROG_ERR | PTA_XTALK);
  pta_twin_write32(t, PTA_TWIN_BITS, bits_v1(b));
  pta_twin_write32(t, PTA_TWIN_SIGMA_TH, 0x0080u);
  pta_twin_write32(t, PTA_TWIN_SIGMA_SH, 0x0030u);
  pta_twin_write32(t, PTA_TWIN_SIGMA_PR, 0x0100u);
  pta_twin_write32(t, PTA_TWIN_XTALK, 5u);
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

// A GEMM on a twin through its own call, run to its end: the reference.
int twin_gemm(pta_twin* t, int bank, int M, int N, int K, const std::vector<int32_t>& A,
              const std::vector<int32_t>& B, std::vector<int64_t>* C) {
  C->assign(static_cast<size_t>(M) * N, 0);
  int status = PTA_TWIN_PENDING;
  pta_twin_cmd c;
  std::memset(&c, 0, sizeof c);
  c.bank = bank; c.M = M; c.N = N; c.K = K; c.A = A.data(); c.B = B.data(); c.C = C->data();
  c.status = &status;
  if (pta_twin_submit(t, &c) != 0) return -1;
  for (int i = 0; i < 1 << 20 && status == PTA_TWIN_PENDING; ++i) pta_twin_run(t, 1u << 16);
  return status;
}

// The low 32 bits, sign-extended: what format 0 carries of a sum.
std::vector<int64_t> as_format0(const std::vector<int64_t>& c) {
  std::vector<int64_t> r(c.size());
  for (size_t i = 0; i < c.size(); ++i) r[i] = (int64_t)(int32_t)(uint32_t)(uint64_t)c[i];
  return r;
}

// ---- a ring written by hand ----------------------------------------------------
struct Ring {
  Host host;
  pta_twin* t = nullptr;
  pta_module* m = nullptr;
  uint8_t* ring = nullptr;
  uint32_t slots = 8, next = 0;
  std::vector<void*> bufs;

  Ring(const pta_twin_build& b, uint32_t log2 = 3) : slots(1u << log2) {
    t = pta_twin_new(&b);
    pta_module_host mh = {host_read, host_write, host_msi, &host};
    m = pta_module_new(t, &mh);
    ring = static_cast<uint8_t*>(aligned(4096, slots * PTA_MODULE_SLOT));
    host.m = m; host.ring = ring; host.ring_slots = slots;
    pta_module_write32(m, PTA_MODULE_PTQ_BASE_LO, (uint32_t)addr_of(ring));
    pta_module_write32(m, PTA_MODULE_PTQ_BASE_HI, (uint32_t)(addr_of(ring) >> 32));
    pta_module_write32(m, PTA_MODULE_PTQ_SIZE, log2);
    pta_module_write32(m, PTA_MODULE_PTQ_TAIL, 0);
    pta_module_write32(m, PTA_MODULE_PTQ_CTRL, PTA_MODULE_CTRL_ENABLE);
  }
  ~Ring() {
    pta_module_free(m);
    pta_twin_free(t);
    std::free(ring);
    for (void* p : bufs) std::free(p);
  }
  uint8_t* buf(size_t bytes, uint8_t fill) {
    uint8_t* p = static_cast<uint8_t*>(aligned(64, bytes));
    std::memset(p, fill, (bytes + 63) / 64 * 64);
    bufs.push_back(p);
    return p;
  }
  uint8_t* slot(uint32_t idx) { return ring + (idx % slots) * PTA_MODULE_SLOT; }
  // Write a descriptor into the next slot, status zero. Returns its index.
  uint32_t put(uint32_t sv, uint32_t target, uint32_t fmt, uint32_t M, uint32_t K, uint32_t N,
               uint64_t act, uint64_t wgt, uint64_t out) {
    uint8_t* s = slot(next);
    std::memset(s, 0, PTA_MODULE_SLOT);
    put32(s + 0x00, sv); put32(s + 0x04, target); put32(s + 0x08, fmt);
    put32(s + 0x0C, M); put32(s + 0x10, K); put32(s + 0x14, N);
    put64(s + 0x18, act); put64(s + 0x20, wgt); put64(s + 0x28, out);
    return next++;
  }
  void ring_bell() { pta_module_write32(m, PTA_MODULE_PTQ_TAIL, next); }
  void run_until_head(uint32_t head) {
    for (int i = 0; i < 1 << 16 && pta_module_read32(m, PTA_MODULE_PTQ_HEAD) != head; ++i)
      pta_module_run(m, 1u << 16);
  }
  uint32_t status(uint32_t idx) { return get32(slot(idx) + PTA_MODULE_D_STATUS); }
  uint32_t index(uint32_t idx) { return get32(slot(idx) + PTA_MODULE_D_INDEX); }
};

// A GEMM's operands as the queue carries them, and its result buffer.
struct Gemm {
  int M, N, K;
  std::vector<int32_t> A, B;
  uint8_t *act, *wgt, *out;
  size_t out_bytes;
  Gemm(Ring& r, int M_, int N_, int K_, uint32_t seed) : M(M_), N(N_), K(K_) {
    A = operands(seed, static_cast<size_t>(M) * K);
    B = operands(seed + 1, static_cast<size_t>(K) * N);
    act = r.buf(A.size(), 0);
    wgt = r.buf(B.size(), 0);
    for (size_t i = 0; i < A.size(); ++i) act[i] = (uint8_t)(int8_t)A[i];
    for (size_t i = 0; i < B.size(); ++i) wgt[i] = (uint8_t)(int8_t)B[i];
    out_bytes = (static_cast<size_t>(M) * N * 4 + 63) / 64 * 64;
    out = r.buf(out_bytes, 0xA5);
  }
  uint32_t put(Ring& r, uint32_t target = 0) {
    return r.put(PTA_MODULE_V1, target, 0, M, K, N, addr_of(act), addr_of(wgt), addr_of(out));
  }
  std::vector<int64_t> results() const {
    std::vector<int64_t> c(static_cast<size_t>(M) * N);
    for (size_t i = 0; i < c.size(); ++i) c[i] = (int64_t)(int32_t)get32(out + 4 * i);
    return c;
  }
  bool untouched() const {
    for (size_t i = 0; i < out_bytes; ++i) if (out[i] != 0xA5) return false;
    return true;
  }
  bool pad_zero() const {
    for (size_t i = static_cast<size_t>(M) * N * 4; i < out_bytes; ++i) if (out[i] != 0) return false;
    return true;
  }
  // The record came after every line of results, in the module's own writes.
  bool results_first(const Host& h, uint64_t record_addr) const {
    size_t rec = h.writes.size();
    for (size_t i = 0; i < h.writes.size(); ++i) if (h.writes[i] == record_addr) rec = i;
    if (rec == h.writes.size()) return false;
    for (size_t i = rec + 1; i < h.writes.size(); ++i)
      if (h.writes[i] >= addr_of(out) && h.writes[i] < addr_of(out) + out_bytes) return false;
    size_t lines = 0;
    for (size_t i = 0; i < rec; ++i)
      if (h.writes[i] >= addr_of(out) && h.writes[i] < addr_of(out) + out_bytes) ++lines;
    return lines == out_bytes / 64;
  }
};

// ---------------------------------------------------------------------------------
void t_page(const pta_twin_build& b) {
  section("the module's BAR: the map on page 0, the queue on page 1");
  Ring r(b);
  check("PTQ_ID is \"PTQ\", version 1; PTQ_CAPS says a ring of up to 2^10, version 1 only, two vectors",
        pta_module_read32(r.m, PTA_MODULE_PTQ_ID) == 0x50545101u &&
            pta_module_read32(r.m, PTA_MODULE_PTQ_CAPS) == (10u | (1u << 8) | (2u << 12)));
  check("page 0 is the twin's map, word for word: PTA_ID, PTA_CAPS0, and the bit that says a model",
        pta_module_read32(r.m, 0x000) == PTA_TWIN_ID_VALUE &&
            pta_module_read32(r.m, 0x004) == pta_twin_read32(r.t, 0x004) &&
            (pta_module_read32(r.m, 0x00C) & PTA_TWIN_CAPS2_TWIN));
  check("the ring as programmed reads back, ENABLED, at an empty ring",
        pta_module_read32(r.m, PTA_MODULE_PTQ_BASE_LO) == (uint32_t)addr_of(r.ring) &&
            pta_module_read32(r.m, PTA_MODULE_PTQ_SIZE) == 3u &&
            pta_module_read32(r.m, PTA_MODULE_PTQ_STATUS) == PTA_MODULE_STATUS_ENABLED &&
            pta_module_read32(r.m, PTA_MODULE_PTQ_HEAD) == 0u);
  pta_module_write32(r.m, PTA_MODULE_PTQ_SIZE, 5u);
  pta_module_write32(r.m, PTA_MODULE_PTQ_BASE_LO, 0x12345000u);
  check("BASE and SIZE are not written while the queue is enabled",
        pta_module_read32(r.m, PTA_MODULE_PTQ_SIZE) == 3u &&
            pta_module_read32(r.m, PTA_MODULE_PTQ_BASE_LO) == (uint32_t)addr_of(r.ring));
  pta_module_write32(r.m, PTA_MODULE_PTQ_CTRL, 0);
  pta_module_write32(r.m, PTA_MODULE_PTQ_SIZE, 11u);
  pta_module_write32(r.m, PTA_MODULE_PTQ_BASE_LO, 0x12345678u);
  check("and while it is not: a size past PTQ_CAPS is ignored, and a base's low 12 bits read zero",
        pta_module_read32(r.m, PTA_MODULE_PTQ_SIZE) == 3u &&
            pta_module_read32(r.m, PTA_MODULE_PTQ_BASE_LO) == 0x12345000u &&
            pta_module_read32(r.m, PTA_MODULE_PTQ_STATUS) == 0u);
  check("a word past page 1 reads zero", pta_module_read32(r.m, 0x1030) == 0u &&
                                             pta_module_read32(r.m, 0x2000) == 0u);
}

void t_done(const pta_twin_build& b) {
  section("a GEMM through the ring: the twin's answer, then its record");
  Ring r(b);
  pta_twin* ref = pta_twin_new(&b);
  program_v1(r.t, b, 0x51Du);
  program_v1(ref, b, 0x51Du);

  Gemm g(r, 3, 5, 10, 7);
  const uint32_t i0 = g.put(r);
  uint8_t written[0x30];
  std::memcpy(written, r.slot(i0), sizeof written);
  r.ring_bell();
  r.run_until_head(1);
  std::vector<int64_t> want;
  const int st = twin_gemm(ref, 0, g.M, g.N, g.K, g.A, g.B, &want);
  check("it ends DONE as GEMM 0, and its 15 results are the twin's own call's, low 32 bits",
        st == PTA_TWIN_DONE && r.status(i0) == PTA_MODULE_DONE && r.index(i0) == 0 &&
            g.results() == as_format0(want));
  check("the record keeps every field the host wrote, and PTQ_HEAD has moved past it",
        std::memcmp(r.slot(i0), written, sizeof written) == 0 &&
            pta_module_read32(r.m, PTA_MODULE_PTQ_HEAD) == 1u);
  check("the last line's pad is zeros", g.pad_zero());
  check("and the record went out after the line of results", g.results_first(r.host, addr_of(r.slot(i0))));

  // Results over five lines, from a K that walks two of the tile's K tiles.
  Gemm h(r, 4, 20, b.rows + 9, 11);
  const uint32_t i1 = h.put(r);
  r.ring_bell();
  r.run_until_head(2);
  twin_gemm(ref, 0, h.M, h.N, h.K, h.A, h.B, &want);
  check("a K of two tiles and results over five lines: the twin's answer, GEMM 1",
        r.status(i1) == PTA_MODULE_DONE && r.index(i1) == 1 && h.results() == as_format0(want));
  check("all five lines of results went out before the record", h.results_first(r.host, addr_of(r.slot(i1))));
  pta_twin_free(ref);
}

void t_endings(const pta_twin_build& b) {
  section("refused, not run, not issued: no index, and the results untouched");
  Ring r(b);
  program_v1(r.t, b, 0xBEEu);
  Gemm g(r, 2, 4, 6, 21);

  const uint32_t ref = g.put(r, 2u << 8);        // a bank the tile does not have
  r.ring_bell();
  r.run_until_head(ref + 1);
  check("bank 2 on a two-bank tile: REFUSED by the tile, no index, the results untouched",
        r.status(ref) == PTA_MODULE_REFUSED && r.index(ref) == PTA_MODULE_NO_INDEX && g.untouched() &&
            pta_twin_read32(r.t, PTA_TWIN_GEMM_CT) == 0u);

  const uint32_t nr = g.put(r, PTA_MODULE_AFTER_SUCCESS);
  r.ring_bell();
  r.run_until_head(nr + 1);
  check("bit 16 after the refusal: NOT RUN, no index, untouched",
        r.status(nr) == PTA_MODULE_NOT_RUN && r.index(nr) == PTA_MODULE_NO_INDEX && g.untouched());
  const uint32_t ok = g.put(r);
  const uint32_t after = g.put(r, PTA_MODULE_AFTER_SUCCESS);
  r.ring_bell();                                  // two slots on one doorbell
  r.run_until_head(after + 1);
  check("then a plain one runs, and bit 16 after a success runs too: GEMMs 0 and 1",
        r.status(ok) == PTA_MODULE_DONE && r.index(ok) == 0 && r.status(after) == PTA_MODULE_DONE &&
            r.index(after) == 1);

  struct Bad { const char* what; uint32_t sv, target, fmt, M, K, N; uint64_t act; };
  const uint64_t a = addr_of(g.act);
  const Bad bad[] = {
      {"a version 2 descriptor, which this module does not take", (2u << 16) | 80u, 0, 0, 2, 6, 4, a},
      {"a size and version of zero", 0, 0, 0, 2, 6, 4, a},
      {"format 7", PTA_MODULE_V1, 0, 7, 2, 6, 4, a},
      {"instance 1, on a module of one", PTA_MODULE_V1, 1, 0, 2, 6, 4, a},
      {"a flag it does not know, bit 20", PTA_MODULE_V1, 1u << 20, 0, 2, 6, 4, a},
      {"M of zero", PTA_MODULE_V1, 0, 0, 0, 6, 4, a},
      {"an address off a line", PTA_MODULE_V1, 0, 0, 2, 6, 4, a + 8},
      {"results past the model's bound", PTA_MODULE_V1, 0, 0, 1u << 13, 6, 1u << 12, a},
  };
  for (const Bad& x : bad) {
    std::memset(g.out, 0xA5, g.out_bytes);
    const uint32_t ct = pta_twin_read32(r.t, PTA_TWIN_GEMM_CT);
    const uint32_t i = r.put(x.sv, x.target, x.fmt, x.M, x.K, x.N, x.act, addr_of(g.wgt), addr_of(g.out));
    r.ring_bell();
    char name[200];
    std::snprintf(name, sizeof name, "%-56s NOT ISSUED", x.what);
    check(name, r.status(i) == PTA_MODULE_NOT_ISSUED && r.index(i) == PTA_MODULE_NO_INDEX &&
                    g.untouched() && pta_module_read32(r.m, PTA_MODULE_PTQ_HEAD) == i + 1 &&
                    pta_twin_read32(r.t, PTA_TWIN_GEMM_CT) == ct);
  }
}

void t_wrap(const pta_twin_build& b) {
  section("the ring wraps, and an interrupt finds its record");
  Ring r(b, 2);                                   // four slots
  pta_twin* ref = pta_twin_new(&b);
  program_v1(r.t, b, 0xA11u);
  program_v1(ref, b, 0xA11u);
  bool all = true;
  for (uint32_t k = 0; k < 11; ++k) {
    Gemm g(r, 2, 3, 5, 100 + k);
    const uint32_t i = g.put(r);
    r.ring_bell();
    r.run_until_head(i + 1);
    std::vector<int64_t> want;
    twin_gemm(ref, 0, g.M, g.N, g.K, g.A, g.B, &want);
    all = all && r.status(i) == PTA_MODULE_DONE && r.index(i) == k && g.results() == as_format0(want);
  }
  check("eleven GEMMs through four slots: each the twin's, as GEMMs 0 to 10", all);

  pta_module_write32(r.m, PTA_MODULE_PTQ_CTRL, PTA_MODULE_CTRL_ENABLE | PTA_MODULE_CTRL_IRQ_EN);
  Gemm g(r, 2, 3, 5, 300);
  const uint32_t i = g.put(r);
  pta_module_write32(r.m, PTA_MODULE_PTQ_IRQ_AT, i + 1);
  r.ring_bell();
  r.run_until_head(i + 1);
  check("vector 0, once, when PTQ_HEAD reached PTQ_IRQ_AT, and the slot had its record by then",
        r.host.msi[0] == 1 && r.host.msi_found_record && r.status(i) == PTA_MODULE_DONE);

  pta_twin_write32(r.t, PTA_TWIN_IRQ_MASK, PTA_TWIN_IRQ_ERR);
  const uint32_t j = g.put(r, 2u << 8);
  r.ring_bell();
  r.run_until_head(j + 1);
  check("vector 1 when the map's PTA_IRQ_STATUS rises: a refusal raises ERR",
        r.host.msi[1] == 1 && r.status(j) == PTA_MODULE_REFUSED);
  pta_twin_free(ref);
}

void t_halt(const pta_twin_build& b) {
  section("a doorbell that makes no sense, and memory that fails, halt the queue");
  Ring r(b);
  Gemm g(r, 2, 3, 4, 41);
  g.put(r);
  pta_module_write32(r.m, PTA_MODULE_PTQ_TAIL, 9);   // past an eight-slot ring
  check("a doorbell more than the ring past PTQ_HEAD: HALTED, reason 1, at 0, and nothing taken",
        pta_module_read32(r.m, PTA_MODULE_PTQ_STATUS) == (1u | 2u | (1u << 8)) &&
            pta_module_read32(r.m, PTA_MODULE_PTQ_FAULT_AT) == 0u && r.status(0) == 0 &&
            pta_twin_pending(r.t) == 0);
  pta_module_write32(r.m, PTA_MODULE_PTQ_CTRL, 0);
  pta_module_write32(r.m, PTA_MODULE_PTQ_TAIL, 0);
  pta_module_write32(r.m, PTA_MODULE_PTQ_CTRL, PTA_MODULE_CTRL_ENABLE);
  check("enabling it again clears the halt", pta_module_read32(r.m, PTA_MODULE_PTQ_STATUS) == 1u);
  r.next = 0;
  g.put(r);
  r.ring_bell();
  r.run_until_head(1);
  pta_module_write32(r.m, PTA_MODULE_PTQ_TAIL, 0);
  check("a doorbell that goes back: HALTED, reason 1",
        r.status(0) == PTA_MODULE_DONE && (pta_module_read32(r.m, PTA_MODULE_PTQ_STATUS) >> 8) == 1u);

  Ring s(b);
  Gemm h(s, 2, 3, 4, 43);
  s.host.poison_lo = addr_of(h.wgt);
  s.host.poison_hi = addr_of(h.wgt) + 64;
  h.put(s);
  s.ring_bell();
  check("an operand's line that cannot be read: HALTED, reason 2, the slot not ended",
        (pta_module_read32(s.m, PTA_MODULE_PTQ_STATUS) >> 8) == 2u && s.status(0) == 0 &&
            pta_twin_pending(s.t) == 0);
  // Enabled again, the ring starts empty at index 1; the slot it reads next is 1.
  s.host.poison_lo = addr_of(s.slot(1));
  s.host.poison_hi = addr_of(s.slot(1)) + 64;
  pta_module_write32(s.m, PTA_MODULE_PTQ_CTRL, 0);
  pta_module_write32(s.m, PTA_MODULE_PTQ_CTRL, PTA_MODULE_CTRL_ENABLE);
  pta_module_write32(s.m, PTA_MODULE_PTQ_TAIL, 2);
  check("and a slot that cannot be read: HALTED, reason 2, at index 1",
        (pta_module_read32(s.m, PTA_MODULE_PTQ_STATUS) >> 8) == 2u &&
            pta_module_read32(s.m, PTA_MODULE_PTQ_FAULT_AT) == 1u);
}

void t_teardown(const pta_twin_build& b) {
  section("a teardown drops what is running, and nothing of it is written");
  Ring r(b);
  program_v1(r.t, b, 0x7EAu);
  pta_twin_write32(r.t, PTA_TWIN_TS, 5000);      // long enough to be caught running
  Gemm g(r, 8, 6, 12, 51);
  const uint32_t i = g.put(r);
  r.ring_bell();
  pta_module_run(r.m, 1000);
  const bool running = pta_twin_pending(r.t) == 1 && r.status(i) == 0;
  pta_module_write32(r.m, PTA_MODULE_PTQ_CTRL, 0);
  const bool stopped = (pta_module_read32(r.m, PTA_MODULE_PTQ_STATUS) & 1u) == 0;
  put32(r.slot(i) + PTA_MODULE_D_STATUS, PTA_MODULE_LOST);       // the host's, now
  const size_t writes = r.host.writes.size();
  for (int k = 0; k < 64 && pta_twin_pending(r.t); ++k) pta_module_run(r.m, 1u << 16);
  check("caught running, ENABLED reads 0 at once, and the host ends the slot lost",
        running && stopped);
  check("the tile ran the GEMM out and the module wrote nothing of it: still lost, results untouched",
        pta_twin_pending(r.t) == 0 && r.host.writes.size() == writes &&
            r.status(i) == PTA_MODULE_LOST && g.untouched() &&
            pta_twin_read32(r.t, PTA_TWIN_GEMM_CT) == 1u);
  pta_module_write32(r.m, PTA_MODULE_PTQ_TAIL, r.next);
  pta_module_write32(r.m, PTA_MODULE_PTQ_CTRL, PTA_MODULE_CTRL_ENABLE);
  pta_twin_write32(r.t, PTA_TWIN_TS, 1);
  Gemm h(r, 2, 3, 4, 53);
  const uint32_t j = h.put(r);
  r.ring_bell();
  r.run_until_head(j + 1);
  check("enabled again at an empty ring, the next GEMM is GEMM 1: the dropped one used an index",
        pta_module_read32(r.m, PTA_MODULE_PTQ_HEAD) == j + 1 && r.status(j) == PTA_MODULE_DONE &&
            r.index(j) == 1u);

  // The orphan holds the next slot back until the tile has finished with it.
  pta_twin_write32(r.t, PTA_TWIN_TS, 5000);
  const uint32_t k = g.put(r);
  r.ring_bell();
  pta_module_run(r.m, 1000);
  pta_module_write32(r.m, PTA_MODULE_PTQ_CTRL, 0);
  put32(r.slot(k) + PTA_MODULE_D_STATUS, PTA_MODULE_LOST);
  pta_module_write32(r.m, PTA_MODULE_PTQ_TAIL, r.next);
  pta_module_write32(r.m, PTA_MODULE_PTQ_CTRL, PTA_MODULE_CTRL_ENABLE);
  const uint32_t l = h.put(r);
  r.ring_bell();
  const bool held = r.status(l) == 0 && pta_twin_pending(r.t) == 1;
  r.run_until_head(l + 1);
  check("a slot published behind a dropped GEMM waits for the tile, then runs as the index after it",
        held && r.status(l) == PTA_MODULE_DONE && r.index(l) == 3u && r.status(k) == PTA_MODULE_LOST);

  // A function-level reset, in the middle of a GEMM.
  const uint32_t q = g.put(r);
  r.ring_bell();
  pta_module_run(r.m, 1000);
  const size_t w = r.host.writes.size();
  pta_module_reset(r.m);
  check("a reset: every queue register zero, the tile's GEMM count zero, and nothing written",
        pta_module_read32(r.m, PTA_MODULE_PTQ_STATUS) == 0u &&
            pta_module_read32(r.m, PTA_MODULE_PTQ_SIZE) == 0u &&
            pta_module_read32(r.m, PTA_MODULE_PTQ_HEAD) == 0u &&
            pta_twin_read32(r.t, PTA_TWIN_GEMM_CT) == 0u && pta_twin_pending(r.t) == 0 &&
            r.host.writes.size() == w && r.status(q) == 0);
}

// ---- the driver, through the queue ----------------------------------------------
struct Dma {
  int allocs = 0, frees = 0;
};
int dma_alloc(void* ctx, size_t bytes, pta_chiplet_dma_t* out) {
  void* p = aligned(4096, bytes);
  if (!p) return -1;
  ++static_cast<Dma*>(ctx)->allocs;
  out->cpu = p;
  out->addr = addr_of(p);
  out->bytes = (bytes + 4095) / 4096 * 4096;
  return 0;
}
void dma_free(void* ctx, pta_chiplet_dma_t* buf) {
  ++static_cast<Dma*>(ctx)->frees;
  std::free(buf->cpu);
  buf->cpu = nullptr;
}

// The window: the module's BAR. A read of either status word is a poll, and
// that is where the model's time passes.
struct Board {
  Host host;
  pta_twin* t = nullptr;
  pta_module* m = nullptr;
  bool frozen = false;      // the module's time stands still
  bool deaf = false;        // and it does not hear PTQ_CTRL either
  long status_reads = 0;    // reads of PTQ_STATUS: the driver's polls
};
uint32_t board_read(void* ctx, uint32_t off) {
  Board* b = static_cast<Board*>(ctx);
  if (off == PTA_CHIPLET_PTQ_STATUS) ++b->status_reads;
  if (!b->frozen && (off == PTA_CHIPLET_STATUS || off == PTA_CHIPLET_PTQ_STATUS))
    pta_module_run(b->m, 1u << 16);
  return pta_module_read32(b->m, off);
}
void board_write(void* ctx, uint32_t off, uint32_t v) {
  Board* b = static_cast<Board*>(ctx);
  if (b->deaf && off == PTA_CHIPLET_PTQ_CTRL) return;
  pta_module_write32(b->m, off, v);
}

uint32_t twin_read(void* ctx, uint32_t off) {
  pta_twin* t = static_cast<pta_twin*>(ctx);
  if (off == PTA_CHIPLET_STATUS) pta_twin_run(t, 1u << 16);
  return pta_twin_read32(t, off);
}
void twin_write(void* ctx, uint32_t off, uint32_t v) { pta_twin_write32(static_cast<pta_twin*>(ctx), off, v); }
int twin_submit(void* ctx, int bank, int M, int N, int K, const int32_t* A, const int32_t* B,
                int64_t* C, int* status) {
  pta_twin_cmd c;
  std::memset(&c, 0, sizeof c);
  c.bank = bank; c.M = M; c.N = N; c.K = K; c.A = A; c.B = B; c.C = C; c.status = status;
  return pta_twin_submit(static_cast<pta_twin*>(ctx), &c);
}

const int64_t POISON = 0x5A5A5A5A5A5A5A5All;
bool all_poison(const std::vector<int64_t>& c) {
  for (int64_t v : c) if (v != POISON) return false;
  return true;
}

void board_open(Board* bd, const pta_twin_build& b) {
  bd->t = pta_twin_new(&b);
  pta_module_host mh = {host_read, host_write, host_msi, &bd->host};
  bd->m = pta_module_new(bd->t, &mh);
}
void board_close(Board* bd) {
  pta_module_free(bd->m);
  pta_twin_free(bd->t);
}

void t_driver(const pta_twin_build& b) {
  section("the driver, through the queue: the twin's answer, as through its call");
  Board bd;
  board_open(&bd, b);
  Dma dma;
  pta_chiplet_device_t dev;
  pta_chiplet_attach_window(&dev, board_read, board_write, &bd);
  const bool found = pta_chiplet_detect(&dev) == 1;
  check("the module's page 0 is a tile the driver knows, and its queue attaches on page 1",
        found && pta_chiplet_attach_queue(&dev, dma_alloc, dma_free, &dma, 4) == 1 && dev.queue == 1 &&
            dma.allocs == 1 && pta_module_read32(bd.m, PTA_MODULE_PTQ_STATUS) == 1u);

  pta_twin* plain = pta_twin_new(&b);
  pta_chiplet_device_t other;
  pta_chiplet_attach_window(&other, twin_read, twin_write, plain);
  pta_chiplet_detect(&other);
  Dma none;
  check("a chiplet behind a GPU has no page 1: no queue, NO LINK, and nothing allocated",
        pta_chiplet_attach_queue(&other, dma_alloc, dma_free, &none, 4) == 0 &&
            other.error == PTA_CHIPLET_ERR_NO_LINK && none.allocs == 0 && other.queue == 0);
  Board big;
  board_open(&big, b);
  pta_chiplet_device_t over;
  pta_chiplet_attach_window(&over, board_read, board_write, &big);
  check("a ring larger than PTQ_CAPS allows is not attached",
        pta_chiplet_attach_queue(&over, dma_alloc, dma_free, &none, 11) == 0 && none.allocs == 0);
  board_close(&big);

  // The same three GEMMs through the queue and through the twin's own call,
  // from the same programming: one answer, and the same indices.
  pta_chiplet_attach_link(&other, twin_submit, plain);
  program_v1(bd.t, b, 0x51Du);
  program_v1(plain, b, 0x51Du);
  const int M = 4, N = b.cols + 6, K = b.rows + 44;
  const auto A = operands(1, static_cast<size_t>(M) * K);
  const auto B = operands(2, static_cast<size_t>(K) * N);
  bool same = true, indices = true;
  for (int g = 0; g < 3; ++g) {
    std::vector<int64_t> Cq(static_cast<size_t>(M) * N, POISON), Cl(Cq.size(), POISON);
    const int64_t before = pta_twin_read32(bd.t, PTA_TWIN_GEMM_CT);
    const int rq = pta_chiplet_gemm(&dev, 0, M, N, K, A.data(), B.data(), Cq.data());
    const int rl = pta_chiplet_gemm(&other, 0, M, N, K, A.data(), B.data(), Cl.data());
    same = same && rq == 0 && rl == 0 && Cq == as_format0(Cl);
    indices = indices && dev.last_index == g && before == g && other.last_index == -1;
  }
  char name[200];
  std::snprintf(name, sizeof name,
                "three GEMMs, %d x %d x %d, through the queue: the call's answer, bit for bit", M, N, K);
  check(name, same);
  check("each ran as the index its record says, 0, 1 and 2, which PTA_GEMM_CT read before it", indices);
  check("and the tile walked each as the call's did: 12 programmings, 48 shots",
        pta_twin_read32(bd.t, PTA_TWIN_WLOAD_CT) == 12u && pta_twin_read32(bd.t, PTA_TWIN_SHOT_CT) == 48u);

  // It waits out a calibration in the tile, as the link path does.
  pta_twin_write32(bd.t, PTA_TWIN_TS, 1000);
  pta_twin_write32(bd.t, PTA_TWIN_TRIM, 2u | (0x4000u << 16));
  pta_twin_write32(bd.t, PTA_TWIN_CAL_CFG, 6u | (1u << 4) | (2u << 8));
  pta_twin_write32(bd.t, PTA_TWIN_CTRL, PTA_TWIN_CTRL_EN | PTA_TWIN_CTRL_CAL_NOW);
  std::vector<int64_t> C(static_cast<size_t>(M) * N, POISON);
  const int rc = pta_chiplet_gemm(&dev, 0, M, N, K, A.data(), B.data(), C.data());
  check("asked for during a calibration, it waits in the ring and then runs, as GEMM 3",
        rc == 0 && !all_poison(C) && dev.last_index == 3 && pta_twin_read32(bd.t, PTA_TWIN_CAL_CT) == 1u);
  pta_twin_write32(bd.t, PTA_TWIN_TS, 1);

  section("through the queue, when it cannot run: C untouched, and the error says why");
  const int m2 = 2, n2 = 8, k2 = 8;
  const auto A2 = operands(3, static_cast<size_t>(m2) * k2);
  const auto B2 = operands(4, static_cast<size_t>(k2) * n2);
  std::vector<int64_t> C2(static_cast<size_t>(m2) * n2, POISON);
  pta_twin_write32(bd.t, PTA_TWIN_IMPAIR, PTA_QUANT | PTA_MZM_NL | PTA_THERMAL);
  int r = pta_chiplet_gemm(&dev, 0, m2, n2, k2, A2.data(), B2.data(), C2.data());
  check("an impairment the tile does not build: REFUSED, and the driver names the bit, 0x20",
        r == -1 && dev.error == PTA_CHIPLET_ERR_REFUSED && dev.refused_impairments == 0x20u &&
            all_poison(C2) && dev.last_index == -1);
  pta_twin_write32(bd.t, PTA_TWIN_IMPAIR, PTA_QUANT);
  r = pta_chiplet_gemm(&dev, 2, m2, n2, k2, A2.data(), B2.data(), C2.data());
  check("bank 2: REFUSED, and no impairment is blamed for it",
        r == -1 && dev.error == PTA_CHIPLET_ERR_REFUSED && dev.refused_impairments == 0u && all_poison(C2));

  auto big_a = A2;
  big_a[5] = 128;
  const uint32_t tail = pta_module_read32(bd.m, PTA_MODULE_PTQ_TAIL);
  r = pta_chiplet_gemm(&dev, 0, m2, n2, k2, big_a.data(), B2.data(), C2.data());
  check("an operand of 128, which int8 does not hold: FORMAT, and no doorbell rung",
        r == -1 && dev.error == PTA_CHIPLET_ERR_FORMAT && all_poison(C2) &&
            pta_module_read32(bd.m, PTA_MODULE_PTQ_TAIL) == tail);

  // Memory that fails: the module halts reading the operands, and the driver
  // tears the queue down and ends the GEMM lost. Then it runs again.
  bd.host.poison_lo = dev.stage.addr;
  bd.host.poison_hi = dev.stage.addr + 64;
  const long polls = bd.status_reads;
  r = pta_chiplet_gemm(&dev, 0, m2, n2, k2, A2.data(), B2.data(), C2.data());
  const long polled = bd.status_reads - polls;
  const uint32_t head = pta_module_read32(bd.m, PTA_MODULE_PTQ_HEAD);
  const uint8_t* lost_slot = static_cast<const uint8_t*>(dev.ring.cpu) +
                             ((dev.next - 1u) & ((1u << dev.ring_log2) - 1u)) * PTA_CHIPLET_SLOT;
  check("a queue that halts on the operands: torn down, LOST, C untouched, and the ring empty again",
        r == -1 && dev.error == PTA_CHIPLET_ERR_LOST && all_poison(C2) &&
            pta_module_read32(bd.m, PTA_MODULE_PTQ_STATUS) == 1u && head == dev.next &&
            dev.ended == dev.next);
  check("and the slot it was in says so: the host wrote LOST into its record",
        get32(lost_slot + PTA_CHIPLET_D_STATUS) == PTA_CHIPLET_REC_LOST);
  check("and the halt was seen at the first poll, not waited out: a handful of PTQ_STATUS reads",
        polled > 0 && polled < 16);
  bd.host.poison_lo = bd.host.poison_hi = 0;
  r = pta_chiplet_gemm(&dev, 0, m2, n2, k2, A2.data(), B2.data(), C2.data());
  check("and the next GEMM runs", r == 0 && dev.error == PTA_CHIPLET_OK && !all_poison(C2));

  // A module whose time has stopped: the GEMM never ends, and a teardown ends it.
  std::fill(C2.begin(), C2.end(), POISON);
  bd.frozen = true;
  r = pta_chiplet_gemm(&dev, 0, m2, n2, k2, A2.data(), B2.data(), C2.data());
  check("a GEMM that never ends: the wait runs out, the queue is torn down, and it is LOST",
        r == -1 && dev.error == PTA_CHIPLET_ERR_LOST && all_poison(C2) && dev.queue_dead == 0);
  bd.frozen = false;
  for (int k = 0; k < 64 && pta_twin_pending(bd.t); ++k) pta_module_run(bd.m, 1u << 16);
  r = pta_chiplet_gemm(&dev, 0, m2, n2, k2, A2.data(), B2.data(), C2.data());
  check("and once the tile is free of it, the next one runs", r == 0 && !all_poison(C2));

  // One that will not stop: the queue is dead, and nothing more is sent.
  std::fill(C2.begin(), C2.end(), POISON);
  bd.frozen = bd.deaf = true;
  r = pta_chiplet_gemm(&dev, 0, m2, n2, k2, A2.data(), B2.data(), C2.data());
  const uint32_t tail2 = pta_module_read32(bd.m, PTA_MODULE_PTQ_TAIL);
  const int r2 = pta_chiplet_gemm(&dev, 0, m2, n2, k2, A2.data(), B2.data(), C2.data());
  check("a module that never says it has stopped: WEDGED, the queue dead, and the next not sent",
        r == -1 && dev.error == PTA_CHIPLET_ERR_WEDGED && dev.queue_dead == 1 && r2 == -1 &&
            dev.error == PTA_CHIPLET_ERR_WEDGED && pta_module_read32(bd.m, PTA_MODULE_PTQ_TAIL) == tail2 &&
            all_poison(C2));

  pta_chiplet_detach_queue(&dev);
  check("detached, every buffer it took is given back", dma.allocs == dma.frees && dev.queue == 0);
  bool named = true;
  for (int e = PTA_CHIPLET_OK; e <= PTA_CHIPLET_ERR_FORMAT; ++e)
    named = named && std::strcmp(pta_chiplet_error_string(e), "unknown error") != 0;
  check("every error has a sentence, FORMAT among them", named);
  pta_twin_free(plain);
  board_close(&bd);
}

// How many of a log's lines fell inside [lo, lo + n).
size_t lines_in(const std::vector<uint64_t>& log, uint64_t lo, size_t n) {
  size_t k = 0;
  for (uint64_t a : log) k += (a >= lo && a < lo + n);
  return k;
}

void t_in_place(const pta_twin_build& b) {
  section("the driver, in place: the caller's own buffers, and nothing copied");
  Board bd;
  board_open(&bd, b);
  Dma dma;
  pta_chiplet_device_t dev;
  pta_chiplet_attach_window(&dev, board_read, board_write, &bd);
  pta_chiplet_detect(&dev);
  pta_chiplet_attach_queue(&dev, dma_alloc, dma_free, &dma, 4);
  pta_twin* ref = pta_twin_new(&b);
  program_v1(bd.t, b, 0x1A9u);
  program_v1(ref, b, 0x1A9u);

  // Three buffers of the caller's, in memory the module reaches: int8 operands
  // and int32 results, row-major, as format 0 has them.
  const int M = 3, N = b.cols + 5, K = b.rows + 20;
  const auto A = operands(61, static_cast<size_t>(M) * K);
  const auto B = operands(62, static_cast<size_t>(K) * N);
  pta_chiplet_dma_t act, wgt, out;
  dma_alloc(&dma, A.size(), &act);
  dma_alloc(&dma, B.size(), &wgt);
  dma_alloc(&dma, static_cast<size_t>(M) * N * 4, &out);
  for (size_t i = 0; i < A.size(); ++i) static_cast<uint8_t*>(act.cpu)[i] = (uint8_t)(int8_t)A[i];
  for (size_t i = 0; i < B.size(); ++i) static_cast<uint8_t*>(wgt.cpu)[i] = (uint8_t)(int8_t)B[i];
  std::memset(out.cpu, 0xA5, out.bytes);
  const int allocs = dma.allocs;
  bd.host.reads.clear();
  bd.host.writes.clear();
  int rc = pta_chiplet_gemm_at(&dev, 0, M, N, K, act.addr, wgt.addr, out.addr);
  std::vector<int64_t> want;
  twin_gemm(ref, 0, M, N, K, A, B, &want);
  bool same = rc == 0;
  for (size_t i = 0; same && i < want.size(); ++i)
    same = (int64_t)(int32_t)get32(static_cast<uint8_t*>(out.cpu) + 4 * i) == as_format0(want)[i];
  char name[200];
  std::snprintf(name, sizeof name,
                "a GEMM at three addresses the caller holds, %d x %d x %d: the call's answer, in its buffer",
                M, N, K);
  check(name, same && dev.last_index == 0);
  check("the module read the caller's operands and wrote the caller's results, and nothing else",
        lines_in(bd.host.reads, act.addr, A.size()) > 0 && lines_in(bd.host.reads, wgt.addr, B.size()) > 0 &&
            lines_in(bd.host.writes, out.addr, out.bytes) > 0 &&
            lines_in(bd.host.writes, out.addr, out.bytes) + 1 == bd.host.writes.size());
  check("and the driver staged nothing: it took no memory of its own",
        dma.allocs == allocs && dev.stage.cpu == nullptr);

  const uint32_t tail = pta_module_read32(bd.m, PTA_MODULE_PTQ_TAIL);
  rc = pta_chiplet_gemm_at(&dev, 0, M, N, K, act.addr + 8, wgt.addr, out.addr);
  check("an address off a line: NOT ACCEPTED, and no doorbell rung",
        rc == -1 && dev.error == PTA_CHIPLET_ERR_NOT_ACCEPTED &&
            pta_module_read32(bd.m, PTA_MODULE_PTQ_TAIL) == tail);
  std::memset(out.cpu, 0xA5, out.bytes);
  pta_twin_write32(bd.t, PTA_TWIN_IMPAIR, PTA_QUANT | PTA_MZM_NL);
  rc = pta_chiplet_gemm_at(&dev, 0, M, N, K, act.addr, wgt.addr, out.addr);
  bool untouched = true;
  for (size_t i = 0; i < out.bytes; ++i) untouched = untouched && static_cast<uint8_t*>(out.cpu)[i] == 0xA5;
  check("refused in place: REFUSED, the bit named, and the caller's results untouched",
        rc == -1 && dev.error == PTA_CHIPLET_ERR_REFUSED && dev.refused_impairments == 0x20u && untouched);
  pta_twin_write32(bd.t, PTA_TWIN_IMPAIR, PTA_QUANT);
  bd.host.poison_lo = act.addr;
  bd.host.poison_hi = act.addr + 64;
  rc = pta_chiplet_gemm_at(&dev, 0, M, N, K, act.addr, wgt.addr, out.addr);
  check("a halt in place: torn down and LOST, where what the results hold is undefined",
        rc == -1 && dev.error == PTA_CHIPLET_ERR_LOST && pta_module_read32(bd.m, PTA_MODULE_PTQ_STATUS) == 1u);
  bd.host.poison_lo = bd.host.poison_hi = 0;
  rc = pta_chiplet_gemm_at(&dev, 0, M, N, K, act.addr, wgt.addr, out.addr);
  check("and the next GEMM in place runs", rc == 0 && dev.error == PTA_CHIPLET_OK);

  pta_chiplet_device_t linkless;
  pta_chiplet_attach_window(&linkless, board_read, board_write, &bd);
  pta_chiplet_detect(&linkless);
  rc = pta_chiplet_gemm_at(&linkless, 0, M, N, K, act.addr, wgt.addr, out.addr);
  check("with no queue there is no way to the tile: NO LINK", rc == -1 && linkless.error == PTA_CHIPLET_ERR_NO_LINK);

  dma_free(&dma, &act);
  dma_free(&dma, &wgt);
  dma_free(&dma, &out);
  pta_chiplet_detach_queue(&dev);
  check("detached, every buffer is given back", dma.allocs == dma.frees);
  pta_twin_free(ref);
  board_close(&bd);
}

}  // namespace

int main() {
  std::printf("PTA module's queue, and the driver through it, against the twin. A model, not a module.\n");
  for (const pta_twin_build* b : {&B128x64, &B256x64}) {
    std::snprintf(g_tile, sizeof g_tile, "%3d x %-3d ", b->rows, b->cols);
    t_page(*b);
    t_done(*b);
    t_endings(*b);
    t_wrap(*b);
    t_halt(*b);
    t_teardown(*b);
    t_driver(*b);
    t_in_place(*b);
  }
  std::printf("\n%d checks, %d failed\n", g_checks, g_fail);
  std::printf("%s\n", g_fail ? "FAILED" : "PASSED");
  return g_fail ? 1 : 0;
}
