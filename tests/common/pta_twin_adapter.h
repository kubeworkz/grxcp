// The PTA chiplet's digital twin, as the device the runtime enumerates.
//
// src/backends/pta_chiplet/pta_chiplet_twin.c presents the chiplet's register
// map with grx930's error model behind it. This installs one through the seam
// in pta_chiplet_testing.h, so that grxGetDeviceProperties, grxSetDevice and
// grxblasGemmEx run against a PTA device on a machine that has none -- which is
// every machine, because there is no chiplet.
//
// Two things sit between the twin and the driver's hooks, and both are the
// twin's shape rather than a workaround:
//
//   1. Its clock only advances inside pta_twin_run(), which no host calls. So a
//      read of PTA_STATUS pumps it: that read is where a host's completion poll
//      spends its time, and the driver polls exactly as it would across a link.
//   2. Its command path is a function call, which is what the driver's link
//      hook is too. The two status codes are the same numbers by construction,
//      and the static asserts below are what hold them there.
//
// A MODEL IS NOT HARDWARE. Anything green through this says the runtime and the
// driver use the map correctly and that the answer is the error model's. It
// says nothing about a chiplet, and no result obtained here may be reported as
// one working.

#ifndef GRXCP_TESTS_PTA_TWIN_ADAPTER_H
#define GRXCP_TESTS_PTA_TWIN_ADAPTER_H

#include "pta_chiplet.h"
#include "pta_chiplet_testing.h"

extern "C" {
#include "pta_chiplet_module.h"
#include "pta_chiplet_twin.h"
#include "pta_tile_model.h"
}

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static_assert(PTA_CHIPLET_CMD_PENDING == PTA_TWIN_PENDING &&
              PTA_CHIPLET_CMD_DONE == PTA_TWIN_DONE &&
              PTA_CHIPLET_CMD_REFUSED == PTA_TWIN_REFUSED &&
              PTA_CHIPLET_CMD_LOST == PTA_TWIN_LOST,
              "the driver's link results are the twin's");

namespace grxtest {

// The twin behind the device, for a test that programs it the way another
// holder of the chiplet would: behind the runtime's back.
inline pta_twin*& pta_twin_installed() {
  static pta_twin* t = nullptr;
  return t;
}

// How many of the twin's cycles one PTA_STATUS read advances.
inline uint64_t& pta_twin_quantum() {
  static uint64_t q = 1u << 16;
  return q;
}

inline uint32_t pta_twin_read_hook(void* ctx, uint32_t off) {
  pta_twin* t = static_cast<pta_twin*>(ctx);
  if (off == PTA_CHIPLET_STATUS) pta_twin_run(t, pta_twin_quantum());
  return pta_twin_read32(t, off);
}

inline void pta_twin_write_hook(void* ctx, uint32_t off, uint32_t v) {
  pta_twin_write32(static_cast<pta_twin*>(ctx), off, v);
}

inline int pta_twin_submit_hook(void* ctx, int bank, int M, int N, int K,
                                const int32_t* A, const int32_t* B, int64_t* C,
                                int* status) {
  pta_twin_cmd c;
  std::memset(&c, 0, sizeof(c));
  c.bank = bank;
  c.M = M;
  c.N = N;
  c.K = K;
  c.A = A;
  c.B = B;
  c.C = C;
  c.status = status;
  return pta_twin_submit(static_cast<pta_twin*>(ctx), &c);
}

// WHICH TILE. The twin builds any tile, and a gate that installs one runs on
// one tile a process, because enumeration happens once. So the tile is the
// command line's, and the gate is run once for each: with no argument or
// "128x64" on the chiplet's working geometry (the board plan's B10, as revised
// on 2026-10-05), and with "256x64" on the tile B10 was first settled at.
// Anything else is refused, and the caller exits on it: a gate that quietly ran
// on some other tile has not run the one that was asked for.
constexpr pta_twin_build pta_twin_working_tile = {128, 64, 8, 48, 4, 0};
constexpr pta_twin_build pta_twin_first_tile = {256, 64, 8, 48, 4, 0};

inline bool pta_twin_build_from_args(int argc, char** argv, pta_twin_build* build) {
  *build = pta_twin_working_tile;
  if (argc < 2) return true;
  if (argc == 2 && std::strcmp(argv[1], "128x64") == 0) return true;
  if (argc == 2 && std::strcmp(argv[1], "256x64") == 0) {
    *build = pta_twin_first_tile;
    return true;
  }
  std::printf("usage: %s [128x64 | 256x64]\n"
              "  the tile the twin is built with: 128x64, the chiplet's working "
              "geometry, if none is named\n", argv[0]);
  return false;
}

// The ADC shift that puts one K tile of a build inside an ADC of `adc_bits`
// bits. A K tile sums `rows` products of two DIN_W-bit operands, each at most
// 2^(2 (DIN_W - 1)), and the ADC has 2^(adc_bits - 1) codes each way. So the
// shift follows the tile's rows, and a gate that runs on two tiles derives it
// for each rather than carrying one tile's number to the other.
constexpr uint32_t pta_twin_adc_shift(const pta_twin_build& b, int adc_bits) {
  const int64_t span = static_cast<int64_t>(b.rows) << (2 * (b.din_w - 1));
  uint32_t s = 0;
  while ((static_cast<int64_t>(1) << (adc_bits - 1 + static_cast<int>(s))) < span) ++s;
  return s;
}
static_assert(pta_twin_adc_shift(pta_twin_working_tile, 7) == 15 &&
              pta_twin_adc_shift(pta_twin_working_tile, 6) == 16 &&
              pta_twin_adc_shift(pta_twin_first_tile, 7) == 16 &&
              pta_twin_adc_shift(pta_twin_first_tile, 6) == 17,
              "7- and 6-bit ADCs: LSBs of 2^15 and 2^16 on 128 rows, 2^16 and 2^17 on 256");

// THE QUEUE, the first board's transport. The same twin, behind the module of
// pta_chiplet_module.h: the twin's map on page 0 of the window and the queue on
// page 1, with this process's memory as the host memory the module reaches. A
// line's address is its pointer. A read of PTA_STATUS or of PTQ_STATUS is a
// poll, and that is where the module's time passes.
inline pta_module*& pta_module_installed() {
  static pta_module* m = nullptr;
  return m;
}

inline uint32_t pta_module_read_hook(void* ctx, uint32_t off) {
  pta_module* m = static_cast<pta_module*>(ctx);
  if (off == PTA_CHIPLET_STATUS || off == PTA_CHIPLET_PTQ_STATUS)
    pta_module_run(m, pta_twin_quantum());
  return pta_module_read32(m, off);
}

inline void pta_module_write_hook(void* ctx, uint32_t off, uint32_t v) {
  pta_module_write32(static_cast<pta_module*>(ctx), off, v);
}

// Every line the module read or wrote, in order: how a gate sees where a GEMM's
// operands were taken from and its results put. Cleared by whoever reads it.
struct PtaLine {
  uint64_t addr;
  bool     write;
};
inline std::vector<PtaLine>& pta_host_lines() {
  static std::vector<PtaLine> v;
  return v;
}

// How many of those lines fell inside [p, p + n), read or written.
inline size_t pta_host_lines_in(const void* p, size_t n, bool write) {
  const uint64_t lo = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p)), hi = lo + n;
  size_t k = 0;
  for (const PtaLine& l : pta_host_lines()) k += (l.write == write && l.addr >= lo && l.addr < hi);
  return k;
}

inline int pta_host_read_line(void*, uint64_t addr, uint8_t line[64]) {
  pta_host_lines().push_back(PtaLine{addr, false});
  std::memcpy(line, reinterpret_cast<const void*>(static_cast<uintptr_t>(addr)), 64);
  return 0;
}

inline int pta_host_write_line(void*, uint64_t addr, const uint8_t line[64]) {
  pta_host_lines().push_back(PtaLine{addr, true});
  std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(addr)), line, 64);
  return 0;
}

inline int pta_host_alloc(void*, size_t bytes, pta_chiplet_dma_t* out) {
  const size_t n = (bytes + 4095) / 4096 * 4096;
  void* p = std::aligned_alloc(4096, n ? n : 4096);
  if (!p) return -1;
  out->cpu = p;
  out->addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
  out->bytes = n ? n : 4096;
  return 0;
}

inline void pta_host_free(void*, pta_chiplet_dma_t* buf) {
  std::free(buf->cpu);
  buf->cpu = nullptr;
}

// Which way a gate's GEMMs reach the twin: the link's function call, as on rev
// A off a model; the queue, with the chiplet still behind its GPU; or the first
// board's module, on the host with no GPU behind it, whose operands are host
// memory. The tile is the command line's first word and the transport its
// second, "link" if none is named.
enum class PtaTransport { Link, Queue, Module };

inline bool pta_twin_args(int argc, char** argv, pta_twin_build* build, PtaTransport* how) {
  *build = pta_twin_working_tile;
  *how = PtaTransport::Link;
  bool ok = argc <= 3;
  if (ok && argc >= 2) {
    if (std::strcmp(argv[1], "256x64") == 0) *build = pta_twin_first_tile;
    else ok = std::strcmp(argv[1], "128x64") == 0;
  }
  if (ok && argc == 3) {
    if (std::strcmp(argv[2], "queue") == 0) *how = PtaTransport::Queue;
    else if (std::strcmp(argv[2], "module") == 0) *how = PtaTransport::Module;
    else ok = std::strcmp(argv[2], "link") == 0;
  }
  if (!ok)
    std::printf("usage: %s [128x64 | 256x64] [link | queue | module]\n"
                "  the tile the twin is built with, 128x64 if none is named, and how its\n"
                "  GEMMs reach it: the link's call, the queue behind a GPU, or the first\n"
                "  board's module on the host\n", argv[0]);
  return ok;
}

// Build a twin and install it as the chiplet in device `parent`'s package.
// BEFORE the first grx call. `with_link` false installs the window alone: a
// chiplet that can be read and cannot be given work. Returns false if the twin
// could not be built or the seam refused.
inline bool pta_twin_install(const pta_twin_build& build, int parent = 0,
                             bool with_link = true) {
  pta_twin* t = pta_twin_new(&build);
  if (!t) return false;
  pta_twin_installed() = t;
  if (grxcp_pta_attach_model_for_testing(pta_twin_read_hook, pta_twin_write_hook, t) != 1)
    return false;
  if (with_link && grxcp_pta_attach_link_for_testing(pta_twin_submit_hook, t) != 1)
    return false;
  return grxcp_pta_set_parent_for_testing(parent) == 1;
}

// The same, behind the module's queue: the window is the module's BAR, there
// is no link, and host memory is attached, so the only way a GEMM reaches the
// tile is the ring. Returns false if anything could not be built or attached.
inline bool pta_twin_install_queue(const pta_twin_build& build, int parent = 0,
                                   bool with_memory = true) {
  pta_twin* t = pta_twin_new(&build);
  if (!t) return false;
  pta_twin_installed() = t;
  const pta_module_host host = {pta_host_read_line, pta_host_write_line, nullptr, nullptr};
  pta_module* m = pta_module_new(t, &host);
  if (!m) return false;
  pta_module_installed() = m;
  if (grxcp_pta_attach_model_for_testing(pta_module_read_hook, pta_module_write_hook, m) != 1)
    return false;
  if (with_memory && grxcp_pta_attach_memory_for_testing(pta_host_alloc, pta_host_free, nullptr) != 1)
    return false;
  return grxcp_pta_set_parent_for_testing(parent) == 1;
}

// The first board's: the same module, on the host, with no GPU behind it. Its
// operands are host memory, and grxMallocHost on it is memory its queue reaches.
// `with_memory` false gives it none, which is no device at all.
inline bool pta_twin_install_module(const pta_twin_build& build, bool with_memory = true) {
  return pta_twin_install_queue(build, /*parent=*/-1, with_memory);
}

// What the run went through, from the module's own count and not from the
// command line: a gate that ignored "queue" would leave the ring untouched.
inline void pta_transport_note(PtaTransport how) {
  if (how != PtaTransport::Link && pta_module_installed())
    std::printf("  note  transport %s, %u commands through the ring\n",
                how == PtaTransport::Module ? "module" : "queue",
                pta_module_read32(pta_module_installed(), PTA_MODULE_PTQ_HEAD));
  else
    std::printf("  note  transport link, the twin's call\n");
}

}  // namespace grxtest

#endif  // GRXCP_TESTS_PTA_TWIN_ADAPTER_H
