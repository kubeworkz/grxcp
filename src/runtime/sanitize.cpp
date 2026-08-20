// GRXCP — the host half of grx-sanitize.
//
// Turned on by GRX_SANITIZE in the environment; off it costs one branch per
// allocation and nothing else.
//
// What it does, in order:
//
//   1. Pads every device allocation with a trailing redzone, and registers the
//      allocation's REQUESTED size -- not the rounded-up size the allocator
//      actually hands out. A one-byte overflow of a 100-byte buffer is then a
//      hole in the address map rather than slack nobody notices.
//   2. Quarantines freed memory instead of recycling it, so a use-after-free
//      stays a use-after-free rather than becoming a read of whatever moved in.
//   3. Uploads the resulting map -- the regions the allocator owns, and the
//      live/freed extents inside them -- to the device before each launch.
//   4. Reads back what the device-side runtime recorded, and prints one
//      machine-readable line per finding. `grx-sanitize` turns those lines and
//      the module's ELF into file:line.
//
// The device half is src/device/grx_sanitize_rt.cpp, compiled into any kernel
// built with `ci/build_kernel.sh --sanitize`. Without that flag there is no
// instrumentation and nothing here can see a single memory access -- which is
// why arming an uninstrumented module prints a status line saying so instead
// of letting the run finish "clean".

#include "internal.h"

#include <grx/grx_sanitize_abi.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace grxcp {

namespace {

// Sized for a debugging run, not for scale. Overflowing any of these is
// reported rather than silently truncated.
constexpr uint32_t kMaxRegions = 256;
constexpr uint32_t kMaxExtents = 4096;
// One report slot per grid-linear thread, because the device cannot use an
// atomic counter (see the ABI header). 1024 slots covers the grids a debugging
// run launches; a larger grid records nothing above the limit and is told so.
constexpr uint32_t kMaxReports = 1024;

// One allocation alignment. Big enough that a typical off-by-one-element
// overflow lands inside it; small enough not to dominate a run of small
// allocations.
constexpr uint64_t kRedzone = 256;

constexpr uint64_t kStateOff   = 0;
constexpr uint64_t kRegionsOff = 128;
constexpr uint64_t kExtentsOff = kRegionsOff + (uint64_t)kMaxRegions * sizeof(grxSanRegion);
constexpr uint64_t kReportsOff = kExtentsOff + (uint64_t)kMaxExtents * sizeof(grxSanExtent);
constexpr uint64_t kBlockSize  = kReportsOff + (uint64_t)kMaxReports * sizeof(grxSanReport);

struct DeviceBlock {
  vx_buffer_h buffer = nullptr;
  uint64_t    base   = 0;
};

std::mutex                     g_san_mutex;
std::map<int, DeviceBlock>     g_blocks;      // device -> control block
std::vector<grxSanRegion>      g_regions;
std::map<uint64_t, grxSanExtent> g_extents;   // base -> extent (live or freed)
uint32_t                       g_next_id  = 1;
bool                           g_map_dirty = true;
int                            g_findings  = 0;
uint32_t                       g_dropped   = 0;
bool                           g_overflow_reported = false;

// The kernel currently armed, for attribution in the report line. Set by
// sanitize_arm, read by sanitize_drain.
std::string g_kernel, g_module, g_elf;

int enabled_state = -1;   // -1 unknown, 0 off, 1 on

const char* kind_name(uint32_t kind) {
  switch (kind) {
    case GRX_SAN_KIND_OOB_GLOBAL:     return "oob-global";
    case GRX_SAN_KIND_OOB_STRADDLE:   return "oob-straddle";
    case GRX_SAN_KIND_USE_AFTER_FREE: return "use-after-free";
    case GRX_SAN_KIND_OOB_SHARED:     return "oob-shared";
    default:                          return "unknown";
  }
}

// ---------------------------------------------------------------------------
// ELF: find the anchor
// ---------------------------------------------------------------------------
//
// The .vxbin footer carries kernel entry points only, so the anchor's address
// has to come from the sibling ELF that ci/build_kernel.sh leaves beside it.
// Just enough ELF64-LE to walk the symbol table; anything unexpected is a
// "not found" rather than an error, because the honest fallback (report the
// module as uninstrumented) is better than refusing to run.

#pragma pack(push, 1)
struct Elf64Header {
  uint8_t  ident[16];
  uint16_t type, machine;
  uint32_t version;
  uint64_t entry, phoff, shoff;
  uint32_t flags;
  uint16_t ehsize, phentsize, phnum, shentsize, shnum, shstrndx;
};
struct Elf64Section {
  uint32_t name, type;
  uint64_t flags, addr, offset, size;
  uint32_t link, info;
  uint64_t addralign, entsize;
};
struct Elf64Sym {
  uint32_t name;
  uint8_t  info, other;
  uint16_t shndx;
  uint64_t value, size;
};
#pragma pack(pop)

bool read_file(const char* path, std::vector<uint8_t>* out) {
  std::FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  const long len = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (len <= 0) { std::fclose(f); return false; }
  out->resize((size_t)len);
  const size_t got = std::fread(out->data(), 1, out->size(), f);
  std::fclose(f);
  return got == out->size();
}

bool elf_symbol_value(const std::vector<uint8_t>& elf, const char* want,
                      uint64_t* out_value) {
  if (elf.size() < sizeof(Elf64Header)) return false;
  Elf64Header eh{};
  std::memcpy(&eh, elf.data(), sizeof(eh));
  if (std::memcmp(eh.ident, "\x7f" "ELF", 4) != 0) return false;
  if (eh.ident[4] != 2 || eh.ident[5] != 1) return false;   // ELF64, LE
  if (eh.shentsize != sizeof(Elf64Section))  return false;
  if (eh.shoff + (uint64_t)eh.shnum * eh.shentsize > elf.size()) return false;

  for (uint16_t i = 0; i < eh.shnum; ++i) {
    Elf64Section sh{};
    std::memcpy(&sh, elf.data() + eh.shoff + (uint64_t)i * eh.shentsize, sizeof(sh));
    if (sh.type != 2 /*SHT_SYMTAB*/) continue;
    if (sh.entsize != sizeof(Elf64Sym) || sh.link >= eh.shnum) continue;

    Elf64Section str{};
    std::memcpy(&str, elf.data() + eh.shoff + (uint64_t)sh.link * eh.shentsize,
                sizeof(str));
    if (sh.offset + sh.size > elf.size() || str.offset + str.size > elf.size())
      continue;

    const uint64_t n = sh.size / sizeof(Elf64Sym);
    for (uint64_t k = 0; k < n; ++k) {
      Elf64Sym sym{};
      std::memcpy(&sym, elf.data() + sh.offset + k * sizeof(Elf64Sym), sizeof(sym));
      if (sym.name == 0 || sym.name >= str.size) continue;
      const char* name = (const char*)elf.data() + str.offset + sym.name;
      if (std::strcmp(name, want) == 0) { *out_value = sym.value; return true; }
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// The device-side control block
// ---------------------------------------------------------------------------

grxError_t write_block(Device& d, DeviceBlock& b, uint64_t offset,
                       const void* src, uint64_t size);

grxError_t ensure_block_locked(Device& d, DeviceBlock** out) {
  auto it = g_blocks.find(d.index);
  if (it != g_blocks.end()) { *out = &it->second; return grxSuccess; }

  // Created with the raw driver call rather than through the GRXCP allocator,
  // so the sanitizer's own memory never appears in the map it is checking.
  vx_buffer_h buf = nullptr;
  vx_result_t r = vx_buffer_create(d.handle, kBlockSize, VX_MEM_READ_WRITE, &buf);
  if (r != VX_SUCCESS) return map_result(r);
  uint64_t base = 0;
  r = vx_buffer_address(buf, &base);
  if (r != VX_SUCCESS) { vx_buffer_release(buf); return map_result(r); }

  DeviceBlock b;
  b.buffer = buf;
  b.base   = base;
  auto ins = g_blocks.emplace(d.index, b);
  *out = &ins.first->second;

  // Zero the header immediately. Device memory is not zero on arrival: SimX's
  // RAM hands back a 0xbaadf00d sentinel page for any address never written,
  // so an unwritten control block reads as ~3.1 billion findings. Writing the
  // header here means a drain before the first arm sees a version mismatch and
  // returns, instead of trusting a sentinel.
  grxSanState empty{};
  return write_block(d, **out, kStateOff, &empty, sizeof(empty));
}

// A blocking write into the control block. Sanitized runs are slow by
// construction; overlapping the upload with anything would only make the
// ordering harder to reason about.
grxError_t write_block(Device& d, DeviceBlock& b, uint64_t offset,
                       const void* src, uint64_t size) {
  vx_queue_h q = nullptr;
  grxError_t e = resolve_stream(nullptr, d.index, &q, nullptr);
  if (e != grxSuccess) return e;
  vx_event_h ev = nullptr;
  vx_result_t r = vx_enqueue_write(q, b.buffer, offset, src, size, 0, nullptr, &ev);
  if (r != VX_SUCCESS) return map_result(r);
  r = vx_queue_finish(q, VX_TIMEOUT_INFINITE);
  if (ev) vx_event_release(ev);
  return (r == VX_SUCCESS) ? grxSuccess : map_result(r);
}

grxError_t read_block(Device& d, DeviceBlock& b, uint64_t offset, void* dst,
                      uint64_t size) {
  vx_queue_h q = nullptr;
  grxError_t e = resolve_stream(nullptr, d.index, &q, nullptr);
  if (e != grxSuccess) return e;
  vx_event_h ev = nullptr;
  vx_result_t r = vx_enqueue_read(q, dst, b.buffer, offset, size, 0, nullptr, &ev);
  if (r != VX_SUCCESS) return map_result(r);
  r = vx_queue_finish(q, VX_TIMEOUT_INFINITE);
  if (ev) vx_event_release(ev);
  return (r == VX_SUCCESS) ? grxSuccess : map_result(r);
}

}  // namespace

// ---------------------------------------------------------------------------
// Public to the rest of the runtime
// ---------------------------------------------------------------------------

bool sanitize_enabled() {
  if (enabled_state < 0) {
    const char* v = std::getenv("GRX_SANITIZE");
    enabled_state = (v && v[0] && std::strcmp(v, "0") != 0) ? 1 : 0;
  }
  return enabled_state == 1;
}

uint64_t sanitize_redzone_bytes() { return sanitize_enabled() ? kRedzone : 0; }

void sanitize_note_region(uint64_t base, uint64_t size) {
  if (!sanitize_enabled()) return;
  std::lock_guard<std::mutex> lock(g_san_mutex);
  if (g_regions.size() >= kMaxRegions) {
    if (!g_overflow_reported) {
      std::fprintf(stderr, "GRXSAN|status|regions=overflow|limit=%u\n", kMaxRegions);
      g_overflow_reported = true;
    }
    return;
  }
  g_regions.push_back(grxSanRegion{base, size});
  g_map_dirty = true;
}

void sanitize_forget_region(uint64_t base) {
  if (!sanitize_enabled()) return;
  std::lock_guard<std::mutex> lock(g_san_mutex);
  g_regions.erase(std::remove_if(g_regions.begin(), g_regions.end(),
                                 [&](const grxSanRegion& r) { return r.base == base; }),
                  g_regions.end());
  g_map_dirty = true;
}

void sanitize_note_alloc(uint64_t base, uint64_t requested) {
  if (!sanitize_enabled()) return;
  std::lock_guard<std::mutex> lock(g_san_mutex);
  grxSanExtent e{};
  e.base  = base;
  e.size  = requested;
  e.state = GRX_SAN_EXTENT_LIVE;
  e.id    = g_next_id++;
  g_extents[base] = e;
  g_map_dirty = true;
}

void sanitize_note_free(uint64_t base) {
  if (!sanitize_enabled()) return;
  std::lock_guard<std::mutex> lock(g_san_mutex);
  auto it = g_extents.find(base);
  if (it == g_extents.end()) return;
  it->second.state = GRX_SAN_EXTENT_FREED;
  g_map_dirty = true;
}

void sanitize_forget_all() {
  if (!sanitize_enabled()) return;
  std::lock_guard<std::mutex> lock(g_san_mutex);
  g_extents.clear();
  g_regions.clear();
  g_map_dirty = true;
}

// Patch the anchor into a .vxbin image so the device runtime can find the
// control block. Returns true when the image was instrumented and patched.
//
// .vxbin layout is [min_vma:8][max_vma:8][payload...], so the payload byte for
// a VMA is at 16 + (vma - min_vma). An anchor outside that span means the
// symbol landed in .bss, which the packager does not emit -- that is a build
// bug rather than a user error, and it is reported as one.
bool sanitize_patch_image(std::vector<uint8_t>& image, const char* elf_path,
                          int device) {
  if (!sanitize_enabled() || !elf_path) return false;

  std::vector<uint8_t> elf;
  if (!read_file(elf_path, &elf)) return false;
  uint64_t anchor_vma = 0;
  if (!elf_symbol_value(elf, "__grx_san_anchor", &anchor_vma)) return false;

  if (image.size() < 16) return false;
  uint64_t min_vma = 0, max_vma = 0;
  std::memcpy(&min_vma, image.data() + 0, 8);
  std::memcpy(&max_vma, image.data() + 8, 8);
  const uint64_t off = 16 + (anchor_vma - min_vma);
  if (anchor_vma < min_vma || off + 8 > image.size()) {
    std::fprintf(stderr,
                 "GRXSAN|status|anchor=unreachable|vma=0x%llx|image_bytes=%zu\n",
                 (unsigned long long)anchor_vma, image.size());
    return false;
  }

  Device* d = nullptr;
  if (acquire_device(device, &d) != grxSuccess) return false;

  std::lock_guard<std::mutex> lock(g_san_mutex);
  DeviceBlock* b = nullptr;
  if (ensure_block_locked(*d, &b) != grxSuccess) return false;

  const uint64_t state_addr = b->base + kStateOff;
  std::memcpy(image.data() + off, &state_addr, 8);
  return true;
}

// Upload the map and reset the report counter. Called once per launch.
grxError_t sanitize_arm(Device& d, uint32_t shared_bytes, uint32_t grid_threads,
                        const char* kernel, const char* module_path,
                        const char* elf_path, bool instrumented) {
  if (!sanitize_enabled()) return grxSuccess;

  std::lock_guard<std::mutex> lock(g_san_mutex);
  g_kernel = kernel      ? kernel      : "?";
  g_module = module_path ? module_path : "?";
  g_elf    = elf_path    ? elf_path    : "";

  if (!instrumented) {
    // Said once per kernel, and said loudly: a run whose kernels carry no
    // instrumentation produces no findings for reasons that have nothing to do
    // with the code being correct.
    static std::map<std::string, bool> announced;
    if (!announced[g_kernel]) {
      announced[g_kernel] = true;
      std::fprintf(stderr,
                   "GRXSAN|status|kernel=%s|module=%s|instrumented=0\n",
                   g_kernel.c_str(), g_module.c_str());
    }
    return grxSuccess;
  }

  DeviceBlock* b = nullptr;
  grxError_t e = ensure_block_locked(d, &b);
  if (e != grxSuccess) return e;

  if (g_map_dirty) {
    std::vector<grxSanExtent> extents;
    extents.reserve(g_extents.size());
    for (const auto& kv : g_extents) extents.push_back(kv.second);
    if (extents.size() > kMaxExtents) extents.resize(kMaxExtents);

    if (!extents.empty()) {
      e = write_block(d, *b, kExtentsOff, extents.data(),
                      extents.size() * sizeof(grxSanExtent));
      if (e != grxSuccess) return e;
    }
    if (!g_regions.empty()) {
      e = write_block(d, *b, kRegionsOff, g_regions.data(),
                      g_regions.size() * sizeof(grxSanRegion));
      if (e != grxSuccess) return e;
    }
    g_map_dirty = false;
  }

  // Slots are indexed by grid-linear thread, so they must start empty or last
  // launch's findings would look like this launch's. Only the slots this grid
  // can reach are cleared -- a two-warp launch does not pay for a thousand.
  const uint32_t slots = std::min(grid_threads ? grid_threads : 1u, kMaxReports);
  static const std::vector<grxSanReport> zeros(kMaxReports);
  e = write_block(d, *b, kReportsOff, zeros.data(),
                  (uint64_t)slots * sizeof(grxSanReport));
  if (e != grxSuccess) return e;

  if (grid_threads > kMaxReports) {
    std::fprintf(stderr,
                 "GRXSAN|status|kernel=%s|report_slots=%u|grid_threads=%u"
                 "|coverage=partial\n",
                 g_kernel.c_str(), kMaxReports, grid_threads);
  }

  grxSanState s{};
  s.abi_version  = GRX_SAN_ABI_VERSION;
  s.enabled      = 1;
  s.extents      = b->base + kExtentsOff;
  s.regions      = b->base + kRegionsOff;
  s.reports      = b->base + kReportsOff;
  s.num_extents  = (uint32_t)std::min<size_t>(g_extents.size(), kMaxExtents);
  s.num_regions  = (uint32_t)g_regions.size();
  s.max_reports  = slots;
  s.grid_threads = grid_threads;
  s.shared_bytes = shared_bytes;
  return write_block(d, *b, kStateOff, &s, sizeof(s));
}

// Read back whatever the last launch recorded. Safe to call when nothing was
// armed; it costs one small read.
void sanitize_drain(int device) {
  if (!sanitize_enabled()) return;

  std::lock_guard<std::mutex> lock(g_san_mutex);
  auto it = g_blocks.find(device);
  if (it == g_blocks.end()) return;
  DeviceBlock& b = it->second;

  Device* d = nullptr;
  if (acquire_device(device, &d) != grxSuccess) return;

  grxSanState s{};
  if (read_block(*d, b, kStateOff, &s, sizeof(s)) != grxSuccess) return;

  // Believe the block only if it looks like a block this runtime armed. The
  // alternative is trusting whatever the memory happened to contain, which on
  // an unwritten page is a sentinel that parses as a count.
  if (s.abi_version != GRX_SAN_ABI_VERSION || !s.enabled ||
      s.max_reports == 0 || s.max_reports > kMaxReports)
    return;

  // Slots are read whole and filtered here: there is no device-side counter to
  // consult (the ABI header explains why), so an empty slot is one whose kind
  // is zero.
  std::vector<grxSanReport> slots(s.max_reports);
  if (read_block(*d, b, kReportsOff, slots.data(),
                 slots.size() * sizeof(grxSanReport)) != grxSuccess)
    return;

  std::vector<grxSanReport> reports;
  for (const grxSanReport& r : slots)
    if (r.kind != 0) reports.push_back(r);
  if (reports.empty()) return;

  // Raw facts only. Every derived number -- how far past the end, which side
  // of the allocation -- is computed once, in grx-sanitize, from these. Two
  // places computing the same offset is two places to get it wrong.
  for (const grxSanReport& r : reports) {
    std::fprintf(stderr,
                 "GRXSAN|1|kind=%s|access=%s|size=%u|addr=0x%llx|pc=0x%llx"
                 "|alloc=0x%llx|allocsize=%llu|allocid=%u"
                 "|block=%u|thread=%u|warp=%u|lane=%u|kernel=%s|module=%s|elf=%s\n",
                 kind_name(r.kind),
                 (r.flags & GRX_SAN_FLAG_WRITE) ? "write" : "read",
                 r.size, (unsigned long long)r.addr, (unsigned long long)r.pc,
                 (unsigned long long)r.extent_base,
                 (unsigned long long)r.extent_size, r.extent_id,
                 r.block, r.thread, r.warp, r.lane,
                 g_kernel.c_str(), g_module.c_str(), g_elf.c_str());
    ++g_findings;
  }
  if (s.grid_threads > s.max_reports)
    g_dropped += s.grid_threads - s.max_reports;

  // Clear the slots so the same findings are not re-reported at the next
  // synchronization point. The state header stays armed; only the table is
  // reset, which is also what the next arm would do.
  static const std::vector<grxSanReport> zeros(kMaxReports);
  write_block(*d, b, kReportsOff, zeros.data(),
              (uint64_t)s.max_reports * sizeof(grxSanReport));
}

int sanitize_findings() { return g_findings; }

void sanitize_report_summary() {
  if (!sanitize_enabled()) return;
  std::fprintf(stderr, "GRXSAN|summary|findings=%d|dropped=%u\n", g_findings,
               g_dropped);
}

namespace {
// The summary is what tells grx-sanitize the runtime was actually in sanitizer
// mode. Without it, "no findings" and "GRX_SANITIZE never reached the runtime"
// look identical from outside the process -- so it is emitted at exit, from a
// destructor, rather than from a teardown call a program might not make.
struct SummaryAtExit {
  ~SummaryAtExit() { sanitize_report_summary(); }
};
SummaryAtExit g_summary_at_exit;
}  // namespace

}  // namespace grxcp
