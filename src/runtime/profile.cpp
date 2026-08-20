// GRXCP — the host half of grx-prof.
//
// Turned on by GRX_PROFILE in the environment; off it costs one branch per
// launch and per transfer.
//
// The interesting question for a profiler on this stack is *which clock*.
// There are two, they measure different things, and conflating them would
// produce a beautiful timeline that means nothing:
//
//   the host clock   `vx_event_get_profiling` stamps each command with host
//                    nanoseconds around execution. It orders operations and
//                    shows the gaps between them -- and on a simulator it
//                    measures the simulator, not the device. That is why
//                    grxDeviceProp_t::eventTimingIsDeviceSide reports 0.
//
//   device cycles    the MPM performance counters, read through
//                    `vx_device_mpm_query`. MCYCLE advances only while the
//                    device is running, so its delta across a launch is real
//                    device time, on the simulator and on silicon alike.
//
// So: the timeline is built on the host clock, because only it can place two
// operations relative to each other, and every kernel slice carries its device
// cycle count as an argument. Neither number is presented as the other. The
// counters -- warp occupancy, the stall breakdown, the instruction mix -- come
// from MPM and are device facts.
//
// Profiling serializes. Each measured operation is bracketed by a full device
// sync so the counter deltas belong to it and nothing else. Today that costs
// nothing real, because the command processor runs one queue and streams do
// not overlap anyway (grxStreamCreate's caveat in docs/conformance.md), but it
// is a property of the mode and the report says so.

#include "internal.h"

#include <VX_types.h>   // MPM class ids and counter CSR addresses

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace grxcp {

namespace {

// The counters sampled around every kernel.
//
// `sum` says how to aggregate across cores. Event counts add; MCYCLE does not
// -- cycles are elapsed time, and four cores running for a thousand cycles
// each took a thousand cycles, not four thousand. Getting that wrong would
// quadruple the denominator of every rate in the report.
struct Counter {
  uint32_t    cls;
  uint32_t    addr;
  const char* name;
  bool        sum;
  bool        needs_tcu;
};

const Counter kCounters[] = {
  {VX_DCR_MPM_CLASS_BASE, VX_CSR_MCYCLE,             "cycles",         false, false},
  {VX_DCR_MPM_CLASS_BASE, VX_CSR_MINSTRET,           "instrs",         true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_SCHED_IDLE,     "sched_idle",     true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_ACTIVE_WARPS,   "active_warps",   true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_STALLED_WARPS,  "stalled_warps",  true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_ISSUED_WARPS,   "issued_warps",   true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_ISSUED_THREADS, "issued_threads", true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_STALL_FETCH,    "stall_fetch",    true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_STALL_IBUF,     "stall_ibuf",     true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_STALL_SCRB,     "stall_scrb",     true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_STALL_OPDS,     "stall_opds",     true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_STALL_ALU,      "stall_alu",      true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_STALL_FPU,      "stall_fpu",      true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_STALL_LSU,      "stall_lsu",      true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_STALL_SFU,      "stall_sfu",      true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_STALL_TCU,      "stall_tcu",      true,  true },
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_BRANCHES,       "branches",       true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_DIVERGENCE,     "divergence",     true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_INSTR_ALU,      "instr_alu",      true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_INSTR_FPU,      "instr_fpu",      true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_INSTR_LSU,      "instr_lsu",      true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_INSTR_SFU,      "instr_sfu",      true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_INSTR_TCU,      "instr_tcu",      true,  true },
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_LOADS,          "loads",          true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_STORES,         "stores",         true,  false},
  {VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_IFETCHES,       "ifetches",       true,  false},
};
constexpr size_t kNumCounters = sizeof(kCounters) / sizeof(kCounters[0]);

struct Snapshot {
  uint64_t host_ns = 0;
  uint64_t value[kNumCounters] = {};
  bool     valid[kNumCounters] = {};
};

std::mutex  g_prof_mutex;
int         enabled_state = -1;
uint64_t    g_seq         = 0;
bool        g_device_announced = false;
std::map<std::string, bool> g_occupancy_announced;

// A counter that the backend does not implement should not be reported as
// zero: zero is a measurement. When a query fails the sample is marked invalid
// and the record omits the field entirely, so the tool prints "unavailable"
// rather than a number nobody measured.
bool read_counter(Device& d, const Counter& c, uint32_t cores, uint64_t* out) {
  uint64_t acc = 0;
  for (uint32_t core = 0; core < cores; ++core) {
    uint64_t v = 0;
    if (vx_device_mpm_query(d.handle, c.cls, c.addr, core, &v) != VX_SUCCESS)
      return false;
    acc = c.sum ? (acc + v) : std::max(acc, v);
  }
  *out = acc;
  return true;
}

void take_snapshot(Device& d, Snapshot* s) {
  s->host_ns = host_now_ns();
  const uint32_t cores =
      (d.prop.multiProcessorCount > 0) ? (uint32_t)d.prop.multiProcessorCount : 1u;
  const bool tcu = (d.prop.capabilities & GRX_CAP_TENSOR_CORE) != 0;
  for (size_t i = 0; i < kNumCounters; ++i) {
    if (kCounters[i].needs_tcu && !tcu) { s->valid[i] = false; continue; }
    s->valid[i] = read_counter(d, kCounters[i], cores, &s->value[i]);
  }
}

// Reading a counter is not free: each one is two DCR reads through the command
// processor, and the processor runs while it serves them. So a snapshot is not
// instantaneous, and the cycles that elapse between reading MCYCLE and reading
// the last counter of the same snapshot land inside the next delta.
//
// That noise floor is measured rather than assumed, and reported rather than
// subtracted. Two snapshots back to back with no work between them: whatever
// separates them is what one sample pair costs. A kernel whose cycle count is
// close to this number has not been measured, and the report says so instead
// of quoting it.
void announce_overhead(Device& d) {
  Snapshot a, b;
  take_snapshot(d, &a);
  take_snapshot(d, &b);
  const bool ok = a.valid[0] && b.valid[0] && b.value[0] >= a.value[0];
  std::fprintf(stderr, "GRXPROF|overhead|cycles=%lld|host_ns=%llu\n",
               ok ? (long long)(b.value[0] - a.value[0]) : -1LL,
               (unsigned long long)(b.host_ns - a.host_ns));
}

void announce_device(Device& d) {
  if (g_device_announced) return;
  g_device_announced = true;
  const grxDeviceProp_t& p = d.prop;
  std::fprintf(stderr,
               "GRXPROF|device|name=%s|backend=%s|cores=%d|warps_per_core=%d"
               "|warp_size=%d|shared_per_sm=%zu|clock_mhz=%d"
               "|event_timing_is_device_side=%d\n",
               p.name, backend_name(p.backend), p.multiProcessorCount,
               p.maxWarpsPerMultiProcessor, p.warpSize,
               p.sharedMemPerMultiprocessor, p.clockRateMHz,
               p.eventTimingIsDeviceSide);
  announce_overhead(d);
}

// The occupancy report: what the CTA dispatcher will admit for this launch
// shape. Emitted once per distinct (kernel, block, shared) rather than per
// launch, because it is a property of the shape and repeating it per launch
// would suggest it had been measured again.
void announce_occupancy(Device& d, const char* kernel, uint32_t block,
                        size_t shared) {
  char key[256];
  std::snprintf(key, sizeof(key), "%s/%u/%zu", kernel ? kernel : "?", block,
                shared);
  if (g_occupancy_announced[key]) return;
  g_occupancy_announced[key] = true;

  const int blocks = resident_blocks_per_sm(d.prop, (int)block, shared);
  const int warps_per_block =
      (d.prop.warpSize > 0)
          ? (int)((block + (uint32_t)d.prop.warpSize - 1) / (uint32_t)d.prop.warpSize)
          : 0;
  std::fprintf(stderr,
               "GRXPROF|occupancy|kernel=%s|block=%u|shared=%zu"
               "|resident_blocks=%d|resident_warps=%d|warp_slots=%d\n",
               kernel ? kernel : "?", block, shared, blocks,
               blocks * warps_per_block, d.prop.maxWarpsPerMultiProcessor);
}

void emit_counters(const Snapshot& a, const Snapshot& b) {
  for (size_t i = 0; i < kNumCounters; ++i) {
    if (!a.valid[i] || !b.valid[i]) continue;
    // Counters are free-running and monotonic; a decrease means something
    // reset them underneath us, and reporting the wraparound as a huge count
    // would be worse than saying nothing.
    if (b.value[i] < a.value[i]) continue;
    std::fprintf(stderr, "|c.%s=%llu", kCounters[i].name,
                 (unsigned long long)(b.value[i] - a.value[i]));
  }
}

}  // namespace

bool profile_enabled() {
  if (enabled_state < 0) {
    const char* v = std::getenv("GRX_PROFILE");
    enabled_state = (v && v[0] && std::strcmp(v, "0") != 0) ? 1 : 0;
  }
  return enabled_state == 1;
}

// Bracket a measured operation. The sync is what makes the counter delta
// attributable: without it the numbers would include whatever else the device
// was still finishing.
bool profile_begin(int device, ProfileSample* out) {
  if (!profile_enabled() || !out) return false;
  Device* d = nullptr;
  if (acquire_device(device, &d) != grxSuccess) return false;
  if (sync_all_streams(device) != grxSuccess) return false;

  std::lock_guard<std::mutex> lock(g_prof_mutex);
  announce_device(*d);
  auto* s = new Snapshot();
  take_snapshot(*d, s);
  out->device = device;
  out->opaque = s;
  return true;
}

void profile_end_kernel(ProfileSample* sample, const char* kernel,
                        const char* module_path, uint32_t grid[3],
                        uint32_t block[3], size_t shared, const void* stream) {
  if (!sample || !sample->opaque) return;
  auto* before = static_cast<Snapshot*>(sample->opaque);
  sample->opaque = nullptr;

  Device* d = nullptr;
  if (acquire_device(sample->device, &d) != grxSuccess) { delete before; return; }
  if (sync_all_streams(sample->device) != grxSuccess)   { delete before; return; }

  Snapshot after;
  std::lock_guard<std::mutex> lock(g_prof_mutex);
  take_snapshot(*d, &after);

  const uint32_t threads = block[0] * block[1] * block[2];
  announce_occupancy(*d, kernel, threads, shared);

  std::fprintf(stderr,
               "GRXPROF|1|op=launch|seq=%llu|kernel=%s|module=%s|stream=%llu"
               "|grid=%u.%u.%u|block=%u.%u.%u|shared=%zu"
               "|host_start_ns=%llu|host_end_ns=%llu",
               (unsigned long long)g_seq++, kernel ? kernel : "?",
               module_path ? module_path : "", (unsigned long long)(uintptr_t)stream,
               grid[0], grid[1], grid[2], block[0], block[1], block[2], shared,
               (unsigned long long)before->host_ns,
               (unsigned long long)after.host_ns);
  emit_counters(*before, after);
  std::fprintf(stderr, "\n");
  delete before;
}

void profile_end_transfer(ProfileSample* sample, const char* op, uint64_t bytes,
                          const char* kind, const void* stream) {
  if (!sample || !sample->opaque) return;
  auto* before = static_cast<Snapshot*>(sample->opaque);
  sample->opaque = nullptr;

  Device* d = nullptr;
  if (acquire_device(sample->device, &d) != grxSuccess) { delete before; return; }
  if (sync_all_streams(sample->device) != grxSuccess)   { delete before; return; }

  Snapshot after;
  std::lock_guard<std::mutex> lock(g_prof_mutex);
  take_snapshot(*d, &after);

  // No counter deltas on a transfer. The CP's DMA does not execute on a core,
  // so the core counters do not describe it; printing them would invite the
  // reader to attribute core cycles to a copy that never used one.
  std::fprintf(stderr,
               "GRXPROF|1|op=%s|seq=%llu|bytes=%llu|kind=%s|stream=%llu"
               "|host_start_ns=%llu|host_end_ns=%llu\n",
               op, (unsigned long long)g_seq++, (unsigned long long)bytes,
               kind ? kind : "?", (unsigned long long)(uintptr_t)stream,
               (unsigned long long)before->host_ns,
               (unsigned long long)after.host_ns);
  delete before;
}

void profile_abandon(ProfileSample* sample) {
  if (!sample || !sample->opaque) return;
  delete static_cast<Snapshot*>(sample->opaque);
  sample->opaque = nullptr;
}

namespace {
// Tells grx-prof the runtime really was in profiling mode, the same way the
// sanitizer's summary does: an empty trace and a trace nobody collected look
// identical from outside the process.
struct SummaryAtExit {
  ~SummaryAtExit() {
    if (profile_enabled())
      std::fprintf(stderr, "GRXPROF|summary|records=%llu\n",
                   (unsigned long long)g_seq);
  }
};
SummaryAtExit g_summary_at_exit;
}  // namespace

}  // namespace grxcp
