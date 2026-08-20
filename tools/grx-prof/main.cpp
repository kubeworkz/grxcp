// grx-prof — run a GRXCP program with profiling on, and turn what the runtime
// recorded into a Perfetto trace and a report.
//
//   grx-prof [options] -- <program> [args...]
//
// Same split as grx-sanitize: the runtime collects, because only it can
// bracket an operation and read the device's performance counters; this tool
// formats, because that is all it takes.
//
// Two clocks, kept apart on purpose:
//
//   The TIMELINE is host nanoseconds. It is the only clock that can place two
//   operations relative to each other and show the gaps between them. On a
//   simulator backend it measures the simulator, not the device -- so the
//   trace's own track names say so, and so does the report.
//
//   The COST of a kernel is device cycles, from the MPM performance counters.
//   MCYCLE advances only while the device runs, so a delta across a launch is
//   real device time on the simulator and on silicon alike. Every kernel slice
//   in the trace carries it, and it is what the report ranks by.
//
// Nothing here converts one into the other.

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace {

using Fields = std::map<std::string, std::string>;

struct Record {
  std::string op;          // launch | memcpy | memset
  std::string name;        // kernel name, or the transfer direction
  uint64_t    seq   = 0;
  uint64_t    start = 0, end = 0;   // host ns
  uint64_t    stream = 0;
  uint64_t    bytes = 0;
  std::string grid, block;
  uint64_t    shared = 0;
  std::map<std::string, uint64_t> counters;
};

struct Occupancy {
  std::string kernel;
  uint64_t block = 0, shared = 0;
  long resident_blocks = 0, resident_warps = 0, warp_slots = 0;
};

struct Device {
  std::string name = "?", backend = "?";
  long cores = 1, warps_per_core = 0, warp_size = 0, clock_mhz = 0;
  long shared_per_sm = 0;
  bool event_timing_is_device_side = false;
  bool known = false;
};

void usage() {
  std::fprintf(stderr,
      "usage: grx-prof [--out FILE] [--no-trace] [--quiet] -- <program> [args...]\n"
      "\n"
      "  --out FILE   where to write the Chrome/Perfetto trace"
      " (default grxprof.json)\n"
      "  --no-trace   report only, write no trace file\n"
      "  --quiet      trace only, no report\n");
}

Fields parse(const std::string& line) {
  Fields f;
  size_t pos = 0;
  while (pos < line.size()) {
    const size_t bar = line.find('|', pos);
    const std::string tok = line.substr(pos, (bar == std::string::npos)
                                                 ? std::string::npos
                                                 : bar - pos);
    const size_t eq = tok.find('=');
    if (eq != std::string::npos) f[tok.substr(0, eq)] = tok.substr(eq + 1);
    if (bar == std::string::npos) break;
    pos = bar + 1;
  }
  return f;
}

std::string get(const Fields& f, const char* k, const char* dflt = "") {
  auto it = f.find(k);
  return (it == f.end()) ? std::string(dflt) : it->second;
}

uint64_t getu(const Fields& f, const char* k, uint64_t dflt = 0) {
  auto it = f.find(k);
  return (it == f.end()) ? dflt : std::strtoull(it->second.c_str(), nullptr, 0);
}

// A counter the runtime could not read is absent from the record rather than
// zero, so "missing" and "measured as none" stay distinguishable all the way
// to the report.
bool counter(const Record& r, const char* name, uint64_t* out) {
  auto it = r.counters.find(name);
  if (it == r.counters.end()) return false;
  *out = it->second;
  return true;
}

std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (c == '\n')        { o += "\\n"; }
    else                        { o += c; }
  }
  return o;
}

// ---------------------------------------------------------------------------
// Trace
// ---------------------------------------------------------------------------

bool write_trace(const std::string& path, const std::vector<Record>& records,
                 const Device& dev, uint64_t t0) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) return false;

  std::fprintf(f, "{\"displayTimeUnit\":\"ns\",\"traceEvents\":[\n");
  bool first = true;
  auto comma = [&] { if (!first) std::fprintf(f, ",\n"); first = false; };

  // The track names carry the caveat. Someone who opens this trace six months
  // from now reads the axis before they read any documentation.
  char pname[512];
  std::snprintf(pname, sizeof(pname), "%s  [x-axis: host clock%s]",
                dev.name.c_str(),
                dev.event_timing_is_device_side
                    ? ""
                    : (dev.backend == "simx" || dev.backend == "rtlsim" ||
                       dev.backend == "gem5")
                          ? ", which on this backend measures the simulator"
                          : ", not device timestamps");
  comma();
  std::fprintf(f,
               "{\"ph\":\"M\",\"pid\":1,\"name\":\"process_name\","
               "\"args\":{\"name\":\"%s\"}}", json_escape(pname).c_str());

  std::map<uint64_t, int> tid_of;
  for (const Record& r : records) {
    if (tid_of.count(r.stream)) continue;
    const int tid = (int)tid_of.size() + 1;
    tid_of[r.stream] = tid;
    char tname[128];
    if (r.stream == 0) std::snprintf(tname, sizeof(tname), "null stream");
    else std::snprintf(tname, sizeof(tname), "stream 0x%llx",
                       (unsigned long long)r.stream);
    comma();
    std::fprintf(f,
                 "{\"ph\":\"M\",\"pid\":1,\"tid\":%d,\"name\":\"thread_name\","
                 "\"args\":{\"name\":\"%s\"}}", tid, tname);
  }

  for (const Record& r : records) {
    const double ts  = (double)(r.start - t0) / 1000.0;
    const double dur = (double)(r.end - r.start) / 1000.0;
    const int tid = tid_of[r.stream];

    comma();
    std::fprintf(f,
                 "{\"ph\":\"X\",\"pid\":1,\"tid\":%d,\"cat\":\"%s\","
                 "\"name\":\"%s\",\"ts\":%.3f,\"dur\":%.3f,\"args\":{",
                 tid, r.op.c_str(), json_escape(r.name).c_str(), ts, dur);

    bool a_first = true;
    auto arg = [&](const char* k, const std::string& v) {
      if (!a_first) std::fprintf(f, ",");
      a_first = false;
      std::fprintf(f, "\"%s\":\"%s\"", k, json_escape(v).c_str());
    };
    if (r.op == "launch") {
      arg("grid", r.grid);
      arg("block", r.block);
      arg("shared bytes", std::to_string(r.shared));
    } else {
      arg("bytes", std::to_string(r.bytes));
    }
    for (const auto& kv : r.counters) {
      if (!a_first) std::fprintf(f, ",");
      a_first = false;
      std::fprintf(f, "\"device.%s\":%llu", kv.first.c_str(),
                   (unsigned long long)kv.second);
    }
    std::fprintf(f, "}}");

    // A counter track, so device cost is visible as height in the timeline
    // rather than only in a tooltip. Raised for the slice and dropped after,
    // because cycles are a per-kernel total, not a level that persists.
    uint64_t cycles = 0;
    if (r.op == "launch" && counter(r, "cycles", &cycles)) {
      comma();
      std::fprintf(f,
                   "{\"ph\":\"C\",\"pid\":1,\"name\":\"device cycles\","
                   "\"ts\":%.3f,\"args\":{\"cycles\":%llu}}",
                   ts, (unsigned long long)cycles);
      comma();
      std::fprintf(f,
                   "{\"ph\":\"C\",\"pid\":1,\"name\":\"device cycles\","
                   "\"ts\":%.3f,\"args\":{\"cycles\":0}}", ts + dur);
    }
  }

  std::fprintf(f, "\n]}\n");
  std::fclose(f);
  return true;
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

struct KernelAgg {
  uint64_t launches = 0;
  uint64_t host_ns  = 0;
  std::map<std::string, uint64_t> counters;
  bool     has_counters = false;
};

void report(const std::vector<Record>& records, const Device& dev,
            const std::vector<Occupancy>& occ, long long overhead_cycles) {
  std::map<std::string, KernelAgg> kernels;
  uint64_t transfer_bytes[3] = {0, 0, 0};   // h2d, d2h, d2d
  uint64_t transfer_ns = 0, transfers = 0;
  uint64_t launch_ns = 0;

  for (const Record& r : records) {
    if (r.op == "launch") {
      KernelAgg& a = kernels[r.name];
      ++a.launches;
      a.host_ns += r.end - r.start;
      launch_ns += r.end - r.start;
      for (const auto& kv : r.counters) { a.counters[kv.first] += kv.second;
                                          a.has_counters = true; }
    } else {
      ++transfers;
      transfer_ns += r.end - r.start;
      if      (r.name == "h2d") transfer_bytes[0] += r.bytes;
      else if (r.name == "d2h") transfer_bytes[1] += r.bytes;
      else                      transfer_bytes[2] += r.bytes;
    }
  }

  std::printf("\n=== grx-prof ===\n");
  if (dev.known) {
    // prop.name already carries the backend in parentheses on every backend
    // that has one; repeating it reads like a bug.
    const bool named = dev.name.find(dev.backend) != std::string::npos;
    std::printf("device   %s%s%s%s, %ld core%s x %ld warps x %ld lanes,"
                " %ld KiB shared per SM\n",
                dev.name.c_str(), named ? "" : " (", named ? "" : dev.backend.c_str(),
                named ? "" : ")", dev.cores, dev.cores == 1 ? "" : "s",
                dev.warps_per_core, dev.warp_size, dev.shared_per_sm / 1024);
  }

  const size_t launches = records.size() - (size_t)transfers;
  std::printf("recorded %zu operations: %zu launch%s, %llu transfer%s\n",
              records.size(), launches, launches == 1 ? "" : "es",
              (unsigned long long)transfers, transfers == 1 ? "" : "s");

  if (kernels.empty()) {
    std::printf("\nNo kernel launches were recorded.\n");
  } else {
    std::printf("\nPer kernel, by device cycles\n");
    std::printf("  %-22s %7s %12s %6s %7s %6s %6s\n", "kernel", "calls",
                "cycles", "IPC", "lanes", "occ", "issue");

    std::vector<std::pair<uint64_t, const std::string*>> order;
    for (const auto& kv : kernels) {
      uint64_t c = 0;
      auto it = kv.second.counters.find("cycles");
      if (it != kv.second.counters.end()) c = it->second;
      order.emplace_back(c, &kv.first);
    }
    std::sort(order.begin(), order.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    for (const auto& o : order) {
      const KernelAgg& a = kernels[*o.second];
      auto cget = [&](const char* k, uint64_t* v) {
        auto it = a.counters.find(k);
        if (it == a.counters.end()) return false;
        *v = it->second; return true;
      };
      uint64_t cycles = 0, instrs = 0, iw = 0, it_ = 0, aw = 0;
      const bool have_cycles = cget("cycles", &cycles) && cycles > 0;
      cget("instrs", &instrs);
      cget("issued_warps", &iw);
      cget("issued_threads", &it_);
      cget("active_warps", &aw);

      char ipc[16] = "  -", lanes[16] = "  -", occv[16] = "  -", issue[16] = "  -";
      if (have_cycles) {
        std::snprintf(ipc, sizeof(ipc), "%.3f", (double)instrs / (double)cycles);
        std::snprintf(issue, sizeof(issue), "%.2f",
                      (double)iw / (double)cycles);
        if (iw > 0 && dev.warp_size > 0)
          std::snprintf(lanes, sizeof(lanes), "%4.0f%%",
                        100.0 * (double)it_ / ((double)iw * (double)dev.warp_size));
        const double slots =
            (double)cycles * (double)dev.warps_per_core * (double)dev.cores;
        if (slots > 0 && aw > 0)
          std::snprintf(occv, sizeof(occv), "%4.0f%%", 100.0 * (double)aw / slots);
      }
      std::printf("  %-22s %7llu %12llu %6s %7s %6s %6s\n", o.second->c_str(),
                  (unsigned long long)a.launches,
                  (unsigned long long)cycles, ipc, lanes, occv, issue);
    }
    std::printf("\n  IPC   warp-instructions retired per device cycle\n");
    std::printf("  lanes fraction of a warp's lanes active when it issued"
                " (100%% = no divergence)\n");
    std::printf("  occ   warp slots occupied, averaged over the kernel's cycles\n");
    std::printf("  issue warp-instructions issued per cycle\n");

    // Stall breakdown. Printed per kernel because a stall profile is the
    // kernel's, not the program's.
    static const struct { const char* key; const char* label; } kStalls[] = {
      {"stall_scrb",  "scoreboard (waiting on a result)"},
      {"stall_lsu",   "memory unit back-pressure"},
      {"stall_opds",  "operand collector"},
      {"stall_fetch", "instruction fetch"},
      {"stall_ibuf",  "instruction buffer empty"},
      {"stall_alu",   "ALU busy"},
      {"stall_fpu",   "FPU busy"},
      {"stall_sfu",   "SFU busy"},
      {"stall_tcu",   "tensor unit busy"},
    };
    for (const auto& kv : kernels) {
      const KernelAgg& a = kv.second;
      auto ic = a.counters.find("cycles");
      if (!a.has_counters || ic == a.counters.end() || ic->second == 0) continue;
      const double cycles = (double)ic->second;
      std::printf("\n  %s -- where the cycles went\n", kv.first.c_str());
      auto idle = a.counters.find("sched_idle");
      if (idle != a.counters.end())
        std::printf("    %-36s %5.1f%%  (%llu cycles issued nothing)\n",
                    "scheduler idle", 100.0 * (double)idle->second / cycles,
                    (unsigned long long)idle->second);
      for (const auto& s : kStalls) {
        auto it = a.counters.find(s.key);
        if (it == a.counters.end()) continue;
        if (it->second == 0) continue;
        std::printf("    %-36s %5.1f%%  (%llu)\n", s.label,
                    100.0 * (double)it->second / cycles,
                    (unsigned long long)it->second);
      }
      auto div = a.counters.find("divergence");
      if (div != a.counters.end() && div->second > 0)
        std::printf("    %-36s %llu splits\n", "branch divergence",
                    (unsigned long long)div->second);
    }
  }

  if (!occ.empty()) {
    std::printf("\nOccupancy the dispatcher will admit\n");
    for (const Occupancy& o : occ)
      std::printf("  %-22s block %4llu, shared %5llu B -> %ld block%s"
                  " = %ld of %ld warp slots\n",
                  o.kernel.c_str(), (unsigned long long)o.block,
                  (unsigned long long)o.shared, o.resident_blocks,
                  o.resident_blocks == 1 ? "" : "s", o.resident_warps,
                  o.warp_slots);
  }

  if (transfers) {
    std::printf("\nTransfers (host clock)\n");
    std::printf("  host to device   %llu bytes\n",
                (unsigned long long)transfer_bytes[0]);
    std::printf("  device to host   %llu bytes\n",
                (unsigned long long)transfer_bytes[1]);
    if (transfer_bytes[2])
      std::printf("  device to device %llu bytes\n",
                  (unsigned long long)transfer_bytes[2]);
    std::printf("  %.3f ms of host time in transfers, %.3f ms in launches\n",
                (double)transfer_ns / 1e6, (double)launch_ns / 1e6);
  }

  std::printf("\nWhat these numbers are\n");
  std::printf("  Device cycles, IPC, occupancy and the stall breakdown come"
              " from the MPM\n  performance counters. They are device"
              " measurements.\n");
  if (!dev.event_timing_is_device_side) {
    std::printf("  Every TIME above is the host clock around execution --"
                " the driver writes\n  back no device timestamps.");
    if (dev.backend == "simx" || dev.backend == "rtlsim" || dev.backend == "gem5")
      std::printf(" On %s that measures the simulator,\n  not the device."
                  " Compare kernels by cycles, not by milliseconds.",
                  dev.backend.c_str());
    std::printf("\n");
  }
  std::printf("  Profiling serializes: each operation is bracketed by a full"
              " device sync,\n  so no overlap between operations appears in the"
              " timeline.\n");
  if (overhead_cycles >= 0)
    std::printf("  One sample pair costs %lld device cycles, measured at"
                " startup. A kernel\n  whose cycle count is near that number"
                " has not really been measured.\n", overhead_cycles);
}

}  // namespace

int main(int argc, char** argv) {
  std::string out = "grxprof.json";
  bool want_trace = true, quiet = false;

  int i = 1;
  for (; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--") { ++i; break; }
    else if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--no-trace")            want_trace = false;
    else if (a == "--quiet")               quiet = true;
    else if (a == "-h" || a == "--help")   { usage(); return 0; }
    else { std::fprintf(stderr, "grx-prof: unknown option %s\n", a.c_str());
           usage(); return 2; }
  }
  if (i >= argc) { usage(); return 2; }

  std::vector<char*> child_argv;
  for (int k = i; k < argc; ++k) child_argv.push_back(argv[k]);
  child_argv.push_back(nullptr);

  int pipefd[2];
  if (::pipe(pipefd) != 0) {
    std::fprintf(stderr, "grx-prof: pipe: %s\n", std::strerror(errno));
    return 2;
  }
  const pid_t pid = ::fork();
  if (pid < 0) {
    std::fprintf(stderr, "grx-prof: fork: %s\n", std::strerror(errno));
    return 2;
  }
  if (pid == 0) {
    ::close(pipefd[0]);
    ::dup2(pipefd[1], STDERR_FILENO);
    ::close(pipefd[1]);
    ::setenv("GRX_PROFILE", "1", 1);
    ::execvp(child_argv[0], child_argv.data());
    std::fprintf(stderr, "grx-prof: cannot run %s: %s\n", child_argv[0],
                 std::strerror(errno));
    ::_exit(127);
  }
  ::close(pipefd[1]);
  std::FILE* err = ::fdopen(pipefd[0], "r");

  std::vector<Record>    records;
  std::vector<Occupancy> occupancy;
  Device      dev;
  long long   overhead_cycles = -1;
  bool        saw_summary = false;

  char buf[16384];
  while (err && std::fgets(buf, sizeof(buf), err)) {
    std::string line = buf;
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
      line.pop_back();
    if (line.rfind("GRXPROF|", 0) != 0) {
      std::fprintf(stderr, "%s\n", line.c_str());
      continue;
    }
    const Fields f = parse(line);

    if (line.rfind("GRXPROF|device", 0) == 0) {
      dev.known          = true;
      dev.name           = get(f, "name", "?");
      dev.backend        = get(f, "backend", "?");
      dev.cores          = (long)getu(f, "cores", 1);
      dev.warps_per_core = (long)getu(f, "warps_per_core", 0);
      dev.warp_size      = (long)getu(f, "warp_size", 0);
      dev.shared_per_sm  = (long)getu(f, "shared_per_sm", 0);
      dev.clock_mhz      = (long)getu(f, "clock_mhz", 0);
      dev.event_timing_is_device_side =
          getu(f, "event_timing_is_device_side", 0) != 0;
    } else if (line.rfind("GRXPROF|overhead", 0) == 0) {
      overhead_cycles = std::atoll(get(f, "cycles", "-1").c_str());
    } else if (line.rfind("GRXPROF|occupancy", 0) == 0) {
      Occupancy o;
      o.kernel          = get(f, "kernel", "?");
      o.block           = getu(f, "block");
      o.shared          = getu(f, "shared");
      o.resident_blocks = (long)getu(f, "resident_blocks");
      o.resident_warps  = (long)getu(f, "resident_warps");
      o.warp_slots      = (long)getu(f, "warp_slots");
      occupancy.push_back(o);
    } else if (line.rfind("GRXPROF|summary", 0) == 0) {
      saw_summary = true;
    } else {
      Record r;
      r.op     = get(f, "op", "?");
      r.seq    = getu(f, "seq");
      r.start  = getu(f, "host_start_ns");
      r.end    = getu(f, "host_end_ns");
      r.stream = getu(f, "stream");
      r.bytes  = getu(f, "bytes");
      r.shared = getu(f, "shared");
      r.grid   = get(f, "grid");
      r.block  = get(f, "block");
      r.name   = (r.op == "launch") ? get(f, "kernel", "?")
                                    : get(f, "kind", r.op.c_str());
      for (const auto& kv : f)
        if (kv.first.rfind("c.", 0) == 0)
          r.counters[kv.first.substr(2)] =
              std::strtoull(kv.second.c_str(), nullptr, 0);
      records.push_back(r);
    }
  }
  if (err) std::fclose(err);

  int status = 0;
  ::waitpid(pid, &status, 0);
  const int child_rc = WIFEXITED(status) ? WEXITSTATUS(status) : 128;

  std::sort(records.begin(), records.end(),
            [](const Record& a, const Record& b) { return a.seq < b.seq; });

  if (!saw_summary) {
    std::fprintf(stderr,
                 "grx-prof: the runtime emitted no summary -- it was not in"
                 " profiling mode.\n         Nothing here describes the run.\n");
    return child_rc ? child_rc : 1;
  }
  if (records.empty()) {
    std::fprintf(stderr,
                 "grx-prof: profiling was on, but the program launched no"
                 " kernels and moved no memory.\n");
    return child_rc;
  }

  const uint64_t t0 = records.front().start;
  if (want_trace) {
    if (write_trace(out, records, dev, t0))
      std::printf("\ngrx-prof: wrote %s (%zu events) -- open it at"
                  " https://ui.perfetto.dev\n", out.c_str(), records.size());
    else
      std::fprintf(stderr, "grx-prof: could not write %s: %s\n", out.c_str(),
                   std::strerror(errno));
  }
  if (!quiet) report(records, dev, occupancy, overhead_cycles);
  return child_rc;
}
