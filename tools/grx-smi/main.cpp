// grx-smi — GRX system management interface.
//
// Phase 0 deliverable and its exit gate: print the device inventory with every
// field sourced from the driver's capability queries. If a number appears here
// that grxGetDeviceProperties could not source, that is a bug in the runtime,
// not a formatting choice in this tool.
//
//   grx-smi            human-readable inventory
//   grx-smi --json     machine-readable, for CI comparison
//   grx-smi --caps     add the raw capability-ID dump

#include <grx/grx.h>

#include <cstdio>
#include <cstring>
#include <string>

namespace {

const char* yes_no(int v) { return v ? "yes" : "no"; }

std::string bytes_human(size_t b) {
  static const char* unit[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double v = (double)b;
  int u = 0;
  while (v >= 1024.0 && u < 4) { v /= 1024.0; ++u; }
  char buf[64];
  std::snprintf(buf, sizeof(buf), (u == 0) ? "%.0f %s" : "%.1f %s", v, unit[u]);
  return buf;
}

std::string caps_list(unsigned c) {
  struct { unsigned bit; const char* name; } table[] = {
    {GRX_CAP_KERNEL_LAUNCH,      "launch"},
    {GRX_CAP_STREAMS,            "streams"},
    {GRX_CAP_EVENTS,             "events"},
    {GRX_CAP_MEMCPY,             "memcpy"},
    {GRX_CAP_GEMM,               "gemm"},
    {GRX_CAP_UNIFIED_ADDRESSING, "uva"},
    {GRX_CAP_TENSOR_CORE,        "tensor"},
    {GRX_CAP_ASYNC_COPY,         "async-copy"},
    {GRX_CAP_RAY_TRACING,        "raytrace"},
    {GRX_CAP_COOPERATIVE_LAUNCH, "cooperative"},
  };
  std::string s;
  for (auto& e : table) {
    if (c & e.bit) { if (!s.empty()) s += " "; s += e.name; }
  }
  return s.empty() ? "(none)" : s;
}

// Fields the stack cannot yet source report -1. Print that as "unknown" rather
// than as a number, so nobody reads a sentinel as data.
std::string int_or_unknown(int v) {
  return (v < 0) ? std::string("unknown") : std::to_string(v);
}

void print_human(int index, const grxDeviceProp_t& p) {
  std::printf("Device %d: %s\n", index, p.name);
  std::printf("  compute capability     %d.%d\n",
              p.computeCapabilityMajor, p.computeCapabilityMinor);
  std::printf("  capabilities           %s\n", caps_list(p.capabilities).c_str());
  std::printf("\n  execution\n");
  std::printf("    SMs (cores x clusters) %d  (%d clusters, socket size %d)\n",
              p.multiProcessorCount, p.clusterCount, p.socketSize);
  std::printf("    warp size              %d threads\n", p.warpSize);
  std::printf("    warps per SM           %d  (%d threads resident)\n",
              p.maxWarpsPerMultiProcessor,
              p.warpSize * p.maxWarpsPerMultiProcessor);
  std::printf("    issue width            %d\n", p.issueWidth);
  std::printf("    max threads per block  %d\n", p.maxThreadsPerBlock);
  std::printf("    barriers per core      %s\n",
              int_or_unknown(p.numBarriers).c_str());
  std::printf("\n  memory\n");
  std::printf("    global                 %s", bytes_human(p.totalGlobalMem).c_str());
  if (p.memBankCount > 0)
    std::printf("  (%d banks x %s)", p.memBankCount,
                bytes_human(p.memBankSize).c_str());
  std::printf("\n");
  std::printf("    shared per SM          %s\n",
              bytes_human(p.sharedMemPerMultiprocessor).c_str());
  std::printf("    cache line             %d B\n", p.cacheLineSize);
  std::printf("    peak bandwidth         %zu MB/s\n", p.peakMemoryBandwidthMBs);
  std::printf("    clock                  %d MHz\n", p.clockRateMHz);
  std::printf("    unified addressing     %s\n", yes_no(p.unifiedAddressing));
  std::printf("    managed memory         %s\n", yes_no(p.managedMemory));
  if (p.pinnedMemTotal > 0)
    std::printf("    pinned region          %s free of %s\n",
                bytes_human(p.pinnedMemFree).c_str(),
                bytes_human(p.pinnedMemTotal).c_str());

  // Surfaced prominently and unconditionally. A user comparing GRXCP numbers
  // against CUDA numbers needs to know which of them came from a software
  // stand-in before they draw a conclusion.
  std::printf("\n  software stand-ins in effect\n");
  std::printf("    warp shuffle           %s\n",
              p.warpShuffleIsEmulated ? "EMULATED via local memory (no WSHFL instruction)"
                                      : "native");
  std::printf("    event timing           %s\n",
              p.eventTimingIsDeviceSide
                  ? "device-side"
                  : "HOST CLOCK around execution (CP writes back no timestamps)");
  if (!p.eventTimingIsDeviceSide &&
      (p.backend == GRX_BACKEND_SIMX || p.backend == GRX_BACKEND_RTLSIM ||
       p.backend == GRX_BACKEND_GEM5))
    std::printf("                           on a simulator that measures the "
                "SIMULATOR, not the device\n");
  std::printf("    __constant__           %s\n",
              p.constantMemoryIsGlobal ? "read-only global (no broadcast path)"
                                       : "constant cache");
  std::printf("\n");
}

void print_json(int index, const grxDeviceProp_t& p, bool last) {
  std::printf("  {\n");
  std::printf("    \"index\": %d,\n", index);
  std::printf("    \"name\": \"%s\",\n", p.name);
  std::printf("    \"backend\": \"%s\",\n",
              p.backend == GRX_BACKEND_SIMX    ? "simx"    :
              p.backend == GRX_BACKEND_RTLSIM  ? "rtlsim"  :
              p.backend == GRX_BACKEND_XRT     ? "xrt"     :
              p.backend == GRX_BACKEND_OPAE    ? "opae"    :
              p.backend == GRX_BACKEND_GEM5    ? "gem5"    : "silicon");
  std::printf("    \"computeCapability\": \"%d.%d\",\n",
              p.computeCapabilityMajor, p.computeCapabilityMinor);
  std::printf("    \"warpSize\": %d,\n", p.warpSize);
  std::printf("    \"warpsPerSM\": %d,\n", p.maxWarpsPerMultiProcessor);
  std::printf("    \"multiProcessorCount\": %d,\n", p.multiProcessorCount);
  std::printf("    \"clusterCount\": %d,\n", p.clusterCount);
  std::printf("    \"socketSize\": %d,\n", p.socketSize);
  std::printf("    \"issueWidth\": %d,\n", p.issueWidth);
  std::printf("    \"maxThreadsPerBlock\": %d,\n", p.maxThreadsPerBlock);
  std::printf("    \"totalGlobalMem\": %zu,\n", p.totalGlobalMem);
  std::printf("    \"sharedMemPerSM\": %zu,\n", p.sharedMemPerMultiprocessor);
  std::printf("    \"cacheLineSize\": %d,\n", p.cacheLineSize);
  std::printf("    \"clockRateMHz\": %d,\n", p.clockRateMHz);
  std::printf("    \"peakMemoryBandwidthMBs\": %zu,\n", p.peakMemoryBandwidthMBs);
  std::printf("    \"unifiedAddressing\": %s,\n", p.unifiedAddressing ? "true" : "false");
  std::printf("    \"managedMemory\": %s,\n", p.managedMemory ? "true" : "false");
  std::printf("    \"capabilities\": \"%s\",\n", caps_list(p.capabilities).c_str());
  std::printf("    \"warpShuffleIsEmulated\": %s,\n", p.warpShuffleIsEmulated ? "true" : "false");
  std::printf("    \"eventTimingIsDeviceSide\": %s,\n", p.eventTimingIsDeviceSide ? "true" : "false");
  std::printf("    \"constantMemoryIsGlobal\": %s\n", p.constantMemoryIsGlobal ? "true" : "false");
  std::printf("  }%s\n", last ? "" : ",");
}

}  // namespace

int main(int argc, char** argv) {
  bool json = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--json")) json = true;
    else if (!std::strcmp(argv[i], "--help")) {
      std::printf("usage: grx-smi [--json]\n");
      return 0;
    } else {
      std::fprintf(stderr, "grx-smi: unknown option '%s'\n", argv[i]);
      return 2;
    }
  }

  int count = 0;
  grxError_t err = grxGetDeviceCount(&count);
  if (err != grxSuccess) {
    std::fprintf(stderr, "grx-smi: %s (%s)\n", grxGetErrorString(err),
                 grxGetErrorName(err));
    return 1;
  }
  if (count == 0) {
    std::fprintf(stderr, "grx-smi: no GRX devices found\n");
    return 1;
  }

  if (json) std::printf("[\n");
  for (int i = 0; i < count; ++i) {
    grxDeviceProp_t p{};
    err = grxGetDeviceProperties(&p, i);
    if (err != grxSuccess) {
      std::fprintf(stderr, "grx-smi: device %d: %s\n", i, grxGetErrorString(err));
      return 1;
    }
    if (json) print_json(i, p, i == count - 1);
    else      print_human(i, p);
  }
  if (json) std::printf("]\n");
  return 0;
}
