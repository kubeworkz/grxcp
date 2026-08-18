// Mock GRX-G100 driver — the subset of vortex2.h the runtime uses, backed by
// synthetic capability values.
//
// Purpose: let the GRXCP runtime and tools be compiled, linked, and executed in
// CI without a Vortex sysroot or a simulator. It is a TEST FIXTURE, never a
// fallback: nothing in libgrxrt may prefer it, and it is not installed.
//
// Defaults mirror the GRX-G100 repo's default VX_config.toml (NUM_WARPS=4,
// NUM_THREADS=4). Override any value through the environment to exercise other
// shapes, e.g. the flagship preset:
//
//   GRXMOCK_NUM_THREADS=32 GRXMOCK_NUM_WARPS=64
//   GRXMOCK_NUM_CORES=16   GRXMOCK_NUM_CLUSTERS=8   ./grx-smi

#include <vortex2.h>

#include <cstdlib>
#include <cstring>
#include <string>

namespace {

uint64_t env_u64(const char* name, uint64_t fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  return std::strtoull(v, nullptr, 0);
}

struct MockDevice { uint32_t index; };
MockDevice g_devices[1] = {{0}};

}  // namespace

extern "C" {

vx_result_t vx_device_count(uint32_t* out_count) {
  if (!out_count) return VX_ERR_INVALID_VALUE;
  *out_count = (uint32_t)env_u64("GRXMOCK_DEVICE_COUNT", 1);
  return VX_SUCCESS;
}

vx_result_t vx_device_open(uint32_t index, vx_device_h* out) {
  if (!out) return VX_ERR_INVALID_VALUE;
  if (index >= (uint32_t)env_u64("GRXMOCK_DEVICE_COUNT", 1))
    return VX_ERR_INVALID_VALUE;
  *out = &g_devices[0];
  return VX_SUCCESS;
}

vx_result_t vx_device_retain (vx_device_h) { return VX_SUCCESS; }
vx_result_t vx_device_release(vx_device_h) { return VX_SUCCESS; }

vx_result_t vx_device_query(vx_device_h dev, uint32_t caps_id, uint64_t* out) {
  if (!dev || !out) return VX_ERR_INVALID_HANDLE;

  const uint64_t threads  = env_u64("GRXMOCK_NUM_THREADS",  4);
  const uint64_t warps    = env_u64("GRXMOCK_NUM_WARPS",    4);
  const uint64_t cores    = env_u64("GRXMOCK_NUM_CORES",    1);
  const uint64_t clusters = env_u64("GRXMOCK_NUM_CLUSTERS", 1);

  switch (caps_id) {
    case VX_CAPS_VERSION:         *out = 2; return VX_SUCCESS;
    case VX_CAPS_NUM_THREADS:     *out = threads; return VX_SUCCESS;
    case VX_CAPS_NUM_WARPS:       *out = warps; return VX_SUCCESS;
    // Matches the hardware contract: VX_CSR_NUM_CORES reports cores x clusters.
    case VX_CAPS_NUM_CORES:       *out = cores * clusters; return VX_SUCCESS;
    case VX_CAPS_NUM_CLUSTERS:    *out = clusters; return VX_SUCCESS;
    case VX_CAPS_SOCKET_SIZE:     *out = env_u64("GRXMOCK_SOCKET_SIZE", 1); return VX_SUCCESS;
    case VX_CAPS_ISSUE_WIDTH:     *out = env_u64("GRXMOCK_ISSUE_WIDTH", 1); return VX_SUCCESS;
    case VX_CAPS_CACHE_LINE_SIZE: *out = env_u64("GRXMOCK_CACHE_LINE", 64); return VX_SUCCESS;
    case VX_CAPS_GLOBAL_MEM_SIZE: *out = env_u64("GRXMOCK_GLOBAL_MEM", 1ull << 32); return VX_SUCCESS;
    case VX_CAPS_LOCAL_MEM_SIZE:  *out = env_u64("GRXMOCK_LOCAL_MEM", 16ull << 10); return VX_SUCCESS;
    case VX_CAPS_NUM_MEM_BANKS:   *out = env_u64("GRXMOCK_MEM_BANKS", 2); return VX_SUCCESS;
    case VX_CAPS_MEM_BANK_SIZE:   *out = env_u64("GRXMOCK_MEM_BANK_SIZE", 1ull << 31); return VX_SUCCESS;
    case VX_CAPS_CLOCK_RATE:      *out = env_u64("GRXMOCK_CLOCK_MHZ", 300); return VX_SUCCESS;
    case VX_CAPS_PEAK_MEM_BW:     *out = env_u64("GRXMOCK_PEAK_BW_MBS", 19200); return VX_SUCCESS;
    case VX_CAPS_VM_SUPPORT:      *out = env_u64("GRXMOCK_VM_SUPPORT", 1); return VX_SUCCESS;
    case VX_CAPS_VM_PINNED_SIZE:  *out = env_u64("GRXMOCK_PINNED_SIZE", 0); return VX_SUCCESS;
    case VX_CAPS_VM_PINNED_FREE:  *out = env_u64("GRXMOCK_PINNED_FREE", 0); return VX_SUCCESS;
    case VX_CAPS_ISA_FLAGS: {
      // RV64IMAFDC plus the custom extension bits the default build enables.
      // vortex2.h exposes no VX_ISA_STD_M constant, so the M bit is omitted
      // rather than hardcoded from the misa layout.
      uint64_t f = VX_ISA_STD_I | VX_ISA_STD_A | VX_ISA_STD_F |
                   VX_ISA_STD_D | VX_ISA_STD_C |
                   VX_ISA_EXT_ICACHE | VX_ISA_EXT_DCACHE | VX_ISA_EXT_LMEM;
      if (env_u64("GRXMOCK_TCU", 1)) f |= VX_ISA_EXT_TCU;
      if (env_u64("GRXMOCK_DXA", 1)) f |= VX_ISA_EXT_DXA;
      if (env_u64("GRXMOCK_RTU", 0)) f |= VX_ISA_EXT_RTU;
      *out = f;
      return VX_SUCCESS;
    }
    default:
      return VX_ERR_INVALID_INFO;
  }
}

vx_result_t vx_device_memory_info(vx_device_h dev, uint64_t* freeBytes,
                                  uint64_t* usedBytes) {
  if (!dev) return VX_ERR_INVALID_HANDLE;
  const uint64_t total = env_u64("GRXMOCK_GLOBAL_MEM", 1ull << 32);
  const uint64_t used  = env_u64("GRXMOCK_MEM_USED", 0);
  if (freeBytes) *freeBytes = total - used;
  if (usedBytes) *usedBytes = used;
  return VX_SUCCESS;
}

const char* vx_result_string(vx_result_t r) {
  return (r == VX_SUCCESS) ? "success" : "error";
}

}  // extern "C"
