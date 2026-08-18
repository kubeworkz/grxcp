// Mock GRX-G100 driver — the subset of vortex2.h the runtime uses, backed by
// synthetic capabilities and host memory.
//
// Purpose: let the GRXCP runtime and tools be compiled, linked, and executed in
// CI without a Vortex sysroot or a simulator. It is a TEST FIXTURE, never a
// fallback: nothing in libgrxrt may prefer it, and it is not installed.
//
// WHAT IT MODELS FAITHFULLY
//   * buffer allocation, addressing, and the read/write/copy/fill data paths,
//     so a memcpy test verifies real bytes moving through real offsets
//   * timeline event counters with monotonic, never-decreasing signal
//   * handle lifetimes and refcounts, so leaks and use-after-release surface
//   * the launch descriptor: the grid, block, cluster, shared-memory size and
//     argument blob the runtime hands the driver are recorded for inspection
//     (vortex_mock.h), which is the part of the launch path testable without a
//     simulator
//
// WHAT IT DOES NOT MODEL
//   * asynchrony. Every enqueue executes immediately and its event is signaled
//     before the call returns. Ordering is therefore trivially correct here,
//     which means the mock can prove data correctness but can prove NOTHING
//     about concurrency, overlap, or race conditions. Those need a real
//     backend (ci/README.md, tier 2).
//   * timing. Profiling timestamps are refused, exactly as the real command
//     processor refuses them today, so the runtime exercises its host-clock
//     fallback path rather than a fiction.
//   * kernel execution. There is no RISC-V core here. A launch is recorded and
//     retired; it computes nothing. Anything that depends on a kernel actually
//     running is a tier-2 test.
//
// Defaults mirror the GRX-G100 repo's default VX_config.toml (NUM_WARPS=4,
// NUM_THREADS=4). Override any value through the environment, e.g. the
// flagship preset:
//
//   GRXMOCK_NUM_THREADS=32 GRXMOCK_NUM_WARPS=64
//   GRXMOCK_NUM_CORES=16   GRXMOCK_NUM_CLUSTERS=8   ./grx-smi

#include <vortex2.h>

#include "vortex_mock.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {

uint64_t env_u64(const char* name, uint64_t fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  return std::strtoull(v, nullptr, 0);
}

struct MockDevice { uint32_t index; };
MockDevice g_devices[1] = {{0}};

// ---------------------------------------------------------------------------
// Device address space
//
// Device addresses are synthetic and deliberately do NOT alias host pointers:
// a runtime bug that treats a device address as dereferenceable segfaults here
// instead of silently working, which is the whole point of testing against a
// model rather than against malloc.
// ---------------------------------------------------------------------------

constexpr uint64_t kDeviceBase  = 0x0000'1000'0000'0000ull;
constexpr uint64_t kDeviceAlign = 256;

struct MockBuffer {
  uint64_t             address = 0;
  uint64_t             size    = 0;
  uint32_t             flags   = 0;
  std::vector<uint8_t> storage;
  std::atomic<int>     refcount{1};
  bool                 reserved = false;   // vx_buffer_reserve: no storage owned
};

struct MockQueue {
  vx_queue_info_t  info{};
  std::atomic<int> refcount{1};
};

struct MockEvent {
  std::atomic<uint64_t> value{0};
  std::atomic<int>      refcount{1};
};

std::mutex g_mutex;
uint64_t   g_next_address = kDeviceBase;
uint64_t   g_bytes_in_use = 0;
std::map<uint64_t, MockBuffer*> g_buffers;   // address -> buffer

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

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
  std::lock_guard<std::mutex> lock(g_mutex);
  const uint64_t total = env_u64("GRXMOCK_GLOBAL_MEM", 1ull << 32);
  if (freeBytes) *freeBytes = total - g_bytes_in_use;
  if (usedBytes) *usedBytes = g_bytes_in_use;
  return VX_SUCCESS;
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

vx_result_t vx_buffer_create(vx_device_h dev, uint64_t size, uint32_t flags,
                             vx_buffer_h* out) {
  if (!dev || !out || size == 0) return VX_ERR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(g_mutex);

  const uint64_t total = env_u64("GRXMOCK_GLOBAL_MEM", 1ull << 32);
  if (g_bytes_in_use + size > total) return VX_ERR_OUT_OF_DEVICE_MEMORY;

  auto* b = new MockBuffer();
  b->address = g_next_address;
  b->size    = size;
  b->flags   = flags;
  b->storage.assign((size_t)size, 0);

  g_next_address += (size + kDeviceAlign - 1) & ~(kDeviceAlign - 1);
  g_bytes_in_use += size;
  g_buffers[b->address] = b;

  *out = b;
  return VX_SUCCESS;
}

vx_result_t vx_buffer_reserve(vx_device_h dev, uint64_t address, uint64_t size,
                              uint32_t flags, vx_buffer_h* out) {
  if (!dev || !out || size == 0) return VX_ERR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(g_mutex);
  auto* b = new MockBuffer();
  b->address  = address;
  b->size     = size;
  b->flags    = flags;
  b->reserved = true;
  *out = b;
  return VX_SUCCESS;
}

vx_result_t vx_buffer_retain(vx_buffer_h buf) {
  if (!buf) return VX_ERR_INVALID_HANDLE;
  static_cast<MockBuffer*>(buf)->refcount.fetch_add(1);
  return VX_SUCCESS;
}

vx_result_t vx_buffer_release(vx_buffer_h buf) {
  if (!buf) return VX_ERR_INVALID_HANDLE;
  auto* b = static_cast<MockBuffer*>(buf);
  if (b->refcount.fetch_sub(1) != 1) return VX_SUCCESS;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!b->reserved) {
    g_buffers.erase(b->address);
    g_bytes_in_use -= b->size;
  }
  delete b;
  return VX_SUCCESS;
}

vx_result_t vx_buffer_address(vx_buffer_h buf, uint64_t* out_addr) {
  if (!buf || !out_addr) return VX_ERR_INVALID_HANDLE;
  *out_addr = static_cast<MockBuffer*>(buf)->address;
  return VX_SUCCESS;
}

vx_result_t vx_buffer_access(vx_buffer_h buf, uint64_t, uint64_t, uint32_t) {
  return buf ? VX_SUCCESS : VX_ERR_INVALID_HANDLE;
}

vx_result_t vx_buffer_map(vx_buffer_h buf, uint64_t offset, uint64_t size,
                          uint32_t, void** out_host_ptr) {
  if (!buf || !out_host_ptr) return VX_ERR_INVALID_HANDLE;
  auto* b = static_cast<MockBuffer*>(buf);
  if (b->reserved || offset + size > b->size) return VX_ERR_INVALID_VALUE;
  *out_host_ptr = b->storage.data() + offset;
  return VX_SUCCESS;
}

vx_result_t vx_buffer_unmap(vx_buffer_h buf, void*) {
  return buf ? VX_SUCCESS : VX_ERR_INVALID_HANDLE;
}

// ---------------------------------------------------------------------------
// Queues
// ---------------------------------------------------------------------------

vx_result_t vx_queue_create(vx_device_h dev, const vx_queue_info_t* info,
                            vx_queue_h* out) {
  if (!dev || !out) return VX_ERR_INVALID_VALUE;
  auto* q = new MockQueue();
  if (info) q->info = *info;
  *out = reinterpret_cast<vx_queue_h>(q);
  return VX_SUCCESS;
}

vx_result_t vx_queue_retain(vx_queue_h q) {
  if (!q) return VX_ERR_INVALID_HANDLE;
  reinterpret_cast<MockQueue*>(q)->refcount.fetch_add(1);
  return VX_SUCCESS;
}

vx_result_t vx_queue_release(vx_queue_h q) {
  if (!q) return VX_ERR_INVALID_HANDLE;
  auto* mq = reinterpret_cast<MockQueue*>(q);
  if (mq->refcount.fetch_sub(1) == 1) delete mq;
  return VX_SUCCESS;
}

// Every enqueue below completes before it returns, so flush and finish have
// nothing left to wait for.
vx_result_t vx_queue_flush (vx_queue_h q)            { return q ? VX_SUCCESS : VX_ERR_INVALID_HANDLE; }
vx_result_t vx_queue_finish(vx_queue_h q, uint64_t)  { return q ? VX_SUCCESS : VX_ERR_INVALID_HANDLE; }

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

vx_result_t vx_event_create(vx_device_h dev, vx_event_h* out) {
  if (!dev || !out) return VX_ERR_INVALID_VALUE;
  *out = reinterpret_cast<vx_event_h>(new MockEvent());
  return VX_SUCCESS;
}

vx_result_t vx_event_signal(vx_event_h ev, uint64_t value) {
  if (!ev) return VX_ERR_INVALID_HANDLE;
  auto* e = reinterpret_cast<MockEvent*>(ev);
  uint64_t cur = e->value.load();
  while (value > cur && !e->value.compare_exchange_weak(cur, value)) {}
  return VX_SUCCESS;
}

vx_result_t vx_event_get_value(vx_event_h ev, uint64_t* out_value) {
  if (!ev || !out_value) return VX_ERR_INVALID_HANDLE;
  *out_value = reinterpret_cast<MockEvent*>(ev)->value.load();
  return VX_SUCCESS;
}

vx_result_t vx_event_wait_value(vx_event_h ev, uint64_t value, uint64_t) {
  if (!ev) return VX_ERR_INVALID_HANDLE;
  // Work completes synchronously here, so a wait either is already satisfied
  // or never will be. Reporting a timeout beats spinning forever.
  return (reinterpret_cast<MockEvent*>(ev)->value.load() >= value)
             ? VX_SUCCESS : VX_ERR_TIMEOUT;
}

vx_result_t vx_event_wait_values(uint32_t n, const vx_event_h* evs,
                                 const uint64_t* values, uint64_t timeout) {
  if (n && (!evs || !values)) return VX_ERR_INVALID_VALUE;
  for (uint32_t i = 0; i < n; ++i) {
    vx_result_t r = vx_event_wait_value(evs[i], values[i], timeout);
    if (r != VX_SUCCESS) return r;
  }
  return VX_SUCCESS;
}

vx_result_t vx_event_retain(vx_event_h ev) {
  if (!ev) return VX_ERR_INVALID_HANDLE;
  reinterpret_cast<MockEvent*>(ev)->refcount.fetch_add(1);
  return VX_SUCCESS;
}

vx_result_t vx_event_release(vx_event_h ev) {
  if (!ev) return VX_ERR_INVALID_HANDLE;
  auto* e = reinterpret_cast<MockEvent*>(ev);
  if (e->refcount.fetch_sub(1) == 1) delete e;
  return VX_SUCCESS;
}

// Refused on purpose. The real command processor's profiling writeback is
// still a skeleton, so the runtime must exercise its host-clock fallback here
// rather than be handed a number the hardware cannot yet produce.
vx_result_t vx_event_get_profiling(vx_event_h, vx_profile_info_t*) {
  return VX_ERR_NOT_SUPPORTED;
}

// ---------------------------------------------------------------------------
// Enqueue
// ---------------------------------------------------------------------------

namespace {

// Every enqueue signals its completion event to 1 before returning.
vx_result_t complete(vx_queue_h q, vx_event_h* out_event) {
  if (!out_event) return VX_SUCCESS;
  vx_event_h ev = nullptr;
  vx_result_t r = vx_event_create(&g_devices[0], &ev);
  if (r != VX_SUCCESS) return r;
  vx_event_signal(ev, 1);
  *out_event = ev;
  (void)q;
  return VX_SUCCESS;
}

}  // namespace

vx_result_t vx_enqueue_write(vx_queue_h q, vx_buffer_h dst, uint64_t dst_off,
                             const void* host_src, uint64_t size, uint32_t,
                             const vx_event_h*, vx_event_h* out_event) {
  if (!q || !dst || (!host_src && size)) return VX_ERR_INVALID_HANDLE;
  auto* b = static_cast<MockBuffer*>(dst);
  if (b->reserved || dst_off + size > b->size) return VX_ERR_INVALID_VALUE;
  std::memcpy(b->storage.data() + dst_off, host_src, (size_t)size);
  return complete(q, out_event);
}

vx_result_t vx_enqueue_read(vx_queue_h q, void* host_dst, vx_buffer_h src,
                            uint64_t src_off, uint64_t size, uint32_t,
                            const vx_event_h*, vx_event_h* out_event) {
  if (!q || !src || (!host_dst && size)) return VX_ERR_INVALID_HANDLE;
  auto* b = static_cast<MockBuffer*>(src);
  if (b->reserved || src_off + size > b->size) return VX_ERR_INVALID_VALUE;
  std::memcpy(host_dst, b->storage.data() + src_off, (size_t)size);
  return complete(q, out_event);
}

vx_result_t vx_enqueue_copy(vx_queue_h q, vx_buffer_h dst, uint64_t dst_off,
                            vx_buffer_h src, uint64_t src_off, uint64_t size,
                            uint32_t, const vx_event_h*, vx_event_h* out_event) {
  if (!q || !dst || !src) return VX_ERR_INVALID_HANDLE;
  auto* d = static_cast<MockBuffer*>(dst);
  auto* s = static_cast<MockBuffer*>(src);
  if (d->reserved || s->reserved) return VX_ERR_INVALID_VALUE;
  if (dst_off + size > d->size || src_off + size > s->size)
    return VX_ERR_INVALID_VALUE;
  std::memmove(d->storage.data() + dst_off, s->storage.data() + src_off,
               (size_t)size);
  return complete(q, out_event);
}

vx_result_t vx_enqueue_fill_buffer(vx_queue_h q, vx_buffer_h dst,
                                   uint64_t offset, uint64_t size,
                                   const void* pattern, size_t pattern_size,
                                   uint32_t, const vx_event_h*,
                                   vx_event_h* out_event) {
  if (!q || !dst || !pattern || pattern_size == 0) return VX_ERR_INVALID_HANDLE;
  if (size % pattern_size != 0) return VX_ERR_INVALID_VALUE;
  auto* b = static_cast<MockBuffer*>(dst);
  if (b->reserved || offset + size > b->size) return VX_ERR_INVALID_VALUE;
  for (uint64_t i = 0; i < size; i += pattern_size)
    std::memcpy(b->storage.data() + offset + i, pattern, pattern_size);
  return complete(q, out_event);
}

vx_result_t vx_enqueue_barrier(vx_queue_h q, uint32_t, const vx_event_h*,
                               vx_event_h* out_event) {
  if (!q) return VX_ERR_INVALID_HANDLE;
  return complete(q, out_event);
}

vx_result_t vx_enqueue_signal(vx_queue_h q, vx_event_h ev, uint64_t value,
                              uint32_t, const vx_event_h*,
                              vx_event_h* out_event) {
  if (!q || !ev) return VX_ERR_INVALID_HANDLE;
  vx_event_signal(ev, value);
  return complete(q, out_event);
}

vx_result_t vx_enqueue_wait_value(vx_queue_h q, vx_event_h ev, uint64_t value,
                                  uint32_t, const vx_event_h*,
                                  vx_event_h* out_event) {
  if (!q || !ev) return VX_ERR_INVALID_HANDLE;
  if (reinterpret_cast<MockEvent*>(ev)->value.load() < value)
    return VX_ERR_TIMEOUT;
  return complete(q, out_event);
}

const char* vx_result_string(vx_result_t r) {
  switch (r) {
    case VX_SUCCESS:                  return "success";
    case VX_ERR_INVALID_HANDLE:       return "invalid handle";
    case VX_ERR_INVALID_INFO:         return "invalid info";
    case VX_ERR_INVALID_VALUE:        return "invalid value";
    case VX_ERR_OUT_OF_HOST_MEMORY:   return "out of host memory";
    case VX_ERR_OUT_OF_DEVICE_MEMORY: return "out of device memory";
    case VX_ERR_DEVICE_LOST:          return "device lost";
    case VX_ERR_TIMEOUT:              return "timeout";
    case VX_ERR_EVENT_FAILED:         return "event failed";
    case VX_ERR_NOT_SUPPORTED:        return "not supported";
    case VX_ERR_INTERNAL:             return "internal error";
  }
  return "unknown";
}

}  // extern "C"

// ---------------------------------------------------------------------------
// Modules, kernels, and launch
// ---------------------------------------------------------------------------

namespace {

// Mock module image: "GRXMOCKMOD" followed by NUL-terminated kernel names and
// a terminating empty name. Not the real .vxbin encoding -- the mock models
// the contract (named entries resolve to kernels), not the file format.
constexpr char kModuleMagic[] = "GRXMOCKMOD";

struct MockKernel {
  std::string      name;
  uint64_t         entry_pc = 0;
  std::atomic<int> refcount{1};
};

struct MockModule {
  std::vector<MockKernel*> kernels;
  std::atomic<int>         refcount{1};
};

grxmock_launch_record g_last_launch{};
uint32_t              g_launch_count = 0;

}  // namespace

extern "C" {

size_t grxmock_build_module(void* buffer, size_t capacity,
                            const char* const* kernel_names, uint32_t count) {
  size_t need = sizeof(kModuleMagic);
  for (uint32_t i = 0; i < count; ++i) need += std::strlen(kernel_names[i]) + 1;
  need += 1;   // terminating empty name
  if (!buffer || capacity < need) return 0;

  auto* out = (char*)buffer;
  std::memcpy(out, kModuleMagic, sizeof(kModuleMagic));
  size_t at = sizeof(kModuleMagic);
  for (uint32_t i = 0; i < count; ++i) {
    const size_t n = std::strlen(kernel_names[i]) + 1;
    std::memcpy(out + at, kernel_names[i], n);
    at += n;
  }
  out[at++] = '\0';
  return at;
}

const grxmock_launch_record* grxmock_last_launch(void) { return &g_last_launch; }
uint32_t grxmock_launch_count(void) { return g_launch_count; }
void grxmock_reset_launches(void) {
  g_last_launch = grxmock_launch_record{};
  g_launch_count = 0;
}

vx_result_t vx_module_load_bytes(vx_device_h dev, const void* bytes,
                                 size_t size, vx_module_h* out) {
  if (!dev || !bytes || !out) return VX_ERR_INVALID_VALUE;
  if (size < sizeof(kModuleMagic) ||
      std::memcmp(bytes, kModuleMagic, sizeof(kModuleMagic)) != 0)
    return VX_ERR_INVALID_VALUE;

  auto* m = new MockModule();
  const char* p   = (const char*)bytes + sizeof(kModuleMagic);
  const char* end = (const char*)bytes + size;
  uint64_t pc = 0x8000'0000ull;
  while (p < end && *p) {
    auto* k = new MockKernel();
    k->name     = p;
    k->entry_pc = pc;
    pc += 0x1000;
    m->kernels.push_back(k);
    p += k->name.size() + 1;
  }
  *out = reinterpret_cast<vx_module_h>(m);
  return VX_SUCCESS;
}

vx_result_t vx_module_load_file(vx_device_h dev, const char* path,
                                vx_module_h* out) {
  if (!dev || !path || !out) return VX_ERR_INVALID_VALUE;
  return VX_ERR_NOT_SUPPORTED;   // the runtime reads files itself
}

vx_result_t vx_module_retain(vx_module_h mod) {
  if (!mod) return VX_ERR_INVALID_HANDLE;
  reinterpret_cast<MockModule*>(mod)->refcount.fetch_add(1);
  return VX_SUCCESS;
}

vx_result_t vx_module_release(vx_module_h mod) {
  if (!mod) return VX_ERR_INVALID_HANDLE;
  auto* m = reinterpret_cast<MockModule*>(mod);
  if (m->refcount.fetch_sub(1) != 1) return VX_SUCCESS;
  for (auto* k : m->kernels) if (k->refcount.fetch_sub(1) == 1) delete k;
  delete m;
  return VX_SUCCESS;
}

vx_result_t vx_module_get_kernel(vx_module_h mod, const char* name,
                                 vx_kernel_h* out) {
  if (!mod || !name || !out) return VX_ERR_INVALID_HANDLE;
  auto* m = reinterpret_cast<MockModule*>(mod);
  for (auto* k : m->kernels) {
    if (k->name == name) {
      k->refcount.fetch_add(1);
      *out = reinterpret_cast<vx_kernel_h>(k);
      return VX_SUCCESS;
    }
  }
  return VX_ERR_INVALID_VALUE;
}

vx_result_t vx_kernel_retain(vx_kernel_h k) {
  if (!k) return VX_ERR_INVALID_HANDLE;
  reinterpret_cast<MockKernel*>(k)->refcount.fetch_add(1);
  return VX_SUCCESS;
}

vx_result_t vx_kernel_release(vx_kernel_h k) {
  if (!k) return VX_ERR_INVALID_HANDLE;
  auto* mk = reinterpret_cast<MockKernel*>(k);
  if (mk->refcount.fetch_sub(1) == 1) delete mk;
  return VX_SUCCESS;
}

vx_result_t vx_kernel_address(vx_kernel_h k, uint64_t* out_addr) {
  if (!k || !out_addr) return VX_ERR_INVALID_HANDLE;
  *out_addr = reinterpret_cast<MockKernel*>(k)->entry_pc;
  return VX_SUCCESS;
}

vx_result_t vx_kernel_get_max_block_size(vx_kernel_h k, uint32_t* x,
                                         uint32_t* y, uint32_t* z) {
  if (!k) return VX_ERR_INVALID_HANDLE;
  // The real driver reports the device's natural block dims here until the
  // toolchain records per-kernel metadata, so the mock does the same.
  if (x) *x = (uint32_t)env_u64("GRXMOCK_NUM_THREADS", 4);
  if (y) *y = (uint32_t)env_u64("GRXMOCK_NUM_WARPS", 4);
  if (z) *z = 1;
  return VX_SUCCESS;
}

vx_result_t vx_enqueue_launch(vx_queue_h q, const vx_launch_info_t* info,
                              uint32_t, const vx_event_h*,
                              vx_event_h* out_event) {
  if (!q || !info) return VX_ERR_INVALID_HANDLE;
  if (!info->kernel) return VX_ERR_INVALID_VALUE;

  std::lock_guard<std::mutex> lock(g_mutex);
  auto& r = g_last_launch;
  r = grxmock_launch_record{};
  r.valid    = 1;
  r.entry_pc = reinterpret_cast<MockKernel*>(info->kernel)->entry_pc;
  r.ndim     = info->ndim;
  for (int i = 0; i < 3; ++i) {
    r.grid_dim[i]    = info->grid_dim[i];
    r.block_dim[i]   = info->block_dim[i];
    r.cluster_dim[i] = info->cluster_dim[i];
  }
  r.lmem_size = info->lmem_size;
  r.args_size = (uint32_t)info->args_size;
  if (info->args_host && info->args_size) {
    const size_t n = (info->args_size < GRXMOCK_MAX_ARGS) ? info->args_size
                                                          : GRXMOCK_MAX_ARGS;
    std::memcpy(r.args, info->args_host, n);
  }
  ++g_launch_count;
  return complete(q, out_event);
}

vx_result_t vx_device_max_occupancy_grid(vx_device_h dev, uint32_t ndim,
                                         const uint32_t* global_dim,
                                         uint32_t* grid_out,
                                         uint32_t* block_out) {
  if (!dev || !global_dim || !grid_out || !block_out) return VX_ERR_INVALID_VALUE;
  const uint32_t natural[3] = {(uint32_t)env_u64("GRXMOCK_NUM_THREADS", 4),
                               (uint32_t)env_u64("GRXMOCK_NUM_WARPS", 4), 1};
  for (uint32_t i = 0; i < ndim && i < 3; ++i) {
    block_out[i] = natural[i];
    grid_out[i]  = (global_dim[i] + natural[i] - 1) / natural[i];
  }
  return VX_SUCCESS;
}

}  // extern "C"
