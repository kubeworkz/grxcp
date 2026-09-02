// GRXCP — device memory: the two-tier allocator, the interval map, and the
// copy/fill family.
//
// The problem this file solves: the driver hands out refcounted vx_buffer_h
// handles, but the platform's public currency is a plain void* device address,
// because that is what kernel arguments and grxMemcpy need. The interval map is
// the bridge -- every live allocation is registered by address range, so any
// pointer can be resolved back to its owning handle and offset.
//
// Allocation strategy:
//   slab tier   large vx_buffer_create slabs carved by a best-fit free list
//   direct tier allocations at or above a quarter slab get their own buffer,
//               so a large tensor cannot fragment a slab
//
// Alignment is at least 256 bytes and at least the device cache line, and
// request sizes are rounded up to that alignment. A useful consequence: no two
// allocations ever share a cache line, which is exactly the mitigation the
// unaligned-DMA gap needs on FPGA backends, where the command processor rounds
// transfers up to a 64-byte multiple with all byte strobes set
// (cuda_mapping.md section 7.6). It costs memory on small allocations and buys
// the removal of a silent-corruption class.

#include "internal.h"

#ifdef GRXCP_ENABLE_NPU
#include "npu_c930.h"
#endif

#include <grx/grx_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <vector>

namespace grxcp {

namespace {

constexpr uint64_t kMinAlign        = 256;
constexpr uint64_t kDefaultSlabSize = 64ull << 20;   // 64 MiB
constexpr uint64_t kDirectDivisor   = 4;             // >= slab/4 goes direct

struct Slab {
  vx_buffer_h buffer = nullptr;
  uint64_t    base   = 0;
  uint64_t    size   = 0;
  int         device = 0;
  // The NPU's DDR is a fixed window, not a buffer the driver handed us. It is
  // carried as a slab so the free list, the interval map, grxFree and the
  // sanitizer all work on it unchanged -- but it has no vx_buffer_h to release
  // and it must come BACK whole on a device reset rather than disappearing.
  bool        npu    = false;
};

struct FreeExtent {
  uint64_t size      = 0;
  size_t   slab      = 0;
};

struct Allocation {
  vx_buffer_h buffer  = nullptr;
  uint64_t    base    = 0;
  uint64_t    size    = 0;   // as rounded up, which is what may be reused
  uint64_t    offset  = 0;   // within `buffer`
  int         device  = 0;
  bool        direct  = false;   // owns `buffer` outright
  bool        managed = false;
  bool        physical = false;  // VX_MEM_PHYS: the DMA engine can reach it
  size_t      slab    = 0;
};

struct HostAllocation {
  vx_buffer_h buffer = nullptr;
  uint64_t    address = 0;
  uint64_t    size    = 0;
  int         device  = 0;
};

std::mutex                      g_mem_mutex;
std::vector<Slab>               g_slabs;
// KEYED BY (device, address) AND (slab, address), NOT BY ADDRESS.
//
// Each device's addresses come from its OWN vx_buffer_address over its own
// DDR, so two devices' allocations routinely share an address -- both start
// near the same base. Keyed by address alone, the second insert silently
// overwrote the first and lookup returned the wrong device's buffer handle.
//
// Predicted, then watched: filtering the free list by device (take_best_fit
// above) gave device 1 its own slab, and grxMalloc(4096) on each device then
// returned the SAME pointer, 0x100000000000, with only one of the two records
// surviving in the map.
//
// The pairing also makes free-extent coalescing slab-local by construction
// rather than by a check inside insert_free that had to be remembered.
std::map<std::pair<size_t, uint64_t>, FreeExtent>  g_free;   // (slab, base)
std::map<std::pair<int, uint64_t>, Allocation>     g_live;   // (device, base)
std::map<void*, HostAllocation> g_host;         // host ptr -> pinned allocation

uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

uint64_t alignment_for(const Device& d) {
  const uint64_t line = (d.prop.cacheLineSize > 0)
                            ? (uint64_t)d.prop.cacheLineSize : kMinAlign;
  return (line > kMinAlign) ? align_up(line, kMinAlign) : kMinAlign;
}

uint64_t slab_size() {
  if (const char* v = std::getenv("GRX_SLAB_BYTES")) {
    uint64_t parsed = std::strtoull(v, nullptr, 0);
    if (parsed >= (1ull << 20)) return parsed;
  }
  return kDefaultSlabSize;
}

// Add a free extent, coalescing with the neighbours that belong to the same
// slab. Coalescing across slabs would produce an extent spanning two device
// buffers, which is why the slab index is part of the record.
void insert_free(uint64_t base, uint64_t size, size_t slab) {
  auto next = g_free.lower_bound({slab, base});
  if (next != g_free.end() && next->first.first == slab &&
      base + size == next->first.second) {
    size += next->second.size;
    g_free.erase(next);
  }
  if (!g_free.empty()) {
    auto prev = g_free.lower_bound({slab, base});
    if (prev != g_free.begin()) {
      --prev;
      if (prev->first.first == slab &&
          prev->first.second + prev->second.size == base) {
        base = prev->first.second;
        size += prev->second.size;
        g_free.erase(prev);
      }
    }
  }
  g_free[{slab, base}] = FreeExtent{size, slab};
}

grxError_t add_slab(Device& d, uint64_t bytes, size_t* out_index) {
  vx_buffer_h buf = nullptr;
  vx_result_t r = vx_buffer_create(d.handle, bytes, VX_MEM_READ_WRITE, &buf);
  if (r != VX_SUCCESS) return map_result(r);

  uint64_t base = 0;
  r = vx_buffer_address(buf, &base);
  if (r != VX_SUCCESS) { vx_buffer_release(buf); return map_result(r); }

  Slab s;
  s.buffer = buf;
  s.base   = base;
  s.size   = bytes;
  s.device = d.index;
  g_slabs.push_back(s);
  sanitize_note_region(base, bytes);

  const size_t index = g_slabs.size() - 1;
  insert_free(base, bytes, index);
  if (out_index) *out_index = index;
  return grxSuccess;
}

// Best fit: the smallest free extent that can hold the request. Linear over the
// free list, which stays short because extents coalesce on release.
// The free list spans every device, so the SEARCH has to say which one it is
// allocating for. Without that filter grxMalloc on device 1 carves a slice out
// of device 0's slab and records it as device 1's: the returned pointer is
// device 0 memory wearing device 1's label, every later operation on it targets
// the wrong device, and nothing anywhere reports a problem.
//
// It was not findable until the mock stopped handing every device the same
// handle. Watched: with two devices, grxMalloc(4096) on device 1 returned
// device 0's address one page above device 0's own allocation, and device 1's
// grxMemGetInfo still read zero bytes in use.
bool take_best_fit(int device, uint64_t bytes, uint64_t align,
                   uint64_t* out_base, size_t* out_slab) {
  auto best = g_free.end();
  uint64_t best_size = UINT64_MAX;
  for (auto it = g_free.begin(); it != g_free.end(); ++it) {
    if (g_slabs[it->second.slab].device != device) continue;
    if (it->first.second % align != 0) continue;  // slabs are aligned; extents stay so
    if (it->second.size < bytes) continue;
    if (it->second.size < best_size) { best = it; best_size = it->second.size; }
  }
  if (best == g_free.end()) return false;

  const uint64_t base = best->first.second;
  const uint64_t size = best->second.size;
  const size_t   slab = best->second.slab;
  g_free.erase(best);
  if (size > bytes) insert_free(base + bytes, size - bytes, slab);

  *out_base = base;
  *out_slab = slab;
  return true;
}

// Physically-addressed allocation, for buffers the DXA engine will read.
//
// It always gets its own buffer rather than a slice of a slab. Physical memory
// may be a scarcer resource than ordinary device memory, and handing out a
// piece of a shared slab would keep the whole slab physical for as long as any
// piece of it lived.
#ifdef GRXCP_ENABLE_NPU
// THE FIRST WORD OF DDR IS NEVER HANDED OUT.
//
// The NPU's DDR window starts at byte 0 and a device pointer here IS the DDR
// offset -- there is no MMU and A_BASE/B_BASE/C_BASE take the address
// literally. But grxMalloc's failure signal is a null pointer, so an
// allocation at offset 0 and an allocation that failed are the same value to
// every caller.
//
// The GRX930 team hit the same wall from the other side: their reference
// allocator returned 0 for failure while its documented window base was
// 0x0000, so the first successful allocation and an out-of-memory were
// indistinguishable. They fixed it with an out-of-band sentinel. We cannot --
// the return type is void* and null already means failure -- so the fix is
// structural: reserve the first aligned block and never allocate from it. It
// costs one alignment unit of a 64 KB window and removes the ambiguity
// entirely rather than moving it somewhere a caller has to remember.
constexpr uint64_t kNpuReservedBase = kMinAlign;

// Add (or restore) the device's DDR window as a single free extent.
// Idempotent: a window already carrying free space is left alone.
grxError_t ensure_npu_window(const Device& d) {
  for (size_t i = 0; i < g_slabs.size(); ++i)
    if (g_slabs[i].npu && g_slabs[i].device == d.index) return grxSuccess;

  if (d.prop.totalGlobalMem <= kNpuReservedBase) return grxErrorMemoryAllocation;

  Slab s;
  s.buffer = nullptr;                       // no driver buffer to release
  s.base   = kNpuReservedBase;
  s.size   = (uint64_t)d.prop.totalGlobalMem - kNpuReservedBase;
  s.device = d.index;
  s.npu    = true;
  g_slabs.push_back(s);
  sanitize_note_region(s.base, s.size);
  insert_free(s.base, s.size, g_slabs.size() - 1);
  return grxSuccess;
}

// One fixed window, carved by the same best-fit free list every other device
// uses. No direct tier: the direct tier exists so a large allocation can get
// its own driver buffer, and there is no driver here -- a request the window
// cannot satisfy is out of memory, not a reason to ask for a second window.
grxError_t allocate_npu(Device& d, uint64_t bytes, uint64_t align,
                        uint64_t need, bool managed, uint64_t* out_address) {
  grxError_t e = ensure_npu_window(d);
  if (e != grxSuccess) return e;

  uint64_t base = 0;
  size_t   slab_index = 0;
  if (!take_best_fit(d.index, need, align, &base, &slab_index))
    return grxErrorMemoryAllocation;

  Allocation a;
  a.buffer  = nullptr;
  a.base    = base;
  a.size    = need;
  a.offset  = base - g_slabs[slab_index].base;
  a.device  = d.index;
  a.direct  = false;      // freed back to the window, not released to a driver
  a.managed = managed;
  a.slab    = slab_index;
  g_live[{d.index, base}] = a;
  sanitize_note_alloc(base, bytes);
  *out_address = base;
  return grxSuccess;
}
#endif  // GRXCP_ENABLE_NPU

grxError_t allocate_device_physical(int device, uint64_t bytes,
                                    uint64_t* out_address) {
  Device* d = nullptr;
  grxError_t e = acquire_device(device, &d);
  if (e != grxSuccess) return e;

  const uint64_t align = alignment_for(*d);
  const uint64_t need = align_up(bytes + sanitize_redzone_bytes(), align);

#ifdef GRXCP_ENABLE_NPU
  // On the NPU there is one address space and no MMU, so every allocation is
  // already where the DMA engine can reach it -- VX_MEM_PHYS has no meaning
  // and no separate pool to come from. Same window, and `physical` stays false
  // because it records "asked the driver for a physical buffer", which nobody
  // did.
  if (d->type == DeviceType::NPU) {
    std::lock_guard<std::mutex> lock(g_mem_mutex);
    return allocate_npu(*d, bytes, align, need, /*managed=*/false, out_address);
  }
#endif

  vx_buffer_h buf = nullptr;
  vx_result_t r = vx_buffer_create(d->handle, need,
                                   VX_MEM_READ_WRITE | VX_MEM_PHYS, &buf);
  if (r != VX_SUCCESS) return map_result(r);
  uint64_t base = 0;
  r = vx_buffer_address(buf, &base);
  if (r != VX_SUCCESS) { vx_buffer_release(buf); return map_result(r); }

  std::lock_guard<std::mutex> lock(g_mem_mutex);
  Allocation a;
  a.buffer = buf; a.base = base; a.size = need; a.offset = 0;
  a.device = device; a.direct = true; a.physical = true;
  g_live[{device, base}] = a;
  sanitize_note_region(base, need);
  sanitize_note_alloc(base, bytes);
  *out_address = base;
  return grxSuccess;
}

grxError_t allocate_device(int device, uint64_t bytes, bool managed,
                           uint64_t* out_address) {
  Device* d = nullptr;
  grxError_t e = acquire_device(device, &d);
  if (e != grxSuccess) return e;

  const uint64_t align = alignment_for(*d);
  // Under GRX_SANITIZE every allocation gets a trailing redzone. It is not
  // handed to the caller and belongs to no extent, so an overflow off the end
  // of one allocation lands in a hole rather than in the next allocation --
  // which is the difference between a finding and a corrupted neighbour.
  // Because every allocation has one, an underflow also lands in a redzone.
  const uint64_t need  = align_up(bytes + sanitize_redzone_bytes(), align);
  const uint64_t slab  = slab_size();

  std::lock_guard<std::mutex> lock(g_mem_mutex);

#ifdef GRXCP_ENABLE_NPU
  // AN NPU DEVICE HAS NO vx_device_h, AND EVERY LINE BELOW ASSUMES ONE.
  //
  // This branch did not exist. allocate_device went straight on to
  // vx_buffer_create(d->handle, ...) with handle == nullptr, and the driver
  // answered VX_ERR_INVALID_VALUE -- reported to the caller as "invalid
  // value", blaming the size argument for a device that had no allocator at
  // all. Measured, before the fix, through the seam in npu_c930_testing.h:
  //
  //   grxMalloc(256) -> grxErrorInvalidValue (invalid argument)
  //   totalGlobalMem is 65536 bytes and nothing allocates from it.
  //
  // The same shape as the module-load refusal that used to say "invalid
  // image" on a device with no pipeline, and the reason nothing above the
  // backend had ever run on an NPU: tests/libs/test_grxblas_npu.cpp fails at
  // its first allocation, before it reaches a single register.
  if (d->type == DeviceType::NPU)
    return allocate_npu(*d, bytes, align, need, managed, out_address);
#endif

  // Direct tier: big allocations get their own buffer so they cannot fragment
  // a slab and so freeing them returns memory to the device immediately.
  if (need >= slab / kDirectDivisor) {
    vx_buffer_h buf = nullptr;
    vx_result_t r = vx_buffer_create(d->handle, need, VX_MEM_READ_WRITE, &buf);
    if (r != VX_SUCCESS) return map_result(r);
    uint64_t base = 0;
    r = vx_buffer_address(buf, &base);
    if (r != VX_SUCCESS) { vx_buffer_release(buf); return map_result(r); }

    Allocation a;
    a.buffer = buf; a.base = base; a.size = need; a.offset = 0;
    a.device = device; a.direct = true; a.managed = managed;
    g_live[{device, base}] = a;
    sanitize_note_region(base, need);
    sanitize_note_alloc(base, bytes);
    *out_address = base;
    return grxSuccess;
  }

  uint64_t base = 0;
  size_t   slab_index = 0;
  if (!take_best_fit(device, need, align, &base, &slab_index)) {
    e = add_slab(*d, (need > slab) ? align_up(need, align) : slab, &slab_index);
    if (e != grxSuccess) return e;
    if (!take_best_fit(device, need, align, &base, &slab_index))
      return grxErrorMemoryAllocation;
  }

  Allocation a;
  a.buffer  = g_slabs[slab_index].buffer;
  a.base    = base;
  a.size    = need;
  a.offset  = base - g_slabs[slab_index].base;
  a.device  = device;
  a.direct  = false;
  a.managed = managed;
  a.slab    = slab_index;
  g_live[{device, base}] = a;
  sanitize_note_alloc(base, bytes);
  *out_address = base;
  return grxSuccess;
}

// Resolve an address to its allocation without taking the lock; callers hold it.
const Allocation* find_locked(int device, uint64_t address) {
  if (g_live.empty()) return nullptr;
  auto it = g_live.upper_bound({device, address});
  if (it == g_live.begin()) return nullptr;
  --it;
  if (it->first.first != device) return nullptr;   // fell into another device
  const Allocation& a = it->second;
  if (address < a.base || address >= a.base + a.size) return nullptr;
  return &a;
}

}  // namespace

bool lookup_device_pointer_on(int device, const void* ptr, Mapping* out) {
  const uint64_t address = (uint64_t)(uintptr_t)ptr;
  std::lock_guard<std::mutex> lock(g_mem_mutex);
  const Allocation* a = find_locked(device, address);
  if (!a) return false;
  if (out) {
    const uint64_t delta = address - a->base;
    out->buffer  = a->buffer;
    out->base    = a->base;
    out->offset  = a->offset + delta;
    out->size    = a->size - delta;
    out->device  = a->device;
    out->managed = a->managed;
    out->physical = a->physical;
  }
  return true;
}

// Resolve against the CURRENT device. An address alone no longer identifies an
// allocation, so this is the only answer a bare pointer can have.
bool lookup_device_pointer(const void* ptr, Mapping* out) {
  return lookup_device_pointer_on(current_device_index(), ptr, out);
}

// Which device owns this address, if any -- for DIAGNOSIS only. With
// overlapping spaces an address can be live on several devices at once, so
// this returns the first that claims it and is never used to pick a target.
int owner_device_of(const void* ptr) {
  const uint64_t address = (uint64_t)(uintptr_t)ptr;
  std::lock_guard<std::mutex> lock(g_mem_mutex);
  for (auto it = g_live.begin(); it != g_live.end(); ++it) {
    const Allocation& a = it->second;
    if (address >= a.base && address < a.base + a.size) return it->first.first;
  }
  return -1;
}

bool lookup_host_pointer(const void* ptr, Mapping* out) {
  std::lock_guard<std::mutex> lock(g_mem_mutex);
  if (g_host.empty()) return false;
  auto it = g_host.upper_bound(const_cast<void*>(ptr));
  if (it == g_host.begin()) return false;
  --it;
  auto* base = (uint8_t*)it->first;
  auto* p    = (const uint8_t*)ptr;
  if (p < base || p >= base + it->second.size) return false;
  if (out) {
    out->buffer  = it->second.buffer;
    out->base    = it->second.address;
    out->offset  = (uint64_t)(p - base);
    out->size    = it->second.size - (uint64_t)(p - base);
    out->device  = it->second.device;
    out->managed = false;
  }
  return true;
}

void release_all_allocations(int device) {
  std::lock_guard<std::mutex> lock(g_mem_mutex);
  for (auto it = g_live.begin(); it != g_live.end();) {
    if (it->first.first != device) { ++it; continue; }
    if (it->second.direct) vx_buffer_release(it->second.buffer);
    it = g_live.erase(it);
  }
  for (auto it = g_host.begin(); it != g_host.end();) {
    if (it->second.device != device) { ++it; continue; }
    vx_buffer_unmap(it->second.buffer, it->first);
    vx_buffer_release(it->second.buffer);
    it = g_host.erase(it);
  }
  for (auto it = g_free.begin(); it != g_free.end();) {
    it = (g_slabs[it->second.slab].device == device) ? g_free.erase(it)
                                                     : std::next(it);
  }
  // A DEVICE RESET FREES THE NPU'S DDR; IT DOES NOT TAKE IT AWAY.
  //
  // The loop above erases every free extent, and the loop below releases the
  // driver buffer behind each slab. Neither is right for the NPU window: there
  // is no buffer to release, and the window is a fixed property of the SoC
  // that is still there after the reset. Put it back whole, which is exactly
  // what "every allocation on this device is gone" means for a fixed window.
  for (size_t i = 0; i < g_slabs.size(); ++i)
    if (g_slabs[i].npu && g_slabs[i].device == device)
      insert_free(g_slabs[i].base, g_slabs[i].size, i);
  for (auto& s : g_slabs) {
    if (s.device == device && s.buffer) {
      vx_buffer_release(s.buffer);
      s.buffer = nullptr;
    }
  }
  g_slabs.erase(std::remove_if(g_slabs.begin(), g_slabs.end(),
                               [](const Slab& s) { return s.buffer == nullptr; }),
                g_slabs.end());
  sanitize_forget_all();
}

namespace {

// Direction resolution. The interval map knows which side is device memory, so
// grxMemcpyDefault is always answerable -- and an explicit kind that
// contradicts the map is rejected instead of quietly doing the wrong transfer.
struct Endpoint {
  bool    is_device = false;
  int     foreign   = -1;   // owned by this OTHER device, and by no local one
  Mapping map{};
};

// CROSS-DEVICE POINTERS, and the honest limit of what can be detected.
//
// A pointer that resolves on the current device is used as the current
// device's, full stop. With per-device address spaces the same address is
// frequently live on both devices at once, and in that case nothing in a bare
// void* says which was meant -- "this pointer, on this device" is the only
// defensible reading, and it is also what CUDA does.
//
// What IS unambiguous is a pointer that resolves on NO local allocation but
// does resolve on another device. There is no reading of that which is
// correct, so it is refused, and the refusal names the owner: "device 1's
// pointer" is fixable where "not a device pointer" sends someone hunting a
// dangling free.
//
// There is no peer-access API here. When one arrives this is where it opens.
Endpoint classify(const void* p) {
  Endpoint e;
  if (lookup_device_pointer(p, &e.map)) { e.is_device = true; return e; }
  Mapping host{};
  if (lookup_host_pointer(p, &host)) { e.map = host; return e; }
  e.foreign = owner_device_of(p);
  return e;
}

grxError_t check_kind(grxMemcpyKind kind, bool dst_dev, bool src_dev) {
  switch (kind) {
    case grxMemcpyDefault:        return grxSuccess;
    case grxMemcpyHostToHost:     return (!dst_dev && !src_dev) ? grxSuccess : grxErrorInvalidMemcpyDirection;
    case grxMemcpyHostToDevice:   return ( dst_dev && !src_dev) ? grxSuccess : grxErrorInvalidMemcpyDirection;
    case grxMemcpyDeviceToHost:   return (!dst_dev &&  src_dev) ? grxSuccess : grxErrorInvalidMemcpyDirection;
    case grxMemcpyDeviceToDevice: return ( dst_dev &&  src_dev) ? grxSuccess : grxErrorInvalidMemcpyDirection;
  }
  return grxErrorInvalidValue;
}

grxError_t enqueue_copy(void* dst, const void* src, size_t count,
                        grxMemcpyKind kind, grxStream_t stream, bool blocking) {
  if (count == 0) return grxSuccess;
  if (!dst || !src) return grxErrorInvalidValue;

  Endpoint d = classify(dst);
  Endpoint s = classify(src);

  if (d.foreign >= 0 || s.foreign >= 0) return grxErrorInvalidDevicePointer;

  grxError_t e = check_kind(kind, d.is_device, s.is_device);
  if (e != grxSuccess) return e;

  if (!d.is_device && !s.is_device) {
    std::memcpy(dst, src, count);
    return grxSuccess;
  }
  if (d.is_device && d.map.size < count) return grxErrorInvalidValue;
  if (s.is_device && s.map.size < count) return grxErrorInvalidValue;

  const int device = d.is_device ? d.map.device : s.map.device;

#ifdef GRXCP_ENABLE_NPU
  // THE NPU HAS NO QUEUE, NO EVENTS AND NO vx_buffer_h.
  //
  // Everything below this point is the Vortex path: resolve_stream wants a
  // vx_queue_h, and the three enqueue calls want buffer handles that an NPU
  // allocation does not have (its Mapping::buffer is null and its address IS
  // the DDR offset). So the copy happens here, synchronously, through the
  // device's DDR hooks.
  //
  // A DEVICE WITH NO DDR HOOKS IS REFUSED, and that is every device today
  // except one with a model attached: the hardware path would be an mmap of
  // the DDR aperture or a bounce through the AXI DMA, and neither is written.
  // Refusing is the only honest answer -- an accepted memcpy that moves no
  // bytes leaves the caller reading whatever was in the buffer, which is worse
  // than an error and much harder to find.
  //
  // Synchronous regardless of `blocking`: there is nothing to enqueue onto, so
  // the copy is complete when this returns. That is a stronger guarantee than
  // the caller asked for and needs no note in the stream's event chain.
  {
    Device* nd = nullptr;
    if (acquire_device(device, &nd) == grxSuccess &&
        nd->type == DeviceType::NPU) {
      npu_c930_device* h = npu_device_for(device);
      if (!h || !npu_c930_mem_ready(h)) return grxErrorNotSupported;

      ProfileSample np;
      const bool nprof = profile_begin(device, &np);
      const uint32_t n = (uint32_t)count;
      int rc = 0;
      const char* dir;
      if (d.is_device && s.is_device) {
        dir = "d2d";
        // Through a bounce buffer rather than hook-to-hook: the two extents
        // may overlap, and a byte-by-byte device-to-device copy through a
        // model with no aliasing guarantees is a bug waiting for the first
        // overlapping call.
        std::vector<uint8_t> tmp(count);
        rc = npu_c930_mem_read(h, (uint32_t)(uintptr_t)src, tmp.data(), n);
        if (rc == 0)
          rc = npu_c930_mem_write(h, (uint32_t)(uintptr_t)dst, tmp.data(), n);
      } else if (d.is_device) {
        dir = "h2d";
        rc = npu_c930_mem_write(h, (uint32_t)(uintptr_t)dst, src, n);
      } else {
        dir = "d2h";
        rc = npu_c930_mem_read(h, (uint32_t)(uintptr_t)src, dst, n);
      }
      if (rc != 0) { profile_abandon(&np); return grxErrorInvalidValue; }
      if (nprof) profile_end_transfer(&np, "memcpy", count, dir, stream);
      return grxSuccess;
    }
  }
#endif

  vx_queue_h q = nullptr;
  e = resolve_stream(stream, device, &q, nullptr);
  if (e != grxSuccess) return e;

  std::vector<vx_event_h> waits;
  collect_wait_events(stream, device, &waits);
  const uint32_t nwait = (uint32_t)waits.size();
  const vx_event_h* wp = waits.empty() ? nullptr : waits.data();

  ProfileSample sample;
  const bool profiling = profile_begin(device, &sample);

  vx_event_h completion = nullptr;
  vx_result_t r;
  const char* direction;
  if (d.is_device && s.is_device) {
    direction = "d2d";
    r = vx_enqueue_copy(q, d.map.buffer, d.map.offset,
                        s.map.buffer, s.map.offset, count, nwait, wp,
                        &completion);
  } else if (d.is_device) {
    direction = "h2d";
    r = vx_enqueue_write(q, d.map.buffer, d.map.offset, src, count, nwait, wp,
                         &completion);
  } else {
    direction = "d2h";
    r = vx_enqueue_read(q, dst, s.map.buffer, s.map.offset, count, nwait, wp,
                        &completion);
  }
  if (r != VX_SUCCESS) { profile_abandon(&sample); return map_result(r); }

  set_stream_last_event(stream, device, completion);
  e = blocking ? sync_stream(stream, device) : grxSuccess;
  if (profiling)
    profile_end_transfer(&sample, "memcpy", count, direction, stream);
  return e;
}

}  // namespace
}  // namespace grxcp

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

extern "C" {

grxError_t grxMalloc(void** ptr, size_t size) {
  if (!ptr) return grxcp::set_error(grxErrorInvalidValue);
  if (size == 0) { *ptr = nullptr; return grxSuccess; }
  uint64_t address = 0;
  grxError_t e = grxcp::allocate_device(grxcp::current_device_index(), size,
                                        /*managed=*/false, &address);
  if (e != grxSuccess) return grxcp::set_error(e);
  *ptr = (void*)(uintptr_t)address;
  return grxSuccess;
}

grxError_t grxMallocPhysical(void** ptr, size_t size) {
  if (!ptr) return grxcp::set_error(grxErrorInvalidValue);
  if (size == 0) { *ptr = nullptr; return grxSuccess; }
  uint64_t address = 0;
  grxError_t e = grxcp::allocate_device_physical(grxcp::current_device_index(),
                                                 size, &address);
  if (e != grxSuccess) return grxcp::set_error(e);
  *ptr = (void*)(uintptr_t)address;
  return grxSuccess;
}

grxError_t grxMallocManaged(void** ptr, size_t size, unsigned int flags) {
  (void)flags;
  if (!ptr) return grxcp::set_error(grxErrorInvalidValue);

  grxcp::Device* d = nullptr;
  grxError_t e = grxcp::acquire_device(grxcp::current_device_index(), &d);
  if (e != grxSuccess) return grxcp::set_error(e);

  // Refused rather than emulated where the backend has no working MMU: a
  // managed pointer that is secretly not managed changes the program's meaning
  // (cuda_mapping.md section 7.5).
  if (!d->prop.managedMemory) return grxcp::set_error(grxErrorNotSupported);

  uint64_t address = 0;
  e = grxcp::allocate_device(d->index, size, /*managed=*/true, &address);
  if (e != grxSuccess) return grxcp::set_error(e);
  *ptr = (void*)(uintptr_t)address;
  return grxSuccess;
}

grxError_t grxFree(void* ptr) {
  if (!ptr) return grxSuccess;   // CUDA: freeing null is legal
  const uint64_t address = (uint64_t)(uintptr_t)ptr;

  const int device = grxcp::current_device_index();
  std::lock_guard<std::mutex> lock(grxcp::g_mem_mutex);
  auto it = grxcp::g_live.find({device, address});
  // Only the base address of an allocation may be freed; an interior pointer
  // is a bug worth reporting rather than silently rounding down. And only from
  // the device that owns it -- the same address is a live allocation on
  // another device as often as not, so freeing "by pointer" from the wrong
  // current device would release the wrong memory.
  if (it == grxcp::g_live.end())
    return grxcp::set_error(grxErrorInvalidDevicePointer);

  const grxcp::Allocation a = it->second;
  grxcp::g_live.erase(it);

  // Under GRX_SANITIZE the memory is quarantined rather than recycled: the
  // extent stays in the map marked freed, and neither the free list nor the
  // driver gets it back. That is what makes a use-after-free stay a
  // use-after-free instead of becoming a read of whatever moved in. It also
  // means a sanitized run's peak device memory is the sum of everything it
  // ever allocated -- documented, and the reason --sanitize is not a default.
  if (grxcp::sanitize_enabled()) {
    grxcp::sanitize_note_free(a.base);
    return grxSuccess;
  }

  if (a.direct) vx_buffer_release(a.buffer);
  else          grxcp::insert_free(a.base, a.size, a.slab);
  return grxSuccess;
}

grxError_t grxMallocHost(void** ptr, size_t size) {
  if (!ptr || size == 0) return grxcp::set_error(grxErrorInvalidValue);
  grxcp::Device* d = nullptr;
  grxError_t e = grxcp::acquire_device(grxcp::current_device_index(), &d);
  if (e != grxSuccess) return grxcp::set_error(e);

#ifdef GRXCP_ENABLE_NPU
  // Pinned host memory is a driver mapping (vx_buffer_create | VX_MEM_HOST,
  // then vx_buffer_map). The NPU has no driver and no way to map its DDR into
  // the host's address space, so there is nothing here to pin. Refused by name
  // rather than sent to vx_buffer_create with a null handle, which would come
  // back "invalid value" and blame the size.
  if (d->type == grxcp::DeviceType::NPU) return grxcp::set_error(grxErrorNotSupported);
#endif

  vx_buffer_h buf = nullptr;
  vx_result_t r = vx_buffer_create(d->handle, size,
                                   VX_MEM_READ_WRITE | VX_MEM_HOST, &buf);
  if (r != VX_SUCCESS) return grxcp::set_error(grxcp::map_result(r));

  void* host = nullptr;
  r = vx_buffer_map(buf, 0, size, VX_MEM_READ_WRITE, &host);
  if (r != VX_SUCCESS) {
    vx_buffer_release(buf);
    return grxcp::set_error(grxcp::map_result(r));
  }
  uint64_t address = 0;
  vx_buffer_address(buf, &address);

  {
    std::lock_guard<std::mutex> lock(grxcp::g_mem_mutex);
    grxcp::g_host[host] = grxcp::HostAllocation{buf, address, size, d->index};
  }
  *ptr = host;
  return grxSuccess;
}

grxError_t grxFreeHost(void* ptr) {
  if (!ptr) return grxSuccess;
  std::lock_guard<std::mutex> lock(grxcp::g_mem_mutex);
  auto it = grxcp::g_host.find(ptr);
  if (it == grxcp::g_host.end())
    return grxcp::set_error(grxErrorInvalidValue);
  vx_buffer_unmap(it->second.buffer, ptr);
  vx_buffer_release(it->second.buffer);
  grxcp::g_host.erase(it);
  return grxSuccess;
}

grxError_t grxHostRegister(void* ptr, size_t size, unsigned int flags) {
  (void)ptr; (void)size; (void)flags;
  // Pinning an existing host allocation needs a driver path that can map
  // arbitrary host pages into the command processor's host aperture. There
  // isn't one; grxMallocHost is the supported way to get pinned memory.
  return grxcp::set_error(grxErrorNotSupported);
}

grxError_t grxHostUnregister(void* ptr) {
  (void)ptr;
  return grxcp::set_error(grxErrorNotSupported);
}

grxError_t grxMemcpy(void* dst, const void* src, size_t count,
                     grxMemcpyKind kind) {
  grxError_t e = grxcp::enqueue_copy(dst, src, count, kind, nullptr,
                                     /*blocking=*/true);
  return (e == grxSuccess) ? e : grxcp::set_error(e);
}

grxError_t grxMemcpyAsync(void* dst, const void* src, size_t count,
                          grxMemcpyKind kind, grxStream_t stream) {
  grxError_t e = grxcp::enqueue_copy(dst, src, count, kind, stream,
                                     /*blocking=*/false);
  return (e == grxSuccess) ? e : grxcp::set_error(e);
}

grxError_t grxMemcpy2DAsync(void* dst, size_t dpitch, const void* src,
                            size_t spitch, size_t width, size_t height,
                            grxMemcpyKind kind, grxStream_t stream) {
  if (width == 0 || height == 0) return grxSuccess;
  if (dpitch < width || spitch < width)
    return grxcp::set_error(grxErrorInvalidValue);

  // Decomposed row by row rather than issued as a strided descriptor. The
  // driver's rect entry points would collapse a contiguous rect into one
  // transfer; wiring them up is a follow-up, and doing it wrong would be worse
  // than doing it simply.
  for (size_t row = 0; row < height; ++row) {
    grxError_t e = grxcp::enqueue_copy((uint8_t*)dst + row * dpitch,
                                       (const uint8_t*)src + row * spitch,
                                       width, kind, stream, /*blocking=*/false);
    if (e != grxSuccess) return grxcp::set_error(e);
  }
  return grxSuccess;
}

grxError_t grxMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                       size_t width, size_t height, grxMemcpyKind kind) {
  grxError_t e = grxMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                                  kind, nullptr);
  if (e != grxSuccess) return e;
  return grxcp::sync_stream(nullptr, grxcp::current_device_index());
}

grxError_t grxMemsetAsync(void* dst, int value, size_t count,
                          grxStream_t stream) {
  if (count == 0) return grxSuccess;
  grxcp::Mapping m{};
  if (!grxcp::lookup_device_pointer(dst, &m))
    return grxcp::set_error(grxErrorInvalidDevicePointer);
  if (m.size < count) return grxcp::set_error(grxErrorInvalidValue);

  vx_queue_h q = nullptr;
  grxError_t e = grxcp::resolve_stream(stream, m.device, &q, nullptr);
  if (e != grxSuccess) return grxcp::set_error(e);

  std::vector<vx_event_h> waits;
  grxcp::collect_wait_events(stream, m.device, &waits);

  grxcp::ProfileSample sample;
  const bool profiling = grxcp::profile_begin(m.device, &sample);

  const uint8_t pattern = (uint8_t)value;
  vx_event_h completion = nullptr;
  vx_result_t r = vx_enqueue_fill_buffer(q, m.buffer, m.offset, count,
                                         &pattern, sizeof(pattern),
                                         (uint32_t)waits.size(),
                                         waits.empty() ? nullptr : waits.data(),
                                         &completion);
  if (r != VX_SUCCESS) {
    grxcp::profile_abandon(&sample);
    return grxcp::set_error(grxcp::map_result(r));
  }
  grxcp::set_stream_last_event(stream, m.device, completion);
  if (profiling)
    grxcp::profile_end_transfer(&sample, "memset", count, "fill", stream);
  return grxSuccess;
}

grxError_t grxMemset(void* dst, int value, size_t count) {
  grxError_t e = grxMemsetAsync(dst, value, count, nullptr);
  if (e != grxSuccess) return e;
  return grxcp::sync_stream(nullptr, grxcp::current_device_index());
}

grxError_t grxPointerGetAttributes(grxPointerAttributes* attr,
                                   const void* ptr) {
  if (!attr) return grxcp::set_error(grxErrorInvalidValue);
  *attr = grxPointerAttributes{};

  grxcp::Mapping m{};
  if (grxcp::lookup_device_pointer(ptr, &m)) {
    attr->type           = m.managed ? grxMemoryTypeManaged : grxMemoryTypeDevice;
    attr->device         = m.device;
    attr->devicePointer  = const_cast<void*>(ptr);
    attr->hostPointer    = nullptr;
    attr->allocationSize = m.size;
    return grxSuccess;
  }
  if (grxcp::lookup_host_pointer(ptr, &m)) {
    attr->type           = grxMemoryTypeHost;
    attr->device         = m.device;
    attr->devicePointer  = (void*)(uintptr_t)(m.base + m.offset);
    attr->hostPointer    = const_cast<void*>(ptr);
    attr->allocationSize = m.size;
    return grxSuccess;
  }
  attr->type        = grxMemoryTypeUnregistered;
  attr->hostPointer = const_cast<void*>(ptr);
  return grxSuccess;
}

grxError_t grxDeviceReset(void) {
  const int device = grxcp::current_device_index();
  grxError_t e = grxcp::sync_all_streams(device);
  grxcp::release_all_allocations(device);
  return (e == grxSuccess) ? e : grxcp::set_error(e);
}

}  // extern "C"
