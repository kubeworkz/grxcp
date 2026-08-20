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
std::map<uint64_t, FreeExtent>  g_free;         // base -> free extent
std::map<uint64_t, Allocation>  g_live;         // base -> live allocation
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
  auto next = g_free.lower_bound(base);
  if (next != g_free.end() && next->second.slab == slab &&
      base + size == next->first) {
    size += next->second.size;
    g_free.erase(next);
  }
  if (!g_free.empty()) {
    auto prev = g_free.lower_bound(base);
    if (prev != g_free.begin()) {
      --prev;
      if (prev->second.slab == slab &&
          prev->first + prev->second.size == base) {
        base = prev->first;
        size += prev->second.size;
        g_free.erase(prev);
      }
    }
  }
  g_free[base] = FreeExtent{size, slab};
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
bool take_best_fit(uint64_t bytes, uint64_t align, uint64_t* out_base,
                   size_t* out_slab) {
  auto best = g_free.end();
  uint64_t best_size = UINT64_MAX;
  for (auto it = g_free.begin(); it != g_free.end(); ++it) {
    if (it->first % align != 0) continue;   // slabs are aligned; extents stay so
    if (it->second.size < bytes) continue;
    if (it->second.size < best_size) { best = it; best_size = it->second.size; }
  }
  if (best == g_free.end()) return false;

  const uint64_t base = best->first;
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
grxError_t allocate_device_physical(int device, uint64_t bytes,
                                    uint64_t* out_address) {
  Device* d = nullptr;
  grxError_t e = acquire_device(device, &d);
  if (e != grxSuccess) return e;

  const uint64_t need = align_up(bytes + sanitize_redzone_bytes(),
                                 alignment_for(*d));

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
  g_live[base] = a;
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
    g_live[base] = a;
    sanitize_note_region(base, need);
    sanitize_note_alloc(base, bytes);
    *out_address = base;
    return grxSuccess;
  }

  uint64_t base = 0;
  size_t   slab_index = 0;
  if (!take_best_fit(need, align, &base, &slab_index)) {
    e = add_slab(*d, (need > slab) ? align_up(need, align) : slab, &slab_index);
    if (e != grxSuccess) return e;
    if (!take_best_fit(need, align, &base, &slab_index))
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
  g_live[base] = a;
  sanitize_note_alloc(base, bytes);
  *out_address = base;
  return grxSuccess;
}

// Resolve an address to its allocation without taking the lock; callers hold it.
const Allocation* find_locked(uint64_t address) {
  if (g_live.empty()) return nullptr;
  auto it = g_live.upper_bound(address);
  if (it == g_live.begin()) return nullptr;
  --it;
  const Allocation& a = it->second;
  if (address < a.base || address >= a.base + a.size) return nullptr;
  return &a;
}

}  // namespace

bool lookup_device_pointer(const void* ptr, Mapping* out) {
  const uint64_t address = (uint64_t)(uintptr_t)ptr;
  std::lock_guard<std::mutex> lock(g_mem_mutex);
  const Allocation* a = find_locked(address);
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
    if (it->second.device != device) { ++it; continue; }
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
  Mapping map{};
};

Endpoint classify(const void* p) {
  Endpoint e;
  if (lookup_device_pointer(p, &e.map)) { e.is_device = true; return e; }
  Mapping host{};
  if (lookup_host_pointer(p, &host)) { e.map = host; }
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

  grxError_t e = check_kind(kind, d.is_device, s.is_device);
  if (e != grxSuccess) return e;

  if (!d.is_device && !s.is_device) {
    std::memcpy(dst, src, count);
    return grxSuccess;
  }
  if (d.is_device && d.map.size < count) return grxErrorInvalidValue;
  if (s.is_device && s.map.size < count) return grxErrorInvalidValue;

  const int device = d.is_device ? d.map.device : s.map.device;
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

  std::lock_guard<std::mutex> lock(grxcp::g_mem_mutex);
  auto it = grxcp::g_live.find(address);
  // Only the base address of an allocation may be freed; an interior pointer
  // is a bug worth reporting rather than silently rounding down.
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
