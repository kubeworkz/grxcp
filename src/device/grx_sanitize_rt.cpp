// GRXCP — the device half of grx-sanitize.
//
// This translation unit is linked into a kernel image built with
// `ci/build_kernel.sh --sanitize`, which compiles the *kernel* with
//
//   -fsanitize=address -mllvm -asan-instrumentation-with-call-threshold=0
//
// The threshold of 0 is the whole trick. Ordinarily AddressSanitizer inlines a
// shadow-memory probe at every load and store, which would need a shadow map
// covering the device address space -- 1/8th of everything the kernel touches,
// written into simulated DRAM before the kernel starts. At threshold 0 clang
// emits an outlined call instead: `__asan_load4(addr)`, `__asan_store8(addr)`,
// and so on. Those calls land here, and what the check *is* becomes ours to
// define. This file defines it as a lookup in the allocation table the host
// runtime uploads before each launch, which needs no shadow memory at all.
//
// What that buys, precisely: a check that knows the size the caller actually
// asked grxMalloc for (not the rounded-up size), knows which allocations have
// been freed, and knows how many bytes of shared memory this launch requested.
// What it costs: a call per memory access. On a functional simulator that is
// the right trade; on silicon it would not be.
//
// This file is compiled WITHOUT -fsanitize=address. If it were instrumented,
// the first check would call itself.
//
// Nothing here allocates, blocks, or prints. A finding is a bounded write into
// a report array plus one atomic increment, so a kernel that goes wrong in
// every thread costs a few atomics and keeps running -- the host reports what
// it caught and says how much it dropped.

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <stdint.h>

#include <grx/grx_sanitize_abi.h>

// Where the control block is.
//
// The block itself cannot live in the kernel image, because the host has to
// rewrite it before every launch and the image is not writable through the
// driver: vx_buffer_reserve fails on a range the module loader already owns.
// So the image holds one pointer, and the block lives in a device buffer the
// GRXCP runtime allocates and can write freely.
//
// The pointer is patched into the .vxbin payload at module load, at the file
// offset the anchor's ELF VMA maps to. That forces two things:
//
//   .data, not .bss. A zero-initialized global lands in .bss, which sits past
//   _edata and is NOT part of the .vxbin payload -- there would be nothing at
//   that offset to patch. The explicit section attribute keeps it in the
//   image, where the loader writes it out and the patch lands.
//
//   Zero means disabled. A kernel built with --sanitize but run without the
//   sanitizer armed reads a null anchor and every check returns immediately.
extern "C" __attribute__((section(".data.grxsan"), used, retain))
uint64_t __grx_san_anchor = 0;

namespace {

// The local-memory aperture, from the build's own configuration. Shared memory
// is a fixed-stride slot inside it; this CTA's slot base is in a CSR.
constexpr uint64_t kLmemBase = (uint64_t)VX_MEM_LMEM_BASE_ADDR;
constexpr uint64_t kLmemSize = (uint64_t)1u << VX_CFG_LMEM_LOG_SIZE;

__attribute__((always_inline)) inline uint32_t linear_block() {
  const uint32_t gx = gridDim.x, gy = gridDim.y;
  return (uint32_t)blockIdx.x + gx * ((uint32_t)blockIdx.y + gy * (uint32_t)blockIdx.z);
}

__attribute__((always_inline)) inline uint32_t linear_thread() {
  const uint32_t bx = blockDim.x, by = blockDim.y;
  return (uint32_t)threadIdx.x + bx * ((uint32_t)threadIdx.y + by * (uint32_t)threadIdx.z);
}

__attribute__((always_inline)) inline grxSanState* state() {
  return reinterpret_cast<grxSanState*>(__grx_san_anchor);
}

// Record a finding in this thread's own slot.
//
// No atomics: this device configuration has VX_CFG_EXT_A_ENABLED off and the
// simulator's LSU aborts on any AMO instruction, even though -march=rv64imafd
// tells the compiler atomics exist. One slot per grid-linear thread means the
// only writer to a slot is the thread that owns it.
//
// First finding wins. A thread that has already reported keeps its first
// report rather than overwriting it with the consequences: the earliest bad
// access is the one worth showing.
__attribute__((noinline)) void report(uint32_t kind, uint64_t addr, uint32_t size,
                                      uint32_t flags, uint64_t pc,
                                      uint64_t extent_base, uint64_t extent_size,
                                      uint32_t extent_id) {
  grxSanState& s = *state();
  const uint32_t block  = linear_block();
  const uint32_t thread = linear_thread();
  const uint32_t per_block =
      (uint32_t)blockDim.x * (uint32_t)blockDim.y * (uint32_t)blockDim.z;
  const uint32_t slot = block * per_block + thread;
  if (slot >= s.max_reports) return;

  grxSanReport* r = reinterpret_cast<grxSanReport*>(s.reports) + slot;
  if (r->kind != 0) return;

  r->addr        = addr;
  r->pc          = pc;
  r->extent_base = extent_base;
  r->extent_size = extent_size;
  r->size        = size;
  r->flags       = flags;
  r->extent_id   = extent_id;
  r->block       = block;
  r->thread      = thread;
  r->warp        = get_sub_group_id();
  r->lane        = (uint32_t)vx_thread_id();
  // Written last: kind is what marks the slot occupied, so a host that reads
  // the table mid-flight sees either an empty slot or a complete report.
  r->kind        = kind;
}

// Largest extent whose base is <= addr. The host uploads the table sorted by
// base and non-overlapping, which is what makes one comparison enough.
__attribute__((always_inline)) inline const grxSanExtent* find_extent(uint64_t addr) {
  const grxSanState& s = *state();
  const grxSanExtent* e = reinterpret_cast<const grxSanExtent*>(s.extents);
  uint32_t lo = 0, hi = s.num_extents;
  const grxSanExtent* best = nullptr;
  while (lo < hi) {
    const uint32_t mid = lo + (hi - lo) / 2;
    if (e[mid].base <= addr) { best = &e[mid]; lo = mid + 1; }
    else                     { hi = mid; }
  }
  return best;
}

__attribute__((always_inline)) inline bool in_owned_region(uint64_t addr) {
  const grxSanState& s = *state();
  const grxSanRegion* g = reinterpret_cast<const grxSanRegion*>(s.regions);
  for (uint32_t i = 0; i < s.num_regions; ++i)
    if (addr >= g[i].base && addr < g[i].base + g[i].size) return true;
  return false;
}

// The check itself.
//
// Three address classes, and the honest answer for each:
//
//   shared memory   the CTA's slot is [CTA_LMEM_ADDR, +sharedMem). The stride
//                   the dispatcher uses is rounded up, so an overrun of a few
//                   bytes lands in slack that belongs to nobody and would go
//                   unnoticed without this check.
//   a region the    checked against the allocation table: no allocation means
//   allocator owns  a redzone or a hole, a freed one means use-after-free, and
//                   an access that starts inside but runs past the end is a
//                   straddle.
//   anything else   not judged. The kernel image, the stack, the driver's own
//                   argument buffer and any memory not handed out by this
//                   runtime are addresses whose bounds this runtime does not
//                   know, and guessing at them would produce false findings.
__attribute__((noinline)) void check(uint64_t addr, uint32_t size, uint32_t flags,
                                     uint64_t pc) {
  if (__grx_san_anchor == 0) return;
  const grxSanState& s = *state();
  if (!s.enabled || s.abi_version != GRX_SAN_ABI_VERSION) return;

  if (addr >= kLmemBase && addr < kLmemBase + kLmemSize) {
    const uint64_t slot = (uint64_t)(uintptr_t)__local_mem();
    if (addr < slot || addr + size > slot + s.shared_bytes)
      report(GRX_SAN_KIND_OOB_SHARED, addr, size, flags, pc, slot, s.shared_bytes, 0);
    return;
  }

  if (!in_owned_region(addr)) return;

  const grxSanExtent* e = find_extent(addr);
  if (e == nullptr || addr >= e->base + e->size) {
    report(GRX_SAN_KIND_OOB_GLOBAL, addr, size, flags, pc,
           e ? e->base : 0, e ? e->size : 0, e ? e->id : 0);
    return;
  }
  if (e->state == GRX_SAN_EXTENT_FREED) {
    report(GRX_SAN_KIND_USE_AFTER_FREE, addr, size, flags, pc, e->base, e->size, e->id);
    return;
  }
  if (addr + size > e->base + e->size) {
    report(GRX_SAN_KIND_OOB_STRADDLE, addr, size, flags, pc, e->base, e->size, e->id);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// The callbacks clang emits
// ---------------------------------------------------------------------------
//
// __builtin_return_address(0) is the address the call returns to -- an
// instruction or two past the access. The host subtracts one before
// symbolizing, which is the same convention llvm-symbolizer expects for a
// return address, so the reported line is the access, not the next statement.

#define GRX_SAN_ACCESS(name, bytes, flags)                                   \
  extern "C" __attribute__((noinline, used))                                 \
  void name(uint64_t addr) {                                                 \
    check(addr, bytes, flags, (uint64_t)(uintptr_t)__builtin_return_address(0)); \
  }

GRX_SAN_ACCESS(__asan_load1,   1,  0)
GRX_SAN_ACCESS(__asan_load2,   2,  0)
GRX_SAN_ACCESS(__asan_load4,   4,  0)
GRX_SAN_ACCESS(__asan_load8,   8,  0)
GRX_SAN_ACCESS(__asan_load16, 16,  0)
GRX_SAN_ACCESS(__asan_store1,   1, GRX_SAN_FLAG_WRITE)
GRX_SAN_ACCESS(__asan_store2,   2, GRX_SAN_FLAG_WRITE)
GRX_SAN_ACCESS(__asan_store4,   4, GRX_SAN_FLAG_WRITE)
GRX_SAN_ACCESS(__asan_store8,   8, GRX_SAN_FLAG_WRITE)
GRX_SAN_ACCESS(__asan_store16, 16, GRX_SAN_FLAG_WRITE)

#undef GRX_SAN_ACCESS

extern "C" __attribute__((noinline, used))
void __asan_loadN(uint64_t addr, uint64_t size) {
  check(addr, (uint32_t)size, 0, (uint64_t)(uintptr_t)__builtin_return_address(0));
}

extern "C" __attribute__((noinline, used))
void __asan_storeN(uint64_t addr, uint64_t size) {
  check(addr, (uint32_t)size, GRX_SAN_FLAG_WRITE,
        (uint64_t)(uintptr_t)__builtin_return_address(0));
}

// The "_noabort" spellings are what -fsanitize-recover=address emits. They are
// the same check; the difference is only whether the compiler expects the
// runtime to return, and this one always does.
extern "C" __attribute__((noinline, used))
void __asan_load4_noabort(uint64_t addr) {
  check(addr, 4, 0, (uint64_t)(uintptr_t)__builtin_return_address(0));
}
extern "C" __attribute__((noinline, used))
void __asan_store4_noabort(uint64_t addr) {
  check(addr, 4, GRX_SAN_FLAG_WRITE, (uint64_t)(uintptr_t)__builtin_return_address(0));
}

// Memory intrinsics. ASan rewrites memcpy/memset/memmove into these so the
// range gets checked; the endpoints are what matter, and checking both ends of
// a range is enough to catch a copy that runs off an allocation.
extern "C" __attribute__((noinline, used))
void* __asan_memcpy(void* dst, const void* src, uint64_t n) {
  const uint64_t pc = (uint64_t)(uintptr_t)__builtin_return_address(0);
  if (n) {
    check((uint64_t)(uintptr_t)dst, 1, GRX_SAN_FLAG_WRITE, pc);
    check((uint64_t)(uintptr_t)dst + n - 1, 1, GRX_SAN_FLAG_WRITE, pc);
    check((uint64_t)(uintptr_t)src, 1, 0, pc);
    check((uint64_t)(uintptr_t)src + n - 1, 1, 0, pc);
  }
  return __builtin_memcpy(dst, src, n);
}

extern "C" __attribute__((noinline, used))
void* __asan_memmove(void* dst, const void* src, uint64_t n) {
  const uint64_t pc = (uint64_t)(uintptr_t)__builtin_return_address(0);
  if (n) {
    check((uint64_t)(uintptr_t)dst, 1, GRX_SAN_FLAG_WRITE, pc);
    check((uint64_t)(uintptr_t)dst + n - 1, 1, GRX_SAN_FLAG_WRITE, pc);
    check((uint64_t)(uintptr_t)src, 1, 0, pc);
    check((uint64_t)(uintptr_t)src + n - 1, 1, 0, pc);
  }
  return __builtin_memmove(dst, src, n);
}

extern "C" __attribute__((noinline, used))
void* __asan_memset(void* dst, int c, uint64_t n) {
  const uint64_t pc = (uint64_t)(uintptr_t)__builtin_return_address(0);
  if (n) {
    check((uint64_t)(uintptr_t)dst, 1, GRX_SAN_FLAG_WRITE, pc);
    check((uint64_t)(uintptr_t)dst + n - 1, 1, GRX_SAN_FLAG_WRITE, pc);
  }
  return __builtin_memset(dst, c, n);
}

// ---------------------------------------------------------------------------
// Stubs
// ---------------------------------------------------------------------------
//
// clang emits a module constructor calling __asan_init and the version check,
// and references a handful of globals-registration and stack-poisoning entry
// points. The device has no init_array runner, so the constructor never
// executes -- but it must still link. These are the shape clang expects and
// nothing more; the checks that would need them (redzones around globals,
// stack-use-after-return) are not implemented, and --sanitize does not claim
// them. See docs/designs/grx_sanitize.md.

extern "C" __attribute__((used)) void __asan_init(void) {}
extern "C" __attribute__((used)) void __asan_version_mismatch_check_v8(void) {}
extern "C" __attribute__((used)) void __asan_handle_no_return(void) {}
extern "C" __attribute__((used)) void __asan_before_dynamic_init(uint64_t) {}
extern "C" __attribute__((used)) void __asan_after_dynamic_init(void) {}
extern "C" __attribute__((used)) void __asan_register_globals(uint64_t, uint64_t) {}
extern "C" __attribute__((used)) void __asan_unregister_globals(uint64_t, uint64_t) {}
extern "C" __attribute__((used)) void __asan_register_elf_globals(uint64_t, uint64_t, uint64_t) {}
extern "C" __attribute__((used)) void __asan_unregister_elf_globals(uint64_t, uint64_t, uint64_t) {}
extern "C" __attribute__((used)) void __asan_poison_stack_memory(uint64_t, uint64_t) {}
extern "C" __attribute__((used)) void __asan_unpoison_stack_memory(uint64_t, uint64_t) {}
extern "C" __attribute__((used)) uint64_t __asan_stack_malloc_0(uint64_t) { return 0; }
extern "C" __attribute__((used)) uint64_t __asan_stack_malloc_1(uint64_t) { return 0; }
extern "C" __attribute__((used)) uint64_t __asan_stack_malloc_2(uint64_t) { return 0; }
extern "C" __attribute__((used)) uint64_t __asan_stack_malloc_3(uint64_t) { return 0; }
extern "C" __attribute__((used)) uint64_t __asan_stack_malloc_4(uint64_t) { return 0; }
extern "C" __attribute__((used)) void __asan_stack_free_0(uint64_t, uint64_t) {}
extern "C" __attribute__((used)) void __asan_stack_free_1(uint64_t, uint64_t) {}
extern "C" __attribute__((used)) void __asan_stack_free_2(uint64_t, uint64_t) {}
extern "C" __attribute__((used)) void __asan_stack_free_3(uint64_t, uint64_t) {}
extern "C" __attribute__((used)) void __asan_stack_free_4(uint64_t, uint64_t) {}
extern "C" __attribute__((used)) int  __asan_option_detect_stack_use_after_return = 0;
extern "C" __attribute__((used)) uint64_t __asan_shadow_memory_dynamic_address = 0;
