# GRXCP — GRX Compute Platform Architecture

**Status:** Design specification (pre-implementation).
**Scope:** the full CUDA-class compute software platform for the **GRX-G100**
GPU and the **GRX930** host SoC — layer model, API surface, kernel ABI,
compilation model, memory model, device model, libraries, and tools.
**Substrate:** the GRX-G100 stack in `grxgpu/` (Vortex-derived RISC-V GPGPU:
`vortex2.h` async driver API, hardware command processor, KMU/CTA dispatch,
TCU, DXA, RTU) and the GRX930 SoC in `grx930/` (RV64 core + systolic NPU).
**Reference model:** the CUDA architecture as summarized in
[ajdillhoff — CUDA Architecture](https://ajdillhoff.github.io/notes/cuda_architecture/)
and NVIDIA's Volta→Hopper programming/driver model.
**Companion docs:** [`cuda_mapping.md`](cuda_mapping.md) (concept-by-concept
mapping + gap register), [`grxcp_roadmap.md`](grxcp_roadmap.md) (phases).

---

## 1. What GRXCP is, and what it is not

GRXCP is the **software platform** that makes GRX-G100 programmable the way
CUDA makes an NVIDIA GPU programmable: one source file containing host and
device code, a compiler that splits it, a runtime that manages devices,
memory, streams and events, a set of math libraries, and profiling/debug
tools.

It is **not** a new driver. The GRX-G100 already has a good one:
`vortex2.h` is an async, handle-based, timeline-event driver API in the
shape of CUDA's *driver* API (`cuModule`/`cuFunction`/`cuStream`/
`cuEvent`). GRXCP consumes it unchanged.

It is **not** a fork of `grxgpu`. GRXCP is a separate repository that links
against the installed GRX-G100 sysroot through `pkg-config`
(`vortex-runtime.pc`, `vortex-kernel.pc`), exactly as the GRX-G100 project
requires of downstream consumers (`grxgpu/AGENTS.md` §2, `README.md`
"Building and installing"). GRXCP never reaches into `$VORTEX_HOME` or
`$VORTEX_BUILD_DIR`.

Three decisions, taken up front, define the project:

| Decision | Choice | Rationale |
|---|---|---|
| API identity | **GRX-named API + a CUDA compatibility header** | The HIP playbook. `grxMalloc`/`grxLaunchKernel` is our surface; `grx_cuda_compat.h` maps `cudaX → grxX` so ported code needs a one-line include change. Clean identity, trivial porting, no API-copying exposure. |
| Compilation | **SPIR-V path first, `grxcc` single-source driver second** | The existing chipStar→SPIR-V→POCL path works today and gets us a running platform in Phase 1. It structurally cannot reach TCU/DXA intrinsics (`grxgpu/docs/designs/hip_on_vortex_chipstar.md` §5.1), which is why `grxcc` follows. |
| Device scope | **GPU first; device model multi-device from day one** | `grxGetDeviceCount`/`grxGetDeviceProperties` enumerate heterogeneous devices with capability profiles from the start. The GRX930 systolic NPU joins later as a GEMM-only device without an API break. |

---

## 2. Layer model

```
┌──────────────────────────────────────────────────────────────────────────┐
│ L4  Tools          grx-smi │ grx-prof │ grx-sanitize │ grx-gdb │ grxify   │
├──────────────────────────────────────────────────────────────────────────┤
│ L3  Libraries      grxBLAS │ grxDNN │ grxFFT │ grxRAND │ grxSPARSE │ grx::par│
├──────────────────────────────────────────────────────────────────────────┤
│ L2  Language       grxcc (host/device split driver)                       │
│     + device hdrs  __global__ __device__ __shared__ <<<grid,block,smem,s>>>│
│                    grx::wmma │ grx::pipeline │ grx::cg │ grx::warp        │
├──────────────────────────────────────────────────────────────────────────┤
│ L1  Runtime API    libgrxrt.so — grxMalloc/grxMemcpy/grxLaunchKernel/      │
│     (CUDA Runtime  grxStreamCreate/grxEventRecord/grxGetDeviceProperties   │
│      analogue)     implicit per-thread context, fat-binary registration    │
├──────────────────────────────────────────────────────────────────────────┤
│ L0  Driver API     vortex2.h  (EXISTS — consumed unchanged)                │
│     (CUDA Driver   vx_device/vx_buffer/vx_queue/vx_event/vx_module/vx_kernel│
│      analogue)     vx_enqueue_launch/copy/read/write/signal/wait_value      │
├──────────────────────────────────────────────────────────────────────────┤
│     Hardware       Command Processor ring → KMU grid walk → VX_cta_dispatch │
│                    SM cores │ TCU (WGMMA) │ DXA (async copy/multicast) │ RTU │
│                    Backends: simx │ rtlsim │ xrt │ opae │ gem5              │
└──────────────────────────────────────────────────────────────────────────┘
```

**The load-bearing observation:** L0 already exists and is well-shaped.
GRXCP is L1–L4. The GRX-G100 team's own rule — *"align with mainstream CUDA,
HIP, OpenCL and Vulkan driver stacks"* (`grxgpu/AGENTS.md` §5) — has already
been applied to the driver, which is why the mapping in §6 is unusually
clean.

### 2.1 Deliverables per layer

| Layer | Artifacts | Depends on |
|---|---|---|
| L1 | `libgrxrt.so`, `libgrxrt.a`, `grx_runtime.h`, `grx_types.h`, `grx_cuda_compat.h`, `grxrt.pc` | `vortex-runtime.pc` |
| L2 | `grxcc`, `grxcc-device` (VOLT/clang wrapper), `libgrxdevice.a`, device headers under `include/grx/device/` | `vortex-kernel.pc`, VOLT |
| L3 | `libgrxblas`, `libgrxdnn`, `libgrxfft`, `libgrxrand`, `libgrxsparse`, header-only `grx::par` | L1 + L2 |
| L4 | `grx-smi`, `grx-prof`, `grx-sanitize`, `grx-gdb` wrapper, `grxify` | L1 + MPM counters + OpenOCD |

---

## 3. The runtime API (L1)

### 3.1 Design rules

1. **CUDA Runtime semantics, GRX names.** Every entry point has a CUDA
   analogue with identical semantics (return-code enum, async-by-default on
   the null stream, `grxDeviceSynchronize` as a full barrier). Where CUDA's
   semantics are a historical wart, we keep them anyway — porting fidelity
   beats elegance at this layer, and `grx::` C++ wrappers (§3.7) provide the
   nicer surface.
2. **Implicit context.** CUDA's runtime API hides `cuCtxCreate`. GRXCP
   keeps a per-process device table and a thread-local current-device index;
   `grxSetDevice(i)` selects, everything else infers. `vortex2.h` has no
   context object, so this is a pure host-side construct.
3. **UVA pointers, not handles.** `vortex2.h` hands out `vx_buffer_h`
   handles plus a device address via `vx_buffer_address`. GRXCP's public
   currency is a plain `void*` device address; an interval map recovers the
   owning handle. This is what makes `grxMemcpy(dst, src, n, kind)` and
   raw-pointer kernel args work.
4. **No new driver features required for v1.** Everything in the v1 runtime
   is expressible on the shipped `vortex2.h` surface. Where a CUDA feature
   needs driver or hardware work (concurrent streams, event timing,
   `__constant__`), it is listed in the gap register
   ([`cuda_mapping.md`](cuda_mapping.md) §7), not silently faked.
5. **Errors are sticky and thread-local**, CUDA-style:
   `grxGetLastError()` clears, `grxPeekAtLastError()` does not.

### 3.2 Device management

```c
grxError_t grxGetDeviceCount(int* count);
grxError_t grxSetDevice(int device);
grxError_t grxGetDevice(int* device);
grxError_t grxGetDeviceProperties(grxDeviceProp_t* prop, int device);
grxError_t grxDeviceSynchronize(void);
grxError_t grxDeviceReset(void);
grxError_t grxMemGetInfo(size_t* free, size_t* total);
```

`grxDeviceProp_t` is populated entirely from `vx_device_query` capability
IDs — no invented numbers:

| `grxDeviceProp_t` field | Source |
|---|---|
| `warpSize` | `VX_CAPS_NUM_THREADS` |
| `maxWarpsPerMultiProcessor` | `VX_CAPS_NUM_WARPS` |
| `multiProcessorCount` | `VX_CAPS_NUM_CORES` (already = cores × clusters) |
| `clusterCount` | `VX_CAPS_NUM_CLUSTERS` |
| `socketSize` | `VX_CAPS_SOCKET_SIZE` |
| `issueWidth` | `VX_CAPS_ISSUE_WIDTH` |
| `totalGlobalMem` | `VX_CAPS_GLOBAL_MEM_SIZE` |
| `sharedMemPerMultiprocessor` | `VX_CAPS_LOCAL_MEM_SIZE` |
| `l2CacheSize` / bank layout | `VX_CAPS_NUM_MEM_BANKS` × `VX_CAPS_MEM_BANK_SIZE` |
| `memoryBusWidth` proxy / `memoryBandwidth` | `VX_CAPS_PEAK_MEM_BW` |
| `clockRate` | `VX_CAPS_CLOCK_RATE` |
| `globalL1CacheSupported`, `tccDriver`… | `VX_CAPS_ISA_FLAGS` bits (`VX_ISA_EXT_DCACHE`, `L2CACHE`, `L3CACHE`, `LMEM`) |
| `tensorCoreSupported` | `VX_ISA_EXT_TCU` |
| `asyncCopySupported` | `VX_ISA_EXT_DXA` |
| `rayTracingSupported` | `VX_ISA_EXT_RTU` |
| `unifiedAddressing` / `managedMemory` | `VX_CAPS_VM_SUPPORT` |
| `pinnedMemTotal` / `pinnedMemFree` | `VX_CAPS_VM_PINNED_SIZE` / `_FREE` |

Two GRX-specific additions, deliberately not CUDA-shaped, because pretending
otherwise would mislead:

```c
typedef enum { GRX_DEVICE_TYPE_GPU = 0,
               GRX_DEVICE_TYPE_NPU = 1 } grxDeviceType_t;

// Which execution backend this device is running on. Programs that care
// about wall-clock (benchmarks, timeouts) must check this: a simx device
// is 5–6 orders of magnitude slower than silicon.
typedef enum { GRX_BACKEND_SIMX = 0, GRX_BACKEND_RTLSIM = 1,
               GRX_BACKEND_XRT  = 2, GRX_BACKEND_OPAE   = 3,
               GRX_BACKEND_GEM5 = 4, GRX_BACKEND_SILICON = 5 } grxBackend_t;
```

`grxDeviceProp_t.computeCapability` follows the G100 chip design's declared
target of **10.0** (`grxgpu/docs/designs/gpu_chip_design.md` §2), encoded as
`{major=10, minor=0}` so occupancy/feature gating has the same shape as CUDA.

### 3.3 Memory

```c
grxError_t grxMalloc      (void** ptr, size_t size);
grxError_t grxFree        (void* ptr);
grxError_t grxMallocHost  (void** ptr, size_t size);   // pinned, VX_MEM_HOST
grxError_t grxFreeHost    (void* ptr);
grxError_t grxMallocManaged(void** ptr, size_t size, unsigned flags); // needs VM
grxError_t grxMemcpy      (void* dst, const void* src, size_t n, grxMemcpyKind k);
grxError_t grxMemcpyAsync (void* dst, const void* src, size_t n,
                           grxMemcpyKind k, grxStream_t s);
grxError_t grxMemcpy2D / grxMemcpy3D (…);              // → vx_enqueue_*_rect
grxError_t grxMemset / grxMemsetAsync(…);              // → vx_enqueue_fill_buffer
grxError_t grxHostRegister / grxHostUnregister(…);
grxError_t grxPointerGetAttributes(grxPointerAttributes* a, const void* p);
```

**Allocator design.** `vx_buffer_create` is a device-memory *allocator* call,
not a cheap operation, and every buffer is a refcounted handle. Calling it
per `grxMalloc` is correct but wasteful for the small-allocation patterns
CUDA code produces. GRXCP therefore runs a two-tier allocator:

- **Slab tier** — a set of large `vx_buffer_create` slabs (default 64 MiB,
  `GRX_SLAB_BYTES`) carved by a best-fit free-list with 256-byte alignment
  (≥ the device cache-line, discovered via `VX_CAPS_CACHE_LINE_SIZE`).
- **Direct tier** — allocations ≥ ¼ slab get their own `vx_buffer_create`,
  so huge tensors don't fragment slabs.

Both tiers register `[address, address+size) → {vx_buffer_h, offset}` in a
red-black interval map. `grxMemcpy` resolves the device pointer through the
map to `(handle, offset)` and calls `vx_enqueue_read`/`write`/`copy`.
`grxPointerGetAttributes` is the same lookup. Freeing returns the extent to
the free list; slabs are released on `grxDeviceReset`.

**`grxMemcpyKind` resolution.** `grxMemcpyDefault` is supported and is the
recommended form: the interval map tells us which side is device memory, so
the runtime picks `read`/`write`/`copy`/plain `memcpy` correctly. The
explicit kinds are validated against the map and return
`grxErrorInvalidMemcpyDirection` on mismatch — CUDA silently misbehaves
here; we do not.

**Managed memory.** `grxMallocManaged` is gated on
`VX_CAPS_VM_SUPPORT == 1`. Where the device has an MMU and a single global
VA space (the G100 design's SVM model,
`grxgpu/docs/designs/gpu_chip_design.md` §7.2), managed allocations are
plain SVM allocations. On backends without VM (today: FPGA paths — see the
CP design's §10 item 2, "VM in RTL"), `grxMallocManaged` returns
`grxErrorNotSupported` rather than degrading to a copy-based emulation that
would silently change performance characteristics.

### 3.4 Streams

```c
grxError_t grxStreamCreate           (grxStream_t* s);
grxError_t grxStreamCreateWithFlags  (grxStream_t* s, unsigned flags);
grxError_t grxStreamCreateWithPriority(grxStream_t* s, unsigned f, int prio);
grxError_t grxStreamDestroy          (grxStream_t s);
grxError_t grxStreamSynchronize      (grxStream_t s);
grxError_t grxStreamQuery            (grxStream_t s);
grxError_t grxStreamWaitEvent        (grxStream_t s, grxEvent_t e, unsigned f);
```

`grxStream_t` wraps a `vx_queue_h` 1:1. Priorities map onto
`vx_queue_priority_e`. The **null stream** is a real queue created lazily
per device, with CUDA's legacy-default-stream semantics (implicit sync with
all blocking streams) implemented host-side via the event graph;
`grxStreamNonBlocking` opts out.

> **Honest caveat, carried into the roadmap.** Stream *semantics* are correct
> from Phase 1, but stream *concurrency* is not yet real. The command
> processor is parameterized on `NUM_QUEUES` but defaults to 1, the emulation
> CP models only `q0_`, and the runtime serializes launches; true per-queue
> concurrency is blocked on the QMD-style atomic `CMD_LAUNCH` work
> (`grxgpu/docs/designs/command_processor.md` §10 items 5–6). Until then a
> multi-stream program is correct and portable but runs serialized. This is
> a roadmap item (Phase 5), not a runtime workaround.

### 3.5 Events

```c
grxError_t grxEventCreate         (grxEvent_t* e);
grxError_t grxEventCreateWithFlags(grxEvent_t* e, unsigned flags);
grxError_t grxEventDestroy        (grxEvent_t e);
grxError_t grxEventRecord         (grxEvent_t e, grxStream_t s);
grxError_t grxEventSynchronize    (grxEvent_t e);
grxError_t grxEventQuery          (grxEvent_t e);
grxError_t grxEventElapsedTime    (float* ms, grxEvent_t start, grxEvent_t end);
```

`vx_event_h` is a **timeline** event (monotonic uint64 counter,
`vx_event_signal`/`wait_value`) — strictly more expressive than CUDA's
binary event. `grxEvent_t` = `{vx_event_h ev; uint64_t target;}`; a record
increments and captures the target value, so one `vx_event_h` can back many
recordings without reallocation.

`grxEventElapsedTime` prefers `vx_event_get_profiling`
(`{queued,submit,start,end}` in ns). That path needs CP profiling writeback,
listed as open in `command_processor.md` §10 item 9; until it lands, GRXCP
falls back to host `CLOCK_MONOTONIC` timestamps taken at enqueue and at
completion-poll, and sets `grxDeviceProp_t.eventTimingIsDeviceSide = 0` so
benchmarks can report which clock they used. Silent substitution of a host
clock for a device clock is a correctness bug in a profiling API; we surface
it.

### 3.6 Modules, kernels, and launch

The **runtime** API path (what `<<<>>>` compiles to):

```c
grxError_t grxLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                           void** args, size_t sharedMem, grxStream_t stream);
grxError_t grxFuncGetAttributes(grxFuncAttributes* a, const void* func);
grxError_t grxFuncSetAttribute (const void* func, grxFuncAttribute a, int v);
```

The **module** API path (driver-style, for JIT and language runtimes):

```c
grxError_t grxModuleLoad       (grxModule_t* m, const char* path);
grxError_t grxModuleLoadData   (grxModule_t* m, const void* image);
grxError_t grxModuleGetFunction(grxFunction_t* f, grxModule_t m, const char* name);
grxError_t grxModuleUnload     (grxModule_t m);
```

These are near-passthroughs to `vx_module_load_file`/`vx_module_load_bytes`/
`vx_module_get_kernel`/`vx_kernel_*`. The `.vxbin` `VXSYMTAB` multi-entry
footer (`grxgpu/docs/designs/kernel_entry_and_dispatch.md` §2) is exactly
CUDA's cubin symbol table: named entry points resolved to PCs.

**Launch lowering.** `grxLaunchKernel` builds one `vx_launch_info_t`:

| `vx_launch_info_t` field | From |
|---|---|
| `kernel` | registry lookup: host stub address → `vx_kernel_h` (§4.3) |
| `args_host` / `args_size` | the packed argument blob (§4.4) |
| `ndim` | 3 (or reduced when y=z=1, to let the KMU walk fewer levels) |
| `grid_dim[3]` | `gridDim` |
| `block_dim[3]` | `blockDim` |
| `lmem_size` | `sharedMem` + the kernel's static `__shared__` bytes |
| `cluster_dim[3]` | from `grxLaunchKernelEx` cluster attribute; `{1,1,1}` default |

then calls `vx_enqueue_launch(queue, &info, nwait, waits, &out_event)`.
`vortex2.h` stages the arg blob into a device scratch slot and programs the
KMU ARG DCRs itself — GRXCP never allocates an args buffer. This is why
raw-pointer kernel arguments work with no marshalling layer.

**Extended launch** (`grxLaunchKernelEx`) carries an attribute list, CUDA
12-style. The v1 attribute set:

```c
GRX_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION      // → cluster_dim[3]
GRX_LAUNCH_ATTRIBUTE_COOPERATIVE            // → global-barrier legality check
GRX_LAUNCH_ATTRIBUTE_PRIORITY               // → queue priority
GRX_LAUNCH_ATTRIBUTE_SHARED_MEM_CARVEOUT    // → L1/shared carve-out DCR (Phase 6)
```

Thread-block clusters are **not** an aspirational feature here: the KMU
already walks the grid two levels deep so a cluster's CTAs are emitted
contiguously, `VX_cta_dispatch` reserves K consecutive fixed-stride LMEM
slots, and `VX_CSR_CTA_CLUSTER_SIZE` is readable from the kernel
(`grxgpu/docs/designs/cta_clustering_and_dispatch.md` §2–§4). GRXCP exposes
what the hardware already does.

### 3.7 Occupancy

```c
grxError_t grxOccupancyMaxActiveBlocksPerMultiprocessor(int* n, const void* f,
                                                        int blockSize,
                                                        size_t dynSmem);
grxError_t grxOccupancyMaxPotentialBlockSize(int* minGrid, int* blockSize,
                                             const void* f, size_t dynSmem,
                                             int blockSizeLimit);
```

Unlike most of the runtime, this is **not** a passthrough — and it must not
be guesswork, because the CTA dispatcher's allocator has an exact,
documented formula. Resident CTAs per SM is the min of three bounds:

```
warp bound   : floor(NUM_WARPS / ceil(blockSize / NUM_THREADS))
slot bound   : NUM_CTA_SLOTS == NUM_WARPS
smem bound   : floor(LOCAL_MEM_SIZE / align_up(smem_per_CTA, MEM_BLOCK_SIZE))
```

The smem term is the fixed-stride allocator's `usable_slots` computation
verbatim (`cta_clustering_and_dispatch.md` §3.1) — *not* CUDA's
register-pressure model, because GRX-G100's dispatcher does not gate CTA
admission on register count today. `grxFuncGetAttributes` reports
`numRegs = -1` (unknown) until `grxcc` emits per-kernel register metadata
into the `.vxbin` footer (Phase 4); it never reports a fabricated number.
`vx_device_max_occupancy_grid` supplies the device's natural block dims as
the search seed.

### 3.8 The C++ convenience layer

A header-only `grx::` namespace over the C API — RAII `grx::stream`,
`grx::event`, `grx::device_buffer<T>`, `grx::error` exceptions,
`grx::launch<<<...>>>` helpers. It exists so new GRX-native code is not
forced through C-with-out-params ergonomics. Ported CUDA code ignores it.

---

## 4. The compilation model (L2)

### 4.1 Two paths, one binary format

```
PATH A — SPIR-V (Phase 1, works with today's toolchain)
  kernel.grx.cpp ──chipStar hipcc/clang──► device.spv ──POCL JIT──► .vxbin
                 └─host ELF (embeds .spv fatbin, links libgrxrt)

PATH B — grxcc single-source (Phase 4, the real platform)
  kernel.grx.cpp ──grxcc──┬── device pass: clang --target=riscv{32,64}
                          │     + VOLT SIMT backend  →  ELF → vxbin.py → .vxbin
                          └── host pass:  clang --target={x86_64,riscv64}
                                +  <<<>>> rewrite  +  registration ctors
                          → link:  host.o + .grxfatbin section + libgrxrt
```

Both paths produce the same `.vxbin` and register through the same
mechanism, so a program's runtime behavior does not depend on which
compiled it. Path A gets us a working platform without writing a compiler;
Path B is the one that can reach TCU/DXA intrinsics — the gap the GRX-G100
docs already identify as structural in the SPIR-V route
(`hip_on_vortex_chipstar.md` §5.1).

### 4.2 The fat binary

`.grxfat` is a container, embedded in the host ELF section `.grxfatbin`:

```
struct grx_fatbin_header {
  uint32_t magic;        // 'G','R','X','F'
  uint16_t version;      // 1
  uint16_t num_entries;
  uint64_t total_size;
};
struct grx_fatbin_entry {
  uint32_t kind;         // 0 = .vxbin, 1 = SPIR-V, 2 = LLVM IR (JIT fallback)
  uint32_t xlen;         // 32 or 64
  uint32_t isa_flags;    // required VX_ISA_EXT_* bits (TCU, DXA, RTU…)
  uint32_t reserved;
  uint64_t offset, size;
};
```

Multiple entries let one host binary carry an rv32 and an rv64 image, or a
TCU-enabled and a TCU-free image, and let the runtime pick by matching
`isa_flags` against `vx_device_query(VX_CAPS_ISA_FLAGS)`. This is CUDA's
fatbin/`-gencode` model with GRX's actual capability bits instead of SM
versions.

### 4.3 Registration and the stub-address trick

`grxLaunchKernel` takes `const void* func` — the address of a host-side
stub — because that is what `<<<>>>` can name at the call site. The mapping
from stub address to device kernel is built at static-init time, exactly as
nvcc does it:

```c
// emitted by grxcc (or by the Path-A wrapper) into each TU
static void** __grx_fatbin_handle;
__attribute__((constructor)) static void __grx_register(void) {
  __grx_fatbin_handle = __grxRegisterFatBinary(&__grx_fatbin_blob);
  __grxRegisterFunction(__grx_fatbin_handle,
                        (const char*)vecadd_stub,   // host stub address
                        "vecadd",                   // .vxbin VXSYMTAB name
                        /*minBlocks*/0, /*maxThreads*/0);
}
__attribute__((destructor)) static void __grx_unregister(void) {
  __grxUnregisterFatBinary(__grx_fatbin_handle);
}
```

The runtime keeps `stub_addr → {module, kernel_name, vx_kernel_h}` and
resolves lazily on first launch per device (so a 4-device process loads each
module once per device, not once per launch).

### 4.4 Kernel argument ABI

CUDA passes `void** args` — an array of pointers to argument values.
`vx_launch_info_t` wants a single flat host blob. `grxcc` knows each
kernel's parameter layout, so it emits a per-kernel descriptor:

```c
struct grx_kernel_param { uint16_t offset; uint16_t size; uint8_t is_pointer; };
```

The runtime packs `args[i]` into the blob at `offset` with natural
alignment, matching the device-side struct layout that `__vx_cta_entry`
loads into `a0` from `MSCRATCH`
(`kernel_entry_and_dispatch.md` §1). Buffers appear inline as `uint64_t`
device addresses — already the documented convention in `vortex2.h`.

> **rv32 hazard, inherited.** On a 32-bit device, host `size_t` is 8 bytes
> and device `size_t` is 4. The chipStar path documents this as an accepted
> unfixed risk (`hip_on_vortex_chipstar.md` §5.4). `grxcc` does **not**
> inherit it: because the driver knows the parameter layout, it narrows
> pointer/size_t parameters at pack time and emits a diagnostic when a
> kernel parameter's host and device widths disagree in a way that cannot be
> narrowed losslessly.

### 4.5 `grxcc` driver responsibilities

1. Parse the compilation line; separate host and device flags
   (`-grx-arch=g100`, `--grx-device-debug`, `-grx-max-registers`).
2. Run the device pass: clang with the VOLT SIMT backend, targeting
   `riscv{32,64}` with the `vortex.kernel` annotation convention that
   `vx_spawn2.h`'s `__kernel` macro already establishes; link with
   `link{32,64}.ld`; run `vxbin.py` to produce the `.vxbin` with its
   `VXSYMTAB` footer.
3. Run the host pass: rewrite `f<<<g,b,s,st>>>(a,b)` into
   `__grxPushCallConfiguration(g,b,s,st); f_stub(a,b);` where the stub calls
   `__grxPopCallConfiguration` + `grxLaunchKernel`. Emit the registration
   constructors and the parameter descriptors.
4. Link: host objects + `.grxfatbin` + `-lgrxrt`.
5. **Host-target matrix:** `x86_64-linux-gnu` (bring-up workstation) and
   `riscv64-linux-gnu` (native on GRX930). The runtime has no x86
   assumptions; the only host-arch-sensitive code is the pinned-memory path
   and the timestamp source.

### 4.6 Device-side headers

Where GRX-G100 already provides the mechanism, GRXCP's device headers are
thin renames — a design goal, not a shortcut:

| GRXCP device API | Backed by |
|---|---|
| `threadIdx`, `blockIdx`, `blockDim`, `gridDim` | `vx_spawn2.h` structs over `VX_CSR_CTA_*` |
| `__syncthreads()` | `vx_barrier(cta_id, num_warps)` |
| `__global__` | `__kernel` = `annotate("vortex.kernel"), used, retain` |
| `__shared__` | `__local_mem()` → `VX_CSR_CTA_LMEM_ADDR` carve |
| `printf` in kernels | `vx_printf` (`vx_print.h`) |
| `grx::wmma::{fragment,load_matrix_sync,mma_sync,store_matrix_sync}` | `vortex::tensor::wmma_context` / `wgmma_context` |
| `grx::pipeline` / `memcpy_async` | `vx_dxa_issue_{1..5}d_wg` + `dxa_multicast_*` |
| `grx::barrier` (arrive/wait/expect_tx) | `vx_barrier_arrive` / `_wait` / `_expect_tx` — an mbarrier with transaction counts, already present |
| `grx::cg::this_grid().sync()` | `vx_gbar` global barrier |
| `grx::cluster_rank()/cluster_size()` | `VX_CSR_CTA_CLUSTER_SIZE` + `CTA_ID % size` |
| `__activemask()`, `__ballot_sync()` | `vx_active_threads()` |
| divergence, predication | `vx_split`/`vx_join`/`vx_pred` (compiler-emitted) |

**The one real language gap: general warp shuffle.** `vx_wgather` is a
*4-lane group* gather (its documented use is a 4×4 transpose), not an
arbitrary-lane shuffle. `__shfl_sync`, `__shfl_xor_sync`, `__shfl_up/down_sync`
across a 32-lane warp have no single-instruction backing today. GRXCP ships
a correct LMEM-staged fallback (write lane value to a per-warp shared slot,
`__syncwarp`, read the source lane) and files an ISA RFC for a `WSHFL`
instruction. See [`cuda_mapping.md`](cuda_mapping.md) §7.1.

---

## 5. Memory model

GRXCP adopts the CUDA memory model, mapped onto the G100 spaces the chip
design already defines (`gpu_chip_design.md` §7.1):

| CUDA space | GRXCP qualifier | G100 backing | Status |
|---|---|---|---|
| registers | (automatic) | per-thread register file | exists |
| `__shared__` | `__shared__` | LMEM carve-out, fixed-stride CTA slot | exists |
| distributed shared (cluster) | `grx::cluster::map_shared_rank()` | consecutive LMEM slots + DXA multicast | exists |
| global | `__device__` / `grxMalloc` | HBM via L2 | exists |
| local (spill) | (automatic) | per-thread spill in device memory | exists |
| `__constant__` | `__constant__` | **gap** — no broadcast constant path exposed | Phase 6 |
| texture / surface | `grx::tex<>` | TEX units + TCACHE | exists in HW, not exposed to compute |

**Consistency.** The device-side memory model is release-consistency with
scoped fences, matching `vx_fence` and the AMO-at-LLC design
(`atomic_memory_operations.md`, `multicache_amo_coherence.md`). GRXCP's
`grx::atomic<T, scope>` types name the three scopes the hardware actually
distinguishes: `thread_block` (LMEM/L1), `device` (L2 AMO point), and
`system` (host-visible, requires cache flush — `CMD_CACHE_FLUSH`). CUDA's
`cuda::atomic` shapes map 1:1.

**Host↔device coherence.** Explicit, via the command processor's DMA path.
`grxMemcpy` on the null stream implies the CUDA-mandated synchronization
points. Cache maintenance uses `CMD_CACHE_FLUSH`, which the runtime already
issues after each launch.

---

## 6. Multi-device and the GRX930 NPU

The device table is heterogeneous from v1 even though only one device type
exists. A device entry carries a **capability profile**: a bitmask of which
GRXCP subsystems it implements.

| Profile bit | GPU (G100) | NPU (GRX930 c930) |
|---|---|---|
| `GRX_CAP_KERNEL_LAUNCH` | yes | no — no programmable SIMT pipeline |
| `GRX_CAP_STREAMS`, `GRX_CAP_EVENTS` | yes | yes (MMIO doorbell + IRQ/`STATUS.DONE`) |
| `GRX_CAP_MEMCPY` | yes | yes (`c930_npu_dma`, AXI4 master) |
| `GRX_CAP_GEMM` | yes (TCU) | yes (systolic array, INT8→INT32 today) |
| `GRX_CAP_UNIFIED_ADDRESSING` | with VM | later, via the planned CHI coherent port + SVM |

Consequences, decided now so the NPU can be added without an API break:

- `grxSetDevice(npu)` followed by `grxLaunchKernel` returns
  `grxErrorNotSupported` — it does not silently fall back.
- **grxBLAS is the NPU's real entry point.** `grxblasGemmEx` on an NPU
  device programs `DIM_M/N/K` + `A_BASE/B_BASE/C_BASE` and pulses `CTRL.START`
  in the c930 NPU's MMIO map (`grx930/c930/doc/c930_architecture.md` §5), then
  waits on `STATUS.DONE`. Same API, different backend — the pattern that lets
  a library route work to whichever engine fits.
- Peer access (`grxDeviceEnablePeerAccess`) is declared in v1 and returns
  `grxErrorNotSupported` until the NPU gains its coherent port (c930 roadmap
  step 5) or the G100 gains NVLink-class peer decode
  (`gpu_chip_design.md` §8.1).

The GRX930 is also the eventual **host**: `libgrxrt` must build and run
natively on riscv64. Nothing in the design blocks this — the only host-arch
dependencies are pinned-memory allocation and the monotonic clock.

---

## 7. Libraries (L3)

Ordered by leverage, not by CUDA's alphabet:

| Library | v1 scope | Backing |
|---|---|---|
| **grxBLAS** | GEMM (fp32/fp16/bf16/int8), GEMV, AXPY, batched GEMM | TCU WGMMA + DXA multicast staging; the `sgemm_tcu_wg_dxa_mcast` kernel family already exists as a tuned reference in `grxgpu/tests/` |
| **grxDNN** | conv2d (implicit GEMM), pooling, softmax, layernorm, GELU, attention | grxBLAS + custom kernels |
| **grxRAND** | Philox 4×32-10, XORWOW; host + device APIs | pure compute, no HW dependency |
| **grxFFT** | 1D/2D/3D complex + real, radix-2/4 Stockham | LMEM-resident butterflies |
| **grxSPARSE** | CSR SpMV, SpMM | AMO at LLC for scatter |
| **grx::par** | Thrust-shaped: `transform`, `reduce`, `scan`, `sort`, `for_each` | header-only over grxcc |

grxBLAS is the pivot: it justifies the TCU exposure work, it is what the NPU
plugs into, and it is the single library most ported code touches first.

---

## 8. Tools (L4)

| Tool | Function | Backing (mostly exists) |
|---|---|---|
| `grx-smi` | device enumeration, memory, utilization, backend | `vx_device_query`, `vx_device_memory_info` |
| `grx-prof` | kernel timeline, occupancy, stall breakdown, roofline | MPM counters (`vx_device_mpm_query`), Perfetto export (`grxgpu/ci/perfetto.py`), `grxgpu/perf/roofline.py` |
| `grx-sanitize` | out-of-bounds, uninitialized shared, race detection | SimX instrumentation — cheap on a functional simulator, and a genuine advantage over silicon-only tooling |
| `grx-gdb` | source-level kernel debugging | existing GDB/OpenOCD path (`grxgpu/docs/kernel_debugging.md`) |
| `grxify` | CUDA source → GRXCP source translator | clang-tidy-style rewriter; the `hipify` analogue |

`grx-sanitize` deserves emphasis: because SimX is a functional model with
full memory visibility and is already the project's timing oracle, a
compute-sanitizer equivalent is far cheaper here than on real hardware, and
it can ship in Phase 2 rather than at the end.

---

## 9. Repository layout

```
grxcp/
├── AGENTS.md                     # invariants and footguns (mirrors grxgpu's)
├── README.md
├── CMakeLists.txt                # finds vortex-runtime.pc / vortex-kernel.pc
├── docs/
│   ├── index.md
│   └── designs/
│       ├── grxcp_architecture.md # this document
│       ├── cuda_mapping.md       # concept mapping + gap register
│       └── grxcp_roadmap.md      # phases and exit criteria
├── include/grx/
│   ├── grx.h                     # umbrella
│   ├── grx_types.h               # dim3, grxError_t, enums, structs
│   ├── grx_runtime.h             # the L1 C API
│   ├── grx_cuda_compat.h         # cudaX -> grxX
│   ├── grxblas.h  grxdnn.h  grxrand.h  grxfft.h  grxsparse.h
│   └── device/
│       ├── grx_device.h          # threadIdx/blockIdx/__syncthreads/printf
│       ├── grx_warp.h            # ballot/any/all/shfl (+ LMEM fallback)
│       ├── grx_wmma.h            # grx::wmma over vortex::tensor
│       ├── grx_pipeline.h        # memcpy_async / barrier over DXA
│       └── grx_cg.h              # cooperative groups
├── src/
│   ├── runtime/                  # context, memory, stream, event, module, launch
│   ├── libs/                     # grxblas, grxdnn, ...
│   └── backends/                 # gpu_g100/, npu_c930/
├── tools/
│   ├── grxcc/                    # the compiler driver
│   ├── grx-smi/  grx-prof/  grx-sanitize/  grxify/
├── tests/
│   ├── unit/  conformance/  regression/  benchmarks/
│   └── samples/                  # vecadd, sgemm, reduction, attention
└── ci/
    ├── testcases/                # YAML catalog, same shape as grxgpu's
    └── regression.sh
```

---

## 10. Verification strategy

GRXCP inherits the GRX-G100 project's verification discipline, adapted:

1. **Golden-numerics gate.** Every library kernel has a CPU reference; the
   test asserts bitwise or ULP-bounded agreement. No "close enough."
2. **Cross-backend gate.** Every conformance test runs on `simx` and
   `rtlsim`. A GRXCP-level result that differs between backends is a bug in
   GRXCP or a model-parity bug in GRX-G100 — either way it blocks.
3. **API conformance suite.** A ported subset of the CUDA sample corpus,
   compiled through `grx_cuda_compat.h`, is the porting-fidelity gate. It is
   also the honest measure of how complete the platform is: publish the pass
   rate, the way the chipStar work publishes its ~36% rv32 conformance
   number rather than hiding it.
4. **Perf baselines.** `ci/perf/baselines/*.json`, golden data, never
   hand-edited — the same rule as `grxgpu/AGENTS.md` §4.
5. **No fabricated capability.** A GRXCP entry point either works or returns
   `grxErrorNotSupported`. Emulating a hardware feature in software behind
   an API that implies hardware is banned; it produces performance cliffs
   users cannot see. (The one sanctioned exception is the documented warp
   shuffle fallback, which is explicitly reported through
   `grxDeviceProp_t.warpShuffleIsEmulated`.)

---

## 11. Open questions

1. **`grxcc` frontend strategy** — a real Clang driver (`ToolChain` +
   `Action` graph, upstream-shaped) versus a Python/C++ orchestrator that
   shells out to clang twice. The former is the right end state; the latter
   is a two-week prototype. Recommendation: orchestrator in Phase 4, promote
   to a Clang `ToolChain` in Phase 6 once the flag surface has stopped moving.
2. **`<<<>>>` without a custom frontend** — can Path A support it via a
   Clang plugin/AST rewriter, so Phase 1 users write real CUDA-shaped source
   rather than explicit `grxLaunchKernel` calls? Worth a spike; it changes
   how demo-able Phase 1 is.
3. **Warp shuffle ISA** — `WSHFL` as a new custom instruction (cost: RTL +
   SimX + parity case) versus generalizing `WGATHER` beyond 4 lanes. Needs
   an area/timing estimate from the GRX-G100 side before it can be scheduled.
4. **`__constant__` implementation** — dedicated 64 KB broadcast path (as
   the chip design lists) versus mapping to read-only global with a
   `__ldg`-style non-coherent load hint. The second is nearly free and may be
   enough.
5. **Stream concurrency ordering** — should GRXCP ship Phase 5 (QMD launch +
   multi-queue, requiring GRX-G100 hardware work) before or after the
   libraries? Libraries make the platform useful; concurrency makes it fast.
6. **rv32 support in GRXCP** — the GRX-G100 stack supports both. Carrying
   rv32 doubles the test matrix for a configuration no flagship target uses.
   Recommendation: rv64 only for v1, rv32 kept compiling but untested.

---

## 12. References

- CUDA architecture reference: <https://ajdillhoff.github.io/notes/cuda_architecture/>
- `grxgpu/docs/designs/gpu_chip_design.md` — G100 chip design, CUDA-style integration
- `grxgpu/docs/designs/command_processor.md` — CP control plane, runtime submit path
- `grxgpu/docs/designs/vortex_runtime_api.md` — `vortex2.h` shape lock
- `grxgpu/docs/designs/kernel_entry_and_dispatch.md` — `.vxbin` / `VXSYMTAB`
- `grxgpu/docs/designs/cta_clustering_and_dispatch.md` — KMU grid walk, LMEM slots
- `grxgpu/docs/designs/tensor_core_wgmma_engine.md` — TCU / WGMMA
- `grxgpu/docs/designs/dxa_async_copy_multicast.md` — async copy / multicast
- `grxgpu/docs/designs/hip_on_vortex_chipstar.md` — the SPIR-V path and its limits
- `grx930/c930/doc/c930_architecture.md` — GRX930 SoC and systolic NPU
