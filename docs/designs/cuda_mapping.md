# CUDA → GRXCP Concept Mapping and Gap Register

**Status:** Reference tables for the GRXCP design.
**Purpose:** every CUDA concept, mapped to the GRX-G100 mechanism that backs
it, with an honest status. Nothing in these tables is aspirational unless the
Status column says so.
**Companion:** [`grxcp_architecture.md`](grxcp_architecture.md),
[`grxcp_roadmap.md`](grxcp_roadmap.md).

Status legend:

| Mark | Meaning |
|---|---|
| **HW** | The hardware mechanism exists in GRX-G100 today |
| **SW** | Exists in GRX-G100 software (`vortex2.h`, `vx_*` headers) today |
| **NEW** | GRXCP must build it; no hardware or driver change needed |
| **DRV** | Needs GRX-G100 driver/CP work (tracked in that repo's open items) |
| **ISA** | Needs a hardware/ISA change |
| **OUT** | Deliberately out of scope for v1 |

---

## 1. Execution model

| CUDA | GRX-G100 mechanism | Status |
|---|---|---|
| thread | lane within a warp | HW |
| warp (32 threads) | warp, `VX_CFG_NUM_THREADS` (32 in the flagship config) | HW |
| thread block / CTA | CTA, admitted to a fixed-stride LMEM slot by `VX_cta_dispatch` | HW |
| thread block cluster (Hopper) | CTA cluster — `CLUSTER_DIM_{X,Y,Z}` DCRs, KMU two-level grid walk, K consecutive LMEM slots | HW |
| grid | KMU grid walk | HW |
| SM | core (`VX_CFG_NUM_CORES` × `VX_CFG_NUM_CLUSTERS`) | HW |
| GPC | cluster | HW |
| warp scheduler | `VX_scheduler`, one per SM partition | HW |
| SIMT divergence / reconvergence | IPDOM stack: `SPLIT` / `JOIN` / `PRED` (`vx_split`, `vx_join`, `vx_pred`) | HW+SW |
| zero-overhead warp switching | resident warp state in hardware registers | HW |
| `threadIdx` / `blockIdx` / `blockDim` / `gridDim` | `VX_CSR_CTA_THREAD_ID_{X,Y,Z}` (0xCD3–5), `CTA_BLOCK_ID_*` (0xCD6–8), `CTA_BLOCK_DIM_*` (0xCD9–B), `CTA_GRID_DIM_*` (0xCDC–E), read through `vx_spawn2.h` | HW+SW |
| `warpSize` | `VX_CAPS_NUM_THREADS` | SW |
| `__syncthreads()` | `vx_barrier(cta_id, num_warps)` → `BAR` instruction | HW+SW |
| `__syncwarp()` | implicit within a warp (lockstep SIMT); compiler fence only | HW |
| `this_grid().sync()` (cooperative) | `VX_gbar_unit` global barrier, `vortex::gbarrier` | HW+SW |
| `cuda::barrier` / mbarrier with transaction count | `vx_barrier_arrive` / `_wait` / `_expect_tx` | HW+SW |
| launch bounds / occupancy | fixed-stride slot allocator; `usable_slots = min(NUM_WARPS, ⌊LMEM/stride⌋)` | HW |
| dynamic parallelism (device-side launch) | no device-side KMU programming path | OUT |
| CUDA graphs | — | OUT (v2; `vx_enqueue_commands` batching is the natural substrate) |

---

## 2. Memory model

| CUDA | GRX-G100 mechanism | Status |
|---|---|---|
| registers | per-thread register file, 4 GPR + 2 VGPR banks | HW |
| `__shared__` | LMEM carve-out; base from `VX_CSR_CTA_LMEM_ADDR` (0xCDF) | HW+SW |
| dynamic `__shared__` (extern) | `vx_launch_info_t.lmem_size` → per-kernel `aligned_lmem_size` stride | SW |
| distributed shared memory (cluster) | consecutive fixed-stride slots; peer base = `base + rank·stride`; DXA multicast writes | HW |
| global memory | HBM/device DRAM via L2 | HW |
| local memory (spill) | per-thread spill region | HW |
| `__constant__` | **no exposed broadcast constant path** | NEW/ISA — §7.2 |
| texture / surface | TEX units + TCACHE (graphics path; not exposed to compute) | HW, NEW to expose |
| L1 / shared carve-out (`cudaFuncSetAttribute`) | unified L1/shared carve-out DCR | HW planned (`gpu_chip_design.md` §5.4, Phase 2 of that roadmap) |
| unified virtual addressing | single global VA space, SATP programmed at init | HW (VM builds) |
| managed / unified memory | SVM under the device MMU | HW where `VX_CAPS_VM_SUPPORT`; **not on FPGA paths** (CP §10 item 2) |
| pinned host memory | `VX_MEM_HOST` allocations reachable by the CP host master | SW |
| `cudaMemcpy` H2D/D2H/D2D | `CMD_MEM_WRITE` / `CMD_MEM_READ` / `CMD_MEM_COPY` via CP DMA | HW+SW |
| `cudaMemcpy2D/3D` | `vx_enqueue_{read,write,copy}_rect` | SW |
| `cudaMemset` | `vx_enqueue_fill_buffer` | SW |
| `atomicAdd` and friends | AMO unit at the LLC | HW |
| `cuda::atomic` scopes (block/device/system) | LMEM / L2-AMO / host-visible + `CMD_CACHE_FLUSH` | HW+SW |
| `__threadfence*()` | `vx_fence` | HW+SW |
| `cudaMemcpyAsync` (device-side, `cp.async`) | DXA async copy, `vx_dxa_issue_{1..5}d_wg` | HW+SW |
| TMA (bulk tensor async copy) | DXA multi-dimensional descriptors (up to 5D) + multicast | HW+SW |

---

## 3. Host runtime

| CUDA Runtime | GRXCP | Backed by |
|---|---|---|
| `cudaGetDeviceCount` | `grxGetDeviceCount` | `vx_device_count` |
| `cudaSetDevice` / `cudaGetDevice` | `grxSetDevice` / `grxGetDevice` | host-side table |
| `cudaGetDeviceProperties` | `grxGetDeviceProperties` | `vx_device_query` (all 18 caps IDs) |
| `cudaDeviceSynchronize` | `grxDeviceSynchronize` | `vx_queue_finish` on every queue |
| `cudaMalloc` / `cudaFree` | `grxMalloc` / `grxFree` | slab allocator over `vx_buffer_create` |
| `cudaMallocHost` | `grxMallocHost` | `vx_buffer_create(VX_MEM_HOST)` |
| `cudaMallocManaged` | `grxMallocManaged` | SVM; gated on `VX_CAPS_VM_SUPPORT` |
| `cudaMemcpy*` | `grxMemcpy*` | `vx_enqueue_read/write/copy(+_rect)` |
| `cudaMemset*` | `grxMemset*` | `vx_enqueue_fill_buffer` |
| `cudaHostRegister` | `grxHostRegister` | `vx_buffer_reserve` |
| `cudaPointerGetAttributes` | `grxPointerGetAttributes` | allocator interval map |
| `cudaStreamCreate*` | `grxStreamCreate*` | `vx_queue_create` |
| `cudaStreamSynchronize` | `grxStreamSynchronize` | `vx_queue_finish` |
| `cudaStreamWaitEvent` | `grxStreamWaitEvent` | `vx_enqueue_wait_value` |
| `cudaEventCreate/Record/Synchronize` | `grxEvent*` | `vx_event_create` / `vx_enqueue_signal` / `vx_event_wait_value` |
| `cudaEventElapsedTime` | `grxEventElapsedTime` | `vx_event_get_profiling` (DRV — CP §10 item 9) with host-clock fallback |
| `cudaLaunchKernel` | `grxLaunchKernel` | `vx_enqueue_launch` |
| `cudaLaunchKernelEx` (cluster, priority) | `grxLaunchKernelEx` | `vx_launch_info_t.cluster_dim`, queue priority |
| `cudaLaunchCooperativeKernel` | `grxLaunchCooperativeKernel` | global barrier legality check + launch |
| `cudaModuleLoad` / `cuModuleGetFunction` | `grxModuleLoad` / `grxModuleGetFunction` | `vx_module_load_file` / `vx_module_get_kernel` |
| `cudaFuncGetAttributes` | `grxFuncGetAttributes` | `vx_kernel_get_max_block_size`, `vx_kernel_address`; `numRegs` unknown until Phase 4 |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | `grxOccupancy…` | computed from the documented slot formula |
| `cudaGetLastError` / `cudaPeekAtLastError` | `grxGetLastError` / `grxPeekAtLastError` | thread-local sticky error |
| `cudaDeviceEnablePeerAccess` | `grxDeviceEnablePeerAccess` | returns `NotSupported` in v1 |
| `cudaGraph*` | — | OUT |
| IPC, MPS, multi-process | — | OUT |

---

## 4. Device-side language

| CUDA | GRXCP | Backed by |
|---|---|---|
| `__global__` | `__global__` | `annotate("vortex.kernel")` + `__vx_kentry_<name>` in `.vx_entry` |
| `__device__` | `__device__` | ordinary device-target function |
| `__host__ __device__` | same | dual-emission in `grxcc` |
| `__shared__` | `__shared__` | `__local_mem()` carve at `VX_CSR_CTA_LMEM_ADDR` |
| `__constant__` | `__constant__` | NEW — §7.2 |
| `__restrict__`, `__ldg` | same | load hints |
| `kernel<<<g,b,s,st>>>(…)` | same syntax | `__grxPushCallConfiguration` + stub → `grxLaunchKernel` |
| `printf` in kernel | `printf` | `vx_printf` |
| `assert` in kernel | `assert` | trap + `vx_printf` |
| `__launch_bounds__` | `__launch_bounds__` | metadata into the `.vxbin` footer (Phase 4) |
| `clock()` / `clock64()` | `grx::clock64()` | `vx_rdcycle` |
| `__activemask()` | `__activemask()` | `vx_active_threads()` |
| `__ballot_sync(mask, p)` | same | `vx_active_threads()` under predication |
| `__any_sync` / `__all_sync` | same | derived from ballot |
| `__shfl_sync` and variants | same | **ISA gap — §7.1**; LMEM fallback in v1 |
| `__popc`, `__clz`, `__brev`, `__ffs` | same | RISC-V Zb* bit-manipulation |
| `__fmaf_rn`, fast-math intrinsics | same | FPU |
| `nvcuda::wmma::fragment` etc. | `grx::wmma::fragment` | `vortex::tensor::wmma_context<NT, …>` |
| warp-group MMA (`wgmma`) | `grx::wmma::wgmma_*` | `vortex::tensor::wgmma_context` |
| 2:4 structured sparsity | `grx::wmma::sparse_*` | TCU sparsity path (`VX_CFG_TCU_SPARSE_ENABLE`) |
| MX / block-scaled formats (mxfp8/mxfp4) | `grx::wmma` MX fragment types | TCU MX path — **ahead of CUDA's public surface here** |
| `cuda::pipeline`, `memcpy_async` | `grx::pipeline`, `grx::memcpy_async` | `vx_dxa_issue_*_wg` + `vx_barrier_expect_tx` |
| cooperative groups (`tiled_partition`, `this_grid`) | `grx::cg` | thread mask + `vx_gbar` |
| ray tracing intrinsics | `grx::rt` | RTU (`vx_raytrace.h`) — no CUDA analogue in the runtime API; OptiX-shaped, out of v1 |

---

## 5. Compilation

| CUDA | GRXCP | Notes |
|---|---|---|
| `nvcc` | `grxcc` | host/device split driver; Phase 4 |
| PTX (virtual ISA) | SPIR-V (Path A) / LLVM IR (Path B) | the fat binary can carry either as a JIT fallback |
| SASS (machine ISA) | RISC-V + Vortex SIMT extensions | `TMC`/`WSPAWN`/`SPLIT`/`JOIN`/`PRED`/`BAR`, TCU, DXA |
| cubin | `.vxbin` | `[min_vma][max_vma][image][VXSYMTAB]` |
| fatbin | `.grxfat` in `.grxfatbin` ELF section | multi-entry: xlen × isa_flags |
| `__cudaRegisterFatBinary` / `RegisterFunction` | `__grxRegisterFatBinary` / `__grxRegisterFunction` | static-init constructors |
| `-gencode arch=compute_90` | `-grx-arch=g100`, ISA-flag matching | matches `VX_CAPS_ISA_FLAGS` at load |
| `nvdisasm` | `llvm-objdump` on the `.vxbin` ELF | already works |
| NVRTC (runtime compilation) | `grxrtc` | Phase 6; POCL already JITs SPIR-V today |

---

## 6. Libraries and tools

| CUDA | GRXCP | Notes |
|---|---|---|
| cuBLAS | grxBLAS | TCU + DXA staging; also the NPU's entry point |
| cuDNN | grxDNN | implicit-GEMM conv, attention |
| cuFFT | grxFFT | |
| cuRAND | grxRAND | Philox/XORWOW, no HW dependency |
| cuSPARSE | grxSPARSE | LLC AMO for scatter |
| Thrust / CUB | `grx::par` | header-only |
| `nvidia-smi` | `grx-smi` | |
| Nsight Compute / Systems | `grx-prof` | MPM counters + Perfetto + roofline already exist |
| compute-sanitizer | `grx-sanitize` | SimX instrumentation — cheaper here than on silicon |
| `cuda-gdb` | `grx-gdb` | existing GDB/OpenOCD kernel-debug path |
| `hipify` | `grxify` | source translator |

---

## 7. Gap register

The honest list. Each gap says who must fix it and what it blocks.

### 7.1 General warp shuffle — **CLOSED: the instructions were already there**

This entry used to open "highest impact" and describe `__shfl_sync` as staged
through local memory at roughly an order of magnitude the cost of a register
shuffle, with a proposed WSHFL ISA extension as the fix and
`grxDeviceProp_t.warpShuffleIsEmulated = 1` reporting the state.

That is no longer true, and the reason is worth recording. The ISA has the
instructions and has had them for a while: `SHFL.UP`, `SHFL.DOWN`, `SHFL.BFLY`,
`SHFL.IDX` and `VOTE.ALL / ANY / UNI / BAL`, exposed in `vx_intrinsics.h` with
no `VX_CFG` gate and implemented in the SimX ALU. GRXCP was emulating something
the hardware does in one instruction, and saying so in its device properties —
honest about GRXCP, wrong about the device. Nobody had looked since the header
was written.

`grx_warp.h` now issues them directly and the emulation is deleted rather than
kept as a fallback. The SHFL instructions implement NVIDIA's segmented
semantics exactly — a control word carrying a clamp and a segment mask, with
out-of-segment lanes keeping their own value — so CUDA's `width` argument maps
onto them arithmetically:

    seg_mask = ~(width - 1)      minLane = lane & seg_mask
    clamp    = width - 1         maxLane = minLane | (clamp & ~seg_mask)

`tests/kernels/warp/` checks all four forms against CUDA semantics at two
segment widths on a real device, plus the vote family and a kernel that
shuffles beside its own shared memory. `warpShuffleIsEmulated` reports 0, and
`tests/unit/test_device_props.cpp` asserts it stays that way.

**The lesson, which outlives the entry.** A gap register is a claim about the
present, and it decays. This one cost the platform a documented "highest impact
hardware gap" that had already been closed upstream. Anything in this file
describing a missing capability should be re-checked against the current
sysroot before it is planned around.

### 7.2 `__constant__` memory — **NEW / possibly ISA**

No exposed broadcast-constant path. The chip design lists a 64 KB constant
space with a constant cache (`gpu_chip_design.md` §7.1) but nothing
implements or exposes it.

- **v1 mitigation:** `__constant__` lowers to read-only global memory with a
  non-coherent load hint. Semantically correct; loses the broadcast
  bandwidth advantage.
- **Fix:** either a real constant cache path, or accept the mitigation
  permanently and document it. The mitigation is nearly free and may simply
  be the right answer — see architecture doc §11 open question 4.

### 7.3 Stream concurrency — **DRV**

Streams are semantically correct from Phase 1 but physically serialized: CP
`NUM_QUEUES` defaults to 1, the emulation CP models only `q0_`, the runtime
serializes launches, and real per-queue concurrency depends on the QMD-style
atomic `CMD_LAUNCH` replacing today's ~18-`CMD_DCR_WRITE` launch dance
(`command_processor.md` §10 items 5–6).

- **Blocks:** copy/compute overlap, multi-stream throughput — the reason
  most CUDA code uses streams at all.
- **Fix:** GRX-G100 side. GRXCP's Phase 5 tracks and consumes it.

### 7.4 Device-side event timing — **DRV**

`vx_event_get_profiling` works: the driver stamps queued/submit/start/end for
every command, and `grxEventElapsedTime` uses those rather than the host
timestamps it captures at record, because they bracket execution instead of
submission.

They are still **host** timestamps, taken in the driver's queue worker either
side of the call that runs the command. The command processor writes back no
device timestamps, so `eventTimingIsDeviceSide` stays 0 — and on a simulator
the distinction is the whole story: `end - start` measures how long **SimX**
took, which says nothing about how long the device would take. `grx-smi` and
`grx-conform` say so on simulator backends rather than leaving the number to
be misread.

A kernel that needs real device time reads the device's own cycle counter,
`grx::clock64()`, and writes it out. Any performance claim in this project has
to come from that, or from the simulator's own cycle statistics — never from
event elapsed time on a simulator.

### 7.5 Managed memory on FPGA paths — **DRV**

VM works on simx/rtlsim/gem5 and silently no-ops on FPGA (no `CP_SATP` decode,
no hardware page-table walker in `VX_cp_dma`, §10 item 2). GRXCP returns
`grxErrorNotSupported` for `grxMallocManaged` on those backends rather than
producing a pointer that quietly means something different.

### 7.6 Byte-exact DMA on FPGA — **DRV, correctness**

`VX_cp_dma` rounds transfers up to a 64-byte multiple with `wstrb` all-ones,
so a non-cache-line-aligned `grxMemcpy` can over-write neighbouring bytes on
FPGA (§10 item 1). Until the tail-`wstrb` fix lands, GRXCP's allocator
**pads every allocation to a 64-byte boundary and never sub-allocates within
a line** on FPGA backends. This costs a little memory and removes a
silent-corruption class.

### 7.7 Register-pressure occupancy — **NEW (compiler)**

Occupancy today is bounded by warps, slots, and shared memory — not
registers, because `VX_cta_dispatch` does not gate admission on register
count. `grxFuncGetAttributes.numRegs` returns -1 until `grxcc` emits
per-kernel register metadata into the `.vxbin` footer. CUDA code that tunes
against `numRegs` will need to handle the unknown value.

### 7.8 Texture/surface from compute — **NEW**

TEX units and TCACHE exist and are driven by the graphics path. Nothing
exposes them to compute kernels. `grx::tex<>` is a Phase 6 item; CUDA
texture-object code does not port until then.

### 7.9 WMMA fragment shape — **HW, structural**

`nvcuda::wmma` fixes fragments at 16x16x16 (plus 32x8x16 and 8x32x16), and
every CUDA GPU with tensor cores has those shapes. GRX-G100's tile is
**derived** from the build — warp width, registers per fragment, input element
width — so it is neither 16x16x16 nor constant across configurations. The
configuration this was developed against gives **8 x 4 x 8** for fp16 in /
fp32 out; a wider warp gives something else again.

There is no honest way to paper over this at the fragment level. Emulating a
16x16x16 fragment on an 8x4x8 tile means loading four times the data per
fragment and issuing eight MMA steps behind the caller's back — a reasonable
thing for a *library* to do, and a bad thing for a fragment *type* to do,
because the register budget the caller is reasoning about silently multiplies.

So `grx::wmma::fragment` **checks** the shape it is declared with and refuses
to compile when it is not the shape this build provides. A ported kernel that
hardcodes 16 has to be edited; a kernel written against
`grx::wmma::tile<T>::m` / `::n` / `::k` follows the configuration by itself.
Absorbing the tiling difference is library-level work — grxBLAS — which is
where it belongs.

### 7.10 Static `__shared__` — **NEW (toolchain), was silently wrong**

CUDA's `__shared__ float tile[64];` needs the toolchain to carve a per-CTA slot
of local memory at link time. Nothing in this stack does that: the device link
script has no `.shared` output section, and the CTA's local-memory base is a
runtime value in `VX_CSR_CTA_LMEM_ADDR`, different for every CTA.

GRXCP used to define `__shared__` as `__attribute__((section(".shared")))`,
which compiled, ran, and put the array **in global memory** — the orphan
section landed in the ELF image. Nothing failed, because no kernel had used it
yet. The first one that did was the DXA gate, where the engine wrote to local
memory while the kernel read the symbol from the image; the tile came back as
the poison value it had been pre-filled with.

`__shared__` is now `__attribute__((unavailable(...)))`, so using it is a
compile error naming the alternative. Dynamic shared memory works and is the
supported route: `grx::shared_memory<T>()` returns this CTA's slot, sized by the
launch's `sharedMem` argument — that is CUDA's `extern __shared__`, and porting
a static declaration means moving it there.

### 7.11 Tensor maps are device slots, not values — **HW, structural**

CUDA's `CUtensorMap` is an opaque object the program owns and hands to a kernel
as an argument; each launch carries its own and nothing is shared. GRX-G100's
DXA descriptors live in the device, in `VX_DCR_DXA_DESC_COUNT` slots programmed
through config registers. Two consequences a port has to deal with:

* Slots are a **shared resource**. Two kernels needing different maps in one
  slot must not be in flight together, and no hardware will say otherwise.
  `grxTensorMapProgramAsync` is stream-ordered, so a single stream is
  predictable; two streams sharing a slot are racing.
* The engine's bus master **bypasses the MMU**, so a descriptor's base is a
  physical address. Where the device has virtual memory an ordinary allocation
  will not do, and `grxMallocPhysical` exists to produce one that will. GRXCP
  refuses the descriptor rather than programming an address the engine would
  misread.

Neither has a CUDA analogue, so neither can be hidden. Slot allocation is the
program's job, and it is visible in the API.

### 7.12 Tensor unit deadlocks on a second CTA — **DEVICE STACK, blocking**

`Core::issue` takes a CTA admission slot for **every** TCU micro-op that holds
the FU lock (guarded only by `VX_CFG_EXT_TCU_ENABLE`), while the matching
release lives inside `VX_CFG_TCU_WGMMA_ENABLE` and only fires for ops where
`tcu_is_wgmma()`. On the default configuration — tensor unit on, WGMMA off —
plain WMMA acquires the slot and nothing ever releases it, so the first CTA to
issue a tensor instruction owns the unit for the rest of the kernel and
`wgmma_cta_blocked()` stalls every other CTA at issue. One CTA is fine; two
never return.

Measured, not inferred: `tests/repro/tcu_multi_cta/` runs the same one-WMMA
kernel with a grid of one and a grid of two, in a child process under a
timeout. Tier 2 runs it as a **watch** — it reports whether the defect is still
present without hanging CI, and it will say so when it is fixed.

The fix is to make acquire and release symmetric. Until it lands, grxBLAS's
tensor GEMM is a **persistent single-CTA kernel**: one block, warps walking the
output tiles. On a one-SM configuration that costs nothing; anywhere else it is
a ceiling, and it comes out the day the watch turns green.

### 7.13 One device module at a time — **TOOLCHAIN**

Every `.vxbin` is linked at the same fixed load address (`STARTUP_ADDR`), so
loading a second module fails with an address overlap. CUDA programs routinely
hold several modules open; here a program gets one unless someone hand-assigns
link addresses.

That is why `src/libs/grxblas/kernels/all.cpp` exists: a library offering both
`grxblasSgemm` and `grxblasGemmEx` cannot ship them as separate modules. Found
by trying, when the second `grxModuleLoad` returned "address range overlaps
with existing allocation".

The real fix is relocatable device images. A cheaper one is a per-module link
address, which the toolchain already accepts as a `--defsym`.

### 7.14 DXA pads outer dimensions only — **HW / DOC, sharp edge**

A tile that overhangs the array is padded with the descriptor's fill value
along the **outer** dimensions, and **not** along dimension 0: there the engine
reads straight past `size0` into whatever memory follows. Measured in
`tests/kernels/dxa/`; the gate asserts both halves so the asymmetry cannot
change unnoticed.

It matters more than it looks. The tensor GEMM's ragged-`k` case is exact only
because `k` is an outer dimension of the **A** descriptor, so A's tail is
zeroed and every tail term is `0 * whatever-B-picked-up`. Put `k` in dimension
0 for both operands — which is what a transposed A would do — and the same
kernel silently starts accumulating garbage. It is one reason `grxblasGemmEx`
refuses transposes today rather than assuming they compose.

`grxTensorMapProgramAsync` sizes its bounds check for a full edge tile, so the
unchecked overhang cannot reach outside the caller's allocation.

### 7.15 Out of scope for v1

Dynamic parallelism (no device-side launch path), CUDA graphs, IPC handles,
MPS, multi-process service, `cudaHostAlloc` write-combining hints, and
peer-to-peer copies. Each is listed so a port that needs one gets a clear
"no" instead of a mysterious failure.

---

## 8. Where GRX-G100 is *ahead* of the reference

Worth recording, because the platform should expose these rather than
flatten itself to CUDA's surface:

1. **Timeline events.** `vx_event` is a monotonic counter with
   `wait_value`/`wait_values` — Vulkan timeline-semaphore shaped. CUDA's
   binary events are strictly weaker. `grx::event` exposes the timeline form
   natively; `grxEvent_t` is the compatibility narrowing.
2. **Transaction barriers with `expect_tx`.** `vx_barrier_expect_tx` is
   mbarrier-with-transaction-count, present today and directly usable for
   producer/consumer async-copy pipelines.
3. **Up to 5D DXA descriptors with multicast.** Broader than TMA's typical
   usage, and the cluster-contiguous LMEM contract makes multicast exact.
4. **MX / block-scaled tensor formats** (mxfp8, mxfp4) in the TCU config.
5. **A functional simulator as a first-class backend.** `grx-sanitize` and
   deterministic replay are cheap here and expensive on silicon.
6. **One command that expands a whole pass device-side** (`CMD_DRAW`, and
   the proposed `OP_DISPATCH`) — a natural substrate for a graphs-like API
   later.
