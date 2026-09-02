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
| `this_grid().sync()` (cooperative) | `VX_gbar_unit`, `vortex::gbarrier` — **cluster-scoped**, so grid-wide only on a single-cluster device | HW+SW — §7.17 |
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
| `atomicAdd` and friends | AMO unit at the LLC, **only where `VX_CFG_EXT_A_ENABLED`** | HW/ISA — §7.16 |
| `cuda::atomic` scopes (block/device/system) | LMEM / L2-AMO / host-visible + `CMD_CACHE_FLUSH` | HW+SW where the A extension is built in — §7.16 |
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
| `__shared__` (static) | `grx::shared_memory<T>()` + the launch's `sharedMem` | compile error as written — §7.10 |
| `__constant__` | `__constant__` | NEW — §7.2 |
| `__restrict__`, `__ldg` | same | load hints |
| `kernel<<<g,b,s,st>>>(…)` | same syntax | `__grxPushCallConfiguration` + stub → `grxLaunchKernel` |
| `printf` in kernel | `printf` | `vx_printf` |
| `assert` in kernel | `assert` | trap + `vx_printf` |
| `__launch_bounds__` | `__launch_bounds__` | metadata into the `.vxbin` footer (Phase 4) |
| `clock()` / `clock64()` | `grx::clock64()` | `vx_rdcycle` |
| `__activemask()` | `__activemask()` | `vx_active_threads()` |
| `__ballot_sync(mask, p)` | same | `vx_vote_ballot` |
| `__any_sync` / `__all_sync` | same | derived from ballot |
| `__shfl_sync` and variants | same | `SHFL.IDX/UP/DOWN/BFLY` — §7.1, closed |
| `__popc`, `__clz`, `__brev`, `__ffs` | same | RISC-V Zb* bit-manipulation |
| `__fmaf_rn`, fast-math intrinsics | same | FPU |
| `nvcuda::wmma::fragment` etc. | `grx::wmma::fragment` | `vortex::tensor::wmma_context<NT, …>` |
| `wmma::fragment<…, __half>` / int8 | `grx::wmma::fragment<…, half>` / `int8_t` with an `int32_t` accumulator | int8 is a build-time option (`VX_CFG_TCU_INT8_ENABLE`) and its tile is DEEPER in k — 8x4x16 where fp16 is 8x4x8 — see §7.19 |
| warp-group MMA (`wgmma`) | `grx::wmma::wgmma_*` | `vortex::tensor::wgmma_context` |
| 2:4 structured sparsity | `grx::wmma::sparse_*` | TCU sparsity path (`VX_CFG_TCU_SPARSE_ENABLE`) |
| MX / block-scaled formats (mxfp8/mxfp4) | `grx::wmma` MX fragment types | TCU MX path — **ahead of CUDA's public surface here** |
| `cuda::pipeline`, `memcpy_async` | `grx::pipeline`, `grx::memcpy_async` | `vx_dxa_issue_*_wg` + `vx_barrier_expect_tx` |
| `cooperative_groups::thread_block` | `grx::cg::thread_block` | `__syncthreads()`, CTA CSRs |
| `cooperative_groups::thread_block_tile<N>` | `grx::cg::thread_block_tile<N>` | segmented `SHFL`/`VOTE`; N must be ≤ the warp width — §7.9's problem, same answer |
| tile `reduce` / `inclusive_scan` / `exclusive_scan` | same | butterfly and Hillis-Steele over the shuffle |
| `cooperative_groups::coalesced_threads()` | `grx::cg::coalesced_threads()` | `vx_active_threads()` + popcount ranking |
| `cooperative_groups::cluster_group` | `grx::cg::cluster_group` | `vortex::group_barrier`; `map_shared_rank` needs an explicit stride — §7.18 |
| `cooperative_groups::grid_group` | `grx::cg::grid_group` | `vortex::gbarrier` — §7.17 |
| `cooperative_groups::labeled_partition` / `binary_partition` | — | OUT of v1 |
| `__match_any_sync` / `__match_all_sync` | — | no match instruction | OUT |
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
| cuBLAS | grxBLAS | level 1 (`Saxpy`, `Sscal`) and level 2 (`Sgemv`) are scalar kernels; `Sgemm` scalar; `GemmEx` on TCU + DXA staging; also the NPU's entry point |
| cuDNN | grxDNN | implicit-GEMM conv, attention |
| cuFFT | grxFFT | |
| cuRAND | grxRAND | Philox/XORWOW, no HW dependency |
| cuSPARSE | grxSPARSE | LLC AMO for scatter |
| Thrust / CUB | `grx::par` | header-only |
| `nvidia-smi` | `grx-smi` | |
| Nsight Compute / Systems | `grx-prof` | v1 ships: MPM counters, occupancy, Chrome/Perfetto trace — see [`grx_prof.md`](grx_prof.md); roofline still open |
| compute-sanitizer | `grx-sanitize` | v1 ships: outlined ASan callbacks + the allocator's own map — see [`grx_sanitize.md`](grx_sanitize.md) |
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

**Now measured rather than asserted.** The sentence above gated an entire phase
and nobody had re-checked it since it was written, so
`tests/repro/stream_overlap/` checks it directly. Two kernels rendezvous through
a device global: a `waiter` spins on it with a bounded budget, a `setter` writes
it, one per stream, waiter enqueued first.

The iteration number is the whole answer, and getting that right took two tries:

| result | meaning |
|---|---|
| saw it at iteration **> 0** | **overlap** — the waiter was already spinning when the setter ran |
| never saw it | **serialized** for that run — the waiter ran its whole budget and the setter still had not run |
| saw it at iteration **0** | **inconclusive** — the flag was set on the first read, so the setter finished *before* the waiter began |

The third case is about a third of runs, and a first version that counted it as
overlap reported overlap half the time on a device that has none. Every sighting
was at iteration 0 exactly — never 500, never 1200 — which is what identifies it
as reordering rather than concurrency.

**Result on simx: serialized, with no ordering between independent streams.**
Across trials, zero overlapped and the rest either ran the full budget or were
reordered. Both readings are consistent, and the second is not a defect: CUDA
promises no ordering between independent streams either. It does mean a naive
overlap test is flaky, which is why the repro runs several trials and concludes
from the set.

The repro carries a control for the *detector*, not just for the device: the
waiter can set the flag itself halfway through its own spin, and must then
report that iteration. Without it, "never saw it" could equally mean the
measurement cannot see a mid-spin sighting at all. That control had to be run
with the setter **not launched**, because otherwise the same reordering race
sets the flag before the waiter's first read — which is exactly how its first
version failed.

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

**And that counter has a scope, which is one launch.** See 7.25.

### 7.25 `VX_CSR_MCYCLE` restarts at zero at every launch — **SIM, silent**

The device's own cycle counter is the only clock that measures the device, and
its readings are comparable **within one launch on one core**. Not across
launches: `ProcessorImpl::run()` opens with `reset()`, which assigns a fresh
`PerfStats`, and MCYCLE reads `PerfStats::cycles`
(`sim/simx/processor.cpp`, `sim/simx/csr_unit.cpp`). Three stages sampled from
very different points in a transformer block all report their first warp
starting at ~4900 cycles.

Why it is listed as a gap rather than as a fact about simulators: nothing about
the numbers says so. Two launches produce two sets of small overlapping
timestamps, and `end - start` across them is a maximum over unrelated clocks
that has the shape, the units and the plausibility of a duration. The
cross-*core* case has been refused since `grx_cycles.h` was written; this one
had no detector at all.

It cost a real decision. `tests/bench/block_cycles.cpp` gave each of attention's
four launches its own region of one probe buffer and summarised the buffer,
believing the result was attention's cost. grxBLAS's sgemm kernel-selection rule
was then reverted on a 27.6% "regression" read off that number. Measured per
launch the sign flips: the reverted rule *saves* 5990 cycles on the block.

What catches it now is `grxCycleSummary::maxLive`, the greatest number of slots
live at once. A device holds `maxWarpsPerMultiProcessor × multiProcessorCount`
warps and not one more, so a buffer reporting more than that did not come from
one launch whatever its timestamps say — attention's reported 64 on a machine
that holds 16. The comparison is left to the caller, which is the one holding a
`grxDeviceProp_t`; `block_cycles.cpp` refuses such a span and prints why, and
`tests/unit/test_cycle_summary.cpp` gates the arithmetic in tier 1 where no
simulator is present at all.

This is a property of the SimX model and it has not been checked against
hardware or rtlsim, neither of which this project can run yet (7.1). On silicon
MCYCLE is a free-running counter and the question would not arise in this form —
but a probe buffer holding two launches would still be two launches, and the
`maxLive` check does not depend on which is true.

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

### 7.8 Texture/surface from compute — **NEW, and now emulated**

TEX units and TCACHE exist and are driven by the graphics path. Nothing exposes
them to compute kernels. **That has not changed.** What has changed is that
`grx::tex<>` now samples in SOFTWARE, out of ordinary global memory, in the
calling warp: `include/grx/device/grx_tex.h`, with the host side in
`include/grx/grx_texture.h`.

This is architecture section 10 rule 5's sanctioned exception and nothing
wider — an emulation reported through a device property, the way the warp
shuffle is. `grxDeviceProp_t.textureIsEmulated` reads **1**, `grx-conform`
prints it, and the TEXTURE GATE checks that it still does, so a green run that
quietly stopped reporting the emulation fails.

What a port gets: correct values. What it does not get is the performance shape
texture code is written for — a bilinear fetch is four global loads and the
arithmetic between them, issued by the caller's own warp, with no texture cache
and no free clamping. Two further differences keep the compat entries at
PARTIAL rather than MAPPED: filter weights are full-precision float where NVIDIA
quantizes the fraction to 8 bits, and the only channel formats are `float` and
`float4` — integer formats with normalized reads are not here, and a port that
needs them fails to compile rather than reading garbage.

The hardware path stays open. This entry closes when TEX is reachable from
compute, and `textureIsEmulated` is what will say so.

### 7.24 Rounding builtins do not survive divergent codegen — **TOOLCHAIN**

`__builtin_floorf` on a **divergent** value does not compile for this device:

```
error: unimplemented divergent codegen found!
```

Measured across the family, because which ones is the difference between a
workaround and a superstition:

| builtin | divergent codegen |
|---|---|
| `floorf`, `ceilf`, `truncf`, `roundf`, `rintf` | **fails** |
| `nearbyintf`, `fabsf`, `sqrtf` | compiles |

The five that fail are the ones lowering to a float→int→float sequence with an
**explicit** rounding mode; `nearbyintf` uses the dynamic mode, lowers
differently, and survives. Nothing in the tree had ever needed a rounding
builtin in divergent device code, so the software texture sampler is the first
thing that could have found it.

The workaround is a conversion and a compare — `grx::tex_detail::floor_f` — and
it is correct only over the range a texture coordinate occupies, which is why it
is not offered as a general `floorf`. A kernel that needs one of these five on a
divergent value has to write it out.

Filed against the device toolchain, not GRXCP. Nothing here can fix it.

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

### 7.10 Static `__shared__` — **CLOSED: grxcc carves the slot**

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

`__shared__` became `__attribute__((unavailable(...)))` — a compile error naming
the alternative, which was the honest replacement for a silent lie.

**That refusal is now lifted, because `grxcc` carves the slot.** The missing
piece was never a link-time section; it was somebody computing a per-kernel size
and a per-variable offset, and a source-to-source driver is in exactly the right
place to do it. `grxcc` collects a kernel's static declarations into one struct,
places it over the CTA's local-memory slot, and replaces each declaration with a
reference to a member:

```cpp
__shared__ float As[BLOCK][BLOCK];      // what the author wrote
auto& As = ((__grx_smem_k*)::grx::shared_memory<void>())->As;   // what compiles
```

The compiler computes every size and offset from the author's own declaration
text, so `float tile[TILE][TILE + 1]` works without `grxcc` knowing what `TILE`
is. `sizeof` the struct goes into the kernel descriptor's `static_smem`, which
`src/runtime/launch.cpp` has added to `lmem_size` since phase 1:

```cpp
info.lmem_size = (uint32_t)(shared + k.static_smem);
```

Nothing had ever set that field. Dynamic shared memory — `extern __shared__` —
lands after the static block, so a kernel with both gets both in that order.

Five of the eleven programs in `tests/cuda_samples` failed on this before it was
implemented, which is the measure of how central the construct is: a CUDA
tutorial reaches for a static `__shared__` tile in its second example.

**Still refused:** a `__shared__` inside a nested scope. `grxcc` hoists these
into one per-kernel block and cannot do that for a declaration whose scope is an
`if` or a loop body, so it is diagnosed rather than silently hoisted — silently
hoisting one would change its lifetime.

**Only through `grxcc`.** A kernel compiled by hand with `ci/build_kernel.sh`
sees the unavailable attribute, correctly: nothing in that path computes the
offsets.

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

**Turning `VX_CFG_TCU_WGMMA_ENABLE` on does not fix it**, and that was tested
rather than assumed: a sysroot rebuilt with the flag set still deadlocks. The
release has two conditions and the compile-time flag is only one of them — the
retiring op must also *be* a WGMMA op, which a kernel issuing plain WMMA never
is. Worth stating plainly, because the flag is the obvious thing to try and it
sends whoever fixes this to the wrong line.

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

**Across libraries it is worse, and the obvious fix does not finish the job.**
grxDNN made this bite between libraries rather than inside one: a transformer
layer calls grxBLAS for the GEMM and grxDNN for the norms, in one process, so
the second library to initialise fails to load its kernels. `src/libs/
kernels_all.cpp` answers that with one image carrying both libraries' entry
points — and with that image built and both libraries pointed at it, grxDNN
*still* failed:

```
Error: address range overlaps with existing allocation -
  requested=[0x180000000-0x180002000], existing=[0x180000000, 0x180002000]
```

The same file collides with itself. Each library calls `grxModuleLoad`
independently — neither knows the other exists — and the second call asks for a
range the first is already holding. Sharing the image removes the *different*
images; it does nothing about the second load of the *same* one.

So the runtime counts references: `load_module_tracked` compares the incoming
bytes against every module already resident on the device and, on a match, hands
back the resident one with its count raised; `grxModuleUnload` releases the
driver handle only when the count reaches zero. Identity is the full post-patch
image rather than the path, so a rebuild between two loads is correctly a
different module, and a sanitized load never shares with an unsanitized one.

Both halves are load-bearing and both have been watched failing.
`tests/libs/test_libs_together.cpp` is the gate; `ci/run_real.sh` runs it twice,
once against the shared image and once against a directory holding two separate
per-library images, where it must fail. Ablating the reference count in
`grxModuleUnload` fails exactly one case — the one where grxBLAS is destroyed
while grxDNN is still using the module — which is the failure a fix that unloads
to make room would ship with.

The real fix is still relocatable device images. A cheaper one is a per-module
link address, which the toolchain already accepts as a `--defsym`. The reference
counting stays useful either way: it is what `cuModuleLoad` does with a file
already loaded in the context.

### 7.14 DXA pads outer dimensions only — **HW / DOC, sharp edge**

A tile that overhangs the array is padded with the descriptor's fill value
along the **outer** dimensions, and **not** along dimension 0: there the engine
reads straight past `size0` into whatever memory follows. Measured in
`tests/kernels/dxa/`; the gate asserts both halves so the asymmetry cannot
change unnoticed.

It mattered more than it looked. The tensor GEMM's ragged-`k` case used to be
exact only because `k` was an outer dimension of the **A** descriptor, so A's
tail came back zeroed and every tail term was `0 * whatever-B-picked-up`. That
is an accident of one transpose combination, and transposing an operand moves
`k` between dimension 0 and the outer dimension:

| | A's `k` | B's `k` | tail product |
|---|---|---|---|
| NN | outer, zeroed | dim 0, garbage | `0 * garbage` |
| NT | outer, zeroed | outer, zeroed | `0 * 0` |
| TT | dim 0, garbage | outer, zeroed | `garbage * 0` |
| **TN** | dim 0, garbage | dim 0, garbage | **garbage** |

So `grxblasGemmEx` no longer inherits its correctness from whichever operand
happened to be padded: `kernels/hgemm_tcu.cpp` zeroes the staged tail itself on
the one step that can overhang. It costs a few stores out of `k_steps`, and it
makes all four combinations exact for their own reason. Verified in the
direction that matters — with the zeroing removed, exactly the TN ragged-`k`
cases fail and the other three stay correct, which is what the table predicts.

`grxTensorMapProgramAsync` sizes its bounds check for a full edge tile, so the
unchecked overhang cannot reach outside the caller's allocation.

### 7.15 Out of scope for v1

Dynamic parallelism (no device-side launch path), CUDA graphs, IPC handles,
MPS, multi-process service, `cudaHostAlloc` write-combining hints, and
peer-to-peer copies. Each is listed so a port that needs one gets a clear
"no" instead of a mysterious failure.


### 7.16 Atomics: the toolchain says yes, the hardware says no — **HW / TOOLCHAIN, sharp edge**

CUDA's `atomicAdd` family is unmapped, and the reason is worth stating
carefully because the failure mode is so unpleasant.

The device compiles `-march=rv64imafd`. The **a** in there is the RISC-V atomic
extension, so clang will lower a `std::atomic`, a `__atomic_fetch_add`, or an
`__sync_*` builtin to an AMO instruction without a word of complaint. A
GRX-G100 built with `VX_CFG_EXT_A_ENABLED` off — which is this configuration —
decodes that instruction, routes it to the LSU, and calls `std::abort()` in the
simulator. No message, no line, no trace: the process is gone, and nothing in
the stack has said the word "atomic".

That is exactly how grx-sanitize's first draft died. It counted findings with
`__atomic_fetch_add` on a device-memory word, and the first planted
out-of-bounds write aborted the simulator instead of being reported. The
sanitizer now indexes its report table by grid-linear thread instead — one slot
per thread, one writer per slot, no atomic anywhere.

Two consequences, both live:

- **`grxDeviceProp_t::capabilities` carries `GRX_CAP_GLOBAL_ATOMICS`,** set from
  the device's own `misa` A bit via `VX_CAPS_ISA_FLAGS`. `grx-smi` prints it as
  `atomics`. Anything that wants an atomic must ask, and on this configuration
  the answer is no.
- **A `grx_atomic.h` is not written yet, and when it is, it must `#error` when
  `VX_CFG_EXT_A_ENABLED` is 0** rather than emit an AMO the device will die on.
  A CAS loop over a hardware barrier is not a substitute worth shipping; a
  configuration with the extension enabled is.

The mismatch itself is fixable upstream — the kernel `-march` string should
follow `VX_CFG_EXT_A_ENABLED` rather than always claiming **a** — and until it
does, the compiler will keep offering a rope the hardware does not have.
### 7.17 The grid barrier is cluster-scoped — **HW, structural**

`this_grid().sync()` is a grid-wide rendezvous in CUDA. On GRX-G100 the
hardware barrier that implements it releases per **cluster**, and that is not a
detail a header can paper over.

The mechanism has two stages. A core's barrier unit waits until every one of
its active warps has arrived, then forwards a single arrival to its cluster.
The cluster releases when the arrivals equal the participant count — and there
is no stage above the cluster. `Cluster::global_barrier_arrive` masks the
arriving core id by `cores_per_cluster` and compares against a mask that only
has that many bits.

Three consequences, all live:

- **On a single-cluster device the barrier is exactly a grid barrier**, and
  `grx::cg::grid_group::sync()` is the real thing. The configuration this was
  developed against has one cluster, and `tests/kernels/cg/` exercises the
  barrier there with a control that fails without it.
- **On a multi-cluster device there is nothing to implement a grid barrier
  with.** `sync()` is `__attribute__((unavailable))` rather than a call that
  hangs, and the message points at `this_cluster()` or separate launches.
- **The participant count is cores per cluster, not cores.**
  `VX_CSR_NUM_CORES` reports `VX_CFG_NUM_CORES * VX_CFG_NUM_CLUSTERS` — the
  device total — and passing it to `vortex::gbarrier` waits for arrivals the
  cluster's mask cannot represent. The compile-time `VX_CFG_NUM_CORES` is the
  per-cluster figure and the correct one.

A cooperative launch must therefore put at least one block on **every** core:
a core with no active warps never forwards an arrival and the cluster waits
forever. `grxLaunchCooperativeKernel` and `grxLaunchCooperativeFunction` refuse
a grid smaller than the machine as well as one too large to be resident —
which is worth knowing, because a grid smaller than the machine is exactly the
shape a first test tends to have.

### 7.18 `map_shared_rank` has no stride to work from — **HW/DOC**

A cluster's CTAs land in consecutive fixed-stride local-memory slots, so a
peer's shared-memory base is `base + delta * stride` and distributed shared
memory is genuinely available. The kernel just cannot compute it:
`VX_CSR_CTA_LMEM_ADDR` gives this CTA's base and there is no companion register
for the stride, which the dispatcher derives from the launch's `lmem_size`
rounded up to the cache-line granule.

So `grx::cg::cluster_group::map_shared_rank(addr, rank)` — the CUDA spelling —
is unavailable, and the three-argument form takes the stride the caller already
knows. Guessing it would read another CTA's memory silently, which is why the
argument is required rather than defaulted.

The fix is a one-register addition on the device side: expose the per-CTA LMEM
stride the way `VX_CSR_CTA_LMEM_ADDR` exposes the base.

### 7.19 The tensor unit has no bf16, and its type set is a build option — **HW / CONFIG**

The roadmap's phase 3 line says "GEMM (fp32/fp16/bf16/int8)". Two thirds of that
turned out to be wrong about this hardware, and the difference between the two
wrong parts matters.

**bf16 does not exist.** There is no `VX_CFG_TCU_BF16` knob to enable, in any
configuration — the tensor unit's type list is fp16, tf32, fp8, fp4, int8, int4
and the MX formats. bf16 is not switched off, it is absent. So `grxblas` has no
bf16 entry point and `grxblasTensorType_t` has no bit for one: a bit that is
always zero reads like a build option somebody forgot to turn on, which is the
opposite of the truth.

**int8 exists but is a build-time choice.** `VX_CFG_TCU_INT8_ENABLE` is off in
the sysroot this project builds against, along with tf32, fp8, fp4, int4,
WGMMA and the sparse variants. A different sysroot of the same hardware would
answer differently, which means no host-side table can be right.

So the answer comes from the device. `hgemm_tcu_shape` — the kernel that
already reports the WMMA tile shape, for exactly the same reason — also reports
the enabled type set, and `grxblasGetTensorTypes` returns it. `grxblasGemmEx`
refusing a type it cannot do now says which types the device *does* accept
rather than only "not supported", because a caller told "no" without being told
what "yes" would look like tends to conclude the whole tensor path is missing.

The consequence for the roadmap: int8 GEMM is implementable but needs a sysroot
built with the flag, and it cannot be gated until one exists. bf16 GEMM is not
implementable at all and has been struck rather than deferred.

### 7.20 `__syncthreads()` does not survive divergence — **TOOLCHAIN, silent deadlock**

`vx_barrier()` in `sw/kernel/include/vx_intrinsics.h` is

```c
__asm__ volatile (".insn r %0, 4, 0, x0, %1, %2" :: "i"(RISCV_CUSTOM0),
                  "r"(barried_id), "r"(num_warps) : "memory");
```

and `__syncthreads()` in `vx_spawn2.h` is that and nothing else. `volatile`
means the barrier may not be deleted and may not be reordered against other
volatile operations. It does **not** mean the barrier may not be **duplicated**.
LLVM spells that with the `convergent` and `noduplicate` function attributes,
and an asm statement carries neither.

So at `-O3` the optimizer tail-duplicates the block holding the barrier into
both arms of a preceding divergent branch. From

```c
if (i < n) s[t] = in[i];
__syncthreads();
if (i < n) out[i] = s[blockDim.x - 1 - t];
```

VOLT emits

```
vx_split_n a0, a7          # diverge on (i < n)
beqz  a7, .else
  ... ; vx_bar a5, a7      # copy 1
  j .join
.else:
  vx_bar a1, a2            # copy 2
.join:
vx_join a0
```

On a split/join reconvergence stack a **diverged** warp executes both arms, so
it arrives at the barrier **twice**. A CTA of two warps then posts three
arrivals against a barrier expecting two: the first two release, and the third
opens a generation nobody will ever join. The kernel hangs, with no error, no
diagnostic, and no wrong answer to notice.

**Why it stayed hidden for three phases.** The extra arrival needs a warp to
actually diverge *and* needs more than one warp per CTA — a single-warp CTA's
barrier is satisfied by its own first arrival. Every GRXCP kernel gate before
phase 4 launched one warp per CTA over a grid that divided evenly. The first
program `grxcc` ever compiled had a ragged tail and two warps per CTA, and hung.
The failure is data-dependent: the same kernel passes on `n = 48` and hangs on
`n = 43`.

**What GRXCP does.** `include/grx/device/grx_device.h` `#undef`s the upstream
`__syncthreads()` and redefines it over a wrapper marked
`convergent, noduplicate`; `grx_cg.h` routes `cluster_group::sync()` and
`grid_group::sync()` through the same wrapper for `vortex::group_barrier` and
`vortex::gbarrier`. Measured by counting `vx_bar` in the kernel above:

| attributes on the wrapper | `vx_bar` in the kernel |
|---|---|
| *(none)* | **2** — the bug |
| `convergent` | 1 |
| `noduplicate` | 1 |
| `noinline, convergent, noduplicate` | 1 |

Either attribute alone is sufficient on this compiler; both are kept because
which one a future optimizer respects is not a promise anyone has made.
`noinline` is not what makes it work — the wrapper is inlined anyway at `-O3`.

**This fixes GRXCP's kernels, not the tree.** Anything else calling
`vx_barrier`, `vx_barrier_arrive` or `vx_barrier_wait` through the upstream
headers has the same exposure, including `vortex::barrier` used directly.
`tests/repro/barrier_duplication/` runs both spellings — GRXCP's must pass, and
upstream's is expected to deadlock under a timeout — so CI reports the day it is
fixed. The upstream fix is the same three words on `vx_barrier` and friends, so
callers do not each have to know.

### 7.21 `numRegs` is a fact about the code, not an occupancy input — **SEMANTIC**

`grxFuncGetAttributes.numRegs` reported -1 from phase 1 to phase 4, because
nothing in the toolchain emitted a per-kernel register count: there is no
`ptxas -v` here to print one, and the `.vxbin` footer carries entry points and
nothing else.

It is now a number, measured rather than declared. `grxcc` disassembles the
device ELF it just built, walks each kernel's reachable call graph, and counts
distinct architectural registers. The definition is narrow, and worth stating
because a CUDA programmer will assume the CUDA one:

- Integer and floating-point registers are counted **together**. A GRX-G100
  thread has one file of each, and CUDA's single number has nowhere to put two.
- `x0` (`zero`) is excluded. It is a wire, not storage.
- The count covers the entry point and everything reachable from it by a
  **direct** call.
- An **indirect** call (`jalr`), or a direct call to a symbol not in the image,
  makes the count unknowable, and the kernel reports **-1** rather than a lower
  bound dressed up as a measurement.
- A module loaded from a `.vxbin` that `grxcc` did not build reports -1, because
  nobody measured it. The gate checks that too — the sentinel has to keep
  working after the number arrives, or "unmeasured" quietly becomes "zero".

**What it does not do.** On CUDA hardware, register count bounds occupancy: the
SM's register file is a shared budget and a hungry kernel fits fewer blocks.
Here it does not. `resident_blocks_per_sm` in `src/runtime/launch.cpp` bounds
occupancy by warp slots, CTA slots and shared memory, and deliberately has no
register term, because the CTA dispatcher does not gate admission on register
count. A register bound added to look familiar would report an occupancy the
hardware does not enforce.

The same asymmetry gives `__launch_bounds__` a split answer:

| argument | CUDA | GRXCP |
|---|---|---|
| `maxThreadsPerBlock` | caps the block, informs the compiler | **enforced** — a larger block is `grxErrorLaunchOutOfResources`, and `grxFuncGetAttributes` reports the bound |
| `minBlocksPerMultiprocessor` | asks the compiler to spill until N blocks fit | **nothing to do** — occupancy has no register term, so there is nothing to trade against |

`grxcc` emits a note on the second rather than accepting it silently, because an
author who wrote it is expecting a spill that will not happen. The maximum is
kept as the **source expression** the author wrote, not a parsed integer:
`__launch_bounds__(kBlock * 2)` is legal CUDA, `grxcc` has no constant
evaluator, and the host compiler does. Same principle as `grx_launch_shim.h`'s
`as_dim` overloads — hand the job to the compiler that already has the answer.

### 7.22 What a CUDA file gets without asking — **COMPAT SURFACE**

A `.cu` file writes `__shfl_down_sync`, `cooperative_groups::reduce`,
`atomicAdd`, `warpSize` and `fabsf` without including anything for them: the
CUDA frontend and `cuda_runtime.h` supply those names. `grxcc` has no frontend,
so it supplies them itself, and the list is not a convenience — it is the
difference between "compiles unmodified" and "compiles after you add four
includes".

The device pass includes `grx_device.h`, `grx_warp.h`, `grx_cg.h` and
`grx_atomic.h`. It does **not** include `grx_wmma.h`, `grx_pipeline.h` or
`grx_cycles.h`: those are GRX APIs with no CUDA spelling a source file would
already be using, so including them would only slow every device compile down.

Three things had to be built for this to hold, and each was found by a sample
rather than predicted:

**`grx_cuda_compat.h` includes `<math.h>`,** because `cuda_runtime.h` does. Three
samples call `fabsf` without including `<cmath>`, and they are right to.

**`<cooperative_groups.h>` and `<cooperative_groups/reduce.h>` exist,** as
forwarding headers onto `grx_cg.h` — which already ends with
`namespace cooperative_groups = grx::cg;`, so nothing is papered over and the
documented differences (tile width, grid-barrier scope, `map_shared_rank`) apply
unchanged. They are guarded to the device pass: `grx_cg.h` bottoms out in the
CTA CSRs, and the host half of the same file must still compile. The host gets
an empty `namespace cooperative_groups {}`, which is enough to make a file-scope
`namespace cg = cooperative_groups;` legal — the actual uses are inside kernel
bodies, which the host pass replaces with launch stubs. That is CUDA's
`__CUDA_ARCH__` fence with grxcc's spelling.

**`grx_device.h` includes `<cstdio>` and `<cassert>` before defining `printf`
and `assert` as macros.** The macros are what let a kernel call them; they also
poison the standard headers if those are parsed afterwards:

```
cstdio:127:11: error: no member named 'printf' in the global namespace
```

`grxcc` used to dodge that by choosing where to insert the header, and the dodge
worked only while `grxcc` controlled the order. It does not: a file writing
`#include <cooperative_groups.h>` above `#include <cstdio>` pulls the device
header in through the first and hits the poison on the second. Pulling the
standard headers in first makes the ordering irrelevant — a later `#include
<cstdio>` is a no-op, so there is nothing left to poison.

`warpSize` is an object with a single `operator int()`, not a macro. It is also
a member of `grxDeviceProp_t`, and a macro would rewrite `prop.warpSize` in any
translation unit that saw both. One conversion operator and not two, because
`threadIdx.x` is itself a struct with a user-defined conversion and two
candidates on the other side make `threadIdx.x % warpSize` ambiguous.

`__device__`-only functions are dropped from the host pass, the way `nvcc` drops
them. Without that, a device helper above the kernels reaches a host compiler
that has never heard of `warpSize` or `__shfl_down_sync`.

### 7.23 `cudaMemcpyToSymbol` works for `__constant__` and refuses `__device__` — **DRV, structural**

CUDA's `cudaMemcpyToSymbol(c_filter, taps, sizeof(taps))` takes the HOST address
of a `__constant__` variable and writes the device one. GRXCP now does that, and
the shape of what it does not do is forced by one measured fact.

**The driver gives the host no handle for a loaded module's memory.** After
`vx_module_load_bytes`, `vx_buffer_reserve` over any address inside the image
answers:

```
address range overlaps with existing allocation -
requested=[0x1800017e0-0x1800027e0], existing=[0x180000000, 0x180002000]
```

So there is no way to write a symbol where it lives. What the runtime does
instead is edit **its own copy of the image** — `grxcc` embeds the fat binary as
a `const` array, and `__grxRegisterFatBinary` copies it to a writable buffer for
exactly this reason — and then reload the module.

That divides the feature cleanly:

| | write | read back |
|---|---|---|
| `__constant__` | patch the image; reload if already loaded | **exact** — the device cannot write it, so the host copy is authoritative |
| `__device__` | patch the image; reload if already loaded | **refused** — a kernel can write it, and nothing here can see what it wrote |

Returning the host copy for a `__device__` symbol would answer with the value it
had before the kernel ran. That is a wrong answer rather than a missing feature,
so `grxMemcpyFromSymbol` returns `grxErrorNotSupported` and says why.

**`grxcc` supplies the link.** A declaration gives it a name; only the linker
knows the address and the size, so after the device compile `grxcc` reads them
out of the ELF with `llvm-nm` and registers them against the host stand-in's
address. A variable that no kernel references is dropped by `--gc-sections` and
is not in the table — `grxcc` warns, and `grxMemcpyToSymbol` on it reports
`grxErrorInvalidSymbol`, which is the truth: there is no device symbol.

**The cost is a module reload**, and it is worth knowing where it falls. A write
before the first launch is free — the module has not been loaded, so patching
the image is all there is to do, and that is where a CUDA program sets its
constants. A write afterwards releases every kernel handle from that image and
reloads it on the next launch. `tests/cuda_samples/12_constant_memory.cu` does
both, and the second was watched failing with the invalidation removed: the
first write still landed and the second silently did not.

`grxGetSymbolAddress` returns the link address, which is where the loader puts
it. It is usable as a kernel argument and **not** usable with `grxMemcpy`, which
refuses an address its allocation map does not own — and the map cannot own this
one, which is the same fact this section opened with.

This also gives §7.2's `__constant__` its point. The lowering is still read-only
global memory with no broadcast path, and `constantMemoryIsGlobal` still reports
that — but the variable is now reachable from the host, which is what made
`__constant__` worth writing in the first place.

---

### 7.26 The c930 NPU's `MAX_N` is a buffer size, not a hardware maximum — **HW / DOC, sharp edge**

The GRX930 team's architecture document now carries a grxcp integration guide,
and it tabulates the NPU's limits for this backend to build against. One row is
mislabelled, and it is the row that would have made us report a hardware limit
that does not exist:

| Parameter | SoC instantiates | the table's "NPU core max" | what the RTL does with it |
|---|---|---|---|
| MAX_M | 8 | 64 | sizes `c_mem[0:MAX_M*MAX_N-1]` |
| MAX_K | 16 | 256 | sizes `b_mem[0:MAX_K*MAX_N-1]` |
| **MAX_N** | **12** | **8** | sizes both — and 12 > 8 on its own row |

`MAX_N` appears three times in the NPU core: it sizes `b_mem`, it sizes `c_mem`,
and it is the bound in the legality check `(i_dim_n >= 1) && (i_dim_n <= MAX_N)`.
It is not a structural cap. The note beside it — that N is limited to one column
tile unless N-tiling is used — is wrong in both halves. N-tiling is not optional
in that core, it is unconditional (`num_n_tiles = (i_dim_n + NUM_COLS - 1) /
NUM_COLS`, `n_base = nt_reg * NUM_COLS`), and the SoC already instantiates
`MAX_N = 12` against `NUM_COLS = 4`, which is three column tiles and is exactly
the case the note says needs something extra.

So 8 is the core's DEFAULT, sitting in a column headed "max".

WHY IT IS IN THIS REGISTER AND NOT A BUG REPORT. AGENTS.md forbids fabricated
capability and requires every limit we publish to be one the hardware has. The
same guide asks — correctly — that this backend's decision logic be gated
against the register map rather than against a green simx run. A backend that
took that row at face value would cap N at 8 and report the cap through a device
property. That property would be fiction, and fiction sourced from a document is
the hardest kind to catch later.

NOT CONFIRMED ON HARDWARE. This is read off the RTL, not run: Verilator is not
installed here, so "N = 12 computes correctly" is unverified on this side. The
GRX930 checkout now ships a standalone NPU DPI model and a C++ DPI harness,
which is what could settle it. Until something runs, this entry says what the
source says and no more.

**ANSWERED, and the reading was right.** The GRX930 team's architecture document
now opens its limits section with "MAX_M, MAX_N, MAX_K are buffer sizes, not
hard computational limits" -- they size the on-chip A/B/C SRAM, and a runtime
dimension must be no larger only because the buffer is statically allocated. The
note claiming N was capped at one column tile unless N-tiling was used is gone.
They added BRAM cost formulas per buffer and an external tiling guide for
dimensions that exceed the parameters.

Two things follow for this backend, and they are bigger than the doc fix:

  * `NPU_C930_MAX_M/N/K` in `src/backends/npu_c930/npu_c930.h` are hardcoded to
    one SoC's synthesis defaults, 8 / 12 / 16. If they are buffer sizes chosen
    per FPGA, then a build against a different SoC has different limits and this
    header is wrong by construction rather than by drift. They should come from
    the device, and they are not reported through a device property at all
    today -- `populate_npu_properties` publishes no GEMM dimension limits, which
    AGENTS.md section 3 does not allow for a limit a caller can hit.
  * `npu_c930_gemm` REFUSES a GEMM that exceeds them. With the semantics settled
    that refusal is a missing feature rather than a correct guard: the caller's
    matrix is not too big for the hardware, it is too big for one invocation.
    Tiling it is the backend's job and the guide is now written.

Neither is done. Both are on the phase 7 list, and this entry is the record of
why they are not a doc fix.

STILL INCONSISTENT IN THE SAME TABLE, and worth one message rather than a gap:
the column is still headed "NPU core max" with MAX_N = 8 on a row whose SoC
column says 12. The paragraph above it now explains the semantics, so a careful
reader is no longer misled -- but the row still asserts a maximum smaller than a
shipped instantiation. The build-commands block a section later still carries
`# Full SoC Verilator model (needs DDR init fix)` in a section whose own status
table was updated to Complete.

Stale in the same guide, and noted here because it is the piece that would let
us run the check above: its RTLSIM table lists the Verilator DDR init issue as
open and prescribes replacing the 2-D banked array with a flat 1-D one. The
Verilator DDR stub already declares `logic [7:0] mem [0:MEM_BYTES-1]`, in a
commit made after the one that wrote the guide.


### 7.27 `sgemm_4x4` stops writing when it gains one more live pointer — **TOOLCHAIN, silent wrong answer**

Adding a fused bias to the sgemm epilogue -- one `uint64_t` in the argument
struct, one conditional load, one add per output -- makes `sgemm_4x4` produce
**nothing**. Its outputs come back holding the test's poison value, so the
kernel is not storing at all. `sgemm`, `sgemm_rb`, `sgemm_2d` and `sgemm_4x2`
are all correct with the identical change.

BISECTED BY SUBSTITUTION, because the obvious explanations are all wrong:

| what the epilogue gains | `sgemm_4x4` |
|---|---|
| one more float in the store expression, no load | **correct** |
| a load of a field that has been in the struct since ABI 1 (`stride_c`) | **breaks** |
| a load of the new field at offset 112 | **breaks** |
| the same field moved to offset 80 | **breaks** |
| the load hoisted to the top of the kernel, before any per-lane guard | **breaks** |

So it is not the new field, not the struct growing 112 -> 128 bytes, not the
offset, and not a uniform load under divergence -- both placements fail and an
existing field fails the same way. What separates the passing case from the
failing ones is one additional 64-bit live value and one more load from the
argument pointer.

WHAT IT IS NOT. `ci/check_kernel_loops.py` reports **zero stack traffic in every
kernel** in the failing build, and the k loops are byte-identical to the working
one -- 14, 23 and 35 instructions for 2d, 4x2 and 4x4. `sgemm_4x4`'s stack frame
SHRANK, 288 bytes to 256. If this were ordinary register pressure the frame
would grow and the census would name the spill; it does neither.

THE BOUNDARY IS THE ACCUMULATOR COUNT. `sgemm_2d` holds 4 accumulators and
`sgemm_4x2` holds 8; both take the change. `sgemm_4x4` holds 16 and does not.
All three are the same `micro_tile_body<RM, RN>` template, so the source is
identical and only the instantiation differs.

WHAT IT COSTS US. The fused bias is worth **5.9% of a transformer block at S=16
and 6.8% at S=8** -- the qkv projections apply their bias in six launches over
128 elements each, and a launch costs 2776 cycles before touching an element.
The fusion is written and correct on every kernel that ships; `sgemm_4x4` is
STAGED and the rule can never select it (7.26's neighbour, proved in
`tests/libs/test_grxblas_rb.cpp`). It is held back because the oracle forces
`sgemm_4x4` and a fusion that cannot be checked against the reference on every
kernel is a fusion shipping on an argument.

The work is not lost: the five changed files and this bisection are attached to
the session that found it.

### 7.28 The NPU device is enumerable but not usable: there is no NPU memory path — **OURS, CLOSED**

Found by trying to wire the GRX930 team's register model in, which is the first
thing that ever asked what an NPU device would do after it was enumerated.

`grxMalloc` → `allocate_device` → `vx_buffer_create(d->handle, …)`, and
`probe_npu_device` sets `d.handle = nullptr` because an NPU has no Vortex
handle. There is no `DeviceType::NPU` branch anywhere in `src/runtime/memory.cpp`
— not in `allocate_device`, not in `allocate_device_physical`, not in the
memcpy path. Measured directly against the installed driver:

```
vx_buffer_create(NULL, 64) -> 3     (VX_ERR_INVALID_VALUE)
```

So an allocation on an NPU device does not crash. It returns
`grxErrorInvalidValue` — *invalid value* — which blames the caller's size
argument for a device that simply has no allocator. That is the same class of
mistake this project already fixed once on the module-load path, where a
device with no pipeline was refusing with `grxErrorInvalidImage` and blaming
the binary.

The consequence is the one that matters: **`tests/libs/test_grxblas_npu.cpp`
could never have passed, hardware or not.** Its first act inside
`run_int8_case` is `grxMalloc`. Every one of its twenty-odd cases would fail at
the allocation, before reaching a single register. That test has been compiled
into every NPU build since it was written and has never executed a GEMM — it
skips at `no NPU device found`, which reads like an absent-hardware skip and is
actually hiding a missing subsystem behind it.

Two more, found in the same pass and blocking the same thing:

- **There is no seam to attach a register model to the *enumerated* device.**
  `probe_npu_device` news its own `npu_c930_device_t` and calls
  `npu_c930_detect` on it; `grxblas.cpp` has a *separate* file-static
  `g_npu_dev` and calls `npu_c930_detect` on that. Two handles, two detections,
  no injection point — so `npu_c930_attach_model`, which exists precisely so a
  model can stand in for hardware, cannot reach either of the devices the
  runtime actually uses. The four models in `test_npu_c930_model.cc` and the
  fifth in `test_npu_c930_shim.cc` all drive the backend directly, below the
  runtime, for this reason.
- **`populate_npu_properties` hardcodes `p.backend = GRX_BACKEND_SILICON`** and
  the name `"GRX930 NPU (silicon)"`, with the comment *"NPU is always real
  hardware"*. It is a claim about what the device is, made by a function with
  no way to know, in a codebase that has `GRX_BACKEND_SIMX` and
  `GRX_BACKEND_RTLSIM` for exactly this distinction. Today nothing can attach a
  model to the enumerated device, so the claim happens to hold; the moment the
  seam above exists, the same line reports a software GEMM model as silicon.
  The seam and this field have to land together — adding the first without the
  second builds the fabrication, and per `AGENTS.md` the backend field must be
  *derived* from how the device was reached, not asserted.

None of this is a GRX930 problem and none of it is fixed by their shim. The
shim makes the register half reachable (see `third_party/grx930/README.md`);
the memory half does not exist yet on our side.

**Two of the three are now closed.**

The seam is `src/backends/npu_c930/npu_c930_testing.h`:
`grxcp_npu_attach_model_for_testing()` installs a register model that
`probe_npu_device` detects through, and refuses after enumeration has run --
because a model installed late would flip the property below without changing
the device it describes, which is the fabrication the property exists to
prevent. `grxblas.cpp`'s file-static `npu_c930_device_t` is gone; it asks
`grxcp::npu_device_for(index)` for the handle the device table owns, so the
GEMM now runs on the device `decide_gemm_engine` made its decision about rather
than on a second one detected behind its back.

The backend field is derived. `GRX_BACKEND_MODEL` is appended to `grxBackend_t`
-- a software register model is neither `SIMX` (the Vortex functional
simulator) nor `RTLSIM` (which executes the RTL) nor `SILICON`, because it
executes nothing of the device at all -- and `populate_npu_properties` reads
the same variable `probe_npu_device` detected through. The device also names
itself: `"GRX930 NPU (software register model, NOT hardware)"`, where `grx-smi`
prints it. Watched failing: restoring the old unconditional
`p.backend = GRX_BACKEND_SILICON` turns three checks in
`tests/unit/test_npu_enumerated_model.cpp` red, with the device reporting
backend 5 and the name "(silicon)" while running on a C register file.

That test is also the first thing in this project to reach an NPU device
through the public API. It measures the remaining hole rather than asserting
it, and the reading is the one predicted above:

```
note  grxMalloc(256) -> grxErrorInvalidValue (invalid argument)
      totalGlobalMem is 65536 bytes and nothing allocates from it.
```

**The allocator is done too.** `allocate_device` gets a `DeviceType::NPU`
branch that carves the SoC's fixed 64 KB window with the same best-fit free
list every other device uses -- the window is registered as a slab with a null
`vx_buffer_h`, so `grxFree`, the interval map, `lookup_device_pointer` and the
sanitizer all work on it unchanged. There is no direct tier: that tier exists
so a large allocation can get its own driver buffer, and there is no driver
here, so a request the window cannot satisfy is out of memory rather than a
reason to ask for a second window.

**The first aligned block is reserved and never handed out.** A device pointer
on the NPU *is* the DDR byte offset -- no MMU, and `A_BASE` takes it literally
-- so an allocation at offset 0 would be indistinguishable from `grxMalloc`
failing, because null is already the failure signal. The GRX930 team hit the
same wall from the other side and fixed it with an out-of-band sentinel
(`NPU_DDR_ALLOC_FAILED`); we cannot, because the return type is `void*`, so the
fix is structural. It costs one alignment unit of 65536 and removes the
ambiguity rather than moving it somewhere a caller has to remember. Watched
failing: setting `kNpuReservedBase = 0` puts the first allocation at offset 0
and the null check goes red.

A device reset frees the window rather than taking it away —
`release_all_allocations` re-inserts it whole, because "every allocation on
this device is gone" means something different for a fixed window than for a
driver buffer. `grxMallocPhysical` uses the same window (there is one address
space and `VX_MEM_PHYS` has no meaning without an MMU); `grxMallocHost` is
refused by name, since pinning is a driver mapping and there is no driver.

Measured through the seam: first allocation at offset 256, three live
allocations non-overlapping and 4-byte aligned inside the window, a request for
twice `totalGlobalMem` refused as `grxErrorMemoryAllocation` rather than
`grxErrorInvalidValue`, a freed extent reused at the same offset, and after
freeing everything a single allocation of nearly the whole window fits — which
is only true if the extents coalesced.

**`grxMemcpy` closes it.** The register hooks covered the control path and said
nothing about the data path, which had the same problem for the same reason. So
`npu_c930.h` grows a matching pair — `npu_c930_mem_read_fn` /
`npu_c930_mem_write_fn`, attached with `npu_c930_attach_memory` — and
`enqueue_copy` gets an NPU branch that copies synchronously through them,
before it reaches `resolve_stream` (there is no queue) or the three
`vx_enqueue_*` calls (there are no buffer handles; an NPU allocation's address
*is* its DDR offset). Device-to-device goes through a bounce buffer rather than
hook-to-hook, because the two extents may overlap.

**A device with no memory hooks is refused**, and that is every device today
except one with a model attached: the hardware path would be an mmap of the DDR
aperture or a bounce through the AXI DMA, and neither is written. Refusing is
the only honest answer — an accepted `memcpy` that moves no bytes leaves the
caller reading whatever was in the buffer.

With that, `tests/libs/test_grxblas_npu.cpp` executes for the first time. It
found two bugs in the NPU path (7.32) and two in itself (7.33) in the first
minute, which is the return on making an unreachable path reachable.

### 7.29 Truncating a host pointer into a 32-bit base register fails silently, and not always out of range — **OURS, sharp edge**

`grxblasGemmEx` refuses any A/B/C above `0xFFFFFFFF` before casting, because the
NPU's `A_BASE`/`B_BASE`/`C_BASE` are 32-bit MMIO words. That refusal is right,
and on a 64-bit host it refuses *every* real allocation — which is worth stating
plainly, because it means the NPU GEMM path is unreachable from the public API
on any host we have.

The sharp edge is what happens when something below that guard truncates
anyway. A truncated pointer is not reliably an out-of-range address; it is an
**arbitrary** one. Measured, same machine, same program:

| allocator | pointer | low 32 bits | inside the c930's 64 KB DDR window? |
|---|---|---|---|
| glibc `malloc` | `0x55ef9bfc62a0` | `0x9bfc62a0` | no |
| ASAN `malloc` | `0x506000000020` | `0x00000020` | **yes — offset 32** |

An address decoder can refuse the first. It cannot refuse the second: 32 is a
perfectly ordinary DDR offset, so the DMA is aimed at a valid-looking location
that has nothing to do with the buffer, and the corruption is silent.

This was found the hard way. `test_npu_c930_shim.cc` originally truncated a live
`malloc()` and asserted the result was refused. It passed — under glibc, by
luck. Rebuilt under ASAN during a sabotage run, the same check went red because
the allocator returned a pointer whose low word landed *inside* the window. A
gate whose verdict depends on allocator internals is not a gate; both addresses
are now pinned as constants and both outcomes are asserted, the second one
precisely because it is the dangerous one.

### 7.30 An unrecognised backend printed as `silicon` — **OURS, fixed**

`grx-smi --json` rendered `grxDeviceProp_t.backend` with a chained ternary whose
last arm was `: "silicon"`. Every value that was not one of the five named
before it — including every value outside the enum — printed as silicon. The
field exists so a caller can tell a simulator from a chip, and the fallthrough
answered "chip" for anything it did not recognise.

`grxcp::backend_name` in `context.cpp` had this right the whole time: it is a
switch with a `return "unknown"` after it. There were two copies of one mapping
and only one of them was honest, which is how the other stayed wrong.

Found by measurement, not by reading, and by an outside proposal. The GRX930
team offered three constants for this field so that an attached software
register model would stop being reported as silicon —
`NPU_DPI_BACKEND_EMULATION` 0x10, `SIMULATION` 0x11, `SILICON` 0x00. Assigned
to `p.backend` against the code as it stood, all three made it worse:

| `p.backend = ...` | `grx-smi --json` said | `backend_has_vm` |
|---|---|---|
| `NPU_DPI_BACKEND_EMULATION` (0x10) | `"silicon"` | no |
| `NPU_DPI_BACKEND_SIMULATION` (0x11) | `"silicon"` | no |
| `NPU_DPI_BACKEND_SILICON` (0x00) | `"simx"` | **yes** |

The first two are the fabrication the constants were meant to prevent. The third
collides with `GRX_BACKEND_SIMX == 0` and switches on `backend_has_vm`, which
would advertise managed memory on a device with no MMU — re-opening 7.5.

The JSON arm is a switch with a `return "unknown"` now. The value collision is
not fixable from here and is not ours to fix: **a foreign header must not
assign values for another project's typed field.** What GRX930 can usefully
publish is the distinction — software model / RTL-backed / silicon — and the
mapping into `grxBackend_t` belongs in our code, next to the seam that attaches
the model. That mapping needs one new enum value (a software register model is
neither `SIMX` nor `RTLSIM` nor `SILICON`); their `SIMULATION` maps onto the
existing `GRX_BACKEND_RTLSIM`, and their `SILICON` onto `GRX_BACKEND_SILICON`.
Appending a value is ABI-safe; reusing an existing one would not be.

### 7.31 A guard that wraps is not a guard — the same shape, three times in one exchange — **CROSS-TEAM, pattern**

Recorded as a pattern rather than a defect, because it turned up three times in
four days in three different pieces of code, written by two teams, and each time
it took a run to see.

1. **7.29, ours.** `grxblasGemmEx` casts a device pointer into a 32-bit
   `A_BASE`/`B_BASE`/`C_BASE`. A truncated 64-bit pointer is not an out-of-range
   address, it is an *arbitrary* one: glibc gave `0x9bfc62a0` (outside a 64 KB
   window), ASAN gave `0x00000020` (offset 32, inside it, indistinguishable from
   a real allocation).
2. **The shim's extent check.** `a_end = A_BASE + m*k` compared against
   `NPU_DDR_SIZE`, in `uint32_t`. With `A_BASE = 0xFFFFFFF0` the sum wraps to
   `0x10` and the check passes. Watched: a byte write at `0xFFFFFFFF` landed on
   `ddr[0..2]` with no error raised; the GEMM path segfaulted under ASAN.
3. **The GRX930 allocator's fit test.** `if (padding + size > block->total)
   continue;` — `padding + size` in `uint32_t`. Measured against their
   `1c279ab`: after one small allocation leaves a free block at offset 10,
   `npu_ddr_alloc(&a, 0xFFFFFFFF, 4)` computes `2 + 0xFFFFFFFF = 1`, sails past
   the test, and **returns offset 12 for a four-gigabyte allocation out of a
   64 KB window.** Their `e02f460` refuses it.

Same shape every time: **a bounds check computed in the same width as the value
it is bounding.** `base + extent <= LIMIT` and `extent <= LIMIT - base` are the
same statement in arithmetic and different programs in C, and the first one is
the one everybody writes.

The rule this project takes from it: a bounds check is either done in a wider
type than the address, or written so no addition or subtraction can leave the
representable range — `base < LIMIT && extent <= LIMIT - base`. Our adapter's
`in_window()` does it in 64-bit and keeps doing it even now that the shim checks
too. Two guards for one hazard is not redundancy when they live in different
processes: the one that matters is the one inside the address space being
protected.

The corollary is about testing, not arithmetic. All three were found by running
a value near the edge of the type, and none by reading. A guard that has never
been shown a wrapping input has not been tested; it has been reviewed.

### 7.32 The NPU GEMM was column-major on one side and row-major on the other — **OURS, silent wrong answer, fixed**

Found in the first minute the NPU path ever ran.

`grxblas.h` opens with *"Shaped after cuBLAS, including its column-major
convention."* The c930 reads `A[i*K + p]` and `B[p*N + j]` and writes
`C[i*N + j]` — row-major, with the natural strides. `npu_gemm_path` checked
`lda == m && ldb == k && ldc == m`, which is the column-major contiguous case
and is right, and then passed the three pointers to the engine unchanged. Its
comment told the caller to "ensure the buffers are contiguous and row-major",
which a caller cannot do: the buffers came from a column-major API.

So the NPU path computed the right answer when `m == k == n` and a wrong one,
silently, otherwise. It had never been observed because it had never executed
(7.28).

**The fix costs nothing.** A column-major matrix read row-major *is* its
transpose, so

```
C(col-major, m x n) = A(m x k) . B(k x n)
```

is the same bytes as

```
C^T(row-major, n x m) = B^T(n x k) . A^T(k x m)
```

which is what the engine computes if it is handed B where it expects A, A where
it expects B, and the dimensions swapped. No copy, no transpose pass, no
staging buffer.

**The consequence has to be stated, because it is surprising.** The engine's
bounds now apply to the caller's dimensions *crossed*: the caller's `n` must
fit `MAX_M` (8) and the caller's `m` must fit `MAX_N` (12). The refusal message
says so by name rather than reporting a bare limit. `tests/libs/
test_grxblas_npu.cpp` had `MAX_M x MAX_N x MAX_K` as its "maximum dimensions"
case, which was written against the un-swapped path and is exactly backwards;
the largest legal shape is `m=MAX_N, n=MAX_M, k=MAX_K`.

Watched failing: removing the swap turns **15 of 16 numerical cases red**,
including the square ones, because the reference is column-major now too.

### 7.33 A test that had never run was also wrong — **OURS, fixed**

Two defects in `tests/libs/test_grxblas_npu.cpp` itself, both found by running
it for the first time, and both invisible for as long as it exited 77 before
reaching them. Recorded because the lesson is about the shape of the risk
rather than about either bug: **a test that compiles and never executes is
worse than a test that does not exist, because it looks like coverage.** This
one had a section list, deterministic fills, an exact-integer comparison and
twenty cases, and not one of them had ever been evaluated.

1. **It punned integers through a `float*`.** Every call passed
   `reinterpret_cast<const float*>(&alpha)` with `alpha` an `int32_t` — the
   cuBLAS convention, where the scalar's type follows `computeType`. Ours is
   not that API; `grxblas.h` says alpha and beta are floats in both cases and
   must hold exactly representable integers. So the library was handed the bit
   pattern of an integer and asked to read a float: `alpha = 1` arrives as
   1.4e-45, and the refusal path printed `alpha=0.0 beta=0.0` for a case that
   meant 0 and 1. Every GEMM would have been refused for a non-unit alpha it
   never had.
2. **Its reference used a layout that does not exist.** It indexed row-major,
   `A[i*lda + l]`, with `lda = m` — but row-major `A[m x k]` has row stride
   `k`. The two agree only when `m == k`, so the square cases would have passed
   and every other shape disagreed with the engine for a reason that had
   nothing to do with the engine. First run after the seam landed: `N=1` wrong
   in all four elements, `K=1` wrong in twelve of sixteen. Now column-major
   throughout, matching the API it is testing.

### 7.34 The shim's performance counters are not the RTL's — **CROSS-TEAM, do not quote them**

The GRX930 team's software shim is register-model-accurate and we have said so
repeatedly. Its **counters** are a different claim, and now that the same shapes
have been run against the RTL under Verilator, the two can be compared. At the
engine's M=4 N=4 K=8 INT8:

| register | shim | c930 RTL | |
|---|---|---|---|
| `CYCLE_LO` `0x24` | 22 | **224** | 10.2x |
| `OP_COUNT` `0x2c` | 256 | **1280** | 5x |
| `STALL_CT` `0x30` | 0 | **64** | the shim declares no stalls |
| `DMA_CT` `0x34` | 22 | **320** | 14.5x |

Three of the four disagree by an order of magnitude, and they do not merely
disagree in scale — they count different things. `OP_COUNT` in the shim is
`M*N*K*2`, a property of the problem; the RTL's rises with the tiling, not with
the element count (1x1x1 gives 160, 4x4x4 gives 640, 8x8x8 gives 5120). And the
RTL's `DMA_CT` **exceeds its own `CYCLE_LO`** on every shape measured (320 vs
224; 3077 vs 2592), which means the two counters do not share a time base.

The shim's own header is careful — "NOT cycle-accurate" — so this is not a
defect in it. It is a boundary worth naming, because the numbers are readable
through the same register at the same address and nothing at the call site says
which kind of device answered. **A cycle count read from a model is a model of
a model.** `grxDeviceProp_t.backend` is the discriminator, and it is derived
rather than asserted precisely so a caller can tell.

Nothing in grxcp reads these counters today. The rule this records is for when
something does: a performance number is only reportable alongside the backend it
came from, and `GRX_BACKEND_MODEL` numbers are not the device's.

**Both counters were then explained by the GRX930 team, and the explanations
were checked against the RTL rather than accepted. One held; one did not.**

`DMA_CT >= CYCLE_COUNT` is expected and correct. They are on different time
bases: `DMA_CT` starts at `CTRL.START` and spans the whole DMA phase (A/B load,
core compute, C writeback), while `CYCLE_COUNT` starts later, when the core
receives `i_start` from the DMA. True on all twenty shapes measured. Nothing to
do.

`OP_COUNT` was given as

```
OP_COUNT = ceil(M/NUM_ROWS) * ceil(N/NUM_COLS) * ceil(K/NUM_ROWS) * NUM_ROWS * NUM_COLS
```

— hardware PE firings, `NUM_ROWS x NUM_COLS` per `S_RUN` cycle. **It misses all
twenty shapes**, low by 10x to 40x. The measured relation is

```
OP_COUNT = 10 * M * ceil(N/NUM_COLS) * ceil(K/NUM_ROWS) * NUM_ROWS * NUM_COLS
```

which reproduces every one of the twenty exactly. Two differences: a factor of
10 (the `S_RUN` phase runs ten cycles per pass, not one), and **`M` is not
tiled** — it enters linearly where their formula has `ceil(M/NUM_ROWS)`. That is
the physically sensible shape for a systolic array: `N` and `K` are what the
array's dimensions tile, while `M` rows stream through it.

The method is worth recording as much as the result. A first fit over the
original twelve shapes gave `... * 10 * min(M, NUM_ROWS)`, which also matched
all twelve. It was **not** reported, because a fit to the points it was fitted
to is not evidence. Eight new shapes were chosen to break it -- `M = 3, 5, 6, 7`
straddle the tile boundary where `min()` and linear `M` diverge -- predicted in
advance, and then run:

| shape | first fit predicted | RTL measured |
|---|---|---|
| M=3 N=4 K=4 | 480 | 480 |
| M=5 N=4 K=4 | 1280 | **800** |
| M=6 N=4 K=4 | 1280 | **960** |
| M=7 N=8 K=4 | 2560 | **2240** |
| M=8 N=12 K=5 | 7680 | 7680 |

Three misses, all at a partial second tile, all in the direction that says `M`
is linear. The corrected form was then checked against all twenty and misses
none. **It is still a fit, not a reading of the RTL**, and it is reported to the
team that owns the design as a fit — the same standard we held them to when
their `CYCLE_COUNT` explanation turned out to be right and our guess about it
was wrong (7.28).

### 7.35 `grxblasGemmEx` runs on the c930 RTL — **OURS, done; still not silicon**

`tests/rtl/test_npu_rtl.cpp`, built only with `-DGRXCP_C930_RTL_DIR=<path>`.
The whole stack in one test: `grxMalloc` carves the DDR window, `grxMemcpy`
moves A and B into it, `grxblasGemmEx` routes to the NPU engine and programs the
CSRs, the Verilated RTL runs the GEMM through its own AXI master, and
`grxMemcpy` reads C back. Ten shapes, all matching a column-major host
reference.

**No DPI, no CPU, no firmware.** Every port of `c930_npu_top` is top-level, so
Verilator hands them to C++ as plain members: the harness drives the AXI4-Lite
slave for CSRs and services the AXI4 master against a C++ DDR array. That
sidesteps the GRX930 tree's `verilate-npu` target (whose DPI bridge is not
connected in either direction) and `verilate-grxcp` (which stops on three
Verilator errors in the rv64imac core), neither of which we need.

Two things it settles that no model could:

- **The transposed operands are right on the design, not just on our
  arithmetic.** 7.32's swap was derived from the layout conventions; the RTL
  agrees with a column-major host reference on every legal shape.
- **The crossed bounds are the RTL's, not ours.** The caller's `n = 9` makes the
  engine's `M = 9`, and the library refuses it; the caller's `m = 12, n = 8` is
  accepted and computes correctly. The wall is where 7.32 said it was.

It also confirms, from the RTL rather than from a header, that `0x28` reads 0 —
`ADDR_CYCLE_HI` is declared and never decoded.

**The parameters are passed explicitly and this matters.** `c930_npu_top`
declares `NUM_ROWS=8, NUM_COLS=8, MAX_M=64, MAX_K=256, MAX_N=8` — the "core
defaults" of 7.26. `c930_soc_top` instantiates `4/4/8/16/12`. Verilating with
the defaults builds a different machine than the one `NPU_C930_MAX_*` describes,
which is 7.26 in live form; the build passes `-GNUM_ROWS=4 -GNUM_COLS=4
-GMAX_M=8 -GMAX_K=16 -GMAX_N=12`.

**A simulation is not silicon.** The device reports `GRX_BACKEND_RTLSIM` and
names itself "RTL through Verilator, NOT hardware"; the seam refuses to let
anything attached through it claim `GRX_BACKEND_SILICON` at all. Executing the
design is a real step past executing a model of its interface, and it is not
timing closure, a physical DDR, a clock domain crossing, or a part in a socket.
The phase 7 exit gate is unchanged.

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
