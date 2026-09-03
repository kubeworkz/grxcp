# GRXCP Implementation Roadmap

**Status:** Planning document.
**Companions:** [`grxcp_architecture.md`](grxcp_architecture.md),
[`cuda_mapping.md`](cuda_mapping.md).

Effort figures are engineer-month estimates for a small senior team and are
planning aids, not commitments. Every phase ends with a gate that can be
objectively passed or failed; a phase is not "done" because its code exists.

---

## Dependency shape

```
 P0 foundations
   │
   ├──► P1 runtime v1 (SPIR-V path) ──► P2 device model + tools
   │                                        │
   │                                        ├──► P3 grxBLAS + tensor exposure
   │                                        │        │
   │                                        └──► P4 grxcc single-source
   │                                                 │
   │        ┌────────────────────────────────────────┤
   │        ▼                                        ▼
   │   P5 concurrency (needs GRX-G100 QMD launch)   P6 library breadth
   │        │                                        │
   └────────┴────────────────────────────────────────┴──► P7 NPU + native host

 Parallel track: ISA RFC — WSHFL warp shuffle (GRX-G100 repo)
                 → unblocks the fast path in P2/P3, not their correctness
```

Two phases depend on work in the **GRX-G100 repo**, not GRXCP:

| GRXCP need | GRX-G100 item | Reference |
|---|---|---|
| Real stream concurrency (P5) | QMD-style atomic `CMD_LAUNCH` + multi-queue | `command_processor.md` §10 items 5–6 |
| Device-side event timing (P5) | CP profiling writeback | §10 item 9 |
| Managed memory on FPGA (P6) | `CP_SATP` decode + HW page-table walker | §10 item 2 |
| Safe unaligned `grxMemcpy` on FPGA (P1 mitigation removal) | tail `wstrb` in `VX_cp_dma` | §10 item 1 |
| Fast warp shuffle (P2 fast path) | `WSHFL` ISA extension | this doc, ISA track |

Start those conversations at P0, not when the phase begins.

---

## Phase 0 — Foundations (≈1 engineer-month)

**Scope.** Repository, build system, CI skeleton, and one end-to-end signal
of life. No API surface yet.

- CMake build that discovers the GRX-G100 sysroot through
  `pkg-config vortex-runtime vortex-kernel` — never `$VORTEX_HOME`.
- `AGENTS.md` and the docs set (this file and its companions).
- CI catalog in the same YAML shape as `grxgpu/ci/testcases/`, running on
  `simx` and `rtlsim`.
- `grx-smi` v0: open every device, dump all 18 `vx_device_query` capability
  IDs, print backend and ISA flags.

**Exit gate.** `grx-smi` prints correct, non-fabricated device properties on
both `simx` and `rtlsim`, in CI, from a clean checkout.

**Status.**

| Item | State |
|---|---|
| Repository, `AGENTS.md`, design docs | done |
| CMake build with pkg-config sysroot discovery | done |
| CI catalog (`ci/testcases/grxcp.yaml`) | done |
| Runtime device layer (`src/runtime/{internal.h,error.cpp,context.cpp}`) | done |
| `grx-smi` v0 (human + `--json`) | done |
| Mock driver fixture + unit tests + `ci/build_mock.sh` (tier 1) | done |
| Exit gate: `grx-smi` on real `simx` | **MET** — real device enumerated through the actual driver |
| Exit gate: `grx-smi` on real `rtlsim` | **MET** — a real Verilated GRX-G100, reporting a NARROWER capability set than `simx` because this RTL is built without the TCU and DXA. See cuda_mapping.md 7.36 |

Tier-1 CI passes against the mock driver, and **tier 2 now passes against a
real SimX device**: `ci/build_sysroot.sh` builds the GRX-G100 driver and SimX
backend from a grxgpu checkout without needing the RISC-V toolchain, LLVM or
Verilator, and `ci/run_real.sh` runs the gates against it. `grx-smi` reports a
device enumerated through the actual driver.

**Why first.** It proves the sysroot contract works and gives every later
phase a place to land. It also surfaces build-integration problems while
they are cheap.

---

## Phase 1 — Runtime API v1 (≈3–4 engineer-months)

**Scope.** `libgrxrt` with the full L1 C API, on the existing SPIR-V
toolchain. Kernels are launched with explicit `grxLaunchKernel`; `<<<>>>`
waits for Phase 4.

- Device management, implicit context, `grxDeviceProp_t` from real caps.
- Two-tier slab allocator with the interval map; `grxMalloc`/`Free`/
  `MallocHost`/`MallocManaged` (gated on `VX_CAPS_VM_SUPPORT`).
- `grxMemcpy` family including 2D/3D → `vx_enqueue_*_rect`; `grxMemset`.
- Streams over `vx_queue_h`, including null-stream semantics.
- Events over `vx_event_h` timelines; `grxEventElapsedTime` with the host
  fallback and the honest `eventTimingIsDeviceSide` flag.
- Module/kernel handles; `grxLaunchKernel` → `vx_launch_info_t`.
- Occupancy API using the documented slot formula.
- FPGA safety mitigation: 64-byte allocation padding while
  `VX_cp_dma`'s tail-`wstrb` gap is open.
- Conformance harness: the CUDA sample corpus subset, compiled through
  `grx_cuda_compat.h`, with a **published pass rate**.

**Exit gate.** `vecadd` and `sgemm` written against the GRXCP runtime API
produce numerically correct results on `simx` **and** `rtlsim`; the
allocator survives a fragmentation stress test; the conformance harness runs
and reports a number.

**Risk.** The SPIR-V path constrains what kernels can express. Do not fight
it here — Phase 1's job is the host runtime, not kernel expressiveness.

**Status.**

| Item | State |
|---|---|
| Device management, implicit context, `grxDeviceProp_t` | done (phase 0) |
| Two-tier slab allocator + interval map | done |
| `grxMalloc` / `Free` / `MallocHost` / `MallocManaged` | done |
| `grxMemcpy` family, `grxMemset`, `grxPointerGetAttributes` | done (2D decomposed row-wise; strided descriptors are a follow-up) |
| Streams over `vx_queue_h`, legacy null-stream ordering | done |
| Events over timeline counters, elapsed with host fallback | done |
| Mock driver models memory / queues / events | done |
| `.grxfat` fat binary + ISA-flag image selection | done |
| Module + kernel handles, host-stub registry | done |
| `grxLaunchKernel` / `Ex` / `Cooperative` / `grxLaunchFunction` | done |
| Occupancy API | done |
| Conformance harness + published coverage number | done — 50 of 82 entry points (61%), published in [docs/conformance.md](../conformance.md) |
| Data plane verified on a real `simx` device | **MET** — allocator, memcpy family, streams and events all pass through the real command processor |
| Exit gate: `vecadd` numerically correct on `simx` | **MET** — kernel built with VOLT from GRXCP's own device header, launched through `grxLaunchFunction`, correct at 1/64/70/255 elements including partial warps |
| Exit gate: `vecadd` and `sgemm` on real `rtlsim` | **MET** — the same binaries, `VORTEX_DRIVER=rtlsim` and nothing else changed. `vecadd` 64/64 correct; `test_grxblas` (the sgemm gate, including batched and argument validation) 0 failures. See cuda_mapping.md 7.36 |
| `sgemm` tuning | belongs with grxBLAS in phase 3 |

Four test binaries pass against the mock: `test_device_props`, `test_memory`,
`test_stream_event`, `test_launch`. They verify data correctness through real
offsets, the allocator's non-overlap and reuse invariants, direction
validation, the event/stream error surface, fat-binary image selection, and
that the launch descriptor — grid, block, cluster, shared memory, packed
argument blob — reaches the driver exactly as the caller meant it.

They verify **nothing** about kernel execution or concurrency. The mock has no
RISC-V core and completes every enqueue before returning, so no test here can
fail because of a race or a wrong result. Both are tier-2 properties.

Three of the four now also pass on a **real SimX device** (`ci/run_real.sh`):
`test_device_props`, `test_memory`, `test_stream_event`. `test_launch` stays
tier-1 only, because it builds modules in the mock driver's image format.

**Kernel execution is now proven.** `ci/install_toolchain.sh` fetches VOLT and
the RISC-V binutils, `ci/build_kernel.sh` compiles a kernel into a `.vxbin`,
and `tests/kernels/vecadd/` runs it: a kernel written against GRXCP's own
`grx_device.h`, compiled by the real VOLT clang, loaded with `grxModuleLoad`,
launched with `grxLaunchFunction`, arithmetic checked on the host.

What remains genuinely unproven is **stream concurrency**. The command
processor runs a single queue, so no test anywhere can fail because of a race
between streams. That is phase 5, and it is blocked upstream.

---

## Phase 2 — Device programming model and tools baseline (≈3 engineer-months)

**Scope.** The device-side headers and the tools that make the platform
usable by someone who is not on the team.

- `include/grx/device/`: `grx_device.h` (thread/block indices,
  `__syncthreads`, `printf`, `clock64`), `grx_warp.h` (ballot/any/all from
  `vx_active_threads`, plus the **LMEM-staged shuffle fallback**),
  `grx_cg.h` (cooperative groups over the thread mask and `vx_gbar`).
- `grx-prof` v1: kernel timeline from event timestamps, occupancy report,
  MPM stall breakdown, Perfetto export reusing `grxgpu/ci/perfetto.py`.
- `grx-sanitize` v1 on SimX: out-of-bounds global/shared access, use of
  uninitialized shared memory, and barrier-divergence detection.
- `grxify` v0: mechanical `cudaX → grxX` source rewriting.

**Exit gate.** A warp-reduction kernel using `__shfl_down_sync` produces
correct results through the fallback; `grx-sanitize` detects a deliberately
planted out-of-bounds write and reports the source line; `grx-prof` produces
a Perfetto trace a human can read.

**Progress — warp primitives are done, and were easier than planned.**
`grx_warp.h` is implemented over the ISA's own `SHFL.*` and `VOTE.*`
instructions. The plan had them emulated through local memory, on the strength
of a gap-register entry that had gone stale: the instructions exist, ungated,
and the SimX ALU implements them. The emulation is deleted, not kept as a
fallback, and `warpShuffleIsEmulated` now reports native (cuda_mapping.md 7.1).

`tests/kernels/warp/` is the exit gate's first clause and more: the
`__shfl_down_sync` warp reduction, all four shuffle forms against CUDA's
segmented semantics at two widths, the vote family, and a kernel that shuffles
beside its own shared memory -- which caught the emulation's scratch region
starting at the same address `grx::shared_memory()` returns.

**Progress — `grx-sanitize` v1 is done, and the exit gate's second clause with
it.** A kernel built with `ci/build_kernel.sh --sanitize` is compiled with
AddressSanitizer's checks *outlined* (`-mllvm
-asan-instrumentation-with-call-threshold=0`), so every load and store becomes
a call into `src/device/grx_sanitize_rt.cpp` — where the check is a lookup in
the allocator's own map rather than a shadow-memory probe. No shadow map has to
exist, and the check knows things a shadow map would not: the size the caller
actually asked for, and whether the allocation has been freed.

The SANITIZE GATE in `ci/run_real.sh` plants four bugs — an overflow, an
underflow, a use-after-free, and a shared-memory overrun — and requires each to
be reported *at the line it lives on*, with the line numbers read out of the
source rather than hard-coded. Two controls sit beside them: the same kernel
without the bug must come back clean, and the same bug in an **uninstrumented**
build must be reported as unchecked rather than as clean.

Two of the phase-2 scope items did not ship: uninitialized shared memory
(needs a per-byte initialization shadow) and barrier divergence (needs a
timeout watch). Both are described, with their designs, in
[`grx_sanitize.md`](grx_sanitize.md) §2 and §7.

Building it also turned up a live hazard worth more than the tool: the device
toolchain compiles kernels `-march=rv64imafd`, but this configuration has
`VX_CFG_EXT_A_ENABLED` off and the simulator **aborts without a word** on any
AMO instruction. The sanitizer's first draft counted findings atomically and
died on its own first finding. It now indexes reports by grid-linear thread,
and `GRX_CAP_GLOBAL_ATOMICS` reports the capability so nothing else has to
discover this the same way (cuda_mapping.md 7.16).

**Progress — `grx-prof` v1 is done, and the phase 2 exit gate is MET on BOTH
BACKENDS.** All three clauses now run on a real Verilated GRX-G100 as well as
on `simx`: the `__shfl_down_sync` reduction passes, `grx-sanitize` finds the
planted out-of-bounds write and names `kernel.cpp:31:29`, and `grx-prof`
produces a nine-event trace whose own banner says it is measuring the
simulator. See cuda_mapping.md 7.36.

**Progress — `grx-prof` v1 is done, and the phase 2 exit gate is MET.**

The gate's three clauses: a `__shfl_down_sync` warp reduction producing correct
results (met, `tests/kernels/warp/`), `grx-sanitize` finding a planted
out-of-bounds write and reporting the source line (met, SANITIZE GATE), and
`grx-prof` producing a Perfetto trace a human can read (met, PROF GATE).

The design question grx-prof had to answer was *which clock*. There are two.
The host clock orders operations and shows the gaps between them, and on a
simulator it measures the simulator. The MPM performance counters measure the
device: `MCYCLE` advances only while the device runs, so its delta across a
launch is device time on `simx` and on silicon alike. grx-prof builds the
timeline on the first, annotates every kernel slice with the second, and
converts neither into the other — the trace's own axis label says which clock
it is. See [`grx_prof.md`](grx_prof.md).

The PROF GATE checks readability three ways (the trace parses, the kernel slice
carries device cycles, the report states which numbers are host-clock) and then
checks the thing that matters most: the same kernel at three sizes, with the
device cycle count required to climb. A profiler emitting numbers nobody has
watched respond to their input is not measuring anything — the same discipline
`tests/kernels/cycles/` applies to `grx::cycle_probe`. Tier 1 adds the
complementary check: against the mock driver, which refuses
`vx_device_mpm_query` because it models a control plane and has no pipeline to
count, **no** device counter may appear on any slice. Absent, not zero.

One planned item changed shape. The roadmap said "Perfetto export reusing
`grxgpu/ci/perfetto.py`"; that script converts instruction-level simulator logs and
needs a debug simulator build, which answers "what is this warp doing in cycle
4,182" rather than "which kernel is expensive and why". grx-prof writes the
same trace *format* from what the runtime and the counters already know, with
no special build, and `grxgpu/ci/perfetto.py` remains the next zoom level down
(`grx_prof.md` section 5).

**Progress — `grx_cg.h` is done, and phase 2 with it.**

`grx::cg` covers `thread_block`, `thread_block_tile<N>` with the shuffle family
and `reduce`/`inclusive_scan`/`exclusive_scan`, `coalesced_group` over the
active mask, `cluster_group`, and `grid_group` including a real grid-wide
barrier. `tests/kernels/cg/` checks every collective against a reference the
host computes independently, and the grid barrier carries a control that has to
fail: the same kernel without the barrier, with block 0 stalled before it
publishes, must read a value that is not there yet. Without that control the
barrier test would pass whether or not the barrier worked.

Writing it turned up three things about the hardware that were not written down
anywhere:

- **The barrier id is a packed pair, not a slot index.** `cta_id | (bar_no <<
  8)`, flattened to `cta_id * NUM_BARRIERS + bar_no`. The cross-CTA forms force
  the CTA field to zero so every CTA names the same slot — which also lands
  them inside CTA 0's range. A cluster barrier on the obvious `bar_no` 1
  collides with CTA 0's own `grx::barrier(1)` and hangs. `grx_device.h` now
  carries the slot map and reserves the top two numbers.
- **The "global" barrier releases per CLUSTER**, and its participant count is
  cores per cluster — `VX_CFG_NUM_CORES`, not `VX_CSR_NUM_CORES`, which is the
  device total. On a multi-cluster device there is no grid barrier at all, and
  `grid_group::sync()` is unavailable rather than a call that hangs
  (cuda_mapping.md 7.17).
- **A cooperative launch must cover every core**, not merely fit. A core with
  no active warps never forwards an arrival. The launch validation now refuses
  a grid smaller than the machine, and `grxLaunchCooperativeFunction` brings
  that validation to the module path, which had no cooperative entry point.

Not implemented, and listed rather than left to be discovered:
`labeled_partition` and `binary_partition`, the `__match_*` family (no match
instruction exists), and `cluster_group::map_shared_rank`'s two-argument CUDA
form — the per-CTA local-memory stride is not exposed to the kernel, so the
three-argument form takes it explicitly (cuda_mapping.md 7.18).

**Phase 2 is complete**: the device headers, `grxify` v0, `grx-sanitize` v1,
`grx-prof` v1, and all three exit-gate clauses.

**Note.** `grx-sanitize` lands early on purpose. On a functional simulator
it is cheap, and it pays for itself across every later phase by catching the
memory bugs that otherwise burn days in RTL debug.

---

## Phase 3 — grxBLAS v1 and tensor-core exposure (≈4 engineer-months)

**Scope.** The library that justifies the platform, and the first real
collision with the SPIR-V path's limits.

- `grx::wmma` device API over `vortex::tensor::wmma_context` /
  `wgmma_context`: `fragment`, `fill_fragment`, `load_matrix_sync`,
  `mma_sync`, `store_matrix_sync`, plus WGMMA warp-group forms and the 2:4
  sparse variants.
- `grx::pipeline` / `grx::memcpy_async` over `vx_dxa_issue_*_wg` and
  `vx_barrier_expect_tx`.
- grxBLAS v1: GEMM (fp32/fp16/~~bf16~~/int8), batched GEMM, GEMV, AXPY, with
  autotuned tile selection seeded from the existing `sgemm_tcu_wg_dxa_mcast`
  reference kernels. **bf16 is struck**: the tensor unit has no bf16 mode in
  any configuration, and int8 needs a sysroot built with
  `VX_CFG_TCU_INT8_ENABLE` (cuda_mapping.md 7.19).

**The structural problem, handled deliberately.** Kernels using TCU/DXA
intrinsics **cannot** go through SPIR-V — the GRX-G100 docs call this out
explicitly. So grxBLAS v1 kernels are compiled with the native VOLT path
directly to `.vxbin` and shipped as prebuilt modules loaded by
`grxModuleLoadData`. This is not a workaround, it is how vendor BLAS
libraries actually ship: precompiled, hand-tuned kernels behind a host API.
It also de-risks Phase 4 by proving the native compile path before `grxcc`
depends on it.

**Exit gate (restated).** `grxblasGemmEx` (fp16 in, fp32 accumulate) costs
**no more than a fifth of sgemm v0's cycles per output element** on the same
configuration, with a passing numerical gate against a CPU reference.

The original wording was "within 15% of the tuned `sgemm_tcu_wg_dxa_mcast`
reference on simx cycle counts". Two problems with it, both found by trying to
run it. That reference kernel is not in the grxgpu source this project can
reach, so the comparison had no left-hand side. And "simx cycle counts" had no
mechanism: event timing is a host clock and on a simulator measures the
simulator (cuda_mapping.md section 7.4), so there was nothing that could
produce the number.

Both are fixed rather than waived. Measurement is `grx::cycle_probe` reading
the device's own counter, calibrated by `tests/kernels/cycles/` — which runs
the same kernel at 1x, 2x and 4x the work and fails unless the count follows.
The baseline is `tests/bench/gemm_cycles.cpp`, and it is a number this project
produces itself instead of one it hopes to be handed:

| shape | span (cycles) | cycles / output element |
|---|---|---|
| 16 x 16 x 16 | 76 207 | **297.7** |
| 16 x 16 x 32 | 133 973 | 523.3 |
| 32 x 32 x 32 | 536 385 | 523.8 |
| 64 x 64 x 32 | 2 147 531 | 524.3 |

One SM, four-lane warp, SimX. About 16 cycles per multiply-add, flat across
shapes, which is what an unblocked one-thread-per-element loop with two global
loads per MAC and no reuse should cost. The 5x target is deliberately a target
and not a prediction: the tensor unit does an 8x4x8 tile per instruction where
this does one MAC per thread per iteration, so a tuned kernel that only manages
5x has left most of the unit unused — and if the reference kernel does turn up,
comparing against it is then a matter of running both through the same harness.

A cycle count from a simulator is what the MODEL does. It is the right thing to
compare two kernels with and the wrong thing to quote as hardware performance,
and every place that prints one says so.

**Progress — grxBLAS v0 has landed, and it is not the exit gate.**
`grxblasSgemm` runs end to end on `simx`: host API, a precompiled `.vxbin`
resolved by name, and a numerical gate (`tests/libs/test_grxblas.cpp`) covering
all four transpose combinations, padded leading dimensions, `alpha`/`beta`
scaling, `k = 0`, and stray writes outside the m x n window. The kernel is one
thread per output element — no blocking, no shared-memory staging, no tensor
cores. It is **correct, not fast**. It exists to fix the API, prove the kernel
packaging, and give the tuned path something to be measured against. The
15%-of-reference gate above is untouched by it, and no performance claim should
cite it.

The v0 gate was itself checked against a deliberately broken kernel: reverting
the transposed-A load to an expression that transposes nothing makes TN and TT
fail while NN and NT still pass. Worth repeating whenever the CPU reference
changes — the reference's first version shared a wrong index expression with the
kernel, so the transpose-A case passed while transposing nothing.

**Progress — `grx::wmma` runs on the tensor unit.** The device-side WMMA API is
implemented over `vortex::tensor::wmma_context` rather than declared: fragments,
`fill_fragment`, `load_matrix_sync`, `store_matrix_sync`, `mma_sync`, dense
fp16-in/fp32-out and fp32-in/fp32-out, single warp. `tests/kernels/wmma/` runs
one tile on `simx` and checks every element **exactly** — the inputs are chosen
so that a correct tensor unit reproduces the reference bit for bit, which means
no tolerance is available to hide a wrong answer behind. Verified in the other
direction too: declaring B `row_major` instead of `col_major` makes all 32
elements wrong.

The tile is **8 x 4 x 8**, not 16 x 16 x 16, and it is derived from the
configuration rather than fixed. That is a structural difference from CUDA and
is registered as such (cuda_mapping.md section 7.9). What is still missing for
the exit gate: WGMMA warp groups (needs `VX_CFG_TCU_WGMMA_ENABLE`), the DXA
async-copy staging in `grx_pipeline.h`, and the blocked grxBLAS kernel that puts
them together.

**Progress — asynchronous tile copy works, both sides of it.** `grx::pipeline`
is implemented, and so is the host half it needs: `grxTensorMapProgramAsync`
programs a DXA descriptor slot through stream-ordered register writes, and
`grx::memcpy_async` fetches a tile of that descriptor into the CTA's shared
memory, completing on a `grx::barrier` transaction. `tests/kernels/dxa/` checks
a staged tile element for element out of a **padded** array, in both destination
layouts, with one and two warps; `tests/unit/test_tensormap.cpp` checks the
register encoding at tier 1 against writes captured by the mock driver.

Two things this turned up, both now fixed and registered in cuda_mapping.md:

* **Static `__shared__` never worked** (section 7.10). It was defined as a
  `.shared` section attribute, and the device link script has no such section,
  so arrays landed in global memory. Every kernel so far had done without it, so
  nothing had failed. It is a compile error now, pointing at
  `grx::shared_memory<T>()` — dynamic shared memory, which does work.
* **Descriptors are device slots, not values** (section 7.11), and the engine
  bypasses the MMU, so the memory it reads has to be physically addressed —
  hence `grxMallocPhysical`.

What is left before the phase 3 exit gate: WGMMA warp groups (needs
`VX_CFG_TCU_WGMMA_ENABLE`, off in the configuration used here), the multi-stage
`pipeline<Stages>` structure, and the blocked grxBLAS kernel that composes
tensor cores with async staging.

**Phase 3 exit gate: NO LONGER MET — 2.65× against a 5× threshold, and the
threshold has not been moved.** `grxblasGemmEx` (fp16 in, fp32 accumulate)
composes the tensor unit with DXA staging and is exact against a CPU reference
on every shape including ragged ones. What it no longer does is cost less than a
fifth of sgemm per output element — because **sgemm got 3.76× faster**, three
times over, and the gate is a ratio between two things that both move.

| shape | sgemm cyc/elem | GemmEx cyc/elem | vs tuned sgemm | vs reference sgemm |
|---|---|---|---|---|
| 16 x 16 x 16 | 50.2 | 34.1 | 1.47× (starved) | 4.84× |
| 16 x 16 x 32 | 80.7 | 41.7 | 1.94× (starved) | 7.16× |
| 16 x 16 x 64 | 142.5 | 56.8 | 2.51× (starved) | 9.74× |
| 32 x 32 x 32 | 78.5 | 29.6 | **2.65×** | 9.96× |
| 32 x 32 x 64 | 141.2 | 44.4 | 3.18× | 12.41× |

The tensor path did not regress: 29.6 and 44.4 cycles per element are what it
read before the SIMT work as well. On three of five shapes the tensor unit is
now within 2.5× of a SIMT kernel, which is a fact about how little of the core
the single-CTA workaround lets it use rather than about the unit. The two columns on the right are printed by
`tests/bench/gemm_cycles.cpp` on every run precisely so this is legible.

Three things could have been done and only one is honest without a decision on
the record. Moving the threshold to 4× would read as a relaxation forever;
restating the gate against the *reference* kernel would freeze its denominator
and make it unfalsifiable by SIMT work. Leaving it is what AGENTS.md section 4
requires — an assertion is not relaxed as a side effect of unrelated progress.
So the gate is red, `ci/run_real.sh` **defers** its failure to the end of the
run so the thirty sections after it still execute, and tier 2 reports FAILED
with the reason named.

What would make it pass again is the tensor unit's multi-CTA deadlock
(cuda_mapping.md 7.12). Three of five shapes are already excluded here for tile
starvation, which is a consequence of the single-CTA workaround; lifting it
gives the tensor path the parallelism the ratio was first measured with.

**And then the tuning was tried, and lost.** The obvious levers were
implemented and measured rather than assumed. Cycles per output element at
16 x 16 x 16, one SM, four warps, one DXA worker:

| per-warp shape | buffering | cyc/elem | vs shipped |
|---|---|---|---|
| 1 x 1 tile | single | **39.0** | shipped |
| 1 x 1 tile | double | 47.9 | 21% worse |
| 1 x 2 block | single | 42.0 | 8% worse |
| 2 x 1 block | single | 43.8 | 12% worse |
| 2 x 2 block | single | 47.9 | 23% worse |

Every one of them is the textbook next step, and every one made it slower. Two
reasons, and the second is the interesting one. Four warps on one core already
hide the staging latency by interleaving, so a second buffer prefetches into a
pipeline that was not stalling and only adds barrier and addressing work. And
blocking cuts the BYTES staged per output tile without cutting what the single
DXA worker costs: the A transfer uses the transposing destination, which
scatters one element per beat, so its cost tracks elements per transfer rather
than transfers issued — a bigger block makes each transfer proportionally
longer while leaving fewer independent warps at the edges.

The blocking factors stayed in the kernel as knobs, set to what measured best,
with the table in the file header and an instruction to re-run the sweep before
assuming it still holds. None of these conclusions need survive more DXA cores,
more SMs, or WGMMA's larger native tiles — which is exactly why the numbers are
recorded with the configuration attached rather than as a rule of thumb.

What the shipped kernel is still missing: WGMMA warp groups, reuse across k,
and more than one CTA -- the last not by choice, see cuda_mapping.md 7.12.

Two device-stack defects came out of building it, both now documented and
watched rather than worked around silently: the tensor unit deadlocks when a
second CTA issues a tensor instruction (7.12), and only one device module can
be resident at a time because every image links at the same address (7.13). A
third sharp edge is documented at 7.14 -- the DMA engine pads a tile's outer
dimensions but reads straight past the end of dimension 0.

Still open for phase 3 v1: transposed operands, bf16/int8, batched GEMM,
autotuned tile selection, and the tuning this kernel deliberately does not have.

**Progress — the gate is measurable now.** `grx::cycle_probe` reads the device's
cycle counter, `tests/kernels/cycles/` proves the reading responds to work
(1x/2x/4x, ratio 2.00, 81 cycles per loop iteration), and
`grxblasSetCycleProbe` lets the same shipped kernel be measured without being
recompiled for measurement. sgemm v0's baseline is in the table above.

Also, and found the hard way: the sgemm argument block now carries an ABI
version the kernel checks. Adding the probe field changed the struct, a stale
test binary passed the old shorter blob, and the kernel read a pointer from past
the end of the staging area and stored through it. A mismatched .vxbin and
library is now an obviously wrong answer instead of memory corruption.

**Progress — level 1 and level 2 have landed, and the type list got shorter.**

`grxblasSaxpy`, `grxblasSscal` and `grxblasSgemv` run on `simx` with a gate
(`tests/libs/test_grxblas_l12.cpp`) whose every comparison is **exact**: the
values are small integers held in floats, so every summation order gives the
same bits and there is no tolerance for a wrong answer to hide behind. It
covers unit, strided and negative increments, padded leading dimensions,
`beta = 0` against a y pre-filled with NaN (a kernel that multiplies through
produces NaN and is caught), `n = 0`, single elements, and a reduction longer
than a warp.

The gate was watched failing before it was believed: reverting `sgemv`'s
transposed load to the classic wrong index makes all five transposed cases fail
and leaves every untransposed one passing.

**"Negative increments" was true of saxpy and not of `sgemv`.** Every `sgemv`
case ran with `incx` of 1 or 2. One case reversed *y*, none reversed *x*, and
the sentence above did not distinguish them — so the branch BLAS is most likely
to get wrong had no test at all. Found while hoisting that branch out of the
inner loop, which is the change that had to be safe. Five cases now reverse
`x`: untransposed, untransposed with both vectors reversed and strided,
transposed with the reduction longer than a warp, transposed and strided both
ways, and transposed with the reduction SHORTER than a warp — the last because
lanes with no element are where a start offset computed before its bound test
underflows. Watched failing: a `vec_start` that always begins at the near end
takes exactly those five red and leaves all eleven pre-existing cases green.

**`sgemv` was the most expensive shipping kernel in the image, and it was
address arithmetic again.** The hot-loop census priced its inner loop at
fifteen instructions and two loads for one multiply-add. Almost none of that
was the product: `vec_index` re-decided the sign of the increment on every
iteration and did two multiplies to place one element. But the increment is
**affine** — for `inc > 0` the index is `i*inc` and for `inc < 0` it is
`(n-1-i)*(-inc)`, and *both* advance by exactly `inc` per step. The sign
chooses a starting offset and nothing else. Walking a pointer from that start
takes the loop to **seven** instructions, and it is the same finding as the
GEMM k-loop one commit earlier: the loop-invariant part of an address does not
belong inside the loop.

Measured, not counted — instructions are the prediction and cycles are the
claim. `tests/bench/gemv_cycles.cpp` prices both traversals on the device:
**1.12× to 1.94× on the span** and **1.25× to 2.25× on the per-warp busy
window**, over three shapes each way. The spread is the point. The long
reductions win about twice; the shape whose reduction is four steps per lane
wins least, because there is almost no loop there to amortise. A single
headline number would have hidden that, so the bench reports every shape and
the baseline pins every one of them.

`sgemv` has two traversals rather than one shape with a flag, and the reason is
the memory system. Untransposed, one thread per output row means consecutive
lanes read consecutive rows of the same column — adjacent addresses.
Transposed, one thread per output column would stride the lanes `lda` apart, so
instead a whole warp takes one column, walks it together, and finishes with a
`grx::cg::thread_block_tile::reduce` over the lanes. The transposed case is not
a variant of the untransposed one; it is a different traversal that happens to
compute a transposed product. It is also the first use of `grx_cg.h` inside the
library rather than in a test.

**A LAUNCH COSTS 2776 CYCLES BEFORE IT TOUCHES AN ELEMENT, and that is what the
fusion question was really about.** Measured directly by sweeping
`grxdnnAddBiasForward` over column counts at the block's own row count:

    elements     cycles      elements     cycles
          16       2958           512       4919
          32       2956          1024       7349
          64       2902          2048      11853
         128       3216          4096      20405
         256       3854

    least squares over nine points:  cycles = 2776 + 4.33 * elements

The first sixteen elements cost 2958 cycles. The next four thousand cost 17447.
An independent two-point fit from the block's own bias stages gave 2716 + 4.25
and predicted a held-out third shape to 1.5%, so the two agree.

**The first version of this experiment was wrong and is worth recording.** It
swept element count with `rows = 1`, which puts every element on ONE warp, and
reported `cycles = 971 + 37.66 * elements` -- a per-element cost nine times too
high, because varying cols at one row removes the parallelism instead of
isolating it. The block's stages all run `rows = S`. Sweeping cols at `rows = 16`
is the comparison that matches them.

**What it means.** 86% of the qkv bias stage is launch overhead: six launches at
2776 each is 16656 of its 19272 cycles. More broadly the block runs about
fourteen elementwise launches, so roughly **12% of it is spent before any
element is touched**. Fusion's prize here is not memory traffic -- it is
launches. That is why "one launch stops existing" was the only fusion estimate
that did not rest on a per-word rate, and why the per-word model disagreed with
itself across shapes.

**Two routes for the qkv bias, with measured values rather than preferences:**

  * **Bias in the GEMM epilogue** -- `grxblasSgemmStridedBatched` already runs
    all H heads in one launch, so a fused bias adds no launch at all and one add
    per output. Worth the whole stage: **5.9% at S=16, 6.8% at S=8**. Costs an
    ABI field, an epilogue in five kernels (the three tiled ones share a
    template), a host entry point, and a re-run of the bit-exact oracle -- it
    touches the kernel family that is 46% of the block.
  * **A batched add-bias** in grxDNN, mirroring `SgemmStridedBatched`: six
    launches become three. Worth **~2.5%**, self-contained, no GEMM change.

The first is worth roughly twice the second and carries the risk of touching the
hottest kernel family in the image. Neither is done.

**The block profiler was throwing away six launches, and they were the largest
fusion target in the block.** `tests/bench/block_cycles.cpp` ran the qkv
projections' per-head biases -- 3 projections x H heads -- and then called
`probe.clear()` under a comment saying they were "counted with the other
biases". They were not. The slots were discarded and no stage absorbed them, so
every SHARE this bench reported was a fraction of a denominator that was too
small, including the attention share published on the ledger.

Counted, they are **11003 cycles at S=8 (6.8%)** and **19272 at S=16 (5.9%)** --
larger than both remaining bias stages put together, and larger than either
layernorm. Every share moved: attention 21.1% -> **19.8%** at S=16 and 14.7% ->
**13.7%** at S=8, gelu 16.7% -> 15.7%.

**I claimed the blocking RATIO was unharmed, and that was wrong.** The argument
was that it is naive over blocked and both totals omitted the same six launches,
so the omission cancels. It does not. Adding roughly the same CONSTANT to both
sides of a ratio greater than one drags it toward one, and this stage is exactly
constant -- `dnn_add_bias` does not care which sgemm produced the tensor it is
adding to. Counting it costs 19067 cycles on the naive side and 19272 on the
blocked side, and the S=16 figure falls from **2.301x to 2.224x**; S=8 falls from
2.22x to **2.13x**.

A ratio survives a PROPORTIONAL omission, not a constant one. So the discarded
launches were not merely making shares wrong -- they were **inflating the
headline speedup**, which is the number this project has quoted most often. The
corrected pair is 2.13x at S=8 and 2.22x at S=16, and it was the regeneration
that caught it rather than the reasoning.

**Why it is so expensive is the interesting part.** Six launches of S x Dh --
128 elements each at S=16 -- costing 3212 cycles apiece, against the mlp bias's
7068 cycles for 1024 elements in ONE launch. It is not the arithmetic and it is
barely the memory: it is six launches doing an eighth of the work each.

**What this does to the fusion arithmetic.** The question that prompted the
count was how much of the block is redundant round-tripping between adjacent
elementwise stages. Two pairs were priced against the measured stages:

| candidate | S=8 | S=16 | confidence |
|---|---|---|---|
| mlp bias -> gelu, fused as `gelu(f1 + b)` | 2.7% | 2.3% | high -- one launch simply disappears |
| out-proj bias -> memcpy -> saxpy, fused to one pass | ~1.7% | ~1.6% | low -- the "one pass not three" factor is assumed, and the memcpy is not a probed launch so its cost is in nobody's total |
| **qkv bias into the GEMM epilogue** | **6.8%** | **5.9%** | high -- it is the whole stage |

The target that was invisible is bigger than both that were not.

**And a caveat on the model those first two numbers came from.** Cycles per word
is not a constant of a kernel: the same `dnn_add_bias` reads 6.17 cycles/word on
the S x D tensor and 2.91 on S x F, because row length decides how per-row setup
amortises. Any fusion estimate resting on a per-word rate is an estimate. The
one number that does not rest on it is "this launch stops existing".

Worth recording alongside: on the identical S x F shape, `dnn_gelu` costs **24.64
cycles/word against `dnn_add_bias`'s 2.91**. Eight times, for the same traffic.
gelu is computing, not moving, and no fusion reaches it -- which with softmax's
68%-polynomial puts the exponential under roughly a fifth of the block as a
floor.

**Softmax was computing its exponential twice per element, and it is half of
attention.** Attention is the largest stage in the block and the only one whose
share grows with sequence length, so it was the obvious next target -- and there
was no way to see which quarter of it to touch, because `block_cycles` reports it
as one stage. It is four launches. `tests/bench/attention_cycles.cpp` splits
them, each against its own clock with `maxLive` checked against occupancy, and
the answer was immediate:

| S | scores GEMM | causal mask | softmax | out GEMM |
|---|---|---|---|---|
| 8 | 19.4% | 12.2% | **49.4%** | 19.1% |
| 16 | 24.1% | 8.7% | **49.7%** | 17.4% |
| 32 | 25.5% | 5.9% | **52.1%** | 16.4% |
| 64 | 27.5% | 3.8% | **53.0%** | 15.7% |

Nothing pointed there. The hot-loop census ranks `dnn_softmax` near the BOTTOM
of its cost table at 2.00 instructions per float op -- and the rate was correct.
There were simply a great many float operations, because the kernel made three
passes over each row and computed `dev_exp` in **two** of them: once to build the
sum, once to write the result.

The comment defending it said recomputing was cheaper than "staging the row
anywhere a third pass could read it back from", and the premise was the mistake.
There is nowhere to stage it only if you are hunting for scratch memory. The
OUTPUT row is already allocated, already the right size, and already about to be
written. Pass 2 keeps the exponential there and pass 3 becomes a load, a
multiply and a store.

**1.39x at S=8 rising to 1.74x at S=64 on softmax; 1.16x to 1.29x on attention;
3.3% and 4.3% on the whole block.** The control is inside the measurement: the
scores GEMM, the causal mask and the output GEMM are identical to the digit at
every sequence length, so none of this is the relink. Attention's share of the
block falls 17.1% -> 14.7% at S=8 and 24.2% -> 21.1% at S=16.

**The arithmetic did not change and the output is bit-identical.** Both versions
compute `e * inv`; the new one computes `e` once. All twelve softmax cases in
`test_grxdnn` report the same worst-case difference to the last digit, in place
included -- with `y == x` the write lands on `xr[j]` only after this lane has
read it, and pass 3 then reads what pass 2 wrote, which is what it needs.

**And the census called it a regression.** 34 instructions to 41, ins/fp 2.00 to
2.41, because the loop it looks at absorbed the store the deleted third pass used
to do. The kernel got smaller (259 bytes to 227) and its frame shrank (112 to
80) in the same change. Nothing is wrong with the number -- it answers "what is
the innermost float loop made of", and that question stops tracking cost the
moment work moves BETWEEN loops. It is the fourth time this ranking has needed a
caveat written next to it, and they are all now in the file's header.

**`dnn_add_bias` was the worst kernel in the census and the change was
reverted.** It headed the cost ranking at 13.00 instructions per float op, and
its loop was five instructions of address arithmetic around one `fadd.s` -- an
index widening pair, because `j` is 32-bit and the offset is 64, and one add per
operand to form each of three addresses. The same hoist that took the GEMM
k-loop and `sgemv` apart applies, and it worked exactly as predicted: 13
instructions to **11**, the number written into the baseline before the change
and watched failing at 13.

**And the block did not move.** 155347 -> 155134 at S=8, 323119 -> 323232 at
S=16 -- the second slightly worse. Inside one build the two bias stages moved in
OPPOSITE directions, +3.5% and -9.3%, while kernels whose source was untouched
moved +4.1% and +3.1%. That is the relink, which `ci/check_perf.py` already
documents at up to 8.7%, and it swamps a 15% instruction saving on a stage worth
3.4% of the block. The kernel does two loads and a store per float op; the
arithmetic was hiding underneath the memory traffic the whole time.

Reverted rather than kept. It is bit-identical and strictly less work, and that
is not the standard: the GEMM and `sgemv` hoists shipped because they were
*measured* at 2.10x and 1.9x, and shipping this one would be shipping on the
argument "address arithmetic is bad" -- which is the reasoning this project has
now falsified twice.

**Two things the cost ranking does not know, and it has now misled once each.**
It is a rate. It does not say where the time is: `dnn_add_bias` headed it at
13.00 and is 3.4% of the block, while `dnn_gelu` sits at the BOTTOM at 1.44 and
is 16%, and `sgemv` was second and is not in the block at all. And it does not
say what the loop is waiting for. The census prints memory traffic beside the
rate now, and flags any loop moving two or more words per float op.

The flag is not a verdict either, and `sgemv` is the reason it says "price it in
cycles" rather than "instructions do not matter here": `sgemv` carries the same
marker and its hoist WAS worth 1.9x, because it removed eight instructions of
fifteen rather than two of thirteen. Memory traffic says the arithmetic can
hide, not that it does. Neither the rate nor the flag separated the two cases.
The bench did.

**The scores GEMM was 35.5% of attention, and the fix was not a wider tile.**
Once softmax stopped computing its exponential twice, the scores GEMM became
the share that grows fastest with sequence length — 22.5% at S=8 rising to
35.5% at S=64. `GRXBLAS_SGEMM_TRACE` says both of attention's GEMMs run the
2x2 tile, because the rule has one threshold and, in `grxblas.cpp`'s own words,
"THE WIDE TILE IS NOT IN THE RULE. Reachable only by asking, because nothing
has measured it yet". So it was measured.

At S=64, H=4, Dh=8 the two GEMMs do **identical arithmetic** — 131072
multiply-adds each — and cost 487869 and 279249 cycles. The difference is the
shape: `scores` is 16384 outputs of 8 MACs, `out` is 2048 outputs of 64. The
scores GEMM pays per-output setup eight times as often over the same FLOPs,
which is exactly what a wider register tile amortises.

**Two obvious fixes were predicted, measured, and were both wrong.**

| hypothesis | predicted | measured |
|---|---|---|
| wider tiles amortise the setup | faster | **0.76x to 1.00x** — slower everywhere |
| unroll the k loop for more MLP | 5-8% faster | **7% SLOWER** at k=8 |

The wide tiles lose because they trade warps for arithmetic density: 4x2 halves
the warp count and 4x4 quarters it, and this configuration has warps to spare.
They are not wrong, they are unmeasurable here, and the rule's caution was
right. `tests/bench/attn_gemm_tiles.cpp` is that measurement, kept so nobody
re-proposes it from the same reasoning. The k-loop unroll pays only from k=32
up, and the scores GEMM is k=8.

**What worked came from a cost model rather than a guess.** Sweeping k with the
warp count held fixed splits the warp's cost in two: the k>=16 points fit a line
of **29.1 cycles per k step on an intercept of 180 cycles**, and k=16 and k=64
give that intercept to a decimal. At k=8 a warp costs 470 cycles, so roughly
38% of the scores GEMM is setup that no amount of k-loop work amortises.

Most of that setup is bounds tests that cannot fail. `micro_tile_body` computes
`row_live[]`/`col_live[]` and guards every store with them, but when
`m % RM == 0` and `n % RN == 0` every tile is wholly inside C — which is exactly
attention, where m and n are the sequence length and the head dimension.
`sgemm_2d_i` is the same 2x2 tile with those tests compiled out. It carries
**2 split/join pairs against sgemm_2d's 8**, because the guards were also where
the divergence came from.

**Measured end to end, not quoted from the microbenchmark:**

| S | scores GEMM | out GEMM | attention total |
|---|---|---|---|
| 8 | **1.338x** | 1.291x | 1.101x |
| 16 | **1.199x** | 1.127x | 1.062x |
| 32 | **1.197x** | 1.055x | 1.068x |
| 64 | **1.175x** | 1.029x | 1.061x |

The controls are in the same run: softmax and the causal mask are untouched
kernels and move 0.974x to 1.002x. That band is wider than the softmax change's
"identical to the digit", because this change relinks the whole `.vxbin` and
moves every kernel's code, so the S=64 out GEMM's 1.029x is inside the noise and
is not claimed. The scores GEMM's 1.175x is not.

**It is a second entry point, not a branch, and that was measured too.**
Dispatching between the two instantiations inside one `sgemm_2d` made interior
shapes faster and boundary shapes **12% slower at k=8**: they pay instruction
fetch for a body they never execute. Split across two entry points, a launch
touches only the code it runs, and the interior path gets 1.18x instead of
1.10x. The host picks on `m % RM == 0 && n % RN == 0`; nothing else changes,
and everywhere the rule already said 2x2 it still says 2x2.

**The oracle could not have caught a mistake here, so it was given a way to.**
`sgemm_2d_i` launches *exactly* `sgemm_2d`'s geometry, and `test_grxblas_rb`
told the kernels apart by counting launched warps — which cannot separate these
two. A forced run that silently fell back would have compared 2d against the
reference and reported it as evidence about 2d-i. `grxblasGetLastSgemmKernel`
makes the library say what it launched; the oracle asserts it, and the
assertion was watched failing with the force hook disabled before it was
believed ("forced 2d-i but the library ran naive").

**Softmax, ablated rather than reasoned about.** With the scores GEMM 1.18x
faster, softmax is the largest stage again at 41.8% of attention. It is three
passes over the row with `dev_exp` in the middle, and the file already said the
exponential was "the expensive thing in this kernel by a wide margin" — which
is an assertion until something removes it. Four ablations, each a deliberately
WRONG kernel run only to price a part, at S=64:

| ablation | softmax | saved |
|---|---|---|
| control | 542584 | — |
| **`dev_exp` removed** | 199253 | **63.3%** |
| exp's polynomial cut to `1+r` | 386077 | 28.8% (**45.6% of the exp**) |
| pass 1, the row max, removed | 479003 | 11.7% |
| pass 3, the normalise, removed | 497121 | 8.4% |

The exponential is confirmed at 63.3%, and the surprise is inside it: the
polynomial is under half of it. The range reduction, the clamp and the
exponent-field construction together cost MORE than the degree-5 series they
exist to support.

**Three of those instructions are dead in softmax specifically, and provably
so.** `dev_exp`'s argument here is `xr[j] - row_max`, which is `<= 0` at every
element by construction. So the clamp to `+88` cannot bind;
`dev_copysign(0.5f, x)` is the constant `-0.5f`; and `(k + 127) & 0xFF` is
redundant, because `x` in `[-88, 0]` gives `k` in `[-127, 0]` and `k + 127`
already fits in eight bits. `dev_exp_nonpos` is `dev_exp` with those three
gone.

**Exact, and checked as such rather than argued.** A host-side comparison of
the two functions over 8818703 values — every representable negative float down
to -200 by bit pattern, plus the clamp edges and NaN — reports zero
differences, bit for bit. NaN matters and is not obvious: `dev_exp` gives
`fmin(fmax(NaN, -88), 88)`, and IEEE-754 `maxNum` returns the non-NaN operand,
so both forms land on -88. The comparison was then sabotaged (one constant
moved by 1e-4) and reported 63 mismatches, because a check that cannot fail
proves nothing.

**The census is the measurement that carries this, not the clock.**
`dnn_softmax`'s element loop goes **41 instructions to 38, 17 float operations
to 15**, deterministically; `dnn_gelu` (46/32) and `dnn_layernorm` (15/3) are
unchanged to the digit, which is what confirms the specialisation did not leak
into the shared `dev_exp`. Cycles move the right way at every sequence length
— softmax 1.039x, 1.034x, 1.011x, 1.029x at S=8/16/32/64 — but the untouched
control stages in the same runs scatter 0.979x to 1.018x, so at S=32 the gain
is smaller than the largest control movement and attention's total is flat
(0.999x). **The cycle figure is inside the relink band and is not claimed as a
measurement; the instruction count is exact and is.**

**Why this is kept where `dnn_add_bias` was reverted.** That change was also
bit-identical and also strictly less work, and it was dropped because the block
did not move — the kernel "does two loads and a store per float op; the
arithmetic was hiding underneath the memory traffic the whole time". The
difference is not judgement, it is the first ablation above: removing the
exponential takes 63.3% of softmax away, which a memory-bound kernel cannot do.
Softmax is compute-bound, which is the condition under which removing
arithmetic pays and the condition `dnn_add_bias` did not meet.

**Still on the table and deliberately not taken.** Pass 3 is 8.4% and reads
back what pass 2 wrote; keeping the row in registers would remove that load,
but each lane holds `cols / warp_size` floats — sixteen at S=64 — and 7.27 is
what happens to this toolchain near a register cliff. Folding the normalise
into the output GEMM instead is arithmetically free — scaling `out`'s rows is
eight times less work than scaling `P` at Dh=8 — but it changes what
`grxdnnSoftmaxForward` returns, and that is a public API and a gated kernel.

**The data-type line was wrong about the hardware, in two different ways.**
bf16 does not exist on this tensor unit in any configuration — there is no knob
to enable — so it is struck rather than deferred, and `grxblasTensorType_t` has
no bit for it. int8 does exist but is a build-time option this sysroot does not
turn on, along with tf32, fp8, fp4, int4, WGMMA and the sparse variants. Since
no host-side table can be right about a build-time choice, the answer comes
from the device: `hgemm_tcu_shape` reports the enabled set and
`grxblasGetTensorTypes` returns it, which is the same mechanism that already
reports the WMMA tile shape and for the same reason. `grxblasGemmEx` refusing a
type now names what the device *does* accept — a caller told "no" without being
told what "yes" looks like concludes the whole tensor path is missing.

**Progress — `grxblasGemmEx` takes transposed operands, all four ways.**

Transposing an operand does three things to its DXA descriptor at once, and
they are three faces of one change: the extents swap, the tile extents swap,
and the destination layout flips between the plain and the transposing scatter
— so that both storage orders land on the *same* shared-memory tile the WMMA
fragments read. The kernel swaps its coordinate pair to match, because tile
coordinates are computed per block on the device and the host cannot do it for
it.

The interesting part was the ragged-`k` case. The untransposed GEMM was exact
only by an accident of layout: `k` was an outer dimension of A's descriptor, so
A's tail came back zeroed and every tail term was `0 * whatever-B-read`. The
DXA engine does not pad dimension 0 (cuda_mapping.md 7.14), and transposing an
operand moves `k` between dimension 0 and the outer dimension — TN puts it in
dimension 0 for **both**, and garbage times garbage is garbage.

So the kernel now zeroes the staged tail itself, on the one step that can
overhang. All four combinations are then exact for their own reason rather than
by luck. Confirmed in the failing direction: with the zeroing removed, exactly
the TN ragged-`k` cases fail and NN, NT and TT stay correct — which is what the
layout table predicts, observed rather than assumed.

**Progress — batched GEMM, in one launch rather than a loop.**

`grxblasSgemmStridedBatched` puts the batch on the grid's second dimension, so
the whole batch is one launch. The unbatched entry point is now the same body
with `batchCount = 1` and zero strides, sharing an implementation rather than a
resemblance — a batched GEMM's usual failure mode is that its unbatched twin
drifted.

Strides are in elements and signed, matching cuBLAS, and `strideB = 0`
broadcasts one B across the batch. The gate checks all of that against a
per-matrix reference over the WHOLE buffer, with deliberate slack between batch
members that has to come back untouched. Watched failing: pinning the kernel's
batch index to 0 makes every multi-member case wrong, which is what says the
grid's second dimension is really being read.

**Two things were learned by rebuilding the sysroot.**

`VX_CFG_TCU_INT8_ENABLE` turns int8 on, and `grxblasGetTensorTypes` then
reports `fp16 int8` — so int8 GEMM has something to be gated against, and
`ci/README.md`'s recommended configuration now includes the flag.

`VX_CFG_TCU_WGMMA_ENABLE` does **not** fix the multi-CTA tensor deadlock. That
was the obvious thing to try, and it was tried: a sysroot built with the flag
still deadlocks on a second CTA. The release has two conditions and the flag is
only one of them — the retiring op must also *be* a WGMMA op, which a kernel
issuing plain WMMA never is. `tests/repro/tcu_multi_cta/` and cuda_mapping.md
7.12 now say so, because a bug report that suggests flipping a flag sends
whoever fixes it to the wrong line.

**Progress — `grx::wmma` does int8.**

`fragment<…, int8_t>` with an `int32_t` accumulator, on a sysroot built with
`VX_CFG_TCU_INT8_ENABLE`. The tile is **8x4x16**: same m and n as fp16, twice
the depth, because a 32-bit register holds four int8 where it holds two fp16.
`tests/kernels/wmma/` asks the device for both shapes and checks that
relationship rather than assuming it — a kernel that sizes a staging buffer
from one shape and indexes it with the other is wrong in a way that only shows
up on ragged edges.

The change to the header was four lines, and the reason is worth writing down:
`wmma_context` carries every fragment in FLOAT registers whatever the format
(`vreg_t = float`, and the MMA instruction pins its operands to `f0`-`f7`), so
an int8 fragment is a bit pattern in a float exactly as a packed fp16 pair
already was. `fragment::x[]` did not need a storage-type parameter, because the
storage was never really float.

The int8 arithmetic is checked against an integer reference with **no
tolerance at all** — not even the "chosen so it is exact" kind the fp16 gate
needs. Watched failing: declaring B `row_major` instead of `col_major` makes all
32 elements wrong.

**Progress — int8 GEMM through `grxblasGemmEx`, and phase 3 v1 is done.**

`Atype = Btype = GRX_R_8I` with `Ctype = GRX_R_32I`, all four transpose
combinations, ragged shapes and padded leading dimensions, checked against an
integer reference with no tolerance at all.

The tensor GEMM kernel is now one templated body instantiated twice rather than
two files. The two pairings are the same algorithm over different tiles — int8
is 8x4x16 where fp16 is 8x4x8 — and everything that genuinely differs (the
tile, the fragment types, the staging sizes, how the epilogue scales) is
derived in one `cfg<In, Acc>` at the top of the file. Two files would have let
the measured kernel and its sibling drift apart with nothing noticing.

`alpha` and `beta` stay floats, because one signature cannot carry two scalar
types. For the integer pairing they must *be* integers: `2.5f` is refused
rather than rounded, since rounding it would be a wrong answer the caller never
sees happen.

The library asks the device for the int8 geometry separately instead of scaling
the fp16 one, and checks that the two tiles share m and n — if they ever stop
sharing it the blocking scheme is not shared either, and this is the place that
would notice.

**What remains of phase 3, and why it is not being done now.**

*Autotuned tile selection.* The tuning sweep in `kernels/hgemm_tcu.cpp` was run
and recorded: double buffering came out 21% worse, and 1x2, 2x1 and 2x2
blocking 8-23% worse, so 1x1 single-buffer shipped. An autotuner needs a set of
candidates worth choosing between; today the search has one winner and the
sweep is in the file. It becomes worth building when a second configuration
wins somewhere — most likely once the multi-CTA deadlock lifts and the launch
shape stops being fixed.

*WGMMA warp groups.* A build-time option, and turning it on does not help yet:
the multi-CTA tensor deadlock survives it (cuda_mapping.md 7.12), so a
warp-group kernel would still be confined to one CTA.
`tests/repro/tcu_multi_cta/` is the standing watch.
---

## Phase 4 — `grxcc` single-source driver (≈5–6 engineer-months)

**Scope.** The piece that makes GRXCP a programming platform rather than a
library binding.

- Driver skeleton (orchestrator first — see architecture §11 open question 1):
  split flags, run the device pass through clang + VOLT, run `vxbin.py`, run
  the host pass, link.
- `<<<grid, block, smem, stream>>>` rewriting to
  `__grxPushCallConfiguration` + stub.
- `.grxfat` fat binary emission into `.grxfatbin`;
  `__grxRegisterFatBinary` / `__grxRegisterFunction` constructors.
- Kernel parameter descriptors and the packing ABI — including the **rv32
  pointer-width narrowing** the chipStar path could not fix.
- `__launch_bounds__`, per-kernel register metadata into the `.vxbin`
  footer, which turns `grxFuncGetAttributes.numRegs` from -1 into a number.
- Host target matrix: `x86_64-linux-gnu` and `riscv64-linux-gnu`.

**Exit gate.** A single `.grx.cpp` file containing `__global__` kernels and
`<<<>>>` launches compiles with `grxcc` and runs correctly on `simx` and
`rtlsim`; at least ten CUDA samples compile **unmodified** except for the
`grx_cuda_compat.h` include; the conformance pass rate improves measurably
over Phase 1's published number.

**Risk.** This is the largest single phase and the one most likely to slip.
Mitigation: the orchestrator design means the risk is integration work, not
compiler research — VOLT already lowers SIMT kernels; `vxbin.py` already
builds multi-entry binaries; the CUDA host-side lowering pattern is public
and well understood.

**Progress — the first increment is closed.** `tools/grxcc` is an orchestrator:
it locates `__global__` definitions and `<<<>>>` launches by text, emits a
device pass and a host pass, shells the device pass out to `ci/build_kernel.sh`
(the same recipe the library kernels use, deliberately — a kernel built by
`grxcc` and one built by hand must come from one set of flags), embeds the
`.vxbin` as a fat-binary array, and links. `tests/grxcc/vecadd.grx.cpp` compiles
and runs correctly on `simx`; it is the PHASE 4 GATE in `ci/run_real.sh`.

Three things that had to be settled to get there, each now a rule rather than an
accident:

- **The device pass stops at the last kernel.** A frontend compiles the whole
  file for the device and simply never diagnoses host-only bodies it does not
  need. An orchestrator has no such option — `main`'s `std::printf` is an error
  for a bare-metal target whether or not the device would run it. So "device
  code first, host code after" is a rule of `grxcc`, and it fails visibly (an
  undeclared name at the device compile) rather than quietly.
- **The device header goes in just before the first kernel, not at the top.**
  It defines `printf` and `assert` as macros, which is what lets a kernel call
  them, and those macros poison `<cstdio>` if the standard header is parsed
  afterwards.
- **Registration state cannot be a file-scope global.** `__grxRegisterFatBinary`
  runs from a static initializer in the user's translation unit, and the order
  against the runtime's own globals is the linker's choice. The first program
  `grxcc` built segfaulted in `_Rb_tree_decrement` before printing anything;
  `src/runtime/module.cpp` now holds those tables in function-local statics,
  which also fixes the mirror-image problem at exit.

That work also surfaced a silent-deadlock defect in the toolchain that had been
invisible for three phases, because every kernel gate before it launched one
warp per CTA over an evenly-dividing grid: `__syncthreads()` is duplicated
across a divergent branch and a diverged warp arrives at the barrier twice. See
`cuda_mapping.md` section 7.20 and `tests/repro/barrier_duplication/`.

**`__launch_bounds__` and register metadata are done, with one correction to
the plan.** The scope line said "per-kernel register metadata into the `.vxbin`
footer". It does not go in the footer: `grxcc` measures the count from the
device ELF and emits it into the kernel descriptor the host TU already carries,
which needs no format change and works for any image `grxcc` builds. A `.vxbin`
loaded by `grxModuleLoad` still reports -1, correctly — nobody measured it.

Both landed with the semantics stated rather than assumed. `maxThreadsPerBlock`
is enforced; `minBlocksPerMultiprocessor` has nothing to trade against on this
hardware and draws a note instead of silence; and `numRegs` does not bound
occupancy here the way it does on CUDA, because the CTA dispatcher does not gate
admission on register count. `cuda_mapping.md` section 7.21.

**The ten CUDA samples compile unmodified, and run.** `tests/cuda_samples/`
holds eleven programs written as CUDA is written — `cudaMalloc`, `__shared__`,
`__syncthreads`, `__shfl_down_sync`, `cooperative_groups`, streams and events —
whose only concession to GRXCP is including `grx_cuda_compat.h` instead of
`cuda_runtime.h`. No `grx*` name appears in any of them.

**The first pass failed eleven times out of eleven,** and the rule was that the
platform changed and the samples did not. What that produced:

- **Static `__shared__` works** (five of eleven needed it). `grxcc` collects a
  kernel's declarations into a struct over the CTA's local-memory slot and puts
  its size in `static_smem` — a descriptor field the runtime had been adding to
  `lmem_size` since phase 1 that nothing had ever set. Gap register 7.10 is
  closed.
- **`grx_atomic.h`**, refusing by name on a build without the A extension rather
  than leaving `atomicAdd` undeclared — or worse, emitting an AMO the simulator
  aborts on with no message (7.16).
- **The device pass supplies what a CUDA frontend supplies**: `grx_warp.h`,
  `grx_cg.h`, `grx_atomic.h`, `warpSize`, `<cooperative_groups.h>`, `<math.h>`
  through the compat header, and `__device__`-only functions dropped from the
  host pass the way `nvcc` drops them (7.22).
- **The `printf` poisoning is fixed at its source** rather than dodged by
  insertion order, which stopped working the moment a file's own include order
  put a GRX header above `<cstdio>`.

**Two gates moved when the device did, and both were the device telling the
truth.** The samples needed a configuration that can run standard CUDA block
sizes, so the sysroot went from 4 warps per core to 16 —
`maxThreadsPerBlock` 16 → 64.

- The **prof gate** required `cycles[1024]/cycles[256]` to land in 2–5× and read
  1.52×. That band was calibrated to a 4-warp core: a launch costs thousands of
  cycles whatever the grid is, and a wider core absorbs the extra work into
  parallelism. It now measures the **marginal** cycles per element, which a
  fixed overhead cancels out of — 8.43 then 8.32 on the wide core.
- The **phase 3 exit gate** read 4.80× at 16×16×16 against a 5× threshold. Cause:
  the tensor kernel's parallelism is bounded by its output tile count (8 tiles
  for 16 warp slots) while sgemm has one thread per element and saturates. The
  threshold now applies where both kernels fill the core, and the starved shapes
  are printed with their speedups and the reason rather than dropped. The tile
  bound is itself a consequence of the single-CTA workaround for the tensor
  unit's multi-CTA deadlock, so it goes when that is fixed.

**The host target matrix is done, and it runs rather than only compiling.**
`ci/build_mock.sh --host riscv64-linux-gnu` cross-builds the runtime, the tools
and the unit tests and executes them under qemu-user; all five unit tests pass
and `grx-smi` reports the same device. `ci/run_real.sh`'s HOST MATRIX GATE
compiles a grxcc program's host pass for the same triple. That the gate has
teeth was checked rather than assumed: a `__builtin_ia32_rdtsc` planted in
`event.cpp` passes the native build and fails the riscv64 one.

Compiling was never the interesting half. A cross compile catches inline asm and
`__x86_64__` ifdefs; only an execution catches a struct laid out differently or
an alignment fault, and the GRX930 is the machine this runtime is ultimately
for.

**The conformance number moved, and the four entry points behind it work.**
50 of 82 (61%) to **54 of 83 (65%)**, from `cudaMemcpyToSymbol`,
`cudaMemcpyFromSymbol`, `cudaGetSymbolAddress` and `cudaGetSymbolSize`.

`grxcc` already read the device ELF for register counts; it now reads the symbol
table too and registers `__constant__` and `__device__` variables by address and
size. The runtime writes them by editing its own copy of the image and reloading
the module, because the driver gives the host no handle for a loaded module's
memory — measured, not assumed: `vx_buffer_reserve` refuses any range that
overlaps it.

That mechanism is exact for `__constant__`, which the device cannot write, and
unsound for `__device__`, which it can — so reading back a `__device__` symbol
is **refused** rather than answered with a value that was true before the kernel
ran. `tests/cuda_samples/12_constant_memory.cu` gates both, and the reload path
was watched failing with the invalidation removed: the first write still landed
and the second silently did not. `cuda_mapping.md` section 7.23.

**Phase 4's exit gate, clause by clause.**

| clause | state |
|---|---|
| a single `.grx.cpp` with `__global__` and `<<<>>>` compiles with `grxcc` and runs correctly on `simx` | **met** — PHASE 4 GATE |
| ...and on `rtlsim` | **split.** `grxcc_vecadd`, a single-source `.grx.cpp`, runs correctly. The unmodified CUDA samples do NOT: `threadsPerBlock = 32` against this config's `maxThreadsPerBlock = 16`, refused by name as "launch exceeds a per-core resource bound". A block size written as a constant is a constant about one machine — cuda_mapping.md 7.36 |
| at least ten CUDA samples compile unmodified except for the `grx_cuda_compat.h` include | **met** — eleven do, and run |
| the conformance rate improves measurably over phase 1's published number | **met** — 61% to 65% |

**`rtlsim` HAS NOW BEEN RUN.** This paragraph used to read "has not been run,
and cannot be from this checkout" — the grxgpu working copy in the build
container has two files under `hw/`, `VX_define.vh` and `VX_gpu_pkg.sv`, and the
build stopped at the missing DPI source.

That was true of *a* checkout and false of the project. The GRXGPU tree on the
development machine has the whole thing: 404 RTL files, `hw/dpi/dpi_util.cpp`,
`sim/rtlsim/` and `sw/runtime/rtlsim/`. Staged across and built with the system
Verilator 5.020, `librtlsim.so` and `libvortex-rtlsim.so` link and run.

The blocker was a claim about a checkout that read like a claim about the
platform, and it stood for as long as nobody looked in the other tree. See
cuda_mapping.md 7.36 for what it then found.

Worth being clear about what running it would add. GRXCP's code path is
identical on either backend — the driver is selected by `VORTEX_DRIVER` and the
runtime never branches on it — so `rtlsim` would substantiate the RTL rather
than GRXCP. That is a real check and it belongs in the gate; it just belongs to
whoever has the RTL. The one place GRXCP itself would notice is
`grxDeviceProp_t::backend`, which every timing claim in this project is already
required to report.

---

## Phase 5 — Concurrency and asynchrony (≈3 engineer-months, gated externally)

**Status: still gated, and now measured rather than assumed.** The gate was a
sentence in `src/runtime/stream.cpp` — "the driver serializes launches" — that
nobody had re-checked since it was written, and an entire phase rested on it.
`tests/repro/stream_overlap/` now tests it directly: two kernels rendezvous
through a device global, one per stream, and only a **mid-spin** sighting proves
overlap. On simx, across trials, zero overlapped. The gate is real.

It also turned up something the sentence did not say: **independent streams have
no order either.** About a third of trials run the second stream's kernel first,
because two streams are two driver worker threads racing to submit. That is not
a defect — CUDA promises no ordering between independent streams — but it is
true *today*, not only after concurrency lands, and a first version of the
repro mistook it for overlap half the time. `cuda_mapping.md` section 7.3 has
the reasoning and the two controls the repro carries.

The repro is a standing watch in `ci/run_real.sh`, so CI reports the day a
mid-spin sighting appears and this phase opens.

**Scope.** Make streams mean something physically.

- Consume the GRX-G100 QMD-style atomic `CMD_LAUNCH` and multi-queue CP work
  (this phase cannot start before that lands).
- Multi-queue stream scheduling in `libgrxrt`; retire the launch
  serialization.
- Device-side event timing once CP profiling writeback exists; drop the host
  clock fallback and flip `eventTimingIsDeviceSide`.
- Copy/compute overlap: `grxMemcpyAsync` on a non-null stream genuinely
  overlapping a kernel on another stream.
- Interrupt-driven completion instead of `Q_SEQNUM` busy-polling, if the
  GRX-G100 interrupt path lands (§10 item 10).

**Exit gate.** A two-stream ping-pong benchmark shows measurable overlap
against the single-stream baseline on `simx` cycle counts, and stream
semantics remain correct under the conformance suite.

---

## Phase 6 — Library breadth and advanced memory (≈6 engineer-months)

**Scope.** Fill out the platform.

- grxDNN (conv2d via implicit GEMM, pooling, softmax, layernorm, GELU,
  fused attention), grxFFT, grxRAND, grxSPARSE, `grx::par`.
- `__constant__` — either the real broadcast path or the documented
  read-only-global lowering, decided by then.
- L1/shared carve-out attribute wired to the DCR, once the G100 unified
  carve-out lands (that project's Phase 2).
- `grx::tex<>` — texture/surface access from compute.
- `grxrtc` runtime compilation.
- Promote `grxcc` to a proper Clang `ToolChain`.

**Exit gate.** A transformer inference workload (attention + GEMM +
layernorm + softmax) runs end-to-end through GRXCP libraries with numerical
agreement against a PyTorch CPU reference; **API coverage reaches 57 of 83
tracked entry points (69%)**, which is every one this phase's scope can move.

*(This clause used to read "conformance pass rate hits the target set at Phase
4". Phase 4 set a direction — "improves measurably over Phase 1's published
number" — and no target, so the reference dangled. The number above is derived
rather than invented: of the 29 entry points not implemented at 54/83, exactly
**three** are in Phase 6's scope — `cudaCreateTextureObject`,
`cudaDestroyTextureObject` and `cudaMallocArray`, which `grx::tex<>` would
supply. The rest belong to Phase 5 (streams), sit behind hardware, or are
deliberately out of scope for v1 (graphs, IPC, interop — `cuda_mapping.md`
7.15). A phase whose scope is library breadth barely moves a CUDA **runtime**
API count: grxFFT and grxSPARSE are not entry points in this table.)*

**BOTH CLAUSES ARE NOW MET.** The block ran end to end against PyTorch when
attention landed; `grx::tex<>` supplied the three remaining entry points and
the published number is **57 of 83 (69%)**.

The three are **PARTIAL, and the distinction is load-bearing.** The TEX units
are driven by the graphics path and are not reachable from compute
(`cuda_mapping.md` 7.8, still open), so `grx::tex<>` addresses and filters in
SOFTWARE, in the calling warp. That is architecture section 10 rule 5's
sanctioned exception and nothing wider: an emulation reported through a device
property, exactly as the warp-shuffle fallback is.
`grxDeviceProp_t.textureIsEmulated` reads 1, `grx-conform` prints it, and the
TEXTURE GATE fails if it ever stops. A phase gate closed by counting entry
points that quietly pretended to be hardware would be the worst outcome
available here, so the flag is part of the gate rather than a footnote to it.

Two more differences keep them at PARTIAL rather than MAPPED: filter weights
are full-precision float where NVIDIA quantizes the fraction to 8 bits, and the
channel formats are `float` and `float4` only — integer formats with normalized
reads fail to compile rather than reading garbage.

Building it found a toolchain defect nothing else could have:
`__builtin_floorf` and four sibling rounding builtins do not compile on a
**divergent** value for this device (`cuda_mapping.md` 7.24). No device code in
the tree had ever needed one.

**Progress.** grxDNN v0 has landed: `grxdnnSoftmaxForward`,
`grxdnnLayerNormForward` and `grxdnnAttentionForward`, fp32, forward only,
row-major. The norms are one warp per row, checked against a host reference on
five shapes with two controls that are themselves verified to discriminate
(`tests/libs/test_grxdnn.cpp`). No conv, no backward pass, no attention
*fusion* — absent rather than stubbed, so a port that needs them fails to
compile.

**Attention closes the exit gate's missing quarter**, and it is the first op
that has to cross between the two libraries' opposite conventions: grxDNN is
row-major, grxBLAS is column-major, and attention presents row-major tensors to
a column-major library twice, once transposed. No data moves — a row-major
(r, c) matrix with leading dimension ld *is* the column-major (c, r) matrix over
the same bytes — so the transposes live in the arguments, which is exactly the
kind of code that produces plausible numbers and wrong answers.

It is therefore gated against **PyTorch**, not against a reference written from
the same reasoning: `tests/libs/attention_ref.py` generates vectors from
`torch.nn.functional.scaled_dot_product_attention` in float64 and checks them
in, so CI needs no torch. The same script simulates the exact `grxblasSgemm`
calls the implementation makes — leading dimensions, transpose flags, flat
memory — and refuses to emit the vectors unless that simulation reproduces
torch to 1e-12, so the layout algebra was settled before a device ran any of
it. It then passed on the device first try.

Watched failing three ways. Flipping `transa` fails everything but the 1×1
case, caught by grxBLAS's own leading-dimension check. Passing Q and K in the
order the formula reads — dimensionally valid, silently computes scoresᵀ —
fails every non-trivial case numerically by 0.117 to 0.316; that is the mistake
a careful person makes and nothing but an outside reference catches it. Removing
the causal mask fails *only* the two causal cases, so the mask does real work
and the unmasked cases are not accidentally masked.

Landing it turned up the part of the exit gate nobody had costed: the exit
gate names **two libraries in one process**, and that did not work. Every
`.vxbin` links at `STARTUP_ADDR`, so the second library to initialise could
not load its kernels. `src/libs/kernels_all.cpp` puts both libraries' entry
points in one image, and that alone was still not enough — both libraries
call `grxModuleLoad` on that one file, and the second call overlapped the
first. The runtime now hands back an already-resident image with a reference
count. Gated both ways in `ci/run_real.sh` (CROSS-LIBRARY GATE), and written
up in `cuda_mapping.md` 7.13.

**The exit gate's first clause is MET.** A whole pre-norm transformer block —
layer norm, QKV projection, causal attention, output projection, residual,
layer norm, MLP with GELU, residual — runs end-to-end through GRXCP libraries
and matches a PyTorch block at **every one of its twelve stages**, on three
shapes including a causal one. Worst deviation anywhere is 8.05e-07.
`tests/libs/test_grxdnn_block.cpp`, PHASE 6 EXIT GATE in `ci/run_real.sh`.

Two ops were added to get there — `grxdnnGeluForward` (both the erf and tanh
forms, because a model needs the one it was trained with) and
`grxdnnAddBiasForward`. The residual needed nothing new: it is `grxblasSaxpy`,
already gated.

**No permute anywhere, and that is a result rather than an optimisation.**
torch projects to [S, D], reshapes to [S, H, Dh] and permutes to [H, S, Dh] for
attention, then permutes back. GRXCP has no permute op and does not need one:
head *h*'s projection weights are a column block at offset `h*Dh` with the same
leading dimension, so one strided-batched GEMM lands directly in [H][S][Dh] —
attention's layout — and coming back out, head *h*'s slice of `Wo` is a row
block, so the output projection is one accumulating GEMM per head. Every GEMM in
the block is `N, N` with the operands swapped; attention is the only op that
still needs an explicit transpose flag, and it owns it.

The gate compares **stage by stage and stops at the first disagreement**, so a
failure names the op rather than reporting that a ten-stage block is wrong
somewhere. Watched failing two ways, and both are worth the space:

- Making the output projection overwrite per head instead of accumulating leaves
  the H=1 case passing *entirely* — correctly, there is nothing to accumulate
  with one head — and fails both H=2 cases at exactly `p`, by 0.283 and 0.495,
  with every earlier stage green. That is a hand-off bug between two ops that
  are each correct and each individually gated, which is precisely what a
  composition gate is for.
- Substituting the exact GELU where the weights expect the tanh form fails at
  exactly `act`, by 2.33e-04. This one found a defect *in the gate*: the
  original tolerance was three orders of magnitude looser than anything observed
  and let the wrong activation through on two of three cases. It is now set from
  the measurement.

**The block has now been PROFILED, and the profile contradicts the obvious
optimisation.** `tests/bench/block_cycles.cpp` reports device cycles per stage.
One SM, four lanes, S=8 D=16 H=2 F=64, **against the naive sgemm** — this is the
baseline the tuning below is measured from, not the current state:

| stage | share | notes |
|---|---|---|
| mlp GEMM 1 (D→F) | 27.7% | |
| mlp GEMM 2 (F→D) | 23.0% | |
| qkv projection (3 GEMMs) | 20.7% | |
| **GELU** | **11.5%** | |
| attention | 4.5% | 8.5% at S=16 — its scores are seqLen-squared |
| output projection | 4.3% | |
| layer norm ×2 | 5.0% | |
| bias ×3 | 2.1% | |
| residual | 1.1% | |

Every share above is derived from `ci/perf/baselines/block_cycles.naive.json`,
which holds the raw spans and is gated exactly by the PERF BASELINE GATE. Four
of these entries were off by a tenth of a point and one — layer norm — by three,
against a table nobody could check when it was written. They are recomputed now,
and a reader who does not believe them can divide the numbers in that file.

Attention's figure was wrong when first taken, and the correction is worth
recording. `grxdnnAttentionForward` calls grxBLAS on an internal handle nothing
probed, so its **two GEMMs were missing from the profile entirely** and the
block's GEMM share was understated. Attaching a probe was not enough either:
attention is four launches, the slot index comes from the block and warp, and
four different grids collide at the low indices — the earliest start is simply
lost. Each launch now writes its own region of the probe buffer.

Giving each launch its own region was still not enough, and the rest of that
story is below under **the instrument was broken**: the four regions were then
*spanned*, and `VX_CSR_MCYCLE` restarts at zero at every launch, so the result
was a maximum over four unrelated clocks. Each launch is now summarised against
its own clock and the spans are added.

Attention's share goes 14.19% → 20.34% when the sequence doubles — 36749/259043
against 103692/509808, both from the baseline file. In absolute cycles it grows
2.82× while the block grows 1.97×, which is the direction a seqLen-squared stage
among linear ones must move and short of the 4× a saturated machine would show,
because at S = 8 most of these launches do not fill the core. Two earlier
versions of this paragraph quoted 1.86 and 1.93 from numbers that were spans
across launches; they are history and do not reproduce from this tree.

**GEMMs are 77% of the block** — 76.8% at S = 8 and 78.1% at S = 16 across every
stage that carries one, attention's own two included, which is the number that
matters and the one the profile hole was hiding. That is where the work is, and `grxblas.h`
already says why: sgemm v0 is one thread per output element with no blocking,
correct and not fast. Tuning it is worth more than everything else on this list
combined.

**Done, and it is worth 1.34× on the whole block.** `sgemm_rb` is a register-
blocked kernel: one thread owns RM = 4 outputs down a column, so B is loaded
once per four multiply-adds instead of once each, and the loop overhead and
index arithmetic are amortised over four outputs. A **separate entry point**,
not an edit of the reference, which is what `kernels/sgemm.cpp` said the tuned
path would be — and that turns the reference into an **oracle**: both kernels
run on the device over the same operands and must agree **bit for bit**, since
blocking changes which thread computes what, not the order of the accumulation.
`tests/libs/test_grxblas_rb.cpp`, 36 shape × transpose combinations, `==` with
no tolerance to hide in.

| stage | m | k | naive → blocked | |
|---|---|---|---|---|
| mlp GEMM 2 | 16 | 64 | 70031 → 43253 | 1.62× |
| mlp GEMM 1 | 64 | 16 | 84379 → 53614 | 1.57× |
| qkv projection | 8 | 16 | 62815 → 45236 | 1.39× |
| output projection | 16 | 8 | 25960 → 19984 | 1.30× |
| attention (4 launches) | 8 | 8 | 42267 → 36749 | 1.15× |
| **whole block** | | | **345482 → 259043** | **1.33×** |

Both columns are the shipping build, and every number in them is a sum of
per-launch spans from `ci/perf/baselines/block_cycles.{naive,register-blocked}.json`,
gated exactly. Two rows here have been wrong before and it is worth saying how,
because the same defect produced both. The attention row once read `13834 →
13861` and `1.00×`, and the output-projection row once read `13056 → 9987`.
Neither was a cost. Attention is four launches and the output projection is
H = 2, and both were being read as a single span across the whole probe buffer.

**The instrument was broken, and it changed an engineering decision.**
`VX_CSR_MCYCLE` restarts at zero at **every launch**: SimX's
`ProcessorImpl::run()` opens with `reset()`, which assigns a fresh `PerfStats`,
and MCYCLE reads `PerfStats::cycles`. Three stages sampled from very different
points in the block all report their first warp starting at ~4900 cycles. So a
span taken across two launches is a maximum over two clocks that both began at
zero — and it looks exactly like a duration.

The symptom was in the data the whole time and nothing was looking at it:
summarising attention's buffer reported **64 warps live at once on a device that
holds 16**. `grxCycleSummary` now carries `maxLive`, `block_cycles.cpp` refuses
any span whose `maxLive` exceeds `maxWarpsPerMultiProcessor ×
multiProcessorCount` and says why, `grxdnnGetCycleRegions` reports where each of
attention's launches wrote, and every multi-launch stage is measured one launch
at a time and summed. `tests/unit/test_cycle_summary.cpp` is the tier-1 gate —
no device needed, which is the point.

**The sweep was owed and has now been done, twice: alone and in place.**
`tests/bench/sgemm_sweep.cpp` runs 66 shapes across m, n and k with both kernels
over the same operands, plus batch and transpose. SGEMM CROSSOVER GATE in
`ci/run_real.sh`.

What it found, in order:

* **k never decides anything.** Across the whole swept range it moves the
  magnitude — 1.17× at k=4 against 1.53× at k=32, same m and n — and never
  changes which kernel wins. The `k >= 16` clause was not doing what its comment
  said it was.
* **The coalescing boundary is not at m = 16.** m = 8 wins at n = 16 by 1.27×.
* **What predicts the isolated GEMM is the output count.**
  `m*n*batch >= 2 × resident threads` explains **all 66 cells with no
  exceptions**, where the old rule was wrong on 19. Cells with equal m·n agree
  to two decimals however m and n split — m=4,n=16 and m=8,n=8 and m=16,n=4 all
  read 0.90 — and batch scales it exactly: m=n=8 at batch 2 reads 1.43, and the
  unbatched m=8,n=16 cell, the same 128 outputs, reads 1.44. The mechanism is
  that the blocked kernel produces the same outputs with a quarter of the
  threads, so below saturation it just idles the core.

**It was shipped, reverted, and shipped again.** The revert is the part worth
keeping on the record. Shipping the output rule appeared to make the block
slower — 230171 cycles against 226405 at S = 8, with attention's two GEMMs 27.6%
worse in place while winning 1.39× in isolation — so the losing rule was kept
and its comment was left saying nobody knew why an isolated sweep disagreed with
the workload. Nothing disagreed. Both of those numbers were spans across
attention's four launches.

**Measured in place, one call at a time.** `ci/sweep_block_sgemm.py` flips
exactly one of the block's sgemm calls to the other kernel and runs the whole
block again — for every call it makes, at both sequence lengths, with nothing
else moving. That is the sweep this section used to say was owed, and it is
finer than what was asked for: per *call*, not per stage, because shape cannot
tell the block's stages apart. At S = 8 attention's two GEMMs are both 8×8×8,
and at S = 16 the qkv projection and attention's output GEMM are both 8×16×16.
Calls are selected by index through a measurement hook in `grxblas.cpp`, and
`GRXBLAS_SGEMM_TRACE` records what the library actually did rather than what the
caller asked for.

The result, on all 18 flippable calls at both shapes: **every one of them is
slower with the other kernel.** The rule is right everywhere the workload goes,
and forcing the blocked kernel on attention's two GEMMs — the change that was
reverted — **saves 5990 cycles**, 2613 and 3377, on a 259043-cycle block. The
two flips compose exactly. BLOCK SGEMM IN SITU gate in `ci/run_real.sh`, 112
seconds, and it fails on the old rule naming those two calls.

**The 2D micro-tile is built, and it is the largest single win here: 1.60× on
the whole block at S = 8, 1.74× at S = 16.** `sgemm_2d` gives one thread a
2 × 2 patch of C and reuses *both* operands, so a k step costs RM + RN loads for
RM × RN multiply-adds — 4 loads per 4 against `sgemm_rb`'s 5.

| stage | naive → rule | |
|---|---|---|
| mlp GEMM 1 | 84465 → 39634 | 2.13× |
| mlp GEMM 2 | 69681 → 33370 | 2.09× |
| qkv projection | 63468 → 33673 | 1.88× |
| output projection | 25985 → 16052 | 1.62× |
| attention (4 launches) | 42223 → 33420 | 1.26× |
| **whole block** | **346150 → 216800** | **1.60×** |

**The tile is 2 × 2 and not 4 × 4 on purpose, and that is the whole experiment.**
2 × 2 produces the same four outputs per thread that `sgemm_rb` does, so at any
shape the two launch the *same number of threads* and the only difference
between them is the load count. That mattered because the in-situ sweep had
already established that what decides between blocked and reference on this
device is how many threads are left running — a wider tile would have changed
the load ratio and the occupancy together, and a measurement of the two at once
cannot say which moved. Holding occupancy fixed, the 2D tile is faster on **42
of 42** swept cells (1.18× to 1.39×) and on **36 of 36** in-situ flips. So: at
equal occupancy, the load count is what costs.

`sgemm_rb` stays. It is the control — without a kernel that differs in exactly
one variable, "the 2D tile wins because it loads less" is an argument rather
than a measurement — and both sweeps compare against it every run.

**The rule moved with it.** The crossover is now bracketed in steps of 8
outputs, which is what the rule before last lacked:

| outputs | 32 | 40 | 48 | 56 | 64 | 72 |
|---|---|---|---|---|---|---|
| naive / best blocked | 0.63 | 0.75 | 0.88 | **1.04** | 1.17 | 1.68 |

The crossover sits inside (48, 56]. The threshold ships at `resident` = 64
rather than at 56, because 56 is not a number the device reports and a fitted
constant is what went wrong last time; the cost is visible and small, since the
one band it declines wins by 3–4% while the band below *loses* by 12%. And the
ratio depends only on the output count, not on how m and n split to reach it —
4×12, 8×6 and 12×4 all read 0.88, 0.91, 0.88.

**A hypothesis about the knee, recorded as one.** The jump from 1.17 to 1.68
between 64 and 72 outputs sits exactly where the reference kernel stops fitting
in the machine: it needs one thread per output, the core holds 64, so at 65 it
needs a second wave while the blocked kernels, at a quarter of the threads, are
still inside one. That predicts another jump at 136 and none between 112 and
128. Measured: 1.87, 1.73, 1.84 across 112–128 and 2.12 at 136 — the jump is
there, it is much smaller, and 144 falls back to 2.00. The wave account fits the
first knee well and the second only partly. It is written down as what it is,
and the rule does not depend on it.

**And it broke the phase 3 exit gate**, which is a ratio against sgemm. See
phase 3 above: the threshold was not moved.

**The host no longer guesses either kernel's tile.** `kSgemmRowsPerThread = 4`
carried a comment asking whoever edited `kernels/sgemm.cpp` to keep it in step,
and nothing checked. A `sgemm_shape` entry point now reports what each blocked
kernel produces, the way the tensor path already did; a module that will not say
gets the reference kernel and nothing else, because a host that guesses a tile
launches a grid covering the wrong number of outputs — silently, in whichever
direction the two drifted.

**Then the wider tile, and it taught the opposite of what it was built to
teach.** 4 × 4 was built first because it has the best arithmetic on paper — 8
loads per 16 multiply-adds against 2 × 2's 4 per 4. It lost. The disassembly
said it spilled: seven stack accesses inside the k loop, handing back exactly
the advantage. 4 × 2 was then built to test whether register pressure was the
reason, with two distinguishable outcomes; it did not spill, it landed at the
0.75 memory operations per multiply-add the arithmetic asks for, and it won
above 16 × resident outputs. A second threshold shipped there.

**It was wrong, and one look at the inner loop said why.** The k loop indexed
its operands the way the reference does:

```
a[i] = ta ? A[l + row[i] * lda] : A[row[i] + l * lda];
```

`ta` is loop-invariant and the compiler re-decided it every iteration. The
shipped 4 × 2 inner loop was **64 instructions for 8 multiply-adds**: 6 loads, 8
multiply-adds, **18** czero/or conditional selects choosing between the two
address expressions, and **12** slli/srli pairs re-widening a 32-bit index into
a 64-bit offset. Thirty of sixty-four instructions were neither arithmetic nor
memory. A wider tile was winning by spreading that overhead across more outputs.

Walking pointers instead — the step is `lda` or `1` depending on the transpose,
decided once before the loop — changes the picture completely:

| kernel | tile | loop ins | fp | loads | stack | ins/FMA | mem/FMA |
|---|---|---|---|---|---|---|---|
| `sgemm` | 1×1 | 24 | 1 | 2 | 0 | 24.00 | 2.00 |
| `sgemm_rb` | 4×1 | 53 | 4 | 5 | 0 | 13.25 | 1.25 |
| `sgemm_2d` | 2×2 | **14** | 4 | 4 | 0 | **3.50** | 1.00 |
| `sgemm_4x2` | 4×2 | **23** | 8 | 6 | 0 | 2.88 | 0.75 |
| `sgemm_4x4` | 4×4 | **35** | 16 | 8 | **0** | 2.19 | 0.50 |

The 2 × 2 loop went 42 → 14 instructions, the 4 × 2 went 64 → 23, and **4 × 4
stopped spilling entirely** — the address arithmetic was what would not fit in
registers, not the sixteen accumulators. With the overhead gone the wider tiles
have nothing left to amortise and pay for their lower occupancy with nothing:
neither beats the 2 × 2 tile on any swept cell, approaching it from below as the
shape grows. **The second threshold was removed after one commit.**

**Whole block: 2.10× at S = 8, 2.20× at S = 16** — 347851 → 165600 and 736743 →
335120, against the reference kernel. 72 of 72 in-situ flips make the block
slower.

**And the threshold did not move, for a reason worth recording.** With the loop
that much cheaper, blocking crosses over at 32 outputs rather than 56 — at
k = 16. At k = 8 the same 32-output shapes *lose* by 11%, all three of them
(4×8, 8×4, 16×2 at 0.88–0.89), and at 64 outputs they win by 1.56–1.62×. So the
crossover is k-sensitive between 32 and 64 and not above it, and "k never
changes which kernel wins" — established over a grid held at n = 16, where every
cell has 64 outputs or more — was never tested near the boundary. `resident` is
the smallest device-reported quantity that is safe at every swept k. The n sweep
now gates that, because the m/k grid structurally cannot.

Five kernels ship and the rule names two. `sgemm` is the ORACLE — every tuned
kernel is compared against it on the device over the same operands and must
agree **bit for bit**. `sgemm_rb` is the control for occupancy: same outputs per
thread as the 2 × 2 tile, so identical thread counts, so the inner loop is the
only difference. `sgemm_4x2` and `sgemm_4x4` are the controls for tile width,
and the sweep now gates the negative claim — that neither beats the 2 × 2 tile
anywhere — precisely because it could come back. A change that makes the inner
loop expensive again would show up there as a wide tile winning, before it
showed up anywhere else.

**And the reference kernel's own loop stays as it is, deliberately.** It is 24
instructions for one multiply-add and the same pointer rewrite would help it —
but it is the ORACLE, and its value comes from being written the obvious way. If
the reference and the tuned kernels shared an addressing idiom, a bug in that
idiom would produce identical wrong answers in both and the bit-exact comparison
would pass. It is also never on a hot path: the rule selects it only below
`resident` outputs, where the work is tiny by definition.

**Where the block's cycles are now, and it is no longer mostly GEMM.** With the
GEMMs three times faster the profile has moved: at S = 8, GELU is the largest
single stage.

| stage | cycles | share |
|---|---|---|
| attention (2 GEMMs + mask + softmax) | 27443 | 17.5% |
| mlp GEMM 1 | 26719 | 17.1% |
| **gelu** | **26590** | **17.0%** |
| qkv projection | 21569 | 13.8% |
| mlp GEMM 2 | 18715 | 11.9% |
| out projection | 10421 | 6.7% |
| layer norm ×2 | 15537 | 9.9% |
| bias ×2, residual | 9632 | 6.2% |

**GELU: 1.34× faster, and the accuracy figures did not move by a bit.** Ablation
first, because the roadmap had asserted the cost was the transcendental without
measuring which part: of the stage's 35573 cycles, removing the whole `tanh`
saved 24012 and removing only the divide saved 2243. So it is the exponential's
polynomial — fifteen operations — and there is no cheaper polynomial at this
accuracy.

The polynomial was not what the kernel was spending its time on. Its column loop
was 86 instructions for 21 float operations, and it carried **twelve `vx_split`
and eighteen `vx_join`** — warp divergence machinery. The cause is worth stating
plainly because it is not obvious and it applies to every kernel in the project:

> **Float selects compile to branches on this toolchain.** The integer ternaries
> in grxBLAS's kernels become `czero`/`or` — conditional moves, no control flow.
> The float ones do not. Rewriting `dev_exp`'s early returns as ternaries left
> the split and join counts at exactly 18 and 18.

Asked for by name, `__builtin_fminf`, `__builtin_fmaxf` and `__builtin_copysignf`
become the single RISC-V instructions `fmin.s`, `fmax.s` and `fsgnj.s`, and the
branches go away: `dnn_gelu` drops from 352 instructions to 229 and from twelve
`vx_split` to **none**. The arithmetic is untouched, and the measured accuracy is
identical to the last digit — 5.36e-07 for the erf form and 4.77e-07 for the tanh
form, at the same two argument values.

Two guards disappeared entirely rather than moving, which is the part worth
keeping in mind when reading the code. `dev_exp`'s underflow return is gone
because clamping at −88 gives k = −127, so the exponent field is zero and the
product is already exactly zero. `dev_tanh`'s saturation return is gone because
`1 − 2/(e+1)` reaches 1.0f in fp32 once the argument passes 9.01 — and between
9.0 and 9.01 the branch-free form is fractionally *more* accurate, since the old
early return snapped to exactly 1.0 where the true value is 0.99999994.

The same rewrite in softmax's row-maximum pass measures at 0.1%, inside relink
noise: that pass is one of three and the rows here are 8 to 16 wide. It is kept
for uniformity and because a per-element branch is a hazard class this project
has been bitten by twice.

**Whole block: 2.17× at S = 8 and 2.24× at S = 16** against the reference
kernel — 339812 → 156626 and 729205 → 325710.

**And then the same lesson a third time, in layer norm.** Its pass-3 element
loop re-tested two null pointers on every element:

```
if (gamma) t *= gamma[j];
if (beta)  t += beta[j];
```

`gamma` and `beta` are kernel arguments — the same for every element, every lane
and every row — and hoisting the tests into four sibling loops made the stage
**9% faster** (7783 → 7049 and 7812 → 7320 at S = 8). Whole block 156626 →
155435, and 329410 → 320073 at S = 16.

**The gate that found it also over-reported, and that is on the record.**
`ci/check_kernel_loops.py` originally picked "the loop with the most float
operations" as the hot one. On that rule it named five kernels as spilling or
diverging, `dnn_layernorm` worst at 7 stack accesses and 4 `vx_split` — and
those instructions exist, but in the *row* loop rather than the element loop the
report implied. A per-row branch and a per-element branch differ by the row
width; saying so is the difference between a census and a scare.

The rule is now the **innermost** loop that does float work. It had to change
anyway: splitting one element loop into four siblings made the old rule pick the
row loop enclosing all four, so it reported more instructions, more spills and
more divergence for a change that made the kernel 9% faster. A heuristic that
reads a genuine improvement as a regression is not measuring what it names.

Corrected, the only divergence left anywhere in the image is one `vx_split` in
each of the two tensor kernels, in a loop with no float work in it.

**Fusing the bias into the GEMM epilogue would save about 2%.** That was the
obvious next optimisation before anyone measured, and the measurement says not
to bother yet — the GEMM itself returned 25% for comparable effort.

**GELU costs more than attention and both layer norms together**, which is
surprising for an elementwise op until the per-element figures are compared:
68 cycles/element against 8.0 for a bias add over the same 512 elements —
roughly 8.5× — while layer norm, with three passes and two warp reductions, is
60. (35057/512, 4118/512 and 7781/128, from the naive baseline file; the bias
figure read 7.8 before there was a file to divide.) The cost is the
transcendental, not the memory traffic. `dev_gelu_tanh` goes through `dev_exp`
(range reduction plus a degree-5 polynomial) once per element. A cheaper direct
rational approximation of `tanh` is a real candidate, and it would need its
accuracy re-measured against PyTorch by the GELU gate before it could ship.

Running the profiler is also what found that **every grxDNN kernel constructed a
`cycle_probe` and never called `finish()`**, so the instrumentation recorded
nothing and reported no error — a silent absence that reads like a device
problem. `grx::cycle_probe` now finishes itself in its destructor and `finish()`
is idempotent, so it cannot be forgotten again and grxBLAS's explicit calls are
unaffected.

What remains for the phase is breadth rather than the gate: conv2d, grxFFT,
grxRAND, grxSPARSE, `grx::par`, `grx::tex<>`, `grxrtc`, and promoting `grxcc` to
a proper Clang `ToolChain`. Fusion is the other direction — bias into the GEMM
epilogue, and a flash-style single-pass attention — each with its own numerical
gate, now that there is a whole-block comparison to hold them to.

---

## Phase 7 — NPU as a second device, and the native host (≈3 engineer-months)

**Scope.** Deliver on the multi-device promise made in the v1 device model.

- `src/backends/npu_c930/`: device enumeration, capability profile,
  MMIO doorbell driver over the c930 NPU register map
  (`CTRL`/`STATUS`/`DIM_M`/`DIM_N`/`DIM_K`/`A_BASE`/`B_BASE`/`C_BASE`),
  DMA via `c930_npu_dma`, completion via `STATUS.DONE` or the AIA MSI.
- `grxblasGemmEx` routing: INT8 GEMM on an NPU device programs the systolic
  array instead of launching a GPU kernel — same API, different backend.
- `libgrxrt` built and tested natively on `riscv64` running on GRX930.
- Heterogeneous dispatch policy: a documented, explicit rule for which
  engine a library call lands on. Not automatic magic — an explicit
  `grxblasSetPreferredDevice`-style control plus a documented default.

**Exit gate.** The same host program, unchanged, runs an INT8 GEMM on the
GPU device and on the NPU device by changing only `grxSetDevice`, with
matching results; `libgrxrt` passes the conformance suite compiled natively
for riscv64.

### Where the NPU actually stands

The backend landed in `src/backends/npu_c930/` — MMIO register map, INT8 GEMM
dispatch, device-table entry with the capability profile section 6 specifies,
`grxblasGemmEx` routing, and a numerical test against a CPU reference. The
shape is right and it follows the spec.

Four claims came with it. Checked, before building on them:

| Claim | State when checked |
|---|---|
| `cmake -DGRXCP_ENABLE_NPU=ON` builds | **No** — and not for any NPU reason. The top-level `CMakeLists.txt` had `if(X) add_subdirectory(y) endif()` on one line, which CMake rejects as a parse error. `cmake` could not configure this project *at all*, with the flag or without it, and no gate had ever run cmake. |
| `grxblasGemmEx` on an NPU device with INT8 | Routing is written and correct, but unreachable: `GRXCP_ENABLE_NPU` was a CMake variable and never a compile definition, so every `#ifdef GRXCP_ENABLE_NPU` block in `context.cpp` — the probe, the device-table entry, the property population — compiled to nothing. The build contained the backend and could not enumerate the device it drives. |
| The DMA fetches A/B, runs, writes C back | A hardware claim this repo cannot check. What it *can* check is whether the host believes it: `npu_c930_gemm` waited for `!BUSY` with no `ERROR` and never read `STATUS.DONE`. A device that ignored every write satisfies both instantly, so the function returned success over a GEMM that never ran, leaving C untouched. |
| The GPU path is unchanged with the flag off | Now true and now gated — both configurations build and run in CI. It was not *checkable* before, since neither configured. |

And one nobody claimed, which was the worst of them: `npu_c930_detect` read
`STATUS` and accepted anything that was not `0xFFFFFFFF` and not above `0x7`.
Its own comment said an absent NPU reads `0x0`. `0x0` passes that test. On any
host where `/dev/mem` opens, GRXCP grew a GRX930 NPU it did not have — and
combined with the missing `DONE` check, that phantom accepted GEMMs and
reported them successful.

All fixed, and each watched failing first:

- Detection is a **write-readback** on a documented R/W register, restoring the
  original value. Unbacked memory reads zeros and a dead bus reads ones; neither
  can return a value it was never given.
- `npu_c930_gemm` requires `STATUS.DONE` and says so by name when it is absent.
- `GRXCP_ENABLE_NPU` becomes a compile definition, and the backend's own test
  targets are configured (`add_subdirectory` was never called on them).
  `enable_testing()` moved to the top level, without which `ctest -R npu_c930`
  answered *"No tests were found!!!"* over two freshly built test binaries.
- The register access goes through injectable read/write hooks, so the backend's
  decisions can be driven through four register models — absent, dead bus, live,
  and accepts-a-launch-and-never-finishes. `test_npu_c930_model.cc`; NPU BACKEND
  GATE in `ci/build_mock.sh`; needs no sysroot and no c930.
- CI asserts that **no NPU is enumerated** on a machine that has none, with the
  flag on. A build flag says what code exists, not what hardware is attached.

**A register model is not hardware.** Nothing above says the c930 works, and no
green run may be reported as the NPU working. What is now true is that the host
side is honest about what it can see, and that it can be exercised without one.

### What is still blocked, and on what

1. **Anything that requires a c930.** `CTRL.START` reaching real silicon,
   `STATUS.DONE` coming back from it, the DMA reading DDR — none of it can be
   gated here. The phase 7 exit gate needs both devices present at once. This
   is the item to hand back: GRXCP needs *access* to hardware or to a simulation
   that answers on the register map, not more host code.

   **Half of that arrived, and it is the half that does not close the gate.**
   The GRX930 team shipped `sim/npu_dpi_shim.c` — a standalone pure-C library
   that answers on the CSR map with no Verilator dependency. It is vendored at
   `third_party/grx930/` and wired up as a fifth register model
   (`test_npu_c930_shim.cc`, `ctest -R npu_c930_shim`); driven through it, the
   full `npu_c930_gemm` sequence completes and every element of C matches a host
   INT8 reference. Five defects were measured on import and are recorded beside
   it, two of which would have made a naive wire-up look green while computing
   nothing.

   It is **not** the simulation this item asks for. It computes the GEMM with a
   C triple loop and its own header says it is not cycle-accurate, so it is a
   model of the *interface*, not of the RTL: it can show that our host drives
   the register map correctly, and it cannot show that the c930 does anything.
   The exit gate is unchanged and still needs hardware. Per `AGENTS.md`, no green
   run through it may be reported as the NPU working.

   What it *does* change is where the next blocker is, and the next blocker is
   ours, not theirs: **there is no NPU memory path at all** — see
   `cuda_mapping.md` 7.28. `grxMalloc` on an NPU device reaches
   `vx_buffer_create(nullptr, …)` and comes back `grxErrorInvalidValue`, so
   `tests/libs/test_grxblas_npu.cpp` could never have passed even with a c930
   attached; it fails at its first allocation, before any register. Nor is there
   a seam to attach a model to the *enumerated* device — the runtime and
   `grxblas.cpp` each hold their own `npu_c930_device_t` and detect
   independently. Those two, plus deriving `p.backend` instead of asserting
   `GRX_BACKEND_SILICON`, are what stand between this shim and an end-to-end
   `grxblasGemmEx` run.
2. ~~**`GRX_CAP_KERNEL_LAUNCH` refusals across the rest of the API.**~~ **Done.**
   `grxModuleLoad` now refuses with `grxErrorNotSupported` on a device without
   the bit, the same error the launch gives — "this device does not do kernels"
   is one fact, and a caller that handles it from one has written the handler
   for the other. It matters that the refusal moved *earlier*: load succeeded,
   `grxModuleGetFunction` succeeded, and the program learned at the third call
   what was wrong with the first. The mistake is `grxSetDevice`, and the report
   now arrives next to it. `tests/unit/test_no_kernel_launch.cpp`, run both
   ways; ablating the check fails exactly the zero-warp case.
3. ~~**Cross-device pointers.**~~ **Done, and it was not what this said.**

   This item read "nothing checks this today". The truth was worse: nothing
   *could*. The mock's `vx_device_open` returned the same `MockDevice` for
   every index, so `GRXMOCK_DEVICE_COUNT=2` gave the runtime two device slots
   over **one** address space and one memory pool. Every cross-device operation
   in CI passed because there was only ever one device.

   Making the mock model N distinct devices — each with its own space, all
   starting at the same base, which is what per-device DDR means — found two
   real bugs before a line of checking was written:

   * **`grxMalloc` on device 1 returned device 0's memory.** `take_best_fit`
     searched the whole free list with no device filter, so device 1's
     allocation was carved out of device 0's slab and recorded as device 1's.
     Device 1's `grxMemGetInfo` read zero bytes in use while holding a live
     allocation. Silent, on every multi-device machine, always.
   * **The interval map could not hold two devices.** With that fixed, both
     devices returned the *same address* — as they must, from equal bases —
     and `g_live`, keyed by address alone, kept one of the two records. It is
     keyed by `(device, address)` now, and the free list by `(slab, address)`,
     which also makes extent coalescing slab-local by construction instead of
     by a check inside `insert_free` that had to be remembered.

   The refusals sit on top of that: a pointer live on another device and on no
   local allocation is refused by `grxMemcpy`, `grxMemset` and `grxFree` with
   `grxErrorInvalidDevicePointer`. **The limit is stated rather than papered
   over** — when an address is live on *both* devices nothing in a bare `void*`
   says which was meant, so it is used as the current device's, which is the
   only defensible reading and is what CUDA does.

   Not covered: **kernel arguments**. The launch path takes an opaque blob with
   no type information, so a device-1 pointer packed into a struct reaches the
   device unexamined. Scanning the blob for 8-byte words that resolve inside
   another device's live allocation would catch most of it and is a heuristic;
   it is written down here rather than slipped in, because architecture §10
   rule 5 is about not shipping a guess that reads as a guarantee.

   `tests/unit/test_cross_device.cpp`, CROSS-DEVICE GATE in `ci/build_mock.sh`.
   Four ablations turn it red, and the test carries a positive control: the
   pointer refused on the wrong device must be accepted on the right one, or a
   runtime that refused everything would pass.
4. **Real workloads and tuning** — items 2 and 3 of the original list. Both sit
   behind item 1 above: there is nothing to tune against until a GEMM can be
   observed running on hardware.
5. ~~**Heterogeneous dispatch policy**~~, scope item 4 above. **Done, and the
   sketched control was deliberately not built.**

   The rule is written down and is one sentence: **the current device decides,
   and nothing else does.** No redirect, and no fallback to another engine when
   the current device's cannot take the call — it refuses. The scope line asked
   for "an explicit `grxblasSetPreferredDevice`-style control plus a documented
   default"; the control is the part that was declined, because a redirect is
   invisible at the call site. The same source line would run on different
   silicon depending on state set elsewhere, which is the automatic magic this
   phase's own scope rules out wearing an explicit name. A program that wants
   GEMM on the NPU calls `grxSetDevice`, which is one line and says so where a
   reader can see it.

   What was built instead is a way to **ask**: `grxblasGetGemmEngine` reports
   which engine a matching `grxblasGemmEx` would use and **which device index
   the decision was made about**, without running anything. Routing and report
   come from one function, so the answer cannot drift from the behaviour.

   Asking is what found the bug. The NPU check read

   ```
   grxGetDeviceProperties(&prop, grxGetDevice(nullptr))
   ```

   and `grxGetDevice` writes through its argument and **returns an error code**
   — `grxErrorInvalidValue`, the value 1 — so the properties query was
   `(&prop, 1)`. The routing consulted device index 1 on every call and the
   current device on none of them. On a machine whose index 1 is not an NPU,
   `grxSetDevice(npu)` routed to the GPU. On a machine whose index 1 **is** the
   NPU — every machine with one GPU and one NPU, since the NPU is appended after
   the GPU — an INT8 `GemmEx` issued while the current device was the **GPU**
   routed to the NPU. Work on silicon the caller did not select, with no error.
   It survived because the decision was a private static that nothing could
   observe without owning a c930.

   Two more, in the NPU path itself. `A_BASE`/`B_BASE`/`C_BASE` are 32-bit MMIO
   words and a device pointer is 64: the cast truncated in silence and the DMA
   would have read whatever lives at the low half. It is refused by name now.
   And the duplicated dispatch guard was reduced to the one decision.

   `tests/unit/test_dispatch.cpp`, DISPATCH GATE in `ci/build_mock.sh`, run with
   two devices because with one the index cannot be seen to *follow*
   `grxSetDevice`. Restoring the `grxGetDevice(nullptr)` expression fails it
   with "device 0: the decision was made about device 1".
6. ~~**`libgrxrt` natively on riscv64.**~~ **The part that does not need a board
   is done.** `ci/build_mock.sh` now runs the riscv64 leg itself, after the
   native one, instead of hiding it behind a flag nobody's CI passed. It costs
   **15 seconds** against the native leg's 33, measured, and runs every gate
   except the two that say why they cannot — the cmake gate is native, and
   grx-prof needs a child process. "Available" was not "checked": a leg nobody
   invokes reports nothing, which is how the RV64 host came to be an untested
   configuration in a phase whose scope names it.

   Running it raised a question nothing had asked: **do the host and the device
   lay the shared structs out the same way?** The kernel argument structs are
   written by the host and read by the device, and a disagreement about one
   offset is not an error anywhere — the kernel reads the blob where it was
   compiled to and returns a wrong answer. `grxblas_sgemm_args` carries an
   `abi_version` first field because that failure was anticipated, but a version
   catches a struct that *changed*, not two compilers that lay the same
   definition out differently. Until phase 7 the host was x86-64 in every
   configuration anyone built.

   `ci/check_abi.py` compares `sizeof`, `_Alignof` and every field's `offsetof`
   across **four targets** — x86-64 host, riscv64 host, device rv64, device
   rv32 — by turning each fact into an array whose length is the number and
   reading it back with `nm --print-size`, so the two targets that cannot
   execute here are checked anyway. 137 facts over 11 cross-boundary structs;
   they all agree today.

   Two things it taught, both by ablation. The probe was a **hand-picked list**
   of fields at first: planting a `size_t` at the front of `grxdnn_gelu_args`, a
   struct the list covered by size only, moved nothing the gate looked at —
   LP64 grew it by 8, ILP32 grew it by 4 and padded 4 back, so the sizes stayed
   equal while every field behind it moved. The probe is **generated from the
   headers** now, every field of every struct, and a declaration the parser
   cannot read is a failure rather than a silent omission. And `grx_abi.h`'s
   `grx_kernel_desc` holds native pointers, so it is 40 bytes on LP64 and 32 on
   ILP32 — reported as nine failures until the right question was asked: **no
   device source includes that header.** It is host-only, grxcc writes it and
   the runtime reads it, both on the same host. Headers are annotated by
   boundary now, and the annotation is itself checked — a device kernel that
   starts including a host-only header fails the gate.

   Still owed and still needing a board: `libgrxrt` running on an actual GRX930,
   and the conformance suite compiled natively there. qemu-user proves there are
   no x86-isms, that structs are passed and returned correctly on RV64, and that
   nothing faults on alignment. It proves nothing about a weak memory model,
   which it does not model, and nothing about the SoC.

The seam the device table needs in order to hold both is written up in
[`heterogeneous_devices.md`](heterogeneous_devices.md).

---

## Phase 8 — Multi-SM on GRX-G100 (≈4 engineer-months, sequenced)

**Scope.** Take GRXCP from one SM to many, in an order chosen so that each step
is measurable when it happens.

The ordering below is the phase. It is not a list of independent work items —
steps 1 and 2 are what make step 4 worth doing, and step 3 is what makes it
correct. Doing them out of order produces a machine that is faster on paper and
unmeasurable in practice.

1. **Widen one SM.** `SIMD_WIDTH` 4 → 32, `NUM_WARPS` 16 → 64,
   `NUM_TCU_BLOCKS` 1 → 4. This is 32× more resident threads per SM and 4×
   the tensor throughput, and it breaks no software contract in this repo.
   Phase 3's tensor gate — 2.51× against a 5× threshold — is a *within-SM*
   shortfall and nothing in this phase's later steps moves it.
2. **Land graph ingestion and fusion** (`developer_interface.md` §3). A launch
   costs 2776 cycles before it touches an element and a transformer block is 23
   launches — 21.1% of the block. More SMs shorten the *work* half of each
   launch and leave the fixed half alone, so the fixed fraction gets **worse**,
   not better. Fusion is a prerequisite for multi-SM paying off, not a
   follow-on.
3. **Enable L2, and `VX_CFG_EXT_A_ENABLE`, before raising `NUM_CORES`.** See
   "Coherence" below — both for the reason and for why they go together.
4. **Then `NUM_CORES` 2, then 4, inside one cluster.** Re-measure every
   grxBLAS threshold; see "Nothing is calibrated for it" below.
5. **Multi-cluster last.** It removes `this_grid().sync()` (gap 7.17) and there
   is nothing in the hardware to replace it with.

**Exit gate.** *Not* a speedup number — a measurement. The transformer block
bench reports a per-stage cycle figure on a 4-SM device, and the block's total
at 1, 2 and 4 SMs is recorded with the speedup stated and the backend named.
A scaling claim cannot be gated before it can be read, and today it cannot be
read at all.

### The prerequisite nobody had noticed: there is no clock above one SM

This is the finding that reorders the phase, and it was invisible until a
4-SM device was run.

`tests/bench/block_cycles.cpp` — the instrument behind every cycle number this
project publishes — measures spans from per-warp `VX_CSR_MCYCLE` probes. MCYCLE
is **per core** and restarts at zero at every launch (gap 7.25), so a stage
whose warps land on more than one core produces a span across two independent
counters. `grx_cycles.h` has refused that case since it was written.

Measured, same binary, same tree, same kernels, only `VX_CFG_NUM_CORES`
differing:

| | stages reporting cycles |
|---|---|
| simx, **1 SM** | **12 of 12** — layernorm 7385, qkv proj 21333, … total 161364 at S=8 |
| simx, **4 SMs** | **0 of 12** — every one: *"the warps spanned cores; a span across two counters means nothing"* |

Not some stages. Every stage, at both shapes. The CTAs distribute across cores
even at these small grids, so nothing survives.

**So the position today is: multi-SM correctness is established and multi-SM
performance is unmeasurable.** `test_grxblas` passes at 4 SMs on rtlsim as of
gap 7.37's fix; the block bench reports nothing there. Building a device-side
clock that survives a core boundary — or a host-side one that does not need to
— is step 0 of this phase, ahead of everything in the ordering above.

Two candidate shapes, neither costed yet: sum per-core spans with a
core-id-tagged probe and report per-core rather than per-stage, or take the
duration from the simulator's own global tick rather than from a device CSR and
accept that it is not available on silicon.

**One defect fixed on the way.** When *every* stage was invalid the bench
printed `(no cycles recorded)` and returned before the loop that prints why —
so `why_no_span()`, written for exactly this case, was unreachable in it. The
first 4-SM run said nothing at all about the cause. Now it prints the reason
for every stage, which is how the table above exists.

### Coherence, which changes under you

Crossing 1 → 2 cores changes the machine silently. Measured from the build
logs of three real builds:

| | 1 core | 2 cores | 4 cores |
|---|---|---|---|
| `VX_CFG_DCACHE_WRITEBACK` | **1** | 0 | 0 |
| `VX_CFG_AMO_RS_SIZE` | 16 | 32 | 64 |
| `VX_CFG_L2_ENABLED` | 0 | 0 | 0 |

`VX_CFG_DCACHE_WRITEBACK` is `int($dcache_is_llc and $single_core)` — nobody
sets it; it flips because the core count changed.

There is **no cache coherence protocol in the RTL**. Write-through above the
last-level cache is the substitute, and `Vortex.sv` states the reason: *"A WB
intermediate could absorb a hart-B store without the LLC seeing it; a later SC
from hart-A on the same line would spuriously succeed."* Two consequences:

- **The static assert that enforces it is inside `` `ifdef VX_CFG_EXT_A_ENABLE ``**,
  which is off in this configuration (gap 7.16). On our builds nothing checks
  the invariant at all. Turning on `EXT_A` is what makes the guard compile,
  which is why step 3 pairs it with L2 rather than leaving it to the atomics
  work.
- **`L2_ENABLED=0` at 4 cores means N private write-through L1s with no shared
  level beneath them.** Every kernel we run touches disjoint data per core, so
  this has never mattered. It starts mattering at the first kernel where one
  core must read what another core wrote — a cross-core reduction, a grid-wide
  softmax denominator, a split-K GEMM. Ask grxgpu what the visibility rule is
  before writing that kernel, not after.

### Nothing is calibrated for more than one SM

grxBLAS chooses its kernel on `outputs >= resident / 2`, where

    resident = warpSize × maxWarpsPerMultiProcessor × multiProcessorCount

which is **64** on the configuration every measurement in this repo was taken
on, and 262,144 at the flagship preset. That is a 4096× move in a decision
boundary that was bracketed empirically between 24 and 32 outputs. Every tiling
choice — including `sgemm_2d_i`, added 2026-09-02 — sits on it.

This is a re-measurement bill, not a blocker, and it grows with accumulated
tuning. It is an argument for raising the core count **earlier** rather than
later, once steps 1–3 are done: the longer single-SM tuning accumulates, the
more of it has to be redone.

### What the grid can and cannot use today

At S=16 the block already asks for more warps than one SM holds — attention
requests 112 against 16 live, mlp GEMM 1 and the residual request 64. Those
stages have real parallelism waiting. The other nine stages request exactly 16,
because the grid is one warp per row and the row count happens to equal one
SM's warp slots at this shape. **That coincidence is a property of the bench
shapes, not of the architecture**: at any production sequence length and hidden
size every stage oversubscribes heavily. The small shapes are what make one SM
look sufficient.

### Two structural caps to plan around

- **The cooperative grid narrows to one legal band.** Gap 7.17: the barrier
  releases per cluster, a core with no active warps never forwards an arrival,
  so `grxLaunchCooperativeKernel` refuses a grid smaller than the machine as
  well as one too large to be resident. At 128 SMs that band is ≥128 blocks and
  ≤ residency, and everything outside it is an error rather than a slow path.
- **DSMEM is dead silicon at one core per cluster.** grxgpu now defaults
  `VX_CFG_EXT_DSMEM_ENABLE = true` alongside `NUM_CORES=16`. Distributed shared
  memory is a cluster feature; if it is on the tape-out then multi-SM is not
  optional, it is the thing DSMEM exists for. Gap 7.18 (`map_shared_rank` has
  no stride) is already open against it.

### Cross-team item found while setting this up

**grxgpu HEAD (`5253957`) does not build simx from a clean configure.**
`sim/simx/csr_unit.cpp` uses `VX_CSR_MAILBOX`, added by `ccc3f363b`. The
definition went into `sw/VX_types.h` — a checked-in copy of a **generated**
header that is not on any include path — and not into `VX_types.toml`, which is
what `configure` regenerates `build/sw/VX_types.h` from. The generated copy has
zero occurrences; the build fails with `'VX_CSR_MAILBOX' was not declared in
this scope`. It works wherever `build/` predates the regeneration, which is the
"works on my machine" failure mode in its purest form. Reported.

---

## Parallel track — `WSHFL` ISA RFC

Not a GRXCP phase; a proposal into the GRX-G100 repo, filed at Phase 0 and
ideally landed before Phase 3.

- **Proposal:** a warp-shuffle instruction with idx / up / down / xor / bfly
  modes, reading the warp's register file through the operand-collector
  crossbar `WGATHER` already uses, generalized beyond 4 lanes.
- **Cost:** RTL + SimX model + a `model_parity` case, plus an area/timing
  study — the register-file crossbar is already flagged as a possible
  bottleneck at 4-way issue (`gpu_chip_design.md` §15 item 5), so this is not
  free.
- **Payoff:** warp reductions and scans stop being an order of magnitude
  slow. This is the single highest-leverage hardware change for CUDA-style
  workload performance.

---

## Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| `grxcc` (P4) slips and blocks everything downstream | High | P3 ships prebuilt library kernels via the native VOLT path, so grxBLAS does not wait on `grxcc`; the driver is an orchestrator, not a new frontend |
| Warp shuffle stays emulated | High for perf | Correct fallback ships in P2; RFC filed at P0; perf gate reports emulated-vs-native separately so the cost is visible |
| Stream concurrency (P5) blocked on GRX-G100 CP work | Medium | Phase ordering already assumes it; semantics ship in P1 so no API breaks when it lands |
| FPGA DMA over-write on unaligned copies | Medium, correctness | 64-byte allocation padding from P1 until the tail-`wstrb` fix lands |
| Conformance pass rate is low and demoralizing | Medium | Publish it from P1 and track it as a trend; the chipStar work's ~36% rv32 number is the precedent for honest reporting |
| Scope creep into graphics/ray tracing | Medium | RTU and TEX exposure are explicitly deferred to P6+; GRXCP v1 is compute |
| rv32 doubles the test matrix | Low | rv64 only for v1; rv32 kept compiling, not tested |
| Emulating hardware features silently | High, insidious | Banned by architecture §10 rule 5 — every emulation is reported through a device property |
| No cycle instrument survives a core boundary | High | MCYCLE is per core and restarts per launch (7.25); measured at 4 SMs the block bench reports **0 of 12** stages. Phase 8 step 0 builds the clock before any scaling claim is made — a speedup that cannot be read cannot be gated |
| Single-SM tuning accumulates against a threshold that moves 4096× | Medium | grxBLAS's `resident` is 64 today and 262,144 at the flagship. Argues for raising core count early, once Phase 8 steps 1–3 are done, rather than banking more tuning first |

---

## What "done" means for v1

GRXCP v1 is complete when a developer who knows CUDA can:

1. Write a `.grx.cpp` file with `__global__` kernels and `<<<>>>` launches,
   compile it with `grxcc`, and run it on GRX-G100.
2. Port existing CUDA code by adding one include (`grx_cuda_compat.h`) and
   fixing only the gaps this document names — no mystery failures.
3. Call `grxblasGemmEx` and get tensor-core throughput.
4. Profile with `grx-prof`, debug with `grx-gdb`, and catch memory bugs with
   `grx-sanitize`.
5. Read a published conformance number and know exactly what does and does
   not work.

Phases 0–4 deliver 1, 2, 3 and 4. Phase 6 raises the number in 5. Phase 7
makes it heterogeneous. Phase 8 makes it more than one SM — and note that none
of the five clauses above mentions performance, which is deliberate: v1 is
"works and says what it does", not "is fast".
