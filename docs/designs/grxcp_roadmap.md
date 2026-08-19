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
| Exit gate: `grx-smi` on real `rtlsim` | pending — needs Verilator and the `hw/dpi` sources |

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
| `sgemm` | pending — belongs with grxBLAS in phase 3 |

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
  MPM stall breakdown, Perfetto export reusing `ci/perfetto.py`.
- `grx-sanitize` v1 on SimX: out-of-bounds global/shared access, use of
  uninitialized shared memory, and barrier-divergence detection.
- `grxify` v0: mechanical `cudaX → grxX` source rewriting.

**Exit gate.** A warp-reduction kernel using `__shfl_down_sync` produces
correct results through the fallback; `grx-sanitize` detects a deliberately
planted out-of-bounds write and reports the source line; `grx-prof` produces
a Perfetto trace a human can read.

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
- grxBLAS v1: GEMM (fp32/fp16/bf16/int8), batched GEMM, GEMV, AXPY, with
  autotuned tile selection seeded from the existing `sgemm_tcu_wg_dxa_mcast`
  reference kernels.

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
The baseline is `tests/bench/sgemm_cycles.cpp`, and it is a number this project
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

**Phase 3 exit gate: MET.** `grxblasGemmEx` (fp16 in, fp32 accumulate) composes
the tensor unit with DXA staging, is exact against a CPU reference on every
shape including ragged ones, and costs far less than a fifth of sgemm per
output element:

| shape | sgemm cyc/elem | GemmEx cyc/elem | speedup |
|---|---|---|---|
| 16 x 16 x 16 | 297.8 | 40.0 | **7.45x** |
| 16 x 16 x 32 | 523.4 | 47.1 | 11.11x |
| 16 x 16 x 64 | 974.2 | 64.1 | 15.19x |
| 32 x 32 x 32 | 523.8 | 43.4 | 12.06x |

The gate is enforced in `tests/bench/gemm_cycles.cpp` now that it passes.

What that number is **not**: tuned. This kernel is single buffered and uses each
staged tile once, so the tensor unit waits for the DMA engine and vice versa.
The speedup is what the composition gives before any of the tricks — double
buffering, a larger per-warp tile, WGMMA warp groups. It also runs on ONE CTA,
not by choice: see cuda_mapping.md section 7.12.

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

---

## Phase 5 — Concurrency and asynchrony (≈3 engineer-months, gated externally)

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
agreement against a PyTorch CPU reference; conformance pass rate hits the
target set at Phase 4.

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
makes it heterogeneous.
