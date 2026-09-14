# PTA on the GPU: where a photonic tile goes in the G100

**Companions:** [`pta_cpu_integration.md`](pta_cpu_integration.md),
[`heterogeneous_devices.md`](heterogeneous_devices.md).
**Source analysis:** GRX_PTA_Integration.md in this repository's `docs/`.

**Status: DESIGN, staged behind the CPU path except G1 (§7), nothing built.**
This document exists now rather than later for one reason: the shared IP — the
tile model and its error budget — is being written for the c930 first, and if
it is written without knowing what the GPU needs it will be rewritten.
Everything here is a constraint on that IP, plus the design that follows once
it exists.

**Boundary rule, first.** GRXCP does not patch grxgpu. If any of this lands it
lands as a proposal in that repository's own proposals directory, authored by
whoever owns the RTL, and this document is the argument for it — not a
substitute for it. The dependency is one-directional and stays that way
(`AGENTS.md` §2).

**Scope, same as the CPU path.** FPGA emulation and numerics only. No PDK, no
photonic die. The tile is a deterministic digital model of an analog channel,
and the results are architectural and numerical.

---

## 1. There are two integration points, and only one of them is a photonic accelerator

The G100 offers an unusually clean-looking seam, and the first thing to
establish is that taking it would be a mistake — a productive one, but not the
mistake it appears to be.

### 1.1 The FEDP seam is real

grxgpu/hw/rtl/tcu/VX_tcu_core.sv selects its fused-dot-product backend by
`VX_CFG_TCU_TYPE`, and the config enumerates five: DPI, DSP, BHF, TFR, FPNEW.
Each is instantiated from an `ifdef` chain with an identical port list —
`clk`, `reset`, `enable`, `fmt_s`, `fmt_d`, `a_row`, `b_col`, `c_val`,
`d_val`, plus `sf_a`/`sf_b` under MX and `vld_mask` under TFR — and
parameterized by `INSTANCE_ID`, `LATENCY`, `N`, `SF`.

Critically, each backend declares its **own** `FEDP_LATENCY`, and the
surrounding machinery derives from it: `PIPE_LATENCY = FEDP_LATENCY + 1`,
`MDATA_QUEUE_DEPTH = 1 << $clog2(PIPE_LATENCY)`, and the shift-register
completion pipe sizes itself. A backend with a different latency drops in
without touching the scheduler, the lockstep gate, or the uop sequencer.

Adding a sixth is a new module plus an `elsif` branch plus one entry in the
config's enumeration. That is genuinely about twenty lines.

### 1.2 And it is the wrong place for a photonic engine

Two reasons, both structural.

**Nothing is stationary.** A photonic mesh's entire economics rest on
programming weights once and streaming many activations past them. The FEDP
receives a fresh `a_row` *and* a fresh `b_col` on every micro-op. There is no
operand it holds. A photonic FEDP would reprogram its mesh every issue — a
microsecond-scale operation in the inner loop of a nanosecond-scale pipeline.
It is not a slow design; it is an incoherent one.

**The dimensions are off by an order of magnitude.** At the shipped
configuration — `NUM_THREADS = 4`, `ISSUE_WIDTH = 4`, `FEDP2K` off — the block
geometry works out to `TCU_TC_M = TCU_TC_N = 2` and `TCU_TC_K = 2`, so
`FEDP_K = 2`. **Each FEDP is a length-2 dot product.** A photonic tile whose
mesh has two channels is a laser, two modulators, two DACs, a detector, a TIA
and an ADC amortized over two multiplies. Every fixed cost of going optical
dominates. Photonic MVM tiles are interesting at tens to hundreds of channels;
this cell is two.

### 1.3 What the FEDP seam is actually good for

It is an excellent place to run the **numerics** experiment, and that
experiment is worth doing on its own.

`VX_CFG_TCU_TYPE_ANALOG` → `VX_tcu_fedp_analog`: the same error model as the
c930 tile — quantization, thermal and shot noise, programming error, drift —
applied to the existing dot product, with no claim that anything optical is
happening. It answers a question the CPU path cannot: *what happens to a real
SIMT tensor workload, at real tile geometry, under an analog error channel?*
And it answers it against a test suite that already exists — the SGEMM,
WGMMA and DXA regressions, unchanged.

Call it what it is in the module name and in every result. An
`fedp_analog` backend is honest. An `fedp_photonic` backend that reprograms a
two-channel mesh per micro-op is a claim nobody should make. The naming is the
whole guardrail here, because the code is identical either way.

---

## 2. Where the engine goes: cluster scope, beside the DXA

The photonic tile wants to sit where a B tile is already resident and already
reused. In the G100 that is unambiguous: **LMEM, at cluster scope, next to
VX_dxa_core.**

Three facts make this the answer rather than a preference.

**Reuse lives at the tile, not the instruction.** In `C = A·B`, for a fixed
output column block the same B tile is consumed by every row block of A. That
is the reuse a weight-stationary engine needs, and it exists one level above
the FEDP — at the tile-buffer level, where `VX_tcu_bbuf` is already *shared*
across all `Q` blocks and broadcast, while each block keeps its own
`VX_tcu_abuf`. The hardware already treats B as the shared, reused operand.
A photonic engine is that observation taken to its conclusion.

**The weight-load path already exists.** DXA's `dest_kmajor` mode produces
exactly the K-major SMEM layout the tensor path consumes, scattering one
element per beat. That is the same serial, one-element-per-beat discipline a
weight-DAC scan chain needs, and it is already the documented DXA↔TCU tie-in.
A photonic tile's weight load is a DXA transfer whose destination is the
tile's weight SRAM instead of LMEM.

**The scope matches.** `VX_dxa_core` is instantiated exactly once per cluster,
arbitrating `NUM_SOCKETS` request streams and draining through the cluster's
LMEM path. A photonic tile placed the same way — one per cluster, sharing the
LMEM-DMA priority arbiter that DXA (index 0) and TCU (index 1) already contend
for — is the same shape of object in the same place. It is a third client, not
a new fabric.

---

## 3. The problem the GPU has and the CPU does not

This is the part of the GPU path that is novel, and it is worth stating before
any register map, because it determines whether the rest is worth building.

A SIMT machine's founding assumption is that **any resident warp may run at any
time**. Scheduling is free because every warp sees the same functional units.

A weight-stationary analog tile says: *only work whose weight set is currently
programmed may run, and changing that costs `Tw`.* The scheduler now has a
resource with an enormous, warp-visible switching cost — something the G100 has
never had. A warp scheduler that round-robins across CTAs with different weight
sets will thrash the mesh and lose everything.

The CPU path never meets this: the c930 runs one GEMM at a time from a
four-deep command queue, so weight-set order is whatever the driver enqueued.
On the GPU it is a live scheduling problem.

Three answers, and the interesting work is choosing between them with numbers:

1. **Weight-set affinity in the KMU.** The CTA dispatcher binds CTAs sharing a
   B tile to the same cluster and issues them contiguously. Cheapest in
   hardware, requires the runtime to expose the weight set as a dispatch
   attribute, and interacts with the existing cluster-dispatch and occupancy
   machinery.
2. **A small mesh cache.** `W` weight banks per tile, LRU or software-managed,
   so `W` weight sets are hot. Turns a scheduling problem into a capacity
   problem, which is easier to reason about and more expensive to build. The
   c930's existing two-bank structure is `W = 2`.
3. **Software-declared, hardware-enforced.** The kernel names its weight slot
   in the launch descriptor; a mismatch serializes on the tile's own lock. No
   scheduler change at all, and it makes the cost visible to the programmer
   rather than hiding it — which, for a research vehicle, may be the right
   trade.

All three are measurable in SimX before a line of RTL, because SimX already
models the lockstep gate, the tile buffers and the LMEM port contention. That
is the first GPU deliverable and it is cheap.

---

## 4. ISA surface

grxgpu/docs/designs/custom_accelerator_isa_extensions.md is the governing
document and its rules are not optional here — it exists specifically because a
previous accelerator shipped the per-slot special-register anti-pattern and cost
~16 instructions per work item. Applying its decision checklist:

| Argument | Scope | Mechanism |
|---|---|---|
| Tile geometry, error-model seed, impairment enables, calibration policy | per-dispatch | **DCR**, host-programmed, zero per-call cost |
| A-tile SMEM address, weight slot, barrier id, CTA mask | per-warp uniform | **`vx_wgather`**, four scalars into four lanes of one register |
| The weight block itself | memory-resident | **custom LD into the tile's weight SRAM** — never through the GPR file |
| Completion | — | **async handle + `wait`**, blocking via the scoreboard |

The async rule is not a preference here, it is forced. `Tw` is microseconds and
even a shot plus ADC is tens of nanoseconds; a synchronous launch would stall
the issuing warp for the entire weight program. The launch returns a handle,
the warp does independent work, and `wait(handle)` blocks on the scoreboard —
the guide's async-by-design pattern, unmodified.

The DXA precedent is the model to copy verbatim: `INST_SFU_DXA` is a RISC-V
`custom0` with `funct7=0x3`, the intrinsic packs four lanes via `vx_wgather`
(smem address, `meta = (bar << 4) | desc_slot`, coords, `cta_mask`), the
descriptor lives in a DCR block, and completion releases a barrier
transaction. A PTA launch should be the same instruction shape with a weight
slot where the descriptor slot is. Following an existing, working idiom is
worth more than an optimal novel one.

R-type over R4-type, per that guide's instruction-encoding section: the
operands collapse to one lane-packed register plus one window base, which
leaves `funct7` free for sub-op and format encoding — and this unit needs
sub-ops (load weights, shoot, calibrate, read status).

---

## 5. Numerics: the format already exists

The single most useful thing found while reading both trees.

The G100 tensor unit already implements microscaling formats — mxfp8, mxbf8,
mxfp4, nvfp4 — with per-block scale factors carried in a dedicated per-warp
metadata SRAM, preloaded by `INST_TCU_LD`, with `TCU_MX_MAX_SF` scale bytes
routed per logical row and column into the FEDP.

Block scaling exists to keep a low-precision channel near full scale. That is
*precisely* what an analog tile needs, for a reason that has nothing to do with
digital formats: detector shot noise is signal-dependent, σ ∝ √|y|, so
signal-to-noise improves toward the top of the ADC range. An analog tile
running at 10% of full scale wastes most of its effective bits. A per-K-block
scale factor is the mechanism that keeps it at 80%.

So the numeric format for a photonic tile is not INT8 and not FP16. It is
block-scaled integer, and **the G100 already has the metadata path, the SRAM,
the scale routing and the software surface for it.** The c930 should adopt the
same format rather than invent one (see [`pta_cpu_integration.md`](pta_cpu_integration.md) §5.2),
and one shared format across both devices is worth more than either device's
local optimum — because the compiler that picks scale factors is then written
once.

This is also the answer to "where does the scale factor come from," which the
CPU document leaves open: the same place the MX scales come from today, from a
host-side pass, with the tile's saturation counter available as a feedback
signal for an adaptive version.

---

## 6. Verification

The same ablation discipline as the CPU path, for the same reason.

**With every impairment disabled, `VX_CFG_TCU_TYPE_ANALOG` must be bit-identical
to `TFR`** on the existing tensor regressions. `TFR` is the ASIC/SimX default,
so it is the right reference. If that is not established first, every accuracy
delta measured later is unattributable between the error model and a plumbing
bug, and there is no experiment that separates them afterwards.

Then: RTL and SimX must agree, because this repository's rule is that a
backend divergence blocks (`AGENTS.md` §4). That forces the same determinism
constraint the CPU path takes — no `$random` anywhere in the model, every
stochastic quantity from a seeded LFSR, and the SimX model reproducing the
same LFSR. It is more work than it sounds and it is the thing that makes
everything downstream checkable.

The cluster-scope engine has a third obligation the FEDP swap does not: it is
a new client on the LMEM-DMA arbiter, contending with DXA and TCU. Arbiter
starvation under a weight-load-heavy pattern is the failure mode, and it wants
a directed test rather than a regression sweep.

---

## 7. Staging

Nothing here starts before the CPU path's phase C2 lands, for two reasons: the
tile model and error budget are shared IP and should stabilize on the cheaper
device, and phase C2's `Tw`-to-`Ts` ratio sweep may say that a general-purpose GEMM engine
is the wrong place for a photonic tile at all
([`pta_cpu_integration.md`](pta_cpu_integration.md) §5.5) — which would change
what the GPU path should be, or whether it should exist.

**Re-staged by the program plan (D4 of
[`pta_program_plan.md`](pta_program_plan.md)).** Neither reason touches G1,
which has no RTL and shares no IP, and the question it answers — how many
weight sets a cluster must keep hot — is the one the c930's multi-bank step
asks too. G1 therefore starts now, sweeping bank count `W` beside `Tw`, since at
the Pockels points `Tw` is the host's scan and `W` is what varies. G0 is
numerics only and waits for the error model, so it starts when C1 is green. G2
and G3 keep the C2 dependency.

The order once each starts:

**G0 — the numerics backend.** `VX_tcu_fedp_analog`, sharing the c930 error
model. *Gate:* bit-identical to TFR with impairments off, across the existing
tensor regressions; RTL↔SimX parity at a fixed seed.
*Result:* accuracy-versus-impairment curves for real SGEMM and WGMMA
workloads. Publishable on its own, and it needs no new fabric.

**G1 — the scheduling study, in SimX only.** Model a cluster-scope
weight-stationary tile with parameterized `Tw`, and compare the three §3
policies on real kernels. *Gate:* a measured answer, including the null result
that affinity does not help. No RTL.

**G2 — the engine.** Cluster-scope tile beside the DXA, the ISA of §4, the
DXA weight-load path. *Gate:* end-to-end GEMM through the tile, matching the
model bitwise; LMEM arbiter fairness under weight-load pressure.

**G3 — the comparison.** The same block-scaled GEMM on the c930 tile and the
G100 tile, same seed, same error parameters, agreeing bitwise. That is what
having two devices is actually for, and it is the closing argument for the
shared IP.

---

## 8. Proposed but not yet implemented

1. **Per-socket rather than per-cluster placement.** The DXA design already
   names per-socket relocation as its largest unbuilt item, motivated by
   measured GMEM latency growth with core count. A photonic tile placed at
   cluster scope inherits that decision; if DXA moves, the tile should be
   revisited alongside it rather than independently.
2. **Multicast weight load.** DXA's multicast replays LMEM writes to
   co-resident CTAs. Programming the same weight set into several clusters'
   tiles at once is the same operation and would amortize `Tw` across a
   cluster group. Attractive, and it depends on the multicast LMEM-arbiter
   hoist that is itself unbuilt.
3. **Sparsity.** The TCU supports 2:4 structured sparsity as a distinct opcode
   with a metadata SRAM and a B-column gather. What 2:4 sparsity means for a
   photonic mesh is a genuine open question — the compression is in the
   *operand routing*, which is a mux on a digital path and a physical
   waveguide on an optical one. Out of scope here and worth its own note.
4. **Graphics and RT.** The G100's raster, texture, OM and RT units are
   irrelevant to this work and should stay that way. Recorded only so that
   "photonics for the GPU" is not read as touching them.

---

## 9. Open questions

1. **Does the FEDP-level numerics result transfer?** G0 measures analog error
   applied to a length-2 dot product replicated across a grid. A real
   photonic tile computes a length-64 reduction in one optical shot, where
   errors combine differently — one detector integrating 64 terms is not 32
   independent 2-term errors. The G0 curves may not predict the G2 tile's
   behavior at all, and the comparison between them is itself a result.
2. **Who owns the shared tile IP?** It cannot live in this repository
   (`AGENTS.md` §2 is one-directional), and neither RTL repository depends on
   the other. The honest answer is probably that it lives in grx930, and
   grxgpu vendors it — which is a coupling neither repo has today and which
   somebody has to agree to. *Settled by the program plan (D1):* grx930 owns
   it, and grxgpu vendors tagged releases in one direction, the way grxcp
   vendors grx930's DPI shim.
3. **What happens to warp occupancy?** A tile with `W` weight banks caps the
   number of concurrently schedulable weight sets per cluster, which is a
   second occupancy limit alongside LMEM slots and register pressure. The
   existing occupancy formula, which GRXCP consumes verbatim, would need a
   term. That is a runtime-visible change and it belongs in the gap register
   before it is built, not after.
4. **Is `PTA_TS` even representable?** On the c930 the shot fits in a cycle
   with room to spare. In the G100's pipeline a photonic shot is faster than
   the FEDP latency it would replace, so the model's floor is the emulation's
   floor rather than the device's — and the `Tw`-to-`Ts` ratio sweep that works on the
   CPU may compress at the fast end here. Worth checking before G2 rather than
   after.
