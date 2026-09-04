# The GRXCP developer interface

**Status: a proposal, and the case for a decision that has not been made.**
Nothing here is built. The numbers are measured or marked as estimates, and the
recommendation is separable from them — if the arithmetic in section 3 is wrong,
the recommendation should change.

## 1. The toolkit mostly exists, which is not the interesting part

| GRXCP | CUDA equivalent |
|---|---|
| `grxcc` | `nvcc` |
| `grx-smi` | `nvidia-smi` |
| `grx-prof` | Nsight Compute / Systems |
| `grx-sanitize` | Compute Sanitizer |
| `grxblas` / `grxdnn` | cuBLAS / cuDNN |
| `grx_cg.h`, `grx_wmma.h`, `grx_warp.h` | cooperative_groups, `mma.h` |
| `grxify` | `hipify` |
| **`grx-conform`** | **none — NVIDIA is the specification** |

So the question is not which components to build. It is who the developer is,
and the three candidates want almost disjoint things:

1. **Porting existing CUDA.** Wants `grxify`, `grx_cuda_compat.h`, and the
   conformance number. Served by what we have.
2. **Writing new kernels.** Wants `grxcc`, the device headers, `grx-prof`.
   Smallest group, most toolkit-shaped.
3. **Never writing a kernel.** Calls a framework and wants it to land on our
   silicon.

**This document proposes (3), keeps (2) in reserve on the reasoning in section
8, and does not pursue (1).** The porting audience is a bet that developers have
CUDA they want to move. That bet ages badly and it anchors us to a 2007
execution model; see section 3 for why that model is specifically bad for this
machine.

## 2. What (3) actually requires, and the trap in it

Implementing PyTorch's operator set directly is not the shape of this work.
There are thousands of ATen operators and nobody implements them. The route is
to ingest below the framework, where the set is already decomposed — a
`torch.compile` backend receives a graph reduced to a few hundred core ops and
mostly pointwise, reduction and matmul; StableHLO or ONNX give a closed set.
Either way the primitive count is tens.

**We have most of those primitives already.** grxBLAS gives GEMM in fp32, fp16
via the tensor unit, and int8 via the NPU. grxDNN gives softmax, layer norm,
gelu, add-bias and fused attention. That is not a coincidence — it is the
transformer's primitive set, arrived at by building a transformer block.

What is missing is not compute. It is ingestion.

## 3. Graph-first is forced by the machine, not preferred on taste

**A launch costs 2776 cycles before it touches an element.** Measured by
sweeping `grxdnnAddBiasForward` over column counts at the block's own row count,
least squares over nine points: `cycles = 2776 + 4.33 * elements`. An
independent two-point fit from the block's own bias stages gave `2716 + 4.25`
and predicted a held-out third shape to 1.5%. The two agree.

**One transformer block is 23 launches** at `H=2`:

| stage | launches | | stage | launches |
|---|---|---|---|---|
| layer norm 1 | 1 | | out projection | 2 (`H` GEMMs) |
| qkv projection | 3 GEMMs | | out bias | 1 |
| qkv bias | 6 (`3 x H`) | | residual | 1 |
| attention | 4 (2 GEMMs, mask, softmax) | | layer norm 2 | 1 |
| mlp GEMM 1 | 1 | | mlp bias | 1 |
| gelu | 1 | | mlp GEMM 2 | 1 |

Nine of those are GEMMs, and that half is **measured** — `GRXBLAS_SGEMM_TRACE`
on a block run reports exactly nine, which is what makes the structural count
credible for the other fourteen.

At 2776 cycles each, 23 launches is **63848 cycles of fixed cost against a
302961-cycle block at S=16 — 21% spent before any element is touched.**

**That was the visible half, and the real figure is 51%.** The 2776 is the fixed
cost inside a stage's measured *span*, and a span begins at the first warp's
probe — so everything between the launch and that first warp sits outside every
number above. On simx MCYCLE is zeroed at the launch, so the first warp's own
reading is exactly that interval, and it had simply never been read. Measured on
simx across all 23 launches at S=16: **216621 cycles of preamble**, 9418 each,
against 328987 of spans. Adding the two kinds of fixed cost together:

| | cycles | share of the block |
|---|---|---|
| fixed cost inside spans (23 × 2776) | 63848 | 11.7% |
| launch preamble, outside every span | 216621 | 39.7% |
| **per-launch fixed cost, total** | **280469** | **51.4%** |
| work | 265139 | 48.6% |
| block | 545608 | |

**Just over half of a transformer block is per-launch fixed cost.** The
conclusion below was right and understated by a factor of two, and it gets
worse with parallelism rather than better: measured at 1, 2 and 4 SMs the
preamble per launch *grows* — 9418, 10899, 11383 — so a 1.97× speedup on the
work becomes **1.27× end to end** (see `grxcp_roadmap.md`, phase 8).

**The preamble is not a constant — it grows with occupancy and then stops.** A
kernel that does nothing but timestamp itself, launched at varying grid sizes,
with every block recording its own entry (`tests/repro/launch_preamble/`).
**One launch per process**, for the reason under the table:

| blocks | first block's entry, rtlsim | simx |
|---|---|---|
| 1 | 2910 | 3030 |
| 4 | 7195 | 6865 |
| 16 | 16467 | 26414 |
| 32 | **17687** | **26414** |
| 64 | **17687** | **26414** |

It is the **earliest** block's entry that grows, not merely the last — and it
levels off at the point the machine is full: 64 warp slots, four warps per
block, so sixteen resident blocks. Past that, blocks queue behind retiring ones
and the first one in is no later than before. Two candidates are ruled out by
the same run — the measuring fence costs 27 cycles at entry and the first memory
access 18 — and the one-block floor is ~1850, so device bring-up is real and
small.

**Two earlier versions of this paragraph were wrong, in opposite directions.**
The first called the growth hardware CTA dispatch at ~1500 cycles per CTA and
built a multi-SM scaling wall on it. The second retracted the label after
reading grxgpu's CTA runtime, where block distribution is a **software loop**
(`sw/kernel/src/vx_spawn.c`, `process_thread_groups`) — but kept the numbers,
which showed the earliest entry climbing to 100774 at 64 blocks with no sign of
stopping, and sent grxgpu a question about which backend was right.

**Neither the wall nor the question was real.** Those numbers came from a sweep
that ran all seven grid sizes in one process in ascending order, and MCYCLE does
not restart between launches on rtlsim: the RTL counter (`VX_scheduler.sv`) is
zeroed only under `reset`, which rtlsim issues once in its constructor, and it
then free-runs per core gated on that core's `busy`. Every reading after the
first carried the launches before it. Run the same sweep descending and one
block reports 122873 instead of 2910. Six identical launches climb by a constant
8917. The curve was the sweep's own position.

So the term that would have scaled with the machine does not exist: **the
preamble scales with occupancy, which is bounded, and not with the grid, which
is not.** A 128-SM part does not inherit a serial per-CTA cost from this. What
remains is a real per-launch cost at the shapes these kernels use — 9418 cycles
at S=16 — so **51.4% is a figure for this shape rather than a property**, and
fusion removes one whole instance of it per fused pair.

**And the 51.4% is a simx number for a reason worth stating.** Reading the
preamble as an absolute requires a counter that restarts at the launch, which is
something simx does and silicon will not. On hardware the same quantity has to
be measured as a difference against a reference taken before the launch. The
figure is sound; the *method* does not transfer, and `block_cycles` now
calibrates the counter with four identical launches and reports `-1` rather than
a number on any backend where it does not restart.

An eager operator-by-operator backend does not issue 23 launches per block. It
issues far more, far smaller ones — that is what the decomposed graph in section
2 looks like before fusion. **So eager dispatch is not a slower option for this
machine, it is a non-starter**, and the conclusion does not depend on the exact
launch count: at 2776 cycles fixed, any design whose unit of work is one
operator loses.

Two consequences worth stating plainly:

- The ingestion layer must take a **graph** and fuse it, not a stream of ops.
- The fusion work already in the gap register stops being an optimisation. The
  qkv bias fusion blocked by 7.27 is worth 5.9% of a block at S=16 as a
  point improvement; as the mechanism this strategy rests on it is worth
  considerably more, and the VOLT report should be weighted accordingly.

## 4. Which ingestion point

Not decided here. The two candidates:

**`torch.compile` / Inductor backend.** Meets developers where they are, needs
no model conversion step, and inherits PyTorch's decompositions. Costs a
dependency on a fast-moving internal interface, and the graphs arrive with
dynamic shapes.

**StableHLO or ONNX ingest.** A closed, versioned op set and a stable contract,
which suits a platform that wants to publish coverage honestly. Costs the
developer a conversion step and loses eager-mode debugging.

What would settle it is not argument but a survey: take three or four models the
company actually intends to run, and count how many primitives each route needs
that we do not have. That is a day of work and it replaces a design meeting.

## 5. bf16 is plumbing, not silicon — corrected

**This section originally said the GPU tensor unit had no bf16 and that the
strategy would have to route bf16 through the NPU before tape-out. That was
wrong, and the correction is good news.** It is left visible rather than
rewritten away, because the mistake is instructive: `grxblas.h` stated bf16 was
"not a type this tensor unit has, in any configuration", a sentence sourced
from the configuration schema — which lists what can be *asked for* and says
nothing about what is *built*. It was read as a statement about the hardware.
That comment has since been corrected in the header itself.

Prompted by the grxgpu team, the datapath was checked directly:

| layer | bf16 present? | evidence |
|---|---|---|
| SimX functional model | yes | `FEDP<bf16,fp32>` and `FEDP<bf16,bf16>`, dispatched from `sim/simx/tcu/tcu_unit.cpp` |
| RTL tensor unit (TFR) | **yes, and ungated** | `TCU_BF16_ID` is a case arm in `hw/rtl/tcu/tfr/VX_tcu_tfr_mul_f16.sv` behind no `ifdef` |
| RTL tensor unit (fpnew) | yes | `mult_result_bf16` in `VX_tcu_fedp_fpnew.sv` |
| c930 NPU | yes | `i_precision` mode 3, `c930_bf16_mul.sv` |
| configuration schema | no knob — **and none is needed** | `VX_config.toml` has no `TCU_BF16_ENABLE`, because the bf16 case arms live inside `VX_tcu_tfr_mul_f16.sv`, the module instantiated under `` `ifdef VX_CFG_TCU_FP16_ENABLE ``. bf16 is present wherever fp16 is. |

**The correction needed a correction, which is the more useful half of this
section.** The first version of this page said the tensor unit had no bf16. The
second said bf16 "needs a config knob, a bit in the reported type set, and a
path through `grxblasGemmEx`". The knob was a residue of the original error —
having concluded "no switch, therefore no hardware", the repair kept the switch
and moved it into the to-do list. Measured: there is nothing to switch. grxgpu
ran a bf16 SGEMM end to end (`05a3d84c0`) with `-DITYPE=bf16 -DOTYPE=fp32` and
no bf16 flag of any kind.

So the multiplier is synthesized into every build we make and **nothing on our
side asks for it**. The device reports `fp16` and nothing else, which is what
`grxblasGetTensorTypes` returns — not because the device lacks bf16 but because
`hgemm_tcu.cpp` never sets a bit it has no enum value for.

**What bf16 actually needs is a bit in the reported type set and a path through
`grxblasGemmEx`** — two pieces of our own software, neither on the tape-out
critical path. That moves it from the item on this page with the least schedule
slack to an ordinary piece of enabling work, and it removes the argument that
bf16 forces matmul through the NPU.

Two things survive the correction. The NPU's bf16 is still bounded at
`MAX_M=8, MAX_N=12, MAX_K=16` with a command queue that cannot pipeline two
commands, so the NPU is not a bf16 escape hatch if the GPU path stalls. And the
general lesson is worth more than the specific fact: **a capability can be
absent from the product while present in the silicon, and the two are not the
same finding.** The honest instrument would have been to ask the device, which
this project already does elsewhere and did not do here.

## 6. The metric changes, and improves

`grx-conform` publishes coverage of the CUDA runtime surface — **57 of 83
tracked entry points, 69%**. That is the right honesty instrument for audience
(1), and if we are not pursuing (1) it measures something nobody is buying.

The report says so itself, in its own words: *"This is an API coverage number.
It is not a conformance pass rate: nothing in this report executes a kernel."*
That caveat is the argument for replacing it rather than a defect in it.

Its successor for audience (3) is a **numerics contract**: *this model produces
the same answers as the reference, within a stated tolerance, and here is the
fraction of models for which that holds.* We already do this at the kernel
level — grxDNN is gated against PyTorch-generated vectors in
`tests/libs/*_ref.bin`, and gelu carries a measured 5.36e-07 rather than a
quoted one. Scaling it to whole models keeps the posture and changes the
subject to one the buyer cares about.

It is also a better engineering gate. Coverage counts entry points that exist;
a numerics contract fails when an answer is wrong.

## 7. What this proposal does not build

- **Training and autograd.** Doubles the operator surface and needs the bf16 we
  do not have. Inference first.
- **An eager dispatch backend.** Section 3.
- **A CUDA porting story.** `grxify` and the compat header stay because they
  cost nothing to keep; they stop being a roadmap item.

## 8. The differentiator is not the simulator — it is that the machine is debuggable

NVIDIA's posture toward its own hardware is *trust the silicon, here is a
profiler*. The counter set was fixed at tape-out, the signals do not leave the
die, and no developer will ever run the same binary twice with the hardware
behaving two different ways.

Ours does not have to be that. The model IS the machine, so the machine is an
artifact you can instrument, bisect, and A/B. **That, rather than "we ship a
simulator", is the thing NVIDIA's toolkit structurally cannot offer**, and it
is worth more to audience (2) than any authoring ergonomics.

### What it bought this week, measured rather than argued

Gap 7.37 — `rtlsim` losing kernel arguments above one core — was closed on
2026-09-03. Every step used something unavailable on hardware:

| what was done | why silicon cannot |
|---|---|
| Ran the fix and its control **one environment variable apart in the same binary** — `GRX_NO_SETTLE=1` gave 4/8 and an abort, unset gave 8/8 and PASSED | the machine's behaviour is not a runtime switch |
| Answered a driver-shaped question with a **hardware waveform**: `t=0 busy=1 \| t=1 busy=0 \| t=2 busy=1 \| t=2318 busy=0` | the signal does not exist outside the die |
| **Added counters to the device** in an afternoon — a drain-iteration count, a cross-core clock probe, a per-launch preamble reading | the counter set is frozen at tape-out |
| Rebuilt the same design at 1, 2 and 4 SMs and compared | you cannot buy a 2-SM part |

The launch-preamble result in section 3 is the clearest case. **51% of a
transformer block is per-launch fixed cost**, and it was found because simx
zeroes MCYCLE at the launch, so the first warp's own reading *is* the preamble.
That number sat in plain view for months, it doubled the strength of this
document's central argument, and it is not obtainable on hardware at all.

It also cuts the other way, which is the part worth keeping. That same
convenience is not what the other backend does, and reading a *grid sweep*
through it produced a scaling wall that did not exist — a whole roadmap phase
was written on a curve that turned out to be the sweep's own position. What
caught it was the same property: two backends that can be run against each
other, a config rebuilt at will, and a kernel small enough to launch six times
and watch the number that should not move. **A machine you can interrogate is
one that can mislead you and then be made to say so.** Silicon offers neither
half.

Two more that follow from the same property and are worth naming as product:
**deterministic replay** — every figure in this document reproduces exactly, so
a regression is a diff rather than a statistical argument — and **bisecting the
hardware**, since a config knob is a rebuild rather than a purchase order.

### The precedent that says don't

NVIDIA shipped a GPU simulator and killed it. `nvcc -deviceemu` was deprecated
at CUDA 3.0 and gone shortly after, around 2010, replaced by debugging on real
hardware. Developers petitioned against the removal and lost.

The reason it deserved to die is the one that matters here. Device emulation ran
kernels as host threads. It could not model warps, divergence, coalescing or
occupancy — so it told you your program was **correct** and misled you about
everything else. A model that validates function while lying about cost is worse
than no model, because people quote its numbers. NVIDIA could also afford to
kill it: they always have silicon, and every CUDA developer owns a GPU.

### The precedent that says do

Neither of those is our situation, and the industry that shares our situation
does ship simulators as product. Arm sells Fixed Virtual Platforms as supported,
versioned models precisely so software can be written before parts exist;
Siemens PAVE360 builds pre-silicon environments around Arm cores. Automotive
drives it because a vehicle program cannot wait for tape-out — the same
arithmetic as a software schedule that has to overlap a hardware one.

**We are in Arm's position, not NVIDIA's.** What separates the two precedents is
not fidelity but disclosure: whether the model tells you what it is.

That part we already do, and not by accident. Every bench prints the backend
that produced its cycles. `grx-smi` reports a *narrower* capability set on a
TCU-less `rtlsim` build rather than the one the header advertises. The perf
baselines are per-backend files. `grxDeviceProp_t.backend` is derived rather
than asserted, and the device seam refuses to let a model claim to be silicon.

### The fidelity contract

What we do not yet publish is where the two simulators disagree, and they do —
materially. Measured, each entry with a gap number:

| behaviour | `simx` | `rtlsim` | gap |
|---|---|---|---|
| tensor unit, second CTA | **deadlocks** | completes | 7.12 |
| kernel args at `NUM_CORES` > 1 | 8 of 8 delivered | 4 of 8 → **8 of 8, closed** | 7.37 |
| misaligned 4-byte access | silent, no diagnostic | **assertion, stops** | 7.37 |
| per-core cycle counters | share an origin, skew 365 | share an origin, skew 470 | 7.25 |
| capability set, TCU-less build | n/a | reports the narrower set | 7.36 |

Three rules follow, cheap to state and cheap to enforce:

1. **Agreement is evidence; a `simx`-only pass is not a pass.** Everything in
   the table was found by running both.
2. **Disagreement resolves toward `rtlsim`**, because it executes the design
   rather than a model of it — but an `rtlsim` defect is not thereby a silicon
   defect. 7.37 lived in the simulation shim, not the RTL, and saying so was
   part of closing it.
3. **Neither is authoritative for time.** Every cycle count names its backend,
   and no cycle count in this project is a hardware claim.

The practical shape: **develop on `simx`, gate on `rtlsim`.** One is fast enough
to iterate against; the other is strict enough to be worth waiting for — it is
the backend that complains loudest, and it complained its way into both 7.27 and
7.37.

### The counterweight, which is the same product requirement

Observability is not understanding, and this is not a caution added for balance
— it is the sharpest lesson of the week.

Closing 7.37 produced **one correct root cause and three wrong explanations of
it**, sent to another team in sequence: work carried over to the next launch,
then stale reads from a shared buffer, then a claim that the old probe had been
misleading. The root cause survived from the first report because it came from a
waveform that was actually captured. All three failures were narrative built on
top of it, each produced from data nobody went back and read — one of them
refuted by an instrument built specifically to remove luck from the previous
one, and one refuted by opening the probe and finding a `grxMemset` on line 22.

A machine this observable makes confident wrong answers *cheaper*, not dearer.
Handing developers this much visibility without the discipline to go with it
produces plausible mechanisms faster than hardware ever could.

**And audience (2) is exactly where that bites hardest.** The kernel-writing
audience is small, and assistants plausibly make it larger. What changes is not
the volume of kernels but the ratio: **more kernels, less scrutiny per kernel.**
That inverts what the toolkit owes them — authoring ergonomics matter less when
authoring is cheap; what matters is whether anyone can tell a kernel is wrong.

Not speculative. A correct-looking fused epilogue was added to
`micro_tile_body` and VOLT emitted **zero** `vx_split`/`vx_join` for one
instantiation (7.27) — exactly right on uniform shapes, silently wrong the
moment a branch diverged, no diagnostic at any warning level. Reading the source
would not have found it; counting instructions in the emitted binary did.

So the deliverables for audience (2) are oracles and structural checks by
default, not a nicer `grxcc`, and they are the same three things that would have
caught this week's wrong answers:

- **Differential testing as the default**, the way `test_grxblas_rb` compares
  every tiling against the reference kernel over identical operands. A new
  kernel arrives with a reference and a comparison, or it does not arrive.
- **Structural verification of emitted code.**
  `tests/repro/sgemm_4x4_splits/count_splits.sh` becomes a tool: every kernel
  has reconvergence, no inner loop has stack traffic, entry points resolve, ABI
  versions match. A class of defect CUDA developers do not see, because nvcc is
  eighteen years old and ours is not.
- **A control in the same binary.** `GRX_NO_SETTLE` is the pattern worth
  generalising — a claim about a change should be demonstrable by toggling it,
  not by comparing two builds. Two builds differ in more ways than you listed.

And the standing rule underneath all three: **silence is a bug.** A fallback
that cannot fail is a fallback that cannot report.

### What the FVP precedent says we would still owe

Arm's model is a supported binary you install. Ours is *clone three
repositories and build a custom LLVM at a pin nothing enforces* — the pin in
`grxgpu/docs/building_toolchain.md` names one commit and the toolchain in use is
a different one, because the clone tracks a branch. We also found this week that
a clean configure of grxgpu HEAD does not build `simx` at all, because a CSR was
added to a checked-in generated header rather than to the generator input.

If the product is a developer preview on a simulator, the on-ramp stops being
cosmetic and becomes the product: one installable, a pinned toolchain, a build
that works from a clean checkout, and a fidelity statement shipped beside the
model.

## 9. Open questions

1. Which ingestion point (section 4), settled by the model survey rather than
   by discussion.
2. Whether bf16 can reach the GPU tensor unit, or whether the strategy routes
   all bf16 matmul through the NPU — which needs its bounds and its command
   queue addressed first.
3. What "a model runs" means as a gate. Suggested: end to end against a
   PyTorch reference at a stated tolerance, on a named backend, with the backend
   printed beside every number — the discipline the cycle benches already use.
4. Whether any of this precedes hardware. Phase 7 is unmet, multi-core is
   broken (7.37), and phase 3's tensor gate reads 2.51x against a 5x threshold.
   A developer preview on a simulator and an SDK for a part you can buy are
   different products with different bars, and this document does not say which
   one is being built. Section 8 sets out what the first one would owe.
5. Whether the fidelity contract in section 8 should be published rather than
   kept internal. The argument for publishing it is the same one that governs
   the conformance number: the disagreements exist whether or not we name them,
   and a developer who finds them unaided concludes something worse than what
   the table says.
