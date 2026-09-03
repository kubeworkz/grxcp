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

## 8. Audience (2), and why AI-authored kernels invert its requirements

The kernel-writing audience is small, and there is a plausible argument that
assistants make it larger. If they do, the thing that changes is not the volume
of kernels but the ratio: **more kernels, less scrutiny per kernel.**

That inverts what the toolkit owes them. Authoring ergonomics matter less when
authoring is cheap; what matters is whether anyone can tell a kernel is wrong.

This is not speculative. On 2026-09-02 a correct-looking fused epilogue was
added to `micro_tile_body` and VOLT emitted **zero** `vx_split`/`vx_join` for
one instantiation — the kernel was exactly right on uniform shapes and silently
wrong the moment a branch diverged (gap 7.27). No diagnostic, at any warning
level. Reading the source would not have found it; counting instructions in the
emitted binary did.

So for audience (2) the deliverables are oracles and structural checks by
default, not a nicer `grxcc`:

- **Differential testing as the default**, the way `test_grxblas_rb` compares
  every tiling against the reference kernel over identical operands. A new
  kernel should arrive with a reference and a comparison, or not arrive.
- **Structural verification of emitted code.** `tests/repro/sgemm_4x4_splits/count_splits.sh`
  should become a tool — every kernel has reconvergence, no inner loop has stack
  traffic, entry points resolve, ABI versions match. This is a class of defect
  CUDA developers do not see because nvcc is eighteen years old. Ours is not.
- **Silence is a bug.** The library change on the same day that made
  `read_sgemm_shape` report when a module goes quiet is the pattern: a fallback
  that cannot fail is a fallback that cannot report.

## 9. If the product is an SDK on a simulator

Section 10 leaves open whether any of this precedes hardware. If it does, there
are two precedents and they point opposite ways.

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
Siemens PAVE360 builds pre-silicon environments around Arm cores; there is a
virtual-prototyping industry underneath both. Automotive drives it because a
vehicle program cannot wait for tape-out — which is the same arithmetic as a
software schedule that has to overlap a hardware one.

**We are in Arm's position, not NVIDIA's.** An SDK on a simulator is
well-precedented. What separates the two precedents is not fidelity but
disclosure: whether the model tells you what it is.

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
| kernel args at `NUM_CORES` > 1 | 8 of 8 delivered | **4 of 8** | 7.37 |
| misaligned 4-byte access | silent, no diagnostic | **assertion, stops** | 7.27 |
| capability set, TCU-less build | n/a | reports the narrower set | 7.36 |

A developer who only ever runs `simx` ships kernels that break on RTL. A
developer who only ever runs `rtlsim` chases defects that are not in the design.
Three rules follow, and they are cheap to state and cheap to enforce:

1. **Agreement is evidence; a `simx`-only pass is not a pass.** Everything in
   the table was found by running both.
2. **Disagreement resolves toward `rtlsim`**, because it executes the design
   rather than a model of it — but an `rtlsim` defect is not thereby a silicon
   defect. 7.37's mechanism is still open and may live in the shim rather than
   the RTL, and the entry says so.
3. **Neither is authoritative for time.** Every cycle count names its backend,
   and no cycle count in this project is a hardware claim.

The practical shape that falls out: **develop on `simx`, gate on `rtlsim`.**
`simx` is fast enough to iterate against; `rtlsim` is slow enough that it has to
be a gate rather than an inner loop, and strict enough to be worth it — it is
the backend that complains loudest, and 7.27 was found because it complained.

### What the FVP precedent says we would still owe

Arm's model is a supported binary you install. Ours is *clone three
repositories and build a custom LLVM at a pin nothing enforces* — the pin in
`grxgpu/docs/building_toolchain.md` names one commit and the toolchain in use is
a different one, because the clone tracks a branch. If the product is a
developer preview on a simulator, the on-ramp stops being cosmetic and becomes
the product: one installable, a pinned toolchain, and a fidelity statement
shipped beside the model.

## 10. Open questions

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
   one is being built. Section 9 sets out what the first one would owe.
5. Whether the fidelity contract in section 9 should be published rather than
   kept internal. The argument for publishing it is the same one that governs
   the conformance number: the disagreements exist whether or not we name them,
   and a developer who finds them unaided concludes something worse than what
   the table says.
