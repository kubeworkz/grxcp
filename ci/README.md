# `ci/` — continuous integration

Two tiers, and the distinction matters.

## Tier 1 — mock build (`ci/build_mock.sh`)

Runs on every commit. Compiles the runtime and tools against the mock driver in
`tests/mock/`, then runs the unit tests and `grx-smi`. No Vortex sysroot, no
simulator, no FPGA; a few seconds.

What it proves: the code compiles and links, the device record is internally
consistent, the error surface behaves, and the honesty flags are still set.

The **DOCS GATE** (`ci/check_docs.py`) runs first, because it needs nothing
built. It opens every `path/like/this` written in backticks in every markdown
file, resolves every markdown link relative to the page making it, and checks
that a reference like `cuda_mapping.md 7.13` names a section that document
actually has. It exists because a sweep done by hand found six dangling
references that several careful readings had not — a bench file renamed out from
under two documents, three GRXGPU paths written as if they were ours, and a
phase 6 exit clause quoting a Phase 4 target that was never set. None were wrong
when written. Reading does not catch this; only opening the path does.

Made mechanical, it immediately found two more. A ci/perf/baselines/\*.json tree
was listed in `grxcp_architecture.md` section 10 as one of five verification
gates, alongside three that exist — there is no ci/perf, there never was, and the
AGENTS.md section 4 rule about never hand-editing a baseline governs no
baselines. And `gpu_chip_design.md` §15 exposed a bug in the checker itself,
which demanded a dotted section number and so mistook a reference into another
repository for a section existing nowhere.

The hand sweep globbed thirteen markdown files; this reads all seventeen, so the
five READMEs under `src/`, `tests/` and `tools/` are covered for the first time.
Fenced blocks are deliberately not read — `grxcp_architecture.md` section 9 is
an intended-layout tree listing a dozen files nobody has written yet, and
opening those would report a plan as a defect. Measured: zero reference-shaped
strings currently live inside a fence, so that rule costs no coverage today.

What it does **not** prove: that a reference points at the *right* place. A bare
`section 7.10`, three lines below a sentence naming `cuda_mapping.md`, is only
readable as such by a human; the checker requires the number to exist in some
document here, which is the strongest rule with no false positives. Paths into
other repositories — `grxgpu/...`, `c930/...`, fragments like `hw/dpi` — cannot
be resolved from this checkout and are counted and printed rather than dropped,
so if `tests/` were ever deleted its seventy-odd references would move from
checked to unchecked in plain sight instead of quietly passing.

Watched failing, on every run: `--self-test` plants one broken reference of each
of the four kinds in a fixture tree, next to a sound one of each, and fails
unless all four are caught and none of the sound ones are reported. Five
ablations turn it red — suppressing the missing-path report, reporting every
path as missing, letting a named reference ignore its leading path (which made
`grxgpu/AGENTS.md` §4 resolve against *our* AGENTS.md, silently and
successfully, since ours does have a section 4), accepting any section number,
and reading fenced blocks.

The **CROSS-DEVICE GATE** (`tests/unit/test_cross_device.cpp`) runs the runtime
against two devices on a machine that has one GPU. It is phase 7's device model,
and it needed the fixture rebuilt before it could exist at all.

`vx_device_open` returned the same `MockDevice` for every index, so
`GRXMOCK_DEVICE_COUNT=2` gave the runtime two device **slots** over one address
space and one memory pool. Every cross-device operation in CI passed for the
uninteresting reason that there was only ever one device — which is not the same
as those operations being correct. The mock now models N distinct devices, each
allocating from its own space, all starting at the same base, because that is
what per-device DDR means and because it is the case that makes an address
ambiguous.

That alone found two bugs, before any checking was written. `grxMalloc` on
device 1 returned **device 0's memory**: `take_best_fit` searched the whole free
list with no device filter, so device 1's allocation came out of device 0's slab
and was recorded as device 1's, while device 1's `grxMemGetInfo` still read zero
bytes in use. Fixing that produced the next bug on the next run, as predicted —
with device 1 allocating from its own slab both devices returned the **same
address**, and `g_live`, keyed by address alone, kept one of the two records.
The map is keyed by `(device, address)` now and the free list by
`(slab, address)`.

On top of that: a pointer live on another device and on no local allocation is
refused by `grxMemcpy`, `grxMemset` and `grxFree`. **The limit is stated rather
than hidden** — when an address is live on *both* devices nothing in a bare
`void*` says which was meant, so it is used as the current device's. Kernel
arguments are not covered: the launch path takes an opaque blob with no type
information, and a blob scan would be a heuristic, so it is written down in the
roadmap instead of shipped quietly.

Watched failing, four ways: the free-list device filter removed (device 1 stops
consuming device 1's memory), the `(device, address)` guard in the lookup
removed (all three refusals stop firing), the foreign-pointer refusal in memcpy
removed (two of them stop), and the module capability check removed. A fifth
ablation — resolving `grxFree`'s target by owner rather than by current device —
deadlocked, because `owner_device_of` takes the allocator lock; that hazard is
now written at its declaration.

The test carries its own positive control. Without it a runtime that refused
every pointer on every device would pass all four refusal assertions.

The order of the test's own steps is load-bearing and was wrong at first: device
0 has to allocate **before** device 1, or the free list is empty when device 1
asks and there is nothing to steal, and the first ablation passes. The
device-0-pointer-on-device-1 case has to run while device 1 owns **nothing**, or
the lookup stops at device 1's own entry and the second ablation passes. Both
were found by ablating, not by reading.

The **DISPATCH GATE** (`tests/unit/test_dispatch.cpp`) is phase 7's scope item
4 — "a documented, explicit rule for which engine a library call lands on. Not
automatic magic." The rule is that **the current device decides and nothing else
does**: no redirect, no fallback to another engine, a refusal instead. What was
missing was any way to observe it, and the decision was wrong.

The NPU check read `grxGetDeviceProperties(&prop, grxGetDevice(nullptr))`, and
`grxGetDevice` writes through its argument and **returns an error code** — with
`nullptr`, `grxErrorInvalidValue`, the value 1. So the routing consulted device
index 1 on every call and the current device on none of them. On a one-GPU
one-NPU machine, where the NPU is appended at index 1, that means an INT8
`GemmEx` issued while the current device was the **GPU** routed to the NPU: work
on silicon the caller did not select, with no error anywhere.

`grxblasGetGemmEngine` now reports the engine **and the device index the
decision was made about**, from the same function that makes it, so the answer
cannot drift from the behaviour. That index is the assertion, and it needs no
NPU: the gate runs with `GRXMOCK_DEVICE_COUNT=2` and checks that the reported
index follows `grxSetDevice`. With one device that is unobservable and the test
says so rather than passing quietly on the weaker configuration.

Watched failing two ways: restoring the `grxGetDevice(nullptr)` expression
("device 0: the decision was made about device 1"), and reporting a fixed 0
instead of the index decided about. There is a positive control — a well-formed
question must still be answered — so a build that refused every query would not
pass the three refusal cases.

A `grxblasSetPreferredDevice` was **declined**, and the reason is written in
`grxblas.h` next to the rule: a redirect is invisible at the call site, so the
same source line would run on different silicon depending on state set
elsewhere. Asking where work lands is the need; moving it silently is not.

Linking these tests taught the build system something too. `tests/CMakeLists.txt`
matched library dependencies by what a source **includes**, but only for
`tests/libs/`; a unit test that asks grxBLAS a host-side question linked nothing
and failed at the CMAKE GATE. The same include rule now covers `tests/unit/`,
and `ci/build_mock.sh` uses it as well.

The **ABI GATE** (`ci/check_abi.py`) compares `sizeof`, `_Alignof` and every
field's `offsetof` for the shared structs across four targets: x86-64 host,
riscv64 host, device rv64, device rv32. The kernel argument structs are
**written by the host and read by the device**, and a disagreement about one
offset is not an error anywhere — the kernel reads the blob where it was
compiled to and returns a wrong answer. `grxblas_sgemm_args` carries an
`abi_version` first field because that failure was anticipated; a version
catches a struct that *changed*, not two compilers that lay the same definition
out differently. Nothing had ever compared them, and until phase 7 the host was
x86-64 in every configuration anyone built.

Two of the four targets cannot run here — the device images are bare-metal
RISC-V for a core this container does not have — so each fact becomes an array
whose **length** is the number and `nm --print-size` reads it out of the object
file, executed or not.

The probe is **generated from the headers**, every field of every struct.
It was a hand-picked list first, and ablation killed that: planting a `size_t`
at the front of `grxdnn_gelu_args`, a struct the list covered by size only,
moved nothing the gate looked at. LP64 grew the struct by 8; ILP32 grew it by 4
then padded 4 to realign the `uint64_t` behind it, so both sizes moved by 8 and
stayed equal while every following field sat somewhere else on the device. Sizes
do not catch a layout change; offsets do, and a list of them written by hand is
one new field away from missing the field that breaks. A declaration the parser
cannot read is now a failure, not a silent omission.

Headers are annotated by **boundary**, and the annotation is checked. The first
run reported nine failures in `grx_abi.h`'s `grx_kernel_desc` — 40 bytes on LP64
and 32 on ILP32, because it holds a `const char*` and a `const
grx_kernel_param*`. Not a defect: **no device source includes that header.**
grxcc writes it and the runtime reads it, both compiled for the same host, so a
native pointer there is legitimate. Host-only headers are compared across the
hosts alone, and a device kernel that starts including one fails the gate — the
second ablation.

Where the discriminating power actually is, measured rather than assumed: the
two hosts are both LP64 and agree on every layout these headers can express. The
live edge is host against **device-rv32**. The riscv64 host leg earns its keep by
*executing* — alignment, argument passing, struct return — not by disagreeing
about offsets.

The **HOST MATRIX** now runs by default. `ci/build_mock.sh --host
riscv64-linux-gnu` has been able to cross-build and *run* the whole mock stack
under qemu-user since phase 4, behind a flag nobody's CI passed; "available" is
not "checked", and a leg nobody invokes reports nothing. The native leg takes 33
seconds and the riscv64 leg 15, measured, and the second runs every gate except
the two that say why they cannot. `GRXCP_NO_HOST_MATRIX=1` turns it off.

It proves there are no x86-isms and that structs survive a different ABI. It
proves **nothing** about a weak memory model, which qemu-user does not model,
and nothing about a GRX930 board.

The **NPU BACKEND GATE** drives `src/backends/npu_c930/` through four register
models — a bus with nothing on it, a bus that floats high, a live device, and a
device that accepts a launch and never finishes. The backend has no Vortex
dependency, so this needs no sysroot and no c930; register access goes through
injectable hooks precisely so those four states can be produced on a machine
that has none.

Two of them found bugs. `npu_c930_detect` read `STATUS` and accepted anything
that was not `0xFFFFFFFF` and not above `0x7` — its own comment said an absent
NPU reads `0x0`, and `0x0` passes that test, so any host where `/dev/mem` opens
grew an NPU it did not have. `npu_c930_gemm` waited for `!BUSY` with no `ERROR`
and never read the latched `DONE` bit, so a device that ignored every write
looked finished and the function reported success over a GEMM that never ran,
leaving `C` holding whatever it held before. Both fixed and both watched
failing: ablating the write-readback probe fails exactly the absent-bus case,
ablating the `DONE` check fails exactly the wedged case.

**A register model is not hardware.** A green run here says this backend's logic
is right. It says nothing about the c930, and must never be reported as the NPU
working — the same rule `tests/mock/` lives under.

The **NPU GROUNDWORK** gate is phase 7 work that can be checked before there is
an NPU. `grxcp_architecture.md` section 6 fixes the c930 NPU's profile as
`GRX_CAP_GEMM` without `GRX_CAP_KERNEL_LAUNCH`, and fixes what a launch on it
must do: return `grxErrorNotSupported`, and **not** silently run the work on the
GPU. A fallback is the bad outcome precisely because it is not a wrong answer —
it is the right answer on the wrong engine, invisible until someone measures.

The runtime derives that capability from the device's own warp geometry rather
than from a device-type test, so the mock reaches the condition with
`GRXMOCK_NUM_WARPS=0` and no capability ID has to be invented for hardware that
does not exist yet. The gate runs the binary twice — once with zero warps, once
on the default device — because a refusal on its own would pass against a
runtime that refused every launch everywhere. Watched failing: with the
capability check removed, the zero-warp launch comes back as `launch exceeds a
per-core resource bound`, which describes a grid that does not fit rather than a
device that cannot run grids, and would send someone off to shrink their block
size.

It also runs the **CMAKE GATE**, which configures and builds the project through
its own `CMakeLists.txt` and checks that every library and tool the file claims
to produce exists. Everything else in tier 1 compiles GRXCP by hand, so nothing
had ever run the build system this project ships to its users — and it did not
work. The top level had `if(X) add_subdirectory(y) endif()` on one line, which
CMake rejects as a parse error rather than accepting as a terse spelling of the
same thing, and `src/CMakeLists.txt` and `cmake/grxrt.pc.in` did not exist at
all. That went unnoticed from the first commit until the GRX930 team asked
whether `cmake -DGRXCP_ENABLE_NPU=ON` builds.

That flag is checked here too, and it is checked for **building**: the NPU
backend exists in `src/backends/npu_c930/`, so both configurations are
configured and built, because "the GPU path is unchanged when the flag is off"
is a claim and not an assumption. The gate also asserts that the NPU test
targets were produced — `GRXCP_ENABLE_NPU` used to switch the source glob and
nothing else, so `npu_c930.cpp` went into `libgrxrt` while every
`#ifdef GRXCP_ENABLE_NPU` block in `context.cpp` compiled to nothing — and that
**no NPU is enumerated** on this machine, which has none. A build flag says what
code exists, not what hardware is attached.

What it does **not** prove: anything about hardware. The mock returns synthetic
capability values. A green tier-1 run says the runtime is not broken; it says
nothing about whether the device works.

```sh
./ci/build_mock.sh                 # finds vortex2.h from $VORTEX_PATH
./ci/build_mock.sh --vortex-include <dir containing vortex2.h>
```

## Tier 2 — real backends (`ci/run_real.sh`, `ci/testcases/grxcp.yaml`)

The declarative catalog, in the same shape as the GRX-G100 project's. Every
conformance case runs on **both** `simx` and `rtlsim`; a result that differs
between the two blocks the merge, because it is either a GRXCP bug or an
upstream model-parity bug and both matter.

Tier 2 needs an installed GRX-G100 sysroot. `ci/build_sysroot.sh` builds one
from a grxgpu checkout without needing the RISC-V toolchain, LLVM or Verilator
— the driver and the SimX backend are plain C++:

```sh
./ci/build_sysroot.sh --grxgpu ../grxgpu
export VORTEX_PATH=../grxgpu/build/install
./ci/run_real.sh
```

`ci/run_real.sh` compiles the runtime against the real driver, runs `grx-smi`,
runs the unit tests that do not need a device binary, and translates a CUDA
source with grxify then compiles, links and runs it. `$VORTEX_DRIVER` selects
the backend and defaults to `simx`.

**The Phase 0 exit gate is a tier-2 result**, and it passes on `simx`. `rtlsim`
additionally needs Verilator and the `hw/dpi` sources, so it is not yet part of
this script.

`test_launch` runs in tier 1 only. It builds modules in the mock driver's own
image format, which the real loader has no reason to accept.

## Tier 3 — kernels

Running an actual kernel needs the device toolchain (VOLT plus the RISC-V
binutils), about 580 MB:

```sh
./ci/install_toolchain.sh --tooldir $HOME/tools --grxgpu ../grxgpu
# re-run the sysroot build with the toolchain visible: it then also builds the
# device-side CTA runtime (libvortex2.a) with the same CONFIGS as everything
# else. See "configuration provenance" below for why that matters.
./ci/build_sysroot.sh --grxgpu ../grxgpu --tooldir $HOME/tools \
  --configs "-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE \
             -DVX_CFG_TCU_INT8_ENABLE"
./ci/run_real.sh --grxgpu ../grxgpu --tooldir $HOME/tools
```

`ci/build_kernel.sh` compiles a kernel into a `.vxbin` — what grxcc will do
internally in Phase 4. `tests/kernels/vecadd/` is the Phase 1 exit gate: a
kernel written against GRXCP's own device header, launched through
`grxLaunchFunction`, with the arithmetic checked on the host at sizes that
exercise the partial-warp path.

`tests/kernels/cycles/` is the calibration weight for device-cycle
measurement: it runs the same kernel at 1x, 2x and 4x the work and fails unless
the measured count follows. Nothing else in the tree may quote a cycle figure
without this passing — a measurement nobody has watched respond to its input is
not a measurement. `tests/bench/gemm_cycles.cpp` then reports what sgemm v0
costs, which is the baseline the tuned tensor-core kernel is gated against;
it is a report, and it fails only if the measurement itself is broken.

Event elapsed time is **not** a measurement of the device — the driver's
timestamps are host clocks, so on a simulator they measure the simulator. See
`include/grx/grx_cycles.h`.

`tests/kernels/dxa/` is the DMA gate: the host programs a tensor map, a kernel
stages a tile of a **padded** 2D array into shared memory through the engine,
and the host checks every element — in both the row-major and the transposing
destination layouts. Its tier-1 counterpart is `tests/unit/test_tensormap.cpp`,
which reads back the device-config-register writes the mock driver captured:
that half says the registers hold what they should, this half says the engine
does what the registers say.

`tests/kernels/wmma/` is the tensor gate: one WMMA tile through
`grx::wmma`, compared **exactly** against a CPU reference. The host asks the
device for its tile shape rather than assuming one, and cross-checks the warp
width the kernel was compiled for against the width the runtime reports — a
disagreement there means the module and the runtime came from different
configurations, which is the failure "configuration provenance" below exists to
prevent. It skips when the device reports no tensor unit.

`tests/libs/test_grxblas_ex.cpp` covers `grxblasGemmEx` in all four transpose
combinations, including the ragged-`k` shapes. Those are the interesting ones:
transposing an operand moves `k` between the DXA descriptor's dimension 0 and
its outer dimension, and only the outer one is padded. TN puts `k` in
dimension 0 for both operands, so the kernel zeroes the staged tail itself
rather than inheriting a zero from whichever operand happened to be padded.
Watched in the failing direction — with the zeroing removed, exactly the TN
ragged-`k` cases fail and the other three stay correct.

`tests/libs/test_grxblas_l12.cpp` is the level-1/level-2 gate: saxpy, sscal and
sgemv, with every comparison EXACT. The values are small integers held in
floats, so every summation order gives the same bits and no tolerance is
available to hide a wrong answer behind — unlike the sgemm gate, which needs
one because the device accumulates in a different order. It has been watched
failing: reverting sgemv's transposed load to the classic wrong index makes all
five transposed cases fail and leaves the untransposed ones passing.

`tests/libs/test_grxdnn.cpp` is the grxDNN gate: softmax and layer norm against
a host reference on five shapes — rows shorter than a warp, rows longer than
one, a padded leading dimension, in place, and the argument checks that must
refuse. Two cases carry numerical controls, and the controls are themselves
checked to discriminate: a row that overflows the naive `exp` before the max is
subtracted, and a row whose mean is large next to its spread, which the one-pass
`E[x²] − E[x]²` variance gets wrong. Watched failing for real — the first
`dev_rsqrt` was the `0x5f3759df` estimate with one Newton step and a comment
claiming 2e-6 relative error. It is 1.7e-3. Every layer-norm case failed and
every softmax case passed, which is the signature of the only thing the two do
not share.

The **GELU GATE** (`tests/libs/test_grxdnn_gelu.cpp`) is where the device's own
transcendentals become visible. The device build is `-nostdlib`, so grxDNN
carries its own `exp` and builds `erf` and `tanh` on top of it, and GELU is the
only op that uses them. The gate therefore **prints what it measures** rather
than only passing — the accuracy figures in `kernels/elementwise.cpp` are a
record of that measurement, swept over [-1000, 1000]:

    gelu, exact (erf) form    worst 5.36e-07 absolute, at x = 0.8158
    gelu, tanh form           worst 4.77e-07 absolute, at x = 4.0401

Those are larger than the erf polynomial's own quoted 1.5e-7, so `dev_exp` and
fp32 rounding contribute rather than the polynomial dominating — the sort of
thing only a measurement tells you, and the reason none of these numbers are
quoted from a table. This library has been wrong about exactly that before:
`dev_rsqrt` shipped claiming "about 2e-6 relative" for something that is 1.7e-3.

The two GELU forms are different functions, 4.73e-04 apart. The gate checks both
against PyTorch **and** checks that the device produces measurably different
answers for them — without which a run that passed both would mean the mode
argument was being ignored. Watched failing two ways: indexing the bias by row
instead of column (worst 3.76, and the row-invariance control fails too), and
swapping the two GELU modes (both form checks fail by exactly 4.73e-04). The
second leaves "the two forms differ" passing, which is correct — that check
catches a mode being *ignored*, not one being *inverted*, and neither check
covers the other.

The **PHASE 6 EXIT GATE** (`tests/libs/test_grxdnn_block.cpp`) is the phase the
whole library was building toward: a complete pre-norm transformer block —
layer norm, QKV projection, causal attention, output projection, residual, layer
norm, MLP with GELU, residual — run end to end through GRXCP and compared to a
PyTorch block at **every one of its twelve stages**.

Every op in it is already gated on its own. What no single-op gate can see is
whether four correct ops *composed* are still correct: each fixes its own layout
convention and checks it in isolation, and composing them means a transposed or
mis-strided hand-off between two correct kernels, which produces plausible
numbers that nothing upstream would catch.

`tests/libs/block_ref.py` dumps every intermediate, not just the block's output,
and the gate compares them in order and **stops at the first disagreement** —
so a failure names the op rather than reporting that a ten-stage block is wrong
somewhere.

Watched failing two ways. Making the output projection overwrite per head
instead of accumulating leaves the H=1 case passing entirely — correctly, with
one head there is nothing to accumulate — and fails both H=2 cases at exactly
`p`, with every earlier stage green; that is a hand-off bug between two
individually-gated ops, which is what this gate exists for. Substituting the
exact GELU where the weights expect the tanh form fails at exactly `act` by
2.33e-04, and that ablation found a defect in the gate itself: the original
tolerance was three orders of magnitude looser than anything observed and let
the wrong activation through on two of three cases. It is now set from the
measurement.

**BLOCK CYCLES** (`tests/bench/block_cycles.cpp`) is a report rather than a
gate, except for its last section — which is a gate, and is the only reason the
rest is worth reading. The block runs at two sequence lengths and attention's
**share** must grow, because its scores matrix is seqLen squared while every
other stage is linear in seqLen. "The total went up" would be satisfied by
almost anything, including a counter that was really counting launches; that
differential would not.

Running it found that every grxDNN kernel constructed a `grx::cycle_probe` and
never called `finish()`. The slot was never written, the host summarised an
array of zeros, and nothing reported an error — the instrumentation had been
dead in every kernel since it was written, and only pointing it at a real
workload surfaced that, because nothing else had ever read a grxDNN slot. The
probe now finishes itself in its destructor and `finish()` is idempotent, so it
cannot be forgotten and grxBLAS's explicit calls still behave identically.

The first version of the bench also had every stage reporting zero because its
probe array was a `std::vector` — a host pointer the device cannot write. That
is now stated in `grxdnn.h` where the API is declared, since it produces a
silent record of nothing rather than an error.

What the report says, and it is not what intuition said: the GEMMs are **76%** of
the block in the four stages named as GEMMs — **80.3%** counting attention's own
two — GELU is **11.5%**, more than attention and both layer norms together
because the transcendental costs about 8.5× a bias add per element, and fusing
the bias into the GEMM epilogue, the obvious next optimisation, would save about
**2%**. See the roadmap's phase 6 progress note for the table.

The bench now also writes its raw spans with `--out`, and runs a second time
with `GRXBLAS_SGEMM_NAIVE=1`. That second run is the only bench execution added
for the baselines rather than reused, and it earns its minutes twice: it pins
the register-blocking speedup to a measured pair, and it is the control that
proves the kernel selection does anything at all.

The **SGEMM CROSSOVER GATE** (`tests/bench/sgemm_sweep.cpp`) is the sweep the
roadmap said was owed, and it is the rare gate that exists to record a
contradiction rather than to assert a rule.

The kernel-selection rule — `k >= 16 || ceil(m/RM) >= warpSize` — was fitted to
five points from the block profile. Swept properly, 66 shapes across m, n and k
with both kernels over the same operands: **k never changes which kernel wins**
anywhere in the range (it moves the magnitude, 1.17× at k=4 against 1.53× at
k=32), and the coalescing boundary is not at m = 16 — m = 8 wins at n = 16.
What does predict the isolated GEMM is `m*n*batch >= 2 × resident threads`,
which explains **all 66 cells** where the shipping rule is wrong on 19. Cells
with equal m·n agree to two decimals however m and n split, and batch scales it
exactly.

**And shipping that better rule made the transformer block slower** — 230171
cycles against 226405 at S=8, because attention's scores GEMM loses inside the
block while winning by 1.39× in isolation. Transpose was checked and explains
none of it. So an isolated sweep does not predict the workload, nobody here
knows why yet, and the rule that wins on the block is the one that ships with
its comment corrected to say its stated mechanism is disproven.

What the gate therefore checks is that the **disagreement does not grow**: no
more than the 3 recorded shapes where the rule picks the slower kernel, no more
than the 6 where it declines a faster one, and that the output-count rule still
explains every isolated cell — because that last one is what makes the block's
disagreement worth chasing rather than a rounding artifact. 21 seconds.

`GRXBLAS_SGEMM_RB` exists for this bench and only for it: the rule refuses the
blocked kernel at shapes it dislikes, so measuring whether the refusal was right
needs a way to run it anyway.

The **PERF BASELINE GATE** (`ci/check_perf.py`, `ci/perf/baselines/`) compares
152 measured cycle counts against stored golden files. AGENTS.md section 4 had
said since the first commit that a moved number means real cycles moved, and
`grxcp_architecture.md` section 10 listed `ci/perf/baselines/` among five
verification gates — while no such directory existed and nothing compared any
number to anything. The benches printed and a human was expected to remember
last week's figure.

Tolerance is **zero**, and that is a measurement rather than a preference: three
consecutive runs of both benches were byte-identical, every stage and every
shape, and rebuilding `src/libs/kernels_all.cpp` produced a bit-identical
`.vxbin`. Only raw integers are stored — shares, cycles per element and speedups
are derived at compare time, because a stored `4.6%` drifts against its own
rounding and needs a tolerance to survive it while a stored `13834` does not.

Three things it refuses to do. It will not compare across a **different device
record** — a different core width or SM count is not a regression and not a
pass, it is numbers about another machine, so that outcome is REFUSED rather
than failed. It will not report a **renamed or reordered stage** as a
regression; structure is part of the metric key, so that is a separate verdict
with a separate fix. And it will not run the benches itself: the two steps above
already wrote the files, so what is gated is exactly what was printed.

It is also the one gate here that is **deferred rather than fatal**. Everything
else exits on the spot; a moved cycle count must not hide a correctness
regression in a gate below it, so this one is recorded and re-raised at the end.

Watched failing, twice over. Planting a wasted 32-iteration loop in `dnn_gelu`
turned it red with gelu the largest mover at **+42.3%**, five times the next
entry — and moved all ten other stages too, between **−6.0%** and **+8.7%**,
with their source untouched, because the `.vxbin` relinks and every kernel's
addresses move with it. The planted build was itself reproducible run to run,
and reverting restored all 152 metrics to exact, so that spread is layout and
not noise. The report therefore ranks movers by magnitude and names the largest
on its own line: when the image changed, rank is the signal; when it did not,
the exact value is. A band wide enough to absorb 8.7% of layout would also
swallow a genuine 8% regression, which is the worse trade.

Separately, `check_perf.py --self-test` runs in **tier 1**, where there is no
device: it checks that the comparator refuses a different machine, fails on one
cycle of drift, calls a renamed stage a structural change, honours a declared
tolerance *and says so*, and that two configurations with identical cycles fail
the naive/register-blocked control. Four ablations turn it red. That self-test
was itself wrong once — it read the real baselines directory, so the day real
baselines appeared its missing-baseline case stopped reaching the
missing-baseline path.

Regenerating is `ci/check_perf.py --results build-real --regenerate`, and the
point of it is the diff: a moved number is meant to arrive in review beside the
code that moved it, never as a quiet edit to make a red gate green.

The **TEXTURE GATE** (`tests/kernels/texture/`) closes phase 6's coverage
clause, and what it gates is arithmetic and honesty — never hardware. The TEX
units are driven by the graphics path and unreachable from compute
(`cuda_mapping.md` 7.8, still open), so `grx::tex<>` addresses and filters in
SOFTWARE, in the calling warp. That is architecture section 10 rule 5's
sanctioned exception and nothing wider: an emulation reported through a device
property, as the warp-shuffle fallback is. **The flag is part of the gate** —
`grxDeviceProp_t.textureIsEmulated` must read 1, so a green run that quietly
stopped reporting the emulation fails. A phase gate closed by counting entry
points that pretended to be hardware would be the worst outcome available.

The reference is **PyTorch's `grid_sample`**, for the same reason the attention
gate uses PyTorch. A sampler is almost entirely convention — where a texel
centre sits, which way floor rounds a negative coordinate, what clamp means at
the far edge — and a reference written from the same reasoning as the
implementation agrees with it whether or not either is right. `align_corners
=False` is exactly CUDA's convention, and three of its padding modes are three
of our four address modes. WRAP has no equivalent and is checked against a
from-specification reference instead, labelled as the weaker check it is. 81
coordinates, inside and well outside on every side, against a field with
curvature: a linear ramp is the classic bad texture fixture, since bilinear
interpolation of a linear function is exact whatever weights you use.

Two things building it turned up.

`__builtin_floorf` **does not compile on a divergent value** for this device —
`error: unimplemented divergent codegen found!`. Measured across the family,
because which ones is the difference between a workaround and a superstition:
`floorf`, `ceilf`, `truncf`, `roundf` and `rintf` all fail; `nearbyintf`,
`fabsf` and `sqrtf` all compile. The five that fail lower to a float→int→float
sequence with an explicit rounding mode. No device code in the tree had ever
needed a rounding builtin, so this is the first thing that could have found it
(`cuda_mapping.md` 7.24).

And an **unexplained one, recorded as unexplained.** Written with a divergent
early return, the texel fetch produced wrong values in the border address mode
at 64 threads — three modes agreeing with PyTorch to 2e-7 and border out by
8.9e-1. The branch-free spelling, semantically identical, agrees everywhere. A
single-lane probe of the original is correct with every intermediate correct,
and **three minimal reproducers** — a bare divergent early return, one with a
bool out-parameter, one with the out-parameter set inside a switch — all run
correctly. No compiler defect was isolated and none is claimed; the repro that
did not reproduce was deleted rather than kept as a watch that always reports
FIXED. The workaround ships with that written at it.

The **ATTENTION GATE** (`tests/libs/test_grxdnn_attn.cpp`) is the last quarter
of the phase 6 exit gate, and the only gate here whose reference is a third
party's arithmetic. Attention is where grxDNN's row-major convention meets
grxBLAS's column-major one — twice, once transposed — so it is almost entirely
index bookkeeping, and a reference written from the same reasoning as the
implementation agrees with it whether or not either is right.

So the expected values come from `torch.nn.functional.scaled_dot_product_
attention` in float64. `tests/libs/attention_ref.py` generates them and checks
them in as `attention_ref.bin`, so this gate needs neither Python nor torch.
That script also simulates the exact `grxblasSgemm` calls the implementation
makes — leading dimensions, transpose flags, flat memory — and **refuses to
write the vectors** unless the simulation reproduces torch to 1e-12. The layout
algebra is settled on a laptop before a device sees it, which is why attention
passed on the device on the first run.

Watched failing three ways. Flipping `transa` fails everything but the 1×1 case,
caught by grxBLAS's own leading-dimension check before any arithmetic. Passing Q
and K in the order the formula reads — dimensionally valid, silently computes
scoresᵀ — fails every non-trivial case by 0.117 to 0.316; that is the mistake a
careful person makes, and nothing but an outside reference catches it. Removing
the causal mask fails **only** the two causal cases, so the mask does real work
and the five unmasked cases are not accidentally masked.

The 1×1×1×1 case is in the file to exercise the degenerate path, not the
algebra: every transpose of a 1×1 matrix is itself, so it passes under every
ablation above and proves nothing about layout on its own.

The **CROSS-LIBRARY GATE** (`tests/libs/test_libs_together.cpp`) is the one that
matters for the phase 6 exit gate, because that gate is a transformer layer and
a transformer layer is two libraries in one process. grxBLAS and grxDNN are
called interleaved — blas, dnn, blas, dnn — with the numbers checked on both
sides of the other library's call, and then grxBLAS's handle is destroyed while
grxDNN keeps using the module.

It runs **twice**. Once against the shared image, where it must pass; once
against a directory holding two separate per-library images and no combined one,
where it must fail *and* the log must contain `address range overlaps`. A
control that fails for some other reason is reported as a failure of the gate,
not as a pass. Both halves of the fix were watched failing: without
`kernels_all.cpp` the two images collide, and without the reference count in
`grxModuleUnload` exactly one case fails — the one where grxBLAS is destroyed
first, which is what a fix that unloads to make room would ship with.

`tests/kernels/cg/` is the cooperative-groups gate: `thread_block`,
`thread_block_tile` at two widths, `coalesced_group` taken inside a divergent
branch, the cluster, and `this_grid().sync()` through a cooperative launch, all
against references the host computes independently.

The grid barrier carries its own control, and it is the interesting part. Block
0 stalls before publishing, and the gate then runs the same kernel with the
barrier removed and requires it to get the answer WRONG. Without the stall both
blocks would publish long before either read, and the barrier test would pass
whether or not the barrier worked.

The **PROF GATE** runs `vecadd` under `grx-prof` at three sizes. Three of its
checks are about the trace being readable — it parses, the kernel slice carries
a device cycle count, and the report states which of its numbers are host-clock
— and the fourth is the one that matters: the device cycle count has to climb
with the work. A profiler emitting numbers nobody has watched respond to their
input is not measuring anything, which is the same reason `tests/kernels/cycles/`
exists. Tier 1 checks the other side: the mock driver refuses
`vx_device_mpm_query`, and no `device.*` argument may appear on any slice taken
against it. Absent, not zero. See `docs/designs/grx_prof.md`.

`tests/kernels/sanitize/` is the memory-checking gate. `build_kernel.sh
--sanitize` compiles it with AddressSanitizer's checks outlined into calls that
`src/device/grx_sanitize_rt.cpp` answers from the allocator's own map, and the
gate requires each of four planted bugs — an overflow, an underflow, a
use-after-free, a shared-memory overrun — to be reported *at the line it lives
on*. The line numbers are grepped out of the kernel source by marker comment,
so moving the code does not silently disarm the check.

It carries two controls, because a detector is only as trustworthy as its
negative case. The same kernel with the bug removed must come back clean, and
the same bug in an **uninstrumented** build must be reported as *unchecked*
rather than as clean — otherwise every build that forgot `--sanitize` would
pass this gate forever. See `docs/designs/grx_sanitize.md`.

The **stream-overlap WATCH** (`tests/repro/stream_overlap/`) answers a question
the roadmap gates all of phase 5 on: do two streams run at the same time? Two
kernels rendezvous through a device global, one per stream, waiter enqueued
first. Only a sighting **mid-spin** proves overlap — a sighting at iteration 0
means the setter finished before the waiter began, which is reordering, and a
first version that counted it as overlap reported overlap half the time on a
device that has none.

Its two controls DO gate, and that is the point: one checks the rendezvous works
at all, the other checks the detector can see a mid-spin sighting by having the
waiter set the flag itself halfway through. Without the second, "never saw it"
could equally mean the measurement is blind. The overlap answer itself is a
watch — exit 0 either way, read the message.

The **BARRIER GATE** (`tests/repro/barrier_duplication/`) is half gate and half
watch, because it covers a defect GRXCP works around rather than owns. Two
kernels do the same work behind a divergent branch: one calls GRXCP's
`__syncthreads()`, one calls upstream's bare `vx_barrier`. The first **must**
pass — it is the workaround, and a regression there is ours. The second is
expected to deadlock, and runs in a child under a timeout so CI reports the day
the toolchain stops duplicating it instead of hanging the run. `cuda_mapping.md`
section 7.20 has the disassembly and the attribute ablation.

The **PHASE 4 GATE** compiles `tests/grxcc/vecadd.grx.cpp` with `grxcc` and runs
it. One file, `__global__` kernels and `<<<>>>` launches in the same translation
unit, no module load and no `.vxbin` named anywhere. The sample checks computed
VALUES rather than return codes, because a mispacked argument blob produces a
wrong answer and not an error — a gate that only checked `grxSuccess` would pass
a driver that put every argument at offset zero.

`tests/grxcc/scopes.grx.cpp` covers the parser, which fails differently: a
mis-lexed file produces mangled generated source, usually with an error naming
something the author never wrote. It puts kernels at three scopes — file,
anonymous namespace, nested named namespace — and surrounds them with decoys: a
`__global__` in a comment, a `<<<` in a string and in a raw string, a namespace
alias. It too checks values, so a kernel that was parsed but never registered
cannot pass.

`tests/grxcc/attributes.grx.cpp` covers `__launch_bounds__` and the register
metadata, and both are checked against controls rather than against themselves.
A bounded kernel must REFUSE an oversized block *and* an otherwise identical
unbounded twin must ACCEPT the same launch — without the second half, a runtime
that refused every large launch would pass. A register-hungry kernel must report
MORE registers than a trivial one — without that, a driver returning any
constant would pass, which is exactly the failure the -1 sentinel existed to
prevent. Both ablations were run and both fail the gate. It also loads a
`.vxbin` nobody measured and requires -1 back, so "unmeasured" cannot quietly
become "zero"; that check runs FIRST, because only one module can be resident at
a time and touching the file's own kernels takes the slot.

Four negative controls follow, one per documented limit, because a limit that is
written down but not enforced is a bug with a paragraph attached. `grxcc` must
REJECT a launch of a name that is not a `__global__`, a templated kernel, a
kernel that is not at namespace scope, and two kernels sharing an unqualified
name — the last because the device entry point comes from the unqualified name,
so they would be one symbol on the device.

The **HOST MATRIX GATE** compiles a grxcc program's host pass for
`riscv64-linux-gnu`, because GRX-G100 hangs off a GRX930 and that is a RISC-V
SoC. It compiles and does not link: the driver in the container is x86_64, so a
riscv64 object has nothing to link against. The stronger claim about the runtime
belongs to `ci/build_mock.sh --host riscv64-linux-gnu`, which cross-builds the
whole mock stack and RUNS it under qemu-user — a planted
`__builtin_ia32_rdtsc` was watched passing the native build and failing that one.

The **CUDA SAMPLES GATE** builds and runs `tests/cuda_samples/`: eleven CUDA
programs whose only concession to GRXCP is including `grx_cuda_compat.h`. It is
the phase 4 exit gate's second claim, and its value is entirely in what the
first pass found — eleven failures out of eleven, listed in that directory's
README. The eleventh sample, `11_histogram_atomics.cu`, is checked the other way
round: on a build with no A extension it must REFUSE to compile with a message
naming the reason, because the alternative is an AMO the simulator aborts on
silently. On a build with the extension it is expected to compile and run, and
the gate reads the device's own capability bits to decide which it is.

Without a toolchain, `run_real.sh` **skips** that gate and says so, rather than
reporting a pass over work that never ran. The grxBLAS gate behaves the same
way: no kernels built means it exits 77 (skip), because "nobody compiled it"
must not read as "the GEMM is broken".

## Configuration provenance

A GRX-G100 sysroot is built for a *particular machine*. `VX_config.toml` is the
small FPGA baseline — no tensor unit, no DMA engine — and a real configuration
is a `CONFIGS` override on top of it:

```sh
./ci/build_sysroot.sh --grxgpu ../grxgpu --tooldir $HOME/tools \
  --configs "-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE \
             -DVX_CFG_TCU_INT8_ENABLE"
```

`VX_CFG_TCU_INT8_ENABLE` is in that list because the tensor unit's input types
are a build-time choice and int8 is off by default: without it
`grxblasGetTensorTypes` reports fp16 alone and there is nothing to gate an int8
GEMM against. `VX_CFG_TCU_WGMMA_ENABLE` is deliberately **not** in it — turning
it on was tried, it does not fix the multi-CTA deadlock
(`docs/designs/cuda_mapping.md` 7.12), and enabling a feature nothing exercises
only widens the gap between what the sysroot claims and what has been run.

Here is the trap, and it is a quiet one. The installed sysroot records nothing
about how it was configured: there is no generated config header in the install
tree, and `vortex-kernel.pc`'s `Cflags` carry only an include path. So anything
compiling device code afterwards falls back to the repo's baseline toml. The
result is a runtime that reports `tensor` in its capability list and a kernel
compiled as though the tensor unit does not exist — and a tensor test that
passes having tested nothing.

Three things keep that from happening:

1. `ci/build_sysroot.sh` writes the `CONFIGS` string it used to
   `$VORTEX_PATH/share/grxcp/device_configs`, and builds `libvortex2.a` with
   the same string when a toolchain is available.
2. `ci/build_kernel.sh` reads that file, feeds it to `gen_config.py`, and
   **prints** the resolved feature bits before compiling. `--configs`
   overrides; a sysroot with no record produces a warning, not silence.
3. `grx_wmma.h` and `grx_pipeline.h` refuse to compile when the configuration
   they are being built for lacks the unit they exist to drive. That backstop
   is deliberately at the header, so it holds no matter how the kernel is
   built.

The right long-term fix belongs upstream: the sysroot should describe its own
configuration, ideally through `vortex-kernel.pc`. Item 1 is shaped so that
switching to an upstream mechanism is a one-line change.

### The configuration this project builds against

```
./ci/build_sysroot.sh --grxgpu <path> --tooldir <path> \
  --configs "-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE \
             -DVX_CFG_TCU_INT8_ENABLE -DVX_CFG_NUM_WARPS=16"
```

`VX_CFG_NUM_WARPS=16` is the one that is not about a hardware unit, and it is
here because of `tests/cuda_samples/`. The toml default of 4 warps over a 4-lane
warp gives `maxThreadsPerBlock` **16**, and five of the CUDA samples hard-code a
32-thread block — as CUDA programs do, because no CUDA device has ever had a
maximum block that small. Sixteen warps gives 64, and the samples run.

It is worth knowing that widening the core moved two gates, because both moves
were the device being more truthful rather than anything getting worse: the prof
gate's cycles-ratio band was calibrated to a narrow core and is now a marginal
cost, and the phase 3 GEMM gate exposed that the tensor path's parallelism is
bounded by its output tile count. `docs/designs/grxcp_roadmap.md`'s phase 4
progress note has both.
