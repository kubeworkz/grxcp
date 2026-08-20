# `ci/` — continuous integration

Two tiers, and the distinction matters.

## Tier 1 — mock build (`ci/build_mock.sh`)

Runs on every commit. Compiles the runtime and tools against the mock driver in
`tests/mock/`, then runs the unit tests and `grx-smi`. No Vortex sysroot, no
simulator, no FPGA; a few seconds.

What it proves: the code compiles and links, the device record is internally
consistent, the error surface behaves, and the honesty flags are still set.

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
not a measurement. `tests/bench/sgemm_cycles.cpp` then reports what sgemm v0
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
