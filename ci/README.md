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
  --configs "-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE"
./ci/run_real.sh --grxgpu ../grxgpu --tooldir $HOME/tools
```

`ci/build_kernel.sh` compiles a kernel into a `.vxbin` — what grxcc will do
internally in Phase 4. `tests/kernels/vecadd/` is the Phase 1 exit gate: a
kernel written against GRXCP's own device header, launched through
`grxLaunchFunction`, with the arithmetic checked on the host at sizes that
exercise the partial-warp path.

`tests/kernels/wmma/` is the tensor gate: one WMMA tile through
`grx::wmma`, compared **exactly** against a CPU reference. The host asks the
device for its tile shape rather than assuming one, and cross-checks the warp
width the kernel was compiled for against the width the runtime reports — a
disagreement there means the module and the runtime came from different
configurations, which is the failure "configuration provenance" below exists to
prevent. It skips when the device reports no tensor unit.

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
  --configs "-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE"
```

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
