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
./ci/build_mock.sh --vortex-include $VORTEX_PATH/include
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
image format, which the real loader has no reason to accept — testing launch
for real needs a `.vxbin`, and that needs the device toolchain. That is the
remaining piece of the Phase 1 gate.
