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

## Tier 2 — real backends (`ci/testcases/grxcp.yaml`)

The declarative catalog, in the same shape as the GRX-G100 project's. Every
conformance case runs on **both** `simx` and `rtlsim`; a result that differs
between the two blocks the merge, because it is either a GRXCP bug or an
upstream model-parity bug and both matter.

Tier 2 needs the installed GRX-G100 sysroot on `PKG_CONFIG_PATH` and the
backend libraries (`libvortex-simx.so`, `libvortex-rtlsim.so`) reachable by the
driver stub. `$VORTEX_DRIVER` selects the backend, defaulting to `simx`.

**The Phase 0 exit gate is a tier-2 result**, not a tier-1 one: `grx-smi`
printing correct device properties on `simx` and `rtlsim` from a clean
checkout. Tier 1 is the fast feedback loop underneath it.
