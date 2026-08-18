# GRXCP — GRX Compute Platform

GRXCP is the CUDA-class compute software platform for the **GRX-G100** GPU
and the **GRX930** host SoC. It is what turns the GRX-G100 from a device with
a driver into a device people can program.

```
  your_kernel.grx.cpp  ──grxcc──►  host ELF + .grxfat
                                        │
                                   libgrxrt  (grxMalloc / grxLaunchKernel / streams / events)
                                        │
                                   vortex2.h  (GRX-G100 driver API — consumed unchanged)
                                        │
                                   Command Processor ──► KMU ──► SMs / TCU / DXA
```

## Design in one paragraph

The GRX-G100 already ships a good driver API: `vortex2.h` is async,
handle-based, and timeline-event shaped — the equivalent of CUDA's *driver*
API. GRXCP does not replace it. GRXCP builds the four layers above it: a
CUDA-Runtime-shaped host API (`libgrxrt`), a single-source compiler driver
(`grxcc`) that gives you `__global__` and `<<<grid, block>>>`, a set of math
libraries (grxBLAS, grxDNN, ...), and the tools (`grx-smi`, `grx-prof`,
`grx-sanitize`). Our API is GRX-named; `grx_cuda_compat.h` maps `cudaX` to
`grxX` so porting existing CUDA is a one-line include change.

## Documents

- [Architecture specification](docs/designs/grxcp_architecture.md) — layers,
  API surface, kernel ABI, memory model, device model
- [CUDA mapping and gap register](docs/designs/cuda_mapping.md) — every CUDA
  concept, the GRX mechanism behind it, and an honest status
- [Implementation roadmap](docs/designs/grxcp_roadmap.md) — phases, exit
  gates, dependencies, risks

## Relationship to the other repos

| Repo | Role | Coupling |
|---|---|---|
| `grxgpu` | GRX-G100 GPU: RTL, simulators, driver, kernel headers | GRXCP links against the **installed sysroot** via `pkg-config vortex-runtime vortex-kernel`. Never against `$VORTEX_HOME` or a build tree. |
| `grx930` | GRX930 host SoC + systolic NPU | Future host target (native riscv64) and future second device (Phase 7) |
| `grxcp` | this repo | Depends on both; neither depends on it |

## Status

Design complete, implementation not started. See the roadmap for phase
ordering and exit gates. Phase 0's gate is `grx-smi` printing real device
properties on `simx` and `rtlsim` from a clean checkout.
