# `grxcc` — the single-source compiler driver

`grxcc` is what turns GRXCP from a library binding into a programming
platform: it lets one `.grx.cpp` file contain both host code and
`__global__` kernels, with `<<<grid, block, smem, stream>>>` launches.

## What it does, in order

```
  input.grx.cpp
    │
    ├─ DEVICE PASS ────────────────────────────────────────────────────┐
    │   clang -x grx-device --target=riscv64 (VOLT SIMT backend)       │
    │     include/grx/device/*.h + the GRX-G100 kernel sysroot         │
    │   → device.o → ld (link64.ld) → device.elf → vxbin.py → .vxbin   │
    │     (multi-entry: one __vx_kentry_<name> per __global__,          │
    │      recorded in the VXSYMTAB footer)                            │
    │                                                                  │
    ├─ HOST PASS ──────────────────────────────────────────────────────┤
    │   rewrite  f<<<g,b,s,st>>>(a, b)                                  │
    │       →    __grxPushCallConfiguration(g, b, s, st); f_stub(a, b); │
    │   emit    per-kernel parameter descriptors                        │
    │   emit    __grxRegisterFatBinary / __grxRegisterFunction ctors    │
    │   clang --target={x86_64,riscv64}-linux-gnu                       │
    │                                                                  │
    └─ LINK ───────────────────────────────────────────────────────────┘
        host.o  +  .grxfatbin section (the .grxfat container)  +  -lgrxrt
```

## Why an orchestrator first

The end state is a proper Clang `ToolChain` with a real `Action` graph. The
first implementation is an orchestrator that shells out to clang twice,
because the risk here is integration, not compiler research: VOLT already
lowers SIMT kernels to the GRX-G100 ISA, `vxbin.py` already builds multi-entry
binaries, and the CUDA host-side lowering pattern is public and well
understood. Promote to a `ToolChain` once the flag surface stops moving
(roadmap phase 6).

## The rv32 argument-width fix

The SPIR-V path has an unfixed hazard: host `size_t` is 8 bytes and device
`size_t` is 4 on a 32-bit device, and POD kernel arguments drift. `grxcc` does
not inherit it — the driver knows each kernel's parameter layout, so it
narrows pointer and `size_t` parameters at pack time and emits a diagnostic
when host and device widths disagree in a way that cannot be narrowed
losslessly.

## Flags

| Flag | Meaning |
|---|---|
| `-grx-arch=g100` | target device profile (selects required ISA extension bits) |
| `-grx-xlen=64` | device pointer width (64 default; 32 builds but is untested in v1) |
| `--grx-device-debug` | device-side debug info for `grx-gdb` |
| `-grx-max-registers=N` | device register cap, feeds occupancy metadata |
| `--grx-keep` | keep intermediates (`.spv`, `.vxbin`, device ELF) |
| `--grx-device-only` / `--grx-host-only` | run a single pass |
