# Multi-core argument delivery — reproducer for cuda_mapping.md 7.37

`parity_probe.cpp` launches one kernel N times with an identical argument blob,
an identical output buffer and an identical argument size, and reports per
launch whether the kernel observed its arguments.

`sgemm_shape` is used because its whole body is a version check and seven
stores, which makes the answer binary: the output buffer is either written or
untouched. Nothing about the defect is specific to that kernel — it is the
cheapest oracle we have for "did the arguments arrive".

Build against the runtime and run under each backend:

```
g++ -std=c++17 -Iinclude tests/repro/multicore_args/parity_probe.cpp \
    -o parity_probe -L<build> -lgrxrt -Wl,-rpath,<build>
VORTEX_DRIVER=simx   ./parity_probe 8
VORTEX_DRIVER=rtlsim ./parity_probe 8
```

Expected on a healthy backend: `8/8 launches wrote`.

Measured on `rtlsim` built with `VX_CFG_NUM_CORES=2` or `4`: `4/8`, alternating,
first launch failing, deterministic across processes. `simx` at the same core
count is 8/8. See cuda_mapping.md 7.37 for the full elimination and for what is
still unmeasured.

This is a reproducer, not a gate. It is not wired into `ci/` because the
behaviour it detects lives below grxcp and a red gate here would be a claim
about someone else's tree.
