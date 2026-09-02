# `dev_exp_nonpos` is an exact specialisation — the check for it

`exact.cpp` builds and runs on the HOST. It implements `dev_exp` and
`dev_exp_nonpos` exactly as `src/libs/grxdnn/kernels/dnn_device.h` has them and
compares them bit for bit over every representable negative float down to -200
(sampled by bit pattern), the clamp edges, and NaN.

```
g++ -O2 -std=c++17 tests/repro/softmax_exp_exact/exact.cpp -o exact && ./exact
```

Expected: `8818703 values compared, 0 mismatches`.

It is on the host rather than the device on purpose. The claim is about float
arithmetic, not about the GPU, and checking it here needs no toolchain, no
simulator and no kernel build — so it stays runnable when none of those are
available, which is when a specialisation is most likely to drift from the
function it specialises.

**It was watched failing.** Moving one constant in `dev_exp_nonpos` by 1e-4
(`-0.5f` to `-0.4999f`) reports 63 mismatches. A comparison that cannot fail is
not evidence, and this one was run in both directions before the kernel change
was believed.

## What makes the specialisation legal

`dnn_softmax` calls it as `dev_exp_nonpos(xr[j] - row_max)` where `row_max` is
the maximum over the row, reduced across the warp before any of these calls.
The argument is therefore `<= 0` at every element, always. If that ever stops
being true, this file stops being the right check and the call site is the bug.
