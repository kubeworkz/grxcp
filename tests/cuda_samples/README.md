# `tests/cuda_samples/` — the phase 4 exit gate's "ten CUDA samples"

The roadmap's phase 4 exit gate asks that *at least ten CUDA samples compile
**unmodified** except for the `grx_cuda_compat.h` include*. This directory is
that claim, and the rule it was written under matters more than the count:

> Each program was written as a CUDA programmer writes it, and then compiled.
> Where it failed, **the platform changed and the sample did not.** Any sample
> that still needs an edit says so at the top of the file, in the sample, and
> is not counted.

Twelve files: eleven compile and run, and the twelfth (`11_histogram_atomics.cu`)
must fail to compile on a build with no atomic extension, which is what the gate
checks for it.

The alternative — writing samples against what `grxcc` already supports — would
produce ten files that pass and prove nothing. The value here is entirely in
what broke the first time.

## The rule, precisely

Every file's only concession to GRXCP is:

```c
#include <grx/grx_cuda_compat.h>      // instead of <cuda_runtime.h>
```

Everything else is CUDA: `__global__`, `<<<>>>`, `cudaMalloc`, `__shared__`,
`__syncthreads`, `__shfl_down_sync`, `cooperative_groups`, `dim3`, `blockIdx`.
No `grx*` name appears in any sample. Nothing includes a `grx/device/` header —
a CUDA file does not include one either; the compiler supplies those names, and
after this exercise `grxcc` does too.

Each sample checks computed **values** against a host reference and prints
`PASSED` or `FAILED`. A sample that only checked return codes would pass on a
platform that launched nothing.

## What the exercise found

Recorded in the order it was hit, because the order is the point.

**First pass: eleven samples, eleven failures.** Not one compiled. Five failed
on the same thing.

| sample | what broke | what changed |
|---|---|---|
| 01, 07, 08 | `'fabsf' was not declared` | `cuda_runtime.h` pulls in `<math.h>`; `grx_cuda_compat.h` did not. It does now. |
| 02, 03, 05, 06, 09 | `'As' is unavailable: GRXCP has no static __shared__` | **grxcc implements it.** Static `__shared__` declarations are collected into a per-kernel struct placed over the CTA's local-memory slot, and its size goes into the descriptor's `static_smem` — a field the runtime already added to `lmem_size` and that nothing had ever set. `extern __shared__` lands after it. |
| 04 | `unknown type name '__device__'`, `undeclared identifier 'warpSize'`, `undeclared identifier '__shfl_down_sync'` | Three separate causes. The device header was inserted *after* the user's `__device__` helper — it now goes after the last preprocessor directive instead. `warpSize` did not exist. `grx_warp.h` was not included unless the user asked, which a CUDA file never does. |
| 10 | `'cooperative_groups.h' file not found` | CUDA's spelling now exists as a forwarding header, alongside `cooperative_groups/reduce.h`. |
| 11 | `undeclared identifier 'atomicAdd'` | `grx_atomic.h`, which refuses **by name** on a build without the A extension instead of leaving the identifier undeclared — and on a build with it, works. |

Nothing in that column is a sample edit. The one thing that did change in a
sample is recorded in `10_cooperative_groups.cu`'s own header comment, and it is
a constant, not a construct.

**Second pass: nine of eleven ran.** Two remained.

| sample | what broke | what changed |
|---|---|---|
| 04 | `'__device__' does not name a type` — in the HOST pass | `nvcc` drops `__device__`-only functions from its host pass; grxcc now does too. Without it a device helper reaches a host compiler that has never heard of `warpSize`. |
| 10 | `vx_spawn2.h: No such file or directory` | The forwarding header pulled the device stack into the host pass as well. It is now fenced to the device pass, with an empty `namespace cooperative_groups {}` for the host so a file-scope `namespace cg = cooperative_groups;` still compiles. Same shape as CUDA's `__CUDA_ARCH__` guard. |

**Third pass: ten of ten, plus the refusal.** `11_histogram_atomics.cu` fails to
compile, which is the correct outcome on this build and is what the gate checks.

**Then a twelfth was added**, for a construct the first ten did not reach for:
`12_constant_memory.cu` sets a convolution's filter taps with
`cudaMemcpyToSymbol`. `09_stencil_1d.cu` would normally do the same for its
coefficients, and the reason it does not is that `__constant__` had no host-side
reach until this sample was written. It is also where the platform's one
asymmetry is documented and checked: `__constant__` reads back exactly,
`__device__` is refused, and both are gated. See `cuda_mapping.md` section 7.23.

## The device the samples run on

Five samples hard-code a 32-thread block, and the sysroot this project used to
build had `VX_CFG_NUM_WARPS=4` with a 4-lane warp — `maxThreadsPerBlock` 16. The
launches were refused, correctly, with `cudaErrorLaunchOutOfResources`.

No CUDA device has ever had a maximum block of 16 threads, so a program written
against CUDA has no reason to check. The 4-warp preset was a simulation
convenience, not a claim about GRX-G100, and it was the thing that had to move:
the sysroot is now built with `VX_CFG_NUM_WARPS=16`, giving
`maxThreadsPerBlock` 64.

That change moved two other gates, and both were the wider device telling the
truth rather than a regression — see the phase 4 progress note in
`docs/designs/grxcp_roadmap.md`.

## Running them

`ci/run_real.sh` builds and runs every sample under the **CUDA SAMPLES GATE**.
By hand:

```
grxcc --grxgpu <path> --tooldir <path> --build-kernel ci/build_kernel.sh \
      -I include tests/cuda_samples/01_vector_add.cu \
      <runtime objects> $(pkg-config --libs vortex-runtime) -o /tmp/01
/tmp/01
```

The `.cu` extension is deliberate: these are CUDA files, and naming them
`.grx.cpp` would quietly concede the point the gate is trying to make.
