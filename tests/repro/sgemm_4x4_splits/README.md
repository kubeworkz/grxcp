# `sgemm_4x4` loses every SIMT split/join — reproducer for cuda_mapping.md 7.27

## What this shows

Adding one 64-bit load to `micro_tile_body`'s epilogue makes VOLT emit
**zero** `vx_split` / `vx_join` instructions in the `RM=RN=4` instantiation,
while every other kernel in the same translation unit keeps all of theirs.
The kernel then follows one lane's decision at every divergent branch.

It is a silent wrong answer only when a branch actually diverges. With uniform
control flow the missing reconvergence is invisible, which is why a shape sweep
is part of the reproduction and a single shape is not.

## Reproducing

Apply the fused bias to `src/libs/grxblas/kernels/sgemm.cpp` — one field, one
conditional load, one add per output:

```c
// sgemm_abi.h, appended to grxblas_sgemm_args
uint64_t bias;

// micro_tile_body, epilogue
const float* bias = reinterpret_cast<const float*>(arg->bias);
for (j...) {
  const float bj = bias ? bias[col[j]] : 0.0f;
  for (i...) C[at] = (...) + bj;
}
```

Build, then count the reconvergence instructions per kernel:

```
./ci/build_kernel.sh --grxgpu <grxgpu> src/libs/grxblas/kernels/sgemm.cpp -o k.vxbin
./tests/repro/sgemm_4x4_splits/count_splits.sh k.elf
```

`vx_split` is `.insn r CUSTOM0, 2, 0, rd, rs1, x0` — opcode `0x0b`, funct3 `2`;
`vx_join` is funct3 `3`. `count_splits.sh` decodes them out of `objdump`.

## Measured

| kernel | accumulators | baseline | with the bias load |
|---|---|---|---|
| `sgemm` | – | 2 / 2 | 2 / 2 |
| `sgemm_rb` | – | 6 / 6 | 6 / 6 |
| `sgemm_2d` | 4 | 8 / 8 | 8 / 8 |
| **`sgemm_4x4`** | **16** | **22 / 22** | **0 / 0** |
| `sgemm_4x2` | 8 | 12 / 12 | 12 / 12 |

Shape sweep on the affected build, `n=4`, `k=2`, one block of four threads.
`row_blocks = ceil(m/4)` is how many lanes pass the outer guard:

| m | lanes passing | result |
|---|---|---|
| 4 | 1 of 4 | writes nothing |
| 8 | 2 of 4 | writes nothing |
| 12 | 3 of 4 | writes nothing |
| **16** | **4 of 4 — uniform** | **correct** |
| **20** | **5 of 5 — uniform** | **correct** |

The same kernel, the same 16 accumulators, the same bias load. Only whether the
guard diverges.

## Why this matters beyond one kernel

The trigger is a property of the function, not of the fusion: something about
the `RM=RN=4` instantiation plus one more live 64-bit value crosses a threshold
and the divergence handling is dropped **silently**, with no diagnostic. Any
kernel that crosses the same threshold loses reconvergence the same way, and
will look correct on every shape whose control flow happens to be uniform.

That is the argument for counting `vx_split`/`vx_join` as a build-time check
rather than trusting a correctness gate: the gate can only catch it on a shape
that diverges, and the kernel that has lost divergence handling is exactly the
one whose author did not think divergence was involved.
