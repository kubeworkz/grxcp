# OP_COUNT at the shipped 8×8 array — the fitted constant was `NUM_ROWS + NUM_COLS + 2`

Our twenty `OP_COUNT` measurements were taken at `NUM_ROWS = NUM_COLS = 4`.
GRX930 widened the array to 8×8 and confirmed 8×8 as the shipped default
(`9f3c2b3`), so we invalidated our own table (`cuda_mapping.md` 7.34). This
re-derives it — and tests an *explanation* of it rather than refitting.

## The prediction, made before the run

At 4×4 the measured relation was

```
OP_COUNT = 10 · M · ceil(N/NUM_COLS) · ceil(K/NUM_ROWS) · NUM_ROWS · NUM_COLS
```

and **the 10 was unexplained** — a constant that happened to fit twenty points.
GRX930's timing note then gave the `S_RUN` pass length as `NUM_ROWS + NUM_COLS
+ 2`, which is exactly 10 at 4×4. If that is what our constant was, it must be
**18** at 8×8. Three hypotheses, written into `opcount.cc` before it was run:

| | form | at 8×8 |
|---|---|---|
| **H1** ours, parameterized | `(R+C+2) · M · ceil(N/C) · ceil(K/R) · R · C` | constant 18 |
| **H2** the constant is literally 10 | `10 · M · ceil(N/C) · ceil(K/R) · R · C` | constant 10 |
| **H3** theirs, published | `ceil(M/R) · ceil(N'/C) · ceil(K/R) · R · C` | M drops out entirely |

`MAX_M = 8`, so `ceil(M/8) = 1` for every legal M — which is why H3 predicts
the same number for `1×4×4` and `8×4×4`. The three are separated by factors of
1.8 and M, so one run decides.

## Measured, 8×8, fifteen shapes

| shape | OP_COUNT | H1 | H2 | H3 | CYCLE | their cycle model |
|---|---|---|---|---|---|---|
| 1×1×1 | 1152 | **1152** | 640 | 64 | 20 | **20** |
| 1×4×4 | 1152 | **1152** | 640 | 64 | 38 | **38** |
| 2×4×4 | 2304 | **2304** | 1280 | 64 | 76 | **76** |
| 3×4×4 | 3456 | **3456** | 1920 | 64 | 114 | **114** |
| 4×4×4 | 4608 | **4608** | 2560 | 64 | 152 | **152** |
| 5×4×4 | 5760 | **5760** | 3200 | 64 | 190 | **190** |
| 6×4×4 | 6912 | **6912** | 3840 | 64 | 228 | **228** |
| 7×8×4 | 8064 | **8064** | 4480 | 64 | 406 | **406** |
| 8×4×4 | 9216 | **9216** | 5120 | 64 | 304 | **304** |
| 4×8×4 | 4608 | **4608** | 2560 | 64 | 232 | **232** |
| 4×12×4 | 9216 | **9216** | 5120 | 128 | 384 | **384** |
| 4×4×8 | 4608 | **4608** | 2560 | 64 | 216 | **216** |
| 4×4×16 | 9216 | **9216** | 5120 | 128 | 416 | **416** |
| 8×12×5 | 18432 | **18432** | 10240 | 128 | 864 | **864** |
| 8×12×16 | 36864 | **36864** | 20480 | 256 | 2208 | **2208** |

**H1 15/15. H2 0/15. H3 0/15.** All fifteen computed correct answers against a
host reference, so the engine is doing the arithmetic right at 8×8 too.

## The same harness rebuilt at 4×4

H1 and H2 are the same formula at 4×4 — that is what makes 8×8 the run that
separates them — so the point of this build is different: it checks that this
harness reproduces the *old* measurements, on the current RTL.

| | OP_COUNT | H1 (= H2 here) | H3 theirs | CYCLE | their cycle model |
|---|---|---|---|---|---|
| 1×1×1 | 160 | **160** | 16 | 12 | **12** |
| 4×4×4 | 640 | **640** | 16 | 120 | **120** |
| 4×4×8 | 1280 | **1280** | 32 | 224 | **224** |
| 8×12×16 | 15360 | **15360** | 384 | 2592 | **2592** |

**H1 15/15, H3 0/15, cycle model 15/15** at this width as well. And the spot
values are the ones already in `cuda_mapping.md` 7.34, measured months ago by a
different harness: `1×1×1 → 160`, `4×4×4 → 640`, `8×8×8 → 5120`, and for
`M=4 N=4 K=8` the pair `OP_COUNT 1280, CYCLE_LO 224`. The `2592` in 7.34's
`DMA_CT ≥ CYCLE_COUNT` note is this table's `8×12×16` cycle count. Nothing had
to be adjusted to make them agree.

**Totals across both widths: H1 30/30, their published OP_COUNT 0/30, their
cycle model 30/30.**

## What this settles

**The constant is not a constant.** `OP_COUNT` is now parameterized rather than
fitted:

```
OP_COUNT = (NUM_ROWS + NUM_COLS + 2) · M · ceil(N/NUM_COLS) · ceil(K/NUM_ROWS)
           · NUM_ROWS · NUM_COLS
```

and it holds at two array widths, which one width could never establish.

**Their cycle model is exactly right — 15/15**, across a hundred-fold range
from 20 cycles to 2208:

```
CYCLE = M × [ (K+1)·N + ceil(K/NUM_ROWS)·(NUM_ROWS+NUM_COLS+2)·ceil(N/NUM_COLS) ]
```

That matters for how to read the disagreement. It is not that their model of
the engine is wrong; it is right to the cycle. It is that **the two formulas in
their document are inconsistent with each other**, and the cycle one is the
correct one.

**And the reconciliation falls out of their own model.** Divide H1 by `R·C`:

```
OP_COUNT / (R·C) = (R+C+2) · M · ceil(N/C) · ceil(K/R)
                 = the S_RUN term of their cycle formula
```

So **`OP_COUNT` is the array's area times the number of cycles it spends in
`S_RUN`** — every PE counted on every `S_RUN` cycle, pipeline fill and drain
included. It is not a count of useful MACs and should not be read as one: at
8×8 the fill/drain overhead is `(R+C+2)/R = 2.25×` on a `K = R` pass. Their
published `ceil(M/R)·…` form counts one firing per pass instead of `R·C` per
cycle, and tiles M where the array streams it.

## Calibrate before trusting the counter

`--calibrate` runs one shape three times on one instance with **no reset
between**. `OP_COUNT` is per-command here — 4608, 4608, 4608 — so the sweep
measures shapes and not the run. The check is in the harness because the same
class of error, a counter that does not restart where the harness assumed it
did, cost this project a roadmap phase (`cuda_mapping.md` 7.25).

It also caught a defect in this harness on its first run. The poll was

```c
for (…) { if (!(status & BUSY)) break; }      // wrong
```

which exits on its first read, because right after `START` the engine has not
asserted `BUSY` yet. The first command after reset reported `OP_COUNT = 0`,
`CYCLE = 0` and wrong answers; every command after a *completed* one was fine,
because `BUSY` was still high on entry. One shape in fifteen would have been
silently zero. **This is the same defect, in the same shape, as the rtlsim
`run()` bug we reported to grxgpu** — sample a level signal before its
transient and you measure the wrong side of it. Fixed by waiting for `BUSY` to
rise first, bounded so a fast command cannot hang the harness.

## Reproducing

```
verilator --cc --exe --build -Wno-fatal --Mdir obj_dir_oc \
  --top-module c930_npu_top \
  -GNUM_ROWS=8 -GNUM_COLS=8 -GMAX_M=8 -GMAX_K=16 -GMAX_N=12 \
  -CFLAGS "-DNUM_ROWS_P=8 -DNUM_COLS_P=8" \
  <grx930>/c930/rtl/*.sv opcount.cc

./obj_dir_oc/Vc930_npu_top --calibrate   # is the counter per-command?
./obj_dir_oc/Vc930_npu_top               # the sweep
```

Rebuild with `-GNUM_ROWS=4 -GNUM_COLS=4 -DNUM_ROWS_P=4 -DNUM_COLS_P=4` for the
old width. Both widths are the point: a relation that holds at one array size
is a fit, and at two it is a model.
