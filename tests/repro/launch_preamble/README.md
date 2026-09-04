# The launch preamble grows with occupancy and stops at residency

*The first two versions of this file measured something else.*

The block bench reports a per-launch **preamble** — the interval between the
launch and the first warp reaching its cycle probe. At S=16 it measured 9418
cycles per launch, 39.7% of a transformer block, and looked near-constant across
twelve different kernels. That constancy is a coincidence of the shapes those
kernels launch. It is not a fixed cost — but neither is it unbounded, which is
what this file said for two revisions.

## The measurement, run correctly

A kernel that does nothing but timestamp itself, launched at varying grid sizes.
Every block records its own entry time, so "does work overlap dispatch" is
answerable rather than assumed.

**One launch per process.** That is the whole correction; see below.

4 cores, 16 threads per block:

| blocks | rtlsim first | rtlsim last | simx first | simx last |
|---|---|---|---|---|
| 1 | 2910 | 2910 | 3030 | 3030 |
| 2 | 4458 | 4476 | 4470 | 4548 |
| 4 | 7195 | 7961 | 6865 | 7674 |
| 8 | 12899 | 14809 | 13487 | 15100 |
| 16 | 16467 | 28115 | 26414 | 29412 |
| 32 | **17687** | 33584 | **26414** | 33163 |
| 64 | **17687** | 34127 | **26414** | 34754 |

**Both backends plateau at residency** — 4 cores × 16 warps = 64 slots, 4 warps
per block, sixteen resident blocks. Past that, blocks queue behind retiring ones
and the earliest entry stops moving. rtlsim reaches its first block *sooner*
than simx at large grids.

The earliest block's entry does grow between 1 block and residency — 5.7× on
rtlsim, 8.7× on simx. That part is real on both. What gates it is asked of
grxgpu; `process_thread_groups` explains the spread but not the floor.

## What it is not

| candidate | measured |
|---|---|
| the instrument — `vx_rdcycle_sync`'s fence at entry | **27 cycles.** Not it. |
| the first memory access, cold dcache | **18 cycles.** Not it. |
| fixed device bring-up | **1828 cycles at one block** — real, but a floor |
| hardware CTA dispatch at ~1500 cycles/CTA | **retracted.** Block distribution is a software loop (`sw/kernel/src/vx_spawn.c`, `process_thread_groups`) |
| a serial per-CTA cost that scales with the grid | **refuted.** It stops at residency on both backends |

The kernel takes an unserialized `csrr` as its very first instruction, so the
first number has nothing of ours in front of it.

## The correction: MCYCLE does not restart per launch on rtlsim

The first two versions of this file printed this table:

| blocks | first entry | last entry |
|---|---|---|
| 1 | 2910 | 2910 |
| 16 | 45221 | 57615 |
| 64 | 100774 | 124372 |

and concluded that rtlsim serialises the whole grid while simx plateaus — "a
fidelity-contract row", worth resolving before any grid-sizing decision. A
question was sent to grxgpu on it.

**Every value after the first is the sum of the launches before it.** The sweep
ran all seven grid sizes in one process in ascending order, and:

* `hw/rtl/core/VX_scheduler.sv` — `cycles` is zeroed only under `if (reset)` and
  incremented under `if (busy)`. Per core, busy-gated, free-running.
* `sim/rtlsim/processor.cpp` — `reset()` is called from the `Impl` constructor.
  `run()` pulses `start` and drains, and never resets.
* `sim/simx/processor.cpp` — `ProcessorImpl::run()` opens with `this->reset()`.
  **simx is the outlier**; silicon free-runs `mcycle` too.

Three ways to watch it fail, all in this directory:

1. **`repeat_probe N T R`** — the same grid, R times. rtlsim climbs by a
   constant **8917** per launch at 4 blocks; simx reads 6865, 6860, 6860, 6860.
2. **`grid_sweep desc`** — the same sweep descending. The one-block case reports
   **122873** instead of 2910. Whichever grid runs first reports the truth.
3. The first three ascending points matched the true values to within 4 cycles,
   which is why this survived review. MCYCLE is **per core**: at 1 block only
   core 0 runs, at 2 blocks core 1 is still fresh, at 4 blocks cores 2 and 3
   are, so the minimum came from a core with no history. From 8 blocks on, the
   first step is +8920 — the 4-block frame again, to within 3 cycles of the
   independently measured 8917.

**Spans are unaffected on both backends.** A span is a subtraction inside one
launch and the offset cancels. Only absolute readings are at risk, which is
exactly why this hid for as long as it did.

`tests/bench/block_cycles.cpp` now calibrates before trusting one: four
identical launches, and if the readings climb it reports `-1` for the preamble
instead of a number. Run it with `--calibrate-only` to see the verdict without
paying for the block.

## What it means for the roadmap

**The multi-SM scaling wall is gone.** The preamble scales with occupancy, which
the machine bounds, not with the grid, which it does not. A 128-SM part does not
inherit a serial per-CTA cost from this.

**The 51.4% stands, as a figure for these shapes.** `developer_interface.md`
section 3 prices per-launch fixed cost at 51.4% of a block from a 9418-cycle
preamble measured on simx. Still true, still shape-dependent, and no longer
expected to grow without bound at production sequence lengths.

**Fusion still pays** — it removes one whole instance of the per-launch cost per
fused pair. The caveat that it "cannot address the term that grows with the
machine" is moot; there is no such term here.

## Reproducing

```
./ci/build_kernel.sh --grxgpu <grxgpu> tests/repro/launch_preamble/kernel.cpp \
    -o build-real/preamble.vxbin

# the artifact, and the two ways to see it
VORTEX_DRIVER=rtlsim ./build-real/grid_sweep          # ascending, one process
VORTEX_DRIVER=rtlsim ./build-real/grid_sweep desc     # inverts the table
VORTEX_DRIVER=rtlsim ./build-real/repeat_probe 4 16 6 # same grid, six times

# the honest sweep: one launch per process
for nb in 1 2 4 8 16 32 64; do ./build-real/repeat_probe $nb 16 1; done
```

The kernel writes `{t_raw, t_sync, t_arg, core+1}` per block. `t_raw` is a plain
`csrr` on the first instruction; `t_sync` the serialized read the real probe
uses; `t_arg` after one load from the argument blob.
