# `Processor::run()` ends the frame on the wrong edge — the mechanism behind gap 7.37

`rtlsim` at `NUM_CORES > 1` appeared to deliver kernel arguments on every *other*
launch. It does not. Arguments arrive correctly on every launch. **The frame is
never executed on half of them**, and the host then reads a buffer that the
previous frame filled.

The defect is thirteen lines from being fixed and it is in grxgpu's
`sim/rtlsim/processor.cpp`, a file that was byte-identical between our tree and
theirs throughout the entire investigation — which is why nothing that looked at
the argument path was ever going to find it.

## The signal

`busy`, sampled every tick over a fixed window after the start pulse. Identical
on every launch, including the first, on a build with no prior frame:

```
t=0 busy=1 | t=1 busy=0 | t=2 busy=1 | t=2318 busy=0
```

`busy` is **already high on entry** — out of reset, and again at the end of every
frame — and dips **low for exactly one cycle** after the start pulse, before the
new frame asserts it.

## Why that breaks `run()`

```cpp
device_->start = 1; this->tick(); device_->start = 0;

for (uint32_t i = 0; !device_->busy && i < NO_WORK_TIMEOUT; ++i)   // (a)
  this->tick();

while (device_->busy) { this->tick(); ... }                        // (b)
```

* (a) "wait for device to go busy" **never runs** — busy is already 1.
  Measured `wait_i = 0` on every launch, every configuration.
* (b) takes its first tick onto `t=1`, finds busy low, and exits.
  Measured `busy_ticks = 1`.

`run()` therefore executes **one cycle of a ~2300-cycle frame** and returns. The
work is not lost; it is still pending, and the next `run()`'s ticks carry it out.
That is the whole of the alternation — launch N's work completes during launch
N+1, so odd launches read a correct-looking buffer and even ones read nothing.
With an identical argument blob every launch this is invisible. With differing
arguments it is a silent wrong answer.

## The measurements

Same binary, same tree (grxgpu `5253957`, which contains `b09ca185e`), one
environment variable apart — the fix compiled in and switchable, so the control
is not a different build:

| | `test_grxblas` | `sgemm_shape` × 8 |
|---|---|---|
| without the settle loop | args never arrive, then `VX_lsu_slice.sv:233` misaligned abort at `addr=0x100000001` | **4/8** |
| with it | **PASSED (0 failures)** | **8/8** |

Per-launch, with the fix: `settle=1`, `wait_i=1`, `busy_ticks=2300` — the settle
loop rides out the dip in one tick, loop (a) then does the job it was written to
do in one more, and the full frame runs.

An independent confirmation that the frame is only starved of ticks: a
diagnostic that ticks 4000 times after the start pulse *before* the wait loops
get a chance to exit early makes every launch pass, 3/3, including launch 0.

## What this rules out

**grxgpu's `b09ca185e` does not fix it**, and cannot. Instrumenting the hook it
guards:

```
[hook] L0 dcr_write addr=0x010 val=0x80000000 future_valid=0
[hook] L0 run() returned after 0 ms        launch 0: *** silent ***
[hook] L1 dcr_write addr=0x010 val=0x80000000 future_valid=1
[hook] L1 run() returned after 736 ms      launch 1: wrote
[hook] L2 dcr_write addr=0x010 val=0x80000000 future_valid=1
[hook] L2 run() returned after 0 ms        launch 2: *** silent ***
```

Launch 0 fails with `future_valid=0` — nothing to drain, so `future_.wait()` is a
no-op and the launch still fails. Launch 2 fails with the wait executed. And the
DCR payload is byte-identical across failing and passing launches, all twenty
registers `0x010`–`0x023`, so nothing is corrupted at the DCR interface in the
first place.

Extending the same `future_.wait()` to `dram_read` and `dram_write`, on the
theory that the argument blob travels by `CMD_MEM_WRITE`, is also 4/8,
identically.

## Reproducing

```
# build rtlsim from grxgpu at >= 2 cores
CONFIGS="-DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=4 -DVX_CFG_NUM_WARPS=16"
(cd build/sim/rtlsim && make -j4) && (cd build/sw/runtime/rtlsim && make -j4)
(cd build && make install)

# the probe: one kernel, one argument blob, N launches, nothing varying but index
g++ -std=c++17 -Iinclude parity_probe.cpp -Lbuild-real -lgrx -o parity_probe
VORTEX_DRIVER=rtlsim LD_LIBRARY_PATH=$VORTEX_PATH/runtime/lib ./parity_probe 8
```

Expected without the patch at 4 cores: `4/8 launches wrote`, strict alternation,
launch 0 among the failures. With `run_settle.patch` applied to grxgpu's
`sim/rtlsim/processor.cpp`: `8/8`.

## A note on the confound, because it nearly cost the result

The first attempt applied grxgpu's three-line fix to a local copy of their tree
that was two weeks old, measured 4/8, and was about to be reported as "your fix
does not work". Comparing file hashes against their HEAD found four RTL files
different, including `VX_dcr_data.sv` — the DCR register file, which is what the
fix is about. Every number above was re-taken after re-staging their tree at
`5253957` and building with their `VX_config.toml` unmodified except for the
cluster, core and warp counts.

A negative result about someone else's fix is worth exactly as much as the
provenance of the tree it was measured on.
