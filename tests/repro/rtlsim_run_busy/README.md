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

`run()` therefore executes **one cycle of a ~2300-cycle frame** and returns.

*This paragraph used to continue: "the work is not lost; it is still pending,
and the next `run()`'s ticks carry it out … with differing arguments it is a
silent wrong answer." **That was wrong**, and `parity_probe2.cpp` — buffer per
launch, re-read at the end — is what caught it. On the unfixed build:*

```
4/8 written by the time the host read them
4/8 written by the end of the run
4 launches did not run at all.
```

**Zero buffers filled late.** The work of a failing launch is not deferred, it
is never done. What produced the "wrote" on launches 1, 3, 5, 7 in the
one-buffer probe was the *shared buffer still holding the last successful
launch's output* — a stale read, not a carried-over frame. The distinction
matters: there is no silent wrong answer here, there are launches that do not
execute, and half a shared buffer's readings are stale.

The carry-over story was inferred from an instrument that could not see the
difference, and it was inferred twice — once here and once in the report sent to
grxgpu, who had it in hand before it was corrected.

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

## Postscript: the second probe, and why it exists

`parity_probe.cpp` reuses one output buffer. That is enough to detect a launch
that does not run, and not enough to detect work that lands in a later call —
launch N-1's output goes into the buffer launch N was going to write, and the
host reads a correct-looking value either way.

grxgpu's first fix (`506e21321`) drained residual busy *before* the start pulse
rather than after. Against the one-buffer probe it scored **7/8** and read as
very nearly right; `test_grxblas` still aborted. `parity_probe2.cpp` gives every
launch its own buffer, checks it immediately after that launch's sync, and
re-reads every buffer at the end.

Run against that same build it reported **zero buffers filled late** — which
refuted the explanation we had already sent grxgpu, that results were one launch
behind. They are not. The pre-pulse drain breaks exactly launch 0, and launch 0
is the one `grxblas` uses to read tile geometry.

The probe was built to remove luck from a finding and its first act was to
correct the finding. That is the argument for building it.

Final state: `3d8785f11` upstream, byte-identical to what is measured here,
8/8 on both probes and `test_grxblas` PASSED at 4 cores.
