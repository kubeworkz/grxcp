# grx-prof v1 — kernel timeline, device counters, occupancy

**Status:** implemented; gated by the PROF GATE in `ci/run_real.sh` and a
counter-honesty check in `ci/build_mock.sh`.
**Companion:** [`grxcp_roadmap.md`](grxcp_roadmap.md) phase 2,
[`grx_sanitize.md`](grx_sanitize.md) (same host/tool split).

---

## 1. The two clocks

Everything about this tool follows from one fact: GRX-G100 has two clocks that
measure different things, and a profiler that mixes them produces a beautiful
timeline that means nothing.

**The host clock.** `vx_event_get_profiling` stamps each command with host
nanoseconds around execution. It is the only clock that can place two
operations relative to each other, and the only one that shows the *gaps*
between them — the time a program spends doing nothing on the device. It is
also, on a simulator backend, a measurement of the simulator. A kernel that
"takes 28 ms" on `simx` tells you how long your workstation spent simulating
it. `grxDeviceProp_t::eventTimingIsDeviceSide` reports 0 for exactly this
reason.

**Device cycles.** The MPM performance counters, read through
`vx_device_mpm_query`. `MCYCLE` advances only while the device is running, so
its delta across a launch is device time — on the simulator and on silicon
alike. This is the number to compare kernels by.

So the timeline is built on the host clock, because nothing else can order
operations; every kernel slice carries its device cycle count as an argument;
and the report ranks by cycles and says in plain words which of its numbers are
which. The trace's own process track is named
`… [x-axis: host clock, which on this backend measures the simulator]`, because
someone opening the trace in six months reads the axis before they read this
document.

Nothing converts one clock into the other.

## 2. What it measures

Sampled around every launch, from the MPM counters:

| Group | Counters |
|---|---|
| Time and work | `cycles` (MCYCLE), `instrs` (MINSTRET) |
| Scheduling | `sched_idle`, `active_warps`, `stalled_warps`, `issued_warps`, `issued_threads` |
| Stalls | `stall_fetch`, `stall_ibuf`, `stall_scrb`, `stall_opds`, `stall_alu`, `stall_fpu`, `stall_lsu`, `stall_sfu`, `stall_tcu` |
| Mix | `instr_alu`, `instr_fpu`, `instr_lsu`, `instr_sfu`, `instr_tcu`, `branches`, `divergence` |
| Memory | `loads`, `stores`, `ifetches` |

The report derives four rates from them:

- **IPC** — warp-instructions retired per device cycle.
- **lanes** — `issued_threads / (issued_warps × warpSize)`: the fraction of a
  warp's lanes active when it issued. 100% means no divergence, and it is
  usually the first number worth looking at on a ported CUDA kernel.
- **occ** — `active_warps / (cycles × warps_per_core × cores)`: warp slots
  occupied, averaged over the kernel's cycles. This is *achieved* occupancy,
  measured — distinct from the *admitted* occupancy printed separately, which
  is what the CTA dispatcher's slot arithmetic allows for the launch shape.
- **issue** — warp-instructions issued per cycle.

Aggregation across cores matters and is not uniform: event counts add, but
`MCYCLE` does not. Four cores running a thousand cycles each took a thousand
cycles, not four thousand, so cycles aggregate as a maximum. Getting that
backwards would quarter every rate in the report.

## 3. What it does not measure

- **Concurrency.** Profiling serializes: each operation is bracketed by a full
  device sync so its counter delta belongs to it and nothing else. No overlap
  appears in the timeline because none is allowed to happen. Today that costs
  nothing real — the command processor runs one queue and streams do not
  overlap anyway (`grxStreamCreate`'s caveat in `docs/conformance.md`) — but
  when they do, this becomes a real limitation and the mode will need a
  counter-free "timeline only" option.
- **Transfers, in device cycles.** A DMA runs on the command processor, not on
  a core, so the core counters do not describe it. Transfer slices carry host
  time and bytes and no counters, rather than counters that would invite the
  reader to attribute core cycles to a copy that never used one.
- **Anything below the kernel.** There are no per-instruction, per-warp or
  per-cache-line events here. That is `grxgpu/ci/perfetto.py`'s job — see §5.
- **Backends without counters.** A backend that refuses `vx_device_mpm_query`
  produces slices with no counters at all. Not zeros: the record omits the
  field, the report prints `-`, and `ci/build_mock.sh` asserts that no
  `device.*` argument ever appears on a trace taken against the mock driver.

## 4. Using it

```
grx-prof [--out FILE] [--no-trace] [--quiet] -- <program> [args...]
```

`grx-prof` sets `GRX_PROFILE=1`, passes the program's own output through,
writes `grxprof.json` (Chrome Trace JSON, which https://ui.perfetto.dev loads
directly), and prints a report:

```
=== grx-prof ===
device   GRX-G100 (simx), 1 core x 4 warps x 4 lanes, 16 KiB shared per SM
recorded 5 operations: 1 launch, 4 transfers

Per kernel, by device cycles
  kernel                   calls       cycles    IPC   lanes    occ  issue
  vecadd                       1         8311  0.287    100%    99%   0.29

  vecadd -- where the cycles went
    scheduler idle                        70.6%  (5867 cycles issued nothing)
    scoreboard (waiting on a result)      23.7%  (1970)
    operand collector                      1.7%  (140)
    memory unit back-pressure              0.9%  (77)
```

Setting `GRX_PROFILE=1` without the tool works too; the runtime's `GRXPROF|`
lines are readable as key/value pairs.

A program that never reaches the GRXCP runtime emits no summary line, and the
tool says *"the runtime emitted no summary — it was not in profiling mode"*
and exits non-zero, rather than presenting an empty trace as a quiet run.

### The measurement floor

Reading a counter is two DCR reads through the command processor, and the
processor runs while it serves them — so a snapshot is not instantaneous, and
whatever elapses inside one lands in the next delta. That floor is **measured
at startup**, by taking two snapshots back to back with no work between them,
and reported rather than subtracted. On `simx` it comes out at 0 device cycles,
because the simulator does not tick the core to serve a DCR read. On a backend
where it does not, the report says what one sample pair costs and a kernel
whose cycle count is near that number has not really been measured.

## 5. Why not `ci/perfetto.py`

The roadmap's phase 2 line says "Perfetto export reusing `ci/perfetto.py`".
That turned out to be the wrong tool for this job, and the difference is worth
recording rather than quietly ignoring.

`grxgpu/ci/perfetto.py` converts *instruction-level simulator log traces* into
Perfetto JSON: every instruction, every pipeline stage, every cache event. It
needs a debug (non-`NDEBUG`) simulator build emitting `DT(...)` traces, and it
produces a trace whose unit is one instruction.

That is a microarchitecture tool, and it is the right one when the question is
"what is this warp doing in cycle 4,182". It is the wrong one when the question
is "which kernel is expensive and why", because it requires a different
simulator build and the answer is buried in millions of events.

So the two are complementary and both stay: `grx-prof` writes the same trace
*format* — Chrome Trace JSON, so both load in the same viewer — from what the
runtime and the performance counters already know, with no special build. When
`grx-prof` narrows a problem to one kernel, `perfetto.py` on a debug build is
the next zoom level.

## 6. Where it goes next

- Cache counters: the ICACHE / DCACHE / L2 / L3 / MEM classes are implemented
  in SimX and not yet sampled. Hit rate per kernel is the obvious next column.
- DXA and TCU class counters, for the tensor path.
- A roofline: the counters for arithmetic intensity are all here, but a
  meaningful roofline needs a measured peak, not a datasheet number.
- Timeline-only mode with no serialization, for when streams really overlap.
- Per-launch cycle attribution across cores rather than the max, once
  multi-core configurations are exercised.
