# PTA on the CPU: a photonic tensor tile in the c930 NPU

**Companions:** [`heterogeneous_devices.md`](heterogeneous_devices.md),
[`pta_gpu_integration.md`](pta_gpu_integration.md).
**Source analysis:** GRX_PTA_Integration.md in this repository's `docs/`.

**Status: DESIGN, partly built.** The C2 loop interchange is in grx930 (§6), and
so are the S_ACT activation stage, PTM-C, exact at C0, and all six of the
error model's impairments, with C1 closed on its accuracy sweep (§6); PTM-B and
the PTA CSRs are not.
[`pta_program_plan.md`](pta_program_plan.md) orders what comes next. This
document fixes what gets built, in what order, and — more importantly — what
each stage is allowed to claim.

**Scope decision, made before anything else.** This program terminates at FPGA
emulation and numerics. There is no photonic PDK, no MPW shuttle, no
mixed-signal tape-out. That is not a reduced version of the photonic project;
it is a different project with a different deliverable, and the difference has
to be stated once, loudly, or every result downstream gets read as a claim
about silicon photonics that it is not. *Still true of this document,
2026-09-22:* the development board does build a photonic chiplet, but it is the
GPU's, and [`board_program_plan.md`](board_program_plan.md) owns it.

What an emulation-only program can honestly produce:

1. **The architecture that photonic constraints force.** A weight-stationary
   analog tile has a weight-programming cost between two and six orders of
   magnitude above its compute cost, depending on the tuning mechanism — a
   thermo-optic phase shifter settles in tens of microseconds against a shot of
   nanoseconds. That single ratio invalidates the loop nest the
   c930 NPU ships today. An electro-optic tile on thin-film lithium niobate,
   which settles in picoseconds, is the check on that claim, and §2.1 runs it:
   with its weights scanned in by the host it still wants the fix; with every
   weight set held at the tile it wants the shipped loop nest back. Its
   lower-drift sibling, thin-film lithium tantalate, is the tile this program
   targets (§4.4). Finding that, fixing it, and measuring the fix is a real
   result, and it is entirely a digital-RTL result.
2. **The numerics.** How much accuracy survives a 4–6 effective-bit analog
   channel with drift, and what digital correction buys it back. Answerable
   against a seeded, bit-deterministic model.
3. **An open emulation framework.** Photonic-accelerator architectural
   simulators are immature and mostly unpublished. A synthesizable, parameterized,
   deterministic tile model with a real error budget, attached to a working SoC,
   does not currently exist in public.

What it cannot produce, and must never be reported as producing: any statement
about photonic device physics, any throughput number for a photonic chip, any
validation of a mesh topology. The model is a *hypothesis about* published
devices. A green run against it proves the digital architecture, exactly the
way `test_npu_c930_model.cc` proves the backend logic and nothing whatsoever
about the register map (see [`heterogeneous_devices.md`](heterogeneous_devices.md) §5).

---

## 1. Why the c930 NPU is the right host

Not for convenience. Because the dataflow already matches.

A photonic matrix-vector multiplier holds its weights in a slow physical
resource — thermo-optic phase shifters, microring resonances, phase-change
cells — and streams activations past them. A weight-stationary systolic array
holds its weights in a PE register and streams activations past them. These
are the same dataflow, arrived at from opposite directions: both put the
reused operand in the resource that is expensive to change.

grx930/c930/rtl/c930_systolic_array.sv already has every port a photonic tile
needs, under names that translate one-for-one:

| c930 systolic array | Photonic tile | Note |
|---|---|---|
| `i_wen` / `i_wrow` / `i_wcol` / `i_wdata` | weight-DAC scan, one element per beat | A real mesh is programmed serially too; the serial load is not an emulation artifact |
| `i_wbank` (write) / `i_bank_sel` (compute) | shadow bank vs. active mesh | Double-buffered weights are how you hide a slow program behind a fast compute |
| `i_act`, one per row | activation DAC → modulator array | |
| `o_ps_out`, one per column | photodetector → TIA → ADC | Optical summation lands on the detector; the column *is* the reduction |
| `i_precision` broadcast | analog mode: bits, integration time | 3 bits wide, five of eight codes used |

So the tile is a module swap, not an architecture. That matters for the first
milestone (§6) and it is the reason this path is cheap.

**One structural mismatch, and it is the interesting one.** The systolic array
is temporally *skewed*: row `r`'s activation fires at cycle `r`
(`(t == r) && (r < kr_reg)` in the core's feed logic) and column `n`'s result
is captured at `t = n + NUM_ROWS + 2`. A photonic MVM is *broadside*: all N
modulators are driven at once, all M detectors integrate at once, one shot,
one latency. Reconciling those two is §4.

---

## 2. What the existing loop nest costs, before photonics enters

This is the finding the rest of the document hangs on, and it is worth stating
in digital terms first because it is true without any photonics at all.

grx930/c930/rtl/c930_npu_core.sv walks `m` outermost, then the N tile, then
the K tile. Its inner sequence per K tile, in steady state, is:

```
  S_PRELOAD   preload_kr * nc cycles   (load tile k+2's weights into the idle bank)
  S_RUN       NUM_ROWS + NUM_COLS + 2  (compute tile k+1)
```

At the 200T configuration — NUM_ROWS = NUM_COLS = 8 — that is **64 cycles of
weight movement per 18 cycles of compute**. Over one `(m, nt)` pass the machine
spends `64·Kt` cycles moving weights and `18·Kt` computing, plus 8 writing:
the array is doing arithmetic **22% of the time**.

**And the existing counter does not show this.** `stall_cnt` increments only in
`S_WLOAD`, and in steady state the weight movement happens in `S_PRELOAD`,
which is not counted. `STALL_CT` therefore reports `64` per `(m, nt)` pass
where the true figure is `64·Kt` — under-reporting weight movement by a factor
of `Kt`, which is 32 at `K = 256`. Extending `stall_cnt` to cover `S_PRELOAD`
is a one-line change and it is a prerequisite for every measurement in this
document, because otherwise phase C2's improvement will appear to come from
nowhere.

Worse, the weights are reloaded per output row. `S_WRITE`'s exit resets
`nt_reg` and `kt_reg` to zero on every `m` increment, so the same B tile is
programmed `M` times. At `MAX_M = 64` that is 64 redundant programmings of
every weight tile.

Digitally this is a 4.5× inefficiency and nobody has minded. Introduce a
weight-programming time `Tw` measured in microseconds and it stops being an
inefficiency and becomes the entire runtime.

### 2.1 The cost model

Let `Tw` be weight program plus settle, `Ts` everything a shot costs that both
loop orders pay — operand feed, shot, ADC conversion — and `Td` one row write
into `c_mem`, the traffic §2.2 counts. With `Kt = ceil(K / NUM_ROWS)`,
`Nt = ceil(N / NUM_COLS)`, and the accumulator restore folded into the write
(§2.3's first step):

| Loop order | Weight programmings | Row writes | Total time |
|---|---|---|---|
| `m` outer (as shipped) | `M · Nt · Kt` | `M · Nt` | `M · Nt · (Kt · (Tw + Ts) + Td)` |
| `(kt, nt)` outer, `m` inner | `Nt · Kt` | `M · Nt · Kt` | `Nt · Kt · (Tw + M · (Ts + Td))` |

The weight term drops by exactly `M`; the write term grows by exactly `Kt`.
The model is checked, not asserted: with the digital array's own constants —
`Tw` = 64 cycles of weight scan, `Ts` = 18 of `S_RUN`, `Td` = 8 — it reproduces
both of §2.3's measured totals to the cycle, 168,448 as shipped and 71,168
interchanged once the unfolded restore's `M · Nt · (Kt − 1) · Td` is added back
([`pta_tw_sweep.py`](pta_tw_sweep.py) asserts it, and computes every derived
figure in this section and §6.2).

For a plausible thermo-optic mesh — `Tw` = 10 µs, `Ts` = 50 ns — and
`M = 64, N = 8, K = 256`, taking `Td` = 0:

- as shipped: 64 · 1 · 32 · 10.05 µs ≈ **20.6 ms**
- interchanged: 32 · (10 µs + 64 · 50 ns) ≈ **0.42 ms**

Roughly 49×, and the gap widens with `M`. **For a thermally tuned tile, the
loop interchange is the photonic integration.** Everything else in this
document is instrumentation for measuring it or numerics for making the result
trustworthy.

**`Ts` does not decide which order wins.** Subtract the two totals and it
cancels. The interchange pays exactly when

```
  Tw  >  Td · M · (Kt − 1) / ((M − 1) · Kt)   ≈ Td      (0.98 · Td at M = 64, K = 256)
```

— when one weight program costs more than one row write, or two until the
restore is folded. The `Tw`-to-`Ts` ratio sets how much the winner wins by;
`Tw` against `Td` sets which order it is.

**Pockels-class tiles.** An electro-optic tile on thin-film lithium niobate
(TFLN) or lithium tantalate (TFLT) is the case that tests this. Its phase
shifters are Pockels modulators — for TFLN, 45 GHz of 3-dB bandwidth from a
20 mm device in the 2018 CMOS-voltage demonstration (Wang et al., *Nature* 562,
101, 2018) — and a single-pole response at that bandwidth settles to half an
8-bit LSB in ln(512) / (2π · 45 GHz) ≈ 22 ps, or 25 ps at TFLT's 40 GHz. The
settle is gone. What is left of `Tw` is whatever delivers the weights, and that
splits the regime in two:

- **Scanned.** The host writes the tile's 64 weights one per beat, as
  `i_wen`/`i_wrow`/`i_wcol` already do (§1). `Tw` is the scan: 64 cycles,
  640 ns at 100 MHz, and 22 ps of settle does not register. This is the digital
  array with its 18-cycle run cut to a one-cycle shot.
- **Resident.** Every weight set the GEMM uses is already at the tile, so a
  program is a bank select — roughly one DAC update, inside the shot's own
  cycle, counted as zero. That takes `Nt · Kt` banks: §8 item 3's multi-bank
  tile.

Priced with the c930 as host at 100 MHz — `Td` = 80 ns, and `Ts` at least one
cycle, because the host presents one operand vector per cycle however short the
shot:

| Tile | `Tw` | `Ts` | As shipped | Interchanged | Faster order |
|---|---|---|---|---|---|
| thermo-optic | 10.6 µs (scan + 10 µs settle) | 50 ns | 21.9 ms | 0.61 ms | interchange, 36× |
| Pockels, scanned | 640 ns (scan) | 10 ns | 1.34 ms | 0.20 ms | interchange, 6.5× |
| Pockels, resident | 0 (sub-cycle) | 10 ns | 25.6 µs | 184 µs | **as shipped, 7.2×** |

The c930's own row write costs the thermal tile some of its 49× and none of the
argument. A scanned Pockels tile still wants the interchange, by an amount the
host sets — 64 cycles of scan against 8 of row write, whatever the modulator
does. A resident one wants the shipped order back: there is no weight cost left
to amortize, only row writes to multiply. Its 25.6 µs is 20.5 µs of one-cycle
shots and 5.1 µs of row writes, so it is bound by the host — its feed and its
writes — not by the tile: the review's operand-supply finding
([`pta_tpaqcn_review.md`](pta_tpaqcn_review.md) §7) arriving from the tile
side.

What a Pockels tile pays instead of settle is bias drift, and it is where the
two materials part. Lithium niobate modulators drift under a held DC bias: side
by side, a quadrature-biased TFLN modulator's output power fluctuated by 5 dB
over 46 hours, against under 1 dB for thin-film lithium tantalate (Powell et
al., *Opt. Express* 32, 44115, 2024). At this end, accuracy over a long run is
a calibration question (§5.1), not a loop-nest one — and that drift gap is most
of why §4.4 picks lithium tantalate.

[`pta_material_scorecard.py`](pta_material_scorecard.py) puts thermo-optic
silicon, thin-film lithium tantalate and non-volatile barium titanate through
the same model. Resident, every one of them lands on the same 25.6 µs. What
differs is what holding the weights costs — about 25 W of heater power for one
GEMM's 2,048 thermo-optic shifters, under a milliwatt for barium titanate — and
how far they drift. Rewritten per tile, barium titanate's 80 ns switch lands
within 2% of the 78.7 ns break-even. §4.4 turns the comparison into the choice
of material.

### 2.2 What the interchange costs

Accumulator state. Today `acc[0..NUM_COLS-1]` is one output row's running
K-accumulator — `NUM_COLS × ACC_W` = 8 × 48 bits, in flops. With `m`
innermost you need `M` accumulators live at once.

That storage already exists. `c_mem` is `MAX_M * MAX_N` words, declared
`(* ram_style = "block" *)` and already BRAM-mapped, on a device with 357
unused BRAMs. The register accumulator disappears into it.

**The cost is accumulator traffic, and the first estimate here under-counted
it.** Two terms appear that the `m`-outer order does not pay:

- `S_WRITE` currently runs once per `(m, nt)`. With the K accumulation living
  in `c_mem` it runs once per `(m, nt, kt)` — `Kt` times as often.
- Restoring the running sum before each run costs another `nc` cycles per
  `(m, nt, kt)`.

Together those are `2·nc` cycles against 18 cycles of compute per run.

The correctness obligation is the FP16/BF16 path, and it decides the shape of
the fix. FP32 addition is not associative, so accumulating a whole K tile and
then adding it to `c_mem` gives a different answer from threading the running
sum through the array. Keeping `i_ps_in` fed from a restored `acc[]` — rather
than zeroing it and adding afterwards — preserves the addition order exactly,
which is why the restore state exists at all. It buys bit-identical results in
every precision at the price of those `nc` cycles.

### 2.3 The interchange helps the digital array too — measured

Under the shipped loop order the digital array is 22% efficient. Measured on
`c930_npu_core` at `M=64, N=8, K=256`, driving the core's data-plane port
directly (`tb/tb_core_m64.sv`):

| | cycles | weight+accumulator cycles | PE ops |
|---|---|---|---|
| `m` outer, as shipped | 168,448 | 131,072 | 2,359,296 |
| `m` inner, sums restored from C | 71,168 | 17,920 | 2,359,296 |

**2.37× on the shipping NPU, with no photonics involved**, results
bit-identical and `op_count` unchanged to the last operation — the array does
exactly the same arithmetic, just without reloading the same weights 64 times.

These totals, and the asserts in `pta_tw_sweep.py` that check them, are for the
core before grx930's half-rate hop (0afeb6e), which runs `S_RUN` at 64 cycles a
K tile for every precision instead of 18. On that core the same interchanged
GEMM takes 165,375 cycles: compute grows 3.6-fold and the 17,920 cycles of
weight and accumulator movement do not change (program plan §3.3).

The remaining 71,168 splits as 36,864 cycles of compute, 16,384 of result
writeback and 17,920 of weight and accumulator movement, so two further steps
are available and each is separately measurable:

- Fold the accumulator restore into the previous row's `S_WRITE`, whose read
  port is idle — removes 15,872 cycles, giving **3.05×**.
- Fold the writeback into the tail of `S_RUN`, where columns already complete
  one per cycle — removes 16,384 more, giving **4.33×**.

4.33× is the ceiling this loop order allows, not the first step. Quoting it as
the headline before the first step is measured is how a plan starts lying to
itself.

---

## 3. Precision codes and the CSR address space

Two hard constraints found by reading, not assumed.

**`i_precision` has room.** It is `[2:0]` at every level — `o_precision` in
the CSR, `i_precision` in core, array and PE — and codes 0–4 are used
(INT8, INT16, FP16, BF16, INT4). Codes 5, 6, 7 are free, which is enough for
the analog modes proposed in §5.2. No datapath widening.

The catch is the DMA. INT4 is nibble-packed and the cross-GEMM prefetch path
(PF2) was enabled specifically for that mode, so a new precision code is not
free in grx930/c930/rtl/c930_npu_dma.sv — it needs an unpack rule and a
`dims_ok` clause. Budget for it; do not describe it as a three-line change.

**The CSR decode is full; the address space is not.** grx930/c930/rtl/c930_npu_csr.sv
decodes `s_axi_awaddr[5:2]`, sixteen words, and all sixteen codes are assigned
(CTRL, STATUS, DIM_M/N/K, A/B/C_BASE, PREC, CYCLE_LO, DMA_LAST, OP_COUNT,
STALL_CT, DMA_CT, QUEUE_STAT, QUEUE_MAX). There is no spare offset inside the
register file.

Outside it there is plenty. The SoC crossbar decodes the MMIO slave over
`0x4000_0000`–`0x4000_FFFF` — 64 KB — with the UART carved out at
`0x4000_1000`–`0x4000_100F` by an earlier branch in the same decode function.
So widening the CSR's internal decode from `[5:2]` to `[7:2]` — 64 words,
`0x00`–`0xFC` — leaves `0x00`–`0x3C` bit-identical, puts PTA state at
`0x40`–`0xCC`, and **requires no crossbar change at all**.

One thing to fix while there. The c930 architecture document's memory map
declares the NPU MMIO region as `0x4000_0000`–`0x4000_001F`, 32 bytes, with
everything above reserved. That is already wrong in two directions: the
crossbar decodes 64 KB, and the same document's own performance-counter table
lists offsets at `0x2C`, `0x30` and `0x34`. The map should be corrected to the
implemented 64 bytes before it is extended to 256, or the extension inherits a
document nobody trusts.

### 3.1 Proposed PTA register block

| Offset | Name | Access | Description |
|---|---|---|---|
| 0x40 | `PTA_CTRL` | RW | bit0 EN, bit1 CAL_NOW, bit2 CAL_AUTO, bit3 MODEL_RST, bits[6:4] CAL_SCHED |
| 0x44 | `PTA_STATUS` | R | bit0 CAL_BUSY, bit1 CAL_VALID, bit2 SAT_STICKY, bit3 DRIFT_ALARM, bits[23:8] last calibration residual |
| 0x48 | `PTA_IMPAIR` | RW | one enable bit per impairment: QUANT, THERMAL, SHOT, DRIFT, XTALK, MZM_NL, PROG_ERR |
| 0x4C | `PTA_BITS` | RW | [3:0] activation bits, [7:4] weight bits, [11:8] ADC bits, [17:12] ADC shift `S`, so `LSB_adc` = 2^S |
| 0x50 | `PTA_SEED` | RW | seed for every noise generator; each reloads from it at a GEMM start (§4.3) |
| 0x54 | `PTA_SIGMA_TH` | RW | thermal/TIA noise σ, Q8.8 in ADC LSB |
| 0x58 | `PTA_SIGMA_SH` | RW | shot-noise coefficient `k`, Q8.8, so σ_shot = k·√\|y\| with σ and y in ADC LSB |
| 0x5C | `PTA_SIGMA_PR` | RW | weight-programming error σ, Q8.8 in weight LSB |
| 0x60 | `PTA_DRIFT` | RW | [15:0] drift step σ, Q8.8 in weight LSB; [31:16] log2 optical shots per step. Defaults to TFLT's fit: σ 55, a step every 2^31 shots (§4.4) |
| 0x64 | `PTA_XTALK` | RW | coupling between neighbouring inputs of a bank, Q0.8 (§4.3) |
| 0x68 | `PTA_TW` | RW | settle after a program's last weight write, in core cycles; 0 is legal (§6.2) |
| 0x6C | `PTA_TS` | RW | shot + ADC latency, in core cycles |
| 0x70 | `PTA_CAL_PER` | RW | calibration period for the periodic scheduler |
| 0x74 | `PTA_CAL_THR` | RW | predicted-error threshold for the drift-predictive scheduler |
| 0x78 | `PTA_CAL_CT` | R | calibrations run |
| 0x7C | `PTA_CAL_CYC` | R | cycles spent calibrating |
| 0x80 | `PTA_SHOT_CT` | R | optical shots issued |
| 0x84 | `PTA_WLOAD_CT` | R | weight-bank programmings |
| 0x88 | `PTA_SAT_CT` | R | ADC saturation events |
| 0x8C | `PTA_ERR_MAX` | R | max \|measured − expected\| from the last calibration |
| 0x90–0xAC | `PTA_GAIN[j]` | RW | per-column gain, Q8.8 |
| 0xB0–0xCC | `PTA_OFFS[j]` | RW | per-column offset, signed |
| 0xD0 | `PTA_DRIFT_MAX` | RW | drift clamp, Q8.8 in weight LSB, defaulting to TFLT's 8,643 (§4.4); the first word past the block, shared with S_ACT's scalars at C4 |

On the development board this block is not a c930 CSR at all: it is MMIO in the
GPU's BAR, reached over CXL.io
([`board_program_plan.md`](board_program_plan.md), B7). The block is the same;
what changes is who decodes it.

`PTA_CAL_CYC` and `PTA_WLOAD_CT` are not diagnostics. They exist so a reported
GEMM time can be decomposed into compute, weight programming, and calibration.
Without them `CYCLE_LO` folds all three into "compute" and every speedup number
this project produces is unfalsifiable. They are the honesty instrumentation,
and they should be read as the same class of object as
`ci/check_perf.py`'s zero-tolerance baselines.

### 3.2 Calibration must not break the completion contract

grx930/c930/rtl/c930_npu_csr.sv spends fourteen lines of header warning that
polling `DONE` then `BUSY` is invalid for a batch, and that the only correct
test is `QUEUE_STAT` occupancy == 0 **and** `STATUS.BUSY` == 0. Calibration
introduces a third machine state and can silently invalidate that predicate.

The failure, precisely: if calibration blocks dispatch without setting `BUSY`,
then a `CTRL.START` arriving during calibration takes the dispatcher's
`D_IDLE` / `start_requested` / `fifo_empty` branch, which dispatches directly
from the live CSRs — into a tile that is not available. If instead the START
is dropped, occupancy stays 0 and `BUSY` stays 0 and a driver correctly
applying the documented predicate concludes the batch finished.

The fix is minimal and it preserves the predicate exactly. Introduce
`cal_busy` and use it to widen the existing "cannot dispatch now" condition:

```
wire cannot_dispatch = i_busy || cal_busy;
wire do_push = (start_requested && !fifo_full && (cannot_dispatch || !fifo_empty))
            || (pending_start && !pending_pushed && !fifo_full && cannot_dispatch);
```

and gate all three `D_IDLE` dispatch branches (pending, start_requested, drain)
on `!cal_busy`. A START during calibration then goes to the FIFO, occupancy
becomes non-zero, and `occupancy == 0 && busy == 0` correctly reports
not-done. `STATUS.BUSY` keeps its per-command meaning; calibration is visible
only through `PTA_STATUS.CAL_BUSY`.

Getting this wrong reintroduces exactly the queue-drain class of bug the CSR
header documents, so it wants its own regression: START during an in-progress
calibration, three times back to back, occupancy checked at each step.

*Built in C3(b), 2026-09-23*, as written above and with that regression:
`grx930/c930/tb/tb_npu_cal_queue.sv`, `make cal_queue`. Three STARTs during a
calibration queue at occupancy 1, 2 and 3, nothing dispatches into the tile, and
`occupancy == 0 && busy == 0` never reads "finished" with work outstanding.
`CAL_GUARD_ABLATE` drops calibration from the dispatch condition and the bench
fails exactly as described — the first START dispatches into the calibrating
tile and the queue strands. One thing the CPU document did not say: a
calibration between GEMMs must not report BUSY, or the guard is never exercised,
so the core's `o_busy` excludes it and only one that interrupted a GEMM keeps
BUSY set.

---

## 4. The tile: two variants, and why both exist

### 4.1 PTM-C — the compatibility shim

A module with the port list of `c930_systolic_array`, parameter for parameter,
that absorbs the skew at its own boundary. It is built, as grx930's
`c930/rtl/pta/c930_ptm_c.sv`.

The array's ports carry no start strobe, and PTM-C needs none, because the
array is a fixed transform of its input streams. Since grx930's half-rate hop
(§2.3), every PE register that carries the activation or the partial sum
updates on hop edges, so each PE delays both by two hop windows, and the core
changes `i_act` and `i_ps_in` only on hop edges. With `R = NUM_ROWS`, column
`c`'s output in hop window `i` is the seed from window `i − 2R`, plus each row
`r`'s product with the activation from window `i − 2(R − r) − 2c`, added top
row first. PTM-C keeps each input's history on hop edges and takes exactly
those samples (the de-skew), evaluates each column in one combinational pass
(the shot), and registers the result on the hop edge (the re-skew). It reads
weights and the bank select at the shot, which is exact because the core
writes weights only in `S_WLOAD` and holds the bank through `S_RUN`. Float
columns chain a combinational copy of `c930_fp16_acc` in the cascade's row
order, since IEEE addition does not associate. As built, PTM-C is a simulation
model: a float column is `NUM_ROWS` multiplies and additions in one pass, and
nothing has timed it.

Its purpose is not performance. **With every impairment disabled it must be
bit-identical to the systolic array on the existing self-checking testbench.**
That single property is what makes every later numerical difference
attributable: if the shim is not proven exact first, an accuracy delta observed
in phase C1 could be the error model or could be a de-skew bug, and there is no
experiment that separates them after the fact. C0's gate (§6) says it is.

Zero changes to core, DMA or CSR. `make <bench> PTM_C=1` builds any NPU bench
with the shim in place of `rtl/c930_systolic_array.sv`, into its own build
directory.

### 4.2 PTM-B — the broadside tile

The tile the architecture actually wants: takes a whole K-tile activation
vector, returns a whole column of results after `PTA_TS`. Requires the core's
`S_RUN` to become a shot-and-wait rather than a skewed drain (18 cycles a K
tile before grx930's half-rate hop, 64 after), and it is the variant the §2
loop interchange is written against.

PTM-B is where the speedup lives and PTM-C is where the trust lives. Build
both; keep both; run the numerics on whichever is convenient, since with
identical error parameters they must agree.

### 4.3 The error model

The model in the source analysis — `sum += ($signed(lfsr[3:0]) - 8)`, a
uniform ±8 LSB — is not defensible in a paper and should not be built. The
first-order model that is:

```
  y_j = g_j · Σ_i  (q_w(w_ij) + eps_ij + delta_ij(t)) · q_a(x_i)
      + Σ_i Σ_{i'~i} chi · (q_w(w_i'j) + eps_i'j + delta_i'j(t)) · q_a(x_i)
                                            crosstalk, neighbouring inputs
      + n_th                                            thermal / TIA, sigma_th
      + n_sh(y)                                         shot, sigma = k*sqrt(|y|)
      + o_j
  d_j = sat( round( y_j / LSB_adc ), 2^(B_adc-1) )
```

with `q_a` an activation quantizer at `B_a` bits (plus the MZM's sinusoidal
residual when MZM_NL is enabled), `q_w` a weight quantizer at `B_w` bits,
`eps_ij` a fixed programming error drawn once per weight load,
`delta_ij(t)` a bounded random walk updated every `2^PTA_DRIFT[31:16]` optical
shots and reset by calibration, and `i' ~ i` the inputs either side of `i` in
output `j`'s bank.

**Settled for C1** (2026-09-14). The weight errors act after the DAC, as the
formula now reads: a Pockels weight is a DAC level that the device's error then
moves, and the source analysis's `q_w(w + eps + delta)` would erase any error
smaller than one DAC step. `LSB_adc` is `2^S`, with the shift `S` set per GEMM:
the activation-scaling knob below. The model is integer and covers the integer
precisions only. Its generators reload from `PTA_SEED` at every GEMM start and
draw in the core's loop order — N tile, K tile, output row, column — for the
results the core marks as captured, so a result depends on the seed, the shape
and the operands, and on nothing the DMA's timing can move. grx930's
`c930/doc/pta_error_model_design_note.md` holds the fixed-point contract.

**Drift and crosstalk, settled** (2026-09-15). Drift's clock is optical shots —
one core run, one output row over one K tile — not cycles, which would bring
back the timing dependence above. Drift is device state: it accumulates across
GEMMs until a model reset (`PTA_CTRL.MODEL_RST`, and in C3 a calibration), so a
drifting GEMM also depends on the GEMMs run since, and is clamped at a set
bound. Crosstalk couples neighbouring inputs in the same output's bank: input
`i`'s light also passes the rings of `i − 1` and `i + 1`, so it sees `chi`
times their analog weights, as the formula now reads. The version before
multiplied each neighbour's weight by the neighbour's own input, which is a gain
error, not crosstalk. An input outside the current K tile holds no weight for
its neighbour, as a ring tuned off resonance holds none.

**Shot noise is signal-dependent and that is the whole point.** Modelling
detector noise as a constant additive term — which every casual model does —
removes the one property that drives the architecture: SNR improves toward
full scale, so effective precision depends on where in the dynamic range the
tile is operating, so *activation scaling becomes an architectural knob*
(§5.2). A constant-noise model produces a project with no interesting
questions in it.

Synthesis cost is modest. Gaussian noise from the sum of three or four LFSR
draws (Irwin–Hall) is a standard trick and needs no multiplier. The `√|y|`
term is a leading-zero count plus a four-entry interpolation LUT — c930 already
has the comparator/leading-zero primitives in `c930_cla_comp`. Per-column gain
and offset are one DSP and one adder each.

**Determinism is a hard rule.** No `$random`, no `$urandom`, no
`$urandom_range`, anywhere in the model. Every stochastic quantity comes from
an LFSR seeded by `PTA_SEED`. The reason is not aesthetic: grxcp's rule is that
every conformance test runs on two backends and a divergence blocks
(`AGENTS.md` §4). An analog device with a non-reproducible model makes that
rule unenforceable, and the project quietly loses its ability to distinguish a
model bug from a backend bug. Seeded determinism keeps RTL↔C parity a bitwise
question.

### 4.4 The target material: thin-film lithium tantalate

**Decision.** The tile this program targets is an electro-optic mesh on
thin-film lithium tantalate (TFLT). The Pockels points of the §6.2 sweep stand
for it, and the error model's drift term (§4.3) and the §5.1 calibration
schedulers are tuned to it. Thermo-optic silicon and thin-film lithium niobate
stay in, as stress cases.

**The criterion is holding, not speed.** §2.1 and §5.5 put the design's
destination at resident weights, and once weights are resident every candidate
gives the same GEMM time — 25.6 µs at `M = 64, N = 8, K = 256` with the c930 as
host, bound by its operand feed and row writes, not by the tile. Speed matters
only until a material settles inside a host cycle, after which the host's scan
sets `Tw` (§2.1). What separates the candidates beyond that is what holding a
weight costs: power, drift and precision.
[`pta_material_scorecard.py`](pta_material_scorecard.py) prices each:

| | Thermo-optic Si | TFLN | TFLT | BTO, non-volatile |
|---|---|---|---|---|
| Settles inside a 10 ns host cycle | no: 7.6 µs | yes: 22 ps | yes: 25 ps | no: 80 ns, on the §2.1 break-even |
| Holding one GEMM's 2,048 resident shifters | 25 W of heaters | a DAC-held voltage each | a DAC-held voltage each | 0.57 mW static |
| Drift over the same 46 h test, in 6-bit LSBs | not measured | 11–15 | under 2–4 | "stable", not quantified |
| Weight levels demonstrated | continuous | continuous | continuous | eight |

- **Thermo-optic silicon** fails on both settling and holding. The interchange
  keeps it workable (36×, §2.1), which is why it stays as the loop-order stress
  case — and on a foundry PDK it remains the cheapest physical demonstration.
- **TFLN** has the speed and a commercial product behind it, and the worse
  drift of the two Pockels materials. It stays as the drift stress case for
  §5.1's calibration schedulers.
- **TFLT** keeps TFLN's speed — 40 GHz of 3-dB bandwidth at 1.96 V·cm
  (C. Wang et al., *Nature* 629, 784, 2024) — with a fraction of its drift
  (§2.1), on wafers already made in volume for 5G radio-frequency filters.
- **Non-volatile barium titanate** holds best, at 0.57 mW of static power
  across all 2,048 shifters. But eight levels fall short of the 4–6 effective
  bits the numerics study assumes (§1), and a tiled GEMM rewrites 2,048
  weights per call, an endurance question nothing in hand answers. It is the
  candidate for a mesh that holds a whole layer untiled, not for this tile.

**What it sets.**

- *Timing.* TFLT is Pockels-class: a 40 GHz single-pole response settles in
  25 ps, so the Pockels points of §6.2 apply unchanged. Scanned, the tile still
  wants the interchange (6.5×); with resident banks it wants the shipped order
  (7.2×).
- *Drift.* The error model's `delta_ij(t)` (§4.3) — stepped and timed by
  `PTA_DRIFT` — defaults to a setting fitted to TFLT, with a TFLN setting as
  the stress case. C1 fitted both at the EO-res shot rate, 80 M shots/s, with
  a step every 2^31 shots and σ set so the RMS drift at 46 hours matches the
  material's swing: σ 55 and 200 in Q8.8 weight LSB, clamped at 8,643 and
  31,413. On the D3 network, accuracy holds within half a point for about an
  hour at TFLT's fit and loses about a point in six minutes at TFLN's
  (grx930's `c930/doc/pta_error_model_design_note.md`, §5). The 46-hour
  comparison stays their anchor, not their value; what they size is
  calibration (§5.1).
- *Priorities.* A Pockels weight's cost is set by its DAC and its host, not by
  the material, so the multi-bank tile (§8 item 3) and the host's operand feed
  (§8 item 2) move ahead of further loop-order work.
- *Nonlinearity.* Unchanged by the material: electronic, at the tile boundary,
  as the review concludes — on measured numbers no candidate beats an analog
  optoelectronic neuron ([`pta_tpaqcn_review.md`](pta_tpaqcn_review.md) §4.5).
  The all-optical branch waits on the S_ACT chain experiment (grx930
  `c930/doc/npu_act_stage_design_note.md`, A3), which measures how many
  optical nonlinearities can run between digital resets.

**What would reopen it.**

- A partner or product that needs a platform now: TFLN, the commercial choice,
  with calibration carrying its drift.
- A physical demonstration on the cheapest route: thermo-optic silicon.
- Non-volatile barium titanate at 4–6 bits, with an endurance figure that
  covers a rewrite on every GEMM call.
- A3 finding a long reset interval, which makes the χ(2) platform matter — and
  poled lithium niobate leads there.

**Caveats.** TFLT is the youngest platform in this comparison: modulators,
low-loss waveguides and soliton microcombs are demonstrated, but no
programmable mesh at scale in the sources checked. Neither lithium tantalate
nor lithium niobate is CMOS-monolithic. The integration route to watch is
heterogeneous: TFLT on silicon photonics chips, with modulators past 70 GHz and
no change to the standard silicon photonics process design kit (M. Niels et
al., *Nat. Photonics*, 2025; arXiv:2503.10557).

---

## 5. Ideas worth building that are not in the source analysis

Five. Each is cheap on hardware already owned, and each answers a question the
source document does not ask.

### 5.1 Calibration in the memory shadow

Calibration costs time. Drift costs accuracy. The naive answer is a fixed
period, which is either too often (wasted) or too rare (inaccurate) and is
never right for two different workloads.

Better: calibrate when the tile would be idle anyway. The DMA already stalls
the array while fetching A and B, and the machine already measures it —
`DMA_CT` is a live counter and `DMA_LAST` latches the last GEMM's total. Under
the §2 interchange those stalls get *longer* per weight tile, because a tile now
feeds `M` rows. A calibration that fits inside the PF2 prefetch window is free
in wall-clock terms.

Four schedulers, selected by `PTA_CTRL[6:4]`, so they can be compared rather
than argued about: off, periodic, drift-predictive (extrapolate from the last
two calibration residuals and fire when predicted error crosses
`PTA_CAL_THR`), and shadow (fire only inside a DMA stall, with the predictive
threshold as a backstop). `PTA_CAL_CYC` measures what each actually costs.

*Built in C3(b), 2026-09-23.* `grx930/c930/rtl/pta/c930_pta_cal.sv`, with the
stall the core can see for itself: a row the DMA has not landed yet, which is
`S_AROW`. **One sentence above is wrong and building it is what showed that.**
"Extrapolate from the last two calibration residuals" cannot work: a residual is
what a calibration *leaves*, and one that worked leaves almost nothing, so the
rate it implies is almost zero and the scheduler stops scheduling. The engine
extrapolates what a calibration *found* instead — the widest correction its first
pass had to make, before any of it was applied, published as `PTA_ERR_FOUND` —
and `PTA_ERR_MAX` keeps its meaning as what is left. Two more things the
implementation had to settle: the rule is compared as two products, `E·r ≥
thr·L`, so there is no divider on the tile; and `PTA_CAL_PER` becomes a floor on
the interval for the two predicting modes, because a short interval reads as a
steep rate and the extrapolation runs away without one.
[`pta_chiplet_calibration.md`](pta_chiplet_calibration.md) §5 has the reasoning
and §8 the measurement.

This is the clearest publishable contribution in the CPU path: *calibration
scheduling for analog accelerators, measured against a real memory system
rather than an assumed one.*

### 5.2 Block scaling, borrowed from the GPU

The G100 tensor unit already implements microscaling formats — mxfp8, mxbf8,
mxfp4, nvfp4 — with per-block scale factors carried in a separate metadata
SRAM. Those formats exist to keep a low-precision channel near full scale.

That is exactly, and non-obviously, what an analog tile needs. Shot noise makes
SNR position-dependent in the dynamic range; a block-scaled integer format with
a per-K-tile exponent keeps every shot near the top of the ADC range where the
SNR is best. So the analog precision codes proposed for `i_precision` should be
**block-scaled**, not plain INT: code 5 = 4-bit activations with a per-tile
scale, code 6 = 8-bit with a per-tile scale.

This has a second payoff. The scale factor per tile is exactly the quantity a
compiler can compute statically or a calibration pass can measure, which turns
"pick the operating point" from a hardware problem into a software one — and it
is the same problem the GPU path already has machinery for. One numeric format
across both devices is worth more than either device's local optimum.

### 5.3 Chopper stabilization across the K loop

A GEMM accumulates over `Kt` tiles. Random error grows as `√Kt`; systematic
error — the per-column offset `o_j`, drift bias — grows as `Kt`. After 32 K
tiles the systematic term dominates by 5.7×.

Borrow from analog circuit design: negate the activation encoding on odd K
tiles and negate the ADC result on odd K tiles. Signal adds; a constant offset
cancels exactly over each tile pair. Cost in RTL: two XOR-with-sign stages and
one bit of state. Cost in accuracy: nothing. It does not touch gain error,
which is why per-column gain still needs calibration.

Chopping is textbook in ADCs and, as far as the literature reviewed for this
document goes, unreported as a K-loop technique in tensor accelerators. It is
nearly free to build and it is directly measurable with the drift model on.

### 5.4 Sparse digital residual correction

Analog error is roughly proportional to the magnitude of the terms being
summed, so a small number of large weights carry most of the error. Keep the
top-`k` weights per tile in a digital side path, subtract their analog
contribution and add the exact one.

On the c930 this is unusually cheap, because if PTM-C rather than PTM-B is
instantiated the digital multipliers have not been deleted — PTM-C removes the
PEs' skew, not their arithmetic — so the tile can compute the `k`-term
correction on the same hardware in `k` extra cycles. The experiment is:
sweep `k` from 0 to 8 and measure recovered accuracy per cycle spent. If the
curve is steep at small `k`, mixed analog/digital tensor units are a result; if
flat, that is a negative result worth publishing too.

### 5.5 State the hypothesis that could kill the project

If the `Tw`-to-`Ts` ratio is large enough, the optimal machine is not "a photonic tile inside
a general GEMM engine" — it is a weight-resident inference engine that never
reprograms during a model's execution, with the digital array handling
everything else. That conclusion is *available from this program*: sweep
`PTA_TW` and find the crossover.

The sweep has a second end, and the Pockels-class points of §2.1 reach it. There
the hypothesis inverts: a resident electro-optic tile wants the shipped
`m`-outer order, 7.2× ahead of the interchange, because a bank select leaves no
weight cost to amortize. Weight residency is the destination at both ends, for
opposite reasons — thermal weights are too slow to reprogram, and electro-optic
ones switch fast enough that keeping every set at the tile costs only memory.
The interchange is the right loop nest in between: wherever weights must be
reprogrammed and a program costs more than a row write.

Writing it down as a hypothesis under test, before the sweep, is the
difference between a measurement and a justification. It also determines what
the CPU path should be optimizing for if the crossover lands badly: weight-set
residency and capacity, not GEMM throughput — and at the electro-optic end, the
host's per-shot feed and write path as well.

---

## 6. Phases and exit gates

Each gate is a thing that can fail, and the ablation is named, in the style of
[`heterogeneous_devices.md`](heterogeneous_devices.md) §5.

**C0 — the swap is exact.** PTM-C built, pin-compatible, all impairments off.
*Gate:* the existing self-checking NPU testbench passes with PTM-C substituted
for the systolic array, with zero output differences across all five precision
modes. *Ablation:* flip one de-skew index and confirm the testbench goes red.
Nothing else in this document may start before this gate is green — every later
number depends on it.
*Met in grx930:* built with `PTM_C=1`, `tb_c930_npu` passes, `tb_npu_float_prec`
passes 24/24 across the five precisions, and `tb_npu_feed`'s four
`M=64, N=8, K=256` GEMMs check, with a log — every DMA phase and core state
count — byte-identical to the array's. A lockstep bench, `tb_ptm_c_lockstep`,
drives PTM-C and the array with the same random weights, row enables and input
streams, and finds no difference on any of 12,195 cycles at 8×8 in all five
precisions, or at 4×4 in the integer ones; it also holds the float adder to
`c930_fp16_acc` over 80,196 operand pairs. *Ablation, red:* with row 3's
de-skew one window late, `tb_c930_npu` passes its K = 2 GEMM and stops at the
next, K = 5, and `tb_npu_float_prec` fails 22 of 24, passing only the two with
K = 1. No GEMM whose K tile reaches row 3 survives it.

**C1 — the error model, one impairment at a time.** Quantization, then
thermal, then shot, then programming error, then drift, then crosstalk. Each
behind its `PTA_IMPAIR` bit, each with its own directed test.
*Gate:* (a) an impairment sweep on a small MLP with published accuracy-vs-bits
curves; (b) RTL and the C model agree **bitwise** for a fixed `PTA_SEED`,
driven through the existing NPU DPI wrapper. *Ablation:* corrupt one LFSR tap
and confirm the parity check fails.
*Built in grx930:* all six impairments are in PTM-C, with a C reference,
`c930/sim/pta_tile_model.c`. At all 14 shapes of the core-level Verilator
harness, `M=64, N=8, K=256` included, and at 8- and 16-bit operands, C and the
ADC saturation count agree bitwise with the C reference for each impairment
alone and for all six together. Hand-worked cases pin the quantisers' and the
ADC's rounding, crosstalk's coupling, and drift's semantics: zero after a model
reset, clamped at its bound, unmoved when its step is zero, and zero again
after another reset. Drift's fourteen GEMMs run on one device, since drift
accumulates across them, and a model reset pulsed while the core is busy is
ignored, as it must be. With every impairment clear, C0's benches still pass
and the feed log is still byte-identical to the array's.
*Ablations, red:* one xorshift32 shift changed fails every shape carrying
thermal, shot, programming or drift noise; dropping the quantiser's rounding
term fails both hand-worked quantiser cases and 10 of the 14 quantised shapes;
clearing drift at every GEMM start fails 13 of the 14 drift shapes; and letting
rows outside the K tile couple fails the stale-row case and exactly the three
shapes whose K tiles are partial. Gate (b) runs in that harness, not the DPI
wrapper, until C4 puts the configuration on a CSR (§9, question 3).
*Gate (a), not met at one width:* the D3 network, a 784-100-10 MLP with ReLU on
MNIST, was trained five times at each weight width from 1 to 10 bits, as
Gorsline, Smith and Merkel trained theirs (arXiv:2105.00227) but from a 16-bit
network of the same seed, and run through the C reference at 16-bit operands.
Against their Fig. 3(c), the five-network mean was within 0.5 points at every
width from 4 to 10 bits and within 1.0 at 2 bits, but 0.61 under at 3 bits,
outside the 0.5 fixed before training. The tile's accuracy equalled the digital
network's on 48 of the 50 networks and was one image off on the other two, so
the miss is in the training, against a paper that states neither its quantiser
nor its stopping rule. It is recorded, and C1 closes on it. *Ablation, red:* a
quantiser that truncates fails every width from 2 to 7 bits. *Reported beside
it,* at 8-bit operands and 6-bit weights: 5 activation bits and a 6-bit ADC
each stay within a quarter point of 97.45%, and thermal noise of 1 ADC LSB,
shot noise at 3 photons per ADC LSB, programming error of 4 weight LSB and 10%
crosstalk each cost under half a point; drift's cost is in §4.4.

**C2 — loop interchange and the broadside tile.** The `m`-inner FSM landed
first, on the digital array, ahead of any tile work — see §2.3. PTM-B then
drops into it.
*Gate, met on the digital array:* weight loads drop from `M·Nt·Kt` to `Nt·Kt`
exactly; `op_count` is unchanged to the last operation; results are
bit-identical across every shape in the NPU testbench and at `M=64, K=256` in
`tb/tb_core_m64.sv`, at both 8- and 16-bit operand width.
*FP16/BF16, closed by test rather than construction:* `tb_npu_float_prec`
checks float GEMMs bit-exactly, including running sums carried across K tiles
through C, and passes 24/24 on the interchanged core; the UART GEMM sweep on
the full SoC passes 18/18. Getting there fixed two float bugs that predate the
interchange — stale weights in rows outside a K tile, which IEEE multiplication
turns into NaN, and NPU operand reads served from L2 lines nothing invalidates
— plus a silent-corruption bug the A-row interlock introduced, a core left
waiting after the DMA abandons a GEMM.
*Remaining for the tile:* total cycles matching the §2.1 model, `Td` term
included, at every §6.2 sweep point in the interchanged order — the shipped
order returns with MB — affine in `PTA_TW`, with slope `PTA_WLOAD_CT`; and,
with the DMA counted, the A-row wait and `DMA_CT` that
[`pta_feed_model.py`](pta_feed_model.py) predicted before the tile existed.

**MB — the multi-bank tile.** `Nt · Kt` resident weight banks, and a loop order
selectable at launch that restores the `m`-outer FSM the interchange replaced,
so that EO-res is measured rather than modelled (§8 item 3).
*Gate:* EO-res totals match §2.1 in both orders, with `Tw` set to the RTL's
bank-select cycles, and C is bit-identical between orders; with the DMA
counted, they match the feed model's predictions for both orders.

**C3 — calibration.** Per-column affine correction, calibration FSM,
four schedulers, and the `cal_busy` dispatch guard.
*Gate:* with drift enabled at a stated rate, accuracy recovers to within a
stated margin of the no-drift case, and `PTA_CAL_CYC` shows the shadow
scheduler costing measurably less wall-clock than the periodic one at equal
accuracy. *Ablation:* the START-during-calibration regression from §3.2.
*Met, 2026-09-23.* C3(a) measured the correction on the D3 network — a tile at
chance after 46 hours of TFLN drift comes back to within 0.01 points of the
no-drift case, against the 0.2 the gate allows — and C3(b) built the RTL: the
two correction paths in PTM-C, the engine and its four schedulers in
`c930/rtl/pta/c930_pta_cal.sv`, and the guard. Gates P7 to P9 of grx930's design
note §5, and [`pta_chiplet_calibration.md`](pta_chiplet_calibration.md) §8 has
the numbers. The affine has no estimator: the error model has no per-column
gain or offset error for one to find, so the store is host-written and the
engine's one loop is the cell trim.

**C4 — SoC, firmware, numbers.** CSR decode widening, firmware, the full-SoC
test suite, and a Vivado run on the Arty A7-200T. *Board note, 2026-09-22:* on
the development board the same register block is reached as MMIO behind the
GPU's CXL.io ([`board_program_plan.md`](board_program_plan.md), B7 and X4).
Whether the c930 keeps a tile of its own is that plan's §8, question 3.
*Gate:* real utilization and timing, not estimates; the five existing full-SoC
tests still pass; the §6.2 sweep produced at both ends, thermo-optic and
Pockels-class, with EO-res measured on MB's tile rather than modelled.

### 6.1 Does it fit?

The 200T baseline is 78,750 / 134,600 LUTs (58.5%), 27,097 FFs (10.1%), 8 / 365
BRAM, 148 / 740 DSP. Estimated PTA additions:

| Block | LUTs | DSP | BRAM |
|---|---|---|---|
| Quantizers (in + out) | ~1,300 | 0 | 0 |
| Broadside MVM (replaces PE multipliers) | ~0 net | ~0 net | 0 |
| Noise generation, 8 columns | ~2,000 | 8 | 0 |
| Drift, 64 weights | ~2,600 | 0 | 0 |
| Calibration FSM + pattern ROM + LMS update | ~2,000 | 2 | 1 |
| `m`-inner accumulator (reuses `c_mem`) | ~200 | 0 | 0 |
| Widened CSR decode + PTA registers | ~800 | 0 | 0 |
| **Total** | **~9,000** | **~10** | **~1** |

Against roughly 55,850 LUTs, 592 DSPs and 357 BRAMs of headroom. It fits with
room to spare, on hardware already on the desk. Timing is the risk, not area —
the shot path adds a multiply, a square-root approximation and two adds where
the systolic PE had one registered product, and the array feed logic already
carries comments about paths that had to be broken by registration.

One cost the table leaves out: the baseline is the SoC as synthesized, with
`MAX_M` 8 and `MAX_K` 16, and the §6.2 shape needs `MAX_M` 64 and `MAX_K` 256.
A's storage grows 128-fold, `MAX_M · MAX_K` elements in each bank and again in
PF2's staging buffer, and one GEMM's operands take 20 KB of the 64 KB DDR window
the crossbar decodes, so a second operand set does not fit beside the firmware.
F0 ran that shape only on a resized Verilator build (program plan §3.3); C4 has
to size and price the synthesized SoC for it.

### 6.2 The cycle-time honesty note

The FPGA runs the tile at ~100 MHz, 10 ns per cycle. A real photonic shot is
1–5 ns — *sub-cycle* — a thermo-optic weight program is 10 µs to 1 ms, and an
electro-optic one is tens of picoseconds plus whatever delivers the weights. So
the emulation cannot represent absolute photonic throughput at all, and must
not be reported as doing so.

What it represents exactly is the **ratio**. Set `PTA_TS` to its floor and
`PTA_TW` to the tile's settle × `f_fpga` — at 10 µs and 100 MHz, 1,000 cycles —
and the thermal range of that ratio, up to about 10⁵, is reachable by sweeping
one register. The sweep is the experiment; the absolute cycle counts are not a
claim about anything.

**The bottom of the range is not reachable, although this note used to say it
was.** `PTA_TW` counts from a program's last weight write, on top of the core's
own 64-cycle scan, so with `PTA_TS` at its floor the smallest `Tw`-to-`Ts` ratio
a two-bank tile can show is 64, not 1. That floor is not an artifact to design
out. It is the host's cost of delivering weights, and it is exactly what a
scanned Pockels-class tile runs into (§2.1). So the sweep names four points and
sweeps between them:

| Point | Models | `PTA_TW` | `PTA_TS` | `Tw` | As shipped | Interchanged | Faster order |
|---|---|---|---|---|---|---|---|
| TO-1ms | thermo-optic, 1 ms settle | 100,000 | 5 | 100,064 | 205 M | 3.23 M | interchange, 63× |
| TO-10µs | thermo-optic, 10 µs settle | 1,000 | 5 | 1,064 | 2.19 M | 60.7 k | interchange, 36× |
| EO-scan | Pockels-class (TFLT, §4.4), scanned | 0 | 1 | 64 | 134 k | 20.5 k | interchange, 6.5× |
| EO-res | Pockels-class (TFLT, §4.4), resident | 0 | 1 | 0 | 2,560 | 18.4 k | as shipped, 7.2× |

Cycles as §2.1 predicts them at `M = 64, N = 8, K = 256`, `Td` = 8, restore
folded; at 10 ns a cycle, TO-10µs, EO-scan and EO-res are §2.1's table.
Sweeping `PTA_TW` between the points is what checks C2's affine claim, but no
value of it crosses the §2.1 break-even: that sits at 8 cycles, and a two-bank
tile's `Tw` never drops below the scan's 64.

Three rules hold at the Pockels-class end:

- **EO-res is a model evaluation until a multi-bank tile exists.** It needs
  `Nt · Kt` resident banks and the tile has two (§8 item 3). Report it as the
  §2.1 formula with the scan removed, labelled as such.
- **`DMA_CT` is part of the result.** A GEMM of a few thousand cycles is no
  longer long beside the DMA fetch of its 18,432 operands, so the fetch cannot
  be subtracted as overhead at this end.
- **Dilation is a claim about the host.** Raising `PTA_TS` above its floor to
  stretch a sub-cycle shot is the same as assuming a host that many times
  faster than the FPGA, because the scan, the row write and the fetch keep
  their real cycle counts. That is a legitimate experiment, but a different
  one, and it is labelled as that one.

---

## 7. What this does to GRXCP

Two of this repository's own rules are implicated, and both need an answer
before phase C1 lands, not after.

**"Every sanctioned emulation is reported through a device property"**
(`AGENTS.md` §3). A c930 running a PTA tile is not computing the GEMM the
caller asked for; it is computing a noisy approximation of it. That is exactly
the class of thing `warpShuffleIsEmulated` exists for. It needs a property —
`gemmIsAnalogEmulated`, or better a small struct carrying the effective bits
and the seed — and `grx-smi` should print it in the same "software stand-ins in
effect" section that already exists for this purpose. The program plan settled
the shape (D2): a struct carrying the effective activation, weight and ADC bits
(`PTA_BITS`), the seed (`PTA_SEED`) and the impairment mask (`PTA_IMPAIR`), not
a flag.

**"Every library kernel has a CPU reference and a numerical gate (bitwise or
ULP-bounded); 'close enough' is not a gate"** (`AGENTS.md` §4). An analog
device breaks this as written, and the temptation will be to loosen the rule
into a tolerance, which destroys it.

The resolution that keeps the rule intact is two gates rather than one loose
one:

1. **Bitwise against the model.** The model is deterministic given
   `PTA_SEED`, so `grxblasGemmEx` on a PTA-enabled c930 has an exact expected
   answer and the existing gate applies unchanged. This is the gate that
   catches bugs.
2. **Distributional against fp32.** Separately, and reported rather than
   gated at zero tolerance: error distribution versus the fp32 reference, with
   a stated bound that moves as the error model is tuned. This is the number
   that goes in the paper.

Conflating them produces a gate that is neither. Keeping them apart means a
model change moves gate 2 visibly and leaves gate 1 red until the golden data
is regenerated as a reviewed step — which is precisely how
`ci/check_perf.py` already treats baselines.

`heterogeneous_devices.md` §6 asks which engine `grxblasGemmEx` picks when both
can do the work, and answers "current-device-only, because it makes the choice
the caller's." A PTA c930 sharpens that: the two devices no longer compute the
same function, so automatic selection would silently change a program's
numerics. Current-device-only stops being the conservative default and becomes
the only defensible one.

---

## 8. Proposed but not yet implemented

Recorded so the next reader knows what was considered and deliberately deferred.

1. **An embedded control core inside the tile.** The source analysis suggests
   replacing the calibration FSM with a RISC-V core so calibration policy is
   software. Correct for a product, wrong for this phase: it puts a second
   toolchain and a second binary between a change and a measurement, and the
   calibration algorithms are small. Revisit if §5.1's scheduler comparison
   turns into a search over many policies.
2. **CXL, coherent access, an LSU replacing the DMA.** A large share of the
   source analysis. It is orthogonal to everything measurable here: the c930's
   AXI4 master already feeds the tile faster than a thermally-limited weight
   bank can consume, so coherence changes no result this program can produce.
   That holds at the thermal end of the §6.2 sweep only. At the Pockels-class
   end — where the TFLT target sits (§4.4) — the host's feed is the binding
   cost (§2.1), so this item reopens as soon as those points are reported, with
   their `DMA_CT` as the evidence. It belongs to the GRXIConnect work, not here.
3. **Multiple weight banks beyond two.** §5.5's hypothesis may make weight-set
   capacity the interesting axis, in which case `N` banks and a bank-allocation
   policy is the follow-on. Not before the `Tw`-to-`Ts` ratio sweep says so.
   The exception is §6.2's EO-res point, which cannot be measured at all
   without `Nt · Kt` banks. With TFLT the target (§4.4), the Pockels-class end
   is the one that matters, so that part is not deferred: measuring the
   target's resident point waits on the multi-bank tile, step MB of §6.
4. **A second tile.** The grx930 team's notes observe 41.5% LUT headroom is
   "enough for a second NPU tile." Two tiles with independent weight sets is
   how a real machine hides `Tw` completely, and it is the obvious C5. It is
   deferred because a two-tile result is uninterpretable until the one-tile
   loop nest is right.
5. **Crosstalk topology.** The model couples nearest-neighbour inputs within
   each output's bank, which is what a linear microring bank looks like (§4.3).
   An MZI mesh couples along its triangular structure instead, and the two are
   not the same matrix. Since no mesh is being built, the topology is a
   parameter with no ground truth; the model should carry a pluggable coupling
   matrix and the document should keep saying it is a hypothesis.

---

## 9. Open questions

1. **Does the interchange survive real B-tile reuse patterns?** The §2.1 model
   assumes `M` rows all use the same B tile, which holds for a dense GEMM and
   not for a batched or strided one. The DMA's address generation may need a
   second traversal order, and that is not in the C2 estimate.
2. **Where does the activation scale factor come from?** §5.2 makes it a
   software quantity, but nothing in the c930 firmware computes one today. Is
   it static (from the model), measured (a calibration pass), or adaptive (from
   `PTA_SAT_CT`)? The adaptive answer is the most interesting and the one most
   likely to oscillate.
3. **Is bitwise RTL↔C parity actually achievable through the DPI wrapper?**
   The NPU DPI path exists for grxcp backend testing, but nothing currently
   crosses it that is sensitive to bit-level arithmetic ordering. If the
   floating-point accumulator paths cannot be made to agree exactly, gate C1(b)
   has to weaken, and it should weaken deliberately rather than by discovery.
   *Partly answered by C1:* the error model is integer, so no floating-point
   ordering reaches the parity gate, and parity holds bitwise in the core
   harness. The DPI wrapper cannot reach the configuration until C4 maps it
   onto a CSR, so the wrapper half of the question waits for C4.
4. **What is the right small model for the accuracy sweep?** It needs
   published quantization curves to compare against, small enough to run in
   RTL simulation, and dominated by GEMM. A two-layer MLP on MNIST is
   defensible and boring; anything transformer-shaped will not fit the
   `MAX_N = 8` output width without tiling that muddies the measurement.
   *Settled by the program plan (D3):* that MLP, fixed once so that C1, C3 and
   A3 report on the same network. C1 fixed it as 784-100-10 with ReLU, compared
   with Gorsline, Smith and Merkel's Fig. 3(c); C3 and A3 take its five
   networks at 8-bit operands and 6-bit weights.
5. **Which weight drive would a real TFLT tile have?** §2.1 shows the loop nest
   turns on it — scanned wants the interchange, resident wants the shipped
   order — and nothing in an emulation-only program can find out. Until
   something does, the sweep carries both points and claims neither.
