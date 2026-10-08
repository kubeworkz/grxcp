# Calibrating the PTA chiplet

**Companions:** [`board_program_plan.md`](board_program_plan.md) (X3, and B4's
split), [`pta_chiplet_regmap.md`](pta_chiplet_regmap.md),
[`pta_cpu_integration.md`](pta_cpu_integration.md) §5.1 and its C3, and
grx930's `c930/doc/pta_error_model_design_note.md` §4, which defines the error
this corrects.

**Status: X3 of the board plan, drafted 2026-09-22, measured the same day in
C3(a), and built on 2026-09-23 in C3(b)** — §8 has both sets of numbers. Four
claims in these pages did not survive being measured or built: one in §4, which
C3(a) corrected, and three in §5, which C3(b) did. The CPU document's C3 —
per-column affine correction, a calibration FSM, four schedulers and the
`cal_busy` guard — specified for a chiplet behind a link. Two things change
there, and both are in this document's favour: the tile has more idle time, not
less, and the correction has somewhere better to go.

---

## 1. What it is for, in numbers

C1 measured drift's cost on the D3 network, and X1 measured it beside
everything else. At TFLT's fitted rate, an hour of drift costs about 0.44
points inside version 1's budget — 0.81 points with drift against 0.37 with
almost none. At TFLN's rate the same hour costs far more: C1 put a point of
loss at six minutes. Left alone for four hours, TFLT's fit costs 4.8 points and
TFLN's is already at chance.

*Those are the c930 core's 8 × 8 tile, 2026-10-04.* Every figure in this
document was measured on it, because it was the only tile grx930's harness had.
The harness now takes the tile as an option, and on a 256 × 64 tile the same
drift costs far less (the board plan's §4.3): an hour of TFLT's is 0.04 points
beyond version 1 and not 0.55, four hours 0.77 and not 5.18, an hour of TFLN's
1.9 and not 22.6. Drift's error is 2.8 to 3.2 times smaller at every age.

The reason is how a small tile is used. All 78,400 weights of D3's first layer
pass through the same 64 cells, and a cell's drift is the same error on every
one of them; at 256 × 64 a cell carries 8. **And the model draws every cell's
drift independently.** A real tile whose neighbouring cells drift together gains
less from being large, and nothing measured says how much less. So this
document's intervals are not relaxed. What changes is what they are: sized on
the worst tile there is, and on the safe side for any other.

*And they are a modulator's, 2026-10-05.* The drift fits behind every interval
here are a Mach-Zehnder's bias drifting (grx930's design note, §7). At the
working geometry the weight cells are rings (the board plan's B10 and §8,
question 1), and a ring turns the same change of index into a weight error in
proportion to its Q: a weight's LSB is 0.42 pm of resonance on a ring that
needs 5 V and 0.084 on one that needs 1 V
([`pta_ring.py`](pta_ring.py)). Nothing has refitted drift for a ring. Until
something does, these intervals describe a tile of modulators.

*And they are MNIST's, 2026-10-05.* Every cost of drift above is on D3
trained on MNIST. On two more data sets (the board plan's §8, question 7;
[`pta_workload.py`](pta_workload.py)) the same drift costs more. On the
working tile's 128 × 64, in points:

| What TFLT's drift adds to version 1 | MNIST | Fashion-MNIST | MNIST, inverted |
|---|---|---|---|
| Six minutes | −0.01 | +0.08 | +0.19 |
| An hour | +0.17 | +0.68 | +1.95 |
| Four hours | +0.40 | +2.34 | +16.5 |
| An hour, then calibrated | −0.13 | +0.02 | −0.09 |

Calibration undoes all of it on all three, so the method stands. The interval
does not carry. An hour costs a sixth of a point on MNIST and two points on a
workload that lights three rows in four, because a drifted weight is wrong by
what it is lit with. An interval is a workload's as much as a material's.

*And they are version 1's, 2026-10-06.* The board plan's B14 holds the
interface chip to version 2: an 8-bit ADC, and half of each noise row. Drift
was not rerun at it. Version 2 loses half what version 1 does before any
drift. If the same drift costs what it did, it is a larger share of what is
left, and whether it does, and so whether an interval that held at version 1
holds at version 2, is not known. One thing
moves the right way: a probe is read against the receiver's noise, which is a
quarter of an 8-bit ADC's LSB where §3 has it at half, so the same averaging
measures a cell twice as finely.

*It was rerun the same day* ([`pta_version2.py`](pta_version2.py); the board
plan's §4.3, at the end of its budget). The same drift does cost what it did.
What TFLT's drift adds to each version, in points, on the working tile:

| | MNIST, version 1 | Version 2 | Fashion-MNIST, version 1 | Version 2 | MNIST inverted, version 1 | Version 2 |
|---|---|---|---|---|---|---|
| As budgeted, points lost | 0.34 | 0.15 | 1.16 | 0.54 | 1.23 | 0.62 |
| Six minutes adds | −0.01 ± 0.02 | −0.06 ± 0.04 | 0.08 ± 0.10 | 0.08 ± 0.06 | 0.20 ± 0.26 | 0.28 ± 0.27 |
| An hour | 0.17 ± 0.05 | 0.12 ± 0.05 | 0.69 ± 0.33 | 0.67 ± 0.31 | 1.95 ± 1.29 | 1.99 ± 1.26 |
| Four hours | 0.40 ± 0.08 | 0.37 ± 0.08 | 2.34 ± 0.43 | 2.36 ± 0.57 | 16.54 ± 3.01 | 16.07 ± 3.09 |
| An hour, then calibrated | −0.13 ± 0.03 | −0.06 ± 0.03 | 0.03 ± 0.15 | 0.02 ± 0.11 | −0.09 ± 0.08 | −0.01 ± 0.05 |

So the method stands at version 2: calibration returns it to its budget on all
three. The interval does not. An hour's drift is 80% of version 2's budget on
MNIST, 124% on Fashion-MNIST and three times it on the inverted set, where it
was 50%, 59% and 159% of version 1's. An interval that keeps drift inside a
tenth of a point is six minutes on the first two and under that on the third.
Which interval version 2 is held to is the board plan's to decide, and it has
not.

*It has since: B15, every six minutes, the same day.* With a source at the
plan's B16 rows as well, version 2 at the end of such an interval loses 0.19,
0.60 and 1.09 points on the three data sets: 0.04, 0.06 and 0.48 over its
budget.

So the question is not whether to calibrate but how often, how long it takes,
and what it can actually undo.

## 2. Two error classes, two loops

The error model gives every cell its own programming error and its own drift,
and gives the column its ADC, its receiver and its share of the laser. A
per-column affine correction, which is what C3 specifies, can only fix what is
common to a column.

| Loop | Corrects | Cannot correct |
|---|---|---|
| **Column affine** — a gain and an offset per column, `PTA_GAIN[j]` and `PTA_OFFS[j]` | Receiver gain and offset, ADC offset, laser power drift, anything common to the column | Per-cell programming error and per-cell drift, which is most of what C1 measured |
| **Cell trim** — a correction per weight cell, applied when the weight is written | Each cell's programming error and accumulated drift | Anything that changes faster than the weight is rewritten |

The chiplet's weights sit in DAC-held voltages (B5), so it can have the second,
and that is the one that addresses drift. On the c930 the tile is a model and its
weight DAC is a register, so C3(b) built both there.

*What C3(b) settled.* Both stores are built and the tile applies both, but only
the cell loop has an estimator. The error model has no per-column gain error and
no per-column offset error — nothing in grx930's contract §4 is common to a
column — so a column loop run against it would be measuring its own receiver
noise and would make the tile worse, not better. The affine is therefore
host-written, exercised by directed cases in gate P7, and waiting for an error
class the model does not yet emulate. That is a gap in the error model, not in
the engine, and it is the one thing in this document that silicon will need and
the twin cannot yet give.

## 3. Measuring

**Cell trim, by one-hot probes.** Drive one row at full scale with every other
row at zero, and each column reports that row's cell. One shot measures a whole
row of cells across all `n` columns at once, so `k` shots measure the tile, and
`k · m` shots measure it with `m`-fold averaging against the receiver noise X1
budgets at half an 8-bit ADC LSB (a quarter of its 7-bit ADC's, which is how
§8's runs set it).

At a 256-row tile with 16-fold averaging that is 4,096 shots: 4 µs at one shot
a nanosecond, for one bank, 8 µs for both. (Half of each at the 128 rows of
the working geometry, since the board plan revised B10 on 2026-10-05.) Against an hourly interval that is a
duty cycle of about two parts in a billion. **Calibration's cost on this chiplet
is not its shots.** It is the interruption: draining the tile, rewriting the
weights and restarting.

*At six minutes, since the board plan's B15 (2026-10-06).* On the working
tile's 128 rows both banks are 4,096 shots, 4 µs, and against six minutes
that is one part in 88 million. It is still not the shots. It is 240
interruptions a day where an hour was 24, and what one costs is still not
priced.

*Priced, 2026-10-08* ([`pta_interruption.py`](pta_interruption.py)). Counted,
and not measured. The twin holds the tile for `passes × repeats × (PTA_TW +
rows × PTA_TS)` cycles a bank ([`pta_chiplet_regmap.md`](pta_chiplet_regmap.md)
§5), and on the working tile a shot is a beat and a bank programs in 64,
the board plan's write path being 128 cells a beat. Both banks at 16 probes
are 6.1 µs at one pass and 18.4 µs at three: 12,288 shots, and the 96
zeroings before them, which are half as much again as the shots and which
the paragraph above did not count. With the estimator's walk, the restore
and the drain it is 68 µs at the most, one part in 5.3 million of six
minutes, and 16 ms a day. For 240 a day to take a thousandth of the tile's
time an interruption would have to last 0.36 s. **So on this chiplet the
interruption is not the cost either.** Nothing that can be counted is, at six
minutes or at three, and what sets the interval is what drift costs.

**Column affine, by two references.** A zero-input shot gives the column's
offset; a known full-scale pattern gives its gain. A handful of shots, taken
far more often than the cell loop.

## 4. Correcting, and what the DAC needs

The cell trim has to be applied where the weight is held, which is the weight
DAC. That imposes a requirement X1's budget does not yet carry:

> **The weight DAC wants resolution below the weight code's LSB.** Weights are
> 6-bit (X1), and a trim held on that same grid can only move a cell by whole
> codes. Enough bits below to resolve a quarter of a code is the comfortable
> figure, which makes an 8-bit DAC behind a 6-bit weight code.

*What C3(a) measured, and where this section was wrong.* An earlier draft said
that without those bits the per-cell drift stays uncorrected, and called it the
most important sentence here. It is not true. Trimming on the bare 6-bit grid
still recovers almost everything, because drift is many codes wide and a whole
code is a fine enough step to chase it. §8 has the figures: a quarter of a code
costs nothing measurable, half a code 0.04 points, a whole code 0.16, and two
codes 1.05. Sub-code resolution is worth buying; it is not a precondition.

The column affine is applied digitally, after the ADC, as the CPU document
intends. Both corrections saturate rather than wrap, and a cell whose trim
cannot reach its measured error raises `PTA_STATUS.DRIFT_ALARM` — the tile is
telling the host it needs a weight rewrite or a service call.

## 5. Scheduling

The CPU document's four modes stay, selected by `PTA_CTRL[6:4]`: off, periodic
(`PTA_CAL_PER`), drift-predictive (extrapolate the last two residuals, fire at
`PTA_CAL_THR`), and shadow. What changes is what shadow means.

On the c930 the shadow is a DMA stall: calibrate while the memory system is
fetching. On the chiplet, X2 found that the link, not the optics, sets the shot
rate — one x16 module holds a 256 × 64 tile to 0.22 G shots a second where the
tile could run faster. **The idle windows are therefore larger and more
frequent than on the c930, and they are visible locally**: the chiplet watches
its own activation buffer draining rather than a host counter it cannot see.
The shadow scheduler fires when the input buffer falls below a watermark and
the predicted error is above a floor, with the predictive threshold as the
backstop the CPU document specifies.

`PTA_CAL_CT` and `PTA_CAL_CYC` are what make the four modes comparable rather
than arguable. `PTA_CAL_CYC` is 64 bits on this map. `PTA_CAL_CT` is not — this
sentence said both were until X5 built the map, which has an upper half for the
cycles and none for the count — and four billion calibrations is enough.

*What C3(b) built, and what it needed that this section did not say.* The four
modes are in `grx930/c930/rtl/pta/c930_pta_cal.sv`, with the c930's window being
a row the DMA has not landed yet. What building them settled:

- **It has to extrapolate what a calibration *found*, not what it left.** This
  section and the CPU document's §5.1 both say to extrapolate the last two
  *residuals*. A residual is what a calibration leaves behind, and one that
  worked leaves almost nothing — so the rate it implies is almost zero and the
  scheduler stops scheduling. P9 measured exactly that: one calibration in fifty
  thousand cycles where the periodic scheduler took four, and no recovery to show
  for it. The quantity the scheduler needs is the widest correction the first
  pass had to make, before any of it was applied, which is the error that
  accumulated over the interval. The engine publishes both, and they answer
  different questions: `PTA_ERR_FOUND` is what accumulated and drives the
  scheduler, `PTA_ERR_MAX` is what is left and is what this document's gate
  reports.
- **The predictive rule is a comparison of products.** With `r1` what the last
  calibration found over `L1` cycles, and `r0`, `L0` the pair before it, the
  error predicted after `E` cycles is `E · max(r1/L1, r0/L0)`, and the scheduler
  fires when it reaches `PTA_CAL_THR`. Compared as `E·r ≥ thr·L` there is no
  divider on the tile.
- **Zero counts as one unit**, because a rate of zero predicts no error however
  long the wait and would switch the scheduler off permanently the first time a
  calibration came back clean.
- **`PTA_CAL_PER` is a floor on the interval** for both predicting modes, not
  only a period for the periodic one. Without it the extrapolation runs away: the
  measurement has a noise floor that a short interval does not divide out, so a
  short interval reads as a steep rate, which fires again sooner, which shortens
  the interval further. With a grant point at every output row C3(b) watched it
  calibrate continuously and never finish the work. A prediction can now only ask
  for *fewer* calibrations than the periodic scheduler would take, which is also
  what makes the comparison in §8 a comparison of policy rather than of budget.
- **The shadow's floor is `PTA_CAL_THR/4`**, derived rather than given a register
  of its own, with the predictive rule as the backstop this section asks for. The
  request is *withdrawn* when the window closes, which is what keeps a shadow
  calibration inside one; a periodic request is held until the tile can take it,
  which is what makes it pay.

*The period, since the board plan's B15 (2026-10-06).* Version 2 is
calibrated every six minutes, so that is `PTA_CAL_PER` for the periodic
scheduler and the floor under the other two. It does not fit: 32 bits of
cycles is 4.3 seconds of a 1 GHz shot clock, and six minutes is 39 bits. The
register map records it as open (its §7, item 11). The two schedulers that
predict were built to take fewer calibrations than the period allows, and
neither has been run at version 2.

*Closed the same day.* The map counts `PTA_CAL_PER` in units of 2¹⁶ cycles on
the chiplet (its §4): 66 µs a unit at 1 GHz, six minutes is 5,493,164 of them,
and the word reaches 78 hours. On the c930 it is still the core's cycles, and
grx930's engine compares it so; a chiplet's engine would compare the cycles
since the last calibration, less their low sixteen bits.

## 6. The engine, and the contract

The FSM lives on the interface chip (B4) and owns: the probe sequence, the
estimator, the two correction stores, and the scheduler. Its obligations under
[`pta_chiplet_regmap.md`](pta_chiplet_regmap.md) §5:

- `PTA_STATUS.CAL_BUSY` is set for the whole of a calibration, and BUSY and
  CAL_BUSY are readable in one snapshot.
- Commands arriving during a calibration **queue**; they are never dropped and
  never dispatched into a tile that is unavailable. This is the c930's
  `cal_busy` dispatch guard, moved to the chiplet's own queue.
- A `MODEL_RST` during a calibration is refused, and raises
  `PTA_IRQ_STATUS.ERR`.
- Completion raises `PTA_IRQ_STATUS.CAL_DONE`, and `PTA_STATUS.CAL_VALID` says
  whether the result was used.
- The residual goes to `PTA_ERR_MAX`, which C3's gate reports, and the error the
  calibration found goes to `PTA_ERR_FOUND`, which is what the predictive
  scheduler reads. An earlier draft had the scheduler reading the residual; §5
  says why that does not work.

*Against C3(b), 2026-09-23.* On the c930 all of this is built except the
interrupt, which has nowhere to go until C4 maps the register block: the engine
raises a completion pulse and the core carries `CAL_BUSY`, `CAL_VALID`,
`DRIFT_ALARM`, `PTA_ERR_MAX`, `PTA_CAL_CT` and `PTA_CAL_CYC` as ports. Two
additions the list did not have. **A calibration must not report BUSY** — the
c930's `o_busy` excludes one that ran between GEMMs, because a calibration is not
a command and BUSY is per-command; without that the dispatch guard is never
exercised and `CAL_BUSY` means nothing. And **a START that reaches the tile
during a calibration anyway is reported**, not dropped and not taken: the guard
is what prevents it, and `PTA_IRQ_STATUS.ERR` is what says the guard failed. A
`MODEL_RST` during a calibration raises the same bit.

The engine also needs configuration no map had a place for — the probe's
amplitude, its repeat count, the DAC's step and clamp, and its own seed. Three
words, in [`pta_chiplet_regmap.md`](pta_chiplet_regmap.md) §4.

*Against X5's twin, 2026-10-04.* The five obligations above are built on the
chiplet's map and held by its gate: CAL_BUSY for the whole of a calibration and
in one word with BUSY; three commands during one queued at occupancy 1, 2 and 3
and run after it in order; a `MODEL_RST` during one refused with
`PTA_IRQ_STATUS.ERR`; `CAL_DONE` at its end and `CAL_VALID` only if its result
was used; and the residual and the error found in their two registers, equal to
what grx930's `pta_cal_bank()` returns. The twin's engine is that function. What
the twin adds is where BUSY stands while work waits behind a calibration
([`pta_chiplet_regmap.md`](pta_chiplet_regmap.md) §5).

## 7. The gate, for the chiplet

C3's gate is that accuracy recovers to within a stated margin of the no-drift
case, and that the shadow scheduler costs measurably less wall-clock than the
periodic one at equal accuracy. For the chiplet:

- **The margin is 0.2 points on the D3 network**, at `DIN_W` 8 and `B_w` 6,
  against the same configuration with drift off. The figure comes from X1: an
  hour of TFLT drift costs 0.44 points inside version 1's budget, and a
  correction that leaves more than half of that standing is not worth its
  silicon.
- **Both fits are run**, TFLT's and TFLN's, as the PTA plan's C3 row says.
- **The comparison of schedulers** runs on the twin first, where wall-clock is
  countable exactly, and then in RTL. *On the c930 those are the same thing* —
  PTM-C is the twin — so C3(b) ran it once, in RTL, and §8 has it. The chiplet's
  own twin is X5's. *X5 built it on 2026-10-04 with the scheduler off:* a
  calibration runs when `CAL_NOW` asks. So this comparison is still owed. It
  needs the three scheduled modes, and for the shadow an idle window the twin can
  see, which waits on what link 2 carries.
- *Ablation:* the START-during-calibration regression from the CPU document's
  §3.2, which on this chiplet means commands arriving while CAL_BUSY is set —
  three back to back, with the queue's occupancy checked at each step.

## 8. What C3(a) measured

The C reference gained both corrections on 2026-09-22 — a trim per cell and an
affine per column — and `sim/pta_mnist.sh calib` in grx930 ran the probe this
document specifies. Every figure below is five networks at `DIN_W` 8 and
`B_w` 6, at version 1's settings from the plan's §4.3, each network on its own
seed. **Version 1 with no drift reads 97.20**, and that is what the margin is
measured against.

**Recovery is complete, at both fits and every age.**

| Drift | Uncalibrated | Calibrated |
|---|---|---|
| TFLT, 1 hour | 96.64 | 97.23 |
| TFLT, 4 hours | 92.27 | 97.24 |
| TFLT, 46 hours | 31.91 | 97.21 |
| TFLN, 1 hour | 74.83 | 97.19 |
| TFLN, 4 hours | 40.79 | 97.21 |
| TFLN, 46 hours | 10.81 | 97.20 |

The worst calibrated figure is 0.01 points off the no-drift baseline, against
the 0.2 the gate allows. A tile at chance — TFLN after 46 hours — comes back
whole, and its ADC saturations fall from 3.3 per thousand elements to 1.4 per
hundred thousand, which is the baseline's own rate.

**Four probes are enough.** At TFLT's four-hour point: one probe gives 97.00,
four give 97.25, sixteen give 97.24. The averaging is there to beat the
programming error redrawn at every weight write, and four draws beat it.

**The trim's resolution, measured at TFLN's 46 hours**, the hardest case. Steps
are in 8-bit weight LSB, and a 6-bit weight code's LSB is four of them:

| Trim step | ¼ code | ½ code | 1 code | 2 codes |
|---|---|---|---|---|
| Accuracy | 97.20 | 97.16 | 97.04 | 96.15 |

**How long a calibration holds**, at TFLT's fit, calibrating at zero and then
ageing:

| After | 15 min | 30 min | 1 hour | 4 hours | 46 hours |
|---|---|---|---|---|---|
| Accuracy | 97.06 | 96.82 | 96.56 | 92.15 | 32.03 |

So the interval that holds the gate's 0.2 points is **about a quarter of an
hour** at TFLT's fitted drift, not the hour §1 assumed from C1's sweep. At
TFLN's it will be far shorter, and that is what the schedulers are for.

### What C3(b) built, and what it measured, 2026-09-23

The two correction paths are in grx930's `c930/rtl/pta/c930_ptm_c.sv`, the engine
in `c930/rtl/pta/c930_pta_cal.sv`, and the dispatch guard in
`c930/rtl/c930_npu_csr.sv`; the gates are P7 to P9 of that repository's design
note §5, at `DIN_W` 8 and 16.

**The engine writes what the C reference writes, cell for cell.** Drift and
programming error accumulated over three GEMMs; one calibration of four repeats
over three passes took 7,321 cycles; every trim it wrote matched
`pta_cal_bank()`, checked by reading the tile back one cell at a time where a
single disagreement shows, and so did both numbers it publishes — 3,584 found and
4,608 left, Q.8 weight LSB. C1's fourteen shapes then matched the model again
with those trims in place. Parity is only possible because a calibration's noise
is reproducible: the streams load once from a seed of its own and then run
through every repeat and pass, which is also what makes the repeats differ.

**Against drift alone the recovery is total.** A trim cannot anticipate a
programming error redrawn at every weight write, so the measurement of what a
trim is *for* turns that off: the tile read back a cell at a time goes from a mean
of 408 to **0.00** at the ADC's resolution, 3,328 found and 256 left. With the
programming error on, the same calibration halves the error and no more — that is
the floor the redraw sets, not the trim's limit, and it is why §3's averaging is
there.

**The scheduling, in wall-clock.** Four GEMMs whose A rows arrive every 1,200
cycles against rows that take about 80 to compute, so the tile waits on its
operands — which is what X2 says the link does to this tile anyway, and what makes
an idle window long enough to hide a calibration in. Every mode ran the same work
with the same operands and the same arrivals, and calibration was switched off
for the read-back so the measurement is of the tile, not of the scheduler:

| Scheduler | Wall-clock cycles | Calibrations | Cycles calibrating | Last found | Mean \|cell\| left |
|---|---|---|---|---|---|
| off | 48,801 | 0 | 0 | 0 | 340.00 |
| periodic | 57,593 | 4 | 8,540 | 2,304 | 172.00 |
| drift-predictive | 53,197 | 2 | 4,270 | 3,584 | 284.00 |
| shadow | 50,999 | 4 | 8,540 | 2,816 | 76.00 |

The periodic scheduler and the shadow one ran the same number of calibrations
and spent the same 8,540 cycles inside them, and that is the whole of the
difference: periodic cost 8,792 cycles of wall-clock where the shadow cost
2,198, so **75% of the calibration was free** — hidden in stalls the tile was
waiting through anyway. It was also the more accurate of the two, 76.00 against
172.00 per cell against 340.00 uncalibrated, because it calibrated at moments
the tile was not using.

**What C3(b) had to correct.** Three things in this document and the CPU
document's §5.1 did not survive being built, and §5 has them: the predictive
scheduler must extrapolate what a calibration found rather than what it left, a
found error of zero must count as one unit, and `PTA_CAL_PER` must floor the
interval or the extrapolation runs away. A fourth was in the RTL rather than the
specification: `CAL_BUSY` has to cover the handover back to the core, not just
the work, or there is one cycle in which a dispatcher believes the tile is free
and the command it sends is lost — the guard's own failure, one cycle wide, found
by the only scheduler that fires between GEMMs.

**What is still unmeasured.** The affine loop corrects error classes the model
does not emulate — there is no per-column gain error in the contract — so it is
implemented for the RTL to match and exercised only by directed cases.
Crosstalk is left at version 1's 2% and not corrected: undoing it means solving
a tridiagonal system per column, which is a different piece of work. And all of
this is the C reference; the RTL, the FSM and the schedulers are C3(b).

## 9. Open

1. ~~The averaging depth `m`.~~ **Four**, measured in §8, at X1's budgeted
   receiver noise. A noisier receiver moves it.
2. **How often the cell loop must run against the column loop.** §8 dates the
   cell loop at about every quarter hour under TFLT's fit; the column loop's
   own interval needs an error class the model does not yet emulate.
3. **The watermark** the shadow scheduler fires on, which follows from the
   chiplet's buffer sizes (X2) and the link's round trip.
4. **Whether a failed trim is recoverable in the field** — the DRIFT_ALARM path
   assumes someone is listening.
5. **What the source does, which neither loop was specified for** (2026-10-05;
   [`pta_source_noise.py`](pta_source_noise.py), the board plan's §4.3). Since
   B12 the tile is lit by a comb, and a comb's power can move three ways. *All
   lines together, slowly:* that is a gain on every column at once, which is
   the column loop's to take out, and is a candidate for the error class item
   2 says the model lacks. *All lines together, shot to shot:* no loop reaches
   it, and through a balanced pair 2% rms is the budget's row for it.
   *Line by line:* a line that is not level puts one error on every weight in
   its row, and a column's gain is one number for all its rows, so the column
   loop cannot see it. Nor can the cell loop as built: grx930's engine probes
   each cell through a weight of zero, which a line's power multiplies. So a
   line's level is unmeasured. A probe with weights in it would read it.

   *Since the board plan's B16 (2026-10-06) the shot-to-shot row is 1% for
   version 2, and no loop reaches it still.*
6. ~~**What an interruption costs**~~ (2026-10-06). B15 takes 240 calibrations a
   day. §3 says the cost of one is draining the tile, rewriting the weights
   and restarting, and no model prices that. Until one does, six minutes is a
   figure for accuracy with no figure for throughput beside it.
   **18 to 68 µs, counted** (2026-10-08;
   [`pta_interruption.py`](pta_interruption.py), §3). The figure for throughput
   is one part in 5.3 million at the most. What it leaves open is three
   things: the estimator's width, which is most of the 68 µs and is
   specified nowhere; what the light does to a ring written to zero and back,
   which no model here has; and the schedulers of §5, whose reason on the c930
   was a calibration's cycles and which have at most half a large layer's
   time to hide on this tile.
