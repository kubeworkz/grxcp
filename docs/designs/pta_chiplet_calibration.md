# Calibrating the PTA chiplet

**Companions:** [`board_program_plan.md`](board_program_plan.md) (X3, and B4's
split), [`pta_chiplet_regmap.md`](pta_chiplet_regmap.md),
[`pta_cpu_integration.md`](pta_cpu_integration.md) §5.1 and its C3, and
grx930's `c930/doc/pta_error_model_design_note.md` §4, which defines the error
this corrects.

**Status: X3 of the board plan, drafted 2026-09-22.** The CPU document's C3 —
per-column affine correction, a calibration FSM, three schedulers and the
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

C3 on the c930 has the first. A chiplet whose weights sit in DAC-held voltages
(B5) can have the second, and it is the one that addresses drift.

## 3. Measuring

**Cell trim, by one-hot probes.** Drive one row at full scale with every other
row at zero, and each column reports that row's cell. One shot measures a whole
row of cells across all `n` columns at once, so `k` shots measure the tile, and
`k · m` shots measure it with `m`-fold averaging against the receiver noise X1
budgets at a quarter of an 8-bit ADC LSB.

At a 256-row tile with 16-fold averaging that is 4,096 shots: 4 µs at one shot
a nanosecond, for one bank, 8 µs for both. Against an hourly interval that is a
duty cycle of about two parts in a billion. **Calibration's cost on this chiplet
is not its shots.** It is the interruption: draining the tile, rewriting the
weights and restarting.

**Column affine, by two references.** A zero-input shot gives the column's
offset; a known full-scale pattern gives its gain. A handful of shots, taken
far more often than the cell loop.

## 4. Correcting, and what the DAC needs

The cell trim has to be applied where the weight is held, which is the weight
DAC. That imposes a requirement X1's budget does not yet carry:

> **The weight DAC needs resolution below the weight code's LSB.** Weights are
> 6-bit (X1); trimming a cell's drift on that same grid would snap the
> correction to a whole weight LSB and undo the point of it. Enough bits below
> to resolve a quarter of a weight LSB is the starting figure, which makes an
> 8-bit DAC behind a 6-bit weight code.

Without those bits, the only correction available is the column affine, and the
per-cell drift C1 measured stays uncorrected. That is the single most important
sentence in this document.

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

`PTA_CAL_CT` and `PTA_CAL_CYC`, 64-bit on this map, are what make the four
modes comparable rather than arguable.

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
- The residual goes to `PTA_ERR_MAX`, which the predictive scheduler reads and
  C3's gate reports.

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
  countable exactly, and then in RTL.
- *Ablation:* the START-during-calibration regression from the CPU document's
  §3.2, which on this chiplet means commands arriving while CAL_BUSY is set —
  three back to back, with the queue's occupancy checked at each step.

## 8. What has to exist before the margin can be measured

The C reference implements the error, not its correction: there is no trim path
and no per-column affine in `pta_tile_model.c` today, so **no recovery number
exists yet, and this document does not report one**. C3 adds both, in grx930,
and the harness that measured drift's cost — `sim/pta_mnist.sh`, with the
`joint` phase X1 added — is what then measures the recovery. Until that lands,
every figure here is a cost or a budget, never a result.

## 9. Open

1. **The averaging depth `m`**, which trades calibration time against the
   residual, and follows from the receiver's actual noise rather than X1's
   budget for it.
2. **How often the cell loop must run against the column loop**, which needs
   the recovery measurement of §8.
3. **The watermark** the shadow scheduler fires on, which follows from the
   chiplet's buffer sizes (X2) and the link's round trip.
4. **Whether a failed trim is recoverable in the field** — the DRIFT_ALARM path
   assumes someone is listening.
