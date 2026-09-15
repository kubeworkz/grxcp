# PTA program plan: after the TFLT decision

**Companions:** [`pta_cpu_integration.md`](pta_cpu_integration.md),
[`pta_gpu_integration.md`](pta_gpu_integration.md),
[`pta_tpaqcn_review.md`](pta_tpaqcn_review.md).

**Status: PLAN, in progress.** The decisions of §2 were settled on 2026-09-14,
all four as recommended, and §7 lists the edits that carried them into the
integration documents. F0 is measured, F1 has made its predictions (§3.3), C0
is green, and C1 is under way (§3.1).
This document orders the next stretch of photonic-tile work across the c930,
the G100 and grxcp. It designs nothing new; where it changes an earlier
document's staging, it says so.

**Scope, unchanged.** FPGA emulation and numerics, as the CPU document states
before anything else. This plan covers the tile on both devices, the operand
feed that the TFLT decision makes binding, and the grxcp surface that reports
an analog-emulated device. The coherent CPU–GPU fabric of
[GRX_GCPU.md](../GRX_GCPU.md) and [GRXIConnect.md](../GRXIConnect.md) gets a
plan of its own; this one hands it requirements (§3.3, step F3).

**Section numbers.** §2.1 and §6.2 are the CPU document's cost model and
sweep, cited often enough here to go unqualified. Every other section number
either names its document or is this one's.

---

## 1. Where things stand

| Work | State |
|---|---|
| Loop interchange (C2 on the digital array) | Landed in grx930 and gated: weight loads drop to `Nt·Kt`, results bit-identical |
| FP16/BF16 and DMA-abort fixes found on the way | Landed, with `tb_npu_float_prec` and the DDR-timeout TEST 4 |
| S_ACT activation stage | RTL through gate A2, tied off in `c930_npu_top`; not synthesized |
| Target material | TFLT, [`pta_cpu_integration.md`](pta_cpu_integration.md) §4.4 |
| Models | [`pta_tw_sweep.py`](pta_tw_sweep.py), [`pta_material_scorecard.py`](pta_material_scorecard.py), [`pta_operand_supply.py`](pta_operand_supply.py) |
| Feed, measured and modelled (F0, F1) | Done: `make npu_feed` and the SoC's F0 mode in grx930, and [`pta_feed_model.py`](pta_feed_model.py) (§3.3) |
| PTM-C, the compatibility shim (C0) | Done: bit-identical to the systolic array on grx930's NPU benches under `PTM_C=1` (§3.1) |
| Error model (C1) | In progress: QUANT, THERMAL, SHOT and PROG_ERR built in grx930 and bitwise against the C reference; drift, crosstalk and the accuracy sweep remain (§3.1) |
| Tile: PTM-B, calibration (C2 tile, C3) | Not started |
| SoC integration, firmware, Vivado (C4) | Not started |
| G100 tile (G0–G3) | Not started; staged behind C2 |
| grxcp NPU backend | Host side built and gated against register models and the vendored grx930 DPI shim; no PTA surface |

Three things the TFLT decision changed, and this plan hangs on them.

1. **The target's resident point cannot be measured yet.** EO-res needs
   `Nt · Kt` resident weight banks — 32 at `M = 64, N = 8, K = 256` — and the
   tile has two. It also wants the `m`-outer loop order, 7.2× ahead of the
   interchange, and the core no longer has that order: the interchange
   replaced it. Both move onto the critical path (§3.1, step MB).
2. **At the Pockels points the tile waits on its operands.** A resident GEMM
   is 25.6 µs, of which 20.5 µs is one-cycle shots and 5.1 µs row writes
   (§2.1), and the fetch of its 18,432 operands is no longer small beside it
   (§6.2). The core already counts the wait: under the interchanged order it
   walks every row of A in the first K tile and sits in `S_AROW` until the DMA
   lands the next one. F0 found that wait on the digital array before grx930's
   half-rate hop, when the core finished a row every 26 cycles and the DMA's
   row prefetch delivered one every 36; since the hop the core takes 72 and
   never waits. A one-cycle shot finishes a row every 9, and F1 puts the wait
   at about 1,600 cycles a GEMM. At system scale the review's operand-supply
   model puts one PCIe 6.0 x16 link at 0.011 POPS of feed, against the
   1.07 POPS the proposal's chip computes unfed.
3. **Accuracy over time is a calibration question.** Drift, not settle, is
   what a Pockels tile pays, so C3's schedulers are tuned to TFLT's drift and
   stressed with TFLN's.

---

## 2. Decisions to settle first

Each one blocked building and needed no measurement. **All four were settled on
2026-09-14 as recommended below.**

**D1 — Who owns the shared tile IP.** The tile model and its error budget are
written once and used by both devices, and neither RTL repository depends on
the other ([`pta_gpu_integration.md`](pta_gpu_integration.md) §9, question 2).
*Recommended:* grx930 owns it, because the c930 builds it first, and grxgpu
vendors tagged releases the way grxcp already vendors grx930's DPI shim under
`third_party/grx930/`. *Needed by:* C1, before any error-model RTL.

**D2 — How grxcp reports an analog-emulated GEMM.** The CPU document's §7 sets
the direction: a device property, a `grx-smi` line, two gates (bitwise against
the model; distributional against fp32, reported), and current-device-only
dispatch. What is left is the property's shape. *Recommended:* a struct, not a
flag — effective activation, weight and ADC bits (`PTA_BITS`), the seed
(`PTA_SEED`) and the impairment mask (`PTA_IMPAIR`) — because a flag says that
a device is analog and not how analog, and the distributional gate needs the
how. *Needed by:* C1, which is when the CPU document's §7 says both rules
must have answers.

**D3 — The accuracy workload.** C1's gate (a), C3's recovery margin and A3's
second stage all need one network with published quantization curves, small
enough for RTL simulation (§9, question 4 of the CPU document).
*Recommended:* the two-layer MLP on MNIST that question calls defensible, fixed
once so all three report on the same network. *Needed by:* C1.

**D4 — Re-stage the GPU work: G1 now, G0 after C1.** The GPU document holds
every phase behind C2, for two reasons: the shared IP should stabilize on the
cheaper device, and C2's sweep might say a GEMM engine is the wrong home for a
tile. Neither reason touches G1, a SimX study with no RTL and no shared IP, and
the question it answers — how many weight sets a cluster must keep hot — is
the capacity question step MB now asks of the c930. G0 is numerics only; what
it waits for is the error model, which C1 finishes. *Recommended:* start G1
now, with bank count `W` as an axis beside `Tw`, since at the Pockels points
`Tw` is the host's scan and `W` is what varies; start G0 when C1 is green.
G2 and G3 stay where they are. *Needed by:* nothing — it is what lets the GPU
track run beside the tile.

---

## 3. The tracks

Five tracks, by repository. A gate below is the existing documents' gate unless
it is marked *new*.

### 3.1 Track T — the tile on the c930 (grx930)

| Step | What | Gate | Needs |
|---|---|---|---|
| C0 | **Done, below.** PTM-C swapped in for the systolic array, impairments off | CPU document §6, unchanged | — |
| C1 | **In progress, below.** Error model, one impairment at a time; `PTA_DRIFT` defaults fitted to TFLT, with a TFLN setting as the stress case | CPU document §6, unchanged | C0, D1–D3 |
| C2 tile | PTM-B in the interchanged core | Total cycles match §2.1 at every §6.2 point, affine in `PTA_TW` | C1 |
| MB | Multi-bank tile with `Nt·Kt` resident banks, and a selectable loop order that restores `m`-outer | *New:* EO-res totals match §2.1 in both orders, with `Tw` set to the RTL's bank-select cycles, and C bit-identical between orders | C2 tile |
| C3 | Calibration FSM and the three schedulers | CPU document §6, run with drift at TFLT's fitted rate and again at TFLN's | C1, D3 |
| C4 | CSR decode, firmware, full SoC, Vivado on the Arty A7-200T | CPU document §6, with EO-res measured rather than modelled | C2 tile, MB, C3, A-synth |

**Step MB, in more detail.** At `Tw = 0` the §2.1 model gives EO-res 2,560
cycles in the `m`-outer order and 18.4 k interchanged. The `m`-outer FSM does
not need re-deriving: it is the core as it stood before grx930 commit 0c5df42,
which replaced it. Restoring it as a mode rather than a revert keeps the
interchange's C2 gate green while giving EO-res the order the model says it
wants. Storage is small — 2,048 weights — and the cost is the select path. If a
bank select takes a cycle in RTL, the gate counts that cycle as `Tw`; the §2.1
break-even sits at 8 cycles, so one cycle still leaves EO-res in the resident
regime.

**C0, met.** PTM-C needs no start strobe: the systolic array is a fixed
transform of its input streams, so the shim keeps each input's history on hop
edges and takes the samples each column's cascade would meet (CPU document
§4.1). In grx930, `PTM_C=1` builds any NPU bench with it in place of the array.
`tb_c930_npu`, `tb_npu_float_prec` and `tb_npu_feed` pass with it, the feed
bench's log is byte-identical to the array's, and a lockstep bench finds no
difference on any cycle. With one row's de-skew a window late, the benches go
red. The CPU document's §6 has the counts.

**C1, decided.** Four choices were settled on 2026-09-14, all as recommended,
before any error-model RTL. They are recorded with the fixed-point contract in
grx930's `c930/doc/pta_error_model_design_note.md`.

- *E1:* the core marks each result it captures, and the tile draws in the
  core's loop order — N tile, K tile, output row, column. A result then
  depends only on the seed, the shape and the operands, which is what lets
  grxcp's host predict it (CPU document §7, gate 1).
- *E2:* the generators reload from `PTA_SEED` at every GEMM start, as S_ACT's
  do, so queued GEMMs stay independent.
- *E3:* weight errors act after the DAC, `q_w(w) + eps + delta`, correcting
  the CPU document's §4.3.
- *E4:* the ADC's scale is a shift chosen per GEMM, a new field in
  `PTA_BITS`.

The model is integer and covers the integer precisions only. Drift's clock and
crosstalk's topology stay open until those impairments are built.

**C1, first four impairments.** QUANT, THERMAL, SHOT and PROG_ERR are built into
PTM-C, and a C reference, `c930/sim/pta_tile_model.c`, agrees with the RTL bit
for bit: C and the ADC saturation count match at every shape of the core-level
harness, the sweep's shape included, at both operand widths. With every
impairment clear, C0's benches still pass. The CPU document's §6 has the
counts and ablations. Still to come: drift, crosstalk, and gate (a), the
accuracy sweep on the D3 network.

### 3.2 Track A — activation (grx930)

| Step | What | Gate | Needs |
|---|---|---|---|
| A3 | Chain mode: reset interval × photons × detuning × curve shape | Reported, not gated: the reset-interval-versus-photons curve the review's kill criterion is evaluated on | A2 (done); D3 for its second stage |
| A-synth | Synthesize `c930_npu_act.sv` on its own | *New:* 100 MHz on the 200T, or a named pipeline cut with `ACT_P` updated to match | — |
| A-CSR | CSR mapping, snapshot bit, firmware | The S_ACT design note's order: only after A3 reports | A3 |

A-synth runs now because stage 1 multiplies the 48-bit accumulator by a
32-bit scale, in one cycle, and nothing has timed it. A3's result routes the
all-optical branch: a short reset interval closes it without touching the
mainline, whose nonlinearity stays electronic; a long one triggers the
reopening clause of the CPU document's §4.4.

### 3.3 Track F — the feed (grx930 measurement, grxcp models)

The track this plan adds. Its question is how long the tile waits on its
operands, measured where this program can measure it and handed on as
requirements where it cannot.

| Step | What | Gate | Needs |
|---|---|---|---|
| F0 | **Done, below.** Run the `M = 64, N = 8, K = 256` GEMM on the Verilator SoC and record `DMA_LAST`, `STALL_CT` and the core's A-row wait (`arow_stall_cnt`, not yet on a CSR), split into the initial load, PF1 (A rows fetched during compute) and PF2 (the next GEMM's operands, fetched while C is written) | *New:* fetch cycles per operand, and the A-row wait on the digital array, identical run to run | — |
| F1 | **Predictions made, below; checked at C2 tile and MB.** Add a feed term to the §2.1 model, built from F0's fetch rate | *New:* the model predicts the A-row wait and `DMA_CT` at C2 tile and MB, before they are measured | F0 |
| F2 | Feed options at the Pockels points: PF1 and PF2 as built; B tiles prefetched straight into resident banks; the whole GEMM staged before launch | *New:* chosen by measured total cycles, A-row wait included, at EO-scan and EO-res | C2 tile, MB |
| F3 | Requirements for the fabric plan: operands per second at each §6.2 point, which operands are resident and which streamed, and how long the tile can wait | *New:* every number traced to F0–F2 or to a model F1 checked | F1, F2, C4 |

F0's numbers belong to this SoC and its simulated DDR, not to any fabric, and
F3 says so: the fabric plan gets rates and access patterns, not this SoC's
latency. The G100 tile is fed differently — from LMEM, by the DXA, at cluster
scope — so G2 reports its DXA transfer cycles as its feed term, and G3 sets the
two feeds side by side.

**F0, measured.** In grx930, `make npu_feed` runs the NPU bench
(`c930/tb/tb_npu_feed.sv`), and the Verilator SoC built with `F0=1` runs the
same four queued GEMMs through the DMA arbiter, the crossbar and the L2. Both
repeat byte for byte and check C. Two cores were measured: grx930 8cceb33, and
the core after the grx930 team's half-rate hop (0afeb6e), which runs `S_RUN`
at 64 cycles a K tile for every precision instead of 18. Nothing on the DMA
side moved between them. One GEMM on the digital array:

| | NPU bench | SoC |
|---|---|---|
| Initial load: A row 0 and B, 2,304 operands | 580 cycles | 586 |
| PF1: A rows 1–63, 16,128 operands | a row every 36 cycles, 34 of them busy | every 39, 37 busy |
| Read request to first beat | 1 cycle | 2 |
| C writeback, 512 words | 1,283 | 1,410 |
| A-row wait (`S_AROW`), before / after the hop | 566 / 0 | 755 / 0 |
| Whole GEMM, before / after the hop | 73,601 / 167,242 | 73,923 / 167,375 |

- Before the hop, the A-row wait was all in the first K tile, where the core
  finished a row every 26 cycles and PF1 delivered one every 36 or 39. After
  it the core takes 72 cycles a row, slower than PF1, and never waits.
- C writeback, about five cycles a beat, is the largest fixed feed cost.
- PF2 as built is a net loss at this shape. It unpacks one element per cycle,
  so when writeback ends it has fetched 143 of the 288 beats of the next
  GEMM's first A row and B. `P_DONE` then drains the abandoned burst for 145
  cycles (132 on the SoC), holding `o_done` high, and the next GEMM loads cold
  anyway.
- The SoC path costs 322 cycles a GEMM, with no throttling inside a burst. Its
  first GEMM after boot took 8 more.

**F1, predicted.** [`pta_feed_model.py`](pta_feed_model.py) adds the feed to
the §2.1 model: load, core, A-row wait, hand-off and writeback, with the wait
taken from a row-by-row race between the core and PF1 that F0's per-row
timeline pins down with no free constant. It reproduces all four measurements —
both levels, both cores — to the hop's one-cycle phase alignment. Its first
version did not: taking PF1's period as its busy cycles alone, and fitting a
row offset to the pre-hop wait, it predicted 27 cycles of wait on the hop core,
where the RTL measured none. The two idle cycles a row were what the fit had
hidden. Its predictions, for C2 tile and MB to check, with the restore
unfolded as built (NPU bench / SoC):

| Point, loop order | A-row wait | Whole GEMM | Feed share |
|---|---|---|---|
| TO-10µs, interchanged | 385 / 574 | 78,796 / 79,118 | 2.9% / 3.3% |
| EO-scan, interchanged | 1,637 / 1,826 | 39,856 / 40,178 | 8.8% / 9.5% |
| EO-res, interchanged | 1,701 / 1,890 | 37,872 / 38,194 | 9.4% / 10.2% |
| EO-res, shipped | 0 / 0 | 4,427 / 4,560 | 42% / 44% |

The shipped order still wins at EO-res once the feed counts, but by less: 5.0×
with the restore folded (4.9× on the SoC), against 7.2× for the core alone,
because 1,900 to 2,000 fixed feed cycles now sit beside a 2,560-cycle core.
That puts writeback and PF2 first among F2's candidates, ahead of anything in
the core.

### 3.4 Track G — the G100 (grxgpu, by proposal)

Every step lands as a proposal in `grxgpu/docs/proposals/`, written with
grxgpu's RTL owners, under the boundary rule of `AGENTS.md` §2.

| Step | What | Gate | Needs |
|---|---|---|---|
| G1 | SimX study of the three weight-set policies, with bank count `W` beside `Tw` | GPU document §7, plus the `W` axis | D4 |
| G0 | `VX_tcu_fedp_analog`, vendoring the c930 error model | GPU document §7, unchanged | C1, D1 |
| G2 | Cluster-scope tile beside the DXA | GPU document §7, with DXA transfer cycles reported as the feed term | C2 tile, MB, G1 |
| G3 | The same block-scaled GEMM on both tiles | GPU document §7, with the two feeds compared | G2, C4 |

### 3.5 Track S — grxcp

| Step | What | Gate | Needs |
|---|---|---|---|
| S0 | The D2 property specified: its fields, what an NPU without a tile reports, and the `grx-smi` line | Review; no code | D2 |
| S1 | The property populated from the PTA CSRs, and the vendored DPI shim extended to answer on them | `AGENTS.md` §3: every field sourced or reported unknown (−1); the NPU BACKEND GATE in `ci/build_mock.sh` green | S0, C4's CSR map |
| S2 | The two gates: bitwise against the model, its golden data regenerated only as a reviewed step, and the distributional report | CPU document §7 | S1 |

---

## 4. Order

| Wave | Starts when | Steps |
|---|---|---|
| Now | Immediately, in parallel | S0; A3; A-synth; G1 (D1–D4 settled; F0 and C0 done) |
| Next | C0 green, as it now is | C1; G0 and C3 once C1 is green (F1 done) |
| Then | C1 green | C2 tile, then MB; F2 once both are in; G2 once MB and G1 have reported |
| Last | C2 tile, MB, C3 and A-synth green | C4; then S1 and S2 on its CSR map, F3 on its numbers, and G3 |

The critical path is C0 → C1 → C2 tile → MB → C4. Everything else runs beside
it or hangs off one of its gates, and nothing on it waits for the GPU.

---

## 5. What the program will have produced

Architectural and numerical results, as the scope allows, and none of them a
claim about a photonic device.

1. The §6.2 sweep measured at all four points — EO-res in both loop orders —
   with `DMA_CT` and the A-row wait beside every total, on the SoC and on the
   200T.
2. The reset-interval-versus-photons curve, which settles the all-optical
   branch against the review's kill criterion.
3. The calibration schedulers compared under TFLT's drift and under TFLN's.
4. One block-scaled GEMM agreeing bitwise across the c930 and G100 tiles, with
   their feeds compared.
5. A requirements note for the fabric plan, every number measured or checked.
6. grxcp reporting analog emulation through a device property, with both
   gates in CI.

**Recorded for a later physical track, without committing to one:** what a
TFLT tile would have to provide for these results to transfer. That is
resident banks per GEMM shape (MB), DAC and ADC bits (C1's accuracy against
bits), a drift budget and calibration interval (C3), the activation reset
interval (A3), and an operand feed rate (F3). Each is the output of a step
above.

---

## 6. Risks

| Risk | Where it bites | Response |
|---|---|---|
| Timing at 100 MHz: the shot path adds a multiply, a square-root approximation and two adds (CPU document §6.1), and S_ACT's stage 1 is a wide one-cycle multiply | C4 | A-synth now. A pipeline cut goes into `PTA_TS`, `ACT_P` and the §2.1 model, never around them |
| The SoC as built cannot run the §6.2 shape: `MAX_M` 8, `MAX_K` 16, and a 64 KB DDR window fixed in the crossbar | C4 | F0 runs the shape on a Verilator build resized with `F0=1`. C4 sizes the synthesized SoC for it and prices the area, which the CPU document's §6.1 does not |
| grx930's half-rate hop changed the systolic array's timing contract: `S_RUN` went from 18 to 64 cycles a K tile for every precision, with rows and seeds presented on hop windows | C0 | Closed: PTM-C is built on the hop schedule and C0's exact-swap gate passed. The CPU document's §2.3 cycles and `pta_tw_sweep.py`'s asserts stay as the record of the core before the hop |
| A bank select is not free in RTL | MB | The gate takes the RTL's select cycles as `Tw`; one cycle leaves EO-res resident |
| Bitwise RTL↔C parity may not survive the DPI path (CPU document §9, question 3) | C1, S2 | Keep the tile model's arithmetic integer, and weaken the gate deliberately, in writing, if it has to weaken |
| F0 measures this SoC's simulated DDR | F3 | F3 labels its numbers as this SoC's and passes on rates and access patterns |
| D1 couples two repositories that have no dependency today | G0 | Tagged vendoring in one direction, as grxcp already does with the DPI shim |
| A3 finds a short reset interval | A-CSR | A result, not a setback: the mainline's electronic nonlinearity is unaffected, and A-CSR is dropped |

---

## 7. Edits that followed §2

Made on 2026-09-14, when §2 was settled:

- [`pta_cpu_integration.md`](pta_cpu_integration.md): the status line; step MB
  in §6 between C2 and C3, with F1's predictions in C2's and MB's gates; D2 in
  §7; §8 item 3 pointed at MB; §9 question 4 answered by D3; a note in §6.1
  that its baseline cannot run the §6.2 shape; and a note in §2.3 that its
  measured cycles predate the half-rate hop.
- [`pta_gpu_integration.md`](pta_gpu_integration.md): the status line and §7's
  staging per D4; §9 question 2 answered by D1.
- The S_ACT design note in grx930: nothing until A3 reports.
