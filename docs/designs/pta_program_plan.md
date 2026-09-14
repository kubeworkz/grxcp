# PTA program plan: after the TFLT decision

**Companions:** [`pta_cpu_integration.md`](pta_cpu_integration.md),
[`pta_gpu_integration.md`](pta_gpu_integration.md),
[`pta_tpaqcn_review.md`](pta_tpaqcn_review.md).

**Status: PLAN, scope agreed, the decisions of §2 open.** This document orders
the next stretch of photonic-tile work across the c930, the G100 and grxcp. It
designs nothing new. Where it changes an earlier document's staging it says so,
and §7 lists the edits that follow once §2 is settled.

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
| Tile: PTM-C, error model, PTM-B, calibration (C0, C1, C2 tile, C3) | Not started |
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
   lands the next one. A comment in the DMA records that an 18× margin between
   the DMA and the core once hid a one-cycle bug there. The digital array
   takes 18 cycles over a K tile where a shot takes one, so at the Pockels
   points that margin is unlikely to survive; F0 and F1 find out. At system
   scale the review's operand-supply model puts one PCIe 6.0 x16 link at
   0.011 POPS of feed, against the 1.07 POPS the proposal's chip computes
   unfed.
3. **Accuracy over time is a calibration question.** Drift, not settle, is
   what a Pockels tile pays, so C3's schedulers are tuned to TFLT's drift and
   stressed with TFLN's.

---

## 2. Decisions to settle first

Each one blocks building, none needs a measurement, and each has a recommended
answer.

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
| C0 | PTM-C swapped in for the systolic array, impairments off | CPU document §6, unchanged | — |
| C1 | Error model, one impairment at a time; `PTA_DRIFT` defaults fitted to TFLT, with a TFLN setting as the stress case | CPU document §6, unchanged | C0, D1–D3 |
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
| F0 | Run the `M = 64, N = 8, K = 256` GEMM on the Verilator SoC and record `DMA_LAST`, `STALL_CT` and the core's A-row wait (`arow_stall_cnt`, not yet on a CSR), split into the initial load, PF1 (A rows fetched during compute) and PF2 (the next GEMM's operands, fetched while C is written) | *New:* fetch cycles per operand, and the A-row wait on the digital array, identical run to run | — |
| F1 | Add a feed term to the §2.1 model, built from F0's fetch rate | *New:* the model predicts the A-row wait and `DMA_CT` at C2 tile and MB, before they are measured | F0 |
| F2 | Feed options at the Pockels points: PF1 and PF2 as built; B tiles prefetched straight into resident banks; the whole GEMM staged before launch | *New:* chosen by measured total cycles, A-row wait included, at EO-scan and EO-res | C2 tile, MB |
| F3 | Requirements for the fabric plan: operands per second at each §6.2 point, which operands are resident and which streamed, and how long the tile can wait | *New:* every number traced to F0–F2 or to a model F1 checked | F1, F2, C4 |

F0's numbers belong to this SoC and its simulated DDR, not to any fabric, and
F3 says so: the fabric plan gets rates and access patterns, not this SoC's
latency. The G100 tile is fed differently — from LMEM, by the DXA, at cluster
scope — so G2 reports its DXA transfer cycles as its feed term, and G3 sets the
two feeds side by side.

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
| Now | Immediately, in parallel | D1–D4 settled, then S0; C0; A3; A-synth; F0; G1 |
| Next | C0 green, D1–D3 settled | C1, with F1 beside it; G0 and C3 once C1 is green |
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
| A bank select is not free in RTL | MB | The gate takes the RTL's select cycles as `Tw`; one cycle leaves EO-res resident |
| Bitwise RTL↔C parity may not survive the DPI path (CPU document §9, question 3) | C1, S2 | Keep the tile model's arithmetic integer, and weaken the gate deliberately, in writing, if it has to weaken |
| F0 measures this SoC's simulated DDR | F3 | F3 labels its numbers as this SoC's and passes on rates and access patterns |
| D1 couples two repositories that have no dependency today | G0 | Tagged vendoring in one direction, as grxcp already does with the DPI shim |
| A3 finds a short reset interval | A-CSR | A result, not a setback: the mainline's electronic nonlinearity is unaffected, and A-CSR is dropped |

---

## 7. Edits that follow

Once §2 is settled:

- [`pta_cpu_integration.md`](pta_cpu_integration.md): the status line, which
  still says nothing is built; step MB added to §6 between C2 and C3; §8
  item 3 pointed at MB; §9 question 4 answered by D3.
- [`pta_gpu_integration.md`](pta_gpu_integration.md): §7's staging per D4;
  §9 question 2 answered by D1.
- The S_ACT design note in grx930: nothing until A3 reports.
