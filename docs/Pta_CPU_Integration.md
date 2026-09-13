# PTA on the CPU: a photonic tensor tile in the c930 NPU

**Companions:** `heterogeneous_devices.md`, `pta_gpu_integration.md`. **Source analysis:** GRX_PTA_Integration.md in this repository's `docs/`.

**Status: DESIGN, nothing built.** No RTL, no model, no CSR. This document fixes what gets built, in what order, and — more importantly — what each stage is allowed to claim.

**Scope decision, made before anything else.** This program terminates at FPGA emulation and numerics. There is no photonic PDK, no MPW shuttle, no mixed-signal tape-out. That is not a reduced version of the photonic project; it is a different project with a different deliverable, and the difference has to be stated once, loudly, or every result downstream gets read as a claim about silicon photonics that it is not.

What an emulation-only program can honestly produce:

1. **The architecture that photonic constraints force.** A weight-stationary analog tile has a weight-programming cost between two and six orders of magnitude above its compute cost, depending on the tuning mechanism — a thermo-optic phase shifter settles in tens of microseconds against a shot of nanoseconds. That single ratio invalidates the loop nest the c930 NPU ships today. Finding that, fixing it, and measuring the fix is a real result, and it is entirely a digital-RTL result.
2. **The numerics.** How much accuracy survives a 4–6 effective-bit analog channel with drift, and what digital correction buys it back. Answerable against a seeded, bit-deterministic model.
3. **An open emulation framework.** Photonic-accelerator architectural simulators are immature and mostly unpublished. A synthesizable, parameterized, deterministic tile model with a real error budget, attached to a working SoC, does not currently exist in public.

What it cannot produce, and must never be reported as producing: any statement about photonic device physics, any throughput number for a photonic chip, any validation of a mesh topology. The model is a *hypothesis about* published devices. A green run against it proves the digital architecture, exactly the way `test_npu_c930_model.cc` proves the backend logic and nothing whatsoever about the register map (see `heterogeneous_devices.md` §5).

---

## 1. Why the c930 NPU is the right host

Not for convenience. Because the dataflow already matches.

A photonic matrix-vector multiplier holds its weights in a slow physical resource — thermo-optic phase shifters, microring resonances, phase-change cells — and streams activations past them. A weight-stationary systolic array holds its weights in a PE register and streams activations past them. These are the same dataflow, arrived at from opposite directions: both put the reused operand in the resource that is expensive to change.

grx930/c930/rtl/c930_systolic_array.sv already has every port a photonic tile needs, under names that translate one-for-one:

<table>
  <tr><th>c930 systolic array</th><th>Photonic tile</th><th>Note</th></tr>
  <tr><td>i_wen / i_wrow / i_wcol / i_wdata</td><td>weight-DAC scan, one element per beat</td><td>A real mesh is programmed serially too; the serial load is not an emulation artifact</td></tr>
  <tr><td>i_wbank (write) / i_bank_sel (compute)</td><td>shadow bank vs. active mesh</td><td>Double-buffered weights are how you hide a slow program behind a fast compute</td></tr>
  <tr><td>i_act, one per row</td><td>activation DAC → modulator array</td><td></td></tr>
  <tr><td>o_ps_out, one per column</td><td>photodetector → TIA → ADC</td><td>Optical summation lands on the detector; the column is the reduction</td></tr>
  <tr><td>i_precision broadcast</td><td>analog mode: bits, integration time</td><td>3 bits wide, five of eight codes used</td></tr>
</table>

So the tile is a module swap, not an architecture. That matters for the first milestone (§6) and it is the reason this path is cheap.

**One structural mismatch, and it is the interesting one.** The systolic array is temporally *skewed*: row `r`'s activation fires at cycle `r` (`(t == r) && (r < kr_reg)` in the core's feed logic) and column `n`'s result is captured at `t = n + NUM_ROWS + 2`. A photonic MVM is *broadside*: all N modulators are driven at once, all M detectors integrate at once, one shot, one latency. Reconciling those two is §4.

---

## 2. What the existing loop nest costs, before photonics enters

This is the finding the rest of the document hangs on, and it is worth stating in digital terms first because it is true without any photonics at all.

grx930/c930/rtl/c930_npu_core.sv walks `m` outermost, then the N tile, then the K tile. Its inner sequence per K tile, in steady state, is:

```plaintext
  S_PRELOAD   preload_kr * nc cycles   (load tile k+2's weights into the idle bank)
  S_RUN       NUM_ROWS + NUM_COLS + 2  (compute tile k+1)


```

At the 200T configuration — NUM_ROWS = NUM_COLS = 8 — that is **64 cycles of weight movement per 18 cycles of compute**. Over one `(m, nt)` pass the machine spends `64·Kt` cycles moving weights and `18·Kt` computing, plus 8 writing: the array is doing arithmetic **22% of the time**.

**And the existing counter does not show this.** `stall_cnt` increments only in `S_WLOAD`, and in steady state the weight movement happens in `S_PRELOAD`, which is not counted. `STALL_CT` therefore reports `64` per `(m, nt)` pass where the true figure is `64·Kt` — under-reporting weight movement by a factor of `Kt`, which is 32 at `K = 256`. Extending `stall_cnt` to cover `S_PRELOAD` is a one-line change and it is a prerequisite for every measurement in this document, because otherwise phase C2's improvement will appear to come from nowhere.

Worse, the weights are reloaded per output row. `S_WRITE`'s exit resets `nt_reg` and `kt_reg` to zero on every `m` increment, so the same B tile is programmed `M` times. At `MAX_M = 64` that is 64 redundant programmings of every weight tile.

Digitally this is a 4.5× inefficiency and nobody has minded. Introduce a weight-programming time `Tw` measured in microseconds and it stops being an inefficiency and becomes the entire runtime.

### 2.1 The cost model

Let `Tw` be weight program plus settle, `Ts` be shot plus ADC conversion, `Kt = ceil(K / NUM_ROWS)`, `Nt = ceil(N / NUM_COLS)`. Then:

<table>
  <tr><th>Loop order</th><th>Weight programmings</th><th>Total time</th></tr>
  <tr><td>m outer (as shipped)</td><td>M · Nt · Kt</td><td>M · Nt · Kt · (Tw + Ts)</td></tr>
  <tr><td>(kt, nt) outer, m inner</td><td>Nt · Kt</td><td>Nt · Kt · (Tw + M · Ts)</td></tr>
</table>

The weight term drops by exactly `M`. For a plausible thermo-optic mesh — `Tw` = 10 µs, `Ts` = 50 ns — and `M = 64, N = 8, K = 256`:

- as shipped: 64 · 1 · 32 · 10.05 µs ≈ **20.6 ms**
- interchanged: 32 · (10 µs + 64 · 50 ns) ≈ **0.42 ms**

Roughly 49×, and the gap widens with `M`. **The loop interchange is the photonic integration.** Everything else in this document is instrumentation for measuring it or numerics for making the result trustworthy.

### 2.2 What the interchange costs

Accumulator state. Today `acc[0..NUM_COLS-1]` is one output row's running K-accumulator — `NUM_COLS × ACC_W` = 8 × 48 bits, in flops. With `m` innermost you need `M` accumulators live at once.

That storage already exists. `c_mem` is `MAX_M * MAX_N` words of 32 bits, declared `(* ram_style = "block" *)` and already BRAM-mapped: 64 × 8 × 32 b = 16 Kb, one RAMB18. Make it read-modify-write per `(m, kt)` instead of write-once per `(m, nt)` and the register accumulator disappears entirely. The cost is one BRAM read port and one adder in the S_WRITE path; the saving is 8 × 48 flops. This is close to free, on a device with 357 unused BRAMs.

The correctness obligation is the FP16/BF16 path: `acc[n]` carries an FP32 value in those modes and `c_mem` is written as `acc[n_cnt][31:0]`, so a read-modify-write accumulator must route through the same FP32 adder (`c930_fp16_acc`) rather than an integer add. That is a real change, not a rewiring, and it is where this phase will actually take time.

### 2.3 The interchange helps the digital array too

Under the shipped loop order the digital array is 22% efficient. Under the interchange the same array amortizes 64 cycles of weight load over `M` rows of 18-cycle compute: 64 + 64·18 = 1216 cycles per weight tile, 95% of it arithmetic. **Roughly 4.3× on the shipping NPU, with no photonics involved.**

That is worth saying plainly to the grx930 team, because it decouples the value of this work from whether photonics ever happens.

---

## 3. Precision codes and the CSR address space

Two hard constraints found by reading, not assumed.

`i_precision`** has room.** It is `[2:0]` at every level — `o_precision` in the CSR, `i_precision` in core, array and PE — and codes 0–4 are used (INT8, INT16, FP16, BF16, INT4). Codes 5, 6, 7 are free, which is enough for the analog modes proposed in §5.2. No datapath widening.

The catch is the DMA. INT4 is nibble-packed and the cross-GEMM prefetch path (PF2) was enabled specifically for that mode, so a new precision code is not free in grx930/c930/rtl/c930_npu_dma.sv — it needs an unpack rule and a `dims_ok` clause. Budget for it; do not describe it as a three-line change.

**The CSR decode is full; the address space is not.** grx930/c930/rtl/c930_npu_csr.sv decodes `s_axi_awaddr[5:2]`, sixteen words, and all sixteen codes are assigned (CTRL, STATUS, DIM_M/N/K, A/B/C_BASE, PREC, CYCLE_LO, DMA_LAST, OP_COUNT, STALL_CT, DMA_CT, QUEUE_STAT, QUEUE_MAX). There is no spare offset inside the register file.

Outside it there is plenty. The SoC crossbar decodes the MMIO slave over `0x4000_0000`–`0x4000_FFFF` — 64 KB — with the UART carved out at `0x4000_1000`–`0x4000_100F` by an earlier branch in the same decode function. So widening the CSR's internal decode from `[5:2]` to `[7:2]` — 64 words, `0x00`–`0xFC` — leaves `0x00`–`0x3C` bit-identical, puts PTA state at `0x40`–`0xCC`, and **requires no crossbar change at all**.

One thing to fix while there. The c930 architecture document's memory map declares the NPU MMIO region as `0x4000_0000`–`0x4000_001F`, 32 bytes, with everything above reserved. That is already wrong in two directions: the crossbar decodes 64 KB, and the same document's own performance-counter table lists offsets at `0x2C`, `0x30` and `0x34`. The map should be corrected to the implemented 64 bytes before it is extended to 256, or the extension inherits a document nobody trusts.

### 3.1 Proposed PTA register block

<table>
  <tr><th>Offset</th><th>Name</th><th>Access</th><th>Description</th></tr>
  <tr><td>0x40</td><td>PTA_CTRL</td><td>RW</td><td>bit0 EN, bit1 CAL_NOW, bit2 CAL_AUTO, bit3 MODEL_RST, bits[6:4] CAL_SCHED</td></tr>
  <tr><td>0x44</td><td>PTA_STATUS</td><td>R</td><td>bit0 CAL_BUSY, bit1 CAL_VALID, bit2 SAT_STICKY, bit3 DRIFT_ALARM, bits[23:8] last calibration residual</td></tr>
  <tr><td>0x48</td><td>PTA_IMPAIR</td><td>RW</td><td>one enable bit per impairment: QUANT, THERMAL, SHOT, DRIFT, XTALK, MZM_NL, PROG_ERR</td></tr>
  <tr><td>0x4C</td><td>PTA_BITS</td><td>RW</td><td>[3:0] activation bits, [7:4] weight bits, [11:8] ADC bits</td></tr>
  <tr><td>0x50</td><td>PTA_SEED</td><td>RW</td><td>LFSR seed; a write resets every noise generator</td></tr>
  <tr><td>0x54</td><td>PTA_SIGMA_TH</td><td>RW</td><td>thermal/TIA noise σ, Q8.8 in ADC LSB</td></tr>
  <tr><td>0x58</td><td>PTA_SIGMA_SH</td><td>RW</td><td>shot-noise coefficient k, so σ_shot = k·√|y|</td></tr>
  <tr><td>0x5C</td><td>PTA_SIGMA_PR</td><td>RW</td><td>weight-programming error σ</td></tr>
  <tr><td>0x60</td><td>PTA_DRIFT</td><td>RW</td><td>[15:0] drift step σ, [31:16] log2 update interval</td></tr>
  <tr><td>0x64</td><td>PTA_XTALK</td><td>RW</td><td>nearest-neighbour coupling, Q0.8</td></tr>
  <tr><td>0x68</td><td>PTA_TW</td><td>RW</td><td>weight program + settle, in core cycles</td></tr>
  <tr><td>0x6C</td><td>PTA_TS</td><td>RW</td><td>shot + ADC latency, in core cycles</td></tr>
  <tr><td>0x70</td><td>PTA_CAL_PER</td><td>RW</td><td>calibration period for the periodic scheduler</td></tr>
  <tr><td>0x74</td><td>PTA_CAL_THR</td><td>RW</td><td>predicted-error threshold for the drift-predictive scheduler</td></tr>
  <tr><td>0x78</td><td>PTA_CAL_CT</td><td>R</td><td>calibrations run</td></tr>
  <tr><td>0x7C</td><td>PTA_CAL_CYC</td><td>R</td><td>cycles spent calibrating</td></tr>
  <tr><td>0x80</td><td>PTA_SHOT_CT</td><td>R</td><td>optical shots issued</td></tr>
  <tr><td>0x84</td><td>PTA_WLOAD_CT</td><td>R</td><td>weight-bank programmings</td></tr>
  <tr><td>0x88</td><td>PTA_SAT_CT</td><td>R</td><td>ADC saturation events</td></tr>
  <tr><td>0x8C</td><td>PTA_ERR_MAX</td><td>R</td><td>max |measured − expected| from the last calibration</td></tr>
  <tr><td>0x90–0xAC</td><td>PTA_GAIN[j]</td><td>RW</td><td>per-column gain, Q8.8</td></tr>
  <tr><td>0xB0–0xCC</td><td>PTA_OFFS[j]</td><td>RW</td><td>per-column offset, signed</td></tr>
</table>

`PTA_CAL_CYC` and `PTA_WLOAD_CT` are not diagnostics. They exist so a reported GEMM time can be decomposed into compute, weight programming, and calibration. Without them `CYCLE_LO` folds all three into "compute" and every speedup number this project produces is unfalsifiable. They are the honesty instrumentation, and they should be read as the same class of object as `ci/check_perf.py`'s zero-tolerance baselines.

### 3.2 Calibration must not break the completion contract

grx930/c930/rtl/c930_npu_csr.sv spends fourteen lines of header warning that polling `DONE` then `BUSY` is invalid for a batch, and that the only correct test is `QUEUE_STAT` occupancy == 0 **and** `STATUS.BUSY` == 0. Calibration introduces a third machine state and can silently invalidate that predicate.

The failure, precisely: if calibration blocks dispatch without setting `BUSY`, then a `CTRL.START` arriving during calibration takes the dispatcher's `D_IDLE` / `start_requested` / `fifo_empty` branch, which dispatches directly from the live CSRs — into a tile that is not available. If instead the START is dropped, occupancy stays 0 and `BUSY` stays 0 and a driver correctly applying the documented predicate concludes the batch finished.

The fix is minimal and it preserves the predicate exactly. Introduce `cal_busy` and use it to widen the existing "cannot dispatch now" condition:

```plaintext
wire cannot_dispatch = i_busy || cal_busy;
wire do_push = (start_requested && !fifo_full && (cannot_dispatch || !fifo_empty))
            || (pending_start && !pending_pushed && !fifo_full && cannot_dispatch);


```

and gate all three `D_IDLE` dispatch branches (pending, start_requested, drain) on `!cal_busy`. A START during calibration then goes to the FIFO, occupancy becomes non-zero, and `occupancy == 0 && busy == 0` correctly reports not-done. `STATUS.BUSY` keeps its per-command meaning; calibration is visible only through `PTA_STATUS.CAL_BUSY`.

Getting this wrong reintroduces exactly the queue-drain class of bug the CSR header documents, so it wants its own regression: START during an in-progress calibration, three times back to back, occupancy checked at each step.

---

## 4. The tile: two variants, and why both exist

### 4.1 PTM-C — the compatibility shim

A module with the port list of `c930_systolic_array`, byte for byte, that absorbs the skew at its own boundary: it de-skews `i_act` into an N-element register over `NUM_ROWS` cycles, fires one broadside shot, and re-emits `o_ps_out` on the exact cycles the core's staggered capture (`t >= NUM_ROWS + 2`) expects.

Its purpose is not performance. **With every impairment disabled it must be bit-identical to the systolic array on the existing self-checking testbench.** That single property is what makes every later numerical difference attributable: if the shim is not proven exact first, an accuracy delta observed in phase C1 could be the error model or could be a de-skew bug, and there is no experiment that separates them after the fact.

Zero changes to core, DMA or CSR. One line in the Makefile's `NPU_RTL` list.

### 4.2 PTM-B — the broadside tile

The tile the architecture actually wants: takes a whole K-tile activation vector, returns a whole column of results after `PTA_TS`. Requires the core's `S_RUN` to become a shot-and-wait rather than an 18-cycle skewed drain, and it is the variant the §2 loop interchange is written against.

PTM-B is where the speedup lives and PTM-C is where the trust lives. Build both; keep both; run the numerics on whichever is convenient, since with identical error parameters they must agree.

### 4.3 The error model

The model in the source analysis — `sum += ($signed(lfsr[3:0]) - 8)`, a uniform ±8 LSB — is not defensible in a paper and should not be built. The first-order model that is:

```plaintext
  y_j = g_j · Σ_i  q_w(w_ij + eps_ij + delta_ij(t)) · q_a(x_i)
      + Σ_{i'~i} chi · q_w(w_i'j) · q_a(x_i')          crosstalk, nearest neighbour
      + n_th                                            thermal / TIA, sigma_th
      + n_sh(y)                                         shot, sigma = k*sqrt(|y|)
      + o_j
  d_j = sat( round( y_j / LSB_adc ), 2^(B_adc-1) )


```

with `q_a` an activation quantizer at `B_a` bits (plus the MZM's sinusoidal residual when MZM_NL is enabled), `q_w` a weight quantizer at `B_w` bits, `eps_ij` a fixed programming error drawn once per weight load, and `delta_ij(t)` a bounded random walk updated every `2^PTA_DRIFT[31:16]` cycles and reset by calibration.

**Shot noise is signal-dependent and that is the whole point.** Modelling detector noise as a constant additive term — which every casual model does — removes the one property that drives the architecture: SNR improves toward full scale, so effective precision depends on where in the dynamic range the tile is operating, so *activation scaling becomes an architectural knob* (§5.2). A constant-noise model produces a project with no interesting questions in it.

Synthesis cost is modest. Gaussian noise from the sum of three or four LFSR draws (Irwin–Hall) is a standard trick and needs no multiplier. The `√|y|` term is a leading-zero count plus a four-entry interpolation LUT — c930 already has the comparator/leading-zero primitives in `c930_cla_comp`. Per-column gain and offset are one DSP and one adder each.

**Determinism is a hard rule.** No `$random`, no `$urandom`, no `$urandom_range`, anywhere in the model. Every stochastic quantity comes from an LFSR seeded by `PTA_SEED`. The reason is not aesthetic: grxcp's rule is that every conformance test runs on two backends and a divergence blocks (`AGENTS.md` §4). An analog device with a non-reproducible model makes that rule unenforceable, and the project quietly loses its ability to distinguish a model bug from a backend bug. Seeded determinism keeps RTL↔C parity a bitwise question.

---

## 5. Ideas worth building that are not in the source analysis

Five. Each is cheap on hardware already owned, and each answers a question the source document does not ask.

### 5.1 Calibration in the memory shadow

Calibration costs time. Drift costs accuracy. The naive answer is a fixed period, which is either too often (wasted) or too rare (inaccurate) and is never right for two different workloads.

Better: calibrate when the tile would be idle anyway. The DMA already stalls the array while fetching A and B, and the machine already measures it — `DMA_CT` is a live counter and `DMA_LAST` latches the last GEMM's total. Under the §2 interchange those stalls get *longer* per weight tile, because a tile now feeds `M` rows. A calibration that fits inside the PF2 prefetch window is free in wall-clock terms.

Three schedulers, selected by `PTA_CTRL[6:4]`, so they can be compared rather than argued about: off, periodic, drift-predictive (extrapolate from the last two calibration residuals and fire when predicted error crosses `PTA_CAL_THR`), and shadow (fire only inside a DMA stall, with the predictive threshold as a backstop). `PTA_CAL_CYC` measures what each actually costs.

This is the clearest publishable contribution in the CPU path: *calibration scheduling for analog accelerators, measured against a real memory system rather than an assumed one.*

### 5.2 Block scaling, borrowed from the GPU

The G100 tensor unit already implements microscaling formats — mxfp8, mxbf8, mxfp4, nvfp4 — with per-block scale factors carried in a separate metadata SRAM. Those formats exist to keep a low-precision channel near full scale.

That is exactly, and non-obviously, what an analog tile needs. Shot noise makes SNR position-dependent in the dynamic range; a block-scaled integer format with a per-K-tile exponent keeps every shot near the top of the ADC range where the SNR is best. So the analog precision codes proposed for `i_precision` should be **block-scaled**, not plain INT: code 5 = 4-bit activations with a per-tile scale, code 6 = 8-bit with a per-tile scale.

This has a second payoff. The scale factor per tile is exactly the quantity a compiler can compute statically or a calibration pass can measure, which turns "pick the operating point" from a hardware problem into a software one — and it is the same problem the GPU path already has machinery for. One numeric format across both devices is worth more than either device's local optimum.

### 5.3 Chopper stabilization across the K loop

A GEMM accumulates over `Kt` tiles. Random error grows as `√Kt`; systematic error — the per-column offset `o_j`, drift bias — grows as `Kt`. After 32 K tiles the systematic term dominates by 5.7×.

Borrow from analog circuit design: negate the activation encoding on odd K tiles and negate the ADC result on odd K tiles. Signal adds; a constant offset cancels exactly over each tile pair. Cost in RTL: two XOR-with-sign stages and one bit of state. Cost in accuracy: nothing. It does not touch gain error, which is why per-column gain still needs calibration.

Chopping is textbook in ADCs and, as far as the literature reviewed for this document goes, unreported as a K-loop technique in tensor accelerators. It is nearly free to build and it is directly measurable with the drift model on.

### 5.4 Sparse digital residual correction

Analog error is roughly proportional to the magnitude of the terms being summed, so a small number of large weights carry most of the error. Keep the top-`k` weights per tile in a digital side path, subtract their analog contribution and add the exact one.

On the c930 this is unusually cheap, because if PTM-C rather than PTM-B is instantiated the digital PEs have not been deleted — the tile can compute the `k`-term correction on the same hardware in `k` extra cycles. The experiment is: sweep `k` from 0 to 8 and measure recovered accuracy per cycle spent. If the curve is steep at small `k`, mixed analog/digital tensor units are a result; if flat, that is a negative result worth publishing too.

### 5.5 State the hypothesis that could kill the project

If the `Tw`-to-`Ts` ratio is large enough, the optimal machine is not "a photonic tile inside a general GEMM engine" — it is a weight-resident inference engine that never reprograms during a model's execution, with the digital array handling everything else. That conclusion is *available from this program*: sweep `PTA_TW` and find the crossover.

Writing it down as a hypothesis under test, before the sweep, is the difference between a measurement and a justification. It also determines what the CPU path should be optimizing for if the crossover lands badly: weight-set residency and capacity, not GEMM throughput.

---

## 6. Phases and exit gates

Each gate is a thing that can fail, and the ablation is named, in the style of `heterogeneous_devices.md` §5.

**C0 — the swap is exact.** PTM-C built, pin-compatible, all impairments off. *Gate:* the existing self-checking NPU testbench passes with PTM-C substituted for the systolic array, with zero output differences across all five precision modes. *Ablation:* flip one de-skew index and confirm the testbench goes red. Nothing else in this document may start before this gate is green — every later number depends on it.

**C1 — the error model, one impairment at a time.** Quantization, then thermal, then shot, then programming error, then drift, then crosstalk. Each behind its `PTA_IMPAIR` bit, each with its own directed test. *Gate:* (a) an impairment sweep on a small MLP with published accuracy-vs-bits curves; (b) RTL and the C model agree **bitwise** for a fixed `PTA_SEED`, driven through the existing NPU DPI wrapper. *Ablation:* corrupt one LFSR tap and confirm the parity check fails.

**C2 — loop interchange and the broadside tile.** PTM-B plus the `m`-inner FSM with the `c_mem` read-modify-write accumulator. *Gate:* for a fixed GEMM, weight-programming count drops from `M·Nt·Kt` to `Nt·Kt` exactly, results are unchanged, and total cycles as a function of `PTA_TW` matches the §2.1 model. The FP16/BF16 accumulator path is the sub-gate most likely to fail; test it first. *Secondary result, reportable on its own:* the same interchange on the digital array, measured.

**C3 — calibration.** Per-column affine correction, calibration FSM, three schedulers, and the `cal_busy` dispatch guard. *Gate:* with drift enabled at a stated rate, accuracy recovers to within a stated margin of the no-drift case, and `PTA_CAL_CYC` shows the shadow scheduler costing measurably less wall-clock than the periodic one at equal accuracy. *Ablation:* the START-during-calibration regression from §3.2.

**C4 — SoC, firmware, numbers.** CSR decode widening, firmware, the full-SoC test suite, and a Vivado run on the Arty A7-200T. *Gate:* real utilization and timing, not estimates; the five existing full-SoC tests still pass; the `Tw`-to-`Ts` ratio sweep of §5.5 produced.

### 6.1 Does it fit?

The 200T baseline is 78,750 / 134,600 LUTs (58.5%), 27,097 FFs (10.1%), 8 / 365 BRAM, 148 / 740 DSP. Estimated PTA additions:

<table>
  <tr><th>Block</th><th>LUTs</th><th>DSP</th><th>BRAM</th></tr>
  <tr><td>Quantizers (in + out)</td><td>~1,300</td><td>0</td><td>0</td></tr>
  <tr><td>Broadside MVM (replaces PE multipliers)</td><td>~0 net</td><td>~0 net</td><td>0</td></tr>
  <tr><td>Noise generation, 8 columns</td><td>~2,000</td><td>8</td><td>0</td></tr>
  <tr><td>Drift, 64 weights</td><td>~2,600</td><td>0</td><td>0</td></tr>
  <tr><td>Calibration FSM + pattern ROM + LMS update</td><td>~2,000</td><td>2</td><td>1</td></tr>
  <tr><td>m-inner accumulator (reuses c_mem)</td><td>~200</td><td>0</td><td>0</td></tr>
  <tr><td>Widened CSR decode + PTA registers</td><td>~800</td><td>0</td><td>0</td></tr>
  <tr><td>Total</td><td>~9,000</td><td>~10</td><td>~1</td></tr>
</table>

Against roughly 55,850 LUTs, 592 DSPs and 357 BRAMs of headroom. It fits with room to spare, on hardware already on the desk. Timing is the risk, not area — the shot path adds a multiply, a square-root approximation and two adds where the systolic PE had one registered product, and the array feed logic already carries comments about paths that had to be broken by registration.

### 6.2 The cycle-time honesty note

The FPGA runs the tile at ~100 MHz, 10 ns per cycle. A real photonic shot is 1–5 ns — *sub-cycle* — and a thermo-optic weight program is 10 µs to 1 ms. So the emulation cannot represent absolute photonic throughput at all, and must not be reported as doing so.

What it represents exactly is the **ratio**. Set `PTA_TS` to its floor and `PTA_TW` to `Tw_real × f_fpga` — at 10 µs and 100 MHz, 1,000 cycles — and the whole architecturally interesting range of that ratio, from 1 to about 10⁵, is reachable by sweeping one register. The sweep is the experiment; the absolute cycle counts are not a claim about anything.

---

## 7. What this does to GRXCP

Two of this repository's own rules are implicated, and both need an answer before phase C1 lands, not after.

**"Every sanctioned emulation is reported through a device property"** (`AGENTS.md` §3). A c930 running a PTA tile is not computing the GEMM the caller asked for; it is computing a noisy approximation of it. That is exactly the class of thing `warpShuffleIsEmulated` exists for. It needs a property — `gemmIsAnalogEmulated`, or better a small struct carrying the effective bits and the seed — and `grx-smi` should print it in the same "software stand-ins in effect" section that already exists for this purpose.

**"Every library kernel has a CPU reference and a numerical gate (bitwise or ULP-bounded); 'close enough' is not a gate"** (`AGENTS.md` §4). An analog device breaks this as written, and the temptation will be to loosen the rule into a tolerance, which destroys it.

The resolution that keeps the rule intact is two gates rather than one loose one:

1. **Bitwise against the model.** The model is deterministic given `PTA_SEED`, so `grxblasGemmEx` on a PTA-enabled c930 has an exact expected answer and the existing gate applies unchanged. This is the gate that catches bugs.
2. **Distributional against fp32.** Separately, and reported rather than gated at zero tolerance: error distribution versus the fp32 reference, with a stated bound that moves as the error model is tuned. This is the number that goes in the paper.

Conflating them produces a gate that is neither. Keeping them apart means a model change moves gate 2 visibly and leaves gate 1 red until the golden data is regenerated as a reviewed step — which is precisely how `ci/check_perf.py` already treats baselines.

`heterogeneous_devices.md` §6 asks which engine `grxblasGemmEx` picks when both can do the work, and answers "current-device-only, because it makes the choice the caller's." A PTA c930 sharpens that: the two devices no longer compute the same function, so automatic selection would silently change a program's numerics. Current-device-only stops being the conservative default and becomes the only defensible one.

---

## 8. Proposed but not yet implemented

Recorded so the next reader knows what was considered and deliberately deferred.

1. **An embedded control core inside the tile.** The source analysis suggests replacing the calibration FSM with a RISC-V core so calibration policy is software. Correct for a product, wrong for this phase: it puts a second toolchain and a second binary between a change and a measurement, and the calibration algorithms are small. Revisit if §5.1's scheduler comparison turns into a search over many policies.
2. **CXL, coherent access, an LSU replacing the DMA.** A large share of the source analysis. It is orthogonal to everything measurable here: the c930's AXI4 master already feeds the tile faster than a thermally-limited weight bank can consume, so coherence changes no result this program can produce. It belongs to the GRXIConnect work, not here.
3. **Multiple weight banks beyond two.** §5.5's hypothesis may make weight-set capacity the interesting axis, in which case `N` banks and a bank-allocation policy is the follow-on. Not before the `Tw`-to-`Ts` ratio sweep says so.
4. **A second tile.** The grx930 team's notes observe 41.5% LUT headroom is "enough for a second NPU tile." Two tiles with independent weight sets is how a real machine hides `Tw` completely, and it is the obvious C5. It is deferred because a two-tile result is uninterpretable until the one-tile loop nest is right.
5. **Crosstalk topology.** The model uses nearest-neighbour coupling on the column index, which is what a linear microring bank looks like. An MZI mesh couples along its triangular structure instead, and the two are not the same matrix. Since no mesh is being built, the topology is a parameter with no ground truth; the model should carry a pluggable coupling matrix and the document should keep saying it is a hypothesis.

---

## 9. Open questions

1. **Does the interchange survive real B-tile reuse patterns?** The §2.1 model assumes `M` rows all use the same B tile, which holds for a dense GEMM and not for a batched or strided one. The DMA's address generation may need a second traversal order, and that is not in the C2 estimate.
2. **Where does the activation scale factor come from?** §5.2 makes it a software quantity, but nothing in the c930 firmware computes one today. Is it static (from the model), measured (a calibration pass), or adaptive (from `PTA_SAT_CT`)? The adaptive answer is the most interesting and the one most likely to oscillate.
3. **Is bitwise RTL↔C parity actually achievable through the DPI wrapper?** The NPU DPI path exists for grxcp backend testing, but nothing currently crosses it that is sensitive to bit-level arithmetic ordering. If the floating-point accumulator paths cannot be made to agree exactly, gate C1(b) has to weaken, and it should weaken deliberately rather than by discovery.
4. **What is the right small model for the accuracy sweep?** It needs published quantization curves to compare against, small enough to run in RTL simulation, and dominated by GEMM. A two-layer MLP on MNIST is defensible and boring; anything transformer-shaped will not fit the `MAX_N = 8` output width without tiling that muddies the measurement.
