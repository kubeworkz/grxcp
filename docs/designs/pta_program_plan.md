# PTA program plan: after the TFLT decision

**Companions:** [`pta_cpu_integration.md`](pta_cpu_integration.md),
[`pta_gpu_integration.md`](pta_gpu_integration.md),
[`pta_tpaqcn_review.md`](pta_tpaqcn_review.md).

**Status: PLAN, in progress.** The decisions of §2 were settled on 2026-09-14,
all four as recommended, and §7 lists the edits that carried them into the
integration documents. F0 is measured, F1 has made its predictions (§3.3), C0
is green, and C1 is closed (§3.1). The development board now has a plan of its
own, [`board_program_plan.md`](board_program_plan.md), which carries the fabric
requirements and the photonic chiplet; §6 of that plan lists what it changes
here.
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
| Error model (C1) | Closed: all six impairments built in grx930 and bitwise against the C reference; the accuracy sweep ran, missed its allowance at 3 bits, and the miss is recorded (§3.1) |
| Tile: PTM-B (C2 tile) | Landed and gated: `make core_c2`. §2.1's form holds; two of its constants and F1's predictions did not |
| Multi-bank tile, both loop orders (MB) | Landed and gated: `make core_mb`. §6.2's EO-res point is measured in both orders — the shipped one wins by 4.42×, against the 7.2× the table claimed |
| The shot's floor (2026-09-30) | Two cycles, not six: the hop came out of the broadside path. Every §2.1 term is an equality now, and EO-res got a third of its margin back |
| §6.2's sweep on the SoC (C4(c)) | Landed and gated: `make pta_sweep`. All four points measured through MMIO. §2.1's terms account for every total exactly; PTM-C's drain is 81–88% of the GEMM at the Pockels-class end |
| PTM-B SoC build (SoC-B) | Landed and gated: `make pta_sweep PTM_B=1`. §6.2's `Ts` axis is reachable — the shot moves with `PTA_TS` at §2.1's slope, and the Pockels-class end gets 3.8–4.8× faster |
| Feed options at the Pockels points (F2) | Landed and gated: `make pta_feed PTM_B=1`. None of the three options was the answer — 81% of the feed was the C write burst. Rebuilt, it is 1,283 → 260 cycles at F0's shape and the feed 333 → 145 at the SoC's. What binds now is the host's MMIO path, at 2.0× the GEMM |
| Calibration (C3) | Landed and gated, on both tiles: the engine has a probe per tile since 2026-09-30, so C3's apparatus runs where the Pockels-class numbers come from |
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
once so all three report on the same network. *Needed by:* C1. *Fixed in C1:*
a 784-100-10 MLP with ReLU, trained as Gorsline, Smith and Merkel trained the
network of their Fig. 3(c), the curve gate (a) compares against, but from a
16-bit start. C3 and A3 take its five networks at 8-bit operands and 6-bit
weights.

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
| C1 | **Closed, below.** Error model, one impairment at a time; `PTA_DRIFT` defaults fitted to TFLT, with a TFLN setting as the stress case | CPU document §6, unchanged | C0, D1–D3 |
| C2 tile | **Done, 2026-09-29.** PTM-B in the interchanged core: the broadside tile is `c930_ptm_c`'s arithmetic under `BROADSIDE = 1` with a shot-and-wait schedule around it, so the two variants agree by construction rather than by comparison. The gate measures each §2.1 term against the counter that makes it up — `o_stall_count`, `o_op_count / 64`, the write residual — and holds the scan, the restore and the write to equality at six shapes × 25 `(PTA_TW, PTA_TS)` points. It corrected §2.1 twice and re-derived F1 (below). The broadside calibration probe this row used to owe was built on 2026-09-30 (below) | `make core_c2`: total cycles match §2.1 at every runnable §6.2 point, affine in `PTA_TW` | C1 |
| MB | **Done, 2026-09-29.** The tile takes a `NUM_BANKS` parameter (two, the DMA's double buffer, is the default and changes nothing); the core can address a bank per (N tile, K tile) and skip the scan when they are already loaded; and the loop order is selectable, `m`-outer restored without the `S_PRELOAD` double-buffering it originally needed. A shape with more tiles than banks is refused, not aliased | `make core_mb`: EO-res totals match §2.1 in both orders with `Tw` at the RTL's bank-select cycle, and both orders return the same C from the same banks | C2 tile |
| C3 | **Done, 2026-09-23.** C3(a) measured the correction in the C reference — the board plan's X3 — and C3(b) built the RTL: the trim and the affine in PTM-C, the calibration engine and its four schedulers, and the `cal_busy` dispatch guard. Gates P7 to P9 | CPU document §6, run with drift at TFLT's fitted rate and again at TFLN's | C1, D3 |
| C4(a) | **Done, 2026-09-24, both halves.** The widened CSR decode, the PTA register block at `0x100` (not `0x40`: that is NPU1's window), and the two counters it reads and nothing produced. It cost three fixes outside the block: the CPU's M unit deadlocked against a store in MEM, C could not be cleared from the CPU, because the L2 stopped tracking a line the CPU wrote — since fixed, 2026-09-30 (below), and the probe amplitude is a bit position with no way to learn its bounds — CPU document §3.3 | `make npu` in both builds and `make pta_fw PTM_C=1`: the same seven checks over AXI-Lite and from a RISC-V program, plus `make mul_store` for the CPU fix | C3 |
| C4(b) | **Part done, 2026-09-24.** Vivado 2026.1 out of context on `xc7a200tfbg484-1`: the NPU baseline, the calibration engine, the CSR and S_ACT measured and placed against §6.1; §6.1's stated baseline shown to be no run's; A-synth's cut named, taken (`ACT_P` 7 → 8) and measured at 55.5 MHz. **Owed:** the tile's own row — `c930_ptm_c` and the PTM-C NPU both exceed this VM's memory | §6.1's table against what the tools say, and 100 MHz or a named pipeline cut | C4(a), A-synth |
| C4(c) | **Done, 2026-09-29, for the `Tw` axis.** MB's three modes reach the tile from `PTA_CTRL` bits 9:7, `arow_stall_cnt` reads back at `NPU_REG_AROW_CT`, and `sw/pta_sweep.c` drives all four §6.2 points from a RISC-V program through MMIO. EO-res is measured, not modelled, and §2.1's terms account for every total exactly. It cost two fixes in C4(a)'s own code: the DMA's core watchdog was sized from the shape alone and aborted TO-1ms as a hung core, and `PTA_WLOAD_CT` counted a level rather than an event so it read 3,004 programmings where the shape makes four. The `Ts` axis this row owed — the SoC built PTM-C, whose drain is fixed, so `PTA_TS` was inert on it — was discharged by SoC-B the next day (below) | `make pta_sweep`: every point runs, C exact at each, EO-res among them | C2 tile, MB, C4(a) |
| SoC-B | **Done, 2026-09-30.** The SoC's Verilator flags defined `PTM_C` and not `PTM_B`, so the file list carried `c930_ptm_b.sv` while the core still elaborated PTM-C. One define. `make pta_sweep PTM_B=1` now sweeps §6.2 on a broadside SoC and the drain becomes a shot: EO-scan 2,528 → 672 cycles, EO-res 2,337 → 487, and the range widens from 159× to 596×. The gate is the slope — the total moves by `M·Nt·Kt` = 32 cycles a unit of `PTA_TS`, 449 against the model's 448 from `PTA_TS` 2 to 16 | `make pta_sweep PTM_B=1`: the shot moves with `PTA_TS` at §2.1's slope, and the same run on `PTM_C=1` reports the register inert rather than passing quietly | C4(c) |

**The shot's floor, 2026-09-30.** §2.1 recorded the shot's six cycles as open
and named two suspects: a hop gate serving a de-skew the broadside tile does not
need, and a registered valid. Both were right, and the hop was the bigger of them.

Traced at `PTA_TS` = 1 the six were one cycle for the start strobe's register, three
for two hop edges — one waiting for the core's hop-gated feed to land, one aligning
the capture — one for an `S_DONE` state that did nothing but raise `o_valid`, and
one for that register. The three hop cycles and `S_DONE` were all there because PTM-C
emulates a half-rate systolic array and PTM-B was borrowing the machinery. A broadside
tile is not emulating one, so the hop came out of that path and four of the six went
with it. What remains is two: the register at the tile's input and the register at
its output, which do not go without making the tile combinational — and C4(b) settled
that direction, since the activation stage's Fmax is what the board plan's 100 MHz
rests on.

PTM-C is untouched, which is the whole safety argument: its nine error-model modes
still pass and its SoC sweep is byte-identical. What moved:

| | before | after |
|---|---|---|
| EO-scan, core bench | 46,592 | 40,448 |
| EO-res, interchanged | 44,608 | 38,432 |
| EO-res, `m`-outer | 16,896 | 8,704 |
| the two orders | 2.64× | **4.42×** |
| EO-res on the SoC | 487 | 388 |

Two things beyond the cycles. The shot is **deterministic** now — the hop's entry
parity used to move it a cycle either way, which is why C2 gated it as a bound and MB
gated its totals as bands; every term of §2.1 is an equality on this tile, held that
way on purpose because an equality is what would catch a hop dependence coming back.
And one of §6.2's two named `Ts` values is reachable for the first time: `Ts` = 5
is `PTA_TS` = 3. `Ts` = 1 still is not, the minimum being three.

**The L2 directory hole, 2026-09-30.** C4(a)'s second finding is closed. The L2
recorded a sharer on a read fill and, on a write-through with no allocate, dropped
the line's tag and sharer vector along with its data — while the writer's L1 kept
the line. So the directory stopped describing reality and a later write to that
line invalidated nobody, leaving the writer reading its own stale value. The rule
"firmware must not write a buffer the accelerator writes" had been living in two
firmwares' comments since.

Of the three candidates the CPU document listed, keeping the writer as a sharer is
what was built: it is the only one that adds no new hazard. Invalidating the
writer's own line would race its next read against the write-through still in
flight to DDR, and making the L1 write no-allocate reaches past this SoC. The two
validities are separate now — the directory's tag survives the data it no longer
has.

Keeping the entry opened a second hole, recorded because it is the same bug one
step removed: a read miss checks the data's validity, so it would allocate the
same tag into another way and split the line's sharers between two entries. A read
miss reuses a way that already holds its tag now, and the install ORs the reader in
rather than replacing.

Three tests, each failing without the fix: `tb_l2_coherent` T10 and T11, and
`make l2_coh` — a firmware reproducer whose control and read-only cases passed
while its written case returned the CPU's own poison, which is what identified the
directory rather than the GEMM. **One of the bench's own tests had been passing
because of the bug:** T5 wrote `0x30` and called it an untouched line, but the line
is 32 bytes and `0x30` shares it with the `0x20` the tests above it use. It passed
only while a write wiped the directory entry.

**The broadside calibration probe, 2026-09-30.** The engine's probe was PTM-C's —
`C_SHOT` walked `t` to `2R+2C−1` and captured one column per two steps, emulating the
staggered readout itself. A broadside tile answers every column on one hop edge, so
the walk read nothing, the residual came back zero, and both benches SKIPped the
calibration under `PTM_B`. That mattered more after SoC-B than before it: PTM-B had
become the build the Pockels-class numbers come from, so the speed story and the
accuracy story could not be measured on the same tile.

The probe is now four `t` steps instead of `2R+2C` — present the row, let the core's
hop-gated register take it, shoot, read every column off the one shot. At eight
repeats on the SoC, `PTA_CAL_CYC` falls from **14,980 cycles to 3,780**, and both
tiles report the same residual (`ERR_FOUND` 2,688, `ERR_MAX` 5,056). The probe phase
is eight times shorter; the zeroing and the estimator do not shrink, which is the
rest of the difference.

The agreement is about the device, not the dither. A broadside shot draws once per
column where the skewed one drew once per shot, so the noise realisations differ by
construction (E1 ties every draw to the loop order); at eight repeats that averages
out and what is left is drift and programming error, which the probe does not
perturb. That is the sense in which CPU document §4.2 says the two tiles must agree.

**One knock-on, and it is a test's and not the guard's.** `pta_test.c`'s T5 checks
that a START arriving during a calibration queues rather than dispatching into a busy
tile. Its only gap between `CAL_NOW` and that START is the descriptor — eight MMIO
writes — and the shorter calibration finished inside that gap, so `CAL_BUSY` was
already clear, nothing queued, and T5 failed while the guard it tests was intact. The
test now asks for eight repeats, which puts the race back where it is winnable on
either tile and averages the probe's noise into the bargain.

**Still not covered:** the core bench's `--pta engine` and `--pta sched` modes need
`--tile ptm_c`, so the broadside probe's only coverage is the iverilog bench and the
SoC firmware. Widening the core bench to drive the engine against PTM-B would give
the schedulers of §5.1 a second tile to run on, and nothing needs it yet.

**What SoC-B found.** Two things, neither of them the RTL's.

1. **The whole obstacle was a missing `-DPTM_B`** in the SoC's Verilator flags. The
   Makefile already added `c930_ptm_b.sv` to the file list and already put the build
   in `build/ptm_b`, so everything looked right while the core elaborated PTM-C. The
   sweep now derives the shot's cost per run from each total rather than assuming a
   tile, so a build that silently answers with the wrong one shows up as a flat 64
   cycles instead of hiding.
2. **The PTA firmware image had to be split per tile.** `pta_test.c` now takes
   `-DPTM_B` so it can skip the calibration test — the engine's probe walks PTM-C's
   staggered readout, which a broadside tile has none of, and `tb_c930_npu` already
   SKIPs it for the same reason. Both builds had shared `sw/pta_prog.hex`, so make
   reused whichever was compiled last and the other build reported the wrong tile's
   skips. There is an image per tile now, and the harness is told which to load.

   The skips print as SKIP rather than PASS. The firmware still sets their DIAG bits,
   because `RESULT` has to stay meaningful, but a test reporting PASS on something it
   did not do is worse than a gap that says so. Those skips are gone again: the
   broadside probe was built the same day (below).

**What C4(c) found.** The sweep's own numbers are in the CPU document §6.2. Two
of its results belong here because they change what other steps should expect.

**What binds the Pockels-class end is the tile's own drain, and the SoC cannot
show otherwise until it builds PTM-B.** `arow_stall_cnt` now reads back on a CSR
(`NPU_REG_AROW_CT`), and the A-row wait is **zero at every point** — the DMA keeps
this core fed at this shape, which is what F0 already measured on the hop core. With
that settled, §2.1's terms account for the totals exactly:

| Point | Total | Settle | Scan | Restore | Write | Drain | A-row | Model − measured |
|---|---|---|---|---|---|---|---|---|
| TO-1ms | 402,528 | 400,000 | 192 | 96 | 192 | 2,048 | 0 | 0 |
| TO-10µs | 6,528 | 4,000 | 192 | 96 | 192 | 2,048 | 0 | 0 |
| EO-scan | 2,528 | 0 | 192 | 96 | 192 | 2,048 | 0 | 0 |
| EO-res | 2,337 | 0 | ~0 | 96 | 192 | 2,048 | 0 | −3 |

The drain is 64 cycles a run because the SoC builds PTM-C, whose readout is the
array's skewed one. So it is **81% of the GEMM at EO-scan and 88% at EO-res** — and
it is exactly what PTM-B replaces with a shot.

**This corrects an earlier reading of the same measurement.** The first pass priced
the shot as PTM-B's `PTA_TS + 6`, found ~1,800 cycles left over, and attributed them
to the feed without the counter that would have checked it. The counter says zero.
The residual was the drain the model had mispriced.

Two consequences:

1. **`PTA_TS` is inert on the SoC.** The `Tw` axis is real — the thermo-optic points
   move with the register, 159× across the range — but the `Ts` axis needs PTM-B,
   which the SoC does not build. So C4(c) has swept half of §6.2 on the SoC and the
   other half is waiting on a **PTM-B SoC build**, which is not a large change: the
   core already selects its tile on `PTM_B`. That is the step to add.
2. **F2's brief is *not* what the first pass said.** There is no ~1,700-cycle feed to
   cut: the feed is zero here and the writes are 192 cycles of 2,337. F2's question —
   whether the host's feed binds the Pockels-class end — only becomes testable once
   PTM-B cuts the drain to a shot, because until then the fetch finishes in the
   drain's shadow. F2 therefore follows the PTM-B SoC build rather than leading it.

**What C2 tile found, beyond its own gate.**

**The shape is the SoC's, not §6.2's**, and that is not fixable at C4: this NPU is
`MAX_M = 8, MAX_K = 16, MAX_N = 12` and cannot be asked for `M = 64, N = 8,
K = 256`, the same gap §6.1 records about its baseline. So C4(c)'s absolute cycle
counts and MB's are not comparable; what the two share is the model, which holds at
both shapes. A SoC that could run §6.2's shape is a separate change with its own
memory cost, and nothing in the plan needs it.

**What C2 tile found, beyond its own gate.** All three are the model's, not the
core's, and all three are recorded in the CPU document:

1. **§2.1's `Tw` = 64 and `Td` = 8 are the full-tile values.** `S_WLOAD` walks
   only the `nc` columns and `kr` rows the tile uses and `S_WRITE` only the `nc`
   columns, so `Tw = nc·kr + PTA_TW` and `Td = nc`. Summed over tiles they reduce
   to `N·K` and `M·Kt·N`, which is why the form survives; a ragged `N` tile is
   *cheaper* than the table. CPU document §2.1.
2. **A shot costs `PTA_TS + 2` cycles**, the register at the tile's input and the
   register at its output. The floor is paid `M·Nt·Kt` times in both loop orders,
   so §2.1's resident row falls from **7.2× to 3.4×** — the verdict survives and
   most of the margin does. CPU document §2.1 and §6.2.

   C2 first measured this at **six**, and recorded the six as open: this handshake's
   cost rather than photonics'. Four of them were the array's— the core fed the tile
   on half-rate hop edges and the capture waited for one, which is what PTM-C needs
   to emulate a systolic array and PTM-B was borrowing. Retiring that on 2026-09-30
   also made the shot deterministic, so §2.1's terms are equalities rather than a
   band (below).
3. **F1's predictions were re-derived.** They took the core term as
   `interchanged(Tw, PTA_TS, Td)`, so its TO-10µs whole GEMM, 78,796 cycles, came
   out *below* what the core alone measures.
   [`pta_feed_model.py`](pta_feed_model.py) carries the floor and cross-checks its
   core term against C2's measurement, so the two models cannot drift again. The
   A-row wait falls as the total rises — a slower core gives PF1 more time — so
   both columns moved:

| Point, loop order | A-row wait, was → now | Whole GEMM, was → now |
|---|---|---|
| TO-10µs, interchanged | 385 / 574 → 259 / 448 | 78,796 / 79,118 → 82,766 / 83,088 |
| EO-scan, interchanged | 1,637 / 1,826 → 1,511 / 1,700 | 39,856 / 40,178 → 43,826 / 44,148 |
| EO-res, interchanged | 1,701 / 1,890 → 1,575 / 1,764 | 37,872 / 38,194 → 41,842 / 42,164 |
| EO-res, shipped | 0 / 0 | 4,427 / 4,560 → 8,523 / 8,656 |

   (NPU bench / SoC. The core term is the band's upper end; the hop's entry parity
   can take up to one cycle a shot off it.) F1's *structure* held — §3.3's race
   model still reproduces all four F0 measurements with no fitted constant, and
   that section is untouched — it was fed a wrong constant. What it says about F2
   is unchanged in direction: writeback and PF2 still lead, and by more than the
   first version had it — the feed's share at EO-res shipped is 22%, against 42% in
   that version and 11% at the shot's six-cycle floor. The shot is no longer the
   larger term; that floor came down on 2026-09-30 (below).

**Step MB, as built.** The select costs one cycle, which the gate counts as
`Tw`; the §2.1 break-even sits at 8, so EO-res stays in the resident regime.
Storage was small as expected — 2,048 weights at 32 banks. Measured at
`M = 64, N = 8, K = 256`:

| Order | Model | Measured |
|---|---|---|
| interchanged, resident | 38,432 | **38,432** |
| `m`-outer, resident | 8,704 | **8,704** |

**The shipped order wins by 4.42×**, and the model predicts both totals
exactly rather than bounding them — the shot has no hop left in it, so nothing
jitters. §6.2's EO-res row claimed 7.2×, which assumed `Ts` = 1; at the
floor of two, §2.1 gives 4.42× unfolded and that is what the core gives.
The verdict survives being built and most of its margin survives with it.

It read 44,608 against 16,896 and 2.6× until the shot's floor came down
on 2026-09-30. A fixed per-shot cost is a larger share of `m`-outer's smaller
total, so the floor had been masking this point's advantage rather than
reducing it evenly — which is the argument for having gone after it.

Three things this step found that the paragraph above did not expect:

1. **`S_PRELOAD` did not come back.** The plan said the `m`-outer FSM "does not
   need re-deriving: it is the core as it stood before grx930 commit 0c5df42".
   Most of it did not need restoring at all. That FSM used a bank swap and a
   preload state to hide the weight load behind the previous tile's run, and with
   resident banks there is nothing to hide — so `m`-outer came back as three
   changes to the shipped FSM (where `S_WLOAD` hands off, where the K loop
   advances, and where a write's end goes) and no new state.
2. **`m`-outer works without resident banks too**, which is §2.1's "as shipped"
   column as written: the weights are reloaded per (row, N tile, K tile), `M`
   times the traffic. Measured at `M = 4, N = 8, K = 32`: 1,152 cycles and 1,024
   cycles of weight movement, against the interchanged order's 576 and 352. So the
   mode is not tied to MB and C4(c) can sweep either order at any point.
3. **The orders agree about arithmetic, not about noise.** Both return the same C
   from the same banks with the error model off. With it on they are not expected
   to match bitwise, and this is by construction rather than by accident: E1 ties
   every draw to the core's loop order, so changing the order changes the draw
   sequence. Worth stating because "C bit-identical between orders" was this step's
   gate, and it holds in the sense that matters and cannot hold in the other.

One limitation is recorded rather than fixed: the calibration hook on a row
advance (§5.1's memory shadow, `cal_take_row`) stays on the interchanged order.
Its resume state is derived from that order's advance, and MB's gate needs no
calibration mid-GEMM, so it is gated off under `m`-outer rather than left to fire
into the wrong state.

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

The model is integer and covers the integer precisions only.

Four more were settled on 2026-09-15, again all as recommended, before drift
and crosstalk were built:

- *E5:* drift's clock is optical shots — one core run, one output row over one
  K tile — so it tracks the tile's work without bringing back the timing E1
  removed.
- *E6:* drift is device state, accumulating across GEMMs until a model reset
  (and, from C3, a calibration). It is the one exception to E2.
- *E7:* drift is clamped at a set bound, a new configuration field.
- *E8:* crosstalk couples neighbouring inputs within an output's bank, and an
  input outside the K tile holds no weight for its neighbour. This corrects the
  CPU document's §4.3 formula and §8 item 5.

**C1, closed.** Quantisation, thermal noise, shot noise,
programming error, drift and crosstalk are in PTM-C, and a C reference,
`c930/sim/pta_tile_model.c`, agrees with the RTL bit for bit: C and the ADC
saturation count match at every shape of the core-level harness, the sweep's
shape included, at both operand widths, for each impairment alone and for all
six together. Drift is device state, so its shapes run as one sequence on one
modelled device. With every impairment clear, C0's benches still pass. The CPU
document's §6 has the counts and the four ablations.

Gate (a), the accuracy sweep on the D3 network, ran on 2026-09-16 against a
criterion fixed before any network was trained. It was not met: at 3 weight
bits the five-network mean fell 0.61 points under the published curve, outside
the 0.5 allowed, while every other gated width was inside its band. The tile
gave exactly the digital networks' accuracy on 48 of 50 networks, and one image
different on the other two, so the miss belongs to the training, not the error
model; it is recorded, and C1 closes on it. The drift settings are fitted at
EO-res, run flat out: 80 M shots/s. At TFLT's fit, accuracy on the D3 network
holds within half a point for about an hour; at TFLN's, it loses about a point
in six minutes. That is what C3's schedulers are sized against. grx930's
design note, §5, has the tables, the ablation and the reported sweeps.

### 3.2 Track A — activation (grx930)

| Step | What | Gate | Needs |
|---|---|---|---|
| A3 | **Done, below, for its first stage.** Chain mode: reset interval × photons × detuning × curve shape | Reported, not gated, as specified: `make core_act_chain` and `sim/act_chain_sweep.py` in grx930, 400 points with every layer bitwise against the C reference and 64 excluded for a fault in A3's own harness, since found and fixed -- the transport exceeded the operand width; there is no RTL bug and A2's gate stands. The curve says the crossover is 10⁴ photons at the knee | A2 (done); D3 for its second stage |
| A-synth | **Measured, 2026-09-24: 41.2 MHz, not 100.** Routed out of context on the 200T. The limiting cone is stage 6 — three variable shifts, a clamp against bounds recomputed per element, and the saturation counter hanging off the end — not stage 1's wide multiply. Lifting the configuration-fixed parts out cost no latency and gave 51.2 MHz with 12% fewer LUTs; the cut for the rest is **taken** — stage 6 splits after the multiply, `ACT_P` 7 → 8, A2 bitwise unchanged with the saturation counts exact, and gate A1 down from 13 failures to 5 — and measured at **55.5 MHz**, the path moving to stage 1's wide multiply exactly as this plan's risk row predicted | 100 MHz on the 200T, or a named pipeline cut with `ACT_P` updated to match | — |
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
| F2 | **Done, below.** Feed options at the Pockels points: PF1 and PF2 as built; B tiles prefetched straight into resident banks; the whole GEMM staged before launch | `make pta_feed PTM_B=1`: nine batches at both points, two batch sizes each, every C exact including an odd `M·N` tail; the A-row wait is checked to be zero rather than assumed, and `STAGE_A` must lengthen the DMA or the bit is reported inert | C2 tile, MB |
| F3 | **Done, below.** Requirements for the fabric plan: operands per second at each §6.2 point, which operands are resident and which streamed, and how long the tile can wait. They are handed to the board plan's X2, which adds the die-to-die term | Delivered into [`pta_chiplet_link.py`](pta_chiplet_link.py) §1, every number traced: the core is §2.1 with C2's shot floor, the feed is F0's measured loads with F2's rebuilt write burst, and the per-row margin is F1's race with `S_AROW` as its check — zero exactly where the margin is positive | F1, F2, C4 |

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
unfolded as built (NPU bench / SoC) — **re-derived after C2 tile measured the
shot's six-cycle floor, which these numbers did not have; see §3.1**:

| Point, loop order | A-row wait | Whole GEMM | Feed share |
|---|---|---|---|
| TO-10µs, interchanged | 259 / 448 | 82,766 / 83,088 | 2.6% / 2.9% |
| EO-scan, interchanged | 1,511 / 1,700 | 43,826 / 44,148 | 7.7% / 8.4% |
| EO-res, interchanged | 1,575 / 1,764 | 41,842 / 42,164 | 8.2% / 8.9% |
| EO-res, shipped | 0 / 0 | 8,523 / 8,656 | 21.9% / 23.1% |

The shipped order still wins at EO-res once the feed counts: 4.9× unfolded
and 3.0× with the restore folded, against 5.8× and 3.4× for the core alone.
The feed's share is what has grown — 22% of a shipped-order EO-res GEMM,
against 11% when the shot's floor was six — because the same 1,900 fixed
feed cycles now sit beside a 6,656-cycle core rather than a 14,848-cycle one.
So writeback and PF2 lead F2's candidates by more than they did, and the shot
is no longer the larger term: it was cut from six cycles to two on 2026-09-30
(§3.1).

These predictions are left as they were made, which is the point of having
them. Two things in them have since been measured otherwise, and F3 found the
second: the feed is 844 cycles at this shape, not ~1,900, because F2 rebuilt
the C write burst; and the shipped-order EO-res core is 8,704, not 6,656,
because §6.2's `Tw = 0` at the resident point is an idealisation and MB
measured the bank select's own cycle. Both move the feed's share the same way,
down from 22% to 9%, and neither disturbs the verdict. The current numbers are
F3's table below.

**F2, measured.** `make pta_feed PTM_B=1` runs nine batches on a broadside SoC:
EO-scan and EO-res, each with the feed as built and with a new
`PTA_CTRL.STAGE_A`, each of those at `Q = 1` and `Q = 4`, plus the resident fill.
Two batch sizes because every per-GEMM counter — `CYCLE_LO`, `DMA_CT`,
`STALL_CT`, `AROW_CT` — resets on `START`, so a drained batch of four reports only
its last GEMM; `4·wall(Q=1) − wall(Q=4)` is then what queueing buys, measured
rather than modelled. The shape is this SoC's `M = 8, N = 12, K = 16`, the same
as C4(c)'s, so the two sets of numbers sit side by side.

The three options in F2's brief map onto this as: *as built* is `modes = 0`;
*B tiles prefetched straight into resident banks* **is** the EO-res point, since
EO-scan with `RESIDENT|WSKIP` would be EO-res by definition, so the two levels
span that option rather than it being a third axis; and *the whole GEMM staged
before launch* is `STAGE_A`, which takes the path INT4 has always needed —
nibble packing makes rows share bytes — at a precision that does not.

| Point | core | GEMM | feed | `S_AROW` |
|---|---|---|---|---|
| EO-scan, as built | 576 | 909 → **721** | 333 → **145** | 0 |
| EO-scan, `STAGE_A` | 576 | 937 → 749 | 361 → 173 | 0 |
| EO-res, as built | 388 | 721 → **533** | 333 → **145** | 0 |
| EO-res, `STAGE_A` | 388 | 749 → 561 | 361 → 173 | 0 |

**None of the three options was the answer.** `S_AROW` is zero at every point in
every mode, so PF1 never makes this core wait at this shape and staging cannot
win by removing a wait — it only adds the 28 cycles of 14 more A beats at the
read path's two cycles a beat, which is what it measures, exactly, at both
levels. Residency cuts the *core* (576 → 388, and weight movement 288 → 100
cycles, exactly the gap), not the feed, which is 333 cycles at both points.

**81% of that feed was the C write burst**, and it was five cycles a beat to put
8 bytes on a channel that takes 8 bytes a cycle. CPU document §2.4 rebuilt it;
the arrow in each cell above is that change. `make npu_feed` gives the clean
reading at F0's shape: **1,283 → 260 cycles**, the model's prediction to the
cycle, with the whole GEMM falling by exactly 1,023 and 256 AXI beats either way.
The fabric did take a beat a cycle — the feed's excess over the structural floor
was +30 cycles before and +33 after, so those are burst setup, paid once a GEMM.

**PF2 costs almost nothing, and the first attempt to price it was wrong.**
F2 began by inferring about 22 cycles a queued GEMM from `STAGE_A`'s refund:
`STAGE_A` turns PF2 off as well, so its `Q = 4` penalty should be 4 × 28 = 112
and measured +17 to +26, the difference being PF2's cost returned. That
arithmetic prices `STAGE_A` by its `DMA_CT` delta (+28) when its *wall* cost is
+12 — the rest hides behind the host's ~1,090 cycles of submit and poll — so
it subtracted quantities that do not subtract.

`PTA_CTRL.PF2_OFF` (bit 11, 2026-10-01) turns PF2 off on its own and measures it
directly: **+4 cycles over four GEMMs at EO-scan, −8 at EO-res**. Both sit
inside the harness's own floor, which the same run established: a batch with
extra batches ahead of it moved by up to **45 cycles** on the wall while every
counter stayed bit-identical, because the C block addresses and the cache state
travel with execution history. So this harness cannot resolve PF2 at all — and
neither could its "EO-res, `PF2_OFF` wins by 8" verdict, which is now reported
as the tie it is.

`tb_npu_feed.sv` can, from phase counters rather than a wall: a GEMM with a next
one queued takes `done = 4` against `done = 1`, with `drain_beats = 3`. **PF2
costs 3 cycles.** F0's recorded 145 was the old write burst's — with 1,283
cycles to run, PF2 issued several bursts and left many beats in flight when it
was abandoned; it now issues one `AR`, gets 29 of its 32 beats and leaves 3. So
rebuilding the write burst (CPU document §2.4) cut PF2's cost from 145 cycles
to 3 incidentally, while making it more pointless still: 29 of the 288 beats it
needs instead of 143.

So of everything the engine can now be asked for, **as built and `PF2_OFF` are
indistinguishable**, and `STAGE_A` loses by more than the floor. `PF2_OFF` is
not a performance win. What it buys is that the claim about PF2 is a reading
rather than a subtraction, and it gates as an exact no-op at `Q = 1`, where an
empty queue means PF2 never starts.

**What binds now is the host.** The wall clock is the CPU's, so `wall − DMA_CT`
is spent outside the engine, on the submit's MMIO writes and the drain's poll:
1,091 cycles a GEMM, and the write burst did not move it (1,086 before), which
is the check that it is really outside. At EO-res that is 67% of the wall and
**2.0× the whole GEMM**, more than the core and the feed together. Cutting the
feed 56% moved the wall 10%. F3 carries that as this SoC's MMIO path, not as a
fabric number — which is the distinction §3.3's preamble already insists on.

**F3, delivered.** The handoff lives in the board plan's own model,
[`pta_chiplet_link.py`](pta_chiplet_link.py) §1, because X2 is what consumes it. At
§6.2's shape, `M = 64 N = 8 K = 256`, 10 ns a cycle. Every core in it has
been measured at that shape — the three interchanged ones by C2, the resident
one by MB — and the model asserts against those readings rather than quoting
them:

| Point | Order | core | GEMM | shots/s | in GB/s | out GB/s | margin |
|---|---|---|---|---|---|---|---|
| TO-1ms | interchanged | 3,248,640 | 3,249,484 | 63,025 | 0.00 | 0.00 | 98,742 |
| TO-10µs | interchanged | 80,640 | 81,743 | 2,505,413 | 0.02 | 0.00 | –20 |
| EO-scan | interchanged | 40,448 | 42,803 | 4,784,711 | 0.04 | 0.00 | –24 |
| EO-res | shipped | 8,704 | 9,548 | 21,449,518 | 0.19 | 0.02 | **101** |

**Which operands are resident and which stream**, which is what decides where a
link's bandwidth goes. **B**, the weights: resident at EO-res in a bank per
(N tile, K tile) (step MB), scanned every tile at the other three — so at those
points B crosses `Nt·Kt` times a GEMM, not once. **A**, the rows: streamed one
row at a time during compute, at a beat a cycle once the burst is open. **C**:
once a GEMM, in one burst, now also at a beat a cycle. **The next GEMM's A and
B**: PF2 fetches these during C's writeback and does not finish — 29 of 288
beats — so a fabric should not count on it.

**How long the tile can wait** is the margin column, and it is the number F3
owed that F1 had not isolated. `S_AROW` says how long the core *did* wait; the
margin says how much extra per-row latency it would absorb before it starts.
Adding L cycles to every row's arrival shifts every landing by L, so the core
stalls when L passes the tightest row's margin. At EO-res with the shipped order
— the point TFLT targets — that is **101 cycles, 1,010 ns**, against the 100 ns
request-and-return X2 assumes, so the link has an order of magnitude in hand. So latency is not what binds the link; bandwidth
is. The two thermo-optic points have *negative* margin and the core already
waits there, which is F1's finding restated rather than a new one. The check
that this is the same race F1 built: `S_AROW` is zero at exactly the points
where the margin is positive, asserted at all five.

**It corrected the rate X2 was built on, by 2.2×.** The handoff first read
0.42 GB/s in and 0.05 out at EO-res; it is 0.19 and 0.02. Its EO-res GEMM was
4,427 cycles and is 9,548, and the difference is three separate things, two of
them pulling the same way:

- **The shot floor.** It took the core from §6.2's table, whose shot costs
  `PTA_TS` and nothing else, where C2 measured `PTA_TS + 2`. At 2,048 shots a
  GEMM that is 4,096 cycles.
- **`Tw` at the resident point.** §6.2's table writes `Tw = 0` there, as though
  selecting a bank were free. MB measured the select's own cycle, and `m`-outer
  programs once a shot, so that is another 2,048 cycles. This one is the easiest
  to miss, because `Tw = 0` is what makes EO-res look like the point of the
  exercise, and it is still the point — just not by as much.
- **The write burst**, which F2 rebuilt from five cycles a beat to one. This
  pulled the other way, by 1,023 cycles, which is how the first two stayed
  hidden: +6,144 against —1,023 nets to a GEMM a little over twice as long.

X2's verdict — that the c930's tile would never trouble a link — is unchanged
and holds more comfortably, since the tile is slower than the handoff claimed.
The margin moved the same way: `Tw` costs the core a cycle a shot, which is a
cycle more for PF1 to land a row in, so 69 cycles of slack became 101.

**What the handoff does not cover**, stated in it rather than left to be
discovered: these are the DMA's rates. On the measured SoC the wall is dominated
by the host, at ~1,090 cycles a GEMM against EO-res's 533-cycle GEMM, and a link
sized to this table still leaves that as the limiter. Whoever owns the chiplet's
command path has to size it separately, and it is not a number this program can
hand them — it belongs to whatever replaces `npu_drv_submit` over a fabric.

### 3.4 Track G — the G100 (grxgpu, by proposal)

Every step lands as a proposal in `grxgpu/docs/proposals/`, written with
grxgpu's RTL owners, under the boundary rule of `AGENTS.md` §2.

| Step | What | Gate | Needs |
|---|---|---|---|
| G1 | SimX study of the three weight-set policies, with bank count `W` beside `Tw` | GPU document §7, plus the `W` axis | D4 |
| G0 | `VX_tcu_fedp_analog`, vendoring the c930 error model | GPU document §7, unchanged | C1, D1 |
| G2 | Cluster-scope tile beside the DXA. On the board this becomes the chiplet attach (board plan, B4) | GPU document §7, with DXA transfer cycles reported as the feed term | C2 tile, MB, G1 |
| G3 | The same block-scaled GEMM on both tiles | GPU document §7, with the two feeds compared | G2, C4 |

### 3.5 Track S — grxcp

| Step | What | Gate | Needs |
|---|---|---|---|
| S0 | **Done, below.** The D2 property specified: its fields, what an NPU without a tile reports, and the `grx-smi` line | Review; no code. CPU document §7.1, which also states what S1 must do to satisfy it, so the specification is testable rather than agreeable | D2 |
| S1 | The property populated from the PTA CSRs, and the vendored DPI shim extended to answer on them | `AGENTS.md` §3: every field sourced or reported unknown (−1); the NPU BACKEND GATE in `ci/build_mock.sh` green | S0, C4's CSR map |
| S2 | The two gates: bitwise against the model, its golden data regenerated only as a reviewed step, and the distributional report | CPU document §7 | S1 |

**S0, specified.** CPU document §7.1. D2 had settled that the property is a
struct rather than a flag; S0 is its shape, and three things came out of writing
it down that the one-line brief did not contain.

**The no-tile case is three cases, not two.** A tile that is present with
`PTA_CTRL.EN` clear is not the same as no tile, and both are "not emulated". So
the struct carries `tileIsPresent` beside `gemmIsAnalogEmulated`, and a user
asking why their GEMM is not analog can tell *this build has no tile* from *you
did not enable it* — which is otherwise a silent difference, and the kind of
thing that gets diagnosed twice.

**In both inactive cases every other field is `-1`, not the CSR's contents.**
With `EN` clear the registers still read back whatever was last written, and
those values describe a model that is not running. Reporting them would invent a
provenance for a result that does not have one. `-1` for not-applicable is
already the house convention — `grxFuncAttributes` guards `numRegs` and
`ptxVersion` the same way — and it is why two of the fields are signed 64-bit:
a full 32-bit register value and `-1` both have to fit.

**The impairment mask needs a second mask beside it.** `PTA_IMPAIR` defines
seven bits and the tile implements six: bit 5, MZM_NL, has no phase in this
build, and a START with it set is *refused* rather than ignored (CPU document
§4.3). Without `impairmentsImplemented` a caller cannot tell a bit that is off
from a bit that cannot be on, and would read that refusal as a driver bug.
`impairments & ~impairmentsImplemented` is exactly the set that will refuse.

Two things S0 deliberately does not settle, both left where there will be code
to look at. Whether `grxblasGemmEx` should refuse a GEMM carrying an
unimplemented impairment or pass it down and let the START refuse it — the
second is what the hardware does today, the first duplicates a rule in two
places, and that is how two rules drift apart. And the wording of the
distributional report, which is S2's and depends on numbers C1 has not finished
moving.

---

## 4. Order

| Wave | Starts when | Steps |
|---|---|---|
| Now | Immediately, in parallel | A3; G1 (D1–D4 settled; F0 and C0 done). S0 and A-synth are done |
| Next | C0 green, as it now is | C1; G0 and C3 once C1 is green (F1 done) |
| Then | C1, C2 tile, MB, C4, SoC-B, F2 and F3 now green | G2 once G1 has reported. Track F is complete: F3's handoff is in the board plan's X2 §1, with the host's ~1,090 cycles labelled there as this SoC's MMIO path rather than a fabric rate |
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
| Drift at TFLN's fit costs about a point on the D3 network in six minutes, and at TFLT's half a point in an hour | C3 | Size the schedulers against both, and keep in mind that the fits rest on a shot rate (EO-res, flat out) and a 46-hour swing, not a measured drift walk |

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
