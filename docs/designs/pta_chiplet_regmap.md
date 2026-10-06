# The PTA chiplet's register map, over CXL.io

**Companions:** [`board_program_plan.md`](board_program_plan.md) (B4, B7, X4),
[`board_icd.md`](board_icd.md) (link 2),
[`pta_cpu_integration.md`](pta_cpu_integration.md) §3.1 and §3.2, and grx930's
`c930/doc/pta_error_model_design_note.md` §4, which gives every field its
meaning.

**Status: X4 of the board plan, drafted 2026-09-22.** The CPU document's §3.1
block, placed in the GPU's BAR and reached over CXL.io. The fields keep their
names, offsets and meanings. What changes is everything a link makes different:
how the device is discovered, how wide its counters have to be, how completion
is observed when two reads can straddle a change, and what happens to a command
the device cannot honour.

*X5 built the twin of this map on 2026-10-04* (§6). Building it found things the
map does not say, and §4, §5 and §7 carry them where they belong.

*X6 specified the activation stage the same day* (§8), and built it into the
twin. It is the last section because it was §7's last open item, and §7 keeps
its number so that references to it still land.

---

## 1. Where the window sits

- A BAR of the GPU's CXL.io function (B7). The GPU's driver owns it, and the
  CPU reaches the PTA through the GPU rather than through a link of its own.
- One window per PTA instance, on a 4 KiB stride, so a window is a page. B4 puts
  one chiplet on the board; the stride is there so a second needs no new map.
- Registers are 32 bits, naturally aligned, little-endian. Reserved bits read
  zero and are written zero.
- **Offsets 0x040–0x0D0 are the CPU document's block, unchanged**, so one driver
  can address a c930 tile and a chiplet with the same offsets. *C4(a) built the
  c930's half of that:* its window is `0x4000_0100`, and the block inside it is
  laid out exactly as this one, so the offsets below are the same on both and only
  the base differs. Not `0x4000_0040`, which is the second NPU's — see that
  document's §3. What the c930
  uses 0x000–0x03C for, the chiplet uses for identity and interrupts; 0x0D4–0x0DC
  are the calibration engine's own configuration, which C3(b) found neither map
  had (§4); 0x0E0 up carries the upper halves of the counters and, at 0x0F0, the
  error a calibration found.

*Proposed otherwise, 2026-10-04.* grxcp's proposal to grxgpu for the chiplet's
host path (`grxgpu/docs/proposals/pta_chiplet_host_path.md`) asks for this
window as a range of the GPU's DCR addresses,
`base + instance × stride + (offset >> 2)`, reached by the driver's existing
register commands, and offers the page above as the alternative. The offsets and
everything in §2 to §4 are the same either way. If grxgpu takes the range, the
first two bullets here change, and §5's ordering rule comes from the GPU's
command ring instead of from CXL.io. Nothing here has been changed on the
strength of a proposal.

---

## 2. Identity and capability, new

| Offset | Name | Access | Description |
|---|---|---|---|
| 0x000 | `PTA_ID` | R | [31:8] the magic `0x505441`, "PTA"; [7:0] the version of this map, 1 |
| 0x004 | `PTA_CAPS0` | R | The tile: [9:0] rows `k`, [19:10] columns `n`, [25:20] `DIN_W`, [31:26] `ACC_W` |
| 0x008 | `PTA_CAPS1` | R | [6:0] impairments built, in `PTA_IMPAIR`'s bit order; [15:8] weight banks; [19:16], [23:20], [27:24] the widest activation, weight and ADC settings the hardware accepts |
| 0x00C | `PTA_CAPS2` | R | [15:0] shot rate as built, in MHz; [16] the calibration engine and [17] the activation stage are present; [19:18] the tile, 0 none, 1 word-serial, 2 broadside; [24:20] the log2 of the operands the activation stage can hold (§8); [31] this is the twin and not silicon |

A driver reads the geometry rather than assuming it. X2 sized the link against
candidate geometries precisely because the real one is not settled (the board
plan's §8), so the map has to carry it, and `heterogeneous_devices.md`'s rule —
no invented device numbers — applies to a chiplet as much as to a GPU. grxcp's
device property (the PTA plan's D2, step S4) is filled from these four
registers, and `PTA_CAPS2`'s emulation bit is what makes the twin honest under
`AGENTS.md` §3.

*The fields above were given their bits on 2026-10-03, when the c930 built them*
(the PTA plan's S1). They are at the head of its block, `0x4000_0100`, laid out
as here, and three things came out of building them that this section did not
say.

**The magic is what a driver reads first, and it is not decoration.** A register
file older than the block decodes fewer address bits and aliases the block's
addresses onto its own registers, so `PTA_IMPAIR` reads back a matrix dimension.
A driver that finds no magic knows nothing, and has to say so rather than report
"no tile".

**`PTA_CAPS1`'s mask has to be the refusal's own constant.** On the c930 the core
drives the word from the same parameter its START refusal tests, and a bench
tries each of the seven bits alone and holds the word to which of them ran — all
seven refused on the digital array, MZM_NL alone on a tile. A capability word
kept in step with the hardware by hand is a second rule, and second rules drift.

**Three fields mean something narrower on an emulated tile than they will on
silicon.** The widest settings are the widest that *quantise*: the c930's
quantiser takes a setting at or above `DIN_W` as unquantised rather than refusing
it, so the field reports `DIN_W - 1`, capped by the four bits `PTA_BITS` has. The
shot rate reads zero, because an emulated shot is `PTA_TS` core cycles and has no
rate of its own. And `PTA_CAPS0` reports the systolic array's geometry on a build
with no tile, where `PTA_CAPS1` says nothing is built; a driver reads `PTA_CAPS1`
before it believes there is a tile to have a geometry.

*A fourth tile kind, 2026-10-03.* `PTA_CAPS2[19:18]` reads 3 on grx930's register
model when it is built with the reference error model behind it (the PTA plan's
S2): the model on its own, its arithmetic with no tile's timing. No RTL build
reports 3. The board plan's X5 twin is the same kind of object.

---

## 3. Interrupts, new

| Offset | Name | Access | Description |
|---|---|---|---|
| 0x010 | `PTA_IRQ_STATUS` | RW1C | bit0 CAL_DONE, bit1 ERR, bit2 DRIFT_ALARM, bit3 SAT_THRESHOLD |
| 0x014 | `PTA_IRQ_MASK` | RW | One mask bit per status bit |

Delivered as MSI through the GPU's function, which is why the board plan asks
grx930 for AIA's message-signalled interrupts (§4.1). Polling stays valid and
is what the twin supports first; the interrupt exists because calibration takes
long enough that polling it across a link wastes the link.

---

## 4. The block itself, 0x040–0x0F0

Unchanged from the CPU document's §3.1, and restated here only so that this
document is a map rather than a diff: `PTA_CTRL`, `PTA_STATUS`, `PTA_IMPAIR`,
`PTA_BITS`, `PTA_SEED`, `PTA_SIGMA_TH`, `PTA_SIGMA_SH`, `PTA_SIGMA_PR`,
`PTA_DRIFT`, `PTA_XTALK`, `PTA_TW`, `PTA_TS`, `PTA_CAL_PER`, `PTA_CAL_THR`,
`PTA_CAL_CT`, `PTA_CAL_CYC`, `PTA_SHOT_CT`, `PTA_WLOAD_CT`, `PTA_SAT_CT`,
`PTA_ERR_MAX`, `PTA_GAIN[j]`, `PTA_OFFS[j]`, `PTA_DRIFT_MAX`.

Four of them behave differently behind a link.

**`PTA_STATUS` gains bit4, BUSY.** On the c930 the dispatcher's BUSY lives in
the NPU's own CSRs. On the board the engine that issues work is the GPU's and
the tile is across a link, so the chiplet publishes its own BUSY beside
CAL_BUSY. One read of `PTA_STATUS` is then a consistent snapshot of BUSY,
CAL_BUSY, CAL_VALID, SAT_STICKY and DRIFT_ALARM. §5 explains why that matters.

**`PTA_SEED` is written once, not per GEMM.** The contract reloads every
per-GEMM stream at a GEMM start, and on the c930 the host wrote the seed each
time. At the chiplet's rates that would be a write per GEMM across the link —
X2 counts 65,536 shots in a single 4096-square layer at one geometry, and a
GEMM is far shorter than that. So the chiplet derives each GEMM's seed from
`PTA_SEED` and its own GEMM counter, exactly as `pta_mnist.c`'s `gemm_seed()`
does in the C1 harness, and publishes the counter so a run stays reproducible:

| Offset | Name | Access | Description |
|---|---|---|---|
| 0x018 | `PTA_GEMM_CT` | R | GEMMs started since the last `PTA_SEED` write; the seed of GEMM *i* is a stated function of `PTA_SEED` and *i* |
| 0x01C | `PTA_ACT_CLIP_CT` | R | Outputs the activation stage clamped, since the last GEMM start (§8). Zero after a GEMM that did not ask for the stage |

*The function, as X5 built it.* The seed of GEMM *i* is the upper half of one
step of SplitMix64 from the state `PTA_SEED · 2³² + i`:

```c
uint64_t z = (((uint64_t)PTA_SEED << 32) | i) + 0x9E3779B97F4A7C15;
z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
seed_i = (uint32_t)((z ^ (z >> 31)) >> 32);
```

That is `gemm_seed()` of grx930's `c930/sim/pta_mnist.c` to the letter, and the
twin's gate pins it to three values computed from that source. Two things follow
that this section did not say. **A refused GEMM takes no index**: the count is of
GEMMs *started*, so the GEMM behind a refused one is the next in the sequence and
not the one after. And **`PTA_SEED` alone does not reproduce a result**: GEMM 0
under seed *s* runs on `seed_0`, not on *s*. A report of what a GEMM ran on needs
its index as well, which the CPU document's D2 property does not carry — it was
built for the c930, where the host writes the seed each time. *S4 added it,
2026-10-04:* `grxAnalogGemm_t.gemmIndex`, and a gate that reproduces a chiplet's
GEMM from the property with it and fails to without it.

**Counters are 64 bits.** A 32-bit `PTA_SHOT_CT` wraps in 4.3 seconds at one
shot a nanosecond, which is inside X2's candidate range. The low half keeps its
§3.1 offset; reading it latches the high half, which is read next:

| Offset | Name | Access | Description |
|---|---|---|---|
| 0x0E0 | `PTA_SHOT_CT_HI` | R | Latched when `PTA_SHOT_CT` is read |
| 0x0E4 | `PTA_WLOAD_CT_HI` | R | Latched when `PTA_WLOAD_CT` is read |
| 0x0E8 | `PTA_SAT_CT_HI` | R | Latched when `PTA_SAT_CT` is read |
| 0x0EC | `PTA_CAL_CYC_HI` | R | Latched when `PTA_CAL_CYC` is read |

*As X5 built them.* `PTA_SAT_CT` restarts at a GEMM start, as the c930's does, so
`SAT_STICKY` means "the last GEMM saturated, or a calibration since it" and lasts
one GEMM. Its upper half is in the map and reads zero until one GEMM saturates
four billion times. `PTA_SHOT_CT` counts a calibration's probe shots with a
GEMM's; `PTA_WLOAD_CT` counts a GEMM's programmings and not a calibration's own
weight writes, which do not pass through the core's weight-load state.
`PTA_CAL_CT` has no upper half here and needs none.

**`PTA_TW` and `PTA_TS` are the emulation's.** They set the modelled settle and
shot latency on the twin. What they mean on silicon — a read-back of what the
hardware does, or nothing at all — is open (§7).

**Three words the engine needs, which C3(b) found missing.** Building the
calibration engine in grx930 (`c930/rtl/pta/c930_pta_cal.sv`) turned up
configuration this map had nowhere to put: the probe's own parameters. `PTA_CTRL`
selects a scheduler and `PTA_CAL_PER`/`PTA_CAL_THR` tune two of them, but nothing
said how hard to drive the probe, how many times, or what the weight DAC can
hold — and the engine cannot be written without all three. The CPU document's
§3.1 needs the same words; where they land in its decode is C4's question, since
`0x0D4` upward is spoken for there by S_ACT's scalars.

| Offset | Name | Access | Description |
|---|---|---|---|
| 0x0D4 | `PTA_CAL_CFG` | RW | [3:0] probe amplitude, `1 <<` this, bounded by `DIN_W - B_a` and `DIN_W - 2` (see the note below: nothing here reports `DIN_W`); [7:4] repeats a pass, `1 <<` this; [9:8] auto-ranging passes; [10] the bank to calibrate |
| 0x0D8 | `PTA_TRIM` | RW | [3:0] the weight DAC's step below the weight code, `1 <<` this in Q.8 weight LSB; [31:16] its clamp, Q.8 weight LSB |
| 0x0DC | `PTA_CAL_SEED` | RW | the calibration's noise seed. Calibration *j* draws from `PTA_CAL_SEED ^ (j · 0x9E3779B1)`, `j` being `PTA_CAL_CT` before it runs, so no two draw the same noise and any of them can be reproduced from one word |
| 0x0F0 | `PTA_ERR_FOUND` | R | the error the last calibration **found**, Q.8 weight LSB: the widest correction its first pass had to make, before any of it was applied. This, not `PTA_ERR_MAX`, is what the drift-predictive scheduler extrapolates — see [`pta_chiplet_calibration.md`](pta_chiplet_calibration.md) §5 |

Every field is a power of two or a log2 of one, which is not tidiness: it is what
lets the estimator divide with a shift and the DAC round with a mask, so the tile
carries no divider (grx930's design note §4).

**`PTA_CAL_CFG`'s amplitude is bounded by something this map does not report.**
*C4(a), 2026-09-24.* The bound `[DIN_W - B_a, DIN_W - 2]` is on the datapath's
operand width, and no register here carries it — `PTA_BITS` gives the quantiser's
bit counts, not the width they quantise into. So the same value is right on one
build and refused on another: 6 is right for grx930's eight-bit bench tile and
refused by its sixteen-bit SoC tile, which ends the calibration in two cycles
with `CAL_ERR` set and `CAL_CT` unmoved. Firmware can find a value the tile
accepts — the refusal is observable and `MODEL_RST` clears it, and grx930's
`sw/pta_test.c` does exactly that — but a driver searching for a number the
hardware already knows is a gap in this table, not a technique. Two ways to
close it: a read-only field for the datapath width, next to whatever else a
capability word should carry, or specify the amplitude *relative* to the width
(`DIN_W - 2 - n`) and let the tile do the arithmetic. The second costs no
register and cannot be read wrong; the first is more use to a driver that wants
to size anything else. Either is better than the search.

*Closed 2026-10-03, by the first.* `PTA_CAPS0` carries `DIN_W` (§2), which is also
what a driver needs to read the quantiser's limits. grx930's firmware still
searches, as a check on the register rather than in place of one: the search has
to stop at `DIN_W - 2`, and does.

---

## 5. The completion contract, behind a link

The CPU document's §3.2 is emphatic: the only correct test that a batch has
finished is queue occupancy zero **and** BUSY clear, and calibration must not
be able to falsify it. Three things change when the tile is across a link.

**A predicate may not span two reads.** Two reads of two registers can straddle
a state change, and across a link they are far apart in time. So the device
side of the predicate is one read of `PTA_STATUS`: BUSY, CAL_BUSY and the rest
in one snapshot. The queue side stays where the queue is, in the GPU's engine.
The driver's test is: the GPU's queue is empty, and one read of `PTA_STATUS`
shows neither BUSY nor CAL_BUSY.

**A command the device cannot honour is refused, not dropped.** The design note
has the model reset gated to idle, and the RTL ignores one that arrives while
the core is busy. Silence is untestable across a link, so here a `MODEL_RST`
written while BUSY or CAL_BUSY is set leaves the device unchanged and raises
`PTA_IRQ_STATUS.ERR`. The driver learns that its reset did not happen.

**Calibration queues work, it does not lose it.** §3.2's fix on the c930 widens
"cannot dispatch now" with `cal_busy` so a START during calibration goes to the
FIFO and occupancy stays honest. The chiplet keeps a command queue of its own
for the same reason: while CAL_BUSY is set, commands queue rather than
dispatching into a tile that is unavailable, and never complete silently. The
queue's depth is open (§7).

*As X5 built it: BUSY covers the queue.* With a queue on the chiplet a command can
be waiting there while nothing runs — for the whole of a calibration, at least —
and a `PTA_STATUS` showing neither bit would then read as finished. The c930
covers that case with a second register, `QUEUE_STAT`, and this section's first
rule is that the chiplet's predicate may not span two reads. So on the twin BUSY
is set while a command is running *or queued*. During a calibration with work
waiting it shows CAL_BUSY and BUSY together, where the c930 shows CAL_BUSY, BUSY
clear and an occupancy of 1. A calibration with nothing waiting still does not
set BUSY.

**Ordering.** Writes over CXL.io are posted and reads are not, so a read of any
register in the window orders behind the driver's earlier writes to it. That is
the flush before a launch. Configuration registers take effect at the next GEMM
start, as the contract's streams do, so changing one mid-GEMM is a driver bug
and not a race the device can resolve.

---

## 6. What the twin answers

X5's digital twin presents this exact map, with `pta_tile_model.c` behind it, so
drivers, grxcp and the dispatch model can be brought up before silicon (the
board plan's P2 gate). `PTA_CAPS2` reports it as the twin, and every emulated
behaviour is visible in a register rather than assumed.

*Built 2026-10-04:* `src/backends/pta_chiplet/pta_chiplet_twin.c`, with its gate
beside it and in tier 1 of `ci/build_mock.sh`. It is a register file, the
chiplet's own command queue, a clock and the calibration contract, in front of
grx930's `pta_gemm()` and `pta_cal_bank()`. None of the tile's arithmetic is
written there. Its geometry is the caller's to name. That was because §7's
first question was open, and it stays so because the gate runs four tiles. The
working geometry, 128 × 64 (the board plan's B10, as revised on 2026-10-05),
is one of them, and so is the 256 × 64 it was first settled at.

**What the gate holds**, in 237 checks at 4 × 4, 8 × 8, 128 × 64 and 256 × 64,
of which 55 are the activation stage's (§8):

- *The map.* Every section above: identity, the seed and its counter, 64-bit
  counters with a latched upper half, interrupts, the engine's three words, the
  completion contract, refusal, reset.
- *The model.* Every GEMM through the twin is `pta_gemm()` called directly, bit
  for bit, on a device the twin never sees, and every calibration is
  `pta_cal_bank()`. The clear case is held to an integer product computed
  without the model.
- *§1's claim that one driver addresses both.* The c930 backend's own reader,
  `npu_c930_read_analog()`, pointed at the twin through a change of base and
  nothing else, identifies it and reports its tile. Two GEMMs are then
  reproduced from what that driver read and `PTA_GEMM_CT`, and from nothing more.

Three twins that are each wrong in one way — a GEMM run on `PTA_SEED` itself, a
command taken into a calibrating tile, a `MODEL_RST` honoured under a running
command — are built by the same script, and the gate has to fail against each.

**The D3 network, through it.** The board plan's P2 gate asks that the network
run "bit-identical to `pta_mnist`'s C reference". grx930's harness reaches its
device through three of the model's functions — a reset, an ageing and
`pta_gemm()` — and, unedited, was compiled with those three redirected to
functions that reach the model only through the twin. It was then given every
evaluation of the accuracy budget of 2026-10-03 again: 44 settings on five
networks, each network on its own seed, 220 test-set passes, 1,865,160 GEMMs.
All 220 printed the line the harness had written then, byte for byte. That is
the 8 × 8 tile the budget was measured on. The reset is `PTA_SEED` and
`MODEL_RST`, each GEMM's configuration is written through the registers before
it, and completion is read from `PTA_STATUS`. Two of the 44 settings age the
tile first, by six minutes of drift and by an hour, which no register does:
those 10 runs go through `pta_twin_age()`, the model's own fast-forward, which
is the twin's in the way its clock is. It is not a CI gate: the harness and the
MNIST files are not in this tree.

**What the twin had to decide, because this map does not.**

| | The map | The twin |
|---|---|---|
| How work is issued | Nothing. This is the control window, and no document says what a command on link 2 looks like | A function call. A command is a whole integer GEMM, in `pta_gemm()`'s own terms: a stand-in, not a proposal |
| The queue's depth | Open (§7) | A build parameter. A full queue does not accept a command, and says so. No register reports the depth |
| BUSY | Per command | Running or queued (§5) |
| `PTA_GAIN[j]`, `PTA_OFFS[j]` | Eight words each | The eight. A tile of 64 columns has 56 no word reaches |
| `SAT_THRESHOLD` | An interrupt with no threshold to cross | Never raised |
| A calibration's refusal | `PTA_IRQ_STATUS.ERR` here; `PTA_STATUS` bit 5 on the c930, which has no interrupt block | `PTA_IRQ_STATUS.ERR` only. One driver reads it in two places |
| A host's trim write | No register, on either map (the CPU document's §3.1) | None. A saved calibration cannot be restored |
| Hours of drift | Nothing: the model clocks drift by shots, and says of its own fast-forward that the RTL has no such port | `pta_twin_age()`, beside the map and not in it. An idle twin does not drift |

**What it does not model**, each reading zero in the register that would say
otherwise: the calibration scheduler, so only `CAL_NOW` starts one
(`PTA_CTRL[6:4]`); the loop-order and residency modes (`PTA_CTRL[9:7]`); the
activation stage unless the build asks for one (`PTA_CAPS2[17]`, §8); a shot
rate (`PTA_CAPS2[15:0]`). A
calibration does not interrupt a GEMM, because `pta_gemm()` is one call. Its
timing is two formulas — a GEMM holds the tile for
`programmings × PTA_TW + shots × PTA_TS` cycles, a calibration for
`passes × repeats × (PTA_TW + rows × PTA_TS)` — and nothing else.

It was not, at first, a device grxcp could see. *S4 made it one on 2026-10-04*
(the board plan's §3.4): a GEMM-only device whose parent is its GPU, enumerated
only when a model is attached, with the D2 property read from this map through
`src/backends/pta_chiplet/pta_chiplet.cpp` and `grxblasGemmEx` routed to it.
That driver keeps its own transcription of the offsets above, and a test holds
it to the twin's. The driver's gate and S4's two, the device and its GEMM
through the runtime, ran on a 256 × 64 twin alone until the board plan revised
B10. Since 2026-10-05 each runs on 128 × 64 as well, the working geometry: 54
checks in the driver's where there were 26, and 44 and 49 on each tile in the
other two where there were 44 and 48 on the one.

---

## 7. Open

1. ~~**The geometry and shot rate**~~ that `PTA_CAPS0` and `PTA_CAPS2`
   report. Board plan §8, question 1. *The geometry is a working one since
   2026-10-05:* 128 rows and 64 columns, the board plan's B10 as revised that
   day from 256 rows, so `PTA_CAPS0` reads those on a chiplet built to the
   plan. `DIN_W` and `ACC_W` in the same
   register are not part of that decision. *The shot rate is a working one
   since the same day:* 1 GS/s, the board plan's B11, so `PTA_CAPS2[15:0]`
   reads 1,000 on such a chiplet. It still reads zero on an emulated tile (§2),
   and the twin still has no rate of its own.
2. **Whether `PTA_TW` and `PTA_TS` mean anything on silicon**, or stay the
   twin's.
3. **The chiplet's command queue depth**, which the GPU's dispatcher has to
   know. The twin takes it as a build parameter and reports it nowhere; the c930
   has a `QUEUE_MAX` register for the same number. *The host-path proposal would
   remove the question:* the GPU's command ring would be the queue, and the
   chiplet would be handed one command at a time.
4. **How many MSI vectors** the function offers, and whether the PTA shares the
   GPU's or has its own.
5. **The seed function** — written out in §4 now, and implemented by the twin
   as well as by the harness it came from. It becomes normative the moment
   silicon implements it, as this item has always said.
6. **What a command on link 2 is.** The map has no way to issue work and nothing
   else specifies one. It decides what the chiplet's queue holds, what BUSY
   counts, and how a refusal reaches whoever sent the command. *The host's half
   was proposed to grxgpu on 2026-10-04* (`grxgpu/docs/proposals/pta_chiplet_host_path.md`):
   one command, a GEMM with its operands in the GPU's memory, which ends done,
   refused or lost and reports the index it took. How the link frames it is
   still nobody's.
7. **The affine past eight columns.** `PTA_GAIN[j]` and `PTA_OFFS[j]` have eight
   words each. An index register and a data register would reach any width; so
   would a second page.
8. **`SAT_THRESHOLD`'s threshold**, which no register holds.
9. **Where a host writes a trim**, if a saved calibration is ever to be
   restored. Open on the c930's map too.
10. ~~**The activation stage's registers.**~~ *Closed by §8, 2026-10-04*, which
    says what it computes, and that a command configures it and no register
    does. What §8 leaves open is its own list. As this item stood:
    `PTA_CAPS2[17]` says whether the stage
    is present and nothing here says what it computes or how it is configured.
    The c930's has its own block, a breakpoint table and a requantisation, and
    B4's addendum put one on the chiplet. The board plan's S3 gives this a price:
    a two-layer network whose intermediate goes through the stage is 14.7 µs
    in one round trip, where with a launch on the GPU between the layers it is
    21.6 to 45.1. The host-path
    proposal has reserved the flag bits such a command would use and cannot
    define them until this map does. The twin does not model the stage either
    (§6), and those figures use a stand-in for the host's round trip.
11. **`PTA_CAL_PER` cannot hold the interval it is for** (2026-10-06). It is
    32 bits of cycles, as on the c930, where a cycle is the core's. Behind a
    shot clock of 1 GHz that is 4.3 seconds. The board plan's B15 calibrates
    version 2 every six minutes, which is 3.6 × 10¹¹ cycles and 39 bits, and
    the hour it replaced was 42. §4 widened the counters for this reason and
    left the period as it was. Either it gains an upper half as they did, or
    its unit becomes a power of two of cycles: a unit of 2¹⁶ cycles, 66 µs at
    1 GHz, holds 78 hours in 32 bits. Nothing here chooses. The twin stores
    the period and no scheduler in it reads it, so it does not show this.

---

## 8. The activation stage

*Specified 2026-10-04 and built into the twin the same day* (the board plan's
X6). B4's addendum put an activation stage on the chiplet so that one layer's
outputs need not cross the link to become the next layer's inputs. This section
says what it computes and how a command asks for it.

**What it computes.** For each of a GEMM's `M × N` complete sums, with the bias
of its output `n`:

```
v = max(sum + bias[n], 0)
v = round(v / 2^shift)              a half rounds up
a = min(v, 2^(bits-1) - 1)
```

That is the step grx930's accuracy harness takes on the host between a network's
layers (`c930/sim/pta_mnist.c`, `tile_batch`), and nothing more: a bias, a ReLU,
a rescale back to an operand, and the operand's clamp. It applies to complete
sums, after the last K tile has been accumulated, which is where grx930's S_ACT
note found a stage has to sit.

**The shift is the laser's too** (2026-10-05; the board plan's B5, at its
end). grx930's harness picks `shift` by a rule that lets one firing unit in
ten thousand reach the clamp. A command that asks for one bit less hands the
next layer operands twice as large: 0.65% of the units that fire clamp, D3
loses 0.02 of a point, and the next layer's sums are twice the size beside the
light they are read out of, which halves the laser the working tile needs.
Two bits less clamp 13% and lose a quarter of a point for nothing. The stage
does not change for it. `PTA_ACT_CLIP_CT` already counts what a shift clamps,
which is what a program choosing one would watch.

*That bit is MNIST's, 2026-10-05.* On two more data sets (the board plan's §8,
question 7) one bit less still halves the laser, and clamps 0.90% and 2.75% of
the units that fire. A second bit, which bought nothing above, pays on
Fashion-MNIST at every laser tried and clamps 10%. On MNIST inverted it clamps
22%, and costs at 8 times B5's laser and above. So the shift that is right is
a workload's, which is the case for its being a field of a command, and for
the counter.

**It is not S_ACT.** B4's addendum called the cost "a nonlinearity fixed in
silicon, which S_ACT has designed once already". S_ACT
(`npu_act_stage_design_note.md` in grx930) is an experiment on all-optical
activation: a transfer curve from a coupled-mode solver in a 1,025-entry table,
shot noise on the light entering the unit, a per-unit detuning, and a
requantisation standing in for an O-E-O reset. It has no bias. So it cannot take
the step between D3's layers, and D3 is the one network with a measured
accuracy. Two things S_ACT settled do carry over: where the stage sits, and
that what it does belongs to a command and not to a live register.

**A command asks for it, and no register does.** S_ACT's note records the
hazard: a live enable applies to whichever command dispatches next. Here the
stage's settings ride in the command that uses them and apply to that command
alone.

| In the command | Meaning |
|---|---|
| ACT | The results go through the stage. What comes back is operands, not sums |
| HOLD | And they stay on the chiplet, to be the next command's activations. Nothing comes back. Needs ACT |
| FROM_HELD | This command's activations are the ones held. It brings none |
| shift | 0 to 62 |
| bits | The operand's width, clamp included: 2 to `DIN_W` |
| bias | `N` values in the sums' own units, or none |

One command may set all three flags: a middle layer. A command that asks for
nothing gets sums, whatever the command before it asked. So grxcp's gap 7.39, a
stage somebody else left enabled changing what a GEMM returns, cannot arise on
the chiplet.

**Held operands are the next command's or nobody's.** Any command's start takes
them or discards them. An unrelated command between two layers leaves nothing
for the second, and so does a layer that was refused. A calibration between two
layers does not take them, because a host cannot keep the engine from running
one. A reset of the chiplet drops them. It is the strictest rule that still lets
a network run, and it is strict on purpose: a command can never run on what is
left of somebody else's network.

**What it refuses**, as §5 has a refusal: accepted, ended as refused,
`PTA_IRQ_STATUS.ERR` raised, nothing returned and nothing counted.

- Any of the three flags on a chiplet without the stage, or a flag it does not
  have.
- HOLD without ACT: sums are not operands.
- A shift or a width out of range.
- More operands to hold than `PTA_CAPS2[24:20]` says there is room for.
- FROM_HELD when nothing is held, or when what is held is not this command's
  `M × K`.

**What reports it.**

| Where | What |
|---|---|
| `PTA_CAPS2[17]` | The stage is built |
| `PTA_CAPS2[24:20]` | The log2 of the operands it can hold. 13 holds D3's hidden layer at a batch of 64; X2's 16.4 kB activation buffer is 14 |
| `PTA_ACT_CLIP_CT`, 0x01C | Outputs the stage clamped, since the last GEMM start |
| The command's result | The same count for that command, beside its ADC saturations |

**What it is held to.** The twin's gate holds the function to fifteen cases
worked by hand and 20,000 draws against the harness's three lines, and holds a
network kept on the chiplet to the same network brought out at every layer: on
an 8 × 8 tile through three layers and on 128 × 64 and 256 × 64 tiles at D3's
shape, with
every impairment the tile builds enabled, the last layer's sums agree element
for element with each other and with a device the twins never see. Two twins
that are each wrong in one way fail it: one whose shift truncates, and one
whose held operands outlive the command after them.

And it is held to grx930's harness itself, compiled into a program beside the
twin (`pta_mnist_act_via_twin.c`; grx930 at `838c7cd`, 2026-10-04, not in CI).
The harness cuts D3's shape into 54 GEMMs and takes its step on the host; the
twin is given two commands with the hidden layer held. On four random networks,
of one and three hidden layers, every operand a hidden layer hands on and every
sum out of the last layer is the same: 6,400 and 640 at D3's shape. Two are the
exact product on a 256 × 64 tile, and two have the three quantisers on, on the
harness's own 8 × 8 tile, where a shot sees the same operands however the work
is cut. **With noise the two are different runs and are not compared**: each
GEMM draws from its own seed, and they cut the work into different GEMMs.

**What it leaves open.**

1. **Its time.** The twin adds none. It takes the stage to sit in the shot's own
   pipeline, one unit a column, at the shot rate. A stage that took a beat an
   output would spend 6.4 µs on D3's hidden layer at a batch of 64 and 1 GS/s,
   against 3.74 µs for both of the network's GEMMs (the board plan's S3). So
   "one unit a column" is a requirement on the interface chip, stated here and
   not measured anywhere.
2. **Any other nonlinearity.** ReLU is what D3 uses. S_ACT's table is how
   another would be carried, and nothing has asked for one.
3. **How it crosses link 2.** The host-path proposal to grxgpu reserved two
   flag bits, for FROM_HELD and HOLD, and defined neither. It needs a third for
   ACT and somewhere for the shift, the width and the bias. That is a further
   amendment, and it can now be written. *Written, 2026-10-04*
   ([kubeworkz/grxgpu#4](https://github.com/kubeworkz/grxgpu/pull/4)): the three
   flags, and a second version of the command's descriptor with the shift, the
   width, a bias vector's address and the clamp count. That is what the GPU is
   asked to carry. What the link itself carries is still nobody's.
4. **Operands wider than a byte.** `bits` goes to `DIN_W`, and the link model
   prices an operand at one byte.
5. **`sum + bias` past 64 bits.** The harness's addition is undefined there. The
   stage saturates. No sum a tile produces is near it.
6. **Whose turn is next, with more than one queue on the GPU.** "The next
   command" above is the chiplet's next. With one command queue on the GPU that
   is the host's next as well. With several, another queue's GEMM can land
   between two layers, and the second is refused. The amendment asks grxgpu to
   keep the tile for a queue whose last command is holding. Until something
   does, a host that shares a chiplet between queues cannot hold a network on
   it.
