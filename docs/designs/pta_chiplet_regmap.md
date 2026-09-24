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

---

## 1. Where the window sits

- A BAR of the GPU's CXL.io function (B7). The GPU's driver owns it, and the
  CPU reaches the PTA through the GPU rather than through a link of its own.
- One window per PTA instance, on a 4 KiB stride, so a window is a page. B4 puts
  one chiplet on the board; the stride is there so a second needs no new map.
- Registers are 32 bits, naturally aligned, little-endian. Reserved bits read
  zero and are written zero.
- **Offsets 0x040–0x0D0 are the CPU document's block, unchanged**, so one driver
  can address a c930 tile and a chiplet with the same offsets. What the c930
  uses 0x000–0x03C for, the chiplet uses for identity and interrupts; 0x0D4–0x0DC
  are the calibration engine's own configuration, which C3(b) found neither map
  had (§4); 0x0E0 up carries the upper halves of the counters and, at 0x0F0, the
  error a calibration found.

---

## 2. Identity and capability, new

| Offset | Name | Access | Description |
|---|---|---|---|
| 0x000 | `PTA_ID` | R | Magic, then the version of this map |
| 0x004 | `PTA_CAPS0` | R | The tile: rows `k`, columns `n`, `DIN_W`, `ACC_W` |
| 0x008 | `PTA_CAPS1` | R | Weight banks; impairments built, in `PTA_IMPAIR`'s bit order; the widest activation, weight and ADC settings the hardware accepts |
| 0x00C | `PTA_CAPS2` | R | Shot rate as built, in MHz; whether the calibration engine and the activation stage are present; whether this is silicon or the twin |

A driver reads the geometry rather than assuming it. X2 sized the link against
candidate geometries precisely because the real one is not settled (the board
plan's §8), so the map has to carry it, and `heterogeneous_devices.md`'s rule —
no invented device numbers — applies to a chiplet as much as to a GPU. grxcp's
device property (the PTA plan's D2, step S4) is filled from these four
registers, and `PTA_CAPS2`'s emulation bit is what makes the twin honest under
`AGENTS.md` §3.

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

**Counters are 64 bits.** A 32-bit `PTA_SHOT_CT` wraps in 4.3 seconds at one
shot a nanosecond, which is inside X2's candidate range. The low half keeps its
§3.1 offset; reading it latches the high half, which is read next:

| Offset | Name | Access | Description |
|---|---|---|---|
| 0x0E0 | `PTA_SHOT_CT_HI` | R | Latched when `PTA_SHOT_CT` is read |
| 0x0E4 | `PTA_WLOAD_CT_HI` | R | Latched when `PTA_WLOAD_CT` is read |
| 0x0E8 | `PTA_SAT_CT_HI` | R | Latched when `PTA_SAT_CT` is read |
| 0x0EC | `PTA_CAL_CYC_HI` | R | Latched when `PTA_CAL_CYC` is read |

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
| 0x0D4 | `PTA_CAL_CFG` | RW | [3:0] probe amplitude, `1 <<` this, bounded by `DIN_W - B_a` and `DIN_W - 2`; [7:4] repeats a pass, `1 <<` this; [9:8] auto-ranging passes; [10] the bank to calibrate |
| 0x0D8 | `PTA_TRIM` | RW | [3:0] the weight DAC's step below the weight code, `1 <<` this in Q.8 weight LSB; [31:16] its clamp, Q.8 weight LSB |
| 0x0DC | `PTA_CAL_SEED` | RW | the calibration's noise seed. Calibration *j* draws from `PTA_CAL_SEED ^ (j · 0x9E3779B1)`, `j` being `PTA_CAL_CT` before it runs, so no two draw the same noise and any of them can be reproduced from one word |
| 0x0F0 | `PTA_ERR_FOUND` | R | the error the last calibration **found**, Q.8 weight LSB: the widest correction its first pass had to make, before any of it was applied. This, not `PTA_ERR_MAX`, is what the drift-predictive scheduler extrapolates — see [`pta_chiplet_calibration.md`](pta_chiplet_calibration.md) §5 |

Every field is a power of two or a log2 of one, which is not tidiness: it is what
lets the estimator divide with a shift and the DAC round with a mask, so the tile
carries no divider (grx930's design note §4).

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

---

## 7. Open

1. **The geometry and shot rate** that `PTA_CAPS0` and `PTA_CAPS2` report.
   Board plan §8, question 1.
2. **Whether `PTA_TW` and `PTA_TS` mean anything on silicon**, or stay the
   twin's.
3. **The chiplet's command queue depth**, which the GPU's dispatcher has to
   know.
4. **How many MSI vectors** the function offers, and whether the PTA shares the
   GPU's or has its own.
5. **The seed function** — `gemm_seed()`'s exact form is in the C1 harness; it
   becomes normative here the moment silicon implements it.
