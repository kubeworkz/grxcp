# PTA first-testchip baseline for GF Fotonix

**Companions:** [`pta_chiplet_regmap.md`](pta_chiplet_regmap.md),
[`board_icd.md`](board_icd.md),
[`board_program_plan.md`](board_program_plan.md),
[`pta_cpu_integration.md`](pta_cpu_integration.md),
[`pta_chiplet_calibration.md`](pta_chiplet_calibration.md),
[`pta_chiplet_link.py`](pta_chiplet_link.py),
[`pta_shot_rate.py`](pta_shot_rate.py).

**Status: BASELINE v0, drafted 2026-10-16; v0.1 2026-10-16.** Freezes the chiplet interface
starting from the regmap and the ICD so GF Fotonix preparation has a fixed
target before the PDK lands. Fixes nothing the companions have not already
decided; every number below is traced to one of them, to a public GF
platform fact, or marked open. Resolves the scope line the PTA documents
carry ("FPGA emulation and numerics only, no PDK, no MPW"): this document
is the first one whose deliverable is a photonic die.

**Decisions taken as input.** TFLT Pockels target, heterogeneously
integrated; GPU+PTA package (G100 die + PTA chiplet over UCIe inside one
package, CPU over CXL 2.0 / PCIe 5.0 on the board); no licensed PDK yet.
Consequence: the first vehicle is GF-native silicon photonics only, with
TFLT as a bonded annex. Nothing tapes out waiting on a material GF does
not ship natively.

---

## 1. What GF Fotonix is, publicly

GF Fotonix (foundry designation 45SPCLO in access listings) is a monolithic
45nm SOI photonics-plus-CMOS platform: 300mm wafers, 9-level metal stack,
Ge photodiodes, Si and SiN waveguides, MZM and ring modulators, grating and
edge couplers, C-band and O-band operation. Platform data-rate claims (100G
per lambda first generation, 200G second) are system claims, not per-device
PDK specs — no Vpi, responsivity, heater power, or layout-rule value is
used from them here.

Qualified EDA, per vendor announcements: Cadence Virtuoso / Virtuoso
Photonics / Spectre, Synopsys photonic tools and IC Validator curvilinear
DRC/LVS, Siemens Calibre nmDRC/nmLVS photonics-aware decks, Ansys Lumerical
FDTD / MODE / CHARGE / HEAT certified for Fotonix. Access is GF Connect /
GlobalShuttle MPW, or EUROPRACTICE (listed 5x5mm general blocks, ~5-6
month turnaround, dummy fill by the customer) and CMC for eligible
academic runs.

TFLT/TFLN is not a native Fotonix layer in any public source. The
established path is heterogeneous: wafer bonding or micro-transfer-printing
of the electro-optic film onto the silicon circuit (imec has demonstrated
both TFLN and LiTaO3 transfer-printing onto silicon photonics, not
qualified on GF). So GF approval, contamination control, thermal budget,
alignment, post-process rules, and reliability are a co-development with GF
and the integration partner — tracked as open items in §7, not assumed.

---

## 2. Frozen interface (from the regmap)

The register contract below is frozen at v0. RTL, firmware, drivers, and
the twin implement exactly this; any change is a versioned edit with a
migration note, not a silent drift.

- **Window.** One 4 KiB page per PTA instance in the GPU's CXL.io BAR
  (board plan B7). GPU driver owns it; CPU reaches the PTA through the GPU.
- **Format.** 32-bit registers, naturally aligned, little-endian. Reserved
  bits read zero, written zero.
- **Offsets 0x040-0x0D0 are the CPU document's block, unchanged**
  (`pta_cpu_integration.md` §3.1): `PTA_CTRL`, `PTA_STATUS`, `PTA_IMPAIR`,
  `PTA_BITS`, `PTA_SEED`, `PTA_SIGMA_TH/SH/PR`, `PTA_DRIFT`, `PTA_XTALK`,
  `PTA_TW`, `PTA_TS`, `PTA_CAL_PER/THR/CT/CYC`, `PTA_SHOT_CT`,
  `PTA_WLOAD_CT`, `PTA_SAT_CT`, `PTA_ERR_MAX`, `PTA_GAIN[j]`,
  `PTA_OFFS[j]`, `PTA_DRIFT_MAX`. The c930 window base is `0x4000_0100`;
  only the base differs per target.
- **Identity (new on the chiplet).** `PTA_ID` 0x000 (magic + map version),
  `PTA_CAPS0` 0x004 (rows k, columns n, `DIN_W`, `ACC_W`), `PTA_CAPS1`
  0x008 (weight banks, impairments built in `PTA_IMPAIR` bit order,
  widest activation/weight/ADC settings), `PTA_CAPS2` 0x00C (shot rate as
  built in MHz, calibration-engine and activation-stage presence,
  silicon-vs-twin bit). grxcp fills its D2 device property from these four;
  the twin bit is the honesty mechanism.
- **Interrupts (new).** `PTA_IRQ_STATUS` 0x010 (RW1C: CAL_DONE, ERR,
  DRIFT_ALARM, SAT_THRESHOLD), `PTA_IRQ_MASK` 0x014. MSI through the GPU
  function; polling stays valid and is what the twin supports first.
- **Seed discipline.** `PTA_SEED` written once; GEMM *i*'s seed is a stated
  function of `PTA_SEED` and the published `PTA_GEMM_CT` (0x018), matching
  the C1 harness `gemm_seed()`. The exact function becomes normative at v1
  (§7).
- **Counters are 64-bit.** Low halves keep their §3.1 offsets; reading a
  low half latches the high half at 0x0E0-0x0EC (`SHOT`, `WLOAD`, `SAT`,
  `CAL_CYC`). A 32-bit shot counter wraps in ~4 s at 1 shot/ns — inside
  the candidate range, so this is required, not optional.
- **Calibration configuration.** `PTA_CAL_CFG` 0x0D4 (probe amplitude as a
  shift, repeats as a shift, auto-ranging passes, bank select), `PTA_TRIM`
  0x0D8 (weight-DAC step + clamp, Q.8 weight LSB), `PTA_CAL_SEED` 0x0DC
  (calibration *j* draws from `PTA_CAL_SEED ^ (j * 0x9E3779B1)`),
  `PTA_ERR_FOUND` 0x0F0 (widest first-pass correction found, Q.8 weight
  LSB — this, not `PTA_ERR_MAX`, is what the drift-predictive scheduler
  extrapolates). All power-of-two / log2 fields so the tile carries no
  divider.
- **Known gap, carried not hidden.** `PTA_CAL_CFG`'s amplitude is bounded by
  the datapath width `DIN_W`, which no register reports — the same value
  is legal on one build and refused on another. v0 keeps the firmware
  search (refusal is observable via CAL_ERR, cleared by MODEL_RST) and
  records the two candidate fixes: a read-only datapath-width field, or a
  width-relative amplitude encoding. One of them lands at v1.
- **Completion contract behind a link.** Driver test is GPU queue empty
  AND one read of `PTA_STATUS` showing neither BUSY nor CAL_BUSY — the
  device side of the predicate is a single read, never two reads that can
  straddle a change. `MODEL_RST` while BUSY/CAL_BUSY leaves the device
  unchanged and raises `IRQ_STATUS.ERR` (refused, not dropped).
  Calibration queues work rather than losing it; chiplet keeps its own
  command queue (depth open, §7). CXL.io posted-write / non-posted-read
  ordering is the launch flush; config registers take effect at the next
  GEMM start.
- **`PTA_TW` / `PTA_TS` on silicon: open.** They drive the twin's modelled
  settle/shot latency. Whether silicon reports back measured values or
  nothing at all is a v1 decision (§7).

---

## 3. First-testchip vehicle (GF-native only)

The product tile (TFLT, resident banks, 1 GS/s class) is not the first
tapeout. The first tapeout proves the E/O/E chain, the register contract,
and the package — on devices GF ships natively.

- **Die.** 5x5mm MPW block (EUROPRACTICE-class envelope); EIC + PIC in one
  PTA package beside the G100 die, UCIe-S between them, PM fiber to an
  off-package laser module (ICD links 2-4).
- **PIC content, GF-native.** Si waveguides / splitters / MMIs, one
  modulator array (MZM-first; ring-bank topology stays a hypothesis until
  the source-kind question in §7 closes), Ge photodiodes (one per column),
  grating + edge coupler test structures, fiber attach. No TFLT film on
  this vehicle.
- **EIC content.** 6-bit activation DACs, 7-bit ADCs, receiver noise budget
  a quarter of an 8-bit ADC LSB, programming error within one weight LSB,
  neighbour crosstalk under 2% (ICD link 3 / X1 v1). Calibration engine
  with the v0 probe, trim DAC, and the four schedulers. UCIe-S endpoint
  with adapter CRC + retry kept on (raw BER ~1e-15 at 16-32 GT/s is
  absorbable in operands but not in register writes).
- **Tile geometry for the vehicle.** Start at the emulated 8x8 tile at
  0.1 GS/s bring-up, 1 GS/s stretch — one UCIe-S x16 module (57.6 GB/s a
  direction at 0.9 efficiency) carries it with margin per
  `pta_chiplet_link.py` §2. Wider geometries (64x8, 128x64, 256x64) are
  link-sized in that model, not claimed here; module count follows the
  frozen `PTA_CAPS0` geometry (§7).
- **TFLT annex (separate).** Bonded / transfer-printed TFLT modulator
  section over a GF waveguide coupon on a parallel MPW or partner run.
  Success criterion is process compatibility + Pockels modulation + drift
  comparison vs TFLN, not GEMM throughput. It does not gate the vehicle.
- **Laser.** Off-package module owned by the board team; PIC owns fiber
  attach + polarization. Power/kind/wavelength open (§7): budget 0.66-6.6 W
  at 1 GS/s for a 256x64 tile behind 10-20 dB loss, RIN ~-150 dB/Hz at
  1 GS/s (10 dB tighter per decade of rate), per `pta_shot_rate.py`. No
  part picked here; comparison belongs to whoever holds a datasheet.

---

## 4. Power, clocks, bring-up (from the ICD)

Rails and owners per `board_icd.md` §3, fixed at v0: EIC digital, EIC
analog (DAC references + receivers — never shares a regulator with any
digital rail), PIC bias (DAC-held Pockels bias), laser module (diode +
TEC), controller/clocks/fans. Sequencing is a dependency list: controller
first; laser emission only after PIC bias is set and the interlock is
satisfied; CXL trains after both packages leave reset; UCIe sideband
before mainband; PTA reset through its block (idle only) then calibrate
before first GEMM.

Clocks: 100 MHz PCIe reference arrangement open; UCIe forwarded clock per
spec (not read out here); shot clock owned by the chiplet — 0.1 and
1 GS/s are the sized candidates, rate open (§7). Debug: JTAG order open,
UART console from GRX930, CXL state in config space, UCIe DFx surface open,
PTA counters as MMIO over CXL.io.

---

## 5. What the twin covers (and what it must not claim)

The digital twin presents this exact map with `pta_tile_model.c` behind it
for driver, grxcp, and dispatch bring-up. It reports itself via
`PTA_CAPS2`'s emulation bit. Per the honesty rules: a green twin run
proves the digital architecture and the register contract — never device
physics, photonic throughput, or mesh-topology validation. Throughput
numbers for a photonic chip wait on silicon (§7).

---

## 6. Immediate work this baseline unblocks (no PDK needed)

1. Version-stamp the map: `PTA_ID` value + `PTA_CAPS2` emulation bit
   wiring in the twin and the c930 CSR block.
2. Close the `DIN_W` reporting gap one of the two recorded ways.
3. Normative seed function: promote the C1 harness `gemm_seed()` text
   into this baseline.
4. gdsfactory/KLayout parameterized PIC cells with GF-rule placeholders
   (bend, density, marker layers as parameters, not constants).
5. Lumerical forward-model harness (MODE/FDTD/CHARGE/HEAT) for the
   vehicle PIC content, checked into the same shape as the existing
   `pta_*.py` models.
6. Calibre deck stubs (curvilinear DRC/LVS) + dummy-fill flow ownership.
7. GF access thread: GF Connect Atlas account created 2026-10-16,
   verification pending; then PDK request + MPW slot inquiry
   (GlobalShuttle or EUROPRACTICE), so §7's device numbers get sources.
8. Open-source flow skeleton: GDSFactory project PDK with GF-rule
   placeholders, KLayout DRC stubs, SAX harness — see photonic/README.md.
   The skeleton runs end to end on placeholders; the foundry layer map
   and rule values drop in on PDK access without restructuring.

---

## 7. Open (needs PDK, partner, or part)

1. Tile geometry + shot rate that `PTA_CAPS0`/`PTA_CAPS2` will report.
2. `PTA_TW`/`PTA_TS` meaning on silicon (read-back vs unimplemented).
3. Chiplet command-queue depth for the GPU dispatcher.
4. MSI vector count and sharing with the GPU function.
5. Normative seed function text.
6. Source kind for link 4 (single line vs line-per-input — the error
   model as written assumes a ring bank summing powers, i.e. many lines),
   wavelength (1550 vs 1310), power, RIN, and the picked laser part.
7. UCIe lane composition + DFx surface, read out of the spec.
8. GF-native device data: Vpi, responsivity/dark current, heater/tuner
   power, bend/density/marker rules, deck revisions, waiver process.
9. TFLT integration flow: bonding vs transfer-print, contamination and
   thermal-budget clearance with GF, alignment and post-process rules.
10. Named chiplet team + board team owners.
