# GRX development board: program plan

**Companions:** [`chip_vs_board_strategy.md`](chip_vs_board_strategy.md),
[`pta_program_plan.md`](pta_program_plan.md),
[`pta_cpu_integration.md`](pta_cpu_integration.md),
[`pta_gpu_integration.md`](pta_gpu_integration.md),
[`ai_motherboard_design_capabilities.md`](ai_motherboard_design_capabilities.md),
[`heterogeneous_devices.md`](heterogeneous_devices.md),
[GRX_GCPU.md](../GRX_GCPU.md).

**Status: PLAN, drafted 2026-09-21. All seven decisions of §2 are settled, each
as recommended: B1 and B3 that day, and B2, B4, B5, B6 and B7 on 2026-09-22.
X2 has made its predictions (§3.3), X1 its budget (§4.3), and P0, X3 and X4
are drafted as [`board_icd.md`](board_icd.md),
[`pta_chiplet_calibration.md`](pta_chiplet_calibration.md) and
[`pta_chiplet_regmap.md`](pta_chiplet_regmap.md).**

The strategy document settles the product: a PCB development board carrying
the GRX930 SoC (the c930 RV64 cores and their NPU), the GRX-G100 GPU, and a
photonic tensor accelerator (PTA) chiplet that serves the GPU as a
co-processor, with UCIe as the chiplet interconnect and CXL as the protocol
above it. grxcp coordinates the board. This document records what the board
needs decided, what it hands to grx930 and grxgpu, and what it changes in the
PTA program. It designs no RTL, and where it overturns an earlier document it
names the document and the section.

**Boundary rules, unchanged.** grx930 and grxgpu own their silicon. A
requirement for grxgpu lands as a proposal in `grxgpu/docs/proposals/`
(`AGENTS.md` §2); one for grx930 goes to that team, as the PTA program's did.
The honesty rules of `AGENTS.md` §3 apply to every number here: each is traced
to a document, to a measurement, or to an assumption stated beside it.

**Scope, changed.** The PTA documents stop at FPGA emulation and numerics,
with no photonic die. A board with a PTA chiplet is the program's first
physical product, so that scope line changes (§6). Emulation stays the first
vehicle: nothing here commits to silicon, or to a package, before §2 settles.

**Section numbers.** §2.1 and §6.2 are the CPU document's cost model and
sweep, as in the PTA plan. Every other section number either names its
document or is this one's.

---

## 1. What the decision adds, and what it runs into

Most of the board is already architected somewhere in grxcp. The strategy
document adds the product and the interconnect choice, and five decisions
already on record now collide with it.

| Topic | Already on record | The board decision | Here |
|---|---|---|---|
| CPU–GPU coherence protocol | [GRX_GCPU.md](../GRX_GCPU.md) §2: TileLink, with CXL.cache "too heavyweight for on-package" | CXL | Reversed; B3, settled |
| Where the GPU's PTA sits | [`pta_gpu_integration.md`](pta_gpu_integration.md) §2: inside the G100, one per cluster, beside the DXA | A separate chiplet | Re-architected; B4, settled |
| Photonic platform | [`pta_cpu_integration.md`](pta_cpu_integration.md) §4.4: TFLT, a Pockels material | The strategy and motherboard documents assume heater-tuned silicon rings | Aligned to TFLT; B5, settled |
| The CPU's silicon | grx930's `c930/doc/GRX930_manufacturing_plan.md`: SKY130, then TSMC N28, with the RTL frozen | UCIe and CXL need PHYs neither node offers readily | B6, settled |
| Memory | The motherboard document: HBM | A development board | B2, settled |
| Addresses in grxcp | [`heterogeneous_devices.md`](heterogeneous_devices.md) §4.1: every device has its own address space, and the spaces overlap | A CXL coherent region is one space | Extended; S2 |
| Scope | The PTA documents: emulation and numerics only | A photonic chiplet | Rewritten; §6 |

Two points in the strategy document need correcting before anything is built
on them; §6 lists the rest.

- **UCIe joins dies inside one package.** UCIe-S reaches 25 mm across an
  organic package substrate, and UCIe-A 2 mm across a silicon interposer or
  bridge. It is not a link between packages on a PCB. Between packages, CXL
  runs over its native PCIe PHY.
- **The PTA's bulk data is electrical until the modulators.** Operands start
  in GPU memory and reach the interface chip's DACs over wires; light exists
  only between the modulators and the detectors on the photonic die. The link
  from the GPU to the chiplet therefore carries the tensor traffic, not just
  control, and has to be sized for it. Optical links between boards, and
  optically attached HBM, are optical I/O, a different product from photonic
  compute.

---

## 2. Decisions to settle first

Each one blocks the tracks of §3 and needs no measurement to decide. B1 and B3
come first; the rest follow from them. **All seven were settled as recommended
below: B1 and B3 on 2026-09-21, and B2, B4, B5, B6 and B7 on 2026-09-22.** §9
lists the edits that followed.

**B1 — Where the packages end.** UCIe only joins dies that share a package, so
"UCIe for the interconnect" is a statement about packaging. There are three
layouts. Everything can share one package, with every link UCIe, as the
strategy document's implementation section draws it. Every chip can have its
own package, with every link CXL over PCIe on the board. Or the two can be
split.
*Recommended:* the split, for board rev A. The CPU and the GPU are separate
packages, joined by CXL over a PCIe 5.0 x16 link on the board; inside the GPU
package, the GRX-G100 die and the PTA chiplet are joined by UCIe. The PTA stays
beside the device it serves, the CPU stays out of the GPU's advanced package,
and the CPU–GPU link can be prototyped on FPGA CXL hard IP before any package
exists (B6). The price is CPU–GPU bandwidth: about 64 GB/s each way at x16,
an order of magnitude under the NVLink-C2C figure the strategy document
proposes as a target, which a development kit can live with. The single
package holding all three, with CXL over UCIe between the CPU and the GPU, is
the Phase 2 module. *Needed by:* everything; B2, B6 and P0 cannot start
without it. *Settled on 2026-09-21, as recommended.*

**B2 — The first board's memory.** HBM needs a silicon interposer or bridge,
which puts every die beside it into a 2.5D package — the packaging the
strategy document names as the constrained resource whose allocation is
priced.
*Recommended:* GDDR6 or LPDDR5X for the GPU and DDR5 or LPDDR5 for the CPU, on
organic substrates, for rev A; HBM for the Phase 2 module. A development kit's
job is to prove the stack, and it can do that at lower memory bandwidth.
*Needed by:* P0 and P1. *Settled on 2026-09-22:* GDDR6 for the GPU and DDR5 for
the CPU, both packages on organic substrates, so UCIe-S carries the link to the
chiplet and no interposer is needed before the Phase 2 module.

**B3 — Which CXL, and each chip's role in it.** CXL is asymmetric: a host holds
the home agent, and devices cache host memory (CXL.cache), expose memory of
their own (CXL.mem), or both.
*Recommended:* CXL 2.0 on the PCIe 5.0 PHY for rev A, since CXL 3.x needs PCIe
6.0's 64 GT/s PAM4 signalling. The GRX930 is the host: a root port, the home
agent, decoders for host-managed device memory (HDM), and an L2 that answers
snoops from outside the chip. The GRX-G100 is a Type-2 device: CXL.io, CXL.cache
to cache the shared region coherently, and CXL.mem to expose its memory to the
CPU. The PTA is not a CXL agent at all; the CPU reaches it through the GPU (B4,
B7).

This keeps GRX_GCPU.md's hybrid memory model — a coherent shared region beside
private GPU memory — and changes only its protocol. It does reverse that
document's TileLink recommendation, and the reasons belong there: CXL devices
are enumerated and driven through standard PCIe configuration space and the
Linux CXL subsystem; CXL controller and PHY IP can be bought; and a standard
edge keeps the chiplets swappable, which is the strategy document's point.
Inside each chip the fabric stays its own — the SoC specification's roadmap
names CHI for the NPU's coherent port (its step 5) — and is bridged to CXL at
the edge. *Needed by:* L1 and L2. *Settled on 2026-09-21, as recommended.*

**B4 — Where the PTA attaches, and what its interface chip holds.** The PTA
chiplet is a photonic die (PIC) and an electronic interface chip (EIC),
stacked or side by side, attached to the GPU die over UCIe. That retires the
GPU document's placement — one tile per cluster, a third client of the
LMEM-DMA arbiter — in favor of a device-level engine that the whole GPU
shares, behind a link.

The PTA program's feed finding sharpens across that link. F0 and F1 found a
Pockels-class tile bound by its host's operand supply (the PTA plan, §3.3),
and a die-to-die link is a harder supply than an on-chip DMA. For scale: the
§6.2 shape moves under 1 GB/s, but a 64-input INT8 tile firing at 1 GS/s needs
64 GB/s of activations — a whole x16 UCIe-S module at 32 GT/s — unless the
activations are reused on the chiplet. Both figures are illustrations; the
chiplet's size is an open question (§8).

*Recommended:* the EIC holds everything that would otherwise cross the link on
every shot:
- the DACs and ADCs;
- resident weight storage — a DAC-held voltage per resident weight, as the
  PTA plan's step MB sizes it;
- activation buffers, and accumulation across K tiles;
- the calibration engine (the PTA plan's C3);
- as an option, the activation stage, so that a whole layer can stay on the
  chiplet.

The GPU feeds the chiplet from device memory with a copy engine: the DXA's
job, moved from cluster LMEM to the UCIe port. The link runs a UCIe streaming
protocol in a FLIT format, keeping the die-to-die adapter's CRC and retry. Raw
mode drops both, and nothing yet justifies that. X2 sizes the link.
*Needed by:* L3, X2 and X4. *Settled on 2026-09-22, as recommended:* the
interface chip holds the converters, the resident weights, the activation
buffers, the accumulation and the calibration engine. Whether the activation
stage joins them is left until X2 has sized the link. *Addendum, 2026-09-22,
once X2 had:* it joins them. X2 puts the saving at 38% of the traffic that is
not weights on the D3 network and 75% on a four-layer block, and that share
grows with depth and with weight residency — which is where this design is
going. The cost is chiplet area and a nonlinearity fixed in silicon, which
S_ACT has designed once already.
*Corrected 2026-10-04, by X6:* S_ACT had not designed it. S_ACT is grx930's
experiment on all-optical activation, a transfer curve with shot noise and no
bias, and the step between D3's layers is a bias, a ReLU, a rescale and a
clamp. The stage is that step
([`pta_chiplet_regmap.md`](pta_chiplet_regmap.md) §8). The decision is
unchanged: the stage is on the chiplet.
*Addendum, 2026-10-03, once the floorplan had been bounded (§8, question 1):*
stacked, for any tile past several hundred cells. The interface chip holds a
voltage a weight and the photonic die has nothing to hold one with, so a
256 × 64 tile has 16,704 lines between the two. Side by side they want 251 mm
of shared edge at the finest wiring B2's packaging has, and a 13 mm edge
carries 866. Stacked puts the converters directly over the tile, which is P1's
to model.

**B5 — The photonic platform, and the laser.** The strategy and motherboard
documents assume silicon photonics tuned by heaters: ring resonators held on
resonance at 70–80 pm/°C, with heater efficiency lost to hybrid bonding. The
program chose TFLT instead (the CPU document, §4.4) because thermo-optic
silicon cannot hold weights: it takes 7.6 µs to settle and 25 W of heaters for
2,048 shifters, where TFLT settles in 25 ps and holds each weight with a
DAC-held voltage. Neither document decides where the laser goes.

*Recommended:* TFLT, with TFLN as the fallback if TFLT dies cannot be had.
TFLT is the youngest platform in the §4.4 comparison, by that section's own
caveat, and TFLN's drift is the stress case C1 measured. The laser lives off
the package, fiber-fed through a polarization-maintaining fiber array, because
thin-film lithium niobate and tantalate modulators are polarization-sensitive.
The board carries the laser module, its driver and temperature control, fiber
management and an eye-safety interlock. The package's thermal study (P1)
models what TFLT is sensitive to — bias drift, whose cost C1 measured, and the
laser's wavelength — rather than heater locking, and still keeps the PIC out
of the GPU's hot spots. *Settled on 2026-09-22, as recommended:* TFLT, with
TFLN as the fallback, and the laser off the package on polarization-maintaining
fiber.

*Laser power, for scale.* C1's sweep allows receiver noise of one 8-bit ADC
LSB, and shot noise down to 3 photons per ADC LSB (§4.3). Take a receiver with
about 1 µA rms of input noise and 1 A/W of responsivity. The receiver limit
then puts a detector's full scale at 256 µA, about 0.26 mW, while the shot-noise
limit asks only about 0.1 µW at 1 GS/s and 1550 nm. So the receiver, not shot
noise, sets the laser power, by three orders of magnitude: 64 channels behind
10–20 dB of optical loss need 0.16–1.6 W of laser. Every input is an
assumption until the chiplet is sized (§8), but a laser of that order is a
board-level thermal and safety item. *Needed by:* P1 and X1.

*Revisited 2026-10-03: for the same receiver it is twice that.* The arithmetic
above uses C1's allowance of one LSB of receiver noise, which is §4.3's version
0. Version 1 holds the interface chip to half of that LSB. A detector's full
scale is then 0.51 mW rather than 0.26, and 64 channels behind 10–20 dB need
**0.33–3.3 W** at 1 GS/s. [`pta_shot_rate.py`](pta_shot_rate.py) reproduces the
figures above from version 0 before it revises them, so the method is this
paragraph's and only the allowance has moved. Every one of them is linear in
the assumed 1 µA, which makes a measured receiver noise the first number worth
having. What this does to the shot rate is §8, question 1, and what kind of
source supplies it is question 8.

*It said four times, and 0.66–6.6 W, until the unit was found.* That read
version 1's "0.25 LSB" as a quarter of an **8-bit** LSB. It is a quarter of the
7-bit ADC's that version 1 was run with, which is half of an 8-bit one's (§4.3,
corrected the same day). Everything downstream moved with it — the loss ceiling
from 13.9 dB to 16.9, the emitter powers, the intensity-noise limit — and is
restated where it stands.

*And at a receiver that has been measured, 2026-10-04, it is a quarter of that
again.* Every figure above is linear in the 1 µA this paragraph assumed. A
published 40 nm receiver measures 7.2 pA/√Hz up to 1.5 GHz (M. Atef and H.
Zimmermann, *IEEE Trans. Circuits Syst. I* 60, 2013), which over the bandwidth
a 1 GS/s shot needs is 0.27 µA. [`pta_power.py`](pta_power.py) carries B5's
method to it: **0.09–0.88 W** under version 1, and a 1.6 W laser stands 22.6 dB
of loss, which is past the whole of the 10–20 dB this decision ranged over. The
1 µA figures stay beside these as the conservative end. That receiver had its
photodiode beside it and a limiting amplifier after it, and whether it keeps
its noise a bond away from the detector, or stays linear over seven bits, its
paper does not say. That is now the measurement to ask for.

**B6 — Silicon nodes, and the board before silicon.** grx930's manufacturing
plan takes the SoC to SKY130 first, then to TSMC N28, and freezes the RTL now.
SKY130 has no SerDes, by that plan's own list of its limits. The PCIe
5.0-class PHYs that CXL 2.0 needs, and UCIe's PHYs, are offered at 16 nm-class
nodes and finer as far as this plan knows; that needs confirming with IP
vendors. Today's FPGA targets, the Artix-7 parts on the Arty A7-200T and the
Nexys Video, stop at PCIe Gen2.

*Recommended:* keep SKY130 as the core-validation shuttle it was planned to be,
but not as this board's CPU. The board's CPU and GPU dies need PCIe 5.0-class
PHYs, and the GPU die needs a UCIe PHY as well. Before silicon, board rev 0 is
an FPGA platform with CXL hard IP — for example, Intel Agilex 7 with its
R-tile, or AMD Versal Premium — which carries the existing emulation program
onto a CXL-capable part, with the PTA as C1's error-model tile. Most FPGA CXL
IP serves the device side, so the RISC-V host's root port may have to be soft
IP; L1 settles that. This plan reads the manufacturing plan's freeze as the
SKY130 tapeout's; the CXL host is new RTL beyond it. *Needed by:* P2, L1 and
L3. *Settled on 2026-09-22, as recommended.* Which CXL-capable FPGA platform
hosts rev 0 is still open (§8).

**B7 — One control path to the PTA.** The strategy document proposes three: a
low-speed I2C or SPI bus from the CPU, the UCIe sideband, and Raw-mode traffic
on the mainband. The sideband exists to train and manage the link. Raw mode has
no CRC or retry, and at 16–32 GT/s the raw error rate is around one in 10^15
bits: operand payloads can tolerate that beside the analog noise, but a
register write cannot.
*Recommended:* the PTA's register block (the CPU document, §3.1) as MMIO behind
the GPU's CXL.io function, so that the GPU's driver owns it. SMBus or I3C
carries board management only — power, temperature and the laser — and the
UCIe sideband carries the link's own management. *Needed by:* X4. *Settled on
2026-09-22, as recommended.*

---

## 3. The tracks

### 3.1 Track P — package and board (grxcp coordinates)

| Step | What | Gate | Needs |
|---|---|---|---|
| P0 | **Drafted:** [`board_icd.md`](board_icd.md), with the block diagram, the nine links, power, clocks, resets and debug, and a list of what it cannot source yet | Review: every link has an owner on each side, and every number a source | B1–B7 |
| P1 | Package study of the GPU package with the PTA chiplet: floorplan, UCIe-S on the organic substrate B2 chose, fiber attach, and a thermal co-simulation with TFLT's drift in place of heater terms. §8, question 1 holds one published package of the kind, for scale, and [`pta_floorplan.py`](pta_floorplan.py) bounds what has to go in it | The model predicts the PIC's temperature range under the GPU's power map, and C1's drift fits say what that costs in calibration | B1, B2, B5 |
| P2 | Board rev 0, on B6's FPGA platform: the GPU partition as a CXL Type-2 device, and the PTA as the error-model tile | Linux on a CXL host enumerates the device, and the D3 network runs through the emulated PTA bit-identical to `pta_mnist`'s C reference | B6, L2, X4, X5 |
| P3 | Board rev A, on silicon | Scoped after P1 and the silicon plans; no gate yet | P1, L1–L4 |

Rev A as B1 recommends it, which is where P0 starts:

```
 CPU package                               GPU package
+-------------------+                     +-----------------------------------------+
| GRX930 SoC        |    CXL 2.0 over     | GRX-G100 die  <--- UCIe --->  PTA       |
| (RV64 cores, NPU) |<------------------->|                               chiplet   |
| DDR5 or LPDDR5    |    PCIe 5.0 x16     | GDDR6 or LPDDR5X              EIC + PIC |
+---------+---------+                     +-------------------------------------+---+
          |                                                                     |
          | SMBus / I3C: power, temperature, laser                     PM fiber |
          |                                                                     |
+---------+---------+                                                 +---------+------+
| board controller  |-------------------------------------------------| laser module   |
+-------------------+                                                 +----------------+
```

### 3.2 Track L — links (grx930 and grxgpu, by requirement)

| Step | What | Gate | Needs |
|---|---|---|---|
| L1 | A CXL 2.0 host in the GRX930: root port, home agent, HDM decoders and an L2 that answers external snoops. Build or license; and where rev 0's root port comes from | The host passes CXL.cache and CXL.mem protocol tests against a device model in simulation | B3, B6 |
| L2 | A CXL 2.0 Type-2 device in the GRX-G100: CXL.io, CXL.cache, and CXL.mem over its memory | The device passes the same tests against a host model | B3 |
| L3 | A UCIe port on the GRX-G100 for the PTA chiplet, fed by a copy engine | Sized by X2, streaming in a FLIT format with CRC and retry | B4, X2 |
| L4 | Firmware and OS: the CXL host bridge described to Linux on RISC-V, and the link brought up | Linux's CXL subsystem enumerates the device and maps its memory | L1, L2 |

Linux's CXL subsystem, as it stands upstream, expects the host bridge to be
described in ACPI's CEDT table, and RISC-V boards mostly boot with a device
tree. So L4 is planned alongside L1, not after it.

### 3.3 Track X — the PTA chiplet (grxcp and the PTA program)

| Step | What | Gate | Needs |
|---|---|---|---|
| X1 | **Budgeted, §4.3.** Version 0 came from C1, one impairment at a time; version 1 from `sim/pta_mnist.sh joint` in grx930, with every impairment on at once | Every number traced to grx930's design note §5 or to a new run | C1, done |
| X2 | **Predicted, below.** Link sizing in [`pta_chiplet_link.py`](pta_chiplet_link.py), which adds the die-to-die term to F1's model and carries the PTA plan's F3 handoff | Predictions stated before any RTL, and every number traced, as F1's were | B4, F1 |
| X3 | **Specified, measured and built:** [`pta_chiplet_calibration.md`](pta_chiplet_calibration.md), with C3(a)'s recovery and C3(b)'s RTL in its §8 — the trim returns a tile at chance to within 0.01 points of the no-drift case, and the engine that writes it agrees with the C reference cell for cell | C3's own gate, at TFLT's and TFLN's drift | Done |
| X4 | **Drafted:** [`pta_chiplet_regmap.md`](pta_chiplet_regmap.md) — the §3.1 block at its own offsets in the GPU's BAR, with identity, interrupts and 64-bit counters added, and §3.2's contract restated for a device behind a link | Review | B4, B7 |
| X5 | **Built, below:** the digital twin, `pta_tile_model.c` behind X4's map, so that drivers and grxcp can bring the PTA up before silicon | grxcp's backend gates pass against it, bitwise against the model. Bitwise against the model: met. Through grxcp's runtime: met by S4 (§3.4) | X4 |
| X6 | **Built, below:** the chiplet's activation stage, specified in X4's §8 and built into the twin, so that a network need not leave the chiplet between its layers | A network held on the chiplet is the network brought out at every layer, bit for bit, and is grx930's harness. Met on the twin; not through grxcp's runtime, which has no call for a network | X5, S3 |

**X2, predicted.** [`pta_chiplet_link.py`](pta_chiplet_link.py) prices the
link the way F1 priced the c930's feed, and carries F3's handoff in its first
section: at the §6.2 shape the emulated tile moves 0.19 GB/s in and 0.02 GB/s
out at EO-res, and less at every thermo-optic point, so the c930's tile would
never trouble a link. The chiplet is a different size of object.

*Revised 2026-10-01, when F3 delivered.* That handoff first read 0.42 and 0.05
GB/s, overstating the rate by 2.2×. Its EO-res GEMM was 4,427 cycles and is
9,548, from three things: §6.2's table gives a shot `PTA_TS` and nothing else
where C2 measured `PTA_TS + 2` (4,096 cycles a GEMM at 2,048 shots); it writes
`Tw = 0` at the resident point as though selecting a bank were free, where MB
measured the select's own cycle and `m`-outer programs once a shot (another
2,048); and it carried the DMA's old five-cycle-a-beat C write burst, which F2
rebuilt to one, pulling the other way by 1,023 and keeping the first two hidden.
Every core in the handoff is now asserted against a measurement at that shape
rather than quoted. The verdict is unchanged and more comfortable, not less,
since the tile is slower than claimed. F3 adds two things X2 did not have: the
per-row latency budget (below) and a caveat that these are the DMA's rates and
not the host's.

A layer of 4096 by 4096 at batch 64, with B4's split, one UCIe-S module taken
as 16 lanes at 32 GT/s and 0.9 of that surviving overhead — 57.6 GB/s a
direction:

| Tile | Shots a layer | At 1 GS/s, in | Out | Modules in |
|---|---|---|---|---|
| 64 × 8 | 2,097,152 | 8.1 GB/s | 0.50 GB/s | 1 |
| 128 × 64 | 131,072 | 130 GB/s | 8.0 GB/s | 3 |
| 256 × 64 | 65,536 | 260 GB/s | 16 GB/s | 5 |
| 256 × 128 | 32,768 | 520 GB/s | 32 GB/s | 10 |

Four things follow, and each is a number this plan can be held to.

- **The tile can wait 1,010 ns for a row, and the link's round trip is 100.**
  F3's margin: at EO-res with the shipped loop order, the tightest A row has 101
  cycles of slack before the core stalls, 1,010 ns at 10 ns a cycle. The model's
  assumed request-and-return is 100 ns, so latency is not what binds —
  bandwidth is, with an order of magnitude in hand. The two thermo-optic points have *negative* margin and the core
  already waits there (259 and 1,511 cycles), which is F1's finding and not a
  link problem. Weights are not in this number: at EO-res they are resident, so
  they cross once a GEMM rather than once a row.
- **The link caps the tile, as B4 expected.** One module holds a 256 × 64 tile
  to 0.22 G shots a second at batch 64, and a 64 × 8 tile to 7.1 G. Whatever
  the optics can do, that is the rate.
- **The batch is the lever on weights.** Inbound is `k·n/mb` bytes of weight a
  shot beside the activations, so the same 256 × 64 tile needs 1,028 GB/s at
  batch 16, 260 at 64 and 68 at 256. Weight residency across batches is the
  other lever, and the one MB already studies on the c930.
- **B4's split earns its area.** A thin interface chip, with neither activation
  buffers nor accumulation, needs two to nine times the inbound and four to
  eight times the outbound of the same geometries.
- **What the chiplet must hold**, at 256 × 64: 16,384 DAC-held weights, 16.4 kB
  of activation buffer for a K tile of a batch, and 16.4 kB of accumulators. A
  module keeps 5.76 kB in flight across an assumed 100 ns round trip, and the
  chiplet needs at least that again before the tile waits.

**The activation stage, which B4 left open.** With the stage on the chiplet, an
intermediate never crosses the link: the layer's outputs are already where the
next layer's inputs are needed. Per inference of a batch of 64, on a 256 × 64
tile:

| Network | Stage on the GPU | Stage on the chiplet | Traffic that is not weights |
|---|---|---|---|
| D3, 784-100-10 | 232 kB | 200 kB | 84.7 kB → 52.7 kB, down 38% |
| Four 4096-square layers | 72.4 MB | 68.4 MB | 5.24 MB → 1.31 MB, down 75% |

At batch 64 the weights are most of the traffic either way, so the stage pays
most when weights stay resident across batches and when the network is deep.
The assumptions — one byte an operand or ADC code, four for an accumulated
output, 0.9 link efficiency, a 100 ns round trip, and candidate geometries
rather than a settled one (§8) — are listed in the script and marked where they
are used.

**X5, built 2026-10-04.** `src/backends/pta_chiplet/` holds the twin: X4's map
as a C register file, the chiplet's own command queue, a clock and the
calibration contract, in front of grx930's error model and calibration
reference. None of the tile's arithmetic is in it. Its gate is in tier 1
(`ci/build_mock.sh`, the PTA CHIPLET TWIN GATE) and
[`pta_chiplet_regmap.md`](pta_chiplet_regmap.md) §6 has what it holds. Four
things came out of building it.

- **The model is behind the map, bit for bit.** 156 checks at three
  geometries: every GEMM through the twin equals `pta_gemm()` called directly
  and every calibration equals `pta_cal_bank()`, on a reference device the twin
  never sees. Three twins that are each wrong in one way fail it.
- **P2's second gate half passes on it now.** grx930's accuracy harness,
  unedited, with the twin where the model was: all 220 evaluations of §4.3's
  budget — 44 settings, five networks, 1,865,160 GEMMs — print the line they
  printed on 2026-10-03, byte for byte. So "the D3 network runs through the
  emulated PTA bit-identical to `pta_mnist`'s C reference" is true of the twin.
  Ten of those runs age the tile first, by six minutes of drift or by an hour,
  and that goes through a call that is the twin's and not the map's: no register
  ages a device. P2's other half, a CXL host enumerating the device, still needs
  a board.
- **One driver does address both maps.** The c930 backend's register reader
  identifies the twin and reports its tile through a change of base and nothing
  else, which is X4's §1 as a test and no longer as a claim.
- **The map cannot issue work.** X4 is the control window, link 2 carries the
  work, and nothing says what a command on that link is. The twin stands a
  function call in its place and takes a whole GEMM at once. So it says nothing
  about the link: what X2 priced is what crosses it each shot, and that is the
  part the twin does not have. The gap is L3's and the chiplet's
  ([`board_icd.md`](board_icd.md) §7).

Six smaller things are recorded against the map in its §6 and §7: BUSY has to
cover the chiplet's queue; the affine has eight words and the tile 64 columns;
`SAT_THRESHOLD` has no threshold; a calibration's refusal is read in two places
by one driver; a host cannot restore a saved trim; and a GEMM's seed cannot be
reported without its index.

What it is not. It was not, when this was written, a device grxcp could see:
nothing enumerated it, so `grxblasGemmEx` did not reach it and X5's gate was met
against the model and not through the runtime. *S4 has since made it one*
(§3.4). Its calibration scheduler is off, so the
comparison of schedulers that
[`pta_chiplet_calibration.md`](pta_chiplet_calibration.md) §7 assigns to this
twin is still owed. And it is a model: a result through it is a statement about
the error model's arithmetic and a driver's use of the map, and not about a
chiplet.

**X6, built 2026-10-04: the activation stage.** X4 had a presence bit for the
stage and nothing that said what it computes. S3 priced what that was worth:
the last row of its "one list for the network" (§3.4), where a network's
intermediate never returns to the GPU and no launch stands between its layers.
[`pta_chiplet_regmap.md`](pta_chiplet_regmap.md) §8 is now the specification and
the twin runs it. Five things.

- **It is the step between D3's layers, and not S_ACT.** A bias, a ReLU, a
  rounding shift back to an operand, and the operand's clamp: what grx930's
  harness does on the host between layers. B4's addendum expected S_ACT's design
  to serve. It cannot, because S_ACT has no bias, and D3 is the one network with
  an accuracy to keep (§2, B4, corrected).
- **A command asks for it and no register does.** Its settings ride in the
  command and apply to that command alone, which is S_ACT's own lesson about a
  live enable. A command either returns operands where it would return sums, or
  holds them on the chiplet for the next command, which takes them as its
  activations. Held operands are the next command's or nobody's: any command's
  start takes or discards them, a calibration between two layers does not, and a
  layer that was refused leaves nothing behind.
- **Held, a network is the network brought out at every layer.** The twin's gate
  grew from 156 checks to 201. On an 8 × 8 tile through three layers and a
  256 × 64 tile at D3's shape, with every impairment the tile builds enabled, the
  last layer's sums agree element for element: held, brought out with the step
  taken by the test, and on a device the twins never see. Two more twins that are
  each wrong in one way fail it, one whose shift truncates and one whose held
  operands outlive the command after them.
- **And it is grx930's harness, not a copy of it.** The function is the one piece
  of arithmetic in the twin that is not grx930's, so it is held to grx930 from
  outside: `pta_mnist_act_via_twin.c` compiles the harness's source into itself
  and runs the harness's own batch routine beside the twin. The harness cuts D3's
  shape into 54 GEMMs and takes its step on the host; the twin gets two commands.
  On four random networks every operand a hidden layer hands on and every sum
  out of the last layer is the same, 6,400 and 640 at D3's shape. Against a twin
  whose shift truncates, 1,646 of those 6,400 differ. grx930 at `838c7cd`; not in
  CI, because the harness is not in this tree.
- **Two things it does not show.** Nothing with noise is compared against the
  harness: the two cut a layer into different GEMMs, each GEMM draws from its own
  seed, and so they are different runs. And the networks are random. D3 on the
  trained networks, held on the chiplet, has not been run, so there is no
  accuracy figure for a held network.

What it adds to the interface chip's requirements (§4.3): an activation unit a
column, at the shot rate, and room to hold a layer's operands. The twin gives
the stage no time. One that took a beat an output would spend 6.4 µs on D3's
hidden layer at a batch of 64, against 3.74 µs for both of the network's GEMMs.

What it leaves. grxcp's runtime does not use it: grxBLAS runs one GEMM and has
no notion of a network, so the stage is reachable through the twin's own call
and nothing above it. grxgpu has not been asked to carry it: the host-path
proposal reserved two flag bits and the stage needs a third and three fields
(§4.2). And its time is a requirement and not a measurement.

### 3.4 Track S — grxcp

| Step | What | Gate | Needs |
|---|---|---|---|
| S1 | Enumeration over CXL: the GPU, and its PTA, found through configuration space | [`heterogeneous_devices.md`](heterogeneous_devices.md) §4's rule: a device that is not present is not enumerated | L4, X4 |
| S2 | A coherent shared pool: pointers valid on both the CPU and the GPU, beside the per-device spaces of §4.1 | Each coherent allocation reported through a device property, and no pointer resolved to the wrong device | L1, L2 |
| S3 | **Predicted, below.** The dispatch cost model, [`pta_dispatch.py`](pta_dispatch.py): the §2.1 model with X2's link terms and C1's accuracy, placing each GEMM on the PTA, the GPU or the NPU | Its predictions checked on rev 0. Not met: there is no rev 0. What is below is what it will be checking | X2, P2 |
| S4 | **Built, below.** The PTA reported through the PTA plan's D2 property — effective bits, seed, impairment mask — from the twin now and from silicon later. X5 found one thing this step had to add: on the chiplet a GEMM's seed is derived from `PTA_SEED` and the GEMM's index, so the property needs the index or it describes a run and not a result | `AGENTS.md` §3: every field sourced or reported unknown (−1) | X5 |

**S4, built 2026-10-04.** The chiplet is a device grxcp can see, as B9 settles
it. `src/backends/pta_chiplet/pta_chiplet.cpp` is its driver, the runtime
enumerates it behind `-DGRXCP_ENABLE_PTA=ON`, and grxBLAS routes an int8 GEMM
on it to the tile. The only chiplet there is to attach is X5's twin, through a
seam, and the device says it is a model. Four things.

- **The property carries the GEMM's index.** `grxAnalogGemm_t.gemmIndex` is
  `PTA_GEMM_CT`, read before the GEMM it describes, and `-1` on a c930, whose
  GEMMs run on the seed as written. The gate is
  `tests/libs/test_grxblas_pta_chiplet.cpp`: eight cases on a 256 × 64 tile,
  4,164 results through `grxSetDevice` and `grxblasGemmEx`, each equal to
  the model built from the property and nothing else. A reference built on the
  seed alone, as a c930's would be, differs in 285 elements of 600.
- **X5's gate is met through the runtime.** "grxcp's backend gates pass against
  it, bitwise against the model" was met by X5 against the model called
  directly, and is now met through the same calls a program makes.
- **Its operands are its parent's.** `parentDevice` names the GPU, `grxMalloc`
  on the chiplet is refused, and a GEMM on it takes the GPU's pointers. A
  pointer that is not live on the parent is refused by name. Outside a GEMM the
  exception does not exist: `grxMemcpy` on the chiplet refuses the same pointer.
- **It has no hardware path, and says so.** With the backend built and nothing
  attached, no PTA device is enumerated, on any machine. Two things are missing
  and neither is grxcp's to supply. The GPU's driver has no call that reaches
  the chiplet's page of its BAR, which was a proposal owed to grxgpu and has
  since been made (§4.2). And link 2
  has no command format ([`board_icd.md`](board_icd.md) §7, item 7), so the
  driver's path for work is a hook that only a model fills. A chiplet with a
  window and no link is the state every one is in off a model: it is enumerated
  and reported, and a GEMM on it returns "not supported" with C untouched.

What stands in, where a model is attached: the operands make a host round trip
in place of the copy engine and the link. It is reachable only through the seam.

What it leaves. S1, finding the chiplet through configuration space, which
waits on L4. S3, which device a GEMM should go to, *predicted since and below*.
And a way to configure the tile from grxcp, which is `cuda_mapping.md` 7.40 for
the chiplet as it was for the c930: the gate programs the registers behind the
runtime's back.

**S3, predicted 2026-10-04.** [`pta_dispatch.py`](pta_dispatch.py) is the model:
the CPU document's §2.1 for the NPU, X2's link and
[`pta_shot_rate.py`](pta_shot_rate.py) §5's weight path for the chiplet, a GEMM
rate grxgpu measured for the GPU, and §4.3's accuracy. A GEMM is one layer on one
batch, `M` rows of `K` inputs against `K × N` weights. The step's gate is that
its predictions are checked on rev 0, so it is not met; this is what rev 0 will
have to check.

The terms are not the same kind of number, and the model's first table says
which is which:

| Device | A MAC | Fixed, a GEMM | What the numbers are |
|---|---|---|---|
| NPU | 5.54 ns | 10.9 µs | **Measured in RTL.** F0's GEMM at `M = 64, N = 8, K = 256`, and F2's host share of a GEMM on the SoC |
| GPU | 257 ps | 6.9 or 30.5 µs | **Measured in SimX** by grxgpu, in fp16, at two shapes that agree to 0.4%: 0.1028 cycles a MAC (`grxgpu/docs/proposals/grxgpu_tensor_engine.md` §8). The clock is **configured**, 400 MHz, and not measured. The launch is grxcp's own simx figure on other kernels ([`developer_interface.md`](developer_interface.md) §3), without and with a preamble that scales with occupancy: an indication |
| PTA chiplet | 0.28 ps | not known | **Predicted.** 256 × 64 at 1 GS/s, two banks, one link module, on X2's 4096-square layer. There is no chiplet, and its command path is proposed and not built |

Each GEMM with no fixed cost on either side, the chiplet as §4.3 has it:

| GEMM, batch 64 | MACs | GPU's arithmetic | Chiplet | Ratio | Slack |
|---|---|---|---|---|---|
| D3's first layer, 784 → 100 | 5,017,600 | 1.29 ms | 3.25 µs | 397 | 1.29 ms |
| D3's second, 100 → 10 | 64,000 | 16.5 µs | 496 ns | 33 | 16 µs |
| §6.2's shape, 256 → 8 | 131,072 | 33.7 µs | 669 ns | 50 | 33 µs |
| The SoC NPU's largest, 16 → 12 at batch 8 | 1,536 | 395 ns | 387 ns | 1.0 | 8 ns |
| X2's layer, 4096 → 4096 | 1,073,741,824 | 276 ms | 296 µs | 933 | 276 ms |

*Slack* is the GPU's arithmetic less the chiplet's whole time: how much more than
a launch a command to the chiplet may cost before the GPU is the quicker of the
two. Both are reached through the GPU's command processor, so whatever the two
paths share cancels.

Seven things follow.

- **Above tens of thousands of MACs, time does not choose.** The chiplet is 33 to
  933 times the GPU's arithmetic on the four layers, and its command may cost
  16 µs to 276 ms more than a launch before that turns. What chooses is whether
  the GEMM may go there at all: int8 operands, and a network whose loss at this
  budget has been measured and accepted. That is one network, D3, at 0.81 points
  with hourly calibration and 0.37 at six minutes (§4.3).
- **At 1,536 MACs they are level.** The chiplet spends its time moving one padded
  weight set, 16,384 bytes for a layer of 192 weights, and the GPU's arithmetic
  takes as long.
- **The chiplet's GEMM is its command.** D3's two layers at batch 64 are 3.74 µs
  on the chiplet. The program has measured two costs of a command, and neither
  is the chiplet's: 10.9 µs for the c930's MMIO path to its own NPU, and 30.5 µs
  for a launch on the GPU in simx. Two commands at either are 5.8 or 16 times
  the work they carry. The tile is half of its own GEMM only from a batch of 628
  on D3's first layer and 6,063 on its second, at the smaller cost. So the speed
  rev 0 can measure for the chiplet is its command path's, and the proposal to
  grxgpu (§4.2) is where that is decided. That proposal asked for one call a
  GEMM. ~~A descriptor that carries a network's layers, as the GPU's draw
  descriptor carries a pass's stages, is the obvious answer to this, and nobody
  has proposed it.~~ *Wrong, and corrected the same day.* grxgpu's runtime
  already submits a list of commands under one doorbell and one completion poll
  (`vx_enqueue_commands`, and `vx_enqueue_draw` as one command the device
  expands). The proposal had been written without using it. It is amended to ask
  for the GEMM as a member of that list (§4.2, and "one list for the network"
  below).
- **The link binds before the tile does**, at one module, on every shape, and
  what crosses is weights. D3's first layer programs eight whole sets to use 60%
  of them. X2 counts a set whole, padding and all; without the padding the
  smallest shape turns tile-bound and the chiplet is 2.3 times the GPU there, so
  that convention is worth up to 2.3 times on a small layer and nothing on X2's.
  The batch is the lever, as X2 said: doubling it adds under 40% to D3's first
  layer, because the weights cross once.
- **The write path is worth as much as the shot rate.** A tenth of the shot rate
  leaves D3's second layer 12 times the GPU. The c930's one-cell scan on one bank
  leaves it level. §4.3's two banks and 256 cells a beat are what the chiplet's
  column rests on.
- **The NPU's share is not settled.** Its fixed cost is the host's and is
  measured. The GPU's is the device's, and what the driver and link 1 add to a
  launch has not been measured. The NPU keeps its largest command if a launch
  costs more than 24.4 µs in all, 9,758 cycles at 400 MHz. The two launch figures
  in hand are 6.9 and 30.5 µs, one on each side of it. Past a hundred thousand
  MACs the GPU wins whichever is right, and the SoC's NPU takes 1,536.
- **Placement is a rule and not a search.** The chiplet takes an int8 GEMM in a
  network measured at this budget, batched. The NPU takes a GEMM it can be asked
  for, if rev 0 puts a launch above 24.4 µs. The GPU takes everything else. The
  rule is applied above grxBLAS, by whoever sets the device (B9): the library
  runs a GEMM where it is told and does not fall back.

D3 end to end, a batch of 64, fixed costs included:

| Where | Time | Accuracy | |
|---|---|---|---|
| NPU | 29.3 ms | 97.45% | The array's rate. The SoC's NPU cannot be asked for either layer in one command |
| GPU | 1.37 ms | 97.45% | Two launches at 30.5 µs, which is an indication |
| Chiplet, hourly calibration | 3.74 µs and two commands | 96.64% | 25.6 to 64.7 µs with a command at the two measured costs |
| Chiplet, calibrated every six minutes | the same | 97.08% | |

An activation stage on the GPU would put a launch between the two GEMMs: 30.5 µs,
eight times what the chiplet spends on both. B4's addendum put the stage on the
chiplet for the link's sake, and this is the same answer from the time side. With
one command queue on the GPU, which is what the host-path proposal would leave
(§4.2), the launch and the GEMMs are serial and nothing overlaps to hide it.

**One list for the network.** The model's §7 prices D3 at a batch of 64 as a
host would submit it. The host's round trip is the c930's measured 10.9 µs, a
stand-in, since nobody has measured the G100's. The step between the two layers
is a launch on the GPU, 6.9 to 30.5 µs on the device, and its arithmetic is not
priced.

| How it is submitted | Round trips | Total | The chiplet is |
|---|---|---|---|
| A call a GEMM, and the launch between them | 3 | 43.4 to 67 µs | 9% to 6% |
| One list, as the amended proposal asks | 1 | 21.6 to 45.1 µs | 17% to 8% |
| One list, with the step between the layers on the chiplet | 1 | 14.7 µs | 26% |

The list removes two round trips, 21.8 µs, whatever the launch costs. The last
row removes the launch as well and cannot be asked for yet: it needs the
chiplet's activation stage, and X4 has a presence bit for that and no registers
([`pta_chiplet_regmap.md`](pta_chiplet_regmap.md) §7, item 10). The intermediate
that would stop crossing the link is not what that row saves. It is 6,400 bytes
and a tenth of a microsecond. *X6 has since specified the stage and built it
into the twin* (§3.3). It is the stage that removes the launch, whether its
operands are held or returned. grxgpu has still not been asked to carry it.

**What rev 0 can check, and what it cannot.** It can check the GPU's rate and
its launch, the NPU's, the figure that separates them, and what a command to the
PTA costs. It cannot check the chiplet's column: on rev 0 the PTA is the
error-model tile (P2), and its time is whatever `PTA_TW` and `PTA_TS` are set to.

**What it does not price.** Energy: [`pta_power.py`](pta_power.py) has the
chiplet's watts and nothing in the three repositories has the GPU's. The GPU's
int8 path: grxgpu's measurement is fp16 and stands in for it. And any network
but D3: nothing else has an accuracy figure, and the model does not supply one.

Every figure above is an assert in the model, and ten errors planted in its
inputs and its arithmetic each fail it: the GPU's clock halved, one of its runs
misquoted by 2%, the launch's preamble dropped, the NPU's host cost doubled, the
write path narrowed to a cell a beat, the round trip left out, the link added to
the tile instead of capping it, the weights sent without padding, D3's first
layer mis-sized, and the NPU's read-ahead dropped.

---

## 4. What this hands on

### 4.1 To grx930

- The CXL 2.0 host of L1, and the firmware of L4.
- A node with PCIe 5.0-class PHYs for the board's CPU (B6), a PCIe 5.0 x16 root
  port, and a DDR5 controller (B2).
- A RISC-V IOMMU for device DMA, and AIA's message-signalled interrupts for
  devices, both of which [GRX_GCPU.md](../GRX_GCPU.md)'s summary already lists.
- The SoC specification's roadmap already names DDR5/HBM, PCIe/CXL and
  IOMMU/AIA as its step 6. The board makes step 6 the next silicon.

### 4.2 To grxgpu, as proposals

- The CXL 2.0 Type-2 device of L2, and the UCIe port of L3.
- The PTA as a device-level engine behind that port, replacing the
  cluster-scope tile of [`pta_gpu_integration.md`](pta_gpu_integration.md) §2
  and its step G2.
- The G100's microscaling path (the GPU document, §5) as the PTA's numeric
  format: FP8-class inputs reach the analog tile as block-scaled integers.
- A GDDR6 controller and PHY (B2).
- *Made, 2026-10-04:* the chiplet's host path,
  `grxgpu/docs/proposals/pta_chiplet_host_path.md`
  ([kubeworkz/grxgpu#2](https://github.com/kubeworkz/grxgpu/pull/2)), open
  until grxgpu answers. It sits in front of the first two items and replaces
  neither. It asks for no RTL, no port and no copy engine, only for how a host
  tells the GPU to use the chiplet: the registers as a range of the GPU's DCR
  addresses, reached by its driver's existing register commands, and a GEMM as
  one command of its command processor with the operands in device memory, in
  the simulators' command processor first.

  Three things here would follow if it is taken as written. None has been
  changed yet.

  - **X4's window** is a page of the GPU's BAR and would become that range.
    What B7 chose is untouched: one control path, owned by the GPU's driver,
    with the link's CRC and retry. Its words "as MMIO" would no longer describe
    it, so this is B7's to confirm and not only grxgpu's to accept.
  - **The chiplet's own queue** would go. The command processor's ring would be
    the queue, the depth X4 leaves open (its §7, item 3) would not be a number
    anyone needs, and BUSY would mean running.
  - *Amended, 2026-10-04* ([kubeworkz/grxgpu#3](https://github.com/kubeworkz/grxgpu/pull/3)),
    open until grxgpu answers. S3 found that a call a GEMM makes the host's round
    trip the cost of a small network, and that grxgpu's runtime already has the
    answer: a list of commands under one doorbell. The amendment asks for the
    chiplet's GEMM as a member of that list and as a step of the device-expanded
    form, with one flag, "only after success", so that a layer whose input was
    refused does not run on whatever was in memory. It withdraws the proposal's
    question about where results go: in the descriptor, since a list has many.
    It reserves three flag bits and defines none: two for a GEMM that takes the
    previous one's results through the activation stage and keeps its own on the
    chiplet, and one for resident weights. **The first two wait on grxcp**: X4
    has to say what the activation stage computes before anyone can be asked to
    carry a command that uses it. *X4's §8 now says* (X6, §3.3). A further
    amendment is owed, and it is larger than the two bits foresaw: a third bit
    to ask for the stage at all, and a shift, a width and a bias vector in the
    command.
  - **A launch and a GEMM on the chiplet would be serial** while the GPU has one
    command queue, which is its default. *S3 priced it* (§3.4): a launch between
    D3's two GEMMs costs eight times what the chiplet spends on both, which is a
    reason the activation stage is on the chiplet and not a reason to ask grxgpu
    for a second queue.

### 4.3 To the PTA chiplet: EIC requirements

**Corrected 2026-10-03: version 0's rows add, and what "compounded" was the
unit.** This section concluded from X1's joint runs that version 0's rows, each
costing half a point or less alone, cost 11 to 13 points together — that
"analog error does not add, it compounds" — and tightened every row about
fourfold in response. The joint runs were not version 0's rows together. The
harness's noise options are in LSB of whichever ADC a run configures. C1
measured the two noise rows at an 8-bit ADC, and X1 ran them at version 0's
6-bit ADC as the same numbers, which there is **four times the receiver noise
and a quarter of the light**. grx930's `sim/pta_mnist.sh budget` reruns them in
one unit (its `c930/doc/pta_error_model_design_note.md`, §5):

| Setting | Accuracy | Loss |
|---|---|---|
| Version 0's five rows, one at a time, at the 8-bit ADC | 96.97 to 97.28 | 1.28 points summed |
| All five at once, at that ADC | 96.04 | 1.38 |
| All five at version 0's own 6-bit ADC, the same noise | 95.96 | 1.49, where the rows and the ADC's own 0.21 sum to 1.49 |
| As X1 ran "v0 entire": receiver noise of 4 LSB of an 8-bit ADC, 0.75 photons per such LSB | 86.37 | 11.08 |
| Version 1 as X1 ran it: 0.5 LSB, 15 photons | 97.20 | 0.25 |

So on this network the rows **add**, to a tenth of a point, and version 0 is a
budget: one that costs a point and a half before drift, and 2.6 after an hour
of TFLT's. What follows is kept as it was written, with each claim that fell
marked where it stands. The row-by-row prices that replace "fourfold" are at
the end of the section.

**Version 0** came from C1's sweep on the D3 network — a 784-100-10 MLP on
MNIST, at 8-bit operands and 6-bit weights, 97.45% with nothing else impaired —
in grx930's `c930/doc/pta_error_model_design_note.md` §5. Each row costs about
half a point or less **on its own**. The noise rows were measured with an
8-bit ADC, at 97.42% before noise.

| Parameter | Requirement | Cost at that setting |
|---|---|---|
| ADC | 6 bits | 0.21 points; 5 bits costs 0.79 |
| Activation DAC | 5 bits | 0.16 |
| Weight resolution | 6 bits | none: the networks are trained at 6 |
| Receiver noise | 1 LSB of an 8-bit ADC, rms | 0.26 |
| Light at each detector | 3 photons per ADC LSB | 0.45 |
| Weight programming error | 4 LSB of an 8-bit weight, rms | 0.27 |
| Crosstalk between neighbouring inputs | 10% | 0.16 |
| Recalibration | about hourly at TFLT's drift fit; within minutes at TFLN's | TFLT's fit costs 0.46 points in an hour, TFLN's 0.96 in six minutes |

**Version 1, from X1's joint runs** (2026-09-22). `sim/pta_mnist.sh joint`
runs every impairment at once on the same five networks, each on its own seed.
The settings' names are as they were written; the last column, added with the
correction, is what each one was in the unit the rows were measured in:

| Setting | Mean | Loss | Receiver noise and photons, in LSB of an 8-bit ADC |
|---|---|---|---|
| v0's converters alone: 5 activation bits, 6-bit ADC | 97.00 | 0.45 | none |
| v0 entire, no drift | 86.37 | 11.08 | 4, 0.75 — v0's rows are 1 and 3 |
| v0 entire, an hour of TFLT drift | 84.61 | 12.84 | 4, 0.75 |
| v0 entire, six minutes of it | 85.97 | 11.48 | 4, 0.75 |
| v0's noise halved, v0's converters, an hour | 93.88 | 3.57 | 2, 2.5 |
| v0's noise, 6 activation bits and a 7-bit ADC, an hour | 93.39 | 4.06 | 2, 1.5 |
| Both — noise halved, converters widened — an hour | 96.04 | 1.41 | 1, 5 |
| Noise quartered, 6 and 7 bits, an hour | 96.64 | 0.81 | 0.5, 15 |
| The same, recalibrated every six minutes | 97.08 | 0.37 | 0.5, 15 |

No row of it is version 0. The nearest is "both", which has version 0's
receiver noise exactly, with more light, half its programming error and half
its crosstalk, at version 1's converters and after an hour of drift.

~~**Version 0 was never a budget.**~~ *Withdrawn.* This paragraph said that
version 0's items cost about two points summed and 12.8 together, that analog
error does not add but compounds, and that a network's slack is spent once. The
12.8 was four times the noise, and none of the three stands. The interface chip
is still held to version 1 below, which is the set C3's and X3's measurements
were made at — as one point on a priced menu, and no longer as the only set
that works:

| Parameter | v0, each alone | v1, all together |
|---|---|---|
| Activation DAC | 5 bits | 6 bits |
| ADC | 6 bits | 7 bits |
| Weight resolution | 6 bits | 6 bits, where the networks are trained. The DAC wants two bits below the code for trimming — measured worth 0.16 points, not the precondition X3 first called it ([`pta_chiplet_calibration.md`](pta_chiplet_calibration.md) §8) |
| Receiver noise | 1 LSB of an 8-bit ADC, rms | 0.5 LSB of an 8-bit ADC. X1 ran a quarter of its 7-bit ADC's LSB, and this cell read "0.25 LSB" until 2026-10-03 |
| Light at each detector | 3 photons per LSB of an 8-bit ADC | 15 per such LSB, which is the 30 per 7-bit LSB that was run. Not a requirement in practice: see the end of the section |
| Weight programming error | 4 LSB of an 8-bit weight, rms | 1 LSB |
| Crosstalk between neighbouring inputs | 10% | 2% |
| Recalibration | about hourly at TFLT's fit | hourly costs 0.81 points, six minutes 0.37 |

v1 costs 0.81 points on the D3 network at hourly calibration, and 0.37 if the
schedulers can recalibrate every six minutes — which is what C3 had to price.
It is still one small network (§8). This paragraph went on to say that the
shape of the result — error compounds, every allowance tightens fourfold —
would travel further than its numbers. It was the numbers that were wrong, and
the shape went with them.

*What the two versions are on a single GEMM, 2026-10-03.* The PTA plan's S2
measured the error of one GEMM against the exact product, at these two sets of
allowances, on grxcp's register model of the tile
([`pta_cpu_integration.md`](pta_cpu_integration.md) §7). As a fraction of the
product's RMS:

| | v0 | v1 |
|---|---|---|
| Quantisers | 6.5% | 3.6% |
| Receiver noise | 18.6% | 2.3% |
| Light at the detector | 21.6% | 4.9% |
| Weight programming | 5.4% | 1.3% |
| Crosstalk | 12.4% | 2.4% |
| All together | 32.6%, 9.7 dB | 7.1%, 23.0 dB |

*Read this with the correction at the head of the section.* Its "v0" column is
X1's setting as it was run, four LSB of an 8-bit ADC and 0.75 photons, and not
version 0's rows.

Two things in it bear on this section. **On a GEMM the parts add**, in
quadrature, to within 1%: the compounding above was not in the tile's sums, and
has since turned out not to be in the network's accuracy either. And **the noise rows are in ADC LSB, which is not a fixed
quantity**: v1's receiver noise is a quarter of v0's in LSB and an eighth in the
product's units, because v1's ADC has another bit and so half the LSB. A receiver
is designed to volts, so the row it is held to has to name the ADC it stands
beside. The numbers are one 4 × 4 tile, one shape and uniform int8 operands, and
they move with all three.

*What C3 priced, 2026-09-23.* Both halves. C3(a) measured the interval: a
calibration holds about a quarter of an hour at TFLT's fit, not the hour this
table assumed, and the correction itself is complete at every age and both fits
([`pta_chiplet_calibration.md`](pta_chiplet_calibration.md) §8). C3(b) measured
the cost: in RTL a calibration is a few thousand cycles, and the shadow scheduler
hid three quarters of that in stalls the tile was waiting through anyway — so the
interval this table wants is affordable, and the recalibration row is a schedule
rather than a tax.

*What the shot rate asks of the weight path, 2026-10-03.* Neither table says how
fast weights have to be written, and it decides whether a shot rate is a
throughput. A weight set — 16,384 cells at 256 × 64 — is shot for one batch and
then replaced, so the tile alternates between programming and shooting.
[`pta_shot_rate.py`](pta_shot_rate.py) §5 prices the split, taking a write beat as
one shot period and sweeping the write's width rather than assuming one:

| Cells written a beat | Batch 16 | Batch 64 | Batch 256 |
|---|---|---|---|
| 1, as the c930's scan does | 0.1% | 0.4% | 1.5% |
| 64, one input's row | 5.9% | 20% | 50% |
| 256, one output's column | 20% | 50% | 80% |
| 16,384, the whole set | 94.1% | 98.5% | 99.6% |

That is the share of its time a **one-bank** tile spends shooting. The c930's
serial scan does not carry over: it would leave this tile shooting 0.4% of the
time at batch 64. With one bank, 90% takes 2,341 cells a beat at batch 64, and at
batch 16 the whole set in a single beat. **A second bank changes the requirement
rather than relaxing it**: the next set loads behind the current set's shots, so
the tile never waits as long as a set programs within one batch, which is
`k×n / batch` cells a beat — 1,024, 256 and 64 at batches of 16, 64 and 256. So
the interface chip is held to two weight banks and a write path one output's
column wide at batch 64, each DAC rewritten at the shot rate over the batch,
15.6 MHz at 1 GS/s. The batch is the lever here as it was on the link: every
quadrupling of it quarters the write path.

The converters are stated as a rate and no further: 64 ADCs at the shot rate is
64 GS/s of 7-bit conversion at 1 GS/s. Turning that into watts needs a device
figure this program does not hold.

*Priced, 2026-10-04* ([`pta_power.py`](pta_power.py)), from published parts and
not from a design. For 256 × 64 at 1 GS/s under version 1:

| On the interface chip | Watts | From |
|---|---|---|
| 64 receivers | 0.26 | A measured 40 nm amplifier, 4.1 mW |
| 64 ADCs at 7 bits | 0.07–0.20 | The survey of published converters: the best and the fifth-best of the 70 that do 7 bits at 1 GS/s or faster, 1.1 to 3.1 pJ a sample (below) |
| Driving 256 input modulators | 0.08–1.5 at a 2 V swing, 0.21–3.8 at 5 V | TFLT's 1.96 V·cm, and an electrode capacitance that is **assumed** and swept |
| Rewriting 16,384 weights a batch | 0.002–0.32 | A cell capacitance that is **assumed** and swept |
| The link, weights re-sent with each batch | 1.8–3.1 | UCIe's own target of 0.75–1.25 pJ a bit, both ends |
| The link, weights resident | 0.13–0.22 | The same |
| **In all** | **0.6 to 8** | A floor: no DAC, weight store, clock or ring control is in it |

Four things in it.

- **The converters are 0.33 to 0.46 W, and most of that is the receivers.**
  The ADCs are priced from every published converter that can do the job and
  the receiver from one paper, so the larger figure is now the less well
  founded. The ADC's seventh bit is nearly free at the head of the field and
  doubles the ADCs behind it (below).
- **Two things can each cost more than all the converters.** The link, if the
  weights are re-sent with every batch, and the inputs' drive at a high swing.
  Residency is worth nearly fourteen times the link's power, as it was worth sixteen
  times the feed (§8, question 1), and the weight store that buys it is still
  unsized.
- **The swing buys area with power, one for one.** A Pockels modulator shortens
  as its swing rises and draws more by the same factor. The floorplan's "a 1 V
  chip does not fit" has its other half here: a 5 V chip pays five times a 1 V
  one's drive.
- **With the laser it comes to 39 to 520 fJ a multiply-accumulate.** For
  comparison only: a microring design's estimate of itself at four bits is
  28 fJ, and the same paper puts a 7 nm digital part at 1,140 fJ at eight (T.-C.
  Hsueh, Y. Fainman and B. Lin, arXiv:2402.08192). Neither is a measurement of
  a tile like this one, and this figure is a floor.

*The ADCs, widened to the survey the same day.* The table first priced them
from three converters read one at a time from their papers, at 34 to 72 fJ a
step: 0.28–0.59 W. None of the three reaches 7 effective bits.
[`pta_adc_survey.py`](pta_adc_survey.py) replaces them with B. Murmann's ADC
Performance Survey, every converter published at ISSCC and the VLSI Symposium
from 1997 to 2026: 763 operating points, from the spreadsheet its author keeps
(github.com/bmurmann/ADC-survey).

A column needs a converter of at least so many effective bits, at the shot rate
or faster. Any published part that does both can do the job and no other can,
so the price is taken from those parts, by the energy each spends on a sample:

| 64 converters of | at | Published parts that can | Best, a sample | Fifth-best | The 64 |
|---|---|---|---|---|---|
| 6 bits (version 0) | 1 GS/s | 89 | 1.06 pJ | 1.48 | 68–95 mW |
| 7 bits (version 1) | 0.1 GS/s | 193 | 1.11 | 2.42 | 7–15 mW |
| 7 bits | 1 GS/s | 70 | 1.11 | 3.14 | **71–201 mW** |
| 7 bits | 10 GS/s | 11 | 5.89 | 35.0 | 3.8–22 W |
| 8 bits | 1 GS/s | 47 | 4.19 | 5.35 | 0.27–0.34 W |

A part faster than the shot rate is priced at the shot rate, which **assumes**
its power follows its clock down. Five is the number of points the survey draws
its own envelope through.

Four things in it.

- **Version 1's converter exists as a part.** 45.5 dB — 7.3 effective bits —
  at exactly 1.00 GS/s for 2.55 mW as measured (D.-R. Oh et al., VLSI 2020,
  28 nm): inside the range above, with nothing assumed. Three cheaper ones
  have been published since, the best of them 3.0 mW at 2.7 GS/s (J.-C. Wang
  and T.-H. Kuo, ISSCC 2023, 28 nm).
- **The seventh bit is nearly free at the head of the field, and doubles the
  ADCs behind it**: 3 to 106 mW for the 64. The cheapest converters at this
  rate are 8-bit designs that reach 7, and the cheapest 6-bit part is within
  5% of them. An eighth bit is not free anywhere: 0.14 to 0.20 W more.
- **The ADCs follow the shot rate to 2.7 GS/s and not past 5.** The best part
  is the same one up to 2.7 GS/s, and the energy a sample is within half as
  much again to 4.8. At 10 GS/s it is five times as much, which is over fifty
  times the power for ten times the rate: the 64 would be 4 to 22 W, and only
  eleven published converters do 7 bits at 10 GS/s at all. The modulator
  allowed 51 GS/s and the receiver under half of that (§8, question 1).
  Neither of those bounds says what the rate costs, and this one does.
- **The receivers are now the larger half of the converters**: 0.26 W from one
  paper's amplifier, against ADCs priced from seventy.

The survey has been read here twice. Before the spreadsheet was to hand it was
read from the plots of its author's ISSCC 2022 short course, whose markers are
vector drawings: 546 converters matched across three plots. Held to the
spreadsheet, 541 are rows of it, to 0.04 dB and 0.1% in power, and the other
five are chance agreements of three different parts' markers, none of them in
the tile's corner. The pricing that reading gave does not stand, all the same.
It said 0.14–0.33 W, a seventh bit as dear as the first six, and fifteen to
sixteen times the power for ten times the rate. The plots stop at 2021, and
three of the four cheapest 7-bit converters are later. The least efficient
parts were off one plot's axis, so its "leading quarter" was a count of what
was missing: 40 fJ a step, where the whole record says 86. And it priced by
the parts *near* a rate, scaled to seven bits by a figure of merit — where a
4.8 GS/s part is near 10 GS/s and cannot sample at it.

The power is each paper's own, which usually leaves out the reference, the
clock and whatever drives the input. These are conference papers, each the
best its authors could show, and not parts that can be bought.

*What the floorplan asks, 2026-10-03.* Two things neither table has a row for
([`pta_floorplan.py`](pta_floorplan.py)). **The swing.** On TFLT a modulator's
length is 1.96 V·cm over the voltage its DAC swings, so the interface chip's
drive voltage sets the photonic die's length: 3.9 mm at 5 V, 9.8 at 2 V, 19.6
at 1 V. A 1 V chip and a 256 × 64 tile do not fit one standard die at any
pitch. **The bond.** The chip reaches every weight through a pad of its own, so
it spans the pad field whatever its circuits need — 10 mm² at a 25 µm pitch,
167 at 100 — and for the tile to fit a standard die the pitch has to be 53 µm
or finer at 5 V and 33 µm at 2 V. And the chip sits on the tile: its
converters' power, unpriced above, is heat in the one place the weights are
sensitive to it.

*What each row is worth, 2026-10-03.* With the unit fixed the two versions are
1.24 points apart on the D3 network, and grx930's budget run prices the six
rows between them one at a time, from both ends:

| Row | v0 → v1 | Buys, tightened from v0 | Costs, relaxed from v1 | What it costs to build |
|---|---|---|---|---|
| Activation DAC | 5 → 6 bits | 0.10 | 0.10 | A bit on every input's DAC |
| ADC | 6 → 7 bits | 0.09 | 0.12 | A bit on every column's converter, at the shot rate: **3 to 106 mW** for the 64, on the tile. Nearly nothing if the converter is one of the best published, and as much again as the six if it is not (priced above from the survey, 2026-10-04) |
| Receiver noise | 1 → 0.5 LSB of an 8-bit ADC | 0.18 | 0.19 | **Twice the laser**: 0.33–3.3 W in place of 0.16–1.6 at the receiver B5 assumed, 0.09–0.88 in place of 0.04–0.44 at a measured one (B5) |
| Light at each detector | 3 → 15 photons per such LSB | 0.34 | 0.34 | Nothing: see below |
| Weight programming error | 4 → 1 LSB of an 8-bit weight | 0.42 | 0.33 | The weight DAC's precision, or C3's trim |
| Crosstalk | 10% → 2% | 0.08 | 0.16 | Layout on the photonic die |
| All six | | 1.21 | 1.24 | |

Three things in it.

- **The rows add, so the budget is a menu.** Two settings were predicted by
  adding these before they were run, and came in at 96.97 for 96.98 and 96.65
  for 96.72. A row can be bought or left at its own price. §7's "stated
  jointly, never item by item" was the response to an effect that was not there.
- **The light row is not a requirement.** By B5's own arithmetic the receiver
  row asks for over a thousand times the light the photon row does, so no tile
  that meets the first is anywhere near the second: at B5's receiver an 8-bit
  LSB is some 6,000 electrons a shot under version 0 and twice that under
  version 1. The 0.34 points this row costs at 3 photons are points a real tile
  does not lose. With no shot noise at all version 0 is 96.30 and version 1 is
  97.29. The row can go, and with it the largest term in the single-GEMM table
  above.
- **Crosstalk is cheap, and it is not noise.** Ten per cent of it puts 26% of
  error on the network's outputs and costs 0.16 points, where thermal noise of
  that size costs about three. Most of it is a gain, which an argmax does not
  see. So an error fraction overstates it, and the single-GEMM table's 12.4%
  for crosstalk is not comparable with the rows beside it.

What this leaves to decide is which rows to buy. Version 1 buys all six for a
quarter of a point. The two whose cost this plan can name are the receiver
noise, which is the laser, and the ADC's bit, which is power on top of the
tile; each is worth under a fifth of a point here. That is one network's
answer (§8, question 7). A deeper network is where a layer's error really is
the next layer's input many times over, and where "compounds" could yet be
true.

*With depth, 2026-10-03.* Something deeper has been run: the same network with
two, four and eight hidden layers of 100, five of each, through grx930's
`sim/pta_mnist.sh depth` (its `c930/doc/pta_error_model_design_note.md`, §5).
Loss in points against the same networks on their host:

| | 1 hidden layer | 2 | 4 | 8 |
|---|---|---|---|---|
| Version 1 | 0.26 | 0.23 | 0.30 | 0.30 |
| Version 0 | 1.49 | 1.96 | 2.44 | 3.19 |
| Version 0's five rows together, less what they sum to | +0.08 ± 0.16 | +0.28 ± 0.10 | +0.34 ± 0.21 | **+0.81 ± 0.15** |

And the table above again, each of version 1's rows relaxed alone to version
0's, by depth:

| Row | 1 hidden layer | 2 | 4 | 8 |
|---|---|---|---|---|
| Activation DAC, 6 → 5 bits | 0.09 | 0.27 | 0.32 | 0.54 |
| ADC, 7 → 6 bits | 0.12 | 0.21 | 0.11 | 0.20 |
| Receiver noise, 0.5 → 1 LSB | 0.18 | 0.38 | 0.23 | 0.22 |
| Light, 15 → 3 photons | 0.34 | 0.31 | 0.23 | 0.21 |
| Weight programming error, 1 → 4 LSB | 0.33 | 0.42 | 0.53 | 0.59 |
| Crosstalk, 2% → 10% | 0.16 | 0.35 | 0.35 | 0.34 |

Four things, and the first is why version 1 stays the requirement.

- **Version 1 holds at every depth, and version 0 does not.** Version 1 costs a
  quarter to a third of a point from one hidden layer to eight. Version 0 goes
  from a point and a half to over three.
- **"Compounds" is true at depth, in a small way.** At eight hidden layers
  version 0's rows together cost 1.4 times what they sum to, five standard
  errors clear of adding. So the menu is exact for a shallow network and
  optimistic for a deep one: relaxing several rows at once costs more than
  their prices say. The factor is 1.4, and not the six that was withdrawn.
- **Noise does not accumulate, and deterministic error does.** Thermal and shot
  noise put the same error on the outputs of a nine-layer network as of a
  two-layer one, and cost no more: a layer's noise does not survive the next
  layer undiminished. The activation quantiser's error is the same function of
  the signal at every layer, and grows by three points and more a layer.
- **So the menu moves.** At eight hidden layers the two dearest rows to relax
  are the weight programming error and the activation DAC's sixth bit, which at
  one hidden layer was the cheapest row there is. The two whose cost this plan
  can name — the receiver noise, which is the laser, and the ADC's bit — stay
  between a tenth and two fifths of a point, with no trend in depth.

That sharpens what there is to decide and does not decide it. If a row is to be
relaxed for the sake of the laser, or of the converters that sit on the tile,
those two are the rows at every depth tried, and the two DACs are the rows to
leave alone. It is still fully connected layers on MNIST, and networks trained
without the impairments in the loop (§8, question 7).

*The ADC's row, since the survey (2026-10-04).* Relaxing it saves 3 to 106 mW,
and the low end is the published state of the art: the cheapest converters that
sample this fast already have the seventh bit. So of the two rows, relaxing the
receiver's saves 0.04 to 0.44 W of laser at a measured receiver whatever
converter is built. Relaxing the ADC's saves 0.1 W if the converter built is
the fifth-best published, and next to nothing if it is the best.

### 4.4 To the PTA program

- ~~C3, the calibration engine, continues, and becomes X3.~~ **Done**, both
  halves: C3(a) measured, C3(b) built (X3's §8).
- C4 splits. **C4(a) is done, both halves, 2026-09-24**: the map X4 shares with
  the chiplet is an implemented one now, checked over AXI-Lite and by a RISC-V
  program driving it through a crossbar, a D-cache and a DMA. Two of its findings
  outlive it and belong to any board driver — a buffer the accelerator writes must
  not be written by the CPU first, and the calibration's probe amplitude is a bit
  position whose bounds no register reports (CPU document §3.3).
- **C4(b), the FPGA numbers, is part done, 2026-09-24.** Routed out of context on
  the 200T, the digital baseline is 58.0 MHz — limited by the FP16 accumulator
  chain between PEs, which predates the photonics — so the 100 MHz the plan asks
  for was never within reach of this NPU, with or without a tile. §6.1's stated
  baseline turns out to be no run's: one NPU alone uses more FFs and more DSPs
  than the whole SoC is credited with. What this means for the board is that a
  rev-0 FPGA platform should be sized from measurements, not from that table, and
  that the clock target in §4.3's pricing needs revisiting. The PTA tile's own
  area is still owed: it does not synthesize on the 5.9 GB VM this work runs on.
  C4(c), the §6.2 sweep, is still ahead and waits on MB.
- C4's target moves: the CSR map goes to CXL.io behind the GPU (X4), not to the
  Arty A7 SoC's bus.
- Step F3's requirements for the fabric go to X2.
- G2's cluster-scope tile gives way to the chiplet attach (B4).
- The error-model tile becomes the chiplet's digital twin (X5). *Built
  2026-10-04*, §3.3.
- The scope lines of both integration documents are rewritten (§6).

---

## 5. Order

1. ~~Settle §2.~~ **Done:** B1 and B3 on 2026-09-21, the rest on 2026-09-22.
2. ~~On paper, now: P0, X1, X2 and X4.~~ **All four are in:**
   [`board_icd.md`](board_icd.md),
   [`pta_chiplet_regmap.md`](pta_chiplet_regmap.md), §4.3 and §3.3.
3. C3 continues in the PTA program, building what X3 specifies.
4. P2, as soon as B6 names the FPGA platform. X5's twin is built, and the half
   of P2's gate that needs no board already passes on it (§3.3).
5. L1–L4, in grx930's and grxgpu's silicon plans.
6. P1, before any package is committed.
7. P3, last.

---

## 6. Edits this plan asks for

Each lands when its decision settles, in its own document's repository.

- [`chip_vs_board_strategy.md`](chip_vs_board_strategy.md), whose author is
  still writing it: UCIe joins dies within one package (§1); the PTA's bulk
  data path is electrical up to the modulators (§1); optical I/O is not
  photonic compute; Raw mode has no CRC or retry, and one control path is
  enough (B7); FP8 reaches the tile block-scaled (§4.2).
- [GRX_GCPU.md](../GRX_GCPU.md) §2: CXL, and why it reverses the TileLink
  recommendation (B3). *Made; §9.*
- [`pta_gpu_integration.md`](pta_gpu_integration.md): its scope line; §2's
  cluster-scope placement, superseded by the chiplet (B4); G2, re-scoped.
  *Made; §9.*
- [`pta_cpu_integration.md`](pta_cpu_integration.md): its scope line; C4's
  target; the home of the §3.1 block (X4). *Made; §9.*
- [`pta_program_plan.md`](pta_program_plan.md): F3 pointed at X2; C4 and G2
  re-scoped; a pointer to this plan. *Made; §9.*
- [`ai_motherboard_design_capabilities.md`](ai_motherboard_design_capabilities.md):
  the thermal section's thermo-optic assumptions replaced with TFLT's, and the
  laser placed (B5). *Made; §9.*
- [`heterogeneous_devices.md`](heterogeneous_devices.md) §4.1: the coherent
  shared pool (S2).
- grx930's `c930/doc/GRX930_manufacturing_plan.md`, which that team owns: the
  board CPU's node (B6), as a requirement.

---

## 7. Risks

| Risk | Where it bites | Response |
|---|---|---|
| UCIe and PCIe 5.0-class PHY IP: which nodes, availability, license cost | B6, L1–L3 | Source it early, and prototype on FPGA hard IP first |
| Advanced-packaging capacity and cost | B2, P3 | Settled for rev A by B2: organic substrates, GDDR6 and DDR5, with HBM left to the Phase 2 module |
| A requirement in LSB of an ADC it does not name. X1's joint runs carried C1's noise rows from an 8-bit ADC to a 6-bit one as the same numbers, which is four times the noise, and until 2026-10-03 this row read "analog error compounds" on the strength of it | X1, B5, and every figure sized from §4.3 | Corrected (§4.3). A noise row names an 8-bit ADC wherever it appears, and grx930's harness reports every run in that unit beside the one it was asked in |
| The budget's rows stop adding with depth: version 0's cost 1.4 times their sum at eight hidden layers, and the rows that are cheap in a shallow network are not the ones that are cheap in a deep one | Any relaxation of §4.3 for a deep workload | Version 1 itself holds to eight layers. A relaxed set is priced jointly, at the depth it is for, as §4.3's depth tables do — not from the shallow menu |
| Drift needs calibrating about every quarter hour at TFLT's fit to hold the gate's margin, not hourly as C1's sweep suggested | C3, the schedulers | C3(a) measured the hold curve ([`pta_chiplet_calibration.md`](pta_chiplet_calibration.md) §8); the shadow scheduler has more idle windows to hide in than the c930 did (X2) |
| TFLT dies not available in the volume or quality needed | B5, Track X | TFLN as the fallback, with calibration sized to its drift |
| RISC-V support in Linux's CXL subsystem | L4, S1 | Firmware planned alongside L1, not after it |
| Coherence verified across a chip boundary, in the c930's L2 and the G100's | L1, L2 | Begin with GRX_GCPU.md's small, configurable coherent region, and grow it |
| The link, not the optics, sets the PTA's throughput | B4, X2 | Keep weights, activations and accumulation on the EIC, and size the link from the feed model |
| Laser power, reliability, fiber attach and eye safety | B5, P1 | An off-package laser with an interlock, and a power budget from X1 |
| Export controls on board-level products (the strategy document) | Phase 2 | Classify the product before the first shipment |

---

**B8 — The die-to-die link's signalling: NRZ, not PAM4.** *Settled
2026-10-01.*

Mostly settled already by B4, and worth writing down because the arguments
usually offered for it are not the ones that apply. UCIe-S standard-package
signalling at 32 GT/s is NRZ, and X2's model is parameterised on exactly that
— one module of 16 lanes at 32 GT/s, 57.6 GB/s a direction at 90% efficiency.
PAM4 enters only at UCIe's higher-rate modes or on a custom link, and leaving
UCIe costs the interoperability B4 chose it for.

*What actually supports it:* bandwidth is the weakest lever this design has.
PAM4 buys 2×. On X2's 256 × 64 tile at 1 GS/s, the batch buys 15× — 1,028
GB/s inbound at batch 16, 260 at 64, 68 at 256. B4's own split removes 38 to 75%
of the traffic that is not weights (X2 §4). Weight residency, which step MB
built and measured on the c930, removes the weight traffic across batches, and
at batch 64 the weights are most of the traffic either way. So the ordering is
residency, then batch, then where the activation stage sits, and modulation
last. At batch 64 a 256 × 64 tile is five modules inbound, which is a sane
number for a link we are not otherwise straining.

*Two arguments not to use,* recorded so they are not re-imported later:

- **FEC latency.** The usual case against PAM4 is that forward error correction
  adds latency. The PTA plan's F3 measured this program's budget: the tile waits
  101 cycles — **1,010 ns** — for an A row before it stalls, against the 100 ns
  request-and-return X2 assumes. There is roughly 900 ns of slack and FEC fits
  inside it comfortably. Our own measurement refutes this argument.
- **The 9.5 dB SNR penalty reaching the computation.** It does not. The link is
  digital and FEC-protected, so the penalty lands in the SerDes and not on the
  tile's analog budget. That budget is separately tight — C1's accuracy sweep
  missed its allowance at 3 bits, and A3 measured a 25 to 57% uncalibrated
  detuning residual at the knee — but PAM4 would not have made it worse.

*What would flip it:* a 256 × 128 geometry with the batch capped at 16. Inbound
passes 2 TB/s and the module count passes thirty, and there modulation stops
being the smallest lever and becomes the difference between feasible and not.
Even then the first response is batch and residency.

*What this does not settle:* the tile's **optical** shot rate, which is open
question 1 below and a different quantity entirely — how often the photonic
tile fires one MAC, `M·Nt·Kt` of them a GEMM, with `PTA_TS` its duration.
X2 quotes shots in GS/s and the link in GB/s separately for that reason. The two
share the phrase "shot rate" and share nothing else, and a source that defines
it as "the per-lane modulation rate of your photonic transceivers" is answering
the link's question, not the tile's. Question 1 has since been bounded
([`pta_shot_rate.py`](pta_shot_rate.py)), and the modulator turns out not to be
among the bounds that bind.

**B9 — How grxcp sees the chiplet.** *Settled 2026-10-04.*

Two documents answer this two ways. B4 calls the PTA "a device-level engine that
the whole GPU shares" and B7 gives its register block to the GPU's driver, which
reads as an engine of the GPU device. grxBLAS's dispatch rule reads the other
way: the current device decides which engine a GEMM runs on, nothing consults a
preference, and nothing falls back, "because the alternative is the same source
line running on different silicon depending on state set somewhere else". A GPU
device with two GEMM engines needs exactly that state. And `analogGemm` would
stop meaning what it means on a c930, that a GEMM on this device is impaired.

*Recommended, and settled as recommended:* the chiplet is a **device of its own**
in grxcp's table — `GRX_DEVICE_TYPE_PTA`, GEMM-only, appended after the GPUs —
and its GPU is its **parent**. That keeps both rules whole and costs one
exception, stated once. The chiplet has no memory, so a GEMM on it takes pointers
allocated on its parent, and `grxDeviceProp_t.parentDevice` says which device
that is. That is B4's copy engine seen from the API: the operands are in the
GPU's memory because the GPU is what feeds the tile. Everything else on the
chiplet is refused. S3's placement becomes a choice of device, made above
grxBLAS, where [`heterogeneous_devices.md`](heterogeneous_devices.md) §6 already
put it. *Needed by:* S4, S3 and S1.

---

## 8. Open questions

1. **How big is the PTA chiplet, and how fast?** Its inputs, outputs and
   **optical** shot rate set X2, the laser (B5) and the EIC's area. Nothing here
   fixes them, and X2 sweeps candidate geometries rather than claiming one
   because of it. The *link's* signalling is no longer part of this question:
   B8 settles that as NRZ. Geometry and batch are what decide the module count,
   and the module count is what could reopen B8.

   **The shot rate is bounded, 2026-10-03**
   ([`pta_shot_rate.py`](pta_shot_rate.py)). Three things cap it, and only one
   of them is still open. For X2's 256 × 64 tile at batch 64, under §4.3's
   version 1:

   | Bound | What it allows | Does it bind |
   |---|---|---|
   | The modulator, settling to half an LSB | 51 GS/s at the 45 GHz TFLN anchor | Never. The receiver allows under half of it at any laser B5 planned on |
   | The feed, weights re-sent with each batch | 0.22 GS/s a module: 1.1 at X2's five, 6.6 at the thirty where B8 reopens | Only with loss near 10 dB and the laser at the top of B5's range |
   | The feed, weights resident on the interface chip | 3.6 GS/s a module, and it is the outbound direction that limits | Hardly: only at the top of B5's laser range with loss near 10 dB, and then only under the kinder noise law. And it needs the 16.8 MB a layer like this one holds, which is a digital store, not the DAC-held residency B4 lists, and nothing has sized it |
   | The receiver | 1 GS/s needs 0.33 W of laser behind 10 dB of loss and 3.3 W behind 20 | Yes, in every case except the laser at the top of B5's range with the loss near 10 dB |

   So the question has changed shape. It is no longer how fast the tile is but
   **how much light reaches each detector**. With the largest laser B5 planned
   on, 1.6 W, X2's 1 GS/s holds only if laser-to-detector loss stays under
   **16.9 dB**, and B5's own range runs to 20. Past that each 3 dB costs between
   1.6× and 4× in rate, depending on a receiver nobody has designed: at 20 dB
   the same laser gives 0.24 to 0.62 GS/s. (These read 13.9 dB, 0.06 to 0.4,
   an eighth and 0.66–6.6 W until version 1's receiver noise was restated in the
   unit it was run in; §4.3.) The number to ask a foundry for is the
   loss budget, and the number to measure is the receiver's noise, which B5
   assumed at 1 µA and every figure here is linear in. Geometry is still open
   and moves this directly — a 256 × 128 tile has twice the detectors and wants
   twice the laser.

   *At a measured receiver the ceiling is 22.6 dB, 2026-10-04.* All of the above
   is at the 1 µA B5 assumed. A published receiver's noise is 0.27 µA over the
   same bandwidth (B5; [`pta_power.py`](pta_power.py)), and the ceiling moves
   with it, past the top of B5's range: a laser of 0.88 W or more reaches 1 GS/s
   at any loss in it. So the receiver would bind only for a smaller laser than
   that, and the feed becomes the bound that matters. The number to measure is
   still the receiver's noise. It is now one where a published part says the
   answer may be nearly four times better than this plan assumed.

   A modulator's line rate does not enter. A table of NRZ benchmarks was offered
   as the tile's shot rate — 56 Gbaud in production, 100 to 180 Gb/s in
   research — and it is B8's confusion again: those rates are recovered by an
   equaliser and protected by FEC, and an analog level has neither.

   A shot rate is only a throughput if the weights keep up, and that is now a
   requirement on the interface chip rather than an unknown: two weight banks
   and a write path `k×n / batch` cells wide (§4.3). This paragraph ended
   "still not priced is the converters' power, which needs a device figure"
   until 2026-10-04. It has been priced from published parts, in §4.3: 0.33 to
   0.46 W at 1 GS/s, and not the largest thing on the chip. The ADCs' share of
   it follows the shot rate to 2.7 GS/s and stops following it at about 5: at
   10 GS/s they alone are 4 to 22 W, where the drive and the link only scale
   with the rate.

   **A package with millimeters on it, 2026-10-03.** Nothing above has a length
   in it. One published package does: X. Li et al., "1.6 Tbps FOWLP-Based
   Silicon Photonic Engine for Co-Packaged Optics", *J. Lightwave Technol.*
   43(4), 1979 (2025), doi:10.1109/JLT.2024.3493855, from Rain Tree Photonics,
   Singapore's Institute of Microelectronics and Advanced Micro Foundry. It is a
   transceiver and not a tile, and silicon photonics and not TFLT, so it is a
   reference for the **package** and says nothing about the computation.

   | | What the paper reports | What it bears on |
   |---|---|---|
   | The package | 9.5 mm × 13 mm, fan-out wafer-level, made on a 300 mm line. The photonic die is molded in and the electronic dies flip-chip on top, directly over its RF pads. No wire bonds and no through-silicon vias | B4's "stacked or side by side" |
   | Down to the board | C4 bumps of 120 µm at 250 µm pitch onto an organic substrate; through-mold vias of 150 µm at 300 µm pitch, 300 µm tall; redistribution at 15 µm line and space, two layers above the die and one below | B2's substrate, P1's floorplan |
   | The photonic die | 8 travelling-wave Mach–Zehnder modulators and 8 germanium waveguide detectors, with thermo-optic and passive circuits. **Its size is not given**, nor the electronic dies', nor the micro-bump pitch between them | The size this question asks for, which it leaves unanswered |
   | Light | Edge couplers along one edge to a fiber array at 250 µm pitch: **under 2 dB a facet** after packaging, without index-matching epoxy, and comparable to the bare die. Grating couplers through windows in the redistribution dielectric, for wafer-level test | The loss budget above, P1's fiber attach |
   | The electrical path | Simulated, 1.1 dB at 56 GHz through 2 mm of substrate line, a via and the redistribution. Measured, a via alone: under 0.5 dB to 50 GHz | A link that crosses such a package |
   | Rate | 112 GBaud a lane: NRZ open on a 5-tap transmit equaliser, PAM4 at a TDECQ of 2.44 dB through 9 receive taps, at an extinction ratio of 4.07 dB | B8's question and not this one: an eye read through an equaliser is not a settled level |

   **What was measured is the package, not an engine running.** The electronic
   dies are in the photograph and not in the signal path: the modulators were
   driven by a waveform generator and an external amplifier through probes, and
   the detectors were read at probe pads, with light from an external source
   through the fiber array. The abstract's 1.79 Tb/s is eight times one lane's
   224 Gb/s — the title's 1.6 is the same eight lanes at their nominal 200 — and
   the text does not report eight lanes driven at once. No power and no
   temperature is reported, and neither the wavelength nor the fiber's
   polarization is stated.

   Three things in it bear on this plan, and each is smaller than it looks.

   - **The loss budget has its first sourced term.** The tile's light crosses
     one facet on the way in and, with the detectors on the die, none on the way
     out. Under 2 dB of the 16.9 dB ceiling above leaves about 15 for the fiber,
     the excess loss of the split to the rows, the modulators and the tile. What
     the measurement supports is that molding a die into a package cost its
     coupler nothing. It is a silicon spot-size converter: it is not what a TFLT
     facet loses, nor what one loses on the polarization-maintaining fiber B5
     needs.
   - **A fiber count, and it collides with question 8.** At 250 µm a 13 mm edge
     holds 52 fibers and a 9.5 mm edge 38, were the whole edge facet. An emitter
     an input row, brought from off the package on a fiber a row, is 256 fibers:
     64 mm of facet, five times this package's longer side. So at this pitch
     B5's "off the package" and question 8's "an emitter an input row" hold
     together only if the rows' lines are combined onto a few fibers before they
     arrive, which wavelength allows a ring bank. Otherwise the emitters come
     onto the package, or the pitch is finer. 250 µm is this paper's pitch and
     not a limit.
   - **B4's "stacked" has been built**, without through-silicon vias in the
     photonic die and onto the kind of substrate B2 chose. What the tile asks of
     such a stack is not what eight lanes ask. B4 keeps a DAC-held voltage for
     each resident weight on the interface chip, and each needs its own way down
     to an electrode on the photonic die: at least 16,384 connections at
     256 × 64, before the 256 inputs and the 64 outputs. That count times the
     square of a pad pitch is an area neither die can be smaller than — the
     first bound on this question that is not optical — and the paper gives no
     pitch to work it with.

   What it does not give is the answer. There is no die size, so no area a
   modulator or a detector; no power; nothing on heat but a sentence in its
   introduction; and the molding, the thinning to 300 µm and the laser drilling
   were shown around a silicon photonic die, so whether a die carrying lithium
   tantalate stands the same steps is not something it tests. The summary this
   paper arrived with closed by saying an NRZ geometry needs about half the lanes
   of a PAM4 one. The paper does not say so and its figures say the reverse: the
   same eight lanes carry 896 Gb/s as NRZ and 1.79 Tb/s as PAM4.

   **Pitches, and what they do to that count, 2026-10-03.** A second batch of
   sources came with pitches in it, so the connection count above has something
   to be multiplied by. Each was read, or its abstract was. What each is, and is
   not, is in the last column.

   | Figure | Source | What it is |
   |---|---|---|
   | 100 µm pad pitch on modulators and detectors, so that bare logic dies flip-chip onto them | Y. Urino et al., *Proc. SPIE* 9010, 901006 (2014), doi:10.1117/12.2041418 | Designed and built: a silicon optical interposer, 20 Gb/s links, 30 Tb/s/cm² |
   | 40 µm pitch: 37 grating couplers in a hexagon, under one multicore fiber | V. Kopp et al., *J. Lightwave Technol.* 33(3) (2015), doi:10.1109/JLT.2014.2364579 | Built, with 0.7 dB of spread across the channels. Only the abstract could be read, so the mean loss is not held here |
   | 127 or 250 µm fiber pitch, 1 to 24 channels an array, two arrays a die on opposite edges; 150 µm wire-bond pitch and at most 120 pads; dies of 3 × 3 to 10 × 10 mm; an 8 W cooler holding ±0.01 °C | Europractice packaging design rules v1.7 (Tyndall, September 2024) | What a multi-project packaging service sells as standard: a floor on what can be bought, not a limit on what can be made |
   | 25 × 25 µm² a cell, for a 64 × 64 weight bank | US 2024/0370050 A1 (University of Pittsburgh), a coherent crossbar array | **An assumption**, made once, to say that 400 such banks pass 10 cm². A pending application, whose built device is a single cell |
   | 256 fibers on more than 4,000 mm², 8 lines a fiber; 16 lines a fiber on a stacked engine; a remote source of 16 lines | Lightmatter's M1000 and L200 and Ayar Labs' TeraPHY announcements, all of 2025-03-31 | Press releases for interconnect products. None states a die-to-die bond pitch, the engine's die size or a power |

   **The bond pitch sizes the die, and the optics do not.** At least 16,384
   connections at 256 × 64:

   | Pitch of a connection | Area of the weights' pads alone | |
   |---|---|---|
   | 100 µm, the flip-chip pitch Urino's interposer was designed to | 164 mm², 12.8 mm square | More than the whole 9.5 × 13 mm package above, which is 124 mm² |
   | 25 µm, the crossbar application's cell | 10.2 mm², 6.4 × 1.6 mm | A pad a cell means the bond pitch **is** the cell pitch |
   | 150 µm wire bond, 120 pads | — | Out by two orders. A perimeter does not carry an area's worth of weights |

   A factor of sixteen in area hangs on one number, the pitch at which the
   interface chip bonds to the photonic die, and no source here states one: the
   stacked products say "chip-on-wafer" and stop. That is the figure to ask a
   packaging house for, ahead of anything optical. And 25 µm is nobody's
   measurement: it is another topology's assumption, for a cell built round a
   splitter and a pair of detectors that holds no weight. No source here sizes a
   TFLT weight.

   The bound is B4 and B5 together: a weight held on one die and applied on
   another needs a way between them. A process with transistors beside the
   optics has no such bound — the TeraPHY announcement calls its GlobalFoundries
   process monolithic — but that process is silicon, which B5 turned down for
   what holding a weight costs it.

   **The fiber count, with edges on it.** The collision above stands.

   - What a standard service attaches is two arrays of 24, and an edge-coupled
     array gives four channels to alignment: 40 usable fibers, on a die of at
     most 10 mm. A 127 µm pitch halves the facet a fiber takes, and 256 fibers
     are still 32.5 mm of it.
   - 256 fibers on one part does exist: on the M1000's 4,000 mm² and more of
     interposer, thirty-two times the package above.
   - Both other ways out have a built instance. Every one of the products
     combines lines before they arrive, 8 or 16 to a fiber, and at 16 a fiber
     256 lines are 16 fibers and 4 mm of facet. And a finer pitch exists in two
     dimensions: 37 couplers at 40 µm under one multicore fiber is seven fibers
     for 256 rows. Those are grating couplers, whose 1 dB bandwidth the
     Europractice rules put at about 30 nm, so the lines a fiber carries have to
     fit inside that.

   **The thermal figures that came with these belong to the platform B5 turned
   down.** Silicon microrings at 70 to 80 pm/°C is B5's own opening line. The
   0.1 nm/°C quoted beside it is the same quantity rounded — the Europractice
   rules give 1 nm for 10 °C — and 70 pm is the smaller number, not the stricter
   one the summary called it. "A 4°C excursion ... isn't degradation, it's
   outright system failure", control to ±0.1 to 0.2 °C and keep-outs "from the
   first floorplan" are one contributed article's sentences, without citations,
   about ring modulators on a link beside a compute die of over 200 W/cm² (A. S.
   Chandrasekaran, *Semiconductor Digest*, 2026). Against them stands a vendor's
   own report: Lightmatter has a ring link of 16 lines holding a raw error rate
   under 10⁻⁹ through cycles from 25 to 105 °C, and through an "800 °C/s"
   aggressor that was a laser swept at 50 nm/s and not a temperature.

   None of it transfers as a number, for two reasons. This tile's weights are
   held by a voltage and not by a heater, so P1 models bias drift and the
   laser's wavelength (B5). And these are a **link's** tolerances: a link's ring
   has to stay inside its linewidth, where a weight has to hold a level to an
   LSB. If the tile is a ring bank (question 8) its rings move with temperature
   too, by a coefficient nothing here gives for TFLT, and its allowance is
   tighter than a link's by roughly the number of levels it holds — an estimate,
   not a figure. What does transfer is where the heat comes from. The stacked
   examples put the interface chip directly over the photonic die, so here the
   converters would sit on the tile, and their power is the figure this question
   still has not priced.

   **The three assembled products are links.** They move bits and none of them
   multiplies, so none is a candidate for the tile; and none is a link this board
   has, since B1 keeps the PTA beside the GPU and inside UCIe-S's reach. What
   they are precedent for is the tile's shell, and the TeraPHY is the nearest: a
   photonic chiplet behind UCIe, lit by a remote source of several lines, which
   is B4 and B5 as somebody else built them.

   **Not used.** "HEXA-PHOTON" (Zenodo, 2026) was offered as a concrete example
   of a 12 × 12 mesh on 12 wavelengths. It is a self-published design with
   nothing built or simulated: by its own abstract every parameter is "derived
   from the arithmetic functions of the perfect number n = 6", and its twelve is
   the sum of 6's divisors. The summary's "34 chiplets" for the M1000 and "NRZ
   with no FEC" for the TeraPHY are not in the announcements cited for them.

   **The floorplan, bounded, 2026-10-03**
   ([`pta_floorplan.py`](pta_floorplan.py)). The pitches above, UCIe's own
   module dimensions and the one Pockels figure the program holds — TFLT's
   1.96 V·cm (the CPU document, §4.4) — put together as a model of what has to
   be on the photonic die and under the interface chip. It is a floorplan in the
   sense P1 needs first and not a layout: every area is a lower bound. What is
   on the die is read off the error model, not chosen: a Mach–Zehnder modulator
   an input row (`MZM_NL`), a cell a weight, a detector a column. For 256 × 64:

   | What takes room | How much | What sets it |
   |---|---|---|
   | The inputs, a modulator a row | 3.9 mm long at a 5 V swing, 9.8 at 2 V, 19.6 at 1 V. All 256 at 5 V: 25 mm² if they sit 25 µm apart, 100 mm² at 100 µm | The swing a DAC holds, which nothing has specified. The lateral pitch is swept, not known |
   | The weights, if not resonant | As long as a modulator each: 1,606 mm² at 5 V and 25 µm, sixteen standard dies | The platform |
   | The weights, if resonant | 10.2 mm² at 25 µm to 164 at 100 | The bond pitch, standing in for a ring nobody has sized |
   | The lines between the dies | 16,704. Stacked, a pad each: 10.4 to 167 mm². Side by side, 251 mm of shared edge at 15 µm line and space | B4 and B5 together |
   | The facet | One fiber, or 16 at sixteen lines each, which is 4 mm | Does not bind |
   | The link | A UCIe-S module's PHY is 571.5 × 1,540 µm: 4.4 mm² and 2.9 mm of edge for X2's five. Brought down through the mold at 300 µm, 33 mm² | The package, not the PHY |
   | The light, as heat | 0.09 W/mm² at most, were all of a 20 dB laser absorbed on the densest die | Does not bind |

   Five things follow. The first two change what this section said a day ago.

   - **On this platform the weights are resonant, or the tile is not a
     chiplet.** A Pockels weight that attenuates by interference is as long as
     a modulator, and 16,384 of those are 40% of the largest photonic part in
     the references above. Short of a smaller tile, another material or a cell
     no figure here describes, the weights sit on resonances. So that much of
     question 8 is no longer open, and a level held on the side of a resonance
     moves with temperature and with the laser's wavelength. The thermal
     figures set aside above as belonging to "the platform B5 turned down"
     therefore come back, as a kind of sensitivity if not as numbers: B5
     escaped the heaters, not the rings. What a non-resonant tile can be is
     about a thousand cells — of X2's candidates, 8 × 8 and 64 × 8.
   - **The inputs are the largest optics on the die, not the weights.** "The
     bond pitch sizes the die, and the optics do not", above, was written about
     the weights and had not counted the inputs. At a 25 µm pitch the 256
     modulators outweigh the weights at every swing, 25 mm² against 10.2 at
     5 V, and the die is one modulator long whatever else is on it. If an input
     is a ring and not a Mach–Zehnder this goes away, and `MZM_NL` is then the
     wrong impairment in the error model.
   - **Two numbers size the die: the bond pitch and the swing.** The tile fits
     the largest die a standard service packages at a bond pitch of 53 µm or
     finer with 5 V inputs, 33 µm with 2 V ones and 78 µm if the inputs are
     resonant too. The first two are inside UCIe's advanced-package range of
     25 to 55 µm, and none reaches the 100 µm of flip-chip. At 1 V the
     modulator is 19.6 mm long and no pitch helps. So the voltage the interface
     chip's DACs swing is a floorplan input, and §4.3 had no row for it.
   - **"Stacked or side by side" is not open at this size**, within B2. B4's
     addendum has it.
   - **The module count is a package-area question before it is a signalling
     one.** X2's five modules are small as PHYs and 33 mm² as vias, three times
     the dense tile's weights, and B8's thirty have more bumps than a package
     the reference one's size has C4 sites.

   Not priced, and now more exactly located: the interface chip's own circuits,
   which sit on top of the tile; and behind the resonant reading, a TFLT ring's
   size, its linewidth and the swing that moves it by one. That a modulator's
   lateral pitch can be the bond's, and that a cell is no smaller than its pad,
   are assumptions the script marks where it uses them. (The first of those was
   priced the next day, in §4.3: the converters, the drive and the link. The
   weight store, the DACs and whatever holds a ring on its line were not.)
2. **What does the development kit cost, and how many are built?** That settles
   B1 and B2 more than any technical argument does.
3. **Does the GRX930's NPU keep a PTA of its own?** The c930 PTM work is built
   and gated. It can be a product feature, or only the development vehicle.
4. **Does the PTA ever need coherent access to host memory?** B3 gives it none;
   revisit that if the dispatch model (S3) says otherwise.
5. **Where does the board controller come from, and who writes its firmware?**
6. **Which CXL-capable FPGA platform hosts rev 0?** B6 settles that there is
   one; the part, its CXL IP and whether that IP can act as a host rather than
   a device are P2's first question.
7. **Does X1's budget hold on a second workload?** It is one 784-100-10 MLP.
   This question used to say that compounding was a property of analog sums and
   not of this network. It was a property of a unit (§4.3). What the corrected
   runs show is that on this network the rows **add**, and that is the network's
   property and not a law: with one hidden layer a layer's error is the next
   layer's input once. The numbers in §4.3, and the adding, belong to this
   network until something deeper is run.

   *Something deeper was run the same day, and the answer is: partly.* Version 1
   holds at two, four and eight hidden layers. Version 0 costs twice as much at
   eight as at one, its rows together cost 1.4 times their sum there, and the
   rows that were cheap to relax are not the same ones (§4.3). So depth is a
   second workload the budget survives and the menu does not. What has still
   not been run is anything other than a stack of fully connected layers — a
   convolution, a residual path, attention — or anything but MNIST, or a
   network trained with the impairments in the loop, which may tolerate more.
   Depth was the cheapest second workload to try and it changed which rows are
   dear; another kind of network may do so again.
8. **What kind of light source does the tile need?** Asked on 2026-10-03, about a
   quantum dot laser, and the answer turned out to rest on something this plan
   never decided. [`pta_shot_rate.py`](pta_shot_rate.py) §6 writes down what any
   source is held to, as requirements and not as a choice of part:

   | | What the tile asks | Where it comes from |
   |---|---|---|
   | Power | 0.33–3.3 W at 1 GS/s for 256 × 64, as one laser or as 1.3–12.8 mW from each of 256 emitters | B5's method at §4.3's version 1 |
   | Noise | Intensity noise within about −144 dB/Hz at 1 GS/s, ten tighter a decade of rate | The receiver's allowance, applied to the source. **Assumed**: §4.3 has no row for it |
   | Wavelength | 18% more light at 1310 nm than at 1550; within 2.4 nm of 1550 if the all-optical branch reopens | The detector's quantum efficiency; the TPA-QCN device's 12 nm of phase matching |
   | Kind | A line an input if a column sums powers; one coherent line if it sums fields | The tile's topology, below |

   **The kind is the open part, and it is open because the topology is.** The
   error model's crosstalk is written for a ring bank — an input's light passes
   its neighbours' rings (the CPU document, §4.3) — and in a ring bank the inputs
   are told apart by wavelength and a column sums powers. Such a tile does not
   merely allow a source of many lines, it requires one, 256 of them at this
   geometry, and a single-line laser is not a source for it. A mesh sums fields
   from one coherent source, and there an array or a comb is none. The CPU
   document's §8 calls the ring-bank topology a hypothesis with no ground truth,
   which was harmless while it only shaped a crosstalk term. It now decides what
   B5's laser is.

   So before a part: **which topology is the tile**, and if it is a ring bank,
   can one bank tell 256 lines apart — a device question no figure here answers.
   Three things follow whichever way that goes. The error model has no term for
   the source's intensity noise, and it is the only impairment that scales with
   the signal from outside the tile. B5's placement, off the package on one
   fiber, was decided for one laser. And an emitter an input row is the one
   arrangement that turns B5's watt-class laser into milliwatt parts, at the same
   total light: 10 mW a row stands 18.9 dB of loss where 1.6 W in one laser stands
   16.9. (The power and noise rows above were twice and 6 dB tighter, and these
   two ceilings 3 dB lower, until §4.3's correction.)

   *Two figures on that device question, 2026-10-03,* from the sources question
   1 now lists. Neither is a measurement of this tile. The Pittsburgh
   application says the wavelength channels on one bus waveguide "can be limited
   to k ≲ 56 based on crosstalk between nearest neighbors", of
   wavelength-multiplexed banks as a class and with no derivation given. And
   every source of many lines in those announcements carries 8 or 16. So 256
   lines on one bus is four and a half times the only limit anyone states and
   sixteen times the largest source. A ring bank at 256 inputs would then be
   several buses that reuse a few lines, with each column summing across buses —
   which changes who is whose neighbour in the error model's crosstalk term.

   *Narrowed by area, 2026-10-03.* The floorplan model
   ([`pta_floorplan.py`](pta_floorplan.py); question 1) gets half an answer by
   another route. A TFLT weight that is not resonant is millimeters long and a
   256 × 64 tile of them is not a chiplet, so the weights are resonant. That is
   the half of the ring-bank hypothesis that concerns the cell, and it brings
   the cell's sensitivities with it: to temperature and to the source's
   wavelength, by figures nobody holds. The other half — that inputs are told
   apart by wavelength, which is what decides the source's kind — area does not
   reach, and it stays open here.

---

## 9. Edits that followed B1 and B3

Made on 2026-09-21, when B1 and B3 settled:

- [GRX_GCPU.md](../GRX_GCPU.md) §2: a note that CXL 2.0 supersedes its
  TileLink recommendation for the board, and why. B1 moves the CPU–GPU link
  between packages, where that document's objection to CXL.cache — too heavy
  for on-package — no longer applies.
- [`chip_vs_board_strategy.md`](chip_vs_board_strategy.md): its author is still
  writing it, so B1's correction is handed over rather than made. UCIe joins
  dies within one package, and on rev A the CPU–GPU link is CXL over PCIe.

And on 2026-09-22, when B2 and B6 settled:

- This document: P1 now names UCIe-S on an organic substrate; §4.1 asks grx930
  for a DDR5 controller and §4.2 asks grxgpu for a GDDR6 one; the packaging
  risk records what B2 settled; and §8 gains the question of which FPGA
  platform hosts rev 0.
- Nothing else changes yet. The board's node (B6) reaches grx930 as the
  requirement in §4.1, not as an edit to that team's manufacturing plan.

And on 2026-09-22, when B4, B5 and B7 settled, the edits they had been holding:

- [`pta_gpu_integration.md`](pta_gpu_integration.md): a note on its scope, and
  one at the head of §2 — on the board the engine is a chiplet the whole GPU
  shares, though that section's reuse and weight-load arguments carry over
  unchanged. Its §7 staging note points G2 at the chiplet.
- [`pta_cpu_integration.md`](pta_cpu_integration.md): a note on its scope
  decision, one under the §3.1 register block on where that block lives on the
  board, and one on C4.
- [`pta_program_plan.md`](pta_program_plan.md): its status line points here, and
  F3, C4 and G2 carry notes.
- [`ai_motherboard_design_capabilities.md`](ai_motherboard_design_capabilities.md):
  a note at the head of its thermal section that the platform is Pockels, not
  thermo-optic, with what that changes and where the laser goes.

And later that day, when X2 had run:

- This document: X2's predictions in §3.3, and B4's addendum — the activation
  stage goes on the chiplet, on the traffic X2 measured.
- [`pta_chiplet_link.py`](pta_chiplet_link.py) is new: the die-to-die term, and
  F3's handoff from [`pta_program_plan.md`](pta_program_plan.md) §3.3.
- X1's joint budget in §4.3, with the risk it exposed, and grx930's
  `sim/pta_mnist.sh` gains the `joint` phase that produced it.
- [`board_icd.md`](board_icd.md) is new: P0's interface control document, whose
  §7 lists what it cannot source yet and §8 answers P0's gate.
- [`pta_chiplet_regmap.md`](pta_chiplet_regmap.md) is new: X4's map, which keeps
  the CPU document's offsets and adds what a link needs — identity, interrupts,
  64-bit counters, a seed the chiplet derives per GEMM, and a completion test
  that never spans two reads.
- [`pta_chiplet_calibration.md`](pta_chiplet_calibration.md) is new: X3's
  specification, and with it §4.3 gains the trim bits the weight DAC needs
  below the weight code's LSB.
