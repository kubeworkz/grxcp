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
Ten more were raised and settled later and are in §7: B8, B9, and on
2026-10-05 B10, the chiplet's working geometry, B11, its working shot rate,
1 GS/s, B12, its topology, a ring bank lit by a comb, and B13, how it signs a
weight, a balanced pair of photodiodes; and on 2026-10-06 B14, the interface
chip's budget, version 2, B15, how often it is calibrated, every six minutes,
B16, the source's shared row, 1%, and B17, the networks its budget is run on,
trained for eight epochs with noise of 10%. **B10 was settled at 256 × 64 and
revised the same day to 128 × 64.** Two things found that day moved it.
Published rings put the larger tile past a standard die. And B5's laser, sized
on a detector's full scale, was low by sixteen times at 256 rows and eight at
128 (B5, at its end), which left the larger tile little cheaper a MAC. Half of
that eight is the host's to give back, in the activation stage's shift.
And X1's budget was run on a second data set that day and does not hold (§8,
question 7): version 1 costs 1.2 points on Fashion-MNIST and on MNIST
inverted, where it costs a third of a point on MNIST, and the working tile's
laser is twice and four times MNIST's. What a tighter budget would buy was run
on 2026-10-06 (§4.3, at the end of its budget): one more bit of ADC with both
noise rows halved takes Fashion-MNIST from 1.16 points to 0.54, and all six
rows a notch tighter to 0.39. The first was adopted that day as version 2
(B14): an 8-bit ADC, receiver noise within a quarter of an 8-bit ADC's LSB,
30 photons per such LSB, and version 1's other three rows. Drift and the
source's rows were rerun at it the same day (§4.3, at the end of its budget).
Drift adds to version 2 what it added to version 1, so an hourly calibration
takes back more than version 2 bought on the two harder data sets. The
source's rows hold at version 2 on MNIST and Fashion-MNIST, and not on the
inverted set at either version. B15 and B16 chose from that: version 2 is
calibrated every six minutes, and its source's shared row is 1%. So held, it
loses 0.19, 0.60 and 1.09 points on the three sets. That was one draw of the
tile's noise: over ten it is 0.22, 0.68 and 1.40 (the sixth sweep, below).
Networks trained for the tile were run that day too (§4.3, at the end of its
budget). Training them for eight epochs, where a rule had stopped them after
two to seven, is worth two points on the inverted set, whose accuracies in
this plan are that much low. Noise of the tile's size on their sums while
they train is worth 0.3 to 0.5 of a point on Fashion-MNIST and nothing five
networks can tell on the other two. What a tile costs a network in points
lost is unchanged, and version 2 buys a trained network what it bought the
others. B14's first test is met as B14 worded it. B14 was kept that day all
the same, because version 2 is worth as much to those networks as to the old
ones, and they were made the reference for every later sweep (B17). Held at
version 2 they are right 97.64, 87.27 and 94.63% of the time and lose 0.11,
0.62 and 0.76 points to it. That was one draw too: over ten they are right
97.59, 87.30 and 94.65% of the time and lose 0.17, 0.59 and 0.74.
The first sweep on those networks, on 2026-10-07, ran them at B16's 1% and at
the 2% it replaced (§4.3, at the end of its budget). Held as the chip is held,
the 1% buys them nothing on any of the three sets, where it bought the old
networks a tenth of a point on the inverted set. B16 keeps the 1% all the
same, decided that day: nothing says it is dear, and it still buys that tenth
for a network that was not trained for the tile. If a comb cannot be had at
−130 dB/Hz, 2% is known to cost the reference networks nothing (§7).
The second sweep on them, the same day, put version 2's rows under a laser
(§4.3, at the end of its budget). Training with noise does not buy laser: over
its own budget a reference network loses what an old one does, and by the
plan's rule the three sets need 8, 8 and 16 times B5's where the old networks
need 8, 16 and 16. The laser a board has to place is its brightest workload's
and is unchanged, 1.4 to 14 W.
The third sweep on them, the same day, drifted both kinds of network for
three minutes to four hours (§4.3, at the end of its budget). In the mean,
B15's six minutes holds a tenth of a point on all three sets for the
reference networks, where it did not on the inverted set for the old ones,
and nothing longer does. Under heavy drift a reference network loses half to
two thirds of what an old one does. And a calibration leaves the reference
networks about a tenth of a point short of their budget on two sets, which
no row in this plan had looked for.
The fourth, the same day, ran the operating cycle: a tile aged an interval,
calibrated, and aged the interval again (§4.3, at the end of its budget). It
read a calibration as costing the reference networks 0.08 of a point on MNIST
by itself. The fifth, on 2026-10-08, took that apart, and it was not the
calibration. A calibrated run meets other draws of noise than the run as
budgeted, the tile as budgeted on those draws is as far short, and the trims
a calibration writes add 0.02 ± 0.02. On its own draws a six-minute cycle
ends 0.02, 0.02 and 0.11 of a point over the reference networks' budget. A
row of this plan moves by 0.02 to 0.07 of a point when nothing changes but
its draw, and its rows as budgeted are one draw each: over thirteen, the
reference networks lose 0.14, 0.54 and 0.48 of a point to the tile, which on
MNIST and Fashion-MNIST is what the networks trained before lose.
The sixth, the same day, ran the working point on ten draws (§4.3, at the end
of its budget). On the inverted set, which sizes B15's interval, the
reference networks' six-minute cycle ends 0.12 ± 0.03 of a point over the tile
as budgeted on the same draws, and a three-minute one 0.08 ± 0.02. By the rule
a calibration holds three minutes there and not six, and neither is far from
the tenth. On MNIST and Fashion-MNIST six minutes holds, at −0.01 and 0.07.
What training for the tile buys is in what holding the chip adds: six minutes
of drift and a source's rows cost the reference networks 0.03, 0.06 and 0.25
of a point, and the networks trained before 0.08, 0.20 and 0.68.
B15 was kept at six minutes on that (2026-10-08). And what one of its 240
interruptions a day costs, which nothing had priced, is counted (B15, at its
end): 18 µs by the twin's formula and 68 µs with everything the formula leaves
out, one part in 5.3 million of the tile's time. For it to matter an
interruption would have to last a third of a second. Asked again with that
counted, the choice was six minutes again (2026-10-08).
**And a comb line's level, which nothing built measured, can be read**
(§4.3, at the end of its budget; 2026-10-08). grx930's harness has a
probe for it: a row at full scale through a full-scale weight. Left alone,
lines 20% off cost the reference networks 0.35, 0.40 and 1.63 of a point on
the three sets and lines 40% off 2.01, 2.26 and 7.79, where the tile itself
costs them 0.13, 0.50 and 0.48. Read first with sixteen shots a row they cost
nothing that can be seen, and a read takes 2.2 µs. It is a reading and not
yet a correction: where a chip would apply it is not modelled.
*It was, the same day* (§4.3, at the end of its budget): asked what to do
next, the choice was to model the correction where a chip can apply it.
**Applied on the weights and written at the 8 bits the weight DAC has, the
reading takes lines 20% off to within 0.11 of a point of level lines**, on
all three sets. At the tile's 6 bits it leaves 0.07, 0.13 and 0.21, and at
this plan's 5% it buys nothing. Scaled to the dimmest line so that nothing is
raised, a correction costs 2.8 dB of light and on Fashion-MNIST more than the
comb did. So a comb 20% off, read and corrected, costs the reference networks
0.05, 0.06 and −0.05 of a point more than this plan's comb left alone. Whether
the plan's row moves on that is B16's to take, and is not taken.
*It was taken on 2026-10-09:* **B16's third row is now lines level to 20% as
they reach the tile, read at each calibration and corrected on the weights,
which are written at the weight DAC's 8 bits.** grx930's harness then ran the
chip as that holds it (§4.3, at the end of its budget). Held, the reference
networks lose 0.23, 0.59 and 0.74 of a point to the tile where with the row
as it was they lost 0.14, 0.58 and 0.78: 0.09 ± 0.03 more on MNIST, which is
clear, and nothing that can be told on the other two. Left alone, the comb
the row now allows would cost them 0.40, 0.41 and 1.47 more, so the read and
the correction are part of the row. A cycle with a source lit had been run
on one draw and not read over its own draws. Over five, as it is now run, a
six-minute cycle ends 0.07, 0.04 and 0.30 over the tile as budgeted, and it
is the source's rows and not the move that put the inverted set there.
**And the trim of every cycle in this plan was held
two bits finer than the 8-bit DAC the calibration note asks for.** On that
DAC a cycle as it is now run ends 0.05, 0.09 and 0.33 over. Which DAC the
chiplet is held to is B16's to take, and is not taken.
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

*And sized on the light a column is actually sent, 2026-10-05, it is sixteen
times that* ([`pta_laser.py`](pta_laser.py); grx930's design note, §5, "What
does a laser of a given size cost?"). Every figure above takes the light at a
detector when its converter reads full scale to be the light the column was
sent. It is not. A column is sent every line at full power whatever the inputs
are, because a source does not know them. An input passes its activation's
share of a line, and a ring sends what is left to one photodiode of a pair or
the other. The converter's full scale is set where the sums fall, and on D3 at
256 × 64 that is an eighth of what 256 lines send on the first layer and a
thirty-second on the second.

So a laser of the size above leaves the receiver 4 and 16 LSB of noise, where
§4.3 holds it to half of one. grx930's harness ran it:

| Laser | 256 × 64, points lost | 128 × 64 |
|---|---|---|
| Version 1 as budgeted | 0.20 ± 0.05 | 0.34 ± 0.07 |
| This decision's laser | 39.19 ± 1.87 | 13.56 ± 1.14 |
| 4 times it | 2.44 ± 0.24 | 0.75 ± 0.07 |
| 8 times | 0.60 ± 0.07 | 0.35 ± 0.06 |
| 16 times | 0.28 ± 0.06 | 0.22 ± 0.04 |
| 32 times | 0.23 ± 0.06 | 0.20 ± 0.06 |

The working tile is within a tenth of a point of version 1 at 16 times this
decision's laser: **1.4–14 W** at the measured receiver, behind 10–20 dB. At
128 × 64 it is 8 times, 0.7–7 W. If a receiver that averages over a shot (B12)
keeps the measured one's noise density it has 0.60 of its noise, and the
working tile's laser is 0.84–8.4 W.

The method was right where it was made. On the core's 8 × 8 tile, which is
where C1's sweep ran, eight lines are a full scale. Sums do not grow as a
tile's rows do, and the light does. Every laser figure in this plan dated
before this one is the method above and low by the same factor: B10's and
B11's rows, B12's source, the two tables of §8's question 1, and the ICD's
link 4.

A laser of 1.4 to 14 W is not the item this decision placed. Its placement
stands, off the package and on fiber. What it is on the board, thermally and
for eye safety, has to be asked again, and so does what it is on the photonic
die: 22 to 220 mW a line at four buses, where B12 has 1.4 to 14.

It is one network, on MNIST, whose images light a tenth of a tile's rows at
the mean. A workload that lights more of its inputs fills more of the light,
and no workload makes the fill one.

*B10 was revised to 128 × 64 on this, the same day.* The working tile's laser
is then the 0.7–7 W above: eight times this decision's.

*And half of that is the host's to give back, 2026-10-05*
([`pta_laser.py`](pta_laser.py), its last section; grx930's design note, §5,
"How a network is put on the tile"). The fill is the network's as much as the
tile's: a layer's sums fall where its operands do, and the host sets two of
them. grx930's harness ran both on the working tile.

| How the network is put on the tile | What clips | That alone | At 2 times this decision's laser | 4 times | 8 times |
|---|---|---|---|---|---|
| As the harness's rule has it | One firing unit in ten thousand | 0.00 ± 0.01 | 2.74 ± 0.19 | 0.75 ± 0.07 | 0.35 ± 0.06 |
| **The hidden layer's rescale one bit lower** | 0.65% of the units that fire | 0.02 ± 0.01 | 0.92 ± 0.06 | **0.35 ± 0.06** | 0.22 ± 0.06 |
| Two bits lower | 13% of them | 0.26 ± 0.05 | 0.99 ± 0.11 | 0.60 ± 0.08 | 0.46 ± 0.06 |
| The first layer's weights written twice as large | 0.38% of the weights | 0.14 ± 0.08 | 2.90 ± 0.21 | 0.83 ± 0.12 | 0.50 ± 0.16 |

Version 1 as budgeted loses 0.34 ± 0.07 on this tile. With the hidden layer's
rescale one bit under the rule it is within a tenth of a point of that at 4
times this decision's laser, where it took 8: **0.35–3.5 W**. The second
layer's operands are twice as large, so its sums are, and its share of the
light with them. A second bit buys nothing, and a gain on the first layer's
weights does not pay.

That rescale is the activation stage's shift, which X6 made a field of a
command. So it costs no hardware, and it is a choice a program makes with each
layer. The rule it undercuts was a converter's: it wasted none of an operand's
range, and under a laser the range is not what is short.

It was run at 128 × 64 and nowhere else, on networks trained with no clip in
the loop. And it gives back the second layer's share only: at twice this
decision's laser the first layer's noise is an LSB, and nothing done to the
hidden operands reaches it.

*And all of that is MNIST's, 2026-10-05* ([`pta_workload.py`](pta_workload.py);
grx930's design note, §5, "Does the budget hold on another workload?"). Two
more data sets went through grx930's harness on the working tile:
Fashion-MNIST, and MNIST with every pixel inverted (§8, question 7). Points
lost, five networks each, with the receiver's noise as a laser of that many
times this decision's fixes it:

| | MNIST | Fashion-MNIST | MNIST, inverted |
|---|---|---|---|
| Version 1 as budgeted | 0.34 ± 0.07 | 1.16 ± 0.14 | 1.23 ± 0.13 |
| 4 times this decision's laser | 0.75 ± 0.07 | 4.16 ± 0.89 | 5.35 ± 0.90 |
| 8 times | **0.35 ± 0.06** | 1.93 ± 0.29 | 1.99 ± 0.27 |
| 16 times | 0.22 ± 0.04 | 1.26 ± 0.14 | **1.27 ± 0.18** |
| 32 times | 0.20 ± 0.06 | **1.04 ± 0.13** | 1.15 ± 0.16 |
| The hidden rescale a bit down: 4 times | **0.35 ± 0.06** | 1.93 ± 0.34 | 2.29 ± 0.29 |
| 8 times | 0.22 ± 0.06 | **1.19 ± 0.13** | 1.37 ± 0.16 |
| 16 times | not run | 0.96 ± 0.07 | **1.17 ± 0.12** |
| Within a tenth of a point of its own version 1, the rescale a bit down | 4 times: 0.35–3.5 W | 8 times: 0.7–7 W | 16 times: 1.4–14 W |
| A MAC at that, every cell in use | 97–922 fJ | 140–1,351 fJ | 226–2,209 fJ |

The rescale still gives back half on both. What it halves is twice MNIST's on
Fashion-MNIST and four times on the inverted set. Fashion-MNIST at the rule's
rescale is 0.10 over at 16 times, which is the line and inside its scatter.

So the working tile's laser is 0.35–3.5 W for MNIST and up to 1.4–14 W for a
workload this plan has run. The lower figure is not withdrawn. It is one
workload's, and so is every figure built on it: B10's and B11's rows, a MAC's
energy, and the ICD's link 4.

It did not move the way this decision expected. "A workload that lights more
of its inputs fills more of the light" held for the first layer of four
networks in ten. The second layer's inputs are the hidden units, and both sets
light those about half as much as MNIST does. On Fashion-MNIST the fill hardly
moved, and the same noise in a converter's LSB costs that network more points.
On the inverted set the second layer's fill fell on all five networks, by two
to four times.

*And a tighter budget takes more of it, 2026-10-06* (§4.3, at the end of its
budget; [`pta_tighten.py`](pta_tighten.py)). Version 1 with an 8-bit ADC and
both noise rows halved, or with all six rows a notch tighter, is within a
tenth of a point of its own budget at 8, 16 and 16 times this decision's
laser on the three data sets, the shift a bit down: 0.7–7 W, 1.4–14 W and
1.4–14 W. That is twice version 1's on MNIST and on Fashion-MNIST, and no
more on the inverted set.

*B14 adopted the first of those budgets the same day, as version 2 (§7).* So
the working tile's laser is 0.7–7 W for MNIST and 1.4–14 W for either of the
other two data sets, 11 to 110 mW and 22 to 220 mW a line of the comb, with
the receiver's noise at a thirty-second and a sixty-fourth of one line's
light ([`pta_working_point.py`](pta_working_point.py), §4).

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
  grew from 156 checks to 201 (and to 237 on 2026-10-05, when it took a
  128 × 64 build for B10's revision). On an 8 × 8 tile through three layers and a
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
  accuracy figure for a held network. *There is one since* (§4.3, the chiplet's
  own tile): 0.20 ± 0.05 points at 256 × 64, by a chain of two exact
  comparisons and not by a run of its own.

What it adds to the interface chip's requirements (§4.3): an activation unit a
column, at the shot rate, and room to hold a layer's operands. The twin gives
the stage no time. One that took a beat an output would spend 6.4 µs on D3's
hidden layer at a batch of 64, against 3.74 µs for both of the network's GEMMs.

What it leaves. grxcp's runtime does not use it: grxBLAS runs one GEMM and has
no notion of a network, so the stage is reachable through the twin's own call
and nothing above it. grxgpu had not been asked to carry it when this was
written: the host-path proposal reserved two flag bits and the stage needs a
third and three fields. *It has been asked since* (§4.2). And its time is a
requirement and not a measurement.

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
  seed alone, as a c930's would be, differs in 285 elements of 600. (Since
  2026-10-05 it is run on a 128 × 64 tile as well, for B10's revision: the same
  eight cases, with their ADC shifts and the K of those that walk two K tiles
  derived for 128 rows, and 4,164 results again. There the reference on the
  seed alone differs in 289 of 600.)
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
  budget has been measured and accepted. That is one network, D3, at 0.24 points
  with hourly calibration and 0.21 at six minutes on the 256 × 64 tile this
  model prices. *It read 0.81 and 0.37 until 2026-10-04*, which are the 8 × 8
  tile's and were the only figures there were (§4.3). *And it is MNIST's: on
  the two other data sets run on 2026-10-05 the same budget costs over a
  point, and an hour's drift more again (§8, question 7).*
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
| Chiplet, hourly calibration | 3.74 µs and two commands | 97.21% | 25.6 to 64.7 µs with a command at the two measured costs |
| Chiplet, calibrated every six minutes | the same | 97.24% | |

The chiplet's two accuracies are a 256 × 64 tile's. This table gave 96.64 and
97.08 until 2026-10-04: the c930 core's 8 × 8 tile, on which drift costs more
(§4.3).

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
operands are held or returned. ~~grxgpu has still not been asked to carry it.~~
*Asked, the same day* (§4.2), and open until grxgpu answers.

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
  - *Amended again, 2026-10-04* ([kubeworkz/grxgpu#4](https://github.com/kubeworkz/grxgpu/pull/4)),
    open until grxgpu answers. It defines the two reserved bits as "from held"
    and "hold", adds "activate", and asks for a second version of the command's
    descriptor, 16 bytes longer, carrying the shift, the width, the bias
    vector's address and the count of what the stage clamped. The chiplet
    computes the stage; the GPU's command processor carries the settings.
    Three things came out of writing it.
    - **The first amendment had the stage on the wrong command.** It reserved a
      bit for "activations are the previous command's results, through the
      activation stage". The stage belongs on the command that produces the
      operands, whose bias and shift it uses, and X4's §8 has it there. The bit
      now means only that the activations are the held ones.
    - **"Activate" alone removes the launch.** A command that returns operands
      where it would return sums writes a buffer the next command can read as
      it stands. Holding them saves the link crossing on top, a tenth of a
      microsecond on D3.
    - **A chain needs the tile kept for it once the GPU has more than one
      command queue.** Held operands are for the chiplet's next command, so
      another queue's GEMM between two layers gets the second layer refused.
      That fails safe, by X4's rule, and it fails. With one queue, which is the
      GPU's default, it cannot happen. It is recorded as a question for grxgpu
      and as something S3's model does not price.
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

*Version 2, since 2026-10-06 (B14, §7).* The interface chip is held to this
table's version 1 with three rows tighter: an **8-bit ADC**, receiver noise of
**0.25 LSB of an 8-bit ADC**, and **30 photons per such LSB**. The activation
DAC, the weight resolution, the programming error and the crosstalk are
version 1's. Why those three is at the end of this section's budget. *And two
rows more, the same day:* it is recalibrated **every six minutes** at TFLT's
fit (B15), and its source's noise is within **1% rms a shot for the lines
together** (B16), with 5% for a line on its own and the lines level to 5% as
before. *The last of those was moved on 2026-10-09 (B16):* the lines are
level to **20%** as they reach the tile, read at each calibration and
corrected on the weights a row at a time, which are written at the weight
DAC's 8 bits.

v1 costs 0.81 points on the D3 network at hourly calibration, and 0.37 if the
schedulers can recalibrate every six minutes — which is what C3 had to price.
It is still one small network (§8). This paragraph went on to say that the
shape of the result — error compounds, every allowance tightens fourfold —
would travel further than its numbers. It was the numbers that were wrong, and
the shape went with them.

*On the chiplet's own tile, 2026-10-04.* Every accuracy in this section was
measured on the c930 core's 8 × 8 tile, in GEMMs that core accepts, and this
plan's chiplet runs to 256 × 128 and takes a layer as one command. Nothing had
asked whether the figures carry. grx930's harness now takes the tile as an
option ([kubeworkz/grx930#39](https://github.com/kubeworkz/grx930/pull/39)), and its
design note has the budget on four more of them. Loss in points on D3, five
networks, a layer a GEMM off the core's tile:

| | 8 × 8, as above | 64 × 8 | 128 × 64 | 256 × 64 | 256 × 128 |
|---|---|---|---|---|---|
| v1 | 0.26 ± 0.06 | 0.33 ± 0.11 | 0.34 ± 0.07 | 0.20 ± 0.05 | 0.23 ± 0.05 |
| v0, at its 6-bit ADC | 1.49 ± 0.06 | 2.05 ± 0.40 | 2.17 ± 0.12 | 1.86 ± 0.17 | 1.78 ± 0.13 |
| v1, six minutes of TFLT's drift | 0.37 ± 0.04 | 0.32 ± 0.12 | 0.33 ± 0.09 | 0.21 ± 0.04 | 0.23 ± 0.06 |
| v1, an hour of it | 0.81 ± 0.21 | 0.56 ± 0.09 | 0.51 ± 0.07 | 0.24 ± 0.06 | 0.43 ± 0.15 |
| v1, four hours of it | 5.18 ± 2.31 | 1.17 ± 0.07 | 0.74 ± 0.07 | 0.77 ± 0.16 | 1.18 ± 0.30 |
| v1, an hour of TFLN's | 22.62 ± 4.79 | 4.24 ± 0.28 | 2.81 ± 0.22 | 1.92 ± 0.25 | 3.53 ± 0.99 |
| v1, an hour of TFLT's, then calibrated | 0.22 ± 0.05 | 0.32 ± 0.08 | 0.21 ± 0.05 | 0.20 ± 0.05 | 0.25 ± 0.07 |

Four things for this plan.

- **v1 holds on every tile.** The interface chip's requirement stands as it is
  written, on every candidate geometry.
- **v0 does not carry.** It costs a quarter to a half more off the 8 × 8 tile,
  five standard errors at 128 × 64. The menu above prices each of v0's rows on
  the 8 × 8 tile, and on a larger one the light costs more and the receiver's
  noise less.
- **A noise row is the same noise from tile to tile only within a factor of
  two.** The rows are in LSB of the tile's own ADC, and the ADC's shift is a
  whole number of bits set by the tile's sums. The error a row leaves on a layer
  goes as `√conversions × 2^shift`, and that predicts the receiver-noise row on
  every tile to a few percent. Of the candidates, 128 inputs is the worst for
  receiver noise and 256 the best: 128 needs the shift 256 needs and takes
  nearly twice the conversions. That bears on §8's first question, the geometry,
  and on B5's laser, which is sized from the receiver's noise in LSB.
- **Drift costs far less on a large tile: 2.8 to 3.2 times less error at every
  age.** An hour of TFLT's drift costs 0.55 points beyond v1 on the 8 × 8 tile
  and 0.04 at 256 × 64. So the recalibration row above, and the interval X3 and
  C3 worked to, were sized on the worst tile there is. **They are not relaxed
  here**, for one reason. The model draws every cell's drift independently, and
  the 8 × 8 tile is hurt because all 78,400 of a layer's weights pass through
  the same 64 cells. If a real tile's neighbouring cells drift together, as
  under a temperature they would, a larger tile gains less than this, and
  nothing measured says how much less.

The calibration itself works at every size: an hour of drift and then C3's
correction returns every tile to its own v1. That needed a fix in grx930's
harness, which had been returning without a word on a tile of more than 64
cells. On the 8 × 8 tile that was every cell, so nothing above was affected.

*And through the twin.* The same runs, with the twin where the model was, print
the harness's lines byte for byte on every tile (`src/backends/pta_chiplet/`,
its README). On a tile that is not the core's the harness gives a layer as one
GEMM, so that is the twin being given what a chiplet would be given. With X6's
gate, which holds a network kept on the chiplet to the same commands with every
intermediate brought out, **the 256 × 64 column is the accuracy of D3 held on
the chiplet.** X6 left that as not run.

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

*On a second data set, 2026-10-05: neither the price nor the menu holds.*
([`pta_workload.py`](pta_workload.py); grx930's design note, §5, "Does the
budget hold on another workload?") Version 1 costs 1.06 ± 0.26 points on
Fashion-MNIST and 1.10 ± 0.22 on MNIST inverted on the core's tile, where it
costs 0.26 on MNIST. On the working tile it is 1.16 ± 0.14 and 1.23 ± 0.13 for
0.34. The table above again, each of version 1's rows relaxed alone to version
0's, on the core's tile:

| Row | MNIST | Fashion-MNIST | MNIST, inverted |
|---|---|---|---|
| Activation DAC, 6 → 5 bits | 0.10 | 0.30 | 0.59 |
| ADC, 7 → 6 bits | 0.12 | 0.59 | 0.79 |
| Receiver noise, 0.5 → 1 LSB of an 8-bit ADC | 0.19 | **0.86** | 0.65 |
| Light, 15 → 3 photons per such LSB | **0.34** | 0.69 | 1.35 |
| Weight programming error, 1 → 4 LSB | 0.33 | 0.65 | **1.87** |
| Crosstalk, 2% → 10% | 0.16 | 0.78 | 1.05 |
| Summed | 1.24 | 3.87 | 6.30 |
| Version 0's five rows at once, over what they sum to | 1.08 | 0.87 | 1.19 |

Every row costs two to seven times what it cost. And the advice above does
not carry. It was to relax the receiver's noise or the ADC's bit, if any row,
and to leave the two DACs alone. On Fashion-MNIST the receiver's noise is the
dearest row there is and the activation DAC's bit the cheapest. On the
inverted set the weight DAC's row is the dearest, as the advice had it, and
the ADC's bit is 0.79.

So none of this section's prices is a budget's. Each is a budget's on a
workload. Version 1 is still the tightest set of rows this plan has and still
what the interface chip is held to. What it buys is a quarter of a point on
MNIST and a point on either of the other two.

*What tightening it would buy, 2026-10-06.* ([`pta_tighten.py`](pta_tighten.py);
grx930's design note, §5, "What would tightening v1 buy?") The table above
prices each row loosened. grx930's harness has now run version 1 with each row
made better, alone and together, on the working tile and all three data sets.
What a row buys of version 1's loss when it is taken away altogether, which is
the most that tightening it could buy:

| Row, gone | MNIST | Fashion-MNIST | MNIST, inverted |
|---|---|---|---|
| Version 1 loses | 0.34 ± 0.07 | 1.16 ± 0.14 | 1.23 ± 0.13 |
| Activation DAC | 0.06 ± 0.03 | −0.01 ± 0.03 | 0.06 ± 0.05 |
| ADC | 0.08 ± 0.08 | **0.26 ± 0.09** | 0.15 ± 0.06 |
| Receiver noise | **0.17 ± 0.03** | **0.31 ± 0.14** | 0.20 ± 0.07 |
| Shot noise | **0.20 ± 0.03** | **0.26 ± 0.11** | **0.52 ± 0.14** |
| Weight programming error | 0.05 ± 0.03 | −0.04 ± 0.07 | 0.12 ± 0.06 |
| Crosstalk | −0.01 ± 0.03 | −0.08 ± 0.11 | 0.13 ± 0.09 |

On Fashion-MNIST the point is in three rows. The other three are worth
nothing there: the activation DAC's sixth bit, programming error at 1 LSB and
crosstalk at 2%. By the table before this one they are dear to loosen, and by
this one they are free to hold. The inverted set spreads its point over all
six, with the shot noise half a point by itself. On MNIST it is the two noise
rows.

So two sets of rows tighter than version 1 were run whole. A notch is a bit
more in a converter and half the noise or the error: receiver noise of a
quarter of an 8-bit ADC's LSB, and 30 photons per such LSB.

| | Version 1 | Version 1 with an 8-bit ADC and both noise rows halved | All six rows a notch tighter |
|---|---|---|---|
| Points lost: MNIST | 0.34 ± 0.07 | 0.15 ± 0.02 | 0.06 ± 0.02 |
| Fashion-MNIST | 1.16 ± 0.14 | 0.54 ± 0.22 | 0.39 ± 0.15 |
| MNIST, inverted | 1.23 ± 0.13 | 0.62 ± 0.07 | 0.50 ± 0.08 |
| ADC | 7 bits | 8 bits | 8 bits |
| Published parts that reach it at 1 GS/s | 70 | 47 | 47 |
| Its 64 converters, best to fifth-best published | 0.07–0.20 W | 0.27–0.34 W | 0.27–0.34 W |
| The interface chip | 0.44–4.04 W | 0.64–4.18 W | 0.64–4.18 W, and what is not priced below |
| A weight ring's line, at least, to settle in a shot | 1.99 GHz | 2.21 GHz | 2.21 GHz |
| Lines a bus, and buses | 77, 2 | 69, 2 | 69, 2 |
| A ring's Q, between settling in a shot and keeping two buses | 80,200 to 97,400: 21% of room | 80,200 to 87,700: 9% | 80,200 to 87,700: 9% |
| Its swing | 2.27 to 2.76 V | 2.53 to 2.76 V | 2.53 to 2.76 V |
| Laser, the activation stage's shift a bit down: MNIST | 4 times B5's, 0.35–3.5 W | 8 times, 0.7–7 W | 8 times, 0.7–7 W |
| Fashion-MNIST | 8 times, 0.7–7 W | 16 times, 1.4–14 W | 16 times, 1.4–14 W |
| MNIST, inverted | 16 times, 1.4–14 W | 16 times, 1.4–14 W | 16 times, 1.4–14 W |
| A MAC, every cell in use: MNIST | 97–922 fJ | 164–1,368 fJ | 164–1,368 fJ |
| Fashion-MNIST | 140–1,351 fJ | 250–2,226 fJ | 250–2,226 fJ |
| MNIST, inverted | 226–2,209 fJ | 250–2,226 fJ | 250–2,226 fJ |
| Activation DAC | 6 bits | 6 bits | 7 bits. **Not priced** |
| Weight programming error | 1 LSB | 1 LSB | 0.5 LSB. **Not priced** |
| Crosstalk | 2% | 2% | 1.2%, which B12's grid already is |

Five things.

- **A notch on all six puts Fashion-MNIST where version 1 puts MNIST**, 0.39
  for 0.34. Two notches halve it again, to 0.18, and ask a ninth bit of ADC,
  at which a bus holds 63 lines and the tile needs a third.
- **Three rows are four fifths of it.** One more bit of ADC and half of each
  noise row buy 82% and 85% of what all six buy on the two harder sets, and
  68% on MNIST. A notch on the other three, by itself, buys 0.04 to 0.11.
- **The bit is a seventh to a fifth of a watt, and a ring that is harder to
  hit.** A level read to a bit more has to have settled further in the same
  shot, so B11's bound on a ring moves: the Q ceiling falls by a tenth. Two
  buses of 64 lines want a Q of at least 80,200. So the room a ring's Q has
  between the two goes from 21% to 9%. A ninth bit closes it.
- **Half the noise is twice the laser, except where the laser was already
  large.** And at one laser the rows are worth more than more laser. At 8
  times B5's, Fashion-MNIST loses 1.19 at version 1's rows and 0.74 with the
  bit and 30 photons, and 64 times at version 1's rows gets it only to 0.84.
- **The crosstalk's notch costs nothing new.** B12 packs a bus at 4.61
  linewidths, where a neighbour is seen at 1.2%, and on its 11 GHz grid a
  ring at the rate's Q ceiling sees 0.8%, or 1.0% if it settles to 8 bits. A
  ring of lower Q on that grid is a different matter, and was one at 2%.

Neither set is adopted. Version 1 is what the interface chip is held to, and
this section's table of requirements has not changed. What is open is whether
to hold it to one of these, and that is a choice between a point on a harder
data set and a fifth of a watt at most, a ring whose Q has half the room, and
a laser of twice the size.

*The first was adopted the same day: B14, §7.* The middle column above is
version 2, and it is what the interface chip is held to.

*Drift and the source's rows at version 2, 2026-10-06.*
([`pta_version2.py`](pta_version2.py); grx930's design note, §5, "Drift and a
source's rows at grxcp's version 2") B14 listed what had been measured at
version 1 and not run again. grx930's harness has run the first two on its
list at both versions, on the working tile on two buses, and on all three data
sets, where the source's rows had been run on MNIST alone. What a row adds to
its own version as budgeted, in points, network by network:

| | MNIST, version 1 | Version 2 | Fashion-MNIST, version 1 | Version 2 | MNIST inverted, version 1 | Version 2 |
|---|---|---|---|---|---|---|
| As budgeted, points lost | 0.34 ± 0.07 | 0.15 ± 0.02 | 1.16 ± 0.14 | 0.54 ± 0.22 | 1.23 ± 0.13 | 0.62 ± 0.07 |
| Six minutes of TFLT's drift adds | −0.01 ± 0.02 | −0.06 ± 0.04 | 0.08 ± 0.10 | 0.08 ± 0.06 | 0.20 ± 0.26 | 0.28 ± 0.27 |
| An hour | 0.17 ± 0.05 | 0.12 ± 0.05 | 0.69 ± 0.33 | 0.67 ± 0.31 | 1.95 ± 1.29 | 1.99 ± 1.26 |
| Four hours | 0.40 ± 0.08 | 0.37 ± 0.08 | 2.34 ± 0.43 | 2.36 ± 0.57 | 16.54 ± 3.01 | 16.07 ± 3.09 |
| An hour of TFLN's | 2.47 ± 0.20 | 2.35 ± 0.17 | 7.22 ± 1.48 | 7.29 ± 1.50 | 37.35 ± 5.63 | 37.38 ± 5.74 |
| An hour of TFLT's, then calibrated | −0.13 ± 0.03 | −0.06 ± 0.03 | 0.03 ± 0.15 | 0.02 ± 0.11 | −0.09 ± 0.08 | −0.01 ± 0.05 |
| The source: the lines together at 1% | 0.00 ± 0.02 | 0.01 ± 0.02 | 0.06 ± 0.06 | 0.04 ± 0.04 | 0.10 ± 0.06 | 0.13 ± 0.08 |
| At 2%, which is the row | −0.06 ± 0.03 | 0.02 ± 0.02 | 0.05 ± 0.02 | 0.03 ± 0.04 | 0.16 ± 0.06 | 0.24 ± 0.05 |
| At 5% | 0.04 ± 0.04 | 0.02 ± 0.05 | 0.12 ± 0.04 | 0.01 ± 0.08 | 0.82 ± 0.05 | 0.90 ± 0.13 |
| A line on its own at 5%, the row | 0.09 ± 0.07 | 0.05 ± 0.05 | 0.11 ± 0.14 | 0.14 ± 0.12 | 0.11 ± 0.13 | 0.25 ± 0.07 |
| The lines' level at 5%, the row | 0.05 ± 0.06 | 0.03 ± 0.03 | 0.09 ± 0.08 | 0.01 ± 0.07 | 0.05 ± 0.11 | 0.17 ± 0.06 |
| The three rows together | 0.03 ± 0.11 | 0.06 ± 0.03 | 0.06 ± 0.11 | 0.09 ± 0.13 | 0.29 ± 0.17 | 0.37 ± 0.07 |

Four things.

- **Drift adds to version 2 what it added to version 1.** At every age and on
  every data set the two are within eight hundredths of a point where drift
  adds little, and within half a standard error where it adds much. A better
  converter and a quieter receiver neither hide drift nor expose it. And
  calibration returns version 2 to its budget, to within 0.06 of a point.
- **So an hourly calibration takes back more than version 2 bought on the
  harder sets.** Version 2 bought 0.62 and 0.61 of a point on Fashion-MNIST
  and the inverted set, and an hour's drift adds 0.67 and 1.99. At six minutes
  it adds 0.08 and 0.28. The "about hourly" of this section's table was
  version 1's on MNIST, and at version 2 an hour is 80% of the budget there
  too.
- **The source's rows hold at version 2 on MNIST and on Fashion-MNIST.** The
  three add 0.06 and 0.09, under the tenth of a point they were sized to.
- **They do not hold on the inverted set, at either version.** There the three
  add 0.29 and 0.37, and the dear row is the one the lines share: at 2% it
  adds 0.16 and 0.24, and at 5% over four fifths of a point, where on the
  other two sets even 10% adds about a fifth. For it to add about a tenth
  there it has to be 1%, which is −130 dB/Hz over the shot rate where 2% is
  −124. The likely reason is the one behind every figure on that set: its sums
  are small differences of a great deal of light, the lines together are
  drawn once a shot and scale that shot's sum, and a layer is seven shots.
  That was not measured shot by shot.

Two things this leaves open, and neither is decided here. How often version 2
is calibrated: the table of requirements still says about hourly, and on a
harder data set that gives back what version 2 was adopted for. And whether
the source's shared row stays at 2%: it is right for MNIST and Fashion-MNIST,
and a workload that lights most of its rows wants 1%, which is still 14 dB
easier than this plan first assumed (§8, question 8).

*Both were decided the same day: B15, every six minutes, and B16, 1% (§7).*
grx930's harness then ran version 2 with both. The source's three rows at 1%,
5% and 5%, and those at the end of six minutes of drift, which is version 2
with everything it is held to:

| | MNIST, version 1 | Version 2 | Fashion-MNIST, version 1 | Version 2 | MNIST inverted, version 1 | Version 2 |
|---|---|---|---|---|---|---|
| The three rows at 2%, 5% and 5% add | 0.03 ± 0.11 | 0.06 ± 0.03 | 0.06 ± 0.11 | 0.09 ± 0.13 | 0.29 ± 0.17 | 0.37 ± 0.07 |
| At 1%, 5% and 5% | 0.04 ± 0.10 | 0.04 ± 0.04 | 0.14 ± 0.11 | 0.04 ± 0.11 | 0.15 ± 0.12 | 0.28 ± 0.06 |
| And after six minutes of drift | 0.06 ± 0.06 | 0.04 ± 0.02 | 0.21 ± 0.13 | 0.06 ± 0.15 | 0.37 ± 0.40 | 0.48 ± 0.29 |
| Which is, in points lost | 0.40 ± 0.03 | **0.19 ± 0.03** | 1.37 ± 0.23 | **0.60 ± 0.23** | 1.60 ± 0.50 | **1.09 ± 0.36** |

On MNIST and on Fashion-MNIST, version 2 with a source and an interval's drift
is within a tenth of a point of its budget. On the inverted set it is half a
point over, and the 1% bought 0.09 of that: the two rows a line carries are
0.25 and 0.17 by themselves there, and are now most of what a source costs.

It is the same model and the same networks as everything above. The drift is
a Mach-Zehnder's fit with every cell drifting on its own, the source's term is
first order on the host's side of the line, and the inverted set's networks
were not trained well.
The activation DAC's seventh bit and half the programming error are what all
six rows ask beyond the three, they buy the last fifth, and no model here
prices either.

It is still a model, on networks that were not trained for the tile (§8,
question 7). And the light's row is the budget's, photons per LSB, held where
a row puts it while the laser multiplies: a real laser moves both, and a
balanced pair's shot noise follows all the light it is lit with, which
[`pta_laser.py`](pta_laser.py) prices and the harness does not have.

*Networks trained for the tile, 2026-10-06.* ([`pta_trained.py`](pta_trained.py);
grx930's design note, §5, "A network trained for the tile".) Every network
above was trained on its host and met the tile afterwards. grx930's trainer
now takes two options and its harness a mode: each seed's 6-bit network
again, from the same 8-bit one, for eight epochs, with Gaussian noise of 0, 5,
10 and 20% of a layer's rms on every sum the layer forms while it trains. 10%
is the tile's size: the probe puts a v1 tile's error at about a tenth of a
sum's rms. The network with no noise is a control, and it turned out to be
needed. The trainer has always stopped at the first epoch whose held-out
accuracy fails to rise, which was after two to seven of them.

Accuracy on the working tile, percent, five networks a row. Held is with the
source's three rows at B16's sizes, at the end of B15's six minutes:

| | On its host | v1 | Version 2 | v1, held | Version 2, held |
|---|---|---|---|---|---|
| MNIST, trained as before | 97.45 ± 0.10 | 97.11 ± 0.13 | 97.30 ± 0.11 | 97.05 ± 0.10 | 97.26 ± 0.10 |
| Eight epochs, no noise | 97.74 ± 0.10 | 97.52 ± 0.10 | 97.66 ± 0.10 | 97.46 ± 0.10 | 97.57 ± 0.09 |
| Eight epochs, noise of 10% | 97.75 ± 0.06 | 97.49 ± 0.06 | 97.68 ± 0.04 | 97.44 ± 0.04 | 97.64 ± 0.07 |
| Fashion-MNIST, trained as before | 87.48 ± 0.19 | 86.32 ± 0.18 | 86.94 ± 0.20 | 86.11 ± 0.14 | 86.88 ± 0.09 |
| Eight epochs, no noise | 87.66 ± 0.16 | 86.30 ± 0.22 | 87.13 ± 0.13 | 86.28 ± 0.18 | 87.04 ± 0.16 |
| Eight epochs, noise of 10% | 87.89 ± 0.16 | 86.77 ± 0.26 | 87.41 ± 0.13 | 86.69 ± 0.25 | 87.27 ± 0.13 |
| MNIST inverted, trained as before | 93.36 ± 0.54 | 92.13 ± 0.56 | 92.74 ± 0.55 | 91.76 ± 0.65 | 92.27 ± 0.63 |
| Eight epochs, no noise | 95.55 ± 0.36 | 94.17 ± 0.59 | 94.88 ± 0.46 | 93.63 ± 0.70 | 94.51 ± 0.53 |
| Eight epochs, noise of 10% | 95.38 ± 0.33 | 94.42 ± 0.37 | 94.86 ± 0.36 | 94.18 ± 0.54 | 94.63 ± 0.53 |

A row's error is the scatter of its five networks, and most of that scatter
is the networks' own. So each comparison is taken seed by seed, and its error
is the error of the five differences:

| | MNIST | Fashion-MNIST | MNIST, inverted |
|---|---|---|---|
| Eight epochs with no noise, over trained as before: on its host | +0.29 ± 0.14 | +0.18 ± 0.28 | **+2.19 ± 0.49** |
| On a v1 tile | +0.41 ± 0.15 | −0.02 ± 0.22 | **+2.04 ± 0.68** |
| Noise of 10%, over none: on a v1 tile | −0.03 ± 0.07 | +0.46 ± 0.18 | +0.24 ± 0.28 |
| At version 2 | +0.02 ± 0.06 | **+0.29 ± 0.07** | −0.02 ± 0.18 |
| Version 2 over v1, the same network: trained as before | **+0.19 ± 0.06** | **+0.62 ± 0.10** | **+0.61 ± 0.06** |
| Trained with noise of 10% | **+0.19 ± 0.04** | **+0.65 ± 0.15** | **+0.44 ± 0.03** |
| The 10% network at v1, over the old one at version 2 | +0.19 ± 0.10 | −0.17 ± 0.24 | **+1.67 ± 0.46** |
| The 10% network at version 2, over the old one at v1 | **+0.57 ± 0.12** | **+1.09 ± 0.24** | **+2.73 ± 0.45** |

With five networks a difference has to be 2.8 of its own errors to be outside
chance at one in twenty. Those in bold are. The model prints well over a
hundred such differences, so a few would reach that with nothing behind them,
and none of what follows rests on one cell.

- **The epochs were the larger thing, and nothing had predicted them.** The
  inverted set's networks had stopped after three to seven epochs. At eight
  they are two points better on their host and on every tile. On MNIST it is
  a third of a point, at two to three of its errors, and on Fashion-MNIST
  nothing that can be told. So every accuracy this plan gives for the
  inverted set is about two points low for it. And half of what §8's question
  7 put down to that set's inputs was this: its networks were 4.1 points
  behind MNIST's on the host, and at eight epochs they are 2.2 behind.
- **What a tile costs a network is what was said.** Held at version 2 the
  better trained networks lose 0.17, 0.61 and 1.04 points, where the old ones
  lose 0.19, 0.60 and 1.09. As budgeted at v1 it is 0.21, 1.35 and 1.37 for
  0.34, 1.16 and 1.23, and none of those moves by two of its errors. This
  section's prices are in points lost, and they stand. B14, B15 and B16 were
  chosen on them.
- **Noise of the tile's size is worth 0.3 to 0.5 of a point on Fashion-MNIST,
  and is not shown on the other two.** It costs nothing that can be told on
  the host. 5% does nothing. 20% is a different trade: such a network loses
  0.46 of a point to a v1 tile on Fashion-MNIST where the one with no noise
  loses 1.35, and it starts 0.38 lower on its host, and 1.43 lower on the
  inverted set. On a version 2 tile it is no better than no noise at all on
  Fashion-MNIST and 1.14 points worse on the inverted set. Points lost is the
  wrong score for a network that was trained to lose fewer.
- **Version 2 buys a trained network what it bought the others.** The same
  network on the two tiles: 0.19, 0.65 and 0.44 of a point with 10% of noise,
  for 0.19, 0.62 and 0.61 as the networks were. It is less on the inverted
  set and the same on the other two, and every one of the six is clear.
- **So training does not stand in for version 2. It adds to it.** The 10%
  network on a v1 tile is level with the old one at version 2 on
  Fashion-MNIST, 0.17 ± 0.24 short, and ahead of it on the other two, where
  it is the epochs that put it ahead. That is the test B14 set, and as B14
  worded it the test is met (§7). But the same network at version 2 is
  better again by what version 2 was adopted for. Held at version 2 the 10%
  networks are at 97.64, 87.27 and 94.63%, which is 0.38, 0.39 and 2.36
  points over the old ones there.

This is not the tile in the loop. The noise is Gaussian, the same fraction on
every layer and independent from sum to sum, and a tile's error has a
quantiser's steps, crosstalk that follows the image and a programming error
that stays put for a GEMM. Eight epochs and the three sizes of noise were
chosen once and not searched. The networks trained here were held at B16's 1%
and not at 2%, so whether a trained network needs B16 was not run. And
nothing above this block was rerun on them.

*Both things it raised were decided the same day (§7).* B14 stands. And the
networks trained for eight epochs with 10% of noise are the reference from
here on (B17): a sweep run after 2026-10-06 is of those, and every figure
above this block is of the networks trained before.

*The reference networks at the 2% and at the 1%, 2026-10-07.*
([`pta_shared_row.py`](pta_shared_row.py); grx930's design note, §5, "Does a
trained network need the 1%?".) B16 holds the row a source's lines share to 1%
and not 2%, on what the networks trained before lost on the inverted set, and
lists a network trained for the tile as one that may not need it. B17's
reference networks had been held at the 1% only. grx930's harness has now run
them at both, in the first sweep on the reference networks, with the networks
trained before beside them row for row. Version 2 on the working tile, five
networks of each kind.

What a row adds to version 2 as budgeted, in points, network by network:

| | MNIST: trained before | Reference | Fashion-MNIST: trained before | Reference | MNIST inverted: trained before | Reference |
|---|---|---|---|---|---|---|
| The lines together, 1% | 0.01 ± 0.02 | 0.03 ± 0.02 | 0.04 ± 0.04 | 0.02 ± 0.03 | 0.13 ± 0.08 | 0.00 ± 0.03 |
| 2% | 0.02 ± 0.02 | 0.06 ± 0.03 | 0.03 ± 0.04 | −0.06 ± 0.02 | 0.24 ± 0.05 | 0.05 ± 0.05 |
| 5% | 0.02 ± 0.05 | 0.08 ± 0.03 | 0.01 ± 0.08 | 0.06 ± 0.03 | 0.90 ± 0.13 | 0.46 ± 0.10 |
| A line on its own, 5% | 0.05 ± 0.05 | 0.04 ± 0.04 | 0.14 ± 0.12 | 0.16 ± 0.08 | 0.25 ± 0.07 | 0.00 ± 0.03 |
| The lines' level, 5% | 0.03 ± 0.03 | 0.02 ± 0.02 | 0.01 ± 0.07 | 0.02 ± 0.05 | 0.17 ± 0.06 | 0.12 ± 0.20 |
| All three: 2%, 5%, 5% | 0.06 ± 0.03 | 0.06 ± 0.03 | 0.09 ± 0.13 | 0.06 ± 0.06 | 0.37 ± 0.07 | 0.24 ± 0.24 |
| All three: 1%, 5%, 5% | 0.04 ± 0.04 | 0.07 ± 0.04 | 0.04 ± 0.11 | 0.03 ± 0.05 | 0.28 ± 0.06 | 0.16 ± 0.19 |
| Six minutes of TFLT's drift | −0.06 ± 0.04 | 0.01 ± 0.01 | 0.08 ± 0.06 | 0.07 ± 0.09 | 0.28 ± 0.27 | 0.08 ± 0.11 |
| Six minutes and 2%, 5%, 5% | 0.09 ± 0.02 | 0.02 ± 0.04 | 0.07 ± 0.14 | 0.11 ± 0.09 | 0.59 ± 0.32 | 0.20 ± 0.20 |
| Six minutes and 1%, 5%, 5% | 0.04 ± 0.02 | 0.04 ± 0.03 | 0.06 ± 0.15 | 0.15 ± 0.08 | 0.48 ± 0.29 | 0.24 ± 0.20 |

And the question itself: how often a network is right with the shared row at
1%, less how often at 2%, in points, network by network.

| | MNIST: trained before | Reference | Fashion-MNIST: trained before | Reference | MNIST inverted: trained before | Reference |
|---|---|---|---|---|---|---|
| The lines together, alone | +0.02 ± 0.02 | +0.03 ± 0.01 | −0.01 ± 0.03 | **−0.08 ± 0.02** | +0.11 ± 0.06 | +0.04 ± 0.03 |
| Among the three | +0.02 ± 0.02 | −0.01 ± 0.01 | +0.05 ± 0.02 | +0.03 ± 0.03 | **+0.09 ± 0.02** | +0.08 ± 0.06 |
| Held: the three, six minutes on | **+0.04 ± 0.00** | −0.02 ± 0.02 | +0.01 ± 0.04 | −0.03 ± 0.04 | +0.12 ± 0.06 | −0.04 ± 0.04 |

The two rows of a pair run from one seed and share the source's draws at half
the size, so a pair's difference has an error of a few hundredths. A hundredth
of a point is one image in ten thousand. Those in bold are outside chance at
one in twenty for five networks. One of them is four images. And one is the
other way: with the row alone, on Fashion-MNIST, the reference networks are
right eight images more often at 2% than at 1%.

- **Held as the chip is held, the 1% buys the reference networks nothing.** At
  the end of six minutes, with a line and the level at 5%, they are right
  97.66, 87.30 and 94.67% of the time with the shared row at 2%, and 97.64,
  87.27 and 94.63% at 1%. The difference is the wrong way on all three sets
  and clear on none, and five networks put it under 0.02, 0.08 and 0.08 of a
  point at one in twenty. A row of the budget is sized to a tenth.
- **As budgeted it may buy them what it bought the old ones, and that is not
  shown.** With no drift, among the three rows, the 1% is worth 0.08 ± 0.06 on
  the inverted set, where it was worth 0.09 ± 0.02 to the networks trained
  before. The same size at nearly four times the error.
- **It did buy the old networks what B16 said.** The 0.09 on the inverted set
  is five of its errors. Held it is 0.12 ± 0.06 there.
- **The rows themselves cost a trained network less on the inverted set.** The
  lines together at 2% add 0.05 where they added 0.24, and at 5% 0.46 for
  0.90. A line on its own at 5% adds nothing where it added 0.25, and that is
  the one difference of the ten that is clear, 0.24 ± 0.05 seed by seed. For
  the shared row to add about a tenth there it can be 2% for the reference
  networks, where it had to be 1%.
- **And the three together are still not shown inside a tenth there, at either
  size.** 0.24 ± 0.24 at 2% and 0.16 ± 0.19 at 1%. It is one network of the
  five: the lines' level at 5% costs seed 4 0.86 of a point, and the other
  four between a quarter of a point gained and 0.17 lost. On MNIST and on
  Fashion-MNIST the three add 0.06 at 2%.

It is the same model as everything above. The source's term is first order
and on the host's side of the line, the drift is a Mach-Zehnder's fit, and the
reference networks were trained with Gaussian noise and not with a source's.
The two rows a line carries were run at 5% and at no other size. ~~Whether B16
keeps its 1% is not decided here (§7).~~ *It keeps it: decided the same day
(B16, at its end).*

*The laser the reference networks need, 2026-10-07.*
([`pta_reference_laser.py`](pta_reference_laser.py); grx930's design note, §5,
"The laser a trained network needs".) The laser is most of what a MAC costs,
and B14's table has it at 8, 16 and 16 times B5's on the three data sets. Those
are the networks trained before. B17's reference networks were trained with
Gaussian noise on their sums, and a receiver's noise is Gaussian noise on a
sum, so of everything in this plan they were the likeliest to need less laser.
grx930's harness put version 2's rows under lasers of 2 to 32 times B5's on
both kinds of network, five of each, with the hidden layer's rescale a bit down
and at the rule's.

| | MNIST: trained before | Reference | Fashion-MNIST: trained before | Reference | MNIST inverted: trained before | Reference |
|---|---|---|---|---|---|---|
| As budgeted, points lost | 0.15 ± 0.02 | 0.07 ± 0.05 | 0.54 ± 0.22 | 0.47 ± 0.13 | 0.62 ± 0.07 | 0.52 ± 0.05 |
| Over that, the rescale a bit down: 2 times B5's laser | +0.67 ± 0.06 | +0.65 ± 0.10 | +3.53 ± 0.89 | +4.06 ± 0.32 | +4.68 ± 0.82 | +4.23 ± 0.35 |
| 4 times | +0.10 ± 0.05 | +0.18 ± 0.03 | +1.03 ± 0.30 | +1.10 ± 0.13 | +1.14 ± 0.16 | +0.96 ± 0.12 |
| 8 times | −0.04 ± 0.03 | +0.03 ± 0.02 | +0.20 ± 0.14 | +0.09 ± 0.06 | +0.30 ± 0.10 | +0.17 ± 0.07 |
| 16 times | −0.06 ± 0.04 | +0.03 ± 0.02 | +0.01 ± 0.08 | −0.10 ± 0.10 | +0.02 ± 0.08 | −0.02 ± 0.06 |
| 32 times | −0.03 ± 0.04 | +0.02 ± 0.03 | −0.12 ± 0.06 | −0.17 ± 0.08 | +0.01 ± 0.07 | −0.08 ± 0.04 |
| **The laser it needs, by the rule** | **8 times** | **8 times** | **16 times** | **8 times** | **16 times** | **16 times** |
| Which is | 0.7–7 W | 0.7–7 W | 1.4–14 W | 0.7–7 W | 1.4–14 W | 1.4–14 W |
| A MAC at it, every cell in use | 164–1,368 fJ | 164–1,368 fJ | 250–2,226 fJ | 164–1,368 fJ | 250–2,226 fJ | 250–2,226 fJ |
| At the rule's rescale it needs | 8 times | 16 times | 32 times | 32 times | 32 times | 32 times |

The rule is [`pta_tighten.py`](pta_tighten.py)'s, as B14 used it: the least
laser at which a kind of network is within a tenth of a point of what it loses
as budgeted, every larger one being so too.

| | MNIST | Fashion-MNIST | MNIST, inverted |
|---|---|---|---|
| Right, percent: the reference networks at 16 times | 97.65 | 87.51 | 94.88 |
| At 8 times | 97.65 | 87.33 | 94.70 |
| 8 times less 16, network by network | −0.01 ± 0.02 | −0.19 ± 0.08 | **−0.19 ± 0.02** |
| The networks trained before, at 16 times | 97.36 | 86.92 | 92.72 |
| The reference at 8, less those at 16, seed by seed | **+0.29 ± 0.09** | +0.40 ± 0.17 | **+1.97 ± 0.56** |
| What 8 times costs over the budget: the reference less the old, seed by seed | +0.07 ± 0.04 | −0.11 ± 0.13 | −0.13 ± 0.13 |

- **Training with noise does not buy laser.** Under each laser a reference
  network is over its own budget by what an old one is over its own. At 4
  times it is 0.18, 1.10 and 0.96 of a point, for 0.10, 1.03 and 1.14. Seed by
  seed, on the two harder sets, none of the twenty differences is clear. Why
  not is not shown here. It is not that the laser's noise was the wrong size
  to have trained for: grx930's note has what its probe says a small laser
  puts on a sum, and it is about the tenth of an rms these networks were
  trained with.
- **So by the rule they need 8, 8 and 16 times, where the old ones need 8, 16
  and 16.** One of three is halved, and by a hundredth of a point: on
  Fashion-MNIST at 8 times they are 0.09 ± 0.06 over their budget, against the
  rule's tenth. That is not a result to build on. And the laser a board has to
  place is its brightest workload's, which is 16 times and 1.4 to 14 W for
  both kinds.
- **The rule reads means to the hundredth, and one figure in B14's table turns
  on that too.** On MNIST at 4 times the old networks are 0.096 of a point
  over, which the rule reads as a tenth and not within it. Read exactly they
  would need 4 times and not 8. B14's table has 8, and it stays: MNIST does not
  size the laser.
- **Half the laser costs a reference network a fifth of a point on the two
  harder sets.** 0.19 on each, where it costs the old networks 0.19 and 0.28.
- **And at half the laser they are still ahead of the old networks at all of
  it**, by 0.29, 0.40 and 1.97 points. That is what B17 bought, and none of it
  is the laser's doing. It is there to spend if a laser of 1.4 to 14 W cannot
  be placed: 8 times is 0.7 to 7 W and a MAC of 164 to 1,368 fJ on every set.
- **The rescale a bit down still halves the laser at version 2, for both
  kinds.** At the rule's rescale the old networks need 8, 32 and 32 times and
  the reference 16, 32 and 32. Only the old networks on MNIST get nothing from
  it. That is the laser at the rule's shift, which B14 listed as not rerun.

It is the same model as everything above. Under a laser the receiver's noise
is Gaussian, and the light's row stays the budget's 30 photons while the laser
multiplies, where a real laser moves both. The multiples are octaves: "8
times" is somewhere above 4 and no more than 8. Nothing under a laser was
held: no drift and no source's noise. Version 2 and the working tile only.
Whether to spend what B17 bought on a smaller laser is not decided here, and
nothing in this plan asks for it yet (B14, at its end).

*How long a calibration holds for the reference networks, 2026-10-07.*
([`pta_reference_drift.py`](pta_reference_drift.py); grx930's design note, §5,
"How long a calibration holds for a trained network".) B15's six minutes was
chosen on the networks trained before, and it left one thing unrun even for
them: the interval that would hold a tenth of a point on the inverted set,
where six minutes adds 0.28. B17's reference networks had been drifted for six
minutes and no longer. grx930's harness drifted both kinds for three minutes
to four hours at version 2, five networks of each, by TFLT's fit.

| | MNIST: trained before | Reference | Fashion-MNIST: trained before | Reference | MNIST inverted: trained before | Reference |
|---|---|---|---|---|---|---|
| As budgeted, points lost | 0.15 ± 0.02 | 0.07 ± 0.05 | 0.54 ± 0.22 | 0.47 ± 0.13 | 0.62 ± 0.07 | 0.52 ± 0.05 |
| What drift adds to the budget: 3 minutes of TFLT's | +0.00 ± 0.03 | +0.03 ± 0.01 | −0.01 ± 0.10 | +0.08 ± 0.04 | +0.14 ± 0.11 | −0.01 ± 0.07 |
| **6 minutes, B15's interval** | −0.06 ± 0.04 | +0.01 ± 0.01 | +0.08 ± 0.06 | +0.07 ± 0.09 | +0.28 ± 0.27 | +0.08 ± 0.11 |
| 15 minutes | +0.00 ± 0.03 | +0.07 ± 0.03 | +0.19 ± 0.08 | +0.08 ± 0.04 | +0.33 ± 0.44 | +0.46 ± 0.25 |
| 30 minutes | +0.04 ± 0.03 | +0.11 ± 0.02 | +0.27 ± 0.14 | +0.18 ± 0.08 | +0.54 ± 0.56 | +0.64 ± 0.22 |
| An hour | +0.12 ± 0.05 | +0.07 ± 0.03 | +0.67 ± 0.31 | +0.32 ± 0.09 | +1.99 ± 1.26 | +1.26 ± 0.46 |
| 2 hours | +0.25 ± 0.04 | +0.13 ± 0.04 | +1.41 ± 0.28 | +0.57 ± 0.04 | +5.75 ± 1.75 | +3.30 ± 0.93 |
| 4 hours | +0.37 ± 0.08 | +0.36 ± 0.03 | +2.36 ± 0.57 | +1.31 ± 0.26 | +16.07 ± 3.09 | +7.18 ± 1.53 |
| An hour, then calibrated | −0.06 ± 0.03 | +0.09 ± 0.02 | +0.02 ± 0.11 | +0.13 ± 0.10 | −0.01 ± 0.05 | −0.01 ± 0.05 |
| An hour of TFLN's | +2.35 ± 0.17 | +1.11 ± 0.18 | +7.29 ± 1.50 | +4.23 ± 1.00 | +37.38 ± 5.74 | +25.12 ± 5.09 |
| Held, with B16's source: at the end of 6 minutes | +0.04 ± 0.02 | +0.04 ± 0.03 | +0.06 ± 0.15 | +0.15 ± 0.08 | +0.48 ± 0.29 | +0.24 ± 0.20 |
| 30 minutes | +0.15 ± 0.06 | +0.08 ± 0.04 | +0.39 ± 0.13 | +0.20 ± 0.08 | +0.76 ± 0.67 | +0.87 ± 0.39 |
| An hour | +0.19 ± 0.03 | +0.12 ± 0.03 | +0.82 ± 0.34 | +0.29 ± 0.12 | +2.26 ± 1.34 | +1.53 ± 0.60 |
| **How long a calibration holds, by the rule** | **30 minutes** | **15 minutes** | **6 minutes** | **15 minutes** | **under 3 minutes** | **6 minutes** |

The rule is the one a row of the budget is sized by: the longest interval run
at which drift adds under a tenth of a point, every shorter one doing so too.

| | MNIST | Fashion-MNIST | MNIST, inverted |
|---|---|---|---|
| What drift adds, the reference less the old, seed by seed: 30 minutes | +0.06 ± 0.04 | −0.09 ± 0.12 | +0.10 ± 0.45 |
| What drift adds, the reference less the old, seed by seed: an hour | −0.06 ± 0.08 | −0.35 ± 0.23 | −0.74 ± 0.90 |
| What drift adds, the reference less the old, seed by seed: 2 hours | −0.11 ± 0.05 | **−0.83 ± 0.25** | −2.45 ± 1.46 |
| What drift adds, the reference less the old, seed by seed: 4 hours | −0.02 ± 0.08 | −1.05 ± 0.77 | −8.89 ± 3.63 |
| What drift adds, the reference less the old, seed by seed: an hour of TFLN's | **−1.24 ± 0.30** | **−3.06 ± 0.79** | **−12.26 ± 4.30** |
| Held, right: the reference at the end of 30 minutes, less at the end of 6 minutes | −0.03 ± 0.06 | −0.05 ± 0.09 | **−0.63 ± 0.20** |
| Held, right: the reference at the end of an hour, less at the end of 6 minutes | **−0.07 ± 0.02** | −0.15 ± 0.13 | **−1.30 ± 0.43** |
| The reference held at an hour, less the old held at 6 minutes | +0.31 ± 0.12 | +0.24 ± 0.13 | +1.06 ± 1.04 |

- **In the mean, B15's six minutes holds a tenth on all three sets for the
  reference networks.** It adds 0.01, 0.07 and 0.08 of a point to their
  budget. Two of the three have errors their own size: five networks put them
  under 0.04, 0.32 and 0.38 at one in twenty, and no nearer a tenth than that.
  For the old networks on the inverted set no interval that was run holds a
  tenth even in the mean: three minutes adds 0.14 ± 0.11. That is the thing
  B15 left unrun.
- **And nothing longer does.** A quarter of an hour adds 0.46 ± 0.25 on the
  inverted set. By the rule a calibration holds 15, 15 and 6 minutes for the
  reference networks, and 30 minutes, 6 minutes and under 3 for the old ones.
  The inverted set sizes the interval for both, and the interval is B15's.
- **Under heavy drift a reference network loses half to two thirds of what
  an old one does.** An hour of TFLN's adds 1.11, 4.23 and 25.12 points for
  2.35, 7.29 and 37.38, and seed by seed that is clear on all three sets. Two
  hours of TFLT's on Fashion-MNIST is clear, 0.57 for 1.41. An hour of it is
  0.32 for 0.67 and 1.26 for 1.99, and is not clear. So unlike the laser,
  drift is something the training buys back, where there is a lot of it.
- **At half an hour and under nothing can be told between them.** None of the
  twelve differences is clear, and on the inverted set the reference networks
  are no better: 0.46 for 0.33 at a quarter of an hour. What training buys
  against drift, it buys at intervals this plan does not mean to run at.
- **A calibration leaves the reference networks about a tenth short on two
  sets.** After an hour and a calibration they are 0.09 ± 0.02, 0.13 ± 0.10
  and −0.01 ± 0.05 from their budget, where the old ones are −0.06, 0.02 and
  −0.01. MNIST's is clear, and is nine images in ten thousand. Every drift
  row in this plan starts from weights as they were written and not from a
  calibration, so what a calibrated and then drifted tile costs has not been
  run for either kind.
- **A way back, if 240 calibrations a day cost too much.** Held, with B16's
  source at the end of the interval, half an hour costs the reference networks
  0.03, 0.05 and 0.63 of a point against six minutes, and an hour 0.07, 0.15
  and 1.30. At an hour, 24 a day, they are 0.31, 0.24 and 1.06 points ahead of
  the old networks at six minutes in the mean, and clear on none of the three.

It is the same model as everything above. Both fits are a Mach-Zehnder's
bias, every cell drifts on its own and none together, and the intervals are
the seven that were run. The reference networks were trained with Gaussian
noise on their sums and not against a drifted weight. By the rule the
interval is six minutes for the reference networks as it was for the old
ones, so nothing here asks B15 to move.

*The operating cycle, 2026-10-07.*
([`pta_reference_cycle.py`](pta_reference_cycle.py); grx930's design note, §5,
"An interval that starts from a calibration".) Every drift row above aged a
tile from weights as they were written. A tile in use is never that: it is
calibrated, drifts for an interval and is calibrated again, so what B15's
interval ends in is a calibrated tile and six minutes of drift. The block
above found a calibration leaves the reference networks about a tenth short
on two sets, and could not say whether that was the calibration or the hour
of drift before it. grx930's harness ran the cycle: a tile aged an interval,
calibrated, and aged the interval again, at 3 minutes to an hour, with both
kinds of network, five of each, at version 2 by TFLT's fit and C3's
calibration at 16 probes a cell.

**Corrected 2026-10-08, by the block after this one.** A calibrated row below
and a row that is not do not meet the same draws of noise. So every figure
here that sets one against the other carries the draws as well as what it
names: a calibration's cost, a cycle over the budget, a cycle against weights
as written, and held against held. What this block reads as the calibration's
own cost was the draws. Its figures are grx930's and stand, and so does what
it reads between two calibrated rows.

| | MNIST: trained before | Reference | Fashion-MNIST: trained before | Reference | MNIST inverted: trained before | Reference |
|---|---|---|---|---|---|---|
| As budgeted, points lost | 0.15 ± 0.02 | 0.07 ± 0.05 | 0.54 ± 0.22 | 0.47 ± 0.13 | 0.62 ± 0.07 | 0.52 ± 0.05 |
| What a calibration adds to the budget: with no drift at all | −0.05 ± 0.04 | +0.08 ± 0.01 | +0.03 ± 0.11 | +0.15 ± 0.12 | +0.06 ± 0.06 | −0.05 ± 0.04 |
| after 6 minutes of drift | −0.04 ± 0.05 | +0.09 ± 0.01 | +0.02 ± 0.08 | +0.13 ± 0.13 | +0.14 ± 0.07 | +0.00 ± 0.06 |
| after an hour of it | −0.06 ± 0.03 | +0.09 ± 0.02 | +0.02 ± 0.11 | +0.13 ± 0.10 | −0.01 ± 0.05 | −0.01 ± 0.05 |
| The end of a cycle, over the budget: 3 minutes | −0.04 ± 0.04 | +0.07 ± 0.03 | +0.06 ± 0.07 | +0.14 ± 0.10 | +0.34 ± 0.33 | +0.02 ± 0.07 |
| **6 minutes, B15's interval** | −0.05 ± 0.04 | +0.08 ± 0.04 | +0.16 ± 0.06 | +0.16 ± 0.08 | +0.14 ± 0.25 | +0.09 ± 0.08 |
| 15 minutes | −0.02 ± 0.06 | +0.08 ± 0.04 | +0.10 ± 0.22 | +0.23 ± 0.12 | +0.14 ± 0.32 | +0.25 ± 0.10 |
| 30 minutes | −0.01 ± 0.05 | +0.14 ± 0.02 | +0.37 ± 0.12 | +0.05 ± 0.10 | +0.98 ± 0.64 | +0.45 ± 0.27 |
| An hour | +0.08 ± 0.08 | +0.14 ± 0.04 | +0.50 ± 0.18 | +0.38 ± 0.12 | +2.34 ± 0.21 | +1.58 ± 0.39 |
| An hour, calibrated, and 6 minutes on | −0.04 ± 0.06 | +0.09 ± 0.02 | −0.04 ± 0.12 | +0.10 ± 0.13 | +0.35 ± 0.24 | +0.28 ± 0.10 |
| Held, with B16's source, at the end of a cycle: 6 minutes | +0.02 ± 0.03 | +0.11 ± 0.06 | +0.12 ± 0.11 | +0.11 ± 0.10 | +0.29 ± 0.31 | +0.28 ± 0.21 |
| 30 minutes | +0.07 ± 0.04 | +0.12 ± 0.04 | +0.48 ± 0.19 | +0.25 ± 0.09 | +1.30 ± 0.66 | +0.61 ± 0.24 |
| An hour | +0.14 ± 0.08 | +0.17 ± 0.04 | +0.55 ± 0.18 | +0.33 ± 0.13 | +2.39 ± 0.28 | +1.87 ± 0.38 |
| **How long a calibration holds in the cycle, by the rule** | **an hour** | **15 minutes** | **3 minutes** | **none that was run** | **none that was run** | **6 minutes** |

The rule is the one above, read on the end of a cycle: the longest interval
whose cycle ends under a tenth of a point over the budget, every shorter one
doing so too.

| | MNIST | Fashion-MNIST | MNIST, inverted |
|---|---|---|---|
| What a calibration adds with no drift, the reference less the old, seed by seed | **+0.13 ± 0.04** | +0.12 ± 0.15 | −0.11 ± 0.06 |
| What an hour of drift before the calibration adds to it, the reference | +0.01 ± 0.01 | −0.02 ± 0.04 | +0.04 ± 0.02 |
| The six minutes' own: a 6-minute cycle less the calibrated tile, the reference | +0.00 ± 0.04 | +0.02 ± 0.04 | +0.14 ± 0.09 |
| A cycle's end less the same interval from weights as written, the reference: 6 minutes | +0.08 ± 0.04 | +0.09 ± 0.10 | +0.02 ± 0.04 |
| A cycle's end less the same interval from weights as written, the reference: 30 minutes | +0.04 ± 0.04 | −0.13 ± 0.10 | −0.19 ± 0.28 |
| A cycle's end less the same interval from weights as written, the reference: an hour | +0.07 ± 0.04 | +0.06 ± 0.12 | +0.33 ± 0.68 |
| An hour, calibrated and 6 minutes on, less the 6-minute cycle: the reference | +0.01 ± 0.03 | −0.06 ± 0.08 | +0.18 ± 0.10 |
| and the old networks | +0.01 ± 0.04 | −0.20 ± 0.10 | +0.21 ± 0.29 |
| Held at the end of a 6-minute cycle, the reference: right, percent | 97.57 | 87.30 | 94.58 |
| less than held from weights as written, by | +0.06 ± 0.05 | −0.03 ± 0.11 | +0.05 ± 0.02 |
| and ahead of the old networks held the same way, by | +0.30 ± 0.12 | **+0.48 ± 0.15** | **+2.12 ± 0.72** |

- ~~**The shortfall is the calibration's own, and not the hour's.**~~ *It was
  the draws, and not the calibration: the block after this one.* A tile
  calibrated as it was written, with no drift at all, leaves the reference
  networks 0.08 ± 0.01, 0.15 ± 0.12 and −0.05 ± 0.04 of a point from their
  budget, and the old ones −0.05, 0.03 and 0.06. MNIST's is clear at 7.8 of
  its errors: each of the five networks loses, 5 to 11 images in ten
  thousand. An hour of drift before the calibration adds 0.01 ± 0.01, −0.02 ±
  0.04 and 0.04 ± 0.02 to that. On the first two sets it is the size of a row
  of the budget, and no row of the budget carries it.
- **At the end of a six-minute cycle the reference networks are within a
  tenth on two sets and over it on Fashion-MNIST.** *Over the row as budgeted,
  which is the draws as well; over their own draws, in the block after this
  one, it is 0.02, 0.02 and 0.11.* 0.08 ± 0.04, 0.16 ± 0.08
  and 0.09 ± 0.08 over their budget, where six minutes from weights as written
  left them 0.01, 0.07 and 0.08 over. None of the three is clear. The six
  minutes themselves add 0.00 ± 0.04, 0.02 ± 0.04 and 0.14 ± 0.09 to the
  calibrated tile: on MNIST and Fashion-MNIST what the cycle ends over by is
  the calibration's, and on the inverted set it is the interval's. The old
  networks end −0.05, 0.16 and 0.14 over theirs.
- **A shorter interval does not buy it back.** A three-minute cycle ends 0.07,
  0.14 and 0.02 over. By the rule a calibration holds 15 minutes on MNIST and
  6 on the inverted set for the reference networks in the cycle, and on
  Fashion-MNIST no interval that was run, where from weights as written it
  held 15, 15 and 6. Fashion-MNIST's is lost to the calibration and not to the
  interval, and the rule there is on a mean five networks do not fix: the
  half-hour cycle ends 0.05 ± 0.10 over.
- **A cycle ends where an interval from weights as written does, to what five
  networks tell.** None of the eighteen differences, both kinds at six
  minutes, half an hour and an hour, is clear. So the drift rows of this plan
  stand as the ends of intervals. MNIST's three for the reference networks
  are all over, by the calibration's eight hundredths or less.
- **What a tile was before its calibration is not seen to matter, and on the
  inverted set a tenth is not pinned.** Aged an hour, calibrated and six
  minutes on, less the six-minute cycle, is over a tenth either way in three
  of six cells and clear in none. In grx930's drift a cell walks at random,
  and what it walked before a calibration does not enter what it walks after,
  so those are two draws of the same six minutes. The reference networks' six
  minutes on the inverted set is now drawn three times: 0.08 ± 0.11 from
  weights as written, 0.09 ± 0.08 at the end of the cycle, and 0.28 ± 0.10
  after the hour's calibration.
- **Held at the end of a six-minute cycle, the reference networks are where
  this plan has them.** Right 97.57, 87.30 and 94.58% of the time, for 97.64,
  87.27 and 94.63 held from weights as written, and 0.30, 0.48 and 2.12 points
  ahead of the old networks held the same way. They lose 0.18, 0.59 and 0.80
  of a point to the tile so held. In all 19 rows on all three sets they are
  ahead of the old networks in the mean, and clear of chance in 38 of the 57.
- ~~**The calibration takes a third of the reference networks' lead on MNIST,
  and why is not shown.**~~ *The draws did: the block after this one.* As
  budgeted they are 0.38 ± 0.11 ahead there and
  calibrated 0.25 ± 0.10: what a calibration adds to them less to the old
  ones is 0.13 ± 0.04, clear at 3.1 of its errors. On MNIST grx930's probe has
  the calibrated tile's sums as far from the host's as the budgeted tile's,
  to a hundredth of the error itself, and the accuracy its own model
  predicts from that error unmoved (its note). What a calibration
  leaves in a cell stays there until the next one, where a programming error
  is drawn again at every write. That is a difference between the two tiles
  and is not shown to be the cause.

It is one calibration, where a tile in use has had hundreds, and nothing here
says the shortfall does or does not build. It is C3's calibration as grx930's
harness has it, 16 probes a cell and a trim step of a quarter of a weight's
LSB, taken before the light is lit, and no other. By the rule six minutes
still holds on the inverted set, which is the set that sized it, and where a
six-minute cycle ends over a tenth a three-minute one does too. So nothing
here asks B15 to move. ~~What it asks is what in a calibration costs a network
trained for the tile, and whether that becomes a row of the budget.~~ *Asked,
and answered on 2026-10-08 in the block after this one: nothing in the
calibration does.*

*What a calibration costs: its trims, and its draws, 2026-10-08.*
([`pta_reference_calibration.py`](pta_reference_calibration.py); grx930's
design note, §5, "What a calibration costs: its trims, and its draws".) The
block above read a calibration's cost off two rows, a tile calibrated as it
was written and the tile as budgeted: 0.08 ± 0.01 of a point on MNIST for the
reference networks. Asked whether to make that a row of the budget or to find
its cause first, the choice was to find the cause. The two runs differ in two
things. A calibration writes trims: with no drift a cell's trim is minus the
mean of the programming errors its probes happened to meet, about a quarter
of an LSB, and nothing the cell will meet again. And a calibration's probes
are GEMMs, six a probe, each of which takes the run's next seed, so every
image of a calibrated run meets other noise and other programming errors than
it does as budgeted. grx930's harness took the two apart: the same probes
taken and no trim written is the calibrated run's draws on the tile as
budgeted. It ran that at 1 to 64 probes a cell, with finer trim steps, and on
other draws, with both kinds of network, five of each, at version 2 and with
no drift anywhere.

| | MNIST: trained before | Reference | Fashion-MNIST: trained before | Reference | MNIST inverted: trained before | Reference |
|---|---|---|---|---|---|---|
| What the trims add, on the same draws: one probe a cell | −0.03 ± 0.03 | +0.01 ± 0.04 | +0.18 ± 0.09 | +0.15 ± 0.13 | +0.70 ± 0.58 | +0.24 ± 0.19 |
| 4 probes | −0.03 ± 0.02 | +0.01 ± 0.02 | +0.12 ± 0.05 | −0.09 ± 0.05 | +0.06 ± 0.14 | +0.02 ± 0.05 |
| **16 probes, C3's** | +0.02 ± 0.02 | +0.02 ± 0.02 | +0.06 ± 0.02 | +0.01 ± 0.05 | −0.05 ± 0.06 | −0.03 ± 0.05 |
| 64 probes | +0.00 ± 0.02 | +0.00 ± 0.01 | −0.02 ± 0.03 | −0.04 ± 0.04 | +0.05 ± 0.06 | +0.01 ± 0.02 |
| 16 probes, a trim step of a sixteenth of an LSB | +0.02 ± 0.02 | +0.01 ± 0.02 | +0.01 ± 0.03 | +0.02 ± 0.04 | −0.04 ± 0.04 | +0.00 ± 0.06 |
| 16 probes, a 256th | +0.01 ± 0.02 | +0.01 ± 0.01 | +0.00 ± 0.02 | +0.00 ± 0.04 | −0.03 ± 0.04 | −0.01 ± 0.06 |
| 64 probes, a 256th | +0.01 ± 0.02 | +0.01 ± 0.02 | −0.02 ± 0.03 | −0.02 ± 0.05 | +0.08 ± 0.03 | +0.03 ± 0.02 |
| 16 probes, over the three draws calibrated on | +0.01 ± 0.01 | −0.01 ± 0.01 | +0.05 ± 0.03 | +0.00 ± 0.03 | +0.02 ± 0.04 | +0.01 ± 0.03 |
| A calibrated row less the row as budgeted, as the block above read it | −0.05 ± 0.04 | **+0.08 ± 0.01** | +0.03 ± 0.11 | +0.15 ± 0.12 | +0.06 ± 0.06 | −0.05 ± 0.04 |
| of which its draws: the same probes, nothing written | −0.07 ± 0.05 | **+0.07 ± 0.02** | −0.03 ± 0.12 | +0.14 ± 0.09 | +0.11 ± 0.05 | −0.02 ± 0.08 |
| The plan's row as budgeted, less the mean of twelve other draws | −0.01 ± 0.04 | **+0.08 ± 0.03** | −0.04 ± 0.12 | +0.07 ± 0.08 | +0.11 ± 0.05 | −0.04 ± 0.05 |
| A network from draw to draw, one standard deviation | 0.06 | 0.05 | 0.14 | 0.14 | 0.09 | 0.11 |
| The five networks' mean, the same | 0.04 | 0.02 | 0.07 | 0.06 | 0.06 | 0.04 |
| Points lost to the tile as budgeted: the plan's row | 0.15 ± 0.02 | 0.07 ± 0.05 | 0.54 ± 0.22 | 0.47 ± 0.13 | 0.62 ± 0.07 | 0.52 ± 0.05 |
| over thirteen draws | 0.14 ± 0.04 | 0.14 ± 0.03 | 0.51 ± 0.12 | 0.54 ± 0.07 | 0.72 ± 0.11 | 0.48 ± 0.06 |

In bold, what five networks put outside chance at one in twenty.

| | MNIST: trained before | Reference | Fashion-MNIST: trained before | Reference | MNIST inverted: trained before | Reference |
|---|---|---|---|---|---|---|
| The cycle's rows over the tile as budgeted on their own draws: calibrated as written | +0.02 ± 0.02 | +0.02 ± 0.02 | +0.06 ± 0.02 | +0.01 ± 0.05 | −0.05 ± 0.06 | −0.03 ± 0.05 |
| aged 6 minutes, then calibrated | +0.03 ± 0.02 | +0.02 ± 0.01 | +0.05 ± 0.04 | −0.01 ± 0.06 | +0.03 ± 0.05 | +0.01 ± 0.05 |
| aged an hour, then calibrated | +0.01 ± 0.02 | +0.02 ± 0.02 | +0.06 ± 0.02 | −0.01 ± 0.02 | −0.12 ± 0.04 | +0.01 ± 0.05 |
| The end of a cycle: 3 minutes | +0.03 ± 0.02 | +0.00 ± 0.03 | +0.09 ± 0.10 | +0.00 ± 0.03 | +0.23 ± 0.31 | +0.04 ± 0.04 |
| **6 minutes, B15's interval** | +0.02 ± 0.03 | +0.02 ± 0.04 | +0.19 ± 0.07 | +0.02 ± 0.02 | +0.03 ± 0.21 | +0.11 ± 0.09 |
| 15 minutes | +0.05 ± 0.04 | +0.01 ± 0.05 | +0.14 ± 0.14 | +0.09 ± 0.07 | +0.04 ± 0.34 | **+0.26 ± 0.07** |
| 30 minutes | **+0.06 ± 0.01** | **+0.08 ± 0.02** | **+0.41 ± 0.08** | −0.09 ± 0.06 | +0.87 ± 0.63 | +0.47 ± 0.27 |
| An hour | +0.14 ± 0.06 | +0.07 ± 0.04 | **+0.54 ± 0.17** | +0.24 ± 0.09 | **+2.23 ± 0.24** | **+1.60 ± 0.42** |
| An hour, calibrated, and 6 minutes on | +0.03 ± 0.02 | +0.03 ± 0.02 | +0.00 ± 0.05 | −0.04 ± 0.09 | +0.24 ± 0.25 | **+0.30 ± 0.09** |
| **How long a calibration holds in the cycle, by the rule** | **30 minutes** | **an hour** | **3 minutes** | **30 minutes** | **none that was run** | **3 minutes** |

- **It was not the calibration.** For the reference networks the calibrated
  row is 0.08, 0.15 and −0.05 of a point from the row as budgeted. The same
  probes with nothing written are 0.07 ± 0.02, 0.14 ± 0.09 and −0.02 ± 0.08
  from it, and the trims are the rest: 0.02 ± 0.02, 0.01 ± 0.05 and −0.03 ±
  0.05. Over the three draws that were calibrated on they add −0.01 ± 0.01,
  0.00 ± 0.03 and 0.01 ± 0.03, and 0.01, 0.05 and 0.02 to the old networks.
  There is no cost to make a row of.
- **More probes and a finer trim buy nothing that can be seen.** With 64
  probes at a 256th of an LSB the trims add 0.01 ± 0.02, −0.02 ± 0.05 and 0.03
  ± 0.02 to the reference networks. With one probe a cell, where the trim a
  cell is left with is as large as its programming error, they add 0.01 ±
  0.04, 0.15 ± 0.13 and 0.24 ± 0.19, and 0.70 ± 0.58 to the old networks on the
  inverted set: more on the harder sets, and clear on none. Of 54 such
  differences, one is outside chance at one in twenty, where chance gives 2.7.
- **A row moves when nothing changes but its draw.** Over twelve other draws
  of the tile as budgeted, a reference network's accuracy has a standard
  deviation of 0.05, 0.14 and 0.11 of a point, and the five networks' mean of
  0.02, 0.06 and 0.04; the old networks' mean, 0.04, 0.07 and 0.06. Two rows
  that do not share their draws differ by 0.03, 0.08 and 0.05 at one standard
  deviation with nothing else changed. A calibrated row and one that is not
  never share theirs.
- **The plan's rows as budgeted are one draw each, and on MNIST the reference
  networks' is a favourable one.** It is 0.08 ± 0.03 above their mean over
  the twelve others, 2.9 of its errors. A tile's seed in that row is its
  network's, and nothing in grx930's trainer or its model ties the two. Five
  networks no sweep had run, seeds 6 to 10, are 0.02 ± 0.03 above on their own
  seed. Over thirteen draws the reference networks lose 0.14, 0.54 and 0.48 of
  a point to the tile, where the row has 0.07, 0.47 and 0.52, and the old ones
  0.14, 0.51 and 0.72, where it has 0.15, 0.54 and 0.62. Seed by seed the
  difference is 0.01 ± 0.04, 0.03 ± 0.10 and −0.24 ± 0.13: on MNIST and
  Fashion-MNIST a network trained for the tile loses to the tile as budgeted
  what one that was not does. It is right 0.29 ± 0.09, 0.38 ± 0.17 and 2.26 ±
  0.52 points more often, for the row's 0.38, 0.48 and 2.12.
- **On its own draws a six-minute cycle ends within a tenth on MNIST and
  Fashion-MNIST, and a hundredth over it on the inverted set.** 0.02 ± 0.04,
  0.02 ± 0.02 and 0.11 ± 0.09 for the reference networks, where the block
  above had 0.08, 0.16 and 0.09 over the row as budgeted. By the rule a
  calibration holds an hour, 30 minutes and 3 minutes for them in the cycle.
  Fashion-MNIST, where no interval held, went with the draws. The inverted
  set sizes the interval again, and its six minutes is now drawn three times,
  each over its own draws: 0.08 ± 0.11 from weights as written, 0.11 ± 0.09 at
  the end of the cycle, and 0.30 ± 0.09 after the hour's calibration. Whether
  six minutes holds a tenth there is not settled by five networks and three
  draws of the drift.
- **How it was missed, and what changes.** The test was not wrong about the
  0.08: one in twenty is wrong once in twenty, and this plan has read some
  hundreds of such differences. What was wrong was to take the calibrated
  rows, which share their draws, for separate findings of it. From here a
  calibrated row is read against the row that took its probes and wrote
  nothing, and a difference between two rows that do not share their draws is
  read against what the draws alone do.

It is a calibration with nothing to correct: nothing here drifts, and what a
calibration is worth when there is drift to take out is the cycle's rows. It
is C3's calibration as grx930's harness has it, and no other. The held rows
of this plan, with a source's rows, are one draw each as well, and were not
run on others. Version 2 and the working tile only.

*The working point over draws, 2026-10-08.*
([`pta_reference_draws.py`](pta_reference_draws.py); grx930's design note, §5,
"The working point over draws".) The block above left two things on one draw
each. The inverted set's six minutes for the reference networks, which sizes
B15's interval, had been drawn three times, at 0.08, 0.11 and 0.30 of a point
over, and a tenth was not pinned between them. And the rows this plan quotes
for the chip as it is held were one draw. Asked which to do, the choice was
to pin the first before deciding anything, and to fold in the second.
grx930's harness ran five rows on each of ten draws, with both kinds of
network, five of each, at version 2: the tile as budgeted; held as B14, B15
and B16 hold the chip, which is six minutes of drift and a source's three
rows; a calibration's probes taken and nothing written; and a cycle of three
minutes and of six. A draw seeds the tile anew, which moves its noise, its
drift's walk and its source. A cycle is read over the probes-only row of its
own draw and the held row over the as-budgeted row of its own draw, so that
each pair meets the same noise.

| | MNIST: trained before | Reference | Fashion-MNIST: trained before | Reference | MNIST inverted: trained before | Reference |
|---|---|---|---|---|---|---|
| A three-minute cycle over the tile as budgeted on the same draws: the mean of ten draws | −0.01 ± 0.01 | +0.00 ± 0.01 | +0.02 ± 0.02 | +0.03 ± 0.02 | **+0.30 ± 0.06** | **+0.08 ± 0.02** |
| the draws' standard deviation | 0.04 | 0.03 | 0.07 | 0.08 | 0.16 | 0.06 |
| draws of the ten at a tenth or over | 0 | 0 | 1 | 2 | 10 | 4 |
| **A six-minute cycle, B15's interval: the mean of ten draws** | +0.01 ± 0.02 | −0.01 ± 0.01 | +0.09 ± 0.04 | **+0.07 ± 0.02** | **+0.41 ± 0.08** | **+0.12 ± 0.03** |
| the draws' standard deviation | 0.03 | 0.02 | 0.12 | 0.08 | 0.21 | 0.06 |
| draws of the ten at a tenth or over | 0 | 0 | 5 | 5 | 8 | 8 |
| on the plan's draw alone, as the block above had it | +0.02 ± 0.03 | +0.02 ± 0.04 | +0.19 ± 0.07 | +0.02 ± 0.02 | +0.03 ± 0.21 | +0.11 ± 0.09 |
| **How long a calibration holds, by the rule** | **6 minutes** | **6 minutes** | **6 minutes** | **6 minutes** | **under 3 minutes** | **3 minutes** |

The rule is the one above, read on the mean over the draws: the longer
interval whose cycle ends under a tenth of a point, the shorter doing so too.
The errors are five networks', each averaged over its ten draws. In bold, what
they put outside chance at one in twenty.

| | MNIST: trained before | Reference | Fashion-MNIST: trained before | Reference | MNIST inverted: trained before | Reference |
|---|---|---|---|---|---|---|
| Held as the chip is: right, percent, on the plan's draw | 97.26 | 97.64 | 86.88 | 87.27 | 92.27 | 94.63 |
| **over ten draws** | 97.23 ± 0.10 | 97.59 ± 0.05 | 86.79 ± 0.14 | 87.30 ± 0.12 | 91.96 ± 0.64 | 94.65 ± 0.36 |
| Points lost to the tile so held: on the plan's draw | 0.19 ± 0.03 | 0.11 ± 0.05 | 0.60 ± 0.23 | 0.62 ± 0.13 | 1.09 ± 0.36 | 0.76 ± 0.22 |
| **over ten draws** | 0.22 ± 0.01 | 0.17 ± 0.03 | 0.68 ± 0.11 | 0.59 ± 0.09 | 1.40 ± 0.24 | 0.74 ± 0.08 |
| The plan's draw, less a network's mean over the other nine | +0.03 ± 0.03 | +0.06 ± 0.04 | +0.10 ± 0.15 | −0.03 ± 0.05 | +0.34 ± 0.20 | −0.02 ± 0.22 |
| What holding adds over the tile as budgeted on the same draws | **+0.08 ± 0.03** | **+0.03 ± 0.01** | **+0.20 ± 0.04** | +0.06 ± 0.04 | **+0.68 ± 0.14** | **+0.25 ± 0.04** |
| Points lost as budgeted, over the ten draws | 0.15 ± 0.04 | 0.14 ± 0.03 | 0.48 ± 0.12 | 0.53 ± 0.07 | 0.71 ± 0.10 | 0.49 ± 0.05 |

| | MNIST | Fashion-MNIST | MNIST, inverted |
|---|---|---|---|
| The reference networks less the old, seed by seed, over ten draws: right, held | **+0.36 ± 0.09** | **+0.50 ± 0.12** | **+2.68 ± 0.59** |
| right, as budgeted | **+0.31 ± 0.09** | +0.36 ± 0.17 | **+2.25 ± 0.51** |
| what holding adds | −0.05 ± 0.02 | **−0.14 ± 0.04** | −0.43 ± 0.17 |
| what a six-minute cycle adds | −0.02 ± 0.01 | −0.02 ± 0.05 | **−0.29 ± 0.07** |
| A six-minute cycle over a three-minute one, the reference | +0.00 ± 0.01 | +0.04 ± 0.03 | +0.04 ± 0.04 |

- **On the inverted set a six-minute cycle ends over a tenth of a point in the
  mean, and a three-minute one under it.** 0.12 ± 0.03 and 0.08 ± 0.02 for the
  reference networks. The five networks' mean is at a tenth or over on 8 of
  the ten draws at six minutes and on 4 at three. By the rule a calibration
  holds three minutes there for them, and not six. Neither is far from the
  tenth: six minutes is 0.6 of its errors over it and three 1.3 under, ten
  draws put the six-minute mean between 0.08 and 0.16, and five networks
  between 0.04 and 0.20. Six minutes costs 0.04 ± 0.04 more than three.
- **On MNIST and Fashion-MNIST six minutes holds.** −0.01 ± 0.01 and 0.07 ±
  0.02. On Fashion-MNIST the five networks' mean is at a tenth or over on 5 of
  the ten draws, and under it in their mean.
- **One draw did not pin it.** From draw to draw a six-minute cycle's cost has
  a standard deviation of 0.02, 0.08 and 0.06 of a point, and on the inverted
  set it runs from 0.01 to 0.19. The plan's draw has 0.11.
- **For the networks trained before, no interval that was run holds on the
  inverted set.** A three-minute cycle ends 0.30 ± 0.06 over and a six-minute
  one 0.41 ± 0.08, and five networks put both over a tenth. That is the set
  B15 was settled without. A six-minute cycle costs a reference network 0.29
  ± 0.07 of a point less there than the old one of its seed.
- **The chip as it is held, over ten draws.** The reference networks are right
  97.59, 87.30 and 94.65% of the time and lose 0.17, 0.59 and 0.74 of a point
  to the tile, where the plan's draw has 97.64, 87.27 and 94.63, and 0.11, 0.62
  and 0.76. The networks trained before are right 97.23, 86.79 and 91.96% of
  the time and lose 0.22, 0.68 and 1.40, for the draw's 97.26, 86.88 and
  92.27, and 0.19, 0.60 and 1.09. In none of the six cells is the plan's draw
  clear of the other nine; the farthest is the old networks' on the inverted
  set, 0.34 ± 0.20 above them.
- **What training for the tile buys is in what holding adds.** The block above
  found that as budgeted a network trained for the tile loses what one that
  was not does. Six minutes of drift and a source's rows add 0.03, 0.06 and
  0.25 of a point to what the reference networks lose on the same draws, and
  0.08, 0.20 and 0.68 to the old ones: seed by seed −0.05 ± 0.02, −0.14 ± 0.04
  and −0.43 ± 0.17, less on all three sets and clear on Fashion-MNIST. Held,
  the reference networks are right 0.36 ± 0.09, 0.50 ± 0.12 and 2.68 ± 0.59
  points more often, clear on all three, where as budgeted they are 0.31,
  0.36 and 2.25.

The ten draws are of the same five networks, and the intervals are three
minutes and six and no other. A draw moves the noise, the walk and the source
together; the pairing takes out the noise and leaves the other two. It is one
calibration and not a schedule, at version 2 on the working tile. B15 is not
changed here: what this asks of it is under B15.

*A line's level, read, 2026-10-08.*
([`pta_reference_level.py`](pta_reference_level.py); grx930's design note, §5,
"A line's level, and a probe that reads it".) This section's source rows ask
for a comb's lines level to 5%, and say below that nothing built measures a
line's level. The calibration note's open question 5 said a probe with
weights in it would. grx930's harness has that probe now. It lights one row
at full scale against a full-scale weight on every column, with the rows
either side at a weight of zero so that crosstalk brings it nothing, and a
shot's sum over what was asked for is one plus that line's error. What it
reads is taken off the line, to a step of 1/256. It draws from a seed of its
own, so a run meets the same noise with it and without it, which the cell
calibration does not do (the block before the last).

It was run at version 2 on the working tile, on three draws, with both kinds
of network. Every lit row has the source's noise at B16's two rows, 1% for
the lines together and 5% for a line on its own. The lines are then level, or
off by this plan's 5%, by 20% or by 40% rms, each left alone and each read
first. What a row costs is read over the level row of its own draw.

| | MNIST: trained before | Reference | Fashion-MNIST: trained before | Reference | MNIST inverted: trained before | Reference |
|---|---|---|---|---|---|---|
| Lines 5% off, 0.2 dB, this plan's row: left alone | −0.01 ± 0.03 | +0.00 ± 0.01 | −0.04 ± 0.05 | −0.04 ± 0.04 | +0.10 ± 0.11 | +0.16 ± 0.08 |
| read first, sixteen shots a row | +0.01 ± 0.01 | **−0.02 ± 0.00** | −0.02 ± 0.02 | +0.00 ± 0.01 | +0.04 ± 0.03 | +0.01 ± 0.01 |
| Lines 20% off, 0.8 dB: left alone | **+0.41 ± 0.05** | **+0.35 ± 0.01** | **+0.62 ± 0.16** | **+0.40 ± 0.13** | **+1.69 ± 0.38** | **+1.63 ± 0.54** |
| read first, sixteen shots a row | +0.00 ± 0.00 | **−0.02 ± 0.00** | −0.01 ± 0.02 | +0.01 ± 0.02 | +0.05 ± 0.03 | −0.01 ± 0.01 |
| Lines 40% off, 1.5 dB: left alone | **+2.43 ± 0.19** | **+2.01 ± 0.07** | **+3.19 ± 0.75** | **+2.26 ± 0.45** | **+8.38 ± 1.16** | **+7.79 ± 2.26** |
| read first, sixteen shots a row | +0.01 ± 0.01 | **−0.02 ± 0.00** | −0.03 ± 0.01 | +0.00 ± 0.01 | +0.05 ± 0.03 | +0.00 ± 0.01 |
| Level lines, read | +0.01 ± 0.01 | **−0.02 ± 0.00** | −0.03 ± 0.02 | +0.00 ± 0.01 | +0.04 ± 0.03 | +0.00 ± 0.01 |
| Lines 20% off, read with one shot a row | **+0.02 ± 0.01** | +0.00 ± 0.02 | −0.03 ± 0.04 | +0.05 ± 0.05 | +0.15 ± 0.11 | +0.05 ± 0.02 |
| read with sixty-four | +0.01 ± 0.01 | +0.00 ± 0.00 | +0.00 ± 0.01 | **+0.02 ± 0.00** | +0.02 ± 0.01 | −0.01 ± 0.01 |

The errors are five networks', each averaged over its three draws. In bold,
what they put outside chance at one in twenty.

| What a read of the lines… | Finds, rms | Leaves, rms | Should leave, by count | Shots | Beats, one bank |
|---|---|---|---|---|---|
| sixteen shots a row, lines 5% off | 5.0% | 0.92% | 0.91% | 2,048 | 2,240 |
| lines 20% off | 20.1% | 0.91% | 0.91% | 2,048 | 2,240 |
| lines 40% off | 40.2% | 0.92% | 0.91% | 2,048 | 2,240 |
| level lines | 0.0% | 0.91% | 0.91% | 2,048 | 2,240 |
| one shot a row, lines 20% off | 20.1% | 3.61% | 3.61% | 128 | 320 |
| sixty-four, lines 20% off | 20.1% | 0.45% | 0.46% | 8,192 | 8,384 |

| | MNIST: trained before | Reference | Fashion-MNIST: trained before | Reference | MNIST inverted: trained before | Reference |
|---|---|---|---|---|---|---|
| Held as the chip is, over the tile as budgeted on the same draws | +0.07 ± 0.03 | +0.01 ± 0.02 | +0.18 ± 0.11 | +0.04 ± 0.07 | **+0.53 ± 0.17** | **+0.28 ± 0.07** |
| held, with the lines read first | +0.06 ± 0.03 | +0.01 ± 0.03 | **+0.21 ± 0.07** | +0.05 ± 0.04 | **+0.54 ± 0.14** | **+0.10 ± 0.03** |
| What the read buys, held | +0.01 ± 0.01 | +0.00 ± 0.03 | −0.03 ± 0.05 | −0.01 ± 0.05 | −0.01 ± 0.11 | +0.18 ± 0.09 |
| Points lost to the tile: held | 0.20 ± 0.01 | 0.14 ± 0.03 | 0.68 ± 0.15 | 0.54 ± 0.12 | 1.23 ± 0.25 | 0.76 ± 0.12 |
| held and read | 0.19 ± 0.02 | 0.14 ± 0.04 | 0.71 ± 0.16 | 0.55 ± 0.08 | 1.24 ± 0.19 | 0.58 ± 0.04 |

- **Left alone, a comb that is not level costs more than the tile does.** With
  lines 20% off the reference networks are right 0.35 ± 0.01, 0.40 ± 0.13 and
  1.63 ± 0.54 of a point less often than with level lines, and with lines 40%
  off 2.01 ± 0.07, 2.26 ± 0.45 and 7.79 ± 2.26. Version 2 as budgeted costs
  them 0.13, 0.50 and 0.48 on these draws: lines 40% off cost more than that
  on every set, and lines 20% off on two. Twice as far from level costs 5.8,
  5.7 and 4.8 times as much. One comb at 40% costs one reference network 22.4
  points on the inverted set. And training for the tile's noise buys little
  against it: seed by seed, lines 20% off cost a reference network −0.06 ±
  0.05, −0.22 ± 0.17 and −0.06 ± 0.62 against the old one, none of them clear.
- **At this plan's 5% it costs nothing that can be told on two sets, and may
  cost on the third.** 0.00 ± 0.01, −0.04 ± 0.04 and 0.16 ± 0.08 for the
  reference networks, and none of the six cells is clear. On one draw with the
  source's noise off this plan had 0.02, 0.02 and 0.12.
- **Read first, a comb costs nothing that can be seen, however far off it
  was.** At 5%, 20% and 40% alike the reference networks end within 0.03 of a
  point of level lines on every set, and the old ones within 0.06. So the read
  buys what the comb cost: 0.37, 0.39 and 1.64 at 20%, and 2.03, 2.26 and 7.79
  at 40%.
- **What the probe leaves is the source's noise over its shots, and not the
  comb.** Lines 5%, 20% and 40% off are left 0.92%, 0.91% and 0.92% off, and
  level lines 0.91%. Counted, sixteen shots a row on two buses should leave
  0.91%: a shot's noise of 5.1% over the root of 32, and the rounding of the
  step. One shot a row leaves 3.61%, which is inside this plan's 5%, and by
  the same count 14 leave a line within 1%.
- **One shot a row already takes back nearly all of it.** With lines 20% off
  and read with one shot the reference networks end 0.00 ± 0.02, 0.05 ± 0.05
  and 0.05 ± 0.02 over level lines, and the old ones 0.02 ± 0.01, −0.03 ± 0.04
  and 0.15 ± 0.11.
- **Held as the chip is, the read buys nothing on two sets and may buy on the
  third.** Six minutes of drift and the source's three rows add 0.01 ± 0.02,
  0.04 ± 0.07 and 0.28 ± 0.07 to what the reference networks lose as budgeted,
  and 0.01 ± 0.03, 0.05 ± 0.04 and 0.10 ± 0.03 with the lines read first. The
  read buys 0.18 ± 0.09 on the inverted set, which is 2.0 of its errors and
  not clear at one in twenty, and nothing on the other two. So read, they
  lose 0.14, 0.55 and 0.58 of a point to the tile where held they lose 0.14,
  0.54 and 0.76. It buys the old networks nothing on any set.
- **A read costs level lines nothing.** With lines that are level already the
  six cells are all within 0.05 of a point, read or not. One of them is clear
  by five networks' rule, the reference networks on MNIST at 0.020 ± 0.003 for
  the read, which is two images in ten thousand and should not be there: what
  a probe leaves on a comb is centred on nothing, 0.007% in the mean over the
  fifteen combs. On seven more draws of those two rows, by hand, it is −0.011
  ± 0.006, the other way, and over all ten −0.002 ± 0.004. Of the 36 read rows
  set beside level lines 6 are clear by that rule, none by more than 0.03 of
  a point, and they go both ways. Four of the six are that one cell seen four
  times: the sixteen-shot rows of a draw take the same probe draws, so what
  is left on the lines is the same in all four. At that size the rule
  misfires, as it did in the block before the last.
- **It takes two microseconds.** Two patterns programmed, 2,048 shots and the
  bank's weights programmed again are 2,240 beats on one bank, 2.2 µs: 12% of
  a cell calibration by the twin's formula and 3% of the most counted (B15,
  at its end). One shot a row is 320 beats.

What it is not. **A correction that is built.** What the probe reads is taken
off the line's level in the model, exactly, to its step. Where a chip would
apply it is not modelled, and each place would cost something this does not
count: on the weights as they are written, range, since a dim line's row
cannot be written above full scale; on a row's drive, hardware this plan does
not have; on a row's inputs at the host, their range in the same way. It is
first order, as the light's own term is, and a line's light does not pass
the converter in this model. A line 40% rms off is a Gaussian here, which
puts one line in 160 at less than no light: that row is a stress and not a
comb. The levels are fixed for a run, and how fast a comb's lines move, which
is what would say how often to read them, is in no document here. An error
that scales a row and is not its line's would read the same. And it is three
draws of five networks, on one tile at version 2. None of this plan's rows is
changed by it: what it asks of them is under B16.

*A line's level, corrected where a chip could, 2026-10-08.*
([`pta_reference_fix.py`](pta_reference_fix.py); grx930's design note, §5, "A
line's level, corrected where a chip could".) The block above read a comb's
lines and took the reading off the model's own record of each line, which no
chip can do, and named three places a chip could apply it with a cost that
was reasoned and not counted. Asked what to do next, the choice was to model
the correction where a chip can apply it, the weights as written first.
grx930's harness now applies the reading a row at a time, in five ways:

- **On the weights, at the tile's 6 bits.** Every weight of the row is
  written as w / (1 + r), and the tile quantises it as it does any weight.
- **On the weights, at the DAC's 8 bits.** The 6-bit weights the network was
  trained for are scaled and written at 8 bits, which is the DAC the
  calibration note's §4 puts behind a 6-bit weight code.
- **On the inputs.** Every input of the row is sent as a / (1 + r).
- **And two that raise nothing.** A dim line's row has to be raised, and a
  weight or an input at the rail cannot be. In the three above it is held
  there. Or every row is scaled down to the dimmest line's, on the 8-bit
  weights or on the inputs, and the sums are divided back by what that took
  off. That costs light, and the converter's range is left where it was.

Version 2 on the working tile, the three draws of the block above and its
rows again where the two share them, the source's noise at B16's two rows,
both kinds of network. Lines 5% and 20% off: lines 40% off were not run,
since a Gaussian comb that uneven has lines at no light, which nothing scales
back. What a row costs is read over the level row of its own draw; no
correction takes a seed of the run's.

| | MNIST: trained before | Reference | Fashion-MNIST: trained before | Reference | MNIST inverted: trained before | Reference |
|---|---|---|---|---|---|---|
| Lines 5% off, 0.2 dB, this plan's row: left alone | −0.01 ± 0.03 | +0.00 ± 0.01 | −0.04 ± 0.05 | −0.04 ± 0.04 | +0.10 ± 0.11 | +0.16 ± 0.08 |
| taken off the model's own line, as the block above did | +0.01 ± 0.01 | **−0.02 ± 0.00** | −0.02 ± 0.02 | +0.00 ± 0.01 | +0.04 ± 0.03 | +0.01 ± 0.01 |
| on the weights, at the tile's 6 bits | +0.00 ± 0.02 | +0.00 ± 0.01 | −0.06 ± 0.07 | +0.06 ± 0.03 | +0.16 ± 0.13 | +0.07 ± 0.06 |
| **on the weights, at the DAC's 8 bits** | −0.01 ± 0.01 | **−0.02 ± 0.01** | −0.03 ± 0.04 | +0.05 ± 0.03 | +0.01 ± 0.06 | +0.04 ± 0.03 |
| at 8 bits, scaled to the dimmest line | +0.02 ± 0.02 | +0.00 ± 0.02 | +0.07 ± 0.06 | **+0.10 ± 0.04** | **+0.16 ± 0.04** | **+0.15 ± 0.02** |
| on the inputs, held at full scale | +0.00 ± 0.01 | −0.02 ± 0.01 | −0.07 ± 0.05 | +0.06 ± 0.04 | −0.03 ± 0.03 | +0.08 ± 0.03 |
| on the inputs, scaled to the dimmest line | +0.01 ± 0.02 | **+0.08 ± 0.01** | +0.02 ± 0.04 | +0.06 ± 0.06 | −0.03 ± 0.04 | +0.01 ± 0.02 |
| Lines 20% off, 0.8 dB: left alone | **+0.41 ± 0.05** | **+0.35 ± 0.01** | **+0.62 ± 0.16** | **+0.40 ± 0.13** | **+1.69 ± 0.38** | **+1.63 ± 0.54** |
| taken off the model's own line, as the block above did | +0.00 ± 0.00 | **−0.02 ± 0.00** | −0.01 ± 0.02 | +0.01 ± 0.02 | +0.05 ± 0.03 | −0.01 ± 0.01 |
| on the weights, at the tile's 6 bits | +0.04 ± 0.04 | **+0.07 ± 0.02** | −0.13 ± 0.08 | +0.13 ± 0.06 | +0.03 ± 0.20 | +0.21 ± 0.14 |
| **on the weights, at the DAC's 8 bits** | +0.03 ± 0.03 | **+0.05 ± 0.02** | −0.03 ± 0.02 | +0.02 ± 0.03 | **+0.09 ± 0.03** | +0.10 ± 0.04 |
| at 8 bits, scaled to the dimmest line | **+0.27 ± 0.04** | **+0.13 ± 0.04** | **+0.72 ± 0.10** | **+0.65 ± 0.08** | **+0.89 ± 0.17** | **+0.75 ± 0.09** |
| on the inputs, held at full scale | +0.08 ± 0.03 | +0.03 ± 0.02 | +0.06 ± 0.07 | +0.09 ± 0.04 | +0.27 ± 0.13 | **+0.44 ± 0.14** |
| on the inputs, scaled to the dimmest line | **+0.33 ± 0.07** | **+0.24 ± 0.02** | **+0.87 ± 0.11** | **+0.80 ± 0.13** | **+0.84 ± 0.18** | **+0.69 ± 0.10** |

The errors are five networks', each averaged over its three draws. In bold in
the cells, what they put outside chance at one in twenty.

| With lines 20% off, a correction… | Leaves of every sum | Costs in light | Holds at a rail: MNIST | Fashion-MNIST | MNIST, inverted |
|---|---|---|---|---|---|
| on the weights, at the tile's 6 bits | 1.00 | none | 0.12% | 0.22% | 0.04% |
| on the weights, at the DAC's 8 bits | 1.00 | none | 0.14% | 0.25% | 0.05% |
| at 8 bits, scaled to the dimmest line | 0.52 | 2.8 dB | 0.00% | 0.00% | 0.00% |
| on the inputs, held at full scale | 1.00 | none | 20.52% | 11.57% | 42.84% |
| on the inputs, scaled to the dimmest line | 0.52 | 2.8 dB | 0.00% | 0.00% | 0.00% |

| | MNIST: trained before | Reference | Fashion-MNIST: trained before | Reference | MNIST inverted: trained before | Reference |
|---|---|---|---|---|---|---|
| Held as the chip is, lines 5% off: over the tile as budgeted on the same draws | +0.07 ± 0.03 | +0.01 ± 0.02 | +0.18 ± 0.11 | +0.04 ± 0.07 | **+0.53 ± 0.17** | **+0.28 ± 0.07** |
| corrected in the model | +0.06 ± 0.03 | +0.01 ± 0.03 | **+0.21 ± 0.07** | +0.05 ± 0.04 | **+0.54 ± 0.14** | **+0.10 ± 0.03** |
| **corrected on the weights, at 8 bits** | +0.06 ± 0.03 | **+0.03 ± 0.01** | +0.15 ± 0.09 | +0.07 ± 0.05 | **+0.56 ± 0.18** | **+0.12 ± 0.02** |
| corrected on the inputs, scaled to the dimmest line | +0.09 ± 0.03 | **+0.07 ± 0.01** | **+0.25 ± 0.08** | +0.07 ± 0.08 | **+0.54 ± 0.18** | **+0.15 ± 0.01** |
| What the weights at 8 bits buy, held | +0.00 ± 0.01 | −0.02 ± 0.01 | +0.03 ± 0.03 | −0.02 ± 0.04 | −0.03 ± 0.15 | +0.16 ± 0.08 |

- **Written at the DAC's 8 bits, the weights take back nearly all a comb
  cost.** With lines 20% off they end 0.05 ± 0.02, 0.02 ± 0.03 and 0.10 ± 0.04
  over level lines for the reference networks: within 0.11 of a point on
  every set, at 20% and at 5% alike, and 86%, 94% and 94% of what the comb
  cost. Against the model's own correction it leaves 0.07 ± 0.02, 0.02 ± 0.03
  and 0.11 ± 0.04. Of the five it is the only one that ends at a tenth or
  under on all three sets at 20%: the 6-bit weights do on one, the inputs held
  at full scale on two, and the two scaled to the dimmest line on none.
- **At the tile's 6 bits the grid does not eat the gain at 20%, and at 5%
  there is none to eat.** Left to the tile's quantiser the scaled weights end
  0.07 ± 0.02, 0.13 ± 0.06 and 0.21 ± 0.14 over level lines at 20%: 81%, 66%
  and 87% taken back. The DAC's two more bits buy 0.02, 0.11 and 0.10 on top,
  none of them clear. At 5% the 6-bit correction buys nothing, and on
  Fashion-MNIST it costs a tenth, which is clear. Counted, a weight has to be
  moved half a code before the grid takes off more than it leaves, and a line
  5% off moves 10%, 9% and 3% of the first layer's weights that far. Past a
  code the grid leaves 1.63 LSB of an 8-bit weight, which is what lines 7%, 7%
  and 11% off would do; the DAC's 8 bits leave 0.29.
- **Scaled to the dimmest line a correction pays in light, and on one set
  more than it buys.** With lines 20% off the dimmest line leaves 0.52 of
  every sum, 2.8 dB. So scaled, the 8-bit weights end 0.13, 0.65 and 0.75
  over level lines and the inputs 0.24, 0.80 and 0.69. On Fashion-MNIST both
  are worse than leaving the comb alone, by 0.25 ± 0.14 and 0.41 ± 0.16. Held
  at the rail does better than scaled to the dimmest on every set, in either
  place. At 5% the dimmest line leaves 0.88 of a sum, 0.6 dB.
- **On the inputs, held at full scale, it works where few inputs are at full
  scale.** At 20% it ends 0.03 ± 0.02, 0.09 ± 0.04 and 0.44 ± 0.14, with 21%,
  12% and 43% of the inputs that are not zero held at the rail. On the
  inverted set, where 81% of pixels are at full scale, it still takes back
  73% of what the comb cost.
- **At this plan's 5% there is little to buy, and what is clear is mostly
  cost.** Of the fifteen figures for what a place buys the reference networks
  there, four are clear and three of those are costs: the 6-bit weights and
  the 8-bit ones to the dimmest line on Fashion-MNIST, and the inputs to the
  dimmest on MNIST. The one that buys is the two images in ten thousand on
  MNIST that the block above found a read to buy, and then did not.
- **Held as the chip is, the weights at 8 bits buy what the model's own
  correction bought.** 0.16 ± 0.08 on the inverted set, for the model's 0.18
  ± 0.09; neither is clear. So corrected, the held chip ends 0.03, 0.07 and
  0.12 over the tile as budgeted, where left alone it ends 0.01, 0.04 and
  0.28, and the reference networks lose 0.16, 0.57 and 0.60 of a point to it.
- **For the networks trained before, the 6 bits do as well as the 8.** At 20%
  their weights at the tile's bits end 0.04 ± 0.04, −0.13 ± 0.08 and 0.03 ±
  0.20 over level lines.
- **A comb 20% off, read and corrected on 8-bit weights, costs half a tenth
  more than this plan's comb left alone on two sets, and no more on the
  third.** Lines 5% off and left alone cost the reference networks 0.00,
  −0.04 and 0.16. Lines four times as far off, corrected, cost them 0.05,
  0.02 and 0.10: seed by seed 0.05 ± 0.01, 0.06 ± 0.01 and −0.05 ± 0.06 more,
  clear on MNIST and Fashion-MNIST and not on the inverted set. At the tile's
  6 bits it is 0.06, 0.17 and 0.05 more.

What it is not. **A chip.** The weights and the inputs are scaled at the host
in the harness, with the tile's own quantisers after them. Written at 8 bits
the weights take the two bits below the code that the calibration note gives
the cell trim, and what the two do to each other there is not modelled; nor
is a write path that carries 8 bits a cell where it carried 6, a third more.
The correction is to the line's reading alone, and its own error over what it
scaled by is left. The source's noise is a share of a line's nominal light in
this model, so a dim line's row that is raised has its noise raised with it,
which a noise that went with the line's own power would not do. Scaled to
the dimmest line the lost light is lost: a laser turned up to give it back is
not modelled. The levels are fixed for a run. And it is three draws of five
networks on one tile at version 2. No row of this plan is changed by it: what
it asks of them is under B16.

*The working point, with the source's third row moved, 2026-10-09.*
([`pta_reference_point.py`](pta_reference_point.py); grx930's design note, §5,
"The working point, with the source's third row moved".) B16's third row
became lines level to 20%, read at each calibration and corrected on the
weights (§7). Two things in this plan then wanted running. Every figure for
the chip as it is held had its lines 5% off and left alone. And a cycle of
B15's had been run with a source lit once, on this plan's one draw ("The
operating cycle", above), and read as what the tile costs and not over its
own draws: the ten draws the six minutes were kept on have no source, and
the source's rows were sized on a tile that was not calibrated. grx930's
harness ran both, on the working tile at version 2, for both kinds of
network, on the first five of `pta_reference_draws.py`'s ten draws: ten rows
a draw, 500 runs a data set, of which 200 are byte for byte what that sweep
wrote.

Held: six minutes of TFLT's drift from weights as written, and a source at
1% for its lines together and 5% for a line. What a row adds over the tile as
budgeted on the same draws: the mean of five draws, with its error from the
five networks. In bold, what they put outside chance at one in twenty.

| | MNIST: trained before | Reference | Fashion-MNIST: trained before | Reference | MNIST inverted: trained before | Reference |
|---|---|---|---|---|---|---|
| Held as it was: lines 5% off, and left alone | +0.06 ± 0.03 | +0.00 ± 0.01 | **+0.23 ± 0.07** | +0.06 ± 0.06 | **+0.59 ± 0.21** | **+0.30 ± 0.06** |
| Held with lines 20% off, and left alone | **+0.49 ± 0.05** | **+0.40 ± 0.03** | **+0.99 ± 0.15** | **+0.46 ± 0.16** | **+2.20 ± 0.33** | **+1.77 ± 0.44** |
| **Held as it now is: lines 20% off, read, and corrected on the weights at 8 bits** | **+0.10 ± 0.02** | **+0.09 ± 0.02** | **+0.22 ± 0.05** | +0.06 ± 0.05 | **+0.63 ± 0.19** | **+0.27 ± 0.02** |
| The third less the first, seed by seed: what the move costs | +0.04 ± 0.03 | **+0.09 ± 0.03** | −0.01 ± 0.03 | +0.01 ± 0.03 | +0.04 ± 0.09 | −0.04 ± 0.07 |

A six-minute cycle: aged, calibrated, and aged again. What it ends over the
probes-only row of its draw.

| | MNIST: trained before | Reference | Fashion-MNIST: trained before | Reference | MNIST inverted: trained before | Reference |
|---|---|---|---|---|---|---|
| With no source, as B15 was sized | +0.03 ± 0.03 | −0.01 ± 0.02 | +0.10 ± 0.05 | +0.05 ± 0.04 | +0.29 ± 0.14 | **+0.14 ± 0.04** |
| With the source as it was | **+0.08 ± 0.03** | +0.03 ± 0.03 | **+0.22 ± 0.04** | +0.05 ± 0.02 | **+0.62 ± 0.14** | **+0.32 ± 0.07** |
| **As it is now run** | **+0.13 ± 0.01** | +0.07 ± 0.03 | **+0.19 ± 0.05** | +0.04 ± 0.03 | **+0.57 ± 0.18** | **+0.30 ± 0.05** |
| What the source's rows add at its end, as they now are | **+0.10 ± 0.03** | **+0.08 ± 0.02** | **+0.09 ± 0.03** | +0.00 ± 0.06 | **+0.29 ± 0.06** | **+0.16 ± 0.04** |
| What the move adds at its end | +0.04 ± 0.02 | +0.04 ± 0.02 | −0.04 ± 0.04 | −0.01 ± 0.04 | −0.04 ± 0.06 | −0.02 ± 0.07 |
| With no source, its trim at 8 bits | +0.04 ± 0.03 | +0.01 ± 0.02 | +0.07 ± 0.05 | +0.03 ± 0.04 | +0.30 ± 0.13 | **+0.18 ± 0.04** |
| **As it is now run, its trim at 8 bits** | **+0.11 ± 0.02** | +0.05 ± 0.02 | **+0.23 ± 0.06** | +0.09 ± 0.04 | **+0.63 ± 0.18** | **+0.33 ± 0.06** |
| What the trim at 8 bits adds: with no source | +0.01 ± 0.01 | **+0.02 ± 0.01** | −0.02 ± 0.01 | −0.01 ± 0.02 | +0.01 ± 0.03 | +0.04 ± 0.02 |
| as it is now run | −0.02 ± 0.01 | −0.02 ± 0.01 | **+0.04 ± 0.01** | +0.04 ± 0.04 | +0.06 ± 0.03 | +0.03 ± 0.03 |

- **Held, the move costs the reference networks 0.09 ± 0.03 of a point on
  MNIST, and nothing that can be told on the other two**, where it is 0.01 ±
  0.03 and −0.04 ± 0.07. On MNIST it is under a tenth, and five networks put
  it between 0.01 and 0.16. They are right 97.52, 87.30 and 94.64% of the
  time and lose 0.23, 0.59 and 0.74 of a point to the tile, where held as it
  was they lost 0.14, 0.58 and 0.78. `pta_reference_fix.py` had the move at
  0.05, 0.06 and −0.05, with no drift and on three draws. For the networks
  trained before it is 0.04, −0.01 and 0.04, none of them clear.
- **The read and the correction are part of the row.** Left alone, the comb
  the row now allows costs the reference networks 0.40 ± 0.04, 0.41 ± 0.10
  and 1.47 ± 0.39 more than held as it was, and the correction takes 0.32,
  0.40 and 1.50 of that back.
- **A cycle with a source lit ends 0.30 of a point over on the inverted set,
  and under a tenth on the other two.** As it is now run it ends 0.07 ± 0.03,
  0.04 ± 0.03 and 0.30 ± 0.05 over, and with the source as it was 0.03, 0.05
  and 0.32. On this plan's draw that last cycle is the one "The operating
  cycle" ran, which cost the reference networks 0.18, 0.59 and 0.80 of a
  point; over the five draws it costs them 0.18, 0.59 and 0.81, and as it is
  now run 0.21, 0.58 and 0.79. The move is 0.04 ± 0.02, −0.01 ± 0.04 and
  −0.02 ± 0.07 at the end of a cycle, clear on no set. On the inverted set
  drift has 0.14 of the 0.30 and the source's rows 0.16, on every one of the
  five draws a tenth or more together. B15 took a tenth for the first and
  B16's rows were sized at a tenth for the second. Each was known of its own
  row, and they had not been added on runs that share their draws. The
  networks trained before end 0.13, 0.19 and 0.57 over, a tenth or more on
  every set.
- **The trim of every cycle in this plan was held two bits finer than the DAC
  the calibration note asks for.** grx930's harness holds a trim in steps of
  a quarter of the operand's LSB, and the operand is an 8-bit weight: a
  sixteenth of a 6-bit code, which is a DAC of 10 bits. The calibration
  note's §4 asks for 8 bits behind a 6-bit code, a step of a quarter of a
  code, and its §8 measured that step at version 1 alone. Where this section
  says "a trim step of a quarter of a weight's LSB", the weight is the
  harness's 8-bit operand and not the code. With the trim at 8 bits a cycle
  with no source costs the reference networks 0.02 ± 0.01, −0.01 ± 0.02 and
  0.04 ± 0.02 more, and the cycle as it is now run −0.02 ± 0.01, 0.04 ± 0.04
  and 0.03 ± 0.03. Over both kinds of network all twelve such figures are
  within 0.06 of a point, and two are clear. On that DAC the cycle as it is
  now run ends 0.05, 0.09 and 0.33 over and the cycle B15 was sized on 0.01,
  0.03 and 0.18. So what this plan quotes for a cycle stands to a few
  hundredths of a point, and for the reference networks no set changes sides
  of a tenth.
- **A trim and a correction share the DAC's 8 bits and do little to each
  other.** B16 had that as not known. What the coarser trim costs the
  corrected cycle, less what it costs the cycle with no source, is −0.04 ±
  0.01, 0.06 ± 0.05 and −0.01 ± 0.02 for the reference networks and −0.03,
  0.07 and 0.05 for the old ones: within 0.07 of a point, of both signs,
  and clear on MNIST for the first and on Fashion-MNIST for the second.
- **Six minutes of drift do nothing to the read.** The harness reads the
  lines when an evaluation starts, which in a cycle is six minutes after the
  calibration. Its reading leaves the lines 0.99% off there, 1.00% where the
  tile was held from weights as written, and 0.98% on a tile that had not
  drifted. The rail holds 0.14%, 0.25% and 0.05% of the reference networks'
  weights back.
- **With the read, a calibration is 70.0 µs at the most**: 2.2 µs more, and
  one part in 5.1 million of six minutes where it was one in 5.3 million.

What it is not. **A chip.** The weights are scaled at the host in a model,
and the model's DAC has no rail for a code and its trim together. **A comb
whose lines move.** They are fixed for a run, so a read taken at the
calibration and one taken six minutes later differ only in the tile they are
read through. How often the lines have to be read is not measured, and no
document says how fast they move. **A schedule**: it is one calibration.
Lines further off than 20% were not run. And it is five draws of the same
five networks: over the ten, holding as it was adds 0.03, 0.06 and 0.25 and
a cycle with no source ends −0.01, 0.07 and 0.12 over, where these five have
0.00, 0.06 and 0.30, and −0.01, 0.05 and 0.14. B15 is not changed here: what
this asks of it is under B15.

*The ADC's row, since the survey (2026-10-04).* Relaxing it saves 3 to 106 mW,
and the low end is the published state of the art: the cheapest converters that
sample this fast already have the seventh bit. So of the two rows, relaxing the
receiver's saves 0.04 to 0.44 W of laser at a measured receiver whatever
converter is built. Relaxing the ADC's saves 0.1 W if the converter built is
the fifth-best published, and next to nothing if it is the best.

*What B12 adds to this section, 2026-10-05.* The tile is a ring bank on four
buses, and two requirements here were written for something simpler. **A
receiver's input is four photodiodes and not one**, a bus's each, because buses
that reuse lines cannot share one. *(Four pairs, since B13; two pairs at
128 inputs, since B10's revision.)* The interface chip still has 64 receivers
and 64 converters, and what four photodiodes do to a receiver's noise is not
priced. **And a receiver has to reject the beat between lines**, 11 GHz from
the signal at the working grid, which the single pole every figure above
assumes passes 8% of, against half an LSB's 0.4%. One that averages over
exactly one shot passes none of it when the grid is locked to the shot clock.
Its noise bandwidth is also narrower than the single pole's, which bears on the
laser and has not been run. Neither is a row of the budget yet: the error model
has no term for a beat, or for the source.

*The source's rows, measured the same day*
([`pta_source_noise.py`](pta_source_noise.py)). The error model has a term for
the source now. It is in grx930's harness, on the host's side of the line and
outside that team's contract (its design note, §5, "What may the light do?"),
and it was run on the working tile, on four buses, over version 1, which by
itself loses 0.20 ± 0.05. Points lost beyond that:

| What the source does to a line's power | Through a balanced pair | Through an offset |
|---|---|---|
| Every line together, anew each shot | 0.08 at 5% rms; 0.25 at 10% | 0.10 at 0.2%; 0.20 at 0.5%; 0.74 at 1% |
| Each line on its own, anew each shot | 0.06 at 5%; 0.16 at 10% | Nothing at 1%; 0.15 at 2% |
| Each line's level, fixed | 0.07 at 5%; 0.14 at 10% | 0.03 at 1%; 0.12 at 2% |

The two columns are two readings of how the tile signs a weight, which B12
left open: through the weight alone, as a balanced pair of photodiodes has it,
or through the weight and an offset the host takes off again, as one
photodiode would.

**Three rows for version 1, if a column reads its weights through a pair**
(*it does, since B13*)**:** the
source's noise within **2% rms a shot** for the lines together, within **5%**
for a line on its own, and the lines **level to 5%**, which is 0.2 dB. That
set loses 0.30 ± 0.04, a tenth of a point between the three. Each stands
5% alone, and 5% of all three loses 0.35 ± 0.05. They do not compound:
neither set costs more than its three rows summed, and the errors on layer
1's sums add in quadrature.

*At version 2, and on two more data sets, 2026-10-06* (at the end of this
section's budget). The three rows hold where they were sized and on
Fashion-MNIST: together they add 0.06 and 0.09 of a point to version 2 on the
working tile. On MNIST inverted they add 0.29 to version 1 and 0.37 to version
2, and the dear one is the 2% the lines share. *B16 made that 1% for version
2, the same day (§7).*

- **Through a pair the source may be 20 dB noisier than this plan assumed.**
  §8's question 8 held it to the receiver's own allowance, 0.2% of full scale,
  about −144 dB/Hz. The row's 2% is −124 over the shot rate, and −121 for a
  receiver that averages over a shot. Alone the source stands 5%, −116.
- **Through an offset it may not.** There 0.5% costs what 10% costs through a
  pair, with the same error on layer 1's sums, and the assumed 0.2% is about
  what the budget stands. The offset is every lit input at a weight of one,
  beside weights that are small.
- **So how the tile signs a weight is worth 26 dB of the source's noise,**
  twenty times. B12 said a pair would double the photodiodes, to 512 at four
  buses. This is what the other reading costs. A column of rings at a weight
  of zero that reads the same light would take the offset's noise off with the
  offset, and is the pair again.
- **Nothing built measures a line's level.** To a column it is one error on
  every weight in a row. The calibration engine probes each cell through a
  weight of zero, which a line's power multiplies
  ([`pta_chiplet_calibration.md`](pta_chiplet_calibration.md) §9). A comb's
  lines are not level as made, so something has to level them to a fifth of a
  decibel, and nothing here says what. *Something reads them since
  2026-10-08: a probe in grx930's harness, a row at full scale through a
  full-scale weight ("A line's level, read", at the end of this section's
  budget). Sixteen shots a row read a line to 0.9%. That is the measurement
  and not the levelling: where what it reads is applied is still not said.*

It is one network, a first-order term added after the converter, and noise
with no memory from shot to shot. No source's noise is held here, a comb's or
an amplifier's: these rows say what one may be.

*The rows hold on the tile as revised, 2026-10-05.* All of the above was
measured at 256 × 64 on four buses. grx930's `source` mode was run again at
128 × 64 on two, where version 1 by itself loses 0.34 ± 0.07. The three rows
together lose 0.37 ± 0.07, which is 0.03 over it; each alone stands 5%; and 5%
of all three loses 0.46 ± 0.03. Through an offset 1% together loses
0.67 ± 0.16 and 2% loses 2.25 ± 0.46: the offset passes sixteen times what a
pair does there, 24 dB, where it passed twenty at 256 rows: with fewer rows
lit, the offset is smaller beside the sums.

*The receiver's row, as one laser fixes it, 2026-10-05*
([`pta_laser.py`](pta_laser.py); B5, at its end). Version 1 gives the receiver
half an LSB of an 8-bit ADC, and the budget was measured with each layer given
half an LSB of its own converter. That is no one laser. The second layer's
sums are a quarter the size of the first's on every tile, so its LSB is a
quarter the light, and version 1 as budgeted is the first layer at 8 times
B5's laser and the second at 32.

Stated so that one laser can meet it, the row is **the receiver's noise at a
thirty-second of one line's light at a detector**: an input at full scale
through a weight of one. That is 16 times B5's laser on a 256-row tile and 8
times on a 128-row one, and it costs under a tenth of a point over version 1
on both. A sixteenth costs 0.40 on the working tile, and an eighth 2.2.

Under it the shot noise of all the light a pair carries, which B13 left open,
is 0.16 and 0.32 LSB on the two layers, where the receiver's is 0.25 and 0.99.
That is what the photon row already allows a sum of 0.4 and 1.5 LSB, and sums
are larger than that, so it is inside the row. It is not three orders away,
as B5 had the photons: it falls as the root of the laser where the receiver's
falls as the laser.

*And with the rescale a bit down, a sixteenth, 2026-10-05.* The row above is
at the hidden layer's rescale as grx930's harness sets it, by a rule that lets
one firing unit in ten thousand reach full scale. One bit under that rule the
second layer's operands are twice as large, and the same accuracy is had with
**the receiver's noise at a sixteenth of one line's light**: 4 times B5's
laser on the working tile, where the row above is 8 (B5, at its end). It
clips 0.65% of the units that fire and costs 0.02 of a point. On the chiplet
it is the activation stage's shift.

*And on a second data set, a thirty-second and a sixty-fourth, 2026-10-05.*
That sixteenth is MNIST's. The same test, within a tenth of a point of
version 1 with the rescale a bit down, takes 8 times B5's laser on the working
tile for Fashion-MNIST and 16 for MNIST inverted: **the receiver's noise at a
thirty-second and at a sixty-fourth of one line's light** (B5, at its end; §8,
question 7).

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

**B10 — The chiplet's working geometry: 128 × 64.** *Settled 2026-10-05 at
256 × 64, and revised the same day.*

*Revised 2026-10-05: 128 inputs by 64 outputs.* This decision was first made
for 256 × 64, on a cheaper MAC on wide layers. Two things found the same day
took that ground away. The smallest ring read puts 256 × 64 past a standard
die and 128 × 64 inside one. And the laser, sized on the light a column is
actually sent, is twice as large for the larger tile, which leaves it 15%
cheaper a MAC at one end of the power range and 2% at the other, where it had
been 40% and 14%. *Recommended, and settled as recommended:* **128 inputs by
64 outputs is the working geometry.**

What it fixes now, each from a model and none from a device
([`pta_working_point.py`](pta_working_point.py)):

| | At 128 × 64 | As first settled, at 256 × 64 |
|---|---|---|
| Lines between the two dies | 8,384 | 16,704 |
| The die | 62 mm² at the smallest ring read, inside a standard 100. 20.3 by the floorplan's bound | 124 mm², and 39.7 |
| Link 2 | Three modules with the weights re-sent; one with them held | Five; one |
| Interface chip | 0.44 to 4.04 W | 0.55 to 7.61 W |
| Laser | 0.7 to 7.0 W, eight times B5's method. *0.35 to 3.5 W, four times, with the hidden layer's rescale one bit under the clip rule: B5, at its end. That is on MNIST; 0.7 to 7 and 1.4 to 14 W on the two other data sets run: §8, question 7* | 1.4 to 14 W, sixteen times |
| A MAC, every cell in use | 140 to 1,351 fJ. *97 to 922 at that laser* | 120 to 1,323 fJ |
| The 4096-square layer | 131 µs | 65.7 µs |
| What the interface chip holds | 8,192 weights a bank in two banks, written 128 cells a beat; 64 receivers and 7-bit converters | 16,384, written 256 a beat |
| The ring bank | Two buses of 64 lines on an 11 GHz grid | Four buses of the same 64 |
| Photodiodes | 256 | 512 |
| A line of the comb | 11 to 110 mW | 22 to 220 mW |
| v1 on D3 | 0.34 ± 0.07 points as budgeted, 0.35 ± 0.06 at that laser | 0.20 ± 0.05, and 0.28 ± 0.06 |
| `PTA_CAPS0` | 128 rows, 64 columns | 256 rows |

**What the move costs.** Time: a wide layer takes twice as long, and D3's two
layers 1.29 µs for 0.96. And, in grx930's model, accuracy under drift: an hour
of TFLT's costs 0.51 ± 0.07 points here where it cost 0.24 ± 0.06, with four
hours the same on both. v1 itself is a standard error and a half worse as
budgeted, and no worse at the laser.

**What it buys.** Half the laser, half the lines, half the photodiodes, three
link modules for five, and a die that fits a standard service at the smallest
ring read. And a ring bank of two buses on the comb B12 already asked for, by
B12's own rule: the fewest that hold across the published range of line
spacings.

**It fits at one ring.** 62 mm² is rings touching, with nothing added to a
cell, at the smallest ring of five papers. At the next ring read it is 233. So
the fourth thing that would have reopened the first decision has not gone
away. It has gone from certain to possible.

**What would reopen it now:** a workload for which a wide layer's time is the
constraint; a ring that cannot be made near 30 µm; or a loss budget that makes
0.7 to 7 W as hard as 1.4 to 14.

The twin's gate runs a build of this size since the same day, beside the
256 × 64 it already ran: 237 checks where it had 201, every GEMM equal to
grx930's model on a device the twin never sees, and D3's shape held on the
chiplet equal to the same network brought out at every layer.

So do the three gates behind it, which had run on 256 × 64 alone: the driver's,
and S4's two, the device and its GEMM through the runtime (§3.4). The driver's
has 54 checks where it had 26, 27 on each tile. S4's run once a tile, 44 checks
and 49 where they had 44 and 48, and the GEMM's 4,164 results are compared on
each. What follows a tile's rows was derived again for 128 and not carried over:
the ADC shift that puts a K tile of int8 products inside the converter, 15 for
the 7-bit one where 256 rows take 16, and the K that walks two K tiles, 172
where it was 300. That shift is the bound for operands that fill their word,
which is what those gates use. It is not the shift grx930's harness finds on
D3, which is the same on both tiles (§8, question 1).

*As first settled, at 256 × 64, and kept as it was written:*

§8's first question has two halves, how big and how fast, and every model swept
candidate tiles because nothing fixed one. [`pta_geometry.py`](pta_geometry.py)
put the six models side by side (§8, question 1). It leaves two candidates,
128 × 64 and 256 × 64, and says that what separates them is the width of the
layers the board is for: 128 × 64 is the cheaper a MAC on layers up to 128 wide,
and 256 × 64 from 512 wide. Accuracy and drift do not choose between them.

*Recommended, and settled as recommended:* **256 inputs by 64 outputs is the
working geometry.** It is the tile the link sizing, the twin's gates and the
dispatch model were already built on. Wide layers are what this plan has sized
everything for. And nothing measured argues against it.

What it fixes, each from a model and none from a device:

| | At 256 × 64 | From |
|---|---|---|
| Lines between the two dies | 16,704 | The floorplan |
| Cells, inputs and link | 39.7 mm², a lower bound. *124 mm² or more at every ring read since: see below* | The floorplan, at its assumed 25 µm cell |
| Link 2 | Five UCIe-S modules at a batch of 64 with the weights re-sent; one with them resident | X2 |
| Interface chip | 0.55 to 7.6 W | Published parts |
| Laser | 0.09 to 0.88 W at a measured receiver, behind 10 to 20 dB of loss. *1.4 to 14 W on the light a column is sent: B5, at its end* | B5's method |
| What the interface chip holds | 16,384 weights a bank in two banks, with a write path 256 cells a beat; 64 receivers and 7-bit converters; an activation unit a column and 2^14 held operands | §4.3, X6 |
| v1 on D3 | 0.20 ± 0.05 points | grx930's harness |
| `PTA_CAPS0` | 256 rows, 64 columns | X4 |

**What it costs** over 128 × 64: twice the lines between the dies, five link
modules for three, and 0.11 to 3.6 W. On layers under 256 wide it is the dearer
tile a MAC.

**What it does not settle.** The shot rate, the other half of the question,
which stays at X2's 1 GS/s as a planning figure (and has since been put in one
table, [`pta_rate.py`](pta_rate.py), and settled there: B11). The kind of light
source (§8, question 8; settled since, with the topology: B12). And whether a weight cell is resonant, which decides whether a tile
this size fits a die at all (§8, question 1, the floorplan).

*The last of those was worked the same day, and it bears on this decision*
([`pta_ring.py`](pta_ring.py); §8, question 1). The cells are rings, as the
floorplan had already found. But the smallest thin-film lithium niobate ring
in the five papers read is 30 µm in radius, a 60 µm cell, where the floorplan
assumed 25.
At that ring the tile is 124 mm² with its inputs and its link, and at the rings
on which tuning and a high Q have actually been shown it is 466 to 584. A
standard packaging service takes 100. 128 × 64 is 62 mm² at the smallest ring,
and is the largest candidate that fits a standard die there. It fits at no
other ring read either: 233 mm² at the next.

**So B10 was made on an area that none of the devices read supports.** Five
papers are not a survey: the ring that would carry 256 × 64 is 26 µm in radius,
4 µm under the smallest read, and one may be published. But every area here is
a floor, with nothing added to a cell for electrodes, a bus or a gap. It is not
reopened here, because whether this board's photonic die may be larger than a
standard service takes is the program's to say, and P1's to cost. If it may
not, the working geometry is 128 × 64.

*Put to the program the same day, and answered: B10 stands.* The tile stays
256 × 64. That takes the photonic die past what a standard service takes at
every ring read, and what such a die costs is P1's to find.

*And the laser has been asked since* ([`pta_laser.py`](pta_laser.py); B5, at
its end). What chose this tile over 128 × 64 was a cheaper MAC on wide layers,
and that was reckoned on B5's laser, which is the same for both. Sized on the
light each is actually sent, the laser is 1.4–14 W here and 0.7–7 W at
128 × 64, and a MAC with every cell in use then costs 119 to 1,323 fJ here and
140 to 1,351 there: 15% cheaper at one end and 2% at the other, where it had
been 40% and 14%. So the larger tile buys little of a MAC any more.
It buys half the time on a wide layer, for twice the laser, twice the die and
five link modules for three. B10 is not reopened here either: it is the
program's. *The program revised it the same day: 128 × 64, at the head of
this decision.*

**It is a working geometry and not a tape-out.** Three things would reopen it: a
workload of layers 128 wide or less; a loss budget or a receiver that cannot
light 64 columns; or a weight cell that is not resonant. *A fourth, since
[`pta_ring.py`](pta_ring.py):* a photonic die held to what a standard packaging
service takes. *Needed by:* P1, L3, X2's module count and S3.

**B11 — The chiplet's working shot rate: 1 GS/s.** *Settled 2026-10-05.*

B10 fixed the tile and left the rate at X2's 1 GS/s "as a planning figure".
[`pta_rate.py`](pta_rate.py) asked six models at five rates (§8, question 1).
It leaves three candidates, 0.5, 1 and 2 GS/s, and it found a bound nobody had
priced: a weight ring passes a level no faster than its line is wide, so the
rate, the voltage a weight needs, how still its ring has to hold and how many
lines share a bus are one trade.

*Recommended, and settled as recommended:* **1 GS/s is the working shot rate.**
It is the fastest of the three that both measured parts stand behind: the one
receiver with a measurement settles a tile up to 1.7 GS/s, and the converter
that sets the best price runs at 2.7. Most of what a faster tile saves a MAC is
had by it. And it is the rate every model here was already run at.

What it fixes, each from a model and none from a device:

| | At 1 GS/s | From |
|---|---|---|
| Shot clock | 1 GHz. `PTA_CAPS2[15:0]` reads 1,000 | P0, X4 |
| Converters | 64 GS/s of 7-bit conversion in all. 70 published parts can do a column's, at 1.11 to 3.14 pJ a sample | The ADC survey |
| Receivers | 0.88 GHz of bandwidth a column. The one measured has 1.5 | Published |
| Link 2 | Five modules with the weights re-sent, one with them held on the interface chip: B10's count, now at a settled rate | X2 |
| Interface chip | 0.55 to 7.6 W | Published parts |
| Laser | 0.09 to 0.88 W at the measured receiver, behind 10 to 20 dB of loss. *1.4 to 14 W on the light a column is sent: B5, at its end* | B5's method |
| A MAC, every cell in use | 39 to 518 fJ. *119 to 1,323 with that laser* | The two rows above over the rate |
| The weight path | 256 G cells a second, 256 cells a beat, each DAC rewritten at 15.6 MHz | §4.3 |
| **The weight ring** | A Q of 97,000 at most and a swing of 2.3 V at least. A weight's LSB is 0.19 pm of resonance | Derived, [`pta_rate.py`](pta_rate.py) |
| Buses | Three or more for 256 inputs: 102 lines a bus at most, on the smallest ring read | Derived |
| The source's intensity noise | −144 dB/Hz at most | The receiver's allowance, applied to the source: assumed |

**What it costs** over 0.5 GS/s: five link modules for three, 0.15 to 3.7 W on
the interface chip, 0.03 to 0.26 W of laser, a weight swing of 2.3 V where
1.1 would do, and three buses for two. **What it buys:** half the time on a
wide layer, a MAC at 39 to 518 fJ where 0.5 GS/s has 57 to 554, and a ring half
as hard to hold still.

**What it closes.** Of the floorplan's three swings, a weight ring carries this
rate only at 5 V. The 1 V and 2 V rows stay open to the inputs, which are
modulators and not rings.

**What it does not settle.** The kind of light source (§8, question 8), which
this narrows: 102 lines a bus or fewer, on three buses or more. Whether the
tile is a ring bank at all. And whose ring: every ring figure here is derived
from five papers and one tuning efficiency. *The first two were settled the
same day, as a working topology: B12.*

**It is a working rate and not a clock specification.** Four things would
reopen it: a weight driver that cannot swing 2.3 V; a ring whose measured
tuning is far from the 7.0 pm/V assumed, which moves that swing in inverse
proportion; a link held under five modules with no store on the interface chip
for a layer's weights; or a loss budget past 20 dB behind a laser under 0.88 W.
*Needed by:* P0's shot clock, X4's `PTA_CAPS2`, L3, X2's module count and S3.

*At 128 × 64, since B10's revision*
([`pta_working_point.py`](pta_working_point.py)): three link modules with the
weights re-sent, 0.44 to 4.04 W on the interface chip, 0.7 to 7.0 W of laser
(0.35 to 3.5 with the hidden rescale a bit down: B5), 140 to 1,351 fJ a MAC
(97 to 922), 128 G cells written a second at 128 a beat, and two buses. The rate, the shot clock, the converters, the receivers and the weight
ring's Q and swing are as above: none of them knows the tile's rows.

*Under version 2, since B14 (2026-10-06;
[`pta_working_point.py`](pta_working_point.py), §4).* The rate stands. What
it asks of a ring moves, because a level read to 8 bits has to have settled
further in the same shot: a line of at least 2.21 GHz where it was 1.99, a Q
of at most 87,700 where it was 97,400, and a swing of at least 2.53 V where it
was 2.27. The first thing that would reopen this decision is then a driver
that cannot swing 2.5 V. The converters are 8-bit ones, 0.27 to 0.34 W for the
64 on 47 published parts, and the interface chip is 0.64 to 4.18 W.

**B12 — The tile's topology: a ring bank, lit by a comb.** *Settled 2026-10-05.*

§8's question 8 asked what kind of light source the tile needs, and found the
answer rested on something this plan never decided: how a column adds its
inputs. The CPU document's error model assumed a ring bank and called that a
hypothesis with no ground truth. [`pta_source.py`](pta_source.py) followed each
reading to its source. A ring bank wants a comb. One line added as fields wants
a second element in every cell. One line and a photodiode in every cell wants
16,384 photodiodes.

*Recommended, and settled as recommended:* **the tile is a ring bank.** Each
input rides its own line, a ring weighs it, and a column adds powers. It is the
one reading whose cell is a ring and nothing else. It is what the error model,
the crosstalk row and every bus count here were written for. And it is the one
with a published analysis behind it. *The working count that came with the
recommendation is four buses,* the fewest that hold across the published range
of line spacings. *At 128 inputs, since B10's revision, that rule gives two
buses, of the same 64 lines on the same grid. What follows is for four.*

What it fixes, each from a model and none from a device:

| | A ring bank on four buses | From |
|---|---|---|
| Lines | 64 on a bus, and the same 64 on all four | [`pta_source.py`](pta_source.py) |
| Grid | 11 GHz at the widest: eleven times the shot clock, and locked to it. 10 GHz, where a comb has been published, fits too | Derived |
| Span | Inside 5.7 nm, one free spectral range of the ring read | [`pta_ring.py`](pta_ring.py) |
| The weight ring | A Q of 62,000 to 97,000 and a swing of 2.3 to 3.6 V. From 81,000 at the published worst spacing | Derived |
| The source | One comb, a pump before it and an amplifier after: 0.09 to 0.88 W in all, 1.4 to 14 mW a line, on B5's one fiber. *Sixteen times both on the light a column is sent: 1.4 to 14 W, and 22 to 220 mW a line (B5, at its end)* | [`pta_rate.py`](pta_rate.py) |
| Photodiodes | 256, one a bus a column. Still 64 receivers and 64 converters. *512 since B13: a pair a bus a column* | Derived |
| The receiver | One that averages over a shot, or one steeper than a single pole | Derived |
| A weight's LSB | 24 to 38 MHz of a line's position against its ring | Derived |
| The error model's crosstalk | The nearest-neighbour chain it already has, as four chains of 64: 252 pairs of neighbours a column where it has 255 | The CPU document, §4.3 |

**What it costs** against one laser: a comb at a spacing nobody has published
on the plan's own material, where the one read is about 30 GHz; an amplifier; a
filter that picks one line for one input, 64 of them; four photodiodes a column
for one; and a second clock on the board, the comb's drive, locked to the
first.

**What it closes.** The one-laser readings. B5's single laser as the source's
kind, though not its placement: a comb is many lines on one fiber. And the CPU
document's hypothesis, as far as the board goes: it is the working topology
and still not a measured one.

**What it does not settle.** Whose comb. The amplifier and its noise: §4.3's
error model still has no term for the source. *(It has one since, and three
rows, through a balanced pair: §4.3.)* How large the beat between lines
is. How the tile signs a weight *(settled since: a balanced pair, B13)*. Where
a column's four currents are added, on the photonic die or after the bond,
which is 64 lines between the dies or 256.
And whether four is the count: three hold at the kind end of the published
range, and more are always allowed, at a photodiode a column each.

**It is a working topology and not a device.** Four things would reopen it: a
comb that cannot be had near the grid, at the power, on any material; a beat
that neither a clocked grid nor a steeper receiver removes; a detector that
makes a photodiode a cell cheap, which is the one-laser reading that needs no
second optical element; or a ring that cannot be put on its line and kept
there, since the drift this program fits is still a modulator's (§8,
question 1). *Needed by:* L3, P1, P0's link 4, X1's error model and X3.

*Under version 2, since B14 (2026-10-06).* Two buses of 64 lines on the
11 GHz grid still hold: at the line an 8-bit level needs a bus could carry 69,
where it could carry 77. What narrows is the ring. To settle in a shot its Q
is at most 87,700, and for 64 lines to share a bus at this decision's spacing
it is at least 80,200, so a ring's Q has 9% of room where it had 21%
([`pta_tighten.py`](pta_tighten.py)). A ring made outside that room is a third
bus or a slower shot.

**B13 — How the tile signs a weight: a balanced pair.** *Settled 2026-10-05.*

A weight in this plan is signed. The networks are trained with weights between
−1 and 1, and a ring only passes a line's light or does not. B12 made the tile
a ring bank and left open how it gets a sign out of that. There are two ways. A
ring sends its line's light to one of two waveguides, each ends on a
photodiode, and the receiver takes the difference: a balanced pair. Or one
photodiode reads the light a ring passes, and the host takes off the offset
that a weight of zero leaves. grx930's harness ran a source's noise through
both (§4.3): the offset passes twenty times what the pair does of the noise
the lines share.

*Recommended, and settled as recommended:* **a column reads each bus through a
balanced pair of photodiodes.** A weight is how a ring splits its line between
the two: all one way is 1, all the other is −1, and an even split is zero. It
is how the published ring bank does it (Tait et al., §8, question 8), and it
is the reading that lets the source be an amplified comb.

What it fixes, each from a model and none from a device:

| | With a balanced pair | From |
|---|---|---|
| Photodiodes | 512: two a bus a column, at four buses. Still 64 receivers and 64 converters. *256 at the two buses of 128 inputs, since B10's revision* | Counted |
| A bank's waveguides | Two out of every bank: the one its rings drop to and the one they pass | The topology |
| The source's rows of the budget | Version 1's, without the "if": 2% rms a shot for the lines together, 5% for a line on its own, lines level to 5%. A tenth of a point between them. *At version 2 (B14) they add 0.06 and 0.09 on MNIST and Fashion-MNIST and 0.37 on MNIST inverted, where the shared row wants 1%: §4.3, at the end of its budget. B16 made it 1% for version 2, and on 2026-10-09 made the third 20%, read and corrected on the weights* | grx930's harness, §4.3 |
| The source's noise, as a density | About −124 dB/Hz over the shot rate, 20 dB easier than was assumed. *−130 since B16, 14 dB easier* | [`pta_source_noise.py`](pta_source_noise.py) |
| A pair's match | Its two halves alike to about 10%. A mismatch is an offset of half its size, and at 10% the offset passes as much of the source's noise as the pair does | Derived from the two measured readings, and not run |
| The converter's span | The difference alone. Behind an offset it would have had to span the offset too | The topology |

**What it costs** over one photodiode: 256 more photodiodes, a second waveguide
out of every bank and the crossings that brings, and a pair that has to be
matched.

**What it closes.** B12's open item. And the "if" on §4.3's three rows for the
source.

**What it does not settle.** Where a pair's difference is taken, and a
column's four pairs added: on the photonic die, which is 64 lines between the
dies, or after the bond, which is 512. Whether a pair can be matched to 10%
as made, or has to be trimmed. And two things the error model does not have
for either reading. Its shot noise is the difference's, where a pair's
photodiodes carry all the light a column is sent. And B5's laser was sized on
a detector's full scale, where what a column is sent does not depend on its
weights at all. Neither has been rerun. *Both have since*
([`pta_laser.py`](pta_laser.py)). *The pair's shot noise is inside the budget's
photon row at the laser the receiver needs. And that laser is sixteen times
B5's: B5, at its end, and §4.3.*

*On the tile as revised, 128 × 64, the twenty is sixteen, 24 dB, and the
match a pair needs is 12% (§4.3).* The twenty times is this network's. Its first layer's weights have an rms of
0.14 of their range, with a few at the end of it, so the offset is large
beside them. A network whose weights filled a ring's range would make the two
readings closer, and no network makes the offset the quieter one.

**It is a working choice and not a detector.** Three things would reopen it: a
photonic die on which 512 photodiodes cost more than 26 dB of the source's
noise is worth; a source quiet enough, about −144 dB/Hz, that the offset would
do; or a pair that cannot be brought within about 10%. *Needed by:* P1, L3,
X1's rows and X3.

**B14 — The interface chip's budget: version 2.** *Settled 2026-10-06.*

X1 held the interface chip to version 1 (§4.3). On MNIST that costs a third of
a point on the working tile. On the two other data sets this plan has run it
costs 1.16 and 1.23 points (§8, question 7). grx930's harness then ran version
1 with each of its six rows made better, and found where the point is: on
Fashion-MNIST in three rows, the ADC's bit and the two noise rows, with the
other three free to hold where they are (§4.3, at the end of its budget).
Three budgets were put side by side: version 1; version 1 with those three
rows a notch tighter; and all six a notch tighter.

*Recommended, and settled as recommended:* **the interface chip is held to
version 2, which is version 1 with an 8-bit ADC, receiver noise within a
quarter of an 8-bit ADC's LSB, and 30 photons per such LSB.** The other three
rows stay version 1's.

| | Version 1 | **Version 2** | From |
|---|---|---|---|
| Activation DAC | 6 bits | 6 bits | |
| ADC | 7 bits | **8 bits** | |
| Receiver noise, rms | 0.5 LSB of an 8-bit ADC | **0.25 LSB of an 8-bit ADC** | |
| Light at each detector | 15 photons per such LSB | **30 photons per such LSB** | |
| Weight programming error, rms | 1 LSB of an 8-bit weight | 1 LSB of an 8-bit weight | |
| Crosstalk between neighbouring inputs | 2% | 2% | |
| Points lost on the working tile: MNIST | 0.34 ± 0.07 | 0.15 ± 0.02 | grx930's harness. **Measured in a model**, five networks a data set |
| Fashion-MNIST | 1.16 ± 0.14 | 0.54 ± 0.22 | The same |
| MNIST, inverted | 1.23 ± 0.13 | 0.62 ± 0.07 | The same |
| The 64 converters | 0.07–0.20 W | 0.27–0.34 W | The published survey: the best and the fifth-best part, of 70 and of 47 that reach the bits at 1 GS/s |
| The interface chip | 0.44–4.04 W | 0.64–4.18 W | Derived |
| A weight ring's line, at least | 1.99 GHz | 2.21 GHz | Derived: B11's bound, at a bit more |
| Its Q | 80,200 to 97,400 | 80,200 to 87,700 | Derived: between two buses and settling in a shot |
| Its swing | 2.27 to 2.76 V | 2.53 to 2.76 V | Derived, at 7.0 pm/V |
| Buses, lines each, photodiodes | 2, 64, 256 | 2, 64, 256 | Derived: a bus could hold 77, and 69 |
| Laser, the activation stage's shift a bit down: MNIST | 0.35–3.5 W | 0.7–7 W | B5's method, times grx930's multiple: 4 and 8 |
| Fashion-MNIST | 0.7–7 W | 1.4–14 W | 8 and 16 |
| MNIST, inverted | 1.4–14 W | 1.4–14 W | 16 and 16 |
| A MAC, every cell in use: MNIST | 97–922 fJ | 164–1,368 fJ | Derived |
| Fashion-MNIST | 140–1,351 fJ | 250–2,226 fJ | Derived |
| MNIST, inverted | 226–2,209 fJ | 250–2,226 fJ | Derived |

**What it buys.** About half of version 1's loss on every data set run. It
puts Fashion-MNIST and the inverted set at just over half a point, where
version 1 put them at over one.

**What it costs.** A seventh to a fifth of a watt of converters. Twice the
laser on MNIST and on Fashion-MNIST, and so most of a MAC's energy on the
harder set: 250 fJ at the low end where version 1's was 140. And a ring that
is harder to hit. Its Q has 9% of room where it had 21%, between a line too
narrow to settle to 8 bits in a nanosecond and one too wide for 64 lines to
share a bus.

**What it sets aside.** All six rows a notch tighter lose 0.06, 0.39 and 0.50.
That is the last fifth to a third of what tightening buys, and what it asks
beyond version 2 is a seventh bit of activation DAC and half the programming
error, which no model in this plan prices. It is not ruled out. It is not adopted
because its cost is not known. And two notches on all six, 0.03, 0.18 and
0.22, ask a ninth bit of ADC, at which a bus holds 63 lines and the tile
takes a third.

**What it does not settle, because it was not rerun.** Every figure in this
plan dated before this decision is version 1's unless it says otherwise.
[`pta_working_point.py`](pta_working_point.py) §4 restates the working point
under version 2. Not restated, and not run at it:

- ~~Drift, and how long a calibration holds.~~ *Run the same day (§4.3, at
  the end of its budget;* [`pta_version2.py`](pta_version2.py)*). Drift adds
  to version 2 what it added to version 1, and calibration returns version 2
  to its budget. So an hourly calibration takes back more than this decision
  bought on the two harder data sets, and how often version 2 is calibrated
  is open. Settled: B15, every six minutes.*
- ~~The source's three rows (B12, B13).~~ *Run the same day, and on all three
  data sets. They hold at version 2 on MNIST and on Fashion-MNIST. On MNIST
  inverted they add 0.37 of a point, 0.29 at version 1, and the shared row
  would have to be 1% and not 2%. Whether it should be is open. Settled: B16,
  1%.*
- Depth, and every tile but the working one.
- The two scorecards of §8's question 1, and the laser at the rule's shift.
  *The laser at the rule's shift was run at version 2 on 2026-10-07, on the
  working tile (§4.3, at the end of its budget): 8, 32 and 32 times, where a
  bit down it is 8, 16 and 16. The scorecards were not.*
- C3's and X3's measurements, which were made at version 1's settings.

**It is a working budget and not a specification.** Five things would reopen
it: a network trained with the tile's errors in the loop that does as well at
version 1, which is the usual remedy ~~and has not been tried~~ *(tried on
2026-10-06, below)*; a ring process
that cannot hold a Q within 9%; a converter of 8 effective bits at 1 GS/s
that cannot be had near the survey's price; a laser of 1.4 to 14 W that the
board cannot place (B5); or a price for the seventh activation bit and the
programming error that makes all six rows worth having. *Needed by:* X1's
rows, P0's links 3 and 4, P1, L3 and X3.

*On the fourth, 2026-10-07* (§4.3, at the end of its budget;
[`pta_reference_laser.py`](pta_reference_laser.py)). The laser was run on
B17's reference networks, and they need what the old ones need on the
workload that sizes it: 16 times, 1.4 to 14 W. Training does not buy laser.
What it leaves is a way back, if that laser cannot be placed. At 8 times, 0.7
to 7 W, the reference networks give up 0.01, 0.19 and 0.19 of a point, and
are still 0.29, 0.40 and 1.97 points ahead of the old networks at 16. And one
figure in the table above turns on a hundredth of a point: MNIST's 8 times at
version 2 is 4 if the rule is read exactly.

*The first of the five was tried on 2026-10-06* (§4.3, at the end of its
budget; [`pta_trained.py`](pta_trained.py)). As it is worded above, it is
met. A network trained with noise of the tile's size on its sums, on a v1
tile, against the network this decision was made on at version 2: 0.19 ± 0.10
ahead on MNIST, 0.17 ± 0.24 short on Fashion-MNIST, and 1.67 ± 0.46 ahead on
MNIST inverted. On two of the three that is not the noise. It is that the old
networks had been stopped early, and eight epochs with no noise put them
ahead by themselves; on Fashion-MNIST eight epochs alone leave a network
0.63 ± 0.16 short, and the noise closes it.

What the wording did not ask is what version 2 is worth to the trained
network, and that is what it was worth to the old one: 0.19 ± 0.04,
0.65 ± 0.15 and 0.44 ± 0.03 of a point, where this decision had 0.19, 0.62
and 0.61. Its costs above are a chip's and do not change with the network.
So the trade this decision made is the same trade on a better network: one
that starts 0.38, 0.48 and 2.12 points higher at version 2 than the old one
did. ~~Whether that reopens B14 is not decided here.~~ *Decided the same
day: it does not.* **B14 stands**, for the reason in the paragraph above, and
the networks that met its test became the reference (B17).

**B15 — How often version 2 is calibrated: every six minutes.** *Settled
2026-10-06.*

§4.3's table of requirements had "about hourly" for recalibration, from
version 1 on MNIST. grx930's harness then ran drift at version 2 (§4.3, at the
end of its budget). Drift adds to version 2 what it added to version 1, to
within a few hundredths of a point, so an hour of it adds 0.67 and 1.99 points
on Fashion-MNIST and MNIST inverted, where version 2 had bought 0.62 and 0.61.
An hourly calibration gives back what B14 was adopted for.

*Recommended, and settled as recommended:* **version 2 is calibrated every six
minutes, at TFLT's fitted drift.** It is the periodic scheduler's period, and
the floor under the two schedulers that predict
([`pta_chiplet_calibration.md`](pta_chiplet_calibration.md) §5).

| | About hourly | **Every six minutes** | From |
|---|---|---|---|
| What an interval's drift adds to version 2: MNIST | 0.12 ± 0.05 | −0.06 ± 0.04 | grx930's harness. **Measured in a model**, at TFLT's fit |
| Fashion-MNIST | 0.67 ± 0.31 | 0.08 ± 0.06 | The same |
| MNIST, inverted | 1.99 ± 1.26 | 0.28 ± 0.27 | The same |
| Shots between calibrations, at 1 GS/s | 3.6 × 10¹² | 3.6 × 10¹¹ | Counted |
| Calibrations a day | 24 | 240 | Counted |
| A calibration's probes | 4,096 shots, 4 µs | The same | The calibration note's §3: both banks, a row a shot, 16 probes a cell |
| Their share of the tile's shots | One in 880 million | One in 88 million | Derived |
| A whole calibration, counted (2026-10-08): the probes, the 96 zeroings before them over three passes, the estimator, the restore and the drain | 18 to 68 µs | The same | [`pta_interruption.py`](pta_interruption.py). **Counted, not measured** |
| Its share of the tile's time, at the most | One in 53 million | One in 5.3 million | Derived there |
| The period, as a count of shot-clock cycles | 42 bits | 39 bits | Derived. `PTA_CAL_PER` has 32 |
| The period, in `PTA_CAL_PER`'s units of 2¹⁶ cycles, since that day | 54,931,641 | 5,493,164 | The register map's §4. The word reaches 78 hours |

**What it buys.** On MNIST and Fashion-MNIST drift stays inside a tenth of a
point between calibrations, so version 2 keeps what B14 bought. With a source
at B16's rows as well, it loses 0.19 and 0.60 at the end of an interval.

**What it costs.** Ten times the calibrations. Their probes are nothing: a
part in 88 million of the tile's shots. The cost is the one the calibration
note names, the interruption: draining the tile, rewriting its weights and
restarting, 240 times a day. ~~No model in this plan prices that.~~ *One does
since 2026-10-08: at this decision's end.*

**What it shows up.** The period does not fit its register. `PTA_CAL_PER` is
32 bits of cycles, which at a shot a nanosecond is 4.3 seconds. Six minutes is
39 bits, and an hour was 42, so this was true before this decision and nobody
had written the interval down as a count. The register map widened its
counters for the same reason and did not widen this
([`pta_chiplet_regmap.md`](pta_chiplet_regmap.md) §7, item 11).

*Closed the same day.* Of the two ways to fix it, an upper half or a coarser
unit, the unit was chosen: on the chiplet `PTA_CAL_PER` counts 2¹⁶ cycles of
the shot clock (the register map's §4). Six minutes is 5,493,164 units, and
the word reaches 78 hours. The twin carries the unit and the conversion, and
its gate holds both. The map's version is unchanged, since nothing had been
built to a period in cycles.

*On the reference networks, 2026-10-07* (§4.3, at the end of its budget;
[`pta_reference_drift.py`](pta_reference_drift.py)). By the rule above the
interval is still six minutes. For B17's reference networks six minutes adds
0.01, 0.07 and 0.08 of a point on the three sets, which is under a tenth on
all of them in the mean and is not shown to be on the two harder ones, and a
quarter of an hour adds 0.46 on the inverted set. On the third thing that
would reopen this, an interruption that costs too much: half an hour, 48 a
day, costs the reference networks 0.03, 0.05 and 0.63 of a point held, and an
hour, 24 a day, 0.07, 0.15 and 1.30. And one thing it shows up: after a
calibration the reference networks are 0.09 and 0.13 of a point short of
their budget on MNIST and Fashion-MNIST, and no row here starts from a
calibration.

*In the cycle, 2026-10-07* (§4.3, at the end of its budget;
[`pta_reference_cycle.py`](pta_reference_cycle.py)). The row that starts from
a calibration has been run, and the interval is still six minutes. At the end
of a six-minute cycle the reference networks are 0.08, 0.16 and 0.09 of a
point over their budget. On the inverted set, which sized the interval, that
is within a tenth, and it is the interval's. ~~On Fashion-MNIST it is not, and
it is the calibration's: a calibration with no drift at all costs 0.15 ± 0.12
there and 0.08 ± 0.01 on MNIST, and a three-minute cycle ends 0.14 and 0.07
over. A shorter interval buys none of it back.~~ Two things are less sure than
this decision's table had them. The reference networks' six minutes on the
inverted set, drawn three times, is 0.08, 0.09 and 0.28. ~~And the budget an
interval is held to is a calibrated tile's, which for the reference networks
is 0.08 and 0.15 worse than the budgeted one on two sets.~~

*Corrected, 2026-10-08* (§4.3, at the end of its budget;
[`pta_reference_calibration.py`](pta_reference_calibration.py)). A calibration
costs nothing that can be seen. What the paragraph above called its cost was
the draws of the two runs it compared, and the budget an interval is held to
is the budgeted tile's after all. Asked whether to make that cost a row of the
budget or to find its cause first, the choice was to find the cause, and
there is nothing to make a row of. Read on its own draws, a six-minute cycle
ends 0.02, 0.02 and 0.11 of a point over the reference networks' budget, and
by the rule a calibration holds an hour, 30 minutes and 3 minutes. So
Fashion-MNIST is not the set where no interval holds, and the inverted set
sizes the interval again. Its six minutes, drawn three times and each over
its own draws, is 0.08, 0.11 and 0.30; a three-minute cycle ends 0.04 over.
Whether six minutes holds a tenth there is not settled.

*Over ten draws, 2026-10-08* (§4.3, at the end of its budget;
[`pta_reference_draws.py`](pta_reference_draws.py)). Asked whether to keep six
minutes, move to three or pin it first, the choice was to pin it first. On
the inverted set the reference networks' six-minute cycle ends 0.12 ± 0.03 of
a point over the tile as budgeted on the same draws, and a three-minute one
0.08 ± 0.02. By the rule above the interval that holds a tenth there is three
minutes, which is 480 calibrations a day, and six minutes costs 0.04 ± 0.04 of
a point more. Neither is far from the tenth: six is 0.6 of its errors over
it, three 1.3 under. On the other two sets six minutes holds, at −0.01 and
0.07. For the networks this decision was settled on, the old ones, a
six-minute cycle ends 0.41 ± 0.08 over on the inverted set and a three-minute
one 0.30 ± 0.06, so no interval that was run holds for them there, which this
decision knew of six minutes when it was taken. Whether the interval moves to
three minutes is this decision's to take again, and is not taken here.

*Kept, 2026-10-08.* Asked whether to keep six minutes or move to three, the
choice was to keep six minutes. It holds a tenth on MNIST and Fashion-MNIST,
it costs 0.12 ± 0.03 on the inverted set, which is not shown to be over a
tenth, and three minutes would buy 0.04 ± 0.04 of a point there. What twice
the interruptions would cost was not priced when this was kept.

*The interruption, priced the same day*
([`pta_interruption.py`](pta_interruption.py); the calibration note's §3 and
its open question 6). Counted from what the program already holds, and not
measured. The twin's own timing has a calibration hold the tile for `passes ×
repeats × (PTA_TW + rows × PTA_TS)` cycles a bank: each repeat writes the bank
to zero and shoots every row once. On the working tile a shot is a beat and a
bank programs in 64, since §4.3's write path is 128 cells a beat. So both
banks at 16 probes are 6.1 µs at one pass and 18.4 µs at the three passes
grx930's engine was measured at: 12,288 shots and 96 zeroings. What the
formula leaves out is the estimator's walk, the restore and the drain, and
with those it is 68 µs at the most, the estimator a cell a beat, or 19 µs
with it as wide as the write path.

- **240 a day is 16 milliseconds a day**, one part in 5.3 million of the
  tile's time at the most. This decision's table had the probes' shots alone.
- **For it to matter an interruption would have to last a third of a
  second.** 0.36 s takes a thousandth of the tile's time at 240 a day and 3.6 s
  a hundredth, which is 5,312 times the most counted. Nothing in these
  documents is that slow: a weight on TFLT settles in 25 ps, and the write
  path was sized to program a bank inside one batch.
- **So the count does not set the interval.** At three minutes the most
  counted is one part in 2.7 million and 33 ms a day, and every ten seconds
  one part in 147,545. It reaches a thousandth of the tile's time at an
  interval of 68 ms. What sets the interval is accuracy, which is measured at
  three minutes and at six and at no shorter interval.
- **A command that meets a calibration waits 68 µs at the most**, half of the
  131 µs the 4096-square layer takes on this tile, and one command in 5.3
  million meets one.

It prices time and nothing else. What the light does to a ring that was
written to zero and back is in no model here, the estimator's width is
specified nowhere, and the twin builds no scheduler, so on it a calibration
still starts when it is asked for. The interval stays six minutes: that was
kept before this was counted, and this does not change what it was kept on.
What it changes is the reason not to go shorter, which is no longer the
interruption.

*Kept again, 2026-10-08.* The count took away the reason first given for
not going shorter, so the question was put once more with that said. The
choice was six minutes again. It rests on accuracy alone: six minutes holds a
tenth on MNIST and Fashion-MNIST and costs 0.12 ± 0.03 on the inverted set,
and three would buy 0.04 ± 0.04 there.

*And a read of the comb's lines, the same day* (§4.3, at the end of its
budget). The level probe is 2.2 µs on one bank, beside a cell calibration's
18 to 68 µs. Taken at every calibration it adds 3% to 12% to an interruption
that is one part in 5.3 million of the tile's time, so it does not bear on
this interval. How often a comb's lines need reading is not known: it may be
far less often than six minutes, or more.

*With a source lit, and on the DAC as it is specified, 2026-10-09* (§4.3, at
the end of its budget; [`pta_reference_point.py`](pta_reference_point.py)).
Two things about the cycle this interval was kept on, found when B16's third
row moved. It had no source: the one cycle run with a source lit was on one
draw, and was not read over its own draws. With B16's rows lit, as the chip
is now run, the reference networks' six-minute cycle ends 0.07 ± 0.03, 0.04 ±
0.03 and 0.30 ± 0.05 over the tile as budgeted on the same draws: on the
inverted set drift has 0.14 of that on these five draws and the source 0.16.
And its trim was held in steps a quarter of what an 8-bit DAC holds. On the
DAC the calibration note asks for, the cycle with no source ends 0.01 ± 0.02,
0.03 ± 0.04 and 0.18 ± 0.04 over on these five draws, where at the finer
step they have −0.01, 0.05 and 0.14 and the ten draws this was kept on have
−0.01, 0.07 and 0.12. Five networks put the inverted set's 0.18 between 0.06
and 0.29, so it is still not shown to be over a tenth. The read of the lines
is now taken at every calibration, and with it an interruption is 70.0 µs at
the most. This interval is not changed here. What it was kept on, 0.12 ±
0.03 on the inverted set, is a figure of the finer trim, and which DAC it is
to be read on is B16's question.

**What it does not settle.** The inverted set, where six minutes adds 0.28 ±
0.27, and the interval that would hold a tenth there was not run. TFLN, whose
hour adds 2.4 to 37 points and whose interval is still "within minutes". A
ring's drift: every fit here is a Mach-Zehnder's bias (§8, question 1). Cells
that drift together, which the model does not have. And the predictive and
shadow schedulers, which were built to take fewer calibrations than the
period allows and were not rerun at version 2.

**It is a working interval and not a schedule.** Three things would reopen it:
a measured drift, of a ring, that is not TFLT's fit; an interruption that
costs enough for 240 a day to matter; or a workload brighter than
Fashion-MNIST that the board has to serve. *The second is counted since
2026-10-08 and is not met: an interruption is 18 to 68 µs, and it would have
to be a third of a second.* *Needed by:* X3, X4's `PTA_CAL_PER`, S3.

**B16 — The source's shared row: 1%.** *Settled 2026-10-06. Its third row
was moved on 2026-10-09: below.*

B12 and B13 gave the source three rows: its noise within 2% rms a shot for the
lines together, within 5% for a line on its own, and the lines level to 5%.
They were sized on MNIST at version 1, to cost a tenth of a point between
them. grx930's harness then ran them at version 2 and on two more data sets
(§4.3, at the end of its budget). They hold on MNIST and on Fashion-MNIST. On
MNIST inverted the three add 0.37 of a point to version 2, and the dear one is
the row the lines share: 2% adds 0.24 there, and 1% adds 0.13.

*Recommended, and settled as recommended:* **for version 2 the source's noise
is within 1% rms a shot for the lines together.** The other two rows stay: 5%
for a line on its own, and the lines level to 5%. *The last is 20%, read and
corrected, since 2026-10-09: below.*

| | At 2% | **At 1%** | From |
|---|---|---|---|
| As a density over the shot rate | −124 dB/Hz | −130 dB/Hz | [`pta_source_noise.py`](pta_source_noise.py) |
| For a receiver that averages over a shot | −121 dB/Hz | −127 dB/Hz | The same |
| Against what this plan first assumed, −144 dB/Hz | 20 dB easier | 14 dB easier | Derived |
| The three rows add, to version 2: MNIST | 0.06 ± 0.03 | 0.04 ± 0.04 | grx930's harness. **Measured in a model** |
| Fashion-MNIST | 0.09 ± 0.13 | 0.04 ± 0.11 | The same |
| MNIST, inverted | 0.37 ± 0.07 | 0.28 ± 0.06 | The same |

**What it buys, which is less than the row alone suggested.** 0.09 of a point
on the inverted set, and nothing that can be told from the scatter on the
other two. The shared row by itself went from 0.24 to 0.13 there. The three
together went from 0.37 to 0.28, because the two rows a line carries add 0.25
and 0.17 by themselves on that set and are now most of what a source costs.
So the source's rows are inside their tenth on MNIST and on Fashion-MNIST, as
they were, and still not on a workload that lights most of its rows.

**What it costs.** 6 dB of the comb's shared intensity noise. It is still 14
dB easier than the −144 dB/Hz §8's question 8 first held a source to.

**What it does not settle.** The two rows a line carries. On the inverted set
a line at 2% and the lines level to 2% add 0.13 and 0.01 alone, where at 5%
they add 0.25 and 0.17: tightening them was run a row at a time and not
together, and is not adopted. Nothing on the chiplet measures a line's level
(the ICD's link 4). And why the shared row is dear on that set is inferred
from its sums and was not measured shot by shot (§4.3).

*A line's level can be read since 2026-10-08* (§4.3, at the end of its
budget; [`pta_reference_level.py`](pta_reference_level.py)). The third row
was a requirement on a comb with nothing to meet it or to check it. grx930's
harness now has a probe that reads a line: one shot a row leaves a comb's
lines 3.6% off, rms, which is inside this row's 5%, and sixteen leave 0.9%,
whatever they were before, up to the 40% that was run. Left alone, lines 20%
off cost the reference networks 0.35, 0.40 and 1.63 of a point, so the row
matters. **The row is not changed.** It is still lines level to 5% at the
tile. What is open is who levels them: the comb as made, an equaliser in
front of it, or a correction from what the probe reads, and the last is not
modelled where a chip could apply it. On the inverted set, held, a read buys
the reference networks 0.18 ± 0.09 of a point at this row's 5%, which is not
clear.

*The correction, where a chip could apply it, the same day* (§4.3, at the end
of its budget; [`pta_reference_fix.py`](pta_reference_fix.py)). It was
modelled five ways. One of them holds on every set: the reading applied on
the weights a row at a time, and the weights written at the 8 bits the
calibration note puts behind a 6-bit weight code. With lines 20% off that
ends within 0.11 of a point of level lines. So there is now something that
could level a comb, and a figure for what it leaves:

| The third row | MNIST | Fashion-MNIST | MNIST, inverted | From |
|---|---|---|---|---|
| As it stands: lines level to 5% as they reach the tile, and left alone | +0.00 ± 0.01 | −0.04 ± 0.04 | +0.16 ± 0.08 | [`pta_reference_fix.py`](pta_reference_fix.py). **Measured in a model** |
| Lines level to 20%, read at a calibration and corrected on 8-bit weights | +0.05 ± 0.02 | +0.02 ± 0.03 | +0.10 ± 0.04 | The same |
| The second less the first, seed by seed | **+0.05 ± 0.01** | **+0.06 ± 0.01** | −0.05 ± 0.06 | Derived there |
| The same at the tile's 6 bits | **+0.06 ± 0.02** | **+0.17 ± 0.05** | +0.05 ± 0.17 | Derived there |

Points lost to level lines on the same draws, the reference networks.

**What moving the row would buy.** A comb four times less level as made: 0.8
dB where the row asks 0.2 dB, for half a tenth of a point on MNIST and
Fashion-MNIST and nothing that can be told on the inverted set. Nothing in
these documents says a comb can be had at 0.2 dB, or levelled to it.

**What it would cost.** A read of the lines at each calibration, 2.2 µs. The
weights rescaled at the host a row at a time when they are written. A weight
written at 8 bits where its code is 6: the DAC has them, the calibration
note gives them to the cell trim, and the write path would carry a third
more a cell. And the half a tenth.

**What is not known.** How a trim and a level correction share the DAC's two
bits below the code. How fast a comb's lines move. Whether a comb as made is
inside 20%: at 40% the model's own correction still held, and no place a chip
could apply one was run there. And it is a model, three draws of five
networks.

~~**The row is not changed here.** Whether it stays at 5% as the lines reach
the tile, or becomes 20% with the read and the correction beside it, is this
decision's to take again, and is not taken.~~

*Taken 2026-10-09.* Asked whether the row stays at 5% or moves, the choice
was to move it. **The source's third row is lines level to 20% as they reach
the tile, read at each calibration and corrected on the weights.** That is
0.8 dB where it was 0.2. Three things are part of the row and not beside it:
the lines are read with the level probe at every calibration, sixteen shots
a row; every row's weights are scaled by what its line read; and they are
written at the 8 bits of the weight DAC, behind the 6-bit code a network is
trained for. The other two rows stay: 1% rms a shot for the lines together,
and 5% for a line on its own.

grx930's harness then ran the chip as that holds it (§4.3, at the end of its
budget; [`pta_reference_point.py`](pta_reference_point.py)):

| The reference networks, five draws | MNIST | Fashion-MNIST | MNIST, inverted | From |
|---|---|---|---|---|
| Held as it was, lines 5% off and left alone: over the tile as budgeted | +0.00 ± 0.01 | +0.06 ± 0.06 | **+0.30 ± 0.06** | [`pta_reference_point.py`](pta_reference_point.py). **Measured in a model** |
| Held as it now is: lines 20% off, read, corrected on the weights at 8 bits | **+0.09 ± 0.02** | +0.06 ± 0.05 | **+0.27 ± 0.02** | The same |
| The second less the first, seed by seed | **+0.09 ± 0.03** | +0.01 ± 0.03 | −0.04 ± 0.07 | Derived there |
| Held with lines 20% off and left alone, less the first | **+0.40 ± 0.04** | **+0.41 ± 0.10** | **+1.47 ± 0.39** | The same |
| A six-minute cycle as it is now run: over the tile as budgeted on its draws | +0.07 ± 0.03 | +0.04 ± 0.03 | **+0.30 ± 0.05** | Measured |
| The same with the source as it was | +0.03 ± 0.03 | +0.05 ± 0.02 | **+0.32 ± 0.07** | The same |
| The first of those two less the second | +0.04 ± 0.02 | −0.01 ± 0.04 | −0.02 ± 0.07 | Derived there |

**What it cost, measured.** Held, 0.09 ± 0.03 of a point on MNIST, which is
clear and under a tenth, and nothing that can be told on the other two sets
or, at the end of a six-minute cycle, on any. What moving it was expected to
cost was half a tenth on MNIST and Fashion-MNIST.

**What it rests on.** The read and the correction: left alone, a comb 20% off
costs the reference networks 0.40, 0.41 and 1.47 of a point more than the
row as it was. And a read that six minutes of drift do nothing to: it leaves
the lines 1.0% off through a drifted tile as through a fresh one.

**What it puts on the chiplet.** A read of the lines at each calibration,
2.2 µs on one bank, which makes an interruption 70.0 µs at the most. Banks
that hold 8 bits a weight where they held 6, and a write path that carries
them; the link already carried a weight as a byte (§3.3). And a multiply a
weight, by its row's scale, which the model does at the host: whether the
host or the interface chip's write path does it is not chosen, and the
register map has no register to read a line's level from or to write a row's
scale to ([`pta_chiplet_regmap.md`](pta_chiplet_regmap.md) §7, item 12).
grx930's engine has no mode that takes the read.

**The DAC's bits, which the run turned up and this does not settle.** The
row writes a weight at the DAC's 8 bits, and the calibration note holds a
cell's trim on those same 8. Every cycle in this plan was run with a trim
two bits finer than that, because grx930's harness counts a trim's step in
LSB of an 8-bit weight and the note counts it in 6-bit codes. Of what was
not known above, the first now has an answer in a model: on the same 8 bits
a trim and a correction do nothing to each other that passes 0.07 of a
point. And on 8 bits a cycle as it is now run ends 0.05 ± 0.02, 0.09 ± 0.04
and 0.33 ± 0.06 over for the reference networks, where two bits finer it
ends 0.07, 0.04 and 0.30. **Whether the weight DAC stays at the calibration
note's 8 bits, or is asked for the 10 this plan's cycles were run on, is not
taken here.**

**What is still not known.** How fast a comb's lines move, which is what says
how often they have to be read: at each calibration is where the read was
put, and nothing was measured for it. Whether a comb as made is inside 20%.
And it is a model, five draws of five networks.

**Three things would reopen the third row:** a comb whose lines move by more
than the 1% a read leaves, inside the interval they are read at; a comb that
is further off than 20% as made; or a weight DAC that cannot hold 8 bits.

**It is a working row and not a source.** Three things would reopen it: a comb
that cannot be had at −130 dB/Hz; a workload brighter than Fashion-MNIST that
the board has to serve, which would ask for the other two rows as well; or a
network trained for the tile, which may not need it. *Needed by:* P0's link
4, P1, L3.

*Networks were trained for the tile on 2026-10-06* (§4.3, at the end of its
budget). They were held at this decision's 1% and not at 2%, so the third of
those is still untested.

*They were run at both on 2026-10-07* (§4.3, at the end of its budget;
[`pta_shared_row.py`](pta_shared_row.py)), which by then were B17's reference
networks. Held as this plan holds the chip, at the end of six minutes with a
line and the level at 5%, the 1% is worth −0.02 ± 0.02, −0.03 ± 0.04 and
−0.04 ± 0.04 of a point to them on the three sets, and under a tenth on each
at one in twenty. So the third of those is met: held, a network trained for
the tile does not need this decision. For the networks it was made on, the
record above stands, 0.09 ± 0.02 of a point on the inverted set, and 0.12 ±
0.06 held.

What that leaves to weigh. The 1% costs 6 dB of the comb's shared noise and
no comb has been chosen, so nothing here says it is dear. It buys the
reference networks nothing where the chip is held. And it still buys a tenth
of a point for a network that was not trained for the tile, on a workload that
lights most of its rows. ~~Whether B16 keeps the 1% is not decided here.~~

*Decided the same day:* **B16 keeps the 1%.** Nothing says it is dear, and it
is the only one of the source's rows that a network not trained for the tile
is known to need on a bright workload. What the run leaves behind is a way
back. If a comb cannot be had at −130 dB/Hz, which is the first of the three
things above, 2% and −124 dB/Hz cost the reference networks nothing where the
chip is held, and this plan can take that without another sweep.

**B17 — The networks the budget is run on: eight epochs, with noise of 10% on
their sums.** *Settled 2026-10-06.*

Every figure in this plan up to that day is of networks that grx930's trainer
stopped at the first epoch whose held-out accuracy failed to rise, which was
after two to seven, and that had never been shown a tile's errors. That day
the same networks were trained for eight epochs, with and without noise on
their sums, and run beside the old ones (§4.3, at the end of its budget;
[`pta_trained.py`](pta_trained.py)).

*Recommended, and settled as recommended:* **from 2026-10-06 the reference
networks are each seed's 6-bit network trained for eight epochs, with
Gaussian noise of 10% of a layer's rms on every sum the layer forms while it
trains. A sweep run from that day uses them. No sweep before it is run
again.**

| | MNIST | Fashion-MNIST | MNIST, inverted | From |
|---|---|---|---|---|
| Epochs, at the mean: as trained before | 3.0 | 3.8 | 4.2 | grx930's trainer, by its own rule |
| The reference networks | 8 | 8 | 8 | This decision |
| Right, percent, on its host: as trained before | 97.45 | 87.48 | 93.36 | grx930's harness. **Measured in a model** |
| The reference networks | 97.75 | 87.89 | 95.38 | The same |
| Held at version 2 (B14, B15, B16): as trained before | 97.26 | 86.88 | 92.27 | The same |
| The reference networks | **97.64** | **87.27** | **94.63** | The same |
| Those, over the old ones, seed by seed | +0.38 ± 0.11 | +0.39 ± 0.07 | +2.36 ± 0.65 | Derived, [`pta_trained.py`](pta_trained.py) |
| Points lost, held at version 2: as trained before | 0.19 ± 0.03 | 0.60 ± 0.23 | 1.09 ± 0.36 | grx930's harness |
| The reference networks | 0.11 ± 0.05 | 0.62 ± 0.13 | 0.76 ± 0.22 | The same |
| Held at version 2, over ten draws (2026-10-08): right, as trained before | 97.23 | 86.79 | 91.96 | grx930's harness, [`pta_reference_draws.py`](pta_reference_draws.py) |
| The reference networks | 97.59 | 87.30 | 94.65 | The same |
| Those, over the old ones, seed by seed | +0.36 ± 0.09 | +0.50 ± 0.12 | +2.68 ± 0.59 | Derived there |
| Points lost, over ten draws: as trained before | 0.22 ± 0.01 | 0.68 ± 0.11 | 1.40 ± 0.24 | The same |
| The reference networks | 0.17 ± 0.03 | 0.59 ± 0.09 | 0.74 ± 0.08 | The same |

**What it buys.** Where the chip is held, 0.4 of a point on MNIST and on
Fashion-MNIST and 2.4 points on the inverted set, each clear of chance with
five networks. *Over ten draws, 2026-10-08: 0.36, 0.50 and 2.68, each clear.* Most of the 2.4 is the epochs and not the noise. And an
inverted set whose accuracies are no longer two points low.

**What it costs.** Nothing on the chip. About twice the training. And
continuity: a sweep run from here is of other networks than the sweeps before
it, and its accuracies are not to be read against theirs.

**Why 10%, of what was run.** It is the best row held at version 2 on all
three sets, and it is the size B14's own test asked about. Over eight epochs
with no noise it is ahead there by 0.07, 0.22 and 0.11 of a point, and none of
those is clear, so between those two this is a working choice. 5% is not
shown to differ from none. 20% loses least to a tile and starts 0.4 and 1.4
points lower on its host on the two harder sets.

**What it does not settle.** Everything dated before this decision is the old
networks': which of version 1's rows carry its cost, the menu, drift by the
hour, the source's rows one at a time, the laser's multiples, depth, and
every tile but the working one. None of it had been run on the reference
networks when this was decided. *Four things have been since, on 2026-10-07
(§4.3, at the end of its budget): the source's rows; the laser, which is
8, 8 and 16 times for the old networks' 8, 16 and 16; drift by the
interval, which leaves B15's six minutes where it was; and the operating
cycle~~, in which a calibration by itself costs the reference networks 0.08
and 0.15 of a point on MNIST and Fashion-MNIST and the old networks nothing~~.
A fifth, on 2026-10-08, found that cost was the draws of the two runs the
cycle compared and not the calibration. And that over thirteen draws of the
tile as budgeted the reference networks lose 0.14, 0.54 and 0.48 of a point,
where one draw had 0.07, 0.47 and 0.52: on MNIST and Fashion-MNIST what the
networks trained before lose. They are right 0.29, 0.38 and 2.26 points more
often than those. A sixth, the same day, ran the chip as it is held on ten
draws, which are the rows added to the table above. What training for the
tile buys is in what holding adds: six minutes of drift and a source's rows
cost the reference networks 0.03, 0.06 and 0.25 of a point, and the old ones
0.08, 0.20 and 0.68.*
What was run on them is the working tile at v1 and at version 2, as budgeted
and as held. Over those four and the three sets, what the tile costs them in
points lost is smaller than it cost the old ones in 11 cells of 12, and in
none by more than 1.8 of its errors. So the prices B14, B15 and B16 were
chosen on are not shown to be wrong for these networks, and are not shown to
be right for them either. ~~Whether a trained network needs B16's 1% is still
untested.~~ *Run on 2026-10-07: held, it does not (B16, at its end).* And the
accuracies this plan quotes for the inverted set before this
decision stay about two points low.

**It is a working reference and not a network trained for the tile.** Three
things would reopen it: a trainer with the tile's own model in its forward
pass, where this one has Gaussian noise; a search over the epochs and the
noise, which were each chosen once; or a kind of network other than one
hidden layer on 28 × 28 images. *Needed by:* X1's sweeps from here on.

---

## 8. Open questions

1. **How big is the PTA chiplet, and how fast?** *Both halves were settled on
   2026-10-05, as working figures: B10, 128 × 64 after a revision the same
   day, and B11, 1 GS/s (§7). What follows is how the question was worked, and
   reads as it was written.* Its inputs, outputs and
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

   *A fourth bound, 2026-10-05:* the weight ring, which allows 0.44 to 2.2 GS/s
   for a ring that swings on 1 to 5 V. It is more than twenty times tighter
   than the modulator's, and it sits at the planning figure whatever the laser.
   The last block of this question.

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

   **As one table, 2026-10-05** ([`pta_geometry.py`](pta_geometry.py)). Six
   models price the tile, each sweeping the same candidates because none can
   fix one, and until now nothing put them side by side. This does, and adds
   nothing of its own: every figure is one of those models', asked for at each
   candidate, and held to the model it came from.

   | | 64 × 8 | 128 × 64 | 256 × 64 | 256 × 128 | What kind of number |
   |---|---|---|---|---|---|
   | Lines between the dies | 584 | 8,384 | 16,704 | 33,152 | Counted |
   | Cells, inputs and link, mm² | 7.5 | 20.3 | 39.7 | 54.4 | A lower bound, at an assumed 25 µm cell |
   | Link modules | 1 | 3 | 5 | 10 | Predicted, for a 4096-square layer at batch 64 |
   | Interface chip, W | 0.07–1.10 | 0.44–4.04 | 0.55–7.61 | 1.02–11.46 | From published parts |
   | Laser, W | 0.01–0.11 | 0.09–0.88 | 0.09–0.88 | 0.18–1.76 | B5's method, at a measured receiver, 10 to 20 dB of loss |
   | A MAC, every cell in use, fJ | 152–2,372 | 65–600 | 39–518 | 37–403 | The two rows above over the rate |
   | The 4096-square layer | 2.1 ms | 131 µs | 65.7 µs | 32.9 µs | Predicted, at that row's modules |
   | v1 on D3, points lost | 0.33 ± 0.11 | 0.34 ± 0.07 | 0.20 ± 0.05 | 0.23 ± 0.05 | Measured in a model, one network |
   | The same receiver noise, on a layer | 0.73 or 1.46 | 1.07 | 0.81 | 0.81 | Against the 8 × 8 tile's |
   | An hour of drift, on a layer | 10.4% | 7.6% | 6.9% | 7.2% | In a model whose cells drift independently |

   Five things follow.

   - **The choice is between 128 × 64 and 256 × 64.** 64 × 8 is 32 times slower
     than 256 × 64 at four times the energy a MAC. 256 × 128 doubles the
     detectors, the laser and the modules for 0.78 to 0.93 of the energy, and
     its ten modules are a third of the way to where B8 reopens.
   - **What 256 × 64 costs over 128 × 64**: twice the lines between the dies,
     twice the cells' area, five modules for three, and 0.11 to 3.6 W. The laser
     is the same, because it follows the columns.
   - **Accuracy does not choose between them.** v1's two losses are a standard
     error and a half apart, and drift treats them alike. What leans to 256 is
     the receiver: 128 inputs need the ADC shift 256 need and take nearly twice
     the conversions. That lean is the harness's whole-bit shift, and a receiver
     whose gain steps are finer than a factor of two would not have it.
   - **So it is the workload that chooses, and it has a number.** Energy a MAC
     the layer asked for, on square layers at a batch of 64:

     | Layer | 128 × 64 | 256 × 64 | The cheaper |
     |---|---|---|---|
     | 64 wide | 462–4,276 fJ | 557–7,385 | 128 × 64, at both ends of the power range |
     | 128 wide | 148–1,369 | 178–2,365 | 128 × 64, at both ends |
     | 256 wide | 86–792 | 64–850 | One end each |
     | 512 wide | 70–648 | 45–601 | 256 × 64, at both ends |
     | 4096 wide | 65–601 | 39–520 | 256 × 64, at both ends |

     A tile pays for the cells a narrower layer leaves unused. D3 uses 54% of
     what a 256 × 64 tile programs for it and 65% of the smaller tile's, and
     costs 0.61 to 8.1 µJ a batch on the one and 0.68 to 6.3 on the other: no
     difference that can be told.
   - **The plan does not hold a workload, and that is what this question was
     waiting on.** It sizes its link, its power and its time on a 4096-square
     layer nobody has trained, and measures accuracy on a 784-100-10 network
     that fills a fraction of either tile. Layers 512 wide and up pay for
     256 × 64. Layers 128 wide and under do not. The question the board team
     can answer is which the board is for. *Answered the same day: wide layers,
     and 256 × 64 as the working geometry (B10, §7).*

   It does not weigh a watt against a tenth of a point: that is the program's
   to say, and a score with invented weights would hide the choice in a number.
   One thing it found in passing. S3 priced the chiplet on one link module,
   where X2 gives a 256 × 64 tile five. At five the 4096-square layer is 65.7 µs
   and bound by the tile, not 296 µs and bound by the link, and D3's two layers
   are 0.96 µs and not 3.74. S3's readings stand, and the one that says the
   command is the cost of a small layer stands harder.

   **What a published ring makes of the cell, 2026-10-05**
   ([`pta_ring.py`](pta_ring.py)). The floorplan sized a resonant cell at 25 µm,
   which is an assumption in a patent application for another topology, and
   said three figures were missing behind it: the ring's size, its linewidth,
   and the swing that moves it by one. Five published devices now stand in
   those places. None is a weight cell, and nobody here has laid out a ring.
   Nor are five papers a survey: "the smallest" below is the smallest of these.

   | Published ring | Radius | Cell, rings touching | 256 × 64, weights | With inputs and link | 128 × 64, the same |
   |---|---|---|---|---|---|
   | Thin-film lithium niobate, the smallest read (Krasnokutska et al., arXiv:1807.06531) | 30 µm | 60 µm | 59 mm² | 124 mm² | 62 mm² |
   | The ring that paper tuned, at 3 pm/V | 70 µm | 140 µm | 321 mm² | 466 mm² | 233 mm² |
   | The highest Q, 5 million loaded (Zhang et al., arXiv:1712.04479) | 80 µm | 160 µm | 419 mm² | 584 mm² | 293 mm² |
   | Thin-film lithium tantalate, a racetrack (Wang et al., arXiv:2306.16492) | 100 µm apex | 200 × 600 µm | 1,966 mm² | | 983 mm² |

   Every area is a floor: electrodes, a bus and a gap to the next ring all add
   to a cell and none is added. Five things follow.

   - **A cell is 60 to 160 µm, not 25**, so the floorplan's area for the weights
     was 6 to 41 times too small on a ring, and 192 times on lithium tantalate's
     racetrack. Its other reading stands the better for it: a pad a cell at
     flip-chip's 100 µm is enough, and the finest bond is not needed.
   - **None of these rings puts 256 × 64 on a standard die.** The ring that
     would is 26 µm in radius, 4 µm under the smallest read, and that with
     nothing added to a cell. On the plan's own
     platform, lithium tantalate, the one resonator read with dimensions is
     a racetrack, and a tile of those is 20 standard dies of weights alone. That
     bears on B10 (§7).
   - **The swing is in range.** A linewidth of a published racetrack is 4.4 V at
     the 7.0 pm/V measured on it (Wang et al., arXiv:1701.06470), so the
     floorplan's 5 V is a published device's. 2 V and 1 V ask a Q of 111,000 and
     221,000. A Q far past that is published, and not on a small ring with
     electrodes.
   - **Low voltage is paid for in stability, one for one.** A 6-bit weight's LSB
     is 0.42 pm of resonance on a 5 V ring and 0.084 on a 1 V one. The one
     figure read, on lithium tantalate, is 0.1 pm in 25 minutes (Sayem et
     al., arXiv:2602.00922): a quarter of an LSB on the one and more than one on
     the other. It is one device for 25 minutes, and says nothing about
     temperature.
   - **The drift this program fits is not a ring's.** TFLT's and TFLN's fits are
     a Mach-Zehnder's bias drifting. A ring turns the same change of index into
     a weight error in proportion to its Q, and nothing has priced that. §4.3's
     recalibration row, X3's intervals and the geometry sweep's drift row are a
     modulator's until something does.

   The lines a bus carries are question 8's, and are there.

   **The second half as one table, 2026-10-05** ([`pta_rate.py`](pta_rate.py)).
   B10 left the shot rate at X2's 1 GS/s as a planning figure. Six models have a
   say in the rate, and none had been run at more than one. This asks each of
   them at five rates, at the working geometry, a batch of 64 and §4.3's
   version 1. One part of it is new, and it is a fourth bound on the rate.

   | | 0.25 GS/s | 0.5 GS/s | 1 GS/s | 2 GS/s | 4 GS/s | What kind of number |
   |---|---|---|---|---|---|---|
   | Link modules, weights re-sent | 2 | 3 | 5 | 10 | 19 | Predicted, for a 4096-square layer |
   | The same, weights held on the interface chip | 1 | 1 | 1 | 1 | 2 | Predicted |
   | A conversion, pJ | 1.11–2.42 | 1.11–2.55 | 1.11–3.14 | 1.11–4.54 | 1.60–11.0 | Published: the best and the fifth-best part |
   | Receiver bandwidth it needs, GHz | 0.22 | 0.44 | 0.88 | 1.77 | 3.53 | Against the 1.5 of the one measured |
   | Interface chip, W | 0.33–2.09 | 0.41–3.92 | 0.55–7.61 | 0.86–15.1 | 1.68–34.6 | From published parts |
   | Laser, W | 0.04–0.44 | 0.06–0.62 | 0.09–0.88 | 0.12–1.46 | 0.18–4.13 | B5's method at the measured receiver; an extrapolation past 1.7 GS/s |
   | A MAC, every cell in use, fJ | 92–617 | 57–554 | 39–518 | 30–507 | 28–591 | The two rows above over the rate |
   | The 4096-square layer | 263 µs | 131 µs | 65.7 µs | 32.9 µs | 16.5 µs | Predicted, at that rate's modules |
   | **The weight ring's Q, at most** | 390,000 | 195,000 | 97,000 | 49,000 | 24,000 | Derived, and new here |
   | Its swing, at least | 0.57 V | 1.14 V | 2.27 V | 4.55 V | 9.09 V | Derived, at 7.0 pm/V |
   | A weight's LSB, as a shift of the resonance | 0.048 pm | 0.096 pm | 0.19 pm | 0.38 pm | 0.77 pm | Derived |
   | Lines a bus, and buses for 256 inputs | 409, 1 | 204, 2 | 102, 3 | 51, 6 | 25, 11 | Derived, on the smallest ring read |

   **The fourth bound.** The 2026-10-03 block above bounded the rate by the
   modulator, the feed and the receiver, and had no weight cell to ask. The
   cells are rings now, and the ring block above compared a ring's line with
   how often a weight is *rewritten*, 15.6 MHz, which every line clears fifty
   times over. That was the wrong clock. The light a ring weighs carries an
   input, an input changes every shot, and a resonance passes a level no faster
   than its line is wide: its field rings down in 1/(π × linewidth), and the
   detector reads the square of it. So a ring that swings on 2 V carries
   0.88 GS/s, one on 1 V 0.44, and one on 5 V 2.2. The modulator's 51 GS/s was
   never the question. This is, and it sits at the planning figure. It holds
   for any resonant cell, whether or not the tile is a ring bank.

   Six things follow.

   - **1 GS/s needs a weight ring with a Q of 97,000 or less and a swing of
     2.3 V or more.** Of the floorplan's three swings, a ring carries the
     planning figure only at 5 V. A published racetrack with a Q of 50,000
     carries 1.9 GS/s. The ring with the highest Q read, 5 million, carries
     20 MS/s.
   - **Rate, swing, stability and buses are one number, the ring's Q.** A
     faster tile needs a wider line: more volts to cross it, fewer lines a bus,
     and a resonance that may wander further before it costs a weight an LSB.
     Lines a bus times the shot rate is a constant of the ring, 102 lines·GS/s
     on the smallest ring read. So the faster tile is the harder to drive and
     to light, and the easier to hold still: the one stability figure read,
     0.1 pm in 25 minutes, is half an LSB at 1 GS/s and a whole one at 0.5.
   - **The measured parts reach 1.7 and 2.7 GS/s, and no further.** The one
     receiver with a measurement settles a tile up to 1.7 GS/s, and the
     converter that sets the best price runs at 2.7. So 1 GS/s is the fastest
     candidate both stand behind. 2 GS/s is past the receiver, and 4 past both.
   - **A MAC gets cheaper with the rate, and most of that is had by 1 GS/s.**
     The receivers draw the same at every rate and the laser grows as its
     square root, so more shots spread them. From 1 to 2 GS/s a MAC is 23%
     cheaper at one end of the power range and 2% at the other. At 4 GS/s the
     high end is dearer than at any rate but the slowest.
   - **The rate buys time only with its modules, and time is not what is
     short.** On one module the wide layer takes 296 µs at every rate. At a
     quarter of the planning rate the tile is still a thousand times the GPU's
     arithmetic on that layer, by S3's indication of the GPU. And D3's two
     layers are 0.4 to 3 µs of arithmetic at any of these rates, under a
     command of 14.7 µs or more. The rate decides what the speed costs.
   - **So the choice is among 0.5, 1 and 2 GS/s.** 1 GS/s is the fastest rate
     every row can stand behind, and it is not free: it fixes the weight ring
     at a Q under 97,000, a swing over 2.3 V and three buses or more.
     0.5 GS/s takes three modules for five and lets a ring swing on 1.1 V, for
     twice the time and a ring twice as hard to hold. 2 GS/s halves the time,
     for twice the modules, six buses and a receiver nobody has measured.
     *Answered the same day: 1 GS/s, as the working rate (B11, §7).*

   It does not weigh a watt against a microsecond, and it does not choose: that
   is the program's. Every ring figure rests on [`pta_ring.py`](pta_ring.py)'s
   five papers and its assumptions, 7.0 pm/V carried from one racetrack to any
   ring among them.

   **The laser rows of both tables are low, 2026-10-05**
   ([`pta_laser.py`](pta_laser.py); B5, at its end). They are B5's method, which
   takes a converter's full scale for all the light a column is sent. On D3 it
   is an eighth of it at 256 rows and a quarter at 128, and less on the second
   layer. With the laser at what grx930's harness says each tile needs to stay
   within a tenth of a point of version 1:

   | | 128 × 64 | 256 × 64 | As the tables have it |
   |---|---|---|---|
   | Laser, W | 0.7–7.0 | 1.4–14 | 0.09–0.88 for both |
   | A MAC, every cell in use, fJ | 140–1,351 | 119–1,323 | 65–600 and 39–518 |

   Two readings above change with it. The geometry table's "the laser is the
   same, because it follows the columns" does not hold: it follows the rows
   too. And its energy case for the larger tile is gone, since a MAC now costs
   the same on both. The shot-rate table's laser row scales as it did, as the
   root of the rate, from sixteen times as much. Neither table has been rerun
   at other tiles or rates with this laser, and the rows stay as they were
   run.

   **Both tables again, at the corrected laser, 2026-10-05**
   ([`pta_working_point.py`](pta_working_point.py)). The note above said
   neither had been rerun. This is them. First the candidates, at 1 GS/s:

   | | 64 × 8 | 128 × 64 | 256 × 64 | 256 × 128 |
   |---|---|---|---|---|
   | The laser, as a multiple of B5's | 8, derived | 8 | 16 | 16, derived |
   | Laser, W | 0.1–0.9 | 0.7–7.0 | 1.4–14 | 2.8–28 |
   | A MAC, every cell in use, fJ | 302–3,873 | 140–1,351 | 120–1,323 | 117–1,208 |
   | The die at the smallest ring read, mm² | 18 | 62 | 124 | 187 |

   The multiple is grx930's where it ran the tile. Elsewhere it is the one
   that leaves the receiver what those two were left, a quarter of an LSB on
   the first layer, and nobody ran it. A larger tile buys little of a MAC with
   this laser: 256 × 64 is 15% cheaper than 128 × 64 at one end and 2% at the
   other, where the first table had 40% and 14%. It buys time, and pays in
   light, in area and in link. B10 was revised on it (§7).

   And the rates, at 128 × 64:

   | | 0.25 GS/s | 0.5 GS/s | 1 GS/s | 2 GS/s | 4 GS/s |
   |---|---|---|---|---|---|
   | Link modules, weights re-sent | 1 | 2 | 3 | 5 | 10 |
   | Interface chip, W | 0.31–1.19 | 0.35–2.13 | 0.44–4.04 | 0.63–7.99 | 1.18–18.9 |
   | Laser, W | 0.4–3.5 | 0.5–5.0 | 0.7–7.0 | 1.0–11.7 | 1.4–33.1 |
   | A MAC, fJ | 322–2,299 | 207–1,734 | 140–1,351 | 99–1,202 | 79–1,585 |
   | The 4096-square layer | 525 µs | 262 µs | 131 µs | 65.7 µs | 32.9 µs |
   | Buses for 128 inputs | 1 | 1 | 2 | 4 | 7 |

   The laser is the larger part of the power at every rate now, and a MAC
   still gets cheaper with the rate up to 2 GS/s. B11's reason stands: 1 GS/s
   is the fastest rate that both measured parts reach, and neither has moved.
   One thing is new to this tile: at 0.5 GS/s one bus would carry all 128
   inputs, where 1 GS/s takes two.

   Both tables are at the hidden layer's rescale as grx930's harness sets it.
   One bit under that, the working tile's laser is half of what they show,
   0.35–3.5 W at 1 GS/s, and a MAC 97–922 fJ (B5, at its end). That was run at
   128 × 64 alone, so no other column moves here.

   And both tables are MNIST's. On the two other data sets since run, the
   working tile's laser with the rescale a bit down is 0.7–7 W and 1.4–14 W,
   and a MAC 140–1,351 and 226–2,209 fJ (question 7). The first of those is
   what the tables show for this tile, and the second is twice it.
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

   *Another data set was run on 2026-10-05, and the answer is: no.*
   ([`pta_workload.py`](pta_workload.py); grx930's design note, §5, "Does the
   budget hold on another workload?") Two went through grx930's harness, with
   its own trainer and nothing retuned. Fashion-MNIST is MNIST's size and
   shape, and harder. MNIST with every pixel inverted was meant as a control
   for light, and is not a clean one, because the trainer does worse on it.

   | | MNIST | Fashion-MNIST | MNIST, inverted |
   |---|---|---|---|
   | The networks, on their host | 97.2–97.8% | 86.9–88.0% | 91.3–94.5% |
   | Rows of the working tile a shot lights on layer 1, at the mean | 12% | 25% | 74% |
   | Version 1, points lost: the core's tile | 0.26 ± 0.06 | 1.06 ± 0.26 | 1.10 ± 0.22 |
   | The working tile | 0.34 ± 0.07 | 1.16 ± 0.14 | 1.23 ± 0.13 |
   | Version 0, the working tile | 2.17 ± 0.12 | 5.27 ± 0.43 | 8.57 ± 0.63 |
   | Error at the outputs under version 1, of their rms | 7.9% | 7.8% | 12.4% |
   | The median image's lead, in multiples of that error | 7.8 | 3.6 | 5.4 |
   | An hour of TFLT's drift, over version 1 | +0.17 | +0.68 | +1.95 |
   | Four hours | +0.40 | +2.34 | +16.5 |
   | An hour, then calibrated | −0.13 | +0.02 | −0.09 |
   | The laser, the rescale a bit down (B5, at its end) | 4 times B5's | 8 times | 16 times |

   Six things.

   - **The budget does not hold.** Version 1 costs three and a half times
     MNIST's on both, five and six standard errors clear on the working tile.
   - **On Fashion-MNIST the tile is no worse.** Its outputs come back 7.8%
     wrong where MNIST's come back 7.9%. The network is right 87% of the time
     and its answers are closer together: the median image leads by 3.6 errors
     where MNIST's leads by 7.8. The same error turns more of them.
   - **The menu does not hold either** (§4.3, at the end of its budget). Every
     row costs two to seven times as much to relax. The dearest is the
     receiver's noise on one and programming error on the other, where on
     MNIST it was the light.
   - **Drift is dearer, and on a bright workload far dearer.** An hour costs
     four times what it cost on MNIST on Fashion-MNIST, and over eleven times
     on the inverted set. Calibration returns all three to version 1. The
     intervals of [`pta_chiplet_calibration.md`](pta_chiplet_calibration.md)
     were priced on MNIST.
   - **The laser is twice and four times MNIST's** (B5, at its end).
   - **The activation stage's shift still gives half of it back, and writing
     weights larger is worse than useless.** On the inverted set a gain that
     saturates 0.15% of the first layer's weights costs three points.

   What this does not say is what to do about it. No network here was trained
   with the tile's errors in the loop, which is the usual remedy and may give
   much of the point back. Nothing was run to say what tightening which rows
   would buy: inside version 1, both noise rows halved again buy a fifth of a
   point on either set, so most of the rest is in the converters, the
   programming error and the crosstalk. And both sets are still 28 × 28 images
   through fully connected layers. So version 1 stays the requirement on the
   interface chip, as the tightest set of rows this plan has. What is
   withdrawn is its price: "a quarter of a point" is MNIST's.

   *What tightening buys was run on 2026-10-06* (§4.3, at the end of its
   budget; [`pta_tighten.py`](pta_tighten.py)). On Fashion-MNIST version 1's
   point is in three of its six rows, the ADC's bit and the two noise rows.
   Version 1 with an 8-bit ADC and both noise rows halved loses 0.15, 0.54 and
   0.62 on the three sets, and all six rows a notch tighter 0.06, 0.39 and
   0.50. The first costs a seventh to a fifth of a watt of converters, a ring
   whose Q has 9% of room where it had 21%, and twice the laser on two of the
   three sets. Whether the interface chip is held to either is not decided
   here. *It was, the same day: B14 (§7) holds it to the first, as version 2.*

   *Networks were trained for the tile on 2026-10-06* (§4.3, at the end of
   its budget; [`pta_trained.py`](pta_trained.py)). The remedy this answer
   twice called usual gives back under half of Fashion-MNIST's point and
   nothing that can be told elsewhere: noise of the tile's size on a
   network's sums while it trains is worth 0.46 ± 0.18 of a point at v1
   there, 0.24 ± 0.28 on the inverted set and nothing on MNIST. Something
   this answer did not look for gives back more. The trainer's rule had
   stopped every network after two to seven epochs, and at eight the inverted
   set's are two points better: the table above has them at 91.3–94.5% on
   their host, and they are at 94.3–96.4%. Neither changes what a tile costs
   in points lost, so the budget's prices stand. And version 2 is worth to a
   trained network what it was worth to the others. *B14 was kept the same
   day, and those networks were made the reference (B17, §7).*
8. **What kind of light source does the tile need?** *Settled on 2026-10-05, as
   a working topology: a ring bank, lit by a comb (B12, §7). What follows is
   how the question was worked, and reads as it was written.* Asked on
   2026-10-03, about a
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

   *One bank does not tell 256 apart, 2026-10-05*
   ([`pta_ring.py`](pta_ring.py)). The question above was whether one bank can,
   and published rings answer it, for a ring bank. The error model's crosstalk
   is a neighbour's ring seen from a line away, and a resonance is a Lorentzian,
   so §4.3's two crosstalk rows are distances between lines: v1's 2% is 3.5
   linewidths and v0's 10% is 1.5. A ring's free spectral range over that
   distance is how many lines one bus holds.

   | Ring | Finesse | Lines a bus at v1 | Buses for 256 |
   |---|---|---|---|
   | 30 µm, a 5 V swing | 163 | 46 | 6 |
   | 30 µm, a 2 V swing | 407 | 116 | 3 |
   | 30 µm, the best its bend loss allows | 503 | 143 | 2 |
   | 70 µm, a 5 V swing | 71 | 20 | 13 |

   Three things follow.

   - **256 inputs are two to six buses**, on the smallest ring read. A
     larger ring holds fewer lines, because its lines are as wide and its range
     is shorter. So a ring bank's source is tens of lines, reused across buses:
     neither B5's one laser nor one line an input.
   - **The patent application's "k ≲ 56" has a derivation now**, where it gave
     none: it is what a 30 µm ring holds at v1's crosstalk with a Q of 53,000,
     which is a 4.2 V swing. One derivation, and not necessarily theirs.
   - **The voltage, the stability and the source are one trade.** A lower swing
     is a narrower line: more lines a bus and fewer buses, and a resonance that
     has to hold proportionally stiller (question 1).

   What is still open is the first half of the question: whether the tile is a
   ring bank at all. Area says its cells are rings. Nothing has said that a
   column sums powers and not fields.

   *And the shot rate is in the same trade, 2026-10-05*
   ([`pta_rate.py`](pta_rate.py); question 1). The table above took a ring's Q
   as free. It is not: the light through a ring carries an input, and a ring
   passes a level no faster than its line is wide. So lines a bus times the
   shot rate is a constant of the ring, 102 lines·GS/s on the 30 µm one at v1's
   crosstalk. At 1 GS/s a bus holds 102 lines at most. The row above that put
   256 inputs on two buses needs a ring that carries only 0.71 GS/s. So at the
   planning rate 256 inputs are three to six buses and not two to six, and one
   bus carries them all only at 0.40 GS/s and under.

   **Each reading, followed to its source, 2026-10-05**
   ([`pta_source.py`](pta_source.py)). The tile is settled now, 256 × 64 at
   1 GS/s (B10, B11), and its cells are rings. That is enough to say what each
   reading of the topology asks for, which is what this question has waited on.

   | Reading | A column adds | Its source | What the tile needs with it |
   |---|---|---|---|
   | A ring bank | Powers, of lines told apart by wavelength | A comb: 43 to 86 lines, 8 to 16 GHz apart, inside 5.7 nm | Three buses or more, and a photodiode a bus a column |
   | One line, added as fields | Fields | One laser | A second element in every cell, to undo the angle its ring turns the field by |
   | One line, a photodiode a cell | Currents | One laser | 16,384 photodiodes, 256 of them on each receiver's input |

   *The ring bank's limit is published, and it is the one derived above.* Tait
   et al., "Microring Weight Banks" (IEEE JSTQE 22(6), 2016), analyse the
   topology the error model assumes. A bank holds its rings' finesse over the
   channel spacing in linewidths, and at a 3 dB penalty that spacing is 3.41 to
   4.61 linewidths, by the length of the bus. The 3.5 this plan got from v1's
   crosstalk is inside that range and at its kind end. At the other end a bus
   holds 77 lines at the working rate and not 102, and 256 inputs are four
   buses and not three. The paper also reads each bank with a balanced pair of
   photodiodes, for signed weights, and assumes the lines far enough apart that
   they do not beat within the signal's band. This plan had neither.

   A ring bank at the working point, by its buses. The grid is the widest whole
   multiple of the shot rate that still fits a bus's share of the inputs in one
   of the ring's ranges:

   | Buses | Lines each | Grid | The weight ring's Q | Its swing | At the paper's worst spacing | Each line | Photodiodes |
   |---|---|---|---|---|---|---|---|
   | 3 | 86 | 8 GHz | 85,000–97,000 | 2.3–2.6 V | No ring does both | 1.0–10 mW | 192 |
   | 4 | 64 | 11 GHz | 62,000–97,000 | 2.3–3.6 V | A Q from 81,000 | 1.4–14 mW | 256 |
   | 5 | 52 | 13 GHz | 52,000–97,000 | 2.3–4.3 V | From 69,000 | 1.7–17 mW | 320 |
   | 6 | 43 | 16 GHz | 42,000–97,000 | 2.3–5.2 V | From 56,000 | 2.0–20 mW | 384 |

   Seven things follow.

   - **A ring bank here is three buses at the least, and four is the first that
     holds across the published range.** The upper end of every Q window is
     B11's rate and the lower end is the grid. Three buses leave a ring 15% of
     room at the kind spacing and none at the worst.
   - **So its source is a comb, and no announced part is one.** The sources of
     8 and 16 lines this question was first asked about are 32 and 16 buses.
     The telecom grid (ITU-T G.694.1) is 37, 19, 10 and 5 buses at 100, 50, 25
     and 12.5 GHz. An electro-optic comb has been published at about 10 GHz on
     lithium niobate (Zhang et al., arXiv:1809.08636): 71 lines in this ring's
     range, four buses. The one on the plan's own material (Zhang et al.,
     *Nature* 637, 2025) is about 30 GHz, by its span over its line count:
     twelve.
   - **The beat between lines is not out of band by itself.** Lines on one
     photodiode beat at their spacing. The power model's receiver, a single
     pole, passes 8% of a beat 11 GHz away, and half an LSB is 0.4%. A receiver
     that averages over a shot passes none of it when the spacing is a whole
     number of shot rates, and 0.45% when it is 50 MHz off. An electro-optic
     comb's spacing is a microwave drive, so it can be locked to the shot
     clock. An array's lines are each their own laser. The other way out is a
     receiver steeper than one pole, which nothing here has priced.
   - **Buses that reuse lines cannot share a photodiode.** The same line on two
     buses is the same light, and brought together in one waveguide it adds as
     fields. So a column reads each bus on its own photodiode: 256 at four
     buses, where the laser, the power and the area were all sized on 64.
   - **A line carries milliwatts, and a comb comes with an amplifier.** The
     tile's light is the same 0.09 to 0.88 W however it is cut: 0.34 to 3.4 mW
     an input. None of the combs read states a line's power, and the most
     efficient of them (Hu et al., arXiv:2111.14743) turns 30% of its pump into lines
     over 23 times the span the tile can use. The amplifier's noise is then the
     source's, which is the term §4.3's error model still does not have.
   - **A weight's LSB is 24 MHz of a line's position.** That is how far a line
     may sit from where its ring was calibrated. A comb's lines move together,
     on two numbers: where its pump sits and how far apart they are. An
     array's move one by one.
   - **One laser is still possible, and costs a device a cell either way.** A
     ring that passes half a line's power turns its field by 45°, and by up to
     90° as the weight falls, so a column that added fields would add them at
     angles: a second element a cell to undo it, and one that is not resonant
     is millimeters long. A column that adds currents needs a photodiode a
     cell. The ring bank is the one reading whose cell is a ring and nothing
     else, and the one the error model and every bus count here were written
     for.

   B5's placement survives the first reading: a comb is many lines on one
   fiber. Its sizing does not, quite: one laser becomes a pump, a comb and an
   amplifier.

   **What is still open is which reading the tile is, and that is now a choice
   with a price on each side** and not a gap. It is the program's to make. Not
   priced in any of them: the amplifier and its noise, whatever picks one line
   off a comb for one input, what puts a ring on its line in the first place,
   how the tile signs a weight, and a receiver with more than one photodiode on
   its input. *Answered the same day: a ring bank, on four buses as the working
   count (B12, §7); two buses at the 128 inputs of B10's revision.*

   *And the noise row has been measured, 2026-10-05*
   ([`pta_source_noise.py`](pta_source_noise.py); §4.3). The table at the head
   of this question held the source to about −144 dB/Hz and marked it assumed.
   Run through grx930's harness on the working tile, a source may be 20 dB
   noisier than that, −124 dB/Hz, if a column reads its weights through a
   balanced pair, and no noisier than assumed if it reads them through an
   offset. Its lines have to be level to within a few percent either way.

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
