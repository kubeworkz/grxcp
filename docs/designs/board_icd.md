# GRX development board: interfaces, rev A

**Companions:** [`board_program_plan.md`](board_program_plan.md),
[`chip_vs_board_strategy.md`](chip_vs_board_strategy.md),
[`pta_cpu_integration.md`](pta_cpu_integration.md),
[`pta_chiplet_link.py`](pta_chiplet_link.py).

**Status: P0 of the board plan, drafted 2026-09-22.** This is the interface
control document for board rev A as §2 of that plan settles it: every link,
what carries it, who owns each end, and where its numbers come from. It fixes
nothing the plan has not already decided, and where a number is not yet
sourced it says so rather than inventing one — §7 collects those.

**How to read the source column.** `B1`–`B7` are the board plan's decisions,
`X1` and `X2` its chiplet work, `C1` grx930's error-model gate. A standard
named without a clause number means the number is the standard's to give and
has not been read out of it here. *Open* means nobody has sourced it yet.

---

## 1. The board

```
 CPU package                               GPU package
+-------------------+                     +-----------------------------------------+
| GRX930 SoC        |    CXL 2.0 over     | GRX-G100 die  <--- UCIe --->  PTA       |
| (RV64 cores, NPU) |<------------------->|                               chiplet   |
| DDR5 or LPDDR5    |    PCIe 5.0 x16     | GDDR6                         EIC + PIC |
+---------+---------+                     +-------------------------------------+---+
          |                                                                     |
          | SMBus / I3C: power, temperature, laser                     PM fiber |
          |                                                                     |
+---------+---------+                                                 +---------+------+
| board controller  |-------------------------------------------------| laser module   |
+-------------------+                                                 +----------------+
```

Four parties own the board between them:

| Party | Owns |
|---|---|
| grx930 team | The GRX930 die, its firmware, its memory and its CXL host |
| grxgpu team | The GRX-G100 die, its memory, its CXL device and its UCIe port |
| Chiplet team | The interface chip (EIC), the photonic die (PIC) and the package they share. **Not yet named** (§7) |
| Board team | The PCB, power, clocks, the controller, the laser module and fiber. **Not yet named** (§7) |
| grxcp | The runtime, the device model and this document |

---

## 2. Links

| # | Link | Carries | Width and rate | Owner, A end | Owner, B end | Source |
|---|---|---|---|---|---|---|
| 1 | GRX930 ↔ GRX-G100 | CXL 2.0 (CXL.io, .cache, .mem) on the PCIe 5.0 PHY | x16, 32 GT/s a lane, about 64 GB/s a direction before overhead | grx930: root port, home agent, HDM decoders | grxgpu: Type-2 device | B1, B3; rate from X2 |
| 2 | GRX-G100 ↔ PTA chiplet | UCIe-S, a streaming protocol in a FLIT format, with the adapter's CRC and retry | One x16 module at 32 GT/s is 57.6 GB/s a direction at X2's assumed 0.9 efficiency. **Module count follows the chiplet's geometry** (§7) | grxgpu: UCIe port, fed by a copy engine | Chiplet team: EIC | B4; rates and counts from X2 |
| 3 | EIC ↔ PIC | Analog: DAC drive to the modulators, photocurrent back from the detectors | One drive line a weight cell and one an input modulator, one detector a column: `k·n + k + n` lines, 16,704 at 256 × 64. The geometry is still §7's | Chiplet team | Chiplet team | B4, B5; resolutions from X1 |
| 4 | Laser module → PIC | Light, on polarization-maintaining fiber | Power, wavelength, noise and **kind** all **open** (board plan §8, question 8). For scale: a 64-column tile at 1 GS/s wants 0.33–3.3 W behind 10–20 dB of loss at the receiver B5 assumed, and 0.09–0.88 W at one that has been measured (board plan B5). It is set by the receiver's noise and not by X1's photon row, and intensity noise within about −144 dB/Hz. This row has said 0.16–1.6 W, sized from X1's version 0, and then 0.66–6.6 W and −150, from version 1's receiver noise read in the wrong ADC's LSB (board plan §4.3) | Board team | Chiplet team | B5, X1; figures from [`pta_shot_rate.py`](pta_shot_rate.py) §3 and §6 |
| 5 | Board controller ↔ all | SMBus or I3C: rails, temperatures, the laser and its interlock | Rate **open**, with the controller (§7) | Board team | Each die's management pins | B7 |
| 6 | GRX930 ↔ DDR5 | DDR5 (or LPDDR5) | Channels, width and speed grade **open**, with part selection | grx930 team | Board team | B2 |
| 7 | GRX-G100 ↔ GDDR6 | GDDR6 | Channels, width and speed grade **open**, with part selection | grxgpu team | Board team | B2 |
| 8 | Debug | JTAG chain and UART console | JTAG chain order **open** | Board team | grx930 and grxgpu teams | grx930's manufacturing plan, which asks a test board for both |
| 9 | Board I/O | Boot flash, console, network and storage for a self-hosted board | **Open**: the GRX930 is the CXL host, so the board boots itself rather than plugging into one | Board team | grx930 team | B3 |

### Notes that do not fit the table

**Link 1.** CXL is asymmetric and the roles are B3's: the GRX930 holds the home
agent and the memory decoders, and its L2 must answer snoops from outside the
chip. The GRX-G100 caches host memory over CXL.cache and exposes its own over
CXL.mem. Linux's CXL subsystem expects the host bridge in ACPI's CEDT table,
and RISC-V boards mostly boot with a device tree, so firmware carries that gap
(the plan's L4).

**Link 2.** Raw mode would drop the adapter's CRC and retry. B7 keeps them: at
16–32 GT/s the raw error rate is around one in 10^15 bits, which an operand
stream can absorb beside the analog noise and a register write cannot. The
PTA's register block — the CPU document's §3.1 — is MMIO in the GPU's BAR over
CXL.io, so the GPU driver owns it, and
[`pta_chiplet_regmap.md`](pta_chiplet_regmap.md) lays it out. The UCIe sideband
carries link management only.

*Its size, 2026-10-03.* By the UCIe Consortium's own tutorial figures (Hot
Chips 2023), a standard-package x16 module's PHY at 32 GT/s is 571.5 µm of die
edge by 1,540 µm deep, on bumps of 100 to 130 µm. X2's five modules are
4.4 mm² and 2.9 mm of edge. In a stacked chiplet the interface chip sits on top
of the photonic die, so this link comes down through the package to reach the
substrate, at the package's pitch and not the PHY's
([`pta_floorplan.py`](pta_floorplan.py) §5). That is a tutorial's summary and
not the specification, so item 5 of §7 stands.

**Link 3.** This is the only interface where the error budget is a wiring
requirement rather than a protocol: X1's version 1 asks for 6-bit activation
DACs, a 7-bit ADC, receiver noise within half an 8-bit ADC LSB — a quarter of
the 7-bit ADC's own, which is how it was run and how this note first put it — and
programming error within one weight LSB. Crosstalk between neighbouring inputs
must stay under 2%, which is as much a layout constraint on the PIC as an
electrical one.

*What the line count does to the package, 2026-10-03.* It decides how the two
dies sit. The photonic die cannot hold a voltage, so every weight is a line of
its own, and [`pta_floorplan.py`](pta_floorplan.py) puts 16,704 of them at
251 mm of shared edge if the dies sit side by side on fan-out wiring. So they
are stacked, with a pad a line, and the pad's pitch sizes both dies: 10 mm² at
25 µm, 167 at 100. The drive voltage belongs to this link too. On TFLT it sets
a modulator's length — 3.9 mm at 5 V, 19.6 at 1 V — which makes it a mechanical
figure as well as an electrical one, and it is **open**.

**Link 4.** The laser sits off the package (B5). The board owns the module, its
driver, its temperature control and the interlock; the chiplet owns the fiber
attach and the polarization the modulators need.

That was decided for one laser on one fiber, and what the tile asks of its source
has since been written down ([`pta_shot_rate.py`](pta_shot_rate.py) §6). Four
things, of which this document held one:

- **Power.** The same light whichever way it is made: one laser of 0.33–3.3 W at
  1 GS/s for a 256 × 64 tile, or an emitter an input row at 1.3–12.8 mW each. An
  emitter of 10 mW a row stands 18.9 dB of loss, where the largest single laser
  B5 planned on stands 16.9.
- **Noise.** The error model has no term for the source. Held to the receiver's
  own allowance its intensity noise is about −144 dB/Hz at 1 GS/s, ten tighter
  for each decade of rate, and independent of power and loss. An emitter a row
  relaxes that by up to 24 dB if the emitters are independent.
- **Wavelength.** B5's arithmetic is at 1550 nm; the same detector at 1310 nm
  needs 18% more light. The all-optical branch would pin it to 1550 nm within
  2.4 nm, and ring weights would pin it by a figure nobody has.
- **Kind**, which decides the rest. The error model is written for a ring bank,
  which sums powers and needs a line an input — so under the model as it stands
  this link carries many lines, not one, and a single-line laser is not a source
  for it. The CPU document calls that topology a hypothesis.

None of this picks a laser. No part's output power or noise is held here, and
the comparison with one belongs to whoever has its datasheet.

*How many fibers, 2026-10-03.* Not decided, and not free. The one published
package the board plan holds for scale (its §8, question 1) attaches fiber at
250 µm pitch along one edge, which is 52 fibers on the package's longer, 13 mm
side if the whole side is facet. A fiber an input row is 256 at X2's geometry,
64 mm of facet. So an emitter a row reaches this link as a few fibers with the
rows' lines combined on the board, or it does not cross this link at all and
sits on the package. Which is the kind's question again, with a length on it.

---

## 3. Power, at block level

Values are open until P1's package study and part selection; what this fixes is
the rails, their owners and what must not share them.

| Rail group | On | Owner | Notes |
|---|---|---|---|
| CPU core and uncore | CPU package | grx930 team | Current **open** (B6's node decides it) |
| CPU memory and PHY | CPU package | grx930 team | DDR5 needs its own supply and reference |
| CPU SerDes | CPU package | grx930 team | PCIe 5.0 analog, quiet by requirement |
| GPU core | GPU package | grxgpu team | The board's largest rail; current **open** |
| GPU memory and PHY | GPU package | grxgpu team | GDDR6 |
| GPU SerDes and UCIe PHY | GPU package | grxgpu team | Two PHYs, one package |
| EIC digital | PTA chiplet | Chiplet team | |
| EIC analog | PTA chiplet | Chiplet team | DAC references and receivers. **Must not share a regulator with any digital rail**, because X1's budget is half an 8-bit LSB of receiver noise |
| PIC bias | PTA chiplet | Chiplet team | Pockels bias, held by DACs (B5) |
| Laser module | Board | Board team | Diode current, and temperature control if the module needs it |
| Controller, clocks, fans | Board | Board team | Up first, down last |

Sequencing is a dependency list, not a schedule: the controller comes up first
and brings up the rails; the laser may not emit before the PIC's bias is set
and the interlock is satisfied (B5); the CXL link trains only after both
packages are out of reset.

*For scale, 2026-10-04.* [`pta_power.py`](pta_power.py) puts a floor under the
chiplet's rails from published parts, for a 256 × 64 tile at 1 GS/s. The analog
rail's receivers and ADCs are 0.4 to 0.6 W. The drive to the modulators and the
weights is 0.1 to 4 W, depending on a swing and a capacitance that are both
open. The link is 0.13–0.22 W with the weights resident and 1.8–3.1 W with them
re-sent, which is both ends of it. None of this is a current. It says which
rail is large and what makes it so, not what regulator it needs.

---

## 4. Clocks

| Clock | Feeds | Source |
|---|---|---|
| 100 MHz reference | The PCIe 5.0 PHYs at both ends of link 1 | PCIe. Whether the two packages share it or run separately, with the spread-spectrum arrangement that implies, is **open** |
| Forwarded, inside link 2 | The UCIe module | UCIe, which also fixes how many lanes carry clock, valid and sideband — **not read out here** |
| Memory clocks | Links 6 and 7 | Each die's PLL from a board reference; frequencies **open** with the parts |
| Shot clock | The tile's shots, and so the ADCs | The chiplet. X2 sized the link at 0.1 and 1 GS/s as candidates; the rate is **open** (§7) |
| Controller | The board controller | Its own oscillator |

---

## 5. Resets and bring-up order

1. Controller rails, then the controller.
2. Package rails in the order §3 fixes, digital before analog.
3. Package resets released; each die runs its own boot.
4. Link 1 trains: PERST# to the device, then PCIe and CXL link-up and
   enumeration. Firmware describes the host bridge (L4).
5. Link 2 trains: UCIe sideband first, then the mainband.
6. The laser is enabled only after the PIC's bias is set and the interlock is
   satisfied.
7. The PTA is reset through its register block, which the core's contract
   allows only while idle — the CPU document's §3.2 — and calibrated before
   its first GEMM.

---

## 6. Debug and telemetry

| What | Where it comes out | Owner |
|---|---|---|
| CPU and GPU debug | JTAG chain; chain order **open** | Board team |
| Console | UART from the GRX930 | grx930 team |
| CXL link state | Standard configuration space | grx930 and grxgpu teams |
| UCIe link state and DFx | The UCIe management path; what it exposes is **open** | grxgpu and chiplet teams |
| PTA counters | `PTA_SHOT_CT`, `PTA_SAT_CT`, `PTA_CAL_CT`, `PTA_CAL_CYC`, `PTA_ERR_MAX`, read as MMIO over CXL.io | Chiplet team, read by grxcp |
| Rails, temperatures, laser | The controller, over link 5 | Board team |

The PTA's counters are not diagnostics in the optional sense: the CPU document
keeps them so a reported GEMM time can be split into compute, weight
programming and calibration, and C3's schedulers read `PTA_ERR_MAX`.

---

## 7. Not yet sourced

Each of these is a number or an owner this document cannot supply yet. The
board plan's §8 holds the ones that are questions rather than gaps.

1. **The chiplet team and the board team are not named.** Every link with an
   *open* owner in §2 waits on that.
2. **The chiplet's geometry and shot rate** — inputs, outputs, GS/s — which set
   link 2's module count, link 3's line count, link 4's laser power and the
   shot clock. Board plan §8, question 1.
   **And the kind of source link 4 carries** — one line or one an input — which
   follows from how a column sums. Board plan §8, question 8.
3. **Part selection** for DDR5, GDDR6, the controller, the laser module and the
   clock sources, with everything that follows from it.
4. **Currents** on every rail in §3.
5. **UCIe lane composition and DFx surface**, to be read out of the
   specification rather than assumed here.
6. **The FPGA platform for rev 0**, which will have its own, shorter version of
   this document. Board plan §8, question 6.

---

## 8. Against P0's gate

P0 asks that every link have an owner on each side and every number a source.

- Links 1, 2, 3, 6, 7 and 8 have owners on both ends, except that the chiplet
  team and the board team are roles rather than names (§7, item 1).
- Every rate, width and budget figure in §2 carries a source, and every figure
  that has none is marked open rather than guessed.
- Power, clocks, resets and debug are here at block level: rails and their
  owners, clock domains and their sources, a bring-up order as dependencies,
  and the telemetry each party reads.

What this document cannot close is §7. P1 and part selection close most of it;
the chiplet's geometry and its topology close the rest — the topology because it
decides what link 4 carries.
