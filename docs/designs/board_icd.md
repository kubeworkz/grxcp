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
| 3 | EIC ↔ PIC | Analog: DAC drive to the modulators, photocurrent back from the detectors | One drive line a weight cell, one detector a column. Counts follow the geometry (§7) | Chiplet team | Chiplet team | B4, B5; resolutions from X1 |
| 4 | Laser module → PIC | Light, on polarization-maintaining fiber | Wavelength and power **open**; X1's budget asks for at least 30 photons an ADC LSB, and B5's sizing put a 64-channel tile at 0.16–1.6 W on stated assumptions | Board team | Chiplet team | B5, X1 |
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
CXL.io, so the GPU driver owns it, and X4 will lay it out. The UCIe sideband
carries link management only.

**Link 3.** This is the only interface where the error budget is a wiring
requirement rather than a protocol: X1's version 1 asks for 6-bit activation
DACs, a 7-bit ADC, receiver noise within a quarter of an 8-bit ADC LSB, and
programming error within one weight LSB. Crosstalk between neighbouring inputs
must stay under 2%, which is as much a layout constraint on the PIC as an
electrical one.

**Link 4.** The laser sits off the package (B5). The board owns the module, its
driver, its temperature control and the interlock; the chiplet owns the fiber
attach and the polarization the modulators need.

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
| EIC analog | PTA chiplet | Chiplet team | DAC references and receivers. **Must not share a regulator with any digital rail**, because X1's budget is a quarter of an 8-bit LSB of receiver noise |
| PIC bias | PTA chiplet | Chiplet team | Pockels bias, held by DACs (B5) |
| Laser module | Board | Board team | Diode current, and temperature control if the module needs it |
| Controller, clocks, fans | Board | Board team | Up first, down last |

Sequencing is a dependency list, not a schedule: the controller comes up first
and brings up the rails; the laser may not emit before the PIC's bias is set
and the interlock is satisfied (B5); the CXL link trains only after both
packages are out of reset.

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
the chiplet's geometry closes the rest.
