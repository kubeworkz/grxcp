# GRX development board: program plan

**Companions:** [`chip_vs_board_strategy.md`](chip_vs_board_strategy.md),
[`pta_program_plan.md`](pta_program_plan.md),
[`pta_cpu_integration.md`](pta_cpu_integration.md),
[`pta_gpu_integration.md`](pta_gpu_integration.md),
[`ai_motherboard_design_capabilities.md`](ai_motherboard_design_capabilities.md),
[`heterogeneous_devices.md`](heterogeneous_devices.md),
[GRX_GCPU.md](../GRX_GCPU.md).

**Status: PLAN, drafted 2026-09-21. None of §2's decisions is settled yet.**

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
| CPU–GPU coherence protocol | [GRX_GCPU.md](../GRX_GCPU.md) §2: TileLink, with CXL.cache "too heavyweight for on-package" | CXL | Reversed; B3 |
| Where the GPU's PTA sits | [`pta_gpu_integration.md`](pta_gpu_integration.md) §2: inside the G100, one per cluster, beside the DXA | A separate chiplet | Re-architected; B4 |
| Photonic platform | [`pta_cpu_integration.md`](pta_cpu_integration.md) §4.4: TFLT, a Pockels material | The strategy and motherboard documents assume heater-tuned silicon rings | Aligned to TFLT; B5 |
| The CPU's silicon | grx930's `c930/doc/GRX930_manufacturing_plan.md`: SKY130, then TSMC N28, with the RTL frozen | UCIe and CXL need PHYs neither node offers readily | B6 |
| Memory | The motherboard document: HBM | A development board | B2 |
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
come first; the rest follow from them.

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
without it.

**B2 — The first board's memory.** HBM needs a silicon interposer or bridge,
which puts every die beside it into a 2.5D package — the packaging the
strategy document names as the constrained resource whose allocation is
priced.
*Recommended:* GDDR6 or LPDDR5X for the GPU and DDR5 or LPDDR5 for the CPU, on
organic substrates, for rev A; HBM for the Phase 2 module. A development kit's
job is to prove the stack, and it can do that at lower memory bandwidth.
*Needed by:* P0 and P1.

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
the edge. *Needed by:* L1 and L2.

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
*Needed by:* L3, X2 and X4.

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
of the GPU's hot spots.

*Laser power, for scale.* C1's sweep allows receiver noise of one 8-bit ADC
LSB, and shot noise down to 3 photons per ADC LSB (§4.3). Take a receiver with
about 1 µA rms of input noise and 1 A/W of responsivity. The receiver limit
then puts a detector's full scale at 256 µA, about 0.26 mW, while the shot-noise
limit asks only about 0.1 µW at 1 GS/s and 1550 nm. So the receiver, not shot
noise, sets the laser power, by three orders of magnitude: 64 channels behind
10–20 dB of optical loss need 0.16–1.6 W of laser. Every input is an
assumption until the chiplet is sized (§8), but a laser of that order is a
board-level thermal and safety item. *Needed by:* P1 and X1.

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
L3.

**B7 — One control path to the PTA.** The strategy document proposes three: a
low-speed I2C or SPI bus from the CPU, the UCIe sideband, and Raw-mode traffic
on the mainband. The sideband exists to train and manage the link. Raw mode has
no CRC or retry, and at 16–32 GT/s the raw error rate is around one in 10^15
bits: operand payloads can tolerate that beside the analog noise, but a
register write cannot.
*Recommended:* the PTA's register block (the CPU document, §3.1) as MMIO behind
the GPU's CXL.io function, so that the GPU's driver owns it. SMBus or I3C
carries board management only — power, temperature and the laser — and the
UCIe sideband carries the link's own management. *Needed by:* X4.

---

## 3. The tracks

### 3.1 Track P — package and board (grxcp coordinates)

| Step | What | Gate | Needs |
|---|---|---|---|
| P0 | System block diagram and interface control document: every link's protocol, width, rate and owner; power, clock, reset and debug trees at block level | Review: every link has an owner on each side, and every number a source | B1–B7 |
| P1 | Package study of the GPU package with the PTA chiplet: floorplan, UCIe-S or UCIe-A, fiber attach, and a thermal co-simulation with TFLT's drift in place of heater terms | The model predicts the PIC's temperature range under the GPU's power map, and C1's drift fits say what that costs in calibration | B1, B2, B5 |
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
| X1 | EIC requirements. Version 0 is §4.3, from C1; the next is a joint budget, with every impairment on at once | Every number traced to grx930's design note §5 or to a new run | C1, done |
| X2 | Link sizing: [`pta_feed_model.py`](pta_feed_model.py) with a die-to-die term — bandwidth, latency and the EIC's buffers. The PTA plan's F3 hands its requirements here | Predictions stated before any RTL, and every number traced, as F1's were | B4, F1 |
| X3 | The calibration engine on the EIC: the PTA plan's C3, specified for the chiplet | C3's own gate, at TFLT's and TFLN's drift | C3 |
| X4 | The register map over CXL.io: the CPU document's §3.1 block in the GPU's BAR, with §3.2's completion contract restated for a device behind a link | Review | B4, B7 |
| X5 | The digital twin: `pta_tile_model.c` behind X4's map, so that drivers and grxcp can bring the PTA up before silicon | grxcp's backend gates pass against it, bitwise against the model | X4 |

### 3.4 Track S — grxcp

| Step | What | Gate | Needs |
|---|---|---|---|
| S1 | Enumeration over CXL: the GPU, and its PTA, found through configuration space | [`heterogeneous_devices.md`](heterogeneous_devices.md) §4's rule: a device that is not present is not enumerated | L4, X4 |
| S2 | A coherent shared pool: pointers valid on both the CPU and the GPU, beside the per-device spaces of §4.1 | Each coherent allocation reported through a device property, and no pointer resolved to the wrong device | L1, L2 |
| S3 | The dispatch cost model: the §2.1 model with X2's link terms and C1's accuracy, placing each GEMM on the PTA, the GPU or the NPU | Its predictions checked on rev 0 | X2, P2 |
| S4 | The PTA reported through the PTA plan's D2 property — effective bits, seed, impairment mask — from the twin now and from silicon later | `AGENTS.md` §3: every field sourced or reported unknown (−1) | X5 |

---

## 4. What this hands on

### 4.1 To grx930

- The CXL 2.0 host of L1, and the firmware of L4.
- A node with PCIe 5.0-class PHYs for the board's CPU (B6), and a PCIe 5.0 x16
  root port.
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
- The memory controller B2 chooses.

### 4.3 To the PTA chiplet: EIC requirements, version 0

These come from C1's sweep on the D3 network — a 784-100-10 MLP on MNIST, at
8-bit operands and 6-bit weights, 97.45% with nothing else impaired — in
grx930's `c930/doc/pta_error_model_design_note.md` §5. Each row costs about
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

Together these compound, and one small network is a floor, not a sign-off. X1's
joint budget is the next step.

### 4.4 To the PTA program

- C3, the calibration engine, continues, and becomes X3.
- C4's target moves: the CSR map goes to CXL.io behind the GPU (X4), not to the
  Arty A7 SoC's bus.
- Step F3's requirements for the fabric go to X2.
- G2's cluster-scope tile gives way to the chiplet attach (B4).
- The error-model tile becomes the chiplet's digital twin (X5).
- The scope lines of both integration documents are rewritten (§6).

---

## 5. Order

1. Settle §2, with B1 and B3 first.
2. On paper, now: P0, X1's joint budget, X2 and X4.
3. C3 continues in the PTA program, as X3.
4. P2, as soon as B6 names the FPGA platform.
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
  recommendation (B3).
- [`pta_gpu_integration.md`](pta_gpu_integration.md): its scope line; §2's
  cluster-scope placement, superseded by the chiplet (B4); G2, re-scoped.
- [`pta_cpu_integration.md`](pta_cpu_integration.md): its scope line; C4's
  target; the home of the §3.1 block (X4).
- [`pta_program_plan.md`](pta_program_plan.md): F3 pointed at X2; C4 and G2
  re-scoped; a pointer to this plan.
- [`ai_motherboard_design_capabilities.md`](ai_motherboard_design_capabilities.md):
  the thermal section's thermo-optic assumptions replaced with TFLT's, and the
  laser placed (B5).
- [`heterogeneous_devices.md`](heterogeneous_devices.md) §4.1: the coherent
  shared pool (S2).
- grx930's `c930/doc/GRX930_manufacturing_plan.md`, which that team owns: the
  board CPU's node (B6), as a requirement.

---

## 7. Risks

| Risk | Where it bites | Response |
|---|---|---|
| UCIe and PCIe 5.0-class PHY IP: which nodes, availability, license cost | B6, L1–L3 | Source it early, and prototype on FPGA hard IP first |
| Advanced-packaging capacity and cost | B2, P3 | Organic substrates and non-HBM memory for rev A |
| TFLT dies not available in the volume or quality needed | B5, Track X | TFLN as the fallback, with calibration sized to its drift |
| RISC-V support in Linux's CXL subsystem | L4, S1 | Firmware planned alongside L1, not after it |
| Coherence verified across a chip boundary, in the c930's L2 and the G100's | L1, L2 | Begin with GRX_GCPU.md's small, configurable coherent region, and grow it |
| The link, not the optics, sets the PTA's throughput | B4, X2 | Keep weights, activations and accumulation on the EIC, and size the link from the feed model |
| Laser power, reliability, fiber attach and eye safety | B5, P1 | An off-package laser with an interlock, and a power budget from X1 |
| Export controls on board-level products (the strategy document) | Phase 2 | Classify the product before the first shipment |

---

## 8. Open questions

1. **How big is the PTA chiplet, and how fast?** Its inputs, outputs and shot
   rate set X2, the laser (B5) and the EIC's area. Nothing here fixes them.
2. **What does the development kit cost, and how many are built?** That settles
   B1 and B2 more than any technical argument does.
3. **Does the GRX930's NPU keep a PTA of its own?** The c930 PTM work is built
   and gated. It can be a product feature, or only the development vehicle.
4. **Does the PTA ever need coherent access to host memory?** B3 gives it none;
   revisit that if the dispatch model (S3) says otherwise.
5. **Where does the board controller come from, and who writes its firmware?**
