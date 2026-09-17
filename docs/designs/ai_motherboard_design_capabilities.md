**Yes, AI can already help design significant parts of a motherboard — but it can't yet independently design a complete, production-ready one from scratch.**

Here's a more nuanced breakdown.

## What AI can do today

**Component placement & routing**

- Reinforcement learning and optimization algorithms (like Google's work on TPU chip floorplanning) can place components and route traces faster than humans, often with better results.
- These are already used in **chip design** (not just boards) — e.g., Synopsys, Cadence, and Google use ML for floorplanning, placement, and routing.

**Signal integrity & power analysis**

- AI models can predict impedance, crosstalk, and thermal hotspots, flagging problems before simulation.
- Useful for high-speed buses (PCIe, DDR, USB) where layout rules are complex.

**Design rule checking (DRC) & verification**

- ML speeds up DRC by learning which violations matter and predicting errors.
- Can auto-suggest fixes for common layout issues.

**Schematic generation & part selection**

- LLMs can draft schematics, pick components from datasheets, and generate netlists for simple designs.
- Tools like Flux, Jitx, and various EDA startups are pushing toward "describe it, get a board."

## What AI still can't do well

- **Full end-to-end ownership** — no AI today takes "build me a gaming motherboard" and outputs a manufacturable, tested design.
- **Novel architecture decisions** — choosing chipset, power delivery topology, layer stackup for a new product still needs human judgment.
- **EMI/EMC compliance** — passing FCC/CE certification requires real-world testing and experience-based tweaks.
- **Manufacturing constraints** — DFM (design for manufacturing), cost optimization, and supply chain realities need human input.
- **Accountability** — someone has to sign off legally and functionally.

## The realistic picture

Think of it as **AI-assisted design, not AI-autonomous design**.

<table>
  <tr><th>Stage</th><th>AI role today</th></tr>
  <tr><td>Spec &amp; architecture</td><td>Minimal (human-led)</td></tr>
  <tr><td>Schematic capture</td><td>Assistive (LLM drafts)</td></tr>
  <tr><td>Placement &amp; routing</td><td>Strong (RL/optimization)</td></tr>
  <tr><td>Simulation &amp; verification</td><td>Strong (ML accelerators)</td></tr>
  <tr><td>DRC / DFM</td><td>Assistive</td></tr>
  <tr><td>Bring-up &amp; debug</td><td>Minimal</td></tr>
</table>

## Where it's heading

Startups like **Jitx**, **Flux**, and **Celus** are building "AI co-pilot for PCB design." Combined with LLMs reading datasheets and RL optimizing layouts, we're maybe 5–10 years from AI designing **simple boards** (breakout boards, sensor modules) with light human review — and longer for complex motherboards.

So: **AI can design *parts* of a motherboard very well today, and will increasingly design *most* of one — but full autonomy on a complex board is still ahead.**

---

Here's a rundown of the standard items you'll find on most motherboards, grouped by function.

## Core compute & memory

- **CPU socket** — LGA (Intel) or PGA/AM (AMD) where the processor sits.
- **DIMM slots** — for RAM (typically 2 or 4).
- **Chipset** — the support chip that handles I/O, often under a heatsink.

## Power

- **VRM (Voltage Regulator Module)** — stages of MOSFETs, inductors, and capacitors that convert 12V into the low voltages the CPU needs.
- **24-pin ATX power connector** — main power from the PSU.
- **8-pin (or 4+4) EPS connector** — dedicated CPU power.
- **VRM heatsinks** — often near the socket and rear I/O.
- **CMOS battery** — coin cell (CR2032) that keeps BIOS settings and clock alive.

## Expansion & storage

- **PCIe slots** — x16 (usually for GPU), x1/x4 for other cards.
- **M.2 slots** — for NVMe SSDs (sometimes also Wi-Fi cards).
- **SATA ports** — for 2.5"/3.5" drives.

## I/O & rear panel

- **Rear I/O cluster** — USB ports, Ethernet (RJ45), HDMI/DisplayPort, audio jacks, sometimes Wi-Fi antenna connectors, USB-C, PS/2.
- **Front panel headers** — for power button, reset, HDD LED, USB, and audio on the case front.

## Control & monitoring

- **BIOS/UEFI flash chip** — stores firmware.
- **Super I/O chip** — manages legacy ports, fan control, sensors.
- **Fan headers** — for CPU cooler and case fans.
- **Debug tools** — POST code display, debug LEDs, or a speaker header.
- **Clear CMOS jumper/button** — resets BIOS settings.

## Networking & audio

- **Ethernet controller** — often built into chipset, sometimes a discrete chip (e.g., Intel, Realtek).
- **Audio codec** — e.g., Realtek ALC series, plus audio capacitors and jacks.
- **Wi-Fi/Bluetooth module** — soldered or via M.2.

## Other common bits

- **Chipset heatsink** — passive cooling for the chipset.
- **PCB layers & traces** — often 4–8+ layers for signal and power routing.
- **Mounting holes** — for standoffs to the case.
- **VRM/power connectors for fans, RGB headers** — 3-pin/4-pin ARGB and RGB headers.
- **TPM header** — for a discrete TPM module (or built-in fTPM).
- **PCIe/M.2 heatsinks** — increasingly standard on higher-end boards.

## Quick mental model

Think of a motherboard as five zones:

1. **CPU + VRM + RAM** (top/right)
2. **Chipset + storage** (bottom/right)
3. **PCIe expansion** (center/bottom)
4. **Rear I/O + audio/networking** (left edge)
5. **Power + front panel + headers** (edges)

The exact set varies by form factor (ATX, micro-ATX, mini-ITX), budget, and generation — but those are the near-universal ingredients.

---

AI actually adds value vs. where it's still weak. Let me rank the motherboard items by how much AI can meaningfully help design them.

## Tier 1: High AI leverage (AI does real work)

**1. Component placement & trace routing**

- **Where:** CPU socket, DIMM slots, PCIe slots, M.2, chipset — the "big blocks" plus all the traces between them.
- **Why AI wins:** This is a massive combinatorial optimization problem (billions of possible layouts). RL and optimization algorithms (like Google's chip floorplanning work) genuinely beat humans on speed and often on quality.
- **Key constraints AI handles well:** trace length matching (DDR), impedance control (PCIe), differential pairs, crosstalk avoidance.

**2. VRM design & power delivery**

- **Where:** MOSFETs, inductors, capacitors, controller IC.
- **Why AI wins:** Multi-objective optimization — efficiency, thermals, transient response, cost, board area. AI can sweep thousands of configurations fast.
- **Bonus:** AI predicts thermal hotspots before you build anything.

**3. High-speed signal integrity**

- **Where:** DDR4/DDR5, PCIe 4/5/6, USB, Ethernet traces.
- **Why AI wins:** These have brutal design rules. ML models predict eye diagrams, jitter, and impedance mismatches far faster than full SPICE simulation.

**4. Layer stackup & PCB geometry**

- **Where:** Number of layers, copper thickness, dielectric materials, via types.
- **Why AI wins:** Tradeoff between cost, signal integrity, and manufacturability — a perfect optimization target.

## Tier 2: Medium AI leverage (AI assists, human decides)

**5. Schematic capture & netlist generation**

- LLMs can draft schematics from a spec, pick components from datasheets, and generate netlists — but need human review for correctness.
- Good for standard blocks (power trees, USB hubs), weaker for novel architecture.

**6. Component selection (BOM)**

- AI can match specs to parts, check availability, and optimize cost — but supply chain and second-sourcing still need human judgment.

**7. DRC / DFM checking**

- ML speeds up design rule checking and predicts which violations matter. Still needs human sign-off.

**8. Thermal design**

- AI predicts hotspots and suggests heatsink placement, but real-world airflow testing still required.

## Tier 3: Low AI leverage (human-led, AI barely helps)

**9. Architecture & chipset choice**

- "Should this be AMD or Intel? Which chipset? How many PCIe lanes?" — strategic decisions needing market and product context.

**10. BIOS/UEFI firmware**

- Deeply hardware-specific, security-critical, and vendor-locked. AI can help write snippets but not own it.

**11. EMI/EMC compliance**

- Passing FCC/CE requires physical testing and experience-based tweaks. AI can predict but not certify.

**12. Bring-up & debug**

- First power-on, signal probing, fixing "why won't this POST" — still very human.

**13. Connector & mechanical layout**

- Rear I/O, headers, mounting holes — constrained by ATX spec and case design. Little optimization room.

## The practical takeaway

If you were building an **AI-assisted motherboard design tool**, you'd focus on:

<table>
  <tr><th>Priority</th><th>Item</th><th>Why</th></tr>
  <tr><td>1</td><td>Placement &amp; routing</td><td>Biggest time sink, best AI fit</td></tr>
  <tr><td>2</td><td>VRM / power delivery</td><td>High complexity, clear objectives</td></tr>
  <tr><td>3</td><td>High-speed signal integrity</td><td>Hard rules, AI predicts well</td></tr>
  <tr><td>4</td><td>Layer stackup</td><td>Pure optimization</td></tr>
  <tr><td>5</td><td>DRC/DFM</td><td>Automation win</td></tr>
  <tr><td>6</td><td>Schematic/BOM assist</td><td>LLM-friendly, human-reviewed</td></tr>
</table>

**The pattern:** AI excels where the problem is **optimization with clear constraints** (placement, routing, power, signal integrity). It struggles where the problem is **judgment, novelty, or physical reality** (architecture, firmware, EMI, debug).

So the "items that matter most" aren't the flashy ones (CPU socket, PCIe slots) — they're the **invisible infrastructure**: the traces, the power delivery, the layer stack. That's where AI earns its keep.

---

## Our own CPU / GPU

Ah — that changes *everything*. If you've designed your own CPU and GPU, you're no longer designing "a motherboard" in the normal sense. You're designing a **custom compute platform**, and most of the standard motherboard assumptions fly out the window.

Let me walk through what actually changes.

## The big shift: you control the silicon, not just the board

Normally a motherboard designer is **constrained by off-the-shelf chips**:

- Fixed pinouts (LGA1700, AM5, BGA packages)
- Vendor-mandated power delivery specs (Intel VRM guidelines, AMD SVI)
- Fixed PCIe lane counts, memory controllers, chipset interfaces
- Reference designs and NDAs from Intel/AMD/NVIDIA

When you own the CPU and GPU, **you define all of those constraints yourself**. That's both freeing and brutal.

## What you now have to design that you didn't before

### 1. The package & pinout

- **Package substrate** — the organic or ceramic interposer your die sits on. You design the ball map.
- **BGA ballout** — thousands of balls for power, ground, high-speed I/O, clocks, debug.
- **Signal-to-ball assignment** — you decide which signal goes where, which means you *also* own the PCB routing problem at a whole new level.

This is where AI-assisted design gets *very* interesting — ballout + PCB co-optimization is a famously hard problem.

### 2. Power delivery (this becomes enormous)

- **VRM becomes VRD (Voltage Regulator Down)** — you spec the exact voltage rails, transient requirements, and load lines.
- If your CPU has multiple voltage domains (cores, uncore, memory controller, I/O), you're designing **multi-rail power trees**.
- Your GPU likely needs **hundreds of amps at sub-1V** — that's a power delivery problem most motherboard designers never touch.
- You may need **on-package or on-interposer power delivery** (like Intel Foveros or AMD's 3D V-Cache power schemes).

### 3. High-speed interfaces you now own

- **Die-to-die links** if CPU and GPU are separate dies (UCIe, custom SerDes, or something proprietary).
- **Memory interface** — you decide DDR5, LPDDR5X, HBM, or something custom. If HBM, you need an interposer and 2.5D/3D packaging.
- **PCIe/CXL** — you define the PHY, lane count, and whether you even use PCIe at all.
- **Chiplet interconnect** — if you go chiplet-based, the fabric is yours.

### 4. Firmware & bring-up

- **No vendor BIOS to fall back on.** You write the boot ROM, the memory training code, the power sequencing firmware.
- **Bring-up is now a silicon problem**, not a board problem. You'll be debugging why the CPU won't POST at the transistor level.

## What "motherboard" even means now

Depending on your architecture, you might be building:

<table>
  <tr><th>Architecture</th><th>What you&#39;re really designing</th></tr>
  <tr><td>CPU + GPU on one PCB, discrete packages</td><td>A custom high-speed board with your own pinouts</td></tr>
  <tr><td>Chiplet CPU + chiplet GPU on interposer</td><td>A 2.5D package + a simpler PCB</td></tr>
  <tr><td>Monolithic SoC with CPU+GPU</td><td>A package + a &quot;carrier board&quot; — closer to a phone board</td></tr>
  <tr><td>Multi-socket / multi-GPU</td><td>A backplane or fabric topology problem</td></tr>
</table>

In most of these, the **PCB is the easy part**. The hard parts are package, power, signal integrity, and firmware.

## How AI-assisted design changes

Now the tiers I gave you earlier **shift**:

**AI gets *more* valuable:**

- **Ballout + PCB co-optimization** — you're optimizing across two domains simultaneously. Pure AI territory.
- **Power delivery for extreme current** — multi-rail, transient, thermal — massive optimization space.
- **Die-to-die / chiplet routing** — very high-speed, very constrained.
- **Custom SerDes tuning** — ML is already used here in industry.

**AI gets *less* able to help:**

- **Architecture decisions** — you're in novel territory. No training data for your specific chip.
- **Firmware bring-up** — deeply custom, no reference.
- **Silicon bugs** — AI can't debug your RTL from the board side.

## The honest reality check

Designing your own CPU and GPU is a **multi-hundred-million-dollar, multi-year effort** even for a small team with serious funding. The motherboard is genuinely the *least* of your problems. But if you're doing it:

1. **Package + PCB co-design is where AI helps most.**
2. **Power delivery is your biggest board-level challenge.**
3. **Firmware and bring-up will consume most of your time.**
4. **You'll want to reuse standard interfaces** (PCIe, DDR, USB) rather than invent everything — unless you have a very good reason.

## So — which direction are you actually going?

A few questions that would let me give you much more targeted advice:

- **Scale:** Hobbyist FPGA-adjacent project, startup, or large org?
- **CPU/GPU relationship:** Separate dies, chiplets, or monolithic SoC?
- **Memory:** DDR, LPDDR, HBM, or custom?
- **Goal:** General compute, AI accelerator, embedded, or something else?
- **Fabrication:** TSMC/Samsung, or something more accessible?

```plaintext
Scale: Startup 
CPU/GPU relationship: separate dies and a chiplet for a photonic tensor accelerator 
Memory: HBM
Goal AI accelerator
Fabrication: TSMC/Samsung
```

Alright, that context changes everything. You're not just building a motherboard — you're building a **photonic-electronic heterogeneous package** with HBM, which sits at the bleeding edge of advanced packaging. The "board" is almost an afterthought; the real design problem is the **interposer/package**.

Let me reframe the AI-assisted design tiers for your specific architecture.

### The New Core Challenge: Photonic-Electronic Co-Design

Standard AI-assisted design tools and methods assume electrical signals. Photonics introduces a fundamentally different constraint set. The most critical design element is no longer just trace routing — it's **thermal-aware placement** of your photonic tensor chiplet relative to your electronic CPU/GPU dies.

The key issue: **silicon's refractive index changes with temperature**. A photonic ring resonator shifts roughly 70–80 pm per °C . Your electronic dies dissipate watts of heat microns away from optical components that need to stay within a fraction of a degree. As one expert put it, this is "the opposite of what you want thermally, even though it's exactly what you want electrically" .

This means **thermal simulation and photonic-electronic co-simulation become the dominant design activity**, not optional checks. You need to model the bidirectional feedback loop: heat affects optical performance, which may trigger active tuning (heaters), which generates more heat.

### AI's Highest-Leverage Roles for Your Architecture

1. **Interposer Layout & Routing (Critical)**  

With HBM3/4 and UCIe die-to-die links, you're dealing with **hundreds of thousands of interconnects** on the interposer . Manual routing is infeasible. AI-driven auto-routing tools (like those emerging for 2.5D packaging) handle HBM and UCIe routing automatically, with integrated signal integrity validation . This is where AI saves months of engineering time.
2. **HBM Signal Integrity Optimization**  

HBM4 targets 12.8 Gbps per pin with a doubled interface width (2048-bit). Crosstalk and jitter make this brutal . AI/ML models can predict eye diagrams and optimize signal-ground routing patterns far faster than full electromagnetic simulation. Synopsys has demonstrated that AI-assisted routing improves HBM eye opening significantly .
3. **Thermal-Aware Placement**  

This is your unique photonic challenge. AI can explore the massive placement space to find configurations that isolate your photonic chiplet from the hottest electrical dies. Tools can co-optimize for both electrical performance (short links) and thermal isolation (optical stability) .
4. **Power Delivery for HBM**  

HBM stacks require clean, high-current power delivery with tight noise margins. AI optimization of the power distribution network (PDN) is valuable here, especially given the dense bump patterns .

### What Remains Human-Led

- **Photonic architecture decisions**: How many optical channels? What wavelength division scheme? Where do the lasers live? No AI has training data for your specific photonic tensor design.
- **Laser integration strategy**: Lasers are temperature-sensitive and often placed off-package or carefully isolated . This is a system architecture call.
- **Firmware for photonic calibration**: Your optical tuning loops are safety-critical and novel. Formal verification, not AI generation, is the right approach .
- **Bring-up**: Debugging why your photonic tensor core won't lock its resonances is deeply physical and experiential.

### The Practical Stack

For your startup, the AI-assisted design work concentrates in **advanced packaging EDA**. Look at tools like Synopsys 3DIC Compiler or Empyrean Storm , which specifically target HBM and UCIe routing on silicon or organic interposers. These tools are increasingly embedding AI/ML for auto-routing and SI validation.

Your "motherboard" — the organic substrate and PCB — is the simplest layer. The hard AI problems are all in the **2.5D interposer and package**.

---

### The thermal co-simulation challenge for photonics, and how HBM4 signal integrity constraints shape our interposer routing rules

Both of these challenges sit at the core of what makes our photonic accelerator so difficult to build. Let me break down what the research actually says about each, and how AI-assisted design can help — or where you'll need to rely on human judgment.

## Thermal Co-Simulation for Photonics

The fundamental problem is stark: your CPU and GPU dies are heaters. Your photonic tensor chiplet is a precision instrument that measures its own performance in fractions of a degree.

### The Heater Efficiency Collapse

When you hybrid-bond or flip-chip an electronic IC (EIC) directly onto a photonic IC (PIC), the EIC acts as an **unwanted heat spreader**. Research shows heater efficiency drops by **42–62%** after hybrid bonding . Earlier flip-chip studies measured a **43.3% efficiency loss** and a **44.4% increase in thermal crosstalk** between photonic devices .

Why this matters for your architecture: your photonic tensor chiplet almost certainly uses **thermo-optic tuning** (heaters) to lock ring or disk resonators to specific wavelengths. If your heater efficiency drops by half, your tuning power doubles — or your thermal loops can't maintain lock at all.

### The Spatial and Temporal Gradient Problem

A fully populated interposer with an XPU power map creates gradients of **12 K/mm spatially and 1.78 K/ms temporally** within the PIC . Your tensor core's compute happens in microseconds. The thermal time constant of silicon photonics is comparable. This means **your optical performance changes during the computation itself**, not just between operations.

This is where conventional thermal simulation breaks down. HotSpot and similar tools model steady-state or slow transients. For photonic tensors, you need **coupled electro-thermal-optical simulation** that captures the feedback loop: electrical power → temperature rise → refractive index shift → resonance wavelength drift → heater compensation → more heat.

### AI's Role: Multi-Agent Placement and Fast Thermal Feedback

Recent work directly addresses this. **TDPNavigator-Placer** uses multi-agent reinforcement learning to handle the inherently conflicting objectives of wirelength minimization and thermal management . Instead of collapsing these into a weighted sum (which fails when objectives conflict), it assigns **specialized agents** to each objective with distinct reward mechanisms.

**MaskPlace2.5D** takes a different approach: a hybrid thermal evaluation pipeline that couples a leakage-aware Green's-function solver with a learning-based residual corrector. It achieves **2.20°C mean absolute error** relative to HotSpot with a **2886× speed-up** . That speed-up is what enables dense thermal feedback during RL training — you can't run HotSpot thousands of times per episode.

For your photonic chiplet specifically, you'd extend these frameworks with a photonic-specific thermal model: ring/disk resonator temperature sensitivity (typically ~70–80 pm/°C), heater efficiency as a function of local thermal environment, and crosstalk between adjacent optical devices.

### What Humans Still Own

The **architectural decision of where to place the photonic chiplet** relative to the hot electrical dies. AI can optimize within a given floorplan, but deciding whether to put the photonic die on the edge of the interposer, thermally isolated via deep trenches, or accept close integration for shorter electrical links — that's a system architecture call with no training data.

---

## HBM4 Signal Integrity Constraints

HBM4 targets **12.8 Gbps per pin** with a **2048-bit interface** — double the width of HBM3 . This creates a routing problem where you're packing thousands of single-ended, high-speed signals at near-100% utilization .

### Why Crosstalk Dominates

The data is unambiguous: **crosstalk, not insertion loss, is the limiting factor** for HBM interposer routing . Increasing trace width reduces insertion loss but increases crosstalk by narrowing ground traces. The insertion-loss-to-crosstalk ratio (ICR) correlates directly with eye opening .

A comparison of 4-layer vs. 6-layer organic interposers for HBM4 at 10 Gbps shows the brutal reality :

<table>
  <tr><th>Metric</th><th>4 Signal Layers</th><th>6 Signal Layers</th></tr>
  <tr><td>ICR @ 6.4 GHz</td><td>23.6 dB</td><td>32.9 dB</td></tr>
  <tr><td>Write Eye Width</td><td>34 ps</td><td>60.5 ps</td></tr>
  <tr><td>Read Eye Width</td><td>51.5 ps</td><td>74.5 ps</td></tr>
  <tr><td>Margin vs. JEDEC 30 ps</td><td>-13.67 ps (FAIL)</td><td>+16.83 ps (PASS)</td></tr>
</table>

The 4-layer design **fails HBM4 requirements**. The crosstalk-induced jitter alone is **33 ps** in the 4-layer case, versus **6.5 ps** with 6 layers . You're not debugging a marginal design; you're either building with enough layers or you're not shipping.

### The Shielding Pattern Problem

Standard GSG (ground-signal-ground) routing worked for HBM3 up to 7.2 Gbps. For HBM4, you need denser shielding patterns — **GSG in full-coverage wrap mode** or **GSS (ground-signal-signal) configurations** — that trade routing density for crosstalk suppression . Every shielding trace you add is a routing channel you lose.

### AI's Role: Multi-Objective Routing Optimization

This is where AI-assisted design has **demonstrated, measurable value**.

**Synopsys 3DIC Compiler** integrates an AI engine that sweeps wire width, pitch, shield width, and pattern parameters to find the optimum trade-off between crosstalk and insertion loss . The workflow is: prototype the cross-section → route with 45-degree shielded traces → extract with HFSS-IC → let AI search the parameter space. This happens **before** you commit to a layout, not after.

Research on **ML/DL-based signal integrity optimization for InFO routing** shows **7.69–11.27% improvement in worst-case eye height** for HBM3 test cases by integrating trained models directly into the router . The models are trained on eye diagram simulation results, then used to make routing decisions SI-aware from the start.

For **active crosstalk cancellation**, a 2026 ISSCC paper demonstrates an ML approach that learns cross-channel coupling responses and compensates in the digital domain at the receiver. This achieves **15 dB far-end crosstalk cancellation** with **1.92 pJ/bit** energy efficiency in 16nm FinFET . This is a silicon-level technique, but it changes your interposer routing requirements — if your PHY can cancel 15 dB of crosstalk, you can relax shielding constraints.

### The Manufacturing Reality: Stacked Microvias

Here's the constraint that often gets overlooked in design discussions: HBM4 routing on a 20+ layer HDI board requires **stacked microvias** (3–4 levels deep) for BGA escape routing. These are the **primary yield-limiting process step** in AI server PCB manufacturing .

The reliability requirements are brutal:

- **>95% copper fill density** in each via
- **±15 μm registration accuracy** between stacked vias
- **±5 μm planarization flatness**
- **1000+ thermal cycles** between 23°C and 150°C with <5% resistance increase

A defective stacked microvia fails within **100–200 cycles** . Your interposer design choices directly determine whether your manufacturer can actually build it.

### What Humans Still Own

The **PDN layer count decision**. A 700W accelerator at 0.8V draws approximately **875A peak current**. Delivering this through PCB planes without excessive droop requires **3–4 power planes and 4–5 ground planes** — which is what pushes AI server boards from 16 to 20+ layers . No AI is going to make that call for you; it's a systems engineering decision about voltage droop tolerance, thermal headroom, and cost.

---

## Where This Leaves You

**AI-assisted design earns its keep in:**

- **Placement optimization** with competing thermal and wirelength objectives (multi-agent RL)
- **Fast thermal feedback** during design iteration (Green's-function + ML correctors)
- **Routing parameter optimization** for HBM channels (AI-swept wire/shield configurations)
- **SI-aware routing** with trained ML models predicting eye diagram impact

**Human judgment remains essential for:**

- **Photonic-electronic architectural partitioning** (where the light lives, how thermal isolation is achieved)
- **Laser integration strategy** (external, on-package, or hybrid — this is the hardest component to place )
- **PDN and layer stackup decisions** that determine manufacturing feasibility
- **Bring-up and calibration** of the photonic tuning loops

The deepest challenge is that **your thermal and signal integrity problems are coupled**. The routing patterns that minimize crosstalk (dense shielding) also create thermal barriers. The placement that isolates your photonic die thermally may lengthen electrical links. AI can explore this Pareto surface, but you have to define what "good" means for your specific tensor workload — and that definition doesn't exist in any training set.
