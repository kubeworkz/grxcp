The choice between selling bare chips versus integrated boards isn't just a distribution decision—it's a strategic fork that determines your margin, customer relationships, and long-term defensibility. For a startup building RISC-V + NPU + GPU silicon, the evidence strongly favors moving up the stack toward integrated modules or boards, but the exact level depends on your target customers and go-to-market resources.

### Why the Industry Is Shifting Toward Integration

The semiconductor value chain is undergoing a structural shift: the "integration boundary"—the point where architectural control, commercial accountability, and system validation converge—is moving upward from the die to the multi-die package, board-level module, and even full platform.

The economic logic is stark. A standalone GPU die might sell for a few thousand dollars. A fully integrated AI accelerator module combining compute, HBM, advanced packaging, and system-level validation can sell for orders of magnitude more. The incremental value isn't in more transistors—it's in memory stacking, die-to-die interconnect performance, thermal management, and full-system validation.

This shift concentrates power. Validated modules are harder to replace than dies, qualifications shift from component-level to system-level, warranty liability moves upward, and the integrator captures a larger share of the system BOM. When advanced packaging capacity became constrained during AI demand surges, it was the integrators who controlled allocation and pricing leverage.

### The Three Distribution Models and Their Trade-offs

**Model 1: Bare Chip Sales (Lowest Value Capture, Highest Volume Potential)**

Selling chips alone means your customer (likely an OEM or cloud provider) bears the integration burden. You get per-unit pricing with no margin on memory, packaging, or system validation. Your differentiation must be purely at the silicon level—ISA extensions, NPU efficiency, or RISC-V ecosystem advantages. This model works if you have a truly disruptive architecture that customers must have, but it leaves enormous value on the table and makes you substitutable.

**Model 2: Board-Level Modules (The Strategic Sweet Spot for Startups)**

This means integrating your chip with memory, power management, interface connectors, and reference designs into a functional board or module. The DEEPX case illustrates the power of this approach for a startup: within seven months of mass production, they secured 27 commercial orders across 8 countries, partly by launching an M.2 module embedding their AI chip that could "drop into existing industrial PCs and edge servers" without hardware redesign.

Nvidia's trajectory is the definitive proof point. The company didn't just sell H100 dies—it created the HGX baseboard and DGX systems, capturing system-level value and controlling the integration boundary. This required direct relationships with server ODMs (Foxconn for GPU modules, Wistron for mainboards, Quanta/Gigabyte for HGX systems) and gave Nvidia pricing power and ecosystem control that a bare-die strategy never could.

For your startup, a board-level strategy accomplishes several critical things:

- **Lowers adoption friction**: Developers can prototype immediately without designing a PCB around your chip.
- **Validates the full stack**: You prove that your NPU and GPU cores work with real memory, real power delivery, and real thermal constraints.
- **Captures more BOM**: You sell the memory, the connectors, the power management ICs—not just your silicon.
- **Creates switching costs**: A qualified module is harder to replace than a socket-compatible chip.

**Model 3: Full Systems/Platforms (Highest Value, Highest Capital and Support Burden)**

At the top of the stack, you'd sell complete servers, workstations, or cloud instances. Nvidia's DGX and DGX Cloud represent this model. It requires massive support infrastructure, software stack maturity, and sales resources that most startups cannot sustain. The customers for full systems are often enterprises or governments that need turnkey AI infrastructure—not the hyperscalers who will build their own.

### Software Is the Real Moat, Regardless of Form Factor

The hardware form factor decision is secondary to the software ecosystem you build around it. Nvidia's CUDA is repeatedly cited as the strongest barrier against in-house chip efforts by cloud providers—"until a viable CUDA alternative emerges, the OEMs building their own chips will be limited to controlling only half of this hardware-software combination".

If you sell bare chips, you're implicitly asking customers to write their own software stack. If you sell boards, you can package SDKs, documentation, and support services as part of the solution. If you sell full systems, the software is the product.

Your RISC-V + NPU + GPU combination actually creates a unique software story: a unified open ISA foundation with specialized compute engines. But that story only reaches customers if you package it.

### A Phased Approach for Your Startup

**Phase 1: Developer Kits and Reference Boards.** Ship low-volume, high-touch evaluation boards with a complete SDK. Target AI researchers, edge computing developers, and system architects who need to test your NPU/GPU performance. DEEPX's strategy of engaging ~350 companies with proof-of-concept testing before mass production is a model worth emulating—it shortens the 9-18 month sales cycle.

**Phase 2: Standardized Modules for Volume.** Once you understand customer use patterns, productize specific module form factors (M.2, PCIe, OCP accelerator modules). This is where you capture meaningful margin and build channel relationships with distributors like Avnet, DigiKey, or WPG who can extend your reach globally.

**Phase 3: Selective Platform Plays.** For customers who lack integration capability—industrial, non-hyperscale data centers, government—offer a reference system or "turnkey" cluster. But avoid competing with your OEM customers on full systems unless you have a compelling reason.

### Critical Watchpoints

**Export Controls.** Nvidia's filings show that A100/H100 chips, boards, and systems containing them all fall under US export licensing requirements. Board-level or system-level products face the same or stricter controls than bare chips because they're classified with higher performance thresholds. If you're targeting international markets, this affects your distribution geography and customer qualification.

**Channel Management.** Selling modules through distribution (Avnet, etc.) accelerates reach but requires disciplined inventory management, return policies, and margin structures. Selling directly to hyperscalers gives you better margins but longer sales cycles and higher support costs.

**Customer Concentration Risk.** If you land a hyperscaler, they may demand custom silicon variants or even threaten in-house development. The board/module strategy diversifies your customer base across industrial, edge, and enterprise segments that lack in-house chip design capability.

The bottom line: for a startup with limited capital, the board-level module is the highest-leverage distribution model. It captures system value, validates your full stack, builds switching costs, and enables channel distribution—all without the capital and support burden of full systems. Use developer kits to seed the market, transition to standardized modules for scale, and reserve full platforms for selective strategic accounts where customers cannot integrate themselves.

---

## PTA Integration

**Yes, the PTA can absolutely be added to the board.** In fact, your photonics research aligns with a major industry shift: moving optical technology from data-center interconnects down to the board and chip level. For your phased board strategy, the PTA isn't just a component—it's a powerful differentiator.

### 🔬 PTA Integration: "Chiplet" Model

You shouldn't expect to integrate PTA directly on the same silicon die as the GPU. The realistic approach is a **chiplet or co-packaged optics architecture**:

- **Separate Die, Shared Package**: The PTA (likely on a silicon photonics platform) and your GPU/NPU (on advanced CMOS) are fabricated separately and then co-packaged on a common substrate or interposer.
- **Electrical Interface**: The PTA needs high-speed DACs/ADCs to convert data between the optical compute engine and the electronic GPU/NPU. Your board design must include these electronic interface chips and drivers.
- **Optical I/O**: The board needs a way to get light in and out. This means integrating optical fibers, waveguides, or a detachable optical connector into the board-level module.

### 🎯 Strategic Value for Your Board Business

Adding PTA to your board creates a unique value proposition that differentiates you from just selling a generic RISC-V/GPU board:

- **Solves the "Memory Wall"**: The industry is actively exploring separating GPUs and HBM into different packages connected by optical interconnects. This allows placing HBM further away without latency loss, breaking the "shoreline limit" that restricts how much memory can fit around a GPU. Your PTA board could demonstrate this architecture, allowing customers to attach more HBM or memory banks.
- **New Compute Paradigm**: PTA can perform matrix-vector multiplications in a single clock cycle using multiple dimensions of light (wavelength, spatial mode). For AI workloads, this could mean dramatically higher throughput and energy efficiency compared to purely electronic tensor cores.
- **Low-Latency Deterministic Compute**: For specific operations (like convolutions or recurrent networks), photonic compute can offer very low, deterministic latency. This is attractive for real-time AI inference at the edge or in specialized industrial systems.

### ⚠️ Practical Challenges for Board Integration

While promising, integrating PTA into a board introduces engineering complexity that you must plan for:

- **Thermal Management**: Photonic components are sensitive to temperature. The board design must isolate heat from the GPU/NPU and provide stable thermal control for the PTA.
- **Calibration and Control**: PTA systems require electronic control for phase shifters, modulators, and photodetectors. Your SDK must include calibration routines to compensate for thermal drift and fabrication variations.
- **Packaging and Fiber Coupling**: Getting light into and out of the chip efficiently is a major packaging challenge. The board needs robust, low-loss optical connections (fiber arrays or waveguides) that survive real-world handling.

### 💡 How It Fits Our Phased Strategy

- **Phase 1 (Developer Kits)**: A high-end developer kit with both GPU and PTA would be a research magnet. Researchers could benchmark optical vs. electronic tensor operations, validating the value proposition.
- **Phase 2 (Standardized Modules)**: You could offer a premium module variant that includes the PTA as a specialized co-processor for memory-bandwidth-intensive or low-latency workloads.
- **Phase 3 (Platforms)**: The PTA could enable a unique "optical scale-up" fabric, allowing multiple boards to be connected with very high bandwidth and low latency, which is a key challenge for large AI training clusters.

By integrating PTA into your board-level product, you're not just selling silicon; you're selling a novel system architecture that addresses the industry's most pressing bottlenecks.

---

## Interconnects

For your board-level architecture, the interconnect between the RISC-V CPU, GPU, and PTA should not be a single bus. It needs to be a **hybrid, two-tier interconnect fabric**: an electrical die-to-die (D2D) backbone for coherence and control, with optical interconnects layered on top for the PTA's high-bandwidth data paths.

Here is what that interconnect architecture looks like in practice.

### 🔌 The Electrical Backbone: CPU ↔ GPU

For the CPU and GPU, you need a **coherent, low-latency D2D interconnect**. This ensures they can share memory and perform fast synchronization without the overhead of PCIe. The industry-standard approach for chiplet-based designs is **UCIe (Universal Chiplet Interconnect Express)**.

- **Role of UCIe**: It provides a standardized physical layer, link layer, and protocol layer for chiplets from different vendors to interoperate. You would design both your RISC-V CPU and GPU with UCIe interfaces.
- **Performance Targets**: A well-designed UCIe link can achieve high bandwidth density. For example, recent research demonstrates UCIe-compliant links achieving **1.24 Tb/s/mm** over standard organic packages, with energy efficiency around **1.2 pJ/b**. This is significantly more efficient than PCIe for on-board chip-to-chip communication.
- **Protocol Choice**: You can run **CXL** or a similar coherence protocol over UCIe. This allows the CPU to access GPU memory (and vice versa) coherently, which is critical for AI workloads where the CPU orchestrates data movement and the GPU performs compute.

NVIDIA's NVLink-C2C is the commercial benchmark here, offering **900 GB/s** of coherent bandwidth with up to **6x** better energy efficiency than PCIe Gen 6. You should target a similar order of magnitude for your CPU-GPU link.

### 🔦 The Optical Overlay: Integrating the PTA

The PTA is where the interconnect becomes fundamentally different. Photonic tensor accelerators perform matrix-vector multiplications in the optical domain, but they still need **electronic interface circuits** to convert data between the electrical and optical domains.

- **Electronic-to-Optical Conversion**: The PTA will have **DACs** to drive optical modulators (encoding the input data as light) and **photodetectors** to convert the optical results back into electrical signals for accumulation.
- **The Interface IC**: You need a dedicated **electronic interface chip (EIC)** or a retimer chiplet that sits between your GPU/NPU and the PTA. This EIC handles the high-speed electrical signaling, DAC/ADC conversion, and calibration of the photonic components (like phase shifters to counteract thermal drift).
- **Optical Routing**: The interconnect between the EIC and the PTA (and potentially between multiple PTAs) can be **on-package optical waveguides** or short fiber arrays. A patent from the search results describes exactly this: a **planar optical layer** formed over the electronic dies, using **optical waveguides and photonic devices** to interconnect different semiconductor dies.

### 🏗️ Practical Board-Level Implementation

On your board, this translates to a **2.5D or 3D chiplet package** mounted on the PCB:

1. **Package Substrate/Interposer**: The RISC-V CPU, GPU, and the EIC for the PTA are mounted on a common substrate (e.g., an organic package or a silicon interposer). The CPU↔GPU link runs through this substrate via UCIe traces.
2. **PTA Placement**: The PTA (the silicon photonics chip) is co-packaged on or near the same interposer. The EIC bridges the electrical domain of the GPU to the optical domain of the PTA.
3. **Board-Level Connectivity**: The entire package is then mounted on your board. If you need to connect multiple such packages (e.g., for scale-up), you can use **optical connectors** on the board edge, leveraging the fact that your PTA already has optical I/O. This is exactly what research on UCIe Optical I/O Retimers is exploring, achieving **1.024 Tb/s** per optical port for AI scale-up fabrics.

### 💡 Key Engineering Considerations

- **Thermal Isolation**: Your PTA will be sensitive to thermal gradients from the power-hungry GPU. The board and package design must include thermal isolation or active cooling for the photonic chip.
- **Calibration Infrastructure**: The interconnect fabric must include a control path (likely a low-speed electrical I2C/SPI bus) from the CPU to the EIC/PTA for runtime calibration of the optical mesh.
- **Software Stack**: Our SDK needs to abstract this complexity. The programmer shouldn't know if a tensor operation ran on the GPU or the PTA; the runtime should handle data marshalling over the hybrid interconnect.

In short, the CPU-GPU interconnect is a solved problem you can build with UCIe. The novel part is the **electro-optical bridge** to the PTA, which requires careful co-design of the EIC and the packaging to feed data at rates fast enough to keep the photonic compute engine busy.
