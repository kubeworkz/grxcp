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

### A Phased Approach for Our Startup

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

### 🎯 Strategic Value for Our Board Business

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

In short, the CPU-GPU interconnect is a solved problem we can build with UCIe. The novel part is the **electro-optical bridge** to the PTA, which requires careful co-design of the EIC and the packaging to feed data at rates fast enough to keep the photonic compute engine busy.

---

## UCIe Integration

To use UCIe, both your RISC-V CPU and GPU/PTA chiplets must implement a standardized, three-layer stack. The specification defines the requirements at each layer to ensure interoperability.

Here is a detailed breakdown of the elements each chiplet requires.

### 🔌 Physical Layer (PHY): The Electrical Foundation

This is the analog front-end that handles the actual signaling between dies. Both chiplets need identical PHY implementations to communicate.

- **Data Rates & Signaling**: The PHY must support the agreed-upon data rate per lane. UCIe 1.0 supports up to **32 GT/s**, while UCIe 3.0 extends this to **48 GT/s and 64 GT/s** for high-performance AI/HPC applications. The signaling is **single-ended, unidirectional, Non-Return-to-Zero (NRZ)**.
- **Lane Configuration**: You must define the number of lanes. UCIe supports **16 to 64 lanes** for a module. A UCIe link can be composed of one, two, or four modules. For example, a x16 link uses 16 lanes.
- **Clocking Architecture**: At higher speeds (e.g., 64 GT/s), the PHY requires analog I/O blocks to operate at **16 GHz** when using a quadrature clocking architecture for data sampling.
- **Packaging & Channel**: The PHY design must account for the package type. UCIe defines two primary packaging options:
  - **UCIe-S (Standard Package)**: Uses an organic substrate, supporting longer traces (up to **25 mm**) and wider bump pitch (100-130 µm).
  - **UCIe-A (Advanced Package)**: Uses a silicon interposer or bridge, supporting much shorter traces (up to **2 mm**) and finer bump pitch (25-55 µm), enabling higher bandwidth density.
- **Sideband Channel**: A separate, lower-speed sideband channel is required for control, initialization, status, and management information. It operates independently of the main high-speed data path.

### 🔗 Die-to-Die (D2D) Adapter: The Link Layer

This layer sits between the PHY and the protocol layer. It ensures reliable data transfer over the physical link.

- **Flow Control Unit (FLIT) Management**: The adapter packages data into **FLITs** for transmission. UCIe defines multiple FLIT formats (e.g., 68B, 256B) that include payload, CRC, and header bytes.
- **Error Handling (CRC & Retry)**: To guarantee reliable delivery, the adapter performs **Cyclic Redundancy Check (CRC)** on each FLIT. If an error is detected, a **link-level retry mechanism** retransmits the failed FLIT. This is required when the raw Bit Error Rate (BER) is above **1e-27**.
- **Link State Management**: The adapter manages the link's state machine, including initialization, training, and parameter negotiation with the PHY layer.
- **Protocol Arbitration**: It handles arbitration between different protocols that may be mapped onto the UCIe link, such as PCIe and CXL.

### 📋 Protocol Layer: The Language of the Link

This layer defines how the chiplet's internal applications communicate over the UCIe link. You must choose a protocol that both your CPU and GPU/PTA will use.

- **Supported Protocols**: UCIe is protocol-agnostic but defines standard mappings for:
  - **PCI Express (PCIe)**: For general I/O and configuration.
  - **Compute Express Link (CXL)**: For cache-coherent memory access between the CPU and accelerator. This is critical for your CPU-GPU link to share memory efficiently.
  - **Streaming Protocols**: For raw, low-latency data transfer without the overhead of a full protocol stack. This is ideal for the high-bandwidth data paths to your PTA.
- **SoC Interface**: The protocol layer connects to your chiplet's internal **Network-on-Chip (NoC)**. It supports standard interfaces like **AXI, CXS, and CHI C2C** to link the UCIe controller to the SoC application.

### ⚙️ Software, Manageability, and System-Level Requirements

Beyond the hardware stack, UCIe defines a software model and system-level features for integration and management.

- **Software Model**: UCIe builds on PCIe/CXL software constructs. The link can be managed by firmware (for pre-UCIe OS) or natively by a UCIe-aware operating system. The OS must be able to **discover and enumerate** the UCIe links.
- **Manageability & DFx**: UCIe 2.0 introduced a standardized system architecture for **testability, manageability, and debug (DFx)** across multiple chiplets. This includes a management fabric within each chiplet for telemetry and debug functions.
- **Initialization & Training**: The link initialization process involves a sequence of steps, starting with **sideband initialization (SBINIT)** to establish control communication, followed by **mainband initialization** for the high-speed data path.

### 💡 Specific Considerations for Our PTA Chiplet

Integrating your photonic tensor accelerator (PTA) introduces unique requirements:

- **Electronic-Optical Interface**: The PTA chiplet will need a UCIe PHY, but its primary data path will be optical. The UCIe link would likely be used for **control, configuration, and calibration** of the PTA, while the bulk tensor data flows over the optical interconnect. You will need a dedicated **electronic interface chip (EIC)** to bridge the UCIe electrical domain to the PTA's optical domain.
- **Protocol Selection**: For the PTA, the **Streaming Raw Mode** of UCIe is likely the best fit. This mode provides the lowest-latency path by logically connecting the PHY directly to the application layer, bypassing the FLIT formatting overhead of the D2D adapter.
- **Thermal & Calibration Management**: The UCIe sideband channel can be used to carry the low-speed control signals needed for runtime calibration of the PTA's phase shifters and modulators to counteract thermal drift.

In summary, both your CPU and GPU/PTA chiplets must implement a compatible UCIe PHY, D2D Adapter, and Protocol Layer. The CPU-GPU link would likely use **CXL over UCIe** for cache coherence, while the link to our PTA would use **Streaming Raw Mouide** for low-latency control and data transfer. This layered approach ensures your chiplets can interoperate as a cohesive system.

---

## GPU / PTA Responsibilites

**Question: Will all our GPU matrix multplication work be transfered to the PTA?**

No — not all of them. The PTA will **selectively offload a subset** of matrix multiplications, not replace the GPU’s tensor cores wholesale. Think of our board as a heterogeneous compute system: CPU, GPU, NPU, and PTA each handle what they do best, and the runtime decides where each matmul runs.

### What the PTA can realistically take

Assuming your PTA is an analog/hybrid photonic tensor accelerator, its sweet spot is:

- **Dense, large GEMMs / batched GEMMs** in inference.
- **Low-precision** workloads: INT8, INT4, FP8, or similar — not FP32/BF16 training.
- **Weight-stationary** operations where weights can be preloaded into the photonic mesh and activations streamed through.
- **High data reuse** matmuls where the electro-optical conversion overhead is amortized.
- **Latency-sensitive, deterministic** inference kernels.

Examples: the linear projections in a transformer (QKV, FFN), convolution lowering to GEMM, and some attention matmuls if precision allows.

### What stays on the GPU / NPU

The GPU’s tensor cores remain essential for:

- **Training**: gradient matmuls, backpropagation, optimizer updates. These need high precision and dynamic weights.
- **High-precision inference**: FP32, TF32, BF16, FP64.
- **Sparse, irregular, or small matmuls** where PTA conversion overhead kills the benefit.
- **Non-linear and control ops**: softmax, layernorm, activations, reductions, data movement.
- **Fallback**: any shape or precision the PTA cannot handle correctly or profitably.

Our NPU cores would handle other NN primitives — activation, pooling, quantization, some elementwise ops — depending on your design.

### The key constraint: data movement and precision

A matmul is only worth offloading if:

```plaintext
PTA speedup > electro-optical conversion + data marshalling + synchronization overhead
```

For small matrices, the DAC/ADC, optical modulation, photodetection, and digital correction can easily dominate. For high-precision training, the PTA simply cannot match the numerical fidelity of GPU tensor cores. So the compiler/runtime needs a **cost model** that dispatches each matmul to the right engine.

### What this means for our architecture

- **First board**: Keep the GPU’s tensor cores. The PTA is a co-processor, not a replacement.
- **Software**: We need a unified compiler/runtime (e.g., MLIR-based) that recognizes matmul patterns and routes them to PTA, GPU, or NPU based on shape, precision, and data locality.
- **Interconnect**: UCIe + CXL gives us coherent control and fallback; the optical path handles bulk PTA data. But the PTA still needs a digital fallback path for correctness and calibration.
- **Later revisions**: If the PTA proves itself on inference workloads, we could reduce tensor-core area in a future GPU revision and lean harder on photonics. But that’s a second-generation decision, not a first-board one.

So the correct framing is: **the PTA will absorb some matrix multiplications — primarily low-precision inference GEMMs — while the GPU retains the rest.** It’s an offload engine, not a wholesale transfer.
