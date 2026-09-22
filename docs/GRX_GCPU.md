### 1. Shared physical memory and single address space

**Memory controller**

- There must be one logical DRAM address space shared by CPU and GPU.
- The GPU should not have private HBM/VRAM controllers like a discrete H100.
- Either the whole SoC uses shared HBM stacks, or uses wide LPDDR5X/DDR5, with a single set of memory controllers.

**System address map**

- GPU MMIO registers, doorbells, command queues, and frame buffers are mapped into the same physical address space as CPU DRAM.
- CPU and GPU must agree on addresses for shared buffers.

**Optional system-level cache**

- Apple uses a large SLC between the coherent fabric and DRAM.
- This reduces DRAM traffic for CPU/GPU shared data and improves effective bandwidth.

---

### 2. Coherent interconnect and cache coherence

This is the largest hardware change.

**CPU side**

- RISC-V CPU clusters need to export a cache-coherent interface to the fabric.
- Typical options:
  - AMBA CHI/ACE
  - TileLink Coherent
  - Custom coherent NoC
- CPU L2/L3 must support snoops, sharing, dirty ownership, and atomics from other masters.

**GPU side**

- The GPU L2 cache must become a coherent cache, not a private writeback cache.
- It must implement:
  - Snoop hit handling
  - Clean/dirty/invalid ownership transitions
  - Forwarding dirty lines to the CPU
  - Atomic read-modify-write operations
  - Barrier operations
- An H100-style L2 is not normally designed for this. It assumes it owns local HBM. You have to add coherence state machines and remove/replace the local memory-controller interface.

**Interconnect**

- A coherent NoC or crossbar connects:
  - CPU complexes
  - GPU L2
  - SLC
  - Memory controllers
  - Other accelerators
- Needs a directory/snoop filter for scalability.

---

### 3. Address translation and MMU/IOMMU

Apple-style unified memory usually means shared virtual memory as well as shared physical memory.

**CPU side**

- Standard RISC-V paging: Sv39, Sv48, or Sv57.
- Should support Svpbmt for memory attributes.
- Should support A extension atomics.
- Optional H extension if you need hypervisor isolation.

**GPU side**

- GPU must have an MMU/TLB that understands RISC-V page table format.
- It should share the same process page tables if you want the same pointers to work on CPU and GPU.
- Needs PASID/ASID support for multiple user processes.
- If you do not share page tables directly, then you need an IOMMU/SMMU for GPU DMA translation.

**RISC-V IOMMU**

- Many designs add an IOMMU-compatible block for device isolation.
- It can do per-PASID address translation.
- The GPU can use this for contexts that are not fully shared with the CPU.

**Page faults**

- You must decide:
  - Pin all shared GPU memory, or
  - Support recoverable GPU page faults with HMM/mm_notifier-style kernel integration.
- Apple-style systems generally avoid discrete GPU-style page migration because the memory is physically shared.

---

### 4. Memory consistency and atomics

RISC-V and NVIDIA GPUs have different memory models.

**RISC-V side**

- RVWMO is the default weak memory model.
- Uses FENCE, acquire/release annotations, and LR/SC or AMOs for atomics.

**GPU side**

- NVIDIA H100 uses the PTX memory model.
- Your GPU must map PTX scopes/memory ordering to system-scope RISC-V semantics.
- At minimum, the L2/fabric must support:
  - Acquire/release loads/stores
  - Seq-cst atomics
  - System-scope fences
  - Atomic operations from both CPU and GPU to the same cache line

**Cacheability and attributes**

- Use RISC-V Svpbmt/PMA to mark pages as:
  - Coherent cacheable
  - Non-cacheable
  - Device memory
- GPU must respect those attributes.

---

### 5. Interrupts, doorbells, and synchronization

**GPU to CPU**

- The GPU command processor should use MSIs to the RISC-V interrupt controller.
- Use RISC-V AIA: APLIC/IMSIC or PLIC.
- Doorbell registers should be memory-mapped and cacheable only if hardware supports coherent doorbells.

**CPU to GPU**

- CPU submits command buffers through shared memory.
- Use doorbells to notify GPU.
- Completion can be polled in shared memory or delivered via interrupts.

---

### 6. Security and isolation

**CPU side**

- RISC-V PMP/ePMP to protect firmware/secure regions.
- Possibly RISC-V hypervisor H extension to isolate VMs.

**GPU side**

- Per-process PASID/ASID so user processes cannot access each other’s GPU memory.
- IOMMU/SMMU page tables to restrict GPU DMA.
- GPU command submission must be mediated by the kernel driver.

**System**

- Shared memory buffers should be mapped with correct permissions:
  - CPU virtual mapping
  - GPU page table mapping
  - IOMMU mapping if used
- Avoid giving GPU access to arbitrary physical memory.

---

### 7. OS/driver/runtime software stack

Unified memory is not only hardware.

**Kernel driver**

- Must allocate buffers once and map them to both CPU and GPU.
- Use mechanisms like:
  - dmabuf
  - memfd
  - HMM
  - DRM scheduler
- No PCIe BAR mapping or pinned DMA copies.

**Memory manager**

- If fully hardware-coherent, no explicit cache flushes are needed.
- You may still need memory barriers at the programming-model level.
- If any path is non-coherent, you need cache maintenance operations:
  - RISC-V Zicbom for clean/invalidate.

**Programming model**

- Similar to Metal shared mode or CUDA managed memory, but without page migration.
- CPU and GPU can share pointers.
- The runtime must insert correct fences/barriers.

---

### 8. Performance, QoS, and physical constraints

**Bandwidth**

- An H100-class GPU wants HBM-class bandwidth.
- Apple uses wide LPDDR5X, but for H100-class performance you may need shared HBM stacks.
- CPU and GPU will contend for DRAM bandwidth.

**Memory controller QoS**

- Must arbitrate CPU vs GPU traffic.
- Avoid GPU starving CPU or firmware.
- Usually done with traffic classes and priority levels.

**Power/thermal**

- Shared memory subsystem must be designed for both CPU and GPU worst-case power.
- On-package DRAM helps, like Apple M-series.

---

## Summary

In short, you need:

<table>
  <tr><th>Area</th><th>CPU/SoC side</th><th>GPU side</th></tr>
  <tr><td>Memory</td><td>Shared DRAM controller</td><td>Drop private HBM controllers</td></tr>
  <tr><td>Coherency</td><td>Export CHI/ACE/TileLink</td><td>Add coherent L2, snoops, ownership</td></tr>
  <tr><td>MMU</td><td>Sv39/Sv48/Sv57 + IOMMU</td><td>RISC-V page table walker + PASID</td></tr>
  <tr><td>Memory model</td><td>RVWMO + atomics</td><td>System-scope ordering + atomics</td></tr>
  <tr><td>Interrupts</td><td>AIA/PLIC</td><td>MSI/doorbell support</td></tr>
  <tr><td>Software</td><td>dmabuf/HMM/driver</td><td>Shared virtual memory runtime</td></tr>
  <tr><td>Physical</td><td>Shared NoC, SLC, QoS</td><td>Bandwidth/latency tuning</td></tr>
</table>

If you cannot make the GPU L2 fully coherent, you can still do zero-copy unified memory with explicit flush/invalidate, but that is not Apple-like. Apple’s approach requires hardware cache coherence, shared page tables, and a single shared physical memory system.

---

## Pragmatic View

That’s a very pragmatic view. The industry is indeed moving toward chiplet-based, cache-coherent heterogeneous systems, but there’s no single “Apple way” that fits everyone. Since you’re building on RISC‑V and a custom H100‑inspired GPU, you have a lot of freedom to define your own path. The key is to pick the right balance between hardware complexity, software effort, and performance for your target workloads.

Here are some directions and design considerations that might suit a more independent approach:

---

### 1. **Chiplet + coherent fabric instead of monolithic SoC**

You mentioned Intel using NVIDIA chiplets with NVLink, AMD’s MI300, etc. For a RISC‑V + custom GPU, you could:

- Use **UCIe** or **BoW** for die‑to‑die connectivity, with a coherent protocol layered on top (e.g., CXL.cache or a custom CHI/TileLink over UCIe).
- Keep the CPU and GPU on separate dies but share a common memory controller and coherent fabric via an interposer or advanced packaging.
- This gives you:
  - Die reuse: same GPU die can be paired with different CPU/accelerator dies.
  - Yield and cost advantages.
  - Ability to use different process nodes for CPU vs GPU.

But **you still need coherence across the chiplet boundary**, so the GPU L2 and CPU must implement a coherent protocol. You can choose:

- **CXL 3.0 / CXL.cache**: industry standard, but may have overhead for fine‑grained GPU sharing.
- **CHI** (Arm’s coherent hub interface): widely used, good ecosystem, but licensing may be a concern.
- **TileLink** (RISC‑V native): fully open, used by SiFive, Berkeley, etc. You can extend it for GPU needs.
- **Custom protocol** over UCIe: gives ultimate freedom but more verification effort.

---

### 2. **Hybrid memory model – not all memory is equal**

Apple’s unified memory means *one pool* of DRAM with no VRAM. But for an H100‑class GPU, you may need HBM‑class bandwidth that is hard to share with a CPU without huge power/cost. A hybrid approach could be:

- **Coherent shared memory region** (e.g., 16–64 GB of LPDDR5X or HBM) for CPU‑GPU shared data, with full hardware coherence.
- **Optional private high‑bandwidth memory** (e.g., separate HBM stacks) for GPU‑only scratchpads, textures, or streaming data.
- The OS/runtime decides placement: data that needs frequent CPU access goes to the shared pool; GPU‑only data goes to private HBM.

This gives you much of the programmability benefit of unified memory while avoiding the need to make *all* GPU memory coherent and shared. It’s similar to NVIDIA’s “managed memory” but without page migration – you just allocate from different pools.

The hardware complexity is lower because the private GPU memory path can remain non‑coherent, and only the shared region needs L2 coherence.

---

### 3. **Start with software‑managed coherence, then add hardware later**

If full hardware coherence is too big a first step, you can design the GPU L2 to support *selective* coherence:

- Add a **coherent mode** for certain address ranges (e.g., marked by page tables or MMIO control).
- In coherent mode, L2 performs snoops and directory lookups.
- In non‑coherent mode, L2 behaves like a traditional GPU cache, and the driver issues explicit cache flushes/invalidates (using RISC‑V Zicbom or custom instructions) when ownership changes.

This lets you bring up the system, validate performance, and then gradually move more memory into the coherent domain. Many early unified‑memory systems (e.g., AMD’s first APUs) used similar tricks.

---

### 4. **Leverage RISC‑V’s openness for custom coherence and memory attributes**

Because RISC‑V is open, you can:

- **Define custom PMA/PMA‑like attributes** for GPU‑specific memory types (e.g., write‑through for command buffers, streaming, etc.).
- **Add custom instructions** to the CPU for fast cache maintenance, atomics with GPU scope, or doorbell acceleration.
- **Implement a RISC‑V IOMMU** that is tightly coupled with the GPU’s MMU, so both use the same page table format and PASID space.
- **Experiment with a directory‑based coherence protocol** that is optimized for CPU‑GPU sharing patterns, not just generic multi‑core.

This gives you a differentiation opportunity that closed ISAs don’t offer.

---

### 5. **Prioritize the software stack early**

No matter how elegant the hardware, the software will make or break the “unified memory” experience. Plan for:

- **A common memory allocator** that can place buffers in shared or private pools and map them with correct attributes.
- **User‑space libraries** that expose shared pointers and insert the right fences (e.g., a RISC‑V analog of CUDA’s `cudaMemAdvise` or Metal’s `MTLResourceStorageModeShared`).
- **Kernel drivers** that use `dmabuf`, `memfd`, or a custom `/dev` interface to manage physical memory and IOMMU mappings.
- **Debug and profiling tools** that can visualize coherence traffic and identify unnecessary flushes.

---

### 6. **Consider the target workloads**

The optimal design depends heavily on what you’re running:

- **AI training/inference**: GPU does most of the work, CPU only feeds commands and occasionally inspects results. You might get away with mostly private GPU memory and a small coherent region for control.
- **Graphics / gaming**: CPU and GPU share vertex buffers, textures, and command buffers frequently; benefits from full coherence and low‑latency shared memory.
- **HPC / scientific**: mixed – some data is CPU‑produced, some GPU‑produced; often large streaming datasets; software‑managed coherence can work well.

Your “own path” could be a **configurable coherence domain** – e.g., the GPU L2 can be configured at boot to treat X% of memory as coherent and the rest as private, depending on the application.

---

## Final thought

You don’t need to copy Apple. You have the chance to build something more modular – a chiplet‑based system where the RISC‑V CPU and your H100‑inspired GPU communicate over a coherent protocol you control, with a memory architecture that can scale from embedded‑style unified to HPC‑style partitioned. Start with a small coherent region, make the protocol extensible, and lean on RISC‑V’s openness to add exactly the features you need.

---

## Deeper Dive

Let’s dive into a modular, chiplet-based architecture where you control the coherence protocol and can scale the memory model from embedded-style fully unified to HPC-style partitioned. This approach lets you start small, validate the design, and grow the coherent domain as needed.

---

## 1. System Topology: Chiplets and Interconnect

**Core idea:**

Build the CPU complex and GPU as separate chiplets connected by a high‑bandwidth, low‑latency die‑to‑die link that carries a coherent protocol. Keep a central memory controller (or multiple) on a separate die or integrated on one chiplet. Use a common fabric that can route snoops and data.

**Physical interconnect options:**

- **UCIe** (Universal Chiplet Interconnect Express): Industry standard, supports multiple protocols (CXL, PCIe, streaming). You can run a custom coherent protocol on top of UCIe’s raw die‑to‑die transport.
- **BoW** (Bunch of Wires): Simpler, open, often used in research. Good for prototyping if you can control the PHY.
- **Custom SerDes/parallel interface**: If you have the IP and want lower latency, you can design a parallel interface on an interposer.

For a coherent link, you need reliable, ordered, low‑latency delivery. UCIe’s “streaming” mode can be a good base; you then define message types for coherence.

**Fabric topology:**

Start with a simple crossbar or ring connecting CPU, GPU, and memory controller(s). For more chiplets later, you can add a mesh or a hierarchical NoC. The initial design can be a **shared bus or crossbar** with a central directory/snoop filter.

**Memory controller placement:**

- **Centralized**: One or more memory controllers on a separate die or integrated into the CPU die. GPU accesses memory through the coherent fabric. This is easiest for coherence but may add latency for GPU.
- **Distributed**: Memory controllers near each compute die, with a global address map. More complex but can provide higher bandwidth. Start centralized.

---

## 2. Choosing a Coherent Protocol You Control

You have several options, each with tradeoffs:

<table>
  <tr><th>Protocol</th><th>Openness</th><th>Ecosystem</th><th>Complexity</th><th>Suitability</th></tr>
  <tr><td>TileLink</td><td>Fully open (RISC‑V)</td><td>Growing, used in SiFive, lowRISC</td><td>Medium</td><td>Ideal if you want RISC‑V native, no licensing. Can extend for GPU needs.</td></tr>
  <tr><td>CHI (Arm)</td><td>Requires license</td><td>Mature, many IP blocks</td><td>High</td><td>Good if you already have CHI‑based IP. But licensing may restrict.</td></tr>
  <tr><td>CXL.cache</td><td>Standard, but more PCIe‑centric</td><td>Strong in data center</td><td>High</td><td>Too heavyweight for on‑package, but could be used if you plan external memory pooling.</td></tr>
  <tr><td>Custom</td><td>Fully controlled</td><td>None</td><td>Highest</td><td>Maximum flexibility, but you must define everything.</td></tr>
</table>

**Recommendation:**

Use **TileLink** as a starting point. It is open, designed for RISC‑V, and supports multiple cache levels and atomics. You can extend it with GPU‑specific messages (e.g., “clean‑shared”, “flush‑line‑to‑memory”, “barrier‑with‑scope”). Because TileLink is an open standard, you can add custom opcodes without breaking compatibility. Later, if needed, you can map TileLink to a physical UCIe layer.

> **Superseded for the development board, 2026-09-21** ([`board_program_plan.md`](designs/board_program_plan.md), B1 and B3). The board joins the GRX930 and the GRX-G100 with CXL 2.0 over PCIe 5.0, with the GRX930 as the host and the GRX-G100 as a Type-2 device. Two things changed the answer. On the board's first revision the CPU and the GPU sit in separate packages (B1), which is where CXL is at home, so the objection above — too heavy for on-package — no longer applies. And a standard edge buys standard enumeration and drivers, controller and PHY IP that can be licensed, and chiplets that can be swapped. Inside each chip the fabric stays its own, and the extensions below remain candidates for it. The hybrid memory model of the Pragmatic View, a coherent shared region beside private GPU memory, stands.

**Example custom extensions to TileLink:**

- **Range‑based clean/invalidate** for non‑coherent to coherent transitions.
- **GPU‑scope fence** that orders all outstanding GPU transactions before a CPU read.
- **Atomic RMW with return data** that both CPU and GPU can use on shared queues.
- **Cache hints** (e.g., “streaming”, “write‑through”) attached to requests.

---

## 3. Starting Small: A Configurable Coherent Region

You don’t need to make all memory coherent immediately. Instead, define a **memory map** with one or more coherent windows.

### Address partitioning

- Reserve a fixed physical address range, say **0x8000_0000 – 0x8FFF_FFFF** (256 MB) as the initial **coherent shared region**.
- All other memory is either CPU‑private or GPU‑private (non‑coherent). The GPU can still access CPU DRAM via explicit DMA, but not with hardware coherence.
- Use a **base‑and‑bound register** or a set of address decoders in both CPU and GPU to mark the coherent region. This lets you resize the region by changing a register at boot or runtime.

### GPU L2 cache behavior for coherent region

- For addresses inside the coherent region, the GPU L2 must participate in coherence.
- A simple approach: **snoop‑all** or **directory‑based**.
  - *Snoop‑all*: For each cache miss in the coherent region, broadcast a snoop to the CPU L2 and any other coherent caches. Works for small systems (2–4 chiplets), no directory needed.
  - *Directory*: Maintain a small directory (e.g., in the memory controller or SLC) tracking which caches hold each line. Scales better. Start with a simple full‑map directory (one entry per cache line in coherent region) if the region is small. For a 256 MB region with 64‑byte lines, you need 4M entries – maybe too much; use a sparse directory or limit to a subset.

**Recommendation:**

Start with **snoop‑filter/directory at the memory controller** that covers only the coherent region. The directory can be a simple set‑associative structure. For 256 MB, you could have 256K lines; a 16‑way set‑assoc directory with 4096 sets would cover 64K lines—not enough for full coverage. Instead, you can use a **coarse‑grained directory** (track blocks of 4 KB) to reduce area.

### CPU side

- RISC‑V cores already support cache coherence via the L2 controller (if using a coherent hub like TileLink). The CPU L2 must be able to snoop incoming requests and respond with data or ownership.
- Enable the RISC‑V **Zicbom** extension for cache management operations to help with non‑coherent regions if needed.
- The CPU’s page tables must mark coherent‑region pages as cacheable and shareable. For RISC‑V, use the **Svpbmt** extension to set memory attributes (e.g., coherent, non‑coherent, device).

### Transitions between coherent and non‑coherent

When the kernel or runtime wants to move a buffer into the coherent region:

1. Allocate pages from the coherent region.
2. Flush the buffer from GPU L2 (if it was previously used non‑coherently) using a range‑based clean/invalidate.
3. Map those pages with coherent attributes in both CPU and GPU page tables.
4. From then on, hardware coherence handles synchronization.

This “grow as needed” approach lets you start with embedded‑style unified memory for a subset of data and expand later.

---

## 4. GPU L2 Cache Modifications for Partial Coherence

Your H100‑inspired GPU L2 is likely designed as a private, write‑back cache. To support a coherent region, you need to add:

- **Coherence state bits** per line: MESI, MOESI, or TileLink states (e.g., None, Branch, Trunk, Tip). For simplicity, MESI is enough.
- **Snoop handling logic**: When a snoop arrives (e.g., from CPU), the L2 must look up the tag, return data if dirty, change state, and possibly write back to memory.
- **Atomic operation support**: Both CPU and GPU may issue atomics to the same line. The L2 must either forward atomics to the memory controller (which can act as the serialization point) or handle them locally if it owns the line.
- **Barrier/fence handling**: The L2 must respond to coherence barriers (e.g., “ensure all prior writes visible”) by draining or acknowledging.

**Implementation strategy:**

- Add a **coherent mode bit** in the L2 tag or address decoder. For non‑coherent accesses, bypass coherence logic entirely (zero overhead).
- For coherent accesses, redirect requests through the coherence controller. Use a **snoop queue** to process incoming snoops without stalling the main pipeline.
- If the coherent region is small, you could even use a **separate small coherent cache** in front of the main L2, but that adds complexity.

---

## 5. Leveraging RISC‑V Openness for Custom Features

Because you control both CPU and GPU, you can define custom instructions and CSRs that make the software/hardware interface more efficient.

**Examples:**

- **Custom cache maintenance instructions** for range‑based clean/invalidate (like ARM’s DC CVAU but for arbitrary GPU‑owned lines).
- **GPU doorbell acceleration**: A custom CSR that, when written, sends a message directly to the GPU command processor without going through memory.
- **Fence with scope**: A `FENCE.GPU` instruction that ensures all prior CPU writes are visible to the GPU before the GPU starts a kernel.
- **PMA/PMA‑like attributes** that encode “coherent with GPU” so the CPU page tables can signal to hardware to use the coherent path.

The RISC‑V ISA allows custom opcodes and CSRs in the “custom” ranges. You can implement these in your CPU core.

---

## 6. Software Stack for Modular Coherence

### Kernel driver

- Manage the coherent region as a reserved memory pool (e.g., via `memmap` boot parameter or device tree).
- Provide an API to allocate from the coherent region (`mmap` with a special flag).
- For non‑coherent buffers, use existing DMA APIs with explicit cache maintenance.
- Implement an **IOMMU** (e.g., RISC‑V IOMMU spec) so the GPU can access CPU virtual addresses safely.

### Runtime / user space

- Expose two memory types:
  - **Coherent shared**: `malloc_coherent()`, `map_gpu()`, `unmap_gpu()`.
  - **Private GPU memory**: allocated from GPU‑private DRAM, accessed via explicit copies or zero‑copy with flushes.
- Provide a **fence API** that maps to the appropriate hardware barrier for the region type.
- For HPC workloads, you can add a **memory advisor** (like CUDA `cudaMemAdvise`) that lets the application hint whether a buffer will be read/written by CPU or GPU frequently, allowing the driver to move it between coherent and non‑coherent pools.

### Debug/tracing

- Add performance counters for coherence traffic (snoops, directory hits, writebacks).
- Use these to identify hotspots where the coherent region is too small or where false sharing occurs.

---

## 7. Scaling Up: From Embedded to HPC

Your initial design can be small (e.g., 4‑core RISC‑V + GPU, 8 GB shared LPDDR5). As you gain confidence, you can scale in multiple dimensions:

- **Larger coherent region**: Increase the address decoder range; if using a directory, you can expand it or switch to a more scalable directory (e.g., limited pointers + overflow to memory).
- **More chiplets**: Add another CPU chiplet or an accelerator. The same coherent protocol can be used; just add another snooper to the broadcast or another entry in the directory. TileLink supports multi‑master easily.
- **Mixed memory**: Add a separate HBM stack for GPU‑private memory while keeping the coherent region on LPDDR5X. The memory controller can route based on address.
- **CXL for expansion**: Later, you could expose a CXL.mem interface to external memory pools, treating them as non‑coherent or semi‑coherent.

The key is that your **protocol is extensible** because you control it. You can add new message types (e.g., for HBM‑specific flushes) without breaking existing IP.

---

## 8. Challenges and Mitigations

<table>
  <tr><th>Challenge</th><th>Mitigation</th></tr>
  <tr><td>Snoop latency from CPU to GPU</td><td>Use a large SLC (system‑level cache) between chiplets to absorb misses. Place directory near memory controller.</td></tr>
  <tr><td>GPU L2 complexity</td><td>Implement coherence only for a small set of lines; use a separate coherent tag array and bypass for non‑coherent.</td></tr>
  <tr><td>Software bugs in coherence</td><td>Provide a debug mode where all coherent traffic is logged; use formal verification on the protocol state machine.</td></tr>
  <tr><td>Power overhead of snoops</td><td>Use a snoop filter to reduce broadcasts. Start with snoop‑all for small systems, then add filter.</td></tr>
  <tr><td>Page table sharing</td><td>Initially use a dedicated GPU page table managed by the kernel; later add hardware page table walker that understands RISC‑V format.</td></tr>
</table>

---

## Summary: Your Custom Modular Path

You are not constrained to Apple’s monolithic unified memory or NVIDIA’s discrete model. By designing a chiplet system with a coherent protocol you control, you can:

- Start with a **small coherent region** for CPU‑GPU sharing.
- Keep the rest of the memory **partitioned** for performance.
- **Extend** the protocol as needed using RISC‑V’s openness.
- **Scale** from embedded to HPC by adding chiplets, memory types, and coherence domains.

This approach gives you a clear evolution path: validate coherence on a small scale, then gradually increase the coherent footprint while optimizing for your specific workloads. The key is to keep the protocol flexible and the hardware modular.

---

Exploring novel CPU‑GPU communication methods beyond standard coherent buses and shared memory is a great way to differentiate your RISC‑V + H100‑inspired SoC. Given your modular chiplet approach, you have the freedom to mix and match these ideas. Here are several innovative directions, ranging from near‑term practical to long‑term research.

---

## 1. High‑Speed Serial Links with Memory Semantics (CXL, UCIe, NVLink)

While not entirely novel, the *way* these are used is evolving. Instead of treating them as just a PCIe replacement, you can:

- **Use CXL.cache and CXL.mem** between CPU and GPU chiplets to create a *coherent memory pool* without a monolithic fabric.
- **Run a lightweight, custom coherence protocol over UCIe’s streaming mode** – this gives you serial links that carry cache‑line requests, snoops, and data. The novelty is that you can define your own message set, latency budgets, and flow control, optimized for CPU‑GPU sharing rather than generic server workloads.
- **Direct cache‑to‑cache transfers over serial lanes**: both CPU and GPU L2 caches can exchange lines via the serial link *without going to memory*, reducing DRAM traffic and latency. This is effectively a “serial snoop bus”.

---

## 2. Dedicated “Core‑to‑Core” Fine‑Grained Communication Links

Instead of all communication going through shared memory or the memory hierarchy, you can add a **narrow, low‑latency, point‑to‑point serial or parallel link** directly between CPU cores and GPU streaming multiprocessors (SMs).

**How it works:**

- Each CPU core (or a small cluster) has a mailbox that can send small messages (e.g., 64 bytes) directly to a specific GPU SM, bypassing L2, the fabric, and DRAM.
- Used for synchronization, user‑level messaging, or dependency tokens.
- Can be implemented as a set of dedicated FIFOs with credit‑based flow control over UCIe sideband or a custom link.

**Benefits:**

- Extremely low latency (tens of nanoseconds) for control messages.
- Avoids cache pollution and coherence overhead for small, frequent notifications.
- Complements the shared memory model – data goes through memory, but control signals go through the fast link.

**Example:** The CPU sends a “kernel launch” command with pointer arguments directly to the GPU command processor via the link; the GPU responds with completion interrupts on the same link.

---

## 3. Message‑Passing / Mailbox with Hardware Acceleration

Traditional doorbells are just MMIO writes. You can extend this into a **hardware message queue engine**:

- CPU writes a message descriptor to a circular buffer in shared memory, then writes a doorbell to a hardware queue manager.
- The queue manager (on the GPU or fabric) fetches the descriptor, decodes it, and dispatches work to GPU engines.
- Completion is signaled by the GPU writing back to a response queue, which triggers an interrupt or a polling flag.

**Novel twist:** Make the queue manager *coherent* and *user‑space programmable*. Instead of kernel involvement, user processes can enqueue work directly, and the hardware queue manager handles address translation and protection. This reduces latency and CPU overhead.

---

## 4. Dataflow / Streaming Interfaces

For AI and HPC workloads, data often flows from CPU to GPU in a pipeline. You can design a **streaming fabric** that moves data directly from CPU caches (or last‑level cache) to GPU L2 or even into the GPU’s tensor cores.

**Implementation ideas:**

- **Cache‑to‑cache streaming**: The CPU’s last‑level cache (LLC) can be configured to push dirty lines to the GPU’s L2 as soon as they are written, rather than waiting for a snoop. This is like a hardware‑managed producer‑consumer buffer.
- **Direct tensor core feeding**: A dedicated high‑bandwidth path connects the CPU’s vector unit output to the GPU’s systolic array input. This is similar to how some accelerators are attached to CPU data paths (e.g., Intel AMX).

**Benefit:** Eliminates the latency of writing to DRAM and reading back, especially for small/medium data chunks that would otherwise cause cache thrashing.

---

## 5. Optical Interconnects on Package

While still early for mainstream, **silicon photonics** or **micro‑ring resonators** can provide enormous bandwidth per pin with lower energy per bit than electrical serial links.

- You could use optical links between CPU and GPU chiplets, especially if they are on separate packages or in a multi‑chip module.
- Optical links can carry the same coherent protocol but at higher bandwidth and lower latency for long distances.
- Research prototypes (e.g., from Ayar Labs, Intel) show feasibility for chip‑to‑chip optical I/O.

**Challenges:** Cost, thermal sensitivity, and integration complexity. But if you are building a high‑end SoC, this could be a differentiator.

---

## 6. Wireless Chip‑to‑Chip Communication (mm‑Wave)

A more radical idea: use **millimeter‑wave wireless transceivers** integrated on the package to communicate between CPU and GPU dies without physical wires.

- Researchers have demonstrated 100+ Gbps wireless links over a few centimeters using on‑chip antennas.
- Advantages: no interposer routing, flexibility in chip placement, and potential for reconfigurable topologies.
- Disadvantages: interference, power, and reliability concerns.

This is still in the research phase, but if you want to be truly novel, it’s an option to explore in parallel with more conventional approaches.

---

## 7. Shared Register Files / Scratchpad Memory

Instead of sharing main memory only, you can share a **small, fast on‑chip memory** between CPU and GPU, accessible by both with very low latency.

- For example, a **shared L1 scratchpad** that is part of the coherent fabric but optimized for single‑cycle access from both CPU and GPU SMs.
- The CPU can write directly into the GPU’s shared memory (like CUDA `__shared__`) and the GPU can read it without going through L2.
- This is similar to how some heterogeneous SoCs (e.g., TI’s DSP+ARM) use shared on‑chip RAM.

**Hardware:** Add a small SRAM block with dual‑port access from CPU and GPU, integrated into the coherence domain but with lower latency than DRAM.

---

## 8. Instruction‑Level Offload (Tightly Coupled Coprocessor)

Instead of treating the GPU as a separate device, you can integrate it as a **RISC‑V coprocessor** that is invoked via custom instructions.

- Add custom RISC‑V instructions like `gpu.mac` or `gpu.launch` that directly trigger operations on the GPU pipeline.
- The CPU’s instruction stream can issue work to the GPU without any memory‑mapped I/O or driver overhead.
- This is akin to how vector units or matrix engines are integrated in some CPUs (e.g., ARM SVE, RISC‑V V extension), but extended to a full GPU.

**Benefit:** Extremely low latency for small tasks; fine‑grained interleaving of CPU and GPU work.

**Challenge:** The GPU is not a simple functional unit; you need to manage thousands of threads. A hybrid approach could be used: custom instructions to enqueue GPU work into a hardware queue, with the GPU executing asynchronously.

---

## 9. Remote Atomic Operations over the Fabric

Standard coherence already supports atomics, but you can enhance this with **hardware‑accelerated remote atomics** that perform operations directly at the memory controller or in the GPU L2.

- Instead of CPU reading a counter from GPU memory, incrementing, and writing back, the CPU can issue an atomic add that is executed *inside* the GPU L2 or at the memory controller.
- This reduces latency and contention for synchronization primitives like spinlocks, barriers, and work queues.

**Novelty:** Extend RISC‑V’s atomic instructions (LR/SC or AMOs) with a “remote scope” that is recognized by the coherence fabric and executed near the target.

---

## 10. Compression‑Aware Data Transfer

If your workload has compressible data, you can add **hardware compression/decompression engines** in the CPU‑GPU communication path.

- The CPU compresses data before sending to GPU (or vice versa) over a serial link or through shared memory.
- The GPU decompresses on the fly, effectively increasing bandwidth at the cost of some latency and power.
- This is especially useful for sparse data, images, or deep learning activations.

**Implementation:** Add a block‑based compressor (e.g., LZ4, or a custom lightweight algorithm) between the L2 and the fabric. Address ranges can be marked as “compressed” via page table attributes.

---

## 11. Asymmetric Communication via Copy Engines

While not novel per se, you can offload data movement to dedicated **copy engines** that operate asynchronously and can overlap with computation.

- CPU writes a list of copy descriptors to a queue.
- A copy engine (on the GPU or fabric) moves data between CPU memory, GPU memory, and shared memory without involving the CPU cores.
- You can add **bidirectional copy engines** that also perform simple transformations (e.g., transpose, scatter/gather) during transfer, which is useful for AI data layouts.

**Novel twist:** Make the copy engine *coherent* – it understands page tables and can handle page faults, so it can safely touch CPU user pages without pinning.

---

## 12. Shared Virtual Memory with Hardware Page Fault Handling

Move beyond traditional unified memory by allowing **both CPU and GPU to share the same virtual address space** with hardware‑managed page faults.

- The GPU MMU can raise page faults and the OS can service them without killing the process, similar to how modern GPUs handle demand paging.
- This allows on‑demand paging of GPU data from CPU memory or even SSD, reducing memory footprint.
- The RISC‑V IOMMU and advanced interrupt controllers can support this.

**Benefit:** Truly seamless programming model; no need for explicit memory management.

---

## Summary Table

<table>
  <tr><th>Approach</th><th>Novelty</th><th>Complexity</th><th>Best For</th></tr>
  <tr><td>Custom coherence over serial links</td><td>Medium</td><td>Medium</td><td>Chiplet scaling</td></tr>
  <tr><td>Core‑to‑core fine‑grained links</td><td>High</td><td>Low‑Med</td><td>Synchronization, control</td></tr>
  <tr><td>Hardware queue manager</td><td>Medium</td><td>Medium</td><td>Command submission</td></tr>
  <tr><td>Dataflow streaming</td><td>High</td><td>High</td><td>AI/HPC pipelines</td></tr>
  <tr><td>Optical interconnects</td><td>High</td><td>High</td><td>Bandwidth, power</td></tr>
  <tr><td>Wireless mm‑wave</td><td>Very High</td><td>Very High</td><td>Research, flexibility</td></tr>
  <tr><td>Shared scratchpad</td><td>Medium</td><td>Low‑Med</td><td>Low‑latency sharing</td></tr>
  <tr><td>Instruction‑level offload</td><td>High</td><td>High</td><td>Fine‑grained GPU tasks</td></tr>
  <tr><td>Remote atomics</td><td>Medium</td><td>Low‑Med</td><td>Synchronization</td></tr>
  <tr><td>Compression‑aware transfer</td><td>Medium</td><td>Medium</td><td>Bandwidth‑limited data</td></tr>
  <tr><td>Coherent copy engines</td><td>Medium</td><td>Medium</td><td>Data movement</td></tr>
  <tr><td>Shared virtual memory with faults</td><td>Medium</td><td>High</td><td>Ease of programming</td></tr>
</table>

## Building Your Own Path

The beauty of RISC‑V and a custom GPU is that you can **combine several of these ideas**. For example:

- Use a **UCIe serial link with a custom TileLink‑like coherent protocol** for cache‑coherent shared memory.
- Add a **narrow core‑to‑core sideband link** for low‑latency doorbells and messages.
- Include a **shared on‑chip scratchpad** for very hot data.
- Later, integrate **remote atomics** and **hardware queue managers** as you refine the design.

Start with the coherent region approach we discussed earlier, then layer on one or two novel communication mechanisms that match your target workloads. This modular, incremental path reduces risk while keeping the door open for innovation.

---

Great – focusing on the next 3 months with concrete, actionable tasks will help both teams move forward quickly. Since you’re an open‑source project, we’ll prioritise open standards (TileLink, RISC‑V extensions, UCIe where feasible) and open‑source tooling (Chipyard, RocketChip, gem5, Verilator, etc.). The tasks below are designed to be started immediately and produce tangible deliverables (reports, design specs, simulation models, or code prototypes).

---

## 1. RISC‑V CPU Team – 3‑Month Investigation & Design Tasks

### Month 1 – Foundations & Standards

<table>
  <tr><th>Task</th><th>Description</th><th>Deliverable</th></tr>
  <tr><td>R1.1</td><td>Evaluate coherent interconnect options: TileLink, AMBA CHI/ACE, OpenPiton’s NoC, CXL.cache (openness, licensing, ecosystem). Focus on TileLink’s suitability for GPU extension.</td><td>Comparison report with pros/cons, licensing constraints, and recommendation.</td></tr>
  <tr><td>R1.2</td><td>Study RISC‑V memory model &amp; extensions: Svpbmt, Zicbom, Zicboz, atomics (A extension), fence instructions. Map these to CPU‑GPU shared memory needs.</td><td>Internal wiki page / markdown doc summarising required ISA features.</td></tr>
  <tr><td>R1.3</td><td>Define initial system address map for a chiplet‑based SoC: reserve a configurable coherent region (e.g., 256 MB) and separate non‑coherent CPU/GPU private regions. Include MMIO, doorbells, and interrupt spaces.</td><td>Address map diagram + description, with base addresses and sizes.</td></tr>
  <tr><td>R1.4</td><td>Set up a baseline simulation environment: use Chipyard or RocketChip to generate a RISC‑V core with TileLink, boot Linux, and run a simple multi‑core test.</td><td>Working simulation repo with instructions.</td></tr>
</table>

### Month 2 – Coherence & Custom Extensions

<table>
  <tr><th>Task</th><th>Description</th><th>Deliverable</th></tr>
  <tr><td>R2.1</td><td>Propose custom TileLink message extensions for GPU coherence: range‑clean, GPU‑scope fence, remote atomics with return, cache hints. Ensure they are backward‑compatible with standard TileLink.</td><td>Design document with opcode/field definitions and use cases.</td></tr>
  <tr><td>R2.2</td><td>Design a snoop filter / directory for the coherent region. Start with a simple full‑map directory (if region small) or a set‑associative directory. Define its interface to the memory controller and TileLink.</td><td>Architectural spec + RTL pseudocode or Chisel sketch.</td></tr>
  <tr><td>R2.3</td><td>Investigate RISC‑V IOMMU specification (latest draft). Plan how GPU will share CPU page tables or use IOMMU for isolation. Decide on PASID handling and fault model.</td><td>IOMMU integration proposal (can be high‑level).</td></tr>
  <tr><td>R2.4</td><td>Modify the simulated RISC‑V core to recognise custom PMA/PMA attributes for the coherent region, and implement Zicbom cache maintenance ops in simulation.</td><td>Patches to Chipyard/RocketChip, tested in sim.</td></tr>
</table>

### Month 3 – Chiplet & Interconnect Exploration

<table>
  <tr><th>Task</th><th>Description</th><th>Deliverable</th></tr>
  <tr><td>R3.1</td><td>Investigate UCIe: openness of spec, available open‑source PHY/controller implementations, and how to map TileLink onto UCIe’s streaming or flit mode.</td><td>Feasibility report with potential open‑source components.</td></tr>
  <tr><td>R3.2</td><td>If UCIe not immediately viable, define a simpler parallel die‑to‑die interface (e.g., using TileLink directly over an interposer) as a fallback.</td><td>Interface spec with signal list and timing.</td></tr>
  <tr><td>R3.3</td><td>Develop a test plan for CPU‑GPU coherence: define a set of microbenchmarks (cache line sharing, atomics, fences, producer‑consumer) that will run in simulation once GPU model is integrated.</td><td>Test specification document.</td></tr>
</table>

---

## 2. GPU Team – 3‑Month Investigation & Design Tasks

### Month 1 – Current Architecture Analysis & Coherence Requirements

<table>
  <tr><th>Task</th><th>Description</th><th>Deliverable</th></tr>
  <tr><td>G1.1</td><td>Deeply analyse the existing GPU L2 cache and memory subsystem: how tags, states, writeback, and atomics are handled. Identify what must change for coherent operation.</td><td>Architecture review report with block diagrams.</td></tr>
  <tr><td>G1.2</td><td>Define the required coherent region behaviour from the GPU’s perspective: cache line states (MESI or TileLink), snoop handling, atomic execution, and fence semantics.</td><td>Coherence requirements spec.</td></tr>
  <tr><td>G1.3</td><td>Map PTX memory model (scopes, fences, atomics) to RISC‑V RVWMO. Determine how GPU instructions should be mapped to system‑level operations when accessing coherent memory.</td><td>Memory model mapping document.</td></tr>
  <tr><td>G1.4</td><td>Set up a simulation/emulation environment for the GPU (e.g., use an existing open‑source GPU simulator like gem5‑GCN3, or build a simplified model in C++/SystemC).</td><td>Baseline GPU model that can run simple kernels and access memory.</td></tr>
</table>

### Month 2 – Coherent L2 Design & Integration

<table>
  <tr><th>Task</th><th>Description</th><th>Deliverable</th></tr>
  <tr><td>G2.1</td><td>Modify the GPU L2 design to support a coherent mode bit (per address range). For coherent addresses, add MESI‑like state and snoop logic; for non‑coherent, keep original behaviour.</td><td>RTL/Chisel code for L2 modifications, simulated.</td></tr>
  <tr><td>G2.2</td><td>Implement an adapter block that translates between TileLink (or chosen protocol) and the GPU’s internal memory request format. This adapter will sit between L2 and the coherent fabric.</td><td>Adapter design + verification plan.</td></tr>
  <tr><td>G2.3</td><td>Design a hardware queue manager for low‑latency command submission from CPU to GPU. It should handle doorbells, descriptor fetch, and completion signalling.</td><td>Queue manager spec + RTL sketch.</td></tr>
  <tr><td>G2.4</td><td>Investigate shared virtual memory: what changes are needed in the GPU MMU to walk RISC‑V page tables or use IOMMU? Plan for PASID and page fault support.</td><td>SVM design document (can be phased).</td></tr>
</table>

### Month 3 – Integration & Simulation

<table>
  <tr><th>Task</th><th>Description</th><th>Deliverable</th></tr>
  <tr><td>G3.1</td><td>Integrate the modified GPU L2 and adapter into a system‑level simulation with the RISC‑V core (using QEMU, gem5, or a custom SystemC harness).</td><td>Combined simulation environment.</td></tr>
  <tr><td>G3.2</td><td>Run the CPU‑GPU coherence microbenchmarks defined by the CPU team. Measure latency, bandwidth, and correctness.</td><td>Performance report and bug list.</td></tr>
  <tr><td>G3.3</td><td>Evaluate the need for a shared on‑chip scratchpad or SLC between CPU and GPU. Propose a simple dual‑port SRAM block that can be added to the coherent fabric.</td><td>Scratchpad proposal (optional, can be deferred).</td></tr>
  <tr><td>G3.4</td><td>Prepare a detailed design document for the next phase: full coherent region implementation, directory sizing, and power/area estimates.</td><td>Phase 2 design plan.</td></tr>
</table>

---

## 3. Cross‑Team Tasks (Joint Effort)

These tasks require collaboration and should be started early.

<table>
  <tr><th>Task</th><th>Description</th><th>Deliverable</th><th>Owner</th></tr>
  <tr><td>X1</td><td>Define the coherent region configuration mechanism: how the address range is set, how the directory/snoop filter is configured, and how software discovers it.</td><td>Configuration spec (CSRs, device tree entries).</td><td>Both</td></tr>
  <tr><td>X2</td><td>Specify the CPU‑GPU communication protocol for kernel launches, completion, and interrupts. Include MSI/IMSIC usage on RISC‑V side, and doorbell MMIO on GPU side.</td><td>Protocol document with message formats.</td><td>Both</td></tr>
  <tr><td>X3</td><td>Choose and set up the shared repository structure for hardware and software, including CI for simulation tests.</td><td>Public repo with CI pipeline.</td><td>Both</td></tr>
  <tr><td>X4</td><td>Create a glossary and conventions document to avoid terminology mismatches (e.g., “coherent” vs “uncached”, “flush” vs “clean”).</td><td>Wiki page.</td><td>Both</td></tr>
  <tr><td>X5</td><td>Plan a bring‑up roadmap for the next 6–12 months after this initial investigation, including milestones for RTL, FPGA prototyping, and software bring‑up.</td><td>Roadmap document.</td><td>Both</td></tr>
</table>

---

## 4. Priority & Immediate Start

The tasks above can all begin right away, but if resources are limited, focus on these **highest‑priority items** in the first few weeks:

- **R1.1 / G1.1** – Understand your existing building blocks and available standards.
- **R1.3 / X1** – Define the memory map and coherent region early; everything else depends on it.
- **G2.1 / R2.2** – Start designing the coherent L2 and directory; these are the core hardware changes.
- **X3 / X4** – Set up collaboration infrastructure so teams can work efficiently.

These tasks are designed to produce concrete, open‑source artefacts that will guide the next phase. By the end of 3 months, you should have:

- A working simulation of a RISC‑V core + modified GPU L2 communicating over a coherent TileLink‑like protocol within a small coherent region.
- A detailed design document for scaling up to chiplet interconnects and larger coherent domains.
- A public repository with all specifications, code, and test results.

This sets a solid foundation for moving from investigation to implementation. Good luck, and feel free to ask for more detail on any specific task!
