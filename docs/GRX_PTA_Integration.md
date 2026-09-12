Congratulations on getting your GPU to synthesis and FPGA-ready. That’s a huge milestone.

Short answer: **this is not a port or an incremental RTL change**. A photonic NPU is an analog, mixed-signal optical accelerator, not a synchronous digital GPU. You would not “modify the GPU” in the usual FPGA/ASIC sense. You would keep the digital control/instruction front end and **replace or augment the math datapath with photonic tensor cores**.

There is enough academic information to start a serious PhD project, but not enough public design infrastructure to simply synthesize a photonic GPU like an FPGA netlist.

---

## 1. What a photonic NPU actually is

Most photonic NPUs are analog optical matrix-vector multipliers.

The common architecture is:

- Digital input activations → DACs → optical modulators
- Light encodes activations as amplitude or phase
- A photonic weight bank does the matrix multiplication
- Photodetectors convert the result back to current
- TIAs and ADCs digitize the output
- Nonlinear activation is usually done electrically or digitally

Common photonic compute devices:

- Mach–Zehnder interferometer meshes
- Microring resonator weight banks
- Phase-change material weight arrays
- Wavelength-division multiplexed dot-product engines
- Photonic crossbars

Key point: the multiply and accumulate happen in the optical domain. Summation is often literally done by light falling on a photodetector.

---

## 2. What you would need to change in your current GPU

Your current GPU is likely built around:

- SIMT warp scheduling
- Register files
- Shared memory
- FP32/INT8 FMA pipelines
- Load/store units
- Coherent cache hierarchy
- Memory controllers

A photonic NPU changes the center of gravity.

### A. Replace the digital MAC array with photonic tensor tiles

Instead of thousands of digital multiply-accumulate units, you have:

- DACs
- optical modulators
- MZI meshes or microring banks
- photodetectors
- ADCs

The GPU’s FP32 FMA datapath would become an analog optical linear-algebra unit.

### B. Change the execution model

A GPU issues instructions to thousands of threads.

A photonic NPU is more like:

1. Load input vector into DACs
2. Configure photonic weights
3. Fire laser pulses
4. Wait for optical settling
5. Sample photodetectors through ADCs
6. Apply digital nonlinearity

The instruction set would be dominated by operations like:

- `LOAD_ACTIVATION`
- `SET_WEIGHTS`
- `MVM`
- `READ_ADC`
- `APPLY_NONLINEARITY`

You would still have a digital controller, but it would look much more like a neural accelerator sequencer than a GPU warp scheduler.

### C. Memory hierarchy changes

Weights in photonic NPUs are often physically encoded in the optical devices:

- thermo-optic phase shifters
- microring resonance shifts
- phase-change materials

That means:

- Weights are not fetched from registers each cycle
- Weight updates are slow
- You need weight double-buffering
- You may need periodic thermal/phase calibration

Activations still live in digital memory and are streamed through DACs.

### D. Precision and arithmetic change

Your GPU probably uses FP32, FP16, or INT8.

Photonic NPUs are usually analog and effectively limited to roughly 4–8 bits of precision, sometimes less after drift, crosstalk, and noise.

You would need:

- quantization-aware training
- noise-aware retraining
- calibration loops
- possibly digital correction after the ADC
- analog-aware error models

### E. The digital control plane remains, but shrinks

You would keep:

- host interface
- DRAM controller
- command processor
- data layout engines
- calibration and scheduling logic

But large parts of the GPU shader core would be removed or reduced.

---

## 3. If you mean optical interconnects, not optical compute

Germany has multiple projects/startups around photonics:

- Q.ANT, working on photonic AI processors
- Black Semiconductor, working on graphene-based optical interconnects
- Fraunhofer HHI / various universities working on silicon photonics NPUs

If your interest is only “photons instead of electrons for data movement,” then the change is less invasive:

- Keep the GPU digital
- Replace electrical NoC/PHY/memory links with optical chiplets or co-packaged optics
- Add optical I/O controllers
- Redesign package and board

That is a realistic near-term step.

If you want photonic computing, that is a much larger architectural change.

---

## 4. Is there enough public information?

Yes, for research.

Key academic starting points:

- Shen et al., *Deep learning with coherent nanophotonic circuits*, Nature Photonics 2017
- Hamerly et al., *Large-scale optical neural networks based on photoelectric multiplication*, Physical Review X 2019
- Shastri et al., *Photonics for artificial intelligence and neuromorphic computing*, Nature Photonics 2021
- Bogaerts et al., *Programmable photonic circuits*, Nature 2020
- Miller, *Self-configuring universal linear optical component*, Photonics Research 2013
- Wetzstein et al., *Inference in artificial intelligence with deep optics and photonics*, Nature 2020

For silicon photonics design:

- Lukas Chrostowski and Michael Hochberg, *Silicon Photonics Design*
- AIM Photonics PDK
- imec iSiPP50G PDK
- AMF / Tower photonic PDKs
- Tools: Lumerical, IPKISS, SiEPIC, gdsfactory, KLayout

The main gap is that there is no equivalent of an FPGA flow for photonic NPUs. You cannot download a photonic netlist to an FPGA. You can only emulate the digital controller and the numerical behavior.

---

## 5. What a practical PhD roadmap could look like

Given your GPU background, a strong PhD direction would be:

### Phase 1: Hybrid architecture

Keep the digital GPU front end, but model a photonic tensor core as an accelerator.

Define:

- instruction interface
- dataflow
- memory movement
- precision/noise model
- scheduling strategy

You can validate the digital control logic on FPGA.

### Phase 2: FPGA emulation

Build an FPGA-based emulator of the photonic tile.

The FPGA would:

- run the actual control state machine
- inject realistic photonic noise
- emulate DAC/ADC quantization
- model weight drift and crosstalk

This is publishable by itself because photonic NPU architectural simulators are still immature.

### Phase 3: Mixed-signal/photonic co-design

If you can get access to a photonic PDK or an academic fab run, design a small photonic dot-product tile and interface it with your digital controller.

The PhD contribution would be:

- digital/photonic co-design methodology
- calibration-aware architecture
- precision-aware scheduling
- hybrid photonic-GPU execution model

---

## 6. Direct answer to your question

> What theoretically would I need to change in the design of my current GPU?

Not a small change. You would need to:

1. Replace digital FMA arrays with photonic matrix-vector tiles.
2. Add DAC/AFM drivers and ADC/TIA readout.
3. Change the execution model from thread-based SIMT to tensor streaming.
4. Replace register-file weight traffic with photonic weight banks.
5. Introduce analog calibration, thermal control, and noise-aware scheduling.
6. Reduce numerical precision assumptions from FP32/INT8 to analog 4–8 bits.
7. Keep the digital front end, memory controller, and host interface.

Your existing GPU RTL is still valuable: it becomes the control plane and emulation platform, not the compute fabric.

In short:

> You would not turn your GPU into a photonic NPU. You would design a photonic tensor accelerator around the digital GPU infrastructure you already understand.

That is exactly the kind of cross-layer problem that makes a good PhD.

---

## 1. System Architecture Overview

We keep the existing GPU’s SIMT cores, caches, and memory controllers unchanged. The photonic tensor accelerator is added as a **new compute unit** attached to the GPU’s internal interconnect, similar to how modern GPUs attach tensor cores or DMA engines.

```plaintext
+-------------------+
|   Host CPU/PCIe   |
+---------+---------+
          |
+---------v---------+
|  GPU Command      |
|  Processor /      |
|  Scheduler        |
+---------+---------+
          |
   +------+------+
   |             |
+--v--+      +---v---+
| SIMT|      |Photonic|
|Cores|      |Tensor  |
|     |      |Accel.  |
+--+--+      +---+---+
   |             |
+--v-------------v--+
|  Shared L2 / Memory|
|  Controller        |
+--------------------+
```

The PTA is a slave device on the GPU’s internal bus (e.g., AXI, NoC). It receives commands and data from the GPU’s copy engines or dedicated DMA, and returns results to memory. The GPU’s command processor is extended to understand new opcodes that target the PTA.

---

## 2. Required Changes to Existing GPU

### 2.1 Command Processor / Front-End

- **New instruction opcodes**: Add instructions like `PTA_LOAD_WEIGHTS`, `PTA_MVM`, `PTA_READ_RESULTS`. These are not executed by SIMT cores but are dispatched to the PTA.
- **New hardware queue**: The command processor needs a separate queue or a dedicated channel to send commands to the PTA. This can be a simple FIFO with a handshake.
- **Address translation**: The PTA will access memory via the GPU’s MMU. You can reuse existing address translation units or give the PTA its own IOMMU context.

### 2.2 Memory Subsystem

- **DMA engine**: The PTA will stream input activations from memory to its DACs and write results back. You can either:
  - Add a dedicated DMA inside the PTA that uses the GPU’s memory interface, or
  - Reuse the GPU’s existing copy engines (CEs) to move data between memory and PTA’s internal buffers.
  - The first option is cleaner and reduces latency; the second minimizes new RTL but adds control complexity.
- **Coherence**: The PTA will not be coherent with the SIMT cores’ L1s. Results are written to L2 or system memory and then made visible via existing cache flush/invalidate mechanisms. This is similar to how current GPU accelerators work.

### 2.3 Interconnect

- **Add a slave port** on the GPU’s NoC or crossbar for the PTA. The PTA will expose a set of memory-mapped registers for control and status, plus a data port for bulk transfers.
- If using AXI, the PTA can be an AXI slave for register access and an AXI master for direct memory access (DMA).

### 2.4 Power / Clock Domains

- The photonic core is analog/mixed-signal. It will require its own power supply and possibly a different clock domain for DAC/ADC sampling. The digital controller of the PTA runs on the GPU’s clock, but the analog interface needs careful clock domain crossing.
- Add power management: the PTA should be power-gated when unused, and the laser/photonic components may need independent power control.

### 2.5 Calibration and Monitoring

- The photonic weights drift with temperature and aging. You need a calibration controller that can:
  - Read photodetector outputs during known test patterns.
  - Adjust thermal phase shifters or other tuning elements.
  - Report status to the GPU driver.
- This controller can be a small embedded microcontroller or a finite state machine inside the PTA, communicating with the host through registers.

---

## 3. Photonic Tensor Accelerator (PTA) Internal Design

### 3.1 High-Level Block Diagram

```plaintext
+---------------------+
|   Digital Control   |
| - Command decoder   |
| - DMA engine        |
| - Weight buffer     |
| - Calibration FSM   |
+----------+----------+
           |
+----------v----------+
| Analog Front-End    |
| - DACs for input    |
| - Modulators        |
| - Photonic weight   |
|   bank (MZI mesh)   |
| - Photodetectors    |
| - TIAs              |
| - ADCs for output   |
+---------------------+
```

### 3.2 Digital Control

- **Command decoder**: Parses commands from the GPU front-end.
- **Weight buffer**: Stores digital weight values to be programmed into the photonic weight bank. Weights are loaded via `PTA_LOAD_WEIGHTS`.
- **Input buffer / output buffer**: Small SRAMs to hold activations and results before/after conversion.
- **DMA controller**: Moves data between PTA buffers and GPU memory.
- **Calibration FSM**: Periodically runs calibration routines to adjust photonic weights. May use dedicated photodetectors or loopback paths.

### 3.3 Analog / Photonic Core

- **DAC array**: Converts digital activations to analog voltages/currents that drive optical modulators.
- **Optical modulators**: Encode activations onto light (amplitude or phase).
- **Photonic weight bank**: A matrix of Mach-Zehnder interferometers (MZIs) or microring resonators. Each element implements a multiplication by a tunable weight. The weights are set by thermal phase shifters (heater) or carrier injection.
- **Photodetectors**: Convert optical output to electrical current.
- **Transimpedance amplifiers (TIAs)**: Convert current to voltage.
- **ADC array**: Digitizes the analog results.

The matrix multiplication is performed as:

yi=∑jwijxjyi​=j∑​wij​xj​

In the photonic core, each input xjxj​ is modulated onto a separate wavelength or a separate spatial mode, then combined and weighted by the photonic mesh, and the sum is detected.

### 3.4 Interface with Digital

- DAC and ADC bit widths: Typically 8–12 bits, but effective precision is limited by optical noise to 4–6 bits.
- Sampling rate: Photonic computation is fast; the bottleneck is DAC/ADC conversion. The PTA can operate at GHz rates in principle, but for initial integration, we assume moderate rates (e.g., 1–10 GS/s) to match FPGA emulation limits.

---

## 4. Integration Steps: From RTL to FPGA Emulation

### Phase 1: Architectural Modeling & Simulation

- Build a cycle-accurate model of the PTA in SystemC or Python (using PyMTL or similar).
- Integrate with a GPU simulator (e.g., GPGPU-Sim, Accel-Sim) to understand performance.
- Define the instruction set extension and memory model.
- Deliverable: A spec document and simulator.

### Phase 2: Digital RTL Implementation

- Implement the PTA digital controller in Verilog/SystemVerilog.
- Implement the command interface (registers, FIFOs) for the GPU front-end.
- Implement DMA engine.
- Implement a **behavioral model** of the photonic core (i.e., a simple matrix multiplication with noise and quantization).
- Verify with UVM or simple testbenches.

### Phase 3: FPGA Integration

- Replace the behavioral photonic core with a **digital emulation** of the photonic compute tile. The FPGA emulates the DAC/ADC and the photonic matrix multiplication using digital multipliers and adders, but with injected noise and quantization effects to mimic analog behavior.
- Integrate the PTA emulator with your existing GPU on the same FPGA (if it fits) or on a separate FPGA connected via PCIe or a high-speed serial link.
- Write a software driver that can send commands and data to the PTA.
- Run small neural network layers (e.g., matrix-vector products) to validate end-to-end operation.

### Phase 4: Mixed-Signal Co-Simulation

- Use an analog/mixed-signal simulator (e.g., Cadence AMS, Synopsys) to model the actual photonic devices.
- Co-simulate the digital RTL with the analog model using a standard interface (e.g., Verilog-AMS, SystemVerilog DPI).
- Refine the digital controller’s calibration and timing.

### Phase 5: Physical Implementation (Optional / Future)

- If access to a photonic PDK and fabrication is available, design a small photonic tile and package it with your digital ASIC.
- This is a major effort and likely beyond a single PhD, but a tape-out of a test chip with a few photonic MAC units could be a capstone.

---

## 5. Software / Programming Model

### 5.1 Extending the GPU’s Programming Interface

- Add new API functions in the GPU driver (e.g., `cuPTA_MatrixMultiply`).
- The driver will:
  1. Allocate memory for inputs/weights/outputs.
  2. Program the PTA’s weight bank via registers.
  3. Initiate DMA transfers.
  4. Issue the `PTA_MVM` command.
  5. Wait for completion (interrupt or polling).
  6. Return results to the application.
- Provide a library (similar to cuBLAS) that uses the PTA for matrix multiplications.

### 5.2 Performance Considerations

- The PTA is most efficient for large matrix-vector or matrix-matrix multiplications (e.g., fully connected layers, GEMMs). Small ops suffer from setup overhead.
- Weights should be kept resident in the photonic bank for multiple inferences to amortize the slow weight programming.
- Calibration runs may need to be scheduled between inferences or when temperature changes are detected.

---

## 6. Summary of Required Changes to Existing GPU RTL

<table>
  <tr><th>GPU Component</th><th>Required Change</th></tr>
  <tr><td>Command Processor</td><td>Add new opcodes and a queue to dispatch to PTA.</td></tr>
  <tr><td>Memory Subsystem</td><td>Add DMA path for PTA; ensure address translation and cache flushing work.</td></tr>
  <tr><td>Interconnect</td><td>Add a slave port for PTA control; add a master port if PTA uses direct DMA.</td></tr>
  <tr><td>Register Map</td><td>Define new memory-mapped registers for PTA control/status.</td></tr>
  <tr><td>Power Management</td><td>Add power-gating controls for PTA.</td></tr>
  <tr><td>Interrupt Controller</td><td>Add interrupt line for PTA completion/errors.</td></tr>
  <tr><td>Software Driver</td><td>Implement new ioctl calls and memory management for PTA.</td></tr>
</table>

The SIMT cores, caches, and execution pipelines remain untouched. The GPU becomes a heterogeneous system: traditional parallel cores for general compute, plus a photonic accelerator for matrix-heavy workloads.

---

## 7. Research Contributions and Risks

### Contributions

- **Digital/photonic co-design methodology** for heterogeneous accelerators.
- **Calibration-aware scheduling** to hide photonic weight drift and programming overhead.
- **Noise-aware programming model** and precision management.
- **FPGA-based emulation framework** for photonic accelerators, enabling early software development.

### Risks

- Photonic devices are immature; actual performance may not meet theoretical projections.
- Calibration and thermal stability are hard problems.
- Integration of analog and digital on the same die or package introduces significant complexity.
- FPGA emulation may not capture all analog effects, so results could be optimistic.

---

## 8. Conclusion

The path from a traditional GPU to one with an integrated photonic tensor accelerator is well-defined from a digital integration perspective. The major effort lies in designing the photonic core and its analog interface, but the digital control, memory, and software changes are manageable and can be prototyped on FPGA. This is an excellent PhD project that bridges computer architecture, photonics, and VLSI design. You can start by building a cycle-accurate simulator and then move to RTL and FPGA emulation, leaving the actual photonic tape-out as a stretch goal.

Would you like me to elaborate on any specific phase, such as the instruction set definition, DMA design, or the FPGA emulation strategy?

---

## Instruction Set Definition for the Photonic Tensor Accelerator (PTA)

Integrating a photonic tensor accelerator into your existing GPU requires a well-defined command interface between the GPU’s digital control plane and the analog/photonic compute engine. This interface is **not** an extension of the shader ISA; it is a set of **control commands** that tell the PTA to move data, configure weights, execute a matrix operation, and report status.

Below I define a practical instruction set for a PTA that can be integrated into your current GPU architecture. The design assumes the PTA is connected as a memory‑mapped device (or via a dedicated command queue) on the GPU’s internal interconnect.

---

### 1. Design Philosophy

- **Decouple from SIMT cores** – PTA commands are issued by the GPU command processor, not by threads. They run asynchronously, similar to DMA or copy engines.
- **Command queue based** – Commands are written into a ring buffer in memory, and the PTA fetches and executes them one by one. This allows overlapping of command setup and execution.
- **Fixed‑size command descriptors** – Each command is a 64‑byte (or 128‑byte) structure for easy DMA and cache alignment.
- **Memory addressing** – The PTA uses the GPU’s virtual address space, with an IOMMU or SMMU for isolation.
- **Interrupt‑driven completion** – The PTA can raise an interrupt when a command finishes, or the GPU can poll a status register.

---

### 2. Command Format

Each command descriptor contains:

<table>
  <tr><th>Field</th><th>Bits</th><th>Description</th></tr>
  <tr><td>opcode</td><td>8</td><td>Operation code (see table below)</td></tr>
  <tr><td>flags</td><td>8</td><td>Control flags (e.g., precision mode, bypass calibration)</td></tr>
  <tr><td>src_addr</td><td>64</td><td>Virtual address of input activations (or weights, depending on opcode)</td></tr>
  <tr><td>dst_addr</td><td>64</td><td>Virtual address of output results</td></tr>
  <tr><td>size</td><td>32</td><td>Data size in bytes, or matrix dimensions packed</td></tr>
  <tr><td>dim0</td><td>16</td><td>For matrix ops: rows (M) or input length</td></tr>
  <tr><td>dim1</td><td>16</td><td>For matrix ops: columns (N) or output length</td></tr>
  <tr><td>batch</td><td>16</td><td>Number of independent vectors to process (for batched MVM)</td></tr>
  <tr><td>weight_id</td><td>16</td><td>Identifier of the weight matrix already programmed in the photonic bank</td></tr>
  <tr><td>reserved</td><td>32</td><td>Future use, e.g., calibration parameters</td></tr>
</table>

**Total: 256 bits (32 bytes)**. To simplify, we can use a 64‑byte descriptor with alignment padding.

Alternatively, commands can be written directly to memory‑mapped registers if no queue is desired, but a queue is more flexible and scalable.

---

### 3. Opcode Set

The PTA supports the following essential operations. Each opcode maps to a single command descriptor.

<table>
  <tr><th>Opcode Name</th><th>Value</th><th>Description</th></tr>
  <tr><td>PTA_NOP</td><td>0x00</td><td>No operation; used for padding in the queue.</td></tr>
  <tr><td>PTA_LOAD_WEIGHTS</td><td>0x01</td><td>Load a weight matrix from memory into the photonic weight bank.</td></tr>
  <tr><td>PTA_STORE_WEIGHTS</td><td>0x02</td><td>Read back current photonic weights (for debugging/calibration).</td></tr>
  <tr><td>PTA_MVM</td><td>0x10</td><td>Perform matrix‑vector multiplication: y = W * x + bias.</td></tr>
  <tr><td>PTA_GEMM</td><td>0x11</td><td>Perform batched matrix‑matrix multiply (may be decomposed into multiple MVMs).</td></tr>
  <tr><td>PTA_ACTIVATE</td><td>0x20</td><td>Apply a nonlinear activation (ReLU, sigmoid, etc.) to a buffer in digital domain.</td></tr>
  <tr><td>PTA_CONV2D</td><td>0x30</td><td>(Optional) Perform convolution using im2col + MVM.</td></tr>
  <tr><td>PTA_CALIBRATE</td><td>0x40</td><td>Run a calibration routine on the photonic array.</td></tr>
  <tr><td>PTA_SET_MODE</td><td>0x50</td><td>Configure operating mode: precision, power, clock divider, etc.</td></tr>
  <tr><td>PTA_SYNC</td><td>0x60</td><td>Memory fence / synchronization: flush caches, wait for previous ops.</td></tr>
  <tr><td>PTA_STATUS</td><td>0x70</td><td>Write status word (error bits, temperature, etc.) to dst_addr.</td></tr>
  <tr><td>PTA_INTERRUPT</td><td>0x80</td><td>Raise an interrupt on the host GPU (or directly to CPU).</td></tr>
</table>

All opcodes are executed **in order** within the queue. Some may require the PTA to wait for previous operations to complete (e.g., `PTA_MVM` must wait for `PTA_LOAD_WEIGHTS` if the weights are being reloaded).

---

### 4. Detailed Semantics

#### `PTA_LOAD_WEIGHTS`

- **Purpose:** Program the photonic weight bank (e.g., MZI mesh or microring array) with a new weight matrix.
- **Inputs:**
  - `src_addr`: memory address of weight matrix in row‑major format (float32, but quantized to analog precision internally).
  - `size`: total bytes.
  - `dim0`, `dim1`: matrix dimensions (M×N). If `dim0` or `dim1` differ from the hardware’s native size, the PTA may pad or reject.
  - `weight_id`: logical identifier for this weight set. The PTA can cache multiple weight sets in its internal SRAM (if available) to avoid reprogramming.
- **Behavior:**
  - The PTA’s DMA engine reads the weight data from memory.
  - The digital controller converts the floating‑point weights to analog control voltages (or currents) for the photonic devices. This may involve quantization and calibration.
  - The physical programming of the photonic mesh occurs (slow, possibly milliseconds).
- **Completion:** The command is considered complete only after programming is finished and a short stabilization delay has elapsed.

#### `PTA_MVM`

- **Purpose:** Perform `Y = W * X + b` where `W` is the weight matrix already loaded, `X` is a vector or batch of vectors, and `b` is an optional bias.
- **Inputs:**
  - `src_addr`: memory address of input vector(s). Format: for a single vector, length N (matching `dim1` of weights). For batched, `batch` vectors of length N stored contiguously.
  - `dst_addr`: memory address where output vector(s) of length M will be written.
  - `dim0`, `dim1`: M and N of the weight matrix (must match loaded weights).
  - `batch`: number of input vectors (if >1, the PTA performs multiple MVMs, possibly reusing weights).
  - `weight_id`: which previously loaded weight set to use (if multiple are stored).
- **Behavior:**
  - The PTA streams the input vectors from memory through DACs.
  - The optical matrix multiplication happens in the analog domain.
  - The photodetector outputs are digitized by ADCs.
  - Bias, if present, is added digitally after ADC (or with an additional electrical summing circuit).
  - Results are written back to `dst_addr`.
- **Precision:** The analog computation is effectively limited to ~4‑8 bits. The PTA may optionally perform digital correction using known calibration data.

#### `PTA_GEMM`

- **Purpose:** Compute `C = A * B` where one matrix (say `B`) is already in the photonic weight bank and the other (`A`) is streamed from memory. This is essentially a batched MVM: each column of `A` becomes an input vector.
- **Inputs:**
  - `src_addr`: address of matrix A (K×N, stored row‑major).
  - `dst_addr`: address of result matrix C (M×N).
  - `dim0` = M (output rows), `dim1` = K (inner dimension), `batch` = N (number of columns).
- **Behavior:** Equivalent to looping `PTA_MVM` over all columns of A, but the PTA can pipeline the operations to hide ADC latency.

#### `PTA_ACTIVATE`

- **Purpose:** Apply a nonlinear function to a buffer in digital memory. This is necessary because optical activation is difficult; the PTA will offload this to a small digital SIMD unit inside the PTA (or the GPU’s own cores).
- **Inputs:** `src_addr`, `dst_addr`, `size`, `flags` (to select activation type).
- **Behavior:** Reads buffer, applies elementwise activation, writes back.

#### `PTA_CALIBRATE`

- **Purpose:** Run a calibration cycle to compensate for thermal drift, phase errors, etc.
- **Inputs:** `dst_addr` (where calibration report is written), optional parameters in `reserved`.
- **Behavior:**
  - The PTA’s internal calibration controller injects known test vectors.
  - It measures photodetector outputs and compares against expected results.
  - It adjusts the weight control voltages accordingly.
  - Writes a calibration log (e.g., measured error, temperature) to `dst_addr`.
- **Note:** This can be issued by the GPU driver periodically or when temperature sensors trigger.

#### `PTA_SET_MODE`

- **Purpose:** Configure the PTA’s operating parameters: ADC sampling rate, DAC resolution, power‑saving modes, calibration frequency, etc.
- **Inputs:** `flags` and `reserved` contain bit‑encoded settings.

#### `PTA_SYNC`

- **Purpose:** Act as a memory barrier. Ensures all previous PTA commands have completed and all writes are visible to the GPU. May also flush/invalidate caches as needed.
- **Inputs:** None (or a memory range).

#### `PTA_STATUS` / `PTA_INTERRUPT`

- **Purpose:** Read internal status (error flags, temperature, operation count) or raise an interrupt to the CPU.

---

### 5. Command Queue Implementation

- The GPU command processor maintains a **ring buffer** in system memory. Each entry is a 64‑byte command descriptor.
- The PTA has a **doorbell register** that the GPU writes to after enqueuing commands. The PTA fetches the commands via its DMA engine.
- Alternatively, the PTA exposes a set of **memory‑mapped registers** (MMIO) that the GPU writes to directly. For a small number of commands, this is simpler but less scalable.

```plaintext
struct pta_command {
    uint8_t   opcode;
    uint8_t   flags;
    uint16_t  weight_id;
    uint16_t  dim0, dim1, batch;
    uint32_t  size;
    uint64_t  src_addr;
    uint64_t  dst_addr;
    uint32_t  reserved[2];
};
```

---

### 6. Integration with GPU Software Stack

- The GPU driver will expose an API like `pta_submit(cmd_list, num_cmds)`. It builds the command descriptors, writes them to the ring buffer, and rings the doorbell.
- The GPU’s thread scheduler does **not** see these commands. They are entirely handled by the copy engine or a dedicated PTA control unit.
- The driver can choose to execute a neural network layer on the PTA by submitting the appropriate command sequence, while the SIMT cores handle other tasks.

**Typical sequence for a fully‑connected layer:**

1. `PTA_LOAD_WEIGHTS` – load weight matrix from memory.
2. `PTA_MVM` – multiply input vector (or batch) by weights.
3. `PTA_ACTIVATE` – apply ReLU to output.
4. `PTA_SYNC` – ensure completion.
5. `PTA_STATUS` – optional, read error/performance counters.

---

### 7. Error Handling

- The PTA sets error bits in its status register (e.g., ADC overflow, calibration failure, DMA error).
- A `PTA_INTERRUPT` command can be used to notify the CPU.
- The driver can query status and retry or fall back to digital computation.

---

### 8. Extensibility

- Reserved bits in `flags` and `reserved` allow future additions like:
  - Different data formats (int8, bfloat16).
  - Sparse matrix support.
  - Multiple photonic tiles running in parallel.
- The opcode space is large enough to add custom instructions for new photonic operations (e.g., convolution with photonic Fourier transforms).

---

### Next Steps

With the instruction set defined, we can proceed to:

1. **Design the PTA’s digital control unit** that decodes and executes these commands.
2. **Specify the DMA and memory interface** to move data efficiently.
3. **Plan the FPGA emulation**, where we can implement the entire command pipeline and replace the photonic core with a digital model.

Would you like to move on to the DMA design or the FPGA emulation strategy next?

---

## DMA Engine Design for the Photonic Tensor Accelerator (PTA)

The DMA engine is the digital workhorse that moves data between GPU memory and the photonic compute core. Its design must bridge the synchronous digital domain (AXI/NoC) with the analog conversion front‑end (DACs/ADCs), while supporting the command queue and maximizing throughput.

We will design a **streaming DMA controller** optimized for the PTA’s workloads: large contiguous blocks (weights, activations, results) that are accessed in a predictable pattern.

---

### 1. Requirements

<table>
  <tr><th>Requirement</th><th>Description</th></tr>
  <tr><td>Bulk transfers</td><td>Move weight matrices (up to MBs), activation vectors (KB to MB), and result matrices efficiently.</td></tr>
  <tr><td>Low latency</td><td>For small MVM operations, the DMA should not dominate execution time.</td></tr>
  <tr><td>Address translation</td><td>Use the GPU’s MMU (IOMMU/SMMU) to translate virtual addresses to physical, enabling user‑space buffers and protection.</td></tr>
  <tr><td>Cache coherence</td><td>Must interact correctly with GPU caches: either bypass them (non‑cacheable) or perform explicit flushes/invalidates.</td></tr>
  <tr><td>Command fetching</td><td>Retrieve command descriptors from memory (if using a ring buffer) or accept commands via MMIO.</td></tr>
  <tr><td>Multiple channels</td><td>Support concurrent input and output streams (e.g., read input while writing previous result).</td></tr>
  <tr><td>Error reporting</td><td>Detect DMA faults, timeouts, and bus errors.</td></tr>
  <tr><td>Back‑pressure</td><td>Coordinate with photonic core’s readiness (DAC/ADC timing).</td></tr>
</table>

---

### 2. Top‑Level Architecture

```plaintext
+-----------------------+
|   PTA Control Unit    |
|  (Command Decoder,    |
|   State Machines)     |
+-----------+-----------+
            |
            v
+-----------+-----------+
|      DMA Engine        |
| +-------------------+  |
| | Descriptor Fetch  |  |
| | & Queue Mgmt      |  |
| +-------------------+  |
| | Read Channel      |  |
| | (AXI Master)      |  |
| +-------------------+  |
| | Write Channel     |  |
| | (AXI Master)      |  |
| +-------------------+  |
| | Buffer Manager    |  |
| | (SRAM Arbiters)   |  |
| +-------------------+  |
+-----------+-----------+
            |
            v
+-----------+-----------+
| Internal Buffers      |
| - Input Buffer (DAC)  |
| - Weight Buffer       |
| - Output Buffer (ADC) |
+-----------------------+
```

The DMA engine has three main functional blocks:

- **Descriptor Fetch**: Reads command descriptors from the ring buffer in memory (or from MMIO registers if not using a queue).
- **Read Channel**: Performs AXI read bursts from GPU memory to fill internal buffers.
- **Write Channel**: Performs AXI write bursts from internal buffers to GPU memory.

The **Buffer Manager** allocates and arbitrates access to the internal SRAM buffers, which are shared between the DMA and the photonic analog front‑end.

---

### 3. Internal Buffering

The photonic core operates at high speed but has analog settling times and ADC conversion latencies. To decouple memory latency from the analog pipeline, we use **double buffering** for input activations and output results.

<table>
  <tr><th>Buffer</th><th>Purpose</th><th>Size (example)</th><th>Double‑buffered?</th></tr>
  <tr><td>Input Buffer</td><td>Holds activations waiting to be sent to DACs</td><td>64–256 KB</td><td>Yes</td></tr>
  <tr><td>Weight Buffer</td><td>Stores weight matrices in digital form before programming into photonic bank. Also used as cache for multiple weight sets.</td><td>1–4 MB</td><td>No (weights are loaded once and persist)</td></tr>
  <tr><td>Output Buffer</td><td>Holds ADC results before writing back to memory</td><td>64–256 KB</td><td>Yes</td></tr>
</table>

The Weight Buffer may be implemented as a small SRAM (or even part of the GPU’s L2 if the PTA is integrated tightly). It allows the PTA to store several weight sets and switch quickly without re‑fetching from DRAM.

---

### 4. Data Transfer Flows

#### 4.1 Command Fetch

If using a command ring buffer:

1. The GPU writes command descriptors to a memory region and updates a **doorbell** (a register in the PTA).
2. The PTA’s Descriptor Fetch engine reads the doorbell index and issues an AXI read to fetch the next descriptor(s).
3. Descriptors are decoded and passed to the control unit, which orchestrates the data movement.

Alternatively, commands can be pushed directly via MMIO writes to the PTA’s registers, avoiding the fetch latency. For simplicity, we will assume a **memory‑mapped command queue** (MMIO push), where the GPU writes descriptors one by one into a hardware FIFO inside the PTA. This avoids the need for the PTA to read its own command stream and is easier to implement on FPGA initially.

So: **Command Interface = MMIO FIFO**. The GPU writes commands to a set of registers; the PTA pops them in order.

Thus, Descriptor Fetch is simplified to a register interface. But for high throughput, a ring buffer is better because the GPU can enqueue many commands with a single doorbell write. We will mention both options but focus on MMIO for initial integration, as it is simpler.

#### 4.2 Loading Weights (`PTA_LOAD_WEIGHTS`)

1. Control unit receives command with `src_addr`, `size`, matrix dims.
2. Control unit instructs DMA Read Channel to fetch weight data from memory into the Weight Buffer.
  - The DMA issues AXI read bursts, using virtual address translation.
  - Data is written into Weight Buffer SRAM.
3. After all data is fetched, the control unit begins programming the photonic weight bank.
  - This is a slow process (DACs for thermal phase shifters, etc.). The DMA is idle during this time.
4. If `weight_id` is specified, the weight set is tagged and kept in Weight Buffer for future reuse.

*Optimization*: If the weight matrix is already in the Weight Buffer (cache hit), skip the memory fetch and directly program the photonic bank from the buffer.

#### 4.3 Matrix‑Vector Multiplication (`PTA_MVM`)

1. Control unit decodes command: `src_addr` (input), `dst_addr` (output), `batch`, `dim0`, `dim1`, `weight_id`.
2. It checks if the required weights are already programmed. If not, it may trigger an implicit load or signal an error.
3. Input streaming:
  - DMA Read Channel fetches input activations from memory into Input Buffer.
  - For batched mode, it fetches `batch` vectors. The DMA can use double buffering: while one buffer is being consumed by the DACs, the next is being filled.
4. Photonic compute occurs: input data is sent to DACs, optical MVM runs, outputs are captured by ADCs and stored in Output Buffer.
5. DMA Write Channel writes the output buffer contents to `dst_addr` in memory. It may also add bias (if bias is present, it can be handled by the digital controller before writing, or the DMA can fetch bias separately and combine after ADC—simpler to do it before writing with a small ALU).
6. Completion: after all outputs are written, the PTA signals command completion.

*Data flow details*:

- The Read Channel must know the input layout: for a single MVM, the input vector is contiguous. For GEMM, the input matrix is stored row‑major, and each column becomes an input vector. The DMA can be programmed with strides and counts.
- The Write Channel writes the result in a contiguous block, regardless of the input layout.

#### 4.4 Calibration and Status

- `PTA_CALIBRATE` may require reading/writing calibration data from/to memory.
- `PTA_STATUS` writes a small status block to `dst_addr`.

These are similar to MVM but with different buffer usage.

---

### 5. AXI Master Interface

The DMA engine will have one or more AXI master ports (e.g., AXI4) to access GPU memory.

**Address Translation**: The PTA will have an **SMMU/IOMMU** interface. The GPU driver sets up page tables for the PTA’s virtual address space. The DMA issues virtual addresses, and the SMMU translates them. This allows user applications to pass pointers directly to the PTA.

**Cache Coherence**: To avoid stale data and ensure correctness:

- For input data (weights, activations), the GPU driver must ensure data is visible in memory (flush L2 if using write‑back caches). The PTA can also issue cache maintenance operations or use non‑cacheable memory attributes in its AXI transactions (AxCache = 0b0000 for non‑cacheable, non‑bufferable). Simpler: use non‑cacheable transactions for all PTA DMA.
- For output data, the PTA writes to memory with non‑cacheable attributes. Then the GPU driver invalidates any cached copies before reading.

Given that the PTA is an accelerator, we can mark all its memory accesses as **non‑coherent** and rely on driver‑managed flushes/invalidates. This is standard for many accelerators.

**Burst Length**: Use maximum burst length supported by the interconnect (e.g., 256 bytes for AXI4) to maximize efficiency. The DMA engine should align transfers to burst boundaries when possible.

---

### 6. Buffer Manager and Arbitration

The internal buffers are dual‑port SRAMs, with one port for DMA and one for the analog front‑end (DAC/ADC). The Buffer Manager implements:

- **Address generation** for DMA transfers (base address, offset, stride).
- **Read/write arbitration** to avoid conflicts when DMA and photonic core access the same buffer.
- **Double‑buffer control**: pointers to “fill” and “drain” buffers, with handshake signals.

For input buffer:

- DMA writes into Buffer A while the photonic core reads from Buffer B.
- When Buffer B is consumed, the roles swap.

For output buffer:

- Photonic core writes into Buffer C while DMA reads from Buffer D.
- Swap when Buffer D is fully transmitted.

The buffer sizes are chosen so that transfer times are comparable to compute times, hiding memory latency.

---

### 7. Performance Considerations

- **Overlap**: The DMA should run concurrently with photonic computation. This is achieved by double buffering and independent read/write channels.
- **Latency hiding**: For small batches, the startup overhead (programming photonic weights, DAC settling) may dominate. The DMA should prefetch input data while the photonic core is still busy with previous command.
- **Weight reuse**: Loading weights is expensive. The weight cache reduces reloading for multiple inferences with the same weights.
- **Bandwidth**: The photonic core can perform a matrix‑vector product in a few nanoseconds, but DAC/ADC and memory bandwidth limit sustained throughput. The DMA must be able to saturate the memory bus. This is feasible with wide AXI ports (e.g., 512‑bit) and high clock frequency.

---

### 8. Error Handling

- **AXI errors**: If a read or write transaction returns an error response, the DMA aborts the current command, sets an error bit in status, and raises an interrupt.
- **Address translation faults**: The SMMU may report a fault (e.g., page not present). The PTA stops and signals the driver.
- **Buffer overflow/underflow**: If the photonic core consumes data faster than DMA can supply (or vice versa), a stall occurs. Proper sizing and flow control prevent this.
- **Timeout counters**: If a memory transaction does not complete within a programmable timeout, the DMA declares a fault.

---

### 9. Implementation Details for FPGA Emulation

On FPGA, the DMA engine is implemented as standard RTL with AXI master ports. Since there is no real photonic core, the “photonic compute” is replaced by a digital model that reads from the input buffer, performs matrix multiplication using DSPs, and writes to the output buffer. The DAC/ADC behavior is simulated by adding quantization and noise.

The DMA design remains identical; only the analog front‑end is replaced. This allows full validation of the DMA and command handling without photonic hardware.

---

### 10. Summary

The DMA engine for the PTA is a conventional but carefully optimized component. It uses:

- AXI master ports with SMMU for virtual memory.
- Non‑cacheable transactions for simplicity.
- Internal SRAMs with double buffering.
- Separate read/write channels for concurrency.
- A command queue either via MMIO FIFO or memory ring buffer.

The design integrates seamlessly with your existing GPU’s memory subsystem and provides the data movement backbone for the photonic tensor accelerator.

---

### Next Step

With the instruction set and DMA design specified, we can now plan the **FPGA emulation strategy** to validate the entire PTA digital subsystem before any photonic tape‑out. Would you like to proceed to that?

---

## FPGA Emulation Strategy for the Photonic Tensor Accelerator

Now that we have defined the instruction set and the DMA engine, the next step is to validate the entire digital control and data movement architecture **before** any physical photonic hardware is available. FPGA emulation allows us to build a functional prototype of the PTA integrated with your existing GPU, using a digital model of the photonic core that mimics its analog behavior (noise, quantization, latency, and calibration effects).

This phase is critical because it enables:

- Functional verification of the command interface and DMA.
- Early software development and performance estimation.
- Validation of memory coherence and synchronization.
- Identification of architectural bottlenecks (e.g., buffer sizes, command overhead) that are independent of actual photonic device physics.

---

### 1. Emulation Goals

<table>
  <tr><th>Goal</th><th>Description</th></tr>
  <tr><td>Functional correctness</td><td>Ensure commands execute in order, data is moved correctly, and results are computed accurately according to the emulated photonic model.</td></tr>
  <tr><td>Timing estimation</td><td>Measure end‑to‑end latency and throughput of the PTA for realistic neural network workloads.</td></tr>
  <tr><td>Software bring‑up</td><td>Test the GPU driver, command submission, and synchronization logic.</td></tr>
  <tr><td>Fault injection</td><td>Simulate analog errors (weight drift, ADC noise) to validate error handling and calibration routines.</td></tr>
  <tr><td>Scalability testing</td><td>Evaluate different buffer sizes, queue depths, and batch sizes.</td></tr>
</table>

The emulation does **not** need to match the speed of a real photonic core; it only needs to replicate its **functional and timing behavior** at the architectural level. For example, the photonic matrix multiplication is emulated using digital multipliers in the FPGA, but we can inject realistic delays (e.g., DAC settling time, ADC conversion time) to model the analog pipeline.

---

### 2. Overall Emulation Architecture

The emulation platform consists of:

- **Your existing GPU design** (or a simplified version) running on FPGA.
- **The PTA digital controller** (command decoder, DMA, buffer manager) implemented in RTL.
- **A photonic core model** implemented as a digital block that emulates the analog matrix‑vector multiplication, including quantization and noise.

The two components are integrated on the same FPGA if it is large enough, or across multiple FPGAs connected via PCIe or a high‑speed serial link.

```plaintext
+---------------------+
|   Host CPU (x86)    |
+----------+----------+
           | PCIe
+----------v----------+
|       FPGA           |
| +------------------+ |
| |  GPU (SIMT cores)| |
| |  + command proc  | |
| +--------+---------+ |
|          |           |
| +--------v---------+ |
| | PTA Digital      | |
| | - Command FIFO   | |
| | - DMA Engine     | |
| | - Buffer Manager | |
| +--------+---------+ |
|          |           |
| +--------v---------+ |
| | Photonic Core    | |
| | Model (Digital)  | |
| | - MVM emulation  | |
| | - Noise injection| |
| +------------------+ |
+---------------------+
```

If the GPU and PTA do not fit on one FPGA, you can place the PTA on a second FPGA and connect them with PCIe or an AXI‑to‑PCIe bridge. For a PhD prototype, a single large FPGA (e.g., Xilinx Alveo, Intel Agilex) should be sufficient if you use a simplified GPU core or share resources.

---

### 3. Implementing the Photonic Core Model in FPGA

The photonic core is replaced by a digital model that emulates the analog matrix‑vector multiplication. This model must capture:

- **Quantization** of inputs and weights (e.g., 4–8 bits).
- **Noise** from photodetectors and ADCs (thermal, shot, quantization).
- **Weight drift** and programming inaccuracies.
- **Latency** of the optical path and conversion.

#### 3.1 Basic Matrix Multiplication Emulation

For an MVM operation y=Wxy=Wx, where WW is an M×NM×N matrix and xx is an NN-vector, the FPGA model uses:

- Digital multipliers (DSPs) to compute each product wijxjwij​xj​.
- Adders to sum the products.
- Optional bias addition.
- Quantization at input and output to mimic DAC/ADC resolution.

If the photonic core uses wavelength‑division multiplexing or spatial modes, the arithmetic is the same from a functional perspective. We can ignore the physical details and focus on the math.

#### 3.2 Noise and Error Injection

To make the emulation realistic:

- **Add Gaussian noise** to the output of the matrix multiplication to simulate photodetector and TIA noise.
- **Quantize weights** to the target analog precision (e.g., 5 bits) when they are loaded, and perhaps add small random offsets to simulate programming error.
- **Add drift** over time: a slow random walk or temperature‑dependent offset that changes after each calibration.
- **Latency model**: The photonic computation itself is extremely fast (picoseconds), but DAC/ADC conversion takes nanoseconds. The emulation can insert a configurable delay (e.g., 10–100 ns) to reflect this.

The noise parameters can be controlled via registers to allow sweep and fault injection.

#### 3.3 Calibration Emulation

The `PTA_CALIBRATE` command should run a simple digital calibration routine:

- Inject known test vectors (stored in a small ROM).
- Measure outputs and compare with expected.
- Compute correction coefficients (e.g., scale and offset) and apply them to subsequent computations.
- Report the measured error to the driver.

This tests the calibration control flow without needing actual photonic hardware.

#### 3.4 Resource Utilization

A full MVM with large dimensions (e.g., 64×64) may require many DSPs. To keep FPGA resources manageable:

- Use a **time‑multiplexed** architecture: process only a few rows of WW per clock cycle, reusing the same multipliers.
- Or use **tiled computation**: split the matrix into small blocks and accumulate partial sums.
- The emulation does not need to match the high speed of real photonics; a slower, resource‑efficient design is acceptable.

For example, a 16×16 MVM can be implemented with 16 MAC units running for 16 cycles, using only 16 DSPs.

---

### 4. Integration with Existing GPU on FPGA

Your current GPU is already synthesized and ready for FPGA testing. We assume it has an AXI or NoC interface for peripherals. The PTA is attached as a new slave device on the GPU’s internal interconnect.

#### 4.1 Hardware Modifications to GPU

- **Add a memory‑mapped register block** for the PTA in the GPU’s address space.
- **Instantiate the PTA** as a module connected to the GPU’s bus.
- **Add an interrupt line** from the PTA to the GPU’s interrupt controller (or directly to the host CPU via PCIe).
- **Modify the GPU’s command processor** to recognize new opcodes that target the PTA. This may be as simple as detecting an address range and forwarding writes to the PTA.

If your GPU does not have a spare bus port, you can place the PTA behind the GPU’s memory controller as a memory‑mapped device.

#### 4.2 Software Stack

The host driver will:

1. Allocate buffers in host memory (or GPU memory) that are accessible by the PTA via DMA.
2. Build command descriptors and write them to the PTA’s MMIO FIFO.
3. Ring the doorbell (write to a PTA register) to start execution.
4. Wait for completion via interrupt or polling a status register.
5. Verify results against a CPU reference.

A minimal userspace library can be written in C to expose functions like `pta_mvm()` for testing.

#### 4.3 Data Path

- The PTA DMA reads input activations from GPU memory (or host memory through PCIe).
- The emulated photonic core processes them.
- The PTA writes results back to memory.
- The host reads the results and checks against expected values.

To simplify initial testing, the PTA can be given physical addresses directly (bypass MMU) if the driver runs in kernel space. Later, add an IOMMU/SMMU.

---

### 5. Test Plan

#### 5.1 Unit Tests

- **Command decoder**: Verify each opcode is decoded correctly and produces the expected control signals.
- **DMA engine**: Test read/write bursts with various sizes, alignments, and strides.
- **Buffer manager**: Test double‑buffering and handshake under back‑pressure.
- **Photonic core model**: Compare output against a software reference with injected noise.

#### 5.2 Integration Tests

- **Single MVM**: Load weights, perform one matrix‑vector multiply, compare result.
- **Batched MVM**: Test multiple input vectors.
- **Weight reuse**: Load weights once, run several MVMs, ensure no reload.
- **Calibration**: Run calibration and verify it corrects known errors.
- **Concurrent operation**: Run SIMT kernels on the GPU while PTA is busy; ensure no interference.

#### 5.3 Performance Measurements

- Measure command latency (from doorbell to completion).
- Measure DMA throughput.
- Measure effective MVM throughput (ops/sec) and compare with theoretical digital performance.
- Analyze bottlenecks: command overhead, buffer sizes, DMA bandwidth.

---

### 6. Limitations of FPGA Emulation

- **No true analog behavior**: The digital model cannot capture all photonic physics (e.g., crosstalk, wavelength drift, phase noise). However, we can inject random noise to approximate first‑order effects.
- **Speed mismatch**: A real photonic core would compute an MVM in ~1 ns, but the FPGA emulation may take microseconds. This is acceptable for functional testing and architectural exploration.
- **Resource constraints**: Large matrix sizes may not fit on FPGA. Use tiling and small test cases.

These limitations are well understood; the emulation is a stepping stone to a mixed‑signal prototype.

---

### 7. Path to Real Photonic Integration

After the FPGA emulation is validated, the next phase is to replace the digital photonic core model with an actual photonic chip connected to the FPGA.

Possible approaches:

1. **Separate photonic chip + FPGA**: Mount a photonic integrated circuit (PIC) on a custom PCB, connect it to the FPGA via high‑speed DAC/ADC and control signals. The FPGA still hosts the digital controller and DMA.
2. **Co‑packaged optics**: Integrate the photonic chip with the FPGA or ASIC in a multi‑chip module (MCM). This is more complex but offers higher bandwidth.

The RTL for the PTA digital controller, DMA, and command interface will remain largely unchanged; only the analog front‑end interface needs to be adapted from the emulated model to real DAC/ADC drivers and photonic control circuitry.

---

### 8. Summary

FPGA emulation is the natural next step. It allows you to:

- Fully validate the instruction set and DMA design you have just defined.
- Integrate the PTA with your existing GPU and run real software.
- Experiment with architectural parameters (buffer sizes, batch sizes, calibration frequency).
- Produce publishable results on hybrid photonic‑digital architecture even before any photonic tape‑out.

We can now proceed to outline a more concrete implementation plan for the FPGA emulation, including RTL module breakdown and resource estimates, or move on to discussing the analog‑digital interface for a future physical prototype. Which aspect would you like to explore next?

---

## Analog‑Digital Interface for the Photonic Tensor Accelerator

The analog‑digital interface is the critical boundary where the synchronous digital world of your GPU meets the continuous, noisy, and temperature‑sensitive photonic compute fabric. Its design determines the achievable precision, speed, and programmability of the photonic tensor accelerator (PTA). In this discussion we will:

- Identify the physical components that form the interface.
- Describe the signal flow and control loops.
- Explain how the digital controller interacts with the analog domain.
- Outline the timing, precision, and calibration challenges.
- Discuss how to model this interface in FPGA emulation and what changes for a real photonic chip.

---

### 1. Physical Components and Signal Flow

The photonic core performs matrix‑vector multiplication in the optical domain. The analog‑digital interface consists of the following blocks, arranged around the photonic mesh:

```plaintext
Digital Input (from DMA/buffers)
        │
        ▼
   ┌─────────┐
   │   DAC   │  (digital‑to‑analog converters for activations)
   └────┬────┘
        │ analog voltage/current
        ▼
   ┌─────────┐
   │ Optical │  (modulators: e.g., Mach‑Zehnder or microring)
   │Modulators│
   └────┬────┘
        │ modulated light
        ▼
   ┌─────────┐
   │Photonic │  (weight bank: MZI mesh or microring array)
   │ Weight  │
   │  Bank   │
   └────┬────┘
        │ optical signals (summation happens optically)
        ▼
   ┌─────────┐
   │Photode‑ │  (photodetectors, often integrated)
   │tectors  │
   └────┬────┘
        │ photocurrent
        ▼
   ┌─────────┐
   │   TIA   │  (transimpedance amplifier: current‑to‑voltage)
   └────┬────┘
        │ analog voltage
        ▼
   ┌─────────┐
   │   ADC   │  (analog‑to‑digital converter)
   └────┬────┘
        │ digital output
        ▼
Digital Output (to DMA/buffers)
```

Additionally, the photonic weight bank requires **programming** of each weight element. This is usually done via thermal phase shifters (heater) or carrier‑injection modulators, driven by **calibration DACs** or slow control DACs.

---

### 2. Activation Path: DAC → Modulator

- **DAC resolution and speed**: Activations are typically quantized to 4–8 bits, so a low‑resolution but high‑speed DAC is sufficient. Sampling rates can range from hundreds of MS/s to several GS/s depending on the target throughput. The DAC output voltage (or current) drives an optical modulator that encodes the activation onto the light amplitude or phase.
- **Modulator types**: Mach‑Zehnder modulators (MZMs) require a push‑pull drive and often have a nonlinear transfer function, needing pre‑distortion or biasing. Microring modulators are smaller and more energy‑efficient but sensitive to temperature and wavelength.
- **Interface signals**: For each input channel (wavelength or spatial mode), there is one DAC. The digital controller writes activation values into DAC registers, and the DAC continuously updates the analog level for the duration of the optical pulse.

In a real system, the DAC may be integrated on the same photonic chip (monolithic) or on a separate electrical chip (2.5D/3D integration). For FPGA emulation, the DAC is simply modeled as a quantizer.

---

### 3. Weight Programming Path: Digital → Photonic Weight Bank

The weights are stored in the photonic mesh by tuning the phase or amplitude of each interference element.

- **Control element**: Thermo‑optic phase shifters (heaters) are common; they are slow (kHz bandwidth) but stable. Carrier‑injection or electro‑optic modulators are faster but less stable and more lossy.
- **Programming DACs**: Each weight element requires a dedicated DAC (or a shared DAC with sample‑and‑hold). These DACs are often **slow** (10–100 kS/s) but need **high resolution** (10–14 bits) because the phase shift must be set accurately.
- **Interface**: The digital controller writes the desired weight value to a register, which is then converted to an analog voltage/current that drives the heater. Due to thermal time constants, the weight settles in microseconds to milliseconds. The `PTA_LOAD_WEIGHTS` command must account for this settling time.
- **Calibration**: Because phase shifters drift with temperature, a feedback loop is needed. Additional monitor photodiodes or electrical sensors measure the actual phase, and the digital controller adjusts the programming DACs.

In FPGA emulation, we can model the weight programming as a simple register write followed by a configurable delay, and we can simulate drift by slowly varying the effective weight values.

---

### 4. Detection Path: Photodetector → TIA → ADC

- **Photodetectors**: Convert light intensity (or phase difference via balanced detection) into photocurrent. The photocurrent is proportional to the optical power, which represents the sum of products in an MVM.
- **TIA**: Amplifies the tiny photocurrent and converts it to a voltage with a certain gain and bandwidth. TIAs introduce noise (thermal and shot noise) and may have limited dynamic range.
- **ADC**: Digitizes the TIA output. The effective precision is limited by the analog signal‑to‑noise ratio. Typically ADCs of 6–10 bits at high speed (GS/s) are used, but the effective number of bits (ENOB) may be lower due to analog impairments.

**Interface signals**: The ADC outputs are read by the digital controller in parallel, one per output channel (or time‑multiplexed if fewer ADCs are used). The digital controller stores the results in the output buffer.

---

### 5. Timing and Synchronization

The photonic computation itself is nearly instantaneous (light travel time within a chip is picoseconds). However, the surrounding electronics impose timing constraints:

- **DAC settling**: Activation DACs must settle before the optical pulse arrives. This requires a setup time of nanoseconds.
- **Optical pulse width**: The laser may be pulsed or continuous wave. For matrix‑vector multiplication, a short optical pulse (or a continuous wave with a specific integration time) is used. The pulse width determines the time available for photodetection and ADC conversion.
- **ADC conversion latency**: High‑speed ADCs have pipeline delays (tens of nanoseconds). The digital controller must wait for valid data.
- **Weight programming latency**: As mentioned, thermal phase shifters take microseconds to milliseconds. This is the slowest part and must be hidden by overlapping with other operations or by keeping weights resident.

The digital controller runs on its own clock (e.g., the GPU’s clock). It must generate precise control signals to sequence the DAC updates, laser pulse, and ADC sampling. This is typically done with a finite state machine (FSM) that has configurable timing parameters.

In FPGA emulation, we can model these latencies as cycle counts or using delay lines. For example, after the `PTA_MVM` command is decoded, the FSM waits a programmable number of cycles to emulate DAC settling, then pulses the “optical compute” enable, waits for ADC latency, and finally writes results.

---

### 6. Precision, Noise, and Calibration

- **Effective precision**: The overall precision is limited by the least precise component. If weight DACs have 12 bits but optical crosstalk limits the signal‑to‑noise ratio, the effective weight precision may be only 5–6 bits. Similarly, activation DACs at 8 bits may be degraded by modulator nonlinearity.
- **Noise sources**:
  - Laser intensity noise (relative intensity noise, RIN)
  - Shot noise in photodetectors
  - Thermal noise in TIAs
  - Quantization noise from ADCs
  - Thermal crosstalk in the photonic mesh
- **Calibration**: Periodic calibration is mandatory. The `PTA_CALIBRATE` command triggers a routine that sends known test vectors, measures outputs, and computes correction coefficients (e.g., scaling and offset for each output). The digital controller may also adjust weight programming voltages to compensate for drift.

The analog‑digital interface must support two types of data flow:

1. **Fast data path**: Streaming activations in and results out, with minimal latency.
2. **Slow control path**: Programming weights, reading monitor signals, and performing calibration.

These two paths often use different buses: a high‑speed parallel interface for DAC/ADC data, and a slower serial interface (e.g., I²C, SPI) for control and monitoring.

---

### 7. Interface with Digital Controller and DMA

The digital controller (which we designed earlier) manages the analog‑digital interface through memory‑mapped registers and dedicated signals.

**Activation data path**:

- The DMA engine writes activation vectors into an input buffer (SRAM).
- The controller reads from this buffer and writes the values to the activation DACs. This can be done in a burst, with the DACs latching the new values on a trigger.
- Alternatively, the DMA can write directly to the DAC registers if they are memory‑mapped, eliminating the input buffer.

**Weight programming**:

- The controller receives `PTA_LOAD_WEIGHTS` and fetches weight data from memory (or weight cache).
- It then sequentially writes each weight value to the appropriate weight‑programming DAC register.
- After all writes, it waits for the thermal settling time before signaling completion.

**Result readout**:

- The ADCs continuously convert the photodetector outputs.
- The controller triggers a sampling event (if not continuous), waits for ADC latency, and reads the digital outputs into the output buffer.
- The DMA then writes the output buffer to memory.

**Calibration**:

- The controller may have dedicated hardware to inject test patterns and read the ADC outputs.
- It can compute corrections on‑chip (using small ALUs) or delegate to the GPU driver.

**Interrupts/Status**:

- The interface includes status registers for temperature, laser power, ADC overrange, and error flags.
- The controller can raise an interrupt to the host when an abnormal condition occurs.

---

### 8. FPGA Emulation of the Analog‑Digital Interface

In the FPGA emulation phase, we replace the physical analog components with digital models. This is relatively straightforward because the interface is already defined in terms of digital control signals and data buses.

**Modeling the DAC/ADC**:

- The activation DAC is modeled as a quantizer: it takes an 8‑bit digital value and outputs the same value but with possible added noise (if desired). The noise can be generated by a pseudo‑random number generator.
- The weight‑programming DAC is modeled as a register that stores the quantized weight value. We can also add a small random offset to simulate programming error.
- The ADC is modeled as a quantizer with configurable resolution and optional additive Gaussian noise.

**Modeling the photonic MVM**:

- The core computation is y=Wxy=Wx. In FPGA, we implement this as a matrix‑vector multiply using DSP blocks. The weights are the quantized values from the weight registers, and the inputs are the quantized activation values.
- We can add noise to the output before quantization to mimic photodetector/TIA noise.
- Latency is emulated by inserting a pipeline delay of the appropriate number of clock cycles after the DAC update and before the ADC read.

**Modeling weight drift**:

- We can implement a slow drift generator (e.g., a linear feedback shift register that slowly changes a baseline value) that adds a time‑varying offset to the stored weights. This drift can be reset or corrected by the `PTA_CALIBRATE` emulation routine.

**Control signals**:

- The FSM that sequences the analog operations is implemented identically to the real design. It controls the emulated DAC/ADC and the emulated photonic core.

This approach lets us validate the entire digital control logic, timing, and software stack without needing any analog hardware.

---

### 9. Moving to a Physical Photonic Chip

When transitioning from FPGA emulation to a real photonic integrated circuit (PIC), the analog‑digital interface becomes a hardware design problem involving:

- **Chip‑to‑chip interconnects**: The FPGA (or ASIC) must be connected to the photonic chip. High‑speed DAC/ADC signals may require serializer/deserializer (SerDes) or parallel LVDS interfaces.
- **Power and thermal management**: The heaters and lasers require stable power supplies and possibly thermoelectric coolers.
- **Packaging**: Co‑packaging (2.5D interposer, 3D stacking) can reduce parasitics and improve bandwidth.
- **Mixed‑signal verification**: The digital controller and analog circuits must be co‑simulated using tools like Cadence AMS, Synopsys Custom Compiler, or Verilog‑AMS to ensure timing and noise margins.
- **PDK selection**: Use a photonic PDK (e.g., AIM Photonics, imec, Tower) that includes models for modulators, photodetectors, and phase shifters. The digital controller can be synthesized using standard CMOS logic and placed on the same die (if monolithic) or on a separate die.

The RTL for the digital controller, DMA, and command handling can remain largely unchanged; only the low‑level driver modules that interface to the actual DAC/ADC chips or analog IP blocks need to be adapted.

---

### 10. Summary

The analog‑digital interface is a mixed‑signal subsystem that bridges the digital GPU with the analog photonic compute fabric. Its design requires careful consideration of:

- High‑speed DACs/ADCs for activations and results.
- Slow, high‑resolution DACs for weight programming.
- Timing synchronization to hide analog latencies.
- Noise and calibration to achieve useful precision.
- A clear separation between fast data streaming and slow control.

In your PhD work, you can first fully define and implement this interface in FPGA emulation, then later replace the emulated models with actual photonic hardware while retaining the same digital infrastructure. This incremental approach minimizes risk and provides publishable results at each stage.

Would you like to delve deeper into any specific aspect of the analog‑digital interface, such as the DAC/ADC selection criteria, the control FSM design, or calibration algorithms?

---

Yes, absolutely. A RISC‑V 64‑bit quad‑core processor fits naturally into this project in several important roles. It can serve as the **host control processor**, an **embedded management controller** inside the PTA, or as part of your FPGA emulation setup. Its open‑source nature and flexibility make it an ideal companion for your existing GPU and the photonic accelerator.

---

### 1. Host CPU for System Control and Software

In a typical heterogeneous system (like a server or an accelerator card), there is a host CPU that:

- Runs the operating system (e.g., Linux).
- Executes the device drivers for the GPU and PTA.
- Manages memory allocation and address translation.
- Submits commands to the GPU and PTA.

Your quad‑core RISC‑V processor can fill this role. If it is a physical chip (e.g., SiFive U74, T-Head C910, or similar), it can run Linux and communicate with the FPGA (or a future ASIC) over PCIe, AXI, or other interfaces. In the FPGA emulation phase, you can also instantiate a soft‑core RISC‑V (like Rocket or BOOM) inside the FPGA as the host, eliminating the need for an external CPU and simplifying the test bench.

**What changes in our previous design?**

Nothing fundamentally. The GPU command processor and PTA command interface remain as described. The RISC‑V host simply writes command descriptors to the PTA’s memory‑mapped FIFO or ring buffer, just as any host CPU would. The driver code you write for the RISC‑V is almost identical to what you would write for x86, because the PTA’s programming model is host‑agnostic.

---

### 2. Embedded Controller Inside the PTA

The PTA’s digital controller (which handles command decoding, DMA, and calibration sequencing) can be implemented as a small **RISC‑V core** instead of a custom finite state machine. This is a common practice in modern accelerators—using an embedded microcontroller for:

- Boot and initialization.
- Running calibration algorithms (e.g., iterative weight tuning).
- Monitoring temperature, laser power, and error conditions.
- Managing slow control interfaces (I²C/SPI for DACs, ADCs, and sensors).
- Handling complex error recovery and logging.

A RISC‑V core with a small amount of local SRAM and a few peripherals (UART, timers, GPIO) can replace much of the custom control logic we discussed. The advantage is that calibration routines and power‑management policies can be written in C and updated easily, without resynthesizing the whole accelerator. The RISC‑V would run a bare‑metal or lightweight RTOS, and communicate with the main GPU/host through mailboxes or shared memory.

In our previous design, the **digital control unit** and **calibration FSM** could be replaced by a RISC‑V core. The DMA engine and high‑speed data paths would remain dedicated RTL for performance, but the core would orchestrate them.

**How this fits:**

- The command decoder might become a simple hardware block that passes commands to the RISC‑V for interpretation, or the RISC‑V itself polls memory‑mapped registers from the GPU.
- Weight programming sequences (writing to slow DACs, waiting for settling) would be driven by software on the RISC‑V.
- Calibration routines could be implemented as software loops that read ADC outputs and adjust parameters.

This is a very practical use of your existing RISC‑V core and reduces the amount of custom RTL you need to write.

---

### 3. Role in FPGA Emulation

In the emulation phase, you have two options:

- **External RISC‑V host**: Connect a physical RISC‑V board (e.g., a SiFive HiFive Unmatched or a custom board) to the FPGA via PCIe or Ethernet. The host runs the driver and test software, sending commands to the PTA inside the FPGA.
- **On‑chip soft‑core RISC‑V**: Instantiate a RISC‑V core (like PicoRV32 or a full Rocket core) inside the FPGA itself, along with a small Linux or bare‑metal environment. This makes the emulation self‑contained and avoids the complexity of a high‑speed external interface. It also allows you to test the entire software stack on the FPGA without additional hardware.

Given that you already have a quad‑core RISC‑V, it might be easier to use it as an external host. Many FPGA development boards have PCIe connectors that can interface with a host CPU; but if your board lacks PCIe, you can use a simpler interface like UART, Ethernet, or even a memory‑mapped bus.

**In short:** Your RISC‑V processor is a valuable asset. It can serve as the **host CPU** to run the software and control the FPGA, or as an **embedded controller** inside the PTA to handle calibration and management. Either way, it integrates seamlessly with the architecture we’ve outlined.

---

### Next Steps

With the RISC‑V in the picture, you can now plan a concrete emulation platform:

- Use the RISC‑V as the host running Linux.
- Connect it to your FPGA board (which contains your GPU and the PTA emulator).
- Write the driver and test programs on the RISC‑V.
- Validate the entire command flow, DMA, and emulated photonic compute.

Would you like to discuss the specific interfacing between the RISC‑V host and the FPGA (e.g., PCIe, AXI, or memory‑mapped), or would you prefer to continue with the analog‑digital interface details we deferred earlier?

---

## Interfacing a RISC‑V Host with the FPGA‑Based PTA Emulation

Now we focus on the concrete connection between your RISC‑V processor (as host) and the FPGA that contains your GPU and the emulated Photonic Tensor Accelerator (PTA). The goal is to enable the RISC‑V to:

- Boot an operating system (likely Linux) or run bare‑metal software.
- Load device drivers for the PTA.
- Allocate memory buffers accessible by the PTA.
- Submit commands to the PTA (via MMIO or ring buffer).
- Receive completion interrupts or poll status.
- Transfer data between host memory and the FPGA.

There are several standard interfaces, each with different complexity, bandwidth, and latency. The choice depends on your available hardware, development effort, and performance goals.

---

### 1. Interface Options

#### A. PCI Express (PCIe)

**Description:**

PCIe is the natural high‑speed interface for connecting a host processor to an accelerator FPGA. Many FPGA boards (e.g., Xilinx Alveo, Intel Agilex, or custom boards with FMC connectors) have PCIe edge connectors or can be plugged into a PCIe slot. A RISC‑V board with a PCIe root complex (like the SiFive HiFive Unmatched, or a custom board with a RISC‑V SoC that has PCIe) can directly host the FPGA.

**Advantages:**

- High bandwidth (Gen3 x4 ~ 4 GB/s, Gen4 x8 ~ 16 GB/s).
- Low latency (sub‑microsecond for MMIO).
- Standard Linux driver model (PCIe device, BAR spaces, MSI‑X interrupts).
- Allows DMA from the FPGA to host memory (if FPGA has a PCIe endpoint with bus mastering).
- Mature IP cores available for FPGAs (Xilinx PCIe, Intel PCIe).

**Disadvantages:**

- Requires FPGA board with PCIe connector or a carrier board.
- RISC‑V host must have a PCIe root complex; not all RISC‑V SoCs have this (many low‑cost ones don't).
- More complex bring‑up (need to handle PCIe enumeration, BAR mapping, etc.).
- Higher power and cost.

**Implementation Sketch:**

- Instantiate a PCIe endpoint in the FPGA (e.g., Xilinx XDMA, Intel Avalon‑ST PCIe).
- Map the PTA's register space (command FIFO, status, control) into a PCIe BAR.
- Use DMA engines in the FPGA (or the XDMA's own DMA) to move data between host memory and PTA buffers.
- The RISC‑V driver uses standard Linux PCIe APIs to access the device.

**When to choose:** If you need high throughput and low latency, and your RISC‑V board supports PCIe.

#### B. Ethernet (GbE or 10GbE)

**Description:**

Use standard Ethernet to connect the RISC‑V host to the FPGA board. The FPGA implements a soft‑core MAC (or uses a hard MAC if available) and runs a lightweight network stack (e.g., LWIP on a soft‑core processor, or dedicated hardware). The RISC‑V host communicates via TCP/UDP or a custom protocol.

**Advantages:**

- Universal: any RISC‑V board with Ethernet can be used.
- Long cable lengths possible.
- Easy to debug with standard networking tools.
- No special hardware required if both boards have Ethernet ports.

**Disadvantages:**

- Higher latency (tens of microseconds to milliseconds).
- Lower bandwidth than PCIe (1 GbE ~ 125 MB/s, 10 GbE ~ 1.25 GB/s).
- Requires implementing a network stack or offloading to a processor inside the FPGA, adding complexity.
- Not ideal for fine‑grained MMIO or frequent small commands.

**Implementation Sketch:**

- FPGA runs a soft‑core (e.g., MicroBlaze, Nios, or RISC‑V) that manages an Ethernet MAC and a simple command protocol.
- Host sends commands as Ethernet frames; the soft‑core decodes them and writes to PTA registers.
- Data can be streamed over Ethernet directly to PTA buffers using dedicated hardware, or via the soft‑core.

**When to choose:** If you only have Ethernet available, or you want a simple remote control interface for testing (e.g., controlling the FPGA from a PC or a RISC‑V board over a network).

#### C. UART / Serial

**Description:**

A simple UART connection for low‑speed control and debugging. The RISC‑V host sends commands over UART, and a small state machine or soft‑core inside the FPGA interprets them.

**Advantages:**

- Extremely simple to implement on both sides.
- Available on almost every board.
- Good for initial testing and bring‑up.

**Disadvantages:**

- Very low bandwidth (typically 115200 bps to a few Mbps).
- High latency for large data transfers.
- Not suitable for performance evaluation, only functional testing.

**Implementation Sketch:**

- FPGA has a UART receiver that feeds a command parser (could be a soft‑core or a simple FSM).
- Commands are textual or binary; responses are sent back.
- Data is transferred in small chunks, possibly base64 encoded.

**When to choose:** Early stages of development, when you just need to verify the PTA logic without high‑speed interfaces. Often used for debug.

#### D. External Memory‑Mapped Bus (e.g., AXI over FMC, GPIO, or custom parallel bus)

**Description:**

If the RISC‑V board and FPGA are on the same custom board, you can connect them via a parallel bus (e.g., AXI, Wishbone, or a simple SRAM‑like interface) using GPIO pins. The RISC‑V treats the FPGA as a memory‑mapped peripheral.

**Advantages:**

- Low latency (direct bus access).
- Can be high bandwidth if many pins are used.
- Simplifies the software model: just memory reads/writes.

**Disadvantages:**

- Requires custom hardware (board design, connector, level shifters).
- Pin count can be large for wide buses.
- Signal integrity issues at high speeds.

**Implementation Sketch:**

- Use an FMC connector with a large number of I/Os.
- Implement an AXI master in the RISC‑V SoC (if it doesn't already have one) and an AXI slave in the FPGA.
- The RISC‑V can directly read/write PTA registers and memory.

**When to choose:** If you are designing a custom board anyway, or you have a development kit that exposes a high‑speed parallel interface.

#### E. USB

**Description:**

Connect via USB (e.g., USB 2.0/3.0). The RISC‑V host is a USB host, and the FPGA implements a USB device controller (or uses a USB‑to‑FIFO bridge chip like FTDI).

**Advantages:**

- Common and easy to use.
- FTDI chips simplify the FPGA side.
- USB 3.0 offers decent bandwidth (up to 5 Gbps).

**Disadvantages:**

- Latency is moderate.
- Requires USB stack on host (Linux has it built‑in).
- FTDI chips may limit throughput and flexibility.

**Implementation Sketch:**

- Use an FTDI FT600/FT601 (USB 3.0 to FIFO) on the FPGA board.
- The RISC‑V driver communicates via libusb or a custom USB driver.
- Data is streamed through the FIFO to PTA buffers.

**When to choose:** If you want a plug‑and‑play solution without PCIe, and Ethernet is not required.

---

### 2. Software Stack Considerations

Regardless of the physical interface, the software running on the RISC‑V host will have a similar structure:

1. **Kernel driver** (if using Linux) or **bare‑metal library** that:
  - Probes the device (via PCIe, USB, etc.).
  - Maps device registers into memory (mmap).
  - Allocates DMA buffers (contiguous memory or scatter‑gather).
  - Provides an API for user space to submit commands.
  - Handles interrupts (if used) or provides polling.
2. **User‑space library** (e.g., `libpta`) that:
  - Encapsulates command building.
  - Manages weight loading, MVM execution, and result retrieval.
  - Exposes high‑level functions like `pta_mvm(A, B, C)`.

For the FPGA side, you will need a **device controller** that interfaces with the chosen physical layer:

- For PCIe: an IP core like Xilinx XDMA, which provides a simple AXI interface.
- For Ethernet: a soft‑core with LWIP or a hardware packet parser.
- For UART: a simple UART receiver and command FSM.
- For FTDI: a FIFO interface to the PTA's internal buses.

The PTA's internal architecture (command FIFO, DMA, buffers) remains unchanged; only the front‑end interface to the host differs.

---

### 3. Recommended Approach for Your Setup

Given that you have a **RISC‑V64 quad‑core processor** (likely a physical chip on a development board) and an FPGA board (where you plan to implement the GPU and PTA emulation), the most straightforward and high‑performance option is **PCIe**, if your RISC‑V board supports it. Many RISC‑V development boards (e.g., SiFive HiFive Unmatched, Microchip PolarFire SoC Icicle Kit) have PCIe root ports.

If PCIe is not available, **Ethernet** is a good fallback, especially if both boards have Ethernet ports and you don't mind the extra latency.

For initial bring‑up, you might start with **UART** to test the command flow and basic functionality, then move to PCIe/Ethernet for performance.

#### Example: PCIe‑Based Integration

- **FPGA side:**
  - Use a Xilinx FPGA with a PCIe endpoint (e.g., XDMA IP). The XDMA provides:
    - AXI4‑Lite master/slave for register access.
    - AXI Memory‑Mapped or AXI Stream for DMA.
  - Connect the PTA's register file (command FIFO, status, control) to the XDMA's AXI‑Lite slave.
  - Connect the PTA's DMA engine to the XDMA's AXI‑MM master port, enabling direct memory access to host buffers.
  - Map the entire PTA address space into a PCIe BAR.
- **RISC‑V host side:**
  - Boot Linux (e.g., Buildroot or Yocto).
  - Write a PCIe driver that:
    - Probes the device (Vendor/Device ID).
    - Maps BAR0 to access PTA registers.
    - Allocates DMA buffers using `dma_alloc_coherent()`.
    - Implements `mmap` for user space.
    - Registers an interrupt handler for MSI‑X.
  - User‑space library builds commands and writes them to the command FIFO; the driver handles doorbell and completion.
- **Data flow:**
  - User allocates input/output buffers in host memory.
  - Commands reference these buffers by physical address.
  - PTA's DMA engine reads inputs from host memory via PCIe, processes them, and writes outputs back.
  - Completion is signaled via interrupt or poll.

This setup allows high‑speed testing and closely mimics the final system where the PTA would be a PCIe accelerator.

---

### 4. Alternative: Embed RISC‑V Inside the FPGA

If you have the FPGA resources and want a self‑contained system, you can instantiate a **soft‑core RISC‑V** (e.g., Rocket, BOOM, PicoRV32) inside the FPGA alongside your GPU and PTA emulation. The soft‑core runs Linux or bare‑metal, and communicates with the PTA via on‑chip buses (AXI, TileLink). This eliminates the need for external interfaces and gives you full control over timing and debugging. It is also a good option if you want to test the whole stack on a single FPGA without external dependencies.

However, a soft‑core running Linux will consume significant FPGA logic and block RAM, possibly competing with your GPU design. You may need a large FPGA (e.g., Xilinx UltraScale+ or Intel Stratix 10). For a PhD project, this might be overkill if you already have a physical RISC‑V board.

---

### 5. Summary

Your RISC‑V quad‑core processor is an excellent host for the FPGA‑based PTA emulation. The most performant and standard interface is PCIe, but Ethernet or UART are simpler alternatives for early testing. The software and hardware architecture we defined earlier remains valid; only the physical connection changes. I recommend starting with UART to validate the PTA logic, then upgrading to PCIe or Ethernet for performance measurements.

Would you like to go deeper into any specific interface (e.g., detailed PCIe integration steps, or how to implement a simple UART command protocol)? Or shall we return to the analog‑digital interface topics we postponed?

---

## Interconnecting RISC‑V Host with GPU/PTA on a Real Motherboard

When moving from FPGA prototyping to a real multi‑chip system on a motherboard, the interconnect between the RISC‑V host processor and the GPU/PTA accelerator becomes a critical design decision. The goal is to provide high bandwidth, low latency, and a coherent or at least well‑defined memory model for efficient data movement.

While NVLink is NVIDIA’s proprietary coherent interconnect, there are open and standard alternatives that are well suited for RISC‑V ecosystems. Below I outline the realistic options, their trade‑offs, and a recommended path for your project.

---

### 1. Baseline: PCI Express (PCIe)

**What it is:**

PCIe is the universal interconnect for attaching accelerators to a host CPU. It uses high‑speed serial lanes (e.g., 16 GT/s for Gen4, 32 GT/s for Gen5) aggregated into links of ×1, ×4, ×8, ×16.

**Advantages:**

- Mature, widely supported, and available in all modern RISC‑V SoCs that target server/desktop applications (e.g., SiFive Performance P550, Ventana Veyron, Esperanto ET‑SoC‑1).
- Standard software model: PCIe enumeration, BARs, MSI‑X interrupts, DMA.
- Sufficient bandwidth for many AI workloads (PCIe Gen5 ×16 ≈ 64 GB/s per direction).
- Low latency for MMIO (~100 ns) and DMA.

**Disadvantages:**

- Non‑coherent by default (requires explicit cache flushes or using non‑cacheable memory).
- Limited scalability beyond a few devices without switches.
- Latency for small transfers can be higher than a dedicated coherent link.

**Use case:**

This is the most practical and likely initial implementation. Your RISC‑V host can use a standard PCIe root complex, and the GPU/PTA chip can be designed as a PCIe endpoint. The PTA’s digital controller and DMA engine already align with PCIe’s model (BARs for registers, DMA for data movement).

---

### 2. Compute Express Link (CXL)

**What it is:**

CXL is an open industry standard built on the PCIe physical layer but adds three protocols:

- **[CXL.io](https://cxl.io/)** – essentially PCIe for enumeration, configuration, and MMIO.
- **CXL.cache** – allows an accelerator to cache host memory coherently.
- **CXL.mem** – allows the host to access accelerator‑attached memory (or vice versa) with cache coherence.

**Advantages:**

- Low latency, cache‑coherent memory sharing between host and accelerator.
- Ideal for workloads where the accelerator needs to share data structures with the CPU without explicit copying.
- Backed by major industry players (Intel, AMD, Arm, and increasingly RISC‑V vendors).
- Can use the same physical SerDes as PCIe, simplifying board design.

**Disadvantages:**

- Newer, less mature in RISC‑V ecosystem (but adoption is growing).
- Requires more complex hardware controllers and software support (CXL drivers, coherent interconnects).
- Not yet as ubiquitous as PCIe.

**Use case:**

If your GPU/PTA needs fine‑grained memory sharing with the host (e.g., the CPU and photonic accelerator cooperatively process a neural network, with weights and activations in a shared coherent memory space), CXL is the best open alternative to NVLink. It provides cache coherence and low latency, enabling a programming model closer to a unified memory space.

For a photonic tensor accelerator, CXL.cache could allow the PTA to directly read weights and activations from host memory with hardware coherence, avoiding explicit DMA copies.

---

### 3. Custom High‑Speed Serial Link (SerDes)

**What it is:**

If you are designing both the RISC‑V host SoC and the GPU/PTA chip, you can implement a proprietary or custom point‑to‑point serial interconnect using high‑speed SerDes transceivers (e.g., 56 Gbps or 112 Gbps per lane). You define your own protocol for data movement and control.

**Advantages:**

- Maximum flexibility: can optimize for ultra‑low latency, very high bandwidth (multi‑Tbps), and custom coherence or streaming semantics.
- Can avoid overheads of PCIe/CXL protocol layers if not needed.
- Allows tight integration of photonic interconnect if desired (optical SerDes).

**Disadvantages:**

- High development cost and risk; need to design, verify, and validate the entire protocol stack.
- No standard software support; you must write drivers, libraries, and possibly OS support from scratch.
- Board‑level signal integrity and power become major challenges.

**Use case:**

This is akin to what NVIDIA did with NVLink, or what Google did with TPU interconnects. It makes sense only for high‑volume or performance‑critical products, not for a research prototype. However, for a PhD project, exploring a custom lightweight protocol over SerDes could be a novel contribution, especially if combined with photonic interconnects.

---

### 4. Chiplet Interconnects (UCIe, BoW, AIB)

**What it is:**

If the RISC‑V host and the GPU/PTA are integrated as chiplets in the same package (multi‑chip module), you can use die‑to‑die interconnects like **UCIe** (Universal Chiplet Interconnect Express), **BoW** (Bunch of Wires), or **AIB** (Advanced Interface Bus). These are short‑reach, high‑bandwidth, low‑power links designed for in‑package communication.

**Advantages:**

- Extremely high bandwidth density (multiple Tbps per mm of edge).
- Low latency and low power per bit.
- Can implement cache coherence or streaming protocols efficiently.
- UCIe is an open standard gaining industry adoption.

**Disadvantages:**

- Requires advanced packaging (2.5D interposer, 3D stacking), which is expensive and complex.
- Not a board‑level solution; the chips must be co‑packaged.
- Design and verification of chiplet interfaces is still evolving.

**Use case:**

For a future product where the photonic accelerator and host CPU are tightly integrated (e.g., to minimize data movement energy and latency), chiplet integration is attractive. The photonic chip could be a separate chiplet connected via UCIe to a digital logic chiplet containing the RISC‑V cores and GPU control logic. This is a longer‑term goal.

---

### 5. What About NVLink?

NVLink is proprietary to NVIDIA and not available to other vendors. However, its characteristics (high‑bandwidth, low‑latency, coherent) are exactly what CXL aims to provide as an open standard. If you are working with NVIDIA GPUs, NVLink is an option, but your project is building a custom GPU/PTA, so you would not use NVLink. Instead, CXL or a custom coherent link would be the equivalent.

---

### 6. System Architecture Comparison

<table>
  <tr><th>Interconnect</th><th>Bandwidth per link</th><th>Latency</th><th>Coherence</th><th>Open Standard</th><th>Maturity</th></tr>
  <tr><td>PCIe Gen5 ×16</td><td>~64 GB/s per direction</td><td>~100 ns MMIO</td><td>No (but can use ATS)</td><td>Yes</td><td>Very high</td></tr>
  <tr><td>CXL 2.0 (×16)</td><td>Similar to PCIe Gen5</td><td>Lower than PCIe for memory</td><td>Yes (cache/mem)</td><td>Yes</td><td>Growing</td></tr>
  <tr><td>Custom SerDes (e.g., 8×112G)</td><td>~112 GB/s per direction (scalable)</td><td>Can be &lt;50 ns</td><td>Custom</td><td>No</td><td>Low (custom)</td></tr>
  <tr><td>UCIe (chiplet)</td><td>&gt;1 TB/s per mm edge</td><td>&lt;10 ns</td><td>Optional</td><td>Yes</td><td>Emerging</td></tr>
</table>

---

### 7. Recommended Path for Your Project

Given that you are building a GPU to the point of synthesis and now planning to add a photonic tensor accelerator, the most pragmatic approach for a real motherboard is:

1. **Start with PCIe** as the host‑accelerator interconnect.
  - Your RISC‑V host (e.g., a SiFive or Ventana based SoC) will have a PCIe root port.
  - The GPU/PTA chip is designed as a PCIe endpoint with BARs for control/status and DMA engines for data.
  - This is fully supported by standard Linux drivers and is the fastest way to get a working system.
2. **Evaluate CXL** if you need tighter memory coherence and lower latency for fine‑grained sharing.
  - Many upcoming RISC‑V server SoCs are including CXL controllers.
  - CXL could be a differentiator in your research, showing how a photonic accelerator can benefit from coherent memory access.
3. **Consider custom SerDes or chiplet integration only in the far future** if you are designing a full system‑on‑package with custom silicon.

In all cases, the digital architecture we have defined (command queue, DMA, buffers) remains valid; only the physical transport and its protocol adapt.

---

### 8. Summary

On a real motherboard, the RISC‑V host will most likely interconnect with the GPU/PTA via **PCIe** initially, with the option to migrate to **CXL** for cache‑coherent acceleration. Custom point‑to‑point serial links (like NVLink) are possible but require significant engineering effort and are best suited for high‑volume products. Chiplet interconnects become relevant if you co‑package the photonic and digital dies.

Your existing plans for the PTA’s DMA and command interface are already well aligned with PCIe’s programming model, making the transition from FPGA emulation to a real PCIe endpoint straightforward.

Would you like to explore how to design the PTA as a PCIe endpoint, or perhaps dive into CXL’s coherent memory model and how it would change the PTA’s data movement?

---

## Diving into CXL for the RISC‑V Host and Photonic Tensor Accelerator

Compute Express Link (CXL) is an open industry standard that provides cache‑coherent interconnect on top of PCIe physical layer. It enables low‑latency, high‑bandwidth communication between a host CPU and accelerators, memory expanders, or other devices, with hardware‑enforced coherence. For a RISC‑V based system integrating a GPU and a photonic tensor accelerator (PTA), CXL offers a compelling alternative to traditional PCIe DMA, especially when fine‑grained data sharing and low overhead are desired.

---

### 1. CXL Protocol Overview

CXL runs on PCIe electricals but adds three distinct protocols:

- **[CXL.io](https://cxl.io/)** – Essentially PCIe: device discovery, configuration, MMIO, and DMA. This is the control plane.
- **CXL.cache** – Allows an accelerator to cache host memory (or other device memory) coherently. The device can issue cacheable reads/writes to host memory, and the host’s coherence protocol ensures consistency. This enables the accelerator to have a hardware‑coherent view of shared data structures.
- **CXL.mem** – Allows the host to access device‑attached memory (e.g., memory on the accelerator) as if it were system memory, with coherence. The device can expose its local memory to the host, enabling unified memory spaces.

Together, these protocols enable three device types:

- **Type 1**: Accelerator with no local memory (or limited) that relies primarily on caching host memory via CXL.cache.
- **Type 2**: Accelerator with local memory that participates fully in the coherent domain; it can cache host memory and its memory can be accessed by the host via CXL.mem.
- **Type 3**: Memory expander – just provides additional memory to the host, no compute.

A GPU/PTA would typically be a **Type 2** device because it has its own memory (e.g., weight buffers, activation storage) and benefits from coherent sharing with the host.

---

### 2. Why CXL for a Photonic Tensor Accelerator?

- **Low‑latency, fine‑grained sharing**: Instead of staging data through DMA buffers and explicit cache flushes, the PTA could directly read weights and activations from host memory using CXL.cache. This is ideal for small, latency‑sensitive operations or where the working set changes frequently.
- **Reduced data copies**: With CXL.mem, the host can write results directly into PTA‑attached memory, or the PTA can write back to host memory without software‑managed buffers. The programmer sees a unified address space.
- **Cache coherence**: The PTA can operate on the same virtual addresses as the CPU threads, enabling tight coupling in heterogeneous programming models (e.g., OpenMP, CUDA‑like unified memory). The hardware manages consistency, simplifying software.
- **Scalability and future‑proofing**: CXL is emerging as the standard for coherent accelerators, backed by major industry players and increasingly adopted in RISC‑V ecosystem. Using CXL positions your project at the forefront of accelerator integration.

For the photonic accelerator specifically, CXL.cache could allow the photonic weight bank to be treated as a memory‑mapped device that the host writes to directly (via MMIO or cacheable writes), and the results read back coherently. This reduces the need for the complex command queue and DMA engine we designed earlier; instead, the accelerator can be controlled via memory‑mapped registers and shared memory regions.

---

### 3. Changes Required to the PTA Architecture

The PTA we previously defined (command queue, DMA, buffers) can still work over [CXL.io](https://cxl.io/) (which is PCIe‑like). However, to fully leverage CXL’s coherence, the PTA’s digital interface should be redesigned around **memory‑mapped, cacheable accesses** rather than asynchronous DMA commands.

#### 3.1 Replace DMA with Direct Coherent Access

- **Input activations**: The PTA could issue CXL.cache read requests directly to host memory addresses, rather than having the host enqueue DMA descriptors. The PTA’s internal logic would generate memory reads for the required data as needed.
- **Weights**: Weights could be stored in host memory and fetched via CXL.cache, or they could be kept in PTA‑attached memory (CXL.mem) so the host can update them with normal stores.
- **Outputs**: The PTA writes results to a coherent memory region (either host memory or its own memory exposed via CXL.mem), and the host reads them as normal memory.

This eliminates the need for a separate DMA engine and reduces command overhead; the PTA can be controlled via memory‑mapped registers and memory‑resident command structures (e.g., doorbells, status flags).

#### 3.2 Memory Map and Address Translation

- The PTA must have an **IOMMU or SMMU** to translate virtual addresses used by the host processes into physical addresses, and to enforce protection. CXL requires the device to perform address translation for coherent accesses.
- The PTA’s internal address space (e.g., weight buffer, control registers) can be exposed to the host via **CXL.mem**, allowing the host to directly read/write them. This is similar to how GPU local memory is exposed.

#### 3.3 Coherence Protocol Implementation

- The PTA must implement a **CXL.cache agent** (typically a home agent or requester) that handles snoop requests and maintains coherence with the host’s cache hierarchy.
- This is a significant hardware complexity, but there are emerging IP blocks and open‑source efforts (e.g., from Chips Alliance) that could be adapted.

#### 3.4 Software Model

- The driver no longer needs to allocate DMA buffers and issue explicit commands. Instead, it can use **shared virtual memory** (SVM) or simply memory‑map the PTA’s control registers and data buffers into user space.
- The PTA can be programmed by writing to memory‑mapped registers, and execution is triggered by a doorbell write (which is itself a coherent store).
- Completion can be signaled via an interrupt (MSI‑X over [CXL.io](https://cxl.io/)) or by polling a status flag in coherent memory.

This model is much closer to how a CPU core interacts with a tightly coupled accelerator, reducing software overhead and enabling finer‑grained offload.

---

### 4. CXL vs PCIe for This Project

<table>
  <tr><th>Feature</th><th>PCIe DMA (traditional)</th><th>CXL (coherent)</th></tr>
  <tr><td>Latency for small transfers</td><td>High (requires DMA setup, flushes)</td><td>Low (cache‑coherent reads/writes)</td></tr>
  <tr><td>Programming complexity</td><td>Driver manages buffers, commands</td><td>Simpler: shared memory, doorbells</td></tr>
  <tr><td>Cache coherence</td><td>None (explicit flushes)</td><td>Hardware‑enforced</td></tr>
  <tr><td>Data sharing granularity</td><td>Coarse (page or buffer)</td><td>Fine (cache line)</td></tr>
  <tr><td>Bandwidth</td><td>High (but overhead for small)</td><td>High (with lower latency)</td></tr>
  <tr><td>Hardware complexity</td><td>Lower (standard PCIe endpoint)</td><td>Higher (coherence engine, home agent)</td></tr>
  <tr><td>Ecosystem maturity</td><td>Very mature</td><td>Growing, but RISC‑V support emerging</td></tr>
</table>

For a photonic tensor accelerator, the lower latency and coherence are particularly attractive because the photonic core itself is extremely fast; the bottleneck becomes data movement and control overhead. CXL can reduce that overhead.

---

### 5. RISC‑V and CXL: Current Status

The RISC‑V ecosystem is actively adopting CXL:

- Several upcoming RISC‑V server SoCs (e.g., Ventana Veyron, Esperanto ET‑SoC‑1, SiFive Performance P550) include CXL controllers.
- Open‑source RTL for CXL IP is emerging, though not yet as mature as PCIe.
- The **CXL Consortium** has RISC‑V members, and there are efforts to define RISC‑V specific extensions for CXL.
- For academic projects, you can either license a CXL controller IP from vendors (e.g., Synopsys, Cadence, Rambus) or use FPGA implementations provided by Intel/Xilinx (though these are often limited to Type‑3 memory devices currently).

Implementing a full CXL Type‑2 device is a significant undertaking but could be a major contribution if you develop an open‑source CXL.cache agent for RISC‑V.

---

### 6. Practical Implementation Path

Given your FPGA background and the goal of adding a photonic accelerator, here is a pragmatic approach:

1. **Start with PCIe/[CXL.io**](https://cxl.io/) – First, connect the PTA as a standard PCIe endpoint using the command queue and DMA design. This is simpler and allows you to validate the photonic emulation and software.
2. **Explore CXL.mem** – Add CXL.mem capability so the host can directly access PTA’s internal buffers (e.g., weight memory, results) as if they were system memory. This reduces the need for DMA transfers and simplifies the driver.
3. **Prototype on FPGA** – Use a CXL‑capable FPGA platform (e.g., Intel Agilex with CXL IP, or Xilinx Versal with CXL support) to implement a Type‑2 or Type‑3 device. There are development kits that support CXL connectivity to a host CPU (often x86, but soon RISC‑V boards with CXL).
4. **Design a CXL.cache agent** – As a research contribution, implement a simplified CXL.cache requester in RTL that allows the PTA to directly read activation data from host memory without DMA. This can be validated in simulation and on FPGA.
5. **Integrate with RISC‑V host** – Once RISC‑V boards with CXL root ports are available (or if you design your own SoC), connect your PTA as a CXL Type‑2 device and run the full software stack using shared virtual memory.

---

### 7. Research Opportunities

Using CXL opens several novel research directions:

- **Coherent memory management for photonic accelerators** – How to manage weights and activations in a shared coherent space, exploiting the photonic core’s speed while hiding memory latency.
- **Custom CXL.cache policies** – Optimizing cache coherence for analog/photonic compute (e.g., write‑through for weights, read‑once for activations).
- **Hardware/software co‑design** – Developing a programming model that leverages CXL’s coherence to offload fine‑grained operations to the PTA without kernel involvement.
- **Open‑source CXL IP for RISC‑V** – Contributing a lightweight CXL agent for FPGA prototyping, enabling wider adoption in academic projects.

---

### 8. Summary

CXL provides a coherent, low‑latency interconnect that can significantly simplify the integration of a photonic tensor accelerator with a RISC‑V host. While PCIe remains the short‑term practical choice, CXL’s coherent protocols align perfectly with the needs of a high‑speed analog compute engine. By gradually migrating from PCIe to CXL, you can not only improve performance but also produce impactful research in accelerator‑host interconnects.

Would you like to delve into the specifics of implementing a CXL.cache agent, or perhaps explore how the PTA’s internal memory hierarchy would be affected by coherent access?

---

## PTA Internal Memory Hierarchy with CXL Coherent Access

When integrating the Photonic Tensor Accelerator (PTA) via CXL, the internal memory architecture must be re‑evaluated. CXL provides hardware cache coherence between the host CPU and the accelerator, which allows us to move away from explicit DMA and toward a more unified, memory‑centric programming model. This changes how we design the PTA’s internal memories, their access patterns, and the digital control logic.

---

### 1. Internal Memory Components of the PTA

In the earlier DMA‑based design, the PTA had:

- **Input Buffer** – holds activation vectors waiting for DAC conversion.
- **Weight Buffer** – stores digital weight values for programming the photonic mesh; optionally caches multiple weight sets.
- **Output Buffer** – holds ADC results before writing back to memory.
- Possibly larger local memory (e.g., SRAM or DRAM) if the PTA has its own on‑chip storage for weights or intermediate data.

These buffers were filled/drained by a DMA engine, acting as staging areas between the photonic core and the host/GPU memory. With CXL, some of these buffers can be eliminated or transformed into cacheable memory regions that are directly accessible by both host and PTA.

---

### 2. Access Patterns and Coherence Requirements

Different data in the PTA have different access characteristics:

<table>
  <tr><th>Data</th><th>Produced by</th><th>Consumed by</th><th>Access Pattern</th><th>Coherence Requirement</th></tr>
  <tr><td>Weights (digital)</td><td>Host (training, off‑line)</td><td>PTA (to program photonic mesh)</td><td>Written rarely, read often</td><td>Must be coherent if host updates weights; otherwise can be read‑only.</td></tr>
  <tr><td>Activations (input)</td><td>Host or previous layer</td><td>PTA (via DAC)</td><td>Streaming, read once</td><td>Should be coherent to allow CPU to produce them and PTA to consume without explicit copies.</td></tr>
  <tr><td>Results (output)</td><td>PTA (via ADC)</td><td>Host or next layer</td><td>Streaming, write once</td><td>Should be coherent so host can immediately read results.</td></tr>
  <tr><td>Control/Status</td><td>Host / PTA</td><td>Both</td><td>Frequent, small, low‑latency</td><td>Needs coherence for doorbells, status flags.</td></tr>
</table>

CXL.cache allows the PTA to issue coherent reads/writes to host memory. CXL.mem allows the host to issue coherent reads/writes to PTA‑attached memory. Together they can create a single shared address space.

---

### 3. CXL.cache: PTA Reading Host Memory Directly

Instead of using DMA to copy activations from host memory into an internal input buffer, the PTA can issue **cacheable read requests** over CXL.cache to fetch activation data directly from the host’s memory hierarchy. The host’s caches are kept coherent by the CXL home agent. This has several implications:

- **No input buffer needed**: The PTA can stream activations in small chunks (e.g., cache line granularity) directly from host memory to the DACs. A small FIFO may still be used to smooth latency, but the large staging buffer disappears.
- **Lower latency for small batches**: For a single matrix‑vector product, the PTA can begin as soon as the first data arrives, without waiting for a full DMA transfer.
- **Fine‑grained sharing**: If the CPU is producing activations (e.g., from a previous layer), it can write them to a buffer in normal cacheable memory, and the PTA reads that same buffer coherently. No cache flushes or driver involvement needed.
- **Potential cache pollution**: The PTA’s reads may cause cache line fill in the PTA’s own cache (if it has one), or in the host’s LLC. For streaming data, we might want to use non‑temporal hints or mark the data as non‑cacheable to avoid thrashing. CXL allows cache allocation hints.

The PTA will likely need a **load/store unit (LSU)** that can generate CXL.cache requests. This replaces the DMA read channel. The LSU can be relatively simple: it receives a virtual address range and a stride, then issues reads and passes the data to the DAC interface.

---

### 4. CXL.mem: Host Accessing PTA Memory

CXL.mem enables the host to access PTA‑attached memory (e.g., the weight buffer, or even the photonic weight control registers) as if it were system memory. This is particularly useful for:

- **Weight updates**: Instead of sending a `PTA_LOAD_WEIGHTS` command with a DMA descriptor, the host can simply `memcpy` the new weights to a memory region that is mapped to the PTA’s weight buffer. The PTA’s digital controller can monitor a flag or be triggered by a write to a doorbell register, and then program the photonic mesh from that buffer. The write itself is coherent, so no cache flush is required.
- **Result retrieval**: The PTA can write its ADC outputs into a coherent memory region that is also visible to the host (either PTA‑attached memory or host memory via CXL.cache). The host can then read the results with ordinary loads, without a DMA completion interrupt.
- **Control registers**: The command interface can be exposed as memory‑mapped registers in the CXL.mem space. The host writes commands directly, and the PTA observes them (perhaps via a polling loop or an interrupt).

This can dramatically simplify the software stack: the driver becomes a thin layer that just maps the device memory into user space, and the application reads/writes data structures using standard pointers.

---

### 5. Redesigned PTA Internal Memory Hierarchy

Given coherent access, the PTA’s internal memory can be reorganized as follows:

- **Local SRAM/DRAM** exposed via CXL.mem: This could hold weight matrices, input queues, output queues, and command/status structures. The host and PTA share this memory coherently, so both can read/write without explicit DMAs.
- **Small staging FIFOs** for DAC/ADC interfaces: These are necessary because the photonic core operates on a different timing domain (analog) than the digital coherent memory. But they can be much smaller than before (e.g., a few cache lines deep) because the coherent LSU can supply data on demand.
- **Optional cache** for recently accessed host memory: If the PTA frequently reuses certain data, a small cache (e.g., 32–64 KB) could reduce CXL latency. But this is optional; the PTA can operate without a cache, using only the host’s memory system.
- **Weight store** in two parts:
  1. **Digital weight buffer (SRAM)**: Holds the digital values that will be programmed into the photonic mesh. This is exposed to the host via CXL.mem for direct updates.
  2. **Analog weight bank (photonic)**: The actual phase shifters or microrings. This is not directly accessible by the host; it is programmed by the PTA’s control logic from the digital buffer.

The digital weight buffer acts as a shadow copy for fast re‑programming and calibration.

---

### 6. Control Flow with CXL Coherence

A typical operation might look like this:

1. **Host** allocates a shared memory region (e.g., using `posix_memalign` or a custom allocator) for weights, input activations, and output results. These regions are coherently mapped to both host and PTA.
2. **Host** writes the weight matrix into the weight region (a simple `memcpy`). It then writes a command descriptor into a command queue in shared memory, or directly to a doorbell register.
3. **Host** signals the PTA by writing to a doorbell register (or the PTA polls the queue). This write is a coherent store.
4. **PTA** fetches the command from the shared queue (using CXL.cache reads).
5. **PTA** programs the photonic mesh from the weight region (using coherent reads or from its local copy).
6. **PTA** streams input activations from the shared input region (using CXL.cache reads) and applies them to DACs.
7. **PTA** performs the photonic MVM.
8. **PTA** writes the results into the shared output region (using CXL.cache writes or to local memory exposed via CXL.mem).
9. **PTA** updates a completion flag or triggers an interrupt.
10. **Host** reads the output results (ordinary loads) and continues.

No explicit DMA engine, no cache flushes, no kernel‑mode driver for data movement. The entire operation can be done from user space with memory‑mapped I/O.

---

### 7. Trade‑offs and Challenges

- **Coherence overhead**: Every CXL.cache transaction involves snooping the host’s caches. For bulk streaming data, this may add latency and consume coherence bandwidth. However, CXL allows the use of **non‑coherent or write‑through** attributes for data that doesn’t need strict coherence (e.g., streaming inputs). The PTA’s LSU can be programmed to use different memory types for different regions.
- **Cache line granularity**: The PTA must operate on cache line (64‑byte) granularity for coherent accesses. For vector data that is not aligned or sized to cache lines, there may be inefficiencies. The host should allocate buffers with cache‑line alignment.
- **Latency hiding**: The photonic core is extremely fast; memory latency can dominate. The PTA should prefetch data (e.g., using a small hardware prefetcher or by issuing multiple outstanding CXL.cache requests) to keep the DACs busy.
- **Address translation**: The PTA must perform virtual‑to‑physical address translation for host memory. It will need a TLB or use the host’s IOMMU (ATS/PRI) to share page tables. This adds complexity.
- **Security**: Coherent sharing means the PTA can access any host memory mapped to it. Proper IOMMU setup is required to restrict access.
- **Photonic weight programming**: While the digital weight buffer is coherent, the actual programming of the analog weights is still slow and cannot be done directly via CXL. The PTA’s control logic must manage the transfer from digital buffer to photonic mesh, possibly using calibration.

---

### 8. Potential Design Modifications

To fully exploit CXL, the PTA should include:

- **A CXL.cache requester/LSU** that can generate coherent read/write transactions to host memory. This replaces the DMA read/write channels.
- **A CXL.mem slave** that maps the PTA’s local SRAM (weight buffer, command queue, status) into the host’s address space, enabling direct host access.
- **An IOMMU/TLB** for address translation and protection.
- **A small command processor** (could be the RISC‑V core we discussed earlier) that interprets commands and coordinates the coherent memory accesses with the photonic pipeline.

The DMA engine may still exist for bulk transfers, but it becomes optional. For large activations, a DMA‑like burst might be more efficient than cache‑line reads, but CXL.cache can also issue large bursts with appropriate prefetching.

---

### 9. Research Opportunities

This opens up several interesting research questions:

- How to optimally manage coherence for analog accelerators with non‑uniform access patterns?
- Can we use CXL.cache with custom coherence states (e.g., mark activation data as “read‑once” to avoid cache allocation)?
- How to hide memory latency behind photonic computation using prefetching and double‑buffering at the cache line level?
- What is the ideal balance between coherent access and traditional DMA for different layer sizes and batch dimensions?
- Can we extend CXL to support non‑volatile or analog memory (photonic weights) directly?

---

### 10. Summary

CXL transforms the PTA from a DMA‑driven, asynchronous accelerator into a **coherent memory‑mapped device**. This simplifies software and reduces data movement overhead, which is especially beneficial for the photonic core’s high speed. The internal memory hierarchy shifts from large staging buffers to smaller FIFOs and coherent shared memory regions. Implementing this requires a CXL.cache requester and CXL.mem slave, along with address translation, but the payoff is a more tightly integrated and efficient system.

Would you like to explore the design of the CXL.cache requester/LSU in more detail, or discuss how to handle address translation and IOMMU integration?

---

## Design of the CXL.cache Requester / Load‑Store Unit (LSU) for the PTA

The CXL.cache requester is the hardware block that allows the Photonic Tensor Accelerator (PTA) to issue coherent reads and writes to host memory over the CXL link. It replaces the traditional DMA engine for data movement, enabling the PTA to directly access activation and weight data in host memory without explicit copies, and to write results back coherently.

This component is central to leveraging CXL’s benefits: low‑latency, cache‑coherent access to shared memory. We will design it to meet the PTA’s streaming and latency requirements while managing the complexities of the CXL.cache protocol.

---

### 1. Role of the CXL.cache Requester

In CXL, a device that needs to access host memory coherently acts as a **Requester**. It sends requests (Read, Write, etc.) to the host’s **Home Agent** (usually integrated into the CPU or a CXL switch). The host maintains coherence by snooping its caches and returning the most up‑to‑date data.

The PTA’s requester must:

- Generate read requests for input activations and weights.
- Generate write requests for output results.
- Handle responses, including data and completion/error status.
- Respect memory ordering and coherence requirements (e.g., write‑back, write‑through).
- Interface with the PTA’s internal data path that feeds the DACs or consumes the ADCs.

For our PTA, the requester acts as a **Load‑Store Unit (LSU)** because it performs load and store operations on behalf of the PTA’s compute pipeline, similar to how a CPU core’s LSU works but optimized for streaming.

---

### 2. Integration in the PTA Architecture

The LSU sits between the PTA’s internal buffers/control logic and the CXL link interface.

```plaintext
+-----------------------+
|   PTA Control Unit    |
| (RISC-V or FSM)       |
+-----------+-----------+
            |
            v
+-----------+-----------+
|   LSU (CXL.cache      |
|   Requester)          |
| - Address generation  |
| - Request queue       |
| - Response buffer     |
| - Coherence handling  |
+-----------+-----------+
            |
            v
+-----------+-----------+
| CXL Link Layer &      |
| Physical Layer        |
+-----------------------+
```

The PTA’s compute pipeline (DACs, photonic core, ADCs) interacts with the LSU via internal FIFOs. For example, the input activation data stream is read from memory by the LSU and pushed into a small FIFO that feeds the DACs. Conversely, ADC outputs are collected and the LSU writes them back to memory.

---

### 3. Functional Blocks of the LSU

#### 3.1 Address Generation Unit (AGU)

The AGU is responsible for producing the sequence of memory addresses for a given data transfer. The PTA’s control unit programs the AGU with:

- Base virtual address
- Size (number of bytes or elements)
- Stride (for non‑contiguous access, e.g., sub‑matrices)
- Access pattern (contiguous, strided, or custom)
- Cache hints (e.g., streaming, non‑temporal)

For typical PTA operations:

- **Input activations**: contiguous vector or matrix (row‑major). AGU generates sequential addresses in cache‑line‑aligned chunks.
- **Weights**: similar, but possibly with larger granularity (e.g., whole rows).
- **Outputs**: contiguous writes.

The AGU must handle **address translation** (virtual to physical). This is done by an **IOMMU** (or SMMU) that the LSU queries via the Address Translation Service (ATS) or by using a local TLB. The translated physical addresses are placed in the CXL request packets.

#### 3.2 Request Queue and Scheduler

To maximize throughput, the LSU supports multiple outstanding requests. A **request queue** holds pending transactions with their metadata (address, length, type, status). The scheduler selects requests to send to the link, respecting:

- Ordering rules (e.g., reads may be reordered, writes need to maintain order unless relaxed).
- Credit‑based flow control from the CXL link.
- Priority (e.g., prefetch requests lower priority than demand requests).

The queue depth determines how much latency can be hidden. For a photonic accelerator, we might target 16–32 outstanding read requests, each up to 64 bytes (one cache line), to keep the DAC pipeline fed.

#### 3.3 Response Buffer and Data Path

When a read response returns from the host, it contains the requested data (cache line(s)) and status. The LSU stores the data in a **response buffer** (small SRAM or FIFO) and then forwards it to the internal consumer (e.g., DAC interface). The response buffer also handles reordering: responses may arrive out of order, so tags are used to match them to the original requests.

For writes, the LSU sends the data along with the request and may not need a response (posted writes) unless the PTA requires acknowledgment (e.g., for error handling or to know when data is visible). CXL.cache supports both posted and non‑posted writes.

#### 3.4 Coherence and Ordering Logic

The LSU must implement the CXL.cache protocol’s coherence state machine. This involves handling:

- Snoop responses from the host (for writes, the host may invalidate or update its caches).
- Memory attributes: cacheable vs. non‑cacheable, write‑back vs. write‑through.
- Atomic operations (if needed).
- Error reporting (e.g., poisoned data, timeouts).

For simplicity, the initial implementation can assume:

- The host memory regions accessed by the PTA are mapped as **cacheable, write‑back** for normal data.
- The PTA uses non‑temporal hints for streaming data to avoid cache pollution.
- The PTA does not perform atomic operations; synchronization is done via doorbells (which are just stores).

#### 3.5 Interface to Compute Pipeline

The LSU connects to the DAC input path and ADC output path via FIFOs:

- **Read path**: LSU fills an **input FIFO**. The DAC controller pops from this FIFO as needed. The FIFO depth is small (e.g., 4–8 cache lines) to smooth jitter.
- **Write path**: ADC results are pushed into an **output FIFO**. The LSU drains this FIFO and issues write requests. A small FIFO (e.g., 4–8 lines) decouples the ADC rate from the memory write rate.

The control unit orchestrates: it configures the AGU for the current operation, starts the LSU, and monitors completion.

---

### 4. Address Translation and IOMMU Integration

The PTA operates on **virtual addresses** provided by the host (user or kernel space). To access physical memory, the LSU must translate these addresses.

Options:

1. **Local TLB with page table walk**: The PTA includes a small TLB and can walk the host page tables (via [CXL.io](https://cxl.io/) or memory reads). This requires the PTA to have access to the page table structures and is complex.
2. **ATS (Address Translation Services)**: The PTA sends a translation request to the host’s IOMMU (or SMMU). The IOMMU returns the physical address and permissions. The LSU caches the translation in a local TLB for subsequent accesses. This is the standard approach for PCIe/CXL devices.

We will adopt ATS. The LSU has a **Translation Request Unit** that issues ATS requests (over [CXL.io](https://cxl.io/) or CXL.cache control path) and populates a small TLB. The TLB entries include virtual page number, physical page number, permissions, and caching attributes.

The AGU first checks the TLB for a translation; if miss, it stalls and requests a translation. The TLB can be managed by hardware or software (via the embedded RISC‑V core).

---

### 5. Performance Optimizations

#### 5.1 Prefetching

The LSU can implement a simple **sequential prefetcher**: when the AGU issues a read for address X, it also issues speculative reads for X+64, X+128, etc., up to a prefetch depth. This hides memory latency and keeps the input FIFO full. The prefetcher should be throttled when the FIFO is nearly full to avoid wasting bandwidth.

#### 5.2 Multiple Outstanding Requests

The request queue depth directly affects throughput. With 16–32 outstanding cache line reads, the LSU can saturate the CXL link. The CXL link itself has credit mechanisms; the LSU must track available credits to avoid stalling.

#### 5.3 Data Alignment and Bursts

For efficiency, the LSU should issue **full cache line (64‑byte) requests** whenever possible. If the requested data is not aligned or smaller than a cache line, the LSU may need to perform read‑modify‑write or use byte enables. To simplify, the host driver should allocate buffers aligned to 64 bytes and sizes multiples of 64 bytes, which is common practice.

#### 5.4 Cache Hints

CXL allows the requester to specify **cache allocation hints** (e.g., non‑temporal, streaming). For streaming activations and results, we can mark them as non‑temporal to avoid polluting the host’s last‑level cache. For weights that are reused, normal caching may be beneficial. The LSU should support configurable hints per transfer.

---

### 6. Error Handling and Reliability

The LSU must handle:

- **Timeouts**: If a request does not receive a response within a programmable time, the LSU signals an error to the control unit.
- **Data errors**: ECC or parity errors in the CXL link; the LSU may retry or report.
- **Address translation faults**: If the IOMMU returns a fault, the LSU aborts the current operation and raises an interrupt.
- **Protocol violations**: Unexpected responses are flagged.

Errors are logged in a status register accessible via [CXL.io](https://cxl.io/), and the embedded RISC‑V core can read them for diagnostics.

---

### 7. Implementation Complexity and FPGA Prototyping

A full CXL.cache requester is a significant RTL effort. For FPGA prototyping, we can take an incremental approach:

1. **Simulation‑only model**: Use a CXL verification IP (e.g., from Cadence, Synopsys) to test the LSU logic against a model of a host.
2. **Use existing CXL IP**: Some FPGA vendors (Intel, Xilinx) provide CXL IP cores for their devices, often limited to Type‑3 (memory) or basic Type‑1/2 support. For example, Intel Agilex FPGAs with CXL hard IP can be used to implement a Type‑2 device. This dramatically reduces the effort.
3. **Emulate coherent behavior in FPGA**: If a real CXL link is not available, you can emulate the host‑side coherence by implementing a simple home agent in another part of the FPGA, or by using a shared memory model between the RISC‑V soft‑core and the PTA. This allows development of the LSU logic without full CXL.

Given the goal of a PhD, using an existing CXL IP block (if available) is the most practical path. The research contribution would be in the integration and optimization for the photonic accelerator, not in building the CXL protocol from scratch.

---

### 8. Interaction with the Embedded RISC‑V Controller

The embedded RISC‑V core can offload some LSU management tasks:

- Configuring the AGU with complex access patterns.
- Managing the TLB (software‑managed if desired).
- Handling errors and retries.
- Performing address translation setup and IOMMU programming.
- Running calibration routines that may require accessing host memory.

This reduces the complexity of the LSU state machines, as the RISC‑V can handle infrequent or complex tasks while the LSU handles high‑speed data streaming.

---

### 9. Summary

The CXL.cache requester/LSU replaces the DMA engine, providing coherent load/store access to host memory. Its key components are an address generation unit, request queue, response buffer, and coherence logic. Address translation is handled via ATS and a local TLB. Performance is optimized through prefetching, multiple outstanding requests, and cache hints. While complex, using vendor‑provided CXL IP can make this feasible for an FPGA prototype.

This design enables the PTA to operate with a simplified software model: the host writes data to shared memory, and the PTA accesses it directly, leveraging coherence for fine‑grained interaction. The next step would be to define the specific register interface for the LSU and how the control unit programs it for each PTA operation.

Would you like to detail the LSU programming model (registers and commands), or move on to another aspect of the system?

---

## LSU Programming Model for the CXL.cache Requester

The Load‑Store Unit (LSU) is programmed by the PTA’s digital control unit (either a state machine or the embedded RISC‑V core) to perform coherent data transfers over CXL.cache. Its programming model is defined by a set of memory‑mapped registers that specify the transfer parameters, control flags, and status reporting. The LSU executes transfers independently once started, freeing the control unit to manage other tasks.

---

### 1. Register Interface

The LSU exposes a set of registers in the PTA’s local address space. These registers can be written by the PTA’s control logic or, if the PTA is CXL.mem‑capable, directly by the host (though typically the host interacts via higher‑level commands). The register map is grouped by function:

<table>
  <tr><th>Offset</th><th>Name</th><th>Width</th><th>Access</th><th>Description</th></tr>
  <tr><td>0x00</td><td>LSU_CTRL</td><td>32</td><td>RW</td><td>Control and start/abort commands.</td></tr>
  <tr><td>0x04</td><td>LSU_STATUS</td><td>32</td><td>RO</td><td>Status and error flags.</td></tr>
  <tr><td>0x08</td><td>LSU_INT_EN</td><td>32</td><td>RW</td><td>Interrupt enable mask.</td></tr>
  <tr><td>0x0C</td><td>LSU_INT_STATUS</td><td>32</td><td>W1C</td><td>Interrupt status (write‑1‑to‑clear).</td></tr>
  <tr><td>0x10</td><td>LSU_BASE_ADDR_LO</td><td>32</td><td>RW</td><td>Lower 32 bits of virtual base address.</td></tr>
  <tr><td>0x14</td><td>LSU_BASE_ADDR_HI</td><td>32</td><td>RW</td><td>Upper 32 bits of virtual base address.</td></tr>
  <tr><td>0x18</td><td>LSU_LENGTH</td><td>32</td><td>RW</td><td>Transfer length in bytes.</td></tr>
  <tr><td>0x1C</td><td>LSU_STRIDE</td><td>32</td><td>RW</td><td>Stride between successive elements (if strided).</td></tr>
  <tr><td>0x20</td><td>LSU_FIFO_SEL</td><td>8</td><td>RW</td><td>Selects internal FIFO: 0 = input, 1 = output, 2 = weight, etc.</td></tr>
  <tr><td>0x24</td><td>LSU_CACHE_HINTS</td><td>8</td><td>RW</td><td>Cache allocation hints (non‑temporal, streaming, etc.).</td></tr>
  <tr><td>0x28</td><td>LSU_TRANSFER_ID</td><td>16</td><td>RW</td><td>Tag for command tracking.</td></tr>
  <tr><td>0x2C</td><td>LSU_DONE_COUNT</td><td>32</td><td>RO</td><td>Number of bytes transferred so far (for progress).</td></tr>
  <tr><td>0x30</td><td>LSU_ERROR_INFO</td><td>32</td><td>RO</td><td>Detailed error information on fault.</td></tr>
</table>

All registers are accessible via the PTA’s internal bus (e.g., AXI‑Lite). Multiple LSU instances (channels) may exist; in that case, the register set is replicated with a base offset per channel.

---

### 2. Control Register Fields

`LSU_CTRL` bits:

<table>
  <tr><th>Bit(s)</th><th>Field</th><th>Description</th></tr>
  <tr><td>0</td><td>START</td><td>Write 1 to initiate the transfer. Cleared by hardware when done.</td></tr>
  <tr><td>1</td><td>ABORT</td><td>Write 1 to abort current transfer.</td></tr>
  <tr><td>2</td><td>DIR</td><td>Direction: 0 = read from memory to FIFO, 1 = write from FIFO to memory.</td></tr>
  <tr><td>3</td><td>STRIDED</td><td>Enable stride mode (otherwise contiguous).</td></tr>
  <tr><td>4</td><td>PREFETCH</td><td>Enable automatic sequential prefetching.</td></tr>
  <tr><td>5</td><td>FLUSH_FIFO</td><td>Before starting, flush the selected FIFO.</td></tr>
  <tr><td>6</td><td>DONE_INT_EN</td><td>Generate interrupt upon successful completion.</td></tr>
  <tr><td>7</td><td>ERR_INT_EN</td><td>Generate interrupt upon error.</td></tr>
  <tr><td>8‑15</td><td>RESERVED</td><td></td></tr>
  <tr><td>16‑23</td><td>FIFO_THRESH</td><td>Watermark threshold for flow control (used with PREFETCH).</td></tr>
  <tr><td>24‑31</td><td>CHANNEL_ID</td><td>Identifies this LSU channel for status reporting.</td></tr>
</table>

`LSU_STATUS` bits:

<table>
  <tr><th>Bit(s)</th><th>Field</th><th>Description</th></tr>
  <tr><td>0</td><td>BUSY</td><td>1 while a transfer is in progress.</td></tr>
  <tr><td>1</td><td>DONE</td><td>1 after successful completion. Cleared on new START.</td></tr>
  <tr><td>2</td><td>ERROR</td><td>1 if an error occurred. Error details in LSU_ERROR_INFO.</td></tr>
  <tr><td>3</td><td>FIFO_OVERFLOW</td><td>Input FIFO overflow (read direction).</td></tr>
  <tr><td>4</td><td>FIFO_UNDERFLOW</td><td>Output FIFO underflow (write direction).</td></tr>
  <tr><td>5</td><td>TLB_MISS</td><td>A translation was missing and ATS request failed.</td></tr>
  <tr><td>6</td><td>LINK_ERROR</td><td>CXL link protocol error.</td></tr>
  <tr><td>7</td><td>TIMEOUT</td><td>A request timed out.</td></tr>
  <tr><td>8‑31</td><td>RESERVED</td><td></td></tr>
</table>

---

### 3. Programming a Transfer

The control unit performs the following steps to set up and start a data movement:

1. **Write **`LSU_BASE_ADDR` – set the virtual start address.
2. **Write **`LSU_LENGTH` – total bytes to move.
3. **If strided**: write `LSU_STRIDE` and set `STRIDED` bit in `LSU_CTRL`.
4. **Select FIFO** – write `LSU_FIFO_SEL`.
5. **Set direction** – `DIR` bit in `LSU_CTRL`.
6. **Set cache hints** – e.g., non‑temporal for streaming.
7. **Optionally enable prefetch** and set FIFO watermark.
8. **Set interrupt enables** if desired.
9. **Write **`LSU_TRANSFER_ID` – a tag for the control unit to identify completion.
10. **Set `START` bit** – the LSU begins issuing requests.

The LSU fetches data from memory (or writes data from the FIFO) until the length is exhausted. On completion, it sets the `DONE` bit and optionally raises an interrupt. The control unit can poll `LSU_STATUS` or wait for the interrupt.

---

### 4. Interaction with the PTA Control Flow

The PTA’s command decoder (or embedded RISC‑V) translates high‑level PTA instructions into a sequence of LSU operations.

**Example: `PTA_MVM` (Matrix‑Vector Multiply)**

- Input activations: The control unit configures an LSU channel to read from the activation buffer address in host memory into the input FIFO (direction = read). The length equals `N * batch * element_size`. The FIFO feeds the DACs.
- Output results: Another LSU channel (or the same, after input is done) is configured to write from the output FIFO to the result buffer address (direction = write). The length is `M * batch * element_size`. The ADC results are pushed into the output FIFO.
- The two transfers can run concurrently using double‑buffering: while the input LSU is filling one buffer, the photonic core consumes another; similarly for output.

The PTA control unit manages the sequencing: start input read, wait for sufficient data, trigger photonic computation, then start output write as data becomes available. This is exactly the double‑buffering scheme described earlier, but now memory accesses are coherent via CXL.cache.

---

### 5. Address Translation and TLB Management

The LSU uses ATS to translate virtual addresses. Before starting a transfer, the control unit may ensure the required translations are present in the TLB by issuing prefetch translation requests. The LSU itself handles TLB misses transparently: when the AGU encounters a virtual page not in the TLB, it stalls and sends an ATS request. Once the translation returns, the TLB is filled and the transfer resumes.

The TLB can be managed by hardware (automatic fill) or software (the RISC‑V can pre‑load translations via a dedicated `LSU_TLB_INSERT` register). Hardware‑managed is simpler but requires the ATS interface. Software‑managed may be used for performance tuning.

If a translation fault occurs (e.g., page not present, permission denied), the LSU aborts, sets `ERROR` and `TLB_MISS`, and records the faulting address in `LSU_ERROR_INFO`. The control unit or driver can then handle the fault.

---

### 6. Error Handling and Recovery

The LSU reports errors via the status register and optional interrupt. The control unit (or host driver) can take the following actions:

- **TLB fault**: The driver may map the missing page and restart the transfer (by re‑issuing START). The LSU does not automatically retry.
- **Link error**: The CXL link may have suffered a transient error; the driver can reset the link and retry.
- **FIFO overflow/underflow**: Indicates a rate mismatch between the LSU and the photonic pipeline. This may require adjusting FIFO sizes, throttling the photonic core, or changing DMA watermark settings.
- **Timeout**: The transaction did not complete within the expected time; possibly due to a deadlock or host failure.

The `LSU_ERROR_INFO` register contains the transfer ID, the memory address where the error occurred, and a specific error code to aid debugging.

---

### 7. Performance Tuning Registers

To optimize for different workloads, additional registers can be provided:

- `LSU_MAX_OUTSTANDING`: Limits the number of outstanding requests to avoid overwhelming the CXL link or host.
- `LSU_PREFETCH_DIST`: Controls how far ahead the prefetcher issues requests.
- `LSU_CACHE_POLICY`: Overrides default cache allocation (e.g., write‑back, write‑through, non‑cacheable).
- `LSU_BURST_LENGTH`: For non‑cache‑line‑aligned transfers, sets the maximum burst size (e.g., 64B, 128B).

These are typically set once during initialization, but could be adjusted per transfer for fine‑grained control.

---

### 8. Example Programming Sequence (Pseudocode)

```plaintext
// PTA control unit prepares to read input activations
void lsu_start_read(uint64_t src_addr, uint32_t len, uint8_t fifo_id) {
    write_reg(LSU_BASE_ADDR_LO, (uint32_t)src_addr);
    write_reg(LSU_BASE_ADDR_HI, (uint32_t)(src_addr >> 32));
    write_reg(LSU_LENGTH, len);
    write_reg(LSU_FIFO_SEL, fifo_id);
    uint32_t ctrl = (1 << 2) /* DIR=0 for read */ |
                    (1 << 4) /* PREFETCH enable */ |
                    (1 << 6) /* DONE_INT_EN */;
    write_reg(LSU_CTRL, ctrl | (1 << 0)); // set START
}

// For writing results
void lsu_start_write(uint64_t dst_addr, uint32_t len, uint8_t fifo_id) {
    write_reg(LSU_BASE_ADDR_LO, (uint32_t)dst_addr);
    write_reg(LSU_BASE_ADDR_HI, (uint32_t)(dst_addr >> 32));
    write_reg(LSU_LENGTH, len);
    write_reg(LSU_FIFO_SEL, fifo_id);
    uint32_t ctrl = (1 << 3) /* DIR=1 for write */ |
                    (1 << 6); /* DONE_INT_EN */
    write_reg(LSU_CTRL, ctrl | (1 << 0));
}
```

The actual PTA command handler would call these functions (or directly program registers) as part of executing `PTA_MVM`.

---

### 9. Summary

The LSU programming model provides a flexible, low‑level interface for coherent data movement. It abstracts the CXL.cache protocol complexities, allowing the PTA control unit to focus on orchestrating the photonic pipeline. The register‑based interface is suitable for both hardware FSM and embedded RISC‑V control, enabling a wide range of optimizations for streaming and prefetching. This design positions the PTA to fully exploit CXL’s coherence for low‑latency, fine‑grained data sharing with the host.

Would you like to explore the photonic core’s analog control interface next, or return to the overall system integration?

---

## Photonic Core Analog Control Interface

The photonic core performs matrix‑vector multiplication in the analog optical domain, but it must be precisely controlled by the digital subsystem. This interface is responsible for programming the weights, driving the input modulators, reading the output detectors, and maintaining accuracy through calibration. It is a mixed‑signal boundary where digital precision meets analog physics.

### 1. Components Requiring Analog Control

A typical photonic tensor core (based on Mach‑Zehnder interferometer meshes or microring resonators) consists of:

- **Input modulators**: Encode electrical activations onto optical signals (amplitude or phase).
- **Weight elements**: Tunable phase shifters or amplitude modulators that implement the matrix coefficients.
- **Photodetectors**: Convert optical output to electrical current.
- **Bias and monitoring structures**: Ensure operating points and provide feedback for calibration.

Each of these requires dedicated analog control signals.

#### 1.1 Weight Elements

- **Thermo‑optic phase shifters (heaters)**: Most common in silicon photonics. A small resistive heater changes the refractive index of a waveguide, altering the phase. Control is via a continuous voltage or current.
  - **Speed**: kHz to tens of kHz (thermal time constant).
  - **Power**: Milliwatts per element.
  - **Precision required**: To achieve 6‑8 effective bits of weight precision, the phase must be set with sub‑milliradian accuracy. This typically demands a 12‑bit DAC or better.
- **Carrier‑injection/electro‑optic modulators**: Faster (GHz) but less stable, lossy, and often require bias control. Less common for weight storage due to drift.

#### 1.2 Input Modulators

- **Mach‑Zehnder modulators (MZM)**: Use an interferometer to convert voltage to amplitude. They require a push‑pull drive and often a bias voltage to set the operating point.
  - **Speed**: Up to tens of GHz.
  - **Precision**: Activations may be quantized to 4‑8 bits; an 8‑bit DAC is sufficient.
  - **Linearity**: MZMs have a sinusoidal transfer function; pre‑distortion or operation in the linear region is needed.
- **Microring modulators**: Compact, but temperature‑sensitive and need wavelength locking.

#### 1.3 Photodetectors and Readout

- Photodetectors convert light intensity to photocurrent. They are inherently analog but are followed by transimpedance amplifiers (TIAs) and ADCs to produce digital outputs. The analog control interface includes:
  - TIA gain settings (often fixed, but sometimes programmable).
  - ADC reference voltages or ranges.
  - Optional bias for the photodetector.

### 2. Control Signal Types and Requirements

<table>
  <tr><th>Function</th><th>Signal Type</th><th>Resolution</th><th>Speed</th><th>Count (example)</th></tr>
  <tr><td>Weight programming</td><td>Voltage/current</td><td>12‑16 bits</td><td>Slow (kHz)</td><td>M×N (e.g., 64×64 = 4096)</td></tr>
  <tr><td>Input modulation</td><td>Voltage</td><td>8‑10 bits</td><td>Fast (GHz)</td><td>N (e.g., 64)</td></tr>
  <tr><td>TIA gain/bias</td><td>Voltage/current</td><td>8‑12 bits</td><td>Slow</td><td>M (e.g., 64)</td></tr>
  <tr><td>ADC reference</td><td>Voltage</td><td>8‑12 bits</td><td>Slow</td><td>1‑4 (shared)</td></tr>
  <tr><td>Monitoring (power, temp)</td><td>Voltage/current</td><td>12‑16 bits</td><td>Slow</td><td>Several</td></tr>
</table>

The large number of weight elements demands a scalable control scheme: often a **row/column addressing** with sample‑and‑hold or a **serial DAC chain** to reduce pin count.

### 3. DAC/ADC Selection and Integration

#### 3.1 Weight DACs

- **Resolution**: High (12‑16 bits) because the phase‑to‑weight mapping is nonlinear and sensitive. Thermal phase shifters exhibit a quadratic relationship between voltage and phase; therefore, the DAC must provide fine voltage steps near the operating point.
- **Speed**: Low; programming a weight matrix can take microseconds to milliseconds.
- **Architecture**: For large matrices, one DAC per weight is impractical. Options:
  - **Column‑parallel with sample‑and‑hold**: A single high‑speed DAC per column, with per‑row sample‑and‑hold circuits. The digital controller scans rows, loading each weight value.
  - **Serial shift register with per‑element DAC**: Each weight element has a tiny local DAC (often a charge‑redistribution or current‑steering DAC) driven by a digital shift register.
  - **Current‑mode DACs with global reference**: Current outputs are summed or steered.

#### 3.2 Activation DACs

- **Resolution**: 8 bits typical; speed is critical. These DACs must run at the data rate of the photonic engine (potentially multiple GS/s).
- **Architecture**: High‑speed current‑steering DACs, often integrated on the same electrical chip as the digital controller (or as a separate chiplet in 2.5D integration).
- **Number**: Equal to the number of input channels (e.g., 64 for a 64×64 MVM). If wavelength‑division multiplexing is used, each wavelength may have its own modulator and DAC.

#### 3.3 Monitoring ADCs

- Used to measure photodetector outputs during calibration or to read temperature sensors.
- **Resolution**: 12‑16 bits, speed modest (kS/s to MS/s).
- **Sharing**: A multiplexer can select among many monitor points to reduce ADC count.

### 4. Programming Sequence for Weight Loading

The `PTA_LOAD_WEIGHTS` command triggers the following steps, orchestrated by the digital control unit:

1. **Digital weight data** is available in the weight buffer (either via coherent CXL.cache reads or from local SRAM).
2. **Address generation**: The control unit iterates over all weight elements. For each element:
  - Compute the required DAC code from the digital weight value using a lookup table or calibration polynomial.
  - Write the code to the corresponding weight DAC register (or shift register).
3. **Settling delay**: After all DACs are written, the control unit waits for the thermal phase shifters to settle. This delay is programmable (e.g., 10 µs to 1 ms).
4. **Optional verification**: The control unit may read back monitor photodiodes to confirm the phase setting, adjusting if necessary.

Because weight updates are infrequent (relative to inference), the slow speed is acceptable. The digital controller can overlap weight programming with other operations if multiple weight sets are cached in photonic meshes (e.g., using multiple photonic cores).

### 5. Closed‑Loop Calibration and Drift Compensation

Photonic weights drift due to temperature changes, aging, and optical crosstalk. A calibration loop is essential.

#### 5.1 Calibration Hardware

- **Monitor photodetectors**: Placed at strategic points (e.g., after each MZI stage) to measure optical power or phase.
- **Reference signals**: Known test vectors stored in ROM or generated by the digital controller.
- **Calibration DACs**: Same as weight DACs, but maybe with finer resolution for adjustment.
- **Feedback algorithm**: Runs on the embedded RISC‑V core or a dedicated FSM.

#### 5.2 Calibration Routine (PTA_CALIBRATE)

1. **Inject known inputs**: The digital controller applies a set of orthogonal test vectors to the activation DACs.
2. **Measure outputs**: The photodetector outputs are digitized by ADCs and stored in memory.
3. **Compute error**: The RISC‑V core compares measured outputs against expected values (computed from the ideal weight matrix).
4. **Update weight DAC codes**: Using an iterative algorithm (e.g., gradient descent or least‑squares), the core adjusts the weight DAC settings to minimize error.
5. **Store calibration coefficients**: The final DAC codes are saved in non‑volatile memory (if available) or in a local register file for quick reprogramming.

This closed‑loop calibration can be run periodically (e.g., every few milliseconds) or triggered by temperature sensors. The digital control interface must support fast access to the weight DACs and monitoring ADCs for this loop.

### 6. Interface to the Digital Control Unit

The analog control interface is ultimately a set of memory‑mapped registers and perhaps dedicated control signals. In our CXL‑coherent PTA, these registers are local to the PTA and not exposed to the host via CXL.mem unless necessary (the host would not directly manipulate DACs; it only provides digital weights). The digital control unit (FSM or RISC‑V) translates high‑level commands into low‑level register writes.

#### 6.1 Register Map for Analog Control

<table>
  <tr><th>Offset</th><th>Name</th><th>Description</th></tr>
  <tr><td>0x00</td><td>WEIGHT_DAC_BASE</td><td>Base address of weight DAC register array.</td></tr>
  <tr><td>0x04</td><td>ACT_DAC_BASE</td><td>Base address of activation DAC registers.</td></tr>
  <tr><td>0x08</td><td>TIA_CTRL</td><td>TIA gain and bias settings.</td></tr>
  <tr><td>0x0C</td><td>ADC_CTRL</td><td>ADC reference, sampling rate, channel select.</td></tr>
  <tr><td>0x10</td><td>MONITOR_ADC_BASE</td><td>Base address for monitoring ADC results.</td></tr>
  <tr><td>0x14</td><td>CALIB_CTRL</td><td>Calibration mode, test vector select.</td></tr>
  <tr><td>0x18</td><td>TEMP_SENSOR</td><td>Readout of on‑chip temperature sensors.</td></tr>
  <tr><td>0x1C</td><td>LASER_CTRL</td><td>Laser power and wavelength control.</td></tr>
</table>

The weight DACs may be mapped as a large memory region (e.g., 4K entries of 16 bits each), allowing the RISC‑V core to write them in a loop. Similarly, activation DACs are mapped as a smaller array (e.g., 64 entries).

#### 6.2 Low‑Level Drivers

The embedded RISC‑V runs small driver functions:

```plaintext
void write_weight_dac(int index, uint16_t code) {
    volatile uint16_t *dac = (volatile uint16_t *)(WEIGHT_DAC_BASE + index * 2);
    *dac = code;
}

void set_activation_dac(int index, uint8_t code) {
    volatile uint8_t *dac = (volatile uint8_t *)(ACT_DAC_BASE + index);
    *dac = code;
}
```

These functions are used by the calibration routine and the MVM sequencer.

### 7. Timing and Sequencing

The photonic core operates in a pipelined fashion:

- **Activation update**: Write new values to activation DACs; wait for DAC settling (nanoseconds).
- **Optical pulse**: Fire the laser (if pulsed) or enable the modulators for a fixed integration time.
- **Detection**: Photodetectors integrate light; TIAs amplify; ADCs convert (pipeline latency ~10‑50 ns).
- **Result readout**: ADC outputs are latched into the output buffer.

The digital controller manages these steps with precise timing, often using a programmable delay line or a state machine with configurable wait states. For FPGA emulation, these delays are modeled as cycle counts.

### 8. Modeling in FPGA Emulation

In the FPGA emulation phase, the analog control interface is replaced by digital models:

- **Weight DACs**: Registers that store quantized weight values, possibly with injected drift.
- **Activation DACs**: Simple quantizers (8‑bit) that feed the emulated photonic core.
- **Photonic core model**: A digital matrix multiplier using DSP blocks, with additive noise.
- **Monitoring ADCs**: Emulated by reading the output and adding noise, or by providing access to internal signals.

The digital control logic (RISC‑V or FSM) remains exactly the same, issuing register writes to the emulated DACs and reading emulated ADCs. This validates the entire control flow without analog hardware.

### 9. Path to Physical Implementation

When moving to a real photonic chip, the analog control interface becomes a physical design problem:

- **Chip integration**: If the photonic chip is separate from the digital chip, high‑speed DAC/ADC signals must cross the chip boundary. This may require serializer/deserializer (SerDes) or parallel LVDS interfaces.
- **PDK requirements**: Use a photonic PDK (e.g., AIM Photonics, imec) that includes models for heaters, modulators, and photodetectors, and co‑simulate with the digital RTL using mixed‑signal tools.
- **Power management**: The heaters and lasers require stable power supplies and thermal management; the digital controller must include power‑on sequencing and temperature monitoring.
- **Calibration hardware**: On‑chip monitor photodiodes and temperature sensors must be designed into the photonic layout and connected to the digital chip.

The digital architecture and programming model we defined remain unchanged; only the low‑level analog interface changes from emulated registers to actual DAC/ADC IP blocks.

### 10. Summary

The photonic core’s analog control interface is a critical mixed‑signal subsystem that translates digital weights and activations into precise optical signals and digitizes the results. It requires high‑resolution, slow DACs for weight programming; fast, moderate‑resolution DACs for activations; and monitoring ADCs for calibration. The digital control unit (often an embedded RISC‑V) manages this interface via memory‑mapped registers, implementing closed‑loop calibration to counteract drift. In an FPGA emulation, this interface is modeled digitally, enabling full validation of the control logic before committing to silicon photonics.

Would you like to explore the calibration algorithm in more depth, or discuss the design of the high‑speed activation DAC interface?

---

## High‑Speed Activation DAC Interface for the Photonic Tensor Accelerator

The activation DAC interface is responsible for converting digital activation values (typically 4–8 bits) into analog electrical signals that drive the optical modulators in the photonic core. This interface must operate at the full speed of the photonic engine, often multiple GS/s, while maintaining low power, low latency, and sufficient linearity. In the context of our CXL‑coherent PTA, this interface receives data from the coherent memory system (via the LSU) through a small FIFO and streams it to the modulators.

### 1. Requirements and Constraints

<table>
  <tr><th>Parameter</th><th>Typical Value</th><th>Notes</th></tr>
  <tr><td>Resolution</td><td>4–8 bits</td><td>Activations are usually quantized; 8 bits is a practical upper limit.</td></tr>
  <tr><td>Sampling rate</td><td>1–10 GS/s per channel</td><td>Must match the optical pulse rate; higher for fast photonic cores.</td></tr>
  <tr><td>Number of channels</td><td>N (e.g., 64–256)</td><td>Equal to the number of input nodes (or wavelengths).</td></tr>
  <tr><td>Latency</td><td>&lt; 10 ns (DAC settling)</td><td>Critical for pipeline efficiency.</td></tr>
  <tr><td>Power per DAC</td><td>~10–100 mW</td><td>Dominated by output driver and high‑speed logic.</td></tr>
  <tr><td>Interface to digital</td><td>Parallel LVDS or SerDes</td><td>If photonic chip is separate.</td></tr>
  <tr><td>Interface to modulator</td><td>Voltage swing ~1–2 Vpp</td><td>Depends on modulator type (MZM, microring).</td></tr>
</table>

The activation DACs are the **fastest analog components** in the PTA; they directly determine the compute throughput.

### 2. Data Flow and Buffering

The activation data originates from the shared coherent memory (host memory via CXL.cache) and is fetched by the LSU. Due to the high speed of the photonic core, it is impractical to feed each DAC directly from memory; a small FIFO per channel (or a shared buffer) is used to smooth latency and ensure continuous streaming.

```plaintext
CXL.cache (LSU) → Input FIFO (per channel or shared) → Activation DAC → Modulator
```

- **FIFO depth**: 4–16 samples per channel, enough to absorb memory latency jitter.
- **Flow control**: The DAC interface asserts a “ready” signal when it can accept more data; the LSU throttles prefetching based on FIFO occupancy.
- **Data alignment**: Samples are typically 8‑bit; if memory returns 64‑byte cache lines, the FIFO width can be 64 bits and demultiplexed into eight 8‑bit channels.

For batched MVM operations, the input data is streamed sequentially: each sample is applied to the DACs for one optical pulse (or a fixed integration time). The DAC output must settle before the optical pulse occurs.

### 3. DAC Architecture

High‑speed, moderate‑resolution DACs are typically implemented as **current‑steering** or **resistive ladder** structures.

- **Current‑steering DAC**: Binary‑weighted or segmented current sources switched by digital inputs, summed into a load resistor to produce voltage.
  - Advantages: high speed, small area, easy to drive off‑chip.
  - Disadvantages: requires careful matching for linearity.
  - For 8‑bit resolution, a segmented design (e.g., 4 binary + 4 thermometer) improves DNL/INL.
- **Resistive ladder (R‑2R)**: Simpler but slower; not common for GS/s.
- **Capacitive DAC**: Used in charge‑domain, but not for continuous high‑speed output.

In an integrated photonic‑electronic chip (or chiplet), the DAC output driver must provide sufficient voltage swing to the modulator. MZMs typically require a differential drive of ±1 V around a bias point. The DAC may include a **line driver** (e.g., a differential current‑mode logic driver) to achieve the required bandwidth.

For FPGA emulation, we do not implement real DACs; instead, the activation values are used directly in the digital photonic core model, possibly with quantization and noise added.

### 4. Modulator Interface and Pre‑distortion

The most common high‑speed modulator is the **Mach‑Zehnder Modulator (MZM)**. Its transfer function is nonlinear (sinusoidal):

Pout=Pincos⁡2(πV2Vπ+ϕb)Pout​=Pin​cos2(2Vπ​πV​+ϕb​)

where VπVπ​ is the half‑wave voltage and ϕbϕb​ is the bias phase. To achieve linear amplitude modulation, the modulator is biased at the quadrature point (ϕb=π/2ϕb​=π/2), where the slope is maximum and approximately linear for small signals.

- **Bias control**: A separate slow DAC sets the bias voltage to maintain quadrature despite temperature drift.
- **Pre‑distortion**: For large signals, the sinusoidal nonlinearity distorts the activation. The digital controller can apply a lookup table (LUT) to map the desired amplitude to the required voltage, effectively linearizing the modulator. This LUT can be stored in a small ROM and applied on the fly before the DAC.

Microring modulators have a Lorentzian response and require wavelength locking and thermal stabilization, making their drive electronics more complex; they are less common for high‑speed multi‑channel arrays unless integrated with control loops.

### 5. Clocking and Synchronization

The photonic core operates in a pulsed or gated mode. The activation DAC interface must be synchronized with the optical pulse timing.

- **Clock generation**: A high‑speed clock (e.g., 2.5 GHz) is derived from the system clock or a dedicated PLL. This clock times the DAC update and the laser pulse.
- **Phase alignment**: The DAC output must be stable before the optical pulse arrives. This requires a programmable delay between the digital data latch and the laser trigger.
- **Pulse width**: If the photonic computation is performed by integrating light over a short pulse (e.g., 100 ps), the DAC output need only be stable during that window, allowing time‑interleaving of channels.

In a real system, the optical pulse may be generated by a mode‑locked laser or a gated continuous‑wave laser, and the trigger is provided by the digital controller. The activation DACs are updated on the rising edge of the clock, and the laser fires after a fixed delay (e.g., 1–2 ns) to allow settling.

### 6. Channel Count and Integration

For a matrix‑vector multiplication with NN input nodes, we need NN activation DACs. For N=64N=64 or 128128, this is manageable on a dedicated electrical chip. To reduce pin count and power, **serialization** techniques can be employed:

- **Time‑interleaving**: Use a single high‑speed DAC per group of channels with sample‑and‑hold circuits. For example, one DAC at 20 GS/s can drive 4 channels at 5 GS/s each using a demultiplexer and track‑and‑hold.
- **Analog multiplexing**: Not common due to bandwidth limitations; better to keep parallel DACs if integrated on the same chip as the photonics.

If the photonic chip is separate from the digital chip (e.g., 2.5D integration), high‑speed **SerDes** links (e.g., 28 Gbps per lane) can carry activation data, with the DACs located on the photonic chip close to the modulators. This reduces the number of physical interconnects but adds complexity and power.

### 7. Interface to the Digital Control Unit

The activation DAC interface is controlled by the digital control unit (FSM or RISC‑V) via a memory‑mapped register block:

- **Data register**: For each channel, a register holds the next activation value. In streaming mode, the control unit may write a whole vector to the DAC registers before triggering the optical pulse.
- **Trigger register**: A write to a specific address causes all DACs to latch their new values simultaneously and start the pulse sequence.
- **Status register**: Indicates when the DACs are ready for the next sample (e.g., FIFO empty flag, settling complete).

In a high‑throughput design, the DACs are not memory‑mapped individually; instead, a dedicated datapath from the input FIFO writes directly to the DAC array using a wide parallel bus. The control unit only sets up the base address and length, and a hardware sequencer streams the data.

### 8. FPGA Emulation Model

In FPGA emulation, the activation DAC is modeled as:

- A register that stores the 8‑bit activation value for each channel.
- Optionally, a quantizer that adds noise or truncates to a lower precision to mimic analog effects.
- A delay element to simulate DAC settling time (e.g., 2–3 clock cycles).
- The emulated photonic core reads the stored values (after the delay) and performs the matrix multiplication.

The trigger mechanism is implemented as a control signal that, when asserted, causes the emulated core to sample the activation registers and compute the result. This allows validation of the timing and flow control without any analog hardware.

### 9. Path to Physical Implementation

When moving to a real photonic chip, the activation DAC interface becomes a mixed‑signal design requiring:

- **Custom layout** of high‑speed DACs, often in a separate CMOS die or integrated with the photonic process (e.g., 45nm SOI or BiCMOS).
- **Co‑simulation** with the photonic device models (e.g., Verilog‑A models for modulators) to ensure signal integrity and bandwidth.
- **Packaging** to minimize parasitics between the DACs and modulators.
- **Power distribution** to avoid supply noise coupling into the sensitive analog circuits.

The digital control logic and the DAC interface protocol remain unchanged; only the physical implementation and low‑level driver code need adaptation.

### 10. Summary

The high‑speed activation DAC interface is a critical mixed‑signal block that bridges the digital memory system and the analog photonic core. It requires fast, moderate‑resolution DACs with precise timing and optional linearization. In the CXL‑coherent PTA, data is streamed from coherent memory through small FIFOs into the DACs, with flow control handled by the LSU and a hardware sequencer. This interface is faithfully modeled in FPGA emulation, enabling full system validation before committing to silicon photonics.

Would you like to explore the weight programming DAC interface (slow, high‑resolution) in similar detail, or discuss the overall control FSM for the photonic pipeline?

---

## Overall Control FSM for the Photonic Pipeline

The control FSM is the heart of the Photonic Tensor Accelerator’s digital controller. It orchestrates the sequence of operations required to execute commands such as `PTA_MVM`, `PTA_LOAD_WEIGHTS`, and `PTA_CALIBRATE`. The FSM must coordinate:

- Coherent memory access via the LSU (for input/output data streaming).
- Activation DAC updates and optical pulse triggering.
- Weight DAC programming (slow) and settling delays.
- ADC readout and result storage.
- Error detection and recovery.

The FSM can be implemented either as dedicated hardware (for high‑speed, deterministic control) or as software running on the embedded RISC‑V core (for flexibility, especially calibration and error handling). In practice, a hybrid approach is best: a hardware FSM handles the time‑critical compute pipeline, while the RISC‑V manages command decoding, calibration loops, and complex error recovery.

Below we define a hardware FSM for the core operations, with hooks for the RISC‑V to intervene when needed.

### 1. Top‑Level Command Execution Flow

When the PTA receives a command (via the command queue or doorbell), the digital control unit decodes it and dispatches to the appropriate FSM routine. Each routine is a sequence of states that interact with the LSU, DACs, and photonic core.

```plaintext
+-------------------+
| Idle              |
+---------+---------+
          | command received
          v
+-------------------+
| Decode Command    |
+---------+---------+
          |
   +------+------+------+
   |             |      |
   v             v      v
Load Weights   MVM    Calibrate
   |             |      |
   +------+------+------+
          |
          v
+-------------------+
| Wait for completion|
+---------+---------+
          |
          v
+-------------------+
| Report status/IRQ  |
+-------------------+
```

The FSM returns to `Idle` after each command (unless pipelining is implemented, which is advanced).

### 2. MVM Execution FSM (Core Compute Pipeline)

This is the most critical and time‑sensitive FSM. It executes the matrix‑vector multiplication:

Y=W⋅XY=W⋅X

where WW is already programmed in the photonic weight bank, XX is a vector (or batch of vectors) to be streamed from memory, and YY is the result to be written back.

#### 2.1 State Definitions

<table>
  <tr><th>State</th><th>Description</th></tr>
  <tr><td>MVM_IDLE</td><td>No active MVM. Waits for start signal from command decoder.</td></tr>
  <tr><td>MVM_CHECK_WEIGHTS</td><td>Verify that required weights are programmed; if not, signal error or trigger implicit load.</td></tr>
  <tr><td>MVM_SETUP_LSU_READ</td><td>Program the LSU to read input activations from coherent memory into input FIFO. Configure address, length, stride, cache hints.</td></tr>
  <tr><td>MVM_SETUP_LSU_WRITE</td><td>Program the LSU to write output results from output FIFO to coherent memory. Configure address, length, stride.</td></tr>
  <tr><td>MVM_WAIT_INPUT_DATA</td><td>Wait until the input FIFO has sufficient data (at least one vector) or the LSU read has started producing.</td></tr>
  <tr><td>MVM_DAC_UPDATE</td><td>Latch the next input vector from the input FIFO into the activation DAC registers.</td></tr>
  <tr><td>MVM_WAIT_DAC_SETTLE</td><td>Wait for DAC settling time (e.g., 2 ns).</td></tr>
  <tr><td>MVM_FIRE_OPTICAL</td><td>Trigger the optical pulse (or enable the modulators) for a fixed integration time.</td></tr>
  <tr><td>MVM_WAIT_ADC</td><td>Wait for ADC conversion latency.</td></tr>
  <tr><td>MVM_READ_ADC</td><td>Read ADC outputs and push results into the output FIFO.</td></tr>
  <tr><td>MVM_CHECK_INPUT_DONE</td><td>If more input vectors remain, go back to MVM_DAC_UPDATE; else proceed to MVM_WAIT_OUTPUT_DONE.</td></tr>
  <tr><td>MVM_WAIT_OUTPUT_DONE</td><td>Wait until the LSU write has drained the output FIFO and completed.</td></tr>
  <tr><td>MVM_DONE</td><td>Signal command completion.</td></tr>
  <tr><td>MVM_ERROR</td><td>Handle error (e.g., FIFO underflow, LSU fault).</td></tr>
</table>

#### 2.2 State Transition Diagram

```plaintext
MVM_IDLE --start--> MVM_CHECK_WEIGHTS --ok--> MVM_SETUP_LSU_READ
                                     |--fail--> MVM_ERROR
MVM_SETUP_LSU_READ --done--> MVM_SETUP_LSU_WRITE --done--> MVM_WAIT_INPUT_DATA
MVM_WAIT_INPUT_DATA --data_available--> MVM_DAC_UPDATE
MVM_DAC_UPDATE --done--> MVM_WAIT_DAC_SETTLE --timeout--> MVM_FIRE_OPTICAL
MVM_FIRE_OPTICAL --pulse_end--> MVM_WAIT_ADC --timeout--> MVM_READ_ADC
MVM_READ_ADC --done--> MVM_CHECK_INPUT_DONE --more--> MVM_DAC_UPDATE
                                           |--no_more--> MVM_WAIT_OUTPUT_DONE
MVM_WAIT_OUTPUT_DONE --done--> MVM_DONE
MVM_DONE --> MVM_IDLE (or report interrupt)
Any state --error--> MVM_ERROR --handle--> MVM_IDLE
```

#### 2.3 Timing and Pipelining

The FSM can be pipelined to overlap DAC settling, optical pulse, and ADC conversion with LSU data transfers. For example, while one vector is being computed, the next vector can be loaded into the DACs. This is achieved by using double buffering in the input and output FIFOs and by allowing the LSU to run concurrently with the compute states. In the state machine, `MVM_WAIT_INPUT_DATA` and `MVM_WAIT_OUTPUT_DONE` can be active simultaneously with `MVM_DAC_UPDATE` and `MVM_READ_ADC` if the hardware supports parallel execution. A more advanced implementation would use separate threads or a dataflow pipeline.

For simplicity, the initial FSM can be sequential, with later optimization adding concurrency.

### 3. Weight Loading FSM

Weight loading involves programming the photonic weight bank from digital weight data. This is a slow process due to thermal settling.

<table>
  <tr><th>State</th><th>Description</th></tr>
  <tr><td>WL_IDLE</td><td>Wait for PTA_LOAD_WEIGHTS command.</td></tr>
  <tr><td>WL_SETUP_LSU</td><td>Program LSU to read weight data from coherent memory (or from local weight buffer) into a temporary buffer.</td></tr>
  <tr><td>WL_WAIT_DATA</td><td>Wait for weight data to be available in the buffer.</td></tr>
  <tr><td>WL_PROGRAM_DAC</td><td>For each weight element: compute DAC code from digital weight (using LUT) and write to weight DAC register.</td></tr>
  <tr><td>WL_SETTLE</td><td>Wait for thermal phase shifters to settle (programmable delay, e.g., 100 µs).</td></tr>
  <tr><td>WL_VERIFY</td><td>Optional: read monitor photodiodes and adjust DAC codes.</td></tr>
  <tr><td>WL_DONE</td><td>Signal completion.</td></tr>
</table>

This FSM can be implemented on the RISC‑V core because it is not time‑critical relative to the compute pipeline; the RISC‑V can run a loop to write DAC registers and wait. However, for very large matrices, hardware acceleration of the loop may be needed.

### 4. Calibration FSM

Calibration is a periodic maintenance routine. It can be triggered by a command, by a timer, or by temperature drift detection.

<table>
  <tr><th>State</th><th>Description</th></tr>
  <tr><td>CAL_IDLE</td><td>Wait for trigger.</td></tr>
  <tr><td>CAL_SELECT_PATTERN</td><td>Choose a known test vector from ROM.</td></tr>
  <tr><td>CAL_APPLY_INPUT</td><td>Write test vector to activation DACs, fire optical pulse.</td></tr>
  <tr><td>CAL_READ_OUTPUT</td><td>Read ADC outputs.</td></tr>
  <tr><td>CAL_COMPUTE_ERROR</td><td>(On RISC‑V) Compute error between measured and expected outputs.</td></tr>
  <tr><td>CAL_ADJUST_WEIGHTS</td><td>Update weight DAC codes to minimize error.</td></tr>
  <tr><td>CAL_NEXT_PATTERN</td><td>Repeat for all patterns or until convergence.</td></tr>
  <tr><td>CAL_DONE</td><td>Save calibration data, signal completion.</td></tr>
</table>

Because calibration involves iterative computations and complex algorithms, it is best handled by the embedded RISC‑V core, which can call hardware primitives for applying inputs and reading outputs.

### 5. Interaction with LSU and DAC Interfaces

The FSM communicates with other blocks via control/status registers:

- **LSU**: The FSM writes to LSU registers to start transfers, and polls `LSU_STATUS` for completion or errors. Interrupts may also be used.
- **Activation DAC interface**: The FSM writes a vector of activation values to a memory‑mapped DAC register array, then pulses a trigger signal.
- **Weight DAC interface**: The FSM (or RISC‑V) writes individual weight DAC codes; no trigger needed as updates are continuous.
- **Photonic core**: The FSM controls the laser pulse (or modulator enable) and reads ADC outputs.

All these interactions are through the PTA’s internal bus (e.g., AXI‑Lite), which is also accessible by the RISC‑V core. The hardware FSM can be implemented as a simple state machine that accesses these registers directly, or the RISC‑V can perform the same actions with software; hardware is faster for the MVM pipeline.

### 6. Error Handling

The FSM monitors for error conditions:

- **LSU errors**: timeouts, translation faults, FIFO overflow/underflow.
- **DAC/ADC errors**: out‑of‑range, settling timeout.
- **Weight programming failures**: verification mismatch.

On error, the FSM enters an `ERROR` state, records the error code and offending address, and optionally raises an interrupt to the host. The RISC‑V can then read the error status and attempt recovery (e.g., re‑program weights, re‑issue command). For non‑recoverable errors, the PTA may reset.

### 7. Implementation Considerations for FPGA Emulation

In the FPGA emulation, the entire FSM is implemented in RTL and controls the digital emulation of the photonic core. The “optical pulse” is a digital signal that triggers the emulated matrix multiplication, and the “ADC” is a quantizer with added noise. The FSM timing parameters (DAC settling, ADC latency) are implemented as programmable counters, allowing exploration of different analog latencies without real hardware.

The state machine can be written in Verilog/SystemVerilog as a classic Moore or Mealy FSM. It is controlled by the PTA command decoder, which itself is driven by the command queue (or MMIO from host). The embedded RISC‑V core, if present, can override or assist the FSM by writing to control registers.

### 8. Summary

The control FSM for the photonic pipeline is responsible for sequencing coherent memory accesses, DAC updates, optical pulse generation, and ADC readout to execute matrix‑vector multiplications. It is designed to be modular, with dedicated sub‑FSMs for weight loading and calibration. For high performance, the MVM FSM is implemented in hardware, while complex calibration algorithms run on the embedded RISC‑V. This division of labor balances speed and flexibility, enabling the PTA to function as a coherent CXL accelerator with minimal software overhead.

Would you like to see a specific hardware implementation example (e.g., Verilog skeleton) or discuss how to pipeline the MVM FSM for higher throughput?

---

Below is a **Verilog/SystemVerilog skeleton** for the MVM control FSM in an FPGA emulation environment. It illustrates the core sequencing logic: reading activations from a FIFO (fed by the LSU or a simplified memory interface), loading them into the activation DAC registers, triggering the emulated photonic core, waiting for the ADC (model) latency, and writing results to an output FIFO. The FSM is parameterized for analog settling and conversion delays, which can be adjusted to study timing.

```plaintext
module pta_mvm_control #(
    parameter NUM_INPUTS      = 64,          // N (number of input nodes)
    parameter NUM_OUTPUTS     = 64,          // M (number of output nodes)
    parameter DAC_SETTLE_CYCLES = 2,         // Emulated DAC settling time
    parameter ADC_LATENCY_CYCLES = 5,        // Emulated ADC pipeline delay
    parameter DATA_WIDTH      = 8            // Activation bit width
)(
    input  logic clk,
    input  logic rst_n,

    // Command interface
    input  logic start,                     // Pulse to start MVM
    input  logic weights_ready,             // High if weights are programmed
    output logic done,                      // Pulse when MVM complete
    output logic error,                     // Sticky error flag

    // Input FIFO interface (from LSU / memory)
    input  logic                   in_fifo_empty,
    output logic                   in_fifo_rd_en,
    input  logic [DATA_WIDTH-1:0]  in_fifo_data,
    input  logic                   in_fifo_last,   // Last word of a vector

    // Output FIFO interface (to LSU / memory)
    input  logic                   out_fifo_full,
    output logic                   out_fifo_wr_en,
    output logic [DATA_WIDTH-1:0]  out_fifo_data,

    // Activation DAC register file (emulated)
    output logic [DATA_WIDTH-1:0]  dac_data [NUM_INPUTS],
    output logic                   dac_update,     // Pulse to latch all DACs
    output logic                   optical_trigger,// Pulse to fire laser

    // ADC interface (emulated)
    input  logic                   adc_valid,
    input  logic [DATA_WIDTH-1:0]  adc_data [NUM_OUTPUTS],
    output logic                   adc_read_en     // Acknowledge ADC read
);

    // FSM states
    typedef enum logic [3:0] {
        IDLE,
        CHECK_WEIGHTS,
        READ_INPUT,
        LOAD_DAC,
        WAIT_DAC_SETTLE,
        FIRE_OPTICAL,
        WAIT_ADC,
        READ_ADC,
        WRITE_OUTPUT,
        CHECK_DONE,
        DONE_STATE,
        ERROR_STATE
    } state_t;

    state_t state, next_state;

    // Internal counters for settling and latency
    logic [$clog2(ADC_LATENCY_CYCLES+1)-1:0] settle_cnt;
    logic [$clog2(ADC_LATENCY_CYCLES+1)-1:0] adc_cnt;

    // Vector counters
    logic [$clog2(NUM_INPUTS+1)-1:0] input_cnt;
    logic [$clog2(NUM_OUTPUTS+1)-1:0] output_cnt;

    // Temporary vector storage (for DAC or ADC)
    logic [DATA_WIDTH-1:0] dac_reg [NUM_INPUTS];
    logic [DATA_WIDTH-1:0] adc_reg [NUM_OUTPUTS];

    // Output FIFO write control
    logic out_fifo_wr_en_reg;
    logic [DATA_WIDTH-1:0] out_fifo_data_reg;

    // State register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            settle_cnt <= '0;
            adc_cnt <= '0;
            input_cnt <= '0;
            output_cnt <= '0;
            dac_reg <= '{default: '0};
            done <= 1'b0;
            error <= 1'b0;
            dac_update <= 1'b0;
            optical_trigger <= 1'b0;
            adc_read_en <= 1'b0;
            out_fifo_wr_en <= 1'b0;
            out_fifo_data <= '0;
        end else begin
            state <= next_state;

            // Default pulse outputs
            dac_update <= 1'b0;
            optical_trigger <= 1'b0;
            adc_read_en <= 1'b0;
            out_fifo_wr_en <= 1'b0;
            done <= 1'b0;

            case (state)
                IDLE: begin
                    if (start && weights_ready) begin
                        // Begin; clear error
                        error <= 1'b0;
                        input_cnt <= '0;
                        output_cnt <= '0;
                    end else if (start && !weights_ready) begin
                        error <= 1'b1; // weights not loaded
                    end
                end

                LOAD_DAC: begin
                    // Pop one word from input FIFO and store into dac_reg[input_cnt]
                    if (!in_fifo_empty) begin
                        dac_reg[input_cnt] <= in_fifo_data;
                        input_cnt <= input_cnt + 1;
                        if (input_cnt == NUM_INPUTS-1) begin
                            // All inputs for this vector loaded
                            input_cnt <= '0;
                        end
                    end
                end

                WAIT_DAC_SETTLE: begin
                    settle_cnt <= settle_cnt + 1;
                    if (settle_cnt == DAC_SETTLE_CYCLES-1) begin
                        settle_cnt <= '0;
                    end
                end

                FIRE_OPTICAL: begin
                    optical_trigger <= 1'b1;
                    // Activate the emulated photonic core (external)
                end

                WAIT_ADC: begin
                    adc_cnt <= adc_cnt + 1;
                    if (adc_cnt == ADC_LATENCY_CYCLES-1) begin
                        adc_cnt <= '0;
                    end
                end

                READ_ADC: begin
                    if (adc_valid) begin
                        // Capture all ADC outputs for this vector
                        for (int i=0; i<NUM_OUTPUTS; i++) begin
                            adc_reg[i] <= adc_data[i];
                        end
                        adc_read_en <= 1'b1;
                        output_cnt <= '0;
                    end
                end

                WRITE_OUTPUT: begin
                    if (!out_fifo_full && output_cnt < NUM_OUTPUTS) begin
                        out_fifo_wr_en <= 1'b1;
                        out_fifo_data <= adc_reg[output_cnt];
                        output_cnt <= output_cnt + 1;
                    end
                end

                DONE_STATE: begin
                    done <= 1'b1;
                end

                ERROR_STATE: begin
                    error <= 1'b1;
                end
            endcase
        end
    end

    // Next-state logic
    always_comb begin
        next_state = state;
        in_fifo_rd_en = 1'b0;

        case (state)
            IDLE: begin
                if (start) begin
                    if (weights_ready)
                        next_state = LOAD_DAC; // assume input FIFO already has data
                    else
                        next_state = ERROR_STATE;
                end
            end

            LOAD_DAC: begin
                if (!in_fifo_empty) begin
                    in_fifo_rd_en = 1'b1;
                    if (input_cnt == NUM_INPUTS-1)
                        next_state = WAIT_DAC_SETTLE;
                    else
                        next_state = LOAD_DAC; // continue loading next input
                end
            end

            WAIT_DAC_SETTLE: begin
                if (settle_cnt == DAC_SETTLE_CYCLES-1)
                    next_state = FIRE_OPTICAL;
            end

            FIRE_OPTICAL: begin
                next_state = WAIT_ADC;
            end

            WAIT_ADC: begin
                if (adc_cnt == ADC_LATENCY_CYCLES-1)
                    next_state = READ_ADC;
            end

            READ_ADC: begin
                if (adc_valid) begin
                    next_state = WRITE_OUTPUT;
                end
            end

            WRITE_OUTPUT: begin
                if (output_cnt == NUM_OUTPUTS)
                    next_state = CHECK_DONE;
                else if (out_fifo_full)
                    next_state = WRITE_OUTPUT; // stall
                else
                    next_state = WRITE_OUTPUT;
            end

            CHECK_DONE: begin
                // Here you would check if more vectors remain (e.g., batch > 1)
                // For a single vector, go to DONE_STATE
                next_state = DONE_STATE;
            end

            DONE_STATE: begin
                next_state = IDLE;
            end

            ERROR_STATE: begin
                next_state = IDLE; // or stay until reset
            end
        endcase
    end

    // Continuous assignments for DAC register outputs
    assign dac_data = dac_reg;
    assign dac_update = (state == WAIT_DAC_SETTLE && settle_cnt == DAC_SETTLE_CYCLES-1); // pulse at end of settle
    assign optical_trigger = (state == FIRE_OPTICAL); // pulse (combinational, not registered)
    // Note: In actual design, these pulses should be registered to avoid glitches.

endmodule
```

### Explanation

- **States**: The FSM follows the sequence described earlier: load inputs into DAC registers, wait for settling, fire optical pulse, wait for ADC latency, read ADC, write outputs.
- **Emulation specifics**:
  - The DAC register file (`dac_reg`) holds the current activation vector.
  - `dac_update` and `optical_trigger` are pulses; in the code they are combinational from the state, but for clean timing you would register them (e.g., generate a one‑cycle pulse in the `always_ff` block).
  - The ADC latency is emulated with a counter (`adc_cnt`).
  - The input FIFO and output FIFO interfaces are simple handshakes; in a real design, they would connect to the LSU or a memory arbiter.
- **Scalability**: The module is parameterized for matrix dimensions and delay values, allowing quick experimentation.

This skeleton can be integrated into your FPGA emulation environment. The actual photonic core model would instantiate a digital matrix multiplier that reads `dac_reg` and produces `adc_data` after `optical_trigger`. The LSU would fill `in_fifo` and drain `out_fifo` using coherent memory access (or simple DMA in the emulation).

Would you like to see the corresponding **emulated photonic core model** or the **LSU wrapper** that connects to this FSM?

---

## Emulated Photonic Core Model

The emulated photonic core is a digital block that mimics the behavior of an analog photonic matrix‑vector multiplier. It receives digital activation values (already quantized as if from DACs), performs a matrix‑vector multiplication using a stored weight matrix, adds configurable noise and quantization effects to emulate analog impairments, and outputs the results as if they were digitized by ADCs. This model is used in FPGA emulation to validate the digital control logic (FSM, LSU, command handling) without physical photonic hardware.

### 1. Functionality

The module has:

- **Inputs**:
  - `dac_data`: N × `DATA_WIDTH` activation values from the DAC registers.
  - `optical_trigger`: a pulse that indicates the photonic core should “compute” the matrix product.
  - `clk`, `rst_n`.
- **Outputs**:
  - `adc_valid`: asserted when the computed result is ready.
  - `adc_data`: M × `DATA_WIDTH` output vector (after ADC quantization).
  - `adc_read_en`: optional, used by FSM to acknowledge.

The core stores a weight matrix WW of size M×NM×N, with each weight quantized to `WEIGHT_WIDTH` bits (typically 8). The computation is:

yj=∑i=0N−1wj,i⋅xiyj​=i=0∑N−1​wj,i​⋅xi​

where xixi​ are the input activations (quantized to `DATA_WIDTH` bits) and yjyj​ are the output values before ADC quantization. After computing yjyj​, the model adds **Gaussian‑like noise** (using a pseudo‑random LFSR) and then quantizes to `DATA_WIDTH` bits to mimic ADC resolution. Bias can also be added if desired.

The core implements the matrix multiplication in parallel using M×NM×N multiply‑accumulate units (DSPs) if resources allow, or sequentially to save area. For FPGA emulation with small matrix sizes (e.g., 8×8, 16×16), parallel implementation is feasible. The module is parameterized to adjust dimensions and precision.

### 2. Verilog/SystemVerilog Implementation

```plaintext
module photonic_core_model #(
    parameter NUM_INPUTS   = 8,          // N
    parameter NUM_OUTPUTS  = 8,          // M
    parameter DATA_WIDTH   = 8,          // Activation and output bit width
    parameter WEIGHT_WIDTH = 8,          // Stored weight bit width (signed)
    parameter ADC_LATENCY_CYCLES = 3,    // Emulated ADC pipeline delay
    parameter ENABLE_NOISE = 1,          // Enable additive noise
    parameter NOISE_SEED   = 32'h12345678
)(
    input  logic clk,
    input  logic rst_n,

    // Activation inputs (from DAC registers)
    input  logic [DATA_WIDTH-1:0] dac_data [NUM_INPUTS],
    input  logic                  optical_trigger,

    // ADC outputs
    output logic                  adc_valid,
    output logic [DATA_WIDTH-1:0] adc_data [NUM_OUTPUTS],
    input  logic                  adc_read_en      // Optional handshake
);

    // Weight matrix stored in registers (or block RAM)
    // For emulation, we initialize with a simple pattern; in real use, load via a separate interface.
    logic signed [WEIGHT_WIDTH-1:0] weights [NUM_OUTPUTS][NUM_INPUTS];

    // Bias (optional)
    logic signed [WEIGHT_WIDTH-1:0] bias [NUM_OUTPUTS];

    // Internal pipeline registers
    logic signed [DATA_WIDTH+WEIGHT_WIDTH+$clog2(NUM_INPUTS)-1:0] accum [NUM_OUTPUTS];
    logic [DATA_WIDTH-1:0] adc_reg [NUM_OUTPUTS];
    logic adc_valid_reg;

    // LFSR for noise generation
    logic [31:0] lfsr;
    logic noise_enable = ENABLE_NOISE;

    // Initialization of weights (replace with actual loading mechanism)
    initial begin
        for (int i=0; i<NUM_OUTPUTS; i++) begin
            for (int j=0; j<NUM_INPUTS; j++) begin
                weights[i][j] = (i*NUM_INPUTS + j) % (2**(WEIGHT_WIDTH-1)-1); // simple pattern
            end
            bias[i] = i;
        end
    end

    // Pipeline to emulate ADC latency
    logic [ADC_LATENCY_CYCLES-1:0] latency_sr [NUM_OUTPUTS]; // shift register for valid signal

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lfsr <= NOISE_SEED;
            adc_valid_reg <= 1'b0;
            for (int i=0; i<NUM_OUTPUTS; i++) begin
                accum[i] <= '0;
                adc_reg[i] <= '0;
                latency_sr[i] <= '0;
            end
        end else begin
            // Default
            adc_valid_reg <= 1'b0;

            // LFSR update for noise
            lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[0]}; // simple LFSR

            if (optical_trigger) begin
                // Compute matrix-vector product in parallel
                for (int i=0; i<NUM_OUTPUTS; i++) begin
                    // Multiply-accumulate
                    // We use signed multiplication
                    int signed product;
                    int signed sum;
                    sum = bias[i]; // include bias
                    for (int j=0; j<NUM_INPUTS; j++) begin
                        product = $signed(weights[i][j]) * $signed(dac_data[j]);
                        sum += product;
                    end
                    // Add noise (if enabled)
                    if (noise_enable) begin
                        // Approximate Gaussian noise by summing a few LFSR bits
                        // Here we add a small offset based on LFSR
                        sum += ($signed(lfsr[3:0]) - 8); // range -8 to 7
                    end
                    accum[i] <= sum;

                    // Quantize to DATA_WIDTH bits (simulate ADC)
                    if (sum > (2**(DATA_WIDTH-1)-1))
                        adc_reg[i] <= (2**(DATA_WIDTH-1)-1);
                    else if (sum < -(2**(DATA_WIDTH-1)))
                        adc_reg[i] <= -(2**(DATA_WIDTH-1));
                    else
                        adc_reg[i] <= sum[DATA_WIDTH-1:0]; // truncate (should be signed)

                    // Start latency shift register for valid
                    latency_sr[i] <= {latency_sr[i][ADC_LATENCY_CYCLES-2:0], 1'b1};
                end
            end else begin
                // Shift latency registers
                for (int i=0; i<NUM_OUTPUTS; i++) begin
                    latency_sr[i] <= {latency_sr[i][ADC_LATENCY_CYCLES-2:0], 1'b0};
                end
            end

            // Generate adc_valid after latency
            if (latency_sr[0][ADC_LATENCY_CYCLES-1]) begin
                adc_valid_reg <= 1'b1;
                // adc_data already captured
            end
        end
    end

    // Assign outputs
    assign adc_valid = adc_valid_reg;
    assign adc_data = adc_reg;

endmodule
```

### 3. Explanation of the Model

#### Weight Matrix Storage

- `weights` is a 2D array of signed integers (`WEIGHT_WIDTH` bits). In this example, it is initialized to a simple pattern; in a full system, weights are loaded via a separate interface (e.g., from the weight buffer using the `PTA_LOAD_WEIGHTS` command). For emulation, you can replace the initial block with a load mechanism.

#### Computation

- On `optical_trigger` pulse, the module computes all M outputs in parallel using nested loops with signed multiplication. The product of each weight and input is accumulated into `sum`.
- Bias is optionally added.
- Noise is injected by adding a small pseudo‑random value derived from the LFSR. This mimics photodetector and TIA noise. The amount can be adjusted.

#### ADC Quantization

- The final sum is saturated and truncated to `DATA_WIDTH` bits. For signed representation, use appropriate conversion. The code shows a simple truncation; for correct signed handling, you may need to select the lower bits after ensuring sign extension, but for simulation it's acceptable.

#### ADC Latency

- A shift register `latency_sr` per output is used to delay the `adc_valid` signal by `ADC_LATENCY_CYCLES` cycles after the trigger, emulating the ADC conversion pipeline. The output data (`adc_reg`) is already computed at the trigger, but valid is delayed. This matches the FSM that waits for `ADC_LATENCY_CYCLES` before reading.

#### Handshake

- `adc_read_en` is not used in this simple model; the FSM reads the data when `adc_valid` is high. If desired, you can add logic to deassert `adc_valid` after read.

### 4. Integration with the FSM

In the FSM, after `FIRE_OPTICAL`, it transitions to `WAIT_ADC`. The `photonic_core_model` receives the `optical_trigger` pulse (which should be a single‑cycle pulse). The model computes and after `ADC_LATENCY_CYCLES` asserts `adc_valid`. The FSM, after counting the same number of cycles, enters `READ_ADC` and captures `adc_data`. The timing is aligned by design.

### 5. Parameterization and Resource Usage

- `NUM_INPUTS`, `NUM_OUTPUTS`, `DATA_WIDTH` can be adjusted.
- `ADC_LATENCY_CYCLES` must match the FSM’s parameter.
- For larger matrices (e.g., 64×64), the parallel multiplier array may consume many DSPs. In that case, consider a sequential MAC implementation that reuses a few multipliers but requires a longer latency. You can modify the model to perform the multiply‑accumulate over multiple cycles, and set `ADC_LATENCY_CYCLES` accordingly.

### 6. Enhancements for Realism

- **Weight drift**: You can add a slow‑varying offset to each weight, perhaps driven by a second LFSR or a temperature‑dependent parameter.
- **Nonlinearity**: Implement the MZM sinusoidal transfer function for activation modulation, if modeling modulator nonlinearity.
- **Crosstalk**: Add coupling terms between adjacent channels.

These can be incorporated without changing the interface.

### 7. Conclusion

This emulated photonic core model provides a cycle‑accurate digital approximation of an analog photonic matrix‑vector multiplier. It allows the entire PTA digital control system—FSM, LSU, command handling—to be tested and debugged on an FPGA before real photonic hardware is available. The model is fully parameterizable and can be extended to include more detailed analog effects as needed.

---
