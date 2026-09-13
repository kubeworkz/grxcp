# GRX930 SoC — Architecture Specification

**Status:** Implemented (RTL verified, bitstream on Arty A7-35T). **Target audience:** grxcp Phase 7 backend authors, SoC integrators, verification. **Companion:** `c930/rtl/` (RTL), `c930/tb/` (testbenches), `c930/synth_xilinx/` (Vivado flow).

---

## 1. Overview

The GRX930 is a RISC-V 64-bit SoC integrating:

- **RV64IMAC core** — 5-stage in-order, with I/D caches, AMO, LR/SC, MMIO bridge
- **INT8 systolic-array NPU** — weight-stationary GEMM engine with AXI4 DMA
- **Unified DDR** — 64 KB byte-addressable memory (cache-line organized)
- **Clock divider** — configurable ÷N for safe operation below routed Fmax

The NPU is a **memory-mapped accelerator** on the CPU's AXI fabric. The CPU programs it over MMIO; the NPU autonomously fetches A/B from DDR, runs the GEMM, and writes C back — no CPU data staging required.

```plaintext
                    ┌──────────────────────────────────────────────┐
                    │              GRX930 SoC                      │
                    │                                              │
  i_clk (100 MHz)──┤──► clk_div (÷N) ──► core_clk                │
  i_rst_n ─────────┤                                              │
                    │  ┌──────────────┐    ┌──────────────────┐   │
                    │  │ riscv_core   │    │  c930_npu_top    │   │
                    │  │   top        │    │  ┌────────────┐  │   │
                    │  │  (RV64IMAC)  │    │  │ CSR (MMIO) │  │   │
                    │  │              │    │  │ AXI4-Lite  │──┼──► 0x4000_0000
                    │  │  ┌─cache────┐│    │  ├────────────┤  │   │
                    │  │  │ I$/D$    ││    │  │ DMA Master │──┼──► AXI4 full
                    │  │  └────┬─────┘│    │  │ (fetch A/B,│  │   │ (DDR data plane)
                    │  │       │      │    │  │  write C)  │  │   │
                    │  └───┬───┘──────┘    │  ├────────────┤  │   │
                    │      │              │  │ GEMM Core  │  │   │
                    │  ┌───▼───────────┐  │  │ (systolic) │  │   │
                    │  │ MMIO Bridge   │  │  └────────────┘  │   │
                    │  │ (CPU uncached │  └──────────────────┘   │
                    │  │  ↔ AXI4-Lite)│                          │
                    │  └───────────────┘    ┌──────────────────┐ │
                    │                       │ DDR (64 KB)      │ │
                    │  ┌───────────────┐    │ unified byte-    │ │
                    │  │ NPU status    │◄───│ addressable      │ │
                    │  │ o_npu_busy/done│   │ (BRAM banks)     │ │
                    │  │ o_npu_error   │    └──────────────────┘ │
                    │  │ o_npu_irq     │                          │
                    │  └───────────────┘                          │
                    └──────────────────────────────────────────────┘


```

---

## 2. Memory Map

Flat, byte-addressed, 64-bit addresses (zero-extended from 32-bit):

<table>
  <tr><th>Range</th><th>Size</th><th>Region</th><th>Access</th></tr>
  <tr><td>0x0000_0000 .. 0x0000_FFFF</td><td>64 KB</td><td>DDR (code + data + NPU A/B/C buffers)</td><td>Cached (CPU) / AXI4 (NPU DMA)</td></tr>
  <tr><td>0x4000_0000 .. 0x4000_001F</td><td>32 B</td><td>NPU MMIO control/status</td><td>Uncached (CPU MMIO bridge → AXI4-Lite)</td></tr>
  <tr><td>0x4000_0020 .. 0xFFFF_FFFF</td><td>—</td><td>Reserved</td><td>—</td></tr>
</table>

### DDR region (0x0000_0000 – 0x0000_FFFF)

The CPU's data cache and the NPU's AXI4 DMA master both access DDR. The cache port has priority over the AXI4 slave port. Byte-addressed, cache-line organized (32 bytes/line, 8 words/line).

**Suggested layout for grxcp backend:**

<table>
  <tr><th>Offset</th><th>Size</th><th>Content</th></tr>
  <tr><td>0x0000</td><td>—</td><td>Code / stack (CPU firmware)</td></tr>
  <tr><td>0x9000</td><td>M×K bytes</td><td>A matrix (INT8, row-major, packed 4 per 32-bit word)</td></tr>
  <tr><td>0x9100</td><td>K×N bytes</td><td>B matrix (INT8, row-major, packed 4 per 32-bit word)</td></tr>
  <tr><td>0x9200</td><td>M×N×4 bytes</td><td>C result (INT32, one word per element)</td></tr>
</table>

These offsets are conventions from the testbench; the NPU itself only requires that A/B/C buffers fit within the DDR and do not overlap.

---

## 3. CPU Core

The CPU is the reference `riscv_core_top` (RV64IMAC, 5-stage in-order):

- **ISA:** RV64IMA (no F/D/C extensions; 32-bit fixed-width instructions)
- **Pipeline:** IF → ID → EX → MEM → WB, 5 stages
- **Caches:** I-cache and D-cache, parameterized (default 32B lines, 128 lines)
- **Memory ports:** separate icache/dcache read ports + dcache write port to DDR
- **MMIO port:** uncached, for addresses ≥ `MMIO_BASE`
- **AMO/LR/SC:** hardware-supported, byte-level atomics
- **CSR unit:** mstatus, mtvec, mepc, mcause, mie, mip, mcycle, minstret
- **Trap handling:** mtvec-based, supports illegal instruction and ecall traps

### Clock configuration

The `CLK_DIV` parameter on `c930_soc_top` divides the input clock:

<table>
  <tr><th>CLK_DIV</th><th>Core clock</th><th>Notes</th></tr>
  <tr><td>1</td><td>= input clock</td><td>Default (Icarus testbenches, ECP5 flow)</td></tr>
  <tr><td>2</td><td>input / 2</td><td>Arty A7-35T (100 MHz → 50 MHz)</td></tr>
</table>

The `DONT_TOUCH` attribute on the divider FFs preserves the clock-generation hierarchy through Vivado synthesis. Even if the generated-clock XDC constraint cannot find the renamed divider FF, the design passes timing at 100 MHz (WNS > 0), guaranteeing closure at 50 MHz.

---

## 4. NPU Architecture

### 4.1 Systolic array

Weight-stationary dataflow (TPU-style). Default configuration: 8×8 PEs (`NUM_ROWS = NUM_COLS = 8` in `c930_npu_top`). The diagram below shows the first four rows/columns for legibility; the array is 8×8.

```plaintext
         activations (A rows) ──►  left → right
              │
   ┌───────┬───────┬───────┬───────┐  (×2: rows k=4..7)
   │ PE00  │ PE01  │ PE02  │ PE03  │  ◄── row k=0  (weight B[0][*])
   ├───────┼───────┼───────┼───────┤
   │ PE10  │ PE11  │ PE12  │ PE13  │  ◄── row k=1
   ├───────┼───────┼───────┼───────┤
   │ PE20  │ PE21  │ PE22  │ PE23  │  ◄── row k=2
   ├───────┼───────┼───────┼───────┤
   │ PE30  │ PE31  │ PE32  │ PE33  │  ◄── row k=3
   └───────┴───────┴───────┴───────┘
          │       │       │       │      (×2: columns n=4..7)
          ▼       ▼       ▼       ▼
      C[m][0]  C[m][1]  C[m][2]  C[m][3]   ◄── partial sums flow top → bottom


```

- **PE (k, n):** holds weight `B[k][n]`, computes `partial_sum += A[m][k] × B[k][n]`
- **Activation skew:** row k's activation pulse delayed by k cycles
- **Accumulator skew:** column n's accumulator injected delayed by n cycles
- **Result capture:** bottom-edge outputs captured in staggered window over `NUM_ROWS + NUM_COLS` cycles

### 4.2 Tiling

The NPU handles M/N/K through three nested loops **within the on-chip buffer limits**:

- **K-tiling:** `ceil(K / NUM_ROWS)` tiles; accumulator feeds back into top edge
- **N-tiling:** `ceil(N / NUM_COLS)` column tiles; fresh accumulator per tile
- **M-tiling:** output rows looped sequentially

**Internal tiling bounds** — M, N, K must all be ≤ their respective MAX parameter because the A/B/C buffers are statically sized by MAX_M, MAX_N, MAX_K:

<table>
  <tr><th>Buffer</th><th>Size (elements)</th><th>Width</th><th>Purpose</th></tr>
  <tr><td>a_mem</td><td>MAX_M × MAX_K</td><td>DIN_W bits</td><td>A matrix (activation)</td></tr>
  <tr><td>b_mem</td><td>MAX_K × MAX_N</td><td>DIN_W bits</td><td>B matrix (weight)</td></tr>
  <tr><td>c_mem</td><td>MAX_M × MAX_N</td><td>32 bits</td><td>C result (INT32 or FP32)</td></tr>
</table>

If the grxcp backend needs GEMMs larger than MAX_M/MAX_N/MAX_K, it must tile externally — see §14.7.

With default SoC parameters (NUM_ROWS=8, NUM_COLS=8, MAX_M=8, MAX_K=16, MAX_N=12):

<table>
  <tr><th>Dimension</th><th>Range</th><th>Internal tiling passes</th></tr>
  <tr><td>M</td><td>1–8</td><td>M passes (1 row each)</td></tr>
  <tr><td>N</td><td>1–12</td><td>1–2 N-tile passes (8 columns each)</td></tr>
  <tr><td>K</td><td>1–16</td><td>1–2 K-tile passes (8 rows each)</td></tr>
  <tr><td>M&gt;8</td><td>—</td><td>External tiling required (split into ≤8-row chunks)</td></tr>
  <tr><td>N&gt;12</td><td>—</td><td>External tiling required (split into ≤12-col chunks)</td></tr>
  <tr><td>K&gt;16</td><td>—</td><td>External tiling required (split into ≤16-length chunks)</td></tr>
</table>

### 4.3 Precision

- **Input:** INT8 (signed, 2's complement), 4 elements packed per 32-bit AXI beat
- **Accumulation:** INT32 (signed), one word per output element
- **INT16/FP16:** Precision-aware datapath (PREC CSR at 0x20: 0=INT8, 1=INT16, 2=FP16)

### 4.4 Modules

<table>
  <tr><th>File</th><th>Role</th></tr>
  <tr><td>c930_tensor_pe.sv</td><td>One multiply-accumulate PE (weight-stationary)</td></tr>
  <tr><td>c930_systolic_array.sv</td><td>NUM_ROWS × NUM_COLS PE mesh with weight-load addressing</td></tr>
  <tr><td>c930_npu_core.sv</td><td>GEMM FSM, A/B/C buffers, K/N-tiling, skew + capture</td></tr>
  <tr><td>c930_npu_csr.sv</td><td>MMIO register file + AXI4-Lite slave</td></tr>
  <tr><td>c930_npu_dma.sv</td><td>AXI4 full master: burst-fetch A/B, burst-write C</td></tr>
  <tr><td>c930_npu_top.sv</td><td>Accelerator IP top (CSR + DMA + core + IRQ)</td></tr>
</table>

---

## 5. NPU Register Map (AXI4-Lite Slave)

Base address: `0x4000_0000` (byte-addressed; register offsets are word-aligned).

### Register summary

<table>
  <tr><th>Offset</th><th>Name</th><th>Access</th><th>Reset</th><th>Description</th></tr>
  <tr><td>0x00</td><td>CTRL</td><td>W</td><td>0x0000_0000</td><td>Control register</td></tr>
  <tr><td>0x04</td><td>STATUS</td><td>R</td><td>0x0000_0000</td><td>Status register</td></tr>
  <tr><td>0x08</td><td>DIM_M</td><td>R/W</td><td>0x0000_0000</td><td>Output rows (M)</td></tr>
  <tr><td>0x0C</td><td>DIM_N</td><td>R/W</td><td>0x0000_0000</td><td>Output columns (N)</td></tr>
  <tr><td>0x10</td><td>DIM_K</td><td>R/W</td><td>0x0000_0000</td><td>Reduction length (K)</td></tr>
  <tr><td>0x14</td><td>A_BASE</td><td>R/W</td><td>0x0000_0000</td><td>A matrix base address (byte)</td></tr>
  <tr><td>0x18</td><td>B_BASE</td><td>R/W</td><td>0x0000_0000</td><td>B matrix base address (byte)</td></tr>
  <tr><td>0x1C</td><td>C_BASE</td><td>R/W</td><td>0x0000_0000</td><td>C result base address (byte)</td></tr>
  <tr><td>0x20</td><td>PREC</td><td>R/W</td><td>0x0000_0000</td><td>Precision mode (0=INT8, 1=INT16, 2=FP16, 3=BF16)</td></tr>
</table>

### CTRL (0x00) — Write-only

<table>
  <tr><th>Bit</th><th>Name</th><th>Description</th></tr>
  <tr><td>[0]</td><td>START</td><td>Write 1 to snapshot the current DIM/base/PREC registers into the command FIFO and launch. Cleared automatically. If the engine is busy the command queues (does not block; see the completion contract below). Writing START also clears DONE and ERROR.</td></tr>
  <tr><td>[31:1]</td><td>—</td><td>Reserved (read as 0)</td></tr>
</table>

### STATUS (0x04) — Read-only

<table>
  <tr><th>Bit</th><th>Name</th><th>Description</th></tr>
  <tr><td>[0]</td><td>BUSY</td><td>1 while the DMA is executing the current GEMM (fetch → compute → C writeback). Drops to 0 in the short bubble between queued commands, so BUSY alone does not mean the queue is drained.</td></tr>
  <tr><td>[1]</td><td>DONE</td><td>Latched level, set 1 after any GEMM completes; cleared only by writing START. Not a per-command edge: with several commands queued it goes 1 after the first completes and stays 1, so it cannot identify which command finished. See the completion contract.</td></tr>
  <tr><td>[2]</td><td>ERROR</td><td>Latched 1 if an invalid dimension was programmed (any dim = 0 or exceeds MAX). Cleared by writing START.</td></tr>
  <tr><td>[31:3]</td><td>—</td><td>Reserved (read as 0)</td></tr>
</table>

### DIM_M (0x08) — Read/Write

<table>
  <tr><th>Bits</th><th>Description</th></tr>
  <tr><td>[15:0]</td><td>Number of output rows. Must be ≥ 1 and ≤ MAX_M (default 8).</td></tr>
  <tr><td>[31:16]</td><td>Reserved (read as 0; writes ignored)</td></tr>
</table>

### DIM_N (0x0C) — Read/Write

<table>
  <tr><th>Bits</th><th>Description</th></tr>
  <tr><td>[15:0]</td><td>Number of output columns. Must be ≥ 1 and ≤ MAX_N (default 12).</td></tr>
  <tr><td>[31:16]</td><td>Reserved</td></tr>
</table>

### DIM_K (0x10) — Read/Write

<table>
  <tr><th>Bits</th><th>Description</th></tr>
  <tr><td>[15:0]</td><td>Reduction length. Must be ≥ 1 and ≤ MAX_K (default 16).</td></tr>
  <tr><td>[31:16]</td><td>Reserved</td></tr>
</table>

### A_BASE (0x14) — Read/Write

<table>
  <tr><th>Bits</th><th>Description</th></tr>
  <tr><td>[31:0]</td><td>Byte address of the A matrix in DDR. The NPU DMA reads M×K INT8 elements from this address (row-major, packed 4 per 32-bit word, little-endian). Must be word-aligned (bits [1:0] = 0).</td></tr>
</table>

### B_BASE (0x18) — Read/Write

<table>
  <tr><th>Bits</th><th>Description</th></tr>
  <tr><td>[31:0]</td><td>Byte address of the B matrix in DDR. The NPU DMA reads K×N INT8 elements from this address (row-major, packed 4 per 32-bit word, little-endian). Must be word-aligned.</td></tr>
</table>

### C_BASE (0x1C) — Read/Write

<table>
  <tr><th>Bits</th><th>Description</th></tr>
  <tr><td>[31:0]</td><td>Byte address of the C result buffer in DDR. The NPU DMA writes M×N INT32 elements to this address (row-major, one word per element). Must be word-aligned.</td></tr>
</table>

### PREC (0x20) — Read/Write

<table>
  <tr><th>Bits</th><th>Name</th><th>Description</th></tr>
  <tr><td>[1:0]</td><td>MODE</td><td>Precision mode: 0=INT8 (default), 1=INT16, 2=FP16, 3=BF16</td></tr>
  <tr><td>[31:2]</td><td>—</td><td>Reserved (read as 0)</td></tr>
</table>

The precision mode controls the element width of the systolic array PEs:

- **INT8 (0):** 8-bit signed multiply-accumulate, 4 elements per AXI beat
- **INT16 (1):** 16-bit signed multiply-accumulate, 2 elements per AXI beat
- **FP16 (2):** half-precision IEEE 754 × FP32 multiply-accumulate, 2 elements per AXI beat
- **BF16 (3):** bfloat16 (1 sign + 8 exp + 7 mant, bias=127), 2 bytes/element

The precision mode must be set before writing CTRL.START. Changing precision while the engine is busy is undefined.

### CYCLE_COUNT (0x24) — Read-only

<table>
  <tr><th>Bits</th><th>Description</th></tr>
  <tr><td>[31:0]</td><td>Free-running 32-bit cycle counter in the NPU core. Increments every clock cycle while state != S_IDLE. Resets to 0 when the core receives i_start (which fires after the DMA finishes loading A/B).</td></tr>
</table>

**Note:** This is a 32-bit counter. The address 0x28 (`CYCLE_HI`) is defined in the RTL as a localparam but is dead code — it is not wired in any read/write case statement and always reads as 0. Do not depend on 0x28.

### OP_COUNT (0x2C) — Read-only

<table>
  <tr><th>Bits</th><th>Description</th></tr>
  <tr><td>[31:0]</td><td>Hardware PE MAC operation count. Increments by NUM_ROWS × NUM_COLS each cycle during S_RUN (all PEs fire each cycle). Resets to 0 when the core receives i_start.</td></tr>
</table>

**Note:** This counts hardware PE firings, not problem-sized operations. For a problem M×N×K on a `NUM_ROWS×NUM_COLS` array, the RTL count is:

```plaintext
OP_COUNT = m_tiles × n_tiles × k_tiles × NUM_ROWS × NUM_COLS
         = ceil(M/NUM_ROWS) × ceil(N'/NUM_COLS) × ceil(K/NUM_ROWS) × NUM_ROWS × NUM_COLS


```

where N' = min(N, MAX_N). This is **not** `M×N×K×2` — it depends on the array size and tiling. The shim's `M×N×K×2` is a software metric; the RTL's count reflects actual hardware PE firings.

### STALL_COUNT (0x30) — Read-only

<table>
  <tr><th>Bits</th><th>Description</th></tr>
  <tr><td>[31:0]</td><td>Number of cycles the NPU core spent moving weights rather than computing — S_WLOAD plus S_PRELOAD. Resets to 0 when the core receives i_start.</td></tr>
</table>

`S_WLOAD` loads the first K tile's weights into the active bank; every subsequent tile is loaded by `S_PRELOAD` into the idle bank. Both move weights and neither computes, so both are counted. Exactly:

```plaintext
STALL_COUNT = M × Σ(N tiles) Σ(K tiles) kr × nc


```

with `kr = min(NUM_ROWS, K - k_base)` and `nc = min(NUM_COLS, N' - n_base)`. For full tiles this reduces to `M × ceil(N'/NUM_COLS) × ceil(K/NUM_ROWS) × NUM_ROWS × NUM_COLS`; at `M=64, N=8, K=256` on the 8×8 array that is 131,072 cycles against a `CYCLE_COUNT` near 168,000, so the array computes about 22% of the time.

The three counters close an accounting identity, which is the cheapest way to check them:

```plaintext
CYCLE_COUNT = STALL_COUNT + OP_COUNT/(NUM_ROWS × NUM_COLS) + M × Σ(N tiles) nc
              └ weights ─┘   └──── S_RUN ────┘               └─── S_WRITE ───┘


```

It holds on all 36 GEMM shapes in `tb/tb_c930_npu.sv`. Counting only `S_WLOAD`, it held for the 24 shapes with a single K tile and failed on the other 12 — which is why the defect survived: every small test agreed with it. A single-K-tile GEMM never enters `S_PRELOAD`, so its `STALL_COUNT` is unchanged from earlier builds.

### DMA_CT (0x34) — Read-only

<table>
  <tr><th>Bits</th><th>Description</th></tr>
  <tr><td>[31:0]</td><td>Number of cycles the DMA was busy (phase != P_IDLE). Resets to 0 when the DMA receives i_start (from the CSR block).</td></tr>
</table>

**Note:** DMA_CT and CYCLE_COUNT are on **different time bases**. DMA_CT starts counting when the CSR fires `CTRL.START` and counts through the entire DMA phase (A/B load + core compute + C writeback). CYCLE_COUNT starts later (when the core receives `i_start` from the DMA) and only counts during core activity. DMA_CT ≥ CYCLE_COUNT for any GEMM, and the difference includes the DMA overhead for A/B loading and C writeback. |

### QUEUE_STAT (0x38) — Read-only

<table>
  <tr><th>Bits</th><th>Name</th><th>Description</th></tr>
  <tr><td>[3:0]</td><td>OCCUPANCY</td><td>Number of commands currently in the command FIFO (0..CMD_QUEUE_DEPTH, default 4)</td></tr>
  <tr><td>[4]</td><td>FULL</td><td>1 when the FIFO is full (occupancy == CMD_QUEUE_DEPTH)</td></tr>
  <tr><td>[31:5]</td><td>—</td><td>Reserved (read as 0)</td></tr>
</table>

`QUEUE_MAX` (0x3C) reports the compile-time FIFO depth.

### Command queue and completion contract (normative)

`CTRL.START` never blocks the CPU. It **snapshots** the current DIM/A_BASE/ B_BASE/C_BASE/PREC values into a FIFO (depth `CMD_QUEUE_DEPTH` = 4). If the engine is idle and the FIFO is empty the command dispatches immediately; otherwise it waits in the FIFO and dispatches automatically when the engine next becomes idle. This makes three properties of the status interface definitive for software:

1. `STATUS.DONE`** is a latched level, not a per-command edge.** It is set whenever a GEMM completes and is cleared **only** by a write to `CTRL` with `START=1`. Because a START write and the dispatch of that command are decoupled (the write may push to the FIFO while an earlier command is still executing), DONE cannot identify *which* queued command completed. With N > 1 commands in the FIFO, DONE reads 1 from the moment the first command completes until the next START write — it says nothing about the others.
2. `STATUS.BUSY`** is per-command, with idle bubbles between commands.** BUSY is 1 only while the DMA is executing the *current* command (A/B fetch → core compute → C writeback). Between the completion of one queued command and the dispatch of the next there is a short bubble (P_DONE → P_IDLE → dispatch) during which BUSY reads 0 even though the FIFO is not empty.
3. **Therefore, for a batch of queued commands, polling **`DONE`** then **`BUSY`** is invalid.** It can observe DONE=1 (left over from an earlier command in the batch) in a BUSY=0 dispatch bubble and declare the whole batch finished while the last command has only just launched — its C writeback has not happened yet. Any host readback performed after that false completion races the in-flight GEMM. (This exact bug corrupted INT4 results in the SoC regression; see commit `f4de883`.)

The **only** robust completion test for a batch is:

```plaintext
while (QUEUE_STAT.occupancy != 0 || STATUS.BUSY != 0)
    ;   // all commands dispatched AND the last one finished


```

Both conditions are required: occupancy alone reaches 0 while the final command is still executing, and BUSY alone reads 0 in the inter-command bubbles. When the loop exits, every submitted command has fully completed and its C matrix is visible in DDR.

Recommended submission patterns:

- **One at a time (always correct):** submit → wait `DONE` then `!BUSY` → submit next. With a single command in flight, DONE and BUSY are unambiguous.
- **Batched (≤ CMD_QUEUE_DEPTH = 4 outstanding):** submit the batch, then use the occupancy + BUSY drain poll above before touching any C buffer. Never write START while the FIFO is full: START is a one-cycle pulse and its snapshot push requires FIFO space in that same cycle, so a submission against a full FIFO can be silently dropped. Keep at most `CMD_QUEUE_DEPTH` commands outstanding and drain between batches.
- Do not rely on `DONE` alone even for single commands: DONE is set when the DMA *enters* its done phase, which is fine — but the level never clears by itself, so poll for `!BUSY` (or re-arm with a fresh START) before reusing C buffers.

`STATUS.ERROR` is also latched and cleared by a `CTRL.START` write; software should check it after a drain poll and treat a nonzero value as a programming error on the batch.

### Full register index map

<table>
  <tr><th>Index</th><th>Offset</th><th>Name</th><th>RTL localparam</th><th>Notes</th></tr>
  <tr><td>0</td><td>0x00</td><td>CTRL</td><td>ADDR_CTRL</td><td>Write-only</td></tr>
  <tr><td>1</td><td>0x04</td><td>STATUS</td><td>ADDR_STAT</td><td>Read-only</td></tr>
  <tr><td>2</td><td>0x08</td><td>DIM_M</td><td>ADDR_DIM_M</td><td></td></tr>
  <tr><td>3</td><td>0x0C</td><td>DIM_N</td><td>ADDR_DIM_N</td><td></td></tr>
  <tr><td>4</td><td>0x10</td><td>DIM_K</td><td>ADDR_DIM_K</td><td></td></tr>
  <tr><td>5</td><td>0x14</td><td>A_BASE</td><td>ADDR_A_BASE</td><td></td></tr>
  <tr><td>6</td><td>0x18</td><td>B_BASE</td><td>ADDR_B_BASE</td><td></td></tr>
  <tr><td>7</td><td>0x1C</td><td>C_BASE</td><td>ADDR_C_BASE</td><td></td></tr>
  <tr><td>8</td><td>0x20</td><td>PREC</td><td>ADDR_PREC</td><td></td></tr>
  <tr><td>9</td><td>0x24</td><td>CYCLE_COUNT</td><td>ADDR_CYCLE_LO</td><td>32-bit free-running</td></tr>
  <tr><td>10</td><td>0x28</td><td>(reserved)</td><td>ADDR_CYCLE_HI</td><td>Dead code, always reads 0</td></tr>
  <tr><td>11</td><td>0x2C</td><td>OP_COUNT</td><td>ADDR_OP_COUNT</td><td></td></tr>
  <tr><td>12</td><td>0x30</td><td>STALL_COUNT</td><td>ADDR_STALL_CT</td><td></td></tr>
  <tr><td>13</td><td>0x34</td><td>DMA_CT</td><td>ADDR_DMA_CT</td><td></td></tr>
  <tr><td>14</td><td>0x38</td><td>QUEUE_STAT</td><td>ADDR_QUEUE_STAT</td><td>[3:0] occupancy, [4] full</td></tr>
  <tr><td>15</td><td>0x3C</td><td>QUEUE_MAX</td><td>ADDR_QUEUE_MAX</td><td>compile-time FIFO depth</td></tr>
</table>

---

## 6. NPU Data Format and Byte Layout

### A matrix (M × K, INT8, row-major, packed)

Element A[m][k] (signed INT8) lives at byte offset `m * K + k` from `A_BASE`. Four elements are packed per 32-bit AXI word, little-endian:

```plaintext
Word at byte offset (m*K + k) & ~3:
  bits [7:0]   = A[m][k]     where (m*K + k) % 4 == 0
  bits [15:8]  = A[m][k+1]   where (m*K + k) % 4 == 1
  bits [23:16] = A[m][k+2]   where (m*K + k) % 4 == 2
  bits [31:24] = A[m][k+3]   where (m*K + k) % 4 == 3


```

**Total bytes:** `M × K`. **Total AXI read beats:** `ceil(M × K / 4)`.

### B matrix (K × N, INT8, row-major, packed)

Element B[k][n] lives at byte offset `k * N + n` from `B_BASE`. Same packing as A.

**Total bytes:** `K × N`. **Total AXI read beats:** `ceil(K × N / 4)`.

### C result (M × N, INT32, row-major, unpacked)

Element C[m][n] (signed INT32) is one full 32-bit word at offset `(m * N + n) * 4` from `C_BASE`.

**Total bytes:** `M × N × 4`. **Total AXI write beats:** `M × N`.

---

## 7. NPU DMA Master (AXI4 Full)

The DMA is the data-plane engine that makes the NPU autonomous. The CPU never writes operand data — it only programs the CSR registers.

### 7.1 Operation sequence

```plaintext
 1. IDLE          ←等待 START
 2. READ_A        ← burst-read M×K bytes from A_BASE, unpack into A buffer
 3. READ_B        ← burst-read K×N bytes from B_BASE, unpack into B buffer
 4. LAUNCH        ← pulse core_start, wait for core_done
 5. WRITE_C       ← burst-write M×N words to C_BASE
 6. DONE          ← pulse o_done / o_irq


```

### 7.2 AXI4 burst parameters

<table>
  <tr><th>Parameter</th><th>Read (A/B)</th><th>Write (C)</th></tr>
  <tr><td>Burst type</td><td>INCR (2&#39;b01)</td><td>INCR (2&#39;b01)</td></tr>
  <tr><td>Beat size</td><td>4 bytes (BEAT_SIZE = 2)</td><td>4 bytes</td></tr>
  <tr><td>Beat count</td><td>ceil(elements / 4)</td><td>M × N</td></tr>
  <tr><td>Max arlen/awlen</td><td>255 (8-bit)</td><td>255</td></tr>
</table>

### 7.3 Backpressure

- The R channel is back-pressured while each beat's 4 bytes are unpacked into the core's A/B buffers (one byte per cycle). Read throughput is naturally throttled by the unpacking rate.
- The W channel is driven beat-by-beat from the C buffer read port.

### 7.4 Constraints

- A, B, C base addresses must be word-aligned (bits [1:0] = 0).
- DIM_M, DIM_N, DIM_K must each be ≥ 1 and ≤ their respective MAX parameters. Violation sets ERROR in STATUS and skips execution.
- A and B buffers are internal to the core (MAX_M × MAX_K and MAX_K × MAX_N INT8 elements). C buffer is MAX_M × MAX_N INT32 elements.
- The NPU does **not** check for DDR address overlap between A, B, and C. The software driver must ensure non-overlapping buffers.

---

## 8. Interrupt Model

The NPU completion signal:

<table>
  <tr><th>Signal</th><th>Behavior</th></tr>
  <tr><td>o_irq</td><td>Pulses for one core clock cycle when DONE is set (GEMM complete)</td></tr>
  <tr><td>o_done</td><td>Same as o_irq — both are driven by the DMA&#39;s done signal</td></tr>
</table>

The IRQ is a **level-sensitive pulse**, not a sticky interrupt. The CPU must poll STATUS or register an interrupt handler before launching the GEMM.

**Current implementation:** the IRQ output is wired to the SoC top-level LED outputs (`o_npu_busy`, `o_npu_done`, `o_npu_error`, `o_npu_irq`) for board-level visibility. There is no APLIC/IMSIC integration yet — the CPU polls STATUS in the current firmware.

**For grxcp Phase 7:** the recommended polling pattern is:

```c
// 1. Program dimensions and base addresses
NPU_CSR_DIM_M  = M;
NPU_CSR_DIM_N  = N;
NPU_CSR_DIM_K  = K;
NPU_CSR_A_BASE = a_ddr_addr;
NPU_CSR_B_BASE = b_ddr_addr;
NPU_CSR_C_BASE = c_ddr_addr;

// 2. Launch (also clears DONE and ERROR)
NPU_CSR_CTRL = 1;

// 3. Poll until done — single-command flow only. If you submit more than
//    one command before draining, use the occupancy+BUSY drain poll in
//    the command queue completion contract above instead.
while (!(NPU_CSR_STATUS & STATUS_DONE))
    ;

// 4. C is now valid at c_ddr_addr


```

---

## 9. AXI4-Lite Slave Interface (CSR Port)

The CSR slave accepts one outstanding transaction at a time. The AW and W channels are paired: a write commits when both AWVALID and WVALID are simultaneously high.

### 9.1 Write protocol

```plaintext
CPU store → MMIO bridge → AXI4-Lite AW+W (paired) → CSR accepts → BVALID response


```

Latency: 2–3 core clock cycles per MMIO store (bridge staging + CSR commit + response).

### 9.2 Read protocol

```plaintext
CPU load → MMIO bridge → AXI4-Lite AR → CSR returns RDATA → bridge returns to CPU


```

Latency: 2–3 core clock cycles per MMIO load.

### 9.3 Bus errors

All transactions return `BRESP = OKAY` / `RRESP = OKAY`. There is no error response — accessing undefined offsets returns 0 on read and is silently ignored on write.

---

## 10. SoC Integration (for grxcp Phase 7)

### 10.1 What grxcp needs to build

The `src/backends/npu_c930/` backend in grxcp must implement:

1. **Device enumeration** — detect the NPU by probing `STATUS` at `0x4000_0004` (a non-zero read indicates the NPU is present).
2. **Capability profile** — report `GRX_CAP_STREAMS | GRX_CAP_MEMCPY | GRX_CAP_GEMM` (no `GRX_CAP_KERNEL_LAUNCH` — the NPU has no SIMT pipeline).
3. **Memory management** — allocate A/B/C buffers in DDR, translate host pointers to physical DDR addresses for `A_BASE/B_BASE/C_BASE`.
4. **GEMM dispatch** — program DIM_M/N/K and A/B/C_BASE, write CTRL.START. For one command at a time, poll STATUS.DONE then `!BUSY` (or wait on `o_irq` if AIA is wired). For batched submission, use the occupancy + BUSY drain poll from the completion contract in the register section.
5. **Result readback** — C is written to DDR by the DMA; the backend reads it back through the normal memory path.

### 10.2 Register access patterns

All register accesses are 32-bit word-aligned MMIO loads/stores. The CPU's MMIO bridge converts uncached stores into AXI4-Lite write transactions and uncached loads into AXI4-Lite read transactions.

**Byte order:** little-endian (RV64 native). The AXI4-Lite bus is also little-endian. No byte-swapping is needed.

### 10.3 DMA address translation

The NPU DMA issues AXI4 transactions using the physical byte addresses written to `A_BASE`, `B_BASE`, `C_BASE`. In the current implementation (no MMU/IOMMU), these are direct physical DDR addresses.

For grxcp integration, the backend must ensure:

- Base addresses are within the DDR region (`0x0000_0000 – 0x0000_FFFF`)
- A, B, C buffers do not overlap each other
- Base addresses are 4-byte aligned (bits [1:0] = 0)

### 10.4 Streaming interface

The NPU does not have a streaming/DMA-submission interface. All programming is through the MMIO CSR registers. For grxcp's stream-ordered execution model:

1. The backend programs the NPU on the calling stream.
2. It records an event after `STATUS.DONE` is observed.
3. Downstream work on the same stream waits on that event.

This gives correct ordering without hardware stream concurrency (which the NPU does not support).

---

## 11. FPGA Resource Utilization and Board Recommendation

### Target board: Digilent Arty A7-100T

**Part:** XC7A100TCSG324-1 | **Board:** ~$130 | **DDR3L:** 256 MB on-board

The Arty A7-100T is the recommended board for the C930 SoC. The -35T variant (20.8K LUTs) was too small for the full NPU+DMA+64-bit-AXI design (~32K LUTs); the -100T (63.4K LUTs) fits at 52% utilization with room to spare.

### Artix-7-100T (Vivado 2026.1, routed implementation)

<table>
  <tr><th>Resource</th><th>Used</th><th>Available</th><th>Utilization</th></tr>
  <tr><td>Slice LUTs</td><td>32,767</td><td>63,400</td><td>51.7%</td></tr>
  <tr><td>Slice Registers</td><td>14,939</td><td>126,800</td><td>11.8%</td></tr>
  <tr><td>DSP48E1</td><td>43</td><td>240</td><td>17.9%</td></tr>
  <tr><td>RAMB36E1</td><td>8</td><td>135</td><td>5.9%</td></tr>
</table>

**Core clock:** 50 MHz (100 MHz board oscillator / CLK_DIV=2). **Routed Fmax:** **44.6 MHz** (WNS -2.439 ns at 50 MHz constraint). **DRC:** 0 errors (post-implementation).

### ECP5-85F (nextpnr, for comparison)

<table>
  <tr><th>Resource</th><th>Used</th><th>Available</th><th>Utilization</th></tr>
  <tr><td>LUT4</td><td>34,012</td><td>83,640</td><td>41%</td></tr>
  <tr><td>TRELLIS_FF</td><td>10,707</td><td>83,640</td><td>13%</td></tr>
  <tr><td>DP16KD</td><td>16</td><td>156</td><td>10%</td></tr>
  <tr><td>MULT18X18D</td><td>21</td><td>156</td><td>13%</td></tr>
</table>

**Routed Fmax:** 29.6 MHz (seed 2, best of sweep).

### Why Artix-7 is faster

Artix-7 has **dedicated CARRY4 carry-chain primitives** — the same 64-bit arithmetic paths that consumed ~26 ns of logic on ECP5 become ~7 ns on Artix-7. The FP16 accumulator critical path drops from 40 LUT levels (ECP5) to 33 LUT levels + 15 CARRY4 slices.

**Fmax improvement: 44.6 MHz / 29.6 MHz = 1.51× over ECP5.**

### Artix-7 200T (8×8 NPU array)

<table>
  <tr><th>Resource</th><th>Used</th><th>Available</th><th>Utilization</th></tr>
  <tr><td>Slice LUTs</td><td>72,027</td><td>134,600</td><td>53.5%</td></tr>
  <tr><td>Slice Registers</td><td>21,874</td><td>269,200</td><td>8.1%</td></tr>
  <tr><td>DSP48E1</td><td>139</td><td>740</td><td>18.8%</td></tr>
  <tr><td>RAMB36E1</td><td>8</td><td>365</td><td>2.2%</td></tr>
</table>

**Routed Fmax:** **513.8 MHz** (WNS = 8.054 ns, all constraints met). **Throughput:** 64 PEs × 513.8M MAC/s = **65.8 TOPS (INT8)**.

The 8×8 array with FP16 accumulators needs ~72K LUTs — fits comfortably on the 200T (53.5%). The FP16 CLA subtractor and barrel shifter dominate (~800 LUTs/PE × 64 PEs = ~51K LUTs for accumulators alone).

### Other boards considered

<table>
  <tr><th>Board</th><th>Part</th><th>LUTs</th><th>Fit?</th><th>Notes</th></tr>
  <tr><td>Arty A7-100T</td><td>XC7A100TCSG324-1</td><td>63.4K</td><td>✅</td><td>Smaller parameterized array only — 8×8 (~72K LUTs) overflows; the old 52% figure is a pre-widening 4×4-era measurement</td></tr>
  <tr><td>Artix-7 200T</td><td>XC7A200TFBG484-1</td><td>134.6K</td><td>✅ 53%</td><td>8×8 array target, 513.8 MHz</td></tr>
  <tr><td>Arty A7-35T</td><td>XC7A35TCSG324-1</td><td>20.8K</td><td>❌ 76%</td><td>Too small</td></tr>
  <tr><td>Nexys A7-100T</td><td>XC7A100TCSG324-1</td><td>63.4K</td><td>✅</td><td>Same FPGA, more I/O, $180</td></tr>
  <tr><td>Basys 3</td><td>XC7A35TCPG236-1</td><td>20.8K</td><td>❌</td><td>Too small</td></tr>
  <tr><td>Genesys ZU</td><td>ZU3EG</td><td>154K</td><td>✅</td><td>Overkill — Zynq UltraScale+, $350</td></tr>
</table>

### DDR3L integration

The Arty A7-100T has a 256 MB DDR3L chip (MT41K128M16JT-125) directly connected to the FPGA. The `c930_ddr3l` module replaces the behavioral DDR stub with a MIG 7 Series controller, bridging the CPU cache-line ports and NPU DMA to the real DDR3L chip. See Section 14.8 for integration details.

---

## 12. Verification

### 12.1 Testbenches

<table>
  <tr><th>Testbench</th><th>Scope</th><th>What it checks</th></tr>
  <tr><td>tb_c930_soc.sv</td><td>Full SoC</td><td>22 randomized GEMM shapes (M/N/K sweep), MMIO stress, store ordering, LR/SC, AMO, memory consistency, stranded-LR, two-trap handler</td></tr>
  <tr><td>tb_kickoff.sv</td><td>Bitstream design</td><td>Boots firmware from BRAM stub, NPU runs GEMM, C={1,2,5,6} verified</td></tr>
  <tr><td>tb_hazard_csrflush.sv</td><td>Hazard unit</td><td>CSR flush deferral, load-use stall, operand hold</td></tr>
</table>

### 12.2 Sweep coverage

The SoC testbench sweeps GEMM shapes including:

- Edge cases: K=1, M=1, N=1
- Exact tiling boundaries: K=NUM_ROWS, N=NUM_COLS
- N-tiling: N > NUM_COLS (multiple column passes)
- K-tiling: K > NUM_ROWS (multiple reduction passes)
- Maximum dimensions: M=MAX_M, N=MAX_N, K=MAX_K
- Randomized: LCG-seeded M/N/K within parameter bounds

### 12.3 Stress tests

- **MMIO drain stall:** back-to-back MMIO stores to the same CSR register
- **Store ordering:** interleaved AMOs, LR/SC, and regular stores to the same cache line
- **Memory consistency:** multi-line AMO/regular access interleaving
- **Stranded LR/SC:** icache-miss freeze between LR and its dependent read
- **Two-trap handler:** illegal instruction + ecall, cold icache line each time
- **CSR dependency:** trap mepc write under stall, no re-execution corruption

---

## 13. Roadmap

<table>
  <tr><th>Step</th><th>Description</th><th>Status</th></tr>
  <tr><td>1</td><td>INT8 systolic GEMM + CSR + testbench</td><td>✅ Done</td></tr>
  <tr><td>2</td><td>AXI4 DMA master for data plane</td><td>✅ Done</td></tr>
  <tr><td>3</td><td>INT16 / FP16 / BF16 datapaths + precision CSR</td><td>✅ Done</td></tr>
  <tr><td>4</td><td>RVV 1.0 vector unit + fused vector→matrix dispatch</td><td>Planned</td></tr>
  <tr><td>5</td><td>CHI coherent NPU port (SVM with CPU)</td><td>Planned</td></tr>
  <tr><td>6</td><td>Wide out-of-order core, DDR5/HBM, PCIe/CXL, IOMMU/AIA</td><td>Planned</td></tr>
  <tr><td>7</td><td>grxcp Phase 7 backend integration</td><td>Ready (this doc)</td></tr>
</table>

---

## 14. grxcp Integration Guide

### 14.1 Buffer sizes (MAX_M/N/K) and synthesis defaults

**MAX_M, MAX_N, MAX_K are buffer sizes**, not hard computational limits. They determine how much on-chip SRAM (BRAM) the NPU allocates for A, B, and C matrices. The runtime GEMM dimensions (set via DIM_M/N/K CSRs) must be ≤ the corresponding MAX parameter because the buffers are statically sized.

The SoC top-level (`c930_soc_top.sv`) instantiates the NPU with **synthesis defaults** that fit a small DDR stub:

<table>
  <tr><th>Parameter</th><th>SoC default</th><th>Core default</th><th>Notes</th></tr>
  <tr><td>MAX_M</td><td>8</td><td>64</td><td>A/C buffer rows</td></tr>
  <tr><td>MAX_K</td><td>16</td><td>256</td><td>A/B buffer reduction length</td></tr>
  <tr><td>MAX_N</td><td>12</td><td>8</td><td>B/C buffer columns</td></tr>
  <tr><td>NUM_ROWS</td><td>4</td><td>8</td><td>Systolic rows per tile</td></tr>
  <tr><td>NUM_COLS</td><td>4</td><td>8</td><td>Systolic cols per tile</td></tr>
</table>

**BRAM cost formulas** (for the grxcp SoC integrator):

<table>
  <tr><th>Buffer</th><th>Elements</th><th>Bits per element</th><th>Total bits</th></tr>
  <tr><td>a_mem</td><td>MAX_M × MAX_K</td><td>DIN_W (8 or 16)</td><td>MAX_M × MAX_K × DIN_W</td></tr>
  <tr><td>b_mem</td><td>MAX_K × MAX_N</td><td>DIN_W (8 or 16)</td><td>MAX_K × MAX_N × DIN_W</td></tr>
  <tr><td>c_mem</td><td>MAX_M × MAX_N</td><td>32</td><td>MAX_M × MAX_N × 32</td></tr>
</table>

**Example:** With MAX_M=64, MAX_K=256, MAX_N=8, DIN_W=8:

- a_mem: 64 × 256 × 8 = 131,072 bits = 16 KB
- b_mem: 256 × 8 × 8 = 16,384 bits = 2 KB
- c_mem: 64 × 8 × 32 = 16,384 bits = 2 KB
- **Total: ~20 KB** (fits in a single ECP5 BRAM block or Artix-7 RAMB18)

With MAX_M=8, MAX_K=16, MAX_N=12 (SoC defaults):

- a_mem: 8 × 16 × 8 = 1,024 bits = 128 B
- b_mem: 16 × 12 × 8 = 1,536 bits = 192 B
- c_mem: 8 × 12 × 32 = 3,072 bits = 384 B
- **Total: ~704 B** (fits in a single small BRAM)

**Key implication:** The grxcp backend should parameterize MAX_M/N/K to match the target FPGA's BRAM budget. Larger MAX values give bigger on-chip tiles (fewer NPU invocations for a given GEMM) but cost more BRAM. For dimensions exceeding MAX, the backend must tile externally (§14.7).

The A/B/C DDR address layout in the default firmware (`npu_boot.c`) is:

<table>
  <tr><th>Address</th><th>Size</th><th>Content</th></tr>
  <tr><td>0x8000</td><td>M×K bytes</td><td>A matrix (INT8, row-major)</td></tr>
  <tr><td>0x8400</td><td>K×N bytes</td><td>B matrix (INT8, row-major)</td></tr>
  <tr><td>0x8800</td><td>M×N×4 bytes</td><td>C result (INT32, one word per element)</td></tr>
  <tr><td>0x9000</td><td>4 bytes</td><td>DONE magic (0xDEADBEEF)</td></tr>
  <tr><td>0x9400</td><td>24 bytes</td><td>Performance benchmark results</td></tr>
</table>

### 14.2 Register model (for grxcp backend)

The grxcp backend needs a **register-model-accurate** simulation of the NPU. A green run on `simx` (software model) is NOT sufficient — the backend's decision logic must be gated against the actual hardware register map.

**Available register models:**

1. **c930_architecture.md §5** — canonical register map (this document)
2. **c930_npu_csr.sv** — RTL implementation of the CSR block
3. **npu_dpi.h** — C++ header with CSR addresses and high-level API
4. **c930_npu_dpi.sv** — standalone NPU Verilator model with DPI functions

**Recommended approach for grxcp testing:**

```c
#include "npu_dpi.h"

// Load A/B data into DDR
for (int i = 0; i < M*K; i++)
    dpi_npu_mem_write(A_ADDR + i, A[i], 0x1);

// Configure NPU via AXI4-Lite CSRs
dpi_npu_csr_write(NPU_CSR_DIM_M,  M);
dpi_npu_csr_write(NPU_CSR_DIM_N,  N);
dpi_npu_csr_write(NPU_CSR_DIM_K,  K);
dpi_npu_csr_write(NPU_CSR_A_BASE, A_ADDR);
dpi_npu_csr_write(NPU_CSR_B_BASE, B_ADDR);
dpi_npu_csr_write(NPU_CSR_C_BASE, C_ADDR);
dpi_npu_csr_write(NPU_CSR_PREC,   0);  // INT8

// Trigger and wait
dpi_npu_csr_write(NPU_CSR_START, 1);
while (!(dpi_npu_csr_read(NPU_CSR_STATUS) & 0x2))  // poll DONE bit
    ;

// Read C results
for (int i = 0; i < M*N; i++)
    C[i] = dpi_npu_mem_read(C_ADDR + i * 4);


```

### 14.3 RTLSIM requirements ([AGENTS.md](http://AGENTS.md) §4)

Every conformance test must run on both `simx` (software model) and `rtlsim` (cycle-accurate RTL). Current status:

<table>
  <tr><th>Requirement</th><th>Status</th><th>Path</th></tr>
  <tr><td>Icarus RTL testbench</td><td>✅ Complete</td><td>c930/tb/tb_c930_soc.sv — 22-case sweep + stress</td></tr>
  <tr><td>Verilator cycle-accurate</td><td>✅ Complete</td><td>c930/sim/c930_soc_verilator.sv</td></tr>
  <tr><td>Standalone NPU DPI wrapper</td><td>✅ Complete</td><td>c930/sim/c930_npu_dpi.sv</td></tr>
  <tr><td>C++ DPI test harness</td><td>✅ Complete</td><td>c930/sim/npu_dpi_test.cc</td></tr>
</table>

**Build commands:**

```bash
# Full SoC simulation (Icarus)
cd c930 && make soc

# Standalone NPU DPI model (Verilator)
cd c930 && make verilate-npu

# Full SoC Verilator model
cd c930 && make verilate


```

**Note:** The Verilator model uses a flat-1D DDR stub (`c930_ddr_verilator.sv`) instead of the synth stub's 2D banked array, because Verilator doesn't handle 2D memory initialization correctly. Both models produce identical cycle counts (verified: NPU done at cycle 180 in both Icarus and Verilator).

### 14.4 Performance counters

The NPU provides three performance counters accessible via MMIO:

<table>
  <tr><th>CSR</th><th>Address</th><th>Description</th></tr>
  <tr><td>CYCLE_COUNT</td><td>0x4000_0024</td><td>Free-running cycles while NPU is busy</td></tr>
  <tr><td>OP_COUNT</td><td>0x4000_002C</td><td>Total PE MAC operations (NUM_ROWS × NUM_COLS × cycles)</td></tr>
  <tr><td>STALL_COUNT</td><td>0x4000_0030</td><td>Weight-movement cycles (S_WLOAD + S_PRELOAD)</td></tr>
</table>

The three addresses in this table were each one word high; they now match the register map in section 12 and `sw/c930_npu_driver.h`, which were both already correct.

**TOPS calculation:**

```plaintext
TOPS = (2 × M × N × K) / (cycle_count × clock_period)


```

The factor of 2 accounts for multiply + accumulate per MAC.

### 14.5 Precision modes

<table>
  <tr><th>PREC</th><th>Mode</th><th>Input format</th><th>Output format</th><th>Notes</th></tr>
  <tr><td>0</td><td>INT8</td><td>Signed 8-bit, packed 4/word</td><td>INT32, 1/word</td><td>Default, fastest</td></tr>
  <tr><td>1</td><td>INT16</td><td>Signed 16-bit, packed 2/word</td><td>INT32, 1/word</td><td></td></tr>
  <tr><td>2</td><td>FP16</td><td>IEEE 754 half, packed 2/word</td><td>FP32, 1/word</td><td>CLA-accelerated</td></tr>
  <tr><td>3</td><td>BF16</td><td>Brain float16, packed 2/word</td><td>FP32, 1/word</td><td></td></tr>
  <tr><td>4</td><td>INT4</td><td>Signed 4-bit, packed 8/word</td><td>INT32, 1/word</td><td>Experimental</td></tr>
</table>

**FP16/BF16 accumulator** uses a combinational FP32 adder with CLA exponent comparator and subtractor. Fmax on ECP5: ~32 MHz, Artix-7: ~45 MHz.

### 14.6 What the grxcp backend needs from the NPU

The grxcp backend dispatches GEMMs through the NPU by:

1. **Allocating DDR buffers** for A, B, C at known addresses
2. **Loading A/B data** via DMA or direct DDR writes
3. **Writing CSRs** (DIM_M, DIM_N, DIM_K, A_BASE, B_BASE, C_BASE, PREC)
4. **Triggering** (write 1 to START)
5. **Polling** STATUS register until DONE bit is set
6. **Reading C results** from DDR

The NPU is a **blocking accelerator** — only one GEMM runs at a time. The grxcp backend must serialize GEMM dispatches through the NPU.

**For GEMMs larger than MAX_M × MAX_N × MAX_K:** the backend must tile externally — see §14.7 for the tiling algorithm and DDR buffer layout.

### 14.7 External tiling for large GEMMs

When a GEMM dimension exceeds the on-chip MAX parameter, the grxcp backend must split it into tiles that fit within the buffer limits. The NPU's internal tiling (§4.2) only handles dimensions ≤ MAX_M/MAX_N/MAX_K.

**Tiling rules:**

<table>
  <tr><th>Dimension</th><th>Max on-chip</th><th>Tiling strategy</th></tr>
  <tr><td>M</td><td>MAX_M</td><td>Split into chunks of ≤MAX_M rows</td></tr>
  <tr><td>N</td><td>MAX_N</td><td>Split into chunks of ≤MAX_N columns</td></tr>
  <tr><td>K</td><td>MAX_K</td><td>Split into chunks of ≤MAX_K, accumulate partial sums</td></tr>
</table>

**M-tiling** (M > MAX_M):

```plaintext
For each m_chunk in range(0, M, MAX_M):
    mc = min(MAX_M, M - m_chunk)
    Load A[m_chunk:m_chunk+mc, 0:K] into DDR at A_ADDR
    Configure DIM_M = mc, DIM_N = N, DIM_K = K
    Trigger NPU, wait for DONE
    Read C[0:mc, 0:N] from DDR at C_ADDR
    Store to output C[m_chunk:m_chunk+mc, 0:N]


```

**N-tiling** (N > MAX_N):

```plaintext
For each n_chunk in range(0, N, MAX_N):
    nc = min(MAX_N, N - n_chunk)
    Load A[0:M, 0:K] into DDR at A_ADDR
    Load B[0:K, n_chunk:n_chunk+nc] into DDR at B_ADDR
    Configure DIM_M = M, DIM_N = nc, DIM_K = K
    Trigger NPU, wait for DONE
    Read C[0:M, 0:nc] from DDR at C_ADDR
    Store to output C[0:M, n_chunk:n_chunk+nc]


```

**K-tiling** (K > MAX_K) — requires **partial-sum accumulation**:

```plaintext
For each k_chunk in range(0, K, MAX_K):
    kc = min(MAX_K, K - k_chunk)
    Load A[0:M, k_chunk:k_chunk+kc] into DDR at A_ADDR
    Load B[0:kc, 0:N] into DDR at B_ADDR
    if k_chunk == 0:
        Configure DIM_M = M, DIM_N = N, DIM_K = kc
        Trigger NPU, wait for DONE
        Read C[0:M, 0:N] (this is the partial sum)
    else:
        // Accumulate: C_partial += NPU(A_tile, B_tile)
        // The backend must add NPU output to existing C in DDR
        Configure NPU for this tile
        Trigger, wait, read C and ADD to running sum


```

**DDR buffer layout for external tiling:**

The backend must manage DDR addresses to avoid overwriting live data:

- Double-buffer A/B: while NPU processes tile N, load tile N+1
- For K-tiling accumulation: keep partial C in DDR, accumulate in-place
- Minimum DDR requirement: A tile + B tile + C result simultaneously

**Minimum DDR for one tile:**

```plaintext
A_tile = MAX_M × MAX_K × element_size
B_tile = MAX_K × MAX_N × element_size  
C_tile = MAX_M × MAX_N × 4  (always 32-bit output)
Total  = A_tile + B_tile + C_tile


```

With MAX_M=64, MAX_K=256, MAX_N=8, INT8:

- A_tile: 64 × 256 × 1 = 16 KB
- B_tile: 256 × 8 × 1 = 2 KB
- C_tile: 64 × 8 × 4 = 2 KB
- **Total: 20 KB** (fits in 64 KB DDR with room for firmware)
