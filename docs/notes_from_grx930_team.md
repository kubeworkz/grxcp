## What We've Built

### Full SoC Architecture

```plaintext
CPU (RV64IMAC) ──┐
                 ├── AXI4 Crossbar (3M×4S) ──┬── Boot ROM (1KB, 0x0000)
NPU (8×8 systolic) ┘                          ├── DDR Controller + Stub (0x8000+)
                                              ├── UART 16550 (MMIO 0x40001xxx)
                                              └── NPU (MMIO 0x40000xxx)

```

### NPU Improvements (done in previous sessions)

- **CSR command queue drain fix** — back-to-back commands no longer strand in FIFO
- **Cross-GEMM prefetch (PF2)** — enabled for INT4 nibble-packed mode
- **Core timeout watchdog** — raises `o_error` if `i_core_done` doesn't fire within bounded cycles
- **DDR timeout watchdog** — detects no `rvalid` within N cycles of `arvalid`
- **DMA cycle counter CSR** — loadable debug register for performance profiling
- **Dual-buffered A/B memories** — bank switching for pipelined GEMM execution

### Full-SoC Test Suite (5 tests, all passing)

<table>
  <tr><th>Test</th><th>What it verifies</th></tr>
  <tr><td>1</td><td>D-cache LW/SW through crossbar → DDR</td></tr>
  <tr><td>2</td><td>NPU GEMM via CPU firmware (4×4×4)</td></tr>
  <tr><td>3</td><td>UART TX via CPU MMIO → crossbar → UART</td></tr>
  <tr><td>4</td><td>Mixed-precision queue drain (INT8+FP16+BF16) with all 21 C elements verified via CPU D-cache</td></tr>
  <tr><td>5</td><td>6-GEMM queue stress (fill 4-deep queue to capacity, drain twice)</td></tr>
</table>

### Vivado Synthesis (Arty A7-200T)

<table>
  <tr><th>Resource</th><th>Used</th><th>Available</th><th>Util%</th></tr>
  <tr><td>LUTs</td><td>78,750</td><td>134,600</td><td>58.5%</td></tr>
  <tr><td>FFs</td><td>27,097</td><td>269,200</td><td>10.1%</td></tr>
  <tr><td>BRAM</td><td>8</td><td>365</td><td>2.2%</td></tr>
  <tr><td>DSPs</td><td>148</td><td>740</td><td>20.0%</td></tr>
  <tr><td>Latches</td><td>0</td><td>—</td><td>✅</td></tr>
</table>

**41.5% LUT headroom** — enough for a second NPU tile or additional peripherals.

---

## Known Issues to Report

### 1. I-Cache Sensitivity in Icarus Verilog (simulation-only)

Icarus doesn't properly track sensitivity for unpacked array reads (`VALID_MEM[INDEX]`) in `assign` and `always_comb` blocks. After the first cache line fill, subsequent cache line misses go undetected. Workaround: keep firmware within a single 32-byte cache line. **The RTL is correct for synthesis** — this only affects simulation.

The `rv64imac` D-cache controller has a local fix (convert `assign tag_hit` to `always_comb` with mirror signals) that was committed but couldn't be pushed (no write access to the rv64imac repo).

### 2. Signed 12-bit Immediate Overflow in Firmware

`addi x11, x11, 0x800` sign-extends to **-2048**, not +2048. Any address with `lo12 >= 0x800` requires bumping `hi20` by 1. This caused **three test failures** during development. We added `sw/check_imm_encoding.py` to prevent recurrence — run it before any firmware changes.

### 3. DDR Stub TB-Preload Ports

The DDR synth stub needed `INIT_FILE` parameter and `i_tb_wr_*` ports added for synthesis. These are already committed.

---

## Recommendations for GRXCP

1. **Run **`python sw/check_imm_encoding.py` after any firmware changes to catch address encoding bugs
2. **The AXI crossbar** uses round-robin arbitration with combinational decode — timing may need attention at higher clock frequencies
3. **Queue depth is 4** — the firmware must poll `STATUS` and wait for queue space when queuing more than 4 GEMMs back-to-back
4. **The CPU boots from DDR[0x000]** — firmware must fit within the I-cache line constraint for Icarus simulation, or use a proper simulator (VCS/Xcelium) that handles array sensitivity correctly
