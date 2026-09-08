# c930 NPU command queue: back-to-back commands stranded — FIXED, verified here

Drives `c930_npu_top` under Verilator with a C++ AXI memory model and submits
several GEMMs without waiting for each to finish, which is what the queue in
`c930_npu_csr.sv` exists to allow:

> Writing CTRL.START pushes the current CSR snapshot into a FIFO. If the engine
> is busy, the command waits in the queue and dispatches automatically when the
> previous GEMM completes. **The CPU never blocks.**

```
verilator --cc --exe --build -Wno-fatal --top-module c930_npu_top \
  -GNUM_ROWS=8 -GNUM_COLS=8 -GMAX_M=8 -GMAX_K=16 -GMAX_N=12 \
  <c930>/rtl/*.sv b2b.cc
./obj_dir/b2b <ops> <gap-cycles> <sequential:0|1>
```

`./b2b 4 0 1` is the CONTROL: one command at a time, waiting for DONE. It must
pass, and it does — all four results match a host reference. That is what makes
the pipelined failure attributable to the queue rather than to this harness.

`./b2b 2 0 0` is the defect: two commands submitted back to back leave one in
the queue forever.

## Measured, at the SoC's own parameters

| | sequential (control) | 2 pipelined | 3 | 4 (fills the queue) |
|---|---|---|---|---|
| `950b721^` (pre-fix) | 1190 cycles, all correct | **deadlock**, occupancy 1, 1600060 cycles | ERROR, `QUEUE_STATUS=0xc` | — |
| `72b2e9e` (current HEAD) | 1190 cycles, all correct | **drains, 508 cycles, correct** | 696 cycles, correct | 1120 cycles, correct |

**The fix is real and this is our own run of it**, not a reading of the commit
message. `950b721` ("fix command queue drain, add staging buffer, watchdog
timers, and diagnostics") adds the `D_IDLE` branch that was missing — engine
idle, FIFO non-empty, nothing dispatching — which is the state this harness
sat in for 1.6M cycles. The control is unchanged at 1190 cycles on both
builds, so the harness did not move under us.

`QUEUE_STATUS=0xc` on the old build is gone: occupancy now reads 0,1,2,3 as
commands are pushed and returns to 0 when drained.

### What the queue is worth, measured here and bounded below

Four GEMMs sequential is 1190 cycles; the same four queued is 1120. That is
**70 cycles over four commands** — the inter-command dispatch bubble, and
nothing more. Do not read it as the value of batching on a real host: this
harness polls CSRs every cycle with no bus latency, so it is the *floor*. What
a host pays per round trip over MMIO is the part that batching actually
removes, and that is not measurable from here.

## Why the completion signal is the queue and not DONE

`STATUS.DONE` is a latch cleared only by writing CTRL with bit 0 set — that is,
by the *next* START. It cannot be polled to count completions. The first
version of this harness did exactly that and reported four completions four
cycles apart, which is how the mistake surfaced. This one waits for
`QUEUE_STATUS` occupancy to reach zero with BUSY clear, which needs no
assumption about DONE.

## Why `tb_csr_queue.sv` passes while this fails

That testbench instantiates `c930_npu_csr` **alone** — no core, no DMA, no top —
and drives `i_busy` / `i_done` by hand, with busy rising exactly one cycle after
start. Against the real `c930_npu_core` the timing differs and the queue
strands. A unit test of the CSR block against a synthetic engine cannot see
this, which is the whole reason this harness drives the integrated design.
