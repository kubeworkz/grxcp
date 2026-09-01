# `npu_dpi_shim` — vendored from the GRX930 team

`npu_dpi_shim.c` / `.h` are **not ours**. They arrive from the GRX930 team's
tree and are vendored here byte-for-byte so that updating them is a drop-in
copy. Every workaround lives in our adapter
(`src/backends/npu_c930/test_npu_c930_shim.cc`), never in these two files, so
that a re-import cannot silently drop a fix.

| | |
|---|---|
| Source | GRX930 tree, `sim/npu_dpi_shim.{c,h}` |
| Imported at | commit `817eb33` — "Address grxcp team feedback: context pointers, shim, doc fixes" |
| Sizes | `.c` 5607 B, `.h` 3164 B |
| Builds | clean under `gcc -O1 -Wall -Wextra`; `libnpu_dpi_shim.a` = 5516 B |

## What it is, and what it is not

It is a **fifth register model** for the NPU backend, alongside the four in
`test_npu_c930_model.cc` (ABSENT, DEAD, LIVE, WEDGED). Those four are ours and
were written to produce a specific decision. This one comes from the team that
owns the RTL, carries their register addresses, and computes a real INT8 GEMM
in software — so it exercises the whole `npu_c930_gemm` sequence rather than
just the branch a hand-written stub was built to hit.

**It is not hardware.** Its own header says so ("NOT cycle-accurate"), and it
computes the GEMM with a C triple loop. Per `AGENTS.md`, no green run through
it may be reported as the NPU working, and the phase 7 exit gate still needs a
real c930. What it can tell us is whether *our host code* drives the register
map correctly — which is the half that has been wrong twice.

## Defects measured on import

All five reproduced against the imported copy. Nothing below is inferred from
reading the source; each has a run behind it.

1. **`npu_dpi_run_gemm()` never starts the GEMM.** It writes DIM_M/N/K,
   A/B/C_BASE and PREC, then calls `npu_dpi_run()` — but never writes
   `NPU_CSR_CTRL`. `npu_dpi_run()` returns immediately because `npu_busy` is
   still 0. Measured: the call returns `28` (a plausible cycle count), STATUS
   stays `0x00000000`, and C stays zero where the host reference says 35. The
   only entry point that *looks* like an API returns a number and computes
   nothing. **We do not call it.** Our adapter drives the CSRs directly.

2. **`STATUS.BUSY` (bit 0) is never asserted.** The register map documents
   bit0 = BUSY and `npu_c930_wait_idle()` polls it. Against this model, STATUS
   reads `0x00000000` immediately after `CTRL.START`, so a BUSY-waiting host
   cannot tell running from finished. Only `STATUS.DONE` distinguishes them
   here — which our backend does check, and which is the *only* reason wiring
   this in is safe.

3. **The performance counters above `CYCLE_COUNT` are one word low.** The
   header's addresses and the implementation's indices disagree:

   | register | header address | written at | reading the header address gives |
   |---|---|---|---|
   | `CYCLE_COUNT` | `0x24` (idx 9) | idx 9 | correct |
   | `OP_COUNT` | `0x2c` (idx 11) | idx 10 (`0x28`) | `STALL_COUNT` |
   | `STALL_COUNT` | `0x30` (idx 12) | idx 11 (`0x2c`) | `DMA_CT` |
   | `DMA_CT` | `0x34` (idx 13) | idx 12 (`0x30`) | zero, always |

   Measured at M=4 N=4 K=8: `0x24`=34, `0x28`=256 (`= M*N*K*2`, the OP_COUNT),
   `0x2c`=0, `0x30`=34, `0x34`=0. The header's central claim — *"STATUS,
   CYCLE_COUNT, OP_COUNT and STALL_COUNT update correctly"* — holds for one of
   the four. The gap at `0x28` in the header suggests `CYCLE_COUNT` is meant to
   be 64-bit (lo at `0x24`, hi at `0x28`) and the implementation lost the high
   word, which would make the fix `csr[10]/[11]/[12]` → `csr[11]/[12]/[13]`.
   Our backend reads no counters, so this blocks nothing today; it is not
   gated here because a test that goes red on a vendored defect turns our
   suite amber for someone else's bug. It is reported instead, and the gate
   arrives with the fix.

4. **The GEMM path indexes `ddr[]` with no bounds check.** `npu_dpi_mem_write`
   and `npu_dpi_mem_read` both range-check; the DMA emulation inside
   `npu_dpi_csr_write`'s START branch does not. Watched failing under ASAN with
   C_BASE = `0xfff0` and a 4x4 INT32 result:

   ```
   ERROR: AddressSanitizer: global-buffer-overflow ... WRITE of size 1
       #0 npu_dpi_csr_write npu_dpi_shim.c:68
   0x... is located 0 bytes after global variable 'ddr' ... of size 65536
   ```

   This matters more than it looks: `grxblasGemmEx` passes **host pointers** as
   A/B/C_BASE (`(uint32_t)(uintptr_t)A`). A measured `malloc()` landed at
   `0x69ed12b0` — four orders of magnitude outside a window that ends at
   `0xffff`. Handing this model a real allocation is an out-of-bounds write of
   megabytes. Our adapter therefore **refuses a launch whose A/B/C extents
   leave the window**, the way an address decoder would, and raises
   `STATUS.ERROR` instead of forwarding `CTRL.START`.

5. **`PREC` is stored and ignored; the cycle model is not the SoC's.** All four
   precision modes produce the same INT8 result (measured: C[0][0] = 8 for
   PREC 0..3 with all-ones inputs at K=8). The cycle formula tiles 8x8 with
   `MAX_N = 12` hardcoded, while the SoC default the team just corrected is a
   4x4 array (`NPU_NUM_COLS` 8→4 in this same commit). At M=8 N=8 K=16 the shim
   says 36 cycles where the same formula shape over a 4x4 array says 160 —
   4.4x apart. `CYCLE_COUNT` from this model is not the SoC's cycle count and
   must not be quoted as one.

Two smaller notes: `STATUS` is host-writable (a write of 0 clears a latched
DONE), and a START with M=0 returns early without raising `STATUS.ERROR` and
without even storing CTRL — the model goes quiet rather than refusing.

## What it does do correctly

With the CSRs driven directly and model time advanced past the expected cycle
count, the full `npu_c930_gemm` sequence completes and every element of C
matches a host `INT8 x INT8 -> INT32` reference. `npu_c930_detect()`'s
write-readback probe passes against it (STATUS reads 0, DIM_M stores and
returns both probe patterns), so it is correctly seen as present.
