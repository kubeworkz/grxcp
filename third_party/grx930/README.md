# `npu_dpi_shim` — vendored from the GRX930 team

`npu_dpi_shim.c` / `.h` are **not ours**. They arrive from the GRX930 team's
tree and are vendored here byte-for-byte so that updating them is a drop-in
copy. Every workaround lives in our adapter
(`src/backends/npu_c930/test_npu_c930_shim.cc`), never in these two files, so
that a re-import cannot silently drop a fix.

| | |
|---|---|
| Source | GRX930 tree, `sim/npu_dpi_shim.{c,h}` |
| First imported at | `817eb33` — "Address grxcp team feedback: context pointers, shim, doc fixes" |
| Current import | `3070806` — "Fix 5 defects in NPU DPI shim found by grxcp team" |
| Sizes | `.c` 10454 B, `.h` 5644 B |
| Builds | clean under `gcc -O1 -Wall -Wextra -Wpedantic` |

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

## The five defects found on import, and their fixes

All five were measured here at `817eb33`, reported upstream, and fixed at
`3070806`. Each fix has been re-measured against the imported copy rather than
accepted; the numbers below are from that re-run.

| # | Defect at `817eb33` | State now |
|---|---|---|
| 1 | `npu_dpi_run_gemm()` never wrote `CTRL.START` — returned a plausible cycle count and computed nothing | **Fixed.** 2×2×2 INT8 gives C = [19,22;43,50] and DONE |
| 2 | `STATUS.BUSY` never asserted | **Fixed.** BUSY high through cycle 241 of a 240-cycle 8×12×16 GEMM, DONE at 242 |
| 3 | Counters above `CYCLE_COUNT` written one word low | **Fixed.** `OP_COUNT` = 256 at `0x2c` for M=4 N=4 K=8; `0x28` reads 0 |
| 4 | GEMM path indexed `ddr[]` unchecked (ASAN global-buffer-overflow) | **Fixed.** A/B/C extents checked; `C_BASE=0xfff0` gives `STATUS=0x04` |
| 5 | `PREC` stored and ignored; 8×8 cycle model with `MAX_N=12` | **Fixed.** INT16 1×1×2 gives 768; 4×4 array, 160 cycles at M=8 N=8 K=16 |

Two questions went with the report, and both were answered from the RTL rather
than guessed at — one of them against our reading:

- **Is `CYCLE_COUNT` 64-bit?** No. We had guessed the gap at `0x28` was a high
  word and the fix was a re-index. It is neither: `ADDR_CYCLE_HI` exists as a
  localparam in `c930_npu_csr.sv` and appears in no case statement, so the RTL
  has a dead register there and the shim's header had simply omitted it.
  `c930_npu_core.sv:138` declares `logic [31:0] cycle_cnt`.
- **Does the RTL drive `STATUS.BUSY`?** Yes. `c930_npu_csr.sv:180` reads
  `{29'd0, i_error, done_latch, i_busy}`, and their Icarus run shows the core
  holding `i_busy` for 2171 cycles of a 4×4×8 GEMM. The register map was right
  and the model was wrong.

## What is still open

### `STATUS.DONE` is inferred, not latched

Not a slip — a modelling choice, and the one place the model still differs from
the map in a way that matters.

```c
build_status() = (error ? 4 : 0)
               | ((CYCLE_LO || OP_COUNT) && !busy ? 2 : 0)
               | (busy ? 1 : 0)
```

DONE therefore means "some counter is non-zero and I am not busy", which is
true of an idle device that ran *any* earlier GEMM. On hardware DONE is a latch
set at completion — that is the whole reason `npu_c930_gemm` reads it, and the
reason a device that ignored every write cannot fake it.

Measured, in one process, after one good GEMM:

- a `CTRL.START` the model refuses outright (`DIM_M = 0`) reads `STATUS = 0x02`
  — a completion flag for a launch that was never accepted;
- an out-of-window `START` reads `STATUS = 0x06` — ERROR and DONE together,
  which says it both failed and completed.

Their own test for the bounds check saw `0x04`, correctly, because it ran on a
freshly initialised shim where `CYCLE_LO` is still zero. The same START the
second time in a process reads `0x06`. That is worth naming as a test-design
point rather than a carelessness one: a state-dependent register cannot be
characterised from a clean slate alone.

Neither reading can reach a caller through us, and
`case_status_is_inferred_not_latched` gates the two reasons why:
`npu_c930_gemm` validates dimensions *before* touching a register, and it
checks ERROR *before* DONE. Both watched failing.

### The bounds checks are `uint32_t` arithmetic

Defect 4's fix computes `a_end = A_BASE + m*k` and compares against
`NPU_DDR_SIZE`. In `uint32_t` that sum wraps, so a base near 2³² passes a check
it should fail. Watched, safely, on the byte path:

```
guard: addr=0xFFFFFFFF -> addr+3 = 2, refused? NO
ddr[0..3] before: 0 0 0 0
ddr[0..3] after:  204 187 170 0
STATUS = 0x00 (ERROR should have been raised and was NOT)
```

A write through an address four billion bytes out of range landed on
`ddr[0..2]`, silently. The same arithmetic is in the GEMM path, where it
segfaults instead — ASAN, `C_BASE = 0xFFFFFFF0`, `1×8×1`:

```
c_end = 0xfffffff0 + 32 = 0x00000010, <= 65536? yes -- check PASSES
ERROR: AddressSanitizer: SEGV ... WRITE ... npu_dpi_shim.c:159
```

This is `cuda_mapping.md` 7.29 from the other side: a truncated host pointer is
not out of range, it is *arbitrary*, and an `end > SIZE` check in the same width
as the pointer cannot see the half that wraps. Our adapter's `in_window()` does
the arithmetic in 64-bit for exactly this reason, and keeps doing it even though
the shim now checks too — the duplication is the point.

## `npu_ddr_alloc.h` is NOT vendored

The GRX930 team also shipped a first-fit DDR allocator. It is not imported, and
the reasons are measured, not stylistic. (Their `sim/npu_ddr_alloc.h`, at
`1c279ab`.)

**1. The failure sentinel is a valid address.** `npu_ddr_alloc` returns `0` on
failure, and the documented window base is `0x0000`, so the first successful
allocation also returns `0`. Measured: first alloc → 0, out-of-memory → 0,
invalid alignment → 0. A caller cannot tell a buffer at the start of DDR from
no buffer at all, and the wrong branch aims the DMA at offset 0 — which is
where A usually is.

**2. Alignment padding is counted twice, and the arena grows.** The allocated
block's `total` includes the padding *and* the padding is inserted as its own
free block. Their own `npu_ddr_alloc_dump` prints the overlap:

```
alloc(10,4) -> 0    sum(total)=65536
alloc(10,4) -> 12   sum(total)=65538      <- window is 65536
  [2] 0x000c-0x0018  USED  size=10 total=12
  [3] 0x0016-0x10000 FREE  size=65514 total=65514
```

Block [2] claims through `0x18`; block [3] starts at `0x16`. After a free and a
coalesce the merged free block reads `0x000a-0x10002` — two bytes past the end
of a 64 KB window. Twelve padded allocations drift the arena to 65572.

**3. It will hand out a buffer that runs off the end.** Driven to the point of
harm rather than argued:

```
alloc(65526,4) -> offset 12, buffer is [12, 65538)
the DDR window is [0, 65536).  past the end by 2 bytes
```

**4. The "global" allocator is per-translation-unit.**
`static npu_ddr_alloc_t g_npu_ddr;` sits in the header, so every TU that
includes it gets a private copy. Two TUs, both calling `npu_ddr_init_global`
then `npu_ddr_malloc(64, 4)`, both received offset 0 for a live allocation.

**5. The header's own usage block does not compile.** It shows
`npu_ddr_alloc_init(0x0000, 0x10000)`, `npu_ddr_alloc(M * K, 4)` and
`npu_ddr_free(a)`; the declarations take three, three and two arguments, and
the global wrapper is spelled `npu_ddr_mfree`.

None of this is hard to fix and the shape of the allocator is right. But our
NPU allocator has to satisfy `grxMalloc`'s contract, not a firmware one —
alignment from the device property, an out-of-band failure signal, and
sanitizer redzones — so it will be written on our side against
`cuda_mapping.md` 7.28 rather than imported.

## The backend constants are not usable as constants

`npu_dpi_shim.h` now defines `NPU_DPI_BACKEND_EMULATION 0x10`,
`NPU_DPI_BACKEND_SIMULATION 0x11` and `NPU_DPI_BACKEND_SILICON 0x00`, with the
instruction to assign one to the device's backend field. Measured against our
code as it stood, every one of the three made things worse:

| assigned to `grxDeviceProp_t.backend` | `grx-smi --json` said | `backend_has_vm` |
|---|---|---|
| `NPU_DPI_BACKEND_EMULATION` (0x10) | `"silicon"` | no |
| `NPU_DPI_BACKEND_SIMULATION` (0x11) | `"silicon"` | no |
| `NPU_DPI_BACKEND_SILICON` (0x00) | `"simx"` | **yes** |

The first two are the exact fabrication the constants exist to prevent. The
third collides with `GRX_BACKEND_SIMX == 0` and additionally switches on
`backend_has_vm`, which would advertise managed memory on a device with no MMU.

Half of that was our bug and is fixed: `grx-smi`'s JSON arm was a chained
ternary ending in `: "silicon"`, so any unrecognised value printed as silicon.
It is a `switch` with a `return "unknown"` now, matching `grxcp::backend_name`,
which had been right all along — two divergent copies of one mapping is why one
of them could stay wrong. The remaining `0x00 → simx` is a value collision that
no defensive coding on our side can fix.

The mapping belongs in our code, against our enum, and that is where it will go
(`cuda_mapping.md` 7.28).
