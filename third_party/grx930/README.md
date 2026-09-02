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
| Current import | `e02f460` — "Fix STATUS.DONE latch, wrap-safe bounds checks, allocator bugs (grxcp round 2)" |
| Sizes | `.c` 11518 B, `.h` 5943 B |
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

## Round two: what reading the fixes turned up, and its fixes

Two things came out of measuring `3070806` rather than accepting it. Both are
fixed at `e02f460`, and both are recorded because the *shape* of them recurs.

### `STATUS.DONE` was inferred, not latched

```c
build_status() = ... | ((CYCLE_LO || OP_COUNT) && !busy ? 2 : 0) | ...
```

DONE meant "some counter is non-zero and I am not busy" — true of any idle
device that ran an earlier GEMM. On hardware DONE is a latch set at completion,
which is the whole reason `npu_c930_gemm` reads it: it is the one bit a device
that ignored every write cannot fake.

Measured then, in one process, after one good GEMM: a `CTRL.START` the model
refused outright (`DIM_M = 0`) read `STATUS = 0x02`, and an out-of-window START
read `0x06` — ERROR and DONE together, saying it both failed and completed.

Their own test for the bounds check saw the correct `0x04`, because it ran on a
freshly initialised shim where `CYCLE_LO` is still zero. That is the part worth
keeping: **a register whose value depends on prior state cannot be characterised
from a clean slate alone.** Every case in `case_done_is_latched` runs a good
GEMM first and only then asks the awkward question.

Fixed with a real `done_latch`, set when the cycle countdown reaches zero and
cleared on `CTRL.START`. Re-measured: `0x00` and `0x04`.

### The bounds checks were `uint32_t` arithmetic

`a_end = A_BASE + m*k` wraps, so a base near 2³² passed a check it should fail.
Watched safely on the byte path — a write at `0xFFFFFFFF`, four billion bytes
out of range, landed on `ddr[0..2]` with no ERROR — and under ASAN on the GEMM
path, where the same arithmetic segfaults at `npu_dpi_shim.c:159`.

Fixed to `base >= SIZE || size > SIZE - base`, which cannot wrap, plus an
`addr >= NPU_DDR_SIZE` guard in front of the byte path. Re-measured: the
`0xFFFFFFFF` write is refused with ERROR and DDR is untouched; the ASAN case
survives with `STATUS = 0x04`. (The inner `addr + highest` comparison still
wraps in principle, but the guard in front of it makes that unreachable.)

This is `cuda_mapping.md` 7.29 seen from the other side: a truncated 64-bit
pointer is not an out-of-range address, it is an **arbitrary** one, and a check
in the same width as the pointer cannot see the half that wraps. Our adapter's
`in_window()` does the arithmetic in 64-bit and keeps doing it even though the
shim now checks too — the guard that matters is the one in the process being
protected.

## `npu_ddr_alloc.h` — assessed, fixed upstream, still not vendored

The GRX930 team also shipped a first-fit DDR allocator. Six defects were
measured here against `1c279ab` and all six are fixed at `e02f460`:

| Defect at `1c279ab` | State at `e02f460` |
|---|---|
| Failure returned `0`, which is also the first valid offset | `NPU_DDR_ALLOC_FAILED` (`0xFFFFFFFF`) |
| Alignment padding counted twice, so the arena grew past 65536 | `total = size` on the allocated block; padding owns its own free block |
| Could hand out a buffer running past the end of DDR | wrap-safe `size - padding >= size` availability test |
| The "global" allocator was a `static` in a header, one copy per TU | removed; callers hold an `npu_ddr_alloc_t` |
| The header's usage block had the wrong arity | corrected |
| *(found later, in the same pass)* `padding + size > block->total` overflowed: `alloc(0xFFFFFFFF, 4)` against a padded free block returned offset 12 — 4 GB out of a 64 KB window | refused; the fit test is `size < padding \|\| size - padding < request` |

Re-measured rather than accepted. The targeted cases pass, and their own
`npu_ddr_alloc_dump` now prints `sum(total) = 65536 (OK)` where it used to print
the overlap. Beyond those, an invariant fuzz over **79,114 allocations across
3,000 random alloc/free trials** found nothing: blocks tile `[base, base+size)`
exactly with no gap or overlap, every returned offset is aligned and inside the
window, no two live allocations overlap, and draining every allocation restores
a single whole free block.

It is still not vendored, and now for one reason rather than five: **an NPU
allocator here has to satisfy `grxMalloc`'s contract, not a firmware one.**
Alignment comes from the device property rather than a caller argument, failure
has to arrive as a `grxError_t`, and `GRX_SANITIZE` expects a trailing redzone
on every allocation and quarantine rather than reuse on free. That is a
different object with the same shape. Theirs stays the reference for the shape,
and having it made ours obvious.

## The backend constants are not usable as constants

`npu_dpi_shim.h` defined `NPU_DPI_BACKEND_EMULATION 0x10`,
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

`NPU_DPI_BACKEND_SILICON` is dropped at `e02f460`, and the remaining two carry
a comment pointing the mapping back at us. That is the right resolution: what is
useful to publish is the *distinction* — software model, RTL-backed simulation,
silicon — while the mapping into `grxBackend_t` belongs in our code, next to the
seam that attaches the model (`cuda_mapping.md` 7.28). Their `SIMULATION` lands
on the existing `GRX_BACKEND_RTLSIM`; a software register model needs one new
value, appended.
