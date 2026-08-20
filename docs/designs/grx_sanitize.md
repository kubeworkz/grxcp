# grx-sanitize v1 — device memory checking

**Status:** implemented; gated by the SANITIZE GATE in `ci/run_real.sh`.
**Companion:** [`grxcp_roadmap.md`](grxcp_roadmap.md) phase 2,
[`cuda_mapping.md`](cuda_mapping.md) §7.16.

CUDA's `compute-sanitizer memcheck` catches out-of-bounds and use-after-free
accesses from device code and names the source line. This is GRXCP's version of
that. It exists early on purpose: on a functional simulator the checking is
cheap, and every phase after this one is easier when a bad pointer produces a
report instead of a corrupted tensor three kernels later.

---

## 1. What it finds

| Finding | Meaning |
|---|---|
| `oob-global` | An address inside memory the GRXCP allocator owns, but inside no live allocation — a redzone, or a hole between allocations |
| `oob-straddle` | An access that starts inside an allocation and runs past its end |
| `use-after-free` | An access to an allocation that `grxFree` has already taken back |
| `oob-shared` | An access outside this CTA's shared-memory slot, `[CTA_LMEM_ADDR, +sharedMem)` |

Each finding reports the access (read or write, width in bytes), the address,
the allocation it relates to and by how much it misses, the block and thread
that made it, and **the source file, line and column**.

## 2. What it does not find

Stated plainly, because a sanitizer that is quiet about its blind spots is
worse than no sanitizer:

- **Accesses from uninstrumented code.** Only the kernel translation units
  compiled with `--sanitize` are checked. Device library code linked in from
  `libvortex2.a`, and anything the DXA engine copies, are invisible: DMA is
  performed by the engine, not by an instrumented instruction.
- **Wild pointers that land outside the allocator's regions.** The check
  applies to addresses inside a slab or direct buffer this runtime handed out.
  An address in the kernel image, the stack, or unmapped space is not judged,
  because this runtime does not know those bounds and inventing them would
  produce false findings.
- **Uninitialized shared memory.** Detecting a read of never-written shared
  memory needs a per-byte initialization shadow, which the outlined-callback
  design can carry but v1 does not implement.
- **Barrier divergence.** Not implemented. The failure mode today is a hang,
  and `tests/repro/tcu_multi_cta/` shows the shape a timeout-based watch would
  take.
- **Races.** No happens-before tracking of any kind.

The roadmap's phase 2 line named all four of the first list's items; two of
them ship here and two do not, and that is what this section is for.

## 3. How it works

### 3.1 The instrumentation

`ci/build_kernel.sh --sanitize` compiles the kernel with

```
-Xclang -fsanitize=kernel-address -mllvm -asan-instrumentation-with-call-threshold=0
```

The threshold of 0 is the whole design. AddressSanitizer normally inlines a
shadow-memory probe at every load and store, which would mean building and
uploading a shadow map covering an eighth of every address the kernel touches.
At threshold 0 clang outlines the check instead: every access becomes a call to
`__asan_load4(addr)`, `__asan_store8(addr)`, and so on. What those calls *do*
then becomes ours to define, and `src/device/grx_sanitize_rt.cpp` defines them
as a lookup in the allocation map — no shadow memory at all.

`kernel-address` rather than `address`: both emit the same callbacks, but the
hosted spelling also emits a module constructor calling `__asan_init`, aimed at
a runtime that does not exist on a device with no `main`. The `-Xclang` is
needed because the clang driver refuses either spelling for a bare-metal
riscv64 target — it cannot find `libclang_rt.asan-riscv64.a`, which does not
exist. That refusal is about the runtime, and the runtime is the part we
supply.

### 3.2 The map

The host half (`src/runtime/sanitize.cpp`) already knows everything the check
needs, because it is the allocator:

- Every allocation gets a **trailing redzone** of 256 bytes that belongs to no
  extent. Because every allocation has one, an overflow lands in a hole and so
  does an underflow — into the previous allocation's redzone.
- The extent registered for an allocation is its **requested** size, not the
  size the allocator rounded up to. Overflowing a 100-byte buffer by one byte
  is a finding even though 256 bytes were really handed out.
- `grxFree` **quarantines**: the extent stays in the map marked freed, and the
  memory goes back to neither the free list nor the driver. A sanitized run's
  peak device memory is therefore the sum of everything it ever allocated.
  That is the price of a use-after-free staying a use-after-free.

Before each launch the map is uploaded to a device-side control block, and the
launch is serialized behind it: one table, rewritten per launch, cannot be read
by a kernel still in flight.

### 3.3 Finding the control block

The block cannot live in the kernel image — the host has to rewrite it every
launch, and `vx_buffer_reserve` fails on a range the module loader already
owns. So the image holds one 8-byte **anchor** (`__grx_san_anchor`, forced into
`.data` so it is inside the `.vxbin` payload), and `grxModuleLoad` patches the
control block's device address into it at load time. The anchor's address comes
from the sibling `.elf` that `ci/build_kernel.sh` writes beside the `.vxbin`:
the VXSYMTAB footer carries kernel entry points only.

A module loaded with `grxModuleLoadData`, or one whose `.elf` is missing, has
no anchor to patch. That case is reported, not ignored — see §5.

### 3.4 Reporting without atomics

The report table is indexed by **grid-linear thread**: one slot per thread,
each thread writes its first finding and nothing after, and the host counts the
slots whose `kind` is non-zero.

This is not a stylistic choice. A finding counter needs an atomic increment,
and this device configuration is built with `VX_CFG_EXT_A_ENABLED` off — the
simulator's LSU calls `std::abort()` on any AMO instruction, silently, even
though the kernel's `-march=rv64imafd` tells the compiler atomics exist. The
first draft of this file counted findings with `__atomic_fetch_add` and killed
the simulator on the first planted bug. `cuda_mapping.md` §7.16 records the
mismatch; `GRX_CAP_GLOBAL_ATOMICS` reports it.

The per-thread table is better anyway: no contention, and the same run produces
the same report twice. Its cost is a slot per thread, capped at 1024; a larger
grid records nothing above the cap and the runtime prints a `coverage=partial`
status line.

## 4. Using it

```
# build the kernel with instrumentation
./ci/build_kernel.sh --grxgpu <path> --tooldir <path> --sanitize \
    tests/kernels/sanitize/kernel.cpp -o sanitize.vxbin

# run the program under the tool
grx-sanitize -- ./my_program sanitize.vxbin
```

`grx-sanitize` sets `GRX_SANITIZE=1`, passes the program's own output through,
and turns each finding into a located report:

```
=== grx-sanitize: oob-global write of 4 bytes ===
  at tests/kernels/sanitize/kernel.cpp:31:29  in san_oob_write
  address 0x3a240
  no live allocation covers it; the nearest below is #2 (64 bytes at 0x3a200),
  which ends 0 bytes before it
  kernel san_oob_write, block 0, thread 0 (warp 0 lane 0)
```

Exit code: the child's if it failed, otherwise 1 when there were findings and 0
when there were none — so it works as a CI gate directly.

The split of work follows host AddressSanitizer's. The runtime detects, because
only the runtime knows what was allocated and how big it really was. The tool
symbolizes, because only the build tree has the ELF and `llvm-symbolizer`. The
runtime's output is one machine-readable `GRXSAN|` line per finding carrying
raw facts only; every derived number is computed once, in the tool.

Setting `GRX_SANITIZE=1` without the tool works too — the findings are readable
as key/value lines, just without source locations.

## 5. When nothing was checked

A module built without `--sanitize` carries no instrumentation, so no access is
ever examined. The runtime prints

```
GRXSAN|status|kernel=<name>|module=<path>|instrumented=0
```

and `grx-sanitize` turns that into a warning and a **non-zero exit**. "No
findings" from an uninstrumented binary is not a clean bill of health, and the
tool refuses to let it look like one. The SANITIZE GATE checks this explicitly:
the same planted bug in an uninstrumented build must be reported as unchecked
rather than as clean.

## 6. Cost

Every load and store in an instrumented kernel becomes a call, and every launch
serializes behind a map upload. On a functional simulator that is the right
trade — it is why the roadmap puts this in phase 2 rather than after silicon.
It is not a mode to leave on: `--sanitize` is a separate build of the kernel,
and `GRX_SANITIZE` a separate run.

## 7. Where it goes next

- Uninitialized-shared-memory detection, via a per-byte initialization shadow
  for the CTA slot only — the slot is 16 KiB, so the shadow is 2 KiB.
- Barrier-divergence detection, via a timeout watch that reports which warps
  arrived at which barrier PC.
- Instrumenting grxBLAS's own kernels in a debug build, so a library bug is
  found the same way a user's is.
- A `grxSanitizerGetFindings`-style query, so a test can assert on findings
  without parsing stderr.
