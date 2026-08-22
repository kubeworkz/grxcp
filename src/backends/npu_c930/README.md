# GRX930 NPU Backend

**Status:** Implementation complete, unit tests passing.
**Target:** grxcp Phase 7 — NPU as a second device.

## What this is

A standalone MMIO driver for the C930's INT8 systolic-array GEMM accelerator.
It programs the NPU's registers over AXI4-Lite and dispatches GEMMs
autonomously — the NPU's DMA fetches A/B from DDR, runs the GEMM,
and writes C back.

**No dependency on `vortex2.h`.** This is a pure MMIO driver that works
on bare-metal, Linux, or simulation.

## Files

| File | Purpose |
|------|---------|
| `npu_c930.h` | Public header: register map, capability flags, device handle |
| `npu_c930.cpp` | Implementation: MMIO access, detect, GEMM dispatch, wait |
| `test_npu_c930.cc` | Unit tests (register layout, validation, data format, numerics) |
| `CMakeLists.txt` | Build integration |

## How to use it

### Standalone (bare-metal or Linux)

```c
#include "npu_c930.h"

npu_c930_device_t dev;
if (npu_c930_detect(&dev)) {
    // NPU present — program a GEMM
    uint32_t a_addr = 0x1000;  // A in DDR
    uint32_t b_addr = 0x2000;  // B in DDR
    uint32_t c_addr = 0x3000;  // C in DDR
    int rc = npu_c930_gemm(&dev, M, N, K, a_addr, b_addr, c_addr);
    // C is now valid at c_addr
}
```

### Integration with grxblasGemmEx

The NPU hooks into `grxblasGemmEx` when the current device is an NPU.
The dispatch path:

```
grxblasGemmEx(handle, ..., Atype=GRX_R_8I, Btype=GRX_R_8I, Ctype=GRX_R_32I)
  │
  ├── if current device is GPU (vortex2.h):
  │     └── existing path: tensor map descriptors → igemm_tcu kernel
  │
  └── if current device is NPU (npu_c930.h):
        ├── allocate A/B/C buffers in DDR (physical addresses)
        ├── memcpy A, B to DDR buffers
        ├── npu_c930_gemm(&npu_dev, M, N, K, a_phys, b_phys, c_phys)
        └── memcpy C from DDR buffer
```

### Register map reference

Base: `0x4000_0000` (from `c930_soc_top.sv`)

| Offset | Name | Access | Bits | Description |
|--------|------|--------|------|-------------|
| `0x00` | CTRL | W | [0]=START | Write 1 to launch. Clears DONE/ERROR. |
| `0x04` | STATUS | R | [0]=BUSY, [1]=DONE, [2]=ERROR | Read to check state. |
| `0x08` | DIM_M | R/W | [15:0] | Output rows (1–8). |
| `0x0C` | DIM_N | R/W | [15:0] | Output cols (1–12). |
| `0x10` | DIM_K | R/W | [15:0] | Reduction length (1–16). |
| `0x14` | A_BASE | R/W | [31:0] | Byte address of A in DDR. |
| `0x18` | B_BASE | R/W | [31:0] | Byte address of B in DDR. |
| `0x1C` | C_BASE | R/W | [31:0] | Byte address of C in DDR. |

### Data format

- **A** (M×K): INT8, row-major, packed 4 per 32-bit word, little-endian
- **B** (K×N): INT8, row-major, packed 4 per 32-bit word, little-endian
- **C** (M×N): INT32, row-major, one word per element

All base addresses must be 4-byte aligned.

## What grxcp Phase 7 needs to build

See `c930/doc/c930_architecture.md` §10 for the full integration guide.
The short version:

1. **Device enumeration:** probe `STATUS` at `0x4000_0004` — non-zero = NPU present
2. **Capability profile:** `GRX_CAP_STREAMS | GRX_CAP_MEMCPY | GRX_CAP_GEMM`
3. **Memory management:** allocate A/B/C in DDR, translate host pointers to physical addresses
4. **GEMM dispatch:** program registers, write CTRL.START, poll STATUS.DONE
5. **Streaming:** NPU has no hardware streams — program on calling stream, record event after DONE

## Running the tests

```bash
g++ -std=c++17 -O2 test_npu_c930.cc npu_c930.cpp -o test_npu_c930
./test_npu_c930
```

Expected output:
```
=== GRX930 NPU Backend Test Suite ===

[TEST] Register layout...
[PASS] Register layout verified
[TEST] Input validation...
[PASS] Input validation verified
[TEST] Data format (INT8 packing)...
[PASS] Data format verified
[TEST] Tiling parameters...
[PASS] Tiling parameters verified
[TEST] GEMM reference (small INT8)...
[PASS] GEMM reference verified
[TEST] Negative INT8 values...
[PASS] Negative INT8 values verified

=== Results: ALL PASSED ===
```
