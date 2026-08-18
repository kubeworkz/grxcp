# `src/runtime/` — the L1 runtime implementation

One concern per translation unit. Nothing here is public; the public surface
is `include/grx/`.

| File | Owns |
|---|---|
| `context.cpp` | device table, thread-local current device, `grxDeviceProp_t` population from `vx_device_query`, sticky error state |
| `memory.cpp` | the two-tier slab allocator, the address→`{vx_buffer_h, offset}` interval map, `grxMalloc`/`Free`/`Memcpy`/`Memset`/`PointerGetAttributes` |
| `stream.cpp` | `grxStream_t` over `vx_queue_h`, null-stream semantics, the host-side dependency graph that implements legacy-default-stream synchronization |
| `event.cpp` | `grxEvent_t` over `vx_event_h` timelines, record/query/synchronize, elapsed-time with the device-vs-host clock decision |
| `module.cpp` | `.grxfat` parsing, ISA-flag matching against `VX_CAPS_ISA_FLAGS`, `vx_module`/`vx_kernel` lifetime, the stub-address registry |
| `launch.cpp` | argument packing into the flat blob, `vx_launch_info_t` construction, `grxLaunchKernel`/`Ex`/`Cooperative` |
| `occupancy.cpp` | the three-bound resident-CTA formula (warps, slots, shared memory) |

## Invariants

1. **Every `vx_*` result is checked and mapped.** There is no path that
   discards a `vx_result_t`. The mapping table lives in `error.cpp` and is the
   only place a `vx_result_t` becomes a `grxError_t`.
2. **The interval map is the single source of truth for device pointers.** A
   freed extent is unmapped before its address can be handed out again;
   `grxPointerGetAttributes` and `grxMemcpy` direction resolution both read it.
3. **No allocation smaller than the device cache line shares a line** on
   backends where the command processor's DMA rounds transfers up to a 64-byte
   multiple. Remove this padding only when the upstream tail-`wstrb` fix lands.
4. **Nothing here knows about a specific backend.** Backend differences are
   discovered through `vx_device_query` and the backend enum, never through
   `#ifdef`.
