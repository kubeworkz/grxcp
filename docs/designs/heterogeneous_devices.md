# Heterogeneous devices: where the backend seam goes

**Companions:** [`grxcp_architecture.md`](grxcp_architecture.md) §6,
[`grxcp_roadmap.md`](grxcp_roadmap.md) phase 7.

**Status: DESIGN, with one part built.** `src/backends/npu_c930/` exists and
`src/runtime/context.cpp` appends an NPU to the device table when one is
detected — that is section 4 of this document, already done. What is *not* built
is section 3: the runtime still reaches the driver directly everywhere, and the
NPU's presence in the table is carried by `#ifdef` and a device-type branch
rather than by the seam described here.

This document exists because the roadmap says the seam should be written down
before it is written, and because the obvious answer is the expensive one. When
more of it lands, this file should be edited to say what was actually built and
what the first implementation got wrong.

---

## 1. The problem, sized

GRXCP's device table is a Vortex device table. `Device` holds a `vx_device_h`,
`acquire_device` calls `vx_device_open`, and `grxGetDeviceCount` returns
`vx_device_count()`. Nine files under `src/runtime/` call **33 distinct
`vx_*` entry points** between them:

| File | `vx_*` call sites |
|---|---|
| `memory.cpp` | 24 |
| `stream.cpp` | 11 |
| `module.cpp` | 11 |
| `event.cpp` | 11 |
| `sanitize.cpp` | 9 |
| `context.cpp` | 5 |
| `launch.cpp` | 2 |
| `tensormap.cpp`, `profile.cpp` | 1 each |

The GRX930 c930 NPU is not a Vortex device. It is a systolic array behind an
MMIO register block: program `DIM_M/N/K` and `A_BASE/B_BASE/C_BASE`, pulse
`CTRL.START`, wait on `STATUS.DONE` or the AIA MSI. There is no module to load,
no kernel to launch, no shared memory to carve, no tensor map, no performance
counter file.

So: how does a device that is not a `vx_device_h` get into that table?

---

## 2. The rejected answer: abstract the driver

The reflex is a backend vtable that covers the driver surface — replace every
`vx_foo(...)` with `d->ops->foo(...)`, give the GPU an implementation that
forwards, give the NPU one that does whatever it can.

This is wrong here, for two reasons.

**The NPU uses about a quarter of that surface.** Of the 33 calls, the NPU has
an analogue for roughly eight: open, close, query, allocate, free, copy in, copy
out, and submit-with-completion. The other twenty-five are modules, kernels,
launches, DCR writes, tensor maps and MPM counters — things the NPU does not
have and will never have. Building an abstraction whose majority of methods
exist only to return "not supported" on the second implementation is
scaffolding around an empty room. `grxdnn.h` makes the same argument about
cuDNN's descriptor machinery, and for the same reason: the apparatus should
arrive when there is something to describe.

**A vtable shaped like `vortex2.h` is not an abstraction, it is a disguise.**
There is exactly one driver today. An interface extracted from one
implementation reproduces that implementation's assumptions — buffer handles
with device addresses, queues with in-order semantics, DCR writes as the
configuration channel — and the second backend then has to pretend. Worse, it
*looks* general, so the pretending is invisible. A layer that lies about
generality is more expensive than no layer, because the next person believes it.

---

## 3. The seam goes at the capability profile

GRXCP already has the mechanism, and it now has teeth.

`grxDeviceProp_t::capabilities` is a `GRX_CAP_*` bitmask, and
`grxcp_architecture.md` §6 already fixes the c930's profile: `GRX_CAP_GEMM`,
`GRX_CAP_STREAMS`, `GRX_CAP_EVENTS`, `GRX_CAP_MEMCPY` — and **not**
`GRX_CAP_KERNEL_LAUNCH`. As of the phase-7 groundwork commit that bit is derived
from the device rather than assumed, and `validate()` refuses on it first, so a
launch on a pipeline-less device returns `grxErrorNotSupported` instead of
running somewhere else.

That is the load-bearing observation: **most of the 33 calls are unreachable on
an NPU by construction.** A path guarded by a capability the NPU does not have
cannot receive an NPU device, so it does not need to be abstracted — it needs to
be *guarded*, and the guard is a refusal that a test can watch.

The seam is therefore narrow by design:

```
struct DeviceOps {
  grxError_t (*open)      (Device&);
  grxError_t (*close)     (Device&);
  grxError_t (*properties)(Device&);              // fills prop, incl. capabilities
  grxError_t (*alloc)     (Device&, size_t, uint64_t* addr);
  grxError_t (*free)      (Device&, uint64_t addr);
  grxError_t (*copy_in)   (Device&, uint64_t dst, const void* src, size_t);
  grxError_t (*copy_out)  (Device&, void* dst, uint64_t src, size_t);
  grxError_t (*submit)    (Device&, const Job&, Completion*);
};
```

Eight entries, not thirty-three. `Job` is deliberately not "a kernel launch":
for the GPU it carries a `vx_kernel_h` and a grid, for the NPU it carries a GEMM
shape and three base addresses. It is a tagged union, and the tag is checked
against the device's capability profile before dispatch.

Everything else — `module.cpp`, `tensormap.cpp`, `sanitize.cpp`, the DCR path,
the MPM counters — keeps calling `vx_*` directly, unchanged, because those files
only ever run on a device whose profile says they can.

### What this costs

It is not free, and the cost should be stated rather than discovered.

- **`memory.cpp` and `stream.cpp` do get touched**, because both are reachable
  on an NPU. That is roughly 35 of the 33+ call sites, in two files.
- **Two allocators, one address space question.** Today a device pointer is a
  `vx_buffer_h` address. The NPU's DMA is an AXI4 master reading DDR, so its
  addresses are physical rather than driver-issued. `grxMemcpy` between a GPU
  buffer and an NPU buffer is a host round trip until the c930 gains its
  coherent port — and `grxDeviceEnablePeerAccess` already returns
  `grxErrorNotSupported`, which is the right answer and stays the right answer.
- **`grxMalloc` becomes device-scoped in a way it currently is not.** A pointer
  allocated on device 0 and passed to a call on device 1 is already undefined in
  CUDA; here it must be *caught*, because the two devices' address spaces are
  genuinely unrelated and a silent wrong-device pointer would read as garbage
  data rather than as an error.

---

## 4. Enumeration

`grxGetDeviceCount` returns `vx_device_count()`. It becomes the sum of what each
registered backend enumerates, with the GPU backend first so that existing
programs keep the device indices they have today.

Backend registration is static, not dynamic: a build either has the NPU backend
compiled in (`-DGRXCP_ENABLE_NPU=ON`, which builds and is gated both ways in
tier 1) or it does not. There is no plugin
loading, because there is no third-party backend and inventing a plugin ABI for
one in-tree implementation is the same empty room as before.

**An NPU that is not present must not be enumerated.** The backend probes for
the register block and reports zero devices if it is absent. A build flag is a
statement about what code exists, not about what hardware is attached, and
conflating the two is how a device appears in `grx-smi` that nobody can talk to.

### 4.1 Every device has its own address space, and they overlap

A device address is meaningful only together with the device it came from. Each
device's addresses come from its own `vx_buffer_address` over its own DDR, so
two devices routinely hand out the **same** address — both spaces start near the
same base — and a bare `void*` therefore does not identify an allocation.

The runtime did not know that. Its interval map was keyed by address alone with
the owning device as a payload field, and its free-list search had no device
filter at all, so `grxMalloc` on device 1 returned a slice of **device 0's**
slab labelled as device 1's. Nothing reported a problem and device 1's
`grxMemGetInfo` read zero bytes in use. The map is keyed by `(device, address)`
now and the free list by `(slab, address)`.

None of it was findable while the mock returned one `MockDevice` for every
index, which it did from the first commit. A fixture that cannot represent the
thing being promised will report the promise as kept.

**The rule, for anything added at this seam:** resolve a pointer against the
current device, refuse it when it is live only on another, and do not try to
guess when it is live on both — that case is genuinely ambiguous and "this
pointer, on this device" is the only reading with a defence. The NPU makes this
sharper rather than softer: its addresses are physical DDR and the GPU's come
from the driver, so the two spaces have no reason to be disjoint either.

---

## 5. What gets gated, and how it can fail

No NPU exists in any environment GRXCP is developed in today, so most of this
cannot be gated yet, and that limit should be named rather than worked around.

Gateable without a c930, and done:

- A device with no `GRX_CAP_KERNEL_LAUNCH` refuses launches —
  `tests/unit/test_no_kernel_launch.cpp`, run both ways, ablation watched.
- Enumeration with the backend compiled in and no hardware present reports zero
  NPU devices, and `grx-smi` shows one device rather than two — CMAKE GATE in
  `ci/build_mock.sh`.
- The backend's own decisions — detection against an empty bus, completion
  against a device that accepts a launch and never finishes —
  `src/backends/npu_c930/test_npu_c930_model.cc`, driven through injectable
  register hooks. Both defects it covers were found this way.

Gateable without a c930, and not done:

- A device with no `GRX_CAP_KERNEL_LAUNCH` refuses `grxModuleLoad` for the same
  reason: there is nowhere to load it to.
- A pointer from device A used on device B is refused rather than dereferenced.

Not gateable without a c930, and it should say so where the claim is made:

- Anything that writes `CTRL.START` and waits on `STATUS.DONE` on real silicon.
- `grxblasGemmEx` producing the same INT8 result on both devices — the phase 7
  exit gate, which needs both devices to exist at once.

The register model in `test_npu_c930_model.cc` is the bridge, and it is worth
having **only because it is labelled as loudly as `tests/mock/` is**. It lets the
detection, launch and completion logic run end to end in CI. It proves nothing
whatsoever about the register map, the DMA, or the silicon, and a green run
against it must never be reported as the NPU working. `grx-smi`'s "software
stand-ins in effect" section is the precedent for how that gets said out loud.

---

## 6. Open questions

1. **Does the NPU get a stream, or is it synchronous?** §6 gives it
   `GRX_CAP_STREAMS` and `GRX_CAP_EVENTS` via the MMIO doorbell and
   `STATUS.DONE`. One doorbell and one status bit is one in-flight job, so a
   "stream" on the NPU is a queue the host drains rather than hardware
   concurrency. That is a legitimate implementation of the API and a misleading
   one to benchmark against — it should be reported through a device property,
   the way event timing already reports that it is a host clock.
2. **Which engine does `grxblasGemmEx` pick when both can do the work?** §6 says
   explicit control, not automatic magic. The default still has to be written
   down: current-device-only is the honest one, since it makes the choice the
   caller's and matches `grxSetDevice` semantics.
3. **Does `grxDNN` follow?** Layer norm on a systolic array is not a GEMM, so
   the answer is probably no, and a transformer layer would then bounce between
   devices through host memory. That may make the NPU the wrong engine for
   whole-model inference and the right one for offloaded GEMM — worth measuring
   before it is designed around.
