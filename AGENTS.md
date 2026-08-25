# AGENTS Guide for GRXCP Development

Canonical entry point for human contributors and AI coding agents working on
GRXCP. This file carries the *rules* — invariants, conventions, and footguns.
Recipes live in `docs/`.

---

## 1. Documentation map

- [docs/designs/grxcp_architecture.md](docs/designs/grxcp_architecture.md) — the specification
- [docs/designs/cuda_mapping.md](docs/designs/cuda_mapping.md) — concept mapping + gap register
- [docs/designs/grxcp_roadmap.md](docs/designs/grxcp_roadmap.md) — phases and exit gates

---

## 2. The boundary rules

- **GRXCP consumes the GRX-G100 sysroot, never its source tree.** Link
  through `pkg-config vortex-runtime vortex-kernel` against `$VORTEX_PATH`.
  Referencing `$VORTEX_HOME`, `$VORTEX_BUILD_DIR`, `grxgpu/hw/*`,
  `grxgpu/sim/*`, or any header not installed by `make install` is a build
  error, not a style preference. The GRX-G100 project states this contract
  explicitly in its own `AGENTS.md` §2 and README.
- **Never patch `grxgpu` from GRXCP.** If GRXCP needs a driver or hardware
  change, it goes in as a proposal to that repo (`grxgpu/docs/proposals/`).
  The dependency is one-directional and stays that way.
- **`include/grx/` is the public surface; `src/` is not.** Anything under
  `include/grx/` is an ABI commitment once released. Internal helpers live in
  `src/`, never in a public header.
- **Device headers and host headers are separate trees.**
  `include/grx/device/` is compiled by the device toolchain only and must not
  include host runtime headers, and vice versa. The only shared content is
  POD ABI structs, which live in `include/grx/grx_abi.h`.

---

## 3. The honesty rules

These are the rules most likely to be violated by a well-meaning change.

- **No fabricated capability.** An entry point either works or returns
  `grxErrorNotSupported`. Do not emulate a hardware feature behind an API
  that implies hardware — it creates performance cliffs the user cannot see
  and cannot measure.
- **Every sanctioned emulation is reported through a device property.** The
  warp-shuffle fallback sets `warpShuffleIsEmulated`; the host-clock event
  timing clears `eventTimingIsDeviceSide`. If you add a third, add its flag.
- **No invented device numbers.** Everything in `grxDeviceProp_t` comes from
  `vx_device_query` or from a formula documented in an upstream design doc.
  If you cannot source a field, report the "unknown" sentinel (-1), the way
  `numRegs` does until the compiler supplies it.
- **Publish the conformance pass rate.** It goes up over time; hiding it
  helps nobody. The chipStar work's published rv32 number is the precedent.
- **Gaps go in the gap register.** If you discover a CUDA behavior GRXCP
  cannot reproduce, add it to `cuda_mapping.md` §7 with a status and an
  owner. A gap that only exists in someone's head gets rediscovered
  expensively.

---

## 4. Correctness rules

- **Fix root causes, not symptoms.** Diagnose before patching.
- **Every library kernel has a CPU reference** and a numerical gate (bitwise
  or ULP-bounded). "Close enough" is not a gate.
- **Every conformance test runs on `simx` and `rtlsim`.** A result that
  differs between backends blocks — it is either a GRXCP bug or a GRX-G100
  model-parity bug, and both matter.
- **Perf baselines are golden data.** Never hand-edit a baseline to make a
  red gate green. A moved number means real cycles moved: root-cause it, or
  regenerate the baseline as a reviewed step so the delta is visible.
- **Allocator invariants are load-bearing.** The interval map must never
  return a stale mapping; a freed extent is unmapped before it is reusable.
  Add a test with every allocator change.

---

## 5. Coding conventions

- **Default to no comment.** Add one only when the *why* is non-obvious — a
  hidden constraint, a subtle invariant, a workaround for a specific bug.
- **Comments explain *why*, not *what*.** Well-named identifiers carry the
  *what*.
- **Never reference the current task, PR, issue, or caller in code.** Those
  belong in the commit message and rot in the source.
- **C API is C99-clean.** Public host headers compile as C and C++.
- **Error handling is total.** Every `vx_*` call's result is checked and
  mapped; there is no path that drops a `vx_result_t` on the floor.

---

## 6. Living document

Update this file when you find a footgun the next contributor will hit.
Include the *why*. Remove rules that have gone stale — a stale rule is worse
than no rule. Do not duplicate content from `docs/`; link to it.

**In prose, a `backticked/path` is a claim that the file is in this tree**, and
`ci/check_docs.py` opens every one of them on every tier-1 build. Write a path
that does not exist yet without backticks, the way section 10 item 4 of the
architecture document names its unbuilt perf-baseline directory; put
intended-layout trees in a fenced block, which the checker does not read. The
rule exists because six references rotted unnoticed through several careful
readings of these documents, and reading is not what catches that.
