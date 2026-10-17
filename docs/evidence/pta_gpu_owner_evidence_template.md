# PTA GPU-owner evidence packet — fill-in template

**Status: TEMPLATE — not evidence.** A checkbox stays open until the named owner attaches the raw artifact and records its provenance. An owner-approved design target, analytical model, or passing twin test does not prove function-IP capability, OS allocation, physical routing, queue behavior, or measured UCIe performance.

Copy this file for each implementation/configuration under review. Store raw logs, configuration exports, waveforms and test sources alongside the copy; cite paths and SHA-256. Redact credentials and unrelated host identifiers while preserving device IDs, IP/configuration revision, test parameters, timestamps and reproducibility details. Never overwrite an earlier packet: supersede it with a new ID and retain both.

## Run identity and provenance

| Field | Value |
|---|---|
| Packet ID / date | |
| GPU owner / EIC-chiplet owner / board owner | |
| GPU part, stepping, PCI BDF | |
| GPU RTL/IP name, release and configuration hash | |
| EIC RTL/IP name, release and configuration hash | |
| UCIe PHY/adapter part, version, mode and configuration hash | |
| Board/package revision and link topology | |
| Host OS/kernel, firmware, driver commit/build ID | |
| Test source commit, command, environment and test equipment | |
| Artifact index (paths + SHA-256) | |
| Deviations, waivers, unresolved items and decision owners | |

## Project-owner-approved recommendations

The user approved all recommendations in the GPU-owner checklist. These are project-level design targets and evidence/acceptance criteria—not implementation proof, technical-owner sign-off, or closure of any evidence gate.

| Area | Approved target | Evidence/decision still required |
|---|---|---|
| Q1 | Two MSI-X entries: vector 0 CP completion, vector 1 aggregate PTA causes; no silent remapping. | Function capability, host allocation and routing tests; explicit fallback decision if fewer than two. |
| Q2–Q4 | Queue depth and BUSY semantics must be owner-specified; full behavior must prevent silent loss; completion predicate must be tested. | GPU/EIC contract and boundary, full-queue and calibration tests with observed status. |
| Q5 | Retain UCIe FLIT with adapter CRC/retry; tensor traffic uses electrical mainband to EIC/PIC, optics stay within EIC/PIC. | Endpoint configuration/training and fault evidence; specified and tested GPU BAR-to-EIC forwarding route. |
| CP ring | Adopt the lossless monotonic-head/backpressure contract, with ordered chunking or all-or-nothing rejection for oversized batches. | Runtime/RTL implementation and exact-full, wrap, delayed-acceptance, concurrency and progress tests. |
| UCIe data window | Size from measured throughput, RTT, transaction distribution and actual credits/buffers; 57.6 GB/s and 100 ns remain assumptions. | Reproducible per-direction measurements and validated buffer/credit fit. |

### Source-audit finding: CP ring (not a passing behavior test)

At GRX-G100 HEAD `f7e7fdaa9979fa1a5594fedae27ab09b26967bb9`, source inspection records:

| Source | Finding | SHA-256 |
|---|---|---|
| `sw/runtime/common/device.cpp` | Allocates/programs `cp_head_`; `cp_ring_append_` copies and increments tail/sequence without reading head or checking free capacity. | `46ffa94c6f09d463e472c9e600d89c4c77c3d8b6761f67b7ea06c03dd5247220` |
| `hw/rtl/cp/VX_cp_fetch.sv` | Advances `head_r` by 64 after a fetched line is exhausted; `state_in.head_addr` is marked unused. | `97bd2f719d487a4e7c111252c73ba84edcc67d6a092eb12ede65e9267ae5b4e0` |
| `hw/rtl/cp/VX_cp_axil_regfile.sv` | `head_addr` is programmable and passed into queue state; this alone does not publish the head. | `2ccb9da32ab45a4d1aa4ef4fb91622ba987871a67c0604015d780fbcf76072e4` |
| `hw/rtl/cp/VX_cp_core.sv` | Routes fetcher's internal head telemetry to the regfile; no evidence here of AXI writeback to the host head buffer. | `ab176beafb16b382ea8d2dcd1cbe1abe42f2eb75a2149323ef41a7f7cca2e1b3` |

Conclusion: the approved monotonic consumer-head/backpressure contract is **not implemented at this revision**. This is a source audit, not a build or simulation result. It closes the question of current source support only; do not mark the CP-ring implementation/test gates PASS. The GPU runtime/RTL owners must implement coherent, observable head publication and producer-side capacity handling, then attach exact-full, wrap, oversized-batch, delayed-acceptance, concurrency and forward-progress test artifacts.

Approval adopts the recommendations; it does not mark any evidence checkbox PASS or resolve technical questions reserved to GPU/EIC owners.

## Q1 — MSI-X capability, host allocation and routing

Approved design target: two entries; vector 0 CP completion, vector 1 aggregate PTA. This is provisional until the evidence below passes. CP completion remains pending if the CP `irq` has no host ISR.

### Function capability

- [ ] Attach PCI configuration-space dump covering endpoint identity and MSI-X capability, plus vendor/IP configuration export.
- [ ] Record PCI BDF, vendor/device IDs, capability offset, raw Message Control value, Table Size field, decoded entry count (`field + 1`), IP revision and configuration hash.
- [ ] Pass only if selected function/IP exposes at least two entries. Fewer requires a separately approved fallback decision; do not remap silently.

Artifact path(s) / SHA-256:

### Host allocation

- [ ] Attach driver source/build ID and runtime log showing requested vector count, allocation API/result, count granted, Linux IRQ mapping to vector index/use, flags and return code.
- [ ] Record OS/kernel, firmware, PCI BDF and any fallback request/approval.
- [ ] Pass two-vector baseline only when the driver receives and configures two vectors. Fewer is BLOCKED pending fallback decision.

Artifact path(s) / SHA-256:

### Interrupt behavior

- [ ] Attach test source/command and simulation or board log/waveform for CP completion on vector 0 and each PTA cause (`CAL_DONE`, `ERR`, `DRIFT_ALARM`, `SAT_THRESHOLD`) on vector 1.
- [ ] Show `PTA_IRQ_STATUS` before/after service, `PTA_IRQ_MASK` enabled/masked cases, RW1C clearing, handler/vector identity, and no lost, duplicate or misattributed causes.
- [ ] If CP ISR wiring is absent, mark CP routing BLOCKED/PENDING; an internal pulse or twin result is not proof of host delivery.

Artifact path(s) / SHA-256:

## Q2 — PTA queue depth and active-command accounting

- [ ] Attach GPU/EIC interface contract, owner-approved RTL/IP configuration, and reference defining enqueue/accept, active, completion, and dequeue/retire events.
- [ ] State capacity in commands and whether the executing command consumes a slot. Separately name GPU CP-ring storage, GPU dispatch queue, EIC/chiplet queue, and active execution; do not combine these capacities.
- [ ] Attach a test/log/waveform filling to declared capacity and observing accepted, queued, active and retired counts at idle and during execution.
- [ ] Pass only when documentation matches the implementation and independent observations agree at boundaries. Otherwise remain BLOCKED for the queue owner.

Artifact path(s) / SHA-256:

## Q3 — Full-queue behavior and losslessness

- [ ] Specify full handshake/response: backpressure (signal and release condition) or explicit rejection/error (status and recovery). Identify the interface and queue that is full.
- [ ] Attach a test that stalls consumption, fills exactly to capacity, submits additional commands, resumes consumption, and checks command IDs/payloads in order.
- [ ] Prove no overwrite, silent drop, duplicate acceptance, ambiguous retry or false completion. For rejection, prove safe retry without duplicates; for backpressure, prove stable request until accepted and eventual progress.
- [ ] Test reset/calibration arrival behavior where applicable and attach evidence for the specified response.

Artifact path(s) / SHA-256:

## Q4 — `PTA_STATUS.BUSY` and completion predicate

- [ ] Provide normative state/timing table for BUSY during accepted-but-queued, active, completed-but-not-retired, retired, calibration, reset-refusal and error states. Identify assertion and clearing events.
- [ ] Attach timestamped queue occupancy, accept/retire, BUSY, CAL_BUSY and status-read waveform/log; include full-queue and calibration transitions.
- [ ] Exercise exact completion predicate: GPU-side relevant queue empty **and one** `PTA_STATUS` read with BUSY and CAL_BUSY clear. Prove queued, active or calibration work cannot cause false-complete under documented ordering.
- [ ] Identify clock/reset domains and coherent-snapshot/CDC mechanism; attach CDC or protocol evidence as applicable.

Artifact path(s) / SHA-256:

## Q5 — CRC-on protocol, tensor path and BAR-to-EIC register forwarding

CRC-on FLIT + adapter retry and GPU→EIC→PIC are the recorded board-plan decisions. Verify the selected implementation and close the forwarding detail; do not treat the design decision as implementation evidence.

### Normative implementation posture

- [ ] GPU owner and EIC/chiplet owner each record explicit confirmation or a named counter-proposal. A counter-proposal is not adopted until decision owners resolve it.
- [ ] Attach endpoint IP/config exports and link-training/status output showing protocol/FLIT format, CRC enable, retry enable, negotiated lane/rate configuration and revision IDs.
- [ ] Attach fault-injection or vendor verification evidence showing CRC detection, retry/recovery, counters/status and final payload integrity.
- [ ] Raw Mode remains a proposal unless B4/B7 are explicitly reopened and resolved in an owner-approved change record.

Artifact path(s) / SHA-256:

### Tensor and register routes

- [ ] Attach an approved block/interface diagram for GPU memory/copy engine → GPU UCIe endpoint → UCIe mainband → EIC receive/application endpoint → converters/PIC. Show the optical segment only inside EIC/PIC.
- [ ] Attach a separate address/transaction diagram tracing GPU CXL.io BAR request → BAR decode/bridge → named UCIe transport/protocol transaction → EIC endpoint/address decode → PTA register. State addresses, widths, ordering/completion semantics and error propagation. The route is open: do not infer it.
- [ ] Attach `PTA_ID` and safe-register read/write/readback trace with transaction IDs, endpoints, ordering/flush, values and errors; prove a posted write is observed before its ordering read completes.
- [ ] Record review owner confirming the strategy appendix agrees with B4/B7 and ICD without claiming the still-open forwarding mechanism is complete.

Artifact path(s) / SHA-256:

## Approved checklist follow-through — CP ring and UCIe data window

These are distinct from the PTA chiplet-local queue and cannot substitute for Q2–Q4.

### GPU CP ring

- [ ] Attach runtime/RTL revisions and GPU-owner interface agreement for the project-owner-approved lossless full-ring target. Current reviewed state: runtime append has no consumer-head free-space check, and RTL does not publish its internal head to the host.
- [ ] Test exact-full stall/resume, batch >64 KiB (ordered chunking or up-front all-or-nothing rejection), repeated 64-byte-line wrap, delayed engine acceptance/retirement, and concurrent producers. Record IDs, head/tail snapshots, retire sequence and errors; prove exactly-once order and forward progress.
- [ ] Do not claim the contract until implementation publishes a reliable consumer head and every applicable test passes.

Artifact path(s) / SHA-256:

### UCIe data-window measurements

- [ ] Record endpoint IP/config IDs, negotiated lanes/rate, adapter/FLIT mode, counter/equipment source, direction, payload sizes/distribution, concurrency, temperature/voltage, retry/error counters and run duration.
- [ ] Measure useful payload throughput independently in each direction and end-to-end request/response latency distribution unloaded and saturated/concurrent. Define timestamp endpoints and justify the selected design latency/quantile.
- [ ] Record buffering and credits at each relevant boundary in bytes and slots. Calculate required byte window as `measured_payload_bytes_per_second × selected_design_RTT_seconds`; derive slots from actual payload-size distribution (not an assumed 64-byte unit), prove fit against buffers/credits, and validate sustained throughput.
- [ ] Attach raw logs and calculation script/output. Board-plan 57.6 GB/s, 100 ns and earlier 64-byte/128-slot figures are assumptions, not measurements or accepted limits.

Artifact path(s) / SHA-256:

## Closure record

| Question / item | Owner | Artifact IDs | Result (PASS / BLOCKED / FAIL) | Open issue / decision record |
|---|---|---|---|---|
| Q1 | GPU driver/IP owner | | BLOCKED | |
| Q2 | GPU dispatcher + EIC queue owners | | BLOCKED | |
| Q3 | GPU dispatcher + EIC queue owners | | BLOCKED | |
| Q4 | GPU driver + EIC status owner | | BLOCKED | |
| Q5 | GPU UCIe + EIC/chiplet owners | | BLOCKED | |
| CP ring | GPU runtime/RTL owners | | BLOCKED | |
| UCIe data window | GPU link + EIC owners | | BLOCKED | |

Close an item only when its required artifacts are present, reproducible, reviewed by named interface owners, and consistent with the specification. If evidence is unavailable, retain BLOCKED, name the owner, and record the next evidence-producing action; do not substitute a proposed value, prose acknowledgement, or twin result for physical/IP evidence.
