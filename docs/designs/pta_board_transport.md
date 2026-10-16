# The first board's PTA transport: the driver's three hooks over CXL

**Companions:** [`board_program_plan.md`](board_program_plan.md) (B18, B24,
S1, S4), [`board_icd.md`](board_icd.md) §9 (the first board),
[`pta_chiplet_regmap.md`](pta_chiplet_regmap.md) (X4, the map),
[`pta_board_transport.py`](pta_board_transport.py) (the model of this
document), and `grxgpu/docs/proposals/pta_chiplet_host_path.md` (the GEMM
command this keeps).

**Status: S4's transport for the first board, drafted 2026-10-15.** The owner
ruled the first board's PTA link that day (the board plan's B24): CXL 2.0 on
the PCIe 5.0 PHY from the GRX930 host, the PTA module a Type-1 device, with
CXL.io and CXL.cache and no CXL.mem. This is the design that ruling unblocked:
how grxcp's driver reaches the module, finds it, hands it a GEMM and learns how
the GEMM ended. It is a design and a model. No module, no root port and no
driver code exists for it. §9 says what it asks of whom, and §10 what it
leaves open.

*Built in a model, 2026-10-15* (§11). The module's half is a model in front of
the chiplet's twin, the host's half is grxcp's driver, and a gate holds the one
to this design and the other to the twin's own call, bit for bit. There is
still no module and no root port.

**How to read it.** A standard named without a clause number means the term is
the standard's and has not been read out of its text here: CXL 2.0's opcodes
and registers below are named as this document understands them, and L1's
protocol tests are where they get checked (§10, item 4).

---

## 1. What it has to carry

S4's driver (`src/backends/pta_chiplet/pta_chiplet.h`) ends in three hooks, and
only a model fills them today:

| Hook | Today | On the first board |
|---|---|---|
| `read32(offset)` | Reads a register of the map from the twin | A 32-bit load from the module's BAR over CXL.io: one non-posted read |
| `write32(offset, value)` | Writes one | A 32-bit store over CXL.io: a posted write |
| `submit(bank, M, N, K, A, B, C, status)` | Hands the twin a GEMM as a function call | A descriptor in a ring in host memory and a doorbell over CXL.io; the module reads the operands and writes the results over CXL.cache |

The first two are what CXL.io is for, and need nothing new. The third is this
document. On rev A the G100's command processor was to carry it, which is what
the host-path proposal asked of grxgpu. On the first board no GPU stands
between the host and the PTA, so the module carries its own queue.

Three things were fixed before this was written, and are kept.

- **The map, X4, unchanged at version 1.** Identity, capabilities,
  configuration and counters stay at their offsets, so one driver answers the
  twin and a module.
- **The host path's GEMM command, field for field**: its descriptor, its flags
  and its five ways to end. grxgpu's draft decision of 2026-10-14, untracked in
  its agent's worktree, asks that the same map and GEMM contract serve a
  separate module and any later in-package attachment. Keeping the descriptor
  is how the first board does that.
- **The map's §5**: a command the device cannot honour is refused and never
  dropped, and one read of `PTA_STATUS` is the device's half of "has the work
  finished".

---

## 2. The module as a CXL device (S1)

- **One function**, a CXL 2.0 Type-1 device (B24). Configuration space as PCIe
  has it, with the class code of a processing accelerator, base class 12h. The
  CXL DVSEC for devices says it is cache-capable and not memory-capable, and a
  register locator DVSEC points at its CXL component registers. The vendor and
  device IDs are not assigned (§10, item 1).
- **BAR 0**, 64-bit and not prefetchable, because reads of the map have side
  effects: a counter's low half latches its high half.
  - Page 0, offset 0x0000: the map, X4, unchanged.
  - Page 1, offset 0x1000: the queue (§4).
  - Pages 2 and 3: the MSI-X table and its pending bits, two vectors (§6).
- **The CXL component registers** in a BAR of their own, where the register
  locator says, laid out as the standard has them.

**Enumeration.** The host finds the function by its IDs. Then the driver reads
`PTA_ID` and `PTA_CAPS0` to `PTA_CAPS2` from page 0, and `PTQ_ID` and
`PTQ_CAPS` from page 1. S4's detection runs on page 0 as it stands: the magic,
map version 1, and a tile built. `PTA_CAPS2[31]` tells a model or stand-in
from a photonic module before any work is sent, which grxgpu's draft request
for a stand-in asks of the host. S1's gate holds: with no function there is no
PTA device, and a function whose page 0 does not identify is reported unknown
and not used.

**What Linux needs is less than rev A's.** With no CXL.mem there is no HDM
decoder and no CXL memory window. The plan's L4 gap is rev A's: Linux's CXL
subsystem wants the host bridge in ACPI's CEDT, and RISC-V boards boot with a
device tree. Here the module is a PCIe function behind a root complex the
device tree describes, and its driver enables CXL.cache in the DVSEC once the
root port has it enabled. What upstream Linux offers a Type-1 device is not
read out here (§10, item 3).

---

## 3. Host memory, over CXL.cache

The module reads and writes the host's memory coherently, and keeps none of
it.

- **It reads with RdCurr**: the line's current data, and no copy kept. No snoop
  ever finds a line on the module, and a reset never strands one.
- **It writes whole lines with ItoMWr**: ownership and data in one
  transaction, and the line not kept.
- **So every transfer is coherent with the GRX930's caches.** The host flushes
  nothing before a job and invalidates nothing after. That is what Type 1 buys
  over CXL.io alone, and why the owner chose it (B24).

A cache on the module would earn its keep in one place, resident weights. A
bank's weights read with RdShared would be invalidated by a snoop the moment
the host rewrote them. That is the host path's §7, question 7, with bit 19 of
`target` reserved for it, and it is not designed here.

**Addresses are the host's physical addresses**, in the GRX930's memory.
CXL.cache carries them as they are, and there is no IOMMU on the path, so
nothing on the first board confines the module to the buffers a descriptor
names. That is the charter's stance for a first board, a single trusted
workload, and it is said here so that it is not assumed otherwise.

| Buffer | Written by | Alignment | Extent |
|---|---|---|---|
| The ring | The host its descriptors, the module its records | 4 KiB | 2ⁿ slots of 128 bytes |
| Activations, `M × K`, row-major | The host | A line | `M·K` bytes in format 0, read in whole lines |
| Weights, `K × N`, row-major | The host | A line | `K·N` bytes, likewise |
| Bias, version 2, `N` signed 64-bit | The host | A line | `8·N` bytes, likewise |
| Results, `M × N`, row-major | The module | A line | Whole lines: `4·M·N` bytes in format 0, and the last line's pad is the module's to write |

**Ownership.** From the doorbell that publishes a slot until the host reads its
status as not zero, the slot and its result buffer are the module's, and the
operands and bias are read-only to both. After that all of them are the host's.
A buffer stays allocated, and unwritten by the host, until its slot has ended,
through a teardown too (§6).

**Each buffer is physically contiguous.** A descriptor gives one address a
buffer, and nothing on the board translates for the module, so the canary's
thousand vectors need 64 kB of activations and 64 kB of results, each in one
physical run.

**The PTA's parent, in grxcp.** S4's device names its GPU in `parentDevice`
and takes that GPU's pointers, which is B9's exception. On the first board the
operands are the host's, so the device has no parent GPU, and a GEMM on it
takes host buffers whose physical addresses the driver can hand the module.
`grxMallocHost` and `grxHostRegister` are where such buffers would come from.
What `parentDevice` reports then, and how those calls give memory that is
physically contiguous, are S4's to settle as it is built (§10, item 11).

---

## 4. The queue

**The ring** is `2ⁿ` slots of 128 bytes in host memory. A slot holds one
descriptor: version 1 is line 0, and version 2 adds the first 16 bytes of
line 1. Indices are free-running 32-bit counts, and slot `i` is at
`base + (i mod 2ⁿ) · 128`.

| Offset | Field | Bytes | Written by | Line | Version |
|---|---|---|---|---|---|
| 0x00 | `size_version` | 4 | host | 0 | 1 |
| 0x04 | `target` | 4 | host | 0 | 1 |
| 0x08 | `format` | 4 | host | 0 | 1 |
| 0x0C | `M` | 4 | host | 0 | 1 |
| 0x10 | `K` | 4 | host | 0 | 1 |
| 0x14 | `N` | 4 | host | 0 | 1 |
| 0x18 | `act_addr` | 8 | host | 0 | 1 |
| 0x20 | `wgt_addr` | 8 | host | 0 | 1 |
| 0x28 | `out_addr` | 8 | host | 0 | 1 |
| 0x30 | `status` | 4 | module | 0 | 1 |
| 0x34 | `gemm_index` | 4 | module | 0 | 1 |
| 0x38 | `saturations` | 8 | module | 0 | 1 |
| 0x40 | `act` | 4 | host | 1 | 2 |
| 0x44 | `clips` | 4 | module | 1 | 2 |
| 0x48 | `bias_addr` | 8 | host | 1 | 2 |

Every field means what the host path's §4 and §10 say. Where they say "CP",
read "the module". Both lines hold fields of both writers, so the module
rewrites a line whole, with the host's fields as it read them.

**The queue's registers** are page 1 of BAR 0:

| Offset | Name | Access | Description |
|---|---|---|---|
| 0x1000 | `PTQ_ID` | R | [31:8] the magic 0x505451, "PTQ"; [7:0] this page's version, 1 |
| 0x1004 | `PTQ_CAPS` | R | [4:0] log2 of the most slots a ring may have; [8] version 1 descriptors taken, [9] version 2; [15:12] MSI-X vectors |
| 0x1008 | `PTQ_CTRL` | RW | [0] ENABLE; [1] IRQ_EN. ENABLE 0 to 1 sets `PTQ_HEAD` to `PTQ_TAIL` |
| 0x100C | `PTQ_STATUS` | R | [0] ENABLED: the module may still write host memory; [1] HALTED; [15:8] why it halted |
| 0x1010 | `PTQ_BASE_LO` | RW | The ring's host physical address, 4 KiB aligned; written only while ENABLE is 0 |
| 0x1014 | `PTQ_BASE_HI` | RW | Its upper half |
| 0x1018 | `PTQ_SIZE` | RW | log2 of the ring's slots; written only while ENABLE is 0 |
| 0x1020 | `PTQ_TAIL` | RW | The doorbell: one past the last slot published, a free-running index. While ENABLE is 0 it sets where the ring starts |
| 0x1024 | `PTQ_HEAD` | R | The next slot the module will end: every slot behind it has its status |
| 0x1028 | `PTQ_IRQ_AT` | RW | Vector 0 is raised when `PTQ_HEAD` reaches this index |
| 0x102C | `PTQ_FAULT_AT` | R | The index the queue halted at |

The page has its own identity and version, so the map's version stays 1 and
the map is the same object on both boards. Its version is `PTQ_ID`'s to count.

**The host publishes:**

1. Waits for room: fewer than `2ⁿ` slots published and not yet ended.
2. Writes the descriptor into the next slot, with `status` zero.
3. Fences, so that its stores reach memory before the doorbell does. On
   RISC-V that is `fence w,o`, which Linux's `writel` already carries.
4. Writes `PTQ_TAIL`, one past the slot. Several slots may go out on one
   doorbell.

**The module runs them,** one at a time and in index order, from `PTQ_HEAD` to
`PTQ_TAIL`:

1. It reads the slot.
2. It decides as the host path has it. **Not issued** (4) if it cannot use the
   descriptor: a version or size it does not know, a format it does not know, a
   dimension of zero, an address off a line, or a flag bit it does not know.
   **Refused** (2) if the tile cannot do it as configured. **Not run** (5) if
   bit 16 of `target` is set and the PTA command before it in this queue did
   not end done. None of these takes an index, and its record says
   `gemm_index` 0xFFFFFFFF, which the host path did not say.
3. Otherwise it starts the GEMM, which takes `PTA_GEMM_CT` as its index. It
   reads the weights and the activations, runs the shots, and writes each line
   of results as soon as the line is full.
4. **Only once every line of results is ordered,** its GO received, it writes
   the record: line 0 whole, the host's fields as it read them and its own,
   `status`, `gemm_index` and `saturations`. For version 2 line 1 is rewritten
   too, for `clips`, and before line 0.
5. Once the record is ordered, `PTQ_HEAD` moves past the slot. Then, and not
   before, it raises vector 0 if the slot is the one `PTQ_IRQ_AT` names.

**A doorbell that makes no sense halts the queue.** A `PTQ_TAIL` that goes
back, or more than the ring past `PTQ_HEAD`, sets HALTED with reason 1 and
`PTQ_FAULT_AT`, and nothing past it is taken.

**The ring is the queue.** A calibration holds it back and drops nothing, as
the map's §5 has it. How far ahead the module reads is its own affair. `BUSY`
in `PTA_STATUS` covers a command taken from the ring and not yet ended, as it
covers a queued one on the twin. So the map's §7, item 3, the queue depth a
dispatcher had to know, has no answer to give on this board: the ring's size
is the host's.

---

## 5. How a command ends

| `status` | Ends | Results | `gemm_index` |
|---|---|---|---|
| 1 | Done | Written | The GEMM's |
| 2 | Refused | Untouched | None, 0xFFFFFFFF |
| 3 | Lost: a teardown or a reset ended it (§6) | **Undefined** | None reported |
| 4 | Not issued | Untouched | None |
| 5 | Not run | Untouched | None |

**One change from the host path, and why.** There a lost command leaves its
results untouched. Here the module writes lines of results as the shots finish,
so a GEMM a teardown cuts off may have written some of them. Keeping them back
would need the module to hold a whole GEMM's results: 64 kB for the canary's
thousand vectors, a megabyte for X2's 4096-square layer at a batch of 64. So a
lost GEMM's results are undefined, and the host may reuse the buffer once the
slot has ended. S4's driver says "C untouched" of every failure today. On this
transport it would say so of every failure but a lost one (§9). *Ruled so on
2026-10-15* (the board plan's B25).

**The index comes from the record.** S4's property needs a GEMM's index to say
what the GEMM ran on. With a queue, a read of `PTA_GEMM_CT` before a GEMM no
longer gives it, so the driver takes the index from the record. A teardown that
cuts off a started GEMM has used an index, so the next GEMM done may be two
on.

**The rules, each one a check in the model (§8):**

| Rule | Check |
|---|---|
| A status is written only after every line of results is ordered. A host that reads a status that is not zero reads the results that go with it, with no fence and no invalidate | C1, C2 |
| Each command ends as it has to; lost only in a teardown | C3 |
| Indices run on, and a command that did not start has none | C4 |
| `PTQ_HEAD` moves only once the record is ordered | C5 |
| The interrupt is raised only once the record is ordered | C6 |
| The record keeps the host's fields. The module rewrites the line from what it read, so it must have read what the host wrote: the host's fence before the doorbell | C7 |
| The host writes no slot the module still owns | C8 |
| Once `PTQ_STATUS.ENABLED` reads 0, nothing more of the module's lands in host memory | C9 |
| After a teardown no slot is left pending | C10 |
| The block's configuration changes only with the queue drained: the ring empty and one read of `PTA_STATUS` showing neither BUSY nor CAL_BUSY | C11 |
| Every command ends, and a wait ends by a timeout only when the module has stopped answering | C12 |

C11 is the map's rule that configuration takes effect at the next GEMM start,
made safe for a queue. A register write over CXL.io does not wait for the ring,
so the driver writes `PTA_IMPAIR`, `PTA_BITS`, `PTA_SEED` and the rest only
with the queue drained. A network whose layers need different settings pays a
drain between them (§10, item 5). *Ruled so for now, 2026-10-15* (B25).

---

## 6. Interrupts, timeouts, teardown and reset

**Interrupts.** Two MSI-X vectors of the module's own function, which answers
the map's §7, item 4, for this board. Vector 0 is raised when `PTQ_HEAD`
reaches `PTQ_IRQ_AT` and `PTQ_CTRL.IRQ_EN` is set. Vector 1 is the map's §3,
`PTA_IRQ_STATUS` under `PTA_IRQ_MASK`. An MSI-X is a write over CXL.io, which
nothing orders against CXL.cache's writes, so the module raises vector 0 only
after the record is ordered (C6). Polling stays valid and costs the link
nothing: the host spins on the slot's `status` in its own cache, and the line
moves only when the record lands. *As built (§11),* the driver also reads
`PTQ_STATUS` on each pass while the slot is pending, which is how it tells a
halted queue from a slow one. On a board it would spin on its cache between
such reads.

**Timeouts.** The driver bounds every wait. It can work out a GEMM's own time
from its shots, the shot rate `PTA_CAPS2` reports and a calibration's longest
run. The margin over that is the driver's to choose. When a wait runs out, the
driver tears the queue down.

**Teardown:**

1. The host writes `PTQ_CTRL.ENABLE` = 0. The module takes nothing more, drops
   what it was running, and waits until every write it has in flight is
   ordered. Then `PTQ_STATUS.ENABLED` reads 0, and it writes nothing more.
2. The host reads `PTQ_HEAD`. The slots behind it the module ended, and the
   host reads their records. The slots from `PTQ_HEAD` to the host's own next
   index are the host's again, and it writes each one's status 3, lost.
3. The host writes `PTQ_TAIL` with its next index, then `PTQ_CTRL.ENABLE` = 1,
   which sets `PTQ_HEAD` to `PTQ_TAIL`: the ring starts empty.

**Reset.** If `ENABLED` does not read 0 within a bound, the host resets the
module over configuration space, by a function-level reset or a CXL reset.
Then it programs the queue again, writes `PTA_SEED` again, because a reset
restarts `PTA_GEMM_CT`, ends every slot the module had as lost, and enables
the queue. Before resetting a module that may hold lines, CXL 2.0 has the host
ask it to write back and invalidate its cache. This module holds none (§3),
which is one reason it is designed not to. What the standard requires of a
reset with CXL.cache requests still in flight is not read out here (§10,
item 4).

**After a teardown or a reset the next valid command completes,** as the
charter's demonstration asks of every interface. The model's last phase holds
that.

---

## 7. What it costs

**The link.** On the canary's FC3 a shot is a line each way in format 0: 64
one-byte activations in, and 16 four-byte results out. A x16 link at 32 GT/s
is 64 GB/s a direction raw. After PCIe 5.0's 128b/130b coding and the 68-byte
flit's 64 bytes of slots, at most **59.31 GB/s** of lines is left, before any
header slot is sent:

| Width | GB/s, at most | The canary's FC3, results int32 | A full tile, int32 | A full tile, codes |
|---|---|---|---|---|
| x4 | 14.83 | 0.232 GS/s, 23% | 0.058 GS/s, 6% | 0.116 GS/s, 12% |
| x8 | 29.65 | 0.463 GS/s, 46% | 0.116 GS/s, 12% | 0.232 GS/s, 23% |
| x16 | 59.31 | 0.927 GS/s, 93% | 0.232 GS/s, 23% | 0.463 GS/s, 46% |

The percentages are of the tile's working rate, 1 GS/s (B11). So the first
board runs the canary at up to 93% of the PTA's working rate, and both
directions bind at once. Headers, credits and the link's measured efficiency
come off that, and none of them is known. A full 128 × 64 tile sends 64 sums
of four bytes a shot, which hold it to 23%. With the converter's one-byte
codes out it would run at 46%, and its inputs would then bind (§10, item 6).

**A correction to B24.** B24 priced the canary's results as one-byte codes,
16 GB/s out. Format 0 returns four bytes a result, 64 GB/s, as much as goes
in. B24's conclusion stands: at x16 the link holds the PTA to about its
working rate, and no faster.

**A job.** In 64-byte lines, with the weights sent with it:

| Job | M × K × N | To the module | To the host |
|---|---|---|---|
| The canary's FC3, one vector | 1 × 64 × 16 | 18: 1 descriptor, 16 weights, 1 activations | 2: 1 results, 1 record |
| The canary's FC3, the gate's thousand | 1000 × 64 × 16 | 1,017: 1 descriptor, 16 weights, 1,000 activations | 1,001: 1,000 results, 1 record |
| A full tile, a thousand shots | 1000 × 128 × 64 | 2,129 | 4,001 |
| The same, codes out | 1000 × 128 × 64 | 2,129 | 1,001 |

A vector a job moves 20 lines for one shot of work, 16 of them weights. A
thousand vectors in one job move a line in and a line out a shot. That is an
argument for resident weights, or for whole batches a job.

**A job's serial crossings,** counted and not timed:

| Step | Crossings | Hidden behind the job before it |
|---|---|---|
| The doorbell, `PTQ_TAIL`: a posted write over CXL.io | 1 | Yes, rung while that job runs |
| The descriptor: its line read over CXL.cache | 2 | Yes, if the module reads ahead |
| The operands' first line, whose address the descriptor gave | 2 | Yes, likewise |
| The shots, with operands and results streaming | The data | It is the work |
| The last line of results: the module learns it is ordered | 2 | No |
| The record: written, and its line at the host | 3 | No |

Ten crossings, five round trips, besides the work. On a busy queue the module
reads ahead, and what a job still adds is the 5 crossings at its end. An
interrupt in place of polling adds one more. A crossing's time is the board's
to measure, under the demonstration's Gate L.

---

## 8. The model, and what it showed

[`pta_board_transport.py`](pta_board_transport.py), standard library only.
Its first five sections hold the slot, the registers and §7's tables to their
arithmetic. Its sixth runs the protocol: a host and a module on a link where
every read and write is ordered a crossing after it is sent, 4 ticks, or up to
6 ticks later than that, so lines land out of order. The module takes one
command at a time and writes its results a line at a time. The host polls, or
waits on an interrupt, and bounds every wait. Every rule of §5 is checked as
the run goes.

**With nothing planted, every check holds** over 42 commands in seven phases,
2,280 ticks. The phases run one command at a time, polled; then a refusal, a
command that waits on it, one the module cannot use, and two that run; then a
stream with the ring full; then a new seed with the queue drained, and six
commands each waited on by interrupt; then a teardown with a long GEMM running;
then a module that stops answering and is reset; and four more after. They
end: 33 done, 1 refused, 1 not issued, 1 not run, and 6 lost.

**Fourteen faults, planted one at a time, and each named by the check written
for it:**

| Fault | In | What it does | Written for | Fired |
|---|---|---|---|---|
| `status-before-results` | module | Writes a status without waiting to learn its results are ordered | C1 | C1, C9 |
| `results-one-line-short` | module | Writes its results a line short, and ends them done | C1 | C1 |
| `refused-writes-results` | module | Runs a command it refuses, results and all, then ends it refused | C2 | C2, C4 |
| `pad-on-refusal` | module | Writes the pad of a refused command's last line of results | C2 | C2 |
| `refusal-as-done` | module | Ends a refusal as done | C3 | C1, C3, C4 |
| `index-on-refusal` | module | Gives a refused command an index | C4 | C4 |
| `head-before-status` | module | Moves `PTQ_HEAD` on when it writes a status, not when it is ordered | C5 | C5, C6 |
| `irq-before-status` | module | Raises the interrupt when it writes a status, not when it is ordered | C6 | C6 |
| `doorbell-before-stores` | host | Rings before its descriptor is in memory: the fence left out | C7 | C1, C7, C9 |
| `slot-reuse` | host | Counts the ring's room one too many, and writes a slot still pending | C8 | C1, C7, C8, C12 |
| `run-after-failure` | module | Runs a command that asked to follow a success, after a failure | C3 | C3 |
| `disable-before-drain` | module | Reports itself stopped while its GEMM still runs, and its results land | C9 | C9 |
| `lost-left-pending` | host | Ends all but the last slot of a teardown as lost | C10 | C10 |
| `config-while-queued` | host | Writes `PTA_SEED` with GEMMs still queued | C11 | C1, C11 |

**Then the model itself was planted** with twelve errors, one at a time: in
its module, in its host and in its checks. Eleven stopped its run. The twelfth
narrowed C5 to the slot just behind `PTQ_HEAD`, and that is not weaker here:
`PTQ_HEAD` moves a slot at a time and an ended slot never reads pending again,
so every slot it has passed was once just behind it. Three of the eleven
stopped it only once sharper faults had been written. Narrowed, C1 to the first
result, C2 to the first byte of results and C9 to the slots and not the
buffers, each still named every fault first written for it. The faults that
tell them apart are `results-one-line-short`, `pad-on-refusal`, and
`disable-before-drain` as it now stands: a GEMM whose results land after its
slot has ended and whose record never does. One fault written along the way
could not bite, and was replaced: a status written before the last line of
results was ordered. With shots a tick apart, that line was in memory before
the record every time.

**What it is not.** A model of CXL. It has no flits, no credits, no channels
and no snoops. "Ordered" is a time, and GO is a message a crossing after it.
It holds the design's rules against one another, not against the standard,
and its ticks are not nanoseconds.

---

## 9. What it asks, and of whom

| Party | What | Step |
|---|---|---|
| grx930 | A CXL 2.0 root port with CXL.cache: the home agent, an L2 that answers snoops (none will find a line here, but a Type-1 host has to ask), and RdCurr and ItoMWr served. MSI-X delivered to the harts, which is the plan's ask for AIA's message-signalled interrupts. PCIe 5.0-class transceivers (B24) | L1 |
| grx930's firmware | The root complex in the device tree; CXL.cache enabled on the root port | L4 |
| The PTA module, a stand-in or photonic | One function as §2 has it; the map on page 0 and the queue on page 1; §4's order and §5's rules; two MSI-X vectors | The module's owner |
| grxcp | The transport behind S4's hooks: `read32` and `write32` as MMIO, `submit` as this queue; a lost GEMM's results undefined; the index from the record. The PTA device taking its operands from host memory, in buffers that are physically contiguous. And the twin behind a ring in host memory, so that S4's gate runs through this queue before any module exists. *Built, 2026-10-15 (§11), but for the device and its buffers* | S4 |
| grxgpu | Nothing new. The descriptor is kept field for field, so the host path stays one command on both boards | — |

---

## 10. Open

1. **The module's vendor and device IDs.** Nothing is assigned.
2. **The lane count.** x16 is sized here from the canary's traffic and is not
   ruled (B24). At x8 the canary runs at 46% of the PTA's working rate.
3. **Linux and a Type-1 device on RISC-V.** What the kernel's CXL code does for
   a device with no memory, and what is left to the driver, is not read out
   here.
4. **The standard's own text.** RdCurr, ItoMWr and GO; the DVSEC's cache
   enable, write-back and invalidate; resets with CXL.cache traffic in flight.
   L1's protocol tests are where this document's reading of them gets checked.
5. **Register writes in the ring.** Configuration changes only with the queue
   drained (C11), so a network whose layers need different settings, an ADC
   shift for one, drains between them. The alternative is a second kind of
   ring entry that writes a register in order, as the host path's command list
   carries DCR writes.
   *Drained only, for now: ruled 2026-10-15 (B25).* The second kind of entry
   stays open.
6. **Results as codes.** A format that returns the converter's 8-bit codes,
   one byte each, would double a full tile's rate over this link. The canary's
   shape is bound by its inputs either way. The canary's declared reference is
   the converter's codes (B21), and format 0 carries them as sums of four
   bytes.
7. **Resident weights**, bit 19 of `target`, and the module's cache with them
   (§3).
8. ~~**Lost: undefined, or buffered**~~ (§5). *Ruled 2026-10-15 (B25):
   undefined.*
9. **What a crossing, a doorbell and an interrupt cost in time.** Nothing here
   says, and the board measures it.
10. ~~**The stand-in, and the map as the discovery contract**~~: the questions
    1 and 2 put to grxgpu in the reconciliation proposal's §6. *Ruled
    2026-10-15 (B25):* a stand-in for integration only, over F1 and reporting
    `PTA_CAPS2[31]`, and the map the discovery contract on both boards. Who
    builds a stand-in is still open, and no RTL exists at 128 × 64.
11. **The PTA's parent and its buffers in grxcp** (§3). What `parentDevice`
    reports on the first board, and how `grxMallocHost` or `grxHostRegister`
    gives the physically contiguous memory a descriptor's one address needs.

---

## 11. Built, in a model (2026-10-15)

Both halves of §4 to §6 exist as code, and neither is hardware.

- **The module**, `src/backends/pta_chiplet/pta_chiplet_module.c`: the queue's
  page in front of the twin, which it calls through the twin's own interface.
  Host memory is two calls, a whole line read (RdCurr) and a whole line written
  (ItoMWr), and a call that has returned is ordered. One command at a time; NOT
  ISSUED and NOT RUN decided before the tile sees a command; a done GEMM's
  results and then its record; a teardown that drops what is running and writes
  nothing of it; halts on a doorbell that makes no sense and on memory that
  fails; MSI-X vectors 0 and 1 as calls.

- **The host's half**, in the driver,
  `src/backends/pta_chiplet/pta_chiplet.cpp`: `pta_chiplet_attach_queue` checks
  `PTQ_ID` and `PTQ_CAPS`, takes a ring from a pair of hooks that hand out
  memory the module can reach, and enables it. With the queue attached a GEMM
  is staged into that memory as int8, published, fenced and rung. The driver
  polls the record, and a halt or a wait that runs out is a teardown that ends
  the GEMM lost. A module that never says it has stopped leaves the queue dead.
  The index comes from the record.

- **The runtime** attaches the queue when its test seam is given host memory,
  and then every GEMM on the PTA device goes through the ring.

**What holds it.** `test_pta_chiplet_queue.cc`, 110 checks, 55 on each of the
two tiles. It holds the module to this design with descriptors written by hand:
each way a command ends, a record that keeps the host's fields, results over
five lines and every line before the record, a ring that wraps, both
interrupts, both halts, a teardown with the GEMM caught running, the slot
behind it held until the tile is free, and a reset. And it holds the driver's
transport through the module to the twin's own call, bit for bit, with the
index each record reports. Five modules built wrong on purpose each fail it: a
record before its results, a refusal with an index, bit 16 ignored, a dropped
command written after all, and a doorbell taken whatever it says. Nine errors
planted in the driver's queue code, one at a time in a copy and not in CI, each
fail it as well: results not sign-extended, a descriptor written after the
doorbell, a ring mask a slot short, the index read from the wrong field, a
teardown that does not end its slots or does not wait for the module to stop,
an operand of 128 let through, a refusal reported as not accepted, and a halt
not looked for. And grxBLAS's PTA gate, 4,164 results a tile held bit for bit
to the model built from the device property alone, passes through the queue as
it does through the call, with no link installed: 17 commands through the ring
on each tile.

**Where it departs from §4 to §6, and why.**

1. The operands are copied, into memory the driver owns. The driver's interface
   takes the twin's int32 integers, and a caller's own buffers cannot yet be
   handed to the module (§10, item 11). So a lost GEMM leaves the caller's C
   untouched after all: only the driver's copy is undefined.

2. `PTQ_STATUS.ENABLED` reads 0 the moment the queue is told to stop. In a
   model a write call returns ordered, so nothing is ever in flight.

3. The driver reads `PTQ_STATUS` on each pass of its poll (§6). It is also
   where a model's time passes, as a read of `PTA_STATUS` is for the twin.

**What is not built.** Version 2 descriptors and the activation stage through
the queue: `PTQ_CAPS` says version 1 only, and a version 2 descriptor ends NOT
ISSUED. The interrupts, which the module raises and no driver waits on. Reading
ahead. The PTA device with no parent GPU, whose operands are the host's: the
runtime's device is still S4's, behind a GPU, and grxBLAS still copies its
operands out of the parent's memory. And anything CXL: there is no root port,
no flit and no snoop, only what the design's rules say happens at the line.
