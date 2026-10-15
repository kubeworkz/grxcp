"""
The first board's PTA transport: the driver's three hooks over CXL.

B24 put the first board's PTA module on CXL 2.0 from the GRX930 host, as a
Type-1 device: CXL.io for its configuration space and its register block,
and CXL.cache to read its operands from host memory and write its results,
coherently.  pta_board_transport.md is the design, and this file is what of
it can be checked before there is a module.

  1. The slot.  The GEMM descriptor grxcp proposed to grxgpu for rev A
     (grxgpu's docs/proposals/pta_chiplet_host_path.md, sections 4 and 10),
     kept field for field and laid in a ring in host memory: every field
     held to its offset, its size, its writer and its cache line.
  2. The queue's registers, the second page of the module's BAR.
  3. What one job moves across the link, in lines, each way.
  4. What the link carries, as the most shots a second at each width.
  5. One job's serial crossings of the link, counted and not timed.
  6. The protocol, run.  A host and a module on a link that reorders, every
     rule of the design checked as it runs, and then fourteen faults planted
     one at a time, each of which the check written for it has to name.

Assumed, and marked at each use: a cache line is 64 bytes; CXL 2.0's
68-byte flit carries 64 bytes of slots; PCIe 5.0's PHY codes 128 bits in
130; the tile is B10's 128 x 64 at B11's 1 GS/s; operands are one byte and
results four, the descriptor's format 0.  No header slot and no other
traffic is counted, so every rate in section 4 is an upper bound.  GB is
10^9 bytes.  Nothing here was measured, and there is no module.

Standard library only.  Run:  python3 docs/designs/pta_board_transport.py
With --fault NAME the simulation runs with that one fault planted.
"""
import heapq
import random
import struct
import sys

LINE = 64                    # assumed: a cache line, the unit CXL.cache moves
SLOT = 2 * LINE              # a ring slot: room for a version 2 descriptor
LANES, GT_S = 16, 32         # B24: x16 as sized here, at PCIe 5.0's rate
ENCODING = 128 / 130         # assumed: PCIe 5.0's PHY
FLIT = 64 / 68               # assumed: a 68-byte flit's 64 bytes of slots
SHOT_HZ = 1e9                # B11
TILE = (128, 64)             # B10: rows k by columns n
CANARY = (1000, 64, 16)      # M, K, N: FC3 on the gate's thousand vectors
A_BYTES, W_BYTES, R_BYTES, CODE_BYTES = 1, 1, 4, 1
BANKS = 2                    # the plan's S3: two weight banks
M32, M64 = (1 << 32) - 1, (1 << 64) - 1


def lines(nbytes):
    return -(-nbytes // LINE)


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


# ---- 1. The slot ----------------------------------------------------------
# (offset, bytes, field, written by, descriptor version that adds it).  The
# host path's "CP" fields are the module's here: there is no GPU between.
FIELDS = [
    (0x00, 4, "size_version", "host", 1),
    (0x04, 4, "target", "host", 1),
    (0x08, 4, "format", "host", 1),
    (0x0C, 4, "M", "host", 1),
    (0x10, 4, "K", "host", 1),
    (0x14, 4, "N", "host", 1),
    (0x18, 8, "act_addr", "host", 1),
    (0x20, 8, "wgt_addr", "host", 1),
    (0x28, 8, "out_addr", "host", 1),
    (0x30, 4, "status", "module", 1),
    (0x34, 4, "gemm_index", "module", 1),
    (0x38, 8, "saturations", "module", 1),
    (0x40, 4, "act", "host", 2),
    (0x44, 4, "clips", "module", 2),
    (0x48, 8, "bias_addr", "host", 2),
]
DESC = struct.Struct("<IIIIIIQQQIIQ")      # version 1, as the simulation writes it


def slot_layout():
    for version, size in ((1, 64), (2, 80)):
        at = 0
        for off, n, name, who, since in FIELDS:
            if since > version:
                continue
            assert off == at, (name, "not contiguous")
            assert off % n == 0, (name, "not naturally aligned")
            at += n
        assert at == size, (version, at)
    assert DESC.size == 64
    module_lines = {off // LINE for off, n, name, who, since in FIELDS if who == "module"}
    assert module_lines == {0, 1}           # it rewrites both lines, whole
    status = [f for f in FIELDS if f[2] == "status"][0]
    assert status[0] // LINE == 0           # a version 1 slot is one line
    assert 80 <= SLOT and SLOT % LINE == 0
    section("1. The slot: the host path's descriptor, field for field, in a ring of 128-byte slots")
    print("| Offset | Field | Bytes | Written by | Line | Version |")
    print("|---|---|---|---|---|---|")
    for off, n, name, who, since in FIELDS:
        print(f"| 0x{off:02X} | `{name}` | {n} | {who} | {off // LINE} | {since} |")
    print("Version 1 is line 0, 64 bytes; version 2 adds 16 bytes of line 1.  The module")
    print("writes into both lines, so it rewrites each whole: the host's fields as it read them.")


# ---- 2. The queue's registers ---------------------------------------------
PTQ = 0x1000                 # the second page of the module's BAR
REGS = [
    (0x000, "PTQ_ID", "R", "[31:8] the magic 0x505451, \"PTQ\"; [7:0] this page's version, 1"),
    (0x004, "PTQ_CAPS", "R", "[4:0] log2 of the most slots a ring may have; [8] version 1 "
     "descriptors taken, [9] version 2; [15:12] MSI-X vectors"),
    (0x008, "PTQ_CTRL", "RW", "[0] ENABLE; [1] IRQ_EN.  ENABLE 0 to 1 sets PTQ_HEAD to PTQ_TAIL"),
    (0x00C, "PTQ_STATUS", "R", "[0] ENABLED: the module may still write host memory; "
     "[1] HALTED; [15:8] why it halted"),
    (0x010, "PTQ_BASE_LO", "RW", "the ring's host physical address, 4 KiB aligned; "
     "written only while ENABLE is 0"),
    (0x014, "PTQ_BASE_HI", "RW", "its upper half"),
    (0x018, "PTQ_SIZE", "RW", "log2 of the ring's slots; written only while ENABLE is 0"),
    (0x020, "PTQ_TAIL", "RW", "the doorbell: one past the last slot published, a free-running "
     "index.  While ENABLE is 0 it sets where the ring starts"),
    (0x024, "PTQ_HEAD", "R", "the next slot the module will end: every slot behind it has its status"),
    (0x028, "PTQ_IRQ_AT", "RW", "vector 0 is raised when PTQ_HEAD reaches this index"),
    (0x02C, "PTQ_FAULT_AT", "R", "the index the queue halted at"),
]
PTQ_CTRL, PTQ_STATUS = PTQ + 0x008, PTQ + 0x00C
PTQ_BASE_LO, PTQ_BASE_HI, PTQ_SIZE = PTQ + 0x010, PTQ + 0x014, PTQ + 0x018
PTQ_TAIL, PTQ_HEAD, PTQ_IRQ_AT = PTQ + 0x020, PTQ + 0x024, PTQ + 0x028
PTA_SEED = 0x050             # the map's block, page 0
HALT_DOORBELL = 1            # PTQ_STATUS[15:8]: the doorbell moved back, or past the ring


def registers():
    offs = [r[0] for r in REGS]
    assert len(set(offs)) == len(offs) and all(o % 4 == 0 and o < 0x1000 for o in offs)
    assert int.from_bytes(b"PTQ", "big") == 0x505451
    assert [r[1] for r in REGS if r[0] in (0x008, 0x00C, 0x020, 0x024)] == \
        ["PTQ_CTRL", "PTQ_STATUS", "PTQ_TAIL", "PTQ_HEAD"]
    section("2. The queue: the second page of the BAR, beside the map's page")
    print("| Offset | Name | Access | Description |")
    print("|---|---|---|---|")
    for off, name, acc, what in REGS:
        print(f"| 0x{PTQ + off:04X} | `{name}` | {acc} | {what} |")


# ---- 3. One job, in lines -------------------------------------------------
def job(M, K, N, out_bytes=R_BYTES, version=1, weights=True):
    to_module = {"descriptor": version, "weights": lines(K * N * W_BYTES) if weights else 0,
                 "activations": lines(M * K * A_BYTES)}
    to_host = {"results": lines(M * N * out_bytes), "record": version}
    return to_module, to_host


def jobs():
    section("3. One job across the link, in 64-byte lines, the weights sent with it")
    print("| Job | M x K x N | To the module | To the host | Lines a shot, each way |")
    print("|---|---|---|---|---|")
    rows = [("The canary's FC3, one vector", (1,) + CANARY[1:], R_BYTES),
            ("The canary's FC3, the gate's thousand", CANARY, R_BYTES),
            ("A full tile, a thousand shots", (1000,) + TILE, R_BYTES),
            ("The same, the converter's codes out", (1000,) + TILE, CODE_BYTES)]
    out = {}
    for name, (M, K, N), ob in rows:
        a, b = job(M, K, N, ob)
        ta, tb = sum(a.values()), sum(b.values())
        out[name] = (ta, tb)
        print(f"| {name} | {M} x {K} x {N} | {ta}: {a['descriptor']} descriptor, {a['weights']} "
              f"weights, {a['activations']} activations | {tb}: {b['results']} results, "
              f"{b['record']} record | {ta / M:.3f} in, {tb / M:.3f} out |")
    assert out["The canary's FC3, one vector"] == (1 + 16 + 1, 1 + 1)
    assert out["The canary's FC3, the gate's thousand"] == (1 + 16 + 1000, 1000 + 1)
    print("A one-vector job moves 20 lines for one shot's worth of work: the weights are 16 of")
    print("them.  A thousand vectors in one job move a line in and a line out a shot.")
    return out


# ---- 4. What the link carries ---------------------------------------------
def bound_gbs(lanes):
    """A direction's bytes a second, at most: the PHY's coding and the flit's
    framing taken out, and no header slot counted."""
    return lanes * GT_S / 8 * ENCODING * FLIT


def shot_rate(lanes, K, N, out_bytes):
    per_shot = max(K * A_BYTES, N * out_bytes)       # the busier direction, bytes a shot
    return bound_gbs(lanes) * 1e9 / per_shot


def link():
    section("4. What the link carries: the most shots a second, against the tile's 1 GS/s")
    raw = LANES * GT_S / 8
    assert raw == 64.0
    print(f"x16 at {GT_S} GT/s is {raw:.0f} GB/s a direction raw, and at most "
          f"{bound_gbs(16):.2f} GB/s of lines after 128b/130b and the flit's framing.")
    print("| Width | GB/s, at most | Canary's FC3, results int32 | Full tile, int32 | Full tile, codes |")
    print("|---|---|---|---|---|")
    rates = {}
    for lanes in (4, 8, 16):
        r = [shot_rate(lanes, CANARY[1], CANARY[2], R_BYTES), shot_rate(lanes, *TILE, R_BYTES),
             shot_rate(lanes, *TILE, CODE_BYTES)]
        rates[lanes] = r
        print(f"| x{lanes} | {bound_gbs(lanes):.2f} | " +
              " | ".join(f"{x / 1e9:.3f} GS/s, {x / SHOT_HZ:.0%}" for x in r) + " |")
    assert abs(rates[16][0] / 1e9 - 0.927) < 0.001
    assert rates[16][1] < rates[16][2] < rates[16][0]
    print("The canary's shot is 64 bytes each way, so both directions bind at once.  A full")
    print("tile's 64 sums of four bytes are what hold it: one-byte codes double its rate,")
    print("and its inputs then bind.  None of this is the link's own efficiency, which nobody")
    print("has measured; it is what is left before any header is sent.")
    return rates


# ---- 5. One job's serial crossings ----------------------------------------
CROSSINGS = [
    ("The doorbell, `PTQ_TAIL`: a posted write over CXL.io", 1, "yes, rung while it runs"),
    ("The descriptor: its line read over CXL.cache", 2, "yes, if the module reads ahead"),
    ("The operands' first line, whose address the descriptor gave", 2, "yes, likewise"),
    ("The shots, with the operands and results streaming", 0, "it is the work"),
    ("The last result's line: the module learns it is ordered", 2, "no"),
    ("The record: written, and its line at the host", 3, "no"),
]


def crossings():
    section("5. One job's serial crossings of the link, counted and not timed")
    print("| Step | Crossings | Hidden behind the job before it |")
    print("|---|---|---|")
    for step, n, hidden in CROSSINGS:
        print(f"| {step} | {n if n else 'the data'} | {hidden} |")
    total = sum(n for _, n, _ in CROSSINGS)
    tail = sum(n for _, n, h in CROSSINGS if h == "no")
    assert (total, tail) == (10, 5)
    print(f"{total} crossings, {total // 2} round trips, beside the work.  On a busy queue the")
    print(f"module reads ahead, and what a job still adds is the {tail} at its end.  A crossing's")
    print("time is the board's to measure.  An interrupt in place of polling adds one more.")


# ---- 6. The protocol, run --------------------------------------------------
L, JITTER = 4, 6             # ticks: a crossing; how much later a line may be ordered
T_WAIT, T_QUIESCE = 400, 60  # the host's patience: for a slot, and for the queue to stop
RING_LOG2, RING, BUF = 3, 0x10000, 0x20000
SENTINEL = 0xA5              # what the host fills a result buffer with before a job
PENDING, DONE, REFUSED, LOST, NOT_ISSUED, NOT_RUN = range(6)
NAMES = ["pending", "done", "refused", "lost", "not issued", "not run"]
NO_INDEX = M32
AFTER_SUCCESS = 1 << 16      # target bit 16: run only after a success (host path, section 9)

CHECKS = {
    "C1": "a slot that ends done holds its GEMM, as its seed and its index give it",
    "C2": "a slot that ends refused, not issued or not run leaves its results untouched",
    "C3": "each command ends as it has to, and lost only in a teardown",
    "C4": "each done GEMM's index is the last one's plus one; one that did not start has none",
    "C5": "behind PTQ_HEAD every slot has its status",
    "C6": "an interrupt finds the slots it was raised for ended",
    "C7": "an ended slot keeps the fields the host wrote",
    "C8": "the host writes no slot the module still owns",
    "C9": "nothing lands in a slot or a buffer the host has ended",
    "C10": "after a teardown no slot is left pending",
    "C11": "the block's configuration changes only with the queue drained",
    "C12": "every command ends, and only the wedged module's by a timeout",
}
FAULTS = {
    "status-before-results": ("module", "writes a status without waiting to learn its results are ordered", "C1"),
    "results-one-line-short": ("module", "writes its results a line short, and ends them done", "C1"),
    "refused-writes-results": ("module", "runs a command it refuses, results and all, then ends it refused", "C2"),
    "pad-on-refusal": ("module", "writes the pad of a refused command's last line of results", "C2"),
    "refusal-as-done": ("module", "ends a refusal as done", "C3"),
    "index-on-refusal": ("module", "gives a refused command an index", "C4"),
    "head-before-status": ("module", "moves PTQ_HEAD on when it writes a status, not when it is ordered", "C5"),
    "irq-before-status": ("module", "raises the interrupt when it writes a status, not when it is ordered", "C6"),
    "doorbell-before-stores": ("host", "rings before its descriptor is in memory: the fence left out", "C7"),
    "slot-reuse": ("host", "counts the ring's room one too many, and writes a slot still pending", "C8"),
    "run-after-failure": ("module", "runs a command that asked to follow a success, after a failure", "C3"),
    "disable-before-drain": ("module", "reports itself stopped while its GEMM still runs, and its results land", "C9"),
    "lost-left-pending": ("host", "ends all but the last slot of a teardown as lost", "C10"),
    "config-while-queued": ("host", "writes PTA_SEED with GEMMs still queued", "C11"),
}


def gemm_seed(seed, i):
    """The map's section 4: GEMM i's seed under PTA_SEED, SplitMix64 as the twin has it."""
    z = (((seed & M32) << 32 | (i & M32)) + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return ((z ^ (z >> 31)) >> 32) & M32


def gemm(A, W, M, K, N, s):
    """What the module returns: the exact sums, and a small error drawn from the
    GEMM's seed, which stands in for the tile's.  So a result can be reproduced
    only from the seed and the index it ran on, as S4's property has it."""
    return [sum(A[m * K + k] * W[k * N + n] for k in range(K)) + ((s >> ((m * N + n) % 30)) & 3) - 1
            for m in range(M) for n in range(N)]


class Sim:
    def __init__(self, fault=None, seed=20261015):
        assert fault is None or fault in FAULTS, fault
        self.fault, self.rng = fault, random.Random(seed)
        self.t, self.q, self.n = 0, [], 0
        self.mem, self.fired = {}, {}
        self.module = Module(self)
        self.host = Host(self)

    def at(self, dt, prio, fn, *args):
        self.n += 1
        heapq.heappush(self.q, (self.t + dt, prio, self.n, fn, args))

    def fail(self, check, why):
        self.fired.setdefault(check, f"tick {self.t}: {why}")

    def line(self, addr):
        return bytes(self.mem.get(addr, bytes(LINE)))

    def store(self, addr, data):            # the host's own store, in its cache
        assert len(data) == LINE
        self.mem[addr] = bytes(data)

    def land(self, addr, data):             # a line the module wrote, ordered at the host
        self.host.landing(addr)
        self.mem[addr] = bytes(data)

    def run(self, limit=60000):
        script, wake = self.host.script(), 0
        while True:
            while self.q and self.q[0][0] <= self.t:
                _, _, _, fn, args = heapq.heappop(self.q)
                fn(*args)
            self.host.watch()
            if self.t >= wake:
                try:
                    wake = self.t + next(script)
                except StopIteration:
                    break
            if self.t > limit:
                self.fail("C12", "the run did not finish")
                break
            self.t += 1
        self.host.finish()
        return self.fired


class Module:
    """The PTA module's queue and tile, as the design has them.  One command at a
    time, in order; a whole line written at once (ItoMWr) and read at once (RdCurr),
    so it keeps no line of the host's."""

    def __init__(self, sim):
        self.sim, self.gen, self.wedge_at, self.resets = sim, 0, None, 0
        self.reset()

    def reset(self):                        # power-on, and a function-level reset
        self.gen += 1
        self.resets += 1
        self.job = 0
        self.enable = self.enabled = self.irq_en = 0
        self.base = self.size_log2 = 0
        self.tail = self.head = self.irq_at = 0
        self.seed = self.gemm_ct = 0
        self.busy, self.inflight, self.prev = False, 0, None
        self.halted, self.wedged, self.draining, self.ghost = 0, False, False, False

    # CXL.io: posted writes arrive in order a crossing later; a read comes back after two.
    def mmio_write(self, off, val):
        self.sim.at(L, 1, self._write, off, val)

    def mmio_read(self, off, box):
        self.sim.at(L, 1, self._read, off, box)

    def _read(self, off, box):
        v = {PTQ_STATUS: self.enabled | (bool(self.halted) << 1) | (self.halted << 8),
             PTQ_HEAD: self.head}.get(off, 0)
        self.sim.at(L, 2, box.append, v)

    def _write(self, off, val):
        s = self.sim
        if self.wedged:
            return
        if off == PTQ_CTRL:
            en, self.irq_en = val & 1, (val >> 1) & 1
            if en and not self.enable:
                self.enable = self.enabled = 1
                self.head, self.prev, self.halted = self.tail, None, 0
                self.kick()
            elif not en and self.enable:
                self.enable = 0
                if s.fault == "disable-before-drain" and self.busy:
                    self.enabled, self.ghost = 0, True      # and the GEMM runs on
                else:
                    self.job += 1                   # what was running is dropped
                    self.busy, self.draining = False, True
                    self.settle()
        elif off in (PTQ_BASE_LO, PTQ_BASE_HI, PTQ_SIZE):
            if not self.enable:
                if off == PTQ_BASE_LO:
                    self.base = (self.base & ~M32) | val
                elif off == PTQ_BASE_HI:
                    self.base = (self.base & M32) | val << 32
                else:
                    self.size_log2 = val
        elif off == PTQ_TAIL:
            if not self.enable:
                self.tail = val
            elif (val - self.head) & M32 > 1 << self.size_log2 or (val - self.tail) & M32 > 1 << 31:
                self.halted = HALT_DOORBELL
            else:
                self.tail = val
                self.kick()
        elif off == PTQ_IRQ_AT:
            self.irq_at = val
        elif off == PTA_SEED:
            if self.head != self.tail or self.busy:
                s.fail("C11", f"PTA_SEED written with {(self.tail - self.head) & M32} slots queued")
            self.seed, self.gemm_ct = val, 0

    def settle(self):
        if self.draining and self.inflight == 0:
            self.enabled, self.draining = 0, False

    def slot(self, idx):
        return self.base + (idx & ((1 << self.size_log2) - 1)) * SLOT

    def kick(self):
        if self.enable and not (self.busy or self.halted or self.wedged) and self.head != self.tail:
            self.busy = True
            self.read_lines([self.slot(self.head)], self.got_descriptor, self.head)

    def read_lines(self, addrs, then, *args):
        """RdCurr of each line: read at the host a crossing or more later, back one after."""
        job, gen, got = self.job, self.gen, {}

        def at_host(a):
            self.sim.at(L, 1, back, a, self.sim.host.as_read_by_module(a))

        def back(a, data):
            if (job, gen) != (self.job, self.gen):
                return
            got[a] = data
            if len(got) == len(addrs):
                then([got[x] for x in addrs], *args)

        for a in addrs:
            self.sim.at(L + self.sim.rng.randint(0, JITTER), 1, at_host, a)

    def write_line(self, addr, data, then=None):
        """ItoMWr of a whole line: ordered at the host a crossing or more later, and
        the module learns it (GO) a crossing after that."""
        self.inflight += 1
        gen, lag = self.gen, L + self.sim.rng.randint(0, JITTER)

        def go():
            if gen != self.gen:
                return
            self.inflight -= 1
            if then:
                then()
            self.settle()

        self.sim.at(lag, 0, self.sim.land, addr, data)
        self.sim.at(lag + L, 1, go)

    def got_descriptor(self, datas, idx):
        f = self.sim.fault
        if idx == self.wedge_at:
            self.wedged = True                      # it stops answering, and stays stopped
            return
        d = datas[0]
        sv, target, fmt, M, K, N, act, wgt, out = DESC.unpack(d)[:9]
        bank, flags = (target >> 8) & 0xFF, target & ~0xFFFF
        usable = (sv == (1 << 16 | 64) and fmt == 0 and M and K and N and
                  not (act | wgt | out) % LINE and not flags & ~AFTER_SUCCESS)
        if not usable:
            return self.end(idx, d, NOT_ISSUED)
        if bank >= BANKS:
            if f == "index-on-refusal":
                self.gemm_ct += 1
            if f == "refused-writes-results":
                return self.start(idx, d, M, K, N, act, wgt, out, REFUSED)
            return self.end(idx, d, DONE if f == "refusal-as-done" else REFUSED)
        if flags & AFTER_SUCCESS and self.prev not in (None, DONE) and f != "run-after-failure":
            return self.end(idx, d, NOT_RUN)
        self.start(idx, d, M, K, N, act, wgt, out, DONE)

    def start(self, idx, d, M, K, N, act, wgt, out, status):
        index = self.gemm_ct
        self.gemm_ct += 1
        wl = [wgt + i * LINE for i in range(lines(K * N * W_BYTES))]
        al = [act + i * LINE for i in range(lines(M * K * A_BYTES))]
        self.read_lines(wl + al, self.shots, idx, d, M, K, N, out, index, self.seed, len(wl), status)

    def shots(self, datas, idx, d, M, K, N, out, index, seed, nw, status):
        W = struct.unpack(f"<{K * N}b", b"".join(datas[:nw])[:K * N])
        A = struct.unpack(f"<{M * K}b", b"".join(datas[nw:])[:M * K])
        raw = struct.pack(f"<{M * N}i", *gemm(A, W, M, K, N, gemm_seed(seed, index)))
        n = lines(len(raw))
        raw += bytes(n * LINE - len(raw))           # the last line's pad is the module's
        job, state = self.job, {"sent": 0, "ordered": 0}
        early = self.sim.fault == "status-before-results"
        need = n - 1 if self.sim.fault == "results-one-line-short" and n > 1 else n

        def ordered():
            state["ordered"] += 1
            if state["ordered"] == need and not early and job == self.job:
                self.end(idx, d, status, index)

        def shot(m):                                # one shot a tick; a line goes once it is full
            if job != self.job:
                return
            full = min(need, n if m == M - 1 else (m + 1) * N * R_BYTES // LINE)
            while state["sent"] < full:
                i = state["sent"]
                state["sent"] += 1
                self.write_line(out + i * LINE, raw[i * LINE:(i + 1) * LINE], ordered)
            if m + 1 < M:
                self.sim.at(1, 1, shot, m + 1)
            elif early:
                self.end(idx, d, status, index)

        self.sim.at(1, 1, shot, 0)

    def end(self, idx, d, status, index=NO_INDEX):
        """The record: line 0 rewritten whole, the host's fields as the module read
        them.  PTQ_HEAD moves, and the interrupt is raised, once it is ordered."""
        rec = d[:0x30] + struct.pack("<IIQ", status, index, 0)
        job, f = self.job, self.sim.fault
        if f == "disable-before-drain" and self.ghost:
            self.ghost = self.busy = False          # its results landed; its record never does
            self.kick()
            return

        def record():
            if f == "irq-before-status":
                self.irq(idx + 1)
            if f == "head-before-status":
                self.ended(idx, status, job)
                self.write_line(self.slot(idx), rec)
            else:
                self.write_line(self.slot(idx), rec, lambda: self.ended(idx, status, job))

        if f == "pad-on-refusal" and status == REFUSED:
            M, N, out = (DESC.unpack(d)[i] for i in (3, 5, 8))
            self.write_line(out + (lines(M * N * R_BYTES) - 1) * LINE, bytes(LINE), record)
        else:
            record()

    def ended(self, idx, status, job):
        self.busy = False
        if job != self.job or not self.enable:
            return
        self.head, self.prev = (idx + 1) & M32, status
        if self.sim.fault != "irq-before-status":
            self.irq(self.head)
        self.kick()

    def irq(self, head):
        if self.irq_en and head == self.irq_at:
            self.sim.at(L, 2, self.sim.host.msi)


class Host:
    """The driver, as the design has it: publish, fence, ring; poll the oldest slot
    or wait for its interrupt; bound every wait; tear down and reset what wedges."""

    def __init__(self, sim):
        self.sim, self.m = sim, sim.module
        self.size = 1 << RING_LOG2
        self.next = self.hhead = self.origin = 0
        self.cmds, self.slot_of, self.owner, self.stale = {}, {}, {}, {}
        self.ended, self.alloc, self.seed = set(), BUF, 0
        self.last, self.restart, self.slack = None, True, False
        self.msis = self.timeouts = self.teardowns = 0
        self.irq_at, self.ctrl, self.fence_victim = 0, 1, None
        self.ends = [0] * len(NAMES)

    def addr(self, idx):
        return RING + (idx & (self.size - 1)) * SLOT

    def status(self, idx):
        return struct.unpack_from("<I", self.sim.line(self.addr(idx)), 0x30)[0]

    def write(self, off, val):
        if off == PTQ_CTRL:
            self.ctrl = val
        self.m.mmio_write(off, val)

    def read(self, off):
        box = []
        self.m.mmio_read(off, box)
        while not box:
            yield 1
        return box[0]

    def set_seed(self, seed):
        self.write(PTA_SEED, seed)
        self.seed, self.restart = seed, True

    def ring(self):
        self.write(PTQ_TAIL, self.next)

    def take(self, nbytes, fill=0, data=b""):
        a = self.alloc
        n = lines(nbytes)
        raw = data + bytes([fill]) * (n * LINE - len(data))
        for i in range(n):
            self.sim.store(a + i * LINE, raw[i * LINE:(i + 1) * LINE])
        self.alloc += n * LINE
        return a, n

    def cmd(self, rng, M=0, K=0, N=0, bank=None, flags=0, fmt=0, expect=DONE, teardown=False):
        M, K, N = M or rng.randint(1, 4), K or rng.randint(1, 8), N or rng.randint(1, 4)
        return dict(M=M, K=K, N=N, bank=rng.randint(0, 1) if bank is None else bank, flags=flags,
                    fmt=fmt, expect=expect, teardown=teardown,
                    A=[rng.randint(-128, 127) for _ in range(M * K)],
                    W=[rng.randint(-128, 127) for _ in range(K * N)])

    def room(self):
        return self.size - (self.next - self.hhead) + (self.sim.fault == "slot-reuse")

    def publish(self, c):
        while self.room() <= 0:
            if not self.try_end():
                yield 1
        M, K, N, idx = c["M"], c["K"], c["N"], self.next
        act, _ = self.take(M * K, data=struct.pack(f"<{M * K}b", *c["A"]))
        wgt, _ = self.take(K * N, data=struct.pack(f"<{K * N}b", *c["W"]))
        out, n = self.take(M * N * R_BYTES, fill=SENTINEL)
        for i in range(n):
            self.owner[out + i * LINE] = idx
        c.update(out=out, out_lines=n, seed=self.seed, desc=DESC.pack(
            1 << 16 | 64, c["bank"] << 8 | c["flags"], c["fmt"], M, K, N, act, wgt, out, 0, 0, 0))
        slot = idx & (self.size - 1)
        prev = self.slot_of.get(slot)
        if prev is not None and prev not in self.ended:
            self.sim.fail("C8", f"slot {slot} written for command {idx} while command {prev} is pending")
        if self.sim.fault == "doorbell-before-stores" and idx == self.fence_victim:
            self.stale[self.addr(idx)] = self.sim.line(self.addr(idx))
        self.sim.store(self.addr(idx), c["desc"])
        self.slot_of[slot], self.cmds[idx] = idx, c
        self.next += 1

    def as_read_by_module(self, addr):
        """What a read of host memory returns.  With the fence left out, the slot
        the host has just written is read as it was before the store."""
        return self.stale.pop(addr) if addr in self.stale else self.sim.line(addr)

    def try_end(self):
        if self.hhead == self.next or self.status(self.hhead) == PENDING:
            return False
        self.complete(self.hhead, self.sim.line(self.addr(self.hhead)))
        return True

    def complete(self, idx, rec):
        s, c = self.sim, self.cmds[idx]
        status, index = struct.unpack_from("<II", rec, 0x30)
        if rec[:0x30] != c["desc"][:0x30]:
            s.fail("C7", f"command {idx} ended with fields the host did not write")
        if status not in {c["expect"]} | ({LOST} if c["teardown"] else set()):
            s.fail("C3", f"command {idx} ended {NAMES[min(status, 5)]}, not {NAMES[c['expect']]}")
        out = b"".join(s.line(c["out"] + i * LINE) for i in range(c["out_lines"]))
        if status == DONE:
            M, K, N = c["M"], c["K"], c["N"]
            want = gemm(c["A"], c["W"], M, K, N, gemm_seed(c["seed"], index))
            if list(struct.unpack_from(f"<{M * N}i", out)) != want:
                s.fail("C1", f"command {idx} ended done without its results")
            if self.restart:
                ok = index == 0
            elif self.slack:
                ok = index in (self.last + 1, self.last + 2)
            else:
                ok = index == self.last + 1
            if not ok:
                s.fail("C4", f"command {idx} ran as GEMM {index} after GEMM {self.last}")
            self.last, self.restart, self.slack = index, False, False
        elif status in (REFUSED, NOT_ISSUED, NOT_RUN):
            if out != bytes([SENTINEL]) * len(out):
                s.fail("C2", f"command {idx} ended {NAMES[status]} with its results written")
            if index != NO_INDEX:
                s.fail("C4", f"command {idx} ended {NAMES[status]} and reported GEMM {index}")
        self.ends[min(status, 5)] += 1
        self.ended.add(idx)
        self.hhead = idx + 1

    def watch(self):
        """C5, every tick: what a read of PTQ_HEAD would say, against the slots."""
        m = self.m
        if not m.enable:
            return
        for idx in range(max(self.origin, m.head - self.size), m.head):
            if self.slot_of.get(idx & (self.size - 1)) == idx and idx not in self.ended \
                    and self.status(idx) == PENDING:
                self.sim.fail("C5", f"PTQ_HEAD is {m.head} and slot of command {idx} is pending")

    def landing(self, addr):
        """C9: nothing the module writes may land on what the host has ended."""
        idx = self.owner.get(addr)
        if idx is not None and idx in self.ended:
            self.sim.fail("C9", f"a result line of command {idx} landed after it ended")
        if RING <= addr < RING + self.size * SLOT:
            idx = self.slot_of.get((addr - RING) // SLOT)
            if idx is not None and idx in self.ended:
                self.sim.fail("C9", f"the slot of command {idx} was written after it ended")

    def msi(self):
        self.msis += 1
        for idx in range(self.hhead, self.irq_at):
            if idx not in self.ended and self.status(idx) == PENDING:
                self.sim.fail("C6", f"an interrupt for {self.irq_at} found command {idx} pending")

    def drain(self):
        quiet = self.sim.t
        while self.hhead < self.next:
            if self.try_end():
                quiet = self.sim.t
            elif self.sim.t - quiet > T_WAIT:
                self.timeouts += 1
                yield from self.teardown()
                quiet = self.sim.t
            else:
                yield 1

    def await_irq(self, n):
        quiet = self.sim.t
        while self.msis < n and self.sim.t - quiet <= T_WAIT:
            yield 1
        yield from self.drain()

    def end_lost(self):
        """The slots the module will not end are the host's again: it ends them lost."""
        last = self.next - (self.sim.fault == "lost-left-pending")
        for idx in range(self.hhead, last):
            rec = bytearray(self.sim.line(self.addr(idx)))
            struct.pack_into("<I", rec, 0x30, LOST)
            self.sim.store(self.addr(idx), bytes(rec))
        for idx in range(self.hhead, self.next):
            if self.status(idx) == PENDING:
                self.sim.fail("C10", f"command {idx} left pending after a teardown")
                self.ended.add(idx)
                self.hhead = idx + 1
            else:
                self.complete(idx, self.sim.line(self.addr(idx)))
        self.slack = True                           # a GEMM it dropped took an index

    def restart_queue(self):
        self.write(PTQ_TAIL, self.next)             # while disabled: where the ring starts
        self.write(PTQ_CTRL, self.ctrl | 1)
        self.origin = self.next

    def teardown(self):
        self.teardowns += 1
        self.write(PTQ_CTRL, self.ctrl & ~1)
        t0 = self.sim.t
        while (yield from self.read(PTQ_STATUS)) & 1:
            if self.sim.t - t0 > T_QUIESCE:
                yield from self.reset()
                return
        head = yield from self.read(PTQ_HEAD)       # the module has stopped writing
        while self.hhead < head and self.try_end():
            pass
        self.end_lost()
        self.restart_queue()

    def reset(self):
        """A function-level reset, over configuration space; then the queue again."""
        self.sim.at(L, 1, self.m.reset)
        yield 2 * L
        for off, val in ((PTQ_BASE_LO, RING), (PTQ_BASE_HI, 0), (PTQ_SIZE, RING_LOG2)):
            self.write(off, val)
        self.set_seed(self.seed)
        self.end_lost()
        self.restart = True
        self.restart_queue()
        yield 1

    def script(self):
        f, rng = self.sim.fault, random.Random(7)   # the commands' own draws
        for off, val in ((PTQ_BASE_LO, RING), (PTQ_BASE_HI, 0), (PTQ_SIZE, RING_LOG2)):
            self.write(off, val)
        self.set_seed(20261015)
        self.restart_queue()
        yield 1
        # 1. One at a time, polled; two with eight lines of results.
        for i in range(6):
            yield from self.publish(self.cmd(rng, M=8, N=16) if i in (1, 4) else self.cmd(rng))
            self.ring()
            yield from self.drain()
        # 2. A refusal, one that waits on it, one the module cannot use, two that run.
        for c in (self.cmd(rng, M=8, N=16, bank=3, expect=REFUSED), self.cmd(rng, flags=AFTER_SUCCESS, expect=NOT_RUN),
                  self.cmd(rng, fmt=7, expect=NOT_ISSUED), self.cmd(rng), self.cmd(rng, flags=AFTER_SUCCESS)):
            yield from self.publish(c)
        self.ring()
        yield from self.drain()
        # 3. A stream, the ring kept full.
        for i in range(14):
            if i == 9:
                self.fence_victim = self.next
            yield from self.publish(self.cmd(rng))
            self.ring()
        if f == "config-while-queued":
            self.set_seed(20261016)
        yield from self.drain()
        # 4. A new seed with the queue drained; six GEMMs, each waited on by interrupt.
        if f != "config-while-queued":
            self.set_seed(20261016)
        self.write(PTQ_CTRL, 3)
        for _ in range(6):
            yield from self.publish(self.cmd(rng))
            self.irq_at = self.next
            self.write(PTQ_IRQ_AT, self.next)
            self.ring()
            yield from self.await_irq(self.msis + 1)
        self.write(PTQ_CTRL, 1)
        # 5. A teardown with four queued, the first a long GEMM still running.
        for c in [self.cmd(rng, M=40, K=8, N=16, teardown=True)] + \
                [self.cmd(rng, teardown=True) for _ in range(3)]:
            yield from self.publish(c)
        self.ring()
        yield 9 * L
        yield from self.teardown()
        # 6. A module that stops answering: the second of three wedges it.
        self.m.wedge_at = self.next + 1
        for c in (self.cmd(rng), self.cmd(rng, teardown=True), self.cmd(rng, teardown=True)):
            yield from self.publish(c)
        self.ring()
        yield from self.drain()
        # 7. After the reset, four more.
        for _ in range(4):
            yield from self.publish(self.cmd(rng))
        self.ring()
        yield from self.drain()

    def finish(self):
        if self.hhead != self.next:
            self.sim.fail("C12", f"{self.next - self.hhead} commands never ended")
        if self.timeouts != 1:
            self.sim.fail("C12", f"{self.timeouts} timeouts, where the wedged module is the one")


OUTCOME = {"done": 33, "refused": 1, "not issued": 1, "not run": 1, "lost": 6}


def protocol():
    section("6. The protocol, run: 42 commands, a link that reorders, and twelve checks")
    sim = Sim()
    fired = sim.run()
    h = sim.host
    got = {NAMES[s]: h.ends[s] for s in (DONE, REFUSED, NOT_ISSUED, NOT_RUN, LOST)}
    assert not fired, fired
    assert got == OUTCOME, got
    assert (h.msis, h.teardowns, h.timeouts, sim.module.resets - 1) == (6, 2, 1, 1)
    print(f"With nothing planted, after {sim.t:,} ticks of {L} to a crossing: every check holds.")
    print("Ended " + ", ".join(f"{v} {k}" for k, v in got.items()) + ": six by interrupt, two")
    print("teardowns, one of them a module that stopped answering and was reset.")
    print("| Check | What it holds |")
    print("|---|---|")
    for k, v in CHECKS.items():
        print(f"| {k} | {v} |")
    print("\nEach fault planted alone, and the checks that named it:")
    print("| Fault | In | What it does | Written for | Fired |")
    print("|---|---|---|---|---|")
    missed = []
    for name, (who, what, want) in FAULTS.items():
        fired = Sim(name).run()
        if want not in fired:
            missed.append(name)
        print(f"| `{name}` | {who} | {what} | {want} | {', '.join(sorted(fired, key=lambda c: int(c[1:])))} |")
    assert not missed, missed
    print("Every fault was named by the check written for it.")


def main():
    if "--fault" in sys.argv[1:]:
        name = sys.argv[sys.argv.index("--fault") + 1]
        fired = Sim(name).run()
        for k in sorted(fired, key=lambda c: int(c[1:])):
            print(f"{k}  {fired[k]}")
        sys.exit(0 if FAULTS[name][2] in fired else 1)
    assert gemm_seed(1, 0) == 3291240986 and gemm_seed(0xC0FFEE, 7) == 1993064625 \
        and gemm_seed(0, M32) == 1940994978       # the twin's gate pins the same three
    slot_layout()
    registers()
    jobs()
    link()
    crossings()
    protocol()


if __name__ == "__main__":
    main()
