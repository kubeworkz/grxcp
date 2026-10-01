"""
F1 of pta_program_plan.md: the operand feed added to section 2.1's cost model.

Section 2.1 prices a GEMM by the core's own cycles.  F0 measured what the DMA
adds around them on grx930's c930_npu_top at M=64 N=8 K=256 INT8 -- the NPU
bench tb/tb_npu_feed.sv (`make npu_feed`), whose AXI memory answers a read one
cycle after its AR, and the Verilator SoC built with F0=1, whose reads cross the
DMA arbiter, the crossbar and the L2 to the same memory model.  Two cores were
measured: grx930 8cceb33, whose S_RUN is 18 cycles a K tile, and the core after
the half-rate hop (0afeb6e), whose S_RUN is 64.  The DMA did not change
between them, and neither did anything F0 measured of it:

                                   NPU bench     SoC
    initial load: A row 0, B         66 + 514   69 + 517 cycles
    PF1 busy, per A row (rows 1..63)       34         37
    C writeback, 512 words              1,283      1,410
    core -> DMA hand-off                    3          3
    P_DONE                                  1          1
    P_DONE drain when PF2 was running     145        132

    S_RUN 18: core cycles              71,734     71,923
                of which S_AROW           566        755
              whole GEMM               73,601     73,923
    S_RUN 64: core cycles             165,375    165,375
                of which S_AROW             0          0
              whole GEMM              167,242    167,375

(The SoC's first GEMM after boot took 8 more PF1 cycles, and so 8 more S_AROW
cycles at S_RUN 18.)

The model, checked against all of it:

    GEMM = load + core + S_AROW + 3 + writeback + 1   (+ drain if PF2 ran)
    DMA_LAST = GEMM - 1

core is section 2.1's count for the loop order (pta_tw_sweep.py), restore
unfolded as the RTL is built.  S_AROW comes from a race that F0's per-row
timeline on the NPU bench pins down with no free constant.  PF1 lands A row r
at 1 + r*P cycles after launch, where P is its busy cycles per row plus 2 idle
ones -- the deferred watermark increment and the restart -- so rows land every
36 cycles on the bench.  The core starts its first K tile 2 cycles after launch
and wants row r when it finishes row r-1; a core that has to wait starts the
row the cycle after it lands.  Under the interchanged order only the first K
tile races, since by its second every row has landed: its rows start after
that tile's weight program, each taking Ts + Td.  Under the shipped order every
row races, each taking Kt*(Tw + Ts) + Td.

A first version of this model took P as PF1's busy cycles alone and fitted a
row offset to the S_RUN 18 wait; it matched there and predicted 27 cycles of
S_AROW at S_RUN 64, where the RTL measured none.  The two idle cycles a row
were what the fit had been hiding.

Assumed, and checked only when C2 tile and MB report: that PF1 keeps its
period whatever the core does -- as it did across the hop -- that S_AROW gates
the shipped order the way it gates the interchanged one, and that the tile
changes nothing about load and writeback.  A GEMM queued behind another pays
the PF2 drain as well: PF2 as built cannot finish inside a 512-word writeback,
so it saves nothing.  That drain was 145 cycles when F0 measured it and is 3
now -- not because PF2 improved but because the write burst got shorter, so PF2
issues one AXI burst instead of several and leaves 3 beats in flight rather than
many (section 6).

Standard library only.  Run:  python3 docs/designs/pta_feed_model.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_tw_sweep as sweep

M, NT, KT, SCAN, TD = sweep.M, sweep.NT, sweep.KT, sweep.SCAN, sweep.TD
HANDOFF = 3          # P_LAUNCH cycles beyond the core's own
DONE = 1
PF1_IDLE = 2         # idle cycles between one A row landing and the next row's read
FIRST_ROW = 1        # A row r lands at FIRST_ROW + r * period cycles after launch
CORE_START = 2       # the core's first K tile starts this many cycles after launch
HOP_RUN = 64         # S_RUN per K tile after the half-rate hop

# ---- the DMA's own costs, read off c930_npu_dma.sv (F2) --------------------
# Not fitted: each is a count of states the machine passes through per beat.
BYTES_PER_BEAT = 8   # 64-bit AXI
READ_BEAT = 2        # RS_R latches the beat, RS_UNPACK writes it through the
                     # wide port.  PF1's path does a beat a cycle; P_READ_A and
                     # P_READ_B kept the second cycle, since they run once per
                     # GEMM rather than once per output row.
READ_SETUP = 2       # the AR handshake
# The C write burst, before and after F2 rebuilt it.  WALK is what F0 measured:
# WS_ADDR, WS_DATA, WS_PACK, then WS_DRIVE twice -- one cycle to set
# m_axi_wvalid and one for the handshake -- because the beat's two words came
# through one read port a word at a time.  STREAM is the second read port
# (o_c_rdata_hi): one address yields a whole beat, so the burst runs at the W
# channel's own rate and the only extra cycle is the pipeline's first fill.
WRITE_BEAT_WALK = 5
WRITE_BEAT = 1
WRITE_SETUP = 3      # AW, and B at the end
WRITE_FILL = 1       # WS_STREAM presents beat 0 the cycle after it loads it
WORDS_PER_BEAT = 2   # C is INT32, two to a 64-bit beat

F0_SHAPE = (sweep.M, sweep.N, sweep.K)      # 64, 8, 256 -- what F0 measured
SOC_SHAPE = (8, 12, 16)                     # what pta_sweep.c and pta_feed.c run

# ---- measured, F2 (`make pta_feed PTM_B=1`, SOC_SHAPE, 2026-09-30) ---------
# Per GEMM at Q = 1, from NPU_REG_DMA_CT (the whole GEMM: DMA_CT counts
# P_LAUNCH) and NPU_REG_CYCLE_LO (the core).  "walk" is the five-cycle write
# burst, "stream" the one-cycle one.
# wall_walk is None for EO-scan: the only Q = 1 EO-scan batch in the run that
# measured the walking burst was the first one, and it paid ~290 cycles for the
# CPU's cold I-cache.  pta_feed.c now runs a discarded warm-up batch first, but
# that fix and the new burst landed together, so there is no warm pre-change
# wall at this level to compare against.  EO-res's was warm.
MEASURED_F2 = {
    "EO-scan": dict(core=576, gemm_walk=909, gemm=721, wall_walk=None, wall=1_823),
    "EO-res":  dict(core=388, gemm_walk=721, gemm=533, wall_walk=1_807, wall=1_624),
}
# STAGE_A's cost, measured identically at both levels and both write bursts:
# 14 more beats of A read before launch, at the read path's 2 cycles a beat.
MEASURED_F2_STAGE_A = 28
# The A-row wait, at every point, in every mode, both write bursts.  PF1 never
# makes this core wait at this shape, which is why staging cannot help.
MEASURED_F2_AROW = 0
# Weight movement, EO-scan -> EO-res: what WSKIP saves, and exactly the gap
# between the two cores (576 - 388).
MEASURED_F2_WMOVE = (288, 100)


def beats_of(elems, elem_bytes):
    """AXI beats to move `elems` elements of `elem_bytes` each."""
    return -(-(elems * elem_bytes) // BYTES_PER_BEAT)


def c_beats_of(words):
    return -(-words // WORDS_PER_BEAT)


def feed_part(beats, kind, walk=False):
    """Cycles for one DMA phase.  `walk` prices the retired write burst."""
    if kind == "read":
        return beats * READ_BEAT + READ_SETUP
    if walk:
        return beats * WRITE_BEAT_WALK + WRITE_SETUP
    return beats * WRITE_BEAT + WRITE_SETUP + WRITE_FILL

# ---- measured, F0 ----------------------------------------------------------
LEVELS = {
    "NPU bench": dict(read_a=66, read_b=514, pf1_busy=2_142, write_c=1_283, drain=145),
    "SoC": dict(read_a=69, read_b=517, pf1_busy=2_331, write_c=1_410, drain=132),
}
MEASURED = {
    # (level, S_RUN): (core cycles, S_AROW, whole GEMM)
    ("NPU bench", sweep.DIGITAL_RUN): (71_734, 566, 73_601),
    ("SoC", sweep.DIGITAL_RUN): (71_923, 755, 73_923),
    ("NPU bench", HOP_RUN): (165_375, 0, 167_242),
    ("SoC", HOP_RUN): (165_375, 0, 167_375),
}

# ---- measured, F2, on the NPU bench after the write burst was rebuilt -------
# `make npu_feed`, 2026-09-30, same shape and same core as the HOP_RUN rows
# above, so these differ from them by the write burst and nothing else.  The
# bench reports the phase breakdown directly, which is why this is the clean
# reading: 256 beats either way (axi wbeats=256), 1,283 cycles against 260.
BENCH_F2 = dict(write_c=260, gemm=166_219, read_a=66, read_b=514, arow=0,
                wbeats=256,
                # P_DONE with PF2 running against without: done=4 vs done=1,
                # drain_beats=3 vs 0.  This is PF2's whole cost, and the only
                # instrument that can see it (section 6).
                drain=3, pf2_busy=260, pf2_ars=1, pf2_beats=29)

# ---- measured, PF2_OFF (PTA_CTRL bit 11), SoC harness, 2026-10-01 -----------
# Wall cycles over four queued GEMMs, PF2 off against PF2 as built.  Both are
# inside the harness's own floor; MEASURED_F2_WALL_DRIFT is why.
MEASURED_PF2_OFF_Q4 = {"EO-scan": +4, "EO-res": -8}
# The same run, with four batches inserted ahead of the later ones: every core
# and DMA counter held bit-identical while walls moved by up to this much.  The
# C block addresses and the cache state travel with execution history, so a wall
# difference smaller than this says nothing at all.
MEASURED_F2_WALL_DRIFT = 45
# And the no-op that PF2_OFF must be at Q = 1, where i_next_valid is low:
# GEMM and core unchanged at both levels.
MEASURED_PF2_OFF_Q1_IS_NOOP = True


def period(level):
    p = LEVELS[level]
    assert p["pf1_busy"] % (M - 1) == 0, (level, "PF1 is not a whole number of cycles a row")
    return p["pf1_busy"] // (M - 1) + PF1_IDLE


def race(start, row_time, per, m=M):
    """S_AROW cycles: the core wants row r when row r-1 is done, row 0 at `start`."""
    t, wait = start, 0
    for r in range(1, m):
        t += row_time
        lands = FIRST_ROW + r * per
        if lands >= t:
            wait += lands + 1 - t
            t = lands + 1
    return wait


def margin(start, row_time, per, m=M):
    """The race's slack: extra per-row feed latency the core absorbs before stalling.

    race() returns what the core waits when PF1 is late.  This returns how early
    PF1 is at its tightest row, which is the quantity F3 owes the fabric plan: a
    link that adds L cycles to every row's arrival shifts every `lands` by L, so
    for L <= margin no row is late and for L > margin at least one is.  It is a
    first-order bound -- once a row is late the stall cascades -- and that is the
    direction a link wants to be told about.

    Zero or negative means the core already waits, and then race() is the number
    that matters instead.
    """
    t, out = start, None
    for r in range(1, m):
        t += row_time
        lands = FIRST_ROW + r * per
        slack = t - lands
        out = slack if out is None else min(out, slack)
        if lands >= t:
            t = lands + 1
    return out


def arow_interchanged(tw, ts, td, per):
    return race(CORE_START + tw, ts + td, per)


def margin_interchanged(tw, ts, td, per):
    return margin(CORE_START + tw, ts + td, per)


def margin_shipped(tw, ts, td, per):
    return margin(CORE_START, KT * (tw + ts) + td, per)


def arow_shipped(tw, ts, td, per):
    return race(CORE_START, KT * (tw + ts) + td, per)


def gemm(level, core, arow, queued=False):
    p = LEVELS[level]
    total = p["read_a"] + p["read_b"] + core + arow + HANDOFF + p["write_c"] + DONE
    return total + (p["drain"] if queued else 0)


def gemm_stream(level, core, arow, shape=None, queued=False):
    """A GEMM as the RTL stands after F2: F0's measured loads, and the rebuilt
    write burst from the per-beat model (section 5), which the NPU bench measured
    at exactly the model's 260 cycles.

    gemm() above is kept as F0 measured it, because section 1 checks the model
    against those numbers and they belong to the write burst that has been
    retired.  Everything quoting the machine as it is now uses this.
    """
    p = LEVELS[level]
    m, n, _ = shape or F0_SHAPE
    wr = feed_part(c_beats_of(m * n), "write")
    total = p["read_a"] + p["read_b"] + core + arow + HANDOFF + wr + DONE
    # PF2's drain, 3 cycles since the burst shortened, not F0's 145 (section 6).
    return total + (BENCH_F2["drain"] if queued else 0)


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    # 1. The model against F0, both cores.  Exact, or stop.
    section("1. Model against F0, interchanged order, restore unfolded")
    for (level, run), (core_m, arow_m, gemm_m) in MEASURED.items():
        per = period(level)
        core = sweep.interchanged(SCAN, run, TD, restore=True)
        arow = arow_interchanged(SCAN, run, TD, per)
        total = gemm(level, core, arow)
        # The hop's free-running phase can shorten a GEMM's first S_RUN by a cycle.
        slack = 1 if run == HOP_RUN else 0
        print(f"  {level:<10} S_RUN {run:>2}, PF1 every {per}:  S_AROW {arow} (measured {arow_m});"
              f"  core {core + arow:,} (measured {core_m:,});  GEMM {total:,} (measured {gemm_m:,})")
        assert arow == arow_m, (level, run, arow, arow_m)
        assert 0 <= core + arow - core_m <= slack, (level, run, core + arow, core_m)
        assert 0 <= total - gemm_m <= slack, (level, run, total, gemm_m)
    print("  exact at both levels on both cores (to the hop's one-cycle phase), no fitted constant")

    # 2. Predictions at the section 6.2 points, for C2 tile and MB to check.
    points = (
        # name, PTA_TW, PTA_TS, weights scanned by the core?
        ("TO-1ms", 100_000, 5, True),
        ("TO-10us", 1_000, 5, True),
        ("EO-scan", 0, 1, True),
        ("EO-res", 0, 1, False),
        ("EO-res, 1-cycle select", 1, 1, False),
    )
    for order in ("interchanged", "shipped"):
        who = "C2 tile" if order == "interchanged" else "MB"
        section(f"2{'a' if order == 'interchanged' else 'b'}. Predicted, {order} order"
                f" (checked by {who}); restore unfolded, GEMM not queued")
        print(f"  a shot costs PTA_TS + {sweep.SHOT_FLOOR} cycles (C2, measured);"
              f" the core term is the band's upper end, within one cycle a shot")
        print(f"  {'point':<24}{'level':<11}{'core':>13}{'S_AROW':>8}{'GEMM':>13}"
              f"{'DMA_LAST':>13}{'feed':>7}")
        for name, pta_tw, pta_ts, scanned in points:
            if order == "interchanged" and not scanned and pta_tw:
                continue
            tw = (SCAN if scanned else 0) + pta_tw
            # What a shot costs the core, which C2 measured: the register plus the
            # hop-gated feed, the hop-aligned capture and the registered valid.
            ts = pta_ts + sweep.SHOT_FLOOR
            for level in LEVELS:
                per = period(level)
                if order == "interchanged":
                    core = sweep.interchanged(tw, ts, TD, restore=True)
                    arow = arow_interchanged(tw, ts, TD, per)
                else:
                    core = sweep.shipped(tw, ts, TD)
                    arow = arow_shipped(tw, ts, TD, per)
                total = gemm(level, core, arow)
                feed = total - core
                print(f"  {name:<24}{level:<11}{core:>13,}{arow:>8,}{total:>13,}"
                      f"{total - 1:>13,}{feed / total:>7.1%}")

    # 2c. The core term against C2's own gate, so the two models cannot drift.
    section("2c. The interchanged core term against C2's measurement")
    for name, pta_tw, pta_ts, scanned in points:
        if not scanned and pta_tw:
            continue
        tw = (SCAN if scanned else 0) + pta_tw
        here = sweep.interchanged(tw, pta_ts + sweep.SHOT_FLOOR, TD, restore=True)
        there = sweep.core_cycles(pta_tw, pta_ts, scanned=scanned)
        assert here == there, (name, here, there)
        got = sweep.MEASURED_C2.get(name)
        band = sweep.core_band()
        flag = "-"
        if got is not None:
            assert here - band <= got <= here, (name, here, got)
            flag = f"{got:,}"
        print(f"  {name:<24}model {here:>10,}  measured {flag:>10}"
              f"  (band {band:,})")

    # 3. What the feed does to the loop-order verdict at the resident point.
    section("3. EO-res, whole GEMM: shipped against interchanged")
    for level in LEVELS:
        per = period(level)
        ts = 1 + sweep.SHOT_FLOOR
        s_core = sweep.shipped(0, ts, TD)
        i_core = sweep.interchanged(0, ts, TD, restore=True)
        i_fold = sweep.interchanged(0, ts, TD)
        s = gemm(level, s_core, arow_shipped(0, ts, TD, per))
        i = gemm(level, i_core, arow_interchanged(0, ts, TD, per))
        i_f = gemm(level, i_fold, arow_interchanged(0, ts, TD, per))
        print(f"  {level:<10} shipped {s:,} vs interchanged {i:,} ({i / s:.1f}x;"
              f" {i_f:,} and {i_f / s:.1f}x with the restore folded).  Core alone:"
              f" {i_core / s_core:.1f}x, {i_fold / s_core:.1f}x folded")

    section("4. Queued GEMMs")
    for level, p in LEVELS.items():
        print(f"  {level:<10} +{p['drain']} cycles each for PF2's drain, as F0 measured it,"
              " at every point: writeback was 512 words regardless of the tile")
    print(f"  Since the write burst was rebuilt that drain is {BENCH_F2['drain']} cycles"
          f" (section 6), so this row is history.")

    # 5. The feed's parts from the DMA's structure, not from a fit (F2).
    section("5. The feed per beat, read off c930_npu_dma.sv and checked against F0")
    print("  F0's load and writeback were three measured constants per level.  They")
    print("  are not constants: they are beat counts times a per-beat cost the state")
    print("  machine fixes, which is what lets F2 price the feed at a shape F0 never")
    print("  ran and F3 quote a rate rather than one SoC's numbers.")
    print()
    print(f"    reads   {READ_BEAT} cycles a beat   RS_R latches the beat, RS_UNPACK writes")
    print( "                              all of it through the wide port (one cycle")
    print( "                              per beat on PF1's path, which restructured it)")
    print(f"    write   {WRITE_BEAT_WALK} cycles a beat   WS_ADDR, WS_DATA, WS_PACK, then WS_DRIVE")
    print( "            (retired)         twice -- set m_axi_wvalid, then the handshake")
    print(f"    write   {WRITE_BEAT} cycle a beat    WS_STREAM: one address yields the whole")
    print( "            (F2)              beat from c_mem's two read ports")
    print(f"    AR/AW   +{READ_SETUP} / +{WRITE_SETUP}          the address handshake, and B for a write")
    print()
    print(f"  {'part':<22}{'beats':>7}{'model':>9}{'measured':>10}{'level':>12}")
    for level in LEVELS:
        p = LEVELS[level]
        for name, beats, want, kind in (
            ("A row 0, K=256 INT8", beats_of(sweep.K, 1), p["read_a"], "read"),
            ("B, K*N bytes", beats_of(sweep.K * sweep.N, 1), p["read_b"], "read"),
            ("C, M*N INT32 words", c_beats_of(sweep.M * sweep.N), p["write_c"], "write"),
        ):
            # F0 ran the walking write burst, so that is what its numbers check.
            got = feed_part(beats, kind, walk=True)
            print(f"  {name:<22}{beats:>7}{got:>9,}{want:>10,}{level:>12}")
            # The NPU bench's AXI answers a read the cycle after its AR, so the
            # structure is exact there.  The SoC's reads cross the DMA arbiter,
            # the crossbar and the L2, which adds latency the state machine does
            # not contain -- so there it is a floor, and the gap is the fabric's.
            if level == "NPU bench":
                assert got == want, (level, name, got, want)
            else:
                assert want >= got, (level, name, got, want)
    print("  exact on the NPU bench, a floor on the SoC: the difference is the")
    print("  crossbar and L2, which the state machine does not contain.")
    print()
    # The rebuilt burst on the same bench, same shape, same core -- so the only
    # thing that changed is the write burst, and the model has to get it right
    # for the same reason it got the walking one right.
    cb = c_beats_of(sweep.M * sweep.N)
    want = feed_part(cb, "write")
    print(f"  the rebuilt burst, NPU bench, same shape and core:"
          f"  C model {want} measured {BENCH_F2['write_c']}"
          f"   ({cb} beats, AXI reported {BENCH_F2['wbeats']})")
    assert want == BENCH_F2["write_c"], (want, BENCH_F2["write_c"])
    assert BENCH_F2["wbeats"] == cb, (BENCH_F2["wbeats"], cb)
    # And the whole GEMM must fall by exactly the writeback's saving: nothing
    # else in the GEMM was touched, so any other delta would be a side effect.
    walked = feed_part(cb, "write", walk=True)
    before = MEASURED[("NPU bench", HOP_RUN)][2]
    assert before - BENCH_F2["gemm"] == walked - want, \
        (before - BENCH_F2["gemm"], walked - want)
    print(f"  and the whole GEMM fell {before:,} -> {BENCH_F2['gemm']:,}, which is"
          f" {walked - want:,} -- exactly the burst's saving and nothing else")
    print("  exact for BOTH bursts with no fitted constant, which is what makes")
    print("  the per-beat costs usable at shapes nobody has run (F3).")

    section("5b. What rebuilding the write burst is worth (F2)")
    print("  c_mem is read combinationally (o_c_rdata = c_mem[i_c_raddr]) and is")
    print("  distributed RAM precisely so that it can be, so the five cycles were")
    print("  the state machine's, not the memory's.  A second asynchronous read one")
    print("  word up fills a beat from one address; advancing that address only on")
    print("  a W handshake makes the whole thing a pipeline with no skid buffer.")
    print()
    print(f"  {'shape':<28}{'C walk':>9}{'C stream':>10}{'feed':>9}{'->':>4}"
          f"{'feed':>8}{'change':>9}")
    for shape, label in ((F0_SHAPE, "F0's, M=64 N=8 K=256"),
                         (SOC_SHAPE, "the SoC's, M=8 N=12 K=16")):
        m, n, k = shape
        rd_a = feed_part(beats_of(k, 1), "read")
        rd_b = feed_part(beats_of(k * n, 1), "read")
        cb = c_beats_of(m * n)
        wr_walk = feed_part(cb, "write", walk=True)
        wr_strm = feed_part(cb, "write")
        rest = rd_a + rd_b + HANDOFF + DONE
        print(f"  {label:<28}{wr_walk:>9,}{wr_strm:>10,}{rest + wr_walk:>9,}"
              f"{'->':>4}{rest + wr_strm:>8,}"
              f"{(rest + wr_strm) / (rest + wr_walk) - 1:>+9.0%}")
        print(f"    writeback was {wr_walk / (rest + wr_walk):.0%} of the feed and is"
              f" now {wr_strm / (rest + wr_strm):.0%}; A row 0 {rd_a}, B {rd_b},"
              f" hand-off {HANDOFF}, done {DONE}")
    print()
    print("  This is a floor.  At five cycles a beat the burst never asked the")
    print("  crossbar and L2 for more than a fifth of the W channel; at one it asks")
    print("  for all of it, and whatever they will not take shows up as wready low.")

    # 5c. The floor against the SoC, measured both ways.  The question the model
    # could not answer was whether the write path would sustain a beat a cycle.
    section("5c. F2 measured: did the fabric take a beat a cycle?")
    m, n, k = SOC_SHAPE
    rest = (feed_part(beats_of(k, 1), "read") + feed_part(beats_of(k * n, 1), "read")
            + HANDOFF + DONE)
    floor_walk = rest + feed_part(c_beats_of(m * n), "write", walk=True)
    floor_strm = rest + feed_part(c_beats_of(m * n), "write")
    print(f"  {'point':<10}{'feed walk':>11}{'floor':>8}{'over':>7}"
          f"{'feed stream':>13}{'floor':>8}{'over':>7}")
    excess = {}
    for name, d in MEASURED_F2.items():
        fw, fs = d["gemm_walk"] - d["core"], d["gemm"] - d["core"]
        excess[name] = (fw - floor_walk, fs - floor_strm)
        print(f"  {name:<10}{fw:>11,}{floor_walk:>8,}{excess[name][0]:>+7}"
              f"{fs:>13,}{floor_strm:>8,}{excess[name][1]:>+7}")
    # If the write path had throttled, the excess over the floor would have grown
    # with the beat rate -- five times the beats per cycle asked for, five times
    # the wready gaps.  It did not move, so the excess is burst setup latency in
    # the crossbar and L2, paid once per GEMM, not a per-beat ceiling.
    for name, (ew, es) in excess.items():
        assert abs(es - ew) <= 8, (name, ew, es)
    print(f"  the excess over the floor barely moves ({excess['EO-res'][0]:+d} ->"
          f" {excess['EO-res'][1]:+d} at EO-res), so it is burst setup in the")
    print("  crossbar and L2, paid once a GEMM -- NOT a per-beat ceiling.  The W")
    print("  channel did take a beat a cycle, and the model's floor holds.")
    print()
    for name, d in MEASURED_F2.items():
        print(f"  {name:<10} GEMM {d['gemm_walk']:,} -> {d['gemm']:,} "
              f"({d['gemm'] / d['gemm_walk'] - 1:+.0%}), feed "
              f"{d['gemm_walk'] - d['core']:,} -> {d['gemm'] - d['core']:,}, "
              f"feed's share {(d['gemm_walk'] - d['core']) / d['gemm_walk']:.0%}"
              f" -> {(d['gemm'] - d['core']) / d['gemm']:.0%}")

    # 5d. What binds now.  The wall clock is the CPU's, so wall - GEMM is spent
    # outside the engine: the submit's seven MMIO writes and the drain's poll.
    section("5d. What binds after F2: the host, not the feed")
    for name, d in MEASURED_F2.items():
        if d["wall_walk"] is None:
            print(f"  {name:<10} host {d['wall'] - d['gemm']:,} cycles a GEMM"
                  f" ({(d['wall'] - d['gemm']) / d['wall']:.0%} of the wall);"
                  f" no warm pre-change wall at this level (see MEASURED_F2)")
            continue
        hw, hs = d["wall_walk"] - d["gemm_walk"], d["wall"] - d["gemm"]
        print(f"  {name:<10} host {hw:,} -> {hs:,} cycles a GEMM, "
              f"{hs / d['wall']:.0%} of the wall and {hs / d['gemm']:.1f}x the GEMM")
        # The host's cost is outside the engine, so shortening the write burst
        # must not move it.  That it does not is the check that `wall - GEMM` is
        # really the host's and not some of the DMA's leaking in.
        assert abs(hs - hw) <= 16, (name, hw, hs)
    print("  Cutting the feed 56% moved the wall 10%: the submit writes and the")
    print("  drain poll are now the largest term in a GEMM, larger than the core")
    print("  and the feed together.  That is F3's finding to carry, and it is this")
    print("  SoC's MMIO path rather than anything a fabric would fix.")


    # 6. What PF2 costs, and which instrument can say so.
    section("6. PF2, measured twice: the wrong way and the right way")
    print("  F2 first priced PF2 by subtracting STAGE_A's refund.  STAGE_A turns PF2")
    print("  off as well, so its Q=4 penalty should be 4 * 28 = 112 and measured +17")
    print("  to +26; the difference looked like PF2's cost returned, about -22 cycles")
    print("  a GEMM.  It is not: that prices STAGE_A by its DMA_CT delta (+28) when")
    print("  its wall cost is +12, the rest hiding behind the host's ~1,090 cycles of")
    print("  submit and poll.  Quantities that do not subtract.")
    print()
    print("  PTA_CTRL.PF2_OFF turns PF2 off alone.  Measured, four queued GEMMs:")
    for name, d in MEASURED_PF2_OFF_Q4.items():
        print(f"    {name:<10} {d:+d} cycles over four GEMMs ({d / 4:+.1f} a GEMM)")
    print(f"  Both inside this harness's floor of {MEASURED_F2_WALL_DRIFT} cycles: in the same")
    print("  run, batches with extra batches ahead of them moved by up to that much")
    print("  on the wall while every core and DMA counter stayed bit-identical.  PF2")
    print("  only affects a GEMM with a next one queued, and the only queued-batch")
    print("  instrument there is the wall, so the SoC harness cannot see PF2 at all.")
    assert MEASURED_PF2_OFF_Q1_IS_NOOP
    for d in MEASURED_PF2_OFF_Q4.values():
        assert abs(d) < MEASURED_F2_WALL_DRIFT, d
    print()
    print("  tb_npu_feed.sv can, from phase counters rather than a wall:")
    print(f"    P_DONE {BENCH_F2['drain'] + 1} cycles with PF2 running against 1 without,"
          f" drain_beats {BENCH_F2['drain']} against 0")
    print(f"    PF2 ran {BENCH_F2['pf2_busy']} cycles, issued {BENCH_F2['pf2_ars']} AR,"
          f" fetched {BENCH_F2['pf2_beats']} beats of the 288 it needs")
    print(f"  so PF2 costs {BENCH_F2['drain']} cycles a queued GEMM, against F0's 145 --")
    print("  and the difference is the write burst, not PF2.  With 1,283 cycles to run")
    print("  it issued several bursts and left many beats in flight when abandoned;")
    print("  with 260 it issues one and nearly finishes it.")
    print()
    print("  So PF2_OFF saves nothing worth having.  What it buys is that the claim is")
    print("  a reading rather than a subtraction.  The open question is the opposite")
    print("  one -- whether a PF2 that finished would pay -- and at 3 cycles of cost")
    print("  there is no urgency to find out (CPU document section 8, item 6).")


    # 7. What F3 hands the fabric plan, in cycles.  The rates in bytes per second
    # are X2's section 1 (board_program_plan.md); this is the part that belongs
    # to the feed model, because it is the race that decides it.
    section("7. F3: how long the tile can wait, per A row")
    print("  A link that adds L cycles to every row's arrival stalls the core when L")
    print("  exceeds the tightest row's margin.  Checked against F2's measurement:")
    print("  S_AROW is zero exactly where the margin is positive.")
    print()
    print(f"  {'point':<24}{'order':<14}{'margin':>9}{'S_AROW':>9}{'agree':>8}")
    pts = (("TO-1ms", 100_000, 5, True), ("TO-10us", 1_000, 5, True),
           ("EO-scan", 0, 1, True), ("EO-res", 0, 1, False))
    for name, pta_tw, pta_ts, scanned in pts:
        tw = (SCAN if scanned else 0) + pta_tw
        ts = pta_ts + sweep.SHOT_FLOOR
        for order in ("interchanged", "shipped"):
            if order == "interchanged" and not scanned:
                continue
            if order == "shipped" and scanned and pta_tw:
                continue
            per = period("NPU bench")
            if order == "interchanged":
                mg = margin_interchanged(tw, ts, TD, per)
                aw = arow_interchanged(tw, ts, TD, per)
            else:
                mg = margin_shipped(tw, ts, TD, per)
                aw = arow_shipped(tw, ts, TD, per)
            ok = (aw == 0) == (mg > 0)
            print(f"  {name:<24}{order:<14}{mg:>9,}{aw:>9,}{'yes' if ok else 'NO':>8}")
            assert ok, (name, order, mg, aw)
    print()
    print(f"  At EO-res, shipped -- the point TFLT targets -- the margin is")
    print(f"  {margin_shipped(0, 1 + sweep.SHOT_FLOOR, TD, period('NPU bench')):,} cycles a row,"
          f" which at {10} ns a cycle is the latency budget a")
    print("  die-to-die link has before the tile stalls on its activations.")
    print("  The weights are not in this number: at EO-res they are resident (MB), so")
    print("  they cross once and the link's latency for them is amortised, not per row.")


if __name__ == "__main__":
    main()
