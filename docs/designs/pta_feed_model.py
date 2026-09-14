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
so it saves nothing.

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


def arow_interchanged(tw, ts, td, per):
    return race(CORE_START + tw, ts + td, per)


def arow_shipped(tw, ts, td, per):
    return race(CORE_START, KT * (tw + ts) + td, per)


def gemm(level, core, arow, queued=False):
    p = LEVELS[level]
    total = p["read_a"] + p["read_b"] + core + arow + HANDOFF + p["write_c"] + DONE
    return total + (p["drain"] if queued else 0)


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
        print(f"  {'point':<24}{'level':<11}{'core':>13}{'S_AROW':>8}{'GEMM':>13}"
              f"{'DMA_LAST':>13}{'feed':>7}")
        for name, pta_tw, pta_ts, scanned in points:
            if order == "interchanged" and not scanned and pta_tw:
                continue
            tw = (SCAN if scanned else 0) + pta_tw
            for level in LEVELS:
                per = period(level)
                if order == "interchanged":
                    core = sweep.interchanged(tw, pta_ts, TD, restore=True)
                    arow = arow_interchanged(tw, pta_ts, TD, per)
                else:
                    core = sweep.shipped(tw, pta_ts, TD)
                    arow = arow_shipped(tw, pta_ts, TD, per)
                total = gemm(level, core, arow)
                feed = total - core
                print(f"  {name:<24}{level:<11}{core:>13,}{arow:>8,}{total:>13,}"
                      f"{total - 1:>13,}{feed / total:>7.1%}")

    # 3. What the feed does to the loop-order verdict at the resident point.
    section("3. EO-res, whole GEMM: shipped against interchanged")
    for level in LEVELS:
        per = period(level)
        s_core = sweep.shipped(0, 1, TD)
        i_core = sweep.interchanged(0, 1, TD, restore=True)
        i_fold = sweep.interchanged(0, 1, TD)
        s = gemm(level, s_core, arow_shipped(0, 1, TD, per))
        i = gemm(level, i_core, arow_interchanged(0, 1, TD, per))
        i_f = gemm(level, i_fold, arow_interchanged(0, 1, TD, per))
        print(f"  {level:<10} shipped {s:,} vs interchanged {i:,} ({i / s:.1f}x;"
              f" {i_f:,} and {i_f / s:.1f}x with the restore folded).  Core alone:"
              f" {i_core / s_core:.1f}x, {i_fold / s_core:.1f}x folded")

    section("4. Queued GEMMs")
    for level, p in LEVELS.items():
        print(f"  {level:<10} +{p['drain']} cycles each for PF2's drain, at every point:"
              " writeback is 512 words regardless of the tile")


if __name__ == "__main__":
    main()
