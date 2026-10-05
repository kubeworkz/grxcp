"""
X2 of board_program_plan.md: the die-to-die term, and what it costs the tile.

B4 puts the PTA on a chiplet behind a UCIe port, so the operands the c930's DMA
used to fetch now cross a link.  F1 priced the c930's feed in cycles
(pta_feed_model.py); this prices the chiplet's in bytes per second, and says
what the link caps.

Section 1 is F3's handoff, which pta_program_plan.md §3.3 owes this plan: the
operand rates at each §6.2 point for the shape those points are quoted at,
M = 64, N = 8, K = 256, taken from F1's cycle counts at 10 ns a cycle.  Nothing
in it is new; it is the same model in different units.

Sections 2 to 5 are the chiplet.  A layer of `kin` inputs and `nout` outputs
runs on a tile of `k` by `n` cells at `fs` shots a second, a batch of `mb` rows
at a time.  Traffic depends on where the work sits:

  split (B4)  the interface chip buffers activation rows and accumulates across
              K tiles, so each row crosses once per layer and each output
              crosses once.
  thin        neither, so each row crosses once per column group and every
              K tile's partial sum comes back.

Assumed, and marked at each use: operands and ADC codes are one byte, an
accumulated output is four, one UCIe-S module of 16 lanes at 32 GT/s carries
64 GB/s a direction before overhead, 0.9 of that survives flit and protocol
overhead, and a request crosses and returns in 100 ns.  GB is 10^9 bytes.  The
chiplet's own geometry was not settled when this was written
(board_program_plan.md §8), so the tables sweep candidates rather than claiming
one.  The plan's B10 made 256 x 64 the working geometry on 2026-10-05, and the
sweep stays: it is what that choice was made from.

Standard library only.  Run:  python3 docs/designs/pta_chiplet_link.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_feed_model as feed
import pta_tw_sweep as sweep

CYCLE_NS = 10.0          # the c930 host at 100 MHz, §6.2
A_BYTES = 1              # an activation on the wire
W_BYTES = 1              # a weight on the wire
ADC_BYTES = 1            # one ADC code, 6 bits of it used (C1's spec)
ACC_BYTES = 4            # an accumulated output before the activation stage
LANES, GT_S = 16, 32     # one UCIe-S module
LINK_EFF = 0.9           # assumed: flit and protocol overhead
ROUND_TRIP_NS = 100.0    # assumed: request out, data back, both fabrics
MODULE_GBS = LANES * GT_S / 8 * LINK_EFF          # 57.6 GB/s a direction


def modules(rate_gbs):
    """UCIe-S modules needed for a rate, per direction."""
    return -(-int(rate_gbs * 1e9) // int(MODULE_GBS * 1e9))


def layer_traffic(kin, nout, mb, k, n, split=True):
    """Bytes in and out for one layer of one batch, and the shots it takes."""
    ktiles, ngroups = -(-kin // k), -(-nout // n)
    shots = ktiles * ngroups * mb
    weights = ktiles * ngroups * k * n * W_BYTES
    if split:
        acts = mb * kin * A_BYTES
        results = mb * nout * ACC_BYTES
    else:
        acts = mb * kin * A_BYTES * ngroups
        results = mb * nout * ktiles * ADC_BYTES
    return dict(shots=shots, weights=weights, acts=acts, results=results,
                into=weights + acts, out=results)


def rates(kin, nout, mb, k, n, fs, **kw):
    """The same layer as bandwidths, given a shot rate."""
    t = layer_traffic(kin, nout, mb, k, n, **kw)
    seconds = t["shots"] / fs
    return t["into"] / seconds / 1e9, t["out"] / seconds / 1e9


def max_shot_rate(kin, nout, mb, k, n, mods=1, **kw):
    """The shot rate at which `mods` modules saturate in the busier direction."""
    t = layer_traffic(kin, nout, mb, k, n, **kw)
    per_shot = max(t["into"], t["out"]) / t["shots"]
    return mods * MODULE_GBS * 1e9 / per_shot


def fmt_bytes(b):
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if b >= scale:
            return f"{b / scale:.3g} {unit}"
    return f"{b:.0f} B"


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


# The §6.2 points, with the order each one wins in (§2.1, §6.2).
# PTA_TW per point.  At EO-res it is not zero: the core still spends the bank
# select's own cycle, which step MB measured and sweep.MB_TW records.  Section
# 6.2's table writes Tw = 0 there, and that is the idealisation rather than the
# RTL -- 2,048 cycles a GEMM at this shape, since m-outer programs once a shot.
POINTS = (
    ("TO-1ms", 100_000, 5, True, "interchanged"),
    ("TO-10us", 1_000, 5, True, "interchanged"),
    ("EO-scan", 0, 1, True, "interchanged"),
    ("EO-res", sweep.MB_TW, 1, False, "shipped"),
)
# Candidate chiplet geometries: the emulated tile, then wider ones.
GEOMETRIES = ((8, 8), (64, 8), (128, 64), (256, 64), (256, 128))
BIG = dict(kin=4096, nout=4096, mb=64)          # an illustrative dense layer


def main():
    # 1. F3's handoff, in operands per second.
    section("1. F3's handoff: section 6.2's shape, its operand rates, every number traced")
    a_bytes, b_bytes = sweep.M * sweep.K * A_BYTES, sweep.K * sweep.N * W_BYTES
    c_bytes = sweep.M * sweep.N * ACC_BYTES
    shots = sweep.M * sweep.NT * sweep.KT
    assert shots == 2_048, shots                          # section 6.2's count
    assert sweep.shipped(0, 1, sweep.TD) == 2_560         # section 6.2's EO-res core
    print(f"  A {fmt_bytes(a_bytes)} in, B {fmt_bytes(b_bytes)} in, C {fmt_bytes(c_bytes)} out,"
          f" {shots:,} shots a GEMM, {CYCLE_NS:.0f} ns a cycle")
    print(f"  Traced: every core here is MEASURED at this shape -- the three"
          f" interchanged ones by C2 (make core_c2),")
    print(f"  the resident one by MB (make core_mb) -- and asserted against those"
          f" readings, not quoted. Section 2.1")
    print(f"  with C2's shot floor of {sweep.SHOT_FLOOR} cycles a shot and MB's"
          f" bank-select Tw of {sweep.MB_TW} reproduces all four.")
    print(f"  S_AROW and the margin are F1's race, which F2 checked")
    print(f"  (S_AROW is zero exactly where the margin is positive); the feed is"
          f" F0's measured loads plus the rebuilt")
    print(f"  write burst, {feed.feed_part(feed.c_beats_of(sweep.M * sweep.N), 'write')}"
          f" cycles, which the NPU bench measured to the cycle (feed model section 5).")
    print(f"  {'point':<10}{'order':<14}{'core':>12}{'GEMM':>12}{'GEMM us':>10}"
          f"{'shots/s':>12}{'in GB/s':>9}{'out GB/s':>10}{'margin':>9}")
    # Each of the four cores has been measured at THIS shape: the three
    # interchanged ones by C2 (make core_c2) and the resident one by MB
    # (make core_mb, BANKS=32).  F3's gate is that every number traces, so they
    # are asserted here rather than quoted.
    measured_core = dict(sweep.MEASURED_C2)
    measured_core["EO-res"] = sweep.MEASURED_MB["m-outer"]
    out = {}
    for name, tw, ts, scanned, order in POINTS:
        per = feed.period("NPU bench")
        full_tw = (sweep.SCAN if scanned else 0) + tw
        # C2 measured a shot at PTA_TS + 2.  Section 6.2's table has none of it,
        # and this handoff used to inherit that: 2 cycles x 2,048 shots a GEMM.
        shot = ts + sweep.SHOT_FLOOR
        if order == "interchanged":
            core = sweep.interchanged(full_tw, shot, sweep.TD, restore=True)
            arow = feed.arow_interchanged(full_tw, shot, sweep.TD, per)
            mg = feed.margin_interchanged(full_tw, shot, sweep.TD, per)
        else:
            core = sweep.shipped(full_tw, shot, sweep.TD)
            arow = feed.arow_shipped(full_tw, shot, sweep.TD, per)
            mg = feed.margin_shipped(full_tw, shot, sweep.TD, per)
        assert core == measured_core[name], (name, core, measured_core[name])
        total = feed.gemm_stream("NPU bench", core, arow)
        seconds = total * CYCLE_NS * 1e-9
        out[name] = dict(total=total, seconds=seconds, margin=mg, arow=arow)
        print(f"  {name:<10}{order:<14}{core:>12,}{total:>12,}{seconds * 1e6:>10.1f}"
              f"{shots / seconds:>12,.0f}{(a_bytes + b_bytes) / seconds / 1e9:>9.2f}"
              f"{c_bytes / seconds / 1e9:>10.2f}{mg:>9,}")
    print()
    print("  WHICH OPERANDS ARE RESIDENT AND WHICH STREAM, which is the part that")
    print("  decides where the link's bandwidth goes:")
    print("    B, the weights   resident at EO-res, in a bank per (N tile, K tile)")
    print("                     (step MB); scanned every tile at the other three, so")
    print("                     at those points B crosses Nt*Kt times a GEMM, not once")
    print("    A, the rows      streamed, one row at a time, during compute (PF1), at")
    print("                     one beat a cycle once the burst is open")
    print("    C, the outputs   once a GEMM, in one burst, now at a beat a cycle")
    print("    next GEMM's A+B  PF2 fetches these during C's writeback and does not")
    print("                     finish: 29 of 288 beats.  Costs 3 cycles, saves nothing")
    print("                     (feed model section 6).  A fabric should not count on it")
    print()
    print("  HOW LONG THE TILE CAN WAIT, per A row, which is the margin column:")
    eo = out["EO-res"]
    print(f"    EO-res, shipped   {eo['margin']:,} cycles ="
          f" {eo['margin'] * CYCLE_NS:,.0f} ns, against this model's assumed")
    print(f"                      {ROUND_TRIP_NS:.0f} ns round trip -- so the link's latency fits,"
          f" with margin")
    for name in ("TO-10us", "EO-scan"):
        d = out[name]
        print(f"    {name + ', interchanged':<22}{d['margin']:,} cycles: already negative, and the core"
              f" waits {d['arow']:,}")
    print("                      cycles there.  Those points are feed-bound before any")
    print("                      link is added, which is F1's finding, not a new one.")
    print()
    print("  WHAT THIS HANDOFF DOES NOT COVER.  These are the DMA's rates.  On the")
    print("  measured SoC the wall clock is dominated by the host, not the feed: the")
    print("  submit's MMIO writes and the drain's poll cost ~1,090 cycles a GEMM, which")
    print("  at EO-res is 2.0x the whole GEMM and did not move when the feed was cut")
    print("  56% (F2).  That is this SoC's MMIO path, not a fabric rate, and a link")
    print("  sized to the table above still leaves it as the limiter.  Whoever owns the")
    print("  chiplet's command path should size that separately.")

    # 2. The chiplet's link, under B4's split.
    section("2. The die-to-die term, B4's split, one dense layer"
            f" ({BIG['kin']}x{BIG['nout']}, batch {BIG['mb']})")
    print(f"  one UCIe-S module = {LANES} lanes at {GT_S} GT/s ="
          f" {MODULE_GBS:.1f} GB/s a direction at {LINK_EFF:.0%} efficiency")
    print(f"  {'tile':<12}{'shots/layer':>12}{'GS/s':>7}{'in GB/s':>10}{'out GB/s':>10}"
          f"{'modules in':>12}")
    for k, n in GEOMETRIES:
        t = layer_traffic(k=k, n=n, **BIG)
        for fs in (0.1e9, 1e9):
            into, out = rates(k=k, n=n, fs=fs, **BIG)
            print(f"  {f'{k}x{n}':<12}{t['shots']:>12,}{fs / 1e9:>7.1f}{into:>10.1f}"
                  f"{out:>10.2f}{modules(into):>12}")
    print("  Inbound is k*n/mb bytes of weight a shot plus kin/(ktiles*ngroups) of")
    print("  activation, so the batch is the lever on the weights:")
    for mb in (16, 64, 256):
        big = dict(BIG, mb=mb)
        into, out = rates(k=256, n=64, fs=1e9, **big)
        print(f"    256x64 at 1.0 GS/s, batch {mb:>3}: in {into:>6.1f} GB/s,"
              f" out {out:>5.2f} GB/s, {modules(into)} modules in")

    # 3. What the split removes.
    section("3. What B4's split removes: the same layer on a thin interface chip")
    print(f"  {'tile':<12}{'GS/s':>7}{'split in':>10}{'thin in':>10}{'split out':>11}"
          f"{'thin out':>10}{'thin modules':>14}")
    for k, n in GEOMETRIES:
        for fs in (1e9,):
            s_in, s_out = rates(k=k, n=n, fs=fs, **BIG)
            t_in, t_out = rates(k=k, n=n, fs=fs, split=False, **BIG)
            assert t_in >= s_in - 1e-9 and t_out >= s_out - 1e-9, (k, n)
            print(f"  {f'{k}x{n}':<12}{fs / 1e9:>7.1f}{s_in:>10.1f}{t_in:>10.1f}"
                  f"{s_out:>11.2f}{t_out:>10.1f}{modules(max(t_in, t_out)):>14}")

    # 4. The activation stage: B4 left it open, and multi-layer traffic decides it.
    section("4. The activation stage on the chiplet, per inference of a batch")
    d3 = ((784, 100), (100, 10))                      # the D3 network
    block = ((4096, 4096),) * 4                       # an illustrative four-layer block
    print("  With the stage on the chiplet an intermediate never crosses: the layer's")
    print("  outputs become the next layer's inputs where they already are.")
    print(f"  {'network':<26}{'activation stage':<18}{'weights':>9}{'acts in':>9}"
          f"{'out':>9}{'total':>9}{'without weights':>17}")
    for label, net, mb in (("D3, batch 64", d3, 64), ("4 x 4096 square, batch 64", block, 64)):
        k, n = 256, 64
        totals = {}
        for stage in (False, True):
            into = out = weights = 0
            for i, (kin, nout) in enumerate(net):
                t = layer_traffic(kin, nout, mb, k, n)
                weights += t["weights"]
                if not stage or i == 0:
                    into += t["acts"]
                if not stage or i == len(net) - 1:
                    out += t["results"]
            totals[stage] = into + out
            print(f"  {label:<26}{'on the chiplet' if stage else 'on the GPU':<18}"
                  f"{fmt_bytes(weights):>9}{fmt_bytes(into):>9}{fmt_bytes(out):>9}"
                  f"{fmt_bytes(weights + into + out):>9}{fmt_bytes(into + out):>17}")
        assert totals[True] <= totals[False], label
        print(f"  {'':<26}the stage on the chiplet removes"
              f" {fmt_bytes(totals[False] - totals[True])} of the"
              f" {fmt_bytes(totals[False])} that is not weights"
              f" ({1 - totals[True] / totals[False]:.0%})")
    print("  Weights here include the tile's padding: a 256x64 tile holds a 784-input")
    print("  layer in four K tiles, and the last is three quarters empty.  At batch 64")
    print("  the weights are most of the traffic either way; the stage is the lever once")
    print("  they stay resident across batches, and section 2's batch sweep is the other.")

    # 5. What a link of a given width caps, and the buffer it needs.
    section("5. Caps and buffers, B4's split, the dense layer of section 2")
    print(f"  {'tile':<12}{'shots/s, 1 mod':>16}{'2 mods':>10}{'4 mods':>10}"
          f"{'weights held':>14}{'act buffer':>12}{'accumulators':>14}")
    for k, n in GEOMETRIES:
        caps = [max_shot_rate(k=k, n=n, mods=m, **BIG) for m in (1, 2, 4)]
        held = k * n                                   # DAC-held weights, per B4
        act_buf = BIG["mb"] * k * A_BYTES              # one K tile's rows for the batch
        accum = BIG["mb"] * n * ACC_BYTES              # one column group's running sums
        print(f"  {f'{k}x{n}':<12}{caps[0] / 1e9:>15.2f}G{caps[1] / 1e9:>9.2f}G"
              f"{caps[2] / 1e9:>9.2f}G{held:>14,}{fmt_bytes(act_buf):>12}"
              f"{fmt_bytes(accum):>14}")
    in_flight = MODULE_GBS * 1e9 * ROUND_TRIP_NS * 1e-9
    print(f"  A module keeps {fmt_bytes(in_flight)} in flight across the assumed"
          f" {ROUND_TRIP_NS:.0f} ns round")
    print("  trip, so the chiplet needs at least that much again before the tile waits.")


if __name__ == "__main__":
    main()
