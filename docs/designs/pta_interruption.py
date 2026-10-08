"""
What an interruption costs: a calibration, counted in the tile's own time.

B15 calibrates version 2 every six minutes, 240 times a day.  The calibration
note says what one costs is not its shots but the interruption, "draining the
tile, rewriting the weights and restarting", and that no model prices it (its
section 3, and its open question 6).  B15 lists it among the three things that
would reopen the interval: "an interruption that costs enough for 240 a day to
matter".  On 2026-10-08 the interval was kept at six minutes, with a three-minute
one in view that the inverted set would have liked, and the cost of twice the
interruptions was the part nobody could put a number on.

This puts the number on it, from what the program already holds.  Nothing here
is measured, and nothing is a device's figure this program did not have.

COUNTED, from:

  the twin's timing     pta_chiplet_regmap.md section 5, and
                        src/backends/pta_chiplet/pta_chiplet_twin.h: a
                        calibration holds the tile for
                            passes * repeats * (PTA_TW + rows * PTA_TS)
                        cycles, a bank.  Each repeat writes the bank to zero
                        and then shoots every row once.  X5's gate holds the
                        twin to that formula
  what Tw and Ts are    board_program_plan.md section 4.3: the working tile's
  at the chiplet's      write path is 128 cells a beat into either of two
  scale                 banks of 8,192, a beat being one shot period, so a
                        bank programs in the 64 beats one batch is shot in.
                        And B11: a shot is a nanosecond
  the engine's          PTA_CAL_CFG, the same map's section 4: repeats a pass
  settings              and auto-ranging passes.  B15's table counts 16 probes
                        a cell and one pass; grx930's engine was measured at
                        three passes and its harness takes three; the
                        calibration note's section 9 says four probes are
                        enough
  what the formula      the estimator, which walks a bank's cells once a pass;
  leaves out            the restore, a bank's working weights programmed again;
                        and the drain, the batch in flight.  Each is a count of
                        beats.  The estimator's width is specified nowhere, so
                        it is taken a cell a beat, as grx930's engine walks
                        them, and as wide as the write path

DERIVED here:

  a calibration's time   in beats and in seconds, least to most
  its share              of B15's interval, and of shorter ones
  what would matter      how long an interruption would have to be for 240 a
                         day to cost a thousandth or a hundredth of the tile's
                         time, and the interval at which the count would
  what it costs a        how long one that meets a calibration waits, beside
  command                the 4096-square layer this plan times

WHAT THIS IS NOT.  A measurement.  A settling after the restore beyond a
weight's own: TFLT's response settles in 25 ps (pta_cpu_integration.md section
2.1), which is a fortieth of a beat, and what the light does to a ring that was
written to zero and back is in no model here.  A host that has to be told: the
scheduler is on the interface chip (the calibration note's section 6), and the
twin builds none, so CAL_NOW is still the only thing that starts one there.  A
drift that knows a cell's history.  And it prices time and not accuracy: what a
shorter interval would buy is pta_reference_draws.py's, at three minutes and at
six and at no other interval.

Standard library only.  Run:  python3 docs/designs/pta_interruption.py
"""
import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_laser as laser
import pta_shot_rate as shot
import pta_tw_sweep as sweep
import pta_version2 as version2
import pta_working_point as point

with contextlib.redirect_stdout(io.StringIO()):
    import pta_reference_draws as draws

TILE = laser.SMALLER                           # 128 x 64: B10
ROWS, COLS = TILE
CELLS = ROWS * COLS                            # 8,192 a bank
BANKS = version2.BANKS                         # two: section 4.3's write path loads one behind the other's shots
FS = version2.FS                               # 1 GS/s: B11.  A beat is one shot period
BATCH = point.BATCH                            # 64 shots to a weight set
WRITE = shot.per_beat_two_banks(ROWS, COLS, BATCH)     # 128 cells a beat: section 4.3
INTERVAL_S = version2.INTERVAL_S               # 360: B15, kept on 2026-10-08
SHORTER_S = 180                                # the three minutes that was in view
PROBES = (4, version2.CAL_AVERAGE)             # repeats a pass: the note's "four are enough", and B15's sixteen
PASSES = (1, 3)                                # auto-ranging passes: B15's table counts one; grx930's engine and harness take three
TFLT_BW_HZ = 40e9                              # pta_cpu_integration.md 2.1: TFLT's single-pole response
SETTLE_BITS = 8                                # to half an LSB of the weight DAC's eight bits
SHARES = (1e-3, 1e-2)                          # a thousandth and a hundredth of the tile's time
INTERVALS_S = (INTERVAL_S, SHORTER_S, 60, 10, 1)
DAY_S = 86400


# ---- a calibration, in beats ----------------------------------------------------------
def t_w():
    """Beats to program one bank: its cells over the write path.  The twin's PTA_TW."""
    return math.ceil(CELLS / WRITE)


def t_s():
    """Beats to a shot.  The twin's PTA_TS."""
    return 1


def twin(passes, repeats, banks=BANKS):
    """The twin's formula: each repeat of each pass zeroes a bank and shoots its rows."""
    return banks * passes * repeats * (t_w() + ROWS * t_s())


def shots(passes, repeats, banks=BANKS):
    """A calibration's probe shots: a row a shot."""
    return banks * passes * repeats * ROWS


def programmings(passes, repeats, banks=BANKS):
    """How often it writes a bank to zero."""
    return banks * passes * repeats


def estimator(passes, wide, banks=BANKS):
    """The estimator's walk, which the formula leaves out: a bank's cells once a
    pass, a cell a beat or as wide as the write path."""
    return banks * passes * math.ceil(CELLS / (WRITE if wide else 1))


def restore(banks=BANKS):
    """A bank's working weights programmed again.  Between two weight sets the
    next set's programming is the restore, and this is nothing; it is counted."""
    return banks * t_w()


def drain():
    """The batch in flight when a calibration is asked for."""
    return BATCH * t_s()


def interruption(passes, repeats, wide=False):
    """Everything counted: the formula, the estimator, the restore and the drain."""
    return twin(passes, repeats) + estimator(passes, wide) + restore() + drain()


LEAST = twin(PASSES[0], PROBES[0])                          # the formula alone, at its least
B15S = twin(PASSES[0], PROBES[1])                           # at B15's table's settings
ENGINE = twin(PASSES[1], PROBES[1])                         # at grx930's engine's
MOST = interruption(PASSES[1], PROBES[1], wide=False)       # and with all the formula leaves out, the estimator serial
MOST_WIDE = interruption(PASSES[1], PROBES[1], wide=True)   # or as wide as the write path


def seconds(beats):
    return beats / FS


def share(beats, interval_s=INTERVAL_S):
    """A calibration's share of the tile's time, one every interval."""
    return seconds(beats) / interval_s


def a_day(interval_s=INTERVAL_S):
    return DAY_S // interval_s


def day_s(beats, interval_s=INTERVAL_S):
    """Seconds a day spent calibrating."""
    return a_day(interval_s) * seconds(beats)


def to_matter_s(part, interval_s=INTERVAL_S):
    """How long an interruption would have to be to take that part of the tile's time."""
    return part * interval_s


def interval_for_s(beats, part):
    """The interval at which a calibration of that length would take that part."""
    return seconds(beats) / part


def settle_s():
    """A TFLT weight's settle to half an LSB."""
    return sweep.settle_s(TFLT_BW_HZ, SETTLE_BITS)


def layer_s():
    """The 4096-square layer on the working tile: pta_working_point.py's."""
    return point.layer(TILE, FS, BATCH, *point.WIDE)["seconds"]


def us(beats):
    return f"{seconds(beats) * 1e6:.1f} us"


def one_in(x):
    return f"one part in {1 / x / 1e6:.1f} million" if x < 1e-6 else f"one part in {1 / x:,.0f}"


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    print("What an interruption costs: a calibration, counted in the tile's own time.")
    print(f"The working tile, {laser.name(TILE)}, {BANKS} banks of {CELLS:,} cells written {WRITE} a beat, at {FS / 1e9:.0f} GS/s: a beat is a nanosecond.")

    section("1. What a calibration does, in beats")
    print(f"  A bank programs in {t_w()} beats, and a shot is {t_s()}.  Both banks:")
    print(f"  {'passes':>8}{'probes':>8}{'zeroings':>10}{'shots':>8}{'the formula':>13}{'':>4}{'time':>9}")
    for p in PASSES:
        for r in PROBES:
            print(f"  {p:>8}{r:>8}{programmings(p, r):>10}{shots(p, r):>8,}{twin(p, r):>13,}{'':>4}{us(twin(p, r)):>9}")
    p, r = PASSES[1], PROBES[1]
    print(f"  And what the formula leaves out, at {p} passes:")
    print(f"    {'the estimator, a cell a beat':<38}{estimator(p, False):>8,}")
    print(f"    {'the estimator, ' + str(WRITE) + ' cells a beat':<38}{estimator(p, True):>8,}")
    print(f"    {'the restore':<38}{restore():>8,}")
    print(f"    {'the drain':<38}{drain():>8,}")
    print(f"    {'all of it, the estimator serial':<38}{MOST:>8,}   {us(MOST)}")
    print(f"    {'all of it, the estimator wide':<38}{MOST_WIDE:>8,}   {us(MOST_WIDE)}")

    section("2. Against the interval")
    print(f"  {'every':>12}{'a day':>8}{'the formula, B15':>26}{'the engine':>26}{'the most counted':>26}{'a day, the most':>18}")
    for s in INTERVALS_S:
        print(f"  {s:>10} s{a_day(s):>8,}" + "".join(f"{one_in(share(b, s)):>26}" for b in (B15S, ENGINE, MOST)) + f"{day_s(MOST, s) * 1e3:>15.1f} ms")

    section("3. What it would take to matter")
    for part in SHARES:
        print(f"  For {a_day()} a day to take {part:.1%} of the tile's time an interruption has to last {to_matter_s(part):.2f} s:"
              f" {to_matter_s(part) / seconds(MOST):,.0f} times the most counted.")
        print(f"    The most counted takes {part:.1%} at an interval of {interval_for_s(MOST, part) * 1e3:.1f} ms.")
    print(f"  A TFLT weight settles in {settle_s() * 1e12:.0f} ps, a {1 / FS / settle_s():.0f}th of a beat.")

    section("4. What it costs a command")
    print(f"  The 4096-square layer takes {layer_s() * 1e6:.0f} us on this tile.  A command that meets a calibration waits")
    print(f"  {us(MOST)} at the most, {seconds(MOST) / layer_s():.2f} of that layer, and {one_in(share(MOST))} of the tile's time is one.")

    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        findings()
    print(said.getvalue(), end="")
    checks(" ".join(said.getvalue().split()))


def findings():
    i, R = draws.INVERTED, draws.REFERENCE
    print()
    print("What this says, five readings.")
    print()
    print("  1. A CALIBRATION HOLDS THE TILE FOR MICROSECONDS.  By the twin's formula, both banks:")
    print(f"     {us(B15S)} at B15's sixteen probes and one pass, {us(ENGINE)} at the three passes grx930's engine")
    print(f"     was measured at, and {us(LEAST)} at the four probes the calibration note found enough.  Of the")
    print("     engine's")
    print(f"     {ENGINE:,} beats, {shots(PASSES[1], PROBES[1]):,} are shots and {programmings(PASSES[1], PROBES[1]) * t_w():,} are the {programmings(PASSES[1], PROBES[1])} zeroings before them.  With")
    print(f"     what the formula leaves out it is {us(MOST)} at the most: the estimator walking a cell a")
    print(f"     beat is {estimator(PASSES[1], False) / MOST:.0%} of that, and as wide as the write path it would be {us(MOST_WIDE)}.")
    print()
    print(f"  2. {a_day()} A DAY IS {day_s(MOST) * 1e3:.0f} MILLISECONDS A DAY.  The most counted is {one_in(share(MOST))} of")
    print(f"     the tile's time, and the engine's formula {one_in(share(ENGINE))}.  B15's table had the probes'")
    print(f"     shots alone, {one_in(share(shots(PASSES[0], PROBES[1])))}: the zeroings before them are half as much again,")
    print("     and the three passes three times that.")
    print()
    print(f"  3. FOR IT TO MATTER AN INTERRUPTION WOULD HAVE TO LAST A THIRD OF A SECOND.  {to_matter_s(SHARES[0]):.2f} s takes a")
    print(f"     thousandth of the tile's time at {a_day()} a day, and {to_matter_s(SHARES[1]):.1f} s a hundredth: {to_matter_s(SHARES[0]) / seconds(MOST):,.0f} and {to_matter_s(SHARES[1]) / seconds(MOST):,.0f}")
    print(f"     times the most counted.  Nothing in these documents is that slow.  A weight on TFLT")
    print(f"     settles in {settle_s() * 1e12:.0f} ps, and the write path was sized to program a bank inside one batch.")
    print("     B15's second condition for reopening, an interruption that costs enough for 240 a")
    print("     day to matter, is not met by anything that can be counted.")
    print()
    print(f"  4. SO THE COUNT DOES NOT SET THE INTERVAL.  At three minutes, {a_day(SHORTER_S)} a day, the most counted is")
    print(f"     {one_in(share(MOST, SHORTER_S))} and {day_s(MOST, SHORTER_S) * 1e3:.0f} ms a day; every ten seconds, {one_in(share(MOST, 10))}.  It")
    print(f"     reaches a thousandth of the tile's time at an interval of {interval_for_s(MOST, SHARES[0]) * 1e3:.0f} ms.  What a shorter")
    print("     interval would buy is accuracy, and that is measured at two intervals and no other:")
    print(f"     on the inverted set the reference networks' six-minute cycle ends {draws.over(i, R, draws.SIX)[0]:.2f} of a point over")
    print(f"     and their three-minute one {draws.over(i, R, draws.THREE)[0]:.2f}, and six minutes costs {draws.dpm(draws.longer(i, R))} more than three.")
    print()
    print(f"  5. A COMMAND THAT MEETS A CALIBRATION WAITS HALF A LARGE LAYER AT THE MOST.  {us(MOST)}, where the")
    print(f"     4096-square layer takes {layer_s() * 1e6:.0f} us on this tile.  One command in {1 / share(MOST) / 1e6:.1f} million meets one.")
    print("     The two schedulers that predict were built to take fewer calibrations than the period")
    print("     allows, and the shadow one to hide them in idle windows; on the c930 that was worth")
    print("     three quarters of a calibration's cycles.  On this tile there is nothing of that size")
    print("     to hide.")


def checks(said):
    """Every claim above, as an assert.  `said` is the readings as printed, on one line."""
    # 0. The inputs are the plan's.
    assert TILE == (128, 64) and CELLS == 8192 and BANKS == 2 and FS == 1e9 and BATCH == 64 and INTERVAL_S == 360
    assert WRITE == 128 == shot.per_beat_two_banks(*TILE, BATCH)        # section 4.3: "written 128 cells a beat"
    assert PROBES == (4, 16) and version2.CAL_AVERAGE == 16 and PASSES == (1, 3)
    #    A bank programs in the beats one batch is shot in: that is what the write
    #    path was sized for, and it is PTA_TW at the chiplet's scale.
    assert t_w() == 64 == BATCH and t_s() == 1

    # 1. Reading 1.  The counts.
    assert [twin(p, r) for p in PASSES for r in PROBES] == [1536, 6144, 4608, 18432]
    assert (LEAST, B15S, ENGINE) == (1536, 6144, 18432)
    assert all(twin(p, r) == BANKS * p * r * (64 + 128) for p in PASSES for r in PROBES)
    assert (shots(1, 16), shots(3, 16), programmings(3, 16), programmings(1, 16)) == (4096, 12288, 96, 32)
    #    The shots are B15's own count, and pta_version2.py's.
    assert shots(1, 16) == version2.cal_shots() and abs(share(shots(1, 16)) - version2.duty()) < 1e-18
    assert shots(3, 16) + programmings(3, 16) * t_w() == ENGINE and programmings(3, 16) * t_w() == 6144
    assert (estimator(3, False), estimator(3, True), restore(), drain()) == (49152, 384, 128, 64)
    assert MOST == 18432 + 49152 + 128 + 64 == 67776 and MOST_WIDE == 18432 + 384 + 128 + 64 == 19008
    assert [us(b) for b in (LEAST, B15S, ENGINE, MOST_WIDE, MOST)] == ["1.5 us", "6.1 us", "18.4 us", "19.0 us", "67.8 us"]
    assert round(estimator(3, False) / MOST, 2) == 0.73
    #    The twin's formula is for one bank, and both are calibrated.
    assert twin(3, 16, banks=1) * 2 == ENGINE

    # 2. Reading 2.  A day.
    assert a_day() == 240 and a_day(SHORTER_S) == 480 and a_day(3600) == 24
    assert round(day_s(MOST) * 1e3, 1) == 16.3 and round(day_s(ENGINE) * 1e3, 1) == 4.4
    assert [round(1 / share(b) / 1e6, 1) for b in (shots(1, 16), B15S, ENGINE, MOST)] == [87.9, 58.6, 19.5, 5.3]
    assert B15S / shots(1, 16) == 1.5 and ENGINE / B15S == 3

    # 3. Reading 3.  What would matter.
    assert [round(to_matter_s(p), 2) for p in SHARES] == [0.36, 3.6]
    assert [round(to_matter_s(p) / seconds(MOST)) for p in SHARES] == [5312, 53116]
    assert round(settle_s() * 1e12) == 25 and round(1 / FS / settle_s()) == 40
    assert seconds(MOST) < to_matter_s(SHARES[0]) / 5000

    # 4. Reading 4.  The interval.
    assert [round(1 / share(MOST, s)) for s in INTERVALS_S] == [5311615, 2655807, 885269, 147545, 14754]
    assert round(day_s(MOST, SHORTER_S) * 1e3, 1) == 32.5
    assert [round(interval_for_s(MOST, p) * 1e3, 1) for p in SHARES] == [67.8, 6.8]
    assert all(share(MOST, s) < SHARES[0] for s in INTERVALS_S)
    #    What a shorter interval buys: pta_reference_draws.py's, on the inverted set.
    i, R = draws.INVERTED, draws.REFERENCE
    assert (round(draws.over(i, R, draws.SIX)[0], 2), round(draws.over(i, R, draws.THREE)[0], 2)) == (0.12, 0.08)
    assert draws.r2(draws.longer(i, R)) == (0.04, 0.04) and draws.CYCLES == (0.05, 0.1)
    assert draws.B15_H * 3600 == INTERVAL_S and draws.CYCLES[0] * 3600 == SHORTER_S

    # 5. Reading 5.  A command.
    assert round(layer_s() * 1e6) == 131 and round(seconds(MOST) / layer_s(), 2) == 0.52

    # 6. And the readings say those figures, each in its place.
    for words in (
        "6.1 us at B15's sixteen probes and one pass, 18.4 us at the three passes grx930's engine was measured at, and 1.5 us at the four probes",
        "18,432 beats, 12,288 are shots and 6,144 are the 96 zeroings before them.",
        "it is 67.8 us at the most: the estimator walking a cell a beat is 73% of that, and as wide as the write path it would be 19.0 us.",
        "240 A DAY IS 16 MILLISECONDS A DAY.",
        "The most counted is one part in 5.3 million of the tile's time, and the engine's formula one part in 19.5 million.",
        "shots alone, one part in 87.9 million:",
        "0.36 s takes a thousandth of the tile's time at 240 a day, and 3.6 s a hundredth: 5,312 and 53,116 times the most counted.",
        "settles in 25 ps,",
        "At three minutes, 480 a day, the most counted is one part in 2.7 million and 33 ms a day; every ten seconds, one part in 147,545.",
        "at an interval of 68 ms.",
        "six-minute cycle ends 0.12 of a point over and their three-minute one 0.08, and six minutes costs +0.04 +-0.04 more than three.",
        "67.8 us, where the 4096-square layer takes 131 us on this tile. One command in 5.3 million meets one.",
    ):
        assert said.count(words) == 1, words

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
