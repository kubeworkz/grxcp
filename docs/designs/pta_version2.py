"""
Version 2, where version 1 was measured: drift, and the source's rows.

B14 holds the interface chip to version 2: version 1 with an 8-bit ADC,
receiver noise within a quarter of an 8-bit ADC's LSB, and 30 photons such an
LSB.  It listed what had been measured at version 1 and not run again.  The
first two on that list were how long a calibration holds, and the source's
three rows.  grx930's harness has now run both at version 2, with version 1
beside it, on all three data sets.

MEASURED IN A MODEL, by grx930 (c930/doc/pta_error_model_design_note.md section
5, "Drift and a source's rows at grxcp's version 2", 2026-10-06;
`sim/pta_mnist.sh DIR WORK v2`).  On the working tile, 128 x 64 on two buses,
five networks a data set.  For every row, the points lost against the same
weights on the host, and what the row adds to its own version as budgeted,
network by network, each with its standard error.  The drift is TFLT's fit
unless it says TFLN's, and "calibrated" is C3's cell calibration after an hour.
A source's noise is through a balanced pair (B13): the lines together, a line
on its own, and the lines' level, as rms fractions of a line's power.

DERIVED here:

  what drift does   what it adds to version 2, less what it adds to version 1
  to each version
  drift's share     what an interval's drift adds, over the version's budget
  the source's      its three rows as B12 and B13 budget them, 2%, 5% and 5%,
  rows              against the tenth of a point they were sized to
  the shared row    the largest size run at which the lines together add about
                    a tenth, and that as a density: pta_source_noise.py's

THE PLAN THEN CHOSE, 2026-10-06.  B15: version 2 is calibrated every six
minutes.  B16: the row the lines share is 1% and not 2%.  grx930 added the two
rows that say what that is, the source's three at 1%, 5% and 5%, and the same
at the end of six minutes of drift, which is version 2 with everything it is
held to.  Section 4 and readings 6 and 7.

WHAT THIS IS NOT: a measurement of a source or of a drift.  The source's term
is first order and on the host's side of the line.  The drift is a
Mach-Zehnder's fit, and every cell drifts on its own.  A pair only: the offset
reading was not rerun.  And everything pta_workload.py is not: the networks
were not trained for the tile, and the inverted set's were not trained well.

Standard library only.  Run:  python3 docs/designs/pta_version2.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_laser as laser
import pta_shot_rate as shot
import pta_source_noise as noise
import pta_tighten as tighten
import pta_workload as workload

MNIST, FASHION, INVERTED = workload.MNIST, workload.FASHION, workload.INVERTED
WORKLOADS = workload.WORKLOADS
TILE = laser.SMALLER                           # 128 x 64: B10
FS = laser.FS                                  # 1 GS/s: B11
INTERVAL_S = 360                               # B15: version 2 is calibrated every six minutes
HOURLY_S = 3600                                # what section 4.3's table said of version 1: about hourly
ROWS_V2 = (0.01, 0.05, 0.05)                   # B16: the lines together at 1%; a line and the level as they were
PERIOD_BITS = 32                               # PTA_CAL_PER's width: the register map's section 4
PERIOD_UNIT_LOG2 = 16                          # and its unit there, 2^16 cycles, since 2026-10-06
CAL_AVERAGE = 16                               # probes a cell: pta_chiplet_calibration.md's section 3
BANKS = 2
BUDGET = noise.BUDGET                          # a tenth of a point: what a row of the budget costs
ROWS3 = noise.ROWS                             # 2%, 5%, 5%: B12's and B13's three rows
VERSIONS = (1, 2)

# ---- grx930's figures ----------------------------------------------------------
# (data set, version) -> row -> ((points lost, its error), (what the row adds, its error)).
RUN = {
    (FASHION, 1): {
        "budget": ((1.16, 0.14), (0.00, 0.00)), "six": ((1.24, 0.18), (0.08, 0.10)),
        "hour": ((1.84, 0.39), (0.69, 0.33)), "four": ((3.50, 0.41), (2.34, 0.43)),
        "long46": ((23.26, 1.48), (22.10, 1.38)), "tfln": ((8.37, 1.47), (7.22, 1.48)),
        "cal": ((1.18, 0.09), (0.03, 0.15)), ("together", 0.01): ((1.22, 0.12), (0.06, 0.06)),
        ("together", 0.02): ((1.21, 0.12), (0.05, 0.02)), ("together", 0.05): ((1.27, 0.14), (0.12, 0.04)),
        ("together", 0.1): ((1.32, 0.13), (0.16, 0.03)), ("line", 0.02): ((1.20, 0.15), (0.04, 0.09)),
        ("line", 0.05): ((1.26, 0.21), (0.11, 0.14)), ("line", 0.1): ((1.43, 0.20), (0.27, 0.12)),
        ("line", 0.2): ((1.89, 0.25), (0.73, 0.15)), ("level", 0.02): ((1.25, 0.17), (0.09, 0.05)),
        ("level", 0.05): ((1.25, 0.17), (0.09, 0.08)), ("level", 0.1): ((1.38, 0.22), (0.22, 0.12)),
        ("all", "rows"): ((1.22, 0.19), (0.06, 0.11)), ("all", 0.05): ((1.36, 0.22), (0.20, 0.12)),
        ("all", "v2 rows"): ((1.30, 0.21), (0.14, 0.11)), ("interval", "v2 rows"): ((1.37, 0.23), (0.21, 0.13)),
    },
    (FASHION, 2): {
        "budget": ((0.54, 0.22), (0.00, 0.00)), "six": ((0.62, 0.19), (0.08, 0.06)),
        "hour": ((1.21, 0.28), (0.67, 0.31)), "four": ((2.90, 0.49), (2.36, 0.57)),
        "long46": ((23.14, 1.54), (22.60, 1.38)), "tfln": ((7.83, 1.39), (7.29, 1.50)),
        "cal": ((0.56, 0.11), (0.02, 0.11)), ("together", 0.01): ((0.58, 0.20), (0.04, 0.04)),
        ("together", 0.02): ((0.57, 0.20), (0.03, 0.04)), ("together", 0.05): ((0.55, 0.16), (0.01, 0.08)),
        ("together", 0.1): ((0.79, 0.13), (0.25, 0.10)), ("line", 0.02): ((0.53, 0.17), (-0.01, 0.07)),
        ("line", 0.05): ((0.68, 0.19), (0.14, 0.12)), ("line", 0.1): ((0.88, 0.20), (0.34, 0.09)),
        ("line", 0.2): ((1.29, 0.22), (0.75, 0.14)), ("level", 0.02): ((0.53, 0.19), (-0.01, 0.04)),
        ("level", 0.05): ((0.55, 0.20), (0.01, 0.07)), ("level", 0.1): ((0.74, 0.18), (0.20, 0.15)),
        ("all", "rows"): ((0.63, 0.19), (0.09, 0.13)), ("all", 0.05): ((0.69, 0.19), (0.15, 0.12)),
        ("all", "v2 rows"): ((0.58, 0.17), (0.04, 0.11)), ("interval", "v2 rows"): ((0.60, 0.23), (0.06, 0.15)),
    },
    (INVERTED, 1): {
        "budget": ((1.23, 0.13), (0.00, 0.00)), "six": ((1.42, 0.35), (0.20, 0.26)),
        "hour": ((3.18, 1.27), (1.95, 1.29)), "four": ((17.77, 3.02), (16.54, 3.01)),
        "long46": ((68.12, 2.92), (66.89, 2.99)), "tfln": ((38.58, 5.56), (37.35, 5.63)),
        "cal": ((1.14, 0.18), (-0.09, 0.08)), ("together", 0.01): ((1.33, 0.14), (0.10, 0.06)),
        ("together", 0.02): ((1.39, 0.13), (0.16, 0.06)), ("together", 0.05): ((2.05, 0.16), (0.82, 0.05)),
        ("together", 0.1): ((4.43, 0.33), (3.20, 0.21)), ("line", 0.02): ((1.29, 0.19), (0.06, 0.07)),
        ("line", 0.05): ((1.34, 0.23), (0.11, 0.13)), ("line", 0.1): ((1.61, 0.30), (0.39, 0.18)),
        ("line", 0.2): ((3.03, 0.35), (1.81, 0.25)), ("level", 0.02): ((1.20, 0.12), (-0.03, 0.04)),
        ("level", 0.05): ((1.28, 0.20), (0.05, 0.11)), ("level", 0.1): ((1.43, 0.26), (0.21, 0.15)),
        ("all", "rows"): ((1.52, 0.28), (0.29, 0.17)), ("all", 0.05): ((2.20, 0.30), (0.98, 0.18)),
        ("all", "v2 rows"): ((1.38, 0.24), (0.15, 0.12)), ("interval", "v2 rows"): ((1.60, 0.50), (0.37, 0.40)),
    },
    (INVERTED, 2): {
        "budget": ((0.62, 0.07), (0.00, 0.00)), "six": ((0.90, 0.33), (0.28, 0.27)),
        "hour": ((2.61, 1.25), (1.99, 1.26)), "four": ((16.69, 3.09), (16.07, 3.09)),
        "long46": ((68.13, 2.92), (67.51, 2.97)), "tfln": ((37.99, 5.70), (37.38, 5.74)),
        "cal": ((0.61, 0.10), (-0.01, 0.05)), ("together", 0.01): ((0.74, 0.12), (0.13, 0.08)),
        ("together", 0.02): ((0.85, 0.08), (0.24, 0.05)), ("together", 0.05): ((1.51, 0.20), (0.90, 0.13)),
        ("together", 0.1): ((3.84, 0.37), (3.23, 0.31)), ("line", 0.02): ((0.74, 0.10), (0.13, 0.04)),
        ("line", 0.05): ((0.86, 0.12), (0.25, 0.07)), ("line", 0.1): ((1.15, 0.15), (0.54, 0.09)),
        ("line", 0.2): ((2.49, 0.29), (1.88, 0.23)), ("level", 0.02): ((0.63, 0.08), (0.01, 0.02)),
        ("level", 0.05): ((0.78, 0.13), (0.17, 0.06)), ("level", 0.1): ((0.90, 0.16), (0.29, 0.10)),
        ("all", "rows"): ((0.98, 0.13), (0.37, 0.07)), ("all", 0.05): ((1.55, 0.23), (0.93, 0.17)),
        ("all", "v2 rows"): ((0.90, 0.12), (0.28, 0.06)), ("interval", "v2 rows"): ((1.09, 0.36), (0.48, 0.29)),
    },
    (MNIST, 1): {
        "budget": ((0.34, 0.07), (0.00, 0.00)), "six": ((0.33, 0.09), (-0.01, 0.02)),
        "hour": ((0.51, 0.07), (0.17, 0.05)), "four": ((0.74, 0.07), (0.40, 0.08)),
        "long46": ((9.36, 0.96), (9.02, 0.97)), "tfln": ((2.81, 0.22), (2.47, 0.20)),
        "cal": ((0.21, 0.05), (-0.13, 0.03)), ("together", 0.01): ((0.34, 0.07), (0.00, 0.02)),
        ("together", 0.02): ((0.28, 0.08), (-0.06, 0.03)), ("together", 0.05): ((0.38, 0.07), (0.04, 0.04)),
        ("together", 0.1): ((0.53, 0.05), (0.19, 0.04)), ("line", 0.02): ((0.36, 0.06), (0.02, 0.05)),
        ("line", 0.05): ((0.43, 0.06), (0.09, 0.07)), ("line", 0.1): ((0.56, 0.07), (0.22, 0.07)),
        ("line", 0.2): ((0.93, 0.10), (0.59, 0.07)), ("level", 0.02): ((0.37, 0.03), (0.03, 0.05)),
        ("level", 0.05): ((0.40, 0.03), (0.05, 0.06)), ("level", 0.1): ((0.44, 0.05), (0.10, 0.10)),
        ("all", "rows"): ((0.37, 0.07), (0.03, 0.11)), ("all", 0.05): ((0.46, 0.03), (0.12, 0.08)),
        ("all", "v2 rows"): ((0.38, 0.06), (0.04, 0.10)), ("interval", "v2 rows"): ((0.40, 0.03), (0.06, 0.06)),
    },
    (MNIST, 2): {
        "budget": ((0.15, 0.02), (0.00, 0.00)), "six": ((0.09, 0.03), (-0.06, 0.04)),
        "hour": ((0.27, 0.06), (0.12, 0.05)), "four": ((0.52, 0.07), (0.37, 0.08)),
        "long46": ((8.84, 0.86), (8.69, 0.85)), "tfln": ((2.50, 0.17), (2.35, 0.17)),
        "cal": ((0.09, 0.04), (-0.06, 0.03)), ("together", 0.01): ((0.16, 0.03), (0.01, 0.02)),
        ("together", 0.02): ((0.17, 0.03), (0.02, 0.02)), ("together", 0.05): ((0.17, 0.05), (0.02, 0.05)),
        ("together", 0.1): ((0.33, 0.03), (0.18, 0.04)), ("line", 0.02): ((0.20, 0.03), (0.05, 0.03)),
        ("line", 0.05): ((0.20, 0.04), (0.05, 0.05)), ("line", 0.1): ((0.31, 0.03), (0.16, 0.05)),
        ("line", 0.2): ((0.70, 0.08), (0.55, 0.08)), ("level", 0.02): ((0.20, 0.02), (0.05, 0.03)),
        ("level", 0.05): ((0.18, 0.03), (0.03, 0.03)), ("level", 0.1): ((0.27, 0.04), (0.12, 0.04)),
        ("all", "rows"): ((0.21, 0.04), (0.06, 0.03)), ("all", 0.05): ((0.24, 0.04), (0.09, 0.03)),
        ("all", "v2 rows"): ((0.19, 0.04), (0.04, 0.04)), ("interval", "v2 rows"): ((0.19, 0.03), (0.04, 0.02)),
    },
}
DRIFT = (("six", "six minutes of TFLT's drift"), ("hour", "an hour"), ("four", "four hours"),
         ("long46", "46 hours"), ("tfln", "an hour of TFLN's"), ("cal", "an hour of TFLT's, then calibrated"))
SOURCE = ((("together", 0.01), "the lines together, 1%"), (("together", 0.02), "2%"), (("together", 0.05), "5%"),
          (("together", 0.1), "10%"), (("line", 0.02), "a line on its own, 2%"), (("line", 0.05), "5%"),
          (("line", 0.1), "10%"), (("line", 0.2), "20%"), (("level", 0.02), "the lines' level, 2%"),
          (("level", 0.05), "5%"), (("level", 0.1), "10%"),
          (("all", "rows"), "all three, as budgeted: 2%, 5%, 5%"), (("all", 0.05), "all three at 5%"),
          (("all", "v2 rows"), "all three, as B16 has them: 1%, 5%, 5%"))
THREE = (("together", ROWS3[0]), ("line", ROWS3[1]), ("level", ROWS3[2]))
HELD = ("interval", "v2 rows")                 # B16's three rows at the end of B15's interval


# ---- what follows from them ---------------------------------------------------------
def lost(w, v, row="budget"):
    return RUN[(w, v)][row][0]


def adds(w, v, row):
    return RUN[(w, v)][row][1]


def moved(w, row):
    """What a row adds to version 2, less what it adds to version 1: in points,
    and in standard errors of the two."""
    a, b = adds(w, 2, row), adds(w, 1, row)
    err = math.hypot(a[1], b[1])
    return a[0] - b[0], (a[0] - b[0]) / err if err else 0.0


def share(w, v, row):
    """What a row adds, over its version's loss as budgeted."""
    return adds(w, v, row)[0] / lost(w, v)[0]


def kept(w, row):
    """What version 2 still has over version 1 with that row on both."""
    return lost(w, 1, row)[0] - lost(w, 2, row)[0]


def holds(w, v, row=("all", "rows")):
    """Whether a row adds under the budget's tenth."""
    return adds(w, v, row)[0] < BUDGET


def together_for_a_tenth(w, v, slack=0.035):
    """The largest size run at which the lines together add about a tenth: no
    more than it and a third again, which is inside every such row's error."""
    ok = [x for (kind, x) in (k for k, _ in SOURCE) if kind == "together" and adds(w, v, ("together", x))[0] <= BUDGET + slack]
    return max(ok) if ok else None


# ---- B15 and B16 -----------------------------------------------------------------------
def interval_shots(seconds=INTERVAL_S):
    """Shots between two calibrations."""
    return seconds * FS


def period_bits(seconds=INTERVAL_S):
    """Bits a count of shot-clock cycles needs to hold that interval."""
    return math.ceil(math.log2(interval_shots(seconds)))


def period_units(seconds=INTERVAL_S):
    """That interval as PTA_CAL_PER holds it: the nearest whole unit of 2^16 shot-clock cycles."""
    return int(interval_shots(seconds) / 2 ** PERIOD_UNIT_LOG2 + 0.5)


def period_reach_hours():
    """The longest interval the word can hold."""
    return (2 ** PERIOD_BITS - 1) * 2 ** PERIOD_UNIT_LOG2 / FS / 3600


def cal_shots():
    """A calibration's probe shots: a row a shot, both banks, each probe averaged."""
    return BANKS * TILE[0] * CAL_AVERAGE


def duty(seconds=INTERVAL_S):
    """The share of the tile's shots a calibration's probes take."""
    return cal_shots() / interval_shots(seconds)


def held(w, v=2):
    """A version with everything it is held to: B16's three rows of a source, at
    the end of B15's interval.  Points lost, and what that adds to its budget."""
    return RUN[(w, v)][HELD]


def pm(x):
    return f"{x[0]:.2f} +-{x[1]:.2f}"


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def table(rows, what):
    print(f"  {'':<38}" + "".join(f"{w + ', v' + str(v):>22}" for w in WORKLOADS for v in VERSIONS))
    for key, label in rows:
        print(f"  {label:<38}" + "".join(f"{pm(what(w, v, key)):>22}" for w in WORKLOADS for v in VERSIONS))


def main():
    print("Version 2, where version 1 was measured: drift, and the source's rows.")
    print(f"The working tile, {laser.name(TILE)} on two buses.  Five networks a data set.")

    section("1. Each version as budgeted, and what drift adds to it")
    table((("budget", "as budgeted, points lost"),), lambda w, v, k: lost(w, v))
    table(DRIFT, adds)
    table((("hour", "so after an hour, points lost"), ("six", "and after six minutes")), lost)

    section("2. What drift does to version 2, less what it does to version 1")
    print(f"  {'':<38}" + "".join(f"{w:>18}{'in errors':>12}" for w in WORKLOADS))
    for key, label in DRIFT:
        print(f"  {label:<38}" + "".join(f"{moved(w, key)[0]:>+18.2f}{moved(w, key)[1]:>+12.1f}" for w in WORKLOADS))
    hour = "hour"
    print(f"  {'an hour, over the budget':<38}" + "".join(
        f"{f'{share(w, 1, hour):.0%} and {share(w, 2, hour):.0%}':>30}" for w in WORKLOADS))
    print(f"  {'version 2 over version 1: as budgeted':<38}" + "".join(f"{kept(w, 'budget'):>30.2f}" for w in WORKLOADS))
    print(f"  {'after an hour':<38}" + "".join(f"{kept(w, 'hour'):>30.2f}" for w in WORKLOADS))
    print(f"  {'after four':<38}" + "".join(f"{kept(w, 'four'):>30.2f}" for w in WORKLOADS))

    section("3. What a source's noise adds, through a balanced pair")
    table(SOURCE, adds)
    print(f"  {'the three hold their tenth':<38}" + "".join(f"{('yes' if holds(w, v) else 'NO'):>22}" for w in WORKLOADS for v in VERSIONS))
    print(f"  {'the lines together, for a tenth':<38}" + "".join(
        f"{f'{together_for_a_tenth(w, v):.0%}, {noise.db_hz(together_for_a_tenth(w, v)):.0f} dB/Hz':>22}" for w in WORKLOADS for v in VERSIONS))

    section("4. Version 2 as B15 and B16 hold it")
    print(f"  Calibrated every {INTERVAL_S // 60} minutes: {interval_shots():.1e} shots apart, {86400 // INTERVAL_S} times a day.  A calibration's")
    print(f"  probes are {cal_shots():,} shots, {cal_shots() / FS * 1e6:.1f} us, one part in {1 / duty():,.0f} of the tile's.  A count of shot-clock")
    print(f"  cycles needs {period_bits()} bits to hold the interval, and {period_bits(HOURLY_S)} for an hour; PTA_CAL_PER has {PERIOD_BITS}.")
    print(f"  So it counts 2^{PERIOD_UNIT_LOG2} of them: {period_units():,} units for the interval, {period_units(HOURLY_S):,} for an hour,")
    print(f"  and {period_reach_hours():.0f} hours at the most.")
    print(f"  The lines together at {ROWS_V2[0]:.0%}: {noise.db_hz(ROWS_V2[0]):.0f} dB/Hz over the shot rate, where {ROWS3[0]:.0%} is {noise.db_hz(ROWS3[0]):.0f}.")
    table((("budget", "as budgeted, points lost"),), lambda w, v, k: lost(w, v))
    table((("six", "six minutes of drift adds"), (("all", "rows"), "the three at 2%, 5%, 5% add"),
           (("all", "v2 rows"), "at 1%, 5%, 5%"), (HELD, "and after six minutes of drift")), adds)
    table(((HELD, "which is, in points lost"),), lost)

    findings()
    checks()


def findings():
    m, f, i = WORKLOADS
    print()
    print("What this says, seven readings.")
    print()
    print(f"  1. DRIFT ADDS TO VERSION 2 WHAT IT ADDS TO VERSION 1.  An hour adds {adds(m, 2, 'hour')[0]:.2f}, {adds(f, 2, 'hour')[0]:.2f} and {adds(i, 2, 'hour')[0]:.2f}")
    print(f"     points on MNIST, Fashion-MNIST and MNIST inverted, where it added {adds(m, 1, 'hour')[0]:.2f}, {adds(f, 1, 'hour')[0]:.2f} and {adds(i, 1, 'hour')[0]:.2f}.")
    worst = max(abs(moved(w, k)[1]) for w in WORKLOADS for k, _ in DRIFT if k != "cal" and abs(adds(w, 1, k)[0]) > 0.5)
    print(f"     Wherever drift adds over half a point the two versions are within {worst:.1f} of a standard")
    print("     error of each other.  A better converter and a quieter receiver neither hide")
    print("     drift nor expose it.  And calibration returns version 2 to its budget: within")
    print(f"     {max(abs(adds(w, 2, 'cal')[0]) for w in WORKLOADS):.2f} of a point on all three.")
    print()
    print(f"  2. SO AFTER AN HOUR, DRIFT IS MOST OF WHAT THE HARDER SETS LOSE.  It adds {share(f, 2, 'hour'):.0%} of")
    print(f"     version 2's budget on Fashion-MNIST and {share(i, 2, 'hour'):.0%} on the inverted set, where it added")
    print(f"     {share(f, 1, 'hour'):.0%} and {share(i, 1, 'hour'):.0%} of version 1's.  Version 2 bought {kept(f, 'budget'):.2f} and {kept(i, 'budget'):.2f} of a point there,")
    print(f"     and an hour's drift takes back {adds(f, 2, 'hour')[0]:.2f} and {adds(i, 2, 'hour')[0]:.2f}.  At six minutes it takes {adds(f, 2, 'six')[0]:.2f} and {adds(i, 2, 'six')[0]:.2f}.")
    print(f"     On MNIST an hour is {adds(m, 2, 'hour')[0]:.2f}, which is most of a budget of {lost(m, 2)[0]:.2f}.")
    print()
    print(f"  3. THE SOURCE'S ROWS HOLD AT VERSION 2 ON MNIST AND ON FASHION-MNIST.  B12's and B13's")
    print(f"     three rows together add {adds(m, 2, ('all', 'rows'))[0]:.2f} and {adds(f, 2, ('all', 'rows'))[0]:.2f}, where at version 1 they add {adds(m, 1, ('all', 'rows'))[0]:.2f} and {adds(f, 1, ('all', 'rows'))[0]:.2f}:")
    print("     under the tenth of a point they were sized to.")
    print()
    print(f"  4. THEY DO NOT HOLD ON THE INVERTED SET, AT EITHER VERSION.  There the three add {adds(i, 1, ('all', 'rows'))[0]:.2f}")
    print(f"     and {adds(i, 2, ('all', 'rows'))[0]:.2f}.  The dear row is the one the lines share: 2% adds {adds(i, 1, ('together', 0.02))[0]:.2f} and {adds(i, 2, ('together', 0.02))[0]:.2f}, 5% adds")
    print(f"     {adds(i, 1, ('together', 0.05))[0]:.2f} and {adds(i, 2, ('together', 0.05))[0]:.2f}, and 10% over three points, where on the other two sets 10% adds")
    print(f"     about a fifth.  For that row to add about a tenth there it has to be {together_for_a_tenth(i, 2):.0%}, which is")
    print(f"     {noise.db_hz(together_for_a_tenth(i, 2)):.0f} dB/Hz over the shot rate where the row's 2% is {noise.db_hz(ROWS3[0]):.0f}.")
    print()
    print(f"  5. AND ON THAT SET A SOURCE COSTS A LITTLE MORE AT VERSION 2.  A tenth more on each row")
    print(f"     that costs anything: {adds(i, 2, THREE[0])[0]:.2f} for {adds(i, 1, THREE[0])[0]:.2f}, {adds(i, 2, THREE[1])[0]:.2f} for {adds(i, 1, THREE[1])[0]:.2f}, {adds(i, 2, THREE[2])[0]:.2f} for {adds(i, 1, THREE[2])[0]:.2f}.  On the other two sets")
    print("     the versions do not differ one way.  Drift did not do this.")
    print()
    print(f"  6. B15, SIX MINUTES: VERSION 2 KEEPS WHAT IT BOUGHT ON TWO SETS OF THREE.  With a source at")
    print(f"     B16's rows and at the end of an interval it loses {held(m)[0][0]:.2f}, {held(f)[0][0]:.2f} and {held(i)[0][0]:.2f} points, which is")
    print(f"     {held(m)[1][0]:.2f}, {held(f)[1][0]:.2f} and {held(i)[1][0]:.2f} over its budget.  It costs {86400 // INTERVAL_S} calibrations a day where an hour was")
    print(f"     {86400 // HOURLY_S}, and one part in {1 / duty() / 1e6:.0f} million of the tile's shots.  As single cycles it did not fit")
    print(f"     the register: {period_bits()} bits at a shot a nanosecond, where PTA_CAL_PER has {PERIOD_BITS}, and an hour did not")
    print(f"     either.  The map now counts 2^{PERIOD_UNIT_LOG2} of them, and six minutes is {period_units():,} units.")
    print()
    r2, r1 = adds(i, 2, ("all", "rows"))[0], adds(i, 2, ("all", "v2 rows"))[0]
    print(f"  7. B16, 1%: IT BUYS {r2 - r1:.2f} OF A POINT ON THE INVERTED SET AND NOTHING ELSEWHERE.  There the")
    print(f"     three rows add {r1:.2f} where they added {r2:.2f}.  They are not inside a tenth: the two rows a")
    print(f"     line carries add {adds(i, 2, THREE[1])[0]:.2f} and {adds(i, 2, THREE[2])[0]:.2f} by themselves, and are now most of it.  On MNIST")
    print(f"     and Fashion-MNIST the three add {adds(m, 2, ('all', 'v2 rows'))[0]:.2f} and {adds(f, 2, ('all', 'v2 rows'))[0]:.2f}, as they did at 2%, within the scatter.")


def checks():
    """Every claim above, as an assert."""
    m, f, i = WORKLOADS
    assert TILE == (128, 64) and BUDGET == 0.10 and ROWS3 == (0.02, 0.05, 0.05)
    assert set(RUN) == {(w, v) for w in WORKLOADS for v in VERSIONS}
    keys = {"budget", HELD} | {k for k, _ in DRIFT} | {k for k, _ in SOURCE}
    assert all(set(RUN[c]) == keys for c in RUN) and len(keys) == 22

    # 0. The figures agree with what the plan already holds.
    #    Each version as budgeted is pta_tighten.py's, and adds nothing to itself.
    for w in WORKLOADS:
        k = WORKLOADS.index(w)
        assert lost(w, 1) == tighten.LOST["v1"][k] and lost(w, 2) == tighten.LOST["adc_noise"][k]
        assert all(adds(w, v, "budget") == (0.00, 0.00) for v in VERSIONS)
    #    Version 1 under drift is pta_workload.py's, row for row.
    for w in WORKLOADS:
        for key in ("six", "hour", "four", "long46", "tfln", "cal"):
            assert lost(w, 1, key) == workload.ROWS[(w, TILE)][key], (w, key)
    #    Version 1 with a source's noise on MNIST is pta_source_noise.py's, on this tile.
    for x, cells in noise.PAIR_128.items():
        for kind, cell in zip(("together", "line", "level"), cells):
            if cell is not None and (kind, x) in keys:
                assert lost(m, 1, (kind, x)) == cell, (kind, x)
    assert lost(m, 1, ("all", "rows")) == noise.JOINT_PAIR_128[(0.02, 0.05, 0.05)]
    assert lost(m, 1, ("all", 0.05)) == noise.JOINT_PAIR_128[(0.05, 0.05, 0.05)]
    #    What a row adds is its loss less its version's, to the rounding.
    for c in RUN:
        for key in keys:
            assert abs(RUN[c][key][0][0] - RUN[c]["budget"][0][0] - RUN[c][key][1][0]) < 0.011, (c, key)
    assert shot.REQ["v2"] == tighten.NOTCH_REQ
    #    Every figure, counted and summed, as grx930's tables printed them: a
    #    standard error mistyped moves no reading, and would move this.
    figures = [x for c in RUN.values() for cell in c.values() for pair in cell for x in pair]
    assert len(figures) == 528 and abs(sum(figures) - 907.34) < 1e-6

    # 1. Reading 1.  Drift adds the same.
    assert [adds(w, 2, "hour")[0] for w in WORKLOADS] == [0.12, 0.67, 1.99]
    assert [adds(w, 1, "hour")[0] for w in WORKLOADS] == [0.17, 0.69, 1.95]
    assert [adds(w, 2, "four")[0] for w in WORKLOADS] == [0.37, 2.36, 16.07]
    assert [adds(w, 1, "four")[0] for w in WORKLOADS] == [0.40, 2.34, 16.54]
    drift_rows = [k for k, _ in DRIFT]
    #    At most eight hundredths of a point where it adds under half a point,
    #    and under half a standard error where it adds over.
    small = [(w, k) for w in WORKLOADS for k in drift_rows if abs(adds(w, 1, k)[0]) <= 0.5]
    large = [(w, k) for w in WORKLOADS for k in drift_rows if abs(adds(w, 1, k)[0]) > 0.5]
    assert len(small) == 8 and len(large) == 10
    assert max(abs(moved(w, k)[0]) for w, k in small) < 0.085
    assert max(abs(moved(w, k)[1]) for w, k in large) < 0.5
    #    In standard errors of the two: MNIST's calibrated row is the furthest
    #    apart of the eighteen, at 1.6, and its six minutes next.
    assert abs(moved(m, "cal")[1] - 1.65) < 0.01 and abs(moved(m, "six")[1] + 1.12) < 0.01
    assert max(abs(moved(w, k)[1]) for w in WORKLOADS for k in drift_rows) == abs(moved(m, "cal")[1])
    assert all(abs(adds(w, 2, "four")[0] / adds(w, 1, "four")[0] - 1) < 0.08 for w in WORKLOADS)
    #    Calibration returns version 2 to its budget.
    assert max(abs(adds(w, 2, "cal")[0]) for w in WORKLOADS) == 0.06
    assert all(abs(lost(w, 2, "cal")[0] - lost(w, 2)[0]) < 0.07 for w in WORKLOADS)

    # 2. Reading 2.  An hour, against each budget.
    assert [round(share(w, 2, "hour"), 2) for w in WORKLOADS] == [0.80, 1.24, 3.21]
    assert [round(share(w, 1, "hour"), 2) for w in WORKLOADS] == [0.50, 0.59, 1.59]
    assert [round(kept(w, "budget"), 2) for w in WORKLOADS] == [0.19, 0.62, 0.61]
    assert all(adds(w, 2, "hour")[0] > kept(w, "budget") for w in (f, i))
    assert [adds(w, 2, "six")[0] for w in WORKLOADS] == [-0.06, 0.08, 0.28]
    assert [lost(w, 2, "hour")[0] for w in WORKLOADS] == [0.27, 1.21, 2.61]
    assert [lost(w, 1, "hour")[0] for w in WORKLOADS] == [0.51, 1.84, 3.18]
    #    What version 2 has over version 1 is still there after the drift.
    assert all(abs(kept(w, "hour") - kept(w, "budget")) < 0.06 for w in WORKLOADS)
    assert all(abs(kept(w, "four") - kept(w, "budget")) < 0.5 for w in WORKLOADS)

    # 3. Reading 3.  The source's rows on MNIST and Fashion-MNIST.
    rows = ("all", "rows")
    assert [adds(w, 2, rows)[0] for w in WORKLOADS] == [0.06, 0.09, 0.37]
    assert [adds(w, 1, rows)[0] for w in WORKLOADS] == [0.03, 0.06, 0.29]
    assert all(holds(w, v) for w in (m, f) for v in VERSIONS) and not any(holds(i, v) for v in VERSIONS)
    #    Row by row the versions differ by a tenth at most there, and not one way.
    src = [k for k, _ in SOURCE]
    d = [moved(w, k)[0] for w in (m, f) for k in src]
    assert max(abs(x) for x in d) < 0.115 and min(d) < 0 < max(d)

    # 4. Reading 4.  The inverted set, and the row the lines share.
    assert [adds(i, v, ("together", 0.02))[0] for v in VERSIONS] == [0.16, 0.24]
    assert [adds(i, v, ("together", 0.05))[0] for v in VERSIONS] == [0.82, 0.90]
    assert all(adds(i, v, ("together", 0.1))[0] > 3 for v in VERSIONS)
    assert all(0.15 < adds(w, v, ("together", 0.1))[0] < 0.26 for w in (m, f) for v in VERSIONS)
    assert [together_for_a_tenth(i, v) for v in VERSIONS] == [0.01, 0.01]
    assert [adds(i, v, ("together", 0.01))[0] for v in VERSIONS] == [0.10, 0.13]
    assert all(together_for_a_tenth(w, v) == 0.05 for w in (m, f) for v in VERSIONS)
    assert abs(noise.db_hz(0.02) + 124) < 0.5 and abs(noise.db_hz(0.01) + 130) < 0.5
    #    1% is still 14 dB easier than the plan first assumed of a source.
    assert abs(noise.db_hz(0.01) - noise.db_hz(noise.assumed_rms()) - 14.2) < 0.1
    #    Of the three rows it is the dearest there at either version, at the size budgeted.
    assert all(adds(i, 1, THREE[0])[0] > adds(i, 1, k)[0] for k in THREE[1:])
    assert adds(i, 2, THREE[0])[0] > adds(i, 2, THREE[2])[0] and abs(adds(i, 2, THREE[0])[0] - adds(i, 2, THREE[1])[0]) < 0.02
    #    And at one size, 5%, by far.
    assert all(adds(i, v, ("together", 0.05))[0] > 3 * max(adds(i, v, ("line", 0.05))[0], adds(i, v, ("level", 0.05))[0]) for v in VERSIONS)

    # 5. Reading 5.  A tenth more at version 2 on the inverted set.
    assert [(adds(i, 2, k)[0], adds(i, 1, k)[0]) for k in THREE] == [(0.24, 0.16), (0.25, 0.11), (0.17, 0.05)]
    assert all(0.07 < moved(i, k)[0] < 0.15 for k in THREE) and 0.07 < moved(i, rows)[0] < 0.09

    # 6. Reading 6.  B15: six minutes.
    assert INTERVAL_S == 360 and FS == 1e9 and interval_shots() == 3.6e11
    assert 86400 // INTERVAL_S == 240 and 86400 // HOURLY_S == 24
    assert cal_shots() == 4096 and abs(cal_shots() / FS - 4.096e-6) < 1e-12
    assert abs(duty() - 1.14e-8) < 0.01e-8 and 87e6 < 1 / duty() < 89e6 and abs(duty() / duty(HOURLY_S) - 10) < 1e-9
    assert (period_bits(), period_bits(HOURLY_S)) == (39, 42) and period_bits() > PERIOD_BITS
    assert interval_shots(HOURLY_S) == 3.6e12 and 8.7e8 < 1 / duty(HOURLY_S) < 8.9e8
    #    A unit of 2^16 cycles would hold 78 hours in the register's 32 bits.
    assert abs(2 ** (PERIOD_BITS + 16) / FS / 3600 - 78.2) < 0.1 and abs(2 ** 16 / FS - 65.5e-6) < 0.1e-6
    #    TFLN's hour, at version 2: 2.4 to 37 points.
    assert (adds(m, 2, "tfln")[0], adds(i, 2, "tfln")[0]) == (2.35, 37.38)
    assert [adds(w, 2, "hour")[0] for w in WORKLOADS] == [0.12, 0.67, 1.99]
    assert 2 ** PERIOD_BITS / FS < 4.3                           # 32 bits of cycles is 4.3 seconds
    #    In the map's unit both fit, to half a unit, and the word reaches 78 hours.
    assert (period_units(), period_units(HOURLY_S)) == (5_493_164, 54_931_641)
    assert all(period_units(s) < 2 ** PERIOD_BITS for s in (INTERVAL_S, HOURLY_S))
    assert all(abs(period_units(s) * 2 ** PERIOD_UNIT_LOG2 - interval_shots(s)) <= 2 ** (PERIOD_UNIT_LOG2 - 1)
               for s in (INTERVAL_S, HOURLY_S))
    assert 78 < period_reach_hours() < 78.3 and period_bits() - PERIOD_UNIT_LOG2 < PERIOD_BITS
    assert [held(w)[0][0] for w in WORKLOADS] == [0.19, 0.60, 1.09]
    assert [held(w)[1][0] for w in WORKLOADS] == [0.04, 0.06, 0.48]
    assert all(held(w)[1][0] < BUDGET for w in (m, f)) and held(i)[1][0] > 4 * BUDGET
    #    Against an hour with no source at all, six minutes with one is the
    #    better on every set, and by half a point and more on the harder two.
    assert all(held(w)[0][0] < lost(w, 2, "hour")[0] for w in WORKLOADS)
    assert [round(lost(w, 2, "hour")[0] - held(w)[0][0], 2) for w in WORKLOADS] == [0.08, 0.61, 1.52]
    #    Version 2 so held loses a little over half what version 1 does as first
    #    budgeted on two sets, and nine tenths of it on the third.
    assert [round(held(w)[0][0] / lost(w, 1)[0], 2) for w in WORKLOADS] == [0.56, 0.52, 0.89]
    #    Version 1 held the same way: 0.40, 1.37 and 1.60.
    assert [held(w, 1)[0][0] for w in WORKLOADS] == [0.40, 1.37, 1.60]

    # 7. Reading 7.  B16: 1%.
    assert ROWS_V2 == (0.01, ROWS3[1], ROWS3[2]) and ROWS_V2[0] == ROWS3[0] / 2
    assert abs(noise.db_hz(ROWS_V2[0]) + 130) < 0.05 and abs(noise.db_hz(ROWS3[0]) - noise.db_hz(ROWS_V2[0]) - 6.02) < 0.01
    #    For a receiver that averages over a shot: -121 and -127.
    averaged = dict(noise.bands())["a receiver that averages over a shot"]
    assert abs(noise.db_hz(ROWS3[0], averaged) + 121) < 0.05 and abs(noise.db_hz(ROWS_V2[0], averaged) + 127) < 0.05
    assert abs(noise.db_hz(ROWS3[0]) - noise.db_hz(noise.assumed_rms()) - 20.2) < 0.1
    v2rows = ("all", "v2 rows")
    assert [adds(w, 2, v2rows)[0] for w in WORKLOADS] == [0.04, 0.04, 0.28]
    assert [adds(w, 1, v2rows)[0] for w in WORKLOADS] == [0.04, 0.14, 0.15]
    assert round(adds(i, 2, rows)[0] - adds(i, 2, v2rows)[0], 2) == 0.09
    assert adds(i, 2, v2rows)[0] > 2 * BUDGET and not adds(i, 2, v2rows)[0] < BUDGET
    #    The two rows a line carries, alone, are each most of what is left there.
    assert (adds(i, 2, THREE[1])[0], adds(i, 2, THREE[2])[0]) == (0.25, 0.17)
    assert adds(i, 2, THREE[1])[0] > adds(i, 2, ("together", ROWS_V2[0]))[0] and adds(i, 2, ("together", ROWS_V2[0]))[0] == 0.13
    #    On the other two sets 1% and 2% are a draw.
    assert all(abs(moved_rows) < 0.06 for moved_rows in (adds(w, 2, v2rows)[0] - adds(w, 2, rows)[0] for w in (m, f)))
    #    A line at 2% and the level at 2% would be 0.13 and 0.01 there: run, and not adopted.
    assert (adds(i, 2, ("line", 0.02))[0], adds(i, 2, ("level", 0.02))[0]) == (0.13, 0.01)

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
