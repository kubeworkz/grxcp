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
    },
}
DRIFT = (("six", "six minutes of TFLT's drift"), ("hour", "an hour"), ("four", "four hours"),
         ("long46", "46 hours"), ("tfln", "an hour of TFLN's"), ("cal", "an hour of TFLT's, then calibrated"))
SOURCE = ((("together", 0.01), "the lines together, 1%"), (("together", 0.02), "2%"), (("together", 0.05), "5%"),
          (("together", 0.1), "10%"), (("line", 0.02), "a line on its own, 2%"), (("line", 0.05), "5%"),
          (("line", 0.1), "10%"), (("line", 0.2), "20%"), (("level", 0.02), "the lines' level, 2%"),
          (("level", 0.05), "5%"), (("level", 0.1), "10%"),
          (("all", "rows"), "all three, as budgeted: 2%, 5%, 5%"), (("all", 0.05), "all three at 5%"))
THREE = (("together", ROWS3[0]), ("line", ROWS3[1]), ("level", ROWS3[2]))


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

    findings()
    checks()


def findings():
    m, f, i = WORKLOADS
    print()
    print("What this says, five readings.")
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


def checks():
    """Every claim above, as an assert."""
    m, f, i = WORKLOADS
    assert TILE == (128, 64) and BUDGET == 0.10 and ROWS3 == (0.02, 0.05, 0.05)
    assert set(RUN) == {(w, v) for w in WORKLOADS for v in VERSIONS}
    keys = {"budget"} | {k for k, _ in DRIFT} | {k for k, _ in SOURCE}
    assert all(set(RUN[c]) == keys for c in RUN) and len(keys) == 20

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
    assert len(figures) == 480 and abs(sum(figures) - 891.64) < 1e-6

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

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
