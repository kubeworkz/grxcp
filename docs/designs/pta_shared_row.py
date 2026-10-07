"""
The row a source's lines share, on the reference networks: does a trained network need the 1%?

B16 holds a source's noise to 1% rms a shot for the lines together, where B12
and B13 had 2%.  It was chosen on what the networks trained before lost on the
inverted set, and it lists a network trained for the tile among what would
reopen it, "which may not need it".  B17 has since made such networks the
reference, and they had been held at the 1% only.  grx930's harness has now run
them at both.

MEASURED IN A MODEL, by grx930 (c930/doc/pta_error_model_design_note.md section
5, "Does a trained network need the 1%?", 2026-10-07; `sim/pta_mnist.sh DIR WORK
refsource`).  The working tile, 128 x 64 on two buses, at version 2.  Two kinds
of network a data set, five of each, seeds 1 to 5: the ones trained before, and
B17's reference, eight epochs with noise of 10% on their sums.  Eleven rows a
network.  A source's noise is through a balanced pair (B13), in rms fractions of
a line's power: the lines together, a line on its own, and the lines' level.

  as budgeted       version 2 and nothing else
  together          the lines together alone, at 1, 2 and 5%
  a line, level     a line on its own at 5%; the lines' level at 5%
  three             all three: the lines together at 2% or 1%, with 5% and 5%
  six               six minutes of TFLT's drift, B15's interval
  held              six minutes and all three: at 1% it is the chip as B14, B15
                    and B16 hold it

Every network's own figure is here, as in pta_trained.py, because every
comparison is between two rows of the same network or two networks of a seed.

DERIVED here:

  a row's mean and    which have to be the two tables grx930 printed, cell for cell
  its error
  what a row adds     a network's accuracy as budgeted, less in that row
  what the 1% buys    a network's accuracy with the shared row at 1%, less at 2%:
                      alone, among the three, and held
  what a trained      the same for the reference networks, less for the ones
  network needs       trained before, seed by seed
  the most it could   a difference and 2.8 of its errors, which five networks put
  be                  it under at one in twenty

HOW SURE, AND HOW SMALL.  Five networks, so 2.8 errors for one in twenty, as in
pta_trained.py.  But the two rows of a pair run from one seed and share the
source's draws, at half the size, so a pair's difference has a small error: a
few hundredths of a point.  A hundredth of a point is one image in the ten
thousand.  A difference can be clear here and be four images.

WHAT THIS IS NOT: a source.  The term is first order and on the host's side of
the line, as in pta_source_noise.py.  The drift is a Mach-Zehnder's fit.  The
reference networks were trained with Gaussian noise and not with a source's.
Version 2 only, the working tile only, a pair only.  The two rows a line
carries were run at 5% and at no other size.  And it is pta_workload.py's three
data sets.

Standard library only.  Run:  python3 docs/designs/pta_shared_row.py
"""
import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_laser as laser
import pta_source_noise as noise
import pta_trained as trained
import pta_version2 as version2
import pta_workload as workload

MNIST, FASHION, INVERTED = workload.MNIST, workload.FASHION, workload.INVERTED
WORKLOADS = workload.WORKLOADS
TILE = laser.SMALLER                           # 128 x 64: B10
BEFORE, REFERENCE = "before", "reference"      # trained as the plan's networks were; B17's
KINDS = (BEFORE, REFERENCE)
SEEDS = 5
WAS, IS = version2.ROWS3[0], version2.ROWS_V2[0]   # 2%: B12's and B13's row.  1%: B16's
BUDGET = noise.BUDGET                          # a tenth of a point: what a row of the budget costs
T95 = trained.T95
ROWS = (
    ("budget", "as budgeted"),
    (("together", 0.01), "the lines together, 1%"), (("together", 0.02), "2%"), (("together", 0.05), "5%"),
    (("line", 0.05), "a line on its own, 5%"), (("level", 0.05), "the lines' level, 5%"),
    (("three", 0.02), "all three: 2%, 5%, 5%"), (("three", 0.01), "all three: 1%, 5%, 5%"),
    ("six", "six minutes of TFLT's drift"),
    (("held", 0.02), "six minutes and 2%, 5%, 5%"), (("held", 0.01), "six minutes and 1%, 5%, 5%"),
)
WHERE = (("together", "the lines together, alone"), ("three", "among the three"), ("held", "held, six minutes on"))

# ---- grx930's figures ----------------------------------------------------------
# (data set) -> kind -> its five networks' accuracy on their host, percent, seed 1 to 5.
HOSTS = {
    MNIST: {BEFORE: (97.44, 97.81, 97.17, 97.43, 97.41),
            REFERENCE: (97.68, 97.90, 97.57, 97.89, 97.72)},
    FASHION: {BEFORE: (87.72, 87.32, 86.87, 87.99, 87.48),
              REFERENCE: (87.43, 88.33, 87.63, 88.08, 87.96)},
    INVERTED: {BEFORE: (94.46, 93.64, 91.32, 93.37, 94.01),
               REFERENCE: (96.10, 95.68, 94.89, 94.33, 95.92)},
}
# (data set) -> kind -> row -> the same five networks' accuracy on the tile, percent.
RUN = {
    MNIST: {
        BEFORE: {
            "budget": (97.27, 97.70, 97.06, 97.27, 97.21),
            ("together", 0.01): (97.31, 97.70, 97.00, 97.31, 97.16),
            ("together", 0.02): (97.30, 97.63, 97.05, 97.29, 97.12),
            ("together", 0.05): (97.39, 97.67, 96.93, 97.33, 97.10),
            ("line", 0.05): (97.38, 97.61, 96.90, 97.19, 97.19),
            ("level", 0.05): (97.12, 97.67, 97.02, 97.27, 97.27),
            ("three", 0.02): (97.20, 97.71, 96.96, 97.11, 97.22),
            ("three", 0.01): (97.19, 97.70, 97.06, 97.10, 97.25),
            "six": (97.44, 97.63, 97.11, 97.34, 97.31),
            ("held", 0.02): (97.15, 97.60, 97.02, 97.13, 97.18),
            ("held", 0.01): (97.19, 97.63, 97.06, 97.19, 97.22),
        },
        REFERENCE: {
            "budget": (97.61, 97.66, 97.65, 97.84, 97.65),
            ("together", 0.01): (97.64, 97.55, 97.64, 97.84, 97.59),
            ("together", 0.02): (97.63, 97.51, 97.56, 97.81, 97.58),
            ("together", 0.05): (97.48, 97.59, 97.60, 97.85, 97.50),
            ("line", 0.05): (97.61, 97.69, 97.51, 97.88, 97.54),
            ("level", 0.05): (97.53, 97.67, 97.59, 97.83, 97.67),
            ("three", 0.02): (97.50, 97.63, 97.53, 97.90, 97.56),
            ("three", 0.01): (97.45, 97.65, 97.50, 97.90, 97.57),
            "six": (97.60, 97.68, 97.61, 97.82, 97.66),
            ("held", 0.02): (97.47, 97.72, 97.57, 97.91, 97.62),
            ("held", 0.01): (97.48, 97.69, 97.59, 97.88, 97.55),
        },
    },
    FASHION: {
        BEFORE: {
            "budget": (86.45, 86.83, 86.94, 87.65, 86.81),
            ("together", 0.01): (86.47, 86.73, 86.81, 87.71, 86.75),
            ("together", 0.02): (86.44, 86.83, 86.76, 87.72, 86.77),
            ("together", 0.05): (86.67, 86.80, 86.70, 87.71, 86.76),
            ("line", 0.05): (86.63, 86.87, 86.66, 87.52, 86.32),
            ("level", 0.05): (86.58, 86.92, 86.96, 87.37, 86.82),
            ("three", 0.02): (86.71, 86.66, 86.94, 87.12, 86.81),
            ("three", 0.01): (86.73, 86.71, 86.93, 87.24, 86.89),
            "six": (86.54, 86.90, 86.78, 87.48, 86.60),
            ("held", 0.02): (86.77, 86.89, 86.94, 87.16, 86.58),
            ("held", 0.01): (86.74, 86.90, 87.09, 87.07, 86.60),
        },
        REFERENCE: {
            "budget": (87.02, 87.76, 87.63, 87.42, 87.23),
            ("together", 0.01): (86.91, 87.74, 87.59, 87.44, 87.30),
            ("together", 0.02): (87.05, 87.85, 87.63, 87.47, 87.36),
            ("together", 0.05): (87.02, 87.65, 87.53, 87.47, 87.11),
            ("line", 0.05): (86.72, 87.87, 87.33, 87.37, 86.98),
            ("level", 0.05): (86.93, 87.66, 87.53, 87.60, 87.22),
            ("three", 0.02): (86.91, 87.55, 87.51, 87.43, 87.34),
            ("three", 0.01): (86.85, 87.64, 87.57, 87.50, 87.34),
            "six": (87.01, 87.61, 87.41, 87.67, 87.00),
            ("held", 0.02): (87.08, 87.41, 87.40, 87.55, 87.06),
            ("held", 0.01): (86.94, 87.51, 87.36, 87.55, 86.97),
        },
    },
    INVERTED: {
        BEFORE: {
            "budget": (93.65, 93.26, 90.63, 92.71, 93.47),
            ("together", 0.01): (93.55, 93.16, 90.24, 92.57, 93.57),
            ("together", 0.02): (93.49, 93.02, 90.22, 92.57, 93.24),
            ("together", 0.05): (92.47, 92.63, 89.39, 91.92, 92.82),
            ("line", 0.05): (93.44, 93.25, 90.24, 92.42, 93.14),
            ("level", 0.05): (93.43, 93.32, 90.38, 92.45, 93.30),
            ("three", 0.02): (93.05, 93.06, 90.29, 92.40, 93.08),
            ("three", 0.01): (93.17, 93.09, 90.40, 92.51, 93.14),
            "six": (92.43, 93.27, 90.26, 92.47, 93.88),
            ("held", 0.02): (91.94, 92.88, 89.93, 92.32, 93.69),
            ("held", 0.01): (92.15, 93.14, 90.05, 92.27, 93.73),
        },
        REFERENCE: {
            "budget": (95.67, 95.26, 94.38, 93.72, 95.28),
            ("together", 0.01): (95.70, 95.27, 94.44, 93.61, 95.27),
            ("together", 0.02): (95.62, 95.15, 94.47, 93.54, 95.29),
            ("together", 0.05): (95.04, 94.90, 94.24, 93.03, 94.82),
            ("line", 0.05): (95.62, 95.37, 94.35, 93.66, 95.30),
            ("level", 0.05): (95.50, 95.32, 94.51, 92.86, 95.54),
            ("three", 0.02): (95.29, 95.32, 94.51, 92.59, 95.40),
            ("three", 0.01): (95.47, 95.38, 94.39, 92.84, 95.42),
            "six": (95.72, 95.08, 94.24, 93.34, 95.54),
            ("held", 0.02): (95.46, 95.28, 94.38, 92.77, 95.44),
            ("held", 0.01): (95.52, 95.15, 94.24, 92.72, 95.50),
        },
    },
}
# The first table grx930's harness printed, mean and standard error: a row -> what the networks
# trained before lose, what the row adds to them as budgeted, and the same two for the reference.
PRINTED_LOST = {
    MNIST: {
        "budget": ((0.15, 0.02), (0.00, 0.00), (0.07, 0.05), (0.00, 0.00)),
        ("together", 0.01): ((0.16, 0.03), (0.01, 0.02), (0.10, 0.07), (0.03, 0.02)),
        ("together", 0.02): ((0.17, 0.03), (0.02, 0.02), (0.13, 0.07), (0.06, 0.03)),
        ("together", 0.05): ((0.17, 0.05), (0.02, 0.05), (0.15, 0.06), (0.08, 0.03)),
        ("line", 0.05): ((0.20, 0.04), (0.05, 0.05), (0.11, 0.04), (0.04, 0.04)),
        ("level", 0.05): ((0.18, 0.03), (0.03, 0.03), (0.09, 0.04), (0.02, 0.02)),
        ("three", 0.02): ((0.21, 0.04), (0.06, 0.03), (0.13, 0.05), (0.06, 0.03)),
        ("three", 0.01): ((0.19, 0.04), (0.04, 0.04), (0.14, 0.05), (0.07, 0.04)),
        "six": ((0.09, 0.03), (-0.06, 0.04), (0.08, 0.04), (0.01, 0.01)),
        ("held", 0.02): ((0.24, 0.03), (0.09, 0.02), (0.09, 0.05), (0.02, 0.04)),
        ("held", 0.01): ((0.19, 0.03), (0.04, 0.02), (0.11, 0.05), (0.04, 0.03)),
    },
    FASHION: {
        "budget": ((0.54, 0.22), (0.00, 0.00), (0.47, 0.13), (0.00, 0.00)),
        ("together", 0.01): ((0.58, 0.20), (0.04, 0.04), (0.49, 0.12), (0.02, 0.03)),
        ("together", 0.02): ((0.57, 0.20), (0.03, 0.04), (0.41, 0.11), (-0.06, 0.02)),
        ("together", 0.05): ((0.55, 0.16), (0.01, 0.08), (0.53, 0.13), (0.06, 0.03)),
        ("line", 0.05): ((0.68, 0.19), (0.14, 0.12), (0.63, 0.12), (0.16, 0.08)),
        ("level", 0.05): ((0.55, 0.20), (0.01, 0.07), (0.50, 0.11), (0.02, 0.05)),
        ("three", 0.02): ((0.63, 0.19), (0.09, 0.13), (0.54, 0.11), (0.06, 0.06)),
        ("three", 0.01): ((0.58, 0.17), (0.04, 0.11), (0.51, 0.11), (0.03, 0.05)),
        "six": ((0.62, 0.19), (0.08, 0.06), (0.55, 0.13), (0.07, 0.09)),
        ("held", 0.02): ((0.61, 0.19), (0.07, 0.14), (0.59, 0.14), (0.11, 0.09)),
        ("held", 0.01): ((0.60, 0.23), (0.06, 0.15), (0.62, 0.13), (0.15, 0.08)),
    },
    INVERTED: {
        "budget": ((0.62, 0.07), (0.00, 0.00), (0.52, 0.05), (0.00, 0.00)),
        ("together", 0.01): ((0.74, 0.12), (0.13, 0.08), (0.53, 0.07), (0.00, 0.03)),
        ("together", 0.02): ((0.85, 0.08), (0.24, 0.05), (0.57, 0.06), (0.05, 0.05)),
        ("together", 0.05): ((1.51, 0.20), (0.90, 0.13), (0.98, 0.12), (0.46, 0.10)),
        ("line", 0.05): ((0.86, 0.12), (0.25, 0.07), (0.52, 0.06), (0.00, 0.03)),
        ("level", 0.05): ((0.78, 0.13), (0.17, 0.06), (0.64, 0.21), (0.12, 0.20)),
        ("three", 0.02): ((0.98, 0.13), (0.37, 0.07), (0.76, 0.26), (0.24, 0.24)),
        ("three", 0.01): ((0.90, 0.12), (0.28, 0.06), (0.68, 0.21), (0.16, 0.19)),
        "six": ((0.90, 0.33), (0.28, 0.27), (0.60, 0.11), (0.08, 0.11)),
        ("held", 0.02): ((1.21, 0.37), (0.59, 0.32), (0.72, 0.21), (0.20, 0.20)),
        ("held", 0.01): ((1.09, 0.36), (0.48, 0.29), (0.76, 0.22), (0.24, 0.20)),
    },
}
# The second: how often the networks trained before are right with the shared row at 2%, at 1%,
# and the second less the first network by network; and the same three for the reference.
PRINTED_BUYS = {
    MNIST: {
        "together": ((97.28, 0.10), (97.30, 0.12), (0.02, 0.02), (97.62, 0.05), (97.65, 0.05), (0.03, 0.01)),
        "three": ((97.24, 0.13), (97.26, 0.11), (0.02, 0.02), (97.62, 0.07), (97.61, 0.08), (-0.01, 0.01)),
        "held": ((97.22, 0.10), (97.26, 0.10), (0.04, 0.00), (97.66, 0.07), (97.64, 0.07), (-0.02, 0.02)),
    },
    FASHION: {
        "together": ((86.90, 0.22), (86.89, 0.21), (-0.01, 0.03), (87.47, 0.13), (87.40, 0.14), (-0.08, 0.02)),
        "three": ((86.85, 0.08), (86.90, 0.10), (0.05, 0.02), (87.35, 0.12), (87.38, 0.14), (0.03, 0.03)),
        "held": ((86.87, 0.10), (86.88, 0.09), (0.01, 0.04), (87.30, 0.10), (87.27, 0.13), (-0.03, 0.04)),
    },
    INVERTED: {
        "together": ((92.51, 0.59), (92.62, 0.62), (0.11, 0.06), (94.81, 0.37), (94.86, 0.37), (0.04, 0.03)),
        "three": ((92.38, 0.54), (92.46, 0.53), (0.09, 0.02), (94.62, 0.53), (94.70, 0.51), (0.08, 0.06)),
        "held": ((92.15, 0.63), (92.27, 0.63), (0.12, 0.06), (94.67, 0.51), (94.63, 0.53), (-0.04, 0.04)),
    },
}


# ---- what follows from them ---------------------------------------------------------
def stat(xs):
    """Five figures' mean and standard error."""
    n = len(xs)
    m = sum(xs) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def acc(w, kind, row):
    """How often a kind of network is right in a row, percent."""
    return stat(RUN[w][kind][row])


def lost(w, kind, row):
    """What the tile costs a network there: its accuracy on its host less in the row."""
    return stat([h - x for h, x in zip(HOSTS[w][kind], RUN[w][kind][row])])


def adds(w, kind, row):
    """What a row adds to version 2 as budgeted, network by network."""
    return stat([b - x for b, x in zip(RUN[w][kind]["budget"], RUN[w][kind][row])])


def gains(w, kind, where):
    """Each network's accuracy with the shared row at 1%, less at 2%."""
    return [one - two for one, two in zip(RUN[w][kind][(where, IS)], RUN[w][kind][(where, WAS)])]


def buys(w, kind, where):
    """What the 1% buys a kind of network: alone, among the three, or held."""
    return stat(gains(w, kind, where))


def needs_less(w, where):
    """What the 1% buys a reference network, less what it buys the one trained
    before of the same seed."""
    return stat([r - b for r, b in zip(gains(w, REFERENCE, where), gains(w, BEFORE, where))])


def adds_less(w, row):
    """What a row adds to a reference network, less to the one trained before."""
    return stat([(rb - r) - (bb - b) for rb, r, bb, b in zip(
        RUN[w][REFERENCE]["budget"], RUN[w][REFERENCE][row], RUN[w][BEFORE]["budget"], RUN[w][BEFORE][row])])


def errors(x):
    """A difference in its own standard errors."""
    return x[0] / x[1] if x[1] else 0.0


def clear(x):
    """Whether five networks put a difference outside chance at one in twenty."""
    return abs(errors(x)) > T95


def at_most(x):
    """What five networks put a difference under, at one in twenty."""
    return x[0] + T95 * x[1]


def holds(w, kind, size):
    """Whether the source's three rows add under the budget's tenth."""
    return adds(w, kind, ("three", size))[0] < BUDGET


def together_for_a_tenth(w, kind, slack=0.035):
    """The largest size run at which the lines together add about a tenth:
    pta_version2.py's rule, no more than a tenth and a third again."""
    ok = [x for (name, x) in (k for k, _ in ROWS if isinstance(k, tuple)) if name == "together" and adds(w, kind, ("together", x))[0] <= BUDGET + slack]
    return max(ok) if ok else None


def size(x):
    """A size of the shared row, or that none run was small enough."""
    return "none run" if x is None else f"{x:.0%}"


def pm(x):
    return f"{x[0]:.2f} +-{x[1]:.2f}"


def dpm(x):
    return f"{x[0]:+.2f} +-{x[1]:.2f}"


def r2(x):
    return round(x[0], 2), round(x[1], 2)


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    print("The row a source's lines share, on the reference networks: does a trained network need the 1%?")
    print(f"The working tile, {laser.name(TILE)} on two buses, at version 2.  Five networks a kind, mean and standard error.")

    section("1. Points lost against a network's own accuracy on its host, and what a row adds to its budget")
    for w in WORKLOADS:
        print(f"  {w:<30}{'trained before':>16}{'and adds':>16}{'the reference':>16}{'and adds':>16}")
        for key, name in ROWS:
            print(f"    {name:<28}" + "".join(f"{pm(lost(w, k, key)):>16}{dpm(adds(w, k, key)):>16}" for k in KINDS))

    section(f"2. What the 1% buys: right, percent, with the shared row at {WAS:.0%} and at {IS:.0%}, and the second less the first")
    for w in WORKLOADS:
        print(f"  {w}")
        for k in KINDS:
            print(f"    {'the reference' if k == REFERENCE else 'trained before':<28}{'at 2%':>10}{'at 1%':>10}{'the 1% buys':>16}{'in its errors':>15}{'at the most':>13}")
            for where, name in WHERE:
                b = buys(w, k, where)
                print(f"      {name:<26}{acc(w, k, (where, WAS))[0]:>10.2f}{acc(w, k, (where, IS))[0]:>10.2f}{dpm(b):>16}{errors(b):>+15.1f}{at_most(b):>+13.2f}")

    section("3. A reference network less the one trained before, seed by seed")
    print(f"  {'what the 1% buys':<30}" + "".join(f"{w:>20}" for w in WORKLOADS))
    for where, name in WHERE:
        print(f"    {name:<28}" + "".join(f"{dpm(needs_less(w, where)):>20}" for w in WORKLOADS))
    print(f"  {'what a row adds':<30}")
    for key, name in ROWS[1:]:
        print(f"    {name:<28}" + "".join(f"{dpm(adds_less(w, key)):>20}" for w in WORKLOADS))

    section("4. The row's two sizes")
    print(f"  {WAS:.0%} rms a shot is {noise.db_hz(WAS):.0f} dB/Hz over the shot rate, and {IS:.0%} is {noise.db_hz(IS):.0f}: {noise.db_hz(WAS) - noise.db_hz(IS):.0f} dB.")
    print("  The largest size run at which the lines together add about a tenth of a point:")
    print(f"  {'':<30}" + "".join(f"{w:>20}" for w in WORKLOADS))
    for k in KINDS:
        print(f"    {'the reference' if k == REFERENCE else 'trained before':<28}" + "".join(f"{size(together_for_a_tenth(w, k)):>20}" for w in WORKLOADS))

    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        findings()
    print(said.getvalue(), end="")
    checks(" ".join(said.getvalue().split()))


def findings():
    m, f, i = WORKLOADS
    R, B = REFERENCE, BEFORE
    print()
    print("What this says, six readings.")
    print()
    print("  1. HELD AS THE CHIP IS HELD, THE 1% BUYS THE REFERENCE NETWORKS NOTHING.  At the end of six")
    print(f"     minutes, with a line and the level at 5%, they are right {acc(m, R, ('held', WAS))[0]:.2f}, {acc(f, R, ('held', WAS))[0]:.2f} and {acc(i, R, ('held', WAS))[0]:.2f}% of the")
    print(f"     time with the shared row at 2%, and {acc(m, R, ('held', IS))[0]:.2f}, {acc(f, R, ('held', IS))[0]:.2f} and {acc(i, R, ('held', IS))[0]:.2f}% at 1%.  Network by network the 1%")
    print(f"     is worth {dpm(buys(m, R, 'held'))}, {dpm(buys(f, R, 'held'))} and {dpm(buys(i, R, 'held'))} of a point: less than nothing")
    print(f"     on all three, and none of them clear.  Five networks put it under {at_most(buys(m, R, 'held')):+.2f}, {at_most(buys(f, R, 'held')):+.2f}")
    print(f"     and {at_most(buys(i, R, 'held')):+.2f} at one in twenty, which is under the tenth a row is sized to.")
    print()
    print("  2. AS BUDGETED IT MAY BUY THEM WHAT IT BOUGHT THE OLD ONES ON THE INVERTED SET, AND THAT IS NOT")
    print(f"     SHOWN.  Among the three rows with no drift the 1% is worth {dpm(buys(i, R, 'three'))} there, where it was")
    print(f"     worth {dpm(buys(i, B, 'three'))} to the networks trained before.  The same size, at {buys(i, R, 'three')[1] / buys(i, B, 'three')[1]:.1f} times the")
    print(f"     error.  On MNIST and Fashion-MNIST it is {dpm(buys(m, R, 'three'))} and {dpm(buys(f, R, 'three'))}.")
    print()
    print("  3. IT DID BUY THE OLD NETWORKS WHAT B16 SAID.  On the inverted set, network by network:")
    print(f"     {dpm(buys(i, B, 'together'))} alone, {dpm(buys(i, B, 'three'))} among the three and {dpm(buys(i, B, 'held'))} held.  The middle one")
    print(f"     is {errors(buys(i, B, 'three')):.1f} of its errors, and is the 0.09 B16's record has.  On MNIST held it is {dpm(buys(m, B, 'held'))},")
    print("     which is clear and is four images in ten thousand.")
    print()
    print("  4. THE ROWS THEMSELVES COST A TRAINED NETWORK LESS ON THE INVERTED SET.  The lines together at")
    print(f"     2% add {dpm(adds(i, R, ('together', 0.02)))} to its budget where they added {dpm(adds(i, B, ('together', 0.02)))}, and at 5%")
    print(f"     {dpm(adds(i, R, ('together', 0.05)))} for {dpm(adds(i, B, ('together', 0.05)))}.  A line on its own at 5% adds {dpm(adds(i, R, ('line', 0.05)))} for {dpm(adds(i, B, ('line', 0.05)))},")
    print(f"     and that one is clear: {dpm(adds_less(i, ('line', 0.05)))} seed by seed.  So for the shared row to add about a")
    print(f"     tenth there it can be {size(together_for_a_tenth(i, R))} for the reference networks, where it had to be {size(together_for_a_tenth(i, B))}.")
    print()
    print("  5. AND THE THREE TOGETHER ARE STILL NOT SHOWN INSIDE A TENTH THERE, AT EITHER SIZE.  They add")
    print(f"     {dpm(adds(i, R, ('three', WAS)))} at 2% and {dpm(adds(i, R, ('three', IS)))} at 1%, each with an error its own size.  It is")
    level = [b - x for b, x in zip(RUN[i][R]["budget"], RUN[i][R][("level", 0.05)])]
    print(f"     one network: the lines' level at 5% costs seed {level.index(max(level)) + 1} {max(level):.2f} of a point and the other four")
    print(f"     {min(level):+.2f} to {sorted(level)[-2]:+.2f}.  On MNIST and Fashion-MNIST the three add {adds(m, R, ('three', WAS))[0]:.2f} and {adds(f, R, ('three', WAS))[0]:.2f} at 2%.")
    print()
    print("  6. WHETHER A TRAINED NETWORK NEEDS IT LESS THAN THE OLD ONES DID IS SUGGESTED AND NOT SHOWN.")
    print(f"     Held, the 1% is worth {dpm(needs_less(m, 'held'))}, {dpm(needs_less(f, 'held'))} and {dpm(needs_less(i, 'held'))} less to a reference")
    print("     network than to the one trained before of the same seed.  MNIST's is clear and is")
    print("     six images.  The other two are not.  What is shown is reading 1: held, the reference")
    print("     networks do not need the 1%.  Whether B16 keeps it is the plan's to say.")


def checks(said):
    """Every claim above, as an assert.  `said` is the readings as printed, on one line."""
    m, f, i = WORKLOADS
    R, B = REFERENCE, BEFORE
    keys = [k for k, _ in ROWS]
    assert TILE == (128, 64) and (WAS, IS) == (0.02, 0.01) and BUDGET == 0.10 and len(keys) == 11
    assert set(RUN) == set(HOSTS) == set(PRINTED_LOST) == set(PRINTED_BUYS) == set(WORKLOADS)
    assert all(set(RUN[w]) == set(HOSTS[w]) == set(KINDS) for w in WORKLOADS)
    assert all(set(RUN[w][k]) == set(keys) and len(HOSTS[w][k]) == SEEDS for w in WORKLOADS for k in KINDS)
    assert all(len(RUN[w][k][row]) == SEEDS for w in WORKLOADS for k in KINDS for row in keys)

    # 0. The figures are grx930's, and agree with what the plan already holds.
    #    Both tables its harness printed, every cell, from the networks' own figures.
    def same(a, b):
        return abs(a[0] - b[0]) < 0.0051 and abs(a[1] - b[1]) < 0.0051

    cells = 0
    for w in WORKLOADS:
        assert set(PRINTED_LOST[w]) == set(keys) and set(PRINTED_BUYS[w]) == {x for x, _ in WHERE}
        for row in keys:
            got = (lost(w, B, row), adds(w, B, row), lost(w, R, row), adds(w, R, row))
            assert len(PRINTED_LOST[w][row]) == 4 and all(same(a, b) for a, b in zip(got, PRINTED_LOST[w][row])), (w, row)
            cells += 4
        for where, _ in WHERE:
            got = [x for k in KINDS for x in (acc(w, k, (where, WAS)), acc(w, k, (where, IS)), buys(w, k, where))]
            assert len(PRINTED_BUYS[w][where]) == 6 and all(same(a, b) for a, b in zip(got, PRINTED_BUYS[w][where])), (w, where)
            cells += 6
    assert cells == 3 * (11 * 4 + 3 * 6) == 186
    #    Counted and summed, as grx930's lines gave them.
    figures = [x for w in RUN.values() for k in w.values() for row in k.values() for x in row]
    assert len(figures) == 330 and abs(sum(figures) - 30596.03) < 1e-6
    hosts = [x for w in HOSTS.values() for k in w.values() for x in k]
    assert len(hosts) == 30 and abs(sum(hosts) - 2796.55) < 1e-6
    printed = [x for t in (PRINTED_LOST, PRINTED_BUYS) for w in t.values() for row in w.values() for cell in row for x in cell]
    assert len(printed) == 372 and abs(sum(printed) - 3400.74) < 1e-6
    #    The networks are pta_trained.py's, seed for seed: on the host, at version
    #    2 as budgeted, and held at the 1%, which that model has for both kinds.
    assert trained.REFERENCE == 0.1 and trained.TILE == TILE and trained.WORKLOADS == WORKLOADS
    for w in WORKLOADS:
        for k, t in ((B, trained.BEFORE), (R, trained.REFERENCE)):
            nets = trained.NETS[w][t]
            assert tuple(n[trained.HOST] for n in nets) == HOSTS[w][k], (w, k)
            assert tuple(n[trained.V2] for n in nets) == RUN[w][k]["budget"], (w, k)
            assert tuple(n[trained.V2_HELD] for n in nets) == RUN[w][k][("held", IS)], (w, k)
    #    And the networks trained before are pta_version2.py's at version 2, in
    #    every row that model has: what they lose, and what the row adds.
    there = {"budget": "budget", "six": "six", ("three", WAS): ("all", "rows"), ("three", IS): ("all", "v2 rows"),
             ("held", IS): version2.HELD, ("line", 0.05): ("line", 0.05), ("level", 0.05): ("level", 0.05),
             ("together", 0.01): ("together", 0.01), ("together", 0.02): ("together", 0.02), ("together", 0.05): ("together", 0.05)}
    assert set(keys) - set(there) == {("held", WAS)}
    for w in WORKLOADS:
        for row, key in there.items():
            assert (r2(lost(w, B, row)), r2(adds(w, B, row))) == version2.RUN[(w, 2)][key], (w, row)
    assert version2.ROWS3 == (WAS, 0.05, 0.05) and version2.ROWS_V2 == (IS, 0.05, 0.05) and version2.INTERVAL_S == 360
    assert abs(noise.db_hz(WAS) + 124) < 0.5 and abs(noise.db_hz(IS) + 130) < 0.5
    assert abs(noise.db_hz(WAS) - noise.db_hz(IS) - 6.02) < 0.01

    # 1. Reading 1.  Held, the reference networks.
    assert [round(acc(w, R, ("held", WAS))[0], 2) for w in WORKLOADS] == [97.66, 87.30, 94.67]
    assert [round(acc(w, R, ("held", IS))[0], 2) for w in WORKLOADS] == [97.64, 87.27, 94.63]
    assert [r2(buys(w, R, "held")) for w in WORKLOADS] == [(-0.02, 0.02), (-0.03, 0.04), (-0.04, 0.04)]
    assert all(buys(w, R, "held")[0] < 0 and not clear(buys(w, R, "held")) for w in WORKLOADS)
    assert [round(at_most(buys(w, R, "held")), 2) for w in WORKLOADS] == [0.02, 0.08, 0.08]
    assert all(at_most(buys(w, R, "held")) < BUDGET for w in WORKLOADS)
    #    What they lose so held: 0.09, 0.59 and 0.72 at 2%, 0.11, 0.62 and 0.76 at 1%.
    assert [round(lost(w, R, ("held", WAS))[0], 2) for w in WORKLOADS] == [0.09, 0.59, 0.72]
    assert [round(lost(w, R, ("held", IS))[0], 2) for w in WORKLOADS] == [0.11, 0.62, 0.76]
    #    The last is pta_trained.py's reference row, to the hundredth.
    assert [r2(lost(w, R, ("held", IS))) for w in WORKLOADS] == [r2(trained.lost(w, trained.REFERENCE, trained.V2_HELD)) for w in WORKLOADS]

    # 2. Reading 2.  As budgeted.
    assert [r2(buys(w, R, "three")) for w in WORKLOADS] == [(-0.01, 0.01), (0.03, 0.03), (0.08, 0.06)]
    assert r2(buys(i, B, "three")) == (0.09, 0.02) and not any(clear(buys(w, R, "three")) for w in WORKLOADS)
    assert 3 < buys(i, R, "three")[1] / buys(i, B, "three")[1] < 4.5
    #    There five networks do not put it under a tenth: 0.26 at the most on the inverted set.
    assert [round(at_most(buys(w, R, "three")), 2) for w in WORKLOADS] == [0.03, 0.11, 0.26]
    #    Alone, the reference networks: 0.03, -0.08 and 0.04.  Fashion-MNIST's is
    #    clear and is the other way: eight images more at 2% than at 1%.
    assert [r2(buys(w, R, "together")) for w in WORKLOADS] == [(0.03, 0.01), (-0.08, 0.02), (0.04, 0.03)]
    assert [w for w in WORKLOADS for where, _ in WHERE if clear(buys(w, R, where))] == [f] and clear(buys(f, R, "together"))
    #    In no cell of the nine does the 1% buy a reference network a tenth of a point.
    assert max(buys(w, R, where)[0] for w in WORKLOADS for where, _ in WHERE) < 0.08

    # 3. Reading 3.  The networks trained before, which B16 was chosen on.
    assert [r2(buys(i, B, where)) for where, _ in WHERE] == [(0.11, 0.06), (0.09, 0.02), (0.12, 0.06)]
    assert clear(buys(i, B, "three")) and 4.8 < errors(buys(i, B, "three")) < 5.0
    assert not clear(buys(i, B, "together")) and not clear(buys(i, B, "held"))
    assert r2(buys(m, B, "held")) == (0.04, 0.0) and clear(buys(m, B, "held"))
    assert [round(x, 2) for x in gains(m, B, "held")] == [0.04, 0.03, 0.04, 0.06, 0.04]
    #    That 0.09 is what pta_version2.py has as two rows' difference, and B16's record quotes.
    v = version2.adds
    assert round(v(i, 2, ("all", "rows"))[0] - v(i, 2, ("all", "v2 rows"))[0], 2) == round(buys(i, B, "three")[0], 2) == 0.09
    #    On the other two sets it buys them 0.02 to 0.05 among the three, and under 0.02 alone.
    assert [r2(buys(w, B, "three")) for w in (m, f)] == [(0.02, 0.02), (0.05, 0.02)]
    assert all(abs(buys(w, B, "together")[0]) < 0.02 for w in (m, f))

    # 4. Reading 4.  What the rows add on the inverted set.
    assert [r2(adds(i, R, ("together", x))) for x in (0.01, 0.02, 0.05)] == [(0.0, 0.03), (0.05, 0.05), (0.46, 0.10)]
    assert [r2(adds(i, B, ("together", x))) for x in (0.01, 0.02, 0.05)] == [(0.13, 0.08), (0.24, 0.05), (0.90, 0.13)]
    assert (r2(adds(i, R, ("line", 0.05))), r2(adds(i, B, ("line", 0.05)))) == ((0.0, 0.03), (0.25, 0.07))
    assert r2(adds_less(i, ("line", 0.05))) == (-0.24, 0.05) and clear(adds_less(i, ("line", 0.05)))
    #    Of the ten rows there it is the one that is clear; the lines together are two errors.
    assert [k for k in keys[1:] if clear(adds_less(i, k))] == [("line", 0.05)]
    assert all(-2.5 < errors(adds_less(i, ("together", x))) < -2.0 for x in (0.02, 0.05))
    assert 0.45 < adds(i, R, ("together", 0.05))[0] / adds(i, B, ("together", 0.05))[0] < 0.55
    assert [together_for_a_tenth(w, R) for w in WORKLOADS] == [0.05, 0.05, 0.02]
    assert [together_for_a_tenth(w, B) for w in WORKLOADS] == [0.05, 0.05, 0.01]
    assert [together_for_a_tenth(w, B) for w in WORKLOADS] == [version2.together_for_a_tenth(w, 2) for w in WORKLOADS]

    # 5. Reading 5.  The three together.
    assert [r2(adds(i, R, ("three", x))) for x in (WAS, IS)] == [(0.24, 0.24), (0.16, 0.19)]
    assert [r2(adds(i, B, ("three", x))) for x in (WAS, IS)] == [(0.37, 0.07), (0.28, 0.06)]
    assert not holds(i, R, WAS) and not holds(i, R, IS) and not holds(i, B, WAS) and not holds(i, B, IS)
    assert all(holds(w, R, x) and holds(w, B, x) for w in (m, f) for x in (WAS, IS))
    assert [round(adds(w, R, ("three", WAS))[0], 2) for w in (m, f)] == [0.06, 0.06]
    assert not clear(adds(i, R, ("three", WAS))) and not clear(adds(i, R, ("three", IS)))
    level = [b - x for b, x in zip(RUN[i][R]["budget"], RUN[i][R][("level", 0.05)])]
    assert [round(x, 2) for x in level] == [0.17, -0.06, -0.13, 0.86, -0.26] and r2(adds(i, R, ("level", 0.05))) == (0.12, 0.20)

    # 6. Reading 6.  A trained network, less the old one.
    assert [r2(needs_less(w, "held")) for w in WORKLOADS] == [(-0.06, 0.02), (-0.05, 0.06), (-0.16, 0.08)]
    assert [w for w in WORKLOADS if clear(needs_less(w, "held"))] == [m]
    assert all(needs_less(w, "held")[0] < 0 for w in WORKLOADS)
    assert not any(clear(needs_less(w, where)) for w in WORKLOADS for where in ("together", "three"))

    # 7. And the readings say those figures, each in its place.
    for words in (
        "they are right 97.66, 87.30 and 94.67% of the time with the shared row at 2%, and 97.64, 87.27 and 94.63% at 1%",
        "is worth -0.02 +-0.02, -0.03 +-0.04 and -0.04 +-0.04 of a point: less than nothing",
        "put it under +0.02, +0.08 and +0.08 at one in twenty",
        "the 1% is worth +0.08 +-0.06 there, where it was worth +0.09 +-0.02 to the networks trained before",
        "The same size, at 3.7 times the error.",
        "On MNIST and Fashion-MNIST it is -0.01 +-0.01 and +0.03 +-0.03.",
        "network by network: +0.11 +-0.06 alone, +0.09 +-0.02 among the three and +0.12 +-0.06 held",
        "The middle one is 4.9 of its errors",
        "On MNIST held it is +0.04 +-0.00,",
        "2% add +0.05 +-0.05 to its budget where they added +0.24 +-0.05, and at 5% +0.46 +-0.10 for +0.90 +-0.13",
        "A line on its own at 5% adds +0.00 +-0.03 for +0.25 +-0.07",
        "that one is clear: -0.24 +-0.05 seed by seed",
        "it can be 2% for the reference networks, where it had to be 1%",
        "They add +0.24 +-0.24 at 2% and +0.16 +-0.19 at 1%",
        "costs seed 4 0.86 of a point and the other four -0.26 to +0.17",
        "the three add 0.06 and 0.06 at 2%",
        "the 1% is worth -0.06 +-0.02, -0.05 +-0.06 and -0.16 +-0.08 less to a reference network",
    ):
        assert said.count(words) == 1, words

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
