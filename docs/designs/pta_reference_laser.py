"""
The laser the reference networks need: does a network trained with noise buy any of it back?

The laser is most of what a MAC costs.  B14's version 2 takes 8, 16 and 16 times
B5's laser on the three data sets, with the hidden layer's rescale a bit down:
0.7 to 7 W on MNIST and 1.4 to 14 W on the other two.  Those are the networks
trained before.  B17's reference networks were trained with Gaussian noise on
their sums, and a receiver's noise is Gaussian noise on a sum, so they were the
likeliest thing in this plan to need less laser.  grx930's harness has run them.

MEASURED IN A MODEL, by grx930 (c930/doc/pta_error_model_design_note.md section
5, "The laser a trained network needs", 2026-10-07; `sim/pta_mnist.sh DIR WORK
reflaser`).  The working tile, 128 x 64, at version 2's rows.  Two kinds of
network a data set, five of each, seeds 1 to 5: the ones trained before, and
B17's reference.  Eleven rows a network:

  as budgeted       version 2, its receiver's noise the budget's quarter of an LSB
  a bit down        under a laser of 2, 4, 8, 16 and 32 times B5's, the hidden
                    layer's rescale one bit under the clip rule.  This is how
                    every laser in B14's table was sized
  the rule's        the same five lasers, at the rule's rescale

Under a laser the receiver's row is the laser's: its noise is rows / 512 of one
line's light over the multiple, the same in every layer.  The light's row stays
the budget's 30 photons while the laser multiplies, as in pta_tighten.py.

DERIVED here:

  a row's mean and    which have to be the two tables grx930 printed, cell for cell
  its error
  over its budget     what a network loses under a laser, less what it loses as
                      budgeted, network by network
  the laser it        pta_tighten.py's rule: the least multiple at which a kind is
  needs               within a tenth of a point of its own budget, every larger
                      one being so too
  what that is        B5's method times the multiple, and a MAC at it
  a trained network   the same row for a reference network, less for the one
  less the old        trained before, seed by seed

HOW SURE.  Five networks, 2.8 errors for one in twenty, as in pta_trained.py.  And
the rule is read on means to the hundredth, as pta_tighten.py reads it.  It does
not ask how sure a mean is, and two of its twelve answers here turn on a hundredth
of a point.  One of the two is in B14's table already.  Reading 2 says which.

WHAT THIS IS NOT: a laser.  The receiver's noise is Gaussian and the light's row
does not move with it, where a real laser moves both.  The multiples are octaves:
"8 times" is somewhere above 4 and no more than 8.  Nothing is held: no drift and
no source's noise under any laser.  Version 2 and the working tile only.  And it
is pta_workload.py's three data sets.

Standard library only.  Run:  python3 docs/designs/pta_reference_laser.py
"""
import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_laser as laser
import pta_power as power
import pta_shared_row as shared
import pta_tighten as tighten
import pta_trained as trained
import pta_working_point as wp
import pta_workload as workload

MNIST, FASHION, INVERTED = workload.MNIST, workload.FASHION, workload.INVERTED
WORKLOADS = workload.WORKLOADS
TILE, FS = laser.SMALLER, laser.FS             # 128 x 64 (B10), 1 GS/s (B11)
BEFORE, REFERENCE = "before", "reference"      # trained as the plan's networks were; B17's
KINDS = (BEFORE, REFERENCE)
SEEDS = 5
LASERS = (2, 4, 8, 16, 32)                     # multiples of B5's laser
DOWN, RULE = "down", "rule"                    # the hidden rescale a bit under the clip rule; at it
SHIFTS = (DOWN, RULE)
WITHIN = laser.WITHIN                          # a tenth of a point: what a row of the budget costs
BITS2 = tighten.V2["adc_bits"]                 # 8: version 2's ADC
T95 = trained.T95
ROWS = ((("budget", "as budgeted"),)
        + tuple(((DOWN, m), f"a bit down, {m} times") for m in LASERS)
        + tuple(((RULE, m), f"the rule's rescale, {m} times") for m in LASERS))

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
            ("down", 2): (96.80, 97.09, 96.28, 96.46, 96.55),
            ("down", 4): (97.35, 97.51, 97.04, 97.12, 97.01),
            ("down", 8): (97.43, 97.65, 97.06, 97.32, 97.24),
            ("down", 16): (97.45, 97.67, 97.10, 97.38, 97.20),
            ("down", 32): (97.40, 97.60, 97.10, 97.39, 97.18),
            ("rule", 2): (95.22, 95.81, 94.30, 94.69, 94.93),
            ("rule", 4): (97.00, 97.24, 96.62, 96.65, 96.73),
            ("rule", 8): (97.25, 97.59, 97.03, 97.17, 97.12),
            ("rule", 16): (97.35, 97.64, 97.04, 97.36, 97.29),
            ("rule", 32): (97.37, 97.71, 97.11, 97.39, 97.20),
        },
        REFERENCE: {
            "budget": (97.61, 97.66, 97.65, 97.84, 97.65),
            ("down", 2): (96.99, 96.76, 97.15, 97.45, 96.81),
            ("down", 4): (97.48, 97.45, 97.46, 97.73, 97.37),
            ("down", 8): (97.59, 97.67, 97.54, 97.83, 97.61),
            ("down", 16): (97.59, 97.61, 97.55, 97.85, 97.67),
            ("down", 32): (97.61, 97.66, 97.54, 97.81, 97.71),
            ("rule", 2): (94.17, 93.49, 96.72, 96.76, 93.66),
            ("rule", 4): (96.93, 96.95, 97.27, 97.57, 96.90),
            ("rule", 8): (97.47, 97.50, 97.49, 97.78, 97.48),
            ("rule", 16): (97.49, 97.63, 97.56, 97.83, 97.53),
            ("rule", 32): (97.53, 97.61, 97.53, 97.83, 97.59),
        },
    },
    FASHION: {
        BEFORE: {
            "budget": (86.45, 86.83, 86.94, 87.65, 86.81),
            ("down", 2): (84.09, 83.53, 84.56, 85.07, 79.80),
            ("down", 4): (86.06, 85.72, 86.13, 86.96, 84.68),
            ("down", 8): (86.71, 86.60, 86.73, 87.48, 86.17),
            ("down", 16): (86.76, 86.81, 86.89, 87.52, 86.64),
            ("down", 32): (86.77, 87.00, 87.00, 87.63, 86.88),
            ("rule", 2): (78.73, 78.37, 80.50, 80.62, 68.84),
            ("rule", 4): (84.34, 83.96, 84.56, 85.17, 80.05),
            ("rule", 8): (86.23, 85.96, 86.10, 87.17, 84.69),
            ("rule", 16): (86.47, 86.77, 86.74, 87.44, 86.25),
            ("rule", 32): (86.80, 86.82, 86.89, 87.48, 86.54),
        },
        REFERENCE: {
            "budget": (87.02, 87.76, 87.63, 87.42, 87.23),
            ("down", 2): (82.95, 83.48, 82.83, 84.55, 82.96),
            ("down", 4): (85.83, 86.72, 86.09, 86.71, 86.19),
            ("down", 8): (87.10, 87.66, 87.34, 87.35, 87.18),
            ("down", 16): (87.12, 87.80, 87.48, 87.48, 87.68),
            ("down", 32): (87.16, 88.01, 87.59, 87.53, 87.64),
            ("rule", 2): (75.10, 75.72, 74.75, 77.40, 74.37),
            ("rule", 4): (82.83, 83.76, 83.01, 84.79, 83.09),
            ("rule", 8): (85.83, 86.68, 86.34, 86.80, 86.15),
            ("rule", 16): (86.69, 87.44, 87.22, 87.29, 86.98),
            ("rule", 32): (87.11, 87.64, 87.42, 87.31, 87.16),
        },
    },
    INVERTED: {
        BEFORE: {
            "budget": (93.65, 93.26, 90.63, 92.71, 93.47),
            ("down", 2): (87.53, 90.84, 87.43, 86.08, 88.44),
            ("down", 4): (92.24, 92.56, 89.55, 91.13, 92.53),
            ("down", 8): (93.31, 93.02, 89.98, 92.56, 93.37),
            ("down", 16): (93.74, 93.27, 90.31, 92.83, 93.47),
            ("down", 32): (93.56, 93.27, 90.40, 92.82, 93.61),
            ("rule", 2): (73.22, 84.30, 80.68, 68.56, 73.05),
            ("rule", 4): (87.61, 91.06, 87.90, 85.94, 88.43),
            ("rule", 8): (92.22, 92.61, 89.97, 91.15, 92.52),
            ("rule", 16): (93.34, 93.12, 90.33, 92.43, 93.14),
            ("rule", 32): (93.65, 93.14, 90.46, 92.62, 93.51),
        },
        REFERENCE: {
            "budget": (95.67, 95.26, 94.38, 93.72, 95.28),
            ("down", 2): (92.48, 91.45, 89.36, 88.78, 91.08),
            ("down", 4): (94.96, 94.45, 93.39, 92.30, 94.42),
            ("down", 8): (95.49, 95.06, 94.30, 93.33, 95.30),
            ("down", 16): (95.65, 95.27, 94.51, 93.56, 95.43),
            ("down", 32): (95.72, 95.33, 94.50, 93.68, 95.47),
            ("rule", 2): (81.57, 77.29, 74.35, 74.12, 76.08),
            ("rule", 4): (92.76, 91.40, 89.52, 88.92, 91.11),
            ("rule", 8): (95.02, 94.31, 93.54, 92.68, 94.48),
            ("rule", 16): (95.49, 95.02, 94.22, 93.27, 95.16),
            ("rule", 32): (95.60, 95.25, 94.34, 93.53, 95.32),
        },
    },
}
# The first table grx930's harness printed, mean and standard error: a row -> what the networks
# trained before lose, that less what they lose as budgeted, and the same two for the reference.
PRINTED_LOST = {
    MNIST: {
        "budget": ((0.15, 0.02), (0.00, 0.00), (0.07, 0.05), (0.00, 0.00)),
        ("down", 2): ((0.82, 0.06), (0.67, 0.06), (0.72, 0.14), (0.65, 0.10)),
        ("down", 4): ((0.25, 0.06), (0.10, 0.05), (0.25, 0.06), (0.18, 0.03)),
        ("down", 8): ((0.11, 0.03), (-0.04, 0.03), (0.10, 0.03), (0.03, 0.02)),
        ("down", 16): ((0.09, 0.04), (-0.06, 0.04), (0.10, 0.05), (0.03, 0.02)),
        ("down", 32): ((0.12, 0.04), (-0.03, 0.04), (0.09, 0.04), (0.02, 0.03)),
        ("rule", 2): ((2.46, 0.16), (2.31, 0.16), (2.79, 0.75), (2.72, 0.71)),
        ("rule", 4): ((0.60, 0.06), (0.45, 0.06), (0.63, 0.13), (0.56, 0.10)),
        ("rule", 8): ((0.22, 0.03), (0.07, 0.02), (0.21, 0.06), (0.14, 0.02)),
        ("rule", 16): ((0.12, 0.02), (-0.03, 0.03), (0.14, 0.05), (0.07, 0.02)),
        ("rule", 32): ((0.10, 0.03), (-0.05, 0.03), (0.13, 0.04), (0.06, 0.02)),
    },
    FASHION: {
        "budget": ((0.54, 0.22), (0.00, 0.00), (0.47, 0.13), (0.00, 0.00)),
        ("down", 2): ((4.07, 0.94), (3.53, 0.89), (4.53, 0.26), (4.06, 0.32)),
        ("down", 4): ((1.57, 0.35), (1.03, 0.30), (1.58, 0.06), (1.10, 0.13)),
        ("down", 8): ((0.74, 0.20), (0.20, 0.14), (0.56, 0.10), (0.09, 0.06)),
        ("down", 16): ((0.55, 0.17), (0.01, 0.08), (0.37, 0.08), (-0.10, 0.10)),
        ("down", 32): ((0.42, 0.18), (-0.12, 0.06), (0.30, 0.08), (-0.17, 0.08)),
        ("rule", 2): ((10.06, 2.20), (9.52, 2.14), (12.42, 0.48), (11.94, 0.52)),
        ("rule", 4): ((3.86, 0.91), (3.32, 0.87), (4.39, 0.28), (3.92, 0.34)),
        ("rule", 8): ((1.45, 0.37), (0.91, 0.33), (1.53, 0.10), (1.05, 0.11)),
        ("rule", 16): ((0.74, 0.22), (0.20, 0.10), (0.76, 0.10), (0.29, 0.05)),
        ("rule", 32): ((0.57, 0.18), (0.03, 0.11), (0.56, 0.12), (0.08, 0.05)),
    },
    INVERTED: {
        "budget": ((0.62, 0.07), (0.00, 0.00), (0.52, 0.05), (0.00, 0.00)),
        ("down", 2): ((5.30, 0.86), (4.68, 0.82), (4.75, 0.37), (4.23, 0.35)),
        ("down", 4): ((1.76, 0.22), (1.14, 0.16), (1.48, 0.16), (0.96, 0.12)),
        ("down", 8): ((0.91, 0.14), (0.30, 0.10), (0.69, 0.08), (0.17, 0.07)),
        ("down", 16): ((0.64, 0.11), (0.02, 0.08), (0.50, 0.07), (-0.02, 0.06)),
        ("down", 32): ((0.63, 0.12), (0.01, 0.07), (0.44, 0.05), (-0.08, 0.04)),
        ("rule", 2): ((17.40, 3.11), (16.78, 3.07), (18.70, 1.11), (18.18, 1.08)),
        ("rule", 4): ((5.17, 0.95), (4.56, 0.90), (4.64, 0.39), (4.12, 0.36)),
        ("rule", 8): ((1.67, 0.24), (1.05, 0.19), (1.38, 0.09), (0.86, 0.07)),
        ("rule", 16): ((0.89, 0.10), (0.27, 0.03), (0.75, 0.08), (0.23, 0.06)),
        ("rule", 32): ((0.68, 0.08), (0.07, 0.04), (0.58, 0.06), (0.05, 0.04)),
    },
}
# The second: how often the networks trained before are right in a row, how often the reference,
# and the second less the first, seed by seed.
PRINTED_RIGHT = {
    MNIST: {
        "budget": ((97.30, 0.11), (97.68, 0.04), (0.38, 0.11)),
        ("down", 2): ((96.64, 0.14), (97.03, 0.13), (0.40, 0.24)),
        ("down", 4): ((97.21, 0.10), (97.50, 0.06), (0.29, 0.12)),
        ("down", 8): ((97.34, 0.10), (97.65, 0.05), (0.31, 0.09)),
        ("down", 16): ((97.36, 0.10), (97.65, 0.05), (0.29, 0.11)),
        ("down", 32): ((97.33, 0.09), (97.67, 0.05), (0.33, 0.09)),
        ("rule", 2): ((94.99, 0.25), (94.96, 0.74), (-0.03, 0.95)),
        ("rule", 4): ((96.85, 0.12), (97.12, 0.13), (0.28, 0.22)),
        ("rule", 8): ((97.23, 0.10), (97.54, 0.06), (0.31, 0.12)),
        ("rule", 16): ((97.34, 0.10), (97.61, 0.06), (0.27, 0.10)),
        ("rule", 32): ((97.36, 0.10), (97.62, 0.06), (0.26, 0.10)),
    },
    FASHION: {
        "budget": ((86.94, 0.20), (87.41, 0.13), (0.48, 0.20)),
        ("down", 2): ((83.41, 0.94), (83.35, 0.32), (-0.06, 0.85)),
        ("down", 4): ((85.91, 0.37), (86.31, 0.18), (0.40, 0.36)),
        ("down", 8): ((86.74, 0.21), (87.33, 0.10), (0.59, 0.22)),
        ("down", 16): ((86.92, 0.15), (87.51, 0.12), (0.59, 0.20)),
        ("down", 32): ((87.06, 0.15), (87.59, 0.14), (0.53, 0.19)),
        ("rule", 2): ((77.41, 2.19), (75.47, 0.53), (-1.94, 1.94)),
        ("rule", 4): ((83.62, 0.91), (83.50, 0.36), (-0.12, 0.84)),
        ("rule", 8): ((86.03, 0.40), (86.36, 0.18), (0.33, 0.35)),
        ("rule", 16): ((86.73, 0.20), (87.12, 0.13), (0.39, 0.16)),
        ("rule", 32): ((86.91, 0.16), (87.33, 0.10), (0.42, 0.17)),
    },
    INVERTED: {
        "budget": ((92.74, 0.55), (94.86, 0.36), (2.12, 0.45)),
        ("down", 2): ((88.06, 0.79), (90.63, 0.68), (2.57, 0.70)),
        ("down", 4): ((91.60, 0.58), (93.90, 0.48), (2.30, 0.46)),
        ("down", 8): ((92.45, 0.63), (94.70, 0.40), (2.25, 0.58)),
        ("down", 16): ((92.72, 0.62), (94.88, 0.38), (2.16, 0.56)),
        ("down", 32): ((92.73, 0.60), (94.94, 0.38), (2.21, 0.53)),
        ("rule", 2): ((75.96, 2.85), (76.68, 1.35), (0.72, 3.13)),
        ("rule", 4): ((88.19, 0.83), (90.74, 0.69), (2.55, 0.80)),
        ("rule", 8): ((91.69, 0.50), (94.01, 0.41), (2.31, 0.38)),
        ("rule", 16): ((92.47, 0.56), (94.63, 0.40), (2.16, 0.49)),
        ("rule", 32): ((92.68, 0.58), (94.81, 0.38), (2.13, 0.48)),
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


def over(w, kind, row):
    """What a network loses in a row, less what it loses as budgeted, network by network."""
    return stat([b - x for b, x in zip(RUN[w][kind]["budget"], RUN[w][kind][row])])


def over_less(w, row):
    """What a row costs a reference network over its budget, less what it costs the
    one trained before of the same seed over its own."""
    return stat([(rb - r) - (bb - b) for rb, r, bb, b in zip(
        RUN[w][REFERENCE]["budget"], RUN[w][REFERENCE][row], RUN[w][BEFORE]["budget"], RUN[w][BEFORE][row])])


def ahead(w, row, than):
    """How much more often a reference network is right in a row than the one
    trained before of the same seed is in another."""
    return stat([r - b for r, b in zip(RUN[w][REFERENCE][row], RUN[w][BEFORE][than])])


def halving(w, kind, frm=16, to=8, shift=DOWN):
    """What a kind of network gives up for a smaller laser: how often right at the
    smaller, less at the larger, network by network."""
    return stat([a - b for a, b in zip(RUN[w][kind][(shift, to)], RUN[w][kind][(shift, frm)])])


def needs(w, kind, shift=DOWN, budget=WITHIN, exact=False):
    """The laser a kind of network needs, by pta_tighten.py's rule and read as it
    reads it, on means to the hundredth: the least multiple at which what is lost is
    within the budget of what is lost as budgeted, every larger one being so too.
    exact: the same on the means as they are."""
    def figure(row):
        return lost(w, kind, row)[0] if exact else round(lost(w, kind, row)[0], 2)
    base = figure("budget")
    ok = None
    for t in sorted(LASERS, reverse=True):
        if figure((shift, t)) - base >= budget - 1e-9:
            break
        ok = t
    return ok


def laser_w(times):
    """B5's method at the working tile, times a multiple: watts, from and to."""
    assert times is not None, "no laser that was run is enough"
    return laser.laser_w(TILE, times)


def fj_mac(times):
    """A MAC with every cell in use, at version 2's interface chip and that laser."""
    c, l = tighten.chip_w(BITS2), laser_w(times)
    return tuple((c[k] + l[k]) / power.mac_s(*TILE, FS) * 1e15 for k in (0, 1))


def errors(x):
    """A difference in its own standard errors."""
    return x[0] / x[1] if x[1] else 0.0


def clear(x):
    """Whether five networks put a difference outside chance at one in twenty."""
    return abs(errors(x)) > T95


def pm(x):
    return f"{x[0]:.2f} +-{x[1]:.2f}"


def dpm(x):
    return f"{x[0]:+.2f} +-{x[1]:.2f}"


def r2(x):
    return round(x[0], 2), round(x[1], 2)


def watts(x):
    return f"{x[0]:.1f}-{x[1]:.0f} W"


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def need_rows():
    """Section 3's rows: a label, and a cell for each data set and kind."""
    def cell(what):
        return [what(w, k) for w in WORKLOADS for k in KINDS]
    return (
        ("the laser it needs, a bit down", cell(lambda w, k: f"{needs(w, k)} times")),
        ("which is", cell(lambda w, k: watts(laser_w(needs(w, k))))),
        ("a MAC at it", cell(lambda w, k: f"{fj_mac(needs(w, k))[0]:.0f}-{fj_mac(needs(w, k))[1]:,.0f} fJ")),
        ("at the rule's rescale", cell(lambda w, k: f"{needs(w, k, RULE)} times")),
    )


def main():
    print("The laser the reference networks need: does a network trained with noise buy any of it back?")
    print(f"The working tile, {laser.name(TILE)}, at version 2's rows.  Five networks a kind, mean and standard error.")

    section("1. Points lost under a laser, and that less the same networks as budgeted")
    for w in WORKLOADS:
        print(f"  {w:<32}{'trained before':>16}{'over its budget':>18}{'the reference':>16}{'over its budget':>18}")
        for key, name in ROWS:
            print(f"    {name:<30}" + "".join(f"{pm(lost(w, k, key)):>16}{dpm(over(w, k, key)):>18}" for k in KINDS))

    section("2. A reference network less the one trained before, seed by seed: what a laser costs over the budget")
    print(f"  {'':<32}" + "".join(f"{w:>20}" for w in WORKLOADS))
    for key, name in ROWS[1:]:
        print(f"    {name:<30}" + "".join(f"{dpm(over_less(w, key)):>20}" for w in WORKLOADS))

    section("3. The laser each needs, by pta_tighten.py's rule")
    print(f"  {'':<32}" + "".join(f"{(w + ', ' + ('before' if k == BEFORE else 'reference')):>28}" for w in WORKLOADS for k in KINDS))
    for name, cells in need_rows():
        print(f"    {name:<30}" + "".join(f"{c:>28}" for c in cells))

    section("4. How often they are right, percent, and what half the laser costs")
    print(f"  {'':<32}" + "".join(f"{w:>20}" for w in WORKLOADS))
    for k in KINDS:
        who = "the reference" if k == REFERENCE else "trained before"
        for m in (16, 8):
            print(f"    {who + ', ' + str(m) + ' times':<30}" + "".join(f"{acc(w, k, (DOWN, m))[0]:>20.2f}" for w in WORKLOADS))
        print(f"    {'  8 times less 16':<30}" + "".join(f"{dpm(halving(w, k)):>20}" for w in WORKLOADS))
    print(f"    {'the reference at 8, less':<30}")
    print(f"    {'  the old ones at 16':<30}" + "".join(f"{dpm(ahead(w, (DOWN, 8), (DOWN, 16))):>20}" for w in WORKLOADS))

    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        findings()
    print(said.getvalue(), end="")
    checks(" ".join(said.getvalue().split()))


def findings():
    m, f, i = WORKLOADS
    R, B = REFERENCE, BEFORE
    print()
    print("What this says, five readings.")
    print()
    print("  1. UNDER A LASER A REFERENCE NETWORK LOSES WHAT AN OLD ONE DOES.  Over its own budget, with")
    print(f"     the rescale a bit down: at 4 times {over(m, R, (DOWN, 4))[0]:.2f}, {over(f, R, (DOWN, 4))[0]:.2f} and {over(i, R, (DOWN, 4))[0]:.2f} of a point where the old ones are over")
    print(f"     theirs by {over(m, B, (DOWN, 4))[0]:.2f}, {over(f, B, (DOWN, 4))[0]:.2f} and {over(i, B, (DOWN, 4))[0]:.2f}, and at 8 times {over(m, R, (DOWN, 8))[0]:.2f}, {over(f, R, (DOWN, 8))[0]:.2f} and {over(i, R, (DOWN, 8))[0]:.2f} for {over(m, B, (DOWN, 8))[0]:.2f}, {over(f, B, (DOWN, 8))[0]:.2f} and {over(i, B, (DOWN, 8))[0]:.2f}.")
    harder = [over_less(w, (s, t)) for w in (f, i) for s in SHIFTS for t in LASERS]
    print(f"     Seed by seed, on the two harder sets, not one of the {len(harder)} differences is clear, and the")
    print(f"     largest in its errors is {max(abs(errors(x)) for x in harder):.1f}.  Noise of a tenth of a sum's rms in the training does")
    print("     not buy laser.")
    print()
    print(f"  2. SO BY THE PLAN'S RULE THEY NEED {needs(m, R)}, {needs(f, R)} AND {needs(i, R)} TIMES B5'S LASER, WHERE THE OLD ONES NEED {needs(m, B)}, {needs(f, B)}")
    print(f"     AND {needs(i, B)}.  One of the three is halved, and by a hundredth of a point: on Fashion-MNIST at")
    print(f"     8 times they are {dpm(over(f, R, (DOWN, 8)))} over their budget, against the rule's tenth.  The old")
    print(f"     networks' {needs(m, B)} on MNIST turns on one too: at 4 times they are {over(m, B, (DOWN, 4))[0]:.3f} over, which the")
    print(f"     rule reads to the hundredth as a tenth, and read exactly they would need {needs(m, B, exact=True)}.  The laser a")
    print(f"     board has to place is its brightest workload's, and that is {needs(i, R)} times for both kinds:")
    print(f"     {watts(laser_w(needs(i, R)))}, and a MAC of {fj_mac(needs(i, R))[0]:.0f} to {fj_mac(needs(i, R))[1]:,.0f} fJ.")
    print()
    print("  3. HALF THAT LASER COSTS THEM A FIFTH OF A POINT ON THE TWO HARDER SETS.  Right, at 8 times")
    print(f"     less at 16, network by network: {dpm(halving(m, R))}, {dpm(halving(f, R))} and {dpm(halving(i, R))}.  For the")
    print(f"     old networks it is {dpm(halving(m, B))}, {dpm(halving(f, B))} and {dpm(halving(i, B))}.  At 8 times the laser is")
    print(f"     {watts(laser_w(8))} and a MAC {fj_mac(8)[0]:.0f} to {fj_mac(8)[1]:,.0f} fJ.")
    print()
    print("  4. AND AT HALF THE LASER THEY ARE STILL AHEAD OF THE OLD NETWORKS AT ALL OF IT.  The reference")
    print(f"     at 8 times is right {acc(m, R, (DOWN, 8))[0]:.2f}, {acc(f, R, (DOWN, 8))[0]:.2f} and {acc(i, R, (DOWN, 8))[0]:.2f}% of the time, and the old ones at 16 times")
    print(f"     {acc(m, B, (DOWN, 16))[0]:.2f}, {acc(f, B, (DOWN, 16))[0]:.2f} and {acc(i, B, (DOWN, 16))[0]:.2f}%: {dpm(ahead(m, (DOWN, 8), (DOWN, 16)))}, {dpm(ahead(f, (DOWN, 8), (DOWN, 16)))} and {dpm(ahead(i, (DOWN, 8), (DOWN, 16)))} seed by")
    print("     seed.  That is what B17 bought, and none of it is the laser's doing.  It is there")
    print("     to spend on a smaller laser if one has to be: reading 3 is the price.")
    print()
    print("  5. THE RESCALE A BIT DOWN STILL HALVES THE LASER, AT VERSION 2 AND FOR BOTH KINDS.  At the")
    print(f"     rule's rescale the old networks need {needs(m, B, RULE)}, {needs(f, B, RULE)} and {needs(i, B, RULE)} times and the reference {needs(m, R, RULE)}, {needs(f, R, RULE)} and")
    print(f"     {needs(i, R, RULE)}.  A bit down it is {needs(m, B)}, {needs(f, B)} and {needs(i, B)}, and {needs(m, R)}, {needs(f, R)} and {needs(i, R)}.  Only the old networks on MNIST")
    print("     get nothing from it.")


def checks(said):
    """Every claim above, as an assert.  `said` is the readings as printed, on one line."""
    m, f, i = WORKLOADS
    R, B = REFERENCE, BEFORE
    keys = [k for k, _ in ROWS]
    assert TILE == (128, 64) and FS == 1e9 and WITHIN == 0.10 and BITS2 == 8 and len(keys) == 11
    assert set(RUN) == set(HOSTS) == set(PRINTED_LOST) == set(PRINTED_RIGHT) == set(WORKLOADS)
    assert all(set(RUN[w]) == set(HOSTS[w]) == set(KINDS) for w in WORKLOADS)
    assert all(set(RUN[w][k]) == set(keys) and len(HOSTS[w][k]) == SEEDS for w in WORKLOADS for k in KINDS)
    assert all(len(RUN[w][k][row]) == SEEDS for w in WORKLOADS for k in KINDS for row in keys)

    # 0. The figures are grx930's, and agree with what the plan already holds.
    #    Both tables its harness printed, every cell, from the networks' own figures.
    def same(a, b):
        return abs(a[0] - b[0]) < 0.0051 and abs(a[1] - b[1]) < 0.0051

    cells = 0
    for w in WORKLOADS:
        assert set(PRINTED_LOST[w]) == set(PRINTED_RIGHT[w]) == set(keys)
        for row in keys:
            got = (lost(w, B, row), over(w, B, row), lost(w, R, row), over(w, R, row))
            assert len(PRINTED_LOST[w][row]) == 4 and all(same(a, b) for a, b in zip(got, PRINTED_LOST[w][row])), (w, row)
            got = (acc(w, B, row), acc(w, R, row), ahead(w, row, row))
            assert len(PRINTED_RIGHT[w][row]) == 3 and all(same(a, b) for a, b in zip(got, PRINTED_RIGHT[w][row])), (w, row)
            cells += 7
    assert cells == 3 * 11 * 7 == 231
    #    Counted and summed, as grx930's lines gave them.
    figures = [x for w in RUN.values() for k in w.values() for row in k.values() for x in row]
    assert len(figures) == 330 and abs(sum(figures) - 30100.06) < 1e-6
    hosts = [x for w in HOSTS.values() for k in w.values() for x in k]
    assert len(hosts) == 30 and abs(sum(hosts) - 2796.55) < 1e-6
    printed = [x for t in (PRINTED_LOST, PRINTED_RIGHT) for w in t.values() for row in w.values() for cell in row for x in cell]
    assert len(printed) == 462 and abs(sum(printed) - 6363.39) < 1e-6
    #    The networks are pta_shared_row.py's, seed for seed: on the host, and at
    #    version 2 as budgeted.
    assert shared.TILE == TILE and shared.WORKLOADS == WORKLOADS and (shared.BEFORE, shared.REFERENCE) == (B, R)
    for w in WORKLOADS:
        for k in KINDS:
            assert HOSTS[w][k] == shared.HOSTS[w][k] and RUN[w][k]["budget"] == shared.RUN[w][k]["budget"], (w, k)
    #    The networks trained before are pta_tighten.py's under a laser, where it ran
    #    them: as budgeted and at 4 to 32 times, the rescale a bit down.
    for n, w in enumerate(WORKLOADS):
        light = tighten.LASER["light"][n]
        assert r2(lost(w, B, "budget")) == light[0]
        for t in LASERS:
            if t in light:
                assert r2(lost(w, B, (DOWN, t))) == light[t], (w, t)
        assert {t for t in LASERS if t in light} == {4, 8, 16, 32}
    #    And the laser they need is the one pta_tighten.py and the working point have.
    assert [needs(w, B) for w in WORKLOADS] == [tighten.times("light", w) for w in WORKLOADS] == [wp.times_at(2, w) for w in WORKLOADS]
    assert all(laser_w(needs(w, B)) == wp.laser_at_w(2, w) for w in WORKLOADS)
    assert all(max(abs(x - y) for x, y in zip(fj_mac(needs(w, B)), wp.fj_mac_at(2, w))) < 1e-9 for w in WORKLOADS)
    assert abs(T95 - 2.776) < 1e-9

    # 1. Reading 1.  Over its own budget.
    assert [round(over(w, R, (DOWN, 4))[0], 2) for w in WORKLOADS] == [0.18, 1.10, 0.96]
    assert [round(over(w, B, (DOWN, 4))[0], 2) for w in WORKLOADS] == [0.10, 1.03, 1.14]
    assert [round(over(w, R, (DOWN, 8))[0], 2) for w in WORKLOADS] == [0.03, 0.09, 0.17]
    assert [round(over(w, B, (DOWN, 8))[0], 2) for w in WORKLOADS] == [-0.04, 0.20, 0.30]
    harder = [over_less(w, (s, t)) for w in (f, i) for s in SHIFTS for t in LASERS]
    assert len(harder) == 20 and not any(clear(x) for x in harder) and 1.2 < max(abs(errors(x)) for x in harder) < 1.4
    assert [r2(over_less(w, (DOWN, 8))) for w in (f, i)] == [(-0.11, 0.13), (-0.13, 0.13)]
    assert [r2(over_less(w, (DOWN, 4))) for w in (f, i)] == [(0.08, 0.34), (-0.18, 0.14)]
    #    On MNIST the difference is the other way, a tenth of a point at the most:
    #    a larger laser makes the old networks a little better than budgeted and the
    #    reference none.  Clear at the rule's rescale and 32 times, twelve images.
    assert all(over_less(m, (s, t))[0] > 0 for s in SHIFTS for t in LASERS if t >= 4)
    assert r2(over_less(m, (RULE, 32))) == (0.12, 0.03) and clear(over_less(m, (RULE, 32)))
    assert [(s, t) for s in SHIFTS for t in LASERS if clear(over_less(m, (s, t)))] == [(RULE, 32)]
    #    Two times is no laser for either kind: four points and more over, a bit down.
    assert all(over(w, k, (DOWN, 2))[0] > 3.5 for w in (f, i) for k in KINDS)

    # 2. Reading 2.  The laser each needs.
    assert [needs(w, R) for w in WORKLOADS] == [8, 8, 16] and [needs(w, B) for w in WORKLOADS] == [8, 16, 16]
    assert r2(over(f, R, (DOWN, 8))) == (0.09, 0.06) and not clear(over(f, R, (DOWN, 8)))
    #    A hundredth: the rule reads 0.56 less 0.47 against 0.10.
    assert round(lost(f, R, (DOWN, 8))[0], 2) == 0.56 and round(lost(f, R, "budget")[0], 2) == 0.47
    assert abs(round(lost(f, R, (DOWN, 8))[0], 2) - round(lost(f, R, "budget")[0], 2) - 0.09) < 1e-9
    #    The inverted set at 8 times is over by 0.17, and MNIST at 4 by 0.18: neither is near.
    assert r2(over(i, R, (DOWN, 8))) == (0.17, 0.07) and r2(over(m, R, (DOWN, 4))) == (0.18, 0.03)
    assert needs(i, R) == needs(i, B) == max(needs(w, k) for w in WORKLOADS for k in KINDS) == 16
    #    Read exactly and not to the hundredth, one answer of the twelve moves: the old
    #    networks on MNIST, a bit down, 0.096 over at 4 times.  It is B14's 8.
    moved = [(w, k, s) for w in WORKLOADS for k in KINDS for s in SHIFTS if needs(w, k, s, exact=True) != needs(w, k, s)]
    assert moved == [(m, B, DOWN)] and needs(m, B, exact=True) == 4 and abs(over(m, B, (DOWN, 4))[0] - 0.096) < 5e-4
    assert abs(over(f, R, (DOWN, 8))[0] - 0.086) < 5e-4 and needs(f, R, exact=True) == 8
    #    Which of the sixty cells are within three hundredths of the tenth, either side.
    near = [(w, k, s, t) for w in WORKLOADS for k in KINDS for s in SHIFTS for t in LASERS if abs(over(w, k, (s, t))[0] - WITHIN) < 0.03]
    assert near == [(m, B, DOWN, 4), (m, R, RULE, 16), (f, R, DOWN, 8), (f, R, RULE, 32)], near
    assert watts(laser_w(16)) == "1.4-14 W" and watts(laser_w(8)) == "0.7-7 W"
    assert [round(x) for x in fj_mac(16)] == [250, 2226] and [round(x) for x in fj_mac(8)] == [164, 1368]
    #    Section 3's cells, as they print: a data set's two kinds, trained before and then the reference.
    assert {name: cells for name, cells in need_rows()} == {
        "the laser it needs, a bit down": ["8 times", "8 times", "16 times", "8 times", "16 times", "16 times"],
        "which is": ["0.7-7 W", "0.7-7 W", "1.4-14 W", "0.7-7 W", "1.4-14 W", "1.4-14 W"],
        "a MAC at it": ["164-1,368 fJ", "164-1,368 fJ", "250-2,226 fJ", "164-1,368 fJ", "250-2,226 fJ", "250-2,226 fJ"],
        "at the rule's rescale": ["8 times", "16 times", "32 times", "32 times", "32 times", "32 times"],
    }, need_rows()

    # 3. Reading 3.  Half the laser.
    assert [r2(halving(w, R)) for w in WORKLOADS] == [(-0.01, 0.02), (-0.19, 0.08), (-0.19, 0.02)]
    assert [r2(halving(w, B)) for w in WORKLOADS] == [(-0.02, 0.02), (-0.19, 0.08), (-0.28, 0.05)]
    assert [w for w in WORKLOADS if clear(halving(w, R))] == [i] and [w for w in WORKLOADS if clear(halving(w, B))] == [i]
    assert all(0.15 < -halving(w, R)[0] < 0.25 for w in (f, i))

    # 4. Reading 4.  Ahead at half the laser.
    assert [round(acc(w, R, (DOWN, 8))[0], 2) for w in WORKLOADS] == [97.65, 87.33, 94.70]
    assert [round(acc(w, B, (DOWN, 16))[0], 2) for w in WORKLOADS] == [97.36, 86.92, 92.72]
    assert [r2(ahead(w, (DOWN, 8), (DOWN, 16))) for w in WORKLOADS] == [(0.29, 0.09), (0.40, 0.17), (1.97, 0.56)]
    assert [w for w in WORKLOADS if clear(ahead(w, (DOWN, 8), (DOWN, 16)))] == [m, i]
    #    It is B17's gain and not the laser's: as budgeted they are 0.38, 0.48 and 2.12 ahead,
    #    which is pta_trained.py's.
    assert [r2(ahead(w, "budget", "budget")) for w in WORKLOADS] == [(0.38, 0.11), (0.48, 0.20), (2.12, 0.45)]
    assert [r2(ahead(w, "budget", "budget")) for w in WORKLOADS] == [r2(trained.over(w, trained.REFERENCE, trained.V2, trained.BEFORE, trained.V2)) for w in WORKLOADS]

    # 5. Reading 5.  The rule's rescale.
    assert [needs(w, B, RULE) for w in WORKLOADS] == [8, 32, 32] and [needs(w, R, RULE) for w in WORKLOADS] == [16, 32, 32]
    assert all(needs(w, k, RULE) >= 2 * needs(w, k) for w in WORKLOADS for k in KINDS if (w, k) != (m, B))
    assert needs(m, B, RULE) == needs(m, B) and needs(f, R, RULE) == 4 * needs(f, R)
    #    At the rule's rescale and 32 times every one of the six is within its tenth.
    assert all(over(w, k, (RULE, 32))[0] < WITHIN for w in WORKLOADS for k in KINDS)

    # 6. And the readings say those figures, each in its place.
    for words in (
        "at 4 times 0.18, 1.10 and 0.96 of a point where the old ones are over theirs by 0.10, 1.03 and 1.14, and at 8 times 0.03, 0.09 and 0.17 for -0.04, 0.20 and 0.30.",
        "not one of the 20 differences is clear, and the largest in its errors is 1.3.",
        "THEY NEED 8, 8 AND 16 TIMES B5'S LASER, WHERE THE OLD ONES NEED 8, 16 AND 16.",
        "at 8 times they are +0.09 +-0.06 over their budget, against the rule's tenth",
        "that is 16 times for both kinds: 1.4-14 W, and a MAC of 250 to 2,226 fJ.",
        "networks' 8 on MNIST turns on one too: at 4 times they are 0.096 over",
        "read exactly they would need 4.",
        "network by network: -0.01 +-0.02, -0.19 +-0.08 and -0.19 +-0.02.",
        "For the old networks it is -0.02 +-0.02, -0.19 +-0.08 and -0.28 +-0.05.",
        "At 8 times the laser is 0.7-7 W and a MAC 164 to 1,368 fJ.",
        "at 8 times is right 97.65, 87.33 and 94.70% of the time, and the old ones at 16 times 97.36, 86.92 and 92.72%",
        "92.72%: +0.29 +-0.09, +0.40 +-0.17 and +1.97 +-0.56 seed by seed.",
        "the old networks need 8, 32 and 32 times and the reference 16, 32 and 32.",
        "A bit down it is 8, 16 and 16, and 8, 8 and 16.",
    ):
        assert said.count(words) == 1, words

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
