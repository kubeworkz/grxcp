"""
How long a calibration holds for the reference networks: drift, by the interval.

B15 calibrates version 2 every six minutes.  It was chosen on the networks
trained before, for which an hour of TFLT's drift adds 0.67 and 1.99 points on
the two harder data sets, and it left one thing unrun even for them: the
interval that would hold a tenth of a point on the inverted set, where six
minutes adds 0.28.  B17's reference networks had been drifted for six minutes
and no longer.  grx930's harness has now run both kinds from three minutes to
four hours.

MEASURED IN A MODEL, by grx930 (c930/doc/pta_error_model_design_note.md section
5, "How long a calibration holds for a trained network", 2026-10-07;
`sim/pta_mnist.sh DIR WORK refdrift`).  The working tile, 128 x 64 on two buses,
at version 2.  Two kinds of network a data set, five of each, seeds 1 to 5: the
ones trained before, and B17's reference.  Thirteen rows a network:

  as budgeted       version 2 and nothing else
  drift             after 3, 6, 15 and 30 minutes and 1, 2 and 4 hours of TFLT's
                    fitted drift
  calibrated        after an hour of it, and then C3's cell calibration
  TFLN              after an hour of TFLN's fitted drift
  held              a source's three rows at B16's 1%, 5% and 5%, at the end of
                    6 minutes, half an hour and an hour.  At six minutes it is
                    the chip as B14, B15 and B16 hold it

DERIVED here:

  a row's mean and    which have to be the two tables grx930 printed, cell for cell
  its error
  what a row adds     a network's accuracy as budgeted, less in that row
  how long a          the longest interval run at which drift adds under a tenth of
  calibration holds   a point, every shorter one doing so too: read on means to the
                      hundredth, and exactly
  a trained network   what a row adds to a reference network, less to the one
  less the old        trained before, seed by seed
  ahead               how much more often a reference network is right in one row
                      than the old one of its seed is in another

HOW SURE.  Five networks, 2.8 errors for one in twenty, as in pta_trained.py.
Drift is the noisiest thing this plan measures: every cell drifts on its own,
one network's draw is not another's, and an hour's drift has an error as large
as itself on the inverted set.  The readings say what is clear and what is not.

WHAT THIS IS NOT: a ring's drift.  Both fits are a Mach-Zehnder's bias (the
plan's section 8, question 1), every cell drifts on its own, and nothing drifts
together.  The intervals are the seven that were run.  The reference networks
were trained with Gaussian noise on their sums and not against a drifted
weight.  Version 2 and the working tile only.  And it is pta_workload.py's
three data sets.

Standard library only.  Run:  python3 docs/designs/pta_reference_drift.py
"""
import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_laser as laser
import pta_shared_row as shared
import pta_trained as trained
import pta_version2 as version2
import pta_workload as workload

MNIST, FASHION, INVERTED = workload.MNIST, workload.FASHION, workload.INVERTED
WORKLOADS = workload.WORKLOADS
TILE = laser.SMALLER                           # 128 x 64: B10
BEFORE, REFERENCE = "before", "reference"      # trained as the plan's networks were; B17's
KINDS = (BEFORE, REFERENCE)
SEEDS = 5
HOURS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4)        # the intervals drifted for
HELD = (0.1, 0.5, 1)                           # and those a source was held at the end of
B15_H = version2.INTERVAL_S / 3600             # 0.1: six minutes
WITHIN = laser.WITHIN                          # a tenth of a point: what a row of the budget costs
T95 = trained.T95


def minutes(h):
    return f"{h * 60:.0f} minutes" if h < 1 else ("an hour" if h == 1 else f"{h:g} hours")


ROWS = ((("budget", "as budgeted"),)
        + tuple((("drift", h), f"drift, {minutes(h)}") for h in HOURS)
        + (("cal", "an hour, then calibrated"), ("tfln", "an hour of TFLN's"))
        + tuple((("held", h), f"held, {minutes(h)}") for h in HELD))

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
            ("drift", 0.05): (97.41, 97.67, 97.02, 97.26, 97.17),
            ("drift", 0.1): (97.44, 97.63, 97.11, 97.34, 97.31),
            ("drift", 0.25): (97.34, 97.75, 96.96, 97.29, 97.16),
            ("drift", 0.5): (97.32, 97.70, 96.99, 97.17, 97.11),
            ("drift", 1): (97.30, 97.66, 96.82, 97.13, 96.98),
            ("drift", 2): (96.99, 97.46, 96.93, 97.04, 96.86),
            ("drift", 4): (96.89, 97.09, 96.78, 97.11, 96.77),
            "cal": (97.44, 97.69, 97.14, 97.33, 97.21),
            "tfln": (95.25, 95.62, 94.07, 95.01, 94.80),
            ("held", 0.1): (97.19, 97.63, 97.06, 97.19, 97.22),
            ("held", 0.5): (97.28, 97.61, 96.90, 96.90, 97.08),
            ("held", 1): (97.16, 97.56, 96.85, 96.97, 97.02),
        },
        REFERENCE: {
            "budget": (97.61, 97.66, 97.65, 97.84, 97.65),
            ("drift", 0.05): (97.57, 97.68, 97.63, 97.81, 97.58),
            ("drift", 0.1): (97.60, 97.68, 97.61, 97.82, 97.66),
            ("drift", 0.25): (97.52, 97.63, 97.53, 97.87, 97.51),
            ("drift", 0.5): (97.50, 97.59, 97.51, 97.80, 97.47),
            ("drift", 1): (97.50, 97.54, 97.65, 97.75, 97.64),
            ("drift", 2): (97.43, 97.60, 97.58, 97.57, 97.57),
            ("drift", 4): (97.28, 97.22, 97.36, 97.41, 97.35),
            "cal": (97.46, 97.60, 97.57, 97.73, 97.59),
            "tfln": (96.60, 95.84, 96.69, 96.88, 96.84),
            ("held", 0.1): (97.48, 97.69, 97.59, 97.88, 97.55),
            ("held", 0.5): (97.57, 97.56, 97.42, 97.80, 97.68),
            ("held", 1): (97.40, 97.58, 97.60, 97.78, 97.47),
        },
    },
    FASHION: {
        BEFORE: {
            "budget": (86.45, 86.83, 86.94, 87.65, 86.81),
            ("drift", 0.05): (86.62, 87.06, 86.90, 87.66, 86.47),
            ("drift", 0.1): (86.54, 86.90, 86.78, 87.48, 86.60),
            ("drift", 0.25): (86.31, 86.67, 86.45, 87.54, 86.74),
            ("drift", 0.5): (86.23, 86.47, 86.17, 87.69, 86.77),
            ("drift", 1): (86.02, 85.37, 85.67, 87.16, 87.09),
            ("drift", 2): (84.94, 85.58, 85.00, 85.71, 86.41),
            ("drift", 4): (84.20, 85.28, 83.26, 84.07, 86.06),
            "cal": (86.80, 86.82, 86.58, 87.54, 86.83),
            "tfln": (82.54, 74.53, 79.10, 79.79, 82.27),
            ("held", 0.1): (86.74, 86.90, 87.09, 87.07, 86.60),
            ("held", 0.5): (86.37, 86.23, 86.25, 87.13, 86.73),
            ("held", 1): (86.08, 84.90, 85.91, 86.79, 86.89),
        },
        REFERENCE: {
            "budget": (87.02, 87.76, 87.63, 87.42, 87.23),
            ("drift", 0.05): (86.97, 87.63, 87.48, 87.50, 87.08),
            ("drift", 0.1): (87.01, 87.61, 87.41, 87.67, 87.00),
            ("drift", 0.25): (86.96, 87.65, 87.41, 87.44, 87.19),
            ("drift", 0.5): (86.71, 87.63, 87.32, 87.53, 86.98),
            ("drift", 1): (86.70, 87.14, 87.25, 87.22, 87.15),
            ("drift", 2): (86.47, 87.27, 86.96, 86.74, 86.75),
            ("drift", 4): (85.13, 86.03, 86.49, 87.03, 85.82),
            "cal": (86.84, 87.69, 87.15, 87.45, 87.30),
            "tfln": (83.75, 79.57, 84.41, 83.75, 84.45),
            ("held", 0.1): (86.94, 87.51, 87.36, 87.55, 86.97),
            ("held", 0.5): (86.58, 87.60, 87.32, 87.41, 87.15),
            ("held", 1): (86.79, 87.18, 87.07, 87.24, 87.31),
        },
    },
    INVERTED: {
        BEFORE: {
            "budget": (93.65, 93.26, 90.63, 92.71, 93.47),
            ("drift", 0.05): (93.39, 93.32, 90.14, 92.61, 93.55),
            ("drift", 0.1): (92.43, 93.27, 90.26, 92.47, 93.88),
            ("drift", 0.25): (91.80, 93.07, 90.92, 92.09, 94.20),
            ("drift", 0.5): (91.94, 92.47, 91.38, 90.97, 94.24),
            ("drift", 1): (90.98, 88.38, 91.15, 88.40, 94.85),
            ("drift", 2): (86.76, 87.82, 86.86, 81.11, 92.40),
            ("drift", 4): (82.22, 76.55, 63.17, 77.91, 83.51),
            "cal": (93.52, 93.18, 90.66, 92.78, 93.63),
            "tfln": (66.60, 39.42, 51.77, 47.99, 71.06),
            ("held", 0.1): (92.15, 93.14, 90.05, 92.27, 93.73),
            ("held", 0.5): (91.57, 92.28, 91.52, 90.41, 94.12),
            ("held", 1): (90.42, 88.14, 91.28, 87.92, 94.66),
        },
        REFERENCE: {
            "budget": (95.67, 95.26, 94.38, 93.72, 95.28),
            ("drift", 0.05): (95.77, 95.21, 94.35, 93.54, 95.51),
            ("drift", 0.1): (95.72, 95.08, 94.24, 93.34, 95.54),
            ("drift", 0.25): (95.55, 94.71, 94.00, 92.36, 95.41),
            ("drift", 0.5): (95.23, 94.57, 93.88, 92.27, 95.14),
            ("drift", 1): (94.96, 93.22, 93.89, 91.07, 94.89),
            ("drift", 2): (94.55, 93.08, 90.78, 87.11, 92.29),
            ("drift", 4): (92.04, 91.36, 87.22, 83.42, 84.36),
            "cal": (95.58, 95.36, 94.52, 93.65, 95.23),
            "tfln": (83.58, 55.77, 78.60, 61.17, 69.59),
            ("held", 0.1): (95.52, 95.15, 94.24, 92.72, 95.50),
            ("held", 0.5): (94.94, 94.78, 93.95, 91.33, 94.96),
            ("held", 1): (94.73, 93.30, 93.87, 90.03, 94.72),
        },
    },
}
# The first table grx930's harness printed, mean and standard error: a row -> what the networks
# trained before lose, what the row adds to them as budgeted, and the same two for the reference.
PRINTED_LOST = {
    MNIST: {
        "budget": ((0.15, 0.02), (0.00, 0.00), (0.07, 0.05), (0.00, 0.00)),
        ("drift", 0.05): ((0.15, 0.03), (-0.00, 0.03), (0.10, 0.05), (0.03, 0.01)),
        ("drift", 0.1): ((0.09, 0.03), (-0.06, 0.04), (0.08, 0.04), (0.01, 0.01)),
        ("drift", 0.25): ((0.15, 0.03), (0.00, 0.03), (0.14, 0.05), (0.07, 0.03)),
        ("drift", 0.5): ((0.19, 0.04), (0.04, 0.03), (0.18, 0.05), (0.11, 0.02)),
        ("drift", 1): ((0.27, 0.06), (0.12, 0.05), (0.14, 0.07), (0.07, 0.03)),
        ("drift", 2): ((0.40, 0.05), (0.25, 0.04), (0.20, 0.06), (0.13, 0.04)),
        ("drift", 4): ((0.52, 0.07), (0.37, 0.08), (0.43, 0.08), (0.36, 0.03)),
        "cal": ((0.09, 0.04), (-0.06, 0.03), (0.16, 0.05), (0.09, 0.02)),
        "tfln": ((2.50, 0.17), (2.35, 0.17), (1.18, 0.22), (1.11, 0.18)),
        ("held", 0.1): ((0.19, 0.03), (0.04, 0.02), (0.11, 0.05), (0.04, 0.03)),
        ("held", 0.5): ((0.30, 0.06), (0.15, 0.06), (0.15, 0.05), (0.08, 0.04)),
        ("held", 1): ((0.34, 0.04), (0.19, 0.03), (0.19, 0.06), (0.12, 0.03)),
    },
    FASHION: {
        "budget": ((0.54, 0.22), (0.00, 0.00), (0.47, 0.13), (0.00, 0.00)),
        ("drift", 0.05): ((0.53, 0.22), (-0.01, 0.10), (0.55, 0.12), (0.08, 0.04)),
        ("drift", 0.1): ((0.62, 0.19), (0.08, 0.06), (0.55, 0.13), (0.07, 0.09)),
        ("drift", 0.25): ((0.73, 0.18), (0.19, 0.08), (0.56, 0.10), (0.08, 0.04)),
        ("drift", 0.5): ((0.81, 0.19), (0.27, 0.14), (0.65, 0.11), (0.18, 0.08)),
        ("drift", 1): ((1.21, 0.28), (0.67, 0.31), (0.79, 0.13), (0.32, 0.09)),
        ("drift", 2): ((1.95, 0.28), (1.41, 0.28), (1.05, 0.11), (0.57, 0.04)),
        ("drift", 4): ((2.90, 0.49), (2.36, 0.57), (1.79, 0.28), (1.31, 0.26)),
        "cal": ((0.56, 0.11), (0.02, 0.11), (0.60, 0.03), (0.13, 0.10)),
        "tfln": ((7.83, 1.39), (7.29, 1.50), (4.70, 1.03), (4.23, 1.00)),
        ("held", 0.1): ((0.60, 0.23), (0.06, 0.15), (0.62, 0.13), (0.15, 0.08)),
        ("held", 0.5): ((0.93, 0.13), (0.39, 0.13), (0.67, 0.10), (0.20, 0.08)),
        ("held", 1): ((1.36, 0.31), (0.82, 0.34), (0.77, 0.11), (0.29, 0.12)),
    },
    INVERTED: {
        "budget": ((0.62, 0.07), (0.00, 0.00), (0.52, 0.05), (0.00, 0.00)),
        ("drift", 0.05): ((0.76, 0.17), (0.14, 0.11), (0.51, 0.08), (-0.01, 0.07)),
        ("drift", 0.1): ((0.90, 0.33), (0.28, 0.27), (0.60, 0.11), (0.08, 0.11)),
        ("drift", 0.25): ((0.94, 0.49), (0.33, 0.44), (0.98, 0.26), (0.46, 0.25)),
        ("drift", 0.5): ((1.16, 0.58), (0.54, 0.56), (1.17, 0.23), (0.64, 0.22)),
        ("drift", 1): ((2.61, 1.25), (1.99, 1.26), (1.78, 0.46), (1.26, 0.46)),
        ("drift", 2): ((6.37, 1.78), (5.75, 1.75), (3.82, 0.96), (3.30, 0.93)),
        ("drift", 4): ((16.69, 3.09), (16.07, 3.09), (7.70, 1.58), (7.18, 1.53)),
        "cal": ((0.61, 0.10), (-0.01, 0.05), (0.52, 0.08), (-0.01, 0.05)),
        "tfln": ((37.99, 5.70), (37.38, 5.74), (25.64, 5.10), (25.12, 5.09)),
        ("held", 0.1): ((1.09, 0.36), (0.48, 0.29), (0.76, 0.22), (0.24, 0.20)),
        ("held", 0.5): ((1.38, 0.69), (0.76, 0.67), (1.39, 0.40), (0.87, 0.39)),
        ("held", 1): ((2.88, 1.33), (2.26, 1.34), (2.05, 0.61), (1.53, 0.60)),
    },
}
# The second: how often the networks trained before are right in a row, how often the reference,
# and the second less the first, seed by seed.
PRINTED_RIGHT = {
    MNIST: {
        "budget": ((97.30, 0.11), (97.68, 0.04), (0.38, 0.11)),
        ("drift", 0.05): ((97.31, 0.11), (97.65, 0.04), (0.35, 0.11)),
        ("drift", 0.1): ((97.37, 0.09), (97.67, 0.04), (0.31, 0.09)),
        ("drift", 0.25): ((97.30, 0.13), (97.61, 0.07), (0.31, 0.13)),
        ("drift", 0.5): ((97.26, 0.12), (97.57, 0.06), (0.32, 0.13)),
        ("drift", 1): ((97.18, 0.14), (97.62, 0.04), (0.44, 0.17)),
        ("drift", 2): ((97.06, 0.11), (97.55, 0.03), (0.49, 0.10)),
        ("drift", 4): ((96.93, 0.07), (97.32, 0.03), (0.40, 0.09)),
        "cal": ((97.36, 0.10), (97.59, 0.04), (0.23, 0.11)),
        "tfln": ((94.95, 0.26), (96.57, 0.19), (1.62, 0.40)),
        ("held", 0.1): ((97.26, 0.10), (97.64, 0.07), (0.38, 0.11)),
        ("held", 0.5): ((97.15, 0.13), (97.61, 0.06), (0.45, 0.16)),
        ("held", 1): ((97.11, 0.12), (97.57, 0.06), (0.45, 0.15)),
    },
    FASHION: {
        "budget": ((86.94, 0.20), (87.41, 0.13), (0.48, 0.20)),
        ("drift", 0.05): ((86.94, 0.21), (87.33, 0.13), (0.39, 0.15)),
        ("drift", 0.1): ((86.86, 0.17), (87.34, 0.14), (0.48, 0.09)),
        ("drift", 0.25): ((86.74, 0.21), (87.33, 0.12), (0.59, 0.20)),
        ("drift", 0.5): ((86.67, 0.28), (87.23, 0.17), (0.57, 0.26)),
        ("drift", 1): ((86.26, 0.37), (87.09, 0.10), (0.83, 0.36)),
        ("drift", 2): ((85.53, 0.27), (86.84, 0.13), (1.31, 0.29)),
        ("drift", 4): ((84.57, 0.49), (86.10, 0.32), (1.53, 0.67)),
        "cal": ((86.91, 0.16), (87.29, 0.14), (0.37, 0.18)),
        "tfln": ((79.65, 1.44), (83.19, 0.92), (3.54, 0.80)),
        ("held", 0.1): ((86.88, 0.09), (87.27, 0.13), (0.39, 0.07)),
        ("held", 0.5): ((86.54, 0.17), (87.21, 0.17), (0.67, 0.23)),
        ("held", 1): ((86.11, 0.36), (87.12, 0.09), (1.00, 0.35)),
    },
    INVERTED: {
        "budget": ((92.74, 0.55), (94.86, 0.36), (2.12, 0.45)),
        ("drift", 0.05): ((92.60, 0.64), (94.88, 0.41), (2.27, 0.54)),
        ("drift", 0.1): ((92.46, 0.61), (94.78, 0.44), (2.32, 0.57)),
        ("drift", 0.25): ((92.42, 0.56), (94.41, 0.58), (1.99, 0.63)),
        ("drift", 0.5): ((92.20, 0.57), (94.22, 0.54), (2.02, 0.43)),
        ("drift", 1): ((90.75, 1.19), (93.61, 0.71), (2.85, 0.81)),
        ("drift", 2): ((86.99, 1.80), (91.56, 1.27), (4.57, 1.33)),
        ("drift", 4): ((76.67, 3.62), (87.68, 1.76), (11.01, 4.00)),
        "cal": ((92.75, 0.54), (94.87, 0.35), (2.11, 0.49)),
        "tfln": ((55.37, 5.89), (69.74, 5.19), (14.37, 4.57)),
        ("held", 0.1): ((92.27, 0.63), (94.63, 0.53), (2.36, 0.65)),
        ("held", 0.5): ((91.98, 0.61), (93.99, 0.69), (2.01, 0.49)),
        ("held", 1): ((90.48, 1.23), (93.33, 0.87), (2.85, 0.89)),
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


def adds_less(w, row):
    """What a row adds to a reference network, less what it adds to the one trained
    before of the same seed."""
    return stat([(rb - r) - (bb - b) for rb, r, bb, b in zip(
        RUN[w][REFERENCE]["budget"], RUN[w][REFERENCE][row], RUN[w][BEFORE]["budget"], RUN[w][BEFORE][row])])


def ahead(w, row, than):
    """How much more often a reference network is right in a row than the one
    trained before of the same seed is in another."""
    return stat([r - b for r, b in zip(RUN[w][REFERENCE][row], RUN[w][BEFORE][than])])


def longer(w, kind, hours, than=B15_H):
    """What a kind of network gives up for a longer interval, held: how often right
    at the end of the longer, less at the end of B15's, network by network."""
    return stat([a - b for a, b in zip(RUN[w][kind][("held", hours)], RUN[w][kind][("held", than)])])


def holds(w, kind, budget=WITHIN, exact=False):
    """How long a calibration holds: the longest interval run at which drift adds
    under the budget, every shorter one doing so too.  Read on means to the
    hundredth, as the plan's other rules are, or exactly.  None if the shortest
    interval run is already over."""
    ok = None
    for h in HOURS:
        x = adds(w, kind, ("drift", h))[0]
        if (x if exact else round(x, 2)) >= budget - 1e-9:
            break
        ok = h
    return ok


def a_day(hours):
    """Calibrations a day at that interval."""
    return round(24 / hours)


def at_most(x):
    """What five networks put a figure under, at one in twenty."""
    return x[0] + T95 * x[1]


def errors(x):
    """A difference in its own standard errors."""
    return x[0] / x[1] if x[1] else 0.0


def clear(x):
    """Whether five networks put a difference outside chance at one in twenty."""
    return abs(errors(x)) > T95


def interval(h):
    return "none run" if h is None else minutes(h)


def pm(x):
    return f"{x[0]:.2f} +-{x[1]:.2f}"


def dpm(x):
    return f"{x[0]:+.2f} +-{x[1]:.2f}"


def r2(x):
    return round(x[0], 2), round(x[1], 2)


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def hold_rows():
    """Section 3's cells: how long a calibration holds, a data set's two kinds."""
    return [interval(holds(w, k)) for w in WORKLOADS for k in KINDS]


def main():
    print("How long a calibration holds for the reference networks: drift, by the interval.")
    print(f"The working tile, {laser.name(TILE)} on two buses, at version 2.  Five networks a kind, mean and standard error.")

    section("1. Points lost against a network's own accuracy on its host, and what a row adds to its budget")
    for w in WORKLOADS:
        print(f"  {w:<32}{'trained before':>16}{'and adds':>16}{'the reference':>16}{'and adds':>16}")
        for key, name in ROWS:
            print(f"    {name:<30}" + "".join(f"{pm(lost(w, k, key)):>16}{dpm(adds(w, k, key)):>16}" for k in KINDS))

    section("2. What a row adds to a reference network, less to the one trained before, seed by seed")
    print(f"  {'':<32}" + "".join(f"{w:>20}{'in its errors':>15}" for w in WORKLOADS))
    for key, name in ROWS[1:]:
        print(f"    {name:<30}" + "".join(f"{dpm(adds_less(w, key)):>20}{errors(adds_less(w, key)):>+15.1f}" for w in WORKLOADS))

    section("3. How long a calibration holds: the longest interval at which drift adds under a tenth of a point")
    print(f"  {'':<32}" + "".join(f"{(w + ', ' + ('before' if k == BEFORE else 'reference')):>28}" for w in WORKLOADS for k in KINDS))
    print(f"    {'it holds':<30}" + "".join(f"{c:>28}" for c in hold_rows()))
    print(f"  B15's interval is {minutes(B15_H)}, {a_day(B15_H)} calibrations a day; half an hour is {a_day(0.5)}, and an hour {a_day(1)}.")

    section("4. Held, with B16's source at the end of an interval: how often right, percent")
    print(f"  {'':<32}" + "".join(f"{w:>20}" for w in WORKLOADS))
    for h in HELD:
        print(f"    {'the reference, ' + minutes(h):<30}" + "".join(f"{acc(w, REFERENCE, ('held', h))[0]:>20.2f}" for w in WORKLOADS))
    for h in HELD[1:]:
        print(f"    {'  ' + minutes(h) + ' less 6 minutes':<30}" + "".join(f"{dpm(longer(w, REFERENCE, h)):>20}" for w in WORKLOADS))
    print(f"    {'trained before, 6 minutes':<30}" + "".join(f"{acc(w, BEFORE, ('held', B15_H))[0]:>20.2f}" for w in WORKLOADS))
    for h in HELD[1:]:
        print(f"    {'the reference at ' + minutes(h) + ', less':<30}" + "".join(f"{dpm(ahead(w, ('held', h), ('held', B15_H))):>20}" for w in WORKLOADS))

    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        findings()
    print(said.getvalue(), end="")
    checks(" ".join(said.getvalue().split()))


def findings():
    m, f, i = WORKLOADS
    R, B = REFERENCE, BEFORE
    six, quarter, half = ("drift", 0.1), ("drift", 0.25), ("drift", 0.5)
    print()
    print("What this says, six readings.")
    print()
    print("  1. IN THE MEAN, B15'S SIX MINUTES HOLDS A TENTH OF A POINT ON ALL THREE SETS FOR THE REFERENCE")
    print(f"     NETWORKS.  Six minutes of drift adds {dpm(adds(m, R, six))}, {dpm(adds(f, R, six))} and {dpm(adds(i, R, six))} to their budget,")
    print(f"     where it adds {adds(m, B, six)[0]:.2f}, {adds(f, B, six)[0]:.2f} and {adds(i, B, six)[0]:.2f} to the old networks'.  Two of the three have errors")
    print(f"     their own size: five networks put them under {at_most(adds(m, R, six)):.2f}, {at_most(adds(f, R, six)):.2f} and {at_most(adds(i, R, six)):.2f} at one in twenty, and")
    print("     no nearer a tenth than that.  For the old networks, on the inverted set, no interval")
    print(f"     that was run holds a tenth even in the mean: three minutes adds {dpm(adds(i, B, ('drift', 0.05)))}.  That was")
    print("     the thing B15 left unrun.")
    print()
    print("  2. AND NOTHING LONGER DOES.  A quarter of an hour adds")
    print(f"     {dpm(adds(m, R, quarter))}, {dpm(adds(f, R, quarter))} and {dpm(adds(i, R, quarter))} to the reference networks' budget.  By the")
    print(f"     rule a calibration holds {interval(holds(m, R))}, {interval(holds(f, R))} and {interval(holds(i, R))} for them, and {interval(holds(m, B))}, {interval(holds(f, B))}")
    print(f"     and {interval(holds(i, B))} for the old ones.  The set that sizes the interval is the inverted one, for both.")
    print()
    print("  3. UNDER HEAVY DRIFT A REFERENCE NETWORK LOSES HALF TO TWO THIRDS OF WHAT AN OLD ONE DOES.  An")
    print(f"     hour of TFLN's adds {adds(m, R, 'tfln')[0]:.2f}, {adds(f, R, 'tfln')[0]:.2f} and {adds(i, R, 'tfln')[0]:.2f} points for {adds(m, B, 'tfln')[0]:.2f}, {adds(f, B, 'tfln')[0]:.2f} and {adds(i, B, 'tfln')[0]:.2f}, and seed by seed")
    print(f"     that is clear on all three: {dpm(adds_less(m, 'tfln'))}, {dpm(adds_less(f, 'tfln'))} and {dpm(adds_less(i, 'tfln'))}.  Of TFLT's,")
    print(f"     two hours on Fashion-MNIST is clear, {adds(f, R, ('drift', 2))[0]:.2f} for {adds(f, B, ('drift', 2))[0]:.2f}, and four hours on the inverted set is")
    print(f"     {adds(i, R, ('drift', 4))[0]:.2f} for {adds(i, B, ('drift', 4))[0]:.2f} at {-errors(adds_less(i, ('drift', 4))):.1f} of its errors.  An hour is {adds(m, R, ('drift', 1))[0]:.2f}, {adds(f, R, ('drift', 1))[0]:.2f} and {adds(i, R, ('drift', 1))[0]:.2f} for {adds(m, B, ('drift', 1))[0]:.2f}, {adds(f, B, ('drift', 1))[0]:.2f}")
    print(f"     and {adds(i, B, ('drift', 1))[0]:.2f}, and is not clear on any.")
    print()
    short = [adds_less(w, ("drift", h)) for w in WORKLOADS for h in HOURS if h <= 0.5]
    print(f"  4. AT HALF AN HOUR AND UNDER NOTHING CAN BE TOLD BETWEEN THEM.  None of the {len(short)} differences")
    print(f"     is clear.  On the inverted set the reference networks are no better there: {adds(i, R, quarter)[0]:.2f} for")
    print(f"     {adds(i, B, quarter)[0]:.2f} at a quarter of an hour, and {adds(i, R, half)[0]:.2f} for {adds(i, B, half)[0]:.2f} at half.  What training buys against")
    print("     drift it buys where the plan does not mean to be.")
    print()
    print("  5. CALIBRATION RETURNS BOTH KINDS TO THEIR BUDGET ON THE INVERTED SET, AND LEAVES THE REFERENCE")
    print(f"     NETWORKS A TENTH SHORT ON THE OTHER TWO.  After an hour and a calibration they are")
    print(f"     {dpm(adds(m, R, 'cal'))}, {dpm(adds(f, R, 'cal'))} and {dpm(adds(i, R, 'cal'))} from their budget, and the old ones {dpm(adds(m, B, 'cal'))},")
    print(f"     {dpm(adds(f, B, 'cal'))} and {dpm(adds(i, B, 'cal'))}.  MNIST's is clear and is nine images in ten thousand.")
    print("     Every drift row here starts from weights as they were written, and not from a")
    print("     calibration.")
    print()
    print(f"  6. A WAY BACK, IF {a_day(B15_H)} CALIBRATIONS A DAY COST TOO MUCH.  Held, with B16's source at the end of")
    print(f"     the interval, half an hour costs the reference networks {dpm(longer(m, R, 0.5))}, {dpm(longer(f, R, 0.5))} and")
    print(f"     {dpm(longer(i, R, 0.5))} against six minutes, and an hour {dpm(longer(m, R, 1))}, {dpm(longer(f, R, 1))} and {dpm(longer(i, R, 1))}.")
    print(f"     At an hour, {a_day(1)} a day, they are {dpm(ahead(m, ('held', 1), ('held', B15_H)))}, {dpm(ahead(f, ('held', 1), ('held', B15_H)))} and {dpm(ahead(i, ('held', 1), ('held', B15_H)))} against")
    print("     the old networks at six minutes: ahead in the mean on all three, and clear on none.")
    print("     Whether B15 moves is the plan's to say.")


def checks(said):
    """Every claim above, as an assert.  `said` is the readings as printed, on one line."""
    m, f, i = WORKLOADS
    R, B = REFERENCE, BEFORE
    keys = [k for k, _ in ROWS]
    six, quarter, half = ("drift", 0.1), ("drift", 0.25), ("drift", 0.5)
    assert TILE == (128, 64) and WITHIN == 0.10 and B15_H == 0.1 and len(keys) == 13
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
            got = (lost(w, B, row), adds(w, B, row), lost(w, R, row), adds(w, R, row))
            assert len(PRINTED_LOST[w][row]) == 4 and all(same(a, b) for a, b in zip(got, PRINTED_LOST[w][row])), (w, row)
            got = (acc(w, B, row), acc(w, R, row), ahead(w, row, row))
            assert len(PRINTED_RIGHT[w][row]) == 3 and all(same(a, b) for a, b in zip(got, PRINTED_RIGHT[w][row])), (w, row)
            cells += 7
    assert cells == 3 * 13 * 7 == 273
    #    Counted and summed, as grx930's lines gave them.
    figures = [x for w in RUN.values() for k in w.values() for row in k.values() for x in row]
    assert len(figures) == 390 and abs(sum(figures) - 35533.91) < 1e-6
    hosts = [x for w in HOSTS.values() for k in w.values() for x in k]
    assert len(hosts) == 30 and abs(sum(hosts) - 2796.55) < 1e-6
    printed = [x for t in (PRINTED_LOST, PRINTED_RIGHT) for w in t.values() for row in w.values() for cell in row for x in cell]
    assert len(printed) == 546 and abs(sum(printed) - 7605.43) < 1e-6
    #    The networks are pta_shared_row.py's, seed for seed, in the three rows the
    #    two sweeps share: as budgeted, six minutes, and held at six minutes.
    assert shared.TILE == TILE and shared.WORKLOADS == WORKLOADS and (shared.BEFORE, shared.REFERENCE) == (B, R)
    for w in WORKLOADS:
        for k in KINDS:
            assert HOSTS[w][k] == shared.HOSTS[w][k], (w, k)
            for here, there in (("budget", "budget"), (six, "six"), (("held", 0.1), ("held", shared.IS))):
                assert RUN[w][k][here] == shared.RUN[w][k][there], (w, k, here)
    #    The networks trained before are pta_version2.py's at version 2, in every
    #    row that model has: what they lose, and what the row adds.
    there = {"budget": "budget", six: "six", ("drift", 1): "hour", ("drift", 4): "four", "tfln": "tfln", "cal": "cal",
             ("held", 0.1): version2.HELD}
    for w in WORKLOADS:
        for row, key in there.items():
            assert (r2(lost(w, B, row)), r2(adds(w, B, row))) == version2.RUN[(w, 2)][key], (w, row)
    assert version2.INTERVAL_S == 360 and version2.ROWS_V2 == (0.01, 0.05, 0.05) and abs(T95 - 2.776) < 1e-9

    # 1. Reading 1.  Six minutes.
    assert [round(adds(w, R, six)[0], 2) for w in WORKLOADS] == [0.01, 0.07, 0.08]
    assert [round(adds(w, B, six)[0], 2) for w in WORKLOADS] == [-0.06, 0.08, 0.28]
    assert all(adds(w, R, six)[0] < WITHIN for w in WORKLOADS) and not adds(i, B, six)[0] < WITHIN
    assert r2(adds(i, B, ("drift", 0.05))) == (0.14, 0.11) and holds(i, B) is None
    assert not any(clear(adds(w, R, six)) for w in WORKLOADS)
    assert [r2(adds(w, R, six)) for w in WORKLOADS] == [(0.01, 0.01), (0.07, 0.09), (0.08, 0.11)]
    assert [round(at_most(adds(w, R, six)), 2) for w in WORKLOADS] == [0.04, 0.32, 0.38]
    assert at_most(adds(m, R, six)) < WITHIN and all(at_most(adds(w, R, six)) > WITHIN for w in (f, i))
    #    B15's own table: an interval's drift on the old networks, hourly and at six minutes.
    assert [r2(adds(w, B, ("drift", 1))) for w in WORKLOADS] == [(0.12, 0.05), (0.67, 0.31), (1.99, 1.26)]
    assert [r2(adds(w, B, six)) for w in WORKLOADS] == [(-0.06, 0.04), (0.08, 0.06), (0.28, 0.27)]

    # 2. Reading 2.  How long a calibration holds.
    assert [r2(adds(w, R, quarter)) for w in WORKLOADS] == [(0.07, 0.03), (0.08, 0.04), (0.46, 0.25)]
    assert [holds(w, R) for w in WORKLOADS] == [0.25, 0.25, 0.1] and [holds(w, B) for w in WORKLOADS] == [0.5, 0.1, None]
    assert hold_rows() == ["30 minutes", "15 minutes", "6 minutes", "15 minutes", "none run", "6 minutes"]
    #    Read exactly and not to the hundredth, none of the six answers moves.
    assert all(holds(w, k, exact=True) == holds(w, k) for w in WORKLOADS for k in KINDS)
    #    The interval that holds on every set: six minutes for the reference
    #    networks, which is B15's, and none that was run for the old ones.
    assert min(holds(w, R) for w in WORKLOADS) == B15_H and any(holds(w, B) is None for w in WORKLOADS)
    assert (a_day(B15_H), a_day(0.5), a_day(1)) == (240, 48, 24)
    #    Which cells are within three hundredths of the tenth, either side.
    near = [(w, k, h) for w in WORKLOADS for k in KINDS for h in HOURS if abs(adds(w, k, ("drift", h))[0] - WITHIN) < 0.03]
    assert near == [(m, B, 1), (m, R, 0.25), (m, R, 0.5), (f, B, 0.1), (f, R, 0.05), (f, R, 0.1), (f, R, 0.25), (i, R, 0.1)], near

    # 3. Reading 3.  Heavy drift.
    assert [round(adds(w, R, "tfln")[0], 2) for w in WORKLOADS] == [1.11, 4.23, 25.12]
    assert [round(adds(w, B, "tfln")[0], 2) for w in WORKLOADS] == [2.35, 7.29, 37.38]
    assert [r2(adds_less(w, "tfln")) for w in WORKLOADS] == [(-1.24, 0.30), (-3.06, 0.79), (-12.26, 4.30)]
    assert all(clear(adds_less(w, "tfln")) for w in WORKLOADS)
    assert all(0.45 < adds(w, R, "tfln")[0] / adds(w, B, "tfln")[0] < 0.7 for w in WORKLOADS)
    assert (round(adds(f, R, ("drift", 2))[0], 2), round(adds(f, B, ("drift", 2))[0], 2)) == (0.57, 1.41)
    assert r2(adds_less(f, ("drift", 2))) == (-0.83, 0.25) and clear(adds_less(f, ("drift", 2)))
    assert (round(adds(i, R, ("drift", 4))[0], 2), round(adds(i, B, ("drift", 4))[0], 2)) == (7.18, 16.07)
    assert not clear(adds_less(i, ("drift", 4))) and 2.4 < -errors(adds_less(i, ("drift", 4))) < 2.6
    assert [round(adds(w, R, ("drift", 1))[0], 2) for w in WORKLOADS] == [0.07, 0.32, 1.26]
    assert [r2(adds_less(w, ("drift", 1))) for w in WORKLOADS] == [(-0.06, 0.08), (-0.35, 0.23), (-0.74, 0.90)]
    assert not any(clear(adds_less(w, ("drift", 1))) for w in WORKLOADS)
    #    Of TFLT's twenty-one cells, the one that is clear is Fashion-MNIST's two hours.
    assert [(w, h) for w in WORKLOADS for h in HOURS if clear(adds_less(w, ("drift", h)))] == [(f, 2)]
    #    From two hours up the reference adds less on every set, by a half on the two harder.
    assert all(adds_less(w, ("drift", h))[0] < 0 for w in WORKLOADS for h in (1, 2, 4))
    assert all(0.35 < adds(w, R, ("drift", h))[0] / adds(w, B, ("drift", h))[0] < 0.65 for w in (f, i) for h in (2, 4))

    # 4. Reading 4.  Half an hour and under.
    short = [adds_less(w, ("drift", h)) for w in WORKLOADS for h in HOURS if h <= 0.5]
    assert len(short) == 12 and not any(clear(x) for x in short)
    assert (round(adds(i, R, quarter)[0], 2), round(adds(i, B, quarter)[0], 2)) == (0.46, 0.33)
    assert (round(adds(i, R, half)[0], 2), round(adds(i, B, half)[0], 2)) == (0.64, 0.54)
    assert all(abs(errors(adds_less(i, ("drift", h)))) < 1.5 for h in HOURS if h <= 0.5)

    # 5. Reading 5.  Calibration.
    assert [r2(adds(w, R, "cal")) for w in WORKLOADS] == [(0.09, 0.02), (0.13, 0.10), (-0.01, 0.05)]
    assert [r2(adds(w, B, "cal")) for w in WORKLOADS] == [(-0.06, 0.03), (0.02, 0.11), (-0.01, 0.05)]
    assert [w for w in WORKLOADS if clear(adds(w, R, "cal"))] == [m] and not any(clear(adds(w, B, "cal")) for w in WORKLOADS)
    assert r2(adds_less(m, "cal")) == (0.15, 0.05) and clear(adds_less(m, "cal"))
    #    On MNIST a calibrated reference network is no better off than one left to
    #    drift for the hour: 0.09 for 0.07.
    assert adds(m, R, "cal")[0] > adds(m, R, ("drift", 1))[0]

    # 6. Reading 6.  Held, and a longer interval.
    assert [r2(longer(w, R, 0.5)) for w in WORKLOADS] == [(-0.03, 0.06), (-0.05, 0.09), (-0.63, 0.20)]
    assert [r2(longer(w, R, 1)) for w in WORKLOADS] == [(-0.07, 0.02), (-0.15, 0.13), (-1.30, 0.43)]
    assert [w for w in WORKLOADS if clear(longer(w, R, 0.5))] == [i] and [w for w in WORKLOADS if clear(longer(w, R, 1))] == [m, i]
    assert [r2(ahead(w, ("held", 1), ("held", B15_H))) for w in WORKLOADS] == [(0.31, 0.12), (0.24, 0.13), (1.06, 1.04)]
    assert all(ahead(w, ("held", 1), ("held", B15_H))[0] > 0 and not clear(ahead(w, ("held", 1), ("held", B15_H))) for w in WORKLOADS)
    #    At half an hour: 0.35, 0.33 and 1.72 ahead, MNIST's clear.
    assert [r2(ahead(w, ("held", 0.5), ("held", B15_H))) for w in WORKLOADS] == [(0.35, 0.11), (0.33, 0.15), (1.72, 0.81)]
    #    What they lose so held: 0.11, 0.62 and 0.76 at six minutes, which is
    #    pta_trained.py's reference row; 0.15, 0.67 and 1.39 at half an hour; 0.19,
    #    0.77 and 2.05 at an hour.
    assert [round(lost(w, R, ("held", h))[0], 2) for h in HELD for w in WORKLOADS] == [0.11, 0.62, 0.76, 0.15, 0.67, 1.39, 0.19, 0.77, 2.05]
    assert [r2(lost(w, R, ("held", 0.1))) for w in WORKLOADS] == [r2(trained.lost(w, trained.REFERENCE, trained.V2_HELD)) for w in WORKLOADS]

    # 7. And the readings say those figures, each in its place.
    for words in (
        "Six minutes of drift adds +0.01 +-0.01, +0.07 +-0.09 and +0.08 +-0.11 to their budget",
        "where it adds -0.06, 0.08 and 0.28 to the old networks'",
        "put them under 0.04, 0.32 and 0.38 at one in twenty",
        "three minutes adds +0.14 +-0.11.",
        "A quarter of an hour adds +0.07 +-0.03, +0.08 +-0.04 and +0.46 +-0.25 to the reference networks' budget.",
        "holds 15 minutes, 15 minutes and 6 minutes for them, and 30 minutes, 6 minutes and none run for the old ones.",
        "hour of TFLN's adds 1.11, 4.23 and 25.12 points for 2.35, 7.29 and 37.38",
        "clear on all three: -1.24 +-0.30, -3.06 +-0.79 and -12.26 +-4.30.",
        "two hours on Fashion-MNIST is clear, 0.57 for 1.41, and four hours on the inverted set is 7.18 for 16.07 at 2.5 of its errors.",
        "An hour is 0.07, 0.32 and 1.26 for 0.12, 0.67 and 1.99, and is not clear on any.",
        "None of the 12 differences is clear.",
        "no better there: 0.46 for 0.33 at a quarter of an hour, and 0.64 for 0.54 at half.",
        "they are +0.09 +-0.02, +0.13 +-0.10 and -0.01 +-0.05 from their budget, and the old ones -0.06 +-0.03, +0.02 +-0.11 and -0.01 +-0.05.",
        "IF 240 CALIBRATIONS A DAY COST TOO MUCH.",
        "half an hour costs the reference networks -0.03 +-0.06, -0.05 +-0.09 and -0.63 +-0.20 against six minutes, and an hour -0.07 +-0.02, -0.15 +-0.13 and -1.30 +-0.43.",
        "At an hour, 24 a day, they are +0.31 +-0.12, +0.24 +-0.13 and +1.06 +-1.04 against the old networks at six minutes",
    ):
        assert said.count(words) == 1, words

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
