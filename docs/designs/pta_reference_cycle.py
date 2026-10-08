"""
The operating cycle: an interval that starts from a calibration.

Every drift figure in this plan aged a tile from weights as they were written.
A tile in use is never that.  It is calibrated, drifts for an interval, and is
calibrated again, so the state B15's interval ends in is a calibrated tile's
and six minutes of drift.  pta_reference_drift.py found a calibration leaves
B17's reference networks about a tenth of a point short of their budget on two
data sets, and could not say whether that was the calibration or the hour of
drift before it, or what it does to the end of an interval.  grx930's harness
has now run the cycle itself.

CORRECTED, 2026-10-08, by pta_reference_calibration.py.  A calibrated row here and
one that is not do not meet the same draws: a calibration's probes are GEMMs,
and each takes the run's next seed.  So a calibrated row less the row as
budgeted is two things, the trims a calibration writes and the draws its run
meets, and readings 1, 2, 3 and 7 below took the two for the calibration's
cost.  grx930 has since taken them apart, and it was the draws.  The figures
here are grx930's and stand.  So does what is read between two calibrated
rows, which share their draws.  pta_reference_calibration.py has a cycle's
rows over the tile as budgeted on their own draws, and those replace the
"over their budget" of readings 2 and 3 and the three draws of reading 5.

MEASURED IN A MODEL, by grx930 (c930/doc/pta_error_model_design_note.md section
5, "An interval that starts from a calibration", 2026-10-07; `sim/pta_mnist.sh
DIR WORK refcycle`).  The working tile, 128 x 64 on two buses, at version 2,
TFLT's fitted drift, C3's cell calibration with 16 probes a cell.  Two kinds of
network a data set, five of each, seeds 1 to 5: the ones trained before, and
B17's reference.  Nineteen rows a network:

  as budgeted       version 2 and nothing else
  calibrated        a tile calibrated as it was written, with no drift at all:
                    what a calibration does by itself
  aged, calibrated  after six minutes and after an hour of drift, then calibrated:
                    the start of an interval
  a cycle           aged an interval, calibrated, and aged the interval again:
                    the end of an interval that started from a calibration.  At
                    3, 6, 15 and 30 minutes and an hour
  an hour, then     aged an hour, calibrated, and six minutes more: whether what
  six minutes       came before a calibration matters
  as written        aged from weights as written, 6 and 30 minutes and an hour:
                    what the plan has had until now
  held              a source's three rows at B16's 1%, 5% and 5%, at the end of
                    each of those three intervals, as written and as a cycle

DERIVED here:

  a row's mean and    which have to be the two tables grx930 printed, cell for cell
  its error
  what a row adds     a network's accuracy as budgeted, less in that row
  what a calibration  the same, for the calibrated rows; and a cycle less the same
  costs               interval from weights as written, network by network
  how long a          the longest interval at which a cycle adds under a tenth of
  calibration holds   a point, every shorter one doing so too
  a trained network   what a row adds to a reference network, less to the one
  less the old        trained before, seed by seed

HOW SURE.  Five networks, 2.8 errors for one in twenty, as in pta_trained.py.  The
rows of a pair share a seed and so the drift's and the calibration's draws, which
makes a pair's difference small and its error smaller.  A hundredth of a point is
one image in ten thousand.

WHAT THIS IS NOT: a schedule.  One calibration, where a tile in use has had
hundreds; nothing here says the shortfall does or does not build.  C3's
calibration as grx930's harness has it, 16 probes a cell and a trim step of a
quarter of a weight's LSB, and no other.  A calibration that sees no source's
noise: the probes are taken before the light is lit.  A Mach-Zehnder's fit, every
cell drifting on its own.  Version 2 and the working tile only.

Standard library only.  Run:  python3 docs/designs/pta_reference_cycle.py
"""
import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_laser as laser
import pta_reference_drift as drift
import pta_trained as trained
import pta_version2 as version2
import pta_workload as workload

MNIST, FASHION, INVERTED = workload.MNIST, workload.FASHION, workload.INVERTED
WORKLOADS = workload.WORKLOADS
TILE = laser.SMALLER                           # 128 x 64: B10
BEFORE, REFERENCE = "before", "reference"      # trained as the plan's networks were; B17's
KINDS = (BEFORE, REFERENCE)
SEEDS = 5
CYCLES = (0.05, 0.1, 0.25, 0.5, 1)             # the intervals cycled at
HELD = (0.1, 0.5, 1)                           # and those held, and aged from weights as written
B15_H = version2.INTERVAL_S / 3600             # 0.1: six minutes
PROBES = 16                                    # a calibration's probes a cell: pta_chiplet_calibration.md's section 3
WITHIN = laser.WITHIN                          # a tenth of a point: what a row of the budget costs
T95 = trained.T95
minutes = drift.minutes
ROWS = ((("budget", "as budgeted"), ("cal", "calibrated as written"))
        + tuple((("aged", h), f"aged {minutes(h)}, calibrated") for h in (0.1, 1))
        + tuple((("cycle", h), f"a cycle of {minutes(h)}") for h in CYCLES)
        + (("after", "an hour, calibrated, 6 minutes"),)
        + tuple((("fresh", h), f"as written, {minutes(h)}") for h in HELD)
        + tuple((("held", h), f"held, as written, {minutes(h)}") for h in HELD)
        + tuple((("heldcycle", h), f"held, a cycle of {minutes(h)}") for h in HELD))

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
            "cal": (97.43, 97.64, 97.16, 97.28, 97.25),
            ("aged", 0.1): (97.45, 97.57, 97.18, 97.32, 97.19),
            ("aged", 1): (97.44, 97.69, 97.14, 97.33, 97.21),
            ("cycle", 0.05): (97.40, 97.61, 97.17, 97.34, 97.18),
            ("cycle", 0.1): (97.43, 97.61, 97.10, 97.35, 97.25),
            ("cycle", 0.25): (97.41, 97.59, 97.21, 97.15, 97.25),
            ("cycle", 0.5): (97.42, 97.59, 97.14, 97.29, 97.13),
            ("cycle", 1): (97.38, 97.39, 97.07, 97.08, 97.21),
            "after": (97.48, 97.62, 97.22, 97.26, 97.13),
            ("fresh", 0.1): (97.44, 97.63, 97.11, 97.34, 97.31),
            ("fresh", 0.5): (97.32, 97.70, 96.99, 97.17, 97.11),
            ("fresh", 1): (97.30, 97.66, 96.82, 97.13, 96.98),
            ("held", 0.1): (97.19, 97.63, 97.06, 97.19, 97.22),
            ("held", 0.5): (97.28, 97.61, 96.90, 96.90, 97.08),
            ("held", 1): (97.16, 97.56, 96.85, 96.97, 97.02),
            ("heldcycle", 0.1): (97.34, 97.59, 97.01, 97.22, 97.23),
            ("heldcycle", 0.5): (97.28, 97.56, 97.07, 97.06, 97.19),
            ("heldcycle", 1): (97.27, 97.35, 97.12, 96.98, 97.09),
        },
        REFERENCE: {
            "budget": (97.61, 97.66, 97.65, 97.84, 97.65),
            "cal": (97.51, 97.59, 97.56, 97.73, 97.60),
            ("aged", 0.1): (97.54, 97.58, 97.52, 97.73, 97.58),
            ("aged", 1): (97.46, 97.60, 97.57, 97.73, 97.59),
            ("cycle", 0.05): (97.43, 97.64, 97.54, 97.82, 97.63),
            ("cycle", 0.1): (97.51, 97.59, 97.46, 97.89, 97.54),
            ("cycle", 0.25): (97.36, 97.61, 97.58, 97.85, 97.60),
            ("cycle", 0.5): (97.39, 97.51, 97.50, 97.71, 97.58),
            ("cycle", 1): (97.35, 97.56, 97.49, 97.67, 97.66),
            "after": (97.54, 97.59, 97.50, 97.78, 97.53),
            ("fresh", 0.1): (97.60, 97.68, 97.61, 97.82, 97.66),
            ("fresh", 0.5): (97.50, 97.59, 97.51, 97.80, 97.47),
            ("fresh", 1): (97.50, 97.54, 97.65, 97.75, 97.64),
            ("held", 0.1): (97.48, 97.69, 97.59, 97.88, 97.55),
            ("held", 0.5): (97.57, 97.56, 97.42, 97.80, 97.68),
            ("held", 1): (97.40, 97.58, 97.60, 97.78, 97.47),
            ("heldcycle", 0.1): (97.45, 97.58, 97.37, 97.94, 97.53),
            ("heldcycle", 0.5): (97.36, 97.59, 97.48, 97.80, 97.58),
            ("heldcycle", 1): (97.36, 97.48, 97.41, 97.66, 97.65),
        },
    },
    FASHION: {
        BEFORE: {
            "budget": (86.45, 86.83, 86.94, 87.65, 86.81),
            "cal": (86.72, 86.94, 86.59, 87.46, 86.83),
            ("aged", 0.1): (86.66, 86.92, 86.69, 87.54, 86.78),
            ("aged", 1): (86.80, 86.82, 86.58, 87.54, 86.83),
            ("cycle", 0.05): (86.42, 86.87, 86.82, 87.37, 86.91),
            ("cycle", 0.1): (86.51, 86.67, 86.75, 87.36, 86.60),
            ("cycle", 0.25): (86.87, 86.62, 86.06, 87.58, 87.04),
            ("cycle", 0.5): (86.45, 86.22, 86.39, 87.16, 86.60),
            ("cycle", 1): (85.82, 86.56, 85.88, 87.07, 86.83),
            "after": (86.77, 86.85, 86.61, 87.59, 87.05),
            ("fresh", 0.1): (86.54, 86.90, 86.78, 87.48, 86.60),
            ("fresh", 0.5): (86.23, 86.47, 86.17, 87.69, 86.77),
            ("fresh", 1): (86.02, 85.37, 85.67, 87.16, 87.09),
            ("held", 0.1): (86.74, 86.90, 87.09, 87.07, 86.60),
            ("held", 0.5): (86.37, 86.23, 86.25, 87.13, 86.73),
            ("held", 1): (86.08, 84.90, 85.91, 86.79, 86.89),
            ("heldcycle", 0.1): (86.50, 86.75, 87.08, 87.15, 86.61),
            ("heldcycle", 0.5): (86.59, 86.10, 86.46, 86.65, 86.47),
            ("heldcycle", 1): (86.08, 86.66, 85.96, 86.66, 86.57),
        },
        REFERENCE: {
            "budget": (87.02, 87.76, 87.63, 87.42, 87.23),
            "cal": (86.68, 87.67, 87.13, 87.45, 87.40),
            ("aged", 0.1): (86.71, 87.64, 87.13, 87.49, 87.45),
            ("aged", 1): (86.84, 87.69, 87.15, 87.45, 87.30),
            ("cycle", 0.05): (86.75, 87.76, 87.19, 87.52, 87.15),
            ("cycle", 0.1): (86.73, 87.62, 87.23, 87.43, 87.24),
            ("cycle", 0.25): (86.74, 87.60, 87.04, 87.58, 86.93),
            ("cycle", 0.5): (86.83, 87.62, 87.35, 87.52, 87.48),
            ("cycle", 1): (86.85, 87.30, 86.84, 87.33, 86.85),
            "after": (86.66, 87.79, 87.28, 87.74, 87.07),
            ("fresh", 0.1): (87.01, 87.61, 87.41, 87.67, 87.00),
            ("fresh", 0.5): (86.71, 87.63, 87.32, 87.53, 86.98),
            ("fresh", 1): (86.70, 87.14, 87.25, 87.22, 87.15),
            ("held", 0.1): (86.94, 87.51, 87.36, 87.55, 86.97),
            ("held", 0.5): (86.58, 87.60, 87.32, 87.41, 87.15),
            ("held", 1): (86.79, 87.18, 87.07, 87.24, 87.31),
            ("heldcycle", 0.1): (86.77, 87.65, 87.20, 87.51, 87.36),
            ("heldcycle", 0.5): (86.96, 87.54, 87.04, 87.12, 87.13),
            ("heldcycle", 1): (86.84, 87.17, 86.95, 87.30, 87.17),
        },
    },
    INVERTED: {
        BEFORE: {
            "budget": (93.65, 93.26, 90.63, 92.71, 93.47),
            "cal": (93.38, 93.15, 90.65, 92.68, 93.56),
            ("aged", 0.1): (93.33, 93.28, 90.34, 92.57, 93.50),
            ("aged", 1): (93.52, 93.18, 90.66, 92.78, 93.63),
            ("cycle", 0.05): (92.12, 93.26, 90.55, 92.24, 93.84),
            ("cycle", 0.1): (92.77, 92.94, 90.31, 93.11, 93.90),
            ("cycle", 0.25): (93.33, 92.41, 91.54, 92.05, 93.67),
            ("cycle", 0.5): (92.94, 90.05, 89.67, 91.87, 94.28),
            ("cycle", 1): (90.92, 91.10, 88.98, 90.35, 90.69),
            "after": (92.62, 92.87, 91.04, 92.12, 93.32),
            ("fresh", 0.1): (92.43, 93.27, 90.26, 92.47, 93.88),
            ("fresh", 0.5): (91.94, 92.47, 91.38, 90.97, 94.24),
            ("fresh", 1): (90.98, 88.38, 91.15, 88.40, 94.85),
            ("held", 0.1): (92.15, 93.14, 90.05, 92.27, 93.73),
            ("held", 0.5): (91.57, 92.28, 91.52, 90.41, 94.12),
            ("held", 1): (90.42, 88.14, 91.28, 87.92, 94.66),
            ("heldcycle", 0.1): (92.39, 92.93, 90.12, 92.78, 94.07),
            ("heldcycle", 0.5): (92.22, 89.78, 89.46, 91.63, 94.15),
            ("heldcycle", 1): (90.62, 90.82, 89.26, 89.93, 91.13),
        },
        REFERENCE: {
            "budget": (95.67, 95.26, 94.38, 93.72, 95.28),
            "cal": (95.65, 95.33, 94.56, 93.72, 95.30),
            ("aged", 0.1): (95.59, 95.43, 94.51, 93.58, 95.22),
            ("aged", 1): (95.58, 95.36, 94.52, 93.65, 95.23),
            ("cycle", 0.05): (95.63, 95.34, 94.48, 93.44, 95.33),
            ("cycle", 0.1): (95.59, 95.14, 94.22, 93.42, 95.47),
            ("cycle", 0.25): (95.14, 94.98, 94.50, 93.44, 95.02),
            ("cycle", 0.5): (95.60, 93.85, 94.49, 93.10, 95.02),
            ("cycle", 1): (95.06, 94.37, 92.60, 91.92, 92.45),
            "after": (95.47, 95.25, 94.13, 93.09, 94.98),
            ("fresh", 0.1): (95.72, 95.08, 94.24, 93.34, 95.54),
            ("fresh", 0.5): (95.23, 94.57, 93.88, 92.27, 95.14),
            ("fresh", 1): (94.96, 93.22, 93.89, 91.07, 94.89),
            ("held", 0.1): (95.52, 95.15, 94.24, 92.72, 95.50),
            ("held", 0.5): (94.94, 94.78, 93.95, 91.33, 94.96),
            ("held", 1): (94.73, 93.30, 93.87, 90.03, 94.72),
            ("heldcycle", 0.1): (95.45, 95.08, 94.23, 92.64, 95.50),
            ("heldcycle", 0.5): (95.39, 94.15, 94.15, 92.42, 95.13),
            ("heldcycle", 1): (94.68, 94.07, 92.73, 90.83, 92.65),
        },
    },
}
# The first table grx930's harness printed, mean and standard error: a row -> what the networks
# trained before lose, what the row adds to them as budgeted, and the same two for the reference.
PRINTED_LOST = {
    MNIST: {
        "budget": ((0.15, 0.02), (0.00, 0.00), (0.07, 0.05), (0.00, 0.00)),
        "cal": ((0.10, 0.04), (-0.05, 0.04), (0.15, 0.05), (0.08, 0.01)),
        ("aged", 0.1): ((0.11, 0.05), (-0.04, 0.05), (0.16, 0.04), (0.09, 0.01)),
        ("aged", 1): ((0.09, 0.04), (-0.06, 0.03), (0.16, 0.05), (0.09, 0.02)),
        ("cycle", 0.05): ((0.11, 0.04), (-0.04, 0.04), (0.14, 0.05), (0.07, 0.03)),
        ("cycle", 0.1): ((0.10, 0.03), (-0.05, 0.04), (0.15, 0.05), (0.08, 0.04)),
        ("cycle", 0.25): ((0.13, 0.06), (-0.02, 0.06), (0.15, 0.07), (0.08, 0.04)),
        ("cycle", 0.5): ((0.14, 0.05), (-0.01, 0.05), (0.21, 0.06), (0.14, 0.02)),
        ("cycle", 1): ((0.23, 0.07), (0.08, 0.08), (0.21, 0.06), (0.14, 0.04)),
        "after": ((0.11, 0.07), (-0.04, 0.06), (0.16, 0.04), (0.09, 0.02)),
        ("fresh", 0.1): ((0.09, 0.03), (-0.06, 0.04), (0.08, 0.04), (0.01, 0.01)),
        ("fresh", 0.5): ((0.19, 0.04), (0.04, 0.03), (0.18, 0.05), (0.11, 0.02)),
        ("fresh", 1): ((0.27, 0.06), (0.12, 0.05), (0.14, 0.07), (0.07, 0.03)),
        ("held", 0.1): ((0.19, 0.03), (0.04, 0.02), (0.11, 0.05), (0.04, 0.03)),
        ("held", 0.5): ((0.30, 0.06), (0.15, 0.06), (0.15, 0.05), (0.08, 0.04)),
        ("held", 1): ((0.34, 0.04), (0.19, 0.03), (0.19, 0.06), (0.12, 0.03)),
        ("heldcycle", 0.1): ((0.17, 0.02), (0.02, 0.03), (0.18, 0.06), (0.11, 0.06)),
        ("heldcycle", 0.5): ((0.22, 0.05), (0.07, 0.04), (0.19, 0.05), (0.12, 0.04)),
        ("heldcycle", 1): ((0.29, 0.08), (0.14, 0.08), (0.24, 0.06), (0.17, 0.04)),
    },
    FASHION: {
        "budget": ((0.54, 0.22), (0.00, 0.00), (0.47, 0.13), (0.00, 0.00)),
        "cal": ((0.57, 0.13), (0.03, 0.11), (0.62, 0.04), (0.15, 0.12)),
        ("aged", 0.1): ((0.56, 0.15), (0.02, 0.08), (0.60, 0.05), (0.13, 0.13)),
        ("aged", 1): ((0.56, 0.11), (0.02, 0.11), (0.60, 0.03), (0.13, 0.10)),
        ("cycle", 0.05): ((0.60, 0.20), (0.06, 0.07), (0.61, 0.06), (0.14, 0.10)),
        ("cycle", 0.1): ((0.70, 0.18), (0.16, 0.06), (0.64, 0.06), (0.16, 0.08)),
        ("cycle", 0.25): ((0.64, 0.09), (0.10, 0.22), (0.71, 0.09), (0.23, 0.12)),
        ("cycle", 0.5): ((0.91, 0.13), (0.37, 0.12), (0.53, 0.07), (0.05, 0.10)),
        ("cycle", 1): ((1.04, 0.22), (0.50, 0.18), (0.85, 0.10), (0.38, 0.12)),
        "after": ((0.50, 0.12), (-0.04, 0.12), (0.58, 0.11), (0.10, 0.13)),
        ("fresh", 0.1): ((0.62, 0.19), (0.08, 0.06), (0.55, 0.13), (0.07, 0.09)),
        ("fresh", 0.5): ((0.81, 0.19), (0.27, 0.14), (0.65, 0.11), (0.18, 0.08)),
        ("fresh", 1): ((1.21, 0.28), (0.67, 0.31), (0.79, 0.13), (0.32, 0.09)),
        ("held", 0.1): ((0.60, 0.23), (0.06, 0.15), (0.62, 0.13), (0.15, 0.08)),
        ("held", 0.5): ((0.93, 0.13), (0.39, 0.13), (0.67, 0.10), (0.20, 0.08)),
        ("held", 1): ((1.36, 0.31), (0.82, 0.34), (0.77, 0.11), (0.29, 0.12)),
        ("heldcycle", 0.1): ((0.66, 0.24), (0.12, 0.11), (0.59, 0.04), (0.11, 0.10)),
        ("heldcycle", 0.5): ((1.02, 0.16), (0.48, 0.19), (0.73, 0.09), (0.25, 0.09)),
        ("heldcycle", 1): ((1.09, 0.17), (0.55, 0.18), (0.80, 0.10), (0.33, 0.13)),
    },
    INVERTED: {
        "budget": ((0.62, 0.07), (0.00, 0.00), (0.52, 0.05), (0.00, 0.00)),
        "cal": ((0.68, 0.11), (0.06, 0.06), (0.47, 0.06), (-0.05, 0.04)),
        ("aged", 0.1): ((0.76, 0.14), (0.14, 0.07), (0.52, 0.09), (-0.00, 0.06)),
        ("aged", 1): ((0.61, 0.10), (-0.01, 0.05), (0.52, 0.08), (-0.01, 0.05)),
        ("cycle", 0.05): ((0.96, 0.38), (0.34, 0.33), (0.54, 0.10), (0.02, 0.07)),
        ("cycle", 0.1): ((0.75, 0.28), (0.14, 0.25), (0.62, 0.08), (0.09, 0.08)),
        ("cycle", 0.25): ((0.76, 0.30), (0.14, 0.32), (0.77, 0.10), (0.25, 0.10)),
        ("cycle", 0.5): ((1.60, 0.61), (0.98, 0.64), (0.97, 0.26), (0.45, 0.27)),
        ("cycle", 1): ((2.95, 0.23), (2.34, 0.21), (2.10, 0.43), (1.58, 0.39)),
        "after": ((0.97, 0.27), (0.35, 0.24), (0.80, 0.14), (0.28, 0.10)),
        ("fresh", 0.1): ((0.90, 0.33), (0.28, 0.27), (0.60, 0.11), (0.08, 0.11)),
        ("fresh", 0.5): ((1.16, 0.58), (0.54, 0.56), (1.17, 0.23), (0.64, 0.22)),
        ("fresh", 1): ((2.61, 1.25), (1.99, 1.26), (1.78, 0.46), (1.26, 0.46)),
        ("held", 0.1): ((1.09, 0.36), (0.48, 0.29), (0.76, 0.22), (0.24, 0.20)),
        ("held", 0.5): ((1.38, 0.69), (0.76, 0.67), (1.39, 0.40), (0.87, 0.39)),
        ("held", 1): ((2.88, 1.33), (2.26, 1.34), (2.05, 0.61), (1.53, 0.60)),
        ("heldcycle", 0.1): ((0.90, 0.35), (0.29, 0.31), (0.80, 0.23), (0.28, 0.21)),
        ("heldcycle", 0.5): ((1.91, 0.64), (1.30, 0.66), (1.14, 0.25), (0.61, 0.24)),
        ("heldcycle", 1): ((3.01, 0.30), (2.39, 0.28), (2.39, 0.42), (1.87, 0.38)),
    },
}
# The second: how often the networks trained before are right in a row, how often the reference,
# and the second less the first, seed by seed.
PRINTED_RIGHT = {
    MNIST: {
        "budget": ((97.30, 0.11), (97.68, 0.04), (0.38, 0.11)),
        "cal": ((97.35, 0.08), (97.60, 0.04), (0.25, 0.10)),
        ("aged", 0.1): ((97.34, 0.08), (97.59, 0.04), (0.25, 0.08)),
        ("aged", 1): ((97.36, 0.10), (97.59, 0.04), (0.23, 0.11)),
        ("cycle", 0.05): ((97.34, 0.08), (97.61, 0.06), (0.27, 0.10)),
        ("cycle", 0.1): ((97.35, 0.09), (97.60, 0.08), (0.25, 0.10)),
        ("cycle", 0.25): ((97.32, 0.08), (97.60, 0.08), (0.28, 0.14)),
        ("cycle", 0.5): ((97.31, 0.09), (97.54, 0.05), (0.22, 0.12)),
        ("cycle", 1): ((97.23, 0.07), (97.55, 0.06), (0.32, 0.11)),
        "after": ((97.34, 0.09), (97.59, 0.05), (0.25, 0.10)),
        ("fresh", 0.1): ((97.37, 0.09), (97.67, 0.04), (0.31, 0.09)),
        ("fresh", 0.5): ((97.26, 0.12), (97.57, 0.06), (0.32, 0.13)),
        ("fresh", 1): ((97.18, 0.14), (97.62, 0.04), (0.44, 0.17)),
        ("held", 0.1): ((97.26, 0.10), (97.64, 0.07), (0.38, 0.11)),
        ("held", 0.5): ((97.15, 0.13), (97.61, 0.06), (0.45, 0.16)),
        ("held", 1): ((97.11, 0.12), (97.57, 0.06), (0.45, 0.15)),
        ("heldcycle", 0.1): ((97.28, 0.09), (97.57, 0.10), (0.30, 0.12)),
        ("heldcycle", 0.5): ((97.23, 0.09), (97.56, 0.07), (0.33, 0.13)),
        ("heldcycle", 1): ((97.16, 0.07), (97.51, 0.06), (0.35, 0.12)),
    },
    FASHION: {
        "budget": ((86.94, 0.20), (87.41, 0.13), (0.48, 0.20)),
        "cal": ((86.91, 0.15), (87.27, 0.17), (0.36, 0.16)),
        ("aged", 0.1): ((86.92, 0.16), (87.28, 0.17), (0.37, 0.16)),
        ("aged", 1): ((86.91, 0.16), (87.29, 0.14), (0.37, 0.18)),
        ("cycle", 0.05): ((86.88, 0.15), (87.27, 0.17), (0.40, 0.13)),
        ("cycle", 0.1): ((86.78, 0.15), (87.25, 0.15), (0.47, 0.16)),
        ("cycle", 0.25): ((86.83, 0.25), (87.18, 0.17), (0.34, 0.26)),
        ("cycle", 0.5): ((86.56, 0.16), (87.36, 0.14), (0.80, 0.20)),
        ("cycle", 1): ((86.43, 0.25), (87.03, 0.11), (0.60, 0.20)),
        "after": ((86.97, 0.17), (87.31, 0.21), (0.33, 0.20)),
        ("fresh", 0.1): ((86.86, 0.17), (87.34, 0.14), (0.48, 0.09)),
        ("fresh", 0.5): ((86.67, 0.28), (87.23, 0.17), (0.57, 0.26)),
        ("fresh", 1): ((86.26, 0.37), (87.09, 0.10), (0.83, 0.36)),
        ("held", 0.1): ((86.88, 0.09), (87.27, 0.13), (0.39, 0.07)),
        ("held", 0.5): ((86.54, 0.17), (87.21, 0.17), (0.67, 0.23)),
        ("held", 1): ((86.11, 0.36), (87.12, 0.09), (1.00, 0.35)),
        ("heldcycle", 0.1): ((86.82, 0.13), (87.30, 0.15), (0.48, 0.15)),
        ("heldcycle", 0.5): ((86.45, 0.10), (87.16, 0.10), (0.70, 0.19)),
        ("heldcycle", 1): ((86.39, 0.15), (87.09, 0.08), (0.70, 0.08)),
    },
    INVERTED: {
        "budget": ((92.74, 0.55), (94.86, 0.36), (2.12, 0.45)),
        "cal": ((92.68, 0.53), (94.91, 0.35), (2.23, 0.47)),
        ("aged", 0.1): ((92.60, 0.59), (94.87, 0.37), (2.26, 0.53)),
        ("aged", 1): ((92.75, 0.54), (94.87, 0.35), (2.11, 0.49)),
        ("cycle", 0.05): ((92.40, 0.56), (94.84, 0.40), (2.44, 0.54)),
        ("cycle", 0.1): ((92.61, 0.61), (94.77, 0.41), (2.16, 0.60)),
        ("cycle", 0.25): ((92.60, 0.40), (94.62, 0.31), (2.02, 0.32)),
        ("cycle", 0.5): ((91.76, 0.87), (94.41, 0.44), (2.65, 0.76)),
        ("cycle", 1): ((90.41, 0.38), (93.28, 0.61), (2.87, 0.51)),
        "after": ((92.39, 0.39), (94.58, 0.44), (2.19, 0.39)),
        ("fresh", 0.1): ((92.46, 0.61), (94.78, 0.44), (2.32, 0.57)),
        ("fresh", 0.5): ((92.20, 0.57), (94.22, 0.54), (2.02, 0.43)),
        ("fresh", 1): ((90.75, 1.19), (93.61, 0.71), (2.85, 0.81)),
        ("held", 0.1): ((92.27, 0.63), (94.63, 0.53), (2.36, 0.65)),
        ("held", 0.5): ((91.98, 0.61), (93.99, 0.69), (2.01, 0.49)),
        ("held", 1): ((90.48, 1.23), (93.33, 0.87), (2.85, 0.89)),
        ("heldcycle", 0.1): ((92.46, 0.65), (94.58, 0.54), (2.12, 0.72)),
        ("heldcycle", 0.5): ((91.45, 0.86), (94.25, 0.52), (2.80, 0.82)),
        ("heldcycle", 1): ((90.35, 0.34), (92.99, 0.67), (2.64, 0.61)),
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


def less(w, kind, row, than):
    """What a row costs a kind of network over another row: how often right in the
    other, less in the row, network by network."""
    return stat([b - x for b, x in zip(RUN[w][kind][than], RUN[w][kind][row])])


def adds(w, kind, row):
    """What a row adds to version 2 as budgeted, network by network."""
    return less(w, kind, row, "budget")


def interval_adds(w, kind, h):
    """What an interval adds to a calibrated tile: the end of a cycle, less the tile
    calibrated as it was written."""
    return less(w, kind, ("cycle", h), "cal")


def from_written(w, kind, h, held=False):
    """The end of a cycle, less the end of the same interval from weights as written."""
    return less(w, kind, ("heldcycle", h), ("held", h)) if held else less(w, kind, ("cycle", h), ("fresh", h))


def adds_less(w, row):
    """What a row adds to a reference network, less what it adds to the one trained
    before of the same seed."""
    return stat([(rb - r) - (bb - b) for rb, r, bb, b in zip(
        RUN[w][REFERENCE]["budget"], RUN[w][REFERENCE][row], RUN[w][BEFORE]["budget"], RUN[w][BEFORE][row])])


def ahead(w, row):
    """How much more often a reference network is right in a row than the one
    trained before of the same seed."""
    return stat([r - b for r, b in zip(RUN[w][REFERENCE][row], RUN[w][BEFORE][row])])


def holds(w, kind, budget=WITHIN, exact=False, over="budget"):
    """How long a calibration holds in the cycle: the longest interval run whose
    cycle ends under the budget, every shorter one doing so too.  Over the tile as
    budgeted, as the plan's rule has been, or over the calibrated tile.  Read on
    means to the hundredth, or exactly.  None if the shortest is already over."""
    ok = None
    for h in CYCLES:
        x = less(w, kind, ("cycle", h), over)[0]
        if (x if exact else round(x, 2)) >= budget - 1e-9:
            break
        ok = h
    return ok


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


def hold_rows(over="budget"):
    """Section 4's cells: how long a calibration holds in the cycle, a data set's two kinds."""
    return [interval(holds(w, k, over=over)) for w in WORKLOADS for k in KINDS]


def both(f):
    """A row of a table: a figure for each data set's two kinds."""
    return "".join(f"{dpm(f(w, k)):>16}" for w in WORKLOADS for k in KINDS)


def main():
    print("The operating cycle: an interval that starts from a calibration.")
    print(f"The working tile, {laser.name(TILE)} on two buses, at version 2.  Five networks a kind, mean and standard error.")
    heads = (f"  {'':<34}" + "".join(f"{w:>32}" for w in WORKLOADS) + "\n"
             + f"  {'':<34}" + "".join(f"{('before' if k == BEFORE else 'reference'):>16}" for w in WORKLOADS for k in KINDS))

    section("1. Points lost against a network's own accuracy on its host, and what a row adds to its budget")
    for w in WORKLOADS:
        print(f"  {w:<34}{'trained before':>16}{'and adds':>16}{'the reference':>16}{'and adds':>16}")
        for key, name in ROWS:
            print(f"    {name:<32}" + "".join(f"{pm(lost(w, k, key)):>16}{dpm(adds(w, k, key)):>16}" for k in KINDS))

    section("2. What a calibration costs: a calibrated tile, less the tile as budgeted")
    print(heads)
    print(f"    {'calibrated as written':<32}" + both(lambda w, k: adds(w, k, "cal")))
    for h in (0.1, 1):
        print(f"    {'aged ' + minutes(h) + ' first':<32}" + both(lambda w, k: adds(w, k, ("aged", h))))
    for h in (0.1, 1):
        print(f"    {'  and what ' + minutes(h) + ' adds':<32}" + both(lambda w, k: less(w, k, ("aged", h), "cal")))

    section("3. The end of an interval that started from a calibration: what it is over the budget by")
    print(heads)
    for h in CYCLES:
        print(f"    {'a cycle of ' + minutes(h):<32}" + both(lambda w, k: adds(w, k, ("cycle", h))))
    print("  and of that, the interval's own: the end of a cycle, less the tile calibrated as written")
    for h in CYCLES:
        print(f"    {'a cycle of ' + minutes(h):<32}" + both(lambda w, k: interval_adds(w, k, h)))
    print("  and the end of a cycle, less the end of the same interval from weights as written")
    for h in HELD:
        print(f"    {minutes(h):<32}" + both(lambda w, k: from_written(w, k, h)))

    section("4. How long a calibration holds in the cycle: the longest whose end is under a tenth of a point")
    print(heads)
    print(f"    {'over the budget':<32}" + "".join(f"{c:>16}" for c in hold_rows()))
    print(f"    {'over the calibrated tile':<32}" + "".join(f"{c:>16}" for c in hold_rows("cal")))
    print(f"    {'from weights as written':<32}" + "".join(f"{c:>16}" for c in drift.hold_rows()))

    section("5. What came before a calibration: an hour, calibrated and 6 minutes on")
    print(heads)
    print(f"    {'less the 6-minute cycle':<32}" + both(lambda w, k: less(w, k, "after", ("cycle", B15_H))))
    print(f"    {'its 6 minutes add':<32}" + both(lambda w, k: less(w, k, "after", ("aged", 1))))
    print(f"    {'and the 6-minute cycle, its own':<32}" + both(lambda w, k: less(w, k, ("cycle", B15_H), ("aged", B15_H))))

    section("6. Held, with B16's source at the end of a cycle")
    print(heads)
    for h in HELD:
        print(f"    {'right, ' + minutes(h) + ', percent':<32}" + "".join(f"{acc(w, k, ('heldcycle', h))[0]:>16.2f}" for w in WORKLOADS for k in KINDS))
    for h in HELD:
        print(f"    {'less held as written, ' + minutes(h):<32}" + both(lambda w, k: from_written(w, k, h, held=True)))
    for h in HELD:
        print(f"    {'what holding adds, ' + minutes(h):<32}" + both(lambda w, k: less(w, k, ("heldcycle", h), ("cycle", h))))

    section("7. The reference network less the one trained before, seed by seed: how often right, and what a row adds")
    print(f"  {'':<34}" + "".join(f"{w:>18}{'and adds':>16}" for w in WORKLOADS))
    for key, name in ROWS:
        print(f"    {name:<32}" + "".join(f"{dpm(ahead(w, key)):>18}{dpm(adds_less(w, key)):>16}" for w in WORKLOADS))

    print()
    print("CORRECTED by pta_reference_calibration.py, 2026-10-08.  A calibrated row and one that is not do")
    print("not meet the same draws, so a calibrated row less the row as budgeted is the trims and the")
    print("draws both.  Readings 1, 2, 3 and 7 below took that for the calibration's cost, and it was")
    print("the draws.  The figures stand, and so does what is read between two calibrated rows.  A")
    print("cycle over its budget, in readings 2, 3 and 5, is read there over its own draws.")
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        findings()
    print(said.getvalue(), end="")
    checks(" ".join(said.getvalue().split()))


def findings():
    m, f, i = WORKLOADS
    R, B = REFERENCE, BEFORE
    six, three = ("cycle", B15_H), ("cycle", 0.05)
    print()
    print("What this says, seven readings.")
    print()
    print("  1. THE SHORTFALL IS THE CALIBRATION'S OWN, AND NOT THE HOUR'S.  A tile calibrated as it was")
    print(f"     written, with no drift at all, leaves the reference networks {dpm(adds(m, R, 'cal'))}, {dpm(adds(f, R, 'cal'))}")
    print(f"     and {dpm(adds(i, R, 'cal'))} from their budget, and the old ones {dpm(adds(m, B, 'cal'))}, {dpm(adds(f, B, 'cal'))} and")
    print(f"     {dpm(adds(i, B, 'cal'))}.  MNIST's is clear, at {errors(adds(m, R, 'cal')):.1f} of its errors: each of the five loses, {round(100 * min(b - x for b, x in zip(RUN[m][R]['budget'], RUN[m][R]['cal'])))} to {round(100 * max(b - x for b, x in zip(RUN[m][R]['budget'], RUN[m][R]['cal'])))}")
    print(f"     images in ten thousand.  An hour of drift before the calibration adds {dpm(less(m, R, ('aged', 1), 'cal'))},")
    print(f"     {dpm(less(f, R, ('aged', 1), 'cal'))} and {dpm(less(i, R, ('aged', 1), 'cal'))} to that.  On the first two sets it is the size of a row of")
    print("     the budget, and no row of the budget carries it.")
    print()
    print("  2. AT THE END OF A SIX-MINUTE CYCLE THE REFERENCE NETWORKS ARE WITHIN A TENTH ON TWO SETS AND")
    print(f"     OVER IT ON FASHION-MNIST: {dpm(adds(m, R, six))}, {dpm(adds(f, R, six))} and {dpm(adds(i, R, six))} over their budget, where")
    print(f"     six minutes from weights as written left them {adds(m, R, ('fresh', 0.1))[0]:.2f}, {adds(f, R, ('fresh', 0.1))[0]:.2f} and {adds(i, R, ('fresh', 0.1))[0]:.2f} over.  None of the three")
    print(f"     is clear.  The six minutes themselves add {dpm(interval_adds(m, R, B15_H))}, {dpm(interval_adds(f, R, B15_H))} and {dpm(interval_adds(i, R, B15_H))}")
    print("     to the calibrated tile: on MNIST and Fashion-MNIST what the cycle ends over by is the")
    print("     calibration's, and on the inverted set it is the interval's.")
    print()
    print("  3. A SHORTER INTERVAL DOES NOT BUY IT BACK.  A three-minute cycle ends")
    print(f"     {dpm(adds(m, R, three))}, {dpm(adds(f, R, three))} and {dpm(adds(i, R, three))} over: on MNIST and Fashion-MNIST, where six")
    print("     minutes' ends over by the calibration's, no better.  By the rule, read over the budget as")
    print(f"     it has been, a calibration holds {interval(holds(m, R))} on MNIST and {interval(holds(i, R))} on the inverted set for the")
    print("     reference networks in the cycle, and on Fashion-MNIST no interval that was run, where from")
    print(f"     weights as written it held {interval(drift.holds(m, R))}, {interval(drift.holds(f, R))} and {interval(drift.holds(i, R))}.  Fashion-MNIST's is lost to")
    print("     the calibration and not to the interval.  And the rule there is on a mean five networks")
    print(f"     do not fix: the half-hour cycle ends {dpm(adds(f, R, ('cycle', 0.5)))} over.")
    print()
    pairs = [from_written(w, k, h) for w in WORKLOADS for k in KINDS for h in HELD]
    print("  4. A CYCLE ENDS WHERE AN INTERVAL FROM WEIGHTS AS WRITTEN DOES, TO WHAT FIVE NETWORKS TELL.  At")
    print(f"     six minutes a cycle's end is {dpm(from_written(m, R, 0.1))}, {dpm(from_written(f, R, 0.1))} and {dpm(from_written(i, R, 0.1))} worse for the")
    print(f"     reference networks than the same interval from weights as written; at half an hour")
    print(f"     {dpm(from_written(m, R, 0.5))}, {dpm(from_written(f, R, 0.5))} and {dpm(from_written(i, R, 0.5))}; at an hour {dpm(from_written(m, R, 1))}, {dpm(from_written(f, R, 1))}")
    print(f"     and {dpm(from_written(i, R, 1))}.  None of the {len(pairs)} differences, both kinds, is clear.  The plan's drift")
    print("     rows stand as the ends of intervals; MNIST's three for the reference networks are all")
    print("     over, by the calibration's eight hundredths or less.")
    print()
    print("  5. WHAT A TILE WAS BEFORE ITS CALIBRATION IS NOT SEEN TO MATTER, AND WAS NOT PINNED.  Aged an")
    print("     hour, calibrated and six minutes on, less the six-minute cycle: the old networks")
    print(f"     {dpm(less(m, B, 'after', six))}, {dpm(less(f, B, 'after', six))} and {dpm(less(i, B, 'after', six))}; the reference {dpm(less(m, R, 'after', six))},")
    print(f"     {dpm(less(f, R, 'after', six))} and {dpm(less(i, R, 'after', six))}.  Three of the six are over a tenth either way and none is")
    print("     clear.  In grx930's drift a cell walks at random and what it walked before does not")
    print("     enter what it walks after, so the two rows are two draws of the same six minutes, and")
    print("     that is how far apart two draws are.  On the inverted set the reference networks' six")
    print(f"     minutes is now drawn three times: {dpm(adds(i, R, ('fresh', 0.1)))} from weights as written, {dpm(adds(i, R, six))} at the")
    print(f"     end of the cycle, and {dpm(adds(i, R, 'after'))} after the hour's calibration.  A tenth of a point is not")
    print("     pinned between them.")
    print()
    print("  6. HELD AT THE END OF A SIX-MINUTE CYCLE, THE REFERENCE NETWORKS ARE WHERE THE PLAN HAS THEM.")
    print(f"     Right {acc(m, R, ('heldcycle', 0.1))[0]:.2f}, {acc(f, R, ('heldcycle', 0.1))[0]:.2f} and {acc(i, R, ('heldcycle', 0.1))[0]:.2f}% of the time, where held from weights as written they")
    print(f"     were {acc(m, R, ('held', 0.1))[0]:.2f}, {acc(f, R, ('held', 0.1))[0]:.2f} and {acc(i, R, ('held', 0.1))[0]:.2f}: {dpm(from_written(m, R, 0.1, held=True))}, {dpm(from_written(f, R, 0.1, held=True))} and {dpm(from_written(i, R, 0.1, held=True))} worse.")
    print(f"     And ahead of the old networks held the same way by {dpm(ahead(m, ('heldcycle', 0.1)))}, {dpm(ahead(f, ('heldcycle', 0.1)))}")
    print(f"     and {dpm(ahead(i, ('heldcycle', 0.1)))}.  In every one of the {len(ROWS)} rows, on every set, they are ahead in the mean, and")
    print(f"     clear of chance in {sum(clear(ahead(w, key)) for w in WORKLOADS for key, _ in ROWS)} of the {len(WORKLOADS) * len(ROWS)}.")
    print()
    print("  7. THE CALIBRATION TAKES A THIRD OF THE REFERENCE NETWORKS' LEAD ON MNIST.  As budgeted they are")
    print(f"     {dpm(ahead(m, 'budget'))} ahead there, and calibrated {dpm(ahead(m, 'cal'))}: what a calibration adds to them, less")
    print(f"     to the old ones, is {dpm(adds_less(m, 'cal'))}, clear at {errors(adds_less(m, 'cal')):.1f} of its errors.  On Fashion-MNIST and the")
    print(f"     inverted set it is {dpm(adds_less(f, 'cal'))} and {dpm(adds_less(i, 'cal'))}.  Why a calibration costs a network trained")
    print("     for the tile and not one that was not is not shown here.")


def checks(said):
    """Every claim above, as an assert.  `said` is the readings as printed, on one line."""
    m, f, i = WORKLOADS
    R, B = REFERENCE, BEFORE
    keys = [k for k, _ in ROWS]
    six, three = ("cycle", B15_H), ("cycle", 0.05)
    assert TILE == (128, 64) and WITHIN == 0.10 and B15_H == 0.1 and len(keys) == 19 and PROBES == 16
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
            got = (acc(w, B, row), acc(w, R, row), ahead(w, row))
            assert len(PRINTED_RIGHT[w][row]) == 3 and all(same(a, b) for a, b in zip(got, PRINTED_RIGHT[w][row])), (w, row)
            cells += 7
    assert cells == 3 * 19 * 7 == 399
    #    Counted and summed, as grx930's lines gave them.
    figures = [x for w in RUN.values() for k in w.values() for row in k.values() for x in row]
    assert (len(figures), round(sum(figures), 2)) == (570, 52734.18)
    hosts = [x for w in HOSTS.values() for k in w.values() for x in k]
    assert (len(hosts), round(sum(hosts), 2)) == (30, 2796.55)
    printed = [x for t in (PRINTED_LOST, PRINTED_RIGHT) for w in t.values() for row in w.values() for cell in row for x in cell]
    assert (len(printed), round(sum(printed), 2)) == (798, 10808.38)
    #    The networks are pta_reference_drift.py's, seed for seed, in the eight rows
    #    the two sweeps share: grx930 found those lines the same, byte for byte.
    assert drift.TILE == TILE and drift.WORKLOADS == WORKLOADS and (drift.BEFORE, drift.REFERENCE) == (B, R)
    there = {"budget": "budget", ("aged", 1): "cal"}
    there.update({("fresh", h): ("drift", h) for h in HELD})
    there.update({("held", h): ("held", h) for h in HELD})
    assert len(there) == 8 and HELD == drift.HELD and set(CYCLES) <= set(drift.HOURS)
    for w in WORKLOADS:
        for k in KINDS:
            assert HOSTS[w][k] == drift.HOSTS[w][k], (w, k)
            for here, key in there.items():
                assert RUN[w][k][here] == drift.RUN[w][k][key], (w, k, here)
    #    And so pta_trained.py's reference row, held at six minutes.
    assert [r2(lost(w, R, ("held", 0.1))) for w in WORKLOADS] == [r2(trained.lost(w, trained.REFERENCE, trained.V2_HELD)) for w in WORKLOADS]
    assert version2.INTERVAL_S == 360 and version2.ROWS_V2 == (0.01, 0.05, 0.05) and abs(T95 - 2.776) < 1e-9

    # 1. Reading 1.  The calibration, by itself.
    assert [r2(adds(w, R, "cal")) for w in WORKLOADS] == [(0.08, 0.01), (0.15, 0.12), (-0.05, 0.04)]
    assert [r2(adds(w, B, "cal")) for w in WORKLOADS] == [(-0.05, 0.04), (0.03, 0.11), (0.06, 0.06)]
    assert [(w, k) for w in WORKLOADS for k in KINDS if clear(adds(w, k, "cal"))] == [(m, R)]
    assert round(errors(adds(m, R, "cal")), 1) == 7.8
    assert sorted(round(100 * (b - x)) for b, x in zip(RUN[m][R]["budget"], RUN[m][R]["cal"])) == [5, 7, 9, 10, 11]
    assert all(b > x for b, x in zip(RUN[m][R]["budget"], RUN[m][R]["cal"]))
    #    What drift before the calibration adds to it: an hour, and six minutes.
    assert [r2(less(w, R, ("aged", 1), "cal")) for w in WORKLOADS] == [(0.01, 0.01), (-0.02, 0.04), (0.04, 0.02)]
    assert [r2(less(w, R, ("aged", 0.1), "cal")) for w in WORKLOADS] == [(0.01, 0.01), (-0.02, 0.01), (0.05, 0.04)]
    assert [r2(less(w, B, ("aged", 1), "cal")) for w in WORKLOADS] == [(-0.01, 0.02), (-0.01, 0.04), (-0.07, 0.02)]
    assert [r2(less(w, B, ("aged", 0.1), "cal")) for w in WORKLOADS] == [(0.01, 0.02), (-0.01, 0.03), (0.08, 0.07)]
    #    Of those twelve, one is clear, and it is a gain: the old networks on the
    #    inverted set, an hour first.  One in twelve is what chance gives.
    assert [(w, k, h) for w in WORKLOADS for k in KINDS for h in (0.1, 1) if clear(less(w, k, ("aged", h), "cal"))] == [(i, B, 1)]
    assert all(abs(less(w, k, ("aged", h), "cal")[0]) < 0.085 for w in WORKLOADS for k in KINDS for h in (0.1, 1))
    #    The same row as pta_reference_drift.py's reading 5, and its figures.
    assert [r2(adds(w, R, ("aged", 1))) for w in WORKLOADS] == [(0.09, 0.02), (0.13, 0.10), (-0.01, 0.05)]
    #    As large as a row of the budget: within three hundredths of a tenth on MNIST,
    #    and over it on Fashion-MNIST.
    assert abs(adds(m, R, "cal")[0] - WITHIN) < 0.03 and adds(f, R, "cal")[0] > WITHIN

    # 2. Reading 2.  The end of a six-minute cycle.
    assert [r2(adds(w, R, six)) for w in WORKLOADS] == [(0.08, 0.04), (0.16, 0.08), (0.09, 0.08)]
    assert [r2(adds(w, B, six)) for w in WORKLOADS] == [(-0.05, 0.04), (0.16, 0.06), (0.14, 0.25)]
    assert [round(adds(w, R, ("fresh", 0.1))[0], 2) for w in WORKLOADS] == [0.01, 0.07, 0.08]
    assert [w for w in WORKLOADS if round(adds(w, R, six)[0], 2) < WITHIN] == [m, i]
    assert not any(clear(adds(w, R, six)) for w in WORKLOADS)
    assert [round(at_most(adds(w, R, six)), 2) for w in WORKLOADS] == [0.19, 0.39, 0.32]
    assert [r2(interval_adds(w, R, B15_H)) for w in WORKLOADS] == [(0.00, 0.04), (0.02, 0.04), (0.14, 0.09)]
    assert [r2(interval_adds(w, B, B15_H)) for w in WORKLOADS] == [(0.00, 0.02), (0.13, 0.08), (0.08, 0.20)]
    #    The calibration's and the interval's, which is the larger.
    assert all(adds(w, R, "cal")[0] > 4 * abs(interval_adds(w, R, B15_H)[0]) for w in (m, f))
    assert adds(i, R, "cal")[0] < 0 < interval_adds(i, R, B15_H)[0]
    #    Against the start of the same interval, a tile aged six minutes and calibrated.
    assert [r2(less(w, R, six, ("aged", B15_H))) for w in WORKLOADS] == [(-0.01, 0.04), (0.03, 0.05), (0.10, 0.10)]

    # 3. Reading 3.  A shorter interval, and how long a calibration holds.
    assert [r2(adds(w, R, three)) for w in WORKLOADS] == [(0.07, 0.03), (0.14, 0.10), (0.02, 0.07)]
    assert all(abs(adds(w, R, three)[0] - adds(w, R, six)[0]) < 0.05 for w in (m, f))
    assert all(clear(adds(w, R, "cal")) or adds(w, R, "cal")[0] > WITHIN for w in (m, f)) and holds(f, R) is None
    assert [holds(w, R) for w in WORKLOADS] == [0.25, None, 0.1]
    assert [holds(w, B) for w in WORKLOADS] == [1, 0.05, None]
    assert hold_rows() == ['an hour', '15 minutes', '3 minutes', 'none run', 'none run', '6 minutes']
    assert hold_rows("cal") == ['30 minutes', 'an hour', '3 minutes', '30 minutes', 'none run', '3 minutes']
    assert [drift.holds(w, R) for w in WORKLOADS] == [0.25, 0.25, 0.1]
    #    Read exactly and not to the hundredth, none of the six answers moves.
    assert all(holds(w, k, exact=True) == holds(w, k) for w in WORKLOADS for k in KINDS)
    assert r2(adds(f, R, ("cycle", 0.5))) == (0.05, 0.10)
    assert round(adds(f, R, ("cycle", 0.5))[0], 2) < WITHIN < min(round(adds(f, R, ("cycle", h))[0], 2) for h in CYCLES if h != 0.5)
    #    Which cells are within three hundredths of the tenth, either side.
    near = [(w, k, h) for w in WORKLOADS for k in KINDS for h in CYCLES if abs(adds(w, k, ("cycle", h))[0] - WITHIN) < 0.03]
    assert near == [(m, B, 1), (m, R, 0.05), (m, R, 0.1), (m, R, 0.25), (f, B, 0.25), (i, R, 0.1)], near
    #    Every cycle the reference networks ran, over the budget.
    assert [round(adds(w, R, ("cycle", h))[0], 2) for w in WORKLOADS for h in CYCLES] == [0.07, 0.08, 0.08, 0.14, 0.14, 0.14, 0.16, 0.23, 0.05, 0.38, 0.02, 0.09, 0.25, 0.45, 1.58]

    # 4. Reading 4.  A cycle against the same interval from weights as written.
    assert [r2(from_written(w, R, h)) for h in HELD for w in WORKLOADS] == [(0.08, 0.04), (0.09, 0.10), (0.02, 0.04), (0.04, 0.04), (-0.13, 0.10), (-0.19, 0.28), (0.07, 0.04), (0.06, 0.12), (0.33, 0.68)]
    assert [r2(from_written(w, B, h)) for h in HELD for w in WORKLOADS] == [(0.02, 0.01), (0.08, 0.04), (-0.14, 0.16), (-0.06, 0.05), (0.10, 0.14), (0.44, 0.69), (-0.05, 0.10), (-0.17, 0.27), (0.34, 1.28)]
    pairs = [from_written(w, k, h) for w in WORKLOADS for k in KINDS for h in HELD]
    assert len(pairs) == 18 and not any(clear(x) for x in pairs)
    assert all(0 < from_written(m, R, h)[0] < adds(m, R, "cal")[0] for h in HELD)

    # 5. Reading 5.  What came before.
    assert [r2(less(w, B, "after", six)) for w in WORKLOADS] == [(0.01, 0.04), (-0.20, 0.10), (0.21, 0.29)]
    assert [r2(less(w, R, "after", six)) for w in WORKLOADS] == [(0.01, 0.03), (-0.06, 0.08), (0.18, 0.10)]
    after = {(w, k): less(w, k, "after", six) for w in WORKLOADS for k in KINDS}
    assert [wk for wk, x in after.items() if abs(x[0]) > WITHIN] == [(f, B), (i, B), (i, R)]
    assert not any(clear(x) for x in after.values())
    #    The six minutes after an hour and a calibration, and after six minutes and one.
    assert [r2(less(w, R, "after", ("aged", 1))) for w in WORKLOADS] == [(0.00, 0.03), (-0.02, 0.10), (0.28, 0.09)]
    assert [wk for wk in after if clear(less(wk[0], wk[1], "after", ("aged", 1)))] == [(i, R)]
    #    The inverted set's six minutes for the reference networks, three draws of it.
    assert [r2(adds(i, R, row)) for row in (("fresh", 0.1), six, "after")] == [(0.08, 0.11), (0.09, 0.08), (0.28, 0.10)]
    assert not any(clear(adds(i, R, row)) for row in (("fresh", 0.1), six, "after"))
    assert [round(adds(i, R, row)[0], 2) < WITHIN for row in (("fresh", 0.1), six, "after")] == [True, True, False]
    assert [r2(adds(i, B, row)) for row in (("fresh", 0.1), six, "after")] == [(0.28, 0.27), (0.14, 0.25), (0.35, 0.24)]

    # 6. Reading 6.  Held.
    assert [round(acc(w, R, ("heldcycle", 0.1))[0], 2) for w in WORKLOADS] == [97.57, 87.30, 94.58]
    assert [round(acc(w, R, ("held", 0.1))[0], 2) for w in WORKLOADS] == [97.64, 87.27, 94.63]
    assert [r2(from_written(w, R, 0.1, held=True)) for w in WORKLOADS] == [(0.06, 0.05), (-0.03, 0.11), (0.05, 0.02)]
    assert not any(clear(from_written(w, k, h, held=True)) for w in WORKLOADS for k in KINDS for h in HELD)
    assert [r2(ahead(w, ("heldcycle", 0.1))) for w in WORKLOADS] == [(0.30, 0.12), (0.48, 0.15), (2.12, 0.72)]
    assert [w for w in WORKLOADS if clear(ahead(w, ("heldcycle", 0.1)))] == [f, i]
    assert all(ahead(w, row)[0] > 0 for w in WORKLOADS for row in keys)
    assert [round(min(ahead(w, row)[0] for row in keys), 2) for w in WORKLOADS] == [0.22, 0.33, 2.01]
    assert sum(clear(ahead(w, row)) for w in WORKLOADS for row in keys) == 38
    #    What they lose so held, at the end of a cycle: six minutes, half an hour, an hour.
    assert [round(lost(w, R, ("heldcycle", h))[0], 2) for h in HELD for w in WORKLOADS] == [0.18, 0.59, 0.80, 0.19, 0.73, 1.14, 0.24, 0.80, 2.39]
    #    What holding adds at the end of a six-minute cycle.
    assert [r2(less(w, R, ("heldcycle", 0.1), six)) for w in WORKLOADS] == [(0.02, 0.02), (-0.05, 0.03), (0.19, 0.15)]

    # 7. Reading 7.  The lead.
    assert [r2(ahead(m, row)) for row in ("budget", "cal")] == [(0.38, 0.11), (0.25, 0.10)]
    assert [r2(adds_less(w, "cal")) for w in WORKLOADS] == [(0.13, 0.04), (0.12, 0.15), (-0.11, 0.06)]
    assert [w for w in WORKLOADS if clear(adds_less(w, "cal"))] == [m]
    assert round(errors(adds_less(m, "cal")), 1) == 3.1
    assert 0.25 < adds_less(m, "cal")[0] / ahead(m, "budget")[0] < 0.42

    # 8. And the readings say those figures, each in its place.
    for words in (
        "leaves the reference networks +0.08 +-0.01, +0.15 +-0.12 and -0.05 +-0.04 from their budget, and the old ones -0.05 +-0.04, +0.03 +-0.11 and +0.06 +-0.06.",
        "MNIST's is clear, at 7.8 of its errors: each of the five loses, 5 to 11 images in ten thousand.",
        "An hour of drift before the calibration adds +0.01 +-0.01, -0.02 +-0.04 and +0.04 +-0.02 to that.",
        "+0.08 +-0.04, +0.16 +-0.08 and +0.09 +-0.08 over their budget, where six minutes from weights as written left them 0.01, 0.07 and 0.08 over.",
        "The six minutes themselves add +0.00 +-0.04, +0.02 +-0.04 and +0.14 +-0.09 to the calibrated tile",
        "A three-minute cycle ends +0.07 +-0.03, +0.14 +-0.10 and +0.02 +-0.07 over",
        "a calibration holds 15 minutes on MNIST and 6 minutes on the inverted set for the reference networks in the cycle",
        "weights as written it held 15 minutes, 15 minutes and 6 minutes.",
        "the half-hour cycle ends +0.05 +-0.10 over.",
        "At six minutes a cycle's end is +0.08 +-0.04, +0.09 +-0.10 and +0.02 +-0.04 worse for the reference networks",
        "at half an hour +0.04 +-0.04, -0.13 +-0.10 and -0.19 +-0.28; at an hour +0.07 +-0.04, +0.06 +-0.12 and +0.33 +-0.68.",
        "None of the 18 differences, both kinds, is clear.",
        "the old networks +0.01 +-0.04, -0.20 +-0.10 and +0.21 +-0.29; the reference +0.01 +-0.03, -0.06 +-0.08 and +0.18 +-0.10.",
        "drawn three times: +0.08 +-0.11 from weights as written, +0.09 +-0.08 at the end of the cycle, and +0.28 +-0.10 after the hour's calibration.",
        "Right 97.57, 87.30 and 94.58% of the time, where held from weights as written they were 97.64, 87.27 and 94.63: +0.06 +-0.05, -0.03 +-0.11 and +0.05 +-0.02 worse.",
        "ahead of the old networks held the same way by +0.30 +-0.12, +0.48 +-0.15 and +2.12 +-0.72.",
        "In every one of the 19 rows, on every set, they are ahead in the mean, and clear of chance in 38 of the 57.",
        "As budgeted they are +0.38 +-0.11 ahead there, and calibrated +0.25 +-0.10",
        "is +0.13 +-0.04, clear at 3.1 of its errors.",
        "inverted set it is +0.12 +-0.15 and -0.11 +-0.06.",
    ):
        assert said.count(words) == 1, words

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
