"""
What a calibration costs: its trims, and its draws.

pta_reference_cycle.py read a calibration's cost off two rows of grx930's
harness, a tile calibrated as it was written and the tile as budgeted: 0.08
+-0.01 of a point on MNIST for B17's reference networks, outside chance at one
in twenty by a long way, and called the calibration's own.  The plan asked what
in the calibration it was before making it a row of the budget.

The two runs differ in two things, and that model took them for one.  A
calibration writes trims: with no drift a cell's trim is minus the mean of the
programming errors its probes happened to meet, about a quarter of an LSB, and
nothing the cell will meet again.  And a calibration's probes are GEMMs, each
of which takes the run's next seed, so every image of a calibrated run meets
other noise and other programming errors than it does in the run as budgeted.
grx930's harness has now taken the two apart.

MEASURED IN A MODEL, by grx930 (c930/doc/pta_error_model_design_note.md section
5, "What a calibration costs: its trims, and its draws", 2026-10-08;
`sim/pta_mnist.sh DIR WORK refcal`).  The working tile, 128 x 64, at version 2,
with no drift anywhere.  Two kinds of network a data set, five of each, seeds 1
to 5: the ones trained before, and B17's reference.  Twenty-two rows a network:

  as budgeted       version 2 and nothing else, the tile seeded with the
                    network's own seed: the plan's row
  calibrated        C3's cell calibration at 1, 4, 16 and 64 probes a cell
  probes only       the same probes taken and no trim written (a trim that can
                    hold nothing): the calibrated run's draws, on the tile as
                    budgeted
  a finer step      16 probes with a trim step of a sixteenth and a 256th of a
                    weight's LSB, and 64 probes at a 256th
  another draw      the tile seeded with the network's seed and 10 D more: as
                    budgeted on six of them, and calibrated and probes only at
                    16 probes on the first two

DERIVED here:

  a row's mean and    which have to be the four tables grx930 printed, cell for
  its error           cell
  what the trims add  how often right with the probes taken and nothing written,
                      less calibrated, network by network: the two runs meet
                      the same draws and differ in the trims and nothing else
  what the draws do   the tile as budgeted on twelve other draws: how far a
                      network moves from one to the next with nothing changed,
                      and where the as-budgeted row sits among them
  what the tile       a network's accuracy on its host, less over all thirteen
  costs, over draws   draws of the tile as budgeted
  the cycle's rows,   pta_reference_cycle.py's calibrated rows less the row here
  on their own draws  that has their draws and no trims, in place of the
                      as-budgeted row, which has neither

HOW SURE.  Five networks, 2.8 errors for one in twenty, as before.  That test
was not wrong about the 0.08: one in twenty is wrong once in twenty, and this
plan has read some hundreds of such differences.  What was wrong was to take
three rows that share their draws for three findings.

WHAT THIS IS NOT.  A calibration that corrects drift: nothing here drifts, and
what a calibration is worth when there is drift to take out is the cycle's
rows.  Another calibration than C3's as grx930's harness has it.  A schedule.
Version 2 and the working tile only.

Standard library only.  Run:  python3 docs/designs/pta_reference_calibration.py
"""
import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_laser as laser
import pta_reference_cycle as cycle
import pta_trained as trained
import pta_workload as workload

MNIST, FASHION, INVERTED = workload.MNIST, workload.FASHION, workload.INVERTED
WORKLOADS = workload.WORKLOADS
TILE = laser.SMALLER                           # 128 x 64: B10
BEFORE, REFERENCE = "before", "reference"      # trained as the plan's networks were; B17's
KINDS = (BEFORE, REFERENCE)
SEEDS = 5
PROBES = (1, 4, 16, 64)                        # the probes a cell that were calibrated with
C3_PROBES = cycle.PROBES                       # 16: what every calibrated row of the plan has had
STEPS = (0.0625, 0.00390625)                   # the finer trim steps, in weight LSB: a sixteenth, a 256th
QUARTER = 0.25                                 # and the trim step every calibrated row of the plan has had
CAL_DRAWS = (1, 2)                             # the other draws that were calibrated on
DRAWS = (1, 2, 3, 4, 5, 6)                     # and all those the tile as budgeted was run on
WITHIN = laser.WITHIN                          # a tenth of a point: what a row of the budget costs
T95 = trained.T95
ROWS = ((("budget", "as budgeted"),)
        + tuple((("cal", n), f"calibrated, {n} probe{'s' if n > 1 else ''}") for n in PROBES)
        + tuple((("only", n), f"probes only, {n} probe{'s' if n > 1 else ''}") for n in PROBES)
        + tuple((("step", x), f"16 probes, a step of 1/{round(1 / x)}") for x in STEPS)
        + (("best", f"64 probes, a step of 1/{round(1 / STEPS[-1])}"),)
        + tuple(row for d in CAL_DRAWS for row in ((("draw", d), f"as budgeted, draw {d}"),
                                                    (("caldraw", d), f"calibrated, draw {d}"),
                                                    (("onlydraw", d), f"probes only, draw {d}")))
        + tuple((("draw", d), f"as budgeted, draw {d}") for d in DRAWS if d not in CAL_DRAWS))
# The tile as budgeted on a draw that is not the plan's: another seed, or the probes taken and nothing written.
OTHERS = (tuple(("only", n) for n in PROBES) + tuple(("draw", d) for d in DRAWS) + tuple(("onlydraw", d) for d in CAL_DRAWS))
# A calibrated row's partner: the row with its probes, on its draw, and no trim written.
PARTNER = dict([(("cal", n), ("only", n)) for n in PROBES] + [(("step", x), ("only", C3_PROBES)) for x in STEPS]
               + [("best", ("only", PROBES[-1]))] + [(("caldraw", d), ("onlydraw", d)) for d in CAL_DRAWS])

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
            ("cal", 1): (97.44, 97.62, 97.19, 97.22, 97.11),
            ("cal", 4): (97.42, 97.70, 97.12, 97.41, 97.22),
            ("cal", 16): (97.43, 97.64, 97.16, 97.28, 97.25),
            ("cal", 64): (97.34, 97.67, 96.95, 97.33, 97.08),
            ("only", 1): (97.46, 97.58, 97.06, 97.19, 97.16),
            ("only", 4): (97.44, 97.73, 97.05, 97.36, 97.13),
            ("only", 16): (97.47, 97.67, 97.21, 97.33, 97.17),
            ("only", 64): (97.30, 97.65, 96.98, 97.32, 97.12),
            ("step", 0.0625): (97.44, 97.63, 97.13, 97.32, 97.22),
            ("step", 0.00390625): (97.47, 97.63, 97.15, 97.29, 97.24),
            "best": (97.33, 97.62, 97.01, 97.30, 97.06),
            ("draw", 1): (97.43, 97.58, 97.16, 97.29, 97.07),
            ("caldraw", 1): (97.47, 97.64, 97.03, 97.33, 97.09),
            ("onlydraw", 1): (97.45, 97.61, 97.00, 97.29, 97.17),
            ("draw", 2): (97.45, 97.73, 97.13, 97.29, 97.17),
            ("caldraw", 2): (97.45, 97.64, 97.08, 97.31, 97.08),
            ("onlydraw", 2): (97.48, 97.64, 97.15, 97.21, 97.11),
            ("draw", 3): (97.34, 97.70, 97.01, 97.32, 97.15),
            ("draw", 4): (97.42, 97.50, 96.98, 97.30, 97.06),
            ("draw", 5): (97.44, 97.66, 97.09, 97.20, 97.19),
            ("draw", 6): (97.53, 97.71, 97.06, 97.37, 97.14),
        },
        REFERENCE: {
            "budget": (97.61, 97.66, 97.65, 97.84, 97.65),
            ("cal", 1): (97.58, 97.54, 97.49, 97.78, 97.77),
            ("cal", 4): (97.55, 97.59, 97.47, 97.68, 97.64),
            ("cal", 16): (97.51, 97.59, 97.56, 97.73, 97.60),
            ("cal", 64): (97.61, 97.68, 97.40, 97.77, 97.61),
            ("only", 1): (97.59, 97.66, 97.52, 97.80, 97.62),
            ("only", 4): (97.50, 97.62, 97.50, 97.75, 97.63),
            ("only", 16): (97.55, 97.60, 97.52, 97.77, 97.63),
            ("only", 64): (97.61, 97.68, 97.43, 97.73, 97.60),
            ("step", 0.0625): (97.53, 97.61, 97.57, 97.73, 97.60),
            ("step", 0.00390625): (97.52, 97.60, 97.55, 97.73, 97.61),
            "best": (97.56, 97.68, 97.42, 97.78, 97.54),
            ("draw", 1): (97.59, 97.61, 97.51, 97.73, 97.51),
            ("caldraw", 1): (97.61, 97.61, 97.52, 97.76, 97.62),
            ("onlydraw", 1): (97.66, 97.63, 97.44, 97.69, 97.52),
            ("draw", 2): (97.60, 97.64, 97.48, 97.75, 97.52),
            ("caldraw", 2): (97.51, 97.65, 97.53, 97.71, 97.54),
            ("onlydraw", 2): (97.47, 97.66, 97.48, 97.65, 97.56),
            ("draw", 3): (97.52, 97.65, 97.45, 97.73, 97.50),
            ("draw", 4): (97.58, 97.64, 97.50, 97.85, 97.50),
            ("draw", 5): (97.52, 97.70, 97.43, 97.84, 97.61),
            ("draw", 6): (97.61, 97.76, 97.57, 97.68, 97.57),
        },
    },
    FASHION: {
        BEFORE: {
            "budget": (86.45, 86.83, 86.94, 87.65, 86.81),
            ("cal", 1): (86.67, 86.29, 86.34, 87.49, 86.47),
            ("cal", 4): (86.68, 86.62, 86.45, 87.60, 86.66),
            ("cal", 16): (86.72, 86.94, 86.59, 87.46, 86.83),
            ("cal", 64): (86.80, 86.69, 86.59, 87.89, 86.77),
            ("only", 1): (86.69, 86.80, 86.50, 87.53, 86.66),
            ("only", 4): (86.96, 86.62, 86.60, 87.78, 86.67),
            ("only", 16): (86.86, 86.93, 86.67, 87.52, 86.87),
            ("only", 64): (86.72, 86.68, 86.68, 87.80, 86.78),
            ("step", 0.0625): (86.76, 86.95, 86.70, 87.55, 86.82),
            ("step", 0.00390625): (86.81, 86.88, 86.67, 87.60, 86.88),
            "best": (86.77, 86.66, 86.62, 87.91, 86.78),
            ("draw", 1): (86.93, 86.81, 86.56, 87.61, 86.81),
            ("caldraw", 1): (87.00, 86.97, 86.60, 87.65, 87.04),
            ("onlydraw", 1): (87.04, 87.24, 86.68, 87.41, 86.98),
            ("draw", 2): (86.83, 86.91, 86.79, 87.76, 86.93),
            ("caldraw", 2): (86.89, 86.45, 86.80, 87.55, 86.72),
            ("onlydraw", 2): (86.94, 86.54, 86.81, 87.60, 86.83),
            ("draw", 3): (86.92, 87.18, 86.65, 87.40, 86.77),
            ("draw", 4): (86.88, 86.71, 86.54, 87.72, 86.76),
            ("draw", 5): (87.10, 86.94, 86.83, 87.63, 86.83),
            ("draw", 6): (87.00, 86.70, 86.79, 87.86, 86.86),
        },
        REFERENCE: {
            "budget": (87.02, 87.76, 87.63, 87.42, 87.23),
            ("cal", 1): (87.05, 87.60, 87.13, 87.21, 86.59),
            ("cal", 4): (87.14, 87.79, 87.24, 87.63, 87.30),
            ("cal", 16): (86.68, 87.67, 87.13, 87.45, 87.40),
            ("cal", 64): (86.99, 87.68, 87.42, 87.44, 87.33),
            ("only", 1): (86.81, 87.50, 87.52, 87.49, 86.99),
            ("only", 4): (87.04, 87.51, 87.22, 87.52, 87.34),
            ("only", 16): (86.81, 87.69, 87.19, 87.44, 87.23),
            ("only", 64): (86.98, 87.61, 87.25, 87.51, 87.29),
            ("step", 0.0625): (86.75, 87.59, 87.12, 87.41, 87.38),
            ("step", 0.00390625): (86.81, 87.59, 87.12, 87.50, 87.33),
            "best": (87.03, 87.64, 87.44, 87.45, 87.17),
            ("draw", 1): (87.01, 87.83, 87.33, 87.63, 87.15),
            ("caldraw", 1): (86.92, 88.06, 87.30, 87.64, 87.20),
            ("onlydraw", 1): (86.98, 88.08, 87.30, 87.73, 87.06),
            ("draw", 2): (86.95, 87.89, 87.29, 87.60, 87.07),
            ("caldraw", 2): (86.98, 87.71, 87.36, 87.47, 87.28),
            ("onlydraw", 2): (87.07, 87.57, 87.38, 87.39, 87.30),
            ("draw", 3): (86.72, 87.60, 87.18, 87.44, 87.28),
            ("draw", 4): (86.85, 87.64, 87.20, 87.81, 87.40),
            ("draw", 5): (87.16, 87.65, 87.24, 87.60, 87.16),
            ("draw", 6): (86.83, 87.62, 87.46, 87.74, 87.30),
        },
    },
    INVERTED: {
        BEFORE: {
            "budget": (93.65, 93.26, 90.63, 92.71, 93.47),
            ("cal", 1): (91.76, 91.01, 91.59, 92.39, 92.75),
            ("cal", 4): (92.95, 93.21, 90.88, 92.39, 93.42),
            ("cal", 16): (93.38, 93.15, 90.65, 92.68, 93.56),
            ("cal", 64): (93.53, 92.99, 90.53, 92.68, 93.45),
            ("only", 1): (93.42, 93.16, 90.42, 92.63, 93.37),
            ("only", 4): (93.28, 93.16, 90.47, 92.73, 93.53),
            ("only", 16): (93.47, 93.15, 90.39, 92.71, 93.46),
            ("only", 64): (93.53, 93.26, 90.44, 92.80, 93.42),
            ("step", 0.0625): (93.37, 93.25, 90.47, 92.72, 93.57),
            ("step", 0.00390625): (93.36, 93.20, 90.52, 92.70, 93.56),
            "best": (93.52, 93.15, 90.29, 92.69, 93.41),
            ("draw", 1): (93.44, 92.95, 90.36, 92.68, 93.36),
            ("caldraw", 1): (93.31, 93.24, 90.17, 92.57, 93.30),
            ("onlydraw", 1): (93.33, 93.27, 90.38, 92.66, 93.23),
            ("draw", 2): (93.38, 93.23, 90.40, 92.76, 93.61),
            ("caldraw", 2): (93.55, 93.42, 89.96, 92.50, 93.38),
            ("onlydraw", 2): (93.24, 93.33, 90.36, 92.70, 93.44),
            ("draw", 3): (93.24, 93.10, 90.38, 92.67, 93.35),
            ("draw", 4): (93.65, 93.26, 90.49, 92.81, 93.53),
            ("draw", 5): (93.46, 93.29, 90.34, 92.67, 93.36),
            ("draw", 6): (93.60, 93.24, 90.38, 92.69, 93.34),
        },
        REFERENCE: {
            "budget": (95.67, 95.26, 94.38, 93.72, 95.28),
            ("cal", 1): (94.84, 95.35, 94.34, 93.46, 95.48),
            ("cal", 4): (95.51, 95.31, 94.46, 93.75, 95.22),
            ("cal", 16): (95.65, 95.33, 94.56, 93.72, 95.30),
            ("cal", 64): (95.60, 95.26, 94.53, 93.96, 95.37),
            ("only", 1): (95.66, 95.33, 94.71, 93.78, 95.20),
            ("only", 4): (95.53, 95.31, 94.54, 93.62, 95.36),
            ("only", 16): (95.56, 95.33, 94.63, 93.51, 95.37),
            ("only", 64): (95.68, 95.25, 94.52, 93.90, 95.41),
            ("step", 0.0625): (95.59, 95.30, 94.58, 93.72, 95.22),
            ("step", 0.00390625): (95.64, 95.31, 94.57, 93.72, 95.23),
            "best": (95.59, 95.19, 94.56, 93.86, 95.42),
            ("draw", 1): (95.73, 95.27, 94.61, 93.64, 95.31),
            ("caldraw", 1): (95.71, 95.00, 94.76, 93.75, 95.20),
            ("onlydraw", 1): (95.74, 95.11, 94.72, 93.80, 95.34),
            ("draw", 2): (95.60, 95.43, 94.56, 93.56, 95.56),
            ("caldraw", 2): (95.51, 95.46, 94.38, 93.73, 95.22),
            ("onlydraw", 2): (95.50, 95.52, 94.43, 93.66, 95.23),
            ("draw", 3): (95.56, 95.38, 94.66, 93.73, 95.22),
            ("draw", 4): (95.46, 95.43, 94.50, 93.90, 95.33),
            ("draw", 5): (95.65, 95.21, 94.46, 93.64, 95.33),
            ("draw", 6): (95.43, 95.22, 94.64, 93.69, 95.30),
        },
    },
}
# The first table grx930's harness printed, mean and standard error: a row -> what the networks
# trained before lose, what the row adds to them as budgeted, and the same two for the reference.
PRINTED_LOST = {
    MNIST: {
        "budget": ((0.15, 0.02), (0.00, 0.00), (0.07, 0.05), (0.00, 0.00)),
        ("cal", 1): ((0.14, 0.06), (-0.01, 0.06), (0.12, 0.07), (0.05, 0.05)),
        ("cal", 4): ((0.08, 0.03), (-0.07, 0.03), (0.17, 0.04), (0.10, 0.03)),
        ("cal", 16): ((0.10, 0.04), (-0.05, 0.04), (0.15, 0.05), (0.08, 0.01)),
        ("cal", 64): ((0.18, 0.04), (0.03, 0.04), (0.14, 0.03), (0.07, 0.05)),
        ("only", 1): ((0.16, 0.05), (0.01, 0.05), (0.11, 0.03), (0.04, 0.02)),
        ("only", 4): ((0.11, 0.05), (-0.04, 0.04), (0.15, 0.04), (0.08, 0.02)),
        ("only", 16): ((0.08, 0.05), (-0.07, 0.05), (0.14, 0.04), (0.07, 0.02)),
        ("only", 64): ((0.18, 0.03), (0.03, 0.03), (0.14, 0.02), (0.07, 0.04)),
        ("step", 0.0625): ((0.10, 0.04), (-0.05, 0.04), (0.14, 0.05), (0.07, 0.01)),
        ("step", 0.00390625): ((0.10, 0.04), (-0.05, 0.04), (0.15, 0.05), (0.08, 0.01)),
        "best": ((0.19, 0.04), (0.04, 0.04), (0.16, 0.02), (0.09, 0.04)),
        ("draw", 1): ((0.15, 0.06), (-0.00, 0.06), (0.16, 0.04), (0.09, 0.02)),
        ("caldraw", 1): ((0.14, 0.06), (-0.01, 0.06), (0.13, 0.04), (0.06, 0.02)),
        ("onlydraw", 1): ((0.15, 0.04), (-0.00, 0.05), (0.16, 0.04), (0.09, 0.05)),
        ("draw", 2): ((0.10, 0.04), (-0.05, 0.04), (0.15, 0.03), (0.08, 0.03)),
        ("caldraw", 2): ((0.14, 0.06), (-0.01, 0.05), (0.16, 0.03), (0.09, 0.02)),
        ("onlydraw", 2): ((0.13, 0.06), (-0.02, 0.06), (0.19, 0.03), (0.12, 0.03)),
        ("draw", 3): ((0.15, 0.03), (-0.00, 0.03), (0.18, 0.02), (0.11, 0.03)),
        ("draw", 4): ((0.20, 0.06), (0.05, 0.06), (0.14, 0.04), (0.07, 0.03)),
        ("draw", 5): ((0.14, 0.04), (-0.01, 0.04), (0.13, 0.03), (0.06, 0.04)),
        ("draw", 6): ((0.09, 0.06), (-0.06, 0.06), (0.11, 0.04), (0.04, 0.04)),
    },
    FASHION: {
        "budget": ((0.54, 0.22), (0.00, 0.00), (0.47, 0.13), (0.00, 0.00)),
        ("cal", 1): ((0.82, 0.13), (0.28, 0.15), (0.77, 0.17), (0.30, 0.12)),
        ("cal", 4): ((0.67, 0.12), (0.13, 0.12), (0.47, 0.06), (-0.01, 0.10)),
        ("cal", 16): ((0.57, 0.13), (0.03, 0.11), (0.62, 0.04), (0.15, 0.12)),
        ("cal", 64): ((0.53, 0.15), (-0.01, 0.13), (0.51, 0.09), (0.04, 0.05)),
        ("only", 1): ((0.64, 0.12), (0.10, 0.11), (0.62, 0.15), (0.15, 0.06)),
        ("only", 4): ((0.55, 0.13), (0.01, 0.15), (0.56, 0.08), (0.09, 0.10)),
        ("only", 16): ((0.51, 0.11), (-0.03, 0.12), (0.61, 0.05), (0.14, 0.09)),
        ("only", 64): ((0.54, 0.16), (0.00, 0.10), (0.56, 0.06), (0.08, 0.09)),
        ("step", 0.0625): ((0.52, 0.14), (-0.02, 0.09), (0.64, 0.04), (0.16, 0.11)),
        ("step", 0.00390625): ((0.51, 0.12), (-0.03, 0.10), (0.62, 0.04), (0.14, 0.11)),
        "best": ((0.53, 0.16), (-0.01, 0.12), (0.54, 0.11), (0.07, 0.04)),
        ("draw", 1): ((0.53, 0.09), (-0.01, 0.14), (0.50, 0.09), (0.02, 0.08)),
        ("caldraw", 1): ((0.42, 0.08), (-0.12, 0.15), (0.46, 0.09), (-0.01, 0.11)),
        ("onlydraw", 1): ((0.41, 0.12), (-0.13, 0.17), (0.46, 0.12), (-0.02, 0.13)),
        ("draw", 2): ((0.43, 0.14), (-0.11, 0.08), (0.53, 0.09), (0.05, 0.10)),
        ("caldraw", 2): ((0.59, 0.15), (0.05, 0.13), (0.53, 0.07), (0.05, 0.06)),
        ("onlydraw", 2): ((0.53, 0.14), (-0.01, 0.13), (0.54, 0.10), (0.07, 0.06)),
        ("draw", 3): ((0.49, 0.13), (-0.05, 0.15), (0.64, 0.05), (0.17, 0.09)),
        ("draw", 4): ((0.55, 0.11), (0.01, 0.14), (0.51, 0.07), (0.03, 0.14)),
        ("draw", 5): ((0.41, 0.11), (-0.13, 0.13), (0.52, 0.10), (0.05, 0.10)),
        ("draw", 6): ((0.43, 0.14), (-0.11, 0.13), (0.50, 0.10), (0.02, 0.10)),
    },
    INVERTED: {
        "budget": ((0.62, 0.07), (0.00, 0.00), (0.52, 0.05), (0.00, 0.00)),
        ("cal", 1): ((1.46, 0.56), (0.84, 0.58), (0.69, 0.17), (0.17, 0.18)),
        ("cal", 4): ((0.79, 0.21), (0.17, 0.16), (0.53, 0.06), (0.01, 0.04)),
        ("cal", 16): ((0.68, 0.11), (0.06, 0.06), (0.47, 0.06), (-0.05, 0.04)),
        ("cal", 64): ((0.72, 0.06), (0.11, 0.04), (0.44, 0.04), (-0.08, 0.05)),
        ("only", 1): ((0.76, 0.10), (0.14, 0.03), (0.45, 0.09), (-0.07, 0.07)),
        ("only", 4): ((0.73, 0.13), (0.11, 0.08), (0.51, 0.07), (-0.01, 0.06)),
        ("only", 16): ((0.72, 0.10), (0.11, 0.05), (0.50, 0.10), (-0.02, 0.08)),
        ("only", 64): ((0.67, 0.10), (0.05, 0.05), (0.43, 0.02), (-0.09, 0.04)),
        ("step", 0.0625): ((0.68, 0.13), (0.07, 0.07), (0.50, 0.07), (-0.02, 0.05)),
        ("step", 0.00390625): ((0.69, 0.12), (0.08, 0.06), (0.49, 0.07), (-0.03, 0.04)),
        "best": ((0.75, 0.10), (0.13, 0.06), (0.46, 0.03), (-0.06, 0.06)),
        ("draw", 1): ((0.80, 0.08), (0.19, 0.05), (0.47, 0.08), (-0.05, 0.05)),
        ("caldraw", 1): ((0.84, 0.14), (0.23, 0.08), (0.50, 0.11), (-0.02, 0.10)),
        ("onlydraw", 1): ((0.79, 0.13), (0.17, 0.06), (0.44, 0.08), (-0.08, 0.08)),
        ("draw", 2): ((0.68, 0.14), (0.07, 0.08), (0.44, 0.09), (-0.08, 0.08)),
        ("caldraw", 2): ((0.80, 0.19), (0.18, 0.14), (0.52, 0.08), (0.00, 0.06)),
        ("onlydraw", 2): ((0.75, 0.16), (0.13, 0.09), (0.52, 0.10), (-0.01, 0.07)),
        ("draw", 3): ((0.81, 0.12), (0.20, 0.06), (0.47, 0.09), (-0.05, 0.07)),
        ("draw", 4): ((0.61, 0.09), (-0.00, 0.04), (0.46, 0.07), (-0.06, 0.07)),
        ("draw", 5): ((0.74, 0.12), (0.12, 0.06), (0.53, 0.05), (0.00, 0.03)),
        ("draw", 6): ((0.71, 0.09), (0.09, 0.04), (0.53, 0.08), (0.01, 0.08)),
    },
}
# The second: how often the networks trained before are right in a row, how often the reference,
# and the second less the first, seed by seed.
PRINTED_RIGHT = {
    MNIST: {
        "budget": ((97.30, 0.11), (97.68, 0.04), (0.38, 0.11)),
        ("cal", 1): ((97.32, 0.09), (97.63, 0.06), (0.32, 0.14)),
        ("cal", 4): ((97.37, 0.10), (97.59, 0.04), (0.21, 0.09)),
        ("cal", 16): ((97.35, 0.08), (97.60, 0.04), (0.25, 0.10)),
        ("cal", 64): ((97.27, 0.12), (97.61, 0.06), (0.34, 0.09)),
        ("only", 1): ((97.29, 0.10), (97.64, 0.05), (0.35, 0.10)),
        ("only", 4): ((97.34, 0.12), (97.60, 0.05), (0.26, 0.12)),
        ("only", 16): ((97.37, 0.09), (97.61, 0.04), (0.24, 0.10)),
        ("only", 64): ((97.27, 0.11), (97.61, 0.05), (0.34, 0.08)),
        ("step", 0.0625): ((97.35, 0.09), (97.61, 0.03), (0.26, 0.09)),
        ("step", 0.00390625): ((97.36, 0.09), (97.60, 0.04), (0.25, 0.10)),
        "best": ((97.26, 0.11), (97.60, 0.06), (0.33, 0.08)),
        ("draw", 1): ((97.31, 0.09), (97.59, 0.04), (0.28, 0.08)),
        ("caldraw", 1): ((97.31, 0.11), (97.62, 0.04), (0.31, 0.11)),
        ("onlydraw", 1): ((97.30, 0.11), (97.59, 0.05), (0.28, 0.08)),
        ("draw", 2): ((97.35, 0.11), (97.60, 0.05), (0.24, 0.10)),
        ("caldraw", 2): ((97.31, 0.11), (97.59, 0.04), (0.28, 0.10)),
        ("onlydraw", 2): ((97.32, 0.10), (97.56, 0.04), (0.25, 0.10)),
        ("draw", 3): ((97.30, 0.12), (97.57, 0.05), (0.27, 0.09)),
        ("draw", 4): ((97.25, 0.10), (97.61, 0.06), (0.36, 0.09)),
        ("draw", 5): ((97.32, 0.10), (97.62, 0.07), (0.30, 0.11)),
        ("draw", 6): ((97.36, 0.12), (97.64, 0.04), (0.28, 0.09)),
    },
    FASHION: {
        "budget": ((86.94, 0.20), (87.41, 0.13), (0.48, 0.20)),
        ("cal", 1): ((86.65, 0.22), (87.12, 0.16), (0.46, 0.27)),
        ("cal", 4): ((86.80, 0.20), (87.42, 0.12), (0.62, 0.19)),
        ("cal", 16): ((86.91, 0.15), (87.27, 0.17), (0.36, 0.16)),
        ("cal", 64): ((86.95, 0.24), (87.37, 0.11), (0.42, 0.26)),
        ("only", 1): ((86.84, 0.18), (87.26, 0.15), (0.43, 0.19)),
        ("only", 4): ((86.93, 0.22), (87.33, 0.09), (0.40, 0.21)),
        ("only", 16): ((86.97, 0.14), (87.27, 0.15), (0.30, 0.16)),
        ("only", 64): ((86.93, 0.22), (87.33, 0.11), (0.40, 0.20)),
        ("step", 0.0625): ((86.96, 0.15), (87.25, 0.15), (0.29, 0.16)),
        ("step", 0.00390625): ((86.97, 0.16), (87.27, 0.14), (0.30, 0.15)),
        "best": ((86.95, 0.24), (87.35, 0.11), (0.40, 0.25)),
        ("draw", 1): ((86.94, 0.18), (87.39, 0.15), (0.45, 0.20)),
        ("caldraw", 1): ((87.05, 0.17), (87.42, 0.20), (0.37, 0.23)),
        ("onlydraw", 1): ((87.07, 0.12), (87.43, 0.21), (0.36, 0.17)),
        ("draw", 2): ((87.04, 0.18), (87.36, 0.17), (0.32, 0.20)),
        ("caldraw", 2): ((86.88, 0.18), (87.36, 0.12), (0.48, 0.23)),
        ("onlydraw", 2): ((86.94, 0.18), (87.34, 0.08), (0.40, 0.21)),
        ("draw", 3): ((86.98, 0.14), (87.24, 0.15), (0.26, 0.15)),
        ("draw", 4): ((86.92, 0.21), (87.38, 0.17), (0.46, 0.18)),
        ("draw", 5): ((87.07, 0.15), (87.36, 0.11), (0.30, 0.13)),
        ("draw", 6): ((87.04, 0.21), (87.39, 0.16), (0.35, 0.22)),
    },
    INVERTED: {
        "budget": ((92.74, 0.55), (94.86, 0.36), (2.12, 0.45)),
        ("cal", 1): ((91.90, 0.31), (94.69, 0.37), (2.79, 0.52)),
        ("cal", 4): ((92.57, 0.46), (94.85, 0.33), (2.28, 0.38)),
        ("cal", 16): ((92.68, 0.53), (94.91, 0.35), (2.23, 0.47)),
        ("cal", 64): ((92.64, 0.55), (94.94, 0.30), (2.31, 0.45)),
        ("only", 1): ((92.60, 0.56), (94.94, 0.33), (2.34, 0.53)),
        ("only", 4): ((92.63, 0.56), (94.87, 0.36), (2.24, 0.52)),
        ("only", 16): ((92.64, 0.58), (94.88, 0.38), (2.24, 0.56)),
        ("only", 64): ((92.69, 0.58), (94.95, 0.33), (2.26, 0.49)),
        ("step", 0.0625): ((92.68, 0.57), (94.88, 0.33), (2.21, 0.52)),
        ("step", 0.00390625): ((92.67, 0.56), (94.89, 0.34), (2.23, 0.51)),
        "best": ((92.61, 0.60), (94.92, 0.32), (2.31, 0.52)),
        ("draw", 1): ((92.56, 0.57), (94.91, 0.37), (2.35, 0.53)),
        ("caldraw", 1): ((92.52, 0.60), (94.88, 0.32), (2.37, 0.59)),
        ("onlydraw", 1): ((92.57, 0.56), (94.94, 0.33), (2.37, 0.54)),
        ("draw", 2): ((92.68, 0.59), (94.94, 0.39), (2.27, 0.54)),
        ("caldraw", 2): ((92.56, 0.68), (94.86, 0.35), (2.30, 0.55)),
        ("onlydraw", 2): ((92.61, 0.58), (94.87, 0.36), (2.25, 0.51)),
        ("draw", 3): ((92.55, 0.55), (94.91, 0.33), (2.36, 0.53)),
        ("draw", 4): ((92.75, 0.58), (94.92, 0.31), (2.18, 0.49)),
        ("draw", 5): ((92.62, 0.59), (94.86, 0.36), (2.23, 0.52)),
        ("draw", 6): ((92.65, 0.59), (94.86, 0.32), (2.21, 0.54)),
    },
}
# The third: a calibrated row -> what its trims add, which is how often right with the same probes
# taken and nothing written, less calibrated, seed by seed: the networks trained before, the reference.
PRINTED_TRIMS = {
    MNIST: {
        ("cal", 1): ((-0.03, 0.03), (0.01, 0.04)),
        ("cal", 4): ((-0.03, 0.02), (0.01, 0.02)),
        ("cal", 16): ((0.02, 0.02), (0.02, 0.02)),
        ("cal", 64): ((0.00, 0.02), (-0.00, 0.01)),
        ("step", 0.0625): ((0.02, 0.02), (0.01, 0.02)),
        ("step", 0.00390625): ((0.01, 0.02), (0.01, 0.01)),
        "best": ((0.01, 0.02), (0.01, 0.02)),
        ("caldraw", 1): ((-0.01, 0.02), (-0.04, 0.03)),
        ("caldraw", 2): ((0.01, 0.03), (-0.02, 0.02)),
    },
    FASHION: {
        ("cal", 1): ((0.18, 0.09), (0.15, 0.13)),
        ("cal", 4): ((0.12, 0.05), (-0.09, 0.05)),
        ("cal", 16): ((0.06, 0.02), (0.01, 0.05)),
        ("cal", 64): ((-0.02, 0.03), (-0.04, 0.04)),
        ("step", 0.0625): ((0.01, 0.03), (0.02, 0.04)),
        ("step", 0.00390625): ((0.00, 0.02), (0.00, 0.04)),
        "best": ((-0.02, 0.03), (-0.02, 0.05)),
        ("caldraw", 1): ((0.02, 0.08), (0.01, 0.04)),
        ("caldraw", 2): ((0.06, 0.02), (-0.02, 0.04)),
    },
    INVERTED: {
        ("cal", 1): ((0.70, 0.58), (0.24, 0.19)),
        ("cal", 4): ((0.06, 0.14), (0.02, 0.05)),
        ("cal", 16): ((-0.05, 0.06), (-0.03, 0.05)),
        ("cal", 64): ((0.05, 0.06), (0.01, 0.02)),
        ("step", 0.0625): ((-0.04, 0.04), (-0.00, 0.06)),
        ("step", 0.00390625): ((-0.03, 0.04), (-0.01, 0.06)),
        "best": ((0.08, 0.03), (0.03, 0.02)),
        ("caldraw", 1): ((0.06, 0.05), (0.06, 0.03)),
        ("caldraw", 2): ((0.05, 0.12), (0.01, 0.02)),
    },
}
# The fourth: a kind -> the tile as budgeted on its other draws: how many; how often right on them;
# the as-budgeted row less a network's mean over them, mean and standard error; a network's standard
# deviation from draw to draw, rms over the five; and the standard deviation of the five networks' mean.
PRINTED_DRAWS = {
    MNIST: {BEFORE: (12, 97.316, (-0.014, 0.042), 0.063, 0.036),
            REFERENCE: (12, 97.604, (0.078, 0.027), 0.051, 0.023)},
    FASHION: {BEFORE: (12, 86.973, (-0.037, 0.116), 0.145, 0.071),
              REFERENCE: (12, 87.340, (0.072, 0.078), 0.135, 0.057)},
    INVERTED: {BEFORE: (12, 92.629, (0.115, 0.048), 0.094, 0.057),
               REFERENCE: (12, 94.904, (-0.042, 0.047), 0.107, 0.036)},
}

# ---- what follows from them ---------------------------------------------------------
def stat(xs):
    """Five figures' mean and standard error."""
    n = len(xs)
    m = sum(xs) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def spread(xs):
    """Some figures' standard deviation."""
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


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
    """What a row adds to the plan's row, version 2 as budgeted, network by network."""
    return less(w, kind, row, "budget")


def ahead(w, row):
    """How much more often a reference network is right in a row than the one
    trained before of the same seed."""
    return stat([r - b for r, b in zip(RUN[w][REFERENCE][row], RUN[w][BEFORE][row])])


def trims(w, kind, row):
    """What a calibrated row's trims add: how often right with its probes taken
    and nothing written, less calibrated.  The two runs meet the same draws."""
    return less(w, kind, row, PARTNER[row])


def trims_at_c3(w, kind):
    """The same at C3's 16 probes and quarter step, over the three draws that were
    calibrated on: a network's mean over the three, and the five networks'."""
    rows = (("cal", C3_PROBES),) + tuple(("caldraw", d) for d in CAL_DRAWS)
    return stat([sum(RUN[w][kind][PARTNER[r]][i] - RUN[w][kind][r][i] for r in rows) / len(rows) for i in range(SEEDS)])


def over(w, kind, rows):
    """Each network's mean accuracy over some rows."""
    return [sum(RUN[w][kind][r][i] for r in rows) / len(rows) for i in range(SEEDS)]


def own_less(w, kind):
    """The plan's as-budgeted row less a network's mean over the tile's other
    draws, network by network: how favourable a draw the plan's row is."""
    return stat([b - o for b, o in zip(RUN[w][kind]["budget"], over(w, kind, OTHERS))])


def net_sd(w, kind):
    """A network's standard deviation from draw to draw of the tile as budgeted,
    rms over the five networks."""
    return math.sqrt(sum(spread([RUN[w][kind][r][i] for r in OTHERS]) ** 2 for i in range(SEEDS)) / SEEDS)


def mean_sd(w, kind):
    """The same for the five networks' mean: what a row of the plan moves by when
    nothing changes but the draw."""
    return spread([acc(w, kind, r)[0] for r in OTHERS])


def two_draws(w, kind):
    """One standard deviation of the difference between two rows that do not share
    their draws, with nothing else changed."""
    return mean_sd(w, kind) * math.sqrt(2)


def lost_over(w, kind):
    """What the tile as budgeted costs, over all thirteen of its draws."""
    return stat([h - m for h, m in zip(HOSTS[w][kind], over(w, kind, ("budget",) + OTHERS))])


def ahead_over(w):
    """The reference network less the one trained before, over the thirteen draws."""
    rows = ("budget",) + OTHERS
    return stat([r - b for r, b in zip(over(w, REFERENCE, rows), over(w, BEFORE, rows))])


def rebased(w, kind, row):
    """A calibrated row of pta_reference_cycle.py's over the tile as budgeted on
    that row's own draws: the probes taken and nothing written, less the row."""
    return stat([z - c for z, c in zip(RUN[w][kind][("only", C3_PROBES)], cycle.RUN[w][kind][row])])


def holds(w, kind, budget=WITHIN, exact=False):
    """How long a calibration holds in the cycle, on the cycle's own draws: the
    longest interval run whose cycle ends under the budget, every shorter one
    doing so too.  Read on means to the hundredth, or exactly."""
    ok = None
    for h in cycle.CYCLES:
        x = rebased(w, kind, ("cycle", h))[0]
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


def pm(x):
    return f"{x[0]:.2f} +-{x[1]:.2f}"


def dpm(x):
    return f"{x[0]:+.2f} +-{x[1]:.2f}"


def r2(x):
    return round(x[0], 2), round(x[1], 2)


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def both(f):
    """A row of a table: a figure for each data set's two kinds."""
    return "".join(f"{dpm(f(w, k)):>16}" for w in WORKLOADS for k in KINDS)


def hold_rows():
    """Section 5's last row: how long a calibration holds in the cycle, a data set's two kinds."""
    return [cycle.interval(holds(w, k)) for w in WORKLOADS for k in KINDS]


LABEL_OWN, LABEL_MEAN, LABEL_ROW = "the plan's row less that", "the five networks' mean", "points lost, the plan's row"
CYCLE_ROWS = (("cal", "calibrated as written"), (("aged", 0.1), "aged 6 minutes, calibrated"), (("aged", 1), "aged an hour, calibrated")) + tuple(
    (("cycle", h), f"a cycle of {cycle.minutes(h)}") for h in cycle.CYCLES) + (("after", "an hour, calibrated, 6 minutes"),)


def main():
    print("What a calibration costs: its trims, and its draws.")
    print(f"The working tile, {laser.name(TILE)}, at version 2, with no drift.  Five networks a kind, mean and standard error.")
    heads = (f"  {'':<34}" + "".join(f"{w:>32}" for w in WORKLOADS) + "\n"
             + f"  {'':<34}" + "".join(f"{('before' if k == BEFORE else 'reference'):>16}" for w in WORKLOADS for k in KINDS))

    section("1. Points lost against a network's own accuracy on its host, and what a row adds to the plan's row")
    for w in WORKLOADS:
        print(f"  {w:<34}{'trained before':>16}{'and adds':>16}{'the reference':>16}{'and adds':>16}")
        for key, name in ROWS:
            print(f"    {name:<32}" + "".join(f"{pm(lost(w, k, key)):>16}{dpm(adds(w, k, key)):>16}" for k in KINDS))

    section("2. What the trims add: the same probes taken and nothing written, less calibrated, on the same draws")
    print(heads)
    for key, name in ROWS:
        if key in PARTNER:
            print(f"    {name:<32}" + both(lambda w, k: trims(w, k, key)))
    print(f"    {'16 probes, over three draws':<32}" + both(trims_at_c3))

    section("3. What the draws do: the tile as budgeted on twelve other draws, and the plan's row among them")
    print(heads)
    print(f"    {'right over them, percent':<32}" + "".join(f"{stat(over(w, k, OTHERS))[0]:>16.2f}" for w in WORKLOADS for k in KINDS))
    print(f"    {LABEL_OWN:<32}" + both(own_less))
    print(f"    {'a network, draw to draw':<32}" + "".join(f"{net_sd(w, k):>16.3f}" for w in WORKLOADS for k in KINDS))
    print(f"    {LABEL_MEAN:<32}" + "".join(f"{mean_sd(w, k):>16.3f}" for w in WORKLOADS for k in KINDS))
    print(f"    {'two rows on two draws':<32}" + "".join(f"{two_draws(w, k):>16.3f}" for w in WORKLOADS for k in KINDS))
    print("  and pta_reference_cycle.py's cost of a calibration, taken apart:")
    print(f"    {'calibrated less as budgeted':<32}" + both(lambda w, k: adds(w, k, ("cal", C3_PROBES))))
    print(f"    {'  of that, its draws':<32}" + both(lambda w, k: adds(w, k, ("only", C3_PROBES))))
    print(f"    {'  and its trims':<32}" + both(lambda w, k: trims(w, k, ("cal", C3_PROBES))))

    section("4. What the tile as budgeted costs: the plan's row, and over thirteen draws")
    print(heads)
    print(f"    {LABEL_ROW:<32}" + "".join(f"{pm(lost(w, k, 'budget')):>16}" for w in WORKLOADS for k in KINDS))
    print(f"    {'points lost, over the draws':<32}" + "".join(f"{pm(lost_over(w, k)):>16}" for w in WORKLOADS for k in KINDS))
    print(f"    {'reference less before: the row':<32}" + "".join(f"{dpm(ahead(w, 'budget')):>32}" for w in WORKLOADS))
    print(f"    {'reference less before: the draws':<32}" + "".join(f"{dpm(ahead_over(w)):>32}" for w in WORKLOADS))

    section("5. The cycle's calibrated rows, over the tile as budgeted on their own draws")
    print(heads)
    for key, name in CYCLE_ROWS:
        print(f"    {name:<32}" + both(lambda w, k: rebased(w, k, key)))
    print(f"    {'a calibration holds':<32}" + "".join(f"{c:>16}" for c in hold_rows()))
    print(f"    {'and over the as-budgeted row':<32}" + "".join(f"{c:>16}" for c in cycle.hold_rows()))

    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        findings()
    print(said.getvalue(), end="")
    checks(" ".join(said.getvalue().split()))


# ---- a check by hand ------------------------------------------------------------------
# grx930's note, the same section.  MNIST only: five networks of each kind that no sweep has
# run, seeds 6 to 10, trained as the others were, on the tile as budgeted at their own seed and
# then at six others, the seed and 10, 20 ... 60 more.  Percent right; a network a row.
NEW_SEEDS = (6, 7, 8, 9, 10)
NEW_RUNS = {
    BEFORE: ((97.37, 97.46, 97.51, 97.44, 97.41, 97.50, 97.50), (97.13, 97.22, 97.10, 97.01, 97.08, 97.08, 97.34),
             (97.25, 97.21, 97.22, 97.12, 97.11, 97.27, 97.14), (97.09, 97.25, 97.34, 97.20, 97.16, 97.16, 97.31),
             (97.32, 97.38, 97.35, 97.45, 97.46, 97.28, 97.36)),
    REFERENCE: ((97.85, 97.81, 97.89, 97.84, 97.75, 97.84, 97.77), (97.38, 97.40, 97.28, 97.36, 97.25, 97.22, 97.39),
                (97.42, 97.39, 97.45, 97.47, 97.65, 97.52, 97.58), (97.63, 97.62, 97.69, 97.59, 97.60, 97.59, 97.60),
                (97.56, 97.51, 97.47, 97.48, 97.44, 97.42, 97.55)),
}


def new_own_less(kind):
    """For the five new networks: the tile at a network's own seed less its mean
    over the six other seeds, network by network."""
    return stat([row[0] - sum(row[1:]) / len(row[1:]) for row in NEW_RUNS[kind]])


def lost_gap(w):
    """What a reference network loses to the tile as budgeted, less what the one
    trained before of the same seed does, over the thirteen draws."""
    rows = ("budget",) + OTHERS
    return stat([(hr - r) - (hb - b) for hr, r, hb, b in zip(HOSTS[w][REFERENCE], over(w, REFERENCE, rows), HOSTS[w][BEFORE], over(w, BEFORE, rows))])


def findings():
    m, f, i = WORKLOADS
    R, B = REFERENCE, BEFORE
    c3, six = ("cal", C3_PROBES), ("cycle", cycle.B15_H)
    cells = [(w, k, row) for w in WORKLOADS for k in KINDS for row in PARTNER]
    print()
    print("What this says, six readings.")
    print()
    print("  1. IT IS NOT THE TRIMS.  At C3's 16 probes and quarter step, over the three draws that were")
    print(f"     calibrated on, the trims add {dpm(trims_at_c3(m, R))}, {dpm(trims_at_c3(f, R))} and {dpm(trims_at_c3(i, R))} to what the")
    print(f"     reference networks lose and {dpm(trims_at_c3(m, B))}, {dpm(trims_at_c3(f, B))} and {dpm(trims_at_c3(i, B))} to the old ones'.  With")
    print("     one probe a cell, where the trim a cell is left with is as large as its programming")
    print(f"     error, they add {dpm(trims(m, R, ('cal', 1)))}, {dpm(trims(f, R, ('cal', 1)))} and {dpm(trims(i, R, ('cal', 1)))} to the reference networks,")
    print(f"     and {dpm(trims(i, B, ('cal', 1)))} to the old ones on the inverted set: more on the harder sets, and clear on")
    print(f"     none.  With 64 at a 256th of an LSB, {dpm(trims(m, R, 'best'))}, {dpm(trims(f, R, 'best'))} and {dpm(trims(i, R, 'best'))}.  Of the {len(cells)}")
    print(f"     such differences here, both kinds, {sum(clear(trims(*c)) for c in cells)} is outside chance at one in twenty, where chance")
    print(f"     gives {len(cells) / 20:.1f}.")
    print()
    print("  2. WHAT WAS CALLED THE CALIBRATION'S COST WAS ITS DRAWS.  pta_reference_cycle.py's calibrated")
    print(f"     row is {dpm(adds(m, R, c3))}, {dpm(adds(f, R, c3))} and {dpm(adds(i, R, c3))} short of the row as budgeted for the")
    print("     reference networks.  The same probes taken and nothing written, which is the tile as")
    print(f"     budgeted on the calibrated run's draws, are {dpm(adds(m, R, ('only', C3_PROBES)))}, {dpm(adds(f, R, ('only', C3_PROBES)))} and {dpm(adds(i, R, ('only', C3_PROBES)))}")
    print(f"     short of it.  The trims are the rest: {dpm(trims(m, R, c3))}, {dpm(trims(f, R, c3))} and {dpm(trims(i, R, c3))}.")
    print(f"     MNIST's draws are {errors(adds(m, R, ('only', C3_PROBES))):.1f} of their errors from nothing, outside chance by the test, and")
    print("     chance all the same: reading 4.")
    print()
    print("  3. A ROW MOVES WHEN NOTHING CHANGES BUT ITS DRAW.  Over twelve other draws of the tile as")
    print(f"     budgeted a reference network's accuracy has a standard deviation of {net_sd(m, R):.2f}, {net_sd(f, R):.2f} and {net_sd(i, R):.2f}")
    print(f"     of a point, and the five networks' mean of {mean_sd(m, R):.2f}, {mean_sd(f, R):.2f} and {mean_sd(i, R):.2f}; the old networks'")
    print(f"     mean of {mean_sd(m, B):.2f}, {mean_sd(f, B):.2f} and {mean_sd(i, B):.2f}.  Two rows that do not share their draws differ by")
    print(f"     {two_draws(m, R):.2f}, {two_draws(f, R):.2f} and {two_draws(i, R):.2f} at one standard deviation with nothing else changed.  A calibrated")
    print("     row and one that is not never share theirs: each probe is six GEMMs, and each GEMM")
    print("     takes the run's next seed.")
    print()
    print("  4. THE PLAN'S ROW AS BUDGETED IS ONE DRAW, AND ON MNIST A FAVOURABLE ONE FOR THE REFERENCE")
    print(f"     NETWORKS.  It is {dpm(own_less(m, R))} above their mean over the twelve other draws, {errors(own_less(m, R)):.1f} of")
    print(f"     its errors; on the other two sets {dpm(own_less(f, R))} and {dpm(own_less(i, R))}, and for the old networks")
    print(f"     {dpm(own_less(m, B))}, {dpm(own_less(f, B))} and {dpm(own_less(i, B))}.  A tile's seed in the plan's row is its network's,")
    print("     and nothing in grx930's trainer or its model ties the two.  Five networks no sweep had")
    print(f"     run, seeds 6 to 10, on their own seed and six others, are {dpm(new_own_less(R))} above; five")
    print(f"     trained the old way, {dpm(new_own_less(B))}.  Over all thirteen draws the reference networks lose")
    print(f"     {lost_over(m, R)[0]:.2f}, {lost_over(f, R)[0]:.2f} and {lost_over(i, R)[0]:.2f} of a point to the tile where the plan's row has {lost(m, R, 'budget')[0]:.2f}, {lost(f, R, 'budget')[0]:.2f} and {lost(i, R, 'budget')[0]:.2f},")
    print(f"     and the old ones {lost_over(m, B)[0]:.2f}, {lost_over(f, B)[0]:.2f} and {lost_over(i, B)[0]:.2f} where it has {lost(m, B, 'budget')[0]:.2f}, {lost(f, B, 'budget')[0]:.2f} and {lost(i, B, 'budget')[0]:.2f}.  Seed by")
    print(f"     seed the reference networks lose {dpm(lost_gap(m))}, {dpm(lost_gap(f))} and {dpm(lost_gap(i))} more than the old")
    print("     ones over the draws, which is nothing clear: on the first two sets a network trained")
    print("     for the tile loses to the tile as budgeted what one that was not does.  It is right")
    print(f"     {dpm(ahead_over(m))}, {dpm(ahead_over(f))} and {dpm(ahead_over(i))} more often, for the row's {ahead(m, 'budget')[0]:.2f}, {ahead(f, 'budget')[0]:.2f} and")
    print(f"     {ahead(i, 'budget')[0]:.2f}.")
    print()
    print("  5. ON ITS OWN DRAWS A SIX-MINUTE CYCLE ENDS WITHIN A TENTH ON MNIST AND FASHION-MNIST, AND A")
    print(f"     HUNDREDTH OVER IT ON THE INVERTED SET.  {dpm(rebased(m, R, six))}, {dpm(rebased(f, R, six))} and {dpm(rebased(i, R, six))} over the")
    print("     tile as budgeted on the cycle's draws, for the reference networks, where")
    print(f"     pta_reference_cycle.py had {cycle.adds(m, R, six)[0]:.2f}, {cycle.adds(f, R, six)[0]:.2f} and {cycle.adds(i, R, six)[0]:.2f} over the row as budgeted.  Fashion-MNIST's,")
    print("     which no interval held, went with the draws.  By the rule a calibration holds")
    print(f"     {cycle.interval(holds(m, R))}, {cycle.interval(holds(f, R))} and {cycle.interval(holds(i, R))} for them in the cycle.  The inverted set's six minutes")
    print(f"     is now drawn three times on its own draws: {dpm(cycle.adds(i, R, ('fresh', 0.1)))} from weights as written,")
    print(f"     {dpm(rebased(i, R, six))} at the end of the cycle, and {dpm(rebased(i, R, 'after'))} after the hour's calibration.")
    print("     Whether six minutes holds a tenth there is not settled by five networks and three")
    print("     draws of the drift.")
    print()
    print("  6. WHAT STANDS IN pta_reference_cycle.py, AND WHAT DOES NOT.  Its figures are grx930's and")
    print("     stand.  So does what it reads between two calibrated rows, which share their draws: an")
    print("     interval's own, what came before a calibration, what holding adds.  What it reads")
    print("     between a calibrated row and one that is not carries the draws as well: a calibration's")
    print("     own cost, which its readings 1 and 7 took for the calibration's; a cycle over the row")
    print("     as budgeted, its 2, 3 and 5, which reading 5 here replaces; a cycle against weights as")
    print("     written, its 4; and held against held, its 6.  The last two found nothing clear, and")
    print("     that is what they can still say.")


def checks(said):
    """Every claim above, as an assert.  `said` is the readings as printed, on one line."""
    m, f, i = WORKLOADS
    R, B = REFERENCE, BEFORE
    keys = [k for k, _ in ROWS]
    c3, only, six = ("cal", C3_PROBES), ("only", C3_PROBES), ("cycle", cycle.B15_H)
    assert TILE == (128, 64) and WITHIN == 0.10 and C3_PROBES == 16 and len(keys) == 22 and len(set(keys)) == 22
    assert len(OTHERS) == 12 and set(OTHERS) < set(keys) and len(PARTNER) == 9
    assert set(PARTNER) == {k for k in keys if k[0] in ("cal", "step", "caldraw") or k == "best"} and set(PARTNER.values()) < set(OTHERS)
    assert set(RUN) == set(HOSTS) == set(PRINTED_LOST) == set(PRINTED_RIGHT) == set(PRINTED_TRIMS) == set(PRINTED_DRAWS) == set(WORKLOADS)
    assert all(set(RUN[w]) == set(HOSTS[w]) == set(KINDS) for w in WORKLOADS)
    assert all(set(RUN[w][k]) == set(keys) and len(HOSTS[w][k]) == SEEDS for w in WORKLOADS for k in KINDS)
    assert all(len(RUN[w][k][row]) == SEEDS for w in WORKLOADS for k in KINDS for row in keys)

    # 0. The figures are grx930's, and agree with what the plan already holds.
    #    All four tables its harness printed, every cell, from the networks' own figures.
    def same(a, b, tol=0.0051):
        return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol

    cells = 0
    for w in WORKLOADS:
        assert set(PRINTED_LOST[w]) == set(PRINTED_RIGHT[w]) == set(keys) and set(PRINTED_TRIMS[w]) == set(PARTNER)
        for row in keys:
            got = (lost(w, B, row), adds(w, B, row), lost(w, R, row), adds(w, R, row))
            assert len(PRINTED_LOST[w][row]) == 4 and all(same(a, b) for a, b in zip(got, PRINTED_LOST[w][row])), (w, row)
            got = (acc(w, B, row), acc(w, R, row), ahead(w, row))
            assert len(PRINTED_RIGHT[w][row]) == 3 and all(same(a, b) for a, b in zip(got, PRINTED_RIGHT[w][row])), (w, row)
            cells += 7
        for row in PARTNER:
            got = (trims(w, B, row), trims(w, R, row))
            assert len(PRINTED_TRIMS[w][row]) == 2 and all(same(a, b) for a, b in zip(got, PRINTED_TRIMS[w][row])), (w, row)
            cells += 2
        for k in KINDS:
            n, right, less_, a_net, the_mean = PRINTED_DRAWS[w][k]
            assert n == len(OTHERS) and abs(right - stat(over(w, k, OTHERS))[0]) < 0.00051, (w, k)
            assert same(own_less(w, k), less_, 0.00051) and abs(a_net - net_sd(w, k)) < 0.00051 and abs(the_mean - mean_sd(w, k)) < 0.00051, (w, k)
            cells += 5
    assert cells == 3 * (22 * 7 + 9 * 2 + 2 * 5) == 546
    #    Counted and summed, as grx930's lines gave them.
    figures = [x for w in RUN.values() for k in w.values() for row in k.values() for x in row]
    assert (len(figures), round(sum(figures), 2)) == (660, 61235.55)
    hosts = [x for w in HOSTS.values() for k in w.values() for x in k]
    assert (len(hosts), round(sum(hosts), 2)) == (30, 2796.55)
    printed = [x for t in (PRINTED_LOST, PRINTED_RIGHT, PRINTED_TRIMS) for w in t.values() for row in w.values() for cell in row for x in cell]
    assert (len(printed), round(sum(printed), 2)) == (1032, 12449.10)
    #    The networks are pta_reference_cycle.py's, seed for seed, in the two rows the
    #    sweeps share: grx930 found those lines the same, byte for byte.
    assert cycle.TILE == TILE and cycle.WORKLOADS == WORKLOADS and (cycle.BEFORE, cycle.REFERENCE) == (B, R)
    for w in WORKLOADS:
        for k in KINDS:
            assert HOSTS[w][k] == cycle.HOSTS[w][k], (w, k)
            assert RUN[w][k]["budget"] == cycle.RUN[w][k]["budget"] and RUN[w][k][c3] == cycle.RUN[w][k]["cal"], (w, k)
    assert abs(T95 - 2.776) < 1e-9 and cycle.PROBES == 16 and STEPS[-1] == 1 / 256 and all(x < QUARTER for x in STEPS)

    # 1. Reading 1.  The trims.
    assert [r2(trims_at_c3(w, R)) for w in WORKLOADS] == [(-0.01, 0.01), (0.00, 0.03), (0.01, 0.03)]
    assert [r2(trims_at_c3(w, B)) for w in WORKLOADS] == [(0.01, 0.01), (0.05, 0.03), (0.02, 0.04)]
    assert not any(clear(trims_at_c3(w, k)) for w in WORKLOADS for k in KINDS)
    assert [round(at_most(trims_at_c3(w, R)), 2) for w in WORKLOADS] == [0.02, 0.09, 0.09]
    assert [r2(trims(w, R, ("cal", 1))) for w in WORKLOADS] == [(0.01, 0.04), (0.15, 0.13), (0.24, 0.19)]
    assert [r2(trims(w, R, "best")) for w in WORKLOADS] == [(0.01, 0.02), (-0.02, 0.05), (0.03, 0.02)]
    assert r2(trims(i, B, ("cal", 1))) == (0.70, 0.58)
    assert trims(m, R, ("cal", 1))[0] < trims(f, R, ("cal", 1))[0] < trims(i, R, ("cal", 1))[0]
    assert not any(clear(trims(w, k, ("cal", 1))) for w in WORKLOADS for k in KINDS)
    #    The trim one probe leaves is the programming error's size, and 16 leave a quarter of it.
    assert math.sqrt(1 / 1) == 1 and math.sqrt(1 / C3_PROBES) == 0.25
    assert [r2(trims(w, R, c3)) for w in WORKLOADS] == [(0.02, 0.02), (0.01, 0.05), (-0.03, 0.05)]
    assert [r2(trims(w, B, c3)) for w in WORKLOADS] == [(0.02, 0.02), (0.06, 0.02), (-0.05, 0.06)]
    cells = [(w, k, row) for w in WORKLOADS for k in KINDS for row in PARTNER]
    assert len(cells) == 54 and [c for c in cells if clear(trims(*c))] == [(f, B, ('caldraw', 2))]
    assert round(max(abs(trims(*c)[0]) for c in cells if c[0] == m), 2) == 0.04
    assert [round(max(abs(trims(w, k, row)[0]) for k in KINDS for row in PARTNER), 2) for w in WORKLOADS] == [0.04, 0.18, 0.70]
    #    Every number of probes and every step, for the reference networks.
    assert [[round(trims(w, R, ("cal", n))[0], 2) for n in PROBES] for w in WORKLOADS] == [[0.01, 0.01, 0.02, 0.00], [0.15, -0.09, 0.01, -0.04], [0.24, 0.02, -0.03, 0.01]]
    assert [[round(trims(w, R, ("step", x))[0], 2) for x in STEPS] for w in WORKLOADS] == [[0.01, 0.01], [0.02, 0.00], [0.00, -0.01]]

    # 2. Reading 2.  The cycle model's cost of a calibration, taken apart.
    assert [r2(adds(w, R, c3)) for w in WORKLOADS] == [(0.08, 0.01), (0.15, 0.12), (-0.05, 0.04)]
    assert all(adds(w, k, c3) == cycle.adds(w, k, "cal") for w in WORKLOADS for k in KINDS)
    assert [r2(adds(w, R, only)) for w in WORKLOADS] == [(0.07, 0.02), (0.14, 0.09), (-0.02, 0.08)]
    assert [r2(adds(w, B, c3)) for w in WORKLOADS] == [(-0.05, 0.04), (0.03, 0.11), (0.06, 0.06)]
    assert [r2(adds(w, B, only)) for w in WORKLOADS] == [(-0.07, 0.05), (-0.03, 0.12), (0.11, 0.05)]
    #    The two parts are the whole, network by network.
    assert all(abs(adds(w, k, c3)[0] - adds(w, k, only)[0] - trims(w, k, c3)[0]) < 1e-9 for w in WORKLOADS for k in KINDS)
    assert [(w, k) for w in WORKLOADS for k in KINDS if clear(adds(w, k, only))] == [(m, R)]
    #    In the standard deviation of two rows on two draws.
    assert [round(adds(w, R, only)[0] / two_draws(w, R), 1) for w in WORKLOADS] == [2.0, 1.7, -0.4]
    assert round(errors(adds(m, R, only)), 1) == 3.8

    # 3. Reading 3.  The draws.
    assert [round(net_sd(w, R), 2) for w in WORKLOADS] == [0.05, 0.14, 0.11]
    assert [round(net_sd(w, B), 2) for w in WORKLOADS] == [0.06, 0.14, 0.09]
    assert [round(mean_sd(w, R), 2) for w in WORKLOADS] == [0.02, 0.06, 0.04]
    assert [round(mean_sd(w, B), 2) for w in WORKLOADS] == [0.04, 0.07, 0.06]
    assert [round(two_draws(w, R), 2) for w in WORKLOADS] == [0.03, 0.08, 0.05]
    assert [round(two_draws(w, B), 2) for w in WORKLOADS] == [0.05, 0.10, 0.08]
    #    Six GEMMs a probe: three passes on each of two banks.
    assert 3 * 2 == 6

    # 4. Reading 4.  The plan's row among the draws.
    assert [r2(own_less(w, R)) for w in WORKLOADS] == [(0.08, 0.03), (0.07, 0.08), (-0.04, 0.05)]
    assert [r2(own_less(w, B)) for w in WORKLOADS] == [(-0.01, 0.04), (-0.04, 0.12), (0.11, 0.05)]
    assert [(w, k) for w in WORKLOADS for k in KINDS if clear(own_less(w, k))] == [(m, R)]
    assert round(errors(own_less(m, R)), 1) == 2.9
    assert len(NEW_SEEDS) == 5 and all(len(NEW_RUNS[k]) == 5 and all(len(row) == 7 for row in NEW_RUNS[k]) for k in KINDS)
    assert r2(new_own_less(R)) == (0.02, 0.03)
    assert r2(new_own_less(B)) == (-0.05, 0.04)
    assert not clear(new_own_less(R)) and not clear(new_own_less(B)) and abs(new_own_less(R)[0]) < 0.05
    assert [round(lost_over(w, R)[0], 2) for w in WORKLOADS] == [0.14, 0.54, 0.48]
    assert [round(lost(w, R, "budget")[0], 2) for w in WORKLOADS] == [0.07, 0.47, 0.52]
    assert [round(lost_over(w, B)[0], 2) for w in WORKLOADS] == [0.14, 0.51, 0.72]
    assert [round(lost(w, B, "budget")[0], 2) for w in WORKLOADS] == [0.15, 0.54, 0.62]
    assert [r2(ahead_over(w)) for w in WORKLOADS] == [(0.29, 0.09), (0.38, 0.17), (2.26, 0.52)]
    assert [round(ahead(w, "budget")[0], 2) for w in WORKLOADS] == [0.38, 0.48, 2.12]
    assert [w for w in WORKLOADS if clear(ahead_over(w))] == [m, i]
    assert [r2(lost_over(w, R)) for w in WORKLOADS] == [(0.14, 0.03), (0.54, 0.07), (0.48, 0.06)]
    assert [r2(lost_over(w, B)) for w in WORKLOADS] == [(0.14, 0.04), (0.51, 0.12), (0.72, 0.11)]
    #    What the reference networks lose less what the old ones do, over the draws, seed by seed.
    assert [r2(lost_gap(w)) for w in WORKLOADS] == [(0.01, 0.04), (0.03, 0.10), (-0.24, 0.13)]
    assert not any(clear(lost_gap(w)) for w in WORKLOADS)
    assert all(abs(lost_gap(w)[0]) < 0.05 for w in (m, f)) and lost_gap(i)[0] < -0.2
    #    The old networks' row is a favourable draw on the inverted set, at 2.4 of its errors.
    assert round(errors(own_less(i, B)), 1) == 2.4

    # 5. Reading 5.  The cycle's rows on their own draws.
    assert [r2(rebased(w, R, six)) for w in WORKLOADS] == [(0.02, 0.04), (0.02, 0.02), (0.11, 0.09)]
    assert [r2(rebased(w, B, six)) for w in WORKLOADS] == [(0.02, 0.03), (0.19, 0.07), (0.03, 0.21)]
    assert [round(cycle.adds(w, R, six)[0], 2) for w in WORKLOADS] == [0.08, 0.16, 0.09]
    assert [w for w in WORKLOADS if round(rebased(w, R, six)[0], 2) < WITHIN] == [m, f]
    assert round(rebased(i, R, six)[0], 2) == round(WITHIN + 0.01, 2) and not clear(rebased(i, R, six))
    #    The inverted set's six minutes, three draws of the drift, each over the tile as
    #    budgeted on its own draws.  The first is not calibrated, and shares the row's.
    assert [r2(x) for x in (cycle.adds(i, R, ("fresh", 0.1)), rebased(i, R, six), rebased(i, R, "after"))] == [(0.08, 0.11), (0.11, 0.09), (0.30, 0.09)]
    assert [clear(x) for x in (cycle.adds(i, R, ("fresh", 0.1)), rebased(i, R, six), rebased(i, R, "after"))] == [False, False, True]
    assert rebased(i, R, "after")[0] - T95 * rebased(i, R, "after")[1] < WITHIN
    #    Fashion-MNIST: no interval held over the row as budgeted, and half an hour does here.
    assert cycle.holds(f, R) is None and holds(f, R) == 0.5
    assert [holds(w, R) for w in WORKLOADS] == [1, 0.5, 0.05]
    assert [holds(w, B) for w in WORKLOADS] == [0.5, 0.05, None]
    assert hold_rows() == ['30 minutes', 'an hour', '3 minutes', '30 minutes', 'none run', '3 minutes']
    assert all(holds(w, k, exact=True) == holds(w, k) for w in WORKLOADS for k in KINDS)
    assert [cycle.holds(w, R) for w in WORKLOADS] == [0.25, None, 0.1]
    #    A calibrated row of the cycle's over its own draws is that row's trims and what it drifted.
    assert all(abs(rebased(w, k, "cal")[0] - trims(w, k, c3)[0]) < 1e-9 for w in WORKLOADS for k in KINDS)
    #    Every cycle the reference networks ran, on its own draws; and three minutes.
    assert [[round(rebased(w, R, ("cycle", h))[0], 2) for h in cycle.CYCLES] for w in WORKLOADS] == [[0.00, 0.02, 0.01, 0.08, 0.07], [0.00, 0.02, 0.09, -0.09, 0.24], [0.04, 0.11, 0.26, 0.47, 1.60]]
    assert [[round(rebased(w, B, ("cycle", h))[0], 2) for h in cycle.CYCLES] for w in WORKLOADS] == [[0.03, 0.02, 0.05, 0.06, 0.14], [0.09, 0.19, 0.14, 0.41, 0.54], [0.23, 0.03, 0.04, 0.87, 2.23]]
    near = [(w, k, h) for w in WORKLOADS for k in KINDS for h in cycle.CYCLES if abs(rebased(w, k, ("cycle", h))[0] - WITHIN) < 0.03]
    assert near == [(m, R, 0.5), (f, B, 0.05), (f, R, 0.25), (i, R, 0.1)], near
    #    Re-basing a row moves it by what the draws were, the same for every calibrated row of a network.
    for w in WORKLOADS:
        for k in KINDS:
            for row, _ in CYCLE_ROWS:
                assert abs(cycle.adds(w, k, row)[0] - rebased(w, k, row)[0] - adds(w, k, only)[0]) < 1e-9, (w, k, row)

    # 6. And the readings say those figures, each in its place.
    for words in (
        "the trims add -0.01 +-0.01, -0.00 +-0.03 and +0.01 +-0.03 to what the reference networks lose and +0.01 +-0.01, +0.05 +-0.03 and +0.02 +-0.04 to the old ones'.",
        "they add +0.01 +-0.04, +0.15 +-0.13 and +0.24 +-0.19 to the reference networks, and +0.70 +-0.58 to the old ones on the inverted set",
        "With 64 at a 256th of an LSB, +0.01 +-0.02, -0.02 +-0.05 and +0.03 +-0.02.",
        "Of the 54 such differences here, both kinds, 1 is outside chance at one in twenty, where chance gives 2.7.",
        "row is +0.08 +-0.01, +0.15 +-0.12 and -0.05 +-0.04 short of the row as budgeted for the reference networks.",
        "are +0.07 +-0.02, +0.14 +-0.09 and -0.02 +-0.08 short of it.",
        "The trims are the rest: +0.02 +-0.02, +0.01 +-0.05 and -0.03 +-0.05.",
        "MNIST's draws are 3.8 of their errors from nothing",
        "a standard deviation of 0.05, 0.14 and 0.11 of a point, and the five networks' mean of 0.02, 0.06 and 0.04; the old networks' mean of 0.04, 0.07 and 0.06.",
        "differ by 0.03, 0.08 and 0.05 at one standard deviation",
        "It is +0.08 +-0.03 above their mean over the twelve other draws, 2.9 of its errors; on the other two sets +0.07 +-0.08 and -0.04 +-0.05, and for the old networks -0.01 +-0.04, -0.04 +-0.12 and +0.11 +-0.05.",
        "are +0.02 +-0.03 above; five trained the old way, -0.05 +-0.04.",
        "lose 0.14, 0.54 and 0.48 of a point to the tile where the plan's row has 0.07, 0.47 and 0.52, and the old ones 0.14, 0.51 and 0.72 where it has 0.15, 0.54 and 0.62.",
        "lose +0.01 +-0.04, +0.03 +-0.10 and -0.24 +-0.13 more than the old ones over the draws",
        "It is right +0.29 +-0.09, +0.38 +-0.17 and +2.26 +-0.52 more often, for the row's 0.38, 0.48 and 2.12.",
        "HUNDREDTH OVER IT ON THE INVERTED SET. +0.02 +-0.04, +0.02 +-0.02 and +0.11 +-0.09 over the tile as budgeted on the cycle's draws",
        "pta_reference_cycle.py had 0.08, 0.16 and 0.09 over the row as budgeted.",
        "a calibration holds an hour, 30 minutes and 3 minutes for them in the cycle.",
        "drawn three times on its own draws: +0.08 +-0.11 from weights as written, +0.11 +-0.09 at the end of the cycle, and +0.30 +-0.09 after the hour's calibration.",
    ):
        assert said.count(words) == 1, words

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
