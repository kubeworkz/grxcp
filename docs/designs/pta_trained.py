"""
A network trained for the tile: what the training buys, and what version 2 still does.

B14 holds the interface chip to version 2 and lists what would reopen it.  First
on the list: a network trained with the tile's errors in the loop that does as
well at version 1.  Every network this plan had run was trained on its host and
met the tile afterwards.  grx930's harness has now trained them the other way.

MEASURED IN A MODEL, by grx930 (c930/doc/pta_error_model_design_note.md section
5, "A network trained for the tile", 2026-10-06; `sim/pta_mnist.sh DIR WORK
trained`).  On the working tile, 128 x 64 on two buses.  Five kinds of network a
data set, five networks a kind, seeds 1 to 5:

  trained as before   the 6-bit network this plan has always run.  Its training
                      stops at the first epoch whose held-out accuracy fails to rise
  eight epochs        the same seed's network again, from the same 8-bit one, for
                      eight epochs whatever the held-out accuracy does: with no
                      noise, and with Gaussian noise of 5, 10 and 20% of a layer's
                      rms on every sum the layer forms while it trains

The one with no noise is there to tell the noise from the epochs.  Each network's
accuracy on its host and on the tile, at version 1 and at version 2, as budgeted
and held: B16's three rows of a source at the end of B15's six minutes of drift.
Every network's own figure is here and not only its row's mean, because each
comparison below is between two rows of the same five seeds.

DERIVED here:

  a row's mean and    which have to be the two tables grx930 printed, cell for cell
  its error
  what a tile costs   a network's accuracy on its host less on the tile
  what the epochs     the eight-epoch network with no noise, less the one trained
  buy                 before
  what noise buys     a network trained with it, less the one trained with none
  what version 2      a network at version 2, less the same network at version 1
  buys
  B14's test          a network trained here at version 1, less the one trained
                      before at version 2

Each is taken seed by seed, and its error is the standard error of the five
differences.  That is not the two rows' errors in quadrature, which would treat
the seeds as strangers, and it can differ in the last digit from the difference
of two rounded means: grx930's note subtracts the means and says 0.47, 0.25, 0.64
and 1.68 where this says 0.46, 0.24, 0.65 and 1.67.

HOW SURE.  Five networks.  A difference has to be 2.8 of its own errors to be
outside chance at one in twenty, and well over a hundred differences are printed
below, so half a dozen would reach that with nothing behind them.  The readings
say which of theirs do and which do not.

WHAT THIS IS NOT: the tile in the loop.  The noise is Gaussian, the same fraction
on every layer and independent from sum to sum.  A tile's error has a quantiser's
steps, crosstalk that follows the image and a programming error that stays put
for a GEMM, and the trainer was shown none of those.  Eight epochs and the three
sizes of noise were chosen once and not searched.  The networks trained here were
held at B16's 1% and not at 2%, so whether a trained network needs B16 was not
run.  And it is still pta_workload.py's three data sets, 28 x 28 images through
one hidden layer.

Standard library only.  Run:  python3 docs/designs/pta_trained.py
"""
import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_laser as laser
import pta_tighten as tighten
import pta_version2 as version2
import pta_workload as workload

MNIST, FASHION, INVERTED = workload.MNIST, workload.FASHION, workload.INVERTED
WORKLOADS = workload.WORKLOADS
TILE = laser.SMALLER                           # 128 x 64: B10
BEFORE = "before"                              # trained as this plan's networks always were
CONTROL = 0.0                                  # eight epochs, no noise
NOISES = (0.05, 0.1, 0.2)                      # eight epochs, with that much
KINDS = (BEFORE, CONTROL) + NOISES
TILE_SIZED = 0.1                               # the probe puts a v1 tile's error at about a tenth of a sum's rms
EPOCHS = 8
SEEDS = 5
HOST, V1, V2, V1_HELD, V2_HELD = range(5)      # a network's five figures
ON_TILE = (V1, V2, V1_HELD, V2_HELD)
COLUMN = ("on its host", "v1", "version 2", "v1, held", "version 2, held")
T95 = 2.776                                    # Student's t, four degrees of freedom, one in twenty

# ---- grx930's figures ----------------------------------------------------------
# (data set) -> kind of network -> its five networks, seed 1 to 5 -> accuracy, percent:
# on its host, at v1, at version 2, v1 held, version 2 held.
NETS = {
    MNIST: {
        "before": ((97.44, 97.18, 97.27, 97.13, 97.19), (97.81, 97.49, 97.70, 97.39, 97.63),
                   (97.17, 96.84, 97.06, 96.84, 97.06), (97.43, 97.23, 97.27, 96.97, 97.19),
                   (97.41, 96.81, 97.21, 96.93, 97.22)),
        0.0: ((97.56, 97.41, 97.61, 97.41, 97.52), (97.76, 97.38, 97.50, 97.37, 97.44),
              (97.68, 97.49, 97.60, 97.44, 97.47), (98.12, 97.90, 98.04, 97.82, 97.93),
              (97.57, 97.44, 97.54, 97.25, 97.48)),
        0.05: ((97.74, 97.55, 97.56, 97.48, 97.42), (97.67, 97.51, 97.55, 97.36, 97.56),
               (97.66, 97.35, 97.48, 97.27, 97.46), (98.04, 97.82, 97.91, 97.81, 97.84),
               (97.62, 97.37, 97.50, 97.28, 97.42)),
        0.1: ((97.68, 97.43, 97.61, 97.31, 97.48), (97.90, 97.59, 97.66, 97.47, 97.69),
              (97.57, 97.45, 97.65, 97.43, 97.59), (97.89, 97.68, 97.84, 97.49, 97.88),
              (97.72, 97.31, 97.65, 97.52, 97.55)),
        0.2: ((97.74, 97.51, 97.64, 97.47, 97.52), (97.74, 97.50, 97.59, 97.54, 97.63),
              (97.67, 97.49, 97.52, 97.49, 97.59), (97.81, 97.58, 97.73, 97.54, 97.72),
              (97.83, 97.44, 97.68, 97.38, 97.54)),
    },
    FASHION: {
        "before": ((87.72, 86.13, 86.45, 85.72, 86.74), (87.32, 86.02, 86.83, 86.02, 86.90),
                   (86.87, 86.12, 86.94, 86.30, 87.09), (87.99, 87.00, 87.65, 86.53, 87.07),
                   (87.48, 86.33, 86.81, 85.96, 86.60)),
        0.0: ((87.37, 85.86, 86.74, 86.41, 87.19), (88.02, 86.17, 87.26, 86.07, 86.85),
              (87.89, 86.87, 87.34, 86.63, 87.15), (87.82, 86.79, 87.38, 86.61, 87.50),
              (87.19, 85.83, 86.91, 85.66, 86.53)),
        0.05: ((87.23, 85.67, 86.32, 86.37, 86.72), (88.14, 87.07, 87.64, 86.87, 87.39),
               (87.72, 86.95, 87.13, 86.17, 86.89), (87.92, 86.80, 87.66, 86.56, 87.34),
               (87.73, 86.53, 87.31, 86.38, 87.32)),
        0.1: ((87.43, 86.17, 87.02, 86.32, 86.94), (88.33, 87.36, 87.76, 87.22, 87.51),
              (87.63, 87.14, 87.63, 86.86, 87.36), (88.08, 87.06, 87.42, 87.13, 87.55),
              (87.96, 86.10, 87.23, 85.90, 86.97)),
        0.2: ((86.84, 86.31, 86.59, 86.36, 86.70), (87.08, 86.74, 86.97, 86.82, 86.85),
              (87.43, 87.21, 87.57, 87.15, 87.30), (87.42, 87.00, 87.09, 87.03, 87.14),
              (87.60, 86.82, 87.09, 87.02, 87.18)),
    },
    INVERTED: {
        "before": ((94.46, 92.91, 93.65, 91.07, 92.15), (93.64, 92.84, 93.26, 92.92, 93.14),
                   (91.32, 89.98, 90.63, 89.83, 90.05), (93.37, 92.01, 92.71, 91.55, 92.27),
                   (94.01, 92.92, 93.47, 93.44, 93.73)),
        0.0: ((96.36, 95.19, 95.90, 94.52, 95.29), (95.60, 94.39, 94.79, 93.84, 94.72),
              (95.26, 94.13, 94.47, 93.71, 94.30), (94.33, 91.97, 93.43, 91.00, 92.62),
              (96.19, 95.19, 95.82, 95.09, 95.63)),
        0.05: ((96.05, 95.00, 95.77, 94.58, 95.18), (96.08, 94.66, 95.39, 94.57, 95.16),
               (95.26, 94.11, 94.63, 93.76, 94.33), (94.20, 91.95, 93.26, 91.30, 92.50),
               (95.79, 94.16, 95.09, 94.26, 95.10)),
        0.1: ((96.10, 95.32, 95.67, 95.21, 95.52), (95.68, 94.75, 95.26, 94.72, 95.15),
              (94.89, 93.96, 94.38, 93.80, 94.24), (94.33, 93.22, 93.72, 92.24, 92.72),
              (95.92, 94.84, 95.28, 94.91, 95.50)),
        0.2: ((94.99, 94.19, 94.43, 94.05, 94.34), (93.39, 92.84, 93.25, 92.84, 93.11),
              (94.02, 93.61, 93.70, 93.26, 93.42), (92.87, 92.08, 92.30, 91.93, 92.23),
              (95.34, 94.53, 95.01, 94.53, 94.90)),
    },
}
# Epochs the networks trained before ran, seed 1 to 5: the rule stopped them.
STOPPED = {
    MNIST: (2, 5, 3, 2, 3),
    FASHION: (5, 3, 4, 4, 3),
    INVERTED: (3, 4, 3, 7, 4),
}
# The noise a training put on each layer's sums over its last epoch, as a fraction of
# their rms: asked -> seed 1 to 5 -> (layer 1, layer 2).
GOT = {
    MNIST: {
        0.0: ((0.0000, 0.0000), (0.0000, 0.0000), (0.0000, 0.0000), (0.0000, 0.0000), (0.0000, 0.0000)),
        0.05: ((0.0499, 0.0497), (0.0499, 0.0498), (0.0499, 0.0497), (0.0499, 0.0498), (0.0499, 0.0499)),
        0.1: ((0.0998, 0.0995), (0.0998, 0.0997), (0.0998, 0.0996), (0.0999, 0.0995), (0.0998, 0.0998)),
        0.2: ((0.1994, 0.1984), (0.1993, 0.1987), (0.1994, 0.1990), (0.1995, 0.1987), (0.1995, 0.1993)),
    },
    FASHION: {
        0.0: ((0.0000, 0.0000), (0.0000, 0.0000), (0.0000, 0.0000), (0.0000, 0.0000), (0.0000, 0.0000)),
        0.05: ((0.0499, 0.0499), (0.0499, 0.0497), (0.0499, 0.0499), (0.0500, 0.0497), (0.0500, 0.0498)),
        0.1: ((0.0996, 0.0998), (0.0997, 0.0997), (0.0997, 0.1000), (0.0998, 0.0995), (0.0998, 0.0996)),
        0.2: ((0.1993, 0.1988), (0.1993, 0.1984), (0.1993, 0.1985), (0.1995, 0.1987), (0.1992, 0.1992)),
    },
    INVERTED: {
        0.0: ((0.0000, 0.0000), (0.0000, 0.0000), (0.0000, 0.0000), (0.0000, 0.0000), (0.0000, 0.0000)),
        0.05: ((0.0500, 0.0499), (0.0498, 0.0498), (0.0498, 0.0500), (0.0500, 0.0498), (0.0498, 0.0500)),
        0.1: ((0.0992, 0.0999), (0.0999, 0.0996), (0.1000, 0.0999), (0.0998, 0.0997), (0.0993, 0.1000)),
        0.2: ((0.1994, 0.2003), (0.1984, 0.1990), (0.1992, 0.1997), (0.1998, 0.1990), (0.1997, 0.1998)),
    },
}
# The two tables grx930's harness printed of them, mean and standard error.  Accuracy:
# on its host, v1, version 2, v1 held, version 2 held.  Then points lost against the
# network's own accuracy on its host, the same four; the epochs it ran; and, for a
# network trained here, the noise put on each layer, percent of its sums' rms.
PRINTED = {
    MNIST: {
        "before": (((97.45, 0.10), (97.11, 0.13), (97.30, 0.11), (97.05, 0.10), (97.26, 0.10)),
                   ((0.34, 0.07), (0.15, 0.02), (0.40, 0.03), (0.19, 0.03), (3.00, 0.55))),
        0.0: (((97.74, 0.10), (97.52, 0.10), (97.66, 0.10), (97.46, 0.10), (97.57, 0.09)),
              ((0.21, 0.04), (0.08, 0.05), (0.28, 0.04), (0.17, 0.05), (8.00, 0.00), (0.00, 0.00), (0.00, 0.00))),
        0.05: (((97.75, 0.08), (97.52, 0.08), (97.60, 0.08), (97.44, 0.10), (97.54, 0.08)),
               ((0.23, 0.03), (0.15, 0.01), (0.31, 0.03), (0.21, 0.03), (8.00, 0.00), (4.99, 0.00), (4.98, 0.00))),
        0.1: (((97.75, 0.06), (97.49, 0.06), (97.68, 0.04), (97.44, 0.04), (97.64, 0.07)),
              ((0.26, 0.05), (0.07, 0.05), (0.31, 0.06), (0.11, 0.05), (8.00, 0.00), (9.98, 0.00), (9.96, 0.01))),
        0.2: (((97.76, 0.03), (97.50, 0.02), (97.63, 0.04), (97.48, 0.03), (97.60, 0.04)),
              ((0.25, 0.04), (0.13, 0.02), (0.27, 0.05), (0.16, 0.04), (8.00, 0.00), (19.94, 0.00), (19.88, 0.02))),
    },
    FASHION: {
        "before": (((87.48, 0.19), (86.32, 0.18), (86.94, 0.20), (86.11, 0.14), (86.88, 0.09)),
                   ((1.16, 0.14), (0.54, 0.22), (1.37, 0.23), (0.60, 0.23), (3.80, 0.37))),
        0.0: (((87.66, 0.16), (86.30, 0.22), (87.13, 0.13), (86.28, 0.18), (87.04, 0.16)),
              ((1.35, 0.16), (0.53, 0.08), (1.38, 0.17), (0.61, 0.17), (8.00, 0.00), (0.00, 0.00), (0.00, 0.00))),
        0.05: (((87.75, 0.15), (86.60, 0.25), (87.21, 0.24), (86.47, 0.12), (87.13, 0.14)),
               ((1.14, 0.13), (0.54, 0.11), (1.28, 0.11), (0.62, 0.08), (8.00, 0.00), (4.99, 0.00), (4.98, 0.00))),
        0.1: (((87.89, 0.16), (86.77, 0.26), (87.41, 0.13), (86.69, 0.25), (87.27, 0.13)),
              ((1.12, 0.22), (0.47, 0.13), (1.20, 0.22), (0.62, 0.13), (8.00, 0.00), (9.97, 0.00), (9.97, 0.01))),
        0.2: (((87.27, 0.14), (86.82, 0.15), (87.06, 0.16), (86.88, 0.14), (87.03, 0.11)),
              ((0.46, 0.10), (0.21, 0.11), (0.40, 0.06), (0.24, 0.05), (8.00, 0.00), (19.93, 0.00), (19.87, 0.01))),
    },
    INVERTED: {
        "before": (((93.36, 0.54), (92.13, 0.56), (92.74, 0.55), (91.76, 0.65), (92.27, 0.63)),
                   ((1.23, 0.13), (0.62, 0.07), (1.60, 0.50), (1.09, 0.36), (4.20, 0.73))),
        0.0: (((95.55, 0.36), (94.17, 0.59), (94.88, 0.46), (93.63, 0.70), (94.51, 0.53)),
              ((1.37, 0.25), (0.67, 0.11), (1.92, 0.38), (1.04, 0.19), (8.00, 0.00), (0.00, 0.00), (0.00, 0.00))),
        0.05: (((95.48, 0.35), (93.98, 0.53), (94.83, 0.43), (93.69, 0.62), (94.45, 0.51)),
               ((1.50, 0.21), (0.65, 0.11), (1.78, 0.28), (1.02, 0.17), (8.00, 0.00), (4.99, 0.00), (4.99, 0.00))),
        0.1: (((95.38, 0.33), (94.42, 0.37), (94.86, 0.36), (94.18, 0.54), (94.63, 0.53)),
              ((0.97, 0.06), (0.52, 0.05), (1.21, 0.22), (0.76, 0.22), (8.00, 0.00), (9.96, 0.02), (9.98, 0.01))),
        0.2: (((94.12, 0.47), (93.45, 0.45), (93.74, 0.47), (93.32, 0.46), (93.60, 0.47)),
              ((0.67, 0.08), (0.38, 0.08), (0.80, 0.07), (0.52, 0.07), (8.00, 0.00), (19.93, 0.02), (19.96, 0.03))),
    },
}


# ---- what follows from them ---------------------------------------------------------
def stat(xs):
    """Five figures' mean and standard error."""
    n = len(xs)
    m = sum(xs) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def acc(w, kind, col):
    """A kind of network's accuracy in one column."""
    return stat([net[col] for net in NETS[w][kind]])


def over(w, a, ca, b, cb):
    """Kind a in column ca, less kind b in column cb, seed by seed."""
    return stat([x[ca] - y[cb] for x, y in zip(NETS[w][a], NETS[w][b])])


def lost(w, kind, col):
    """What the tile costs a network: its accuracy on its host less on the tile."""
    return over(w, kind, HOST, kind, col)


def lost_moved(w, kind, col, than=BEFORE):
    """What the tile costs one kind, less what it costs another, seed by seed."""
    return stat([(x[HOST] - x[col]) - (y[HOST] - y[col]) for x, y in zip(NETS[w][kind], NETS[w][than])])


def epochs_buy(w, col):
    """Eight epochs with no noise, over the network trained before."""
    return over(w, CONTROL, col, BEFORE, col)


def noise_buys(w, f, col):
    """A network trained with that noise, over the one trained with none."""
    return over(w, f, col, CONTROL, col)


def v2_buys(w, kind, held=False):
    """A network at version 2, over the same network at version 1."""
    return over(w, kind, V2_HELD, kind, V1_HELD) if held else over(w, kind, V2, kind, V1)


def b14_test(w, kind=TILE_SIZED, held=False):
    """A network trained here at version 1, over the one trained before at version 2."""
    return over(w, kind, V1_HELD, BEFORE, V2_HELD) if held else over(w, kind, V1, BEFORE, V2)


def both(w, kind=TILE_SIZED):
    """A network trained here at version 2, over the one trained before at version 1."""
    return over(w, kind, V2, BEFORE, V1)


def holding_adds(w, kind, version):
    """What a source and an interval's drift add to what a version costs a network."""
    return over(w, kind, V1 if version == 1 else V2, kind, V1_HELD if version == 1 else V2_HELD)


def errors(x):
    """A difference in its own standard errors."""
    return x[0] / x[1] if x[1] else 0.0


def clear(x):
    """Whether five networks put a difference outside chance at one in twenty."""
    return abs(errors(x)) > T95


def stopped(w):
    """Epochs the networks trained before ran: the fewest, the most."""
    return min(STOPPED[w]), max(STOPPED[w])


def got(w, f):
    """The noise a training put on a layer's sums, of what was asked: the least, the most."""
    r = [g / f for pair in GOT[w][f] for g in pair]
    return min(r), max(r)


def label(kind):
    if kind == BEFORE:
        return "trained as before"
    return f"{EPOCHS} epochs, no noise" if kind == CONTROL else f"{EPOCHS} epochs, noise of {kind:.0%}"


def pm(x):
    return f"{x[0]:.2f} +-{x[1]:.2f}"


def dpm(x):
    return f"{x[0]:+.2f} +-{x[1]:.2f}"


def r2(x):
    return round(x[0], 2), round(x[1], 2)


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def table(rows, what, cols, fmt=pm):
    for w in WORKLOADS:
        print(f"  {w:<26}" + "".join(f"{COLUMN[c]:>18}" for c in cols))
        for key, name in rows:
            print(f"  {name:<26}" + "".join(f"{fmt(what(w, key, c)):>18}" for c in cols))


def main():
    print("A network trained for the tile: what the training buys, and what version 2 still does.")
    print(f"The working tile, {laser.name(TILE)} on two buses.  Five networks a row, mean and standard error.")
    kinds = [(k, label(k)) for k in KINDS]

    section("1. Accuracy, percent")
    table(kinds, acc, range(5))
    print("  Epochs the networks trained before ran: " + ", ".join(
        f"{stopped(w)[0]} to {stopped(w)[1]} on {w}" for w in WORKLOADS) + ".")
    print("  Noise a training put on a layer's sums, of what was asked: " + ", ".join(
        f"{min(got(w, f)[0] for w in WORKLOADS):.1%} to {max(got(w, f)[1] for w in WORKLOADS):.1%} at {f:.0%}" for f in NOISES) + ".")

    section("2. What the tile costs a network: points lost against its own accuracy on its host")
    table(kinds, lost, ON_TILE)
    print("  And eight epochs with no noise, less trained as before, seed by seed:")
    for w in WORKLOADS:
        print(f"  {w:<26}" + "".join(f"{dpm(lost_moved(w, CONTROL, c)):>18}" for c in ON_TILE))

    section("3. What eight epochs buy: no noise, over the network trained before")
    print(f"  {'':<26}" + "".join(f"{COLUMN[c]:>18}" for c in range(5)))
    for w in WORKLOADS:
        print(f"  {w:<26}" + "".join(f"{dpm(epochs_buy(w, c)):>18}" for c in range(5)))
        print(f"  {'  in its errors':<26}" + "".join(f"{errors(epochs_buy(w, c)):>+18.1f}" for c in range(5)))

    section("4. What noise buys: a network trained with it, over the one trained with none")
    for w in WORKLOADS:
        print(f"  {w:<26}" + "".join(f"{COLUMN[c]:>18}" for c in range(5)))
        for f in NOISES:
            print(f"  {f'noise of {f:.0%}':<26}" + "".join(f"{dpm(noise_buys(w, f, c)):>18}" for c in range(5)))
            print(f"  {'  in its errors':<26}" + "".join(f"{errors(noise_buys(w, f, c)):>+18.1f}" for c in range(5)))

    section("5. What version 2 buys a network: the same network, at version 2 over version 1")
    print(f"  {'':<26}" + "".join(f"{w:>18}{'held':>18}" for w in WORKLOADS))
    for k, name in kinds:
        print(f"  {name:<26}" + "".join(f"{dpm(v2_buys(w, k)):>18}{dpm(v2_buys(w, k, True)):>18}" for w in WORKLOADS))

    section("6. B14's test: a network trained here at version 1, over the one trained before at version 2")
    print(f"  {'':<26}" + "".join(f"{w:>18}{'held':>18}" for w in WORKLOADS))
    for k, name in kinds[1:]:
        print(f"  {name:<26}" + "".join(f"{dpm(b14_test(w, k)):>18}{dpm(b14_test(w, k, True)):>18}" for w in WORKLOADS))
    print("  And at version 2, over the one trained before at version 1:")
    for k, name in kinds[1:]:
        print(f"  {name:<26}" + "".join(f"{dpm(both(w, k)):>18}{'':>18}" for w in WORKLOADS))

    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        findings()
    print(said.getvalue(), end="")
    checks(" ".join(said.getvalue().split()))


def findings():
    m, f, i = WORKLOADS
    ten = TILE_SIZED
    print()
    print("What this says, seven readings.")
    print()
    print("  1. THE RULE THAT STOPS A TRAINING HAD LEFT THE INVERTED SET'S NETWORKS HALF TRAINED.  Eight")
    print(f"     epochs with no noise make them {dpm(epochs_buy(i, HOST))} points better on their host,")
    print(f"     {dpm(epochs_buy(i, V1))} on a v1 tile and {dpm(epochs_buy(i, V2))} at version 2.  They had stopped")
    print(f"     after {stopped(i)[0]} to {stopped(i)[1]} epochs.  On MNIST it is {dpm(epochs_buy(m, HOST))} and {dpm(epochs_buy(m, V1))}, two to three")
    print(f"     of its errors.  On Fashion-MNIST it is {dpm(epochs_buy(f, HOST))} and {dpm(epochs_buy(f, V1))}: nothing that")
    print("     five networks can tell.  Every accuracy this plan gives for the inverted set is")
    print("     about two points low for it.")
    print()
    print("  2. WHAT A TILE COSTS A NETWORK IS WHAT WAS SAID.  Held at version 2, the better")
    print(f"     trained networks lose {lost(m, CONTROL, V2_HELD)[0]:.2f}, {lost(f, CONTROL, V2_HELD)[0]:.2f} and {lost(i, CONTROL, V2_HELD)[0]:.2f} points where the old ones lose {lost(m, BEFORE, V2_HELD)[0]:.2f},")
    print(f"     {lost(f, BEFORE, V2_HELD)[0]:.2f} and {lost(i, BEFORE, V2_HELD)[0]:.2f}.  As budgeted at v1 it is {lost(m, CONTROL, V1)[0]:.2f}, {lost(f, CONTROL, V1)[0]:.2f} and {lost(i, CONTROL, V1)[0]:.2f} for {lost(m, BEFORE, V1)[0]:.2f}, {lost(f, BEFORE, V1)[0]:.2f} and {lost(i, BEFORE, V1)[0]:.2f},")
    worst = max(abs(errors(lost_moved(w, CONTROL, c))) for w in WORKLOADS for c in (V1, V2, V2_HELD))
    print(f"     and none of those nine moves by more than {worst:.1f} of its errors.  The budget's prices")
    print("     in points lost stand.  It was the accuracies that were low.")
    print()
    print(f"  3. NOISE OF THE TILE'S SIZE IS WORTH 0.3 TO 0.5 OF A POINT ON FASHION-MNIST, AND IS NOT")
    print(f"     SHOWN ELSEWHERE.  At {ten:.0%}, over the same network trained with none: {dpm(noise_buys(f, ten, V1))} on a")
    print(f"     v1 tile, which is {errors(noise_buys(f, ten, V1)):.1f} of its errors and short of the {T95:.1f} five networks need, and")
    print(f"     {dpm(noise_buys(f, ten, V2))} at version 2, which is {errors(noise_buys(f, ten, V2)):.1f}.  On the inverted set {dpm(noise_buys(i, ten, V1))} and")
    print(f"     {dpm(noise_buys(i, ten, V2))}, and on MNIST {dpm(noise_buys(m, ten, V1))} and {dpm(noise_buys(m, ten, V2))}.  On the host it moves a")
    print(f"     network by {dpm(noise_buys(m, ten, HOST))}, {dpm(noise_buys(f, ten, HOST))} and {dpm(noise_buys(i, ten, HOST))}.  At 5% nothing moves by two")
    print("     of its errors on any set.")
    print()
    twenty = 0.2
    print(f"  4. TWICE THAT LOSES LESS TO THE TILE AND IS NOT A BETTER NETWORK.  At {twenty:.0%} a network loses")
    print(f"     {lost(f, twenty, V1)[0]:.2f} points to a v1 tile on Fashion-MNIST where the one with no noise loses {lost(f, CONTROL, V1)[0]:.2f}, and")
    print(f"     {lost(i, twenty, V1)[0]:.2f} for {lost(i, CONTROL, V1)[0]:.2f} on the inverted set.  It starts {-noise_buys(f, twenty, HOST)[0]:.2f} and {-noise_buys(i, twenty, HOST)[0]:.2f} lower on its")
    print(f"     host.  On the tile at version 2 it is {dpm(noise_buys(f, twenty, V2))} and {dpm(noise_buys(i, twenty, V2))} against")
    print(f"     no noise at all.  Only at v1 on Fashion-MNIST is it ahead, {dpm(noise_buys(f, twenty, V1))}, and there")
    print(f"     the {ten:.0%} network is level with it.  Points lost is the wrong score for a network")
    print("     that was trained to lose fewer.")
    print()
    print("  5. VERSION 2 BUYS A TRAINED NETWORK WHAT IT BOUGHT THE OTHERS.  The same network on the")
    print(f"     two tiles.  Trained with {ten:.0%} of noise: {dpm(v2_buys(m, ten))}, {dpm(v2_buys(f, ten))} and {dpm(v2_buys(i, ten))}.")
    print(f"     Trained as before: {dpm(v2_buys(m, BEFORE))}, {dpm(v2_buys(f, BEFORE))} and {dpm(v2_buys(i, BEFORE))}.  Eight epochs and")
    print(f"     no noise: {dpm(v2_buys(m, CONTROL))}, {dpm(v2_buys(f, CONTROL))} and {dpm(v2_buys(i, CONTROL))}.  All fifteen rows are over")
    least = min(errors(v2_buys(w, k)) for w in WORKLOADS for k in KINDS)
    print(f"     {least:.1f} of their errors.  It is less on the inverted set, {v2_buys(i, ten)[0]:.2f} for {v2_buys(i, BEFORE)[0]:.2f}.  And on the two")
    print(f"     harder sets it is under half only for the {twenty:.0%} network, {v2_buys(f, twenty)[0]:.2f} and {v2_buys(i, twenty)[0]:.2f}, which gave up")
    print("     accuracy on its host to need it less.")
    print()
    print("  6. B14'S TEST, AS B14 WORDED IT, IS MET.  A network trained with the tile's size of")
    print(f"     noise, at version 1, against the old network at version 2: {dpm(b14_test(m))} on MNIST,")
    print(f"     {dpm(b14_test(f))} on Fashion-MNIST and {dpm(b14_test(i))} on the inverted set.  Level on the")
    print(f"     one, inside its error, and ahead on the other two.  Held, {dpm(b14_test(m, held=True))}, {dpm(b14_test(f, held=True))} and")
    print(f"     {dpm(b14_test(i, held=True))}.  On Fashion-MNIST it is the noise that does it: with eight epochs")
    print(f"     and none the network is {dpm(b14_test(f, CONTROL))}, behind by {-errors(b14_test(f, CONTROL)):.1f} of its errors.  On the")
    print("     other two it is the epochs.")
    print()
    print("  7. AND TRAINING DOES NOT STAND IN FOR VERSION 2.  IT ADDS TO IT.  The test did not ask")
    print("     what version 2 is worth to the trained network, and reading 5 is the answer: what")
    print(f"     it was worth before.  The {ten:.0%} network at version 2 is {dpm(both(m))}, {dpm(both(f))} and")
    print(f"     {dpm(both(i))} over the old one at v1, and {dpm(over(m, ten, V2, BEFORE, V2))}, {dpm(over(f, ten, V2, BEFORE, V2))} and {dpm(over(i, ten, V2, BEFORE, V2))} over")
    print("     the old one at version 2.  Whether that reopens B14 is the plan's to say.")


def checks(said):
    """Every claim above, as an assert.  `said` is the readings as printed, on one line."""
    m, f, i = WORKLOADS
    ten, twenty = TILE_SIZED, 0.2
    assert TILE == (128, 64) and KINDS == ("before", 0.0, 0.05, 0.1, 0.2) and EPOCHS == 8
    assert set(NETS) == set(STOPPED) == set(GOT) == set(PRINTED) == set(WORKLOADS)
    assert all(set(NETS[w]) == set(PRINTED[w]) == set(KINDS) and set(GOT[w]) == set(KINDS[1:]) for w in WORKLOADS)
    assert all(len(NETS[w][k]) == SEEDS and all(len(net) == 5 for net in NETS[w][k]) for w in WORKLOADS for k in KINDS)

    # 0. The figures are grx930's, and agree with what the plan already holds.
    #    Every row's mean and error, computed here from its five networks, is the
    #    cell grx930's harness printed: accuracy, points lost, epochs and noise.
    def same(a, b):
        return abs(a[0] - b[0]) < 0.0051 and abs(a[1] - b[1]) < 0.0051

    cells = 0
    for w in WORKLOADS:
        for k in KINDS:
            accs, losts = PRINTED[w][k]
            assert len(accs) == 5 and len(losts) == (5 if k == BEFORE else 7), (w, k)
            for c in range(5):
                assert same(acc(w, k, c), accs[c]), (w, k, c)
            for n, c in enumerate(ON_TILE):
                assert same(lost(w, k, c), losts[n]), (w, k, c)
            assert same(stat(STOPPED[w]) if k == BEFORE else (EPOCHS, 0.0), losts[4]), (w, k)
            if k != BEFORE:
                for layer in range(2):
                    assert same(stat([100 * g[layer] for g in GOT[w][k]]), losts[5 + layer]), (w, k, layer)
            cells += len(accs) + len(losts)
    assert cells == 3 * (10 + 4 * 12) == 174
    #    Counted and summed, as grx930's lines gave them.
    figures = [x for w in NETS.values() for k in w.values() for net in k for x in net]
    assert len(figures) == 375 and abs(sum(figures) - 34807.31) < 1e-6
    printed = [x for w in PRINTED.values() for k in w.values() for part in k for cell in part for x in cell]
    assert len(printed) == 348 and abs(sum(printed) - 7345.70) < 1e-6
    #    The networks trained before are pta_workload.py's five, on the host.
    assert all(tuple(net[HOST] for net in NETS[w][BEFORE]) == workload.HOST[w] for w in WORKLOADS)
    #    What a tile costs them is pta_tighten.py's at each version as budgeted,
    #    and pta_version2.py's as held: what they lose, and what holding adds.
    for n, w in enumerate(WORKLOADS):
        assert r2(lost(w, BEFORE, V1)) == tighten.LOST["v1"][n] and r2(lost(w, BEFORE, V2)) == tighten.LOST["adc_noise"][n]
        assert r2(lost(w, BEFORE, V1_HELD)) == version2.held(w, 1)[0] and r2(lost(w, BEFORE, V2_HELD)) == version2.held(w, 2)[0]
        assert all(r2(holding_adds(w, BEFORE, v)) == version2.held(w, v)[1] for v in (1, 2))
    assert version2.ROWS_V2 == (0.01, 0.05, 0.05) and version2.INTERVAL_S == 360
    #    A training's noise was what it was asked for, to a part in a hundred, and
    #    none where none was asked.
    assert all(g == (0.0, 0.0) for w in WORKLOADS for g in GOT[w][CONTROL])
    assert all(0.99 < got(w, x)[0] <= got(w, x)[1] < 1.01 for w in WORKLOADS for x in NOISES)
    #    Over both layers and all three sets: 99.2% of it at the least, 100.2% at the most.
    assert [(round(min(got(w, x)[0] for w in WORKLOADS), 4), round(max(got(w, x)[1] for w in WORKLOADS), 4))
            for x in NOISES] == [(0.994, 1.0), (0.992, 1.0), (0.992, 1.0015)]
    assert all(len(STOPPED[w]) == SEEDS for w in WORKLOADS)
    #    Why seed by seed: a row's scatter is mostly its networks' own.  On the
    #    inverted set seed 4 is the lowest of all four rows trained here.
    assert all(min(range(SEEDS), key=lambda n: NETS[i][k][n][HOST]) == 3 for k in KINDS[1:])
    assert abs(T95 - 2.776) < 1e-9

    # 1. Reading 1.  The epochs.
    assert [r2(epochs_buy(i, c)) for c in (HOST, V1, V2)] == [(2.19, 0.49), (2.04, 0.68), (2.14, 0.52)]
    assert [r2(epochs_buy(m, c)) for c in (HOST, V1)] == [(0.29, 0.14), (0.41, 0.15)]
    assert [r2(epochs_buy(f, c)) for c in (HOST, V1)] == [(0.18, 0.28), (-0.02, 0.22)]
    assert [c for c in range(5) if not clear(epochs_buy(i, c))] == [V1_HELD] and errors(epochs_buy(i, V1_HELD)) > 2.2
    assert all(2 < errors(epochs_buy(m, c)) < 3 and not clear(epochs_buy(m, c)) for c in range(5))
    assert all(abs(errors(epochs_buy(f, c))) < 1.6 for c in range(5))
    assert [stopped(w) for w in WORKLOADS] == [(2, 5), (3, 5), (3, 7)]
    assert all(max(STOPPED[w]) < EPOCHS for w in WORKLOADS)
    assert [round(stat(STOPPED[w])[0], 1) for w in WORKLOADS] == [3.0, 3.8, 4.2]
    #    About two points low: on its host and on every tile, 1.9 to 2.2.
    assert all(1.85 < epochs_buy(i, c)[0] < 2.25 for c in range(5))
    assert [round(acc(i, k, HOST)[0], 1) for k in (BEFORE, CONTROL)] == [93.4, 95.5]
    #    Half of what set the inverted set's networks behind MNIST's on the host
    #    was this: 4.1 points behind as they were trained, 2.2 at eight epochs.
    behind = [acc(m, k, HOST)[0] - acc(i, k, HOST)[0] for k in (BEFORE, CONTROL)]
    assert [round(x, 2) for x in behind] == [4.09, 2.19] and 0.45 < 1 - behind[1] / behind[0] < 0.5
    hosts = [net[HOST] for net in NETS[i][CONTROL]]
    assert (min(hosts), max(hosts)) == (94.33, 96.36) and (min(workload.HOST[i]), max(workload.HOST[i])) == (91.32, 94.46)
    #    On that set the network that had run longest gained least, and on MNIST too.
    for w in (m, i):
        gains = [x[HOST] - y[HOST] for x, y in zip(NETS[w][CONTROL], NETS[w][BEFORE])]
        assert gains.index(min(gains)) == STOPPED[w].index(max(STOPPED[w]))

    # 2. Reading 2.  What a tile costs.
    assert [round(lost(w, CONTROL, V2_HELD)[0], 2) for w in WORKLOADS] == [0.17, 0.61, 1.04]
    assert [round(lost(w, BEFORE, V2_HELD)[0], 2) for w in WORKLOADS] == [0.19, 0.60, 1.09]
    assert [round(lost(w, CONTROL, V1)[0], 2) for w in WORKLOADS] == [0.21, 1.35, 1.37]
    assert [round(lost(w, BEFORE, V1)[0], 2) for w in WORKLOADS] == [0.34, 1.16, 1.23]
    assert [r2(lost_moved(w, CONTROL, V2_HELD)) for w in WORKLOADS] == [(-0.02, 0.06), (0.02, 0.36), (-0.06, 0.33)]
    assert [r2(lost_moved(w, CONTROL, V1)) for w in WORKLOADS] == [(-0.13, 0.09), (0.20, 0.11), (0.15, 0.25)]
    nine = [lost_moved(w, CONTROL, c) for w in WORKLOADS for c in (V1, V2, V2_HELD)]
    assert len(nine) == 9 and not any(clear(x) for x in nine) and 1.7 < max(abs(errors(x)) for x in nine) < 1.9
    #    The tenth, MNIST's at v1 held, does move: 0.12 of a point less.
    assert r2(lost_moved(m, CONTROL, V1_HELD)) == (-0.12, 0.03) and clear(lost_moved(m, CONTROL, V1_HELD))
    assert not clear(lost_moved(f, CONTROL, V1_HELD)) and not clear(lost_moved(i, CONTROL, V1_HELD))

    # 3. Reading 3.  Noise of the tile's size.
    assert [r2(noise_buys(f, ten, c)) for c in (V1, V2)] == [(0.46, 0.18), (0.29, 0.07)]
    assert not clear(noise_buys(f, ten, V1)) and 2.4 < errors(noise_buys(f, ten, V1)) < 2.6
    #    Under half of the point v1 costs there.
    assert 0.35 < noise_buys(f, ten, V1)[0] / lost(f, BEFORE, V1)[0] < 0.45
    assert clear(noise_buys(f, ten, V2)) and 3.8 < errors(noise_buys(f, ten, V2)) < 4.0
    assert [r2(noise_buys(i, ten, c)) for c in (V1, V2)] == [(0.24, 0.28), (-0.02, 0.18)]
    assert [r2(noise_buys(m, ten, c)) for c in (V1, V2)] == [(-0.03, 0.07), (0.02, 0.06)]
    assert [r2(noise_buys(w, ten, HOST)) for w in WORKLOADS] == [(0.01, 0.08), (0.23, 0.17), (-0.16, 0.09)]
    assert not any(clear(noise_buys(w, ten, HOST)) for w in WORKLOADS)
    #    Of the fifteen cells at 10%, one is clear: Fashion-MNIST at version 2.
    assert [(w, c) for w in WORKLOADS for c in range(5) if clear(noise_buys(w, ten, c))] == [(f, V2)]
    assert all(abs(noise_buys(m, ten, c)[0]) < 0.075 for c in range(5))
    assert all(abs(errors(noise_buys(w, 0.05, c))) < 2 for w in WORKLOADS for c in range(5))
    #    Held, it is 0.41 and 0.54 at v1, at two of their errors.
    assert [r2(noise_buys(w, ten, V1_HELD)) for w in (f, i)] == [(0.41, 0.21), (0.54, 0.26)]
    assert not any(clear(noise_buys(w, ten, V1_HELD)) for w in (f, i))

    # 4. Reading 4.  Twice the tile's size.
    assert [round(lost(w, twenty, V1)[0], 2) for w in (f, i)] == [0.46, 0.67]
    assert [r2(noise_buys(w, twenty, HOST)) for w in (f, i)] == [(-0.38, 0.22), (-1.43, 0.22)]
    assert [r2(noise_buys(w, twenty, V2)) for w in (f, i)] == [(-0.06, 0.11), (-1.14, 0.16)]
    assert r2(noise_buys(f, twenty, V1)) == (0.51, 0.13) and clear(noise_buys(f, twenty, V1))
    assert abs(acc(f, twenty, V1)[0] - acc(f, ten, V1)[0]) < 0.06
    #    On the inverted set it is behind the network with no noise on every tile.
    assert all(noise_buys(i, twenty, c)[0] < -0.3 for c in range(5))
    assert all(lost(w, twenty, c)[0] < 0.6 * lost(w, CONTROL, c)[0] for w in (f, i) for c in ON_TILE)

    # 5. Reading 5.  Version 2, to each kind of network.
    assert [r2(v2_buys(w, ten)) for w in WORKLOADS] == [(0.19, 0.04), (0.65, 0.15), (0.44, 0.03)]
    assert [r2(v2_buys(w, BEFORE)) for w in WORKLOADS] == [(0.19, 0.06), (0.62, 0.10), (0.61, 0.06)]
    assert [r2(v2_buys(w, CONTROL)) for w in WORKLOADS] == [(0.13, 0.02), (0.82, 0.13), (0.71, 0.20)]
    assert all(clear(v2_buys(w, k)) for w in WORKLOADS for k in KINDS)
    assert 3.0 < min(errors(v2_buys(w, k)) for w in WORKLOADS for k in KINDS) < 3.2
    #    What it bought the old networks is what pta_tighten.py had, in its own table.
    assert [r2(v2_buys(w, BEFORE)) for w in WORKLOADS] == list(tighten.TOGETHER["adc_noise"])
    assert [round(v2_buys(w, twenty)[0], 2) for w in (f, i)] == [0.25, 0.29]
    assert all(v2_buys(w, twenty)[0] < 0.5 * v2_buys(w, BEFORE)[0] for w in (f, i))
    assert all(v2_buys(w, k)[0] > 0.7 * v2_buys(w, BEFORE)[0] for w in (f, i) for k in (CONTROL, 0.05, ten))
    #    On MNIST it is 0.08 to 0.19 of a point whatever the training, and clear every time.
    assert [round(v2_buys(m, k)[0], 2) for k in KINDS] == [0.19, 0.13, 0.08, 0.19, 0.13]
    #    Held, the same: 0.19, 0.58 and 0.45 with 10% of noise.
    assert [r2(v2_buys(w, ten, True)) for w in WORKLOADS] == [(0.19, 0.06), (0.58, 0.13), (0.45, 0.05)]
    assert all(clear(v2_buys(w, ten, True)) for w in WORKLOADS)

    # 6. Reading 6.  B14's test.
    assert [r2(b14_test(w)) for w in WORKLOADS] == [(0.19, 0.10), (-0.17, 0.24), (1.67, 0.46)]
    assert [r2(b14_test(w, held=True)) for w in WORKLOADS] == [(0.19, 0.10), (-0.19, 0.18), (1.91, 0.68)]
    assert abs(errors(b14_test(f))) < 1 and b14_test(m)[0] > 0 and clear(b14_test(i))
    assert not clear(b14_test(m)) and not clear(b14_test(f))
    assert r2(b14_test(f, CONTROL)) == (-0.63, 0.16) and clear(b14_test(f, CONTROL))
    assert 3.9 < -errors(b14_test(f, CONTROL)) < 4.1
    #    On the other two the network with no noise is already ahead of the old one at version 2.
    assert [r2(b14_test(w, CONTROL)) for w in (m, i)] == [(0.22, 0.16), (1.43, 0.68)]
    assert r2(b14_test(f, twenty)) == (-0.12, 0.15)

    # 7. Reading 7.  Training and version 2 add.
    assert [r2(both(w)) for w in WORKLOADS] == [(0.57, 0.12), (1.09, 0.24), (2.73, 0.45)]
    assert [r2(over(w, ten, V2, BEFORE, V2)) for w in WORKLOADS] == [(0.38, 0.11), (0.48, 0.20), (2.12, 0.45)]
    assert all(clear(both(w)) for w in WORKLOADS)
    #    Both together are what each buys alone, summed, by construction of the differences.
    for w in WORKLOADS:
        assert abs(both(w)[0] - (v2_buys(w, ten)[0] + over(w, ten, V1, BEFORE, V1)[0])) < 1e-9
    #    Held at version 2, which is what the chip is held to: 97.64, 87.27 and
    #    94.63%, and that over the old networks there.
    assert [round(acc(w, ten, V2_HELD)[0], 2) for w in WORKLOADS] == [97.64, 87.27, 94.63]
    assert [round(acc(w, BEFORE, V2_HELD)[0], 2) for w in WORKLOADS] == [97.26, 86.88, 92.27]
    assert [r2(over(w, ten, V2_HELD, BEFORE, V2_HELD)) for w in WORKLOADS] == [(0.38, 0.11), (0.39, 0.07), (2.36, 0.65)]
    assert all(clear(over(w, ten, V2_HELD, BEFORE, V2_HELD)) for w in WORKLOADS)
    assert [round(lost(w, ten, V2_HELD)[0], 2) for w in WORKLOADS] == [0.11, 0.62, 0.76]
    #    The best row on a version 2 tile, held, is the 10% network on every set.
    assert all(max(KINDS, key=lambda k: acc(w, k, V2_HELD)[0]) == ten for w in WORKLOADS)

    # 8. And the readings say those figures, each in its place: a reading that
    #    printed another column's would pass everything above.
    for words in (
        "make them +2.19 +-0.49 points better on their host, +2.04 +-0.68 on a v1 tile and +2.14 +-0.52 at version 2",
        "after 3 to 7 epochs",
        "On MNIST it is +0.29 +-0.14 and +0.41 +-0.15, two to three",
        "On Fashion-MNIST it is +0.18 +-0.28 and -0.02 +-0.22: nothing",
        "networks lose 0.17, 0.61 and 1.04 points where the old ones lose 0.19, 0.60 and 1.09",
        "As budgeted at v1 it is 0.21, 1.35 and 1.37 for 0.34, 1.16 and 1.23",
        "by more than 1.8 of its errors",
        "At 10%, over the same network trained with none: +0.46 +-0.18 on a v1 tile, which is 2.5 of its errors",
        "short of the 2.8 five networks need, and +0.29 +-0.07 at version 2, which is 3.9",
        "On the inverted set +0.24 +-0.28 and -0.02 +-0.18, and on MNIST -0.03 +-0.07 and +0.02 +-0.06",
        "network by +0.01 +-0.08, +0.23 +-0.17 and -0.16 +-0.09",
        "At 20% a network loses 0.46 points to a v1 tile on Fashion-MNIST where the one with no noise loses 1.35, and 0.67 for 1.37",
        "It starts 0.38 and 1.43 lower on its host",
        "at version 2 it is -0.06 +-0.11 and -1.14 +-0.16 against",
        "is it ahead, +0.51 +-0.13, and",
        "Trained with 10% of noise: +0.19 +-0.04, +0.65 +-0.15 and +0.44 +-0.03.",
        "Trained as before: +0.19 +-0.06, +0.62 +-0.10 and +0.61 +-0.06.",
        "no noise: +0.13 +-0.02, +0.82 +-0.13 and +0.71 +-0.20.",
        "All fifteen rows are over 3.1 of their errors",
        "on the inverted set, 0.44 for 0.61",
        "for the 20% network, 0.25 and 0.29",
        "at version 2: +0.19 +-0.10 on MNIST, -0.17 +-0.24 on Fashion-MNIST and +1.67 +-0.46 on the inverted set",
        "Held, +0.19 +-0.10, -0.19 +-0.18 and +1.91 +-0.68.",
        "the network is -0.63 +-0.16, behind by 4.0 of its errors",
        "at version 2 is +0.57 +-0.12, +1.09 +-0.24 and +2.73 +-0.45 over the old one at v1",
        "and +0.38 +-0.11, +0.48 +-0.20 and +2.12 +-0.45 over the old one at version 2",
    ):
        assert said.count(words) == 1, words

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
