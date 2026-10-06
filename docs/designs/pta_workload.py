"""
Does the budget hold on a second workload?

Every accuracy figure in the board plan was measured on one network family, a
784-100-10 MLP, on one data set, MNIST.  The plan's open question 7 asked
whether the interface chip's budget survives another.  Depth was tried on
2026-10-03 and the budget held.  This is another data set, and it does not.

Two were run through grx930's harness, with the harness's own trainer and
nothing retuned:

  Fashion-MNIST    Xiao, Rasul and Vollgraf, arXiv:1708.07747.  Ten kinds of
                   clothing in MNIST's format: the same 60,000 and 10,000
                   images of 28 x 28, so the same network takes it.  Harder: the
                   networks reach 87.5% where MNIST's reach 97.5%
  MNIST, inverted  every pixel taken from 255.  The same digits with the page
                   lit and the ink dark.  Meant as a control for light alone.
                   It is not a clean one: the trainer does worse on it, 91.3 to
                   94.5%, so two things differ from MNIST and not one

MEASURED IN A MODEL, by grx930 (c930/doc/pta_error_model_design_note.md section
5, "Does the budget hold on another workload?", 2026-10-05).  `sim/pta_mnist.sh
DIR WORK geometry|budget|laser|fill` with DIR Fashion-MNIST's four files, and
with DIR MNIST's and PIXELS=inverted.  Five networks a workload, means and
standard errors over them; the budget's table at the core's tile is means
alone, as that mode prints it.  MNIST's own figures are the ones this plan
already holds, in pta_geometry.py and pta_laser.py and in grx930's note.

DERIVED here:

  what a row costs    version 1's accuracy less the same run with one row at
                      version 0's value
  the laser           the least multiple of B5's at which a workload is within
                      a tenth of a point of its own version 1, and that in
                      watts on the working tile: pta_laser.py's
  the light           a workload's mean pixel, times the 784 inputs, over the
                      seven shots a 128-row tile takes them in

WHAT THIS IS NOT:

  - a network trained for the tile.  None of these was trained with the tile's
    errors in the loop, and one that was may take more of them
  - a network trained well.  The trainer was MNIST's, unchanged.  It does not
    take its inputs less their mean, which is the likely reason the inverted
    set trains worse and was not tested; and a network that did would still
    need that mean sent as light
  - another kind of network.  Fashion-MNIST is MNIST's size and shape.  No
    convolution, no residual path, no attention has been run
  - a measurement of a tile

Standard library only.  Run:  python3 docs/designs/pta_workload.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_geometry as geometry
import pta_laser as laser

MNIST, FASHION, INVERTED = "MNIST", "Fashion-MNIST", "MNIST, inverted"
WORKLOADS = (MNIST, FASHION, INVERTED)
NEW = (FASHION, INVERTED)
TILE, CORE = laser.SMALLER, geometry.CORE          # 128 x 64, the working tile (B10); 8 x 8
WITHIN = laser.WITHIN
D3_IN = geometry.D3_IN                             # 784

# ---- grx930's figures ----------------------------------------------------------
# The mean pixel, of full scale, in the images a run evaluates.
MEAN_PIXEL = {MNIST: 0.1325, FASHION: 0.2868, INVERTED: 0.8675}
# The five networks on the host, percent.
HOST = {
    MNIST: (97.44, 97.81, 97.17, 97.43, 97.41),
    FASHION: (87.72, 87.32, 86.87, 87.99, 87.48),
    INVERTED: (94.46, 93.64, 91.32, 93.37, 94.01),
}
# The 8-bit converter's shift on the working tile, a layer each, network by network.
S8 = {
    MNIST: ((11, 9),) * 5,
    FASHION: ((12, 9), (11, 9), (11, 9), (11, 9), (11, 8)),
    INVERTED: ((12, 8), (11, 8), (11, 8), (12, 7), (12, 7)),
}
# The light a shot sends a column there, in lines: (mean, most), a layer each.
LIT = {
    MNIST: laser.LIT[TILE],
    FASHION: ((32.31, 115.75), (3.78, 10.51)),
    INVERTED: ((95.02, 124.98), (3.50, 5.63)),
}
# Under version 1 on the working tile: the error that reaches the outputs, in
# percent of their rms, and the median image's lead (its largest output less
# the next) in multiples of the rms error at the outputs.
OUT_ERR = {MNIST: (7.94, 0.15), FASHION: (7.77, 0.74), INVERTED: (12.38, 1.22)}
LEAD = {MNIST: (7.792, 0.177), FASHION: (3.597, 0.247), INVERTED: (5.446, 0.420)}
# The same error at the outputs, there, under one row at a time and under drift:
# MNIST, Fashion-MNIST, MNIST inverted.
ROW_ERR = {
    "thermal1": (8.12, 7.43, 9.83), "photons3": (10.79, 8.95, 20.31), "prog4": (7.11, 6.91, 14.72),
    "xtalk10": (18.83, 16.51, 19.28), "v1": (7.94, 7.77, 12.38), "hour": (9.39, 10.61, 25.11),
    "four": (13.58, 19.83, 42.10),
}

# `geometry`: points lost against the same weights on the host, by tile.
ROW_NAMES = (
    ("quant8", "the quantisers, 8-bit ADC"), ("adc6", "the same, 6-bit ADC"),
    ("act5", "v0's rows alone: 5 activation bits"), ("thermal1", "  receiver noise 1 LSB"),
    ("photons3", "  3 photons an LSB"), ("prog4", "  programming error 4 LSB"),
    ("xtalk10", "  crosstalk 10%"), ("v0_8", "v0's rows together, 8-bit ADC"),
    ("v0", "v0, at its 6-bit ADC"), ("v1", "version 1"), ("six", "v1, six minutes of TFLT's drift"),
    ("hour", "v1, an hour of it"), ("four", "v1, four hours of it"), ("long46", "v1, 46 hours of it"),
    ("tfln", "v1, an hour of TFLN's"), ("cal", "v1, an hour, then calibrated"),
)
ROWS = {
    (MNIST, CORE): dict(
        quant8=(0.03, 0.02), adc6=(0.22, 0.06), act5=(0.17, 0.05), thermal1=(0.29, 0.02),
        photons3=(0.49, 0.03), prog4=(0.30, 0.02), xtalk10=(0.19, 0.07), v0_8=(1.41, 0.09),
        v0=(1.49, 0.06), v1=(0.26, 0.06), six=(0.37, 0.04), hour=(0.81, 0.21), four=(5.18, 2.31),
        long46=(65.54, 5.06), tfln=(22.62, 4.79), cal=(0.22, 0.05)),
    (MNIST, TILE): dict(
        quant8=(0.02, 0.04), adc6=(0.43, 0.18), act5=(0.13, 0.06), thermal1=(0.28, 0.06),
        photons3=(0.68, 0.04), prog4=(0.20, 0.10), xtalk10=(0.21, 0.05), v0_8=(1.83, 0.04),
        v0=(2.17, 0.12), v1=(0.34, 0.07), six=(0.33, 0.09), hour=(0.51, 0.07), four=(0.74, 0.07),
        long46=(9.36, 0.96), tfln=(2.81, 0.22), cal=(0.21, 0.05)),
    (FASHION, CORE): dict(
        quant8=(0.13, 0.06), adc6=(1.24, 0.37), act5=(0.59, 0.13), thermal1=(1.35, 0.39),
        photons3=(1.15, 0.23), prog4=(0.89, 0.11), xtalk10=(1.24, 0.30), v0_8=(4.09, 0.61),
        v0=(4.73, 0.66), v1=(1.06, 0.26), six=(1.18, 0.18), hour=(4.39, 0.89), four=(19.55, 3.07),
        long46=(62.48, 5.86), tfln=(45.48, 4.91), cal=(1.04, 0.20)),
    (FASHION, TILE): dict(
        quant8=(0.06, 0.07), adc6=(1.51, 0.28), act5=(0.51, 0.11), thermal1=(1.38, 0.31),
        photons3=(2.16, 0.14), prog4=(0.80, 0.12), xtalk10=(0.96, 0.24), v0_8=(4.65, 0.50),
        v0=(5.27, 0.43), v1=(1.16, 0.14), six=(1.24, 0.18), hour=(1.84, 0.39), four=(3.50, 0.41),
        long46=(23.26, 1.48), tfln=(8.37, 1.47), cal=(1.18, 0.09)),
    (INVERTED, CORE): dict(
        quant8=(0.17, 0.11), adc6=(0.85, 0.30), act5=(0.71, 0.19), thermal1=(0.82, 0.20),
        photons3=(1.60, 0.19), prog4=(1.83, 0.24), xtalk10=(1.27, 0.44), v0_8=(6.59, 1.00),
        v0=(7.14, 1.18), v1=(1.10, 0.22), six=(2.70, 0.73), hour=(23.52, 5.81), four=(62.76, 5.08),
        long46=(81.98, 0.99), tfln=(75.65, 2.39), cal=(1.31, 0.51)),
    (INVERTED, TILE): dict(
        quant8=(-0.01, 0.05), adc6=(1.08, 0.33), act5=(0.65, 0.12), thermal1=(0.65, 0.12),
        photons3=(3.05, 0.48), prog4=(1.57, 0.16), xtalk10=(1.32, 0.40), v0_8=(8.04, 0.71),
        v0=(8.57, 0.63), v1=(1.23, 0.13), six=(1.42, 0.35), hour=(3.18, 1.27), four=(17.77, 3.02),
        long46=(68.12, 2.92), tfln=(38.58, 5.56), cal=(1.14, 0.18)),
}

# `budget`: accuracies at the core's tile, means of five.  The rows in order:
# the activation DAC's bits, the ADC's, the receiver's noise in 8-bit LSB, the
# photons such an LSB, the programming error in weight LSB, the crosstalk.
ROW6 = ("activation DAC, 5 or 6 bits", "ADC, 6 or 7 bits", "receiver noise, 1 or 0.5 LSB",
        "light, 3 or 15 photons an LSB", "programming error, 4 or 1 LSB", "crosstalk, 10% or 2%")
BUDGET = {
    FASHION: dict(host=87.45, adc8=87.34, adc6=86.23, alone=(86.88, 86.13, 86.32, 86.59, 86.24),
                  all8=83.39, v0=82.74, v1=86.41, v1_halved=86.60, v1_dark=86.64,
                  tighten=(83.08, 83.10, 83.35, 83.62, 83.51, 83.54),
                  relax=(86.11, 85.82, 85.55, 85.72, 85.76, 85.63)),
    INVERTED: dict(host=93.34, adc8=93.19, adc6=92.51, alone=(92.65, 92.54, 91.76, 91.53, 92.09),
                   all8=86.77, v0=86.22, v1=92.26, v1_halved=92.48, v1_dark=92.54,
                   tighten=(86.70, 86.51, 86.76, 87.68, 88.43, 86.92),
                   relax=(91.67, 91.47, 91.61, 90.91, 90.39, 91.21)),
}
# MNIST's, as grx930's note has them: what each row costs relaxed from version
# 1 and buys tightened from version 0, version 0's five rows alone at the 8-bit
# ADC, and all five at once.
MNIST_RELAX = (0.10, 0.12, 0.19, 0.34, 0.33, 0.16)
MNIST_TIGHTEN = (0.10, 0.09, 0.18, 0.34, 0.42, 0.08)
MNIST_ALONE, MNIST_TOGETHER = (0.14, 0.26, 0.45, 0.27, 0.16), 1.38

# `laser`: points lost on the working tile by the laser's size, as a multiple of
# B5's; 0 is version 1 as budgeted, half an LSB a layer at that layer's shift.
LOST = {
    MNIST: laser.LOST[TILE],
    FASHION: {0: (1.16, 0.14), 1: (23.90, 3.49), 2: (10.38, 2.19), 4: (4.16, 0.89), 8: (1.93, 0.29),
              16: (1.26, 0.14), 32: (1.04, 0.13), 64: (1.00, 0.14)},
    INVERTED: {0: (1.23, 0.13), 1: (40.87, 4.46), 2: (17.47, 2.89), 4: (5.35, 0.90), 8: (1.99, 0.27),
               16: (1.27, 0.18), 32: (1.15, 0.16), 64: (1.04, 0.14)},
}
# The receiver's noise a laser of B5's size leaves, in 8-bit LSB, a layer each.
NOISE_B5 = {MNIST: (1.98, 7.94), FASHION: (1.79, 9.53), INVERTED: (1.39, 22.23)}

# `fill`: (bits added to the hidden rescale, bits of gain on the first layer's
# weights) -> points lost by the laser's multiple; and what each clips, in
# percent of the hidden units that fire and of the first layer's weights, with
# what that costs by itself.
PUT = {
    MNIST: laser.PUT,
    FASHION: {
        (0, 0):  {2: (10.38, 2.19), 4: (4.16, 0.89), 8: (1.93, 0.29)},
        (0, 1):  {2: (12.83, 2.42), 4: (5.62, 1.23), 8: (2.67, 0.43)},
        (0, 2):  {2: (16.82, 2.43), 4: (8.42, 1.44), 8: (4.87, 0.75)},
        (-1, 0): {2: (4.39, 0.91), 4: (1.93, 0.34), 8: (1.19, 0.13)},
        (-1, 1): {2: (5.69, 1.23), 4: (2.65, 0.46), 8: (1.72, 0.14)},
        (-1, 2): {2: (8.38, 1.44), 4: (4.84, 0.76), 8: (3.59, 0.69)},
        (-2, 0): {2: (2.28, 0.27), 4: (1.29, 0.14), 8: (0.98, 0.08)},
        (-2, 1): {2: (2.60, 0.41), 4: (1.41, 0.17), 8: (1.05, 0.13)},
        (-2, 2): {2: (5.02, 0.77), 4: (3.67, 0.62), 8: (3.24, 0.61)},
    },
    INVERTED: {
        (0, 0):  {2: (17.47, 2.89), 4: (5.35, 0.90), 8: (1.99, 0.27)},
        (0, 1):  {2: (23.83, 4.77), 4: (10.48, 2.33), 8: (5.83, 1.14)},
        (0, 2):  {2: (45.65, 5.01), 4: (34.19, 4.79), 8: (29.41, 4.69)},
        (-1, 0): {2: (6.01, 0.87), 4: (2.29, 0.29), 8: (1.37, 0.16)},
        (-1, 1): {2: (11.41, 2.57), 4: (6.33, 1.29), 8: (4.99, 1.01)},
        (-1, 2): {2: (34.92, 4.52), 4: (29.73, 4.49), 8: (28.08, 4.54)},
        (-2, 0): {2: (3.57, 0.42), 4: (2.22, 0.37), 8: (1.86, 0.36)},
        (-2, 1): {2: (6.25, 1.34), 4: (4.53, 1.04), 8: (4.08, 0.93)},
        (-2, 2): {2: (29.89, 5.27), 4: (27.63, 5.44), 8: (27.00, 5.50)},
    },
}
PUT_CLIPS = {
    MNIST: laser.PUT_CLIPS,
    FASHION: {
        (0, 0): (0.01, 0.00, (0.03, 0.01)), (0, 1): (0.00, 0.73, (0.34, 0.08)), (0, 2): (0.00, 8.42, (2.33, 0.62)),
        (-1, 0): (0.90, 0.00, (0.02, 0.01)), (-1, 1): (0.75, 0.73, (0.35, 0.06)), (-1, 2): (0.51, 8.42, (2.33, 0.63)),
        (-2, 0): (10.32, 0.00, (0.21, 0.08)), (-2, 1): (8.89, 0.73, (0.39, 0.07)), (-2, 2): (7.01, 8.42, (2.42, 0.71)),
    },
    INVERTED: {
        (0, 0): (0.00, 0.00, (0.02, 0.02)), (0, 1): (0.00, 0.15, (3.05, 0.82)), (0, 2): (0.00, 2.09, (26.40, 5.17)),
        (-1, 0): (2.75, 0.00, (0.04, 0.04)), (-1, 1): (3.33, 0.15, (2.99, 0.81)), (-1, 2): (2.30, 2.09, (26.34, 5.22)),
        (-2, 0): (22.27, 0.00, (0.61, 0.33)), (-2, 1): (22.97, 0.15, (2.70, 0.79)), (-2, 2): (18.45, 2.09, (25.97, 5.94)),
    },
}
# The same at 16 times B5's laser, which the first runs stopped short of and
# the two new workloads were then given (FILL_TIMES=16).  MNIST was not.
PUT16 = {
    FASHION: {
        (0, 0): {16: (1.26, 0.14)}, (0, 1): {16: (1.68, 0.22)}, (0, 2): {16: (3.63, 0.72)},
        (-1, 0): {16: (0.96, 0.07)}, (-1, 1): {16: (1.47, 0.10)}, (-1, 2): {16: (3.12, 0.72)},
        (-2, 0): {16: (0.89, 0.07)}, (-2, 1): {16: (1.04, 0.12)}, (-2, 2): {16: (3.07, 0.64)},
    },
    INVERTED: {
        (0, 0): {16: (1.27, 0.18)}, (0, 1): {16: (4.59, 0.88)}, (0, 2): {16: (27.71, 4.69)},
        (-1, 0): {16: (1.17, 0.12)}, (-1, 1): {16: (4.56, 0.91)}, (-1, 2): {16: (27.57, 4.61)},
        (-2, 0): {16: (1.81, 0.37)}, (-2, 1): {16: (4.03, 0.94)}, (-2, 2): {16: (26.88, 5.52)},
    },
}
AS_SET, BIT_DOWN, TWO_DOWN = laser.AS_SET, laser.BIT_DOWN, (-2, 0)


# ---- what follows from them ---------------------------------------------------------
def mean(xs):
    return sum(xs) / len(xs)


def shots():
    """The shots a 128-row tile takes D3's first layer in: 784 inputs, 128 a shot."""
    return -(-D3_IN // TILE[0])


def lit_expected(w):
    """The lines a shot of the first layer sends a column at the mean, if the
    light is the pixels: the mean pixel, times the inputs, over the shots."""
    return MEAN_PIXEL[w] * D3_IN / shots()


def over(w, times):
    """Points a laser of that multiple loses over the workload's own version 1."""
    return LOST[w][times][0] - LOST[w][0][0]


def times_within(w, budget=WITHIN):
    """The least multiple measured that is within the budget of the workload's
    own version 1, every larger one being so too."""
    ok = None
    for t in sorted((t for t in LOST[w] if t), reverse=True):
        if over(w, t) >= budget - 1e-9:
            break
        ok = t
    return ok


def put(w, how):
    """Points lost by the laser's multiple with the network put on the tile that way."""
    d = dict(PUT[w][how])
    d.update(PUT16.get(w, {}).get(how, {}))
    return d


def put_or_laser(w, how):
    """The same, with the rule's own row filled out from the laser's runs, which
    are that row at more multiples."""
    d = put(w, how)
    if how == AS_SET:
        d.update({m: x for m, x in LOST[w].items() if m > 1})
    return d


def times_put(w, how, budget=WITHIN):
    """The same least multiple, with the network put on the tile that way; None
    if none measured is within the budget."""
    ok = None
    d = put_or_laser(w, how)
    for t in sorted(d, reverse=True):
        if d[t][0] - LOST[w][0][0] >= budget - 1e-9:
            break
        ok = t
    return ok


def half_lsb(w):
    """What the receiver's row costs, half an LSB a layer, over a laser that
    leaves a thirty-second of an LSB or less."""
    return LOST[w][0][0] - LOST[w][64][0]


def relax(w):
    """What each of version 1's six rows costs, relaxed to version 0's value."""
    if w == MNIST:
        return MNIST_RELAX
    b = BUDGET[w]
    return tuple(round(b["v1"] - a, 2) for a in b["relax"])


def tighten(w):
    if w == MNIST:
        return MNIST_TIGHTEN
    b = BUDGET[w]
    return tuple(round(a - b["v0"], 2) for a in b["tighten"])


def alone(w):
    """Version 0's five rows, each alone at the 8-bit ADC, under that ADC."""
    if w == MNIST:
        return MNIST_ALONE
    b = BUDGET[w]
    return tuple(round(b["adc8"] - a, 2) for a in b["alone"])


def together(w):
    if w == MNIST:
        return MNIST_TOGETHER
    return round(BUDGET[w]["adc8"] - BUDGET[w]["all8"], 2)


def drift_over(w, key, tile=TILE):
    """What drift adds to version 1 on that tile."""
    return ROWS[(w, tile)][key][0] - ROWS[(w, tile)]["v1"][0]


def sigmas(a, b):
    """How many standard errors apart two of grx930's means are."""
    return (a[0] - b[0]) / (a[1] ** 2 + b[1] ** 2) ** 0.5


def pm(x):
    return f"{x[0]:.2f} +-{x[1]:.2f}"


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    print("Does the budget hold on a second workload?")
    print(f"The working tile is {laser.name(TILE)} (B10) and the core's is {laser.name(CORE)}.  Five networks a workload.")

    section("1. The three workloads")
    print(f"  {'':<18}{'on the host':>14}{'mean pixel':>12}{'lines a shot, layer 1':>24}{'the most':>10}{'of the rows':>13}"
          f"{'layer 2':>9}{'8-bit shifts, by network':>40}")
    for w in WORKLOADS:
        s = " ".join(f"{a},{b}" for a, b in S8[w])
        print(f"  {w:<18}{f'{min(HOST[w]):.1f}-{max(HOST[w]):.1f}%':>14}{MEAN_PIXEL[w]:>12.4f}{LIT[w][0][0]:>24.2f}"
              f"{LIT[w][0][1]:>10.2f}{LIT[w][0][0] / TILE[0]:>12.0%} {LIT[w][1][0]:>8.2f}{s:>40}")
    print(f"  A {TILE[0]}-row tile takes the first layer's {D3_IN} inputs in {shots()} shots.  The mean pixel, times the")
    print("  inputs, over the shots is " + ", ".join(f"{lit_expected(w):.1f}" for w in WORKLOADS) + " lines: the light is the pixels.")

    section("2. Version 1, on each")
    print(f"  {'':<18}{'the core tile':>16}{'the working tile':>18}{'times MNIST':>13}{'error at the outputs':>23}{'median lead, in errors':>25}")
    for w in WORKLOADS:
        c, t = ROWS[(w, CORE)]["v1"], ROWS[(w, TILE)]["v1"]
        print(f"  {w:<18}{pm(c):>16}{pm(t):>18}{t[0] / ROWS[(MNIST, TILE)]['v1'][0]:>12.1f}x{f'{OUT_ERR[w][0]:.2f}%':>23}"
              f"{LEAD[w][0]:>25.2f}")
    print("  Points lost against the same weights on the host.  The error is what reaches the ten")
    print("  outputs, in percent of their rms; the lead is the median image's largest output less")
    print("  its next, in multiples of the rms error there.")

    section("3. What a row costs, at the core's tile")
    print(f"  {'relaxed from version 1':<34}" + "".join(f"{w:>18}" for w in WORKLOADS))
    for i, r in enumerate(ROW6):
        print(f"  {r:<34}" + "".join(f"{relax(w)[i]:>18.2f}" for w in WORKLOADS))
    print(f"  {'summed':<34}" + "".join(f"{sum(relax(w)):>18.2f}" for w in WORKLOADS))
    print(f"  {'tightened from version 0, summed':<34}" + "".join(f"{sum(tighten(w)):>18.2f}" for w in WORKLOADS))
    print(f"  {'v0\'s five rows alone, summed':<34}" + "".join(f"{sum(alone(w)):>18.2f}" for w in WORKLOADS))
    print(f"  {'the five at once':<34}" + "".join(f"{together(w):>18.2f}" for w in WORKLOADS))
    print(f"  {'at once, over summed':<34}" + "".join(f"{together(w) / sum(alone(w)):>18.2f}" for w in WORKLOADS))

    section("4. The rows on both tiles, and drift")
    cols = [(w, t) for w in WORKLOADS for t in (CORE, TILE)]
    print(f"  {'':<36}" + "".join(f"{(w.split(',')[0] if t == CORE else '') + ' ' + laser.name(t):>21}" for w, t in cols))
    for key, label in ROW_NAMES:
        print(f"  {label:<36}" + "".join(f"{pm(ROWS[c][key]):>21}" for c in cols))
    print("  The third pair of columns is MNIST, inverted.  What drift adds to version 1 on the working tile:")
    for key, label in (("six", "six minutes"), ("hour", "an hour"), ("four", "four hours"), ("cal", "an hour, then calibrated")):
        print(f"    {label:<28}" + "".join(f"{w + f' {drift_over(w, key):+.2f}':>26}" for w in WORKLOADS))
    print("  And the error each puts on the outputs there, in percent of their rms:")
    for key in ("v1", "thermal1", "photons3", "prog4", "xtalk10", "hour", "four"):
        print(f"    {dict(ROW_NAMES)[key].strip():<34}" + "".join(f"{f'{x:.2f}%':>12}" for x in ROW_ERR[key]))

    section("5. The laser")
    mults = sorted(LOST[MNIST])
    print(f"  {'':<26}" + "".join(f"{w:>18}" for w in WORKLOADS) + "".join(f"{'over its own v1':>18}" for _ in WORKLOADS[:1])
          + "".join(f"{'':>10}" for _ in WORKLOADS[1:]))
    for m in mults:
        label = "v1 as budgeted" if m == 0 else f"B5's laser, times {m}"
        print(f"  {label:<26}" + "".join(f"{pm(LOST[w][m]):>18}" for w in WORKLOADS)
              + ("" if m == 0 else "".join(f"{over(w, m):>+10.2f}" for w in WORKLOADS)))
    print(f"  {'noise at B5, layer 1, 2':<26}" + "".join(f"{f'{NOISE_B5[w][0]:.2f}, {NOISE_B5[w][1]:.2f} LSB':>18}" for w in WORKLOADS))
    print(f"  {'within a tenth of v1 at':<26}" + "".join(f"{f'{times_within(w)} times':>18}" for w in WORKLOADS))
    print(f"  {'that, in watts':<26}" + "".join(
        f"{'{:.2f}-{:.1f} W'.format(*laser.laser_w(TILE, times_within(w))):>18}" for w in WORKLOADS))
    print(f"  {'half an LSB, over none':<26}" + "".join(f"{half_lsb(w):>18.2f}" for w in WORKLOADS))
    print(f"  {'the rescale a bit down':<26}" + "".join(f"{f'{times_put(w, BIT_DOWN)} times':>18}" for w in WORKLOADS))
    print(f"  {'that, in watts':<26}" + "".join(
        f"{'{:.2f}-{:.1f} W'.format(*laser.laser_w(TILE, times_put(w, BIT_DOWN))):>18}" for w in WORKLOADS))
    print(f"  {'a MAC, every cell in use':<26}" + "".join(
        f"{'{:.0f}-{:,.0f} fJ'.format(*laser.fj_mac(TILE, times_put(w, BIT_DOWN))):>18}" for w in WORKLOADS))

    section("6. How the network is put on the tile")
    for w in WORKLOADS:
        ms = sorted(put(w, AS_SET))
        print(f"  {w}, where version 1 as budgeted loses {pm(LOST[w][0])}")
        print(f"    {'hidden rescale':<16}{'weight gain':>12}{'units clip':>12}{'weights clip':>14}{'that alone':>14}"
              + "".join(f"{'x ' + str(m):>14}" for m in ms) + f"{'within a tenth at':>19}")
        for how in sorted(PUT[w], key=lambda k: (-k[0], k[1])):
            hc, wc, by_itself = PUT_CLIPS[w][how]
            t = times_put(w, how)
            d = put(w, how)
            print(f"    {('the rule' + chr(39) + 's') if how[0] == 0 else f'{-how[0]} bit(s) less':<16}{f'{how[1]} bit(s)':>12}"
                  f"{hc:>11.2f}%{wc:>13.2f}%{pm(by_itself):>14}"
                  + "".join(f"{(pm(d[m]) if m in d else ''):>14}" for m in ms)
                  + f"{('none measured' if t is None else f'{t} times'):>19}")

    findings()
    checks()


def findings():
    print()
    print("What this says, seven readings.")
    print()
    t, c = [ROWS[(w, TILE)]["v1"] for w in WORKLOADS], [ROWS[(w, CORE)]["v1"] for w in WORKLOADS]
    print(f"  1. THE BUDGET DOES NOT HOLD.  Version 1 loses {pm(t[0])} of a point on MNIST on the working")
    print(f"     tile, {pm(t[1])} on Fashion-MNIST and {pm(t[2])} on MNIST inverted: {t[1][0] / t[0][0]:.1f} and {t[2][0] / t[0][0]:.1f}")
    print(f"     times.  On the core's tile, {c[0][0]:.2f}, {c[1][0]:.2f} and {c[2][0]:.2f}.  The plan accepted version 1 at a")
    print("     quarter of a point.  That was MNIST's price.")
    print()
    print(f"  2. ON FASHION-MNIST THE TILE IS NO WORSE.  THE NETWORK HAS LESS TO SPARE.  Its outputs")
    print(f"     come back {OUT_ERR[FASHION][0]:.1f}% wrong where MNIST's come back {OUT_ERR[MNIST][0]:.1f}%, and it loses {t[1][0] / t[0][0]:.1f} times the")
    print(f"     points.  The median image's lead there is {LEAD[FASHION][0]:.1f} times the error, against {LEAD[MNIST][0]:.1f} on")
    print(f"     MNIST.  The inverted set has both: its outputs are {OUT_ERR[INVERTED][0]:.1f}% wrong and its lead is {LEAD[INVERTED][0]:.1f}.")
    print(f"     And the loss does not follow the light: Fashion-MNIST lights {LIT[FASHION][0][0] / LIT[MNIST][0][0]:.1f} times the rows")
    print(f"     MNIST does and the inverted set {LIT[INVERTED][0][0] / LIT[MNIST][0][0]:.1f} times, and the two lose the same.")
    print()
    rf, ri, rm = relax(FASHION), relax(INVERTED), relax(MNIST)
    print(f"  3. EVERY ROW COSTS MORE, AND THE DEAR ONES ARE NOT THE SAME.  Relaxed from version 1, no")
    print(f"     row was worth more than {max(rm):.2f} on MNIST.  On Fashion-MNIST the receiver's noise is {rf[2]:.2f}")
    print(f"     and crosstalk {rf[5]:.2f}.  On the inverted set programming error is {ri[4]:.2f} and the light")
    print(f"     {ri[3]:.2f}.  The ADC's bit, which the plan weighed giving up for {rm[1]:.2f}, is {rf[1]:.2f} and {ri[1]:.2f}.")
    print(f"     The six sum to {sum(rm):.2f}, {sum(rf):.2f} and {sum(ri):.2f}.")
    print()
    print(f"  4. AND THEY DO NOT ADD THE SAME WAY.  Version 0's five rows at once cost {together(MNIST) / sum(alone(MNIST)):.2f} of their")
    print(f"     sum on MNIST, {together(FASHION) / sum(alone(FASHION)):.2f} on Fashion-MNIST and {together(INVERTED) / sum(alone(INVERTED)):.2f} on the inverted set.  Version 0 itself")
    print(f"     loses {ROWS[(MNIST, TILE)]['v0'][0]:.2f}, {ROWS[(FASHION, TILE)]['v0'][0]:.2f} and {ROWS[(INVERTED, TILE)]['v0'][0]:.2f} on the working tile.")
    print()
    print(f"  5. DRIFT COSTS MORE, AND ON THE INVERTED SET FAR MORE.  An hour of TFLT's drift adds")
    print(f"     {drift_over(MNIST, 'hour'):.2f} to version 1 on MNIST on the working tile, {drift_over(FASHION, 'hour'):.2f} on Fashion-MNIST and {drift_over(INVERTED, 'hour'):.2f} on")
    print(f"     the inverted set; four hours, {drift_over(MNIST, 'four'):.2f}, {drift_over(FASHION, 'four'):.2f} and {drift_over(INVERTED, 'four'):.2f}.  On Fashion-MNIST that is")
    print(f"     about the same error costing more: {ROW_ERR['hour'][1]:.1f}% at the outputs after the hour, for {ROW_ERR['hour'][0]:.1f}%.")
    print(f"     On the inverted set it is more error, {ROW_ERR['hour'][2]:.1f}%, and programming error does the")
    print(f"     same there, {ROW_ERR['prog4'][2]:.1f}% for {ROW_ERR['prog4'][0]:.1f}%: a weight's error reaches a sum by what it is lit")
    print("     with, and that set lights three rows in four.  Calibration returns all three")
    print("     to their version 1.  So how long one holds is the workload's, and an interval")
    print("     set on MNIST is MNIST's.")
    print()
    tw = [times_within(w) for w in WORKLOADS]
    tp = [times_put(w, BIT_DOWN) for w in WORKLOADS]
    print(f"  6. THE LASER IS TWICE MNIST'S, AND FOUR TIMES.  Within a tenth of a point of its own")
    print(f"     version 1 takes {tw[0]} times B5's laser on MNIST, {tw[1]} on Fashion-MNIST and {tw[2]} on the inverted")
    print(f"     set.  Fashion-MNIST at 16 is {over(FASHION, 16):.2f} over, which is the line and inside its scatter.")
    print(f"     At 8, which is MNIST's, each of the others is three quarters of a point over.")
    print(f"     With the hidden rescale a bit down it takes {tp[0]}, {tp[1]} and {tp[2]}, which on the working tile")
    print("     is " + ", ".join("{:.2f}-{:.1f} W".format(*laser.laser_w(TILE, x)) for x in tp) + ", and a MAC")
    print("     " + ", ".join("{:.0f}-{:,.0f} fJ".format(*laser.fj_mac(TILE, x)) for x in tp) + ".  Half an 8-bit LSB of receiver's noise")
    print(f"     a layer, the row itself, costs {half_lsb(MNIST):.2f}, {half_lsb(FASHION):.2f} and {half_lsb(INVERTED):.2f} over none: that much held.  A")
    print(f"     second half does not: {relax(MNIST)[2]:.2f}, {relax(FASHION)[2]:.2f} and {relax(INVERTED)[2]:.2f} on the core's tile.")
    print()
    print(f"  7. THE RESCALE STILL GIVES HALF OF IT BACK, AND THE WEIGHTS' GAIN IS WORSE THAN USELESS.")
    print(f"     A bit under the clip rule, at 4 times B5's laser, loses {put(FASHION, BIT_DOWN)[4][0]:.2f} on Fashion-MNIST where the")
    print(f"     rule at 8 loses {LOST[FASHION][8][0]:.2f}, and {put(INVERTED, BIT_DOWN)[4][0]:.2f} against {LOST[INVERTED][8][0]:.2f} on the inverted set.  A second bit,")
    print(f"     which bought MNIST nothing, pays on Fashion-MNIST at a small laser: {put(FASHION, TWO_DOWN)[4][0]:.2f} at 4 times.")
    print(f"     And writing the first layer's weights twice as large saturates {PUT_CLIPS[INVERTED][(0, 1)][1]:.2f}% of them on")
    print(f"     the inverted set and costs {PUT_CLIPS[INVERTED][(0, 1)][2][0]:.2f} points by itself: its sums are small differences")
    print("     of a great deal of light, and the largest weights are what cancels it.")


def checks():
    """Every claim above, as an assert."""
    assert TILE == (128, 64) and CORE == (8, 8) and D3_IN == 784 and shots() == 7

    # 0. MNIST's column is the one the plan already holds.
    g = geometry.ACC
    for tile in (CORE, TILE):
        for key in ("v1", "v0", "hour", "four", "tfln", "cal"):
            assert ROWS[(MNIST, tile)][key] == g[tile][key], (tile, key)
    assert LOST[MNIST][0] == ROWS[(MNIST, TILE)]["v1"] and all(LOST[w][0] == ROWS[(w, TILE)]["v1"] for w in NEW)
    assert all(set(ROWS[c]) == {k for k, _ in ROW_NAMES} for c in ROWS) and len(ROWS) == 6
    assert all(sorted(LOST[w]) == [0, 1, 2, 4, 8, 16, 32, 64] for w in WORKLOADS)

    # 1. Section 1.  The networks, and that the light is the pixels.
    assert all(len(HOST[w]) == 5 and len(S8[w]) == 5 for w in WORKLOADS)
    assert [round(mean(HOST[w]), 1) for w in WORKLOADS] == [97.5, 87.5, 93.4]
    assert (min(HOST[FASHION]), max(HOST[FASHION])) == (86.87, 87.99)
    assert (min(HOST[INVERTED]), max(HOST[INVERTED])) == (91.32, 94.46)
    assert abs(MEAN_PIXEL[MNIST] + MEAN_PIXEL[INVERTED] - 1) < 1e-9
    assert all(abs(LIT[w][0][0] / lit_expected(w) - 1) < 0.03 for w in WORKLOADS)
    assert [round(LIT[w][0][0] / TILE[0], 2) for w in WORKLOADS] == [0.12, 0.25, 0.74]
    #    The second layer is lit less on both, not more.
    assert all(LIT[w][1][0] < 0.6 * LIT[MNIST][1][0] for w in NEW)
    #    The first layer's shift rose on four networks of ten and the second's
    #    fell on six: on every inverted one, and by two bits on two.
    assert sum(s[0] == 12 for w in NEW for s in S8[w]) == 4 and all(s[0] in (11, 12) for w in NEW for s in S8[w])
    assert all(s[1] < 9 for s in S8[INVERTED]) and sum(s[1] == 7 for s in S8[INVERTED]) == 2
    assert sum(s[1] < 9 for s in S8[FASHION]) == 1
    #    B5's noise is the shift's: rows / 512 of a line's light, in LSB at that shift.
    for w in WORKLOADS:
        for layer in (0, 1):
            lsb = mean([TILE[0] / 512 * laser.line_light() / 2 ** s[layer] for s in S8[w]])
            assert abs(lsb - NOISE_B5[w][layer]) < 0.011, (w, layer, lsb)

    # 2. Reading 1.  Version 1, three and a half times MNIST's, and by five standard errors.
    v1t = [ROWS[(w, TILE)]["v1"] for w in WORKLOADS]
    v1c = [ROWS[(w, CORE)]["v1"] for w in WORKLOADS]
    assert [x[0] for x in v1t] == [0.34, 1.16, 1.23] and [x[0] for x in v1c] == [0.26, 1.06, 1.10]
    assert abs(v1t[1][0] / v1t[0][0] - 3.4) < 0.05 and abs(v1t[2][0] / v1t[0][0] - 3.6) < 0.05
    assert all(sigmas(x, v1t[0]) > 5 for x in v1t[1:]) and all(sigmas(x, v1c[0]) > 2.9 for x in v1c[1:])
    assert all(abs(sigmas(ROWS[(w, TILE)]["v1"], ROWS[(w, CORE)]["v1"])) < 1 for w in WORKLOADS)
    #    The budget mode's version 1 at the core's tile agrees with the geometry mode's.
    assert all(abs(BUDGET[w]["host"] - BUDGET[w]["v1"] - ROWS[(w, CORE)]["v1"][0]) < 0.03 for w in NEW)
    assert all(abs(BUDGET[w]["host"] - BUDGET[w]["v0"] - ROWS[(w, CORE)]["v0"][0]) < 0.03 for w in NEW)

    # 3. Reading 2.  Not the light, not the error: the lead.
    assert abs(LIT[FASHION][0][0] / LIT[MNIST][0][0] - 2.2) < 0.05 and abs(LIT[INVERTED][0][0] / LIT[MNIST][0][0] - 6.4) < 0.05
    assert abs(sigmas(v1t[1], v1t[2])) < 0.5
    assert abs(sigmas(OUT_ERR[FASHION], OUT_ERR[MNIST])) < 0.5 and OUT_ERR[INVERTED][0] > 1.5 * OUT_ERR[MNIST][0]
    assert LEAD[MNIST][0] > LEAD[INVERTED][0] > LEAD[FASHION][0]
    assert sigmas(LEAD[MNIST], LEAD[FASHION]) > 10 and sigmas(LEAD[MNIST], LEAD[INVERTED]) > 5
    assert abs(LEAD[MNIST][0] / LEAD[FASHION][0] - 2.2) < 0.05
    assert [f"{LEAD[w][0]:.1f}" for w in WORKLOADS] == ["7.8", "3.6", "5.4"]

    # 4. Reading 3.  The rows.
    rm, rf, ri = relax(MNIST), relax(FASHION), relax(INVERTED)
    assert rf == (0.30, 0.59, 0.86, 0.69, 0.65, 0.78) and ri == (0.59, 0.79, 0.65, 1.35, 1.87, 1.05)
    assert max(rm) == 0.34 and all(a > b for a, b in zip(rf, rm)) and all(a > b for a, b in zip(ri, rm))
    assert rf.index(max(rf)) == 2 and sorted(rf)[-2] == rf[5]          # receiver, then crosstalk
    assert ri.index(max(ri)) == 4 and sorted(ri)[-2] == ri[3]          # programming, then the light
    assert rm.index(max(rm)) == 3 and sorted(rm)[-2] == rm[4]          # the light, then programming
    assert [round(sum(r), 2) for r in (rm, rf, ri)] == [1.24, 3.87, 6.30]
    #    Two to seven times MNIST's, row by row.
    ratios = [a / b for r in (rf, ri) for a, b in zip(r, rm)]
    assert 1.95 < min(ratios) < 2.05 and 6.5 < max(ratios) < 6.7
    assert [round(sum(tighten(w)), 2) for w in WORKLOADS] == [1.21, 3.76, 5.68]
    #    The two versions are that far apart, to a quarter of a point or so.
    assert all(abs(BUDGET[w]["v1"] - BUDGET[w]["v0"] - sum(tighten(w))) < 0.4 for w in NEW)
    #    The cheapest two on MNIST were the converters' bits, and the plan's
    #    two named rows were each under a fifth of a point.
    assert sorted(rm)[:2] == [rm[0], rm[1]] and rm[1] < 0.2 and rm[2] < 0.2
    assert min(rf[1], ri[1]) > 0.5 and min(rf[2], ri[2]) > 0.6
    #    Shot noise at 15 photons, by itself, inside version 1: a quarter of a point.
    assert [round(BUDGET[w]["v1_dark"] - BUDGET[w]["v1"], 2) for w in NEW] == [0.23, 0.28]
    #    Both noise rows halved again buy a fifth of a point.
    assert [round(BUDGET[w]["v1_halved"] - BUDGET[w]["v1"], 2) for w in NEW] == [0.19, 0.22]

    # 5. Reading 4.  Adding.
    assert [round(sum(alone(w)), 2) for w in WORKLOADS] == [1.28, 4.54, 5.38]
    assert [together(w) for w in WORKLOADS] == [1.38, 3.95, 6.42]
    assert [round(together(w) / sum(alone(w)), 2) for w in WORKLOADS] == [1.08, 0.87, 1.19]
    assert [ROWS[(w, TILE)]["v0"][0] for w in WORKLOADS] == [2.17, 5.27, 8.57]
    #    On the working tile the dearest of version 0's rows is the light, on all three.
    for w in WORKLOADS:
        five = {k: ROWS[(w, TILE)][k][0] for k in ("act5", "thermal1", "photons3", "prog4", "xtalk10")}
        assert max(five, key=five.get) == "photons3", (w, five)

    # 6. Reading 5.  Drift.
    assert [round(drift_over(w, "hour"), 2) for w in WORKLOADS] == [0.17, 0.68, 1.95]
    assert [round(drift_over(w, "four"), 2) for w in WORKLOADS] == [0.40, 2.34, 16.54]
    assert [round(drift_over(w, "six"), 2) for w in WORKLOADS] == [-0.01, 0.08, 0.19]
    assert [round(drift_over(w, "cal"), 2) for w in WORKLOADS] == [-0.13, 0.02, -0.09]
    #    An hour: four times MNIST's on one and over eleven on the other, and an
    #    hour on the inverted set is five times MNIST's four hours.
    hours = [drift_over(w, "hour") for w in WORKLOADS]
    assert abs(hours[1] / hours[0] - 4.0) < 0.05 and 11 < hours[2] / hours[0] < 12
    assert abs(hours[2] / drift_over(MNIST, "four") - 4.9) < 0.05
    assert all(abs(sigmas(ROWS[(w, TILE)]["cal"], ROWS[(w, TILE)]["v1"])) < 1.6 for w in WORKLOADS)
    assert all(ROWS[(w, TILE)]["cal"][0] - ROWS[(w, TILE)]["v1"][0] < 0.03 for w in WORKLOADS)
    #    The error: Fashion-MNIST's is MNIST's, to a sixth; the inverted set's
    #    is twice it under a weight's error, an hour's drift, or shot noise.
    assert all(abs(ROW_ERR[k][1] / ROW_ERR[k][0] - 1) < 0.18
               for k in ("thermal1", "photons3", "prog4", "xtalk10", "v1", "hour"))
    assert all(1.85 < ROW_ERR[k][2] / ROW_ERR[k][0] < 2.7 for k in ("photons3", "prog4", "hour"))
    assert abs(ROW_ERR["xtalk10"][2] / ROW_ERR["xtalk10"][0] - 1) < 0.05
    assert all(ROW_ERR["v1"][i] == OUT_ERR[w][0] for i, w in enumerate(WORKLOADS))
    assert round(LIT[INVERTED][0][0] / TILE[0], 2) == 0.74
    #    And far worse on the core's tile, as it was on MNIST.
    assert all(drift_over(w, "hour", CORE) > 3 * drift_over(w, "hour") for w in WORKLOADS)

    # 7. Reading 6.  The laser.
    assert [times_within(w) for w in WORKLOADS] == [8, 32, 16]
    assert [times_put(w, AS_SET) for w in WORKLOADS] == [8, 32, 16]
    assert [times_put(w, BIT_DOWN) for w in WORKLOADS] == [4, 8, 16]
    assert [times_put(w, TWO_DOWN) for w in WORKLOADS] == [None, 8, None]
    watts = [laser.laser_w(TILE, times_put(w, BIT_DOWN)) for w in WORKLOADS]
    assert all(abs(lo - x) < 0.006 and abs(hi - y) < 0.06 for (lo, hi), (x, y) in zip(watts, ((0.35, 3.5), (0.70, 7.0), (1.40, 14.0))))
    #    With a bit down the inverted set is 0.14 over at 8 and under at 16;
    #    Fashion-MNIST is 0.03 over at 8.
    assert abs(put(INVERTED, BIT_DOWN)[8][0] - LOST[INVERTED][0][0] - 0.14) < 0.005
    assert put(INVERTED, BIT_DOWN)[16][0] < LOST[INVERTED][0][0]
    assert abs(put(FASHION, BIT_DOWN)[8][0] - LOST[FASHION][0][0] - 0.03) < 0.005
    mac = [laser.fj_mac(TILE, times_put(w, BIT_DOWN)) for w in WORKLOADS]
    assert [(round(lo), round(hi)) for lo, hi in mac] == [(97, 922), (140, 1351), (226, 2209)]
    #    The receiver's row as a share of a line's light, at those lasers.
    assert [round(1 / (laser.b5_fraction(TILE) / times_put(w, BIT_DOWN))) for w in WORKLOADS] == [16, 32, 64]
    assert times_within(MNIST) == laser.times_within(TILE)
    assert abs(over(FASHION, 16) - 0.10) < 0.005 and over(FASHION, 16) < LOST[FASHION][16][1]
    assert abs(over(INVERTED, 16) - 0.04) < 0.005 and over(FASHION, 32) < 0 and over(INVERTED, 32) < 0
    assert all(over(w, 8) > 0.7 for w in NEW) and abs(over(MNIST, 8) - 0.01) < 0.005
    assert [round(half_lsb(w), 2) for w in WORKLOADS] == [0.15, 0.16, 0.19]
    #    At 8 times, the second layer's noise is about an LSB on MNIST and on
    #    Fashion-MNIST, and MNIST alone does not mind.
    assert abs(NOISE_B5[MNIST][1] / 8 - 0.99) < 0.005 and abs(NOISE_B5[FASHION][1] / 8 - 1.19) < 0.005
    assert abs(NOISE_B5[INVERTED][1] / 8 - 2.78) < 0.005

    # 8. Reading 7 and section 6.  The first row of each is the laser's own.
    for w in WORKLOADS:
        assert all(put(w, AS_SET)[m] == LOST[w][m] for m in put(w, AS_SET)), w
    assert times_put(MNIST, BIT_DOWN) == laser.times_put(BIT_DOWN) == 4
    assert put(FASHION, BIT_DOWN)[4][0] == LOST[FASHION][8][0] == 1.93
    assert (put(INVERTED, BIT_DOWN)[4][0], LOST[INVERTED][8][0]) == (2.29, 1.99)
    assert abs(sigmas(put(INVERTED, BIT_DOWN)[4], LOST[INVERTED][8])) < 1
    assert all(put(w, BIT_DOWN)[m][0] < 0.5 * put(w, AS_SET)[m][0] for w in NEW for m in (2, 4))
    #    What a bit clips: under 1% of firing units on Fashion-MNIST, 2.75% on
    #    the inverted set, and under 0.05 of a point by itself on both.
    assert PUT_CLIPS[FASHION][BIT_DOWN][0] == 0.90 and PUT_CLIPS[INVERTED][BIT_DOWN][0] == 2.75
    assert all(PUT_CLIPS[w][BIT_DOWN][2][0] < 0.05 for w in WORKLOADS)
    assert [PUT_CLIPS[w][TWO_DOWN][0] for w in WORKLOADS] == [13.45, 10.32, 22.27]
    assert [PUT_CLIPS[w][TWO_DOWN][2][0] for w in WORKLOADS] == [0.26, 0.21, 0.61]
    #    An 8-bit converter's full scale on the inverted set is about 32 or 64
    #    lines' worth of sum, and the mean shot lights 95.
    full = sorted({256 * 2 ** s[0] / laser.line_light() for s in S8[INVERTED]})
    assert abs(full[0] - 32.25) < 0.01 and abs(full[1] - 64.5) < 0.01 and LIT[INVERTED][0][0] > full[-1]
    assert round(LIT[MNIST][0][0]) < full[0]
    #    A second bit: nothing on MNIST, pays on Fashion-MNIST at every laser
    #    run to 8, pays on the inverted set only at 2.
    assert all(put(MNIST, TWO_DOWN)[m][0] > put(MNIST, BIT_DOWN)[m][0] for m in (2, 4, 8))
    assert all(put(FASHION, TWO_DOWN)[m][0] < put(FASHION, BIT_DOWN)[m][0] for m in (2, 4, 8))
    assert put(FASHION, TWO_DOWN)[4][0] == 1.29
    assert put(INVERTED, TWO_DOWN)[2][0] < put(INVERTED, BIT_DOWN)[2][0]
    assert put(INVERTED, TWO_DOWN)[8][0] > put(INVERTED, BIT_DOWN)[8][0]
    assert all(set(put(w, how)) == {2, 4, 8, 16} for w in NEW for how in PUT[w])
    assert put(FASHION, TWO_DOWN)[16][0] < put(FASHION, BIT_DOWN)[16][0] < LOST[FASHION][0][0]
    #    The weights' gain: worse than none at every laser on all three.
    for w in WORKLOADS:
        assert all(put(w, (0, 1))[m][0] > put(w, AS_SET)[m][0] for m in put(w, AS_SET)), w
    assert PUT_CLIPS[INVERTED][(0, 1)][1] == 0.15 and PUT_CLIPS[INVERTED][(0, 1)][2][0] == 3.05
    assert PUT_CLIPS[INVERTED][(0, 2)][2][0] > 25 and PUT_CLIPS[MNIST][(0, 1)][2][0] == 0.14

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
