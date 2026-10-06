"""
What would a tighter budget buy, and what would it cost?

pta_workload.py found the interface chip's budget, version 1, at over a point
on two data sets where it costs a third of one on MNIST, and could not say
which of version 1's six rows the point is in.  grx930's harness has now run
version 1 with each row made better, alone and together, on all three.

MEASURED IN A MODEL, by grx930 (c930/doc/pta_error_model_design_note.md section
5, "What would tightening v1 buy?", 2026-10-06; `sim/pta_mnist.sh DIR WORK
tighten`).  On the working tile, 128 x 64, five networks a data set.  What a
setting buys is version 1's loss less its own, network by network, with the
standard error of that.  "A notch" is a bit more in a converter and half the
noise or the error: 7 activation bits, an 8-bit ADC, receiver noise of a
quarter of an 8-bit LSB, 30 photons such an LSB, programming error of half an
8-bit weight's LSB, and crosstalk of 1.2%, which is the nearest the model's
field comes to half of 2%.

DERIVED here, from the plan's own models, for the working tile at 1 GS/s:

  the ADC's bit    pta_power.py's price of a converter, from the published
                   survey: the best and the fifth-best part of that many
                   effective bits at that rate, 64 of them
  the ring         pta_rate.py's fourth bound.  A level read to a bit more has
                   to have settled further in the same shot, so the narrowest
                   line a ring may have is wider: a lower Q and a larger swing
  the buses        lines a bus at that line and B12's spacing, and so the buses;
                   and the window of Q between a ring too sharp to settle and
                   one too broad for 64 lines to share a bus
  the crosstalk    what B12's grid already is, by the Lorentzian pta_ring.py
                   reads a spacing from, for a ring at the rate's Q ceiling
  the laser        grx930's multiple of B5's, times pta_laser.py's watts
  a MAC            the chip and the laser over the tile's MACs a second

NOT PRICED, because no model here prices them: the activation DAC's seventh
bit, and programming error of half an LSB.  Neither is free.

WHAT THIS IS NOT: a decision.  Version 1 is the requirement until the plan says
otherwise.  And everything pta_workload.py is not: no network here was trained
with the tile's errors in the loop, the trainer was MNIST's, and all three data
sets are 28 x 28 images through fully connected layers.

Standard library only.  Run:  python3 docs/designs/pta_tighten.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_adc_survey as adc
import pta_laser as laser
import pta_power as power
import pta_rate as rate
import pta_ring as ring
import pta_source as source
import pta_working_point as wp
import pta_workload as workload

MNIST, FASHION, INVERTED = workload.MNIST, workload.FASHION, workload.INVERTED
WORKLOADS = workload.WORKLOADS
TILE, FS = wp.TILE, wp.FS                      # 128 x 64 (B10), 1 GS/s (B11)
WITHIN = laser.WITHIN
V1_BITS = wp.BITS                              # 7: version 1's ADC

# ---- grx930's figures ----------------------------------------------------------
# Points lost against the same weights on the host: MNIST, Fashion-MNIST, MNIST inverted.
LOST = {
    "v1": ((0.34, 0.07), (1.16, 0.14), (1.23, 0.13)),
    "adc_noise": ((0.15, 0.02), (0.54, 0.22), (0.62, 0.07)),
    "all": ((0.06, 0.02), (0.39, 0.15), (0.50, 0.08)),
    "two": ((0.03, 0.02), (0.18, 0.08), (0.22, 0.06)),
    "adc_noise_gone": ((0.11, 0.03), (0.22, 0.12), (0.48, 0.06)),
    "converters": ((0.12, 0.07), (0.37, 0.08), (0.27, 0.06)),
    "converters_notch": ((0.02, 0.04), (0.02, 0.09), (0.10, 0.06)),
}
# What a setting buys of version 1, network by network.
ROWS = ("the activation DAC", "the ADC", "the receiver's noise", "the shot noise", "programming error", "crosstalk")
GONE = (
    ((0.06, 0.03), (-0.01, 0.03), (0.06, 0.05)),
    ((0.08, 0.08), (0.26, 0.09), (0.15, 0.06)),
    ((0.17, 0.03), (0.31, 0.14), (0.20, 0.07)),
    ((0.20, 0.03), (0.26, 0.11), (0.52, 0.14)),
    ((0.05, 0.03), (-0.04, 0.07), (0.12, 0.06)),
    ((-0.01, 0.03), (-0.08, 0.11), (0.13, 0.09)),
)
NOTCH = (
    ((0.03, 0.02), (0.02, 0.05), (0.10, 0.08)),
    ((0.07, 0.08), (0.29, 0.06), (0.15, 0.08)),
    ((0.13, 0.05), (0.10, 0.07), (0.17, 0.04)),
    ((0.12, 0.03), (0.11, 0.11), (0.26, 0.07)),
    ((0.04, 0.02), (-0.05, 0.07), (0.13, 0.04)),
    ((0.02, 0.03), (-0.06, 0.08), (0.06, 0.07)),
)
TOGETHER = {
    "converters": ((0.14, 0.05), (0.24, 0.07), (0.26, 0.03)),
    "noise": ((0.17, 0.04), (0.30, 0.12), (0.43, 0.10)),
    "weights": ((0.03, 0.03), (0.08, 0.07), (0.08, 0.02)),
    "adc_noise": ((0.19, 0.06), (0.62, 0.10), (0.61, 0.06)),
    "others": ((0.09, 0.05), (0.04, 0.03), (0.11, 0.07)),
    "all": ((0.28, 0.05), (0.76, 0.07), (0.72, 0.06)),
    "two": ((0.31, 0.07), (0.98, 0.07), (1.01, 0.11)),
    "adc_noise_gone": ((0.23, 0.05), (0.93, 0.09), (0.75, 0.09)),
}
# Under a laser of that many times B5's, the hidden rescale a bit down; 0 is
# the set as budgeted, at the rule's rescale.  "light" is version 1's rows with
# an 8-bit ADC and 30 photons, the laser being the receiver's noise.
LASER = {
    "v1": (
        {0: (0.34, 0.07), 4: (0.35, 0.06), 8: (0.22, 0.06), 16: (0.19, 0.06), 32: (0.19, 0.07), 64: (0.18, 0.09)},
        {0: (1.16, 0.14), 4: (1.93, 0.34), 8: (1.19, 0.13), 16: (0.96, 0.07), 32: (0.88, 0.10), 64: (0.84, 0.10)},
        {0: (1.23, 0.13), 4: (2.29, 0.29), 8: (1.37, 0.16), 16: (1.17, 0.12), 32: (1.12, 0.13), 64: (1.11, 0.11)},
    ),
    "light": (
        {0: (0.15, 0.02), 4: (0.25, 0.06), 8: (0.11, 0.03), 16: (0.09, 0.04), 32: (0.12, 0.04), 64: (0.09, 0.04)},
        {0: (0.54, 0.22), 4: (1.57, 0.35), 8: (0.74, 0.20), 16: (0.55, 0.17), 32: (0.42, 0.18), 64: (0.40, 0.15)},
        {0: (0.62, 0.07), 4: (1.76, 0.22), 8: (0.91, 0.14), 16: (0.64, 0.11), 32: (0.63, 0.12), 64: (0.67, 0.11)},
    ),
    "all": (
        {0: (0.06, 0.02), 4: (0.23, 0.04), 8: (0.08, 0.03), 16: (0.12, 0.03), 32: (0.11, 0.02), 64: (0.09, 0.01)},
        {0: (0.39, 0.15), 4: (1.44, 0.32), 8: (0.68, 0.17), 16: (0.40, 0.15), 32: (0.38, 0.13), 64: (0.31, 0.16)},
        {0: (0.50, 0.08), 4: (1.63, 0.25), 8: (0.75, 0.13), 16: (0.54, 0.12), 32: (0.53, 0.12), 64: (0.49, 0.09)},
    ),
}
VERSIONS = (
    ("v1", "v1", "version 1", V1_BITS),
    ("adc_noise", "light", "the ADC's bit, half the noise", V1_BITS + 1),
    ("all", "all", "all six a notch", V1_BITS + 1),
)
NOTCH_XTALK = 3 / 256                          # what the model ran for "half of 2%"


# ---- what follows from grx930's ----------------------------------------------------
def i(w):
    return WORKLOADS.index(w)


def times(which, w, budget=WITHIN):
    """The least multiple of B5's laser at which that set of rows is within the
    budget of its own figure as budgeted, every larger one being so too."""
    d = LASER[which][i(w)]
    ok = None
    for t in sorted((t for t in d if t), reverse=True):
        if d[t][0] - d[0][0] >= budget - 1e-9:
            break
        ok = t
    return ok


def share(w):
    """What a notch on the ADC and both noise rows buys, of what a notch on all six buys."""
    return TOGETHER["adc_noise"][i(w)][0] / TOGETHER["all"][i(w)][0]


# ---- what a bit costs: the plan's models -------------------------------------------
def adcs_w(bits):
    """The tile's converters: the best and the fifth-best published part."""
    lo, hi = power.adc_w(bits, FS)
    return TILE[1] * lo, TILE[1] * hi


def line_hz(bits):
    """The narrowest line a weight ring may have and settle to that many bits in a shot."""
    return rate.line_hz(FS, bits)


def q_ceiling(bits):
    return ring.linewidth_hz(1.0) / line_hz(bits)


def swing_floor_v(bits):
    return ring.swing_v(q_ceiling(bits))


def lines_a_bus(bits):
    """At that line and B12's spacing, the published worst."""
    return int(source.FSR_HZ / (source.M_SPACING[1] * line_hz(bits)))


def buses(bits):
    return -(-TILE[0] // lines_a_bus(bits))


def q_floor():
    """The lowest Q at which the tile's lines a bus still fit one free spectral
    range at B12's spacing: below it the tile needs another bus."""
    line_max = source.FSR_HZ / (wp.lines_each(TILE) * source.M_SPACING[1])
    return ring.linewidth_hz(1.0) / line_max


def q_window(bits):
    """How far above that floor the ceiling is: the room a ring's Q has."""
    return q_ceiling(bits) / q_floor() - 1


def crosstalk_at(spacing):
    """A neighbour's ring seen from that many linewidths away: pta_ring.py's, turned round."""
    return 1 / (1 + 4 * spacing ** 2)


def grid_crosstalk(bits):
    """What B12's grid is, for a ring at the rate's Q ceiling."""
    return crosstalk_at(wp.grid_hz(TILE) / line_hz(bits))


def chip_parts(bits):
    """pta_working_point.py's parts with the ADC at that many bits, and the
    weight drive at the swing a ring that settles to them needs."""
    p = dict(wp.chip_parts(TILE))
    p["ADCs"] = adcs_w(bits)
    floor = swing_floor_v(bits)
    p["weight drive"] = tuple(
        power.weight_drive_w(TILE[0], TILE[1], max(v, floor), c, a, FS)
        for v, c, a in zip(rate.SWINGS_V, power.C_CELL_F, power.ACTIVITY))
    return p


def chip_w(bits):
    p = chip_parts(bits)
    return sum(x[0] for x in p.values()), sum(x[1] for x in p.values())


def laser_w(which, w):
    return laser.laser_w(TILE, times(which, w))


def fj_mac(which, bits, w):
    c, l = chip_w(bits), laser_w(which, w)
    return tuple((c[k] + l[k]) / power.mac_s(*TILE, FS) * 1e15 for k in (0, 1))


def pm(x):
    return f"{x[0]:.2f} +-{x[1]:.2f}"


def span(x, digits=2):
    return f"{x[0]:.{digits}f}-{x[1]:.{digits}f}"


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    print("What would a tighter budget buy, and what would it cost?")
    print(f"The working tile, {laser.name(TILE)} at {FS / 1e9:.0f} GS/s.  Five networks a data set.")

    section("1. What each of version 1's rows buys, made better alone")
    print(f"  {'':<24}" + "".join(f"{w:>18}" for w in WORKLOADS))
    print(f"  {'version 1 loses':<24}" + "".join(f"{pm(x):>18}" for x in LOST["v1"]))
    print("  taken away:")
    for name, row in zip(ROWS, GONE):
        print(f"    {name:<22}" + "".join(f"{pm(x):>18}" for x in row))
    print(f"    {'summed':<22}" + "".join(f"{sum(r[k][0] for r in GONE):>18.2f}" for k in range(3)))
    print("  a notch tighter:")
    for name, row in zip(ROWS, NOTCH):
        print(f"    {name:<22}" + "".join(f"{pm(x):>18}" for x in row))
    print(f"    {'summed':<22}" + "".join(f"{sum(r[k][0] for r in NOTCH):>18.2f}" for k in range(3)))

    section("2. A notch together")
    for key, label in (("converters", "both converters"), ("noise", "both noise rows"), ("weights", "both of a weight's rows"),
                       ("adc_noise", "the ADC and both noise rows"), ("others", "the other three"),
                       ("all", "all six"), ("two", "all six, two notches"),
                       ("adc_noise_gone", "the ADC and the noise gone")):
        print(f"  {label:<30}" + "".join(f"{pm(x):>18}" for x in TOGETHER[key]))
    print(f"  {'the three, of all six':<30}" + "".join(f"{share(w):>18.0%}" for w in WORKLOADS))

    section("3. Three sets of rows, and what each loses")
    print(f"  {'':<34}" + "".join(f"{w:>18}" for w in WORKLOADS))
    for key, _, label, _ in VERSIONS:
        print(f"  {label:<34}" + "".join(f"{pm(x):>18}" for x in LOST[key]))
    for key, label in (("two", "all six, two notches"), ("converters", "v1's two converters, nothing else"),
                       ("converters_notch", "a notch tighter, nothing else")):
        print(f"  {label:<34}" + "".join(f"{pm(x):>18}" for x in LOST[key]))

    section("4. Under a laser, the hidden rescale a bit down")
    for w in WORKLOADS:
        print(f"  {w}")
        print(f"    {'':<16}" + "".join(f"{label:>32}" for _, _, label, _ in VERSIONS))
        for t in (0, 4, 8, 16, 32, 64):
            print(f"    {('as budgeted' if t == 0 else f'laser x{t}'):<16}"
                  + "".join(f"{pm(LASER[which][i(w)][t]):>32}" for _, which, _, _ in VERSIONS))
        print(f"    {'within a tenth':<16}" + "".join(f"{f'{times(which, w)} times':>32}" for _, which, _, _ in VERSIONS))

    section("5. What the tighter rows cost, by the plan's models")
    print(f"  {'':<38}" + "".join(f"{label:>32}" for _, _, label, _ in VERSIONS))
    print(f"  {'the ADC':<38}" + "".join(f"{f'{bits} bits':>32}" for *_, bits in VERSIONS))
    print(f"  {'published parts that do it':<38}" + "".join(f"{len(adc.able(bits, FS)):>32}" for *_, bits in VERSIONS))
    print(f"  {'the 64 converters':<38}" + "".join(f"{span(adcs_w(bits)) + ' W':>32}" for *_, bits in VERSIONS))
    print(f"  {'the interface chip':<38}" + "".join(f"{span(chip_w(bits)) + ' W':>32}" for *_, bits in VERSIONS))
    print(f"  {'a ring' + chr(39) + 's line, at least':<38}" + "".join(f"{f'{line_hz(bits) / 1e9:.2f} GHz':>32}" for *_, bits in VERSIONS))
    print(f"  {'its Q, at most':<38}" + "".join(f"{q_ceiling(bits):>32,.0f}" for *_, bits in VERSIONS))
    print(f"  {'its swing, at least':<38}" + "".join(f"{f'{swing_floor_v(bits):.2f} V':>32}" for *_, bits in VERSIONS))
    print(f"  {'lines a bus, and buses':<38}" + "".join(f"{f'{lines_a_bus(bits)}, {buses(bits)}':>32}" for *_, bits in VERSIONS))
    print(f"  {'Q that keeps those buses, at least':<38}" + "".join(f"{q_floor():>32,.0f}" for _ in VERSIONS))
    print(f"  {'so the room a ring' + chr(39) + 's Q has':<38}" + "".join(f"{q_window(bits):>32.0%}" for *_, bits in VERSIONS))
    print(f"  {'and its swing, between':<38}" + "".join(
        f"{f'{swing_floor_v(bits):.2f} and {ring.swing_v(q_floor()):.2f} V':>32}" for *_, bits in VERSIONS))
    print(f"  {'crosstalk B12' + chr(39) + 's grid already is':<38}" + "".join(f"{grid_crosstalk(bits):>32.2%}" for *_, bits in VERSIONS))
    for w in WORKLOADS:
        print(f"  {'laser, ' + w:<38}" + "".join(
            f"{f'{times(which, w)} times, ' + span(laser_w(which, w)) + ' W':>32}" for _, which, _, _ in VERSIONS))
    for w in WORKLOADS:
        print(f"  {'a MAC, ' + w:<38}" + "".join(
            f"{'{:.0f}-{:,.0f} fJ'.format(*fj_mac(which, bits, w)):>32}" for _, which, _, bits in VERSIONS))
    print(f"  {'and, not priced':<38}{'':>32}{'':>32}{'a 7-bit activation DAC,':>32}")
    print(f"  {'':<38}{'':>32}{'':>32}{'programming error 0.5 LSB':>32}")
    print(f"  A ninth bit, which two notches ask, holds {lines_a_bus(V1_BITS + 2)} lines a bus: {buses(V1_BITS + 2)} buses.")

    findings()
    checks()


def findings():
    f, m, v = i(FASHION), i(MNIST), i(INVERTED)
    print()
    print("What this says, seven readings.")
    print()
    print(f"  1. ON FASHION-MNIST VERSION 1'S POINT IS IN THREE ROWS.  The ADC's quantisation, the")
    print(f"     receiver's noise and the shot noise buy {GONE[1][f][0]:.2f}, {GONE[2][f][0]:.2f} and {GONE[3][f][0]:.2f} taken away.  The activation")
    print(f"     DAC's sixth bit, programming error and crosstalk buy {GONE[0][f][0]:.2f}, {GONE[4][f][0]:.2f} and {GONE[5][f][0]:.2f}: nothing,")
    print("     each inside its error.  Those three are dear to loosen and free to hold.")
    print()
    print(f"  2. THE INVERTED SET SPREADS IT, AND MNIST IS ITS NOISE.  On the inverted set the shot")
    print(f"     noise is {GONE[3][v][0]:.2f} by itself and every other row {min(r[v][0] for k, r in enumerate(GONE) if k != 3):.2f} to {max(r[v][0] for k, r in enumerate(GONE) if k != 3):.2f}.  On MNIST the two")
    print(f"     noise rows are {GONE[2][m][0]:.2f} and {GONE[3][m][0]:.2f} and no other row is over {max(r[m][0] for k, r in enumerate(GONE) if k not in (2, 3)):.2f}.")
    print()
    print(f"  3. A NOTCH ON ALL SIX PUTS FASHION-MNIST WHERE VERSION 1 PUTS MNIST.  It loses {LOST['all'][m][0]:.2f},")
    print(f"     {LOST['all'][f][0]:.2f} and {LOST['all'][v][0]:.2f}, where version 1 loses {LOST['v1'][m][0]:.2f}, {LOST['v1'][f][0]:.2f} and {LOST['v1'][v][0]:.2f}.  Two notches: {LOST['two'][m][0]:.2f}, {LOST['two'][f][0]:.2f}")
    print(f"     and {LOST['two'][v][0]:.2f}.")
    print()
    print(f"  4. THREE ROWS ARE FOUR FIFTHS OF IT.  A notch on the ADC and both noise rows buys")
    print(f"     {share(FASHION):.0%} of what all six buy on Fashion-MNIST and {share(INVERTED):.0%} on the inverted set, and {share(MNIST):.0%} on")
    print(f"     MNIST: it loses {LOST['adc_noise'][m][0]:.2f}, {LOST['adc_noise'][f][0]:.2f} and {LOST['adc_noise'][v][0]:.2f}.  A notch on the other three, by itself,")
    print(f"     buys {min(x[0] for x in TOGETHER['others']):.2f} to {max(x[0] for x in TOGETHER['others']):.2f}.")
    print()
    a7, a8 = adcs_w(V1_BITS), adcs_w(V1_BITS + 1)
    print(f"  5. THE BIT IS A SEVENTH TO A FIFTH OF A WATT, AND A RING THAT IS HARDER TO HIT.  64")
    print(f"     converters go from {span(a7)} W to {span(a8)} W, on {len(adc.able(V1_BITS + 1, FS))} published parts where there were {len(adc.able(V1_BITS, FS))}.")
    print(f"     A ring has to settle a bit further in the same shot, so its line is at least")
    print(f"     {line_hz(V1_BITS + 1) / 1e9:.2f} GHz for {line_hz(V1_BITS) / 1e9:.2f} and its Q at most {q_ceiling(V1_BITS + 1):,.0f} for {q_ceiling(V1_BITS):,.0f}.  Two buses of {wp.lines_each(TILE)} lines")
    print(f"     want a Q of at least {q_floor():,.0f}.  So the room between the two goes from {q_window(V1_BITS):.0%} to")
    print(f"     {q_window(V1_BITS + 1):.0%}, and a ninth bit closes it: {lines_a_bus(V1_BITS + 2)} lines a bus, and a third bus.")
    print()
    print(f"  6. HALF THE NOISE IS TWICE THE LASER, EXCEPT WHERE THE LASER WAS ALREADY LARGE.  Within")
    print(f"     a tenth of their own budget the tighter rows take {times('light', MNIST)}, {times('light', FASHION)} and {times('light', INVERTED)} times B5's laser, where")
    print(f"     version 1's take {times('v1', MNIST)}, {times('v1', FASHION)} and {times('v1', INVERTED)}: {span(laser_w('light', MNIST))} W, {span(laser_w('light', FASHION), 1)} W and {span(laser_w('light', INVERTED), 1)} W.  And")
    print(f"     at one laser the rows are worth more than more laser: at 8 times, Fashion-MNIST")
    print(f"     loses {LASER['v1'][f][8][0]:.2f} at version 1's rows and {LASER['light'][f][8][0]:.2f} with the bit and 30 photons, and 64 times")
    print(f"     at version 1's rows only gets it to {LASER['v1'][f][64][0]:.2f}.")
    print()
    print(f"  7. THE CROSSTALK'S NOTCH IS ALREADY B12'S, AND TWO THINGS ARE NOT PRICED.  B12 packs a")
    print(f"     bus at {source.M_SPACING[1]} linewidths, where a neighbour is seen at {crosstalk_at(source.M_SPACING[1]):.1%}, and on its {wp.grid_hz(TILE) / 1e9:.0f} GHz grid a ring")
    print(f"     at the rate's Q ceiling sees {grid_crosstalk(V1_BITS):.1%}, or {grid_crosstalk(V1_BITS + 1):.1%} settling to 8 bits.  The notch the model ran")
    print(f"     is {NOTCH_XTALK:.1%}.  The activation DAC's seventh bit and half the programming error are")
    print("     what all six ask beyond the three, and no model here prices either.")


def checks():
    """Every claim above, as an assert."""
    f, m, v = i(FASHION), i(MNIST), i(INVERTED)
    assert TILE == (128, 64) and FS == 1e9 and V1_BITS == 7 and (m, f, v) == (0, 1, 2)

    # 0. The figures agree with each other and with what the plan already holds.
    assert all(LOST["v1"][i(w)] == workload.LOST[w][0] for w in WORKLOADS)
    assert all(LASER[which][k][0] == LOST[key][k] for key, which, _, _ in VERSIONS for k in range(3))
    #    Version 1's rows under a laser with the rescale a bit down are
    #    pta_workload.py's, from grx930's `fill`, where both ran them.
    for w in WORKLOADS:
        put = workload.put(w, workload.BIT_DOWN)
        assert all(LASER["v1"][i(w)][t] == put[t] for t in (4, 8, 16) if t in put), w
    #    What a setting buys is version 1's loss less its own, to the rounding.
    for key in ("adc_noise", "all", "two", "adc_noise_gone"):
        assert all(abs(LOST["v1"][k][0] - LOST[key][k][0] - TOGETHER[key][k][0]) < 0.011 for k in range(3)), key

    # 1. Reading 1.  Fashion-MNIST: three rows, and three that buy nothing.
    assert [GONE[r][f][0] for r in (1, 2, 3)] == [0.26, 0.31, 0.26]
    assert [GONE[r][f][0] for r in (0, 4, 5)] == [-0.01, -0.04, -0.08]
    assert all(abs(GONE[r][f][0]) < 1.4 * GONE[r][f][1] for r in (0, 4, 5))
    assert all(GONE[r][f][0] > 2 * GONE[r][f][1] for r in (1, 2, 3))
    #    One more bit is all the ADC has to give there.
    assert NOTCH[1][f][0] >= GONE[1][f][0] and (LOST["converters"][f][0], LOST["converters_notch"][f][0]) == (0.37, 0.02)
    #    Relaxed, those three rows were dear: pta_workload.py's.
    assert [workload.relax(FASHION)[r] for r in (0, 4, 5)] == [0.30, 0.65, 0.78]

    # 2. Reading 2.
    rest = [GONE[r][v][0] for r in (0, 1, 2, 4, 5)]
    assert GONE[3][v][0] == 0.52 and (min(rest), max(rest)) == (0.06, 0.20)
    assert (GONE[2][m][0], GONE[3][m][0]) == (0.17, 0.20) and max(GONE[r][m][0] for r in (0, 1, 4, 5)) == 0.08
    assert max(range(6), key=lambda r: GONE[r][v][0]) == 3 and max(range(6), key=lambda r: GONE[r][m][0]) == 3
    #    The rows gone one at a time, summed, against what version 1 loses.
    sums = [round(sum(r[k][0] for r in GONE), 2) for k in range(3)]
    assert sums == [0.55, 0.70, 1.18] and sums[m] > LOST["v1"][m][0] and sums[f] < 0.65 * LOST["v1"][f][0]
    #    With the three gone the inverted set still loses half a point.
    assert [x[0] for x in LOST["adc_noise_gone"]] == [0.11, 0.22, 0.48]

    # 3. Reading 3.
    assert [x[0] for x in LOST["all"]] == [0.06, 0.39, 0.50] and [x[0] for x in LOST["two"]] == [0.03, 0.18, 0.22]
    assert abs(LOST["all"][f][0] - LOST["v1"][m][0]) < 0.06
    assert all(LOST["two"][k][0] <= 0.5 * LOST["all"][k][0] + 0.005 for k in range(3))
    assert all(LOST["all"][k][0] < 0.45 * LOST["v1"][k][0] for k in range(3))

    # 4. Reading 4.
    assert [round(share(w), 2) for w in WORKLOADS] == [0.68, 0.82, 0.85]
    assert [x[0] for x in LOST["adc_noise"]] == [0.15, 0.54, 0.62]
    assert (min(x[0] for x in TOGETHER["others"]), max(x[0] for x in TOGETHER["others"])) == (0.04, 0.11)
    #    The notches do not add either: singly they sum to 0.41, 0.41 and 0.87,
    #    and all six together buy 0.28, 0.76 and 0.72.
    assert [round(sum(r[k][0] for r in NOTCH), 2) for k in range(3)] == [0.41, 0.41, 0.87]
    assert [x[0] for x in TOGETHER["all"]] == [0.28, 0.76, 0.72] and [x[0] for x in TOGETHER["two"]] == [0.31, 0.98, 1.01]
    assert [x[0] for x in TOGETHER["converters"]] == [0.14, 0.24, 0.26]
    assert [x[0] for x in TOGETHER["noise"]] == [0.17, 0.30, 0.43]
    assert [x[0] for x in TOGETHER["weights"]] == [0.03, 0.08, 0.08]
    assert [x[0] for x in TOGETHER["adc_noise"]] == [0.19, 0.62, 0.61]
    assert [NOTCH[r][f][0] for r in range(6)] == [0.02, 0.29, 0.10, 0.11, -0.05, -0.06]
    #    The three and the other three, summed, are all six to a tenth.
    assert all(abs(TOGETHER["adc_noise"][k][0] + TOGETHER["others"][k][0] - TOGETHER["all"][k][0]) <= 0.10 for k in range(3))

    # 5. Reading 5.  The bit.
    a7, a8 = adcs_w(7), adcs_w(8)
    assert abs(a7[0] - 0.071) < 0.001 and abs(a7[1] - 0.201) < 0.001
    assert abs(a8[0] - 0.268) < 0.001 and abs(a8[1] - 0.342) < 0.001
    assert a7 == wp.chip_parts(TILE)["ADCs"]
    assert 0.19 < a8[0] - a7[0] < 0.20 and 0.14 < a8[1] - a7[1] < 0.15
    assert len(adc.able(8, FS)) < len(adc.able(7, FS)) and len(adc.able(8, FS)) >= 5
    c7, c8 = chip_w(7), chip_w(8)
    assert all(abs(x - y) < 1e-12 for x, y in zip(c7, wp.chip_w(TILE)))
    assert abs(c8[0] - 0.64) < 0.005 and abs(c8[1] - 4.18) < 0.005
    assert abs(line_hz(7) / 1e9 - 1.99) < 0.005 and abs(line_hz(8) / 1e9 - 2.21) < 0.005
    assert abs(q_ceiling(7) - rate.q_ceiling(FS)) < 1e-6 and abs(swing_floor_v(7) - rate.swing_floor_v(FS)) < 1e-9
    assert abs(q_ceiling(8) - 87_662) < 1 and abs(swing_floor_v(8) - 2.53) < 0.005
    assert (lines_a_bus(7), lines_a_bus(8), lines_a_bus(9)) == (77, 69, 63)
    assert lines_a_bus(7) == wp.lines_a_bus() and (buses(7), buses(8), buses(9)) == (2, 2, 3)
    assert buses(7) == wp.buses(TILE)
    assert (len(adc.able(7, FS)), len(adc.able(8, FS))) == (70, 47)
    assert abs(q_ceiling(8) / q_ceiling(7) - 0.9) < 1e-9 and abs(line_hz(8) / line_hz(7) - 10 / 9) < 1e-9
    #    The window of Q: 21% at 7 bits, 9% at 8, none at 9.
    assert wp.lines_each(TILE) == 64 and abs(q_floor() - 80_230) < 1
    assert abs(q_window(7) - 0.214) < 0.001 and abs(q_window(8) - 0.093) < 0.001 and q_window(9) < 0
    assert abs(ring.swing_v(q_floor()) - 2.76) < 0.005
    #    At the floor a bus holds exactly the tile's lines, and one fewer just under it.
    line_at = lambda q: ring.linewidth_hz(1.0) / q
    assert int(source.FSR_HZ / (source.M_SPACING[1] * line_at(q_floor() * 1.0001))) == 64
    assert int(source.FSR_HZ / (source.M_SPACING[1] * line_at(q_floor() * 0.9999))) == 63

    # 6. Reading 6.  The laser.
    assert [times("v1", w) for w in WORKLOADS] == [4, 8, 16]
    assert [times("v1", w) for w in WORKLOADS] == [workload.times_put(w, workload.BIT_DOWN) for w in WORKLOADS]
    assert [times("light", w) for w in WORKLOADS] == [8, 16, 16] == [times("all", w) for w in WORKLOADS]
    assert all(abs(lo - x) < 0.006 and abs(hi - y) < 0.06 for (lo, hi), (x, y) in
               zip([laser_w("light", w) for w in WORKLOADS], ((0.70, 7.0), (1.40, 14.0), (1.40, 14.0))))
    assert (LASER["v1"][f][8][0], LASER["light"][f][8][0], LASER["v1"][f][64][0]) == (1.19, 0.74, 0.84)
    assert (LASER["v1"][v][8][0], LASER["light"][v][8][0], LASER["v1"][v][64][0]) == (1.37, 0.91, 1.11)
    assert all(LASER["light"][k][8][0] < LASER["v1"][k][64][0] for k in range(3))
    #    Of the bit and the photons, alone and as budgeted: the bit on
    #    Fashion-MNIST, the photons on the inverted set.
    assert NOTCH[1][f][0] > NOTCH[3][f][0] and NOTCH[3][v][0] > NOTCH[1][v][0]
    mac = {which: [fj_mac(which, bits, w) for w in WORKLOADS] for _, which, _, bits in VERSIONS}
    assert [(round(lo), round(hi)) for lo, hi in mac["v1"]] == [(97, 922), (140, 1351), (226, 2209)]
    assert [(round(lo), round(hi)) for lo, hi in mac["light"]] == [(164, 1368), (250, 2226), (250, 2226)], mac["light"]

    # 7. Reading 7.  The crosstalk.
    assert abs(crosstalk_at(ring.spacing(ring.XTALK["v1"])) - ring.XTALK["v1"]) < 1e-12
    assert abs(crosstalk_at(source.M_SPACING[1]) - 0.0116) < 0.0001 < NOTCH_XTALK
    assert abs(NOTCH_XTALK - 0.0117) < 0.0001 and wp.grid_hz(TILE) == 11e9
    assert abs(grid_crosstalk(7) - 0.0081) < 0.0001 and abs(grid_crosstalk(8) - 0.0100) < 0.0001
    assert grid_crosstalk(8) < NOTCH_XTALK < ring.XTALK["v1"]
    #    A ring of lower Q on that grid is over version 1's 2% already: the 5 V one sees 3.8%.
    five = crosstalk_at(wp.grid_hz(TILE) / ring.linewidth_hz(ring.q_for_swing(max(rate.SWINGS_V))))
    assert abs(five - 0.038) < 0.0005 and five > ring.XTALK["v1"]

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
