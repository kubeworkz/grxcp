"""
B5's laser, against the light a column is actually sent.

B5 sized the tile's laser from a detector's full scale.  Take a receiver's
noise, hold it to the budget's allowance of an 8-bit LSB, and the 256 LSB of a
full scale are so much light; a laser is that, times the columns, times the
loss.  Every laser figure in this plan since is that method at some receiver:
0.16 to 1.6 W, then 0.33 to 3.3, then 0.09 to 0.88.

The method takes the light at a detector when its converter reads full scale to
be the light the column was sent.  It is not.  A column is sent every line at
full power, whatever the inputs are, because a source does not know them.  An
input modulator passes its activation's share of a line and dumps the rest.
A ring sends what is left to one photodiode of a pair or to the other.  The
converter's full scale is set where the sums actually fall, and on a tile of
256 inputs they fall far below 256 lines' worth.

So the budget's receiver row, half an LSB, is half an LSB of a converter whose
full scale is a fraction of the light B5 counted.  grx930's harness can now
state that row as one laser fixes it, and has run it.

MEASURED IN A MODEL, by grx930 (c930/doc/pta_error_model_design_note.md section
5, "What does a laser of a given size cost?"; `sim/pta_mnist.sh MNIST WORK
laser`, 2026-10-05).  Points lost on D3 against the same weights on the host,
over version 1's other rows, five networks, mean and standard error, with the
receiver's noise set by a laser of B5's size and of 2 to 64 times it.  And the
8-bit converter's shift the harness's rule gives each layer on each tile, from
its recorded runs.

DERIVED here:

  a line's light   an input at full scale through a weight of one: 127 * 128 of
                   a sum's units at 8-bit operands.  grx930's unit
  what a column    its rows, times that
  can be sent
  the fill         a converter's full scale, 256 LSB at the layer's shift, over
                   what the column can be sent.  B5's method takes it for one
  B5's noise       so a laser of B5's size puts the receiver's noise at
                   rows / 512 of a line's light under version 1's half an LSB,
                   and at 1 / (2 * fill) LSB of the converter the budget means
  the laser        B5's, times whatever multiple the accuracy asks
  a receiver that  its noise bandwidth is half the shot rate where a single
  averages         pole's is pi/2 of its corner, so at one noise density its
                   noise is less by the root of the ratio, and the laser with it

ASSUMED, and marked again where each is used:

  - that the measured receiver's noise density carries to a receiver that
    averages over a shot (B12), and to one with eight photodiodes on its input
  - the harness's rule for a converter's shift: the smallest that clips one sum
    in ten thousand.  Another rule moves every fill
  - one network family, on MNIST, whose images are mostly dark.  A workload
    that lights more of its inputs fills more of the light

B10 WAS REVISED ON THIS, 2026-10-05: the working tile is 128 x 64, which this
file calls SMALLER.  It is kept as it was run, with 256 x 64 as WORKING.

AND HALF OF IT IS THE HOST'S TO GIVE BACK.  The fill is the network's as much as
the tile's: a layer's sums fall where its operands do, and the host sets two of
them.  grx930's harness ran both on 128 x 64 (its design note, section 5, "How
a network is put on the tile"; `sim/pta_mnist.sh MNIST WORK fill`): the hidden
layer's rescale, which on the chiplet is the activation stage's shift, and the
scale the first layer's weights are written at.  Section 5.

Standard library only.  Run:  python3 docs/designs/pta_laser.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_geometry as geometry
import pta_power as power
import pta_rate as rate
import pta_shot_rate as shot
import pta_source as source

FS = shot.X2_FS                            # 1 GS/s: B11
V1 = shot.REQ["v1"]
BITS = V1["adc_bits"]
DIN = 8                                    # the operands' width, as grx930 trains D3
WORKING, SMALLER = shot.X2_TILE, (128, 64)
LOSS_DB = shot.LOSS_DB
WITHIN = 0.10                              # a tenth of a point: what a row of the budget costs

# ---- grx930's figures ----------------------------------------------------------
# The 8-bit converter's shift, a layer of D3 each, on every tile grx930 has run.
# At 64 x 8 the rule gave three networks (10, 8) and one each of the others.
S8 = {
    (8, 8): (9, 7), (64, 8): (10, 8), (128, 64): (11, 9), (256, 64): (11, 9), (256, 128): (11, 9),
}
S8_ALSO = {(64, 8): ((11, 8), (11, 9))}
# Points lost by the laser's size, as a multiple of B5's; 0 is version 1 as
# budgeted, half an LSB a layer at that layer's own shift.
LOST = {
    (256, 64): {0: (0.20, 0.05), 1: (39.19, 1.87), 2: (12.53, 1.14), 4: (2.44, 0.24), 8: (0.60, 0.07),
                16: (0.28, 0.06), 32: (0.23, 0.06), 64: (0.19, 0.06)},
    (128, 64): {0: (0.34, 0.07), 1: (13.56, 1.14), 2: (2.74, 0.19), 4: (0.75, 0.07), 8: (0.35, 0.06),
                16: (0.22, 0.04), 32: (0.20, 0.06), 64: (0.19, 0.05)},
}
# The light a shot sends a column, in lines: (mean, most), a layer each.
LIT = {
    (256, 64): ((25.82, 133.48), (6.68, 12.10)),
    (128, 64): ((14.75, 82.96), (6.68, 11.99)),
}
# How the network is put on the tile, on 128 x 64: (bits added to the hidden
# rescale, bits of gain on the first layer's weights) -> points lost by the
# laser's multiple; and what each clips, in percent of the hidden units that
# fire and of the first layer's weights, with what that costs by itself.
PUT = {
    (0, 0):  {2: (2.74, 0.19), 4: (0.75, 0.07), 8: (0.35, 0.06)},
    (0, 1):  {2: (2.90, 0.21), 4: (0.83, 0.12), 8: (0.50, 0.16)},
    (0, 2):  {2: (5.65, 0.31), 4: (2.62, 0.34), 8: (2.05, 0.33)},
    (-1, 0): {2: (0.92, 0.06), 4: (0.35, 0.06), 8: (0.22, 0.06)},
    (-1, 1): {2: (0.83, 0.12), 4: (0.41, 0.13), 8: (0.34, 0.10)},
    (-1, 2): {2: (2.64, 0.32), 4: (2.04, 0.29), 8: (1.89, 0.30)},
    (-2, 0): {2: (0.99, 0.11), 4: (0.60, 0.08), 8: (0.46, 0.06)},
    (-2, 1): {2: (0.79, 0.12), 4: (0.63, 0.12), 8: (0.52, 0.13)},
    (-2, 2): {2: (2.67, 0.46), 4: (2.50, 0.45), 8: (2.41, 0.47)},
}
PUT_CLIPS = {
    (0, 0): (0.00, 0.00, (0.00, 0.01)), (0, 1): (0.00, 0.38, (0.14, 0.08)), (0, 2): (0.00, 7.91, (1.46, 0.33)),
    (-1, 0): (0.65, 0.00, (0.02, 0.01)), (-1, 1): (0.55, 0.38, (0.14, 0.08)), (-1, 2): (0.30, 7.91, (1.49, 0.35)),
    (-2, 0): (13.45, 0.00, (0.26, 0.05)), (-2, 1): (12.24, 0.38, (0.33, 0.08)), (-2, 2): (9.57, 7.91, (1.94, 0.42)),
}
PUT_S8 = {(0, 0): (11, 9), (-1, 0): (11, 10), (-2, 0): (11, 10)}
AS_SET, BIT_DOWN = (0, 0), (-1, 0)


# ---- the light a column is sent ----------------------------------------------------
def line_light(din=DIN):
    """One line's light at a detector, in a sum's units: grx930's."""
    return (2 ** (din - 1) - 1) * 2 ** (din - 1)


def fill(tile, layer):
    """A converter's full scale over the light the column can be sent."""
    return 2 ** shot.NOISE_LSB_BITS * 2 ** S8[tile][layer] / (tile[0] * line_light())


def b5_fraction(tile, req=V1):
    """A laser of B5's size puts the receiver's noise at this share of a line's light."""
    return tile[0] * req["rx_noise_lsb"] / 2 ** shot.NOISE_LSB_BITS


def noise_lsb(tile, layer, times=1.0, req=V1):
    """And so at this many LSB of the 8-bit converter the budget means, on a layer."""
    return b5_fraction(tile, req) / times * line_light() / 2 ** S8[tile][layer]


def times_for_budget(tile, layer, req=V1):
    """The multiple of B5's laser at which a layer's noise is the budget's row."""
    return noise_lsb(tile, layer, 1.0, req) / req["rx_noise_lsb"]


# ---- what the accuracy asks ------------------------------------------------------------
def over(tile, times):
    """Points lost beyond that tile's own version 1 as budgeted."""
    return LOST[tile][times][0] - LOST[tile][0][0]


def times_within(tile, budget=WITHIN):
    """The least multiple measured that is within the budget of version 1, every
    larger one being so too."""
    ok = None
    for t in sorted((t for t in LOST[tile] if t), reverse=True):
        if over(tile, t) >= budget - 1e-9:
            break
        ok = t
    return ok


def times_put(put, budget=WITHIN):
    """The least multiple at which the smaller tile, put on it this way, is within
    the budget of its own version 1; None if none measured is."""
    ok = None
    for t in sorted(PUT[put], reverse=True):
        if PUT[put][t][0] - LOST[SMALLER][0][0] >= budget - 1e-9:
            break
        ok = t
    return ok


# ---- the laser, in watts ------------------------------------------------------------------
def b5_w(tile):
    """B5's method at the measured receiver, behind 10 and 20 dB: pta_geometry.py's."""
    return geometry.laser_w(*tile)


def laser_w(tile, times):
    lo, hi = b5_w(tile)
    return lo * times, hi * times


def averaging_gain():
    """What a receiver that averages over a shot saves of the laser, at one noise
    density: the root of its noise bandwidth over the single pole's."""
    pole = math.pi / 2 * power.settle_bandwidth_hz(FS, BITS)
    return math.sqrt(FS / 2 / pole)


def total_w(times):
    """The working tile's interface chip and laser, pta_rate.py's with the laser scaled."""
    c, l = rate.chip_w(FS), laser_w(WORKING, times)
    return c[0] + l[0], c[1] + l[1]


def fj_mac(tile, times):
    """Energy a MAC, every cell in use: pta_geometry.py's with the laser scaled."""
    c, l = geometry.chip_w(*tile), laser_w(tile, times)
    return tuple((c[i] + l[i]) / power.mac_s(*tile) * 1e15 for i in (0, 1))


# ---- the pair's own noise ---------------------------------------------------------------------
def photons_a_line(tile, times):
    """Photons one line sends a column in a shot, at the measured receiver."""
    noise_a = power.receiver_noise_a(FS, BITS)
    watts = noise_a / shot.RESPONSIVITY_A_W / (b5_fraction(tile) / times)
    return watts / FS / (shot.H_PLANCK * shot.C_LIGHT / shot.WAVELENGTH_M)


def pair_shot_lsb(tile, layer, times, lines):
    """The shot noise of all the light a pair carries, `lines` of it, in LSB of the
    8-bit converter at that layer's shift."""
    n = photons_a_line(tile, times)
    lsb_photons = n * 2 ** S8[tile][layer] / line_light()
    return math.sqrt(lines * n) / lsb_photons


def row_allows_at(shot_lsb, req=V1):
    """The sum, in LSB, at which the budget's photon row already allows that much
    shot noise: the row's is the root of a sum over its photons an LSB."""
    return shot_lsb * shot_lsb * req["photons_per_lsb"]


def name(t):
    return f"{t[0]}x{t[1]}"


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    print("B5's laser, against the light a column is actually sent.")
    print(f"A line's light is {line_light():,} of a sum's units: an input at full scale through a weight of one.")

    section("1. What the converter's full scale is of the light a column can be sent")
    print(f"  {'tile':<10}{'shifts':>9}{'fill, layer 1':>15}{'layer 2':>10}{'B5: noise, layer 1':>21}{'layer 2':>10}"
          f"{'the laser the row asks':>25}")
    for t in sorted(S8):
        f1, f2 = fill(t, 0), fill(t, 1)
        print(f"  {name(t):<10}{f'{S8[t][0]}, {S8[t][1]}':>9}{f1:>15.3f}{f2:>10.3f}{noise_lsb(t, 0):>17.2f} LSB"
              f"{noise_lsb(t, 1):>6.2f} LSB{f'{times_for_budget(t, 0):.0f} and {times_for_budget(t, 1):.0f} times':>25}")
    print("  The fill is 256 LSB of the 8-bit converter, at the shift the harness's rule gives that")
    print("  layer, over the tile's rows at a line each.  B5's method takes it for one.  The noise")
    print("  is what a laser of B5's size leaves the receiver, where the budget's row is half an LSB.")
    print(f"  At {name((64, 8))} the rule gave two of five networks other shifts: {S8_ALSO[(64, 8)]}.")

    section("2. What grx930 measured: points lost by the laser's size")
    tiles = sorted(LOST, reverse=True)
    print(f"  {'':<24}" + "".join(f"{name(t):>16}" for t in tiles) + "".join(f"{'noise, ' + name(t):>22}" for t in tiles))
    for times in sorted(LOST[WORKING]):
        label = "version 1 as budgeted" if times == 0 else f"B5's laser, times {times}"
        row = "".join(f"{LOST[t][times][0]:>10.2f} +-{LOST[t][times][1]:.2f}" for t in tiles)
        noise = "".join(f"{'0.50, 0.50' if times == 0 else f'{noise_lsb(t, 0, times):.2f}, {noise_lsb(t, 1, times):.2f}':>22}"
                        for t in tiles)
        print(f"  {label:<24}{row}{noise}")
    print("  Noise is the receiver's on each layer, in LSB of the 8-bit converter at its shift.")

    section("3. The laser, in watts")
    print(f"  B5's method, at the measured receiver behind 10 and 20 dB (pta_power.py):")
    for t in tiles:
        need = times_within(t)
        lo, hi = b5_w(t)
        print(f"    {name(t):<8} {lo:.2f} to {hi:.2f} W.  Within a tenth of a point of its own version 1 at {need} times that:"
              f" {laser_w(t, need)[0]:.1f} to {laser_w(t, need)[1]:.1f} W.")
    g = averaging_gain()
    need = times_within(WORKING)
    print(f"  ASSUMED the same noise density, a receiver that averages over a shot has {g:.2f} of a single")
    print(f"  pole's noise, and asks {g:.2f} of the laser: {laser_w(WORKING, need)[0] * g:.1f} to {laser_w(WORKING, need)[1] * g:.1f} W at {name(WORKING)}.")
    print(f"  With the interface chip, {name(WORKING)} draws {total_w(need)[0]:.1f} to {total_w(need)[1]:.1f} W where the scorecards have"
          f" {total_w(1)[0]:.2f} to {total_w(1)[1]:.1f},")
    a, b = fj_mac(WORKING, need), fj_mac(SMALLER, times_within(SMALLER))
    a1, b1 = fj_mac(WORKING, 1), fj_mac(SMALLER, 1)
    print(f"  and a MAC costs {a[0]:.0f} to {a[1]:,.0f} fJ where they have {a1[0]:.0f} to {a1[1]:.0f}.  At {name(SMALLER)}:"
          f" {b[0]:.0f} to {b[1]:,.0f}, for {b1[0]:.0f} to {b1[1]:.0f}.")

    section("4. The light a pair carries, and its shot noise")
    for t in tiles:
        need = times_within(t)
        (m1, x1), (m2, x2) = LIT[t]
        print(f"  {name(t)}: a shot sends a column {m1:.1f} lines on layer 1 at the mean and {x1:.0f} at the most, of"
              f" {t[0]}; {m2:.1f} and {x2:.0f} on layer 2.")
        print(f"    At {need} times B5's laser a line is {photons_a_line(t, need):,.0f} photons a shot, and the shot noise"
              f" of that light is")
        print(f"    {pair_shot_lsb(t, 0, need, m1):.2f} LSB on layer 1 and {pair_shot_lsb(t, 1, need, m2):.2f} on layer 2, where the"
              f" receiver's is {noise_lsb(t, 0, need):.2f} and {noise_lsb(t, 1, need):.2f}.")
    need = times_within(WORKING)
    s1, s2 = (pair_shot_lsb(WORKING, l, need, LIT[WORKING][l][0]) for l in (0, 1))
    print("  That is all the light on both photodiodes, whatever the weights: B13's first open item.")
    print(f"  grx930's model has the shot noise of the difference, at {V1['photons_per_lsb']} photons an LSB, which allows")
    print(f"  that much on a sum of {row_allows_at(s1):.1f} LSB on layer 1 and {row_allows_at(s2):.1f} on layer 2, and more on a larger.")
    print("  So it is inside the budget's photon row, and it is not three orders away: it falls")
    print("  as the root of the laser where the receiver's falls as the laser.")

    section("5. How the network is put on the tile, and what that gives back")
    print(f"  {name(SMALLER)}, where version 1 as budgeted loses {LOST[SMALLER][0][0]:.2f} +-{LOST[SMALLER][0][1]:.2f}.  Points lost by the laser's multiple:")
    print(f"  {'hidden rescale':<16}{'weight gain':>12}{'units clip':>12}{'weights clip':>14}{'that alone':>14}"
          f"{'x 2':>14}{'x 4':>14}{'x 8':>14}{'within a tenth at':>19}")
    for put in sorted(PUT, key=lambda k: (-k[0], k[1])):
        hc, wc, alone = PUT_CLIPS[put]
        t = times_put(put)
        bits = ("none", "one bit", "two bits")
        print(f"  {('the rule' + chr(39) + 's') if put[0] == 0 else bits[-put[0]] + ' less':<16}{bits[put[1]]:>12}{hc:>11.2f}%{wc:>13.2f}%"
              f"{f'{alone[0]:.2f} +-{alone[1]:.2f}':>14}"
              + "".join(f"{f'{PUT[put][m][0]:.2f} +-{PUT[put][m][1]:.2f}':>14}" for m in (2, 4, 8))
              + f"{('none measured' if t is None else f'{t} times'):>19}")
    need, down = times_put(AS_SET), times_put(BIT_DOWN)
    lo, hi = laser_w(SMALLER, down)
    print(f"  The first row is section 2's.  With the hidden rescale one bit under the clip rule the")
    print(f"  second layer's shift is {PUT_S8[BIT_DOWN][1]} and not {PUT_S8[AS_SET][1]}, its fill {2 ** (PUT_S8[BIT_DOWN][1] - PUT_S8[AS_SET][1]):.0f} times what it was, and the laser")
    print(f"  {down} times B5's and not {need}: {lo:.2f} to {hi:.1f} W.  On the chiplet that rescale is the activation")
    print("  stage's shift, a field of the command (the register map's section 8).")

    findings()
    checks()


def findings():
    need, need_s = times_within(WORKING), times_within(SMALLER)
    lo, hi = laser_w(WORKING, need)
    print()
    print("What this says, seven readings.")
    print()
    print("  1. B5'S LASER IS SIZED ON LIGHT THE DETECTORS DO NOT GET.  Its method takes a converter's")
    print(f"     full scale for all the light a column is sent.  On D3 at {name(WORKING)} it is {fill(WORKING, 0):.3f} of it on")
    print(f"     the first layer and {fill(WORKING, 1):.3f} on the second: the rest the inputs did not pass, or the")
    print("     weights sent to both photodiodes of a pair alike.  So a")
    print(f"     laser of B5's size leaves the receiver {noise_lsb(WORKING, 0):.1f} and {noise_lsb(WORKING, 1):.0f} LSB of noise, where the budget's")
    print("     row is half of one.")
    print()
    print(f"  2. AT B5'S LASER THE WORKING TILE LOSES {LOST[WORKING][1][0]:.0f} POINTS.  It is within a tenth of a point of")
    print(f"     version 1 at {need} times that laser, and matches it at {2 * need}.  That is {lo:.1f} to {hi:.0f} W at the")
    print(f"     measured receiver, where every scorecard has {b5_w(WORKING)[0]:.2f} to {b5_w(WORKING)[1]:.2f}.")
    print()
    print(f"  3. THE METHOD WAS RIGHT WHERE IT WAS MADE.  On the core's 8 x 8 tile the first layer's")
    print(f"     fill is {fill((8, 8), 0):.2f}: eight lines are a full scale.  Sums do not grow as the rows do, the")
    print("     light does, and a tile of 256 rows was never run against its laser.")
    print()
    print(f"  4. THE SECOND LAYER ASKS FOUR TIMES THE FIRST'S, ON EVERY TILE.  Its sums are a quarter")
    print("     the size, so its LSB is a quarter the light, and the one laser has to serve it.")
    print("     The budget gave each layer half an LSB of its own, as if each had its own laser.")
    print()
    print(f"  5. THE ROWS COST LIGHT, AND {name(SMALLER)} NEEDS HALF OF IT.  It is within a tenth of a point")
    print(f"     of its version 1 at {need_s} times B5's laser, {laser_w(SMALLER, need_s)[0]:.1f} to {laser_w(SMALLER, need_s)[1]:.0f} W, and at any multiple it")
    print(f"     loses what {name(WORKING)} loses at twice that.  With the laser at what each needs a MAC")
    a, b = fj_mac(WORKING, need), fj_mac(SMALLER, need_s)
    print(f"     costs {a[0]:.0f} to {a[1]:,.0f} fJ on the one and {b[0]:.0f} to {b[1]:,.0f} on the other: B10's energy case for")
    print("     the larger tile is gone.")
    print()
    g = averaging_gain()
    print("  6. WHAT MOVES IT.  The receiver's noise, in proportion: every figure is linear in it.")
    print(f"     A receiver that averages over a shot, {g:.2f} of it if its density is the same.  The")
    print(f"     loss, in proportion.  The rows.  And the workload: on MNIST a shot lights {LIT[WORKING][0][0] / WORKING[0]:.0%} of the")
    print("     tile's rows at the mean, and one that lights more fills more.  What does not")
    print("     move it is the receiver's gain, which scales the noise with the signal.")
    print()
    down, need = times_put(BIT_DOWN), times_put(AS_SET)
    hc, _, alone = PUT_CLIPS[BIT_DOWN]
    lo, hi = laser_w(SMALLER, down)
    print(f"  7. AND HALF OF IT IS THE HOST'S TO GIVE BACK.  The hidden layer's rescale, one bit under")
    print(f"     the rule that lets one unit in ten thousand reach full scale, clips {hc}% of the")
    print(f"     units that fire and costs {alone[0]:.2f} of a point.  The second layer's operands are twice")
    print(f"     as large, and {name(SMALLER)} is within a tenth of a point of version 1 at {down} times B5's")
    print(f"     laser where it took {need}: {lo:.2f} to {hi:.1f} W.  A second bit buys nothing, and a gain on")
    print("     the first layer's weights does not pay.  The rule was a converter's: it wasted")
    print("     none of an operand's range, and under a laser the range is not what is short.")


def checks():
    """Every claim above, as an assert."""
    assert line_light() == 16_256 and WORKING == (256, 64) and FS == 1e9

    # 1. Reading 1.  The fills, and B5's noise: 3.97 and 15.9 LSB at the working tile.
    assert abs(fill(WORKING, 0) - 0.126) < 0.0005 and abs(fill(WORKING, 1) - 0.0315) < 0.0005
    assert abs(b5_fraction(WORKING) - 0.5) < 1e-12 and abs(b5_fraction(SMALLER) - 0.25) < 1e-12
    assert abs(noise_lsb(WORKING, 0) - 3.97) < 0.005 and abs(noise_lsb(WORKING, 1) - 15.875) < 0.005
    assert all(abs(noise_lsb(t, l) * fill(t, l) - 0.5) < 1e-12 for t in S8 for l in (0, 1))
    assert [round(times_for_budget(WORKING, l)) for l in (0, 1)] == [8, 32]
    assert [round(times_for_budget(SMALLER, l)) for l in (0, 1)] == [4, 16]
    #    The laser's size reproduces B5's own figure: 0.5 LSB of a detector whose
    #    full scale is all its rows' light.
    assert abs(b5_fraction(WORKING) * line_light() / (WORKING[0] * line_light() / 256) - 0.5) < 1e-12

    # 2. Reading 2.  39 points at B5's laser; within a tenth at 16 times, matched at 32.
    assert round(LOST[WORKING][1][0]) == 39 and times_within(WORKING) == 16
    assert abs(over(WORKING, 16) - 0.08) < 0.005 and abs(over(WORKING, 32)) < LOST[WORKING][32][1]
    assert over(WORKING, 8) > WITHIN
    lo, hi = laser_w(WORKING, 16)
    assert abs(b5_w(WORKING)[0] - 0.088) < 0.001 and abs(b5_w(WORKING)[1] - 0.878) < 0.001
    assert abs(lo - 1.40) < 0.01 and abs(hi - 14.05) < 0.05
    #    grx930's "as budgeted" is its laser sweep at 8 times on layer 1 and 32 on layer 2.
    assert abs(noise_lsb(WORKING, 0, 8) - 0.5) < 0.005 and abs(noise_lsb(WORKING, 1, 32) - 0.5) < 0.005

    # 3. Reading 3.  The 8 x 8 tile's first layer fills the light, and B5's laser
    #    leaves it the budget's half an LSB.
    assert abs(fill((8, 8), 0) - 1.0) < 0.01 and abs(noise_lsb((8, 8), 0) - 0.5) < 0.005
    assert all(fill(t, 0) < fill((8, 8), 0) / 3.9 for t in S8 if t[0] >= 128)

    # 4. Reading 4.  Two bits between the layers' shifts on every tile.
    assert all(S8[t][0] - S8[t][1] == 2 for t in S8)
    assert all(abs(noise_lsb(t, 1) / noise_lsb(t, 0) - 4.0) < 1e-12 for t in S8)

    # 5. Reading 5.  The smaller tile at 8 times, and at any multiple what the
    #    larger loses at twice it, within their errors.
    assert times_within(SMALLER) == 8 and abs(b5_w(SMALLER)[0] - b5_w(WORKING)[0]) < 1e-12
    for t in (1, 2, 4, 8, 16, 32):
        a, b = LOST[SMALLER][t], LOST[WORKING][2 * t]
        assert abs(a[0] - b[0]) < 2.5 * math.hypot(a[1], b[1]), (t, a, b)
    assert abs(noise_lsb(SMALLER, 0, 4) - noise_lsb(WORKING, 0, 8)) < 1e-12
    big, small = fj_mac(WORKING, 16), fj_mac(SMALLER, 8)
    assert abs(big[0] / small[0] - 1) < 0.2 and abs(big[1] / small[1] - 1) < 0.05
    assert fj_mac(WORKING, 1)[0] < 0.65 * fj_mac(SMALLER, 1)[0]

    # 6. Reading 6.  A receiver that averages has 0.60 of a single pole's noise.
    assert abs(averaging_gain() - 0.60) < 0.005
    assert abs(LIT[WORKING][0][0] / WORKING[0] - 0.10) < 0.005

    # 7. Section 4.  The pair's shot noise at the laser the receiver needs: 0.16
    #    and 0.32 LSB, under the receiver's 0.25 and 0.99, and what the photon row
    #    allows a sum of 0.4 and 1.5 LSB.  It falls as the root of the laser.
    s = [pair_shot_lsb(WORKING, l, 16, LIT[WORKING][l][0]) for l in (0, 1)]
    assert abs(s[0] - 0.16) < 0.005 and abs(s[1] - 0.32) < 0.005
    assert all(s[l] < noise_lsb(WORKING, l, 16) for l in (0, 1))
    assert abs(row_allows_at(s[0]) - 0.4) < 0.05 and abs(row_allows_at(s[1]) - 1.5) < 0.05
    assert abs(pair_shot_lsb(WORKING, 0, 64, 25.82) / s[0] - 0.5) < 1e-9
    assert abs(noise_lsb(WORKING, 0, 64) / noise_lsb(WORKING, 0, 16) - 0.25) < 1e-12
    assert round(photons_a_line(WORKING, 16), -3) == 67_000
    #    The totals: 2.0 to 21.7 W, and 119 to 1,323 fJ a MAC, where the scorecards
    #    have 0.64 to 8.5 and 39 to 518.
    t16, t1 = total_w(16), total_w(1)
    assert abs(t16[0] - 1.96) < 0.01 and abs(t16[1] - 21.66) < 0.05
    assert abs(t1[0] - 0.64) < 0.005 and abs(t1[1] - 8.49) < 0.01
    assert [round(x) for x in fj_mac(WORKING, 16)] == [119, 1323]
    assert [round(x) for x in fj_mac(WORKING, 1)] == [39, 518]
    assert [round(x) for x in fj_mac(SMALLER, 8)] == [140, 1351]
    assert abs(laser_w(WORKING, 16)[0] * averaging_gain() - 0.84) < 0.01
    assert abs(laser_w(WORKING, 16)[1] * averaging_gain() - 8.4) < 0.05

    # 9. Section 5 and reading 7.  The first row is section 2's; a bit less of
    #    rescale is half the laser at 4 and 8 times; a second bit and the weight
    #    gain do not pay.
    assert all(PUT[AS_SET][m] == LOST[SMALLER][m] for m in (2, 4, 8))
    assert (times_put(AS_SET), times_put(BIT_DOWN)) == (8, 4)
    assert PUT[BIT_DOWN][4] == LOST[SMALLER][8] and PUT[BIT_DOWN][8][0] == LOST[SMALLER][16][0]
    assert PUT[BIT_DOWN][2][0] > LOST[SMALLER][4][0]            # the first layer's share is not given back
    assert PUT_CLIPS[BIT_DOWN][0] == 0.65 and PUT_CLIPS[BIT_DOWN][2][0] == 0.02
    assert abs(100 / PUT_CLIPS[BIT_DOWN][0] - 154) < 1
    assert PUT_S8[BIT_DOWN][1] - PUT_S8[AS_SET][1] == 1 and PUT_S8[(-2, 0)] == PUT_S8[BIT_DOWN]
    assert all(PUT[(-2, 0)][m][0] > PUT[BIT_DOWN][m][0] for m in (2, 4, 8))
    assert PUT_CLIPS[(-2, 0)][0] > 13 and abs(PUT_CLIPS[(-2, 0)][2][0] - 0.26) < 0.005
    assert all(PUT[(0, 1)][m][0] > PUT[AS_SET][m][0] for m in (2, 4, 8))
    assert all(PUT[(-1, 1)][m][0] > PUT[BIT_DOWN][m][0] for m in (4, 8))
    assert min(PUT, key=lambda k: PUT[k][4][0]) == BIT_DOWN and min(PUT, key=lambda k: PUT[k][8][0]) == BIT_DOWN
    assert [times_put(k) for k in ((0, 1), (-1, 1), (-2, 0))] == [None, 4, None]
    lo, hi = laser_w(SMALLER, times_put(BIT_DOWN))
    assert abs(lo - 0.35) < 0.005 and abs(hi - 3.51) < 0.01
    assert abs(b5_fraction(SMALLER) / times_put(BIT_DOWN) - 1 / 16) < 1e-12
    half = fj_mac(SMALLER, times_put(BIT_DOWN))
    assert [round(x) for x in half] == [97, 922], half

    # 8. The plan's other figures.  The row as one laser can meet it is a
    #    thirty-second of a line's light, on both tiles; a sixteenth costs 0.40
    #    on the working tile and an eighth 2.2; and a line is 22 to 220 mW.
    assert abs(b5_fraction(WORKING) / times_within(WORKING) - 1 / 32) < 1e-12
    assert abs(b5_fraction(SMALLER) / times_within(SMALLER) - 1 / 32) < 1e-12
    assert abs(over(WORKING, 8) - 0.40) < 0.005 and abs(over(WORKING, 4) - 2.24) < 0.005
    assert abs(b5_fraction(WORKING) / 8 - 1 / 16) < 1e-12 and abs(b5_fraction(WORKING) / 4 - 1 / 8) < 1e-12
    line = [x * 16 * 1e3 for x in source.line_w(4)]
    assert abs(line[0] - 21.9) < 0.1 and abs(line[1] - 219.5) < 0.5
    lo_s, hi_s = laser_w(SMALLER, 8)
    assert abs(lo_s - 0.70) < 0.005 and abs(hi_s - 7.02) < 0.01
    assert abs(fill(SMALLER, 0) - 0.25) < 0.005 and abs(fill(WORKING, 0) - 1 / 8) < 0.002
    assert abs(fill(WORKING, 1) - 1 / 32) < 0.0005

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
