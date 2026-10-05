"""
Open question 1 of board_program_plan.md, as one table: how big is the chiplet?

Six models price the chiplet's tile, each for its own reason, and each sweeps
the same candidate geometries because none of them can fix one:

  pta_chiplet_link.py   what crosses the link, and how many modules carry it
  pta_shot_rate.py      what caps the shot rate, and what the weight path needs
  pta_floorplan.py      what has to be on the two dies, and how much room it takes
  pta_power.py          what the interface chip and the laser draw
  pta_dispatch.py       how long a layer takes, command to result
  grx930's harness      what the budget costs in accuracy, tile by tile
                        (c930/doc/pta_error_model_design_note.md section 5,
                        `pta_mnist.sh geometry`, 2026-10-04)

Nothing here is new physics.  Every figure is one of those models' own, asked
for at each candidate and put beside the others, so that the choice the plan
keeps deferring can be looked at whole.  The checks hold each column to the
model it came from.

WHAT KIND OF NUMBER EACH COLUMN IS, because they differ:

  cells, lines      COUNTED.  A weight is a cell, and every cell, input and
                    output is a line between the two dies (pta_floorplan.py).
  area              A LOWER BOUND, and conditional: resonant weight cells at a
                    25 um pitch, which is an assumption in a patent application
                    for another topology, with a 5 V Mach-Zehnder a row.  Not a
                    layout.  A non-resonant tile of any candidate past 64 x 8
                    does not fit a die at all (pta_floorplan.py section 2).
                    AND TOO SMALL: pta_ring.py has since read published rings,
                    and at the smallest of them a 256 x 64 tile is 124 mm2
                    where this column has 39.7.  The column is kept as it was
                    run, and "fits a standard die" in section 1 is this bound's
                    answer and not a published ring's.
  link modules      PREDICTED by X2 for a 4096-square layer at a batch of 64
                    and 1 GS/s, weights re-sent with every batch.
  watts             FROM PUBLISHED PARTS, as a range: the converters from the
                    ADC survey, the receivers from one measured amplifier, the
                    drive from assumed capacitances and swings (pta_power.py).
                    The laser is B5's method at that amplifier's noise, behind
                    10 and 20 dB of loss.  AND LOW: pta_laser.py has since
                    sized it on the light a column is sent, 8 times this at
                    128 rows and 16 at 256.  The column is kept as it was run.
  time, energy      PREDICTED.  pta_dispatch.py's time for a layer, at that
                    geometry's own module count, times the watts.
  accuracy          MEASURED IN A MODEL, on one network: D3, five trainings.

There is no chiplet, and none of this is a measurement of one.

WHAT IT DOES NOT DO.  It does not weigh the columns against each other.  How
much a watt is worth against a tenth of a point is the program's to say, and a
score with invented weights would hide the choice inside a number.

THE CHOICE WAS MADE ON 2026-10-05: 256 x 64 is the working geometry, the plan's
B10.  This file is what it was made from, and is kept as it was run.

Standard library only.  Run:  python3 docs/designs/pta_geometry.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_chiplet_link as link
import pta_dispatch as dispatch
import pta_floorplan as fp
import pta_power as power
import pta_shot_rate as shot

V1 = shot.REQ["v1"]
BITS = V1["adc_bits"]
FS = shot.X2_FS
BATCH = link.BIG["mb"]
STD_DIE_MM2 = fp.STD_DIE_MM[0] * fp.STD_DIE_MM[1]
LOSS_DB = shot.LOSS_DB                     # B5's range, laser to detector

# The candidates, as every model sweeps them.  The first is the c930 core's
# tile: not a candidate for the chiplet, and the tile every accuracy figure
# before 2026-10-04 was measured on.
TILES = link.GEOMETRIES
CORE = (8, 8)
CANDIDATES = tuple(t for t in TILES if t != CORE)

# ---- accuracy: grx930's design note, section 5, "another tile" ---------------
# Loss in points on D3 against the same weights on the host, mean and standard
# error over five networks; and the error on layer 1's sums, percent of their
# rms.  S is layer 1's ADC shift at the 8-bit ADC and T the conversions a sum.
# At 64 x 8 the clip rule put three networks at S = 10 and two at 11.
ACC = {
    (8, 8):     dict(v1=(0.26, 0.06), v0=(1.49, 0.06), hour=(0.81, 0.21), four=(5.18, 2.31),
                     tfln=(22.62, 4.79), cal=(0.22, 0.05), e_thermal=10.55, e_v1=8.00,
                     e_hour=21.17, shift=(9,)),
    (64, 8):    dict(v1=(0.33, 0.11), v0=(2.05, 0.40), hour=(0.56, 0.09), four=(1.17, 0.07),
                     tfln=(4.24, 0.28), cal=(0.32, 0.08), e_thermal=10.64, e_v1=10.04,
                     e_hour=14.43, shift=(10, 11)),
    (128, 64):  dict(v1=(0.34, 0.07), v0=(2.17, 0.12), hour=(0.51, 0.07), four=(0.74, 0.07),
                     tfln=(2.81, 0.22), cal=(0.21, 0.05), e_thermal=11.23, e_v1=10.72,
                     e_hour=13.12, shift=(11,)),
    (256, 64):  dict(v1=(0.20, 0.05), v0=(1.86, 0.17), hour=(0.24, 0.06), four=(0.77, 0.16),
                     tfln=(1.92, 0.25), cal=(0.20, 0.05), e_thermal=8.50, e_v1=9.14,
                     e_hour=11.44, shift=(11,)),
    (256, 128): dict(v1=(0.23, 0.05), v0=(1.78, 0.13), hour=(0.43, 0.15), four=(1.18, 0.30),
                     tfln=(3.53, 0.99), cal=(0.25, 0.07), e_thermal=8.50, e_v1=9.14,
                     e_hour=11.61, shift=(11,)),
}
D3_IN = dispatch.D3[0][0]                  # 784: layer 1's inputs


def modules(k, n):
    """X2's count for the 4096-square layer at batch 64, weights re-sent."""
    return fp.link_modules(k, n, BATCH)


def modules_resident(k, n):
    """The same with the weights already on the interface chip."""
    return math.ceil(FS / shot.feed_rate(mods=1, resident=True, k=k, n=n, **link.BIG))


def area_mm2(k, n):
    """(weights, inputs, link): the floorplan's lower bounds, at its assumed cell."""
    weights, inputs = fp.die_mm2(k, n, fp.P_CELL, 5.0)
    return weights, inputs, modules(k, n) * fp.module_mm2()


def chip_w(k, n):
    """(low, high) watts on the interface chip: 2 V with the weights resident,
    and 5 V with them re-sent, which are pta_power.py's two ends."""
    (lo, _), _ = power.chip_w(k, n, BITS, 2.0, True)
    (_, hi), _ = power.chip_w(k, n, BITS, 5.0, False)
    return lo, hi


def laser_w(k, n):
    """(low, high) watts of laser at the measured receiver, at B5's two losses."""
    na = power.receiver_noise_a(FS, BITS)
    return tuple(power.laser_w(V1, db, na, n) for db in LOSS_DB)


def total_w(k, n):
    c, l = chip_w(k, n), laser_w(k, n)
    return c[0] + l[0], c[1] + l[1]


def fj_mac(k, n):
    """Energy a MAC with every cell in use: the watts over k * n * the shot rate."""
    lo, hi = total_w(k, n)
    return lo / power.mac_s(k, n) * 1e15, hi / power.mac_s(k, n) * 1e15


def layer(k, n, m, kin, nout):
    """pta_dispatch.py's layer at this tile, on this tile's own module count
    and the write path section 4.3 holds a two-bank tile to."""
    return dispatch.pta(m, kin, nout, tile=(k, n), mods=modules(k, n),
                        per_beat=shot.per_beat_two_banks(k, n, BATCH))


def d3(k, n):
    """D3 at a batch of 64: seconds, MACs, and the share of programmed cells used."""
    runs = [layer(k, n, BATCH, a, b) for a, b in dispatch.D3]
    used = sum(a * b for a, b in dispatch.D3)
    programmed = sum(r["sets"] * k * n for r in runs)
    return sum(r["seconds"] for r in runs), BATCH * used, used / programmed


def fj_useful(tile, width):
    """(low, high) fJ a MAC the layer asked for, on a square layer `width` wide
    at a batch of 64: the watts, times the layer's time, over its own MACs.
    Cells the layer programs and does not use are paid for and not counted."""
    r = layer(*tile, BATCH, width, width)
    lo, hi = total_w(*tile)
    macs = BATCH * width * width
    return lo * r["seconds"] / macs * 1e15, hi * r["seconds"] / macs * 1e15


WIDTHS = (64, 128, 256, 512, 1024, 4096)


def noise_factor(k, shift):
    """sqrt(conversions) * 2^shift on D3's first layer, against the core tile's."""
    conv = -(-D3_IN // k)
    core = math.sqrt(-(-D3_IN // CORE[0])) * 2 ** ACC[CORE]["shift"][0]
    return math.sqrt(conv) * 2 ** shift / core


def drift_error(tile):
    """An hour of drift's own error on layer 1, v1's taken out in quadrature."""
    a = ACC[tile]
    return math.sqrt(a["e_hour"] ** 2 - a["e_v1"] ** 2)


def rng(lo, hi, unit="", scale=1.0, digits=2):
    return f"{lo * scale:.{digits}f}-{hi * scale:.{digits}f}{unit}"


def pm(pair):
    return f"{pair[0]:.2f} +-{pair[1]:.2f}"


def name(t):
    return f"{t[0]}x{t[1]}"


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    section("1. What has to be built")
    print(f"  {'tile':<9}{'cells':>8}{'lines':>8}{'weights':>10}{'inputs':>9}{'link':>7}{'sum':>8}"
          f"{'a standard die':>16}{'modules':>9}{'resident':>10}")
    for t in TILES:
        w, i, l = area_mm2(*t)
        print(f"  {name(t):<9}{t[0] * t[1]:>8,}{fp.connections(*t):>8,}{w:>9.1f} {i:>8.1f} {l:>6.1f} "
              f"{w + i + l:>7.1f} {'fits' if w + i + l <= STD_DIE_MM2 else 'does not':>15}"
              f"{modules(*t):>9}{modules_resident(*t):>10}")
    print("  Areas in mm2, and lower bounds: resonant cells at the floorplan's assumed 25 um,")
    print(f"  a 5 V modulator a row, the link's PHY.  A standard die is {STD_DIE_MM2:.0f} mm2.  Lines are")
    print("  between the two dies, one a cell, an input and an output.  Modules are X2's for a")
    print("  4096-square layer at batch 64 and 1 GS/s; 'resident' is with the weights already there.")

    section("2. What it draws")
    print(f"  {'tile':<9}{'interface chip':>18}{'laser':>16}{'together':>16}{'a MAC, all cells used':>24}")
    for t in TILES:
        c, l, w, e = chip_w(*t), laser_w(*t), total_w(*t), fj_mac(*t)
        print(f"  {name(t):<9}{rng(*c, ' W'):>18}{rng(*l, ' W'):>16}{rng(*w, ' W'):>16}{rng(*e, ' fJ', digits=0):>24}")
    print("  The chip's low end is 2 V swings with resident weights and its high end 5 V with")
    print(f"  weights re-sent.  The laser is behind {LOSS_DB[0]:.0f} and {LOSS_DB[1]:.0f} dB of loss, at the measured")
    print("  receiver, and follows the columns: 64 detectors want what 64 want, on any rows.")

    section("3. What it does with a layer")
    print(f"  {'tile':<9}{'MACs a second':>15}{'4096 square':>13}{'D3':>10}{'cells D3 uses':>15}"
          f"{'D3, a batch':>20}{'a MAC of D3':>18}")
    for t in TILES:
        big = layer(*t, BATCH, link.BIG["kin"], link.BIG["nout"])
        sec, macs, fill = d3(*t)
        lo, hi = total_w(*t)
        print(f"  {name(t):<9}{power.mac_s(*t):>15.2e}{dispatch.fmt_s(big['seconds']):>13}"
              f"{dispatch.fmt_s(sec):>10}{fill:>15.0%}{rng(lo * sec, hi * sec, ' uJ', 1e6):>20}"
              f"{rng(lo * sec / macs, hi * sec / macs, ' fJ', 1e15, 0):>18}")
    print("  Times are the tile and the link at each tile's own module count, a batch of 64,")
    print("  with no command in them (pta_dispatch.py section 4 has what a command costs).")
    print("  A network narrower than the tile pays for cells it does not use: D3 fills")
    print("  three fifths of a 256 x 64 tile's first-layer weights and a sixteenth of its second's.")

    section("4. What it costs in accuracy, on D3")
    print(f"  {'tile':<9}{'v1':>13}{'v0':>13}{'+ an hour':>13}{'+ four hours':>14}{'calibrated':>13}"
          f"{'receiver noise':>17}{'drift, an hour':>16}")
    for t in TILES:
        a = ACC[t]
        nf = "/".join(f"{noise_factor(t[0], s):.2f}" for s in a["shift"])
        print(f"  {name(t):<9}{pm(a['v1']):>13}{pm(a['v0']):>13}{pm(a['hour']):>13}{pm(a['four']):>14}"
              f"{pm(a['cal']):>13}{nf:>17}{drift_error(t):>15.1f}%")
    print("  Points lost against the same weights on the host, five networks.  'Receiver noise'")
    print("  is sqrt(conversions) x 2^shift on layer 1 against the 8 x 8 tile's: what the same")
    print("  LSB of receiver noise leaves on a layer.  Drift is an hour of TFLT's on layer 1,")
    print("  percent of its sums' rms, IN A MODEL WHOSE CELLS DRIFT INDEPENDENTLY.")

    section("5. The layer width at which a larger tile pays")
    print(f"  {'square layer':<14}" + "".join(f"{name(t):>22}" for t in CANDIDATES[1:]) + f"{'the cheaper of the first two':>32}")
    for w in WIDTHS:
        e = [fj_useful(t, w) for t in CANDIDATES[1:]]
        a, b = e[0], e[1]
        verdict = ("128x64 at both ends" if a[0] < b[0] and a[1] < b[1] else
                   "256x64 at both ends" if b[0] < a[0] and b[1] < a[1] else "one end each")
        print(f"  {w:>5} wide    " + "".join(f"{rng(*x, ' fJ', digits=0):>22}" for x in e) + f"{verdict:>32}")
    print("  Energy a MAC the layer asked for, a batch of 64.  The two ends of each range are")
    print("  the two ends of the power range, and which is real is pta_power.py's question.")
    print("  A tile's time on a layer narrower than itself is its first weight set's write")
    print("  and the link's round trip, which is why the small layers cost what they do.")

    findings()
    checks()


def findings():
    a, b = (128, 64), (256, 64)
    wa, wb = total_w(*a), total_w(*b)
    sa, ma, fa = d3(*a)
    sb, mb, fb = d3(*b)
    print()
    print("What this says, six readings.")
    print()
    print("  1. THE CHOICE IS BETWEEN 128 x 64 AND 256 x 64.  64 x 8 is an eighth of the")
    print(f"     laser and one module, and {power.mac_s(*b) / power.mac_s(64, 8):.0f} times slower than 256 x 64 at"
          f" {fj_mac(64, 8)[0] / fj_mac(*b)[0]:.0f} times the")
    print("     energy a MAC.  256 x 128 doubles the detectors, the laser and the modules of")
    print(f"     256 x 64 for {fj_mac(256, 128)[0] / fj_mac(*b)[0]:.2f} to {fj_mac(256, 128)[1] / fj_mac(*b)[1]:.2f}"
          f" of its energy a MAC, and its {modules(256, 128)} modules are a third of")
    print("     the way to where B8 reopens.")
    print()
    print("  2. WHAT 256 x 64 COSTS OVER 128 x 64: twice the lines between the dies")
    print(f"     ({fp.connections(*b):,} for {fp.connections(*a):,}), twice the cells' area,"
          f" {modules(*b)} modules for {modules(*a)}, and")
    print(f"     {wb[0] - wa[0]:.2f} to {wb[1] - wa[1]:.1f} W.  The laser is the same: it follows the columns.")
    print()
    print("  3. WHAT IT BUYS, IF THE LAYERS ARE WIDE: twice the MACs a second, and")
    print(f"     {fj_mac(*b)[0]:.0f}-{fj_mac(*b)[1]:.0f} fJ a MAC for {fj_mac(*a)[0]:.0f}-{fj_mac(*a)[1]:.0f}.  On D3,"
          " which is narrow, it buys nothing")
    print(f"     that can be told: {wb[0] * sb * 1e6:.2f}-{wb[1] * sb * 1e6:.1f} uJ a batch for"
          f" {wa[0] * sa * 1e6:.2f}-{wa[1] * sa * 1e6:.1f}, a tenth less at one end and")
    print(f"     a quarter more at the other.  D3 uses {fb:.0%} of the cells it has programmed"
          f" there and {fa:.0%} at 128 x 64.")
    print()
    print("  4. ACCURACY DOES NOT CHOOSE BETWEEN THEM ON v1.  0.34 +-0.07 and 0.20 +-0.05")
    print("     are a standard error and a half apart.  It leans to 256: the same receiver noise")
    print(f"     leaves {noise_factor(256, 11):.2f} of the 8 x 8 tile's error on a layer there and"
          f" {noise_factor(128, 11):.2f} at 128, because 128")
    print("     inputs need the ADC shift 256 need and take nearly twice the conversions.")
    print("     That lean is the harness's whole-bit shift.  A receiver whose gain steps are")
    print("     finer than a factor of two would not have it.")
    print()
    print("  5. DRIFT DOES NOT CHOOSE EITHER.  An hour leaves"
          f" {drift_error(a):.1f}% on a layer at 128 x 64 and {drift_error(b):.1f}% at")
    print(f"     256 x 64, against {drift_error(CORE):.1f}% on the 8 x 8 tile.  Both are large tiles"
          " to drift, in a model")
    print("     whose cells drift independently.")
    print()
    print("  6. SO THE QUESTION IS THE WORKLOAD'S, AND IT HAS A NUMBER.  On square layers at a")
    print("     batch of 64, 128 x 64 is the cheaper at both ends of the power range up to")
    print("     128 wide, and 256 x 64 from 512 wide.  Between them it is one end each.")
    print("     The plan sizes its link, its power and its time on a 4096-square layer")
    print("     nobody has trained, and measures accuracy on a 784-100-10 network.  What")
    print("     settles question 1 is which of those the board is for: a layer width.")


def checks():
    """Every figure above is the model's it came from, and every reading holds."""
    a, b, c = (128, 64), (256, 64), (256, 128)
    assert TILES == ((8, 8), (64, 8), (128, 64), (256, 64), (256, 128))
    assert (BITS, FS, BATCH) == (7, 1e9, 64) and power.X2_TILE == b

    # 1. The counts and the floorplan's bounds: 16,704 lines at 256 x 64, and its
    #    cells, inputs and link come to under half a standard die.
    assert [fp.connections(*t) for t in TILES] == [80, 584, 8_384, 16_704, 33_152]
    w, i, l = area_mm2(*b)
    assert abs(w - 10.24) < 0.01 and abs(i - 25.09) < 0.01 and abs(l - 4.40) < 0.01
    assert all(sum(area_mm2(*t)) < STD_DIE_MM2 for t in TILES)
    #    X2's module counts, and one module each with the weights resident.
    assert [modules(*t) for t in TILES] == [1, 1, 3, 5, 10]
    assert all(modules_resident(*t) == 1 for t in TILES)

    # 2. pta_power.py's own figures at X2's tile: 0.55 to 7.6 W of chip, and 39
    #    to 518 fJ a MAC with the laser.
    lo, hi = chip_w(*b)
    assert abs(lo - 0.55) < 0.005 and abs(hi - 7.61) < 0.005, (lo, hi)
    e = fj_mac(*b)
    assert 38.5 < e[0] < 39.5 and 515 < e[1] < 521, e
    #    The laser follows the columns and not the rows.
    assert laser_w(*a) == laser_w(*b) and abs(laser_w(*c)[1] / laser_w(*b)[1] - 2) < 1e-9
    assert abs(laser_w(*b)[0] - 0.088) < 0.001 and abs(laser_w(*b)[1] - 0.88) < 0.005

    # 3. pta_dispatch.py's layer: at its own five modules a 256 x 64 tile is
    #    bound by the tile on the 4096-square layer, 65.7 us, where at the one
    #    module that model assumed it is bound by the link at 296.
    big = layer(*b, BATCH, link.BIG["kin"], link.BIG["nout"])
    assert big["binds"] == "tile" and abs(big["seconds"] * 1e6 - 65.7) < 0.1
    one = dispatch.pta(BATCH, link.BIG["kin"], link.BIG["nout"])
    assert one["binds"] == "link" and abs(one["seconds"] * 1e6 - 296) < 1
    sb, mb, fb = d3(*b)
    assert mb == 5_081_600 and abs(sb * 1e6 - 0.96) < 0.01 and abs(fb - 0.54) < 0.01, (sb, fb)

    # 4. Reading 1.  64 x 8 is 32 times slower and four times the energy; 256 x
    #    128 is twice the laser and the modules for most of the energy a MAC.
    assert power.mac_s(*b) / power.mac_s(64, 8) == 32
    assert 3.8 < fj_mac(64, 8)[0] / fj_mac(*b)[0] < 4.0
    assert modules(*c) == 2 * modules(*b) and fp.connections(*c) > 1.98 * fp.connections(*b)
    assert 0.9 < fj_mac(*c)[0] / fj_mac(*b)[0] < 1.0 and 0.75 < fj_mac(*c)[1] / fj_mac(*b)[1] < 0.80

    # 5. Readings 2 and 3.  Twice the lines, 5 modules for 3, 0.11 to 3.6 W more;
    #    twice the rate at 0.60 to 0.86 of the energy a MAC with every cell in
    #    use; and on D3 nothing that can be told, the larger tile's batch a tenth
    #    cheaper at the low end and a quarter dearer at the high one.
    wa, wb = total_w(*a), total_w(*b)
    assert abs(fp.connections(*b) / fp.connections(*a) - 2) < 0.01
    assert abs((wb[0] - wa[0]) - 0.11) < 0.01 and abs((wb[1] - wa[1]) - 3.6) < 0.05, (wa, wb)
    assert power.mac_s(*b) == 2 * power.mac_s(*a)
    assert 0.59 < fj_mac(*b)[0] / fj_mac(*a)[0] < 0.61 and 0.85 < fj_mac(*b)[1] / fj_mac(*a)[1] < 0.87
    sa, ma, fa = d3(*a)
    assert fa > fb and ma == mb
    assert 0.88 < wb[0] * sb / (wa[0] * sa) < 0.92 and 1.25 < wb[1] * sb / (wa[1] * sa) < 1.32

    # 6. Reading 4.  v1's two losses are 1.6 standard errors apart, and the
    #    whole-bit shift's law gives the measured thermal error on both tiles
    #    to within one percent of it.
    va, vb = ACC[a]["v1"], ACC[b]["v1"]
    assert 1.5 < (va[0] - vb[0]) / math.hypot(va[1], vb[1]) < 1.7
    for t in (a, b, c):
        predicted = ACC[CORE]["e_thermal"] * noise_factor(t[0], ACC[t]["shift"][0])
        assert abs(predicted / ACC[t]["e_thermal"] - 1) < 0.01, (t, predicted)
    assert abs(noise_factor(256, 11) - 0.81) < 0.005 and abs(noise_factor(128, 11) - 1.07) < 0.005
    #    At 64 x 8 the two shifts the networks landed on bracket the mean.
    lo64, hi64 = (ACC[CORE]["e_thermal"] * noise_factor(64, s) for s in ACC[(64, 8)]["shift"])
    assert lo64 < ACC[(64, 8)]["e_thermal"] < hi64
    #    v1 is within a standard error and a half of the core tile's on every candidate.
    for t in CANDIDATES:
        v = ACC[t]["v1"]
        assert abs(v[0] - ACC[CORE]["v1"][0]) / math.hypot(v[1], ACC[CORE]["v1"][1]) < 1.5, t

    # 7. Reading 5.  An hour of drift leaves 7.6 and 6.9 percent on the two, and
    #    19.6 on the core's tile.
    assert abs(drift_error(a) - 7.6) < 0.05 and abs(drift_error(b) - 6.9) < 0.05
    assert abs(drift_error(CORE) - 19.6) < 0.05
    assert all(drift_error(t) < 0.55 * drift_error(CORE) for t in CANDIDATES)

    # 8. Reading 6.  D3 fills 54% of what a 256 x 64 tile programs for it and
    #    the 4096-square layer all of it.  And the width at which the larger
    #    tile pays: 128 x 64 is cheaper at both ends up to 128 wide, 256 x 64
    #    from 512, and at 256 wide it is one end each.
    assert big["fill"] == 1.0 and fb < 0.6
    for w in WIDTHS:
        ea, eb = fj_useful(a, w), fj_useful(b, w)
        if w <= 128:
            assert ea[0] < eb[0] and ea[1] < eb[1], w
        elif w >= 512:
            assert eb[0] < ea[0] and eb[1] < ea[1], w
        else:
            assert eb[0] < ea[0] and ea[1] < eb[1], w
    #    On the widest it is section 2's figure: every cell in use.
    wide = fj_useful(b, 4096)
    assert abs(wide[0] / fj_mac(*b)[0] - 1) < 0.01 and abs(wide[1] / fj_mac(*b)[1] - 1) < 0.01

    # 9. board_program_plan.md section 8, question 1, quotes these, figure for figure.
    quoted = {
        (64, 8):    ("7.5", "0.07-1.10", "0.01-0.11", "152-2372", "2.1 ms"),
        (128, 64):  ("20.3", "0.44-4.04", "0.09-0.88", "65-600", "131 us"),
        (256, 64):  ("39.7", "0.55-7.61", "0.09-0.88", "39-518", "65.7 us"),
        (256, 128): ("54.4", "1.02-11.46", "0.18-1.76", "37-403", "32.9 us"),
    }
    for t in CANDIDATES:
        got = (f"{sum(area_mm2(*t)):.1f}", rng(*chip_w(*t)), rng(*laser_w(*t)),
               rng(*fj_mac(*t), digits=0),
               dispatch.fmt_s(layer(*t, BATCH, link.BIG["kin"], link.BIG["nout"])["seconds"]))
        assert got == quoted[t], (t, got)
    widths = {64: ("462-4276", "557-7385"), 128: ("148-1369", "178-2365"), 256: ("86-792", "64-850"),
              512: ("70-648", "45-601"), 4096: ("65-601", "39-520")}
    for w, want in widths.items():
        assert (rng(*fj_useful(a, w), digits=0), rng(*fj_useful(b, w), digits=0)) == want, w
    assert (f"{fa:.0%}", f"{fb:.0%}") == ("65%", "54%")
    assert rng(wb[0] * sb, wb[1] * sb, scale=1e6) == "0.61-8.13"
    assert rng(wa[0] * sa, wa[1] * sa, scale=1e6) == "0.68-6.33"
    assert [f"{drift_error(t):.1f}" for t in CANDIDATES] == ["10.4", "7.6", "6.9", "7.2"]
    assert dispatch.fmt_s(sb) == "957 ns" and abs(sum(dispatch.pta(BATCH, x, y)["seconds"]
                                                      for x, y in dispatch.D3) * 1e6 - 3.74) < 0.005

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
