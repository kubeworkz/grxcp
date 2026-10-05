"""
The working point as it stands, and the two scorecards at the corrected laser.

On 2026-10-05 the board plan fixed the chiplet's tile in four decisions and then
moved it.  B10 made it 256 x 64, B11 gave it 1 GS/s, B12 made it a ring bank lit
by a comb, B13 gave a column a balanced pair of photodiodes, and B5's laser was
found low by the light a column is not sent.  With the laser corrected the
larger tile no longer bought a cheaper MAC, and B10 was revised to 128 x 64.

The models behind those decisions were each run at 256 x 64 and are kept as
they were run.  This asks each of them at the tile as it now stands and puts
the answers in one place, beside what they were.  And it runs the two
scorecards again, pta_geometry.py's and pta_rate.py's, with the laser
pta_laser.py corrected, which is what both were waiting for.

Nothing here is new.  Every figure is one of those models' own, asked for at
another tile, and the checks hold each to the model it came from: at 256 x 64
this reproduces pta_rate.py's rows to the last figure, and at every tile
pta_geometry.py's.

WHAT KIND OF NUMBER EACH ROW IS is what it was in the model it came from, and
those say.  Two things are particular to this file:

  the laser      pta_laser.py's: B5's method, times the multiple at which
                 grx930's harness is within a tenth of a point of version 1.
                 MEASURED IN A MODEL at 128 x 64 and 256 x 64, 8 and 16 times.
                 DERIVED at the other candidates: the multiple that leaves the
                 receiver what those two were left, a quarter of an LSB on the
                 first layer.  Nobody ran those
  the rescale    pta_laser.py's last section: the hidden layer's rescale one bit
                 under the clip rule halves the laser the working tile needs.
                 MEASURED IN A MODEL at 128 x 64 and nowhere else, so every
                 other tile and both scorecards stay at the rule's rescale
  the buses      a ring bank's, for a tile's own inputs: pta_source.py's
                 arithmetic at another input count, by B12's rule, the fewest
                 that hold across the published range of line spacings.  128
                 inputs are two buses of 64 lines, on the comb B12 asked for

Standard library only.  Run:  python3 docs/designs/pta_working_point.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_chiplet_link as link
import pta_dispatch as dispatch
import pta_floorplan as fp
import pta_geometry as geometry
import pta_laser as laser
import pta_power as power
import pta_rate as rate
import pta_ring as ring
import pta_shot_rate as shot
import pta_source as source

TILE = (128, 64)                           # B10, as revised
FIRST = shot.X2_TILE                       # 256 x 64: B10 as first settled, and every model's
FS = shot.X2_FS                            # 1 GS/s: B11
V1 = shot.REQ["v1"]
BITS = V1["adc_bits"]
BATCH = link.BIG["mb"]
WIDE = (link.BIG["kin"], link.BIG["nout"])
LOSS_DB = shot.LOSS_DB
CANDIDATES = geometry.CANDIDATES
RATES = rate.RATES
MEASURED = tuple(laser.LOST)               # the tiles grx930 ran against their laser
LEFT_LSB = laser.noise_lsb(FIRST, 0, laser.times_within(FIRST))    # 0.25: what both were left
SMALLEST_RING_UM = 2 * ring.K30["radius_um"]                       # a 60 um cell


# ---- the laser ---------------------------------------------------------------------
def times(tile):
    """The multiple of B5's laser a tile needs: pta_laser.py's where grx930 ran it,
    and elsewhere the one that leaves the receiver what those were left."""
    if tile in MEASURED:
        return laser.times_within(tile)
    return laser.noise_lsb(tile, 0) / LEFT_LSB


def b5_w(tile, fs=FS):
    """B5's method at a tile and a rate: pta_rate.py's, at another tile's columns."""
    na = power.receiver_noise_a(fs, BITS)
    lo, hi = (power.laser_w(V1, db, na, tile[1]) for db in LOSS_DB)
    past = fs / rate.receiver_carries_hz()
    if past > 1:
        hi *= past ** (shot.NOISE_LAWS[1][1] - shot.NOISE_LAWS[0][1])
    return lo, hi


def laser_w(tile, fs=FS):
    lo, hi = b5_w(tile, fs)
    return lo * times(tile), hi * times(tile)


def times_down(tile):
    """The multiple with the hidden rescale a bit under the clip rule: pta_laser.py's,
    which grx930 ran at the working tile alone.  None anywhere else."""
    return laser.times_put(laser.BIT_DOWN) if tile == laser.SMALLER else None


def laser_down_w(tile, fs=FS):
    lo, hi = b5_w(tile, fs)
    return lo * times_down(tile), hi * times_down(tile)


def fj_mac_down(tile, fs=FS):
    c, l = chip_w(tile, fs), laser_down_w(tile, fs)
    return tuple((c[i] + l[i]) / power.mac_s(*tile, fs) * 1e15 for i in (0, 1))


# ---- the interface chip -------------------------------------------------------------
def chip_parts(tile, fs=FS):
    """pta_rate.py's parts, at another tile."""
    k, n = tile
    _, lo = power.chip_w(k, n, BITS, rate.SWINGS_V[0], True, fs)
    _, hi = power.chip_w(k, n, BITS, rate.SWINGS_V[1], False, fs)
    p = {name: (lo[name][0], hi[name][1]) for name in lo}
    floor = rate.swing_floor_v(fs)
    p["weight drive"] = tuple(
        power.weight_drive_w(k, n, max(v, floor), c, a, fs)
        for v, c, a in zip(rate.SWINGS_V, power.C_CELL_F, power.ACTIVITY))
    return p


def chip_w(tile, fs=FS):
    p = chip_parts(tile, fs)
    return sum(x[0] for x in p.values()), sum(x[1] for x in p.values())


def total_w(tile, fs=FS, corrected=True):
    c, l = chip_w(tile, fs), (laser_w(tile, fs) if corrected else b5_w(tile, fs))
    return c[0] + l[0], c[1] + l[1]


def fj_mac(tile, fs=FS, corrected=True):
    lo, hi = total_w(tile, fs, corrected)
    return lo / power.mac_s(*tile, fs) * 1e15, hi / power.mac_s(*tile, fs) * 1e15


# ---- the link, and the time ----------------------------------------------------------
def modules(tile, fs=FS, resident=False):
    return math.ceil(fs / shot.feed_rate(mods=1, resident=resident, k=tile[0], n=tile[1], **link.BIG))


def layer(tile, fs, m, kin, nout, mods=None):
    mods = modules(tile, fs) if mods is None else mods
    return dispatch.pta(m, kin, nout, tile=tile, fs=fs, mods=mods,
                        per_beat=shot.per_beat_two_banks(*tile, BATCH))


def d3_s(tile, fs=FS):
    return sum(layer(tile, fs, BATCH, a, b)["seconds"] for a, b in dispatch.D3)


# ---- the ring bank --------------------------------------------------------------------
def lines_a_bus(fs=FS, spacing=source.M_SPACING[1]):
    """Lines a bus holds at the narrowest line the rate passes.  At the published
    worst spacing unless told: B12 took the count that holds across the range."""
    return int(source.FSR_HZ / (spacing * rate.line_hz(fs)))


def lines_most(fs=FS):
    """The same at the error model's own spacing, which is the kind end: pta_rate.py's."""
    return lines_a_bus(fs, source.V1_SPACING)


def buses(tile, fs=FS, spacing=source.M_SPACING[1]):
    return -(-tile[0] // lines_a_bus(fs, spacing))


def lines_each(tile, fs=FS):
    return -(-tile[0] // buses(tile, fs))


def grid_hz(tile, fs=FS):
    """The widest whole multiple of the shot rate that fits a bus's lines in one range."""
    return math.floor(source.FSR_HZ / lines_each(tile, fs) / fs) * fs


def photodiodes(tile, fs=FS):
    """B13: a pair a bus a column."""
    return 2 * buses(tile, fs) * tile[1]


def line_w(tile, fs=FS):
    lo, hi = laser_w(tile, fs)
    return lo / lines_each(tile, fs), hi / lines_each(tile, fs)


# ---- the dies --------------------------------------------------------------------------
def lines_between(tile):
    return tile[0] * tile[1] + tile[0] + tile[1]


def die_at_ring_mm2(tile):
    """Weights, inputs and link at the smallest ring pta_ring.py read."""
    return sum(ring.die_mm2(tile, SMALLEST_RING_UM))


def name(t):
    return f"{t[0]}x{t[1]}"


def span(lo, hi, digits=2):
    return f"{lo:.{digits}f}-{hi:.{digits}f}"


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    print("The working point as it stands, and the two scorecards at the corrected laser.")
    print(f"{name(TILE)} (B10, as revised) at {FS / 1e9:g} GS/s (B11), a ring bank (B12) read by balanced pairs (B13).")

    section("1. The working point, beside what it was")
    rows = (
        ("lines between the dies", lambda t: f"{lines_between(t):,}"),
        ("the die: the floorplan's bound", lambda t: f"{sum(geometry.area_mm2(*t)):.1f} mm2"),
        ("the die: at the smallest ring read", lambda t: f"{die_at_ring_mm2(t):.0f} mm2"),
        ("link modules, weights re-sent", lambda t: f"{modules(t)}"),
        ("interface chip", lambda t: span(*chip_w(t)) + " W"),
        ("laser: B5's method", lambda t: span(*b5_w(t)) + " W"),
        ("laser: on the light a column is sent", lambda t: f"{laser_w(t)[0]:.1f}-{laser_w(t)[1]:.1f} W, {times(t):.0f} times"),
        ("a MAC, every cell in use", lambda t: f"{fj_mac(t)[0]:.0f}-{fj_mac(t)[1]:,.0f} fJ"),
        ("laser: the hidden rescale a bit down", lambda t: "not run" if times_down(t) is None else
         f"{laser_down_w(t)[0]:.2f}-{laser_down_w(t)[1]:.1f} W, {times_down(t)} times"),
        ("a MAC, at that", lambda t: "" if times_down(t) is None else
         f"{fj_mac_down(t)[0]:.0f}-{fj_mac_down(t)[1]:,.0f} fJ"),
        ("the 4096-square layer", lambda t: f"{layer(t, FS, BATCH, *WIDE)['seconds'] * 1e6:.1f} us"),
        ("D3's two layers", lambda t: f"{d3_s(t) * 1e6:.2f} us"),
        ("weights a bank, and written a beat", lambda t: f"{t[0] * t[1]:,}, {shot.per_beat_two_banks(*t, BATCH)}"),
        ("the ring bank's buses, and lines each", lambda t: f"{buses(t)} of {lines_each(t)}"),
        ("its grid", lambda t: f"{grid_hz(t) / 1e9:.0f} GHz"),
        ("photodiodes", lambda t: f"{photodiodes(t)}"),
        ("a line of the comb", lambda t: f"{line_w(t)[0] * 1e3:.0f}-{line_w(t)[1] * 1e3:.0f} mW"),
        ("v1 on D3, as budgeted", lambda t: f"{geometry.ACC[t]['v1'][0]:.2f} +-{geometry.ACC[t]['v1'][1]:.2f}"),
        ("v1 on D3, at that laser", lambda t: f"{laser.LOST[t][times(t)][0]:.2f} +-{laser.LOST[t][times(t)][1]:.2f}"),
        ("an hour of TFLT's drift", lambda t: f"{geometry.ACC[t]['hour'][0]:.2f} +-{geometry.ACC[t]['hour'][1]:.2f}"),
    )
    print(f"  {'':<40}{name(TILE):>24}{'as first settled, ' + name(FIRST):>30}")
    for label, f in rows:
        print(f"  {label:<40}{f(TILE):>24}{f(FIRST):>30}")
    print(f"  A standard die is {geometry.STD_DIE_MM2:.0f} mm2.  Buses are by B12's rule, the fewest that hold across the")
    print(f"  published range of spacings.  At its kind end {name(FIRST)} would be {buses(FIRST, FS, source.V1_SPACING)} and"
          f" {name(TILE)} still {buses(TILE, FS, source.V1_SPACING)}.")

    section("2. The geometry scorecard, with the laser corrected")
    print(f"  {'':<36}" + "".join(f"{name(t):>18}" for t in CANDIDATES))
    grows = (
        ("laser, as the scorecard had it, W", lambda t: span(*b5_w(t))),
        ("the multiple it needs", lambda t: f"{times(t):.0f}" + ("" if t in MEASURED else ", derived")),
        ("laser, W", lambda t: f"{laser_w(t)[0]:.1f}-{laser_w(t)[1]:.1f}"),
        ("interface chip, W", lambda t: span(*chip_w(t))),
        ("a MAC as it had it, fJ", lambda t: f"{fj_mac(t, FS, False)[0]:.0f}-{fj_mac(t, FS, False)[1]:,.0f}"),
        ("a MAC, fJ", lambda t: f"{fj_mac(t)[0]:.0f}-{fj_mac(t)[1]:,.0f}"),
        ("the die at the smallest ring, mm2", lambda t: f"{die_at_ring_mm2(t):.0f}"),
        ("the 4096-square layer", lambda t: f"{layer(t, FS, BATCH, *WIDE)['seconds'] * 1e6:.0f} us"),
    )
    for label, f in grows:
        print(f"  {label:<36}" + "".join(f"{f(t):>18}" for t in CANDIDATES))
    print("  The multiple is grx930's where it ran the tile, and elsewhere the one that leaves the")
    print(f"  receiver {LEFT_LSB:.2f} LSB on the first layer, as those two were left.")

    section(f"3. The shot-rate scorecard, at {name(TILE)} with the laser corrected")
    print(f"  {'rate':<11}{'modules':>9}{'chip, W':>14}{'laser, W':>14}{'a MAC, fJ':>14}{'wide layer':>13}{'swing':>9}"
          f"{'lines a bus':>13}{'buses':>7}{'photodiodes':>13}")
    for fs in RATES:
        print(f"  {rate.gs(fs):<11}{modules(TILE, fs):>9}{span(*chip_w(TILE, fs)):>14}"
              f"{f'{laser_w(TILE, fs)[0]:.1f}-{laser_w(TILE, fs)[1]:.1f}':>14}"
              f"{f'{fj_mac(TILE, fs)[0]:.0f}-{fj_mac(TILE, fs)[1]:,.0f}':>14}"
              f"{layer(TILE, fs, BATCH, *WIDE)['seconds'] * 1e6:>10.1f} us{rate.swing_floor_v(fs):>7.2f} V"
              f"{lines_most(fs):>13}{buses(TILE, fs):>7}{photodiodes(TILE, fs):>13}")
    print("  The swing and the lines a bus are the ring's and do not know the tile: pta_rate.py's,")
    print("  the lines at the kind spacing.  Buses are at the worst, as B12 counts them.")
    print(f"  The laser is {times(TILE):.0f} times B5's at every rate: what it needs is a fraction of a line's light.")

    findings()
    checks()


def findings():
    print()
    print("What this says, six readings.")
    print()
    a, b = fj_mac(TILE), fj_mac(FIRST)
    print(f"  1. THE MOVE HALVES WHAT THE TILE COSTS AND BARELY MOVES WHAT A MAC COSTS.  {lines_between(TILE):,} lines for {lines_between(FIRST):,},")
    print(f"     {modules(TILE)} modules for {modules(FIRST)}, {photodiodes(TILE)} photodiodes for {photodiodes(FIRST)}, and {laser_w(TILE)[0]:.1f} to {laser_w(TILE)[1]:.0f} W of laser for"
          f" {laser_w(FIRST)[0]:.1f} to {laser_w(FIRST)[1]:.0f}.")
    print(f"     A MAC is {a[0]:.0f} to {a[1]:,.0f} fJ where it was {b[0]:.0f} to {b[1]:,.0f}.  What it gives up is time: the wide")
    print(f"     layer takes {layer(TILE, FS, BATCH, *WIDE)['seconds'] * 1e6:.0f} us for {layer(FIRST, FS, BATCH, *WIDE)['seconds'] * 1e6:.0f}.")
    print()
    print(f"  2. THE DIE FITS, AT ONE RING.  {die_at_ring_mm2(TILE):.0f} mm2 at the smallest ring pta_ring.py read, inside a")
    print(f"     standard {geometry.STD_DIE_MM2:.0f}, where {name(FIRST)} was {die_at_ring_mm2(FIRST):.0f}.  At the next ring read it is"
          f" {sum(ring.die_mm2(TILE, 140)):.0f}.  So it")
    print("     fits with rings touching and nothing added to a cell, and at no other ring read.")
    print()
    print(f"  3. THE RING BANK IS TWO BUSES, ON THE COMB B12 ASKED FOR.  {lines_each(TILE)} lines on a {grid_hz(TILE) / 1e9:.0f} GHz grid,")
    print(f"     each line lighting two rows and not four: the count that holds across the")
    print(f"     published range of spacings, as four was for {name(FIRST)}.")
    print()
    cheapest = min(CANDIDATES, key=lambda t: fj_mac(t)[0])
    new, old = fj_mac(TILE), fj_mac(TILE, FS, False)
    big_new, big_old = fj_mac(FIRST), fj_mac(FIRST, FS, False)
    print("  4. WITH THE LASER CORRECTED A LARGER TILE BUYS LITTLE OF A MAC.  The laser follows the")
    print("     rows as well as the columns, so most of the scorecard's fall in energy is gone:")
    print(f"     {', '.join(f'{fj_mac(t)[0]:.0f}-{fj_mac(t)[1]:,.0f}' for t in CANDIDATES[1:])} fJ from {name(CANDIDATES[1])} up, where it had"
          f" {', '.join(f'{fj_mac(t, FS, False)[0]:.0f}-{fj_mac(t, FS, False)[1]:.0f}' for t in CANDIDATES[1:])}.")
    print(f"     {name(FIRST)} is {1 - big_new[0] / new[0]:.0%} cheaper than {name(TILE)} at one end and {1 - big_new[1] / new[1]:.0%} at the other, where it was"
          f" {1 - big_old[0] / old[0]:.0%} and {1 - big_old[1] / old[1]:.0%}.")
    print("     What a larger tile buys is time, and it pays in light, in area and in link.")
    print()
    e = [fj_mac(TILE, fs) for fs in RATES]
    print(f"  5. AT {name(TILE)} A MAC STILL GETS CHEAPER WITH THE RATE, and the laser is now most of it.")
    print(f"     {', '.join(f'{x[0]:.0f}' for x in e)} fJ at the low end across the five rates and"
          f" {', '.join(f'{x[1]:,.0f}' for x in e)} at the high.")
    print(f"     At {rate.gs(RATES[1])} one bus carries all {TILE[0]} inputs; at {rate.gs(FS)} it takes {buses(TILE)}.  B11's 1 GS/s was")
    print("     chosen on the measured parts' reach, and that has not moved.")
    print()
    d = fj_mac_down(TILE)
    print(f"  6. AND THE HOST HALVES THE LASER AGAIN.  With the hidden layer's rescale one bit under the")
    print(f"     clip rule the working tile needs {times_down(TILE)} times B5's laser and not {times(TILE)}: {laser_down_w(TILE)[0]:.2f} to {laser_down_w(TILE)[1]:.1f} W, and")
    print(f"     a MAC is {d[0]:.0f} to {d[1]:.0f} fJ.  It is the activation stage's shift, a field of a command,")
    print("     and it was run at this tile alone: the two scorecards above are at the rule's.")


def checks():
    """Every figure above, held to the model it came from."""
    assert TILE == (128, 64) and FIRST == (256, 64) and FS == 1e9 and abs(LEFT_LSB - 0.25) < 0.005

    # 1. At 256 x 64 this is pta_rate.py, row for row, at every rate.
    for fs in RATES:
        assert modules(FIRST, fs) == rate.modules(fs) and modules(FIRST, fs, True) == rate.modules(fs, True)
        for mine, theirs in ((chip_w(FIRST, fs), rate.chip_w(fs)), (b5_w(FIRST, fs), rate.laser_w(fs)),
                             (fj_mac(FIRST, fs, False), rate.fj_mac(fs))):
            assert all(abs(x - y) < 1e-9 * max(1.0, abs(y)) for x, y in zip(mine, theirs)), (fs, mine, theirs)
        assert abs(layer(FIRST, fs, BATCH, *WIDE)["seconds"] - rate.layer(fs, BATCH, *WIDE)["seconds"]) < 1e-15
        assert lines_most(fs) == rate.lines(fs) and buses(FIRST, fs, source.V1_SPACING) == rate.buses(fs)
    #    And at every candidate, at the working rate, pta_geometry.py.
    for t in CANDIDATES:
        assert modules(t) == geometry.modules(*t)
        assert all(abs(x / y - 1) < 0.002 for x, y in zip(chip_w(t), geometry.chip_w(*t))), t
        assert all(abs(x - y) < 1e-12 for x, y in zip(b5_w(t), geometry.laser_w(*t))), t
        assert abs(layer(t, FS, BATCH, *WIDE)["seconds"] - geometry.layer(*t, BATCH, *WIDE)["seconds"]) < 1e-15
    #    The laser's multiple is pta_laser.py's where it was run, and its fill's elsewhere.
    assert times(TILE) == laser.times_within(TILE) == 8 and times(FIRST) == 16
    assert [round(times(t)) for t in CANDIDATES] == [8, 8, 16, 16]
    assert all(abs(laser.noise_lsb(t, 0, times(t)) - LEFT_LSB) < 0.005 for t in CANDIDATES)
    assert all(abs(x - y) < 1e-12 for x, y in zip(laser_w(TILE), laser.laser_w(TILE, 8)))

    # 2. Reading 1.  The working point's own figures.
    assert lines_between(TILE) == 8_384 and lines_between(FIRST) == 16_704
    assert (modules(TILE), modules(FIRST)) == (3, 5) and modules(TILE, FS, True) == 1
    assert [round(x, 2) for x in chip_w(TILE)] == [0.44, 4.04]
    assert abs(laser_w(TILE)[0] - 0.70) < 0.005 and abs(laser_w(TILE)[1] - 7.02) < 0.01
    #    (pta_laser.py has 119 for the larger tile's low end: it takes the scorecard's
    #    chip, whose weights swing 2 V where a ring at 1 GS/s needs 2.27.)
    assert [round(x) for x in fj_mac(TILE)] == [140, 1351] and [round(x) for x in fj_mac(FIRST)] == [120, 1323]
    assert abs(fj_mac(FIRST)[0] - laser.fj_mac(FIRST, 16)[0]) < 0.5
    assert all(abs(x / y - 1) < 0.2 for x, y in zip(fj_mac(TILE), fj_mac(FIRST)))
    wide = [layer(t, FS, BATCH, *WIDE)["seconds"] * 1e6 for t in (TILE, FIRST)]
    assert abs(wide[0] - 131.2) < 0.1 and abs(wide[1] - 65.7) < 0.1
    assert TILE[0] * TILE[1] == 8_192 and shot.per_beat_two_banks(*TILE, BATCH) == 128
    assert abs(shot.weight_updates_per_s(*TILE, BATCH, FS) - 128e9) < 1
    assert (photodiodes(TILE), photodiodes(FIRST)) == (256, 512)
    assert geometry.ACC[TILE]["v1"] == (0.34, 0.07) and laser.LOST[TILE][8] == (0.35, 0.06)

    # 3. Reading 2.  62 mm2 at the smallest ring, inside a standard die; 233 at the next.
    assert abs(die_at_ring_mm2(TILE) - 62.2) < 0.1 and die_at_ring_mm2(TILE) < geometry.STD_DIE_MM2
    assert abs(die_at_ring_mm2(FIRST) - 123.6) < 0.1 and round(sum(ring.die_mm2(TILE, 140))) == 233
    assert abs(sum(geometry.area_mm2(*TILE)) - 20.3) < 0.05

    # 4. Reading 3.  Two buses of 64 lines on an 11 GHz grid, and two at the
    #    published worst spacing; 256 x 64 is three at the kind and four at the worst.
    assert (buses(TILE), lines_each(TILE), grid_hz(TILE)) == (2, 64, 11e9)
    assert buses(TILE, FS, source.V1_SPACING) == 2 and buses(TILE, FS, source.M_SPACING[0]) == 2
    assert buses(FIRST) == 4 and buses(FIRST, FS, source.V1_SPACING) == 3
    assert (lines_each(FIRST), grid_hz(FIRST)) == (64, 11e9)
    first_line = [x * 1e3 for x in line_w(FIRST)]
    assert abs(first_line[0] - 21.9) < 0.1 and abs(first_line[1] - 219.5) < 0.5
    assert grid_hz(TILE) == source.clocked_hz(4) and lines_each(TILE) == source.lines_needed(4)
    line = [x * 1e3 for x in line_w(TILE)]
    assert abs(line[0] - 11.0) < 0.1 and abs(line[1] - 109.7) < 0.3

    # 5. Reading 4.  A MAC at the corrected laser is level from 128 x 64 up, where
    #    the scorecard had it falling.
    new = [fj_mac(t) for t in CANDIDATES[1:]]
    old = [fj_mac(t, FS, False) for t in CANDIDATES[1:]]
    assert old[0][0] > 1.5 * old[-1][0] and max(x[0] for x in new) < 1.25 * min(x[0] for x in new)
    assert old[0][1] > 1.45 * old[-1][1] and max(x[1] for x in new) < 1.15 * min(x[1] for x in new)
    #    256 x 64 against 128 x 64: 15% and 2% cheaper, where it was 40% and 14%.
    a, b = fj_mac(TILE), fj_mac(FIRST)
    a0, b0 = fj_mac(TILE, FS, False), fj_mac(FIRST, FS, False)
    assert [round(100 * (1 - b[i] / a[i])) for i in (0, 1)] == [15, 2]
    assert [round(100 * (1 - b0[i] / a0[i])) for i in (0, 1)] == [40, 14]
    assert [round(x) for x in fj_mac(CANDIDATES[0])] == [302, 3873], fj_mac(CANDIDATES[0])

    # 6. Reading 5.  The rates at 128 x 64.
    assert [modules(TILE, fs) for fs in RATES] == [1, 2, 3, 5, 10]
    assert [buses(TILE, fs) for fs in RATES] == [1, 1, 2, 4, 7]
    e = [fj_mac(TILE, fs) for fs in RATES]
    assert all(e[i][0] > e[i + 1][0] for i in range(4))
    assert all(laser_w(TILE, fs)[0] > chip_w(TILE, fs)[0] for fs in RATES)
    assert all(laser_w(TILE, fs)[1] > chip_w(TILE, fs)[1] for fs in RATES)
    assert e[3][1] < e[2][1] < e[1][1] and e[4][1] > e[3][1]

    #    Reading 6: with the rescale a bit down, 4 times and not 8, at this tile alone.
    assert times_down(TILE) == 4 and times_down(FIRST) is None and times(TILE) == 2 * times_down(TILE)
    assert abs(laser_down_w(TILE)[0] - 0.35) < 0.005 and abs(laser_down_w(TILE)[1] - 3.51) < 0.01
    assert [round(x) for x in fj_mac_down(TILE)] == [97, 922]
    assert all(abs(x - y) < 0.5 for x, y in zip(fj_mac_down(TILE), laser.fj_mac(TILE, 4)))
    assert laser.PUT[laser.BIT_DOWN][4] == laser.LOST[TILE][8]
    line_down = [x / lines_each(TILE) * 1e3 for x in laser_down_w(TILE)]
    assert abs(line_down[0] - 5.5) < 0.05 and abs(line_down[1] - 54.9) < 0.2

    # 7. The plan's two tables, cell by cell.
    assert [round(chip_w(TILE, fs)[0], 2) for fs in RATES] == [0.31, 0.35, 0.44, 0.63, 1.18]
    assert [round(chip_w(TILE, fs)[1], 2) for fs in RATES] == [1.19, 2.13, 4.04, 7.99, 18.85]
    assert [round(laser_w(TILE, fs)[0], 1) for fs in RATES] == [0.4, 0.5, 0.7, 1.0, 1.4]
    assert [round(laser_w(TILE, fs)[1], 1) for fs in RATES] == [3.5, 5.0, 7.0, 11.7, 33.1]
    assert [round(x[0]) for x in e] == [322, 207, 140, 99, 79]
    assert [round(x[1]) for x in e] == [2299, 1734, 1351, 1202, 1585]
    t = [layer(TILE, fs, BATCH, *WIDE)["seconds"] * 1e6 for fs in RATES]
    assert [round(x, 1) for x in t] == [524.6, 262.4, 131.2, 65.7, 32.9]
    assert [f"{laser_w(c)[0]:.1f}-{laser_w(c)[1]:.1f}" for c in CANDIDATES] == ["0.1-0.9", "0.7-7.0", "1.4-14.1", "2.8-28.1"]
    assert [round(x) for x in fj_mac(CANDIDATES[-1])] == [117, 1208]
    assert [round(die_at_ring_mm2(c)) for c in CANDIDATES] == [18, 62, 124, 187]
    #    And B10's own: D3's layers, the drift, and the four hours alike.
    assert abs(d3_s(TILE) * 1e6 - 1.29) < 0.005 and abs(d3_s(FIRST) * 1e6 - 0.96) < 0.005
    assert geometry.ACC[TILE]["hour"] == (0.51, 0.07) and geometry.ACC[FIRST]["hour"] == (0.24, 0.06)
    assert abs(geometry.ACC[TILE]["four"][0] - geometry.ACC[FIRST]["four"][0]) < 0.05
    gap = geometry.ACC[TILE]["v1"][0] - geometry.ACC[FIRST]["v1"][0]
    assert 1.4 < gap / math.hypot(geometry.ACC[TILE]["v1"][1], geometry.ACC[FIRST]["v1"][1]) < 1.8
    assert laser.LOST[TILE][8][0] - laser.LOST[FIRST][16][0] < math.hypot(0.06, 0.06)

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
