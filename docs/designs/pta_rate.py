"""
Open question 1 of board_program_plan.md, its second half as one table: how fast?

B10 made 256 x 64 the chiplet's working geometry and left the shot rate where
X2 put it, at 1 GS/s, "as a planning figure".  Six models have a say in the
rate, each for its own reason, and none of them was run at more than one:

  pta_shot_rate.py      what bounds the rate: the modulator, the feed, the receiver
  pta_chiplet_link.py   how many modules carry a rate
  pta_adc_survey.py     what a conversion costs at a rate
  pta_power.py          what the interface chip and the laser draw
  pta_dispatch.py       how long a layer takes, command to result
  pta_ring.py           the line a weight ring has

This asks each of them at five candidate rates, at the working geometry, and
puts the answers side by side: pta_geometry.py's method on the other axis.  The
checks hold each row to the model it came from.

ONE THING HERE IS NEW, and it is a fourth bound on the rate.  pta_shot_rate.py
bounded the rate by the modulator, the feed and the receiver, and found the
modulator so loose, at 51 GS/s, that it "is never the bound that binds".  It had
no weight cell to ask.  pta_ring.py has since made the cells rings, and compared
a ring's line with how often a weight is REWRITTEN, 15.6 MHz, which every line
clears fifty times over.  That was the wrong clock.  The light a ring weighs is
an input's, and an input changes every shot.

DERIVED here, from a resonance being a Lorentzian line:

  a ring's field      rings down in 1 / (pi * linewidth) seconds: a single pole
                      whose corner is half the linewidth
  a level through it  the detector reads power, the square of the field, so
                      half an LSB of power is a quarter of an LSB of field.  A
                      level has settled after ln(2^(bits+2)) of those time
                      constants: pta_tw_sweep.py's settle, at one bit more
  the ring's rate     one over that settle.  An upper bound, the settle being
                      the whole shot, which is how pta_shot_rate.py takes the
                      modulator's
  lines x rate        a ring's free spectral range holds a fixed number of
                      linewidths, a line a bus costs some of them and a shot a
                      second costs some more, so lines a bus times the shot
                      rate is a constant of the ring

It holds for any resonant cell, whether or not the tile is a ring bank: a
resonance that sharp has a memory that long.

WHAT KIND OF NUMBER EACH ROW IS, because they differ:

  link modules      PREDICTED by X2's accounting, for a 4096-square layer at a
                    batch of 64.
  a conversion      PUBLISHED: the best and the fifth-best converter in the ADC
                    survey that reach seven effective bits at that rate.
  the receiver      MEASURED, one amplifier, 1.5 GHz wide.  Its noise is taken
                    over the bandwidth a rate needs.  Past the rate its own
                    bandwidth settles, the figure is an extrapolation, and the
                    laser's upper end takes the harsher of pta_shot_rate.py's
                    two laws from there.
  watts             pta_power.py's sum at each rate, between pta_geometry.py's
                    two ends: 2 V with the weights held on the interface chip,
                    and 5 V with them re-sent.  A weight cell is driven at the
                    swing its ring needs at that rate where that is more.
  time              PREDICTED by pta_dispatch.py.
  the ring's rows   DERIVED as above, from pta_ring.py's published figures and
                    its assumptions: 7.0 pm/V carried from one racetrack to
                    any ring, and the smallest ring it read, 30 um.

There is no chiplet, and none of this is a measurement of one.

WHAT IT DOES NOT DO.  It does not weigh the rows against each other, and it
does not choose.  It says what each rate asks for and what it buys.

THE CHOICE WAS MADE ON 2026-10-05: 1 GS/s is the working shot rate, the plan's
B11.  This file is what it was made from, and is kept as it was run.

Standard library only.  Run:  python3 docs/designs/pta_rate.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_adc_survey as adc
import pta_chiplet_link as link
import pta_dispatch as dispatch
import pta_floorplan as fp
import pta_geometry as geometry
import pta_power as power
import pta_ring as ring
import pta_shot_rate as shot
import pta_tw_sweep as sweep

TILE = shot.X2_TILE                        # 256 x 64: the plan's B10
K, N = TILE
V1 = shot.REQ["v1"]
BITS = V1["adc_bits"]
BATCH = link.BIG["mb"]
WIDE = (link.BIG["kin"], link.BIG["nout"])  # the 4096-square layer every model sizes on
LOSS_DB = shot.LOSS_DB                     # B5's range, laser to detector
SWINGS_V = (2.0, 5.0)                      # pta_power.py's two ends, as pta_geometry.py takes them

WORKING = shot.X2_FS                       # 1 GS/s: X2's planning figure
RATES = (0.25e9, 0.5e9, 1e9, 2e9, 4e9)     # the candidates: two octaves either side of it


# ---- the link ------------------------------------------------------------------
def modules(fs, resident=False):
    """Modules that carry the wide layer at this rate.  Resident means the layer's
    weights are held on the interface chip and only activations and results cross."""
    return math.ceil(fs / shot.feed_rate(mods=1, resident=resident, k=K, n=N, **link.BIG))


# ---- the converters ------------------------------------------------------------
def adc_j(fs):
    """(best, fifth-best) joules a sample among the published parts that can do it."""
    return adc.price_j(BITS, fs)


def adc_parts(fs):
    return len(adc.able(BITS, fs))


def adc_best_hz():
    """The rate of the part that sets the best price: past it the price moves."""
    return adc.able(BITS, WORKING)[0][adc.FS] * 1e9


# ---- the receiver, and the laser -------------------------------------------------
def receiver_carries_hz():
    """The fastest tile the measured amplifier's own bandwidth settles."""
    return power.TIA_BW_HZ / power.settle_bandwidth_hz(1.0, BITS)


def laser_w(fs):
    """(low, high) watts.  Low: the measured receiver's noise behind 10 dB.  High:
    behind 20 dB, and past the rate that receiver settles, under the harsher of
    pta_shot_rate.py's two laws from there on."""
    na = power.receiver_noise_a(fs, BITS)
    lo, hi = (power.laser_w(V1, db, na, N) for db in LOSS_DB)
    past = fs / receiver_carries_hz()
    if past > 1:
        hi *= past ** (shot.NOISE_LAWS[1][1] - shot.NOISE_LAWS[0][1])
    return lo, hi


# ---- the weight ring: the fourth bound -------------------------------------------
def ring_settle_s(q, bits=BITS):
    """How long a level takes to settle through a ring of this Q.  The field is a
    single pole at half the linewidth; the power is its square, hence one bit more."""
    return sweep.settle_s(ring.linewidth_hz(q) / 2, bits + 1)


def ring_rate(q, bits=BITS):
    """Shots a second if the ring's settle were the whole shot: an upper bound."""
    return 1.0 / ring_settle_s(q, bits)


def line_hz(fs, bits=BITS):
    """The narrowest line that settles within a shot at fs."""
    return 2 * sweep.settle_s(1.0, bits + 1) * fs


def q_ceiling(fs):
    """The highest Q a weight ring may have and still pass this rate."""
    return ring.linewidth_hz(1.0) / line_hz(fs)


def swing_floor_v(fs):
    """And so the least swing that covers a weight's range: a linewidth, at 7.0 pm/V."""
    return ring.swing_v(q_ceiling(fs))


def lines(fs, fsr_nm=ring.K30["fsr_nm"], chi=ring.XTALK["v1"]):
    """Lines a bus at the narrowest line the rate allows, which is the most there can be."""
    return ring.lines_a_bus(fsr_nm, q_ceiling(fs), chi)


def buses(fs, **kw):
    return -(-K // lines(fs, **kw))


def lines_times_rate(fsr_nm, chi=ring.XTALK["v1"], bits=BITS):
    """Lines a bus times shots a second: the ring's constant, in lines Hz."""
    fsr_hz = ring.linewidth_hz(1.0) * fsr_nm * 1e3 / ring.LAMBDA_PM
    return fsr_hz / (ring.spacing(chi) * line_hz(1.0, bits))


# ---- the watts -------------------------------------------------------------------
def parts(fs):
    """pta_power.py's five parts at this rate, each (low end, high end)."""
    _, lo = power.chip_w(K, N, BITS, SWINGS_V[0], True, fs)
    _, hi = power.chip_w(K, N, BITS, SWINGS_V[1], False, fs)
    p = {name: (lo[name][0], hi[name][1]) for name in lo}
    floor = swing_floor_v(fs)
    p["weight drive"] = tuple(
        power.weight_drive_w(K, N, max(v, floor), c, a, fs)
        for v, c, a in zip(SWINGS_V, power.C_CELL_F, power.ACTIVITY))
    return p


def chip_w(fs):
    p = parts(fs)
    return sum(x[0] for x in p.values()), sum(x[1] for x in p.values())


def total_w(fs):
    c, l = chip_w(fs), laser_w(fs)
    return c[0] + l[0], c[1] + l[1]


def fj_mac(fs):
    """Energy a MAC with every cell in use: the watts over k * n * the rate."""
    lo, hi = total_w(fs)
    return lo / power.mac_s(K, N, fs) * 1e15, hi / power.mac_s(K, N, fs) * 1e15


# ---- the time ----------------------------------------------------------------------
def layer(fs, m, kin, nout, mods=None):
    """pta_dispatch.py's layer at this rate: on the rate's own modules unless told,
    and the write path section 4.3 holds a two-bank tile to."""
    mods = modules(fs) if mods is None else mods
    return dispatch.pta(m, kin, nout, tile=TILE, fs=fs, mods=mods,
                        per_beat=shot.per_beat_two_banks(K, N, BATCH))


def d3_s(fs):
    return sum(layer(fs, BATCH, a, b)["seconds"] for a, b in dispatch.D3)


# ---- printing ------------------------------------------------------------------------
def gs(fs):
    return f"{fs / 1e9:g} GS/s"


def span(lo, hi, digits=2):
    return f"{lo:.{digits}f}-{hi:.{digits}f}"


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    print("Open question 1, its second half: the shot rate, at the working geometry.")
    print(f"{K} x {N}, a batch of {BATCH}, version 1 of section 4.3.  {gs(WORKING)} is X2's planning figure.")

    section("1. What a rate asks of the link, the converters and the receiver")
    print(f"  {'rate':<11}{'modules':>9}{'weights held':>14}{'a conversion':>16}{'parts':>7}"
          f"{'converted':>12}{'receiver needs':>16}")
    for fs in RATES:
        lo, hi = adc_j(fs)
        print(f"  {gs(fs):<11}{modules(fs):>9}{modules(fs, True):>14}{span(lo * 1e12, hi * 1e12) + ' pJ':>16}"
              f"{adc_parts(fs):>7}{N * fs / 1e9:>7.0f} GS/s{power.settle_bandwidth_hz(fs, BITS) / 1e9:>12.2f} GHz")
    print(f"  Modules are for the {WIDE[0]}-square layer with its weights re-sent every batch, and")
    print("  with them held on the interface chip.  A conversion is the best and the fifth-best")
    print(f"  published part of {BITS} effective bits at that rate; parts is how many there are.")
    print(f"  The measured receiver is {power.TIA_BW_HZ / 1e9:.1f} GHz wide, which settles a tile up to"
          f" {receiver_carries_hz() / 1e9:.2f} GS/s.")
    print(f"  B8 reopens at {shot.B8_FLIP_MODULES} modules:"
          f" {shot.feed_rate(mods=shot.B8_FLIP_MODULES, resident=False, k=K, n=N, **link.BIG) / 1e9:.2f} GS/s.")

    section("2. What it draws")
    print(f"  {'rate':<11}{'receivers':>11}{'converters':>13}{'drive':>13}{'link':>13}{'chip':>13}"
          f"{'laser':>13}{'a MAC':>13}")
    for fs in RATES:
        p = parts(fs)
        drive = tuple(p["input drive"][i] + p["weight drive"][i] for i in (0, 1))
        print(f"  {gs(fs):<11}{p['receivers'][0]:>9.2f} W{span(*p['ADCs']) + ' W':>13}{span(*drive) + ' W':>13}"
              f"{span(*p['link']) + ' W':>13}{span(*chip_w(fs)) + ' W':>13}{span(*laser_w(fs)) + ' W':>13}"
              f"{span(*fj_mac(fs), digits=0) + ' fJ':>13}")
    print("  Low end: 2 V, the weights held on the interface chip, 10 dB of loss.  High end: 5 V,")
    print("  the weights re-sent, 20 dB.  The receivers are the one measured amplifier at every")
    print("  rate.  A MAC is the chip and the laser over every cell of the tile, each shot.")

    section("3. What it buys")
    gpu = dispatch.gpu_s(BATCH, *WIDE, launch=False)
    print(f"  {'rate':<11}{'the wide layer':>16}{'on one module':>15}{'times the GPU':>16}{'D3':>10}")
    for fs in RATES:
        big, one = layer(fs, BATCH, *WIDE), layer(fs, BATCH, *WIDE, mods=1)
        print(f"  {gs(fs):<11}{big['seconds'] * 1e6:>13.1f} us{one['seconds'] * 1e6:>12.1f} us"
              f"{gpu / big['seconds']:>16,.0f}{d3_s(fs) * 1e6:>7.2f} us")
    print(f"  The wide layer is {WIDE[0]} square at a batch of {BATCH}, on that rate's own modules and then")
    print(f"  on one.  The GPU's arithmetic on it is {gpu * 1e3:.0f} ms (pta_dispatch.py, an indication).")
    print("  D3 is its two layers' arithmetic, with no command: a command is 14.7 us or more.")

    section("4. The weight ring: a fourth bound")
    print("  What a ring carries, by its Q:")
    rows = [("K's 30 um ring, as built", ring.K30["q"]),
            ("W's racetrack, where 7.0 pm/V was measured", ring.W_Q)]
    rows += [(f"a ring that swings on {v:.0f} V", ring.q_for_swing(v)) for v in fp.SWINGS_V[::-1]]
    rows += [("K's 30 um ring, at its bend loss's Q", ring.q_from_loss(ring.K30["bend_db_cm"]) / 2),
             ("Z's 80 um ring", ring.Z80["q"])]
    for label, q in rows:
        r = ring_rate(q)
        rate = f"{r / 1e9:.2f} GS/s" if r >= 1e8 else f"{r / 1e6:.1f} MS/s"
        print(f"    {label:<44} Q {q:>11,.0f}   a line {ring.linewidth_hz(q) / 1e9:>7.3f} GHz wide"
              f"   {rate:>11}")
    print(f"  The modulator allows {shot.modulator_rate(sweep.TFLN_BW_HZ, BITS) / 1e9:.0f} GS/s (pta_shot_rate.py).")
    print()
    print("  And what a rate asks of the ring:")
    print(f"  {'rate':<11}{'a line at least':>17}{'Q at most':>11}{'swing at least':>16}{'an LSB':>10}"
          f"{'0.1 pm is':>11}{'lines a bus':>13}{'buses':>7}")
    for fs in RATES:
        q = q_ceiling(fs)
        print(f"  {gs(fs):<11}{line_hz(fs) / 1e9:>13.2f} GHz{q:>11,.0f}{swing_floor_v(fs):>14.2f} V"
              f"{ring.lsb_pm(q):>7.3f} pm{ring.S_SHIFT_PM / ring.lsb_pm(q):>7.2f} LSB{lines(fs):>13}{buses(fs):>7}")
    c30, c70 = lines_times_rate(ring.K30["fsr_nm"]), lines_times_rate(ring.K70["fsr_nm"])
    print(f"  Lines a bus times the rate is the ring's constant: {c30 / 1e9:.0f} lines GS/s on the 30 um ring")
    print(f"  at v1's crosstalk, and {c70 / 1e9:.0f} on the 70 um one.  One bus carries all {K} at"
          f" {c30 / K / 1e9:.2f} GS/s and under.")
    print("  An LSB is a 6-bit weight's, as a shift of the resonance; 0.1 pm is the one stability")
    print("  figure pta_ring.py read, over 25 minutes.  Lines and buses are at v1's 2% crosstalk.")

    section("5. The weight path and the source")
    print(f"  {'rate':<11}{'cells written':>15}{'each weight DAC':>17}{'source noise at most':>22}")
    for fs in RATES:
        print(f"  {gs(fs):<11}{shot.weight_updates_per_s(K, N, BATCH, fs) / 1e9:>11.0f} G/s"
              f"{fs / BATCH / 1e6:>13.1f} MHz{shot.rin_limit_db_hz(V1, fs):>16.1f} dB/Hz")
    print(f"  Two banks, {shot.per_beat_two_banks(K, N, BATCH)} cells a beat at every rate: the beat shortens and the count does not.")

    findings()
    checks()


def findings():
    v2, v5 = ring.q_for_swing(2.0), ring.q_for_swing(5.0)
    c30 = lines_times_rate(ring.K30["fsr_nm"])
    print()
    print("What this says, six readings.")
    print()
    print("  1. THE WEIGHT RING BOUNDS THE RATE, AND IT BINDS AT THE PLANNING FIGURE ITSELF.")
    print(f"     The modulator allows {shot.modulator_rate(sweep.TFLN_BW_HZ, BITS) / 1e9:.0f} GS/s and was never the question.  A ring that swings on")
    print(f"     2 V allows {ring_rate(v2) / 1e9:.2f}, and one on 5 V {ring_rate(v5) / 1e9:.2f}.  {gs(WORKING)} needs a line"
          f" {line_hz(WORKING) / 1e9:.2f} GHz wide:")
    print(f"     a Q of {round(q_ceiling(WORKING), -3):,.0f} at most and a swing of {swing_floor_v(WORKING):.1f} V at least.  Of the floorplan's")
    print("     three swings, a ring carries the planning figure only at 5 V.")
    print()
    print("  2. RATE, SWING, STABILITY AND BUSES ARE ONE NUMBER, THE RING'S Q.  A faster tile needs")
    print("     a wider line: more volts to cross it and fewer lines a bus, and a resonance that")
    print("     may wander further before it costs a weight an LSB.  So the faster tile is the")
    print("     harder to drive and to light, and the easier to hold still.")
    print(f"     At {gs(WORKING)}: {swing_floor_v(WORKING):.1f} V, {lines(WORKING)} lines a bus, {buses(WORKING)} buses,"
          f" and 0.1 pm is {ring.S_SHIFT_PM / ring.lsb_pm(q_ceiling(WORKING)):.2f} LSB.")
    print(f"     At {gs(RATES[1])}: {swing_floor_v(RATES[1]):.1f} V, {buses(RATES[1])} buses, and the same 0.1 pm is a whole LSB.")
    print(f"     At {gs(RATES[3])}: {swing_floor_v(RATES[3]):.1f} V and {buses(RATES[3])} buses.  One bus carries {K} inputs under"
          f" {c30 / K / 1e9:.2f} GS/s.")
    print()
    print(f"  3. THE MEASURED PARTS REACH {receiver_carries_hz() / 1e9:.1f} AND {adc_best_hz() / 1e9:.1f} GS/s, AND NO FURTHER.  A conversion costs")
    print(f"     {adc_j(RATES[0])[0] * 1e12:.2f} pJ at best up to {adc_best_hz() / 1e9:.1f} GS/s, the rate of the part that sets it, and"
          f" {adc_j(RATES[4])[0] * 1e12:.2f} to {adc_j(RATES[4])[1] * 1e12:.0f}")
    print(f"     at {gs(RATES[4])}.  The one receiver this program has a measurement of settles a tile up")
    print(f"     to {receiver_carries_hz() / 1e9:.1f} GS/s.  So {gs(WORKING)} is the fastest candidate both stand behind: {gs(RATES[3])}")
    print(f"     is past the receiver, and {gs(RATES[4])} past both.")
    print()
    e = [fj_mac(fs) for fs in RATES]
    print("  4. A MAC GETS CHEAPER WITH THE RATE, AND MOST OF THAT IS HAD BY 1 GS/s.  The receivers")
    print("     draw the same at every rate and the laser grows as its square root, so more")
    print("     shots spread them.  At the low end of the power range a MAC is"
          f" {', '.join(f'{x[0]:.0f}' for x in e)} fJ")
    print(f"     across the five rates, and at the high end {', '.join(f'{x[1]:.0f}' for x in e)}.  From {gs(WORKING)} to"
          f" {gs(RATES[3])}")
    print(f"     that is {1 - e[3][0] / e[2][0]:.0%} less at one end and {1 - e[3][1] / e[2][1]:.0%} at the other, and at {gs(RATES[4])} the high")
    print("     end is dearer than at any rate but the slowest: the converters and the swing.")
    print()
    gpu = dispatch.gpu_s(BATCH, *WIDE, launch=False)
    slow = layer(RATES[0], BATCH, *WIDE)["seconds"]
    one = layer(WORKING, BATCH, *WIDE, mods=1)["seconds"]
    print("  5. THE RATE BUYS TIME ONLY WITH ITS MODULES, AND TIME IS NOT WHAT IS SHORT.  Weights")
    print(f"     re-sent, the wide layer needs {', '.join(str(modules(fs)) for fs in RATES)} modules; on one it takes")
    print(f"     {one * 1e6:.0f} us at every rate.  Held on the interface chip, one module does to"
          f" {shot.feed_rate(mods=1, resident=True, k=K, n=N, **link.BIG) / 1e9:.1f} GS/s.")
    print(f"     And at {gs(RATES[0])} the tile is still {gpu / slow:,.0f} times the GPU's arithmetic on that layer,")
    print("     by pta_dispatch.py's indication of the GPU.")
    print("     D3 is the command's at every rate.  The rate decides what the speed costs.")
    print()
    print(f"  6. SO {gs(WORKING)} IS THE FASTEST RATE EVERY ROW HERE CAN STAND BEHIND, and it is not free.")
    print(f"     It fixes the weight ring: a Q under {round(q_ceiling(WORKING), -3):,.0f}, a swing over {swing_floor_v(WORKING):.1f} V, and"
          f" {buses(WORKING)} buses of")
    print(f"     {lines(WORKING)} lines or fewer on the smallest ring read.  {gs(RATES[1])} takes {modules(RATES[1])} modules for {modules(WORKING)} and lets")
    print(f"     a ring swing on {swing_floor_v(RATES[1]):.1f} V, for twice the time and a ring twice as hard to hold.")
    print(f"     {gs(RATES[3])} halves the time, for twice the modules, {buses(RATES[3])} buses and a receiver nobody")
    print("     has measured.  Which of those the board wants is the program's to say.")


def checks():
    """Every claim above, as an assert."""
    assert TILE == (256, 64) and BITS == 7 and BATCH == 64 and WORKING in RATES

    # 1. At the working rate every row is the model's own, as pta_geometry.py has it.
    assert modules(WORKING) == geometry.modules(K, N) == shot.X2_MODULES
    assert modules(WORKING, True) == geometry.modules_resident(K, N) == 1
    for mine, theirs in ((chip_w(WORKING), geometry.chip_w(K, N)), (laser_w(WORKING), geometry.laser_w(K, N)),
                         (fj_mac(WORKING), geometry.fj_mac(K, N))):
        assert all(abs(a / b - 1) < 0.002 for a, b in zip(mine, theirs)), (mine, theirs)
    big = layer(WORKING, BATCH, *WIDE)
    assert abs(big["seconds"] - geometry.layer(K, N, BATCH, *WIDE)["seconds"]) < 1e-12
    assert abs(d3_s(WORKING) - geometry.d3(K, N)[0]) < 1e-12
    #    The modules: 2, 3, 5, 10 and 19, and 1 with the weights held, 2 at the top.
    assert [modules(fs) for fs in RATES] == [2, 3, 5, 10, 19]
    assert [modules(fs, True) for fs in RATES] == [1, 1, 1, 1, 2]
    assert all(modules(fs) < shot.B8_FLIP_MODULES for fs in RATES)

    # 2. Reading 3.  The best conversion is 1.11 pJ through 2 GS/s and dearer
    #    past it; the measured receiver settles 1.70 GS/s, between two candidates.
    assert all(abs(adc_j(fs)[0] * 1e12 - 1.11) < 0.005 for fs in RATES[:4])
    assert adc_j(RATES[4])[0] > 1.4 * adc_j(RATES[3])[0] and adc_j(RATES[4])[1] > 2 * adc_j(RATES[3])[1]
    assert [adc_parts(fs) for fs in RATES] == sorted((adc_parts(fs) for fs in RATES), reverse=True)
    assert [adc_parts(fs) for fs in RATES] == [122, 98, 70, 45, 24]
    #    The part that sets the best price runs at 2.7 GS/s, and past it the price moves.
    assert abs(adc_best_hz() / 1e9 - 2.7) < 1e-9 and RATES[3] < adc_best_hz() < RATES[4]
    assert adc.price_j(BITS, adc_best_hz())[0] == adc_j(WORKING)[0] < adc.price_j(BITS, adc_best_hz() * 1.01)[0]
    assert abs(receiver_carries_hz() / 1e9 - 1.70) < 0.005
    assert WORKING < receiver_carries_hz() < RATES[3]
    #    The laser goes as the root of the rate under it, and faster over it.
    assert abs(laser_w(WORKING)[0] / laser_w(RATES[0])[0] - 2.0) < 1e-9
    assert abs(laser_w(WORKING)[1] / laser_w(RATES[0])[1] - 2.0) < 1e-9
    assert laser_w(RATES[3])[1] / laser_w(WORKING)[1] > math.sqrt(2) * 1.15

    # 3. Reading 1.  A ring's settle is the sweep's, at half its linewidth and one
    #    bit more; a 2 V ring falls short of 1 GS/s and a 5 V one clears it.
    q2, q5 = ring.q_for_swing(2.0), ring.q_for_swing(5.0)
    tau = 1.0 / (math.pi * ring.linewidth_hz(q2))
    assert abs(ring_settle_s(q2) / (tau * math.log(2 ** (BITS + 2))) - 1) < 1e-12
    assert abs(ring_rate(q2) / 1e9 - 0.88) < 0.005 and abs(ring_rate(q5) / 1e9 - 2.20) < 0.005
    assert ring_rate(ring.q_for_swing(1.0)) < ring_rate(q2) < WORKING < ring_rate(q5)
    assert ring_rate(q5) < shot.modulator_rate(sweep.TFLN_BW_HZ, BITS) / 20
    #    The published devices: W's racetrack carries 1.9 GS/s, and Z's 20 MS/s.
    assert abs(ring_rate(ring.W_Q) / 1e9 - 1.95) < 0.005 and ring_rate(ring.Z80["q"]) < 20e6
    #    The working rate: a line of 1.99 GHz, a Q of 97,000 and 2.3 V.
    assert abs(line_hz(WORKING) / 1e9 - 1.99) < 0.005 and abs(q_ceiling(WORKING) / 97_400 - 1) < 0.002
    assert abs(swing_floor_v(WORKING) - 2.27) < 0.005
    assert abs(ring_rate(q_ceiling(WORKING)) / WORKING - 1) < 1e-12
    assert [v for v in fp.SWINGS_V if ring_rate(ring.q_for_swing(v)) >= WORKING] == [5.0]

    # 4. Reading 2.  The trade, rate by rate, and the constant under it.
    assert [round(swing_floor_v(fs), 2) for fs in RATES] == [0.57, 1.14, 2.27, 4.55, 9.09]
    assert [lines(fs) for fs in RATES] == [409, 204, 102, 51, 25]
    assert [buses(fs) for fs in RATES] == [1, 2, 3, 6, 11]
    c30, c70 = lines_times_rate(ring.K30["fsr_nm"]), lines_times_rate(ring.K70["fsr_nm"])
    assert abs(c30 / 1e9 - 102.3) < 0.1 and abs(c70 / 1e9 - 44.9) < 0.1
    assert all(abs(lines(fs) - c30 / fs) < 1 for fs in RATES)
    assert abs(c30 / K / 1e9 - 0.40) < 0.005 and buses(c30 / K * 0.999) == 1 and buses(c30 / K * 1.02) == 2
    still = [ring.S_SHIFT_PM / ring.lsb_pm(q_ceiling(fs)) for fs in RATES]
    assert [round(s, 2) for s in still] == [2.09, 1.04, 0.52, 0.26, 0.13]
    assert swing_floor_v(RATES[3]) < max(fp.SWINGS_V) < swing_floor_v(RATES[4])

    # 5. Reading 4.  The low end falls at every step and flattens: 92, 57, 39, 30
    #    and 28 fJ.  The high end bottoms at 2 GS/s, and at 4 is dearer than at
    #    any rate but the slowest.
    e = [fj_mac(fs) for fs in RATES]
    assert [round(x[0]) for x in e] == [92, 57, 39, 30, 28]
    assert [round(x[1]) for x in e] == [617, 554, 518, 507, 591]
    assert 0.2 < 1 - e[3][0] / e[2][0] < 0.25 and 0.015 < 1 - e[3][1] / e[2][1] < 0.03
    assert e[0][1] > e[4][1] > e[1][1]
    #    The receivers are fixed, and the laser is the root of the rate.
    assert all(parts(fs)["receivers"] == parts(WORKING)["receivers"] for fs in RATES)

    # 6. Reading 5.  On one module the wide layer is the link's at every rate.
    one = [layer(fs, BATCH, *WIDE, mods=1) for fs in RATES]
    assert all(r["binds"] == "link" for r in one)
    assert all(abs(r["link_s"] - one[0]["link_s"]) < 1e-15 for r in one)
    assert all(layer(fs, BATCH, *WIDE)["binds"] == "tile" for fs in RATES)
    gpu = dispatch.gpu_s(BATCH, *WIDE, launch=False)
    assert gpu / layer(RATES[0], BATCH, *WIDE)["seconds"] > 1000
    assert all(d3_s(fs) < 14.7e-6 / 4 for fs in RATES)

    # 7. The plan's table, row by row, and the figures its readings quote.
    assert [round(adc_j(fs)[1] * 1e12, 2) for fs in RATES] == [2.42, 2.55, 3.14, 4.54, 10.96]
    assert abs(adc_j(RATES[4])[0] * 1e12 - 1.60) < 0.005
    assert [round(power.settle_bandwidth_hz(fs, BITS) / 1e9, 2) for fs in RATES] == [0.22, 0.44, 0.88, 1.77, 3.53]
    assert [round(chip_w(fs)[0], 2) for fs in RATES] == [0.33, 0.41, 0.55, 0.86, 1.68]
    assert [round(chip_w(fs)[1], 2) for fs in RATES] == [2.09, 3.92, 7.61, 15.14, 34.62]
    assert [round(laser_w(fs)[0], 2) for fs in RATES] == [0.04, 0.06, 0.09, 0.12, 0.18]
    assert [round(laser_w(fs)[1], 2) for fs in RATES] == [0.44, 0.62, 0.88, 1.46, 4.13]
    times = [layer(fs, BATCH, *WIDE)["seconds"] * 1e6 for fs in RATES]
    assert [round(t, 1) for t in times] == [262.5, 131.3, 65.7, 32.9, 16.5]
    assert abs(one[0]["seconds"] * 1e6 - 295.9) < 0.05
    assert [round(q_ceiling(fs), -3) for fs in RATES] == [390_000, 195_000, 97_000, 49_000, 24_000]
    assert [round(ring.lsb_pm(q_ceiling(fs)), 3) for fs in RATES] == [0.048, 0.096, 0.191, 0.383, 0.766]
    assert [round(shot.rin_limit_db_hz(V1, fs)) for fs in RATES] == [-138, -141, -144, -147, -150]
    assert [round(d3_s(fs) * 1e6, 1) for fs in RATES] == [3.0, 1.6, 1.0, 0.6, 0.4]
    #    Question 8's revision.  The ring that put 256 inputs on two buses carries
    #    0.71 GS/s, so at the planning rate they are three to six.
    qb = ring.q_from_loss(ring.K30["bend_db_cm"]) / 2
    assert abs(ring_rate(qb) / 1e9 - 0.71) < 0.005 and ring_rate(qb) < WORKING
    assert -(-K // ring.lines_a_bus(ring.K30["fsr_nm"], qb, ring.XTALK["v1"])) == 2
    assert buses(WORKING) == 3 and -(-K // ring.lines_a_bus(ring.K30["fsr_nm"], q5, ring.XTALK["v1"])) == 6
    assert ring_rate(q5) > WORKING

    # 8. B11's own figures: what the working rate costs over half of it.
    half = RATES[1]
    dc = [a - b for a, b in zip(chip_w(WORKING), chip_w(half))]
    dl = [a - b for a, b in zip(laser_w(WORKING), laser_w(half))]
    assert [round(x, 2) for x in dc] == [0.15, 3.69] and [round(x, 2) for x in dl] == [0.03, 0.26]
    assert (modules(WORKING), modules(half)) == (5, 3) and (buses(WORKING), buses(half)) == (3, 2)
    assert abs(swing_floor_v(WORKING) / swing_floor_v(half) - 2.0) < 1e-9
    assert [round(x) for x in fj_mac(half)] == [57, 554] and [round(x) for x in fj_mac(WORKING)] == [39, 518]
    assert abs(shot.weight_updates_per_s(K, N, BATCH, WORKING) - 256e9) < 1 and WORKING / BATCH == 15.625e6
    assert shot.per_beat_two_banks(K, N, BATCH) == 256 and N * WORKING == 64e9
    assert adc_parts(WORKING) == 70 and abs(adc_j(WORKING)[1] * 1e12 - 3.14) < 0.005

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
