"""
Open question 8 of board_program_plan.md: what kind of light source is it?

The question was asked on 2026-10-03 and has waited on the tile's topology ever
since: a column that sums powers wants a line an input, a column that sums
fields wants one line, and nothing said which.  Three things have been settled
or found since, and together they are enough to say what each answer asks for:

  B10               the tile is 256 x 64
  B11               it fires at 1 GS/s
  pta_ring.py       its cells are rings, and how many lines a bus of rings holds
  pta_rate.py       a ring passes a level no faster than its line is wide

This follows each reading of the topology to the source it needs.  It is not a
design, and it does not choose between them.

THE CHOICE WAS MADE ON 2026-10-05: a ring bank, the plan's B12, on four buses
as the working count.  This file is what it was made from, and is kept as it
was run.  B10 has since been revised to 128 inputs, where the same rule gives
two buses of the same 64 lines: pta_working_point.py.

PUBLISHED, and quoted as each has it:

  M   A. N. Tait, A. X. Wu, T. Ferreira de Lima, E. Zhou, B. J. Shastri,
      M. A. Nahmias and P. R. Prucnal, "Microring Weight Banks", IEEE J. Sel.
      Top. Quantum Electron. 22(6), 5900214 (2016).  The analysis of the
      topology the error model assumes.  A bank's channel count is its rings'
      finesse over the channel spacing in linewidths; "an allowed 3 dB power
      penalty resulted in a minimum channel spacing between 3.41 and 4.61
      linewidths", which "depends coherently on the optical path length of
      bus" waveguides; a finesse of 368 "could potentially support 108"
      channels and the 8-channel bank they measured, at 133, 39.  It weighs
      each channel between the two ports of a balanced photodetector.  And it
      assumes "that the spacing between the [lines] is significantly greater
      than the bandwidth of" the signal: "in other words, coherent beat noise
      would corrupt the function of weighting".
  E   M. Zhang, B. Buscaino, C. Wang et al., "Broadband electro-optic frequency
      comb generation in an integrated microring resonator", arXiv:1809.08636
      (Nature 568, 2019).  Thin-film lithium niobate.  "Over 900 comb lines
      spaced at ~ 10 GHz", with the spacing "finely controllable over seven
      orders of magnitude (10 Hz to 100 MHz)" of detuning.
  H   Y. Hu, M. Yu, B. Buscaino et al., "High-efficiency and broadband
      electro-optic frequency combs enabled by coupled micro-resonators",
      arXiv:2111.14743.  A conversion efficiency of 30% over 132 nm, "100-times
      higher" than the integrated electro-optic combs before it.
  L   J. Zhang, C. Wang, C. Denney et al., "Ultrabroadband integrated
      electro-optic frequency comb in lithium tantalate", Nature 637, 1096
      (2025).  THIN-FILM LITHIUM TANTALATE, the plan's platform.  "Over 450 nm
      (more than 60 THz) with more than 2,000 lines", driven by a hybrid
      integrated laser diode, in 1 cm2.
  G   ITU-T G.694.1, the telecom grid: channels 12.5, 25, 50 or 100 GHz apart.

  And from pta_floorplan.py: the sources of many lines in the co-packaged
  optics announcements the plan read, 8 or 16 lines each.

DERIVED here:

  a line's spacing    the bank's spacing in linewidths times the ring's line,
                      which B11 holds to 1.99 GHz or more
  lines a bus         the ring's free spectral range over that spacing, as
                      pta_ring.py has it.  M's own formula
  a grid for b buses  256 inputs on b buses are ceil(256 / b) lines each, so
                      the lines are no further apart than the range over that
  a beat              two lines on one photodiode beat at their spacing.  A
                      receiver that averages over one shot passes
                      |sin(pi x) / (pi x)| of it, x the spacing over the shot
                      rate: nothing when the spacing is a whole multiple
  a ring's phase      a ring that passes a fraction w of a line's power turns
                      its field by atan(sqrt(1/w - 1)): 45 degrees at a half
  a weight's LSB      pta_ring.py's, as a frequency: how far a line may sit
                      from where its ring was calibrated

ASSUMED, and marked again where each is used:

  - pta_ring.py's ring: the smallest it read, 30 um, and 7.0 pm/V carried to it
  - that a receiver can be made to average over exactly one shot.  The power
    model's is a single pole, and section 4 says what that one passes
  - that M's spacing, found for silicon rings read by a balanced pair, carries
    to these.  It is a property of Lorentzian lines on a shared bus

NOT PRICED: an amplifier, and its noise; whatever picks one line off a comb
for one input; what puts a ring on its line in the first place; how this tile
signs a weight, which M does with a balanced pair and which would double every
photodiode count here (the plan's B13 has since chosen the pair: 512 at four
buses); and the receiver's noise with more than one photodiode on its input.

Standard library only.  Run:  python3 docs/designs/pta_source.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_floorplan as fp
import pta_power as power
import pta_rate as rate
import pta_ring as ring
import pta_shot_rate as shot

# ---- the program's own -------------------------------------------------------
TILE = shot.X2_TILE                        # 256 x 64: B10
K, N = TILE
FS = shot.X2_FS                            # 1 GS/s: B11
BITS = rate.BITS
RING = ring.K30                            # ASSUMED: the smallest ring pta_ring.py read
F_OPT_HZ = ring.linewidth_hz(1.0)          # 193 THz: the carrier, at 1550 nm
FSR_HZ = F_OPT_HZ * RING["fsr_nm"] * 1e3 / ring.LAMBDA_PM
V1_SPACING = ring.spacing(ring.XTALK["v1"])   # 3.5 linewidths: v1's 2% crosstalk

# ---- published ---------------------------------------------------------------
M_SPACING = (3.41, 4.61)                   # M: linewidths, at a 3 dB cross-weight penalty
M_CHANNELS = {368: 108, 133: 39}           # M: finesse, and the channels it gives for it
E_SPACING_HZ, E_LINES, E_DETUNE_HZ = 10e9, 900, 100e6
H_EFFICIENCY, H_SPAN_NM, H_GAIN = 0.30, 132, 100
L_SPAN_HZ, L_LINES = 60e12, 2000
G_GRID_HZ = (12.5e9, 25e9, 50e9, 100e9)
FEW_LINES = fp.LINES_A_FIBER               # 8 and 16: the announcements' sources

BUSES = (3, 4, 5, 6)                       # pta_rate.py: three to six at the working rate


# ---- 1. a bank at the working point --------------------------------------------
def line_floor_hz():
    """The narrowest line B11's rate passes: pta_rate.py's."""
    return rate.line_hz(FS)


def lines_at(spacing_hz):
    """Lines one free spectral range holds, that far apart."""
    return int(FSR_HZ / spacing_hz)


def buses_for(lines):
    return -(-K // lines)


def lines_needed(buses):
    return -(-K // buses)


def spacing_ceiling_hz(buses):
    """The widest grid that still puts a bus's share of the inputs in one range."""
    return FSR_HZ / lines_needed(buses)


def clocked_hz(buses):
    """The widest grid under that ceiling that is a whole multiple of the shot rate."""
    return math.floor(spacing_ceiling_hz(buses) / FS) * FS


def line_window_hz(buses, spacing=V1_SPACING):
    """(narrowest, widest) line a ring may have on that grid, or None: no narrower
    than the rate passes, no wider than the grid's spacing allows."""
    lo, hi = line_floor_hz(), clocked_hz(buses) / spacing
    return (lo, hi) if hi >= lo else None


def q_of(line_hz):
    return F_OPT_HZ / line_hz


def lsb_hz(line_hz):
    """A 6-bit weight's LSB as a line's distance from its ring: pta_ring.py's, in hertz."""
    return line_hz / (ring.SLOPE * 2 ** ring.W_BITS)


# ---- 2. the light a line carries -------------------------------------------------
def laser_w():
    """The whole tile's light at the working rate: pta_rate.py's (low, high)."""
    return rate.laser_w(FS)


def line_w(buses):
    """(low, high) watts a line, each line lighting one input on every bus."""
    lo, hi = laser_w()
    return lo / lines_needed(buses), hi / lines_needed(buses)


# ---- 4. the beat -------------------------------------------------------------------
def single_pole_pass(spacing_hz, fs=FS):
    """What the power model's receiver, a single pole just fast enough to settle,
    passes of a beat at the line spacing."""
    return 1.0 / math.sqrt(1.0 + (spacing_hz / power.settle_bandwidth_hz(fs, BITS)) ** 2)


def averaged_pass(spacing_hz, fs=FS):
    """What a receiver that averages over exactly one shot passes of it."""
    x = math.pi * spacing_hz / fs
    return abs(math.sin(x) / x)


HALF_LSB = 2.0 ** -(BITS + 1)              # of the converter's full scale


# ---- 5. the other readings ---------------------------------------------------------
def ring_phase_deg(w):
    """The angle a ring turns a line's field by when it passes a fraction w of its power."""
    return math.degrees(math.atan(math.sqrt(1.0 / w - 1.0)))


def coherent_sum(weights):
    """The power a column reads if it adds, as fields, equal inputs weighed by rings."""
    re = sum(math.sqrt(w) * math.cos(math.radians(ring_phase_deg(w))) for w in weights)
    im = sum(math.sqrt(w) * math.sin(math.radians(ring_phase_deg(w))) for w in weights)
    return re * re + im * im


def photodiodes(buses):
    return N * buses


# ---- printing ------------------------------------------------------------------------
def ghz(hz, digits=2):
    return f"{hz / 1e9:.{digits}f} GHz"


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    print("Open question 8: what each reading of the tile asks of its light.")
    print(f"{K} x {N} (B10) at {FS / 1e9:g} GS/s (B11), on the ring pta_ring.py read: {RING['radius_um']} um,"
          f" a range of {ghz(FSR_HZ, 0)}.")

    section("1. A ring bank at the working point")
    floor = line_floor_hz()
    print(f"  B11's rate passes a line {ghz(floor)} wide or wider: a Q of {q_of(floor):,.0f} or less.")
    print("  How far apart two lines sit, in linewidths, and what one bus then holds:")
    print(f"  {'spacing':<44}{'linewidths':>11}{'at the narrowest line':>23}{'lines a bus':>13}{'buses':>7}")
    for label, s in (("M, the kindest bus length", M_SPACING[0]), ("the error model's v1 crosstalk, 2%", V1_SPACING),
                     ("M, the worst bus length", M_SPACING[1])):
        print(f"  {label:<44}{s:>11.2f}{ghz(s * floor):>23}{lines_at(s * floor):>13}{buses_for(lines_at(s * floor)):>7}")
    print(f"  M's own check: a finesse of 368 over {M_SPACING[0]} is {368 / M_SPACING[0]:.0f} channels, and it says 108;"
          f" 133 is {133 / M_SPACING[0]:.0f}, and it says 39.")
    print(f"  Lines sit {M_SPACING[0] * floor / FS:.1f} to {M_SPACING[1] * floor / FS:.1f} shot rates apart at the least, whatever the rate:")
    print("  the narrowest line is a fixed multiple of the rate, and the spacing of that.")

    print()
    print(f"  And {K} inputs on b buses, the grid a whole multiple of the shot rate:")
    print(f"  {'buses':<7}{'lines each':>11}{'grid at most':>14}{'clocked':>10}{'ring Q, at v1':>20}{'its swing':>16}"
          f"{'at the worst':>16}{'photodiodes':>13}")
    for b in BUSES:
        w, worst = line_window_hz(b), line_window_hz(b, M_SPACING[1])
        qs = f"{q_of(w[1]):,.0f}-{q_of(w[0]):,.0f}"
        vs = f"{ring.swing_v(q_of(w[0])):.2f}-{ring.swing_v(q_of(w[1])):.2f} V"
        ws = "none" if worst is None else f"{q_of(worst[1]):,.0f} and up"
        print(f"  {b:<7}{lines_needed(b):>11}{ghz(spacing_ceiling_hz(b)):>14}{ghz(clocked_hz(b), 0):>10}{qs:>20}{vs:>16}"
              f"{ws:>16}{photodiodes(b):>13}")
    print("  The Q's upper end is the rate's; its lower end is the grid's.  'At the worst' is the")
    print("  same window at M's widest spacing.  Photodiodes are one a bus a column: section 4.")

    section("2. What that asks of a source")
    lo, hi = laser_w()
    print(f"  The tile's light is {lo:.2f} to {hi:.2f} W (pta_rate.py), which is {lo / K * 1e3:.2f} to {hi / K * 1e3:.1f} mW an input.")
    print(f"  {'buses':<7}{'lines':>7}{'on a grid of':>14}{'within':>10}{'each line':>17}{'a weight LSB is':>18}")
    for b in BUSES:
        w = line_window_hz(b)
        l = line_w(b)
        print(f"  {b:<7}{lines_needed(b):>7}{ghz(clocked_hz(b), 0):>14}{RING['fsr_nm']:>7} nm"
              f"{f'{l[0] * 1e3:.1f}-{l[1] * 1e3:.1f} mW':>17}{f'{lsb_hz(w[0]) / 1e6:.0f}-{lsb_hz(w[1]) / 1e6:.0f} MHz':>18}")
    print("  Within: a ring answers again one range on, so a source's lines outside one range")
    print("  land on rings that already have a line.  A weight's LSB is how far a line may sit")
    print("  from where its ring was calibrated.  A comb's lines move together, on two numbers:")
    print("  where its pump sits and how far apart they are.  An array's move one by one.")

    section("3. Sources, against that")
    print(f"  {'source':<52}{'lines apart':>12}{'in one range':>14}{'buses':>7}{'photodiodes':>13}")
    rows = [("E: an electro-optic comb on lithium niobate", E_SPACING_HZ),
            ("L: one on lithium tantalate, 60 THz over 2,000 lines", L_SPAN_HZ / L_LINES)]
    rows += [(f"G: the telecom grid at {g / 1e9:g} GHz", g) for g in G_GRID_HZ]
    for label, s in rows:
        n = lines_at(s)
        print(f"  {label:<52}{ghz(s, 1):>12}{n:>14}{buses_for(n):>7}{photodiodes(buses_for(n)):>13}")
    for n in FEW_LINES[::-1]:
        print(f"  {f'an announced source of {n} lines':<52}{'':>12}{n:>14}{buses_for(n):>7}{photodiodes(buses_for(n)):>13}")
    print(f"  E's spacing is its microwave drive, and it follows a drive up to {E_DETUNE_HZ / 1e6:.0f} MHz off its own")
    print(f"  resonator.  H's comb turns {H_EFFICIENCY:.0%} of its pump into lines, {H_GAIN} times the ones before it,")
    print(f"  over {H_SPAN_NM} nm, of which this tile can use {RING['fsr_nm']}.  None of them says what a line carries.")
    print(f"  ({lines_at(G_GRID_HZ[0])} lines at {G_GRID_HZ[0] / 1e9:g} GHz is the patent application's \"k <~ {fp.LINES_A_BUS}\" again: on this ring the")
    print("  finest telecom grid and that limit are one number.  Nothing here says theirs is from it.)")

    section("4. Two things a ring bank needs that the plan does not have")
    print("  A BEAT.  Two lines on one photodiode beat at their spacing.  M assumes the spacing")
    print("  significantly greater than the signal's bandwidth.  Here it is 8 to 16 shot rates:")
    print("  greater, and not by enough for the power model's receiver, which is a single pole.")
    print(f"  Half an LSB is {HALF_LSB:.2%} of full scale.")
    print(f"  {'grid':<10}{'a single pole passes':>22}{'averaged over a shot':>22}{'and 50 MHz off the clock':>26}")
    for b in BUSES:
        s = clocked_hz(b)
        print(f"  {ghz(s, 0):<10}{single_pole_pass(s):>22.1%}{averaged_pass(s):>22.1%}{averaged_pass(s + 50e6):>26.2%}")
    print("  How large the beat is to begin with turns on the lines' phases, and is not priced.")
    print("  Two ways out.  A receiver steeper than one pole, which nothing here has priced.  Or a")
    print("  grid locked to the shot clock and a receiver that averages over the shot: locked, and")
    print("  not merely near, by the last column.")
    print("  A PHOTODIODE A BUS.  Buses reuse lines, and a reused line is the same light: brought")
    print("  together in one waveguide two buses' shares add as fields, not powers.  So a column")
    print(f"  reads each bus on its own photodiode: {photodiodes(BUSES[0])} to {photodiodes(BUSES[-1])}, where every model here has {N}.")

    section("5. The other two readings: one line")
    print("  FIELDS.  A column that adds fields from one laser adds them through rings, and a")
    print("  ring's weight comes with an angle:")
    for w in (1.0, 0.5, 0.25, 0.1):
        print(f"    a weight of {w:<5g} turns the field {ring_phase_deg(w):5.1f} degrees")
    pair = (1.0, 0.5)
    print(f"  Two equal inputs at weights {pair[0]:g} and {pair[1]:g} read {coherent_sum(pair):.2f}.  Their powers add to"
          f" {sum(pair):.2f}, and")
    print(f"  their fields with no angle to {sum(math.sqrt(w) for w in pair) ** 2:.2f}.  It is neither, before any path's own phase.")
    print("  Undoing the angle is a second element a cell, and a phase shifter that is not")
    print("  resonant is millimeters long (pta_floorplan.py).")
    print("  A PHOTODIODE A CELL.  One laser, no lines to tell apart and no bus to fill: every")
    print(f"  cell's light ends on its own photodiode and a column adds currents.  {K * N:,} photodiodes,")
    print(f"  {K} on each receiver's input, on a platform with no detector of its own.")

    findings()
    checks()


def findings():
    floor = line_floor_hz()
    lo, hi = laser_w()
    print()
    print("What this says, seven readings.")
    print()
    print("  1. THE BANK'S LIMIT IS PUBLISHED, AND IT IS THE ONE DERIVED HERE.  M finds a bank holds")
    print(f"     its finesse over {M_SPACING[0]} to {M_SPACING[1]} linewidths, by bus length.  pta_ring.py's {V1_SPACING:.1f}, from the")
    print(f"     error model's crosstalk, is inside that and at its kind end.  At the other end a")
    print(f"     bus holds {lines_at(M_SPACING[1] * floor)} lines at the working rate where pta_rate.py has {rate.lines(FS)}, and {K} inputs")
    print(f"     are {buses_for(lines_at(M_SPACING[1] * floor))} buses and not {rate.buses(FS)}.")
    print()
    print("  2. A RING BANK HERE IS THREE BUSES AT THE LEAST, AND FOUR IS THE FIRST THAT HOLDS ACROSS")
    w3, w4 = line_window_hz(3), line_window_hz(4)
    print(f"     THE PUBLISHED RANGE.  Three need {lines_needed(3)} lines {ghz(clocked_hz(3), 0)} apart and a ring Q between")
    print(f"     {round(q_of(w3[1]), -3):,.0f} and {round(q_of(w3[0]), -3):,.0f}, and at M's worst spacing no ring does both.  Four need {lines_needed(4)}")
    print(f"     lines {ghz(clocked_hz(4), 0)} apart and a Q from {round(q_of(w4[1]), -3):,.0f}: a swing of"
          f" {ring.swing_v(q_of(w4[0])):.1f} to {ring.swing_v(q_of(w4[1])):.1f} V.  More buses")
    print("     widen the window further, and cost a photodiode a column each.")
    print()
    print(f"  3. SO ITS SOURCE IS A COMB: {lines_needed(BUSES[-1])} TO {lines_needed(BUSES[0])} LINES, {clocked_hz(BUSES[0]) / 1e9:.0f} TO {clocked_hz(BUSES[-1]) / 1e9:.0f} GHz APART, IN {RING['fsr_nm']} nm.")
    print(f"     No telecom grid is that fine but the finest, which is {buses_for(lines_at(G_GRID_HZ[0]))} buses; at {G_GRID_HZ[2] / 1e9:.0f} GHz it is"
          f" {buses_for(lines_at(G_GRID_HZ[2]))}.")
    print(f"     The announced sources of {FEW_LINES[0]} and {FEW_LINES[1]} lines are {buses_for(FEW_LINES[0])} and {buses_for(FEW_LINES[1])} buses.  An electro-optic comb")
    print(f"     has been published at about {E_SPACING_HZ / 1e9:.0f} GHz, on lithium niobate: {lines_at(E_SPACING_HZ)} lines in this ring's")
    print(f"     range, {buses_for(lines_at(E_SPACING_HZ))} buses.  The one on the plan's own material is about {L_SPAN_HZ / L_LINES / 1e9:.0f} GHz: {buses_for(lines_at(L_SPAN_HZ / L_LINES))} buses.")
    print()
    print("  4. THE BEAT BETWEEN LINES IS NOT OUT OF BAND BY ITSELF, AND A COMB CAN PUT IT THERE.")
    print(f"     Lines on one photodiode beat at their spacing.  A single-pole receiver passes {single_pole_pass(clocked_hz(4)):.0%}")
    print(f"     of a beat {clocked_hz(4) / 1e9:.0f} GHz away, and half an LSB is {HALF_LSB:.1%}.  A receiver that averages over a")
    print("     shot passes none of it when the spacing is a whole number of shot rates.  An")
    print("     electro-optic comb's spacing is a microwave drive, so it can be locked to the shot")
    print("     clock.  An array's lines are each their own laser, and their beats wander.  The")
    print("     other way out is a steeper receiver, which nothing here has priced.")
    print()
    print("  5. BUSES THAT REUSE LINES CANNOT SHARE A PHOTODIODE.  The same line on two buses is the")
    print(f"     same light, and in one waveguide it adds as fields.  A column needs a photodiode a")
    print(f"     bus: {photodiodes(4)} at four buses, where the laser, the power and the area were sized on {N}.")
    print()
    l3, l6 = line_w(BUSES[0]), line_w(BUSES[-1])
    print(f"  6. A LINE CARRIES MILLIWATTS: {l3[0] * 1e3:.1f} TO {l3[1] * 1e3:.0f} mW AT THREE BUSES, {l6[0] * 1e3:.1f} TO {l6[1] * 1e3:.0f} AT SIX.  The total")
    print(f"     is the same {lo:.2f} to {hi:.2f} W however it is cut.  No comb read here states a line's")
    print("     power, and the best turns 30% of its pump into lines over 23 times the span the")
    print("     tile can use.  So a comb comes with an amplifier, and the amplifier's noise is")
    print("     the source's: the term the error model still does not have.")
    print()
    print("  7. ONE LASER IS STILL POSSIBLE, AND COSTS A DEVICE A CELL EITHER WAY.  A column that")
    print(f"     adds fields adds them through rings that turn each by up to 90 degrees, {ring_phase_deg(0.5):.0f} at a")
    print("     weight of a half: a second element a cell to undo it.  A column that adds")
    print(f"     currents needs a photodiode a cell, {K * N:,} of them.  The ring bank is the one")
    print("     reading whose cell is a ring and nothing else, and the one the error model,")
    print("     the crosstalk row and every bus count here were written for.")


def checks():
    """Every claim above, as an assert."""
    assert TILE == (256, 64) and FS == 1e9 and BITS == 7 and RING["radius_um"] == 30
    floor = line_floor_hz()
    assert abs(FSR_HZ / 1e9 - 711.3) < 0.1 and abs(floor / 1e9 - 1.986) < 0.001
    assert abs(q_of(floor) - rate.q_ceiling(FS)) < 1e-6

    # 1. Reading 1.  M's formula reproduces M's own two figures, and pta_ring.py's
    #    spacing is inside M's range, 3% from its kind end.
    assert round(368 / M_SPACING[0]) == M_CHANNELS[368] and round(133 / M_SPACING[0]) == M_CHANNELS[133]
    assert M_SPACING[0] < V1_SPACING < M_SPACING[1] and V1_SPACING / M_SPACING[0] < 1.03
    assert lines_at(V1_SPACING * floor) == rate.lines(FS) == 102
    assert [lines_at(s * floor) for s in M_SPACING] == [105, 77]
    assert [buses_for(lines_at(s * floor)) for s in M_SPACING] == [3, 4] and rate.buses(FS) == 3
    #    Lines are 6.8 to 9.2 shot rates apart at the least, at any rate.
    assert abs(M_SPACING[0] * floor / FS - 6.8) < 0.05 and abs(M_SPACING[1] * floor / FS - 9.2) < 0.05
    assert abs(rate.line_hz(2 * FS) / (2 * FS) - floor / FS) < 1e-12

    # 2. Reading 2.  The grids: 86, 64, 52 and 43 lines, at most 8.27, 11.1, 13.7
    #    and 16.5 GHz apart, clocked at 8, 11, 13 and 16.
    assert [lines_needed(b) for b in BUSES] == [86, 64, 52, 43]
    assert all(lines_needed(b) * b >= K > lines_needed(b) * (b - 1) for b in BUSES)
    assert [round(spacing_ceiling_hz(b) / 1e9, 2) for b in BUSES] == [8.27, 11.11, 13.68, 16.54]
    assert [clocked_hz(b) / 1e9 for b in BUSES] == [8, 11, 13, 16]
    assert all(lines_at(clocked_hz(b)) >= lines_needed(b) for b in BUSES)
    assert all(lines_at(clocked_hz(b) + FS) < lines_needed(b) for b in BUSES)
    #    Three buses: a Q of 85,000 to 97,000 at v1's spacing, and none at M's worst.
    w3, w4 = line_window_hz(3), line_window_hz(4)
    assert round(q_of(w3[1]), -3) == 85_000 and round(q_of(w3[0]), -3) == 97_000
    assert line_window_hz(3, M_SPACING[1]) is None and line_window_hz(3, M_SPACING[0]) is not None
    #    Four: from 62,000, a swing of 2.3 to 3.6 V, and a window at M's worst too.
    assert round(q_of(w4[1]), -3) == 62_000 and line_window_hz(4, M_SPACING[1]) is not None
    assert abs(ring.swing_v(q_of(w4[0])) - 2.27) < 0.005 and abs(ring.swing_v(q_of(w4[1])) - 3.60) < 0.005
    assert round(q_of(line_window_hz(4, M_SPACING[1])[1]), -3) == 81_000
    assert all(line_window_hz(b, M_SPACING[1]) is not None for b in BUSES[1:])
    #    Every window's swing is inside the floorplan's 5 V but six buses' upper end.
    assert all(ring.swing_v(q_of(line_window_hz(b)[1])) < 5.0 for b in BUSES[:3])
    assert ring.swing_v(q_of(line_window_hz(6)[1])) > 5.0

    # 3. Reading 3.  The sources: E is 71 lines and 4 buses, L about 30 GHz and
    #    12; the telecom grid 56, 28, 14 and 7 lines, 5, 10, 19 and 37 buses; the
    #    announced 16 and 8 lines, 16 and 32.
    assert lines_at(E_SPACING_HZ) == 71 and buses_for(71) == 4
    assert abs(L_SPAN_HZ / L_LINES / 1e9 - 30) < 1e-9 and buses_for(lines_at(L_SPAN_HZ / L_LINES)) == 12
    assert [lines_at(g) for g in G_GRID_HZ] == [56, 28, 14, 7]
    assert [buses_for(lines_at(g)) for g in G_GRID_HZ] == [5, 10, 19, 37]
    assert lines_at(G_GRID_HZ[0]) == fp.LINES_A_BUS
    assert [buses_for(n) for n in FEW_LINES] == [32, 16]
    #    E's comb reaches a multiple of the shot clock: 10 GHz is one.
    assert E_SPACING_HZ % FS == 0 and E_DETUNE_HZ < FS
    assert clocked_hz(4) >= E_SPACING_HZ > clocked_hz(3)

    # 4. Reading 4.  A single pole passes 8% of a beat 11 GHz away and half an LSB
    #    is 0.4%; averaged over a shot a whole multiple passes nothing, and 50 MHz
    #    off the clock passes more than half an LSB.
    assert abs(single_pole_pass(clocked_hz(4)) - 0.080) < 0.001 and abs(HALF_LSB - 0.0039) < 0.0001
    assert all(single_pole_pass(clocked_hz(b)) > 10 * HALF_LSB for b in BUSES)
    assert all(averaged_pass(clocked_hz(b)) < 1e-12 for b in BUSES)
    assert all(averaged_pass(clocked_hz(b) + 50e6) > HALF_LSB for b in BUSES[:2])
    assert averaged_pass(spacing_ceiling_hz(3)) > 5 * HALF_LSB

    # 5. Reading 5.  192 to 384 photodiodes, 256 at four buses.
    assert [photodiodes(b) for b in BUSES] == [192, 256, 320, 384]

    # 6. Reading 6.  The light is pta_rate.py's; an input's share 0.34 to 3.4 mW;
    #    a line 1.0 to 10 mW at three buses and 2.0 to 20 at six.
    lo, hi = laser_w()
    assert abs(lo - 0.088) < 0.001 and abs(hi - 0.878) < 0.001
    assert abs(lo / K * 1e3 - 0.34) < 0.005 and abs(hi / K * 1e3 - 3.43) < 0.005
    l3, l6 = line_w(3), line_w(6)
    assert abs(l3[0] * 1e3 - 1.02) < 0.01 and abs(l3[1] * 1e3 - 10.2) < 0.05
    assert abs(l6[0] * 1e3 - 2.04) < 0.01 and abs(l6[1] * 1e3 - 20.4) < 0.05
    assert all(abs(line_w(b)[0] * lines_needed(b) - lo) < 1e-12 for b in BUSES)
    assert round(H_SPAN_NM / RING["fsr_nm"]) == 23
    #    A weight's LSB is 24 MHz at the narrowest line, pta_ring.py's in hertz.
    assert abs(lsb_hz(floor) / 1e6 - 23.9) < 0.05
    pm = ring.lsb_pm(rate.q_ceiling(FS))
    assert abs(lsb_hz(floor) - pm / ring.LAMBDA_PM * F_OPT_HZ) < 1.0

    # 7. Reading 7.  A ring's angle: 0, 45, 60 and 71.6 degrees; and the pair.
    assert [round(ring_phase_deg(w), 1) for w in (1.0, 0.5, 0.25, 0.1)] == [0.0, 45.0, 60.0, 71.6]
    assert abs(coherent_sum((1.0, 0.5)) - 2.5) < 1e-9
    assert abs(sum(math.sqrt(w) for w in (1.0, 0.5)) ** 2 - 2.914) < 0.001
    assert K * N == 16_384 and K * N > 40 * photodiodes(BUSES[-1])

    # 8. The plan's table, and the figures its readings quote.
    wins = [line_window_hz(b) for b in BUSES]
    assert [round(q_of(w[1]), -3) for w in wins] == [85_000, 62_000, 52_000, 42_000]
    assert all(round(q_of(w[0]), -3) == 97_000 for w in wins)
    assert [round(ring.swing_v(q_of(w[1])), 1) for w in wins] == [2.6, 3.6, 4.3, 5.2]
    worst = [line_window_hz(b, M_SPACING[1]) for b in BUSES[1:]]
    assert [round(q_of(w[1]), -3) for w in worst] == [81_000, 69_000, 56_000]
    assert [round(line_w(b)[0] * 1e3, 1) for b in BUSES] == [1.0, 1.4, 1.7, 2.0]
    assert [round(line_w(b)[1] * 1e3) for b in BUSES] == [10, 14, 17, 20]
    #    Three buses leave a ring 14% of room in Q at the kind spacing.
    assert abs(q_of(wins[0][0]) / q_of(wins[0][1]) - 1.15) < 0.01
    #    The beat, 50 MHz off the clock at 11 GHz: 0.45%.
    assert abs(averaged_pass(clocked_hz(4) + 50e6) - 0.0045) < 0.0001

    # 9. B12's own figures, at four buses.
    assert lines_needed(4) == N == 64 and clocked_hz(4) == 11 * FS and photodiodes(4) == 256
    assert lines_at(E_SPACING_HZ) >= lines_needed(4)            # E's 10 GHz fits too
    assert round(q_of(E_SPACING_HZ / V1_SPACING), -3) == 68_000
    assert [round(lsb_hz(x) / 1e6) for x in line_window_hz(4)] == [24, 38]
    assert 4 * (lines_needed(4) - 1) == 252 and K - 1 == 255    # neighbours a column
    assert abs(line_w(4)[0] * 1e3 - 1.37) < 0.01 and abs(line_w(4)[1] * 1e3 - 13.7) < 0.05

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
