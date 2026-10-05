"""
What a published ring makes of the weight cell.

B10 of board_program_plan.md made 256 x 64 the chiplet's working geometry, and
left open whether a weight cell is resonant.  pta_floorplan.py had already half
answered it: a weight that is not resonant is millimeters long, so at this size
the cells are rings.  But it sized the ring at an assumed 25 um, which is a
figure from a patent application for another topology, and said of the ring
itself that "no source sizes a TFLT ring, and three figures are missing behind
this reading: the ring's size, its linewidth, and the swing that moves it by
one".  Question 8 of the plan asks a fourth: can one bank tell 256 lines apart.

This puts published rings in those four places.  It is a reading of five papers
and not a design: nobody here has laid out a ring, and none of these devices is
a weight cell.  Nor is it a survey.  "The smallest" below is the smallest of
these, and a smaller ring may well be published: one of 26 um radius or under
would change reading 2.

PUBLISHED, and quoted as each paper has it.  The arXiv numbers are where each
was read:

  K   I. Krasnokutska, J.-L. Tambasco and A. Peruzzo, "Tunable large free
      spectral range microring resonators in lithium niobate on insulator",
      arXiv:1807.06531 (Scientific Reports, 2019).  Z-cut thin-film lithium
      niobate.  Rings of 30 to 90 um radius: a free spectral range of 5.7 nm and
      a Q of about 9,000 at 30 um, 2.5 nm and 7,500 at 70 um; bend loss about
      1.5 dB/cm at 30 um; group index 2.33; 3 pm/V of electro-optic tuning on
      the 70 um ring, over 0 to -55 V.
  W   C. Wang, M. Zhang, B. Stern, M. Lipson and M. Loncar, "Nanophotonic
      lithium niobate electro-optic modulators", arXiv:1701.06470.  x-cut
      racetrack resonators: a loaded Q of about 50,000 and 7.0 pm/V "with good
      linearity", between electrodes 3.5 um apart; and a Q of 8,000 for a
      30 GHz bandwidth.
  Z   M. Zhang, C. Wang, R. Cheng, A. Shams-Ansari and M. Loncar, "Monolithic
      ultrahigh-Q lithium niobate microring resonator", arXiv:1712.04479.  A
      bending radius of 80 um, and a loaded Q of 5.0 million.
  T   C. Wang, Z. Li, J. Riemensberger et al., "Lithium tantalate
      electro-optical photonic integrated circuits for high volume
      manufacturing", arXiv:2306.16492.  THIN-FILM LITHIUM TANTALATE, the
      plan's platform: a racetrack of 100 um apex radius with 400 um straight
      sections, at 5.6 dB/m.  (pta_floorplan.py already holds the 5.6.)
  S   A. Sayem, S. Z. Uddin, T.-C. Hu et al., "High-power handling and bias
      stability of thin-film lithium tantalate microring and coupling
      resonators", arXiv:2602.00922.  A TFLT ring's resonance "remains below
      0.1 pm" of shift over 25 minutes.

DERIVED here, from those and from a resonance being a Lorentzian line:

  the linewidth        lambda / Q
  the swing            the linewidth over the tuning efficiency: what moves a
                       ring by one linewidth, which is a weight's full range
  a weight's LSB       the shift that moves the transmission by 2^-6 at the
                       steepest point of the line
  lines a bus          the free spectral range over the spacing between lines,
                       and the spacing from the error model's crosstalk: a
                       neighbour's ring, s linewidths away, is seen at
                       1 / (1 + 4 s^2)

ASSUMED, and marked again where each is used:

  - that a cell is no smaller than the ring is across, so its pitch is at
    least two radii.  Electrodes, a bus and a gap to the next ring all add to
    it and none is added here, so every area below is a floor
  - that a tuning efficiency measured on one ring carries to another of a
    different radius.  7.0 pm/V is the larger of the two read here and is the
    one used
  - a wavelength of 1550 nm, and K's group index
  - that the drift this program fits is a ring's.  It is not: the fits are a
    Mach-Zehnder's bias (grx930's design note section 7).  Finding 6

Standard library only.  Run:  python3 docs/designs/pta_ring.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_floorplan as fp
import pta_shot_rate as shot

LAMBDA_PM = 1_550_000.0          # 1550 nm
NG = 2.33                        # K: the TM mode's group index
C_M_S = 299_792_458.0

# ---- published ---------------------------------------------------------------
K30 = dict(radius_um=30, fsr_nm=5.7, q=9_000, bend_db_cm=1.5)
K70 = dict(radius_um=70, fsr_nm=2.5, q=7_500, tuning_pm_v=3.0)
W_Q, W_TUNING_PM_V, W_FAST_Q, W_FAST_GHZ = 50_000, 7.0, 8_000, 30
Z80 = dict(radius_um=80, q=5.0e6)
T_APEX_UM, T_STRAIGHT_UM = 100, 400
S_SHIFT_PM, S_MINUTES = 0.1, 25

TUNING_PM_V = W_TUNING_PM_V      # ASSUMED to carry from W's racetrack to any ring

# ---- the program's own -------------------------------------------------------
TILE = shot.X2_TILE              # 256 x 64: B10
SMALLER = (128, 64)
W_BITS = 6                       # section 4.3: the weight's resolution
XTALK = {"v1": 0.02, "v0": 0.10} # section 4.3: crosstalk between neighbouring inputs
INPUT_V = 5.0                    # the shortest of the floorplan's three modulators
WEIGHT_HZ = shot.weight_updates_per_s(TILE[0], TILE[1], 64, shot.X2_FS) / (TILE[0] * TILE[1])
STD_DIE_MM2 = fp.STD_DIE_MM[0] * fp.STD_DIE_MM[1]
PATENT_LINES_A_BUS = fp.LINES_A_BUS


# ---- 1. the cell's size ------------------------------------------------------
def cell_um(radius_um):
    """ASSUMED: a ring's cell is no smaller than the ring is across."""
    return 2 * radius_um


def weights_mm2(tile, pitch_um):
    return fp.field_mm2(tile[0] * tile[1], pitch_um)


def racetrack_mm2(tile):
    """T's racetrack as published, one a cell: two apex radii by that and its straights."""
    w, h = 2 * T_APEX_UM, 2 * T_APEX_UM + T_STRAIGHT_UM
    return tile[0] * tile[1] * w * h * fp.MM * fp.MM


def die_mm2(tile, pitch_um):
    """(weights, inputs, link): the floorplan's own sum, at a ring's pitch."""
    w, i = fp.die_mm2(tile[0], tile[1], pitch_um, INPUT_V)
    return w, i, fp.link_modules(tile[0], tile[1], 64) * fp.module_mm2()


def fsr_nm(radius_um):
    """A ring's free spectral range from its circumference: lambda^2 / (ng L)."""
    return LAMBDA_PM ** 2 / (NG * 2 * math.pi * radius_um * 1e6) / 1e3


# ---- 2. the linewidth and the swing -------------------------------------------
def linewidth_pm(q):
    return LAMBDA_PM / q


def swing_v(q, tuning_pm_v=TUNING_PM_V):
    """Volts to move a ring by one linewidth."""
    return linewidth_pm(q) / tuning_pm_v


def q_for_swing(volts, tuning_pm_v=TUNING_PM_V):
    return LAMBDA_PM / (tuning_pm_v * volts)


def linewidth_hz(q):
    return C_M_S / (LAMBDA_PM * 1e-12) / q


def q_from_loss(db_cm):
    """The intrinsic Q a propagation loss allows: 2 pi ng / (lambda alpha)."""
    alpha_m = db_cm * 100 * math.log(10) / 10
    return 2 * math.pi * NG / (LAMBDA_PM * 1e-12 * alpha_m)


# ---- 3. how still it has to hold ------------------------------------------------
SLOPE = 3 * math.sqrt(3) / 4     # a Lorentzian's steepest slope, in units of 1 / linewidth


def lsb_pm(q, bits=W_BITS):
    """The resonance shift that moves a weight by one LSB, where the line is steepest."""
    return linewidth_pm(q) / (SLOPE * 2 ** bits)


# ---- 4. lines a bus ---------------------------------------------------------------
def spacing(chi):
    """Linewidths between two lines for a neighbour's ring to be seen at chi."""
    return 0.5 * math.sqrt(1 / chi - 1)


def lines_a_bus(fsr, q, chi):
    """Lines one free spectral range holds at that crosstalk."""
    return int(fsr * 1e3 / (spacing(chi) * linewidth_pm(q)))


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    k, n = TILE
    section("1. The cell: as large as a published ring is across")
    print(f"  {'ring':<34}{'radius':>8}{'cell':>10}{'256x64 weights':>16}{'with inputs, link':>19}"
          f"{'128x64 the same':>17}")
    rows = (("K, the smallest read here", K30["radius_um"]), ("K, the ring it tuned", K70["radius_um"]),
            ("Z, the highest Q", Z80["radius_um"]))
    for label, r in rows:
        p = cell_um(r)
        print(f"  {label:<34}{r:>5} um{p:>7} um{weights_mm2(TILE, p):>12.0f} mm2"
              f"{sum(die_mm2(TILE, p)):>15.0f} mm2{sum(die_mm2(SMALLER, p)):>13.0f} mm2")
    print(f"  {'T, lithium tantalate, a racetrack':<34}{T_APEX_UM:>5} um{'200x600':>10}"
          f"{racetrack_mm2(TILE):>12.0f} mm2{'':>19}{racetrack_mm2(SMALLER):>13.0f} mm2")
    w25 = sum(die_mm2(TILE, fp.P_CELL))
    print(f"  The floorplan's bound, at its assumed {fp.P_CELL} um cell: {w25:.1f} mm2.  A standard die is"
          f" {STD_DIE_MM2:.0f} mm2.")
    print(f"  Inputs are a {INPUT_V:.0f} V modulator a row at the cell's pitch.  ASSUMED: rings touching,")
    print("  so every figure is a floor.  The coarsest pitch at which 256 x 64 still fits a")
    print(f"  standard die is {fp.pitch_ceiling_um(k, n, INPUT_V, STD_DIE_MM2 - die_mm2(TILE, 60)[2]):.0f} um:"
          f" a ring of {fp.pitch_ceiling_um(k, n, INPUT_V, STD_DIE_MM2 - die_mm2(TILE, 60)[2]) / 2:.0f} um radius,"
          " under the smallest read here.")
    print(f"  K's two rings check the circumference formula: {fsr_nm(30):.2f} nm for {K30['fsr_nm']} measured,"
          f" {fsr_nm(70):.2f} for {K70['fsr_nm']}.")

    section("2. The linewidth, and the swing that moves a ring by one")
    print(f"  {'what':<44}{'Q':>10}{'linewidth':>12}{'at 7.0 pm/V':>13}{'at 3 pm/V':>11}")
    for label, q in (("K's 30 um ring, as built", K30["q"]), ("K's 70 um ring, as built", K70["q"]),
                     ("W's racetrack, where 7.0 pm/V was measured", W_Q), ("Z's 80 um ring", Z80["q"])):
        print(f"  {label:<44}{q:>10,.0f}{linewidth_pm(q):>9.2f} pm{swing_v(q):>11.3g} V"
              f"{swing_v(q, K70['tuning_pm_v']):>9.3g} V")
    print("  And the Q each of the floorplan's three swings asks of a ring:")
    for v in fp.SWINGS_V[::-1]:
        q = q_for_swing(v)
        print(f"    {v:.0f} V  Q {q:>9,.0f}   a line {linewidth_pm(q):5.1f} pm wide,"
              f" {linewidth_hz(q) / 1e6:>6,.0f} MHz: {linewidth_hz(q) / WEIGHT_HZ:,.0f} times the"
              f" {WEIGHT_HZ / 1e6:.1f} MHz a weight is rewritten at")
    print(f"  K's 30 um ring loses {K30['bend_db_cm']} dB/cm to its bend.  DERIVED: that allows an intrinsic Q"
          f" of {q_from_loss(K30['bend_db_cm']):,.0f},")
    print(f"  {q_from_loss(K30['bend_db_cm']) / 2:,.0f} loaded at critical coupling.  It was built at"
          f" {K30['q']:,}, and nobody has shown the other.")

    section("3. How still a ring has to hold")
    print(f"  A {W_BITS}-bit weight's LSB, as a shift of the resonance where the line is steepest:")
    for v in fp.SWINGS_V[::-1]:
        q = q_for_swing(v)
        print(f"    a {v:.0f} V ring   {lsb_pm(q):.3f} pm an LSB   S's {S_SHIFT_PM} pm in {S_MINUTES} minutes is"
              f" {S_SHIFT_PM / lsb_pm(q):.2f} of one")
    print("  The LSB goes as the swing, because both go as the linewidth: a ring that needs")
    print("  a fifth of the voltage has to hold five times as still.")

    section("4. Lines a bus: can one bank tell 256 apart")
    for name, chi in XTALK.items():
        print(f"  {name}'s crosstalk of {chi:.0%} is a neighbour {spacing(chi):.1f} linewidths away.")
    print(f"  {'ring':<38}{'finesse':>9}{'lines, v1':>11}{'lines, v0':>11}{'buses for 256, v1':>19}")
    cases = [(f"30 um, a {v:.0f} V swing", K30["fsr_nm"], q_for_swing(v)) for v in fp.SWINGS_V[::-1]]
    cases.append(("30 um, at its bend loss's Q", K30["fsr_nm"], q_from_loss(K30["bend_db_cm"]) / 2))
    cases.append(("70 um, a 5 V swing", K70["fsr_nm"], q_for_swing(5.0)))
    for label, fsr, q in cases:
        l1, l0 = lines_a_bus(fsr, q, XTALK["v1"]), lines_a_bus(fsr, q, XTALK["v0"])
        print(f"  {label:<38}{fsr * 1e3 / linewidth_pm(q):>9.0f}{l1:>11}{l0:>11}{-(-k // l1):>19}")
    print(f"  The patent application's \"k <~ {PATENT_LINES_A_BUS}\" a bus is what a 30 um ring holds at v1's")
    q56 = LAMBDA_PM / (K30["fsr_nm"] * 1e3 / (PATENT_LINES_A_BUS * spacing(XTALK["v1"])))
    print(f"  crosstalk with a Q of {q56:,.0f}, a {swing_v(q56):.1f} V swing.  It gave no derivation; this is one.")

    findings()
    checks()


def findings():
    k, n = TILE
    q5, q1 = q_for_swing(5.0), q_for_swing(1.0)
    qb = q_from_loss(K30["bend_db_cm"]) / 2
    ceiling = fp.pitch_ceiling_um(k, n, INPUT_V, STD_DIE_MM2 - die_mm2(TILE, 60)[2])
    print()
    print("What this says, six readings.")
    print()
    w25 = weights_mm2(TILE, fp.P_CELL)
    print("  1. THESE RINGS ARE 30 TO 80 um IN RADIUS, SO A CELL IS 60 TO 160 um AND NOT 25.")
    print(f"     The floorplan's area for the weights was {weights_mm2(TILE, 60) / w25:.0f} to"
          f" {weights_mm2(TILE, 160) / w25:.0f} times too small on a ring, and")
    print(f"     {racetrack_mm2(TILE) / w25:.0f} times on lithium tantalate's racetrack.  Its other reading stands:")
    print("     a pad a cell at flip-chip's 100 um is enough, and the finest bond is not needed.")
    print()
    print(f"  2. AT 256 x 64 THE WEIGHTS ARE {weights_mm2(TILE, 60):.0f} TO {weights_mm2(TILE, 160):.0f} mm2, AND NONE OF THESE RINGS FITS A")
    print(f"     STANDARD DIE.  With its inputs and its link the tile is {sum(die_mm2(TILE, 60)):.0f} mm2 at the smallest ring")
    print(f"     and {sum(die_mm2(TILE, 160)):.0f} at the one with the highest Q.  On lithium tantalate as published, a")
    print(f"     racetrack a cell, the weights alone are {racetrack_mm2(TILE):,.0f} mm2.  128 x 64 is {sum(die_mm2(SMALLER, 60)):.0f} mm2")
    print("     at the smallest ring, and is the largest candidate that fits a standard die at one.")
    print(f"     The ring that would carry 256 x 64 is {ceiling / 2:.0f} um in radius, {K30['radius_um'] - ceiling / 2:.0f} under the smallest read,")
    print("     and that with nothing added to a cell.  Five papers are not a survey.")
    print()
    print("  3. THE SWING IS IN RANGE.  A linewidth of W's racetrack is"
          f" {swing_v(W_Q):.1f} V at the tuning measured on it,")
    print(f"     so the floorplan's 5 V is a published device's.  2 V and 1 V ask a Q of {q_for_swing(2.0):,.0f}")
    print(f"     and {q1:,.0f}.  Z's Q is far past that, and is not a small ring's with electrodes.")
    print()
    print("  4. LOW VOLTAGE IS PAID FOR IN STABILITY, ONE FOR ONE.  The one figure read here, on")
    print(f"     lithium tantalate, is {S_SHIFT_PM} pm in {S_MINUTES} minutes: {S_SHIFT_PM / lsb_pm(q5):.2f} of a weight's LSB on a 5 V ring")
    print(f"     and {S_SHIFT_PM / lsb_pm(q1):.2f} on a 1 V one.  It is one device, for {S_MINUTES} minutes, and nothing about temperature.")
    print()
    print(f"  5. ONE BUS DOES NOT CARRY 256 LINES.  At v1's crosstalk a 30 um ring holds {lines_a_bus(K30['fsr_nm'], q5, XTALK['v1'])} lines at a")
    print(f"     5 V swing and {lines_a_bus(K30['fsr_nm'], qb, XTALK['v1'])} at the best its bend loss allows, so 256 inputs are"
          f" {-(-k // lines_a_bus(K30['fsr_nm'], qb, XTALK['v1']))} to {-(-k // lines_a_bus(K30['fsr_nm'], q5, XTALK['v1']))} buses.")
    print("     A larger ring holds fewer: its lines are as wide and its range is shorter.")
    print("     So the source is tens of lines reused across buses, and question 8's answer")
    print("     for a ring bank is neither one laser nor 256.")
    print()
    print("  6. THE DRIFT THIS PROGRAM FITS IS NOT A RING'S.  TFLT's and TFLN's fits are a")
    print("     Mach-Zehnder's bias drifting.  A ring turns the same change of index into a")
    print("     weight error in proportion to its Q, and nothing here has priced that.")
    print("     The calibration intervals, and the geometry sweep's drift row, are a")
    print("     modulator's until it is.")


def checks():
    """Every claim above, as an assert."""
    k, n = TILE
    assert TILE == (256, 64) and abs(WEIGHT_HZ / 1e6 - 15.6) < 0.05

    # 1. The circumference formula gives K's two measured ranges to 7%.
    assert abs(fsr_nm(30) / K30["fsr_nm"] - 1) < 0.05 and abs(fsr_nm(70) / K70["fsr_nm"] - 1) < 0.07
    #    Rings touching: 59 mm2 of weights at the smallest and 419 at Z's.
    assert abs(weights_mm2(TILE, 60) - 59.0) < 0.05 and abs(weights_mm2(TILE, 160) - 419.4) < 0.05
    assert abs(weights_mm2(TILE, 140) - 321.1) < 0.05 and abs(racetrack_mm2(TILE) - 1966) < 1
    #    The floorplan's own bound is what pta_geometry.py quotes: 39.7 mm2.
    assert abs(sum(die_mm2(TILE, fp.P_CELL)) - 39.7) < 0.05
    #    Reading 1: 6 to 41 times in area on a ring, and 192 on T's racetrack,
    #    which is 20 standard dies of weights.
    w25 = weights_mm2(TILE, fp.P_CELL)
    assert round(weights_mm2(TILE, 60) / w25) == 6 and round(weights_mm2(TILE, 160) / w25) == 41
    assert round(racetrack_mm2(TILE) / w25) == 192 and round(racetrack_mm2(TILE) / STD_DIE_MM2) == 20
    assert abs(racetrack_mm2(SMALLER) - 983) < 1

    # 2. Reading 2.  None of these rings puts 256 x 64 on a standard die: 124 mm2
    #    at the smallest, and the ring that would is 26 um in radius.  128 x 64
    #    fits at the smallest and at no other.
    assert abs(sum(die_mm2(TILE, 60)) - 123.6) < 0.1 and abs(sum(die_mm2(TILE, 160)) - 584.4) < 0.1
    assert all(sum(die_mm2(TILE, cell_um(r))) > STD_DIE_MM2 for r in (30, 70, 80))
    ceiling = fp.pitch_ceiling_um(k, n, INPUT_V, STD_DIE_MM2 - die_mm2(TILE, 60)[2])
    assert 51 < ceiling < 54 and ceiling / 2 < K30["radius_um"]
    assert abs(sum(die_mm2(SMALLER, 60)) - 62.2) < 0.1 and sum(die_mm2(SMALLER, 60)) < STD_DIE_MM2
    assert sum(die_mm2(SMALLER, 140)) > STD_DIE_MM2
    #    The plan's table: 466 at K's tuned ring, and the smaller tile's 233 and 293.
    assert round(sum(die_mm2(TILE, 140))) == 466
    assert [round(sum(die_mm2(SMALLER, p))) for p in (140, 160)] == [233, 293]
    #    And 256 x 128, the candidate past the working one, fits at none of them.
    assert sum(die_mm2((256, 128), 60)) > STD_DIE_MM2

    # 3. Reading 3.  W's linewidth is 31 pm and 4.4 V; K's rings as built would
    #    take tens of volts; the three swings ask Qs of 44,000, 111,000 and 221,000.
    assert abs(linewidth_pm(W_Q) - 31.0) < 0.05 and abs(swing_v(W_Q) - 4.43) < 0.005
    assert swing_v(K70["q"], K70["tuning_pm_v"]) > 60 and swing_v(K30["q"]) > 20
    assert [round(q_for_swing(v), -3) for v in (5.0, 2.0, 1.0)] == [44_000, 111_000, 221_000]
    assert q_for_swing(1.0) < Z80["q"] / 20
    #    K's bend loss allows about 137,000 loaded, fifteen times what it was built at.
    qb = q_from_loss(K30["bend_db_cm"]) / 2
    assert 135_000 < qb < 139_000 and qb > 15 * K30["q"]
    #    A weight is rewritten at 15.6 MHz, and every one of these lines is wider.
    assert linewidth_hz(q_for_swing(1.0)) > 50 * WEIGHT_HZ
    #    W's faster ring is the same law: a Q of 8,000 is a line 24 GHz wide.
    assert 20e9 < linewidth_hz(W_FAST_Q) < W_FAST_GHZ * 1e9

    # 4. Reading 4.  An LSB is 0.42 pm on a 5 V ring and 0.084 on a 1 V one, so
    #    S's 0.1 pm is a quarter of one and more than one; and it goes as the swing.
    q5, q1 = q_for_swing(5.0), q_for_swing(1.0)
    assert abs(lsb_pm(q5) - 0.421) < 0.001 and abs(lsb_pm(q1) - 0.0842) < 0.0002
    assert abs(S_SHIFT_PM / lsb_pm(q5) - 0.24) < 0.005 and abs(S_SHIFT_PM / lsb_pm(q1) - 1.19) < 0.005
    assert abs(lsb_pm(q5) / lsb_pm(q1) - 5.0) < 1e-9

    # 5. Reading 5.  v1's 2% is 3.5 linewidths and v0's 10% is 1.5.  A 30 um ring
    #    holds 46 lines at 5 V and 143 at its bend loss's Q: 6 buses and 2.  And
    #    "56 a bus" is that ring at a Q of 53,000 and 4.2 V.
    assert abs(spacing(0.02) - 3.5) < 1e-9 and abs(spacing(0.10) - 1.5) < 1e-9
    l5, lb = lines_a_bus(K30["fsr_nm"], q5, 0.02), lines_a_bus(K30["fsr_nm"], qb, 0.02)
    assert (l5, lb) == (46, 143) and (-(-k // l5), -(-k // lb)) == (6, 2), (l5, lb)
    assert lines_a_bus(K30["fsr_nm"], q1, 0.02) < k
    assert lines_a_bus(K70["fsr_nm"], q5, 0.02) < l5
    #    The plan's table: finesse, lines and buses for its four rings.
    q2 = q_for_swing(2.0)
    table = [(K30["fsr_nm"], q5), (K30["fsr_nm"], q2), (K30["fsr_nm"], qb), (K70["fsr_nm"], q5)]
    assert [round(f * 1e3 / linewidth_pm(q)) for f, q in table] == [163, 407, 503, 71]
    assert [lines_a_bus(f, q, 0.02) for f, q in table] == [46, 116, 143, 20]
    assert [-(-k // lines_a_bus(f, q, 0.02)) for f, q in table] == [6, 3, 2, 13]
    q56 = LAMBDA_PM / (K30["fsr_nm"] * 1e3 / (PATENT_LINES_A_BUS * spacing(0.02)))
    assert 53_000 < q56 < 54_000 and abs(swing_v(q56) - 4.2) < 0.1

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
