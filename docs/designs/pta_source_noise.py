"""
What the light's noise may be: grx930's measurement, in the plan's units.

The plan holds the tile's source to an intensity noise nobody measured.
pta_shot_rate.py section 6 took the receiver's own allowance, half an 8-bit LSB
of full scale, and applied it to the light: about -144 dB/Hz at 1 GS/s.  It said
so: "the error model has ... nothing for the source", and the figure was an
assumption about a budget section 4.3 does not have.  B12 then made the source
a comb with an amplifier behind it, whose noise "is the source's: the term the
error model still does not have".

grx930's accuracy harness has that term now, and has run it.  This puts what it
found beside what the plan assumed.

MEASURED IN A MODEL, by grx930 (c930/doc/pta_error_model_design_note.md section
5, "What may the light do?"; `sim/pta_mnist.sh MNIST WORK source`, 2026-10-05).
Points lost on D3 against the same weights on the host, on the 256 x 64 tile on
four buses, over version 1 of the budget, five networks, mean and standard
error.  Three things a source does to a line's power, each an rms fraction of
it: every line together and anew each shot; each line on its own and anew each
shot; and each line's level, fixed for a run.  And two readings of how a line's
light reaches a column, because how the tile signs a weight is not settled:
through the weight alone, as a balanced pair of photodiodes has it, or through
the weight and an offset the host takes off again, as one photodiode would.

The term is first order and outside grx930's contract: added on the host's side,
after the converter.  That note says what that leaves out.

DERIVED here:

  a density       an rms fraction a shot is rms^2 over the receiver's noise
                  bandwidth.  Three bandwidths, because nothing has chosen the
                  receiver: the shot rate, which is pta_shot_rate.py's; half of
                  it, which is a receiver that averages over a shot (B12); and
                  pi/2 of a single pole's corner, which is pta_power.py's
  what is allowed the largest rms measured that costs under a tenth of a point
                  over version 1, with every smaller one measured doing so too
  what reaches    how much of a noise's own rms arrives at layer 1's sums: from
  the sums        their error with it and without, in quadrature

ASSUMED, as the harness does: noise with no memory from one shot to the next.
A source whose power wanders slowly is a level that moves between calibrations,
and nobody has run that.

No source's noise is held here, a comb's or an amplifier's.  This says what one
may be and not what one is.

Standard library only.  Run:  python3 docs/designs/pta_source_noise.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_power as power
import pta_shot_rate as shot
import pta_source as source

FS = shot.X2_FS                            # 1 GS/s: B11
V1 = shot.REQ["v1"]
BITS = V1["adc_bits"]
BUDGET = 0.10                              # a tenth of a point: what a row of the budget costs

# ---- grx930's figures ----------------------------------------------------------
# rms fraction -> (every line together, each line on its own, a line's level),
# each (points lost, standard error), or None where it was not run.
V1_LOSS, V1_E1 = (0.20, 0.05), 9.14
KINDS = ("together", "a line", "level")
LEVEL = "a line's level"
PAIR = {
    0.002: ((0.21, 0.04), (0.20, 0.05), None),
    0.005: ((0.20, 0.05), None, None),
    0.01:  ((0.25, 0.04), (0.21, 0.05), (0.25, 0.04)),
    0.02:  ((0.24, 0.04), (0.23, 0.05), (0.24, 0.02)),
    0.05:  ((0.28, 0.04), (0.26, 0.05), (0.27, 0.01)),
    0.1:   ((0.45, 0.06), (0.36, 0.03), (0.34, 0.04)),
    0.2:   (None, (0.76, 0.06), (0.65, 0.07)),
}
OFFSET = {
    0.002: ((0.30, 0.08), (0.20, 0.06), None),
    0.005: ((0.40, 0.10), None, None),
    0.01:  ((0.94, 0.21), (0.20, 0.06), (0.23, 0.05)),
    0.02:  ((3.75, 0.70), (0.35, 0.05), (0.32, 0.03)),
    0.05:  (None, (0.71, 0.12), (0.55, 0.18)),
}
# Layer 1's error against the network's own sums, percent of their rms.
PAIR_E1 = {
    0.002: (9.15, 9.14, None), 0.005: (9.16, None, None), 0.01: (9.21, 9.15, 9.13),
    0.02: (9.40, 9.18, 9.15), 0.05: (10.66, 9.38, 9.28), 0.1: (14.28, 10.06, 9.85),
    0.2: (None, 12.41, 11.98),
}
OFFSET_E1 = {
    0.002: (10.14, 9.17, None), 0.005: (14.26, None, None), 0.01: (23.72, 9.81, 10.34),
    0.02: (44.73, 11.59, 12.89), 0.05: (None, 20.01, 23.24),
}
ONE_BUS = dict(rms=0.05, one=((0.28, 0.04), 9.38), four=((0.26, 0.05), 9.38))
# The three at once: (together, a line, level) -> points lost, and through a
# pair layer 1's error as well.
JOINT_PAIR = {(0.02, 0.05, 0.05): (0.30, 0.04), (0.05, 0.05, 0.05): (0.35, 0.05)}
JOINT_PAIR_E1 = {(0.02, 0.05, 0.05): 9.77, (0.05, 0.05, 0.05): 10.98}
JOINT_OFFSET = {(0.002, 0.01, 0.01): (0.33, 0.08)}
ROWS = (0.02, 0.05, 0.05)                  # the set that costs the budget between them


# ---- as a density ---------------------------------------------------------------
def bands():
    """The receiver's noise bandwidth, three ways: (name, hertz)."""
    return (("the shot rate", FS),
            ("a receiver that averages over a shot", FS / 2),
            ("a single pole", math.pi / 2 * power.settle_bandwidth_hz(FS, BITS)))


def db_hz(rms, bandwidth_hz=FS):
    return 10 * math.log10(rms * rms / bandwidth_hz)


def assumed_rms():
    """What the plan assumed: the receiver's allowance of full scale, as a fraction."""
    return V1["rx_noise_lsb"] / 2 ** shot.NOISE_LSB_BITS


def over(cell):
    """Points lost over version 1's own."""
    return cell[0] - V1_LOSS[0]


def allowed(table, kind):
    """The largest rms that costs under the budget, every smaller one doing so too.
    None if the smallest measured already costs it."""
    best = None
    for rms in sorted(table):
        cell = table[rms][kind]
        if cell is None:
            continue
        if over(cell) >= BUDGET - 1e-9:
            break
        best = rms
    return best


def reaches(table_e1, rms, kind):
    """The share of a noise's rms that arrives at layer 1's sums."""
    e = table_e1[rms][kind]
    return math.sqrt(e * e - V1_E1 * V1_E1) / (100 * rms)


def rows_summed(table, triple):
    """What three settings cost over version 1 if their separate costs added."""
    return sum(over(table[rms][k]) for k, rms in enumerate(triple))


def in_quadrature(table_e1, triple):
    """Layer 1's error if the three added in quadrature to version 1's."""
    extra = sum(table_e1[rms][k] ** 2 - V1_E1 ** 2 for k, rms in enumerate(triple))
    return math.sqrt(V1_E1 ** 2 + extra)


def cell_text(cell):
    return "" if cell is None else f"{cell[0]:.2f} +-{cell[1]:.2f}"


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    print("What the light's noise may be: grx930's measurement, in the plan's units.")
    print(f"256 x 64 on four buses, version 1 of the budget, which by itself loses {V1_LOSS[0]:.2f} +-{V1_LOSS[1]:.2f} points.")

    section("1. What grx930 measured: points lost on D3")
    for name, table in (("Through a balanced pair", PAIR), ("Through an offset", OFFSET)):
        print(f"  {name}:")
        print(f"  {'rms a shot':<12}{'every line together':>22}{'each on its own':>18}{LEVEL:>18}")
        for rms in sorted(table):
            print(f"  {rms:<12.1%}" + "".join(f"{cell_text(c):>{w}}" for c, w in zip(table[rms], (22, 18, 18))))
    print(f"  Buses do not enter: {ONE_BUS['rms']:.0%} a line loses {cell_text(ONE_BUS['one'][0])} on one bus and"
          f" {cell_text(ONE_BUS['four'][0])} on four.")
    for t, loss in JOINT_PAIR.items():
        print(f"  All three through a pair, {t[0]:.0%} with {t[1]:.0%} a line and {t[2]:.0%} of level: {cell_text(loss)},"
              f" {over(loss):.2f} over version 1,")
        print(f"    where the three alone sum to {rows_summed(PAIR, t):.2f}; layer 1's error {JOINT_PAIR_E1[t]}%,"
              f" and in quadrature {in_quadrature(PAIR_E1, t):.2f}%.")
    for t, loss in JOINT_OFFSET.items():
        print(f"  And through an offset, {t[0]:.1%} with {t[1]:.0%} a line and {t[2]:.0%} of level: {cell_text(loss)},"
              f" {over(loss):.2f} over, for {rows_summed(OFFSET, t):.2f} summed.")

    section("2. As a density")
    print(f"  The plan assumed {assumed_rms():.3%} rms of full scale: {db_hz(assumed_rms()):.1f} dB/Hz over the shot rate"
          f" (pta_shot_rate.py).")
    print(f"  What costs under {BUDGET:.2f} of a point over version 1, as an rms a shot and as a density:")
    print(f"  {'':<34}{'rms':>7}" + "".join(f"{n:>40}" for n, _ in bands()))
    for name, table in (("a pair", PAIR), ("an offset", OFFSET)):
        for k, kind in enumerate(KINDS):
            r = allowed(table, k)
            row = "none measured" if r is None else f"{r:.1%}"
            dens = "" if r is None or kind == "level" else "".join(
                f"{db_hz(r, hz):>34.1f} dB/Hz" for _, hz in bands())
            print(f"  {'through ' + name + ', ' + kind:<34}{row:>7}{dens}")
    print(f"  A level is not a noise and has no density: {allowed(PAIR, 2):.0%} is {10 * math.log10(1 + allowed(PAIR, 2)):.2f} dB between lines.")
    print(f"  And as three rows of one budget, {ROWS[0]:.0%} together beside {ROWS[1]:.0%} a line and {ROWS[2]:.0%} of level:"
          f" {db_hz(ROWS[0]):.1f} dB/Hz over the shot rate.")

    section("3. What reaches the sums")
    print(f"  Layer 1's error is {V1_E1}% of its sums' rms under version 1.  What each noise adds, as a")
    print("  share of its own rms:")
    print(f"  {'':<22}{'together':>10}{'a line':>10}{'level':>10}")
    for name, table, at in (("through a pair", PAIR_E1, (0.1, 0.2, 0.2)), ("through an offset", OFFSET_E1, (0.01, 0.05, 0.05))):
        print(f"  {name:<22}" + "".join(f"{reaches(table, r, k):>10.2f}" for k, r in enumerate(at)))
    print(f"  An offset passes {reaches(OFFSET_E1, 0.01, 0) / reaches(PAIR_E1, 0.1, 0):.0f} times what a pair does of noise the lines share,")
    print(f"  {20 * math.log10(reaches(OFFSET_E1, 0.01, 0) / reaches(PAIR_E1, 0.1, 0)):.0f} dB: it is every lit input at a weight of one, beside weights that are small.")

    findings()
    checks()


def findings():
    ap, ao = allowed(PAIR, 0), allowed(OFFSET, 0)
    ratio = reaches(OFFSET_E1, 0.01, 0) / reaches(PAIR_E1, 0.1, 0)
    print()
    print("What this says, seven readings.")
    print()
    print(f"  1. THROUGH A BALANCED PAIR THE SOURCE MAY BE {db_hz(ap) - db_hz(assumed_rms()):.0f} dB NOISIER THAN THE PLAN ASSUMED.")
    print(f"     {ap:.0%} rms a shot costs under a tenth of a point, where the plan held it to {assumed_rms():.1%}:")
    print(f"     {db_hz(ap):.0f} dB/Hz over the shot rate and not {db_hz(assumed_rms()):.0f}.  That holds for the lines together,")
    print("     for each on its own, and for their levels, each taken alone.")
    print()
    print(f"  2. AS ROWS OF ONE BUDGET THEY ARE {ROWS[0]:.0%}, {ROWS[1]:.0%} AND {ROWS[2]:.0%}, AND THEY DO NOT COMPOUND.  That set costs")
    print(f"     {over(JOINT_PAIR[ROWS]):.2f} of a point between them, and {ap:.0%} of each {over(JOINT_PAIR[(0.05, 0.05, 0.05)]):.2f}.  Neither is more than its rows")
    print(f"     summed, and layer 1's error is their root sum of squares.  {ROWS[0]:.0%} together is {db_hz(ROWS[0]):.0f} dB/Hz,")
    print(f"     {db_hz(ROWS[0]) - db_hz(assumed_rms()):.0f} dB easier than the plan assumed.")
    print()
    print("  3. THROUGH AN OFFSET THE PLAN'S FIGURE IS ABOUT WHAT THE BUDGET STANDS.  There")
    print(f"     {min(OFFSET):.1%} together already costs {over(OFFSET[min(OFFSET)][0]):.2f}, and {0.005:.1%} what {0.1:.0%} costs through a pair:"
          f" {OFFSET[0.005][0][0]:.2f} for {PAIR[0.1][0][0]:.2f},")
    print(f"     with the same error on layer 1's sums, {OFFSET_E1[0.005][0]}% for {PAIR_E1[0.1][0]}%.")
    print()
    print(f"  4. SO HOW THE TILE SIGNS A WEIGHT IS WORTH {ratio:.0f} TIMES IN THE SOURCE'S NOISE, {20 * math.log10(ratio):.0f} dB.")
    print("     B12 left the sign open and said a pair would double the photodiodes.  This is")
    print("     what the other reading costs instead.  A column of rings at a weight of zero")
    print("     that reads the same light would take the offset's noise off with it, and is")
    print("     the pair again.")
    print()
    print(f"  5. NOISE A LINE COUNTS FOR {reaches(PAIR_E1, 0.2, 1):.2f} OF NOISE THE LINES SHARE, at the sums.  Not the eighth")
    print("     that 64 independent lines on equal weights would leave.  And the buses do not")
    print("     enter.  An amplifier's noise is both kinds, and nothing here says in what parts.")
    print()
    lev = allowed(PAIR, 2)
    print(f"  6. A COMB'S LINES HAVE TO BE LEVEL TO {lev:.0%}, {10 * math.log10(1 + lev):.1f} dB, AND NOTHING BUILT MEASURES IT.  A")
    print("     line's level is one error on every weight in a row.  X3's probe reads each cell")
    print("     through a weight of zero, which a line's power multiplies: it cannot see one.")
    print("     Levelling is somebody else's, or the probe needs weights in it.")
    print()
    print("  7. NONE OF THIS IS A SOURCE.  It is what one may do, on one network, in a term that")
    print("     is first order and outside grx930's contract.  No comb's noise is held here, and")
    print("     no amplifier's.")


def checks():
    """Every claim above, as an assert."""
    assert FS == 1e9 and BITS == 7 and source.photodiodes(4) == 256

    # The plan's assumption: half an 8-bit LSB of full scale, -144 dB/Hz at the shot rate.
    assert abs(assumed_rms() - 0.5 / 256) < 1e-15
    assert abs(db_hz(assumed_rms()) - shot.rin_limit_db_hz(V1, FS)) < 1e-9
    assert abs(db_hz(assumed_rms()) + 144.2) < 0.05
    b = dict(bands())
    assert b["a receiver that averages over a shot"] == FS / 2
    assert abs(b["a single pole"] / 1e9 - 1.386) < 0.001

    # 1. Reading 1.  5% through a pair, all three ways; -116 dB/Hz; 28 dB.
    assert [allowed(PAIR, k) for k in range(3)] == [0.05, 0.05, 0.05]
    assert all(over(PAIR[0.05][k]) < BUDGET <= over(PAIR[0.1][k]) for k in range(3))
    assert abs(db_hz(0.05) + 116.0) < 0.05 and abs(db_hz(0.05) - db_hz(assumed_rms()) - 28.2) < 0.05
    assert abs(db_hz(0.05, FS / 2) + 113.0) < 0.05 and abs(db_hz(0.05, b["a single pole"]) + 117.4) < 0.05
    #    The plan's own figure costs nothing that can be seen, through a pair.
    assert abs(over(PAIR[0.002][0])) < PAIR[0.002][0][1] and abs(over(PAIR[0.002][1])) < PAIR[0.002][1][1]
    #    10% costs 0.25, 0.16 and 0.14.
    assert [round(over(PAIR[0.1][k]), 2) for k in range(3)] == [0.25, 0.16, 0.14]

    # 2. Reading 2.  The set of rows costs a tenth, 5% of each 0.15; neither is
    #    more than its rows summed, and the errors add in quadrature.
    assert ROWS in JOINT_PAIR and abs(over(JOINT_PAIR[ROWS]) - 0.10) < 0.005
    assert abs(over(JOINT_PAIR[(0.05, 0.05, 0.05)]) - 0.15) < 0.005
    assert [round(rows_summed(PAIR, t), 2) for t in JOINT_PAIR] == [0.17, 0.21]
    assert all(over(JOINT_PAIR[t]) <= rows_summed(PAIR, t) for t in JOINT_PAIR)
    assert all(abs(in_quadrature(PAIR_E1, t) - JOINT_PAIR_E1[t]) < 0.01 for t in JOINT_PAIR)
    t_off = (0.002, 0.01, 0.01)
    assert abs(over(JOINT_OFFSET[t_off]) - rows_summed(OFFSET, t_off)) < 0.005
    assert abs(db_hz(ROWS[0]) + 124.0) < 0.05 and abs(db_hz(ROWS[0]) - db_hz(assumed_rms()) - 20.2) < 0.05

    # 3. Reading 3.  Through an offset 0.2% together costs a tenth, and nothing
    #    smaller was measured; a line and a level stand 1%.
    assert allowed(OFFSET, 0) is None and abs(over(OFFSET[0.002][0]) - 0.10) < 0.005
    assert allowed(OFFSET, 1) == 0.01 and allowed(OFFSET, 2) == 0.01
    assert abs(OFFSET[0.005][0][0] - PAIR[0.1][0][0]) < 0.06
    assert abs(OFFSET_E1[0.005][0] - PAIR_E1[0.1][0]) < 0.05

    # 4. Reading 4.  Twenty times, 26 dB, and the same at two rms each.
    r_pair = [reaches(PAIR_E1, x, 0) for x in (0.05, 0.1)]
    r_off = [reaches(OFFSET_E1, x, 0) for x in (0.005, 0.01, 0.02)]
    assert all(abs(x - 1.10) < 0.01 for x in r_pair) and all(abs(x - 21.9) < 0.1 for x in r_off)
    assert round(r_off[1] / r_pair[1]) == 20 and round(20 * math.log10(r_off[1] / r_pair[1])) == 26

    # 5. Reading 5.  A line counts for 0.42, at 10% and at 20%; and the buses.
    assert all(abs(reaches(PAIR_E1, x, 1) - 0.42) < 0.005 for x in (0.1, 0.2))
    assert 1 / 8 < reaches(PAIR_E1, 0.2, 1) < 1
    assert ONE_BUS["one"][1] == ONE_BUS["four"][1]
    assert abs(ONE_BUS["one"][0][0] - ONE_BUS["four"][0][0]) < ONE_BUS["one"][0][1]
    assert ONE_BUS["four"] == (PAIR[0.05][1], PAIR_E1[0.05][1])

    # 6. Reading 6.  5% is 0.2 dB, and a level costs what a line's noise does.
    assert abs(10 * math.log10(1.05) - 0.21) < 0.005
    assert all(abs(PAIR[x][1][0] - PAIR[x][2][0]) < 0.12 for x in (0.01, 0.02, 0.05, 0.1, 0.2))

    # 7. The plan's table, cell by cell, and its densities.
    assert [round(over(PAIR[x][0]), 2) for x in (0.05, 0.1)] == [0.08, 0.25]
    assert [round(over(PAIR[x][1]), 2) for x in (0.05, 0.1)] == [0.06, 0.16]
    assert [round(over(PAIR[x][2]), 2) for x in (0.05, 0.1)] == [0.07, 0.14]
    assert [round(over(OFFSET[x][0]), 2) for x in (0.002, 0.005, 0.01)] == [0.10, 0.20, 0.74]
    assert [round(over(OFFSET[x][1]), 2) for x in (0.01, 0.02)] == [0.00, 0.15]
    assert [round(over(OFFSET[x][2]), 2) for x in (0.01, 0.02)] == [0.03, 0.12]
    assert abs(db_hz(ROWS[0], FS / 2) + 121.0) < 0.05
    assert 2 * source.photodiodes(4) == 512

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
