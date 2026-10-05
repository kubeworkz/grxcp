"""
Open question 1 of board_program_plan.md, its last unpriced item: the watts.

pta_shot_rate.py bounded how fast the tile fires and pta_floorplan.py how large
it is.  Both stopped at the same sentence -- "still not priced is the converters'
power, which needs a device figure" -- and the first of them named one number
as the most leveraged in the model and the first worth replacing with a real
one: the receiver's noise, which B5 assumed at 1 uA.  This prices both from
figures that have been published, and says of each what kind of figure it is.

Six things draw power, and the first is not on the chiplet at all:

  1. the laser        set by the receiver's noise, which is now a measured one
  2. the receivers    an amplifier a column
  3. the ADCs         a converter a column, at the shot rate
  4. the inputs       a modulator's electrode is a capacitor, charged every shot
  5. the weights      a cell's electrode, charged once a batch
  6. the link         every bit that crosses it

MEASURED, on fabricated chips, each from its paper's abstract:
    receiver   M. Atef and H. Zimmermann, IEEE Trans. Circuits Syst. I 60 (2013):
               40 nm CMOS, 1.5 GHz, 7.2 pA/rtHz input-referred up to 1.5 GHz,
               4.1 mW the amplifier (12 mW the chip)
    ADCs       B. Murmann's ADC Performance Survey, 1997 to 2026, from its
               author's spreadsheet as pta_adc_survey.py holds it: the 70
               published converters of at least 7 effective bits at 1 GS/s
               or faster, by the energy each spends a sample.  The best is
               1.11 pJ and the fifth-best 3.14, and one measured at exactly
               1 GS/s is 2.55 mW.
               Three were first taken one at a time from their papers:
               L. Kull et al., IEEE JSSC 48 (2013), 8 bit, 1.2 GS/s, 3.1 mW,
               39.3 dB; H.-Y. Tai et al., A-SSCC 2013, 6 bit, 1.25 GS/s,
               5.3 mW, 37.1 dB; B. Verbruggen et al., IEEE JSSC 45 (2010),
               6 bit, 2.2 GS/s, 2.6 mW, 31.1 dB.  The survey lists the first
               and the third, at those figures.  None of the three reaches 7
               effective bits, and they are kept only as that check.

A STANDARD'S TARGET, not a part: 0.75 to 1.25 pJ a bit for a UCIe standard-
package link at 24 and 32 GT/s (the UCIe Consortium's Hot Chips 2023 tutorial,
its Electrical Summary).  Taken as the whole link's, both ends.

A DESIGN'S OWN ESTIMATE, not a measurement, kept apart and used only to compare:
T.-C. Hsueh, Y. Fainman and B. Lin, arXiv:2402.08192 (2024), a matrix-vector
multiplier of microrings in a 45 nm monolithic silicon-photonics process, 4 bits
at 2 GS/s.  Its Table II at a dimension of 256: 1.10 W of laser, 1.23 W of
heaters, 3.65 W in all, 61 mm2, 27.9 fJ a multiply-accumulate; and for Google's
TPUv4, an 8-bit digital part in 7 nm, 1,142 fJ.

Held by the program already:
    1.96 V cm           TFLT's half-wave voltage-length product (CPU document 4.4)
    pta_shot_rate.py    the laser by B5's method, and v0 and v1 of section 4.3
    pta_chiplet_link.py the bytes a shot moves across the link
    pta_floorplan.py    the die the interface chip sits on
    grx930's `pta_mnist.sh depth`, for what a row is worth in accuracy

ASSUMED, and marked again where each is used:
    - an electrode's capacitance a length.  No source here gives one for TFLT.
      It is swept from 0.5 to 3 pF/cm; 1.5 is what a 50 ohm line has when its
      microwave index is 2.2, C' = n / (c Z0), which is how a travelling-wave
      modulator's line is usually designed and is not a measurement of this one
    - a weight cell's capacitance, pad included: swept from 10 to 100 fF
    - how much of its swing a line moves in a shot.  Random levels move a sixth
      of C V^2 a shot on average; a full swing every other shot moves a half
    - that a converter clocked slower than it was measured draws in
      proportion.  The parts are priced by their energy a sample, P / fs, and
      most of those that can do the tile's job are faster than its shot rate.
      At version 1's seven bits the cheapest part run exactly as it was
      measured, which assumes nothing, is inside the range used
    - the receiver's noise bandwidth: that of a single pole just fast enough to
      settle to half an LSB within a shot.  And that a receiver measured with a
      photodiode beside it keeps its noise with one a bond away
    - one detector, one receiver and one ADC a column.  (The plan's B12 has since
      made it a photodiode a bus a column, four at the working count, on the
      same 64 receivers and ADCs.  What that does to a receiver's noise, and
      what a receiver that averages over a shot would do to the laser, are
      not priced here.)

NOT PRICED: the DACs themselves, apart from what they charge; the interface
chip's weight store, buffers and clocking; calibration; and anything that holds
a ring on its line.  So every total here is a floor.

Standard library only.  Run:  python3 docs/designs/pta_power.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_adc_survey as adc
import pta_chiplet_link as link
import pta_floorplan as fp
import pta_shot_rate as shot
import pta_tw_sweep as sweep

# ---- measured -----------------------------------------------------------------
TIA_NOISE_A_RTHZ, TIA_BW_HZ, TIA_W = 7.2e-12, 1.5e9, 4.1e-3
ADCS = (                    # name, nominal bits, samples a second, watts, SNDR in dB
    ("Kull 2013, 32 nm SOI", 8, 1.2e9, 3.1e-3, 39.3),
    ("Tai 2013, 40 nm", 6, 1.25e9, 5.3e-3, 37.1),
    ("Verbruggen 2010, 40 nm", 6, 2.2e9, 2.6e-3, 31.1),
)

# ---- a standard's target ------------------------------------------------------
UCIE_J_BIT = (0.75e-12, 1.25e-12)

# ---- a design's estimate, for comparison only ---------------------------------
EST = dict(dim=256, laser_w=1.0973, heater_w=1.2312, soc_w=3.6533, mm2=61.12, bits=4,
           clock_hz=2e9, fj_mac=27.9, tia_a_rthz=6.26e-12, tia_w=0.1e-3, adc4_w=1.2e-3)
TPU_FJ_MAC = 1141.9

# ---- assumed ------------------------------------------------------------------
C_LINE_F_CM = (0.5e-12, 1.5e-12, 3.0e-12)
C_LINE_50_OHM = 2.2 / (shot.C_LIGHT * 50.0) / 100.0     # F/cm: n / (c Z0)
C_CELL_F = (10e-15, 100e-15)
ACTIVITY = (1.0 / 6.0, 0.5)

X2_TILE = shot.X2_TILE
FS = shot.X2_FS
MB = link.BIG["mb"]

# ---- what a row is worth, from grx930's depth run -----------------------------
# points lost relaxing v1's row to v0's, at 1 and at 8 hidden layers
ADC_BIT_POINTS = (0.12, 0.20)
RX_NOISE_POINTS = (0.18, 0.22)


# ---- 1. the receiver's noise, and the laser -----------------------------------
def settle_bandwidth_hz(fs, bits):
    """The single-pole bandwidth that settles to half an LSB within one shot."""
    return sweep.settle_s(1.0, bits) * fs


def receiver_noise_a(fs, bits, density=None):
    """Input-referred rms over that pole's noise bandwidth, pi/2 of its corner.
    The measured receiver's density unless another is given."""
    density = TIA_NOISE_A_RTHZ if density is None else density
    return density * math.sqrt(math.pi / 2 * settle_bandwidth_hz(fs, bits))


def laser_w(req, loss_db, noise_a, n=None):
    """B5's method at another receiver: the laser is linear in its noise."""
    n = X2_TILE[1] if n is None else n
    return shot.laser_power(req, FS, n, loss_db, 0.5) * noise_a / shot.RX_NOISE_A


def loss_ceiling_db(req, laser, noise_a, n=None):
    n = X2_TILE[1] if n is None else n
    return shot.loss_ceiling_db(req, laser, n, FS, 0.5) - 10 * math.log10(noise_a / shot.RX_NOISE_A)


# ---- 3. the ADCs ----------------------------------------------------------------
def enob(sndr_db):
    return (sndr_db - 1.76) / 6.02


def walden_j(watts, fs, sndr_db):
    """Joules a conversion step: the part's own figure of merit."""
    return watts / (2 ** enob(sndr_db) * fs)


def adc_w(bits, fs=None):
    """(low, high) watts for one converter: the best and the fifth-best published
    part of at least that many effective bits at that rate or faster, each at
    its own energy a sample."""
    fs = FS if fs is None else fs
    lo, hi = adc.price_j(bits, fs)
    return lo * fs, hi * fs


# ---- 4. the inputs ----------------------------------------------------------------
def input_drive_w(volts, c_f_cm, activity, fs=FS):
    """One modulator's electrode: its length is 1.96 V cm over the swing, so its
    capacitance falls as the swing rises and the power is linear in the swing."""
    length_cm = fp.TFLT_VPI_L_V_CM / volts
    return activity * c_f_cm * length_cm * volts * volts * fs


# ---- 5. the weights ---------------------------------------------------------------
def weight_drive_w(k, n, volts, c_cell_f, activity, fs=FS, mb=MB):
    return k * n * fs / mb * activity * c_cell_f * volts * volts


# ---- 6. the link ------------------------------------------------------------------
def link_bits_s(k, n, resident, fs=None):
    """Bits a second on the lanes, both directions, at the shot rate."""
    fs = FS if fs is None else fs
    into, out = shot.bytes_per_shot(k=k, n=n, resident=resident, **link.BIG)
    return (into + out) * fs * 8 / link.LINK_EFF


def link_w(k, n, resident, fs=None):
    b = link_bits_s(k, n, resident, fs)
    return b * UCIE_J_BIT[0], b * UCIE_J_BIT[1]


# ---- the sum ----------------------------------------------------------------------
def chip_w(k, n, bits, volts, resident, fs=None):
    """(low, high) watts on the interface chip, and the parts, each a (low, high).
    At the plan's shot rate unless another is given (pta_rate.py asks)."""
    fs = FS if fs is None else fs
    a_lo, a_hi = adc_w(bits, fs)
    parts = {
        "receivers": (n * TIA_W, n * TIA_W),
        "ADCs": (n * a_lo, n * a_hi),
        "input drive": (k * input_drive_w(volts, C_LINE_F_CM[0], ACTIVITY[0], fs),
                        k * input_drive_w(volts, C_LINE_F_CM[-1], ACTIVITY[1], fs)),
        "weight drive": (weight_drive_w(k, n, volts, C_CELL_F[0], ACTIVITY[0], fs),
                         weight_drive_w(k, n, volts, C_CELL_F[1], ACTIVITY[1], fs)),
        "link": link_w(k, n, resident, fs),
    }
    return (sum(p[0] for p in parts.values()), sum(p[1] for p in parts.values())), parts


def mac_s(k, n, fs=None):
    return k * n * (FS if fs is None else fs)


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def w(x):
    return f"{x:.2f} W" if x >= 0.995 else f"{x * 1e3:.0f} mW" if x >= 9.95e-3 else f"{x * 1e3:.1f} mW"


def main():
    k, n = X2_TILE
    v0, v1 = shot.REQ["v0"], shot.REQ["v1"]
    bits = v1["adc_bits"]

    print("Open question 1, its last unpriced item: what the tile costs in watts.")
    print(f"{k}x{n} at 1 GS/s under version 1, batch {MB}.  Every total is a floor.")

    # ------------------------------------------------------------------ 1
    section("1. The receiver's noise, measured, and the laser it asks for")
    bw = settle_bandwidth_hz(FS, bits)
    na = receiver_noise_a(FS, bits)
    print(f"  To settle to half a {bits}-bit LSB inside a shot a single pole needs {bw / 1e9:.2f} GHz,")
    print(f"  which the measured receiver has ({TIA_BW_HZ / 1e9:.1f} GHz).  Its {TIA_NOISE_A_RTHZ * 1e12:.1f} pA/rtHz over that")
    print(f"  pole's noise bandwidth is {na * 1e6:.2f} uA rms.  B5 assumed {shot.RX_NOISE_A * 1e6:.0f}.")
    print(f"  {'receiver':<26}{'v1, 10 dB':>11}{'v1, 20 dB':>11}{'v0, 10 dB':>11}{'v0, 20 dB':>11}"
          f"{'loss a 1.6 W laser stands, v1':>32}")
    for name, a in (("B5's, assumed", shot.RX_NOISE_A), ("measured, 40 nm", na),
                    ("the design estimate's", receiver_noise_a(FS, bits, EST["tia_a_rthz"]))):
        cells = "".join(f"{w(laser_w(r, d, a)):>11}" for r in (v1, v0) for d in shot.LOSS_DB)
        print(f"  {name:<26}{cells}{loss_ceiling_db(v1, shot.LASERS_W[-1], a):>29.1f} dB")
    print("  The measured receiver had its photodiode beside it and a limiting")
    print("  amplifier after it.  Whether it keeps that noise a bond away from the")
    print(f"  detector, and stays linear over {bits} bits, its paper does not say.")

    # ------------------------------------------------------------------ 2, 3
    section("2 and 3. The receivers and the ADCs")
    print(f"  {n} receivers at {TIA_W * 1e3:.1f} mW, the measured amplifier alone: {w(n * TIA_W)}.")
    a_lo, a_hi = adc_w(bits)
    able = adc.able(bits, FS)
    best, pub = able[0], adc.as_published(bits, FS)
    print(f"  The survey has {len(able)} published ADCs of at least {bits} effective bits at 1 GS/s or")
    print(f"  faster (pta_adc_survey.py).  By energy a sample the best is {adc.sample_j(best) * 1e12:.2f} pJ --")
    print(f"  {adc.cite(best)}, {best[adc.FS]:.1f} GS/s, {best[adc.SNDR]:.1f} dB, {best[adc.P]:.1f} mW -- and the fifth-best"
          f" {a_hi / FS * 1e12:.2f}.")
    print(f"  Clocked at 1 GS/s that is {a_lo * 1e3:.2f} to {a_hi * 1e3:.2f} mW, ASSUMING a part's power follows its")
    print(f"  clock down.  One needs no assumption: {adc.cite(pub)} is {pub[adc.SNDR]:.1f} dB at {pub[adc.FS]:.2f} GS/s")
    print(f"  for {pub[adc.P]:.2f} mW as measured, which is version 1's converter as a part and not")
    print("  a projection.  The three first taken from their papers, none of them a")
    print(f"  {bits}-bit converter:")
    for name, b, fs, p, s in ADCS:
        print(f"    {name:<24}{b} bit{fs / 1e9:>6.2f} GS/s{p * 1e3:>5.1f} mW  {enob(s):.2f} effective bits"
              f"{walden_j(p, fs, s) * 1e15:>6.0f} fJ a step")
    print(f"  {n} converters at 1 GS/s, of an effective B bits or more:")
    for b in (v0["adc_bits"], v1["adc_bits"], 8):
        lo, hi = adc_w(b)
        print(f"    {b} bits  {lo * 1e3:>5.2f} to {hi * 1e3:>4.2f} mW each  {w(n * lo):>8} to {w(n * hi)}")
    d_lo, d_hi = (n * (adc_w(v1["adc_bits"])[e] - adc_w(v0["adc_bits"])[e]) for e in (0, 1))
    print(f"  So version 1's seventh bit is {w(d_lo)} to {w(d_hi)}.  At the head of the field it")
    print("  is nearly free, because the cheapest converters at this rate are 8-bit")
    print("  designs that reach 7.  Behind the head it doubles the cost.  An eighth")
    e_lo, e_hi = (n * (adc_w(8)[e] - adc_w(v1["adc_bits"])[e]) for e in (0, 1))
    print(f"  bit is not free anywhere: {w(min(e_lo, e_hi))} to {w(max(e_lo, e_hi))} more.")
    print(f"  The same {n} converters at {bits} bits, by shot rate:")
    for fs in (0.1e9, 1e9, 10e9):
        lo, hi = adc_w(bits, fs)
        print(f"    {fs / 1e9:>4.1f} GS/s  {lo / fs * 1e12:>5.2f} to {hi / fs * 1e12:>5.2f} pJ a sample"
              f"  {w(n * lo):>8} to {w(n * hi)}")
    print("  The best part is the same one up to 2.7 GS/s, so that far the ADCs follow")
    print("  the rate.  Past 5 GS/s they do not: ten times the rate is over fifty times")
    print(f"  the power, and only {len(adc.able(bits, 10e9))} published converters do {bits} bits at 10 GS/s at all.")

    # ------------------------------------------------------------------ 4
    section("4. The inputs: an electrode is charged every shot")
    print(f"  A modulator is {fp.TFLT_VPI_L_V_CM} V cm over its swing long, so its capacitance falls as")
    print("  the swing rises and what it draws is LINEAR in the swing.  Capacitance a")
    print(f"  length is ASSUMED: a 50 ohm line at an index of 2.2 has {C_LINE_50_OHM * 1e12:.2f} pF/cm.")
    print(f"  All {k}, in watts, random levels (a sixth of C V^2 a shot) to a full swing")
    print("  every other shot (a half):")
    print(f"  {'swing':>7}{'length':>9}" + "".join(f"{f'{c * 1e12:.1f} pF/cm':>20}" for c in C_LINE_F_CM))
    for v in fp.SWINGS_V:
        cells = "".join(f"{w(k * input_drive_w(v, c, ACTIVITY[0])):>9} to {w(k * input_drive_w(v, c, ACTIVITY[1])):<8}"
                        for c in C_LINE_F_CM)
        print(f"  {v:>5.0f} V{fp.mzm_mm(v):>6.1f} mm{cells}")
    print("  The floorplan found the die shrinks as the swing rises.  The power rises")
    print("  by the same factor, so area times power does not depend on the swing:")
    for v in fp.SWINGS_V:
        a = fp.strip_mm2(k, fp.mzm_mm(v), fp.P_CELL)
        p = k * input_drive_w(v, C_LINE_F_CM[1], ACTIVITY[0])
        print(f"    {v:.0f} V  {a:>6.1f} mm2 x {w(p):>7} = {a * p:.1f} mm2 W")

    # ------------------------------------------------------------------ 5, 6
    section("5 and 6. The weights and the link")
    print(f"  {k * n:,} cells rewritten once a batch of {MB}, each a capacitance that is ASSUMED:")
    for c in C_CELL_F:
        print(f"    {c * 1e15:>4.0f} fF  " + "   ".join(
            f"{v:.0f} V: {w(weight_drive_w(k, n, v, c, ACTIVITY[0]))} to {w(weight_drive_w(k, n, v, c, ACTIVITY[1]))}"
            for v in (2.0, 5.0)))
    print("  The link, at the standard's own target of 0.75 to 1.25 pJ a bit, by where")
    print("  the weights live:")
    for resident in (False, True):
        lo, hi = link_w(k, n, resident)
        print(f"    weights {'resident on the chip' if resident else 're-sent every batch':<22}"
              f"{link_bits_s(k, n, resident) / 1e12:>5.2f} Tb/s  {w(lo)} to {w(hi)}")
    print("  That is both ends of the link, and the standard's target, not a part's.")

    # ------------------------------------------------------------------ 7
    section("7. The interface chip, summed, and what it is against")
    print(f"  {'':<36}{'low':>10}{'high':>10}")
    for label, volts, resident in (("5 V inputs, weights re-sent", 5.0, False),
                                   ("5 V inputs, weights resident", 5.0, True),
                                   ("2 V inputs, weights resident", 2.0, True)):
        (lo, hi), parts = chip_w(k, n, bits, volts, resident)
        print(f"  {label:<36}{w(lo):>10}{w(hi):>10}")
        for name, (a, b) in parts.items():
            print(f"      {name:<32}{w(a):>10}{w(b):>10}")
    print()
    print(f"  A shot is {k * n:,} multiply-accumulates, {mac_s(k, n) / 1e12:.1f} T a second.  Energy for each,")
    print("  chip and laser (measured receiver, 10 to 20 dB) together:")
    for label, volts, resident in (("5 V, re-sent", 5.0, False), ("2 V, resident", 2.0, True)):
        (lo, hi), _ = chip_w(k, n, bits, volts, resident)
        e_lo = (lo + laser_w(v1, shot.LOSS_DB[0], receiver_noise_a(FS, bits))) / mac_s(k, n)
        e_hi = (hi + laser_w(v1, shot.LOSS_DB[1], receiver_noise_a(FS, bits))) / mac_s(k, n)
        print(f"    {label:<16}{e_lo * 1e15:>6.0f} to {e_hi * 1e15:.0f} fJ")
    print(f"  For comparison, and neither is a measurement of a tile like this one: the")
    print(f"  microring design's own estimate is {EST['fj_mac']} fJ at {EST['bits']} bits, and it puts a 7 nm")
    print(f"  digital part at {TPU_FJ_MAC:,.0f} fJ at 8.")
    print()
    print("  As heat, on the floorplan's die (resonant weights, an MZM a row, 25 um):")
    for volts in (5.0, 2.0):
        area = sum(fp.die_mm2(k, n, fp.P_CELL, volts))
        (lo, hi), _ = chip_w(k, n, bits, volts, True)
        print(f"    {volts:.0f} V  {area:>5.1f} mm2  {lo / area:.3f} to {hi / area:.3f} W/mm2"
              f"   (a ring link is reported stable above {fp.RING_LINK_W_MM2})")

    # ------------------------------------------------------------------ 8
    section("8. Section 4.3's two rows that cost watts, priced")
    na = receiver_noise_a(FS, bits)
    print(f"  {'row, v1 relaxed to v0':<34}{'saves':>22}{'costs, in points':>22}")
    print(f"  {'ADC, 7 to 6 bits':<34}{f'{w(d_lo)} to {w(d_hi)}':>22}"
          f"{f'{ADC_BIT_POINTS[0]:.2f} to {ADC_BIT_POINTS[1]:.2f}':>22}")
    s_lo = laser_w(v1, shot.LOSS_DB[0], na) - laser_w(v0, shot.LOSS_DB[0], na)
    s_hi = laser_w(v1, shot.LOSS_DB[1], na) - laser_w(v0, shot.LOSS_DB[1], na)
    print(f"  {'receiver noise, 0.5 to 1 LSB':<34}{f'{w(s_lo)} to {w(s_hi)} of laser':>22}"
          f"{f'{RX_NOISE_POINTS[0]:.2f} to {RX_NOISE_POINTS[1]:.2f}':>22}")
    print("  The points are grx930's, relaxed alone, at one hidden layer and at eight.")

    findings()
    checks()


def findings():
    k, n = X2_TILE
    v0, v1 = shot.REQ["v0"], shot.REQ["v1"]
    bits = v1["adc_bits"]
    na = receiver_noise_a(FS, bits)
    a_lo, a_hi = adc_w(bits)
    (lo_r, hi_r), _ = chip_w(k, n, bits, 5.0, False)
    (lo_s, hi_s), _ = chip_w(k, n, bits, 2.0, True)
    l_re, l_res = link_w(k, n, False), link_w(k, n, True)

    print()
    print("What this says, five readings.")
    print()
    print("1. THE RECEIVER B5 ASSUMED IS NEARLY FOUR TIMES NOISIER THAN ONE THAT HAS")
    print(f"   BEEN BUILT.  A 40 nm receiver measures {na * 1e6:.2f} uA over the bandwidth a shot")
    print(f"   needs, against B5's 1.  The laser is linear in it: {w(laser_w(v1, 10.0, na))} to "
          f"{w(laser_w(v1, 20.0, na))} under")
    print(f"   version 1 and not {w(laser_w(v1, 10.0, shot.RX_NOISE_A))} to {w(laser_w(v1, 20.0, shot.RX_NOISE_A))},"
          f" and a 1.6 W laser stands {loss_ceiling_db(v1, 1.6, na):.1f} dB,")
    print("   which is past the whole of B5's range.  \"The open number is the loss\"")
    print("   was true at an assumed receiver.  At a measured one the loss budget has")
    print("   five or six decibels in hand, if that receiver survives the bond.")
    print()
    conv = (n * TIA_W + n * a_lo, n * TIA_W + n * a_hi)
    print("2. THE CONVERTERS ARE PRICED: 0.33 TO 0.46 W, AND MOST OF IT IS THE RECEIVERS.")
    print(f"   {n} receivers are {w(n * TIA_W)} and {n} seven-bit ADCs {w(n * a_lo)} to"
          f" {w(n * a_hi)}, so {w(conv[0])} to {w(conv[1])}")
    print("   together.  The ADCs are priced from every published converter that can")
    print("   do the job, and the receiver from one paper: it is now the larger of the")
    print("   two and the less well founded.  The ADCs were first priced from three")
    print("   parts read one at a time, at 0.28 to 0.59 W, and none of the three is a")
    print("   7-bit converter.  The seventh bit is nearly free at the head of the field")
    print("   and doubles the ADCs behind it.  The converters are the largest thing on")
    print("   the chip only once the two below have been dealt with.")
    print()
    print("3. TWO THINGS CAN EACH COST MORE THAN ALL THE CONVERTERS.  The link, if the")
    print(f"   weights are re-sent every batch: {w(l_re[0])} to {w(l_re[1])} at the standard's own target,")
    print(f"   against {w(l_res[0])} to {w(l_res[1])} with them resident.  The shot-rate model found")
    print("   residency worth sixteen times the feed; it is worth nearly fourteen times the")
    print("   link's power too, and the weight store that buys it is still unsized.")
    print(f"   And the inputs' drive at a high swing: up to {w(k * input_drive_w(5.0, C_LINE_F_CM[-1], ACTIVITY[1]))}"
          f" at 5 V, at the top of the")
    print("   capacitance and activity swept.  Which of the two leads depends on")
    print("   figures nobody has measured.  That neither is a converter does not.")
    print()
    print("4. THE DRIVE VOLTAGE BUYS AREA WITH POWER, ONE FOR ONE.  A Pockels modulator")
    print("   gets shorter as its swing rises and draws more by the same factor, so")
    print("   the inputs' area times their power does not move.  The floorplan could")
    print("   only say a low-voltage chip does not fit; this says what the high-voltage")
    print("   one pays.  Both rest on a capacitance nobody has measured for TFLT.")
    print()
    e_lo = (lo_s + laser_w(v1, 10.0, na)) / mac_s(k, n)
    e_hi = (hi_r + laser_w(v1, 20.0, na)) / mac_s(k, n)
    print(f"5. A FLOOR OF {lo_s:.1f} TO {hi_r:.0f} W ON THE CHIP, AND {e_lo * 1e15:.0f} TO {e_hi * 1e15:.0f} fJ A"
          f" MULTIPLY-ACCUMULATE with")
    print("   the laser.  That is between a microring design's estimate of itself at")
    print("   four bits and what the same paper charges a digital part at eight.  It")
    print("   is a floor because the DACs, the weight store, the clocks and whatever")
    print("   holds a ring on its line are not in it, and the last of those is the")
    print("   one the floorplan's first reading made unavoidable.")
    print()


def checks():
    """Every claim above, as an assert."""
    k, n = X2_TILE
    v0, v1 = shot.REQ["v0"], shot.REQ["v1"]
    bits = v1["adc_bits"]
    assert (k, n, bits, FS, MB) == (256, 64, 7, 1e9, 64)

    # 1. The receiver: 0.88 GHz to settle seven bits in a nanosecond, inside the
    #    measured part's 1.5; and its noise over that pole is 0.27 uA.
    bw = settle_bandwidth_hz(FS, bits)
    assert abs(bw / 1e9 - 0.883) < 0.001 and bw < TIA_BW_HZ
    na = receiver_noise_a(FS, bits)
    assert abs(na * 1e6 - 0.268) < 0.001, na
    assert 3.7 < shot.RX_NOISE_A / na < 3.8

    # 2. At B5's own receiver laser_w() is pta_shot_rate.py's laser, exactly, so
    #    the method has not moved: only the noise has.
    for req in (v0, v1):
        for d in shot.LOSS_DB:
            assert abs(laser_w(req, d, shot.RX_NOISE_A) - shot.laser_power(req, FS, n, d, 0.5)) < 1e-12

    # 3. The laser at the measured receiver, and the ceiling: 88 mW to 0.88 W
    #    under v1, and a 1.6 W laser stands 22.6 dB, past B5's 20.
    assert abs(laser_w(v1, 10.0, na) - 0.088) < 0.001 and abs(laser_w(v1, 20.0, na) - 0.878) < 0.002
    ceil = loss_ceiling_db(v1, 1.6, na)
    assert abs(ceil - 22.6) < 0.05 and ceil > shot.LOSS_DB[1]
    assert abs(ceil - shot.loss_ceiling_db(v1, 1.6, n, FS, 0.5) - 10 * math.log10(shot.RX_NOISE_A / na)) < 1e-9
    # and the ceiling inverts the laser: at it, the laser is the 1.6 W asked for
    assert abs(laser_w(v1, ceil, na) - 1.6) < 1e-9

    # 4. Each ADC's figure of merit, recomputed from its own power, rate and
    #    SNDR, is the one its paper reports: 34, 73 and 40 fJ a step.
    for (name, _, fs, p, s), want in zip(ADCS, (34, 73, 40)):
        assert abs(walden_j(p, fs, s) * 1e15 - want) < 1.0, (name, walden_j(p, fs, s))
    # None of the three reaches 7 effective bits, which is what the first
    # pricing scaled them to; and the survey's table holds the first of them, at
    # the paper's own figures.
    assert all(enob(s) < 7.0 for _, _, _, _, s in ADCS)
    name, _, fs, p, s = ADCS[0]
    hit = [r for r in adc.TABLE if abs(r[adc.FS] - fs / 1e9) < 0.005 and abs(r[adc.SNDR] - s) < 0.05]
    assert len(hit) == 1 and abs(hit[0][adc.P] / (p * 1e3) - 1.0) < 0.02, (name, hit)

    # 5. The ADCs, from the parts that can do the job: 1.11 to 3.14 mW each at
    #    7 bits and 1 GS/s, 71 to 201 mW for the 64.  The part measured at
    #    exactly this rate is inside that range, so the range does not rest on
    #    clocking a faster part down.
    a_lo, a_hi = adc_w(7)
    assert (a_lo, a_hi) == tuple(j * FS for j in adc.price_j(7, FS))
    assert abs(n * a_lo - 0.071) < 0.001 and abs(n * a_hi - 0.201) < 0.001
    pub = adc.as_published(7, FS)
    assert pub[adc.FS] == 1.0 and a_lo < pub[adc.P] * 1e-3 < a_hi
    # The seventh bit is 3 mW at the head of the field and 106 behind it, and
    # the receivers are more than the ADCs behind them at both ends.
    b_lo, b_hi = adc_w(6)
    assert abs(n * (a_lo - b_lo) - 0.0032) < 0.0005 and abs(n * (a_hi - b_hi) - 0.106) < 0.001
    assert n * TIA_W > n * a_hi
    # an eighth bit is 141 to 197 mW more, and nearly four times the ADCs at the head
    c_lo, c_hi = adc_w(8)
    assert abs(n * (c_hi - a_hi) - 0.141) < 0.001 and abs(n * (c_lo - a_lo) - 0.197) < 0.001
    assert 3.7 < c_lo / a_lo < 3.8
    assert abs(n * (TIA_W + a_lo) - 0.333) < 0.001 and abs(n * (TIA_W + a_hi) - 0.463) < 0.001
    # The ADCs follow the rate to 2.7 GS/s.  At 10 GS/s they are 3.8 to 22 W,
    # over fifty times as much for ten times the rate.
    assert abs(adc_w(7, 2.7e9)[0] / a_lo - 2.7) < 1e-9
    f_lo, f_hi = adc_w(7, 10e9)
    assert abs(n * f_lo - 3.77) < 0.01 and abs(n * f_hi - 22.4) < 0.1
    assert f_lo / a_lo > 50 and f_hi / a_hi > 100

    # 6. The 50 ohm line's capacitance is inside the sweep, and near its middle.
    assert C_LINE_F_CM[0] < C_LINE_50_OHM < C_LINE_F_CM[-1]
    assert abs(C_LINE_50_OHM * 1e12 - 1.47) < 0.01

    # 7. The inputs' power is linear in the swing, and area times power is the
    #    same at every swing.
    for c in C_LINE_F_CM:
        for a in ACTIVITY:
            p1, p5 = input_drive_w(1.0, c, a), input_drive_w(5.0, c, a)
            assert abs(p5 / p1 - 5.0) < 1e-9
    prod = [fp.strip_mm2(k, fp.mzm_mm(v), fp.P_CELL) * k * input_drive_w(v, C_LINE_F_CM[1], ACTIVITY[0])
            for v in fp.SWINGS_V]
    assert max(prod) - min(prod) < 1e-9 * max(prod)

    # 8. The link: 260 and 16 bytes a shot re-sent, 4 and 16 resident, which is
    #    the shot-rate model's own; and residency is worth 13.8 times the power.
    assert shot.bytes_per_shot(k=k, n=n, resident=False, **link.BIG) == (260.0, 16.0)
    assert shot.bytes_per_shot(k=k, n=n, resident=True, **link.BIG) == (4.0, 16.0)
    assert abs(link_bits_s(k, n, False) / link_bits_s(k, n, True) - 13.8) < 1e-9
    assert abs(link_w(k, n, False)[0] - 1.84) < 0.005 and abs(link_w(k, n, False)[1] - 3.07) < 0.005

    # 9. Reading 3.  With weights re-sent the largest part is the link at the low
    #    end of the range and the inputs' drive at the high end -- the first
    #    version of this check said the link at both, and failed -- and at
    #    neither end is it a converter: the link and the drive are over 70% of
    #    the chip between them.  Resident, the link is under a fifth of it.
    _, parts = chip_w(k, n, bits, 5.0, False)
    assert max(parts, key=lambda q: parts[q][0]) == "link"
    assert max(parts, key=lambda q: parts[q][1]) == "input drive"
    for end in (0, 1):
        total = sum(q[end] for q in parts.values())
        assert (parts["link"][end] + parts["input drive"][end]) / total > 0.70
        assert (parts["receivers"][end] + parts["ADCs"][end]) / total < 0.25
        _, res = chip_w(k, n, bits, 5.0, True)
        assert res["link"][end] / sum(q[end] for q in res.values()) < 0.2
    # and resident at 2 V the converters are most of the low end, which is
    # reading 2's last sentence
    _, low = chip_w(k, n, bits, 2.0, True)
    assert (low["receivers"][0] + low["ADCs"][0]) / sum(q[0] for q in low.values()) > 0.55

    # 10. The parts sum to the total, and the total is 0.55 W to under eight.
    for volts, resident in ((5.0, False), (5.0, True), (2.0, True)):
        (lo, hi), parts = chip_w(k, n, bits, volts, resident)
        assert abs(lo - sum(p[0] for p in parts.values())) < 1e-12
        assert 0.55 < lo < hi < 7.7, (volts, resident, lo, hi)

    # 11. Energy a multiply-accumulate sits between the design estimate's 27.9 fJ
    #     at four bits and the 1,142 fJ it gives a digital part at eight.
    (lo_s, _), _ = chip_w(k, n, bits, 2.0, True)
    (_, hi_r), _ = chip_w(k, n, bits, 5.0, False)
    e_lo = (lo_s + laser_w(v1, 10.0, na)) / mac_s(k, n) * 1e15
    e_hi = (hi_r + laser_w(v1, 20.0, na)) / mac_s(k, n) * 1e15
    assert EST["fj_mac"] < e_lo < e_hi < TPU_FJ_MAC, (e_lo, e_hi)
    # and the estimate's own table is self-consistent: its watts over its rate
    assert abs(EST["soc_w"] / (EST["dim"] ** 2 * EST["clock_hz"]) * 1e15 - EST["fj_mac"]) < 0.1

    # 12. As heat the chip is under a fifth of what a ring link is reported
    #     stable above, at the floorplan's densest die.
    area = sum(fp.die_mm2(k, n, fp.P_CELL, 5.0))
    (_, hi), _ = chip_w(k, n, bits, 5.0, True)
    assert hi / area < fp.RING_LINK_W_MM2 / 5

    print("All checks pass.")


if __name__ == "__main__":
    main()
