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
    ADCs       L. Kull et al., IEEE JSSC 48 (2013): 8 bit, 1.2 GS/s, 3.1 mW,
               39.3 dB SNDR, 32 nm SOI
               H.-Y. Tai et al., A-SSCC 2013: 6 bit, 1.25 GS/s, 5.3 mW, 37.1 dB
               peak SNDR, 40 nm
               B. Verbruggen et al., IEEE JSSC 45 (2010): 6 bit, 2.2 GS/s,
               2.6 mW, 31.1 dB SNDR, 40 nm
               They are single converters near this operating point, not a
               survey: B. Murmann's ADC Performance Survey is the place to
               widen this, and was not read here.

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
    - that an ADC of B bits costs what the figure of merit says at an effective
      B bits.  The figure of merit is the measured parts' own, P / (2^ENOB fs),
      and it does not stay put across resolutions; this spans the three
    - the receiver's noise bandwidth: that of a single pole just fast enough to
      settle to half an LSB within a shot.  And that a receiver measured with a
      photodiode beside it keeps its noise with one a bond away
    - one detector, one receiver and one ADC a column

NOT PRICED: the DACs themselves, apart from what they charge; the interface
chip's weight store, buffers and clocking; calibration; and anything that holds
a ring on its line.  So every total here is a floor.

Standard library only.  Run:  python3 docs/designs/pta_power.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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


def fom_range():
    f = [walden_j(w, fs, s) for _, _, fs, w, s in ADCS]
    return min(f), max(f)


def adc_w(bits, fs, fom_j):
    return fom_j * 2 ** bits * fs


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
def link_bits_s(k, n, resident):
    """Bits a second on the lanes, both directions, at the shot rate."""
    into, out = shot.bytes_per_shot(k=k, n=n, resident=resident, **link.BIG)
    return (into + out) * FS * 8 / link.LINK_EFF


def link_w(k, n, resident):
    b = link_bits_s(k, n, resident)
    return b * UCIE_J_BIT[0], b * UCIE_J_BIT[1]


# ---- the sum ----------------------------------------------------------------------
def chip_w(k, n, bits, volts, resident):
    """(low, high) watts on the interface chip, and the parts, each a (low, high)."""
    f_lo, f_hi = fom_range()
    parts = {
        "receivers": (n * TIA_W, n * TIA_W),
        "ADCs": (n * adc_w(bits, FS, f_lo), n * adc_w(bits, FS, f_hi)),
        "input drive": (k * input_drive_w(volts, C_LINE_F_CM[0], ACTIVITY[0]),
                        k * input_drive_w(volts, C_LINE_F_CM[-1], ACTIVITY[1])),
        "weight drive": (weight_drive_w(k, n, volts, C_CELL_F[0], ACTIVITY[0]),
                         weight_drive_w(k, n, volts, C_CELL_F[1], ACTIVITY[1])),
        "link": link_w(k, n, resident),
    }
    return (sum(p[0] for p in parts.values()), sum(p[1] for p in parts.values())), parts


def mac_s(k, n):
    return k * n * FS


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
    print("  The measured ADCs, and the figure of merit each one has:")
    for name, b, fs, p, s in ADCS:
        print(f"    {name:<24}{b} bit{fs / 1e9:>6.2f} GS/s{p * 1e3:>5.1f} mW  {enob(s):.2f} effective bits"
              f"{walden_j(p, fs, s) * 1e15:>6.0f} fJ a step")
    f_lo, f_hi = fom_range()
    print(f"  At {f_lo * 1e15:.0f} to {f_hi * 1e15:.0f} fJ a step, an effective B bits at 1 GS/s, and {n} of them:")
    for b in (v0["adc_bits"], v1["adc_bits"], 8):
        lo, hi = adc_w(b, FS, f_lo), adc_w(b, FS, f_hi)
        print(f"    {b} bits  {lo * 1e3:>5.1f} to {hi * 1e3:>4.1f} mW each  {w(n * lo):>8} to {w(n * hi)}")
    d_lo = n * (adc_w(v1["adc_bits"], FS, f_lo) - adc_w(v0["adc_bits"], FS, f_lo))
    d_hi = n * (adc_w(v1["adc_bits"], FS, f_hi) - adc_w(v0["adc_bits"], FS, f_hi))
    print(f"  So version 1's seventh bit is {w(d_lo)} to {w(d_hi)}: as much again as the six.")

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
    f_lo, f_hi = fom_range()
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
    conv = (n * TIA_W + n * adc_w(bits, FS, f_lo), n * TIA_W + n * adc_w(bits, FS, f_hi))
    print("2. THE CONVERTERS ARE PRICED: HALF A WATT TO UNDER ONE.")
    print(f"   {n} receivers are {w(n * TIA_W)} and {n} seven-bit ADCs {w(n * adc_w(bits, FS, f_lo))} to"
          f" {w(n * adc_w(bits, FS, f_hi))}, so {w(conv[0])} to {w(conv[1])}")
    print("   together.  The seventh bit is as much again as the first six, because a")
    print("   converter's power doubles with a bit.  They are the largest thing on the")
    print("   chip only once the two below have been dealt with.")
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
    f_lo, f_hi = fom_range()

    # 5. A bit doubles a converter, so the seventh costs what the six do.
    for f in (f_lo, f_hi):
        assert abs(adc_w(7, FS, f) - 2 * adc_w(6, FS, f)) < 1e-15
    assert abs(n * adc_w(7, FS, f_lo) - 0.28) < 0.005 and abs(n * adc_w(7, FS, f_hi) - 0.59) < 0.01

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
    assert (low["receivers"][0] + low["ADCs"][0]) / sum(q[0] for q in low.values()) > 0.7

    # 10. The parts sum to the total, and the total is a watt to about nine.
    for volts, resident in ((5.0, False), (5.0, True), (2.0, True)):
        (lo, hi), parts = chip_w(k, n, bits, volts, resident)
        assert abs(lo - sum(p[0] for p in parts.values())) < 1e-12
        assert 0.7 < lo < hi < 12.0, (volts, resident, lo, hi)

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
