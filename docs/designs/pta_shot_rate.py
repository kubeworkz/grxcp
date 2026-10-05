"""
Open question 1 of board_program_plan.md, narrowed: how fast can the tile fire?

The PTA chiplet's OPTICAL shot rate -- how often the tile takes one analog
matrix-vector product -- is the number X2, the laser (B5) and the interface
chip's area all wait on, and nothing has fixed it.  It is also easily mistaken
for a different number.  B8 records the confusion: a modulator's line rate in
Gb/s is the link's question, not the tile's.  A table of NRZ benchmarks -- 56
Gbaud in production, 100 to 180 Gb/s in research -- was offered on 2026-10-03 as
the tile's shot rate, and section 1 says how far apart the two are.

Three things bound the shot rate, and this prices each from numbers the program
already holds:

  1. the modulator   an analog level has to SETTLE, to half an LSB, and there is
                     no equaliser and no FEC to rescue one that has not
  2. the feed        every shot consumes operands that crossed the die-to-die
                     link (X2), so the link's bandwidth caps the rate
  3. the receiver    every detector needs enough light to clear its receiver's
                     noise (B5), and that noise grows with the rate

Reused, not restated:
    pta_tw_sweep.py      settle_s() and the 45 GHz TFLN bandwidth it is anchored
                         on (C. Wang et al., Nature 562, 101, 2018)
    pta_chiplet_link.py  X2's traffic accounting, its 57.6 GB/s module, its
                         candidate geometries and its illustrative dense layer

From board_program_plan.md:
    section 4.3   v0 and v1 of the interface chip's requirements: ADC bits,
                  and receiver noise and photons in LSB of an 8-bit ADC -- which
                  for v1 is half an LSB and 15 photons, not the 0.25 and 30 this
                  model first read (see REQ)
    B5            the receiver it takes "for scale": about 1 uA rms of input
                  noise, 1 A/W, 1550 nm, 64 channels, 10 to 20 dB of loss

Assumed here, and marked again where each is used:
    - that B5's 1 uA is the receiver's noise AT 1 GS/s.  B5 quotes no rate for
      it; 1 GS/s is where it makes its comparison, so that is where this anchors
    - how receiver noise scales with the rate.  Nothing has designed the
      receiver, so two laws bracket it rather than one being claimed
    - one detector per tile column

Section 5 turns the rate into what it asks of the interface chip's weight path:
how many cells a beat have to be written for the tile to spend its time shooting.
That is section 2.1's Tw at the chiplet's scale.  The write parallelism is swept
rather than assumed, and a write beat is taken as one shot period.

Section 6 is the light source, as requirements and not as a choice of laser:
the same total light delivered by one laser or by one emitter per input row, the
intensity noise the source may have, and what the wavelength costs.  It assumes
flat intensity noise over a noise bandwidth equal to the shot rate, and it holds
that noise to the RECEIVER's allowance -- an assumption about a budget section 4.3
does not have, because the error model has no term for the source at all.

Not priced: the converters' power, and any laser.  The ADC array is stated as a
conversion rate, and the source as what it is asked for, because turning either
into a device needs a figure this program does not hold.

What KIND of source the tile needs turns on how a column sums, and the documents
answer that only by implication.  The error model's crosstalk is written for a
ring bank (pta_cpu_integration.md 4.3), where inputs are told apart by wavelength
and a column sums powers; that document's section 8 calls the topology a
hypothesis with no ground truth.  Section 6 says what follows from each reading
and claims neither.

A3's photon counts are deliberately not used.  They are photons at the all-optical
activation's knee -- the nonlinear element's budget on a branch the mainline does
not take -- and not what a detector needs.

Standard library only.  Run:  python3 docs/designs/pta_shot_rate.py
"""
import math
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_chiplet_link as link
import pta_tpaqcn_measured as tpaqcn
import pta_tw_sweep as sweep

# ---- section 4.3: what the interface chip is held to ----------------------
# v1 is the requirement.  v0 is the looser set B5's laser figure was computed
# from, so reproducing that figure is how this model is checked; it costs 1.5
# points on the D3 network where v1 costs a quarter of one.
#
# Both noise rows are in LSB OF AN 8-BIT ADC.  v0's were measured at one.  v1's
# were not: X1 ran them as "--adcbits 7 --thermal 0.25 --photons 30", which is
# in LSB of that 7-bit ADC, and one of those is two of an 8-bit one's.  So v1 is
# half an 8-bit LSB of receiver noise and 15 photons per 8-bit LSB.  This model
# first read "0.25 LSB" and "30" as 8-bit LSB, which made its laser twice too
# large; grx930's `pta_mnist.sh budget` prints every run's noise in this unit
# and is where the two figures come from.
REQ = {
    "v0": dict(adc_bits=6, rx_noise_lsb=1.0, photons_per_lsb=3),
    "v1": dict(adc_bits=7, rx_noise_lsb=0.5, photons_per_lsb=15),
}
NOISE_LSB_BITS = 8

# ---- B5: the receiver "for scale" -----------------------------------------
RX_NOISE_A = 1e-6           # input-referred, rms.  B5's figure.
RX_NOISE_AT_HZ = 1e9        # ASSUMED: the rate that figure belongs to (see above)
RESPONSIVITY_A_W = 1.0      # B5's figure
WAVELENGTH_M = 1550e-9      # B5's figure
LOSS_DB = (10.0, 20.0)      # B5's range, laser to detector
B5_CHANNELS = 64

# ---- how receiver noise scales with the rate ------------------------------
# rms noise ~ rate ** exponent.  No receiver has been designed, so neither is
# claimed; they bracket it.
#   0.5   a fixed front end with white input noise, band-limited to the rate
#   1.5   a front end redesigned for each rate at the transimpedance limit: its
#         feedback resistance falls as rate^2, so its noise density rises with it
# Above 1 GS/s the first is the kinder; below it, the second.
NOISE_LAWS = (("white", 0.5), ("redesigned", 1.5))

# ---- a laser "of that order" ----------------------------------------------
# B5 calls 0.16 to 1.6 W "a board-level thermal and safety item".  Those are the
# powers swept: not a budget, which nothing has set, but the order B5 planned on.
LASERS_W = (0.16, 0.5, 1.6)

H_PLANCK, C_LIGHT = 6.62607015e-34, 2.99792458e8
Q_ELECTRON = 1.602176634e-19

# ---- the source ------------------------------------------------------------
# Emitter powers swept in decades.  NOT devices: this program holds no laser's
# output power, so the table says what loss each decade could stand and leaves
# the comparison with a real part to whoever has its datasheet.
EMITTERS_W = (1e-3, 10e-3, 100e-3)
O_BAND_M = 1310e-9          # the other telecom window, for the comparison only
# The all-optical activation's phase matching: FWHM in pump wavelength, read off
# the TPA-QCN paper's Fig. 3C (pta_tpaqcn_measured.py), about its 1550 nm pump.
PM_FWHM_NM = tpaqcn.FWHM_NM
PM_HALF_ARG = 1.3915574     # sinc^2(x) = 1/2 at this x; its FWHM is 2.783

# ---- X2's working point ---------------------------------------------------
X2_TILE = (256, 64)
X2_FS = 1e9                 # the shot rate X2's tables are quoted at, and since
                            # 2026-10-05 the working rate: the board plan's B11
X2_MODULES = 5              # what its 256x64 tile needs at 1 GS/s, batch 64
B8_FLIP_MODULES = 30        # where B8 says its own decision reopens


# ---- 1. the modulator ------------------------------------------------------
def modulator_rate(bw_hz, bits):
    """Shots a second if the settle were the whole shot: an upper bound."""
    return 1.0 / sweep.settle_s(bw_hz, bits)


# ---- 2. the feed -----------------------------------------------------------
def bytes_per_shot(kin, nout, mb, k, n, resident):
    """(in, out) bytes crossing the link per shot.

    Resident means the weights are already on the interface chip, which B4 says
    it holds and step MB built: they crossed once, at load, and a batch moves
    only its activations in and its results out.
    """
    t = link.layer_traffic(kin, nout, mb, k, n)
    into = t["acts"] if resident else t["into"]
    return into / t["shots"], t["out"] / t["shots"]


def feed_rate(kin, nout, mb, k, n, mods, resident):
    """The shot rate at which `mods` modules saturate in the busier direction."""
    into, out = bytes_per_shot(kin, nout, mb, k, n, resident)
    return mods * link.MODULE_GBS * 1e9 / max(into, out)


# ---- 3. the receiver -------------------------------------------------------
def detector_power(req, fs, exponent):
    """Light one detector needs at full scale, in watts, by B5's method.

    The receiver's rms noise has to sit at the allowed fraction of an 8-bit LSB,
    so full scale is that noise scaled up by 2^8 and by the allowance.
    """
    noise_a = RX_NOISE_A * (fs / RX_NOISE_AT_HZ) ** exponent
    full_scale_a = noise_a / req["rx_noise_lsb"] * 2 ** NOISE_LSB_BITS
    return full_scale_a / RESPONSIVITY_A_W


def shot_noise_power(req, fs):
    """The light the photon allowance alone would ask for, in watts."""
    photons = req["photons_per_lsb"] * 2 ** NOISE_LSB_BITS
    return photons * H_PLANCK * C_LIGHT / WAVELENGTH_M * fs


def laser_power(req, fs, channels, loss_db, exponent):
    return channels * detector_power(req, fs, exponent) * 10 ** (loss_db / 10)


def receiver_rate(req, laser_w, channels, loss_db, exponent):
    """The shot rate a laser supports: laser_power() solved for the rate."""
    at_anchor = laser_power(req, RX_NOISE_AT_HZ, channels, loss_db, exponent)
    return RX_NOISE_AT_HZ * (laser_w / at_anchor) ** (1.0 / exponent)


def loss_ceiling_db(req, laser_w, channels, fs, exponent):
    """The most loss a laser can stand and still light every detector at fs."""
    return 10 * math.log10(laser_w / (channels * detector_power(req, fs, exponent)))


NINE_TENTHS = Fraction(9, 10)

# ---- 5. the weight path ---------------------------------------------------
def weight_updates_per_s(k, n, mb, fs):
    """Cells written a second: a set of k*n is replaced once per batch of shots."""
    return k * n * fs / mb


def duty_one_bank(k, n, mb, per_beat):
    """The share of its time a one-bank tile spends shooting.

    A set of k*n cells goes in `per_beat` at a time, a beat being one shot
    period, and is then shot mb times.  Nothing overlaps: the bank being written
    is the bank being read.
    """
    return mb / (mb + math.ceil(k * n / per_beat))


def per_beat_one_bank(k, n, mb, want):
    """Cells a beat for a one-bank tile to reach a duty of `want`, a Fraction.

    A set takes a whole number of beats, so this is not mb / (mb + k*n/p) solved
    for p: it is the most beats a set may take, then the width that fits in them.
    The first version solved the continuous form and returned widths that fell a
    beat short -- 9,217 at batch 16, which takes two beats and reaches 88.9%.
    """
    beats = int(Fraction(mb) * (1 - want) / want)        # floor: the most allowed
    if beats < 1:
        return None                                      # not reachable at all
    return -(-(k * n) // beats)


def per_beat_two_banks(k, n, mb):
    """Cells a beat for a two-bank tile never to wait.

    With a second bank the next set loads behind the current set's shots, so the
    duty is 1 as long as a set programs within one batch: k*n cells in mb beats.
    """
    return math.ceil(k * n / mb)


# ---- 6. the source ---------------------------------------------------------
def per_emitter_power(req, fs, k, n, loss_db, exponent):
    """Light each emitter supplies when the source is one emitter per input row.

    The total is laser_power()'s: every detector still needs its full scale, so
    an array changes how the light is made and not how much of it there is.
    """
    return laser_power(req, fs, n, loss_db, exponent) / k


def emitter_loss_ceiling_db(req, emitter_w, k, n, fs, exponent):
    """The most loss an array of k emitters of this power can stand at fs."""
    return loss_ceiling_db(req, k * emitter_w, n, fs, exponent)


def rin_limit_db_hz(req, fs):
    """The intensity noise that puts the source at the receiver's allowance.

    Intensity noise is a fraction of the signal, so it is largest at full scale:
    rms / full scale = sqrt(RIN * B).  Held to the receiver's allowance there,
    with RIN flat and the noise bandwidth B equal to the shot rate -- both
    assumed.  Halving B moves this 3 dB.
    """
    rel = req["rx_noise_lsb"] / 2 ** NOISE_LSB_BITS
    return 10 * math.log10(rel * rel / fs)


def participation(contributions):
    """How many independent emitters' noise averages at one detector.

    With one source every row's light fluctuates together and the sum carries the
    source's whole relative noise.  With an independent emitter a row the noises
    add in power while the signals add in amplitude, so the sum's relative noise
    falls by the square root of this count: 1 when one row carries the sum, the
    number of rows when they all carry it equally.
    """
    total = sum(contributions)
    return total * total / sum(c * c for c in contributions)


def quantum_efficiency(responsivity_a_w, wavelength_m):
    return responsivity_a_w * H_PLANCK * C_LIGHT / (Q_ELECTRON * wavelength_m)


def responsivity_at(eta, wavelength_m):
    return eta * Q_ELECTRON * wavelength_m / (H_PLANCK * C_LIGHT)


def phase_match_scale(offset_nm):
    """The all-optical activation's efficiency at a pump offset from its peak.

    sinc^2, with half its peak at half the measured FWHM -- the shape A3's chain
    mode uses for a detuned unit, here against wavelength.
    """
    x = PM_HALF_ARG * abs(offset_nm) / (PM_FWHM_NM / 2)
    return 1.0 if x == 0 else (math.sin(x) / x) ** 2


def phase_match_window_nm(scale):
    """The pump offset at which the efficiency has fallen to `scale`."""
    lo, hi = 0.0, PM_FWHM_NM            # monotone on the main lobe out to here
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if phase_match_scale(mid) > scale:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gs(rate_hz):
    """A rate in GS/s, at a width that reads from MS/s to tens of GS/s."""
    g = rate_hz / 1e9
    return f"{g:.3g}" if g >= 0.01 else f"{g:.2g}"


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    v0, v1 = REQ["v0"], REQ["v1"]
    k, n = X2_TILE
    big = link.BIG

    print("Open question 1, narrowed: three bounds on the tile's optical shot rate.")
    print("A shot is one analog matrix-vector product.  It is not a bit on a link.")

    # ------------------------------------------------------------------ 1
    section("1. The modulator: how soon an analog level has settled")
    print("  A binary link is recovered by an equaliser and protected by FEC, so it")
    print("  runs with its eye nearly closed.  An analog level has neither: it has")
    print("  to be within half an LSB when it is read, and nothing downstream can")
    print("  repair one that was not.  Single pole, as pta_tw_sweep.py has it:")
    print(f"  {'bandwidth':>10}  " + "  ".join(f"{b}-bit settle / rate" for b in (6, 7, 8)))
    for bw in (10e9, 30e9, sweep.TFLN_BW_HZ, 100e9):
        cells = []
        for bits in (6, 7, 8):
            cells.append(f"{sweep.settle_s(bw, bits) * 1e12:>6.1f} ps /"
                         f" {modulator_rate(bw, bits) / 1e9:>5.1f} GS/s")
        mark = "  <- the plan's anchor" if bw == sweep.TFLN_BW_HZ else ""
        print(f"  {bw / 1e9:>7.0f} GHz  " + "  ".join(cells) + mark)
    mod_fs = modulator_rate(sweep.TFLN_BW_HZ, v1["adc_bits"])
    print(f"  At v1's {v1['adc_bits']}-bit ADC the anchor allows {gs(mod_fs)} GS/s:"
          f" {mod_fs / sweep.TFLN_BW_HZ:.2f} shots per hertz of bandwidth.")
    print("  Only the 45 GHz row is a device.  The rest is a sweep: 30 to 100 GHz is")
    print("  what the offered benchmarks quote (not verified here), and nothing below")
    print("  depends on them, because this bound is the loosest of the three.")
    print("  (pta_rate.py, 2026-10-05, found a fourth bound of this kind and far tighter:")
    print("  the weight ring's own line, 0.44 to 2.2 GS/s for a ring that swings on 1 to 5 V.)")

    # ------------------------------------------------------------------ 2
    section(f"2. The feed: what the link can deliver ({big['kin']}x{big['nout']} layer,"
            f" batch {big['mb']})")
    print(f"  one module = {link.MODULE_GBS:.1f} GB/s a direction (X2).  Weights either")
    print("  cross with every batch, as X2's tables have them, or are resident on the")
    print("  interface chip (B4, step MB) and only activations and results cross.")
    print(f"  {'tile':<10}{'re-sent: B/shot':>16}{'GS/s a module':>15}"
          f"{'resident: B/shot':>18}{'GS/s a module':>15}{'modules at 1 GS/s':>19}")
    for gk, gn in link.GEOMETRIES:
        ri, ro = bytes_per_shot(k=gk, n=gn, resident=False, **big)
        si, so = bytes_per_shot(k=gk, n=gn, resident=True, **big)
        f_re = feed_rate(k=gk, n=gn, mods=1, resident=False, **big)
        f_rs = feed_rate(k=gk, n=gn, mods=1, resident=True, **big)
        into, _ = link.rates(k=gk, n=gn, fs=X2_FS, **big)
        print(f"  {f'{gk}x{gn}':<10}{max(ri, ro):>16.3g}{f_re / 1e9:>15.3g}"
              f"{max(si, so):>18.3g}{f_rs / 1e9:>15.3g}{link.modules(into):>19}")
    f5 = feed_rate(k=k, n=n, mods=X2_MODULES, resident=False, **big)
    f30 = feed_rate(k=k, n=n, mods=B8_FLIP_MODULES, resident=False, **big)
    fr1 = feed_rate(k=k, n=n, mods=1, resident=True, **big)
    print(f"  The {k}x{n} tile with weights re-sent: {gs(f5)} GS/s at X2's"
          f" {X2_MODULES} modules, and {gs(f30)} at the")
    print(f"  {B8_FLIP_MODULES} where B8 reopens.  With weights resident, {gs(fr1)}"
          f" GS/s a module -- and it is the")
    print("  OUTBOUND direction that binds then, four bytes a result.")
    print("  The batch moves the re-sent bound and not the resident one:")
    for mb in (16, 64, 256):
        b2 = dict(big, mb=mb)
        print(f"    batch {mb:>3}: re-sent"
              f" {gs(feed_rate(k=k, n=n, mods=1, resident=False, **b2)):>6} GS/s a module,"
              f" resident {gs(feed_rate(k=k, n=n, mods=1, resident=True, **b2))}")

    # ------------------------------------------------------------------ 3
    section("3. The receiver: how much light a detector needs, by B5's own method")
    p0 = detector_power(v0, X2_FS, 0.5)
    sn0 = shot_noise_power(v0, X2_FS)
    lo0, hi0 = (laser_power(v0, X2_FS, B5_CHANNELS, d, 0.5) for d in LOSS_DB)
    print(f"  B5, reproduced (v0's allowance, {B5_CHANNELS} channels, 1 GS/s): a detector's")
    print(f"  full scale is {p0 * 1e3:.2f} mW, the photon allowance alone would ask"
          f" {sn0 * 1e6:.2f} uW,")
    print(f"  and the laser is {lo0:.2f} to {hi0:.1f} W behind {LOSS_DB[0]:.0f} to"
          f" {LOSS_DB[1]:.0f} dB.  Those are B5's figures.")
    p1 = detector_power(v1, X2_FS, 0.5)
    lo1, hi1 = (laser_power(v1, X2_FS, B5_CHANNELS, d, 0.5) for d in LOSS_DB)
    print(f"  v1 allows the receiver {v1['rx_noise_lsb']} of that LSB,"
          f" not {v0['rx_noise_lsb']:.0f}:")
    print(f"  full scale {p1 * 1e3:.2f} mW a detector, and the laser {lo1:.2f} to"
          f" {hi1:.1f} W -- {lo1 / lo0:.0f}x B5's figure.")
    print()
    print("  By geometry, the laser for 1 GS/s under v1 (one detector a column):")
    print(f"  {'tile':<10}{'detectors':>10}{'at 10 dB':>11}{'at 20 dB':>11}")
    for gk, gn in link.GEOMETRIES:
        a, b = (laser_power(v1, X2_FS, gn, d, 0.5) for d in LOSS_DB)
        print(f"  {f'{gk}x{gn}':<10}{gn:>10}{a:>9.2f} W{b:>9.1f} W")
    print()
    print(f"  The most loss a laser can stand and still run the {k}x{n} tile at"
          f" 1 GS/s (v1):")
    for w in LASERS_W:
        print(f"    {w:>5.2f} W  {loss_ceiling_db(v1, w, n, X2_FS, 0.5):>5.1f} dB")
    print(f"  B5 planned on {LOSS_DB[0]:.0f} to {LOSS_DB[1]:.0f} dB.  At 1 GS/s the"
          f" scaling law does not enter, so these")
    print("  are the firmest numbers here.")
    print()
    print(f"  Away from 1 GS/s the law matters.  The {k}x{n} tile under v1, in GS/s,")
    print("  as the range the two noise laws span (white noise is the low end below")
    print("  1 GS/s and the high end above it):")
    print(f"  {'laser':>8}" + "".join(f"{f'{d:.0f} dB':>22}" for d in (10, 15, 20)))
    for w in LASERS_W:
        cells = []
        for d in (10.0, 15.0, 20.0):
            a = receiver_rate(v1, w, n, d, NOISE_LAWS[0][1])
            b = receiver_rate(v1, w, n, d, NOISE_LAWS[1][1])
            cells.append(f"{gs(min(a, b)):>9} to {gs(max(a, b)):<9}")
        print(f"  {w:>6.2f} W" + "".join(cells))
    print("  Under white noise each 3 dB of loss costs a factor of four in rate.")

    # ------------------------------------------------------------------ 4
    section(f"4. Which bound binds, at X2's point: {k}x{n}, batch {big['mb']},"
            f" {X2_MODULES} modules, v1")
    print(f"  the modulator allows {gs(mod_fs)} GS/s throughout")
    print(f"  {'weights':<10}{'laser':>8}{'loss':>7}{'feed':>9}{'receiver':>20}"
          f"   binds")
    for resident in (False, True):
        ff = feed_rate(k=k, n=n, mods=X2_MODULES, resident=resident, **big)
        for w in (LASERS_W[0], LASERS_W[-1]):
            for d in LOSS_DB:
                r = [receiver_rate(v1, w, n, d, e) for _, e in NOISE_LAWS]
                lo, hi = min(r), max(r)
                if hi < min(ff, mod_fs):
                    who = "receiver"
                elif lo > min(ff, mod_fs):
                    who = "feed" if ff < mod_fs else "modulator"
                else:
                    who = "feed or receiver"
                print(f"  {'resident' if resident else 're-sent':<10}{w:>6.2f} W"
                      f"{d:>5.0f} dB{gs(ff):>9}{gs(lo):>9} to {gs(hi):<8}   {who}")

    # ------------------------------------------------------------------ 5
    section(f"5. The weight path: what a shot rate asks of the interface chip"
            f" ({k}x{n})")
    print(f"  A weight set is {k * n:,} cells, shot for one batch and then replaced."
          f"  At 1 GS/s:")
    print(f"  {'batch':>7}{'sets a second':>16}{'cells a second':>17}"
          f"{'each DAC rewritten at':>24}")
    for mb in (16, 64, 256):
        upd = weight_updates_per_s(k, n, mb, X2_FS)
        print(f"  {mb:>7}{X2_FS / mb / 1e6:>13.1f} M{upd / 1e9:>15.0f} G"
              f"{X2_FS / mb / 1e6:>20.1f} MHz")
    print("  Cells a second is X2's inbound seen from the tile: at a byte a weight it")
    print("  is the link's weight traffic, to the byte.")
    print()
    print("  ONE BANK.  The share of its time the tile spends shooting, by how many")
    print("  cells are written a beat (a beat is one shot period):")
    modes = ((1, "1, the c930's own scan"), (n, f"{n}, one input's row"),
             (k, f"{k}, one output's column"), (k * n, f"{k * n:,}, all at once"))
    print(f"  {'cells a beat':<28}" + "".join(f"{f'batch {mb}':>12}" for mb in (16, 64, 256)))
    for per, label in modes:
        print(f"  {label:<28}" + "".join(
            f"{duty_one_bank(k, n, mb, per):>11.1%} " for mb in (16, 64, 256)))
    print()
    print("  What it takes to stop waiting, in cells a beat:")
    print(f"  {'batch':>7}{'one bank, 90% duty':>21}{'two banks, no wait':>21}")
    for mb in (16, 64, 256):
        print(f"  {mb:>7}{per_beat_one_bank(k, n, mb, NINE_TENTHS):>21,}"
              f"{per_beat_two_banks(k, n, mb):>21,}")
    print(f"  At batch 16 one bank has to write the whole set in a single beat: two")
    print("  beats of programming against sixteen of shooting is already under 90%.")
    print(f"  And the converters: {n} ADCs at the shot rate is {n * X2_FS / 1e9:.0f} GS/s of"
          f" {v1['adc_bits']}-bit conversion at 1 GS/s.")

    # ------------------------------------------------------------------ 6
    section(f"6. The source: what the tile asks of its light ({k}x{n}, v1)")
    print("  POWER.  The light is reading 3's whichever way it is made.  One laser")
    print(f"  has to supply all of it; an emitter a row, {k} of them, a share each:")
    print(f"  {'rate':>9}{'loss':>8}{'one laser':>12}{f'each of {k}':>14}")
    for d in LOSS_DB:
        # At 1 GS/s the noise law does not enter, so these need no bracket.
        tot = laser_power(v1, X2_FS, n, d, 0.5)
        each = per_emitter_power(v1, X2_FS, k, n, d, 0.5)
        print(f"  {gs(X2_FS):>4} GS/s{d:>5.0f} dB{tot:>10.2f} W{each * 1e3:>11.1f} mW")
    print()
    print("  By geometry, at 1 GS/s:")
    print(f"  {'tile':<10}{'emitters':>9}{'each at 10 dB':>15}{'each at 20 dB':>15}")
    for gk, gn in link.GEOMETRIES:
        a, b = (per_emitter_power(v1, X2_FS, gk, gn, d, 0.5) for d in LOSS_DB)
        print(f"  {f'{gk}x{gn}':<10}{gk:>9}{a * 1e3:>12.1f} mW{b * 1e3:>12.1f} mW")
    print()
    print(f"  The loss an emitter can stand at 1 GS/s, by its power.  A sweep in")
    print("  decades and not a list of parts: no laser's output is held here.")
    for w in EMITTERS_W:
        print(f"    {w * 1e3:>5.0f} mW a row  {emitter_loss_ceiling_db(v1, w, k, n, X2_FS, 0.5):>5.1f} dB")
    print()
    print("  NOISE.  The error model has quantisation, thermal, shot, drift, crosstalk")
    print("  and programming error, and nothing for the source.  Intensity noise is a")
    print("  fraction of the signal, so it is worst at full scale; held there to the")
    print("  receiver's own allowance, flat, over a bandwidth equal to the shot rate:")
    print(f"  {'rate':>9}{'v1':>14}{'v0':>14}")
    for fs in (0.1e9, 1e9, 10e9):
        print(f"  {gs(fs):>4} GS/s{rin_limit_db_hz(v1, fs):>8.1f} dB/Hz"
              f"{rin_limit_db_hz(v0, fs):>8.1f} dB/Hz")
    print("  It does not depend on the laser's power or on the loss: it is a ratio.")
    print("  Read the other way it is a fourth bound on the rate, once a source's")
    print("  figure is known -- every 10 dB of intensity noise is a decade of rate.")
    relief = 10 * math.log10(k)
    print(f"  An emitter a row relaxes it, by between 0 dB (one row carries the sum)")
    print(f"  and {relief:.1f} dB (all {k} carry it equally), if the emitters are independent.")
    print()
    print("  WAVELENGTH.  B5's arithmetic is at 1550 nm with 1 A/W, which is a quantum")
    eta = quantum_efficiency(RESPONSIVITY_A_W, WAVELENGTH_M)
    r_o = responsivity_at(eta, O_BAND_M)
    print(f"  efficiency of {eta:.2f}.  The same detector at 1310 nm gives {r_o:.3f} A/W, and")
    print(f"  the laser is linear in that: {RESPONSIVITY_A_W / r_o:.3f}x the light for the same current.")
    w90, w50 = phase_match_window_nm(0.9), phase_match_window_nm(0.5)
    print(f"  Two things would pin it, and only the first can be priced here.  The")
    print(f"  all-optical branch: its activation is phase-matched at 1550 nm over")
    print(f"  {PM_FWHM_NM:.0f} nm, so its pump has to sit within {w90:.1f} nm of the peak for 90% of")
    print(f"  the efficiency and within {w50:.1f} for half.  And the weights, if they are")
    print("  rings: B5 lists the laser's wavelength among what the tile is sensitive")
    print("  to and gives no figure for it.")
    print()
    print("  KIND.  What sort of source it must be turns on how a column sums, and the")
    print("  documents answer that only by implication.  The error model's crosstalk")
    print("  is written for a ring bank -- an input's light passes its neighbours'")
    print("  rings (CPU document 4.3) -- and in a ring bank the inputs are told apart")
    print("  by wavelength and a column sums POWERS.  Such a tile does not merely")
    print(f"  allow a source of many lines, it requires one: a line an input, {k} here.")
    print("  Whether one bank can tell that many apart is the device's question, and")
    print("  no figure for it is held.  A mesh sums FIELDS from one coherent source,")
    print("  and there an array or a comb is no source at all.  The CPU document's")
    print("  section 8 calls the ring-bank topology a hypothesis with no ground truth,")
    print("  so this is the model's assumption followed through, not a decision.")
    print("  (pta_source.py, 2026-10-05, follows each reading to its source at the working")
    print("  geometry and rate.  A ring bank's is a comb of 43 to 86 lines reused across buses,")
    print("  and not a line an input.  The plan's B12 took the ring bank the same day.)")

    findings(mod_fs)
    checks()


def findings(mod_fs):
    v1 = REQ["v1"]
    k, n = X2_TILE
    big = link.BIG
    lo1, hi1 = (laser_power(v1, X2_FS, n, d, 0.5) for d in LOSS_DB)
    f5 = feed_rate(k=k, n=n, mods=X2_MODULES, resident=False, **big)
    fr1 = feed_rate(k=k, n=n, mods=1, resident=True, **big)
    ceil = loss_ceiling_db(v1, LASERS_W[-1], n, X2_FS, 0.5)

    print()
    print("What this says, six readings.")
    print()
    print("1. THE MODULATOR IS NOT THE QUESTION.  It allows about"
          f" {mod_fs / 1e9:.0f} GS/s, and at")
    print("   B5's order of laser the receiver never allows half of that, so")
    print("   the modulator is never the bound that binds.  A modulator benchmark --")
    print("   a line rate least of all, since it is recovered by machinery an analog")
    print("   level does not have -- says nothing about how fast this tile fires.")
    print()
    wbytes = link.layer_traffic(k=k, n=n, **big)["weights"]
    print("2. THE FEED DEPENDS ON WHERE THE WEIGHTS LIVE, by a factor of sixteen.")
    print(f"   Re-sent with every batch, X2's {X2_MODULES} modules carry {gs(f5)} GS/s."
          f"  Resident on")
    print(f"   the interface chip, one module carries {gs(fr1)}.  So the feed need not be")
    print("   what holds the tile to 1 GS/s -- but that is not the residency B4 lists.")
    print(f"   B4's is a DAC-held voltage per weight ON THE TILE, {k * n:,} of them at")
    print(f"   this geometry (X2).  Holding a layer is a digital store of {wbytes / 1e6:.1f} MB")
    print("   for this one, which B4 does not list and nothing has sized.")
    print()
    print("3. THE RECEIVER IS, AND B5's LASER IS HALF WHAT v1 NEEDS from the")
    print("   receiver it assumed.  B5 sized it from v0's allowance of one LSB of")
    print("   receiver noise; section 4.3 holds the chip to v1's half.")
    print(f"   At 1 GS/s the {k}x{n} tile needs {lo1:.2f} W behind 10 dB and {hi1:.1f} W")
    print("   behind 20, not 0.16 and 1.6.  Every figure here is linear in B5's")
    print("   assumed 1 uA of receiver noise, which makes that the most leveraged")
    print("   number in the model and the first one worth replacing with a real one.")
    print()
    print("4. SO THE OPEN NUMBER IS THE LOSS, NOT THE RATE.  With the largest laser")
    print(f"   B5 planned on, 1 GS/s holds only if laser-to-detector loss stays under")
    print(f"   {ceil:.1f} dB -- and B5's own range runs to 20.  Every 3 dB past that")
    print("   costs between a factor of 1.6 and a factor of 4 in rate, depending on a")
    print("   receiver nobody has designed.  The question to put to a foundry is the")
    print("   loss budget; the shot rate follows from it.")
    print()
    serial = duty_one_bank(k, n, big['mb'], 1)
    one90 = per_beat_one_bank(k, n, big['mb'], NINE_TENTHS)
    two = per_beat_two_banks(k, n, big['mb'])
    print("5. AND A SHOT RATE IS ONLY A THROUGHPUT IF THE WEIGHTS KEEP UP.  The c930")
    print(f"   scans one weight a cycle; at that rate this tile would shoot {serial:.1%} of")
    print(f"   the time at batch {big['mb']}, so the emulation's weight path does not carry")
    print(f"   over.  One bank needs {one90:,} cells a beat for 90% duty.  A second bank")
    print(f"   needs {two} -- one output's column a shot period -- and never waits, which")
    print("   is the case for two banks that G1 left unpriced.  The batch is the lever")
    print("   here too: every quadrupling of it quarters the write path.")
    print()
    e_lo, e_hi = (per_emitter_power(v1, X2_FS, k, n, d, 0.5) for d in LOSS_DB)
    print("6. A SOURCE IS HELD TO FOUR THINGS, AND THE PLAN HAD ONE OF THEM.  Power:")
    print(f"   reading 3's light is one laser of {lo1:.2f} to {hi1:.1f} W or {k} emitters of")
    print(f"   {e_lo * 1e3:.1f} to {e_hi * 1e3:.1f} mW, and a 10 mW emitter a row stands"
          f" {emitter_loss_ceiling_db(v1, 10e-3, k, n, X2_FS, 0.5):.1f} dB.  Noise: the")
    print(f"   error model has no term for the source; at the receiver's own allowance")
    print(f"   it is {rin_limit_db_hz(v1, X2_FS):.0f} dB/Hz at 1 GS/s, ten tighter per decade of rate.")
    print("   Wavelength: 18% more light at 1310 nm for the same detector; pinned to")
    print(f"   1550 within {phase_match_window_nm(0.9):.1f} nm if the all-optical branch reopens; and held against")
    print("   the weights if they are rings, by a figure nobody has.")
    print("   And KIND, which is the one that decides the rest: the error model is")
    print("   written for a ring bank, which sums powers and needs a line per input,")
    print("   so under the model as it stands an array or a comb is the kind of source")
    print("   required and a single-line laser is not one.  The CPU document calls that")
    print("   topology a hypothesis.  It is now a hypothesis with a laser hanging on it.")
    print()


def checks():
    """Every claim above, as an assert."""
    v0, v1 = REQ["v0"], REQ["v1"]
    k, n = X2_TILE
    big = link.BIG

    # 1. The settle is the one pta_tw_sweep.py prints: 22 ps at 45 GHz, 8 bits.
    assert abs(sweep.settle_s(sweep.TFLN_BW_HZ, 8) * 1e12 - 22.0) < 0.5

    # 2. X2's own numbers, which B8 quotes: inbound at 1 GS/s by batch, and the
    #    module count at batch 64.  If these move, this model's feed has drifted
    #    from the link model's.
    for mb, want in ((16, 1028), (64, 260), (256, 68)):
        into, _ = link.rates(k=k, n=n, fs=X2_FS, **dict(big, mb=mb))
        assert round(into) == want, (mb, into)
    into, _ = link.rates(k=k, n=n, fs=X2_FS, **big)
    assert link.modules(into) == X2_MODULES, link.modules(into)

    # 3. With weights re-sent this is X2's max_shot_rate, exactly.
    for gk, gn in link.GEOMETRIES:
        a = feed_rate(k=gk, n=gn, mods=3, resident=False, **big)
        b = link.max_shot_rate(k=gk, n=gn, mods=3, **big)
        assert abs(a - b) <= 1e-6 * b, (gk, gn, a, b)

    # 4. Residency relaxes the feed sixteen-fold at X2's point, and moves the
    #    binding direction from inbound to outbound.
    re = feed_rate(k=k, n=n, mods=1, resident=False, **big)
    rs = feed_rate(k=k, n=n, mods=1, resident=True, **big)
    assert abs(rs / re - 16.25) < 0.01, rs / re
    si, so = bytes_per_shot(k=k, n=n, resident=True, **big)
    assert so > si, (si, so)

    # 5. B5's figures, reproduced from v0: 0.26 mW a detector, about 0.1 uW from
    #    the photon allowance, 0.16 to 1.6 W of laser, and "three orders" between
    #    the receiver's limit and the shot-noise one.
    p0 = detector_power(v0, X2_FS, 0.5)
    assert abs(p0 * 1e3 - 0.256) < 1e-9, p0
    sn0 = shot_noise_power(v0, X2_FS)
    assert 0.09e-6 < sn0 < 0.11e-6, sn0
    assert p0 / sn0 > 1000, p0 / sn0
    lo0, hi0 = (laser_power(v0, X2_FS, B5_CHANNELS, d, 0.5) for d in LOSS_DB)
    assert round(lo0, 2) == 0.16 and round(hi0, 1) == 1.6, (lo0, hi0)

    # 6. v1 doubles it, exactly, and at 1 GS/s the scaling law does not enter.
    #    v1's two noise rows are X1's as run -- a quarter of a 7-bit LSB and 30
    #    photons per 7-bit LSB -- restated in the 8-bit LSB every figure here uses.
    assert v1["rx_noise_lsb"] == 0.25 * 2 ** (NOISE_LSB_BITS - v1["adc_bits"])
    assert v1["photons_per_lsb"] == 30 / 2 ** (NOISE_LSB_BITS - v1["adc_bits"])
    for _, e in NOISE_LAWS:
        lo1 = laser_power(v1, X2_FS, B5_CHANNELS, LOSS_DB[0], e)
        assert abs(lo1 / lo0 - 2.0) < 1e-9, (e, lo1 / lo0)

    # 7. Shot noise never becomes the limit: across four decades of rate and both
    #    laws the receiver asks for more light than the photon allowance does.
    for fs in (1e7, 1e8, 1e9, 1e10, 1e11):
        for _, e in NOISE_LAWS:
            assert detector_power(v1, fs, e) > shot_noise_power(v1, fs), (fs, e)

    # 8. The modulator never binds at X2's tile: at every laser B5 planned on and
    #    every loss in its range, under either law, the receiver allows less than
    #    half of what the modulator does.
    mod_fs = modulator_rate(sweep.TFLN_BW_HZ, v1["adc_bits"])
    for w in LASERS_W:
        for d in LOSS_DB:
            for _, e in NOISE_LAWS:
                assert receiver_rate(v1, w, n, d, e) * 2 < mod_fs, (w, d, e)

    # 9. The loss ceiling for 1 GS/s at B5's largest laser sits INSIDE B5's loss
    #    range, which is reading 4: part of that range cannot reach X2's rate.
    ceil = loss_ceiling_db(v1, LASERS_W[-1], n, X2_FS, 0.5)
    assert LOSS_DB[0] < ceil < LOSS_DB[1], ceil
    assert abs(ceil - 16.9) < 0.05, ceil

    # 10. receiver_rate() inverts laser_power(): a round trip returns the rate.
    for _, e in NOISE_LAWS:
        for fs in (1e8, 1e9, 1e10):
            w = laser_power(v1, fs, n, 15.0, e)
            assert abs(receiver_rate(v1, w, n, 15.0, e) / fs - 1.0) < 1e-9, (e, fs)

    # 11. The weight path's rate IS X2's inbound weight traffic: cells a second
    #     at a byte a weight equals the link's inbound less its activations.
    for mb in (16, 64, 256):
        b2 = dict(big, mb=mb)
        t = link.layer_traffic(k=k, n=n, **b2)
        into, _ = link.rates(k=k, n=n, fs=X2_FS, **b2)
        acts_gbs = t["acts"] / (t["shots"] / X2_FS) / 1e9
        upd = weight_updates_per_s(k, n, mb, X2_FS) * link.W_BYTES / 1e9
        assert abs(upd - (into - acts_gbs)) < 1e-6 * into, (mb, upd, into, acts_gbs)

    # 12. A serial scan leaves the tile shooting under half a percent of the time
    #     at batch 64, and writing everything at once leaves one beat a batch.
    assert duty_one_bank(k, n, 64, 1) < 0.005
    assert abs(duty_one_bank(k, n, 64, k * n) - 64 / 65) < 1e-12

    # 13. per_beat_one_bank() meets the duty it was asked for, and one cell fewer
    #     does not: the requirement is tight, not padded.  The first version of
    #     that function failed exactly this, which is how its error was found.
    for mb in (16, 64, 256):
        p = per_beat_one_bank(k, n, mb, NINE_TENTHS)
        assert Fraction(mb, mb + -(-(k * n) // p)) >= NINE_TENTHS, (mb, p)
        assert Fraction(mb, mb + -(-(k * n) // (p - 1))) < NINE_TENTHS, (mb, p)
    # At batch 16 that is the whole set in one beat.
    assert per_beat_one_bank(k, n, 16, NINE_TENTHS) == k * n

    # 14. Two banks need at most a ninth of what one bank needs for 90%, and at
    #     batch 64 that is exactly one column of the tile a beat.
    assert per_beat_two_banks(k, n, 64) == k, per_beat_two_banks(k, n, 64)
    for mb in (16, 64, 256):
        assert (per_beat_one_bank(k, n, mb, NINE_TENTHS)
                >= 9 * per_beat_two_banks(k, n, mb)), mb

    # 15. Quadrupling the batch quarters the write path, in both cases.
    assert per_beat_two_banks(k, n, 16) == 4 * per_beat_two_banks(k, n, 64)
    assert per_beat_two_banks(k, n, 64) == 4 * per_beat_two_banks(k, n, 256)

    # 16. An array changes how the light is made, not how much: k emitters'
    #     shares add back to the one laser, at every geometry and loss.
    for gk, gn in link.GEOMETRIES:
        for d in LOSS_DB:
            each = per_emitter_power(v1, X2_FS, gk, gn, d, 0.5)
            assert abs(gk * each - laser_power(v1, X2_FS, gn, d, 0.5)) < 1e-12, (gk, gn, d)

    # 17. Ten times the emitter is ten decibels of loss, and a 10 mW emitter a row
    #     at X2's tile stands 18.9 dB -- past the 16.9 that B5's largest single
    #     laser could.
    c1, c10, c100 = (emitter_loss_ceiling_db(v1, w, k, n, X2_FS, 0.5) for w in EMITTERS_W)
    assert abs((c10 - c1) - 10.0) < 1e-9 and abs((c100 - c10) - 10.0) < 1e-9
    assert abs(c10 - 18.9) < 0.05, c10
    assert c10 > ceil

    # 18. The intensity-noise limit: about -144 dB/Hz at 1 GS/s under v1, exactly
    #     10 dB a decade of rate, and 20 log10(2) looser under v0's allowance.
    assert abs(rin_limit_db_hz(v1, X2_FS) + 144.2) < 0.05, rin_limit_db_hz(v1, X2_FS)
    assert abs(rin_limit_db_hz(v1, 1e8) - rin_limit_db_hz(v1, 1e9) - 10.0) < 1e-9
    assert abs(rin_limit_db_hz(v0, X2_FS) - rin_limit_db_hz(v1, X2_FS)
               - 20 * math.log10(2)) < 1e-9

    # 19. The participation count runs from 1 to the number of rows, which is
    #     the whole range of what an array can buy on noise.
    assert abs(participation([1.0] + [0.0] * (k - 1)) - 1.0) < 1e-12
    assert abs(participation([1.0] * k) - k) < 1e-9
    assert 1.0 < participation([float(i + 1) for i in range(k)]) < k

    # 20. B5's 1 A/W at 1550 nm is a quantum efficiency of 0.80, and at the same
    #     efficiency the O-band costs the ratio of the wavelengths in light.
    eta = quantum_efficiency(RESPONSIVITY_A_W, WAVELENGTH_M)
    assert abs(eta - 0.80) < 0.005, eta
    r_o = responsivity_at(eta, O_BAND_M)
    assert abs(RESPONSIVITY_A_W / r_o - WAVELENGTH_M / O_BAND_M) < 1e-12
    assert abs(responsivity_at(eta, WAVELENGTH_M) - RESPONSIVITY_A_W) < 1e-12

    # 21. The phase-matching curve is half its peak at half the measured FWHM,
    #     which is what FWHM means, and falls monotonically to there.
    assert abs(phase_match_scale(PM_FWHM_NM / 2) - 0.5) < 1e-6
    assert abs(phase_match_window_nm(0.5) - PM_FWHM_NM / 2) < 1e-6
    assert 0 < phase_match_window_nm(0.9) < phase_match_window_nm(0.5)
    assert abs(phase_match_window_nm(0.9) - 2.4) < 0.05, phase_match_window_nm(0.9)

    print("checks: all pass.")


if __name__ == '__main__':
    main()
