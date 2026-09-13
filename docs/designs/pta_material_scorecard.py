"""
The materials question, put to the models this repository already trusts.

A photonic tile needs a material for two different jobs, and each job already
has a model here:

  activation  pta_tpaqcn_review.md sections 4.4-4.5: the input power at which
              half the fundamental converts, linear loss divided out, times the
              pulse (pta_tpaqcn_measured.knee_power), against a 5 pJ digital
              O-E-O path and an analog optoelectronic neuron.
  weights     pta_cpu_integration.md section 2.1: the loop-order cost model
              (pta_tw_sweep), with the c930 as host at 100 MHz.

For each job this prints where every candidate lands, then the spec a new
material would have to meet to change the answer.  That spec is the useful
form of "design a better material": a target, not a guess.

Activation candidates:
    TPA-QCN      eta_L 29 %/W/cm^2 at 20 dB/cm; 5 dB/cm projected once leakage
                 is removed (Sci. Adv. 12, eaeg3170, 2026)
    compound 4   chi_31 ~16 pm/V, twice TPA-QCN, measured in films
                 (arXiv:2511.13682).  That its films guide and lose like
                 TPA-QCN's is assumed, not measured.
    TFLN         periodically poled, ~5000 %/W/cm^2, as the review uses it; and
                 a measured all-optical switch at 80 fJ and ~46 fs
                 (Q. Guo et al., Nat. Photonics 16, 625, 2022), whose switching
                 energy is close to, not identical with, the knee

Weight candidates:
    thermo-optic silicon, 24.77 mW per pi, 130 kHz -3 dB bandwidth
                 (N. C. Harris et al., Opt. Express 22, 10487, 2014)
    TFLN         Pockels, 45 GHz for a 20 mm device (C. Wang et al.,
                 Nature 562, 101, 2018)
    TFLT         Pockels, 40 GHz and 1.96 V cm (C. Wang et al., Nature 629,
                 784, 2024).  Quadrature-biased side by side for 46 hours,
                 TFLN's output power fluctuated 5 dB and TFLT's under 1 dB
                 (K. Powell et al., Opt. Express 32, 44115, 2024).
    BTO          non-volatile ferroelectric phase shifters: 80 ns to switch and
                 560 nW per pi static, in a 58-cell mesh (Nat. Photonics 2026,
                 arXiv:2601.07456); 4.6 pJ per switch across eight levels
                 (J. Geler-Kremer et al., Nat. Photonics, 2022)

Not scored:
    HZO          a non-volatile shift is shown but not yet reversible
                 (Nat. Commun. 15, 3549, 2024), so it is not yet a weight
    TPA-QCN      as a weight: no modulator has been demonstrated
    BTO Pockels  r42 = 923 pm/V (Nat. Mater. 18, 42, 2019) is a low-frequency
                 figure with strong dispersion (Nat. Mater., 2025), and nothing
                 in hand gives a settle time

Assumed, not measured: a single-pole response behind every bandwidth, settling
to half an 8-bit LSB; phases spread evenly over [0, pi], so a shifter holds
half its per-pi power on average; and drift_rad()'s reading of a power
fluctuation as phase.

Standard library only.  Run:  python3 docs/designs/pta_material_scorecard.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_tpaqcn_measured as meas
import pta_tw_sweep as sweep

# ---- activation ------------------------------------------------------------
CHI31_C4 = 16.0                          # pm/V, compound 4
TFLN_ETA = 50.0                          # /W/cm^2 (~5000 %/W/cm^2)
LOSS_BUDGET_DB = 3.0                     # per activation, review section 4.3
GUO_SWITCH_J = 80e-15
PULSES = (("10 ps", 10e-12), ("1 ps", 1e-12), ("100 fs", 100e-15))
NEURONS = ((50, 1.5), (10, 1.0), (1, 0.5))       # fF, V: today's to nanoscale
TARGET_FF = 10                           # the neuron the specs are written against

# ---- weights ---------------------------------------------------------------
THERMO_P_PI, THERMO_BW = 24.77e-3, 130e3         # W, Hz
TFLN_BW, TFLT_BW = 45e9, 40e9                    # Hz
BTO_SWITCH_S, BTO_STATIC_PI, BTO_SWITCH_J = 80e-9, 560e-9, 4.6e-12
DRIFT_TEST_H = 46
TFLN_DRIFT_DB, TFLT_DRIFT_DB = 5.0, 1.0          # TFLT's is an upper bound

SHIFTERS = sweep.NUM_ROWS * sweep.NUM_COLS       # one 8x8 weight set
SETS = sweep.NT * sweep.KT                       # sets an M=64 N=8 K=256 GEMM uses
TS_NS = sweep.CYCLE_NS                           # a shot takes one host cycle
TD_NS = sweep.TD * sweep.CYCLE_NS
SCAN_NS = sweep.SCAN * sweep.CYCLE_NS


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def neuron_j(c_ff, volts):
    """Light to deliver the charge C V, plus C V^2 from the supply (review 4.5)."""
    q = c_ff * 1e-15 * volts
    return q / meas.Q_E * meas.E_PHOTON + q * volts


def kappa_tpaqcn():
    """kappa0 recovered from the measured device, as pta_tpaqcn_measured does."""
    eta0 = meas.ETA_L / meas.loss_factor(meas.ALPHA_DB, meas.L_DEV_CM)
    return math.sqrt(eta0 * 1e4)


def walkoff_dng(tau_s, L_mm):
    """Largest group-index mismatch that walks off less than the pulse."""
    return tau_s * meas.C_LIGHT / (L_mm * 1e-3)


def drift_rad(fluct_db):
    """
    Phase drift read from the output-power fluctuation of a quadrature-biased
    Mach-Zehnder, P = P0 (1 - sin d) / 2.  Two readings bracket it: the
    fluctuation as a swing symmetric about quadrature, or as a fall from it.
    """
    r = 10 ** (fluct_db / 10)
    return math.asin((r - 1) / (r + 1)), math.asin(1 - 1 / r)


def fmt_power(w):
    for unit, scale in (("W", 1.0), ("mW", 1e-3), ("uW", 1e-6), ("nW", 1e-9)):
        if w >= scale:
            return f"{w / scale:.3g} {unit}"
    return f"{w:.3g} W"


def fmt_time(seconds, digits=3):
    if seconds == 0:
        return "0"
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9), ("ps", 1e-12)):
        if seconds >= scale:
            return f"{seconds / scale:.{digits}g} {unit}"
    return f"{seconds:.{digits}g} s"


def faster(tw_ns):
    """GEMM time in the faster loop order, and which order wins by how much."""
    a = sweep.shipped(tw_ns, TS_NS, TD_NS)
    b = sweep.interchanged(tw_ns, TS_NS, TD_NS)
    if b < a:
        return b, f"interchange, {a / b:.3g}x"
    return a, f"as shipped, {b / a:.3g}x"


def main():
    section("1. What an activation has to beat (review 4.5)")
    neurons = {c: neuron_j(c, v) for c, v in NEURONS}
    target = neurons[TARGET_FF]
    print(f"  digital O-E-O path, per activation      {meas.fmt_energy(meas.OEO_J):>9}")
    for c, v in NEURONS:
        print(f"  analog neuron, {c:>2} fF at {v:.1f} V            {meas.fmt_energy(neurons[c]):>9}")

    section("2. Activation candidates through the review's energy model")
    kap0 = kappa_tpaqcn()
    kap4 = kap0 * CHI31_C4 / meas.CHI31
    candidates = (
        ("TPA-QCN as built", kap0, 2.0, meas.ALPHA_DB),
        ("TPA-QCN, leakage removed", kap0, 6.0, 5.0),
        ("compound 4, leakage removed", kap4, 6.0, 5.0),
        ("TFLN-class PPLN, lossless", math.sqrt(TFLN_ETA * 1e4), 10.0, 0.0),
    )
    print("  energy per activation at each pulse; the last two columns are for 100 fs:")
    print(f"  energy over the {meas.fmt_energy(target)} neuron's, and the largest group-index"
          " mismatch that pulse tolerates")
    print(f"  {'candidate':<28}{'L mm':>5}{'loss':>8}{'P_knee':>9}"
          + "".join(f"{name:>10}" for name, _ in PULSES)
          + f"{'/ neuron':>10}{'dn_g':>8}")
    knee = {}
    for name, kappa, L_mm, loss in candidates:
        P = knee[name] = meas.knee_power(L_mm, kappa, loss, loss)
        print(f"  {name:<28}{L_mm:5.1f}{loss * L_mm / 10:5.1f} dB{fmt_power(P):>9}"
              + "".join(f"{meas.fmt_energy(P * tau):>10}" for _, tau in PULSES)
              + f"{P * 100e-15 / target:9.3g}x{walkoff_dng(100e-15, L_mm):8.4f}")
    ideal = knee["TFLN-class PPLN, lossless"] * 100e-15
    print(f"\n  measured TFLN switch (Guo 2022): {meas.fmt_energy(GUO_SWITCH_J)} at ~46 fs"
          f" = {GUO_SWITCH_J / ideal:.0f}x the idealised row at 100 fs,")
    print(f"  {GUO_SWITCH_J / neurons[10]:.1f}x the 10 fF neuron and"
          f" {GUO_SWITCH_J / neurons[50]:.2f}x today's 50 fF one: a peer of the analog"
          " neuron, not a successor")

    section("3. Spec: the chi_31 a TPA-QCN-like film needs, 3 dB of loss per activation")
    print("  The knee goes as 1/kappa^2 at fixed length and loss, and kappa as chi(2) in a")
    print("  fixed guide, so each target is a square root away from compound 4.  chi_31 is")
    print("  in the TPA-QCN papers' convention: compare within it, not with LN's d33.")
    print(f"\n  {'loss':>11}{'L':>10}{'pulse':>8}{'dn_g <=':>9}"
          f"{'to tie the ' + meas.fmt_energy(target) + ' neuron':>26}{'to beat it 10x':>24}")
    for loss in (meas.ALPHA_DB, 5.0, 1.0, 0.3):
        L_mm = LOSS_BUDGET_DB / loss * 10
        P = meas.knee_power(L_mm, kap4, loss, loss)
        for pname, tau in PULSES[1:]:
            tie = CHI31_C4 * math.sqrt(P * tau / target)
            ten = CHI31_C4 * math.sqrt(P * tau / (target / 10))
            print(f"  {loss:5.1f} dB/cm{L_mm:7.1f} mm{pname:>8}{walkoff_dng(tau, L_mm):9.4f}"
                  f"{tie:13.1f} pm/V ({tie / meas.CHI31:4.1f}x){ten:13.1f} pm/V"
                  f" ({ten / meas.CHI31:4.1f}x)")
    print("  (x: multiples of TPA-QCN's 8.1 pm/V.)  Cutting loss n-fold buys what an n-fold")
    print("  chi(2) does, but it is spent as length, and length costs group-velocity matching")
    print("  and fabrication tolerance: the phase-matching acceptance narrows as 1/L (review 5.1).")

    section("4. Weight candidates")
    weights = (
        # name, settle or switch time (s), W held per pi (None: a held voltage), drift dB
        ("thermo-optic silicon", sweep.settle_s(THERMO_BW), THERMO_P_PI, None),
        ("TFLN Pockels", sweep.settle_s(TFLN_BW), None, TFLN_DRIFT_DB),
        ("TFLT Pockels", sweep.settle_s(TFLT_BW), None, TFLT_DRIFT_DB),
        ("BTO non-volatile", BTO_SWITCH_S, BTO_STATIC_PI, None),
    )
    print(f"  {'material':<22}{'settles':>9}  {'holds with':<16}"
          f"{'drift in ' + str(DRIFT_TEST_H) + ' h':>15}{'4-bit LSBs':>12}{'6-bit LSBs':>12}")
    for name, settle, p_pi, drift_db in weights:
        holds = f"{fmt_power(p_pi)}/pi" if p_pi else "a held voltage"
        if drift_db is None:
            drift = lsb4 = lsb6 = "-"
        else:
            lo, hi = drift_rad(drift_db)
            bound = "< " if drift_db == TFLT_DRIFT_DB else ""
            drift = f"{bound}{lo:.2f}-{hi:.2f} rad"
            lsb4 = f"{bound}{lo / (math.pi / 16):.1f}-{hi / (math.pi / 16):.1f}"
            lsb6 = f"{bound}{lo / (math.pi / 64):.0f}-{hi / (math.pi / 64):.0f}"
        print(f"  {name:<22}{fmt_time(settle):>9}  {holds:<16}"
              f"{drift:>15}{lsb4:>12}{lsb6:>12}")
    print(f"  BTO rewrites for {meas.fmt_energy(BTO_SWITCH_J)} a switch and holds eight levels;"
          "\n  its drift is reported stable, not quantified.")

    section("5. Weights through section 2.1: c930 host, M=64 N=8 K=256, one-cycle shots")
    print(f"  {'material':<22}{'weights arrive by':<18}{'Tw':>9}{'GEMM':>10}"
          f"  {'faster order':<20}{'holding one GEMM':>17}")
    for name, settle, p_pi, _ in weights:
        settle_ns = settle * 1e9
        modes = (("scan, two banks", SCAN_NS + settle_ns, SHIFTERS),
                 ("bank select", settle_ns, SHIFTERS),
                 ("a mesh per set", 0.0, SETS * SHIFTERS))
        for i, (mode, tw_ns, held) in enumerate(modes):
            t, order = faster(tw_ns)
            hold = fmt_power(held * p_pi / 2) if p_pi else f"{held:,} voltages"
            print(f"  {name if i == 0 else '':<22}{mode:<18}{fmt_time(tw_ns * 1e-9):>9}"
                  f"{fmt_time(t * 1e-9, 4):>10}  {order:<20}{hold:>17}")

    section("6. Spec: what a weight material has to do on this host")
    bw_min = math.log(2 ** 9) / (2 * math.pi * TS_NS * 1e-9)
    inside = [n for n, s, _, _ in weights if s * 1e9 < TS_NS]
    print(f"  bank select costs nothing only if it settles inside one host cycle,"
          f" {TS_NS:.0f} ns:")
    print(f"    >= {bw_min / 1e6:.0f} MHz single-pole at 8 bits.  Inside: {', '.join(inside)}")
    tw_star = sweep.break_even(TD_NS)
    print(f"  below Tw* = {tw_star:.1f} ns the shipped loop order wins, above it the"
          f" interchange;\n    BTO's {BTO_SWITCH_S * 1e9:.0f} ns sits on it")
    floor = sweep.shipped(0.0, TS_NS, TD_NS)
    print(f"  once resident, every material's GEMM is {fmt_time(floor * 1e-9)}:"
          "\n    the host's feed and row writes, not the material")
    p_pi_1w = 2 * 1.0 / (SETS * SHIFTERS)
    print(f"  for scale, 1 W across one GEMM's {SETS * SHIFTERS:,} resident shifters allows"
          f" {fmt_power(p_pi_1w)} per pi:")
    print(f"    thermo-optic silicon needs {THERMO_P_PI / p_pi_1w:.0f}x that,"
          f" BTO {p_pi_1w / BTO_STATIC_PI:,.0f}x less")
    lo, hi = drift_rad(TFLN_DRIFT_DB)
    lo_t, hi_t = drift_rad(TFLT_DRIFT_DB)
    print("  drift: recalibrate before one LSB accumulates (pta_cpu_integration.md 5.1).")
    print(f"    Over the {DRIFT_TEST_H} h test TFLN moved {lo / (math.pi / 64):.0f}-"
          f"{hi / (math.pi / 64):.0f} LSBs of a 6-bit code and TFLT under"
          f" {lo_t / (math.pi / 64):.0f}-{hi_t / (math.pi / 64):.0f}.")


if __name__ == "__main__":
    main()
