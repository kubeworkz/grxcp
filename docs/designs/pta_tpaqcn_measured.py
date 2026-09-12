"""
The TPA-QCN review, checked against the device that now exists.

    P.-L. Theriault, A. Petit, A. Anand V.S., S. Kena-Cohen, "Poling-free
    integrated second-order nonlinear optics with evaporated organic thin
    films", Sci. Adv. 12, eaeg3170 (27 May 2026), open access.

The same group as the materials paper (arXiv:2511.13682) fabricated the
TE00(w) -> TM00(2w) birefringently phase-matched waveguide the review argues
for.  This script re-derives section 2.3 and section 4.4 of
pta_tpaqcn_review.md from what that paper reports.

From the text:
    chi_31 = 8.1 +/- 0.3 pm/V, chi_33 = 7 +/- 3 pm/V at 1550 nm
    230 nm TPA-QCN on 2 um SiO2, 15 nm TCTA cap, resist strip
    (n = 1.418 at 1550, 1.423 at 775), phase matched at w = 1.9 um, n_eff 1.477
    propagation loss 20 +/- 2 dB/cm at both 1550 and 780 nm
    eta_L = P_2w / (P_w^2 L^2) = 29 %/W/cm^2 for the best 1.7 mm device,
    68 %/W/cm^2 projected once leakage is removed (5 dB/cm)
    +50 nm of strip width moves the phase-matching peak +22 nm

Digitised from the figures (axes calibrated on each figure's own tick labels;
Fig. 2A additionally against its 775 and 1550 nm marker lines, which land
within 1 nm).  Index readings are good to about +/-0.002:
    Fig. 2A   n_o, n_e of the film at 1550 and 775 nm
    Fig. 3C   phase-matching FWHM of about 12 nm in pump wavelength

Assumed, not in the text: SiO2 1.444 / 1.454, TCTA 1.66 / 1.70 (isotropic,
15 nm, so it barely matters).

Standard library only.  Run:  python3 docs/designs/pta_tpaqcn_measured.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pta_tpaqcn_waveguide_pol import UniaxialFilm, find_phase_match, slab_neff

# ---- measured --------------------------------------------------------------
N_O_FF, N_E_FF = 1.7645, 1.5719          # 1550 nm, Fig. 2A
N_O_SH, N_E_SH = 1.8270, 1.5954          # 775 nm,  Fig. 2A
CHI31, CHI33 = 8.1, 7.0                  # pm/V, text
H_FILM, H_TCTA = 0.230, 0.015            # um
N_RESIST = (1.418, 1.423)                # 1550, 775
W_PM, NEFF_PM = 1.9, 1.477
ETA_L, L_DEV_CM, ALPHA_DB = 0.29, 0.17, 20.0
ETA_PROJECTED_5DB = 0.68
FWHM_NM, SHIFT_NM_PER_NM = 12.0, 22.0 / 50.0

# ---- assumed ---------------------------------------------------------------
N_SIO2 = (1.4440, 1.4538)
N_TCTA = (1.66, 1.70)

# ---- conventions shared with pta_tpaqcn_audit.py ---------------------------
TAU = 10e-12                             # one symbol at the claimed 100 GHz
OEO_J = 5e-12                            # the document's O-E-O activation
NP_DB = 10 * math.log10(math.e)          # dB per neper, power


def rule(title):
    print("=" * 76)
    print(title)
    print("=" * 76)


# ---------------------------------------------------------------------------
# Waveguide physics
# ---------------------------------------------------------------------------

def slab_multi(layers, n_sub, n_cov, lam_um, tm):
    """
    Fundamental mode of a multilayer slab by transfer matrix.

    layers: (n_o, n_e, thickness_um) bottom to top, optic axis normal to the
    film.  TE sees n_o.  TM is the extraordinary wave:
        k_y = k0 (n_o / n_e) sqrt(n_e^2 - N^2)
    with 1/eps_z = 1/n_o^2 in the boundary condition, because E_z lies in the
    plane.  pta_tpaqcn_waveguide_pol.py gives TM the isotropic n_e instead;
    that is an approximation this solver does not make.
    """
    k0 = 2 * math.pi / lam_um

    def residual(N):
        g_s = k0 * math.sqrt(N * N - n_sub * n_sub)
        g_c = k0 * math.sqrt(N * N - n_cov * n_cov)
        w_s, w_c = (n_sub ** 2, n_cov ** 2) if tm else (1.0, 1.0)
        U, V = 1.0, g_s / w_s
        for no, ne, d in layers:
            arg = (no / ne) ** 2 * (ne * ne - N * N) if tm else no * no - N * N
            w = no * no if tm else 1.0
            if arg > 0:
                k = k0 * math.sqrt(arg)
                c, s = math.cos(k * d), math.sin(k * d)
                U, V = c * U + (w / k) * s * V, -(k / w) * s * U + c * V
            else:
                g = k0 * math.sqrt(-arg)
                c, s = math.cosh(g * d), math.sinh(g * d)
                U, V = c * U + (w / g) * s * V, (g / w) * s * U + c * V
        return V + (g_c / w_c) * U

    lo = max(n_sub, n_cov) + 1e-9
    hi = max(ne if tm else no for no, ne, _ in layers) - 1e-9
    steps = 4000
    prev_N, prev_r = hi, residual(hi)
    for i in range(1, steps + 1):
        N = hi - (hi - lo) * i / steps
        r = residual(N)
        if prev_r * r <= 0:
            a, b, ra = N, prev_N, r
            for _ in range(100):
                m = 0.5 * (a + b)
                rm = residual(m)
                if ra * rm <= 0:
                    b = m
                else:
                    a, ra = m, rm
            return 0.5 * (a + b)
        prev_N, prev_r = N, r
    return None


def strip_loaded_neff(w_um, lam_um, tm):
    """
    Effective-index model of the paper's strip-loaded guide: the vertical stack
    under the resist strip against the same stack under air, then a lateral
    slab of width w.  A TE-like mode crosses the lateral walls TM-polarised
    and vice versa.
    """
    i = 0 if lam_um > 1.0 else 1
    no, ne = (N_O_FF, N_E_FF) if i == 0 else (N_O_SH, N_E_SH)
    stack = [(no, ne, H_FILM), (N_TCTA[i], N_TCTA[i], H_TCTA)]
    n_in = slab_multi(stack, N_SIO2[i], N_RESIST[i], lam_um, tm)
    n_out = slab_multi(stack, N_SIO2[i], 1.0, lam_um, tm)
    if n_in is None or n_out is None or n_in <= n_out:
        return None
    return slab_neff(n_in, n_out, n_out, w_um, lam_um, tm=not tm)


def te0_cutoff_um(n_film, n_sub, n_cov, lam_um):
    """Thickness below which an asymmetric slab guides no TE mode."""
    v = math.sqrt(n_film ** 2 - n_sub ** 2)
    return math.atan(math.sqrt((n_sub ** 2 - n_cov ** 2) / v ** 2)) / (
        2 * math.pi / lam_um * v)


def loss_factor(alpha_db_cm, L_cm):
    """
    Undepleted SH power with equal power loss alpha at both wavelengths,
    relative to the lossless P_2w = eta0 P_w^2 L^2.
    """
    x = alpha_db_cm / NP_DB * L_cm / 2
    return (math.exp(-x) * (1 - math.exp(-x)) / x) ** 2


def knee_power(L_mm, kappa, a_ff_db=15.0, a_sh_db=10.0, steps=2000):
    """
    The definition in pta_tpaqcn_audit.knee_power, integrated by RK4 instead
    of torchdiffeq: the input power at which half the fundamental has
    converted, with its linear loss divided out.

        a' = -(a_ff/2) a - kappa a b
        b' = -(a_sh/2) b + kappa a^2          (phase matched, P = amplitude^2)
    """
    L = L_mm * 1e-3
    a1, a2 = a_ff_db * 100 / NP_DB, a_sh_db * 100 / NP_DB
    linear = math.exp(-a1 * L)
    h = L / steps

    def f(a, b):
        return -0.5 * a1 * a - kappa * a * b, -0.5 * a2 * b + kappa * a * a

    def converted(P):
        a, b = math.sqrt(P), 0.0
        for _ in range(steps):
            k1 = f(a, b)
            k2 = f(a + 0.5 * h * k1[0], b + 0.5 * h * k1[1])
            k3 = f(a + 0.5 * h * k2[0], b + 0.5 * h * k2[1])
            k4 = f(a + h * k3[0], b + h * k3[1])
            a += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            b += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return 1.0 - a * a / (P * linear)

    lo, hi = 1e-7, 1e7
    if converted(hi) < 0.5:
        return float('inf')
    while hi / lo > 1.00001:
        mid = math.sqrt(lo * hi)
        if converted(mid) < 0.5:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


# ---------------------------------------------------------------------------

def main():
    global N_O_FF, N_E_SH                # section 3 perturbs the two readings
    rule("1. MATERIAL: the window check (review section 2.1)")
    print(f"  1550 nm   n_o {N_O_FF:.3f}   n_e {N_E_FF:.3f}"
          f"   birefringence {N_O_FF - N_E_FF:.3f}")
    print(f"   775 nm   n_o {N_O_SH:.3f}   n_e {N_E_SH:.3f}"
          f"   birefringence {N_O_SH - N_E_SH:.3f}")
    print(f"  ordinary dispersion n_o(2w) - n_o(w)     {N_O_SH - N_O_FF:.3f}")
    print(f"  bulk gap n_o(w) - n_e(2w)                {N_O_FF - N_E_SH:.3f}"
          "   (guiding has to close this)")
    print("  section 2.1's window, interpolated to this dispersion: ~0.10-0.25")

    print()
    rule("2. GEOMETRY: why the review's model wanted a thick film")
    film = UniaxialFilm(N_O_FF, N_E_FF, N_O_SH, N_E_SH, CHI31, CHI33)
    print("  the review's own air-clad channel, measured indices:")
    for h in (0.23, 0.30, 0.40, 0.60, 1.00):
        w = find_phase_match(film, h)
        print(f"    h {h:4.2f} um  ->  "
              f"{'phase matched at w = %.2f um' % w if w else 'no crossing'}")
    c_air = te0_cutoff_um(N_O_FF, N_SIO2[0], 1.0, 1.55)
    c_res = te0_cutoff_um(N_O_FF, N_SIO2[0], N_RESIST[0], 1.55)
    print(f"  TE0 cutoff at 1550 nm, film on oxide:  air cover {c_air*1e3:.0f} nm,"
          f"  resist cover {c_res*1e3:.0f} nm")
    print(f"  a {H_FILM*1e3:.0f} nm film is {H_FILM/c_air:.1f}x cutoff under air and"
          f" {H_FILM/c_res:.1f}x under resist: the cover, not the film, was the limit")

    print()
    rule("3. THE DEVICE, effective-index model (paper: w 1.9 um, n_eff 1.477)")

    def crossing():
        prev = None
        for i in range(0, 121):
            w = 0.5 + 0.05 * i
            te = strip_loaded_neff(w, 1.55, tm=False)
            tm = strip_loaded_neff(w, 0.775, tm=True)
            if te is None or tm is None:
                continue
            if prev is not None and prev[1] * (te - tm) <= 0:
                return w, 0.5 * (te + tm)
            prev = (w, te - tm)
        return None

    te19 = strip_loaded_neff(W_PM, 1.55, tm=False)
    tm19 = strip_loaded_neff(W_PM, 0.775, tm=True)
    print(f"  at w = {W_PM} um: TE00(w) {te19:.4f}, TM00(2w) {tm19:.4f}")
    base = (N_O_FF, N_E_SH)
    for shift in (0.0, 0.005):
        N_O_FF, N_E_SH = base[0] - shift, base[1] + shift
        c = crossing()
        label = ("digitised indices" if shift == 0 else
                 f"n_o(w) -{shift}, n_e(2w) +{shift}")
        print(f"  {label:<32} crossing at "
              f"{'w ~ %.2f um, n_eff %.3f' % c if c else 'none in 0.5-6.5 um'}")
    N_O_FF, N_E_SH = base
    print("  n_eff lands within 0.01 of the paper; the width does not, and pushing")
    print("  the indices 0.005 (past their reading error) moves it 0.45 um.  The")
    print("  phase-matching width is set by index differences of order 1e-3, finer")
    print("  than digitised curves or effective-index theory resolve.  Designing")
    print("  widths needs tabulated indices and an anisotropic mode solver.")

    print()
    rule("4. EFFICIENCY: loss removed from the measurement")
    F20 = loss_factor(ALPHA_DB, L_DEV_CM)
    eta0 = ETA_L / F20
    F5 = loss_factor(5.0, L_DEV_CM)
    kappa0 = math.sqrt(eta0 * 1e4)                    # /W/cm^2 -> /W/m^2
    print(f"  eta_L {ETA_L*100:.0f} %/W/cm^2 at {L_DEV_CM*10:.1f} mm and {ALPHA_DB:.0f} dB/cm;"
          f" loss factor {F20:.3f}")
    print(f"  lossless eta0 = {eta0*100:.0f} %/W/cm^2,  kappa0 = {kappa0:.0f} W^-1/2 m^-1")
    print(f"  cross-check, eta0 at 5 dB/cm: {eta0*F5*100:.0f} %/W/cm^2"
          f"   (paper's own projection: {ETA_PROJECTED_5DB*100:.0f})")
    print(f"  review's corrected model at 8 pm/V: kappa 121, eta0 {121**2/1e2:.0f} %/W/cm^2"
          f"  -> {121**2/1e4/eta0:.1f}x optimistic")

    print()
    rule("5. ENERGY PER ACTIVATION, anchored on the measurement (review 4.4)")
    P = knee_power(2.0, 121.0)
    print(f"  solver check against the review: kappa 121, 2 mm, 15/10 dB/cm ->"
          f" {P*TAU*1e12:.0f} pJ (review: 179)")
    k4 = kappa0 * (16.0 / CHI31)          # compound 4: chi_31 ~16 pm/V at 1550
    cases = (
        ("TPA-QCN as built, best device",             kappa0, 1.7, 20.0),
        ("TPA-QCN as built, the review's 2 mm unit",  kappa0, 2.0, 20.0),
        ("leakage removed (5 dB/cm), 2 mm",           kappa0, 2.0, 5.0),
        ("leakage removed, 3 dB budget -> 6 mm",      kappa0, 6.0, 5.0),
        ("+ compound 4 chi(2), 6 mm",                 k4, 6.0, 5.0),
        ("+ compound 4, 15 dB budget -> 30 mm",       k4, 30.0, 5.0),
    )
    print(f"\n  {'case':<42}{'L mm':>6}{'loss':>8}{'P_knee':>10}{'E':>10}{'/O-E-O':>9}")
    print("  " + "-" * 83)
    for name, k, L, a in cases:
        P = knee_power(L, k, a, a)
        E = P * TAU
        print(f"  {name:<42}{L:6.1f}{a*L/10:6.1f}dB{P:9.3g}W"
              f"{E*1e12:8.3g}pJ{E/OEO_J:8.2g}x")
    eta_tfln = 50.0      # /W/cm^2 (~5000 %/W/cm^2), PPLN on TFLN, as the paper cites
    P_tfln = (0.8814 ** 2) / (eta_tfln * 1.0 ** 2)   # sech^2 midpoint, 1 cm
    print(f"\n  for scale, periodically poled TFLN (~5000 %/W/cm^2), 1 cm, lossless:"
          f" {P_tfln*TAU*1e15:.0f} fJ")
    print(f"  TPA-QCN's lossless efficiency is {5000/(eta0*100):.0f}x below that;"
          f" 100 aJ is {P_tfln*TAU/100e-18:,.0f}x below TFLN")

    print()
    rule("6. TOLERANCE: every unit is its own curve (review 5.1)")
    half_width_nm = FWHM_NM / 2 / SHIFT_NM_PER_NM
    print(f"  phase-matching FWHM ~{FWHM_NM:.0f} nm; tuning {SHIFT_NM_PER_NM:.2f} nm of peak"
          f" per nm of strip width")
    print(f"  at a shared pump wavelength, a unit {half_width_nm:.0f} nm wider or narrower"
          " than nominal is at half efficiency")
    print("  the acceptance scales as 1/L, so the long, low-energy units in 5 above")
    print("  are proportionally less tolerant")


if __name__ == '__main__':
    main()
