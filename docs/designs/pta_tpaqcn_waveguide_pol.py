"""
Polarisation-aware TPA-QCN waveguide model.

This replaces waveguide.py, whose effective-index model carries one core index
per wavelength and no polarisation axis, and therefore cannot express the
birefringent phase matching the whole TPA-QCN premise rests on.  Swept over
38,400 geometries that model never brings n_eff(2w) below n_eff(FF).

Three changes:

  1. The film is a uniaxial medium.  Spontaneously oriented evaporated films of
     TPA-QCN show S ~ -0.30 to -0.35, i.e. a preference for the molecular long
     axis lying IN the plane of the film (Theriault et al., Adv. Opt. Mater.
     2026, arXiv:2511.13682).  In-plane polarisability therefore exceeds
     out-of-plane: n_o > n_e, a NEGATIVE uniaxial film with the optic axis
     normal to the substrate.  That is the sign the mechanism needs.

  2. Confinement comes from a real asymmetric-slab dispersion relation solved
     by bisection, plus the effective-index method for the second dimension,
     rather than a sigmoid in normalised frequency.  TE and TM differ by the
     n^2 ratios in the TM boundary condition, which is where the polarisation
     dependence physically lives.

  3. Type-I here is TE00(w) + TE00(w) -> TM00(2w).  Two in-plane fundamental
     fields driving an out-of-plane polarisation is chi_31, not chi_33.  For
     TPA-QCN the two happen to be close (8 vs 9 pm/V at 1550 nm) but for
     derivatives they are not, and picking the wrong one is a factor of 4 in
     energy.

Measured material values, at 1550 nm, from arXiv:2511.13682:

    compound 1 (TPA-QCN)   chi_31 ~  8 pm/V,  chi_33 ~  9 pm/V,  Tg 110 C
    compound 4 (best)      chi_31 ~ 16 pm/V,  chi_33 ~ 18 pm/V,  Tg  76 C

The refractive indices are in that paper's supplementary material, which is not
reachable from here, so n_o and n_e are inputs rather than constants and the
birefringence needed for phase matching is reported as a prediction to check.

Since checked: pta_tpaqcn_measured.py carries indices digitised from the
published TPA-QCN waveguide (Sci. Adv. 12, eaeg3170, 2026), which sit inside
the window, and chi_31 = 8.1, chi_33 = 7 pm/V as that paper reports.  The
constants below are left as they were so the review's section 4.1 still
reproduces.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

C0 = 299792458.0
EPS0 = 8.8541878128e-12

# Measured chi(2) at 1550 nm, pm/V (arXiv:2511.13682).
CHI2_MEASURED = {
    'TPA-QCN':    {'chi31': 8.0,  'chi33': 9.0,  'Tg_C': 110},
    'compound-4': {'chi31': 16.0, 'chi33': 18.0, 'Tg_C': 76},
}


def slab_neff(n_core: float, n_sub: float, n_cov: float,
              h_um: float, lam_um: float, tm: bool,
              m: int = 0) -> Optional[float]:
    """
    Effective index of mode m of a three-layer asymmetric slab.

    TE :  k h = m pi + atan(g_c / k) + atan(g_s / k)
    TM :  k h = m pi + atan((n_core/n_cov)^2 g_c / k)
                     + atan((n_core/n_sub)^2 g_s / k)

    The n^2 ratios in the TM form are the whole polarisation dependence: they
    push the TM mode out of the core relative to TE at the same geometry.

    Returns None when the mode is below cutoff.
    """
    n_hi = max(n_sub, n_cov)
    if n_core <= n_hi:
        return None
    k0 = 2.0 * math.pi / lam_um

    def residual(N):
        k = k0 * math.sqrt(max(n_core**2 - N**2, 1e-18))
        g_s = k0 * math.sqrt(max(N**2 - n_sub**2, 1e-18))
        g_c = k0 * math.sqrt(max(N**2 - n_cov**2, 1e-18))
        if tm:
            phi_c = math.atan((n_core / n_cov) ** 2 * g_c / k)
            phi_s = math.atan((n_core / n_sub) ** 2 * g_s / k)
        else:
            phi_c = math.atan(g_c / k)
            phi_s = math.atan(g_s / k)
        return k * h_um - (m * math.pi + phi_c + phi_s)

    lo, hi = n_hi + 1e-9, n_core - 1e-9
    if residual(lo) * residual(hi) > 0:
        return None                      # cutoff
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(lo) * residual(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def core_confinement(n_core: float, n_sub: float, n_cov: float,
                     h_um: float, lam_um: float, tm: bool) -> float:
    """
    Fraction of the slab mode's power inside the core, from the analytic
    field of the fundamental mode.  This is what multiplies chi(2): the
    nonlinearity exists only inside the organic film.
    """
    N = slab_neff(n_core, n_sub, n_cov, h_um, lam_um, tm)
    if N is None:
        return 0.0
    k0 = 2.0 * math.pi / lam_um
    k = k0 * math.sqrt(max(n_core**2 - N**2, 1e-18))
    g_s = k0 * math.sqrt(max(N**2 - n_sub**2, 1e-18))
    g_c = k0 * math.sqrt(max(N**2 - n_cov**2, 1e-18))
    # symmetric-form effective thickness: h plus the two evanescent tails
    h_eff = h_um + 1.0 / g_s + 1.0 / g_c
    return h_um / h_eff


@dataclass
class UniaxialFilm:
    """
    Negative uniaxial organic film, optic axis normal to the substrate.

    n_o : in-plane index, seen by TE
    n_e : out-of-plane index, seen by TM
    """
    n_o_ff: float
    n_e_ff: float
    n_o_sh: float
    n_e_sh: float
    chi31_pm_V: float
    chi33_pm_V: float
    name: str = 'TPA-QCN'

    def birefringence_ff(self) -> float:
        return self.n_o_ff - self.n_e_ff

    def birefringence_sh(self) -> float:
        return self.n_o_sh - self.n_e_sh


def tpaqcn_film(n_o_ff: float = 1.75,
                dispersion: float = 0.10,
                birefringence: float = 0.25,
                compound: str = 'TPA-QCN') -> UniaxialFilm:
    """
    Build a film from three physically meaningful knobs instead of four indices:

      n_o_ff        in-plane index at the fundamental
      dispersion    n(2w) - n(w) for the same polarisation; organics with a
                    charge-transfer band near 450-510 nm have strong normal
                    dispersion, so 0.05-0.20 is the plausible range at
                    1550/775 nm
      birefringence n_o - n_e, taken wavelength-independent for simplicity

    Phase matching needs n_e(2w) = n_o(w), i.e. birefringence > dispersion.
    That single inequality is the whole mechanism.
    """
    c = CHI2_MEASURED[compound]
    return UniaxialFilm(
        n_o_ff=n_o_ff,
        n_e_ff=n_o_ff - birefringence,
        n_o_sh=n_o_ff + dispersion,
        n_e_sh=n_o_ff + dispersion - birefringence,
        chi31_pm_V=c['chi31'], chi33_pm_V=c['chi33'], name=compound)


@dataclass
class PolWaveguide:
    """Rectangular organic-core channel on oxide, air-clad."""
    film: UniaxialFilm
    width_um: float
    height_um: float
    n_sub: float = 1.44          # SiO2
    n_cov: float = 1.00          # air
    lam_ff_um: float = 1.55

    def _eim(self, n_core_vert: float, n_core_horiz_ref: float,
             lam_um: float, tm_like: bool) -> Optional[float]:
        """
        Effective-index method.  For a TE-like 2-D mode the vertical slab
        problem is TE and the horizontal one TM; for TM-like, the reverse.
        """
        N_v = slab_neff(n_core_vert, self.n_sub, self.n_cov,
                        self.height_um, lam_um, tm=tm_like)
        if N_v is None:
            return None
        # lateral: core index N_v, sides are the unetched/air region
        n_side = max(self.n_sub, self.n_cov) * 0.999
        if N_v <= n_side:
            return None
        return slab_neff(N_v, n_side, n_side,
                         self.width_um, lam_um, tm=not tm_like)

    def n_eff_te_ff(self) -> Optional[float]:
        """TE00 at the fundamental: E in-plane, sees n_o."""
        return self._eim(self.film.n_o_ff, self.film.n_o_ff,
                         self.lam_ff_um, tm_like=False)

    def n_eff_tm_sh(self) -> Optional[float]:
        """TM00 at the second harmonic: E out-of-plane, sees n_e."""
        return self._eim(self.film.n_e_sh, self.film.n_e_sh,
                         self.lam_ff_um / 2.0, tm_like=True)

    def delta_n(self) -> Optional[float]:
        a, b = self.n_eff_te_ff(), self.n_eff_tm_sh()
        if a is None or b is None:
            return None
        return b - a

    def delta_beta(self) -> Optional[float]:
        """beta(2w) - 2 beta(w), 1/m.  Zero is phase matched."""
        d = self.delta_n()
        if d is None:
            return None
        k0 = 2.0 * math.pi / (self.lam_ff_um * 1e-6)
        return 2.0 * k0 * d

    def nl_overlap(self) -> float:
        """
        Fraction of the interaction inside the film, as the product of the
        FF and SH core confinements.  For a 50-100 nm film -- which is what
        these depositions actually produce -- this is small, and it multiplies
        chi(2) directly.
        """
        g_ff = core_confinement(self.film.n_o_ff, self.n_sub, self.n_cov,
                                self.height_um, self.lam_ff_um, tm=False)
        g_sh = core_confinement(self.film.n_e_sh, self.n_sub, self.n_cov,
                                self.height_um, self.lam_ff_um / 2, tm=True)
        return g_ff * math.sqrt(max(g_sh, 0.0))

    def mode_area_um2(self) -> float:
        """Crude but honest: the guided cross-section plus evanescent tails."""
        g = core_confinement(self.film.n_o_ff, self.n_sub, self.n_cov,
                             self.height_um, self.lam_ff_um, tm=False)
        g = max(g, 1e-3)
        return self.width_um * self.height_um / g

    def kappa(self) -> float:
        """
        kappa = sqrt(eta_norm) with the correct chi(2) tensor element, the
        film overlap, and a real mode area.

            eta_norm = 8 pi^2 d_eff^2 / (eps0 c n_ff^2 n_sh lambda^2 A_eff)

        d_eff = chi_31 / 2 for TE + TE -> TM.
        """
        n_ff = self.n_eff_te_ff()
        n_sh = self.n_eff_tm_sh()
        if n_ff is None or n_sh is None:
            return 0.0
        d_eff = 0.5 * self.film.chi31_pm_V * 1e-12 * self.nl_overlap()
        lam = self.lam_ff_um * 1e-6
        A = self.mode_area_um2() * 1e-12
        eta = (8 * math.pi**2 * d_eff**2) / (
            EPS0 * C0 * n_ff**2 * n_sh * lam**2 * A)
        return math.sqrt(eta)


def find_phase_match(film: UniaxialFilm, height_um: float,
                     w_lo: float = 0.10, w_hi: float = 8.0,
                     n: int = 600) -> Optional[float]:
    """Width at which delta_n crosses zero, or None if it never does."""
    prev_w = prev_d = None
    for i in range(n):
        w = w_lo + (w_hi - w_lo) * i / (n - 1)
        d = PolWaveguide(film, w, height_um).delta_n()
        if d is not None and prev_d is not None and prev_d * d <= 0:
            lo, hi = prev_w, w
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                dm = PolWaveguide(film, mid, height_um).delta_n()
                if dm is None:
                    break
                if prev_d * dm <= 0:
                    hi = mid
                else:
                    lo = mid
            return 0.5 * (lo + hi)
        if d is not None:
            prev_w, prev_d = w, d
    return None
