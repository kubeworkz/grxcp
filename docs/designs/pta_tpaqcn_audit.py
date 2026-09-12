"""
Energy-per-activation audit for the TPA-QCN activation unit.

Three findings, each produced by running the document's own Tier-1 code:

  A. The effective-index model cannot phase-match anywhere in its design space,
     because it carries one index per wavelength and no polarisation axis.
     The birefringent crossing the whole TPA-QCN premise rests on is not
     expressible in it.

  B. The nonlinear coupling coefficient is dimensionally inconsistent and about
     two orders of magnitude too large.

  C. With both repaired -- phase matching imposed and kappa taken from the
     textbook normalised-SHG form -- the energy per activation lands in the
     picojoule range, not the attojoule range the document claims.

Run:  PYTHONPATH=. python3 scripts/energy_audit.py
"""
import math
import torch
from tpaqcn.waveguide import TPAQCNWaveguide
from tpaqcn.cme_solver import solve_cme

torch.set_default_dtype(torch.float64)

EPS0 = 8.8541878128e-12
C0 = 299792458.0
LAM = 1550e-9
F_SYM = 100e9                      # the document's claimed activation bandwidth
TAU = 1.0 / F_SYM


def kappa_standard(chi2_pm_V, n_ff, n_sh, A_eff_m2, lam=LAM):
    """
    kappa from the textbook normalised SHG efficiency.

    Undepleted, phase-matched, the CME gives P2 = kappa^2 P1^2 L^2, and the
    standard result is P2 = eta_norm P1^2 L^2 with

        eta_norm = 8 pi^2 d_eff^2 / (eps0 c n_ff^2 n_sh lambda^2 A_eff)

    so kappa = sqrt(eta_norm).  Units W^-1/2 m^-1.
    """
    d_eff = 0.5 * chi2_pm_V * 1e-12          # chi(2) = 2 d
    eta_norm = (8 * math.pi**2 * d_eff**2) / (
        EPS0 * C0 * n_ff**2 * n_sh * lam**2 * A_eff_m2)
    return math.sqrt(eta_norm), eta_norm


class Shim:
    """Minimal waveguide view with kappa and delta_beta forced."""
    def __init__(self, L_mm, kappa, dbeta, a_ff_db, a_sh_db):
        self.length = torch.tensor(L_mm)
        self._k, self._db = kappa, dbeta
        self._a1, self._a2 = a_ff_db, a_sh_db

    def kappa(self):
        return torch.tensor(self._k)

    def delta_beta(self):
        return torch.tensor(self._db)

    def alpha_ff_np_per_m(self):
        return torch.tensor(self._a1 * 100.0 / (10.0 * math.log10(math.e)))

    def alpha_sh_np_per_m(self):
        return torch.tensor(self._a2 * 100.0 / (10.0 * math.log10(math.e)))


def knee_power(L_mm, kappa, dbeta=0.0, a_ff=15.0, a_sh=10.0,
               lo=1e-6, hi=1e6):
    """Input power at which half the FF has converted, loss divided out."""
    wg = Shim(L_mm, kappa, dbeta, a_ff, a_sh)
    lin = math.exp(-float(wg.alpha_ff_np_per_m()) * L_mm * 1e-3)

    def converted(P):
        out = solve_cme(wg, torch.tensor([P]), n_z=32,
                        method='dopri5', rtol=1e-9, atol=1e-13)
        return 1.0 - float(out['P_ff_out']) / (P * lin)

    if converted(hi) < 0.5:
        return float('inf')
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if converted(mid) < 0.5:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.000001:
            break
    return math.sqrt(lo * hi)


def rule(c='='):
    print(c * 76)


def main():
    ref = TPAQCNWaveguide(
        width=torch.tensor([1.0]), height=torch.tensor([0.4]),
        length=torch.tensor([0.5]), chi2_eff=torch.tensor([50.0]),
        n_core_ff=torch.tensor([1.75]), n_core_sh=torch.tensor([1.85]),
        n_clad=torch.tensor([1.44]),
        alpha_ff_db=torch.tensor([15.0]), alpha_sh_db=torch.tensor([10.0]))
    k_doc = float(ref.kappa())
    n_ff, n_sh = [float(x) for x in ref.effective_indices()]
    db_ref = float(ref.delta_beta())

    rule()
    print("A. PHASE MATCHING: the model has no polarisation axis")
    rule()
    W = torch.linspace(0.15, 6.0, 240)
    H = torch.linspace(0.05, 3.0, 160)
    ww, hh = torch.meshgrid(W, H, indexing='ij')
    ww, hh = ww.reshape(-1), hh.reshape(-1)
    n = ww.numel()
    grid = TPAQCNWaveguide(
        width=ww, height=hh, length=torch.full((n,), 0.5),
        chi2_eff=torch.full((n,), 50.0),
        n_core_ff=torch.full((n,), 1.75), n_core_sh=torch.full((n,), 1.85),
        n_clad=torch.full((n,), 1.44),
        alpha_ff_db=torch.full((n,), 15.0), alpha_sh_db=torch.full((n,), 10.0))
    gff, gsh = grid.effective_indices()
    d = gsh - gff
    print(f"  swept {n:,} geometries, width 0.15-6.0 um x height 0.05-3.0 um")
    print(f"  n_eff(2w) - n_eff(FF):  min {float(d.min()):+.5f}"
          f"   max {float(d.max()):+.5f}")
    print(f"  crosses zero anywhere?  {bool((d <= 0).any())}")
    print(f"  reference unit (1.0 x 0.4 um): delta_beta = {db_ref:.4e} 1/m,")
    print(f"    delta_beta * L = {db_ref * 0.5e-3:.1f} rad over 0.5 mm"
          f"  (phase matched means ~0)")
    print()
    print("  Cause: n_eff = n_clad + (n_core - n_clad) * Gamma(V), one n_core per")
    print("  wavelength, and V ~ 1/lambda so the SH is always both higher-index")
    print("  and better confined.  No TE/TM distinction exists in the model, so")
    print("  the giant negative birefringence that makes TPA-QCN interesting --")
    print("  n_eff(TM00, 2w) crossing BELOW n_eff(TE00, w) -- cannot be")
    print("  represented.  Every transfer function this solver produces comes")
    print("  from deep phase mismatch, not from the designed mechanism.")

    print()
    rule()
    print("B. COUPLING COEFFICIENT: dimensionally inconsistent")
    rule()
    print(f"  waveguide.py: kappa = (w/2) sqrt(eta0) chi2 / (n1 n2) * overlap")
    print(f"                      = {k_doc:.4e}")
    print("    [w]=s^-1  [sqrt(eta0)]=V W^-1/2  [chi2]=m V^-1  ->  m s^-1 W^-1/2")
    print("    the CME needs                                   ->  m^-1 W^-1/2")
    print("    shortfall: a factor of s m^-2, i.e. 1/(c sqrt(A_eff))")
    print()
    print(f"  {'A_eff':>8}{'doc/(c sqrt A)':>17}{'textbook':>12}{'doc/textbook':>15}")
    print("  " + "-" * 52)
    for A_um2 in (0.25, 0.5, 1.0):
        A = A_um2 * 1e-12
        k_std, _ = kappa_standard(50.0, n_ff, n_sh, A)
        print(f"  {A_um2:5.2f} um^2{k_doc/(C0*math.sqrt(A)):17.1f}"
              f"{k_std:12.1f}{k_doc/k_std:14.0f}x")

    A_eff = 0.5e-12
    k_std, eta_norm = kappa_standard(50.0, n_ff, n_sh, A_eff)
    print(f"\n  Taking A_eff = 0.5 um^2: kappa = {k_std:.1f} W^-1/2 m^-1,"
          f" eta_norm = {eta_norm:.3e} /W/m^2")
    print("  The two independent corrections agree to a factor of ~2, which is")
    print("  the chi2-vs-d_eff convention and the overlap factor.")

    print()
    rule()
    print("C. ENERGY PER ACTIVATION, phase matched, textbook kappa")
    rule()
    print(f"  activation = one symbol at {F_SYM/1e9:.0f} GHz = {TAU*1e12:.0f} ps")
    print("  knee = input power at which half the FF has converted (the sigmoid")
    print("  midpoint an activation is biased around), propagation loss removed")
    print()
    print(f"  {'L (mm)':>8}{'kappa':>10}{'P_knee solver':>16}{'analytic':>12}"
          f"{'E/activation':>15}")
    print("  " + "-" * 61)
    for L_mm in (0.1, 0.5, 1.0, 5.0, 10.0):
        P = knee_power(L_mm, k_std, dbeta=0.0)
        P_an = (0.8814 / (k_std * L_mm * 1e-3)) ** 2
        print(f"  {L_mm:8.1f}{k_std:10.0f}{P:14.4g} W{P_an:10.4g} W"
              f"{P*TAU:13.4g} J")
    print("\n  solver and the analytic sech^2 midpoint agree, which validates the")
    print("  ODE integration once delta_beta is actually zero.")

    print()
    rule()
    print("D. AGAINST THE DOCUMENT'S CLAIMS")
    rule()
    P_ref = knee_power(0.5, k_std, dbeta=0.0)
    E_ref = P_ref * TAU
    print(f"  reference 0.5 mm unit, phase matched: {E_ref:.4g} J per activation")
    for label, claim in (("doc claim, activation", 100e-18),
                         ("doc claim, O-E-O baseline", 5e-12)):
        print(f"    vs {label:<26}{claim:9.3g} J   -> {E_ref/claim:12,.0f}x")
    print()
    print("  It is not merely worse than claimed; at 0.5 mm it is worse than the")
    print("  O-E-O activation the architecture exists to remove.")
    print()
    print("  Reaching 100 aJ at 100 GHz needs P_knee = "
          f"{100e-18*F_SYM:.3g} W:")
    k_need = 0.8814 / (math.sqrt(100e-18 * F_SYM) * 0.5e-3)
    print(f"    kappa = {k_need:.4g}  ({(k_need/k_std)**2:.3g}x the eta_norm above)")
    L_need = 0.8814 / (k_std * math.sqrt(100e-18 * F_SYM))
    print(f"    by length alone:  L = {L_need*1e3:,.0f} mm"
          f"  -> {15*L_need*100:,.0f} dB of FF loss at 15 dB/cm")
    print(f"    by chi2 alone:    chi2 = {50.0*(k_need/k_std):,.0f} pm/V"
          f"  (assumed 50; LiNbO3 d33 ~ 27)")
    print(f"    by A_eff alone:   A_eff = {0.5/(k_need/k_std)**2:.3g} um^2"
          f"  (sub-nm^2, unphysical)")
    print()
    print("  Resonant enhancement is the one remaining route, and it collides")
    print("  with the bandwidth claimed in the same table:")
    print(f"    carrier {C0/LAM/1e12:.1f} THz;  >100 GHz activation bandwidth")
    print(f"    caps Q at {C0/LAM/F_SYM:,.0f}.  High Q and wide bandwidth are the")
    print("    same knob turned opposite ways.")
    print()
    print("  The honest operating point is picojoules at 100 GHz, or attojoules")
    print("  at a bandwidth set by whatever Q the enhancement needs.  The")
    print("  document quotes both ends of that trade at once.")


if __name__ == '__main__':
    main()
