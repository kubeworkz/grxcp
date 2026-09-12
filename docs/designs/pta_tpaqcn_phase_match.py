"""
Phase-matching and energy audit with the polarisation-aware model.

Reproduces every number in section 2.1 and section 4 of pta_tpaqcn_review.md.
Run:  PYTHONPATH=. python3 scripts/phase_match_audit.py
"""
from tpaqcn.waveguide_pol import (tpaqcn_film, PolWaveguide, find_phase_match,
                                  CHI2_MEASURED)
from scripts.energy_audit import knee_power, TAU


def has_cross(film, h, n=90):
    prev = None
    for i in range(n):
        w = 0.1 + (6.0 - 0.1) * i / (n - 1)
        d = PolWaveguide(film, w, h).delta_n()
        if d is not None and prev is not None and prev * d <= 0:
            return True
        if d is not None:
            prev = d
    return False


def main():
    print("1. Birefringence window vs material dispersion (h in 0.3-2.0 um)")
    for disp in (0.05, 0.10, 0.15, 0.20):
        ok = [b for b in [round(0.02 * k, 2) for k in range(1, 31)]
              if any(has_cross(tpaqcn_film(birefringence=b, dispersion=disp), h)
                     for h in (0.3, 0.6, 1.0, 1.6, 2.0))]
        if ok:
            print(f"   dispersion {disp:.2f} -> birefringence {min(ok):.2f}"
                  f"..{max(ok):.2f}  ({min(ok)/disp:.1f}x..{max(ok)/disp:.1f}x)")

    print("\n2. Phase-matched design points, measured chi_31 = 8 pm/V")
    film = tpaqcn_film(birefringence=0.20, dispersion=0.10)
    print(f"   {'h um':>7}{'w um':>8}{'overlap':>9}{'A_eff':>9}{'kappa':>8}"
          f"{'E @2mm':>12}")
    best = None
    for h in (0.4, 0.6, 0.8, 1.2, 1.6, 2.0):
        w = find_phase_match(film, h)
        if not w:
            continue
        g = PolWaveguide(film, w, h)
        k = g.kappa()
        if k <= 0:
            continue
        E = knee_power(2.0, k, 0.0) * TAU
        print(f"   {h:7.2f}{w:8.3f}{g.nl_overlap():9.3f}"
              f"{g.mode_area_um2():9.3f}{k:8.1f}{E:12.3g}")
        if best is None or E < best[0]:
            best = (E, k)

    print("\n3. Energy at 2 mm (a 3 dB loss budget) vs the chi(2) assumed")
    _, k8 = best
    for name, chi in (("document assumed", 50.0),
                      ("TPA-QCN measured", CHI2_MEASURED['TPA-QCN']['chi31']),
                      ("compound-4 measured", CHI2_MEASURED['compound-4']['chi31'])):
        kk = k8 * chi / 8.0
        print(f"   {name:<22}chi31={chi:5.1f} pm/V -> "
              f"{knee_power(2.0, kk, 0.0) * TAU:10.3g} J")

    print("\n4. Film thickness")
    thin = [h for h in (0.05, 0.08, 0.10, 0.15, 0.20) if find_phase_match(film, h)]
    print(f"   published films 49-108 nm; phase match at 50-200 nm? "
          f"{thin if thin else 'no'}")
    print("   the window needs 0.4-2.0 um, i.e. 4-20x thicker than deposited")


if __name__ == '__main__':
    main()
