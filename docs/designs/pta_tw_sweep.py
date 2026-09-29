"""
The PTA_TW sweep of pta_cpu_integration.md, evaluated before anything is built.

Section 2.1's cost model with the row-write term, checked against the cycles
section 2.3 measured on c930_npu_core, then priced at the points section 6.2
names: two thermo-optic tiles and two TFLN-class (Pockels) ones.  It prints
every number those sections quote.

Measured, from section 2.3 (tb/tb_core_m64.sv, M=64 N=8 K=256):
    m outer, as shipped           168,448 cycles
    m inner, sums restored        71,168 cycles
    restore folded into S_WRITE   removes 15,872 cycles  (3.05x)

From the c930 RTL, as section 2 reads it:
    S_PRELOAD scans one weight per cycle: NUM_ROWS * NUM_COLS = 64 cycles
    one c_mem row write is NUM_COLS = 8 cycles
    the tile's clock is ~100 MHz, 10 ns per cycle (section 6.2)

Published:
    TFLN Mach-Zehnder modulator, 20 mm, 45 GHz 3-dB electro-optic bandwidth
    (C. Wang et al., Nature 562, 101, 2018)

Assumed, not measured: the thermo-optic settle (10 us and 1 ms), a 50 ns
thermo-optic shot, and a resident bank select that fits inside the shot's own
cycle.  A single-pole response stands in for the modulator's real one.

Standard library only.  Run:  python3 docs/designs/pta_tw_sweep.py
"""
import math

# ---- the c930 configuration of section 2 ----------------------------------
M, N, K = 64, 8, 256
NUM_ROWS = NUM_COLS = 8
KT = -(-K // NUM_ROWS)                  # 32
NT = -(-N // NUM_COLS)                  # 1
SCAN = NUM_ROWS * NUM_COLS              # S_PRELOAD, cycles
TD = NUM_COLS                           # one c_mem row write, cycles
CYCLE_NS = 10.0                         # 100 MHz

# ---- measured, section 2.3 ------------------------------------------------
DIGITAL_RUN = NUM_ROWS + NUM_COLS + 2   # S_RUN, cycles
MEASURED_SHIPPED = 168_448
MEASURED_INTERCHANGED = 71_168
MEASURED_FOLD_SAVES = 15_872

# ---- measured, C2's gate (grx930 `make core_c2`) ---------------------------
# A shot's fixed cost on the c930: the hop-gated feed in, the hop-aligned
# capture, the registered valid.  PTA_TS buys dilation on top of it.
SHOT_FLOOR = 6
# The core's totals at 6.2's shape, from tb_core_verilator --c2.  The hop's entry
# parity leaves a residue of up to one cycle a shot, so these sit at or just
# inside the low end of the model's band.
MEASURED_C2 = {
    "TO-1ms": 3_254_785,
    "TO-10us": 86_784,
    "EO-scan": 46_592,
}

# ---- published ------------------------------------------------------------
TFLN_BW_HZ = 45e9


def shipped(tw, ts, td, m=M, nt=NT, kt=KT):
    """m outer: a weight program per shot, a row write per (m, nt)."""
    return m * nt * (kt * (tw + ts) + td)


def interchanged(tw, ts, td, m=M, nt=NT, kt=KT, restore=False):
    """(kt, nt) outer, m inner: a program per tile, a row write per shot.

    restore=True adds the unfolded restore: one more row of c_mem traffic per
    shot, except on each row's first K tile, which starts from zero.
    """
    t = nt * kt * (tw + m * (ts + td))
    if restore:
        t += m * nt * (kt - 1) * td
    return t


def core_cycles(pta_tw, pta_ts, m=M, n=N, k=K,
                rows=NUM_ROWS, cols=NUM_COLS, scanned=True):
    """What the c930 core spends, term by term, as C2's gate measures it.

    Each term is one of the core's states, and each is `interchanged()`'s
    corresponding term with 2.1's full-tile constant replaced by what the RTL
    walks:

      scan     S_WLOAD, N*K cells -- every weight element programmed once, which
               is Nt*Kt*(nc*kr) summed over tiles, not Nt*Kt*64
      settle   S_WLOAD held, Nt*Kt*PTA_TW, rounded up to the hop
      restore  S_ACCLD, M*(Kt-1)*N -- 2.1's unfolded M*Nt*(Kt-1)*Td with Td = nc
      write    S_WRITE, M*Kt*N     -- 2.1's M*Nt*Kt*Td, likewise
      shot     S_RUN, M*Nt*Kt*(PTA_TS + SHOT_FLOOR)

    Returns the upper end of the model's band; the hop's entry parity can take
    up to one cycle a shot off it, which is `band` below.
    """
    nt = -(-n // cols)
    kt = -(-k // rows)
    tw_eff = pta_tw + (pta_tw & 1)
    scan = (n * k if scanned else 0) + nt * kt * tw_eff
    restore = m * (kt - 1) * n
    write = m * kt * n
    shot = m * nt * kt * (pta_ts + SHOT_FLOOR)
    return scan + restore + write + shot


def core_band(m=M, n=N, k=K, rows=NUM_ROWS, cols=NUM_COLS):
    """One cycle a shot: the hop's entry parity, the same residue A1 needed."""
    return m * (-(-n // cols)) * (-(-k // rows))


def break_even(td, m=M, kt=KT, restore=False):
    """The Tw above which the interchange wins.  Ts cancels out of it."""
    return (2 if restore else 1) * td * m * (kt - 1) / ((m - 1) * kt)


def settle_s(bandwidth_hz, bits=8):
    """Single-pole settle to half an LSB of a `bits`-bit full scale."""
    return math.log(2 ** (bits + 1)) / (2 * math.pi * bandwidth_hz)


def fmt_time_ns(ns):
    if ns >= 1e6:
        return f"{ns / 1e6:.3g} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.3g} us"
    return f"{ns:.3g} ns"


def fmt_cycles(c):
    if c >= 1e6:
        return f"{c / 1e6:.3g} M"
    if c >= 1e4:
        return f"{c / 1e3:.3g} k"
    return f"{c:,.0f}"


def winner(t_shipped, t_inter):
    if t_inter < t_shipped:
        return f"interchange, {t_shipped / t_inter:.2g}x"
    return f"as shipped, {t_inter / t_shipped:.2g}x"


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    # 1. The model against measured RTL.  Exact, or stop.
    section("1. Model against section 2.3's measured cycles")
    s = shipped(SCAN, DIGITAL_RUN, TD)
    i_unfolded = interchanged(SCAN, DIGITAL_RUN, TD, restore=True)
    i_folded = interchanged(SCAN, DIGITAL_RUN, TD)
    print(f"  Tw = {SCAN} (scan), Ts = {DIGITAL_RUN} (S_RUN), Td = {TD} (row write)")
    print(f"  as shipped            model {s:>8,}   measured {MEASURED_SHIPPED:>8,}")
    print(f"  interchanged          model {i_unfolded:>8,}   measured"
          f" {MEASURED_INTERCHANGED:>8,}")
    print(f"  restore folded        model {i_folded:>8,}   measured"
          f" {MEASURED_INTERCHANGED - MEASURED_FOLD_SAVES:>8,} (71,168 - 15,872)")
    assert s == MEASURED_SHIPPED
    assert i_unfolded == MEASURED_INTERCHANGED
    assert i_unfolded - i_folded == MEASURED_FOLD_SAVES
    print(f"  exact.  Gains: {s / i_unfolded:.2f}x measured, {s / i_folded:.2f}x folded")

    # 2. Break-even, and a check that Ts really drops out of it.
    section("2. Break-even between the loop orders")
    for restore in (False, True):
        tw_star = break_even(TD, restore=restore)
        for ts in (1, 18, 1000):
            gap = shipped(tw_star, ts, TD) - interchanged(tw_star, ts, TD, restore=restore)
            assert abs(gap) < 1e-6 * shipped(tw_star, ts, TD), (restore, ts, gap)
        label = "restore unfolded" if restore else "restore folded"
        print(f"  {label:<17} Tw* = {tw_star:.2f} cycles"
              f" = {tw_star / TD:.3f} * Td   (same at Ts = 1, 18, 1000)")

    # 3. Tiles in physical time, the c930 as host.
    section("3. Tiles in physical time, c930 host at 100 MHz (Td = 80 ns)")
    settle_ns = settle_s(TFLN_BW_HZ) * 1e9
    print(f"  TFLN settle, single pole at {TFLN_BW_HZ / 1e9:.0f} GHz, 8-bit:"
          f" {settle_ns * 1e3:.0f} ps")
    scan_ns = SCAN * CYCLE_NS
    td_ns = TD * CYCLE_NS
    tiles = (
        # name, Tw ns, Ts ns
        ("thermo-optic, 10 us settle", scan_ns + 10_000.0, 50.0),
        ("TFLN, scanned weights", scan_ns + settle_ns, CYCLE_NS),
        ("TFLN, resident weights", 0.0, CYCLE_NS),
    )
    print(f"  {'tile':<27}{'Tw':>10}{'Ts':>8}{'as shipped':>12}"
          f"{'interchanged':>14}  winner")
    for name, tw, ts in tiles:
        a, b = shipped(tw, ts, td_ns), interchanged(tw, ts, td_ns)
        print(f"  {name:<27}{fmt_time_ns(tw):>10}{fmt_time_ns(ts):>8}"
              f"{fmt_time_ns(a):>12}{fmt_time_ns(b):>14}  {winner(a, b)}")
    a, b = shipped(10_000.0, 50.0, 0.0), interchanged(10_000.0, 50.0, 0.0)
    print(f"  (section 2.1's idealised example, Tw = 10 us, Td = 0:"
          f" {fmt_time_ns(a)} vs {fmt_time_ns(b)}, {a / b:.0f}x)")
    ts_share = M * NT * KT * CYCLE_NS
    a = shipped(0.0, CYCLE_NS, td_ns)
    print(f"  resident, as shipped: {ts_share / 1e3:.2f} us of one-cycle shots"
          f" + {M * NT * td_ns / 1e3:.2f} us of row writes = {a / 1e3:.1f} us")

    # 4. The named sweep points, in core cycles.
    section("4. Section 6.2 sweep points, core cycles (Td = 8, restore folded)")
    points = (
        # name, PTA_TW, PTA_TS, weights scanned by the core?
        ("TO-1ms", 100_000, 5, True),
        ("TO-10us", 1_000, 5, True),
        ("EO-scan", 0, 1, True),
        ("EO-res", 0, 1, False),
    )
    print(f"  {'point':<9}{'PTA_TW':>8}{'PTA_TS':>8}{'Tw':>9}"
          f"{'as shipped':>12}{'interchanged':>14}  winner")
    for name, pta_tw, pta_ts, scanned in points:
        tw = (SCAN if scanned else 0) + pta_tw
        a, b = shipped(tw, pta_ts, TD), interchanged(tw, pta_ts, TD)
        print(f"  {name:<9}{pta_tw:>8,}{pta_ts:>8}{tw:>9,}"
              f"{fmt_cycles(a):>12}{fmt_cycles(b):>14}  {winner(a, b)}"
              f"   [{a:,} / {b:,}]")
    print(f"  operands the DMA fetches for this GEMM: A {M}x{K} + B {K}x{N}"
          f" = {M * K + K * N:,}")

    # 4b. What the core actually spends at those points, and why it differs.
    section("4b. The same points as the core spends them (C2's gate, measured)")
    band = core_band()
    print(f"  {'point':<9}{'model band':>20}{'measured':>11}"
          f"{'6.2 table':>11}{'delta':>10}  = restore + shot")
    for name, pta_tw, pta_ts, scanned in points:
        hi = core_cycles(pta_tw, pta_ts, scanned=scanned)
        tw = (SCAN if scanned else 0) + pta_tw
        folded = interchanged(tw, pta_ts, TD)
        if name not in MEASURED_C2:
            print(f"  {name:<9}{hi - band:>9,}..{hi:<9,}{'-':>11}"
                  f"{fmt_cycles(folded):>11}{'-':>10}  needs Nt*Kt banks (6.2)")
            continue
        got = MEASURED_C2[name]
        assert hi - band <= got <= hi, (name, hi - band, got, hi)
        restore = M * (KT - 1) * N
        # The shot's excess over 6.2's Ts, taken from the measurement rather than
        # from the band's upper end, so the itemisation below is exact.
        scan = (N * K if scanned else 0) + NT * KT * (pta_tw + (pta_tw & 1))
        shot_over = (got - scan - restore - M * KT * N) - M * NT * KT * pta_ts
        # The two structural differences account for the whole gap, exactly.
        assert got - folded == restore + shot_over, (name, got - folded,
                                                     restore, shot_over)
        print(f"  {name:<9}{hi - band:>9,}..{hi:<9,}{got:>11,}"
              f"{fmt_cycles(folded):>11}{got - folded:>+10,}"
              f"  = {restore:,} + {shot_over:,}")
    print(f"  the restore is 2.1's unfolded M*Nt*(Kt-1)*Td, which 2.3 measures at"
          f" {MEASURED_FOLD_SAVES:,} and 6.2's table folds away")
    print(f"  the shot's floor is {SHOT_FLOOR} cycles, so 6.2's Ts = 1 and Ts = 5"
          f" are both unreachable; the nearest are {SHOT_FLOOR} and {4 + SHOT_FLOOR}")
    print(f"  Tw = nc*kr + PTA_TW and Td = nc; at this shape the tile is full, so"
          f" they are 6.2's {SCAN} and {TD}")

    # 5. Sweeping PTA_TW between them.
    section("5. PTA_TW swept at PTA_TS = 1: interchange gain (shipped / interchanged)")
    print(f"  {'PTA_TW':>8}{'two banks, scanned':>20}{'resident banks':>16}")
    for pta_tw in (0, 1, 2, 4, 8, 16, 32, 64, 256, 1_000, 10_000, 100_000):
        g_scan = shipped(SCAN + pta_tw, 1, TD) / interchanged(SCAN + pta_tw, 1, TD)
        g_res = shipped(pta_tw, 1, TD) / interchanged(pta_tw, 1, TD)
        print(f"  {pta_tw:>8,}{g_scan:>19.2f}x{g_res:>15.2f}x")
    print(f"  the gain tends to M = {M} as PTA_TW grows; the resident column crosses 1"
          f" at Tw* = {break_even(TD):.2f}, the scanned one never can (Tw >= {SCAN})")


if __name__ == "__main__":
    main()
