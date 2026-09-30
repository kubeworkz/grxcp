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

# ---- measured, MB's gate (grx930 `make core_mb`, BANKS=32) -----------------
# 6.2's EO-res point, measured instead of modelled: the tile holds a bank per
# (N tile, K tile), so a weight program is the select.  MB_TW is that select's own
# cycle -- step MB's "if a bank select takes a cycle in RTL, the gate counts that
# cycle as Tw", and the 2.1 break-even sits at 8, so one cycle leaves EO-res in
# the resident regime.
MB_TW = 1
MEASURED_MB = {
    "interchanged": 44_608,
    "m-outer": 16_896,
}

# ---- measured, C4(c) (grx930 `make pta_sweep`) -----------------------------
# The same four points on the SoC, driven by firmware through MMIO.  The shape is
# the SoC's NPU -- MAX_M=8, MAX_K=16, MAX_N=12 -- which cannot be asked for this
# file's M=64 N=8 K=256 at all, the gap 6.1 records about its baseline.  So these
# are NOT comparable with the figures above; what the two shapes share is the
# model, which holds at both.
SOC_M, SOC_N, SOC_K = 8, 12, 16
MEASURED_C4C = {
    # name:      (PTA_TW, PTA_TS, resident, cycles, weight movement)
    "TO-1ms":    (100_000, 5, False, 402_528, 400_288),
    "TO-10us":   (  1_000, 5, False,   6_528,   4_288),
    "EO-scan":   (      0, 1, False,   2_528,     288),
    "EO-res":    (      0, 1, True,    2_336,     100),
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


def core_resident(pta_ts, m=M, n=N, k=K, rows=NUM_ROWS, cols=NUM_COLS):
    """EO-res in the interchanged order, as the core runs it (MB).

    `core_cycles` with the scan replaced by the select: the weights are already at
    the tile, so S_WLOAD spends its bookkeeping cycle and leaves.  Everything else
    is unchanged, which is the point -- resident buys this order only the scan,
    because it already loaded each tile once.
    """
    nt = -(-n // cols)
    kt = -(-k // rows)
    select = nt * kt * MB_TW
    restore = m * (kt - 1) * n
    write = m * kt * n
    shot = m * nt * kt * (pta_ts + SHOT_FLOOR)
    return select + restore + write + shot


def core_shipped_resident(pta_ts, m=M, n=N, k=K, rows=NUM_ROWS, cols=NUM_COLS):
    """EO-res in the m-outer order, as the core runs it (MB).

    A select and a shot per (output row, N tile, K tile), and one row write per
    (output row, N tile) -- nc columns each, so N a row.  No restore: m-outer holds
    the running sum in acc[] across the K loop, which is the whole reason it writes
    once a row, and the reason 2.1 says a resident tile wants this order back.
    """
    nt = -(-n // cols)
    kt = -(-k // rows)
    shots = m * nt * kt
    return shots * MB_TW + shots * (pta_ts + SHOT_FLOOR) + m * n


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

    # 4c. EO-res in both orders, resident (MB's gate, measured).
    section("4c. EO-res with Nt*Kt resident banks, both orders (MB, measured)")
    band = core_band()
    hi_i = core_resident(1)
    hi_o = core_shipped_resident(1)
    got_i = MEASURED_MB["interchanged"]
    got_o = MEASURED_MB["m-outer"]
    assert hi_i - band <= got_i <= hi_i, (got_i, hi_i, band)
    assert hi_o - band <= got_o <= hi_o, (got_o, hi_o, band)
    print(f"  {'order':<14}{'model band':>20}{'measured':>11}  Tw = the select")
    print(f"  {'interchanged':<14}{hi_i - band:>9,}..{hi_i:<9,}{got_i:>11,}")
    print(f"  {'m-outer':<14}{hi_o - band:>9,}..{hi_o:<9,}{got_o:>11,}")
    # The verdict, and the band on it: each order's shots land where the hop's
    # entry parity puts them, and the two orders do not land together.
    print(f"  the shipped order wins by {got_i / got_o:.2f}x"
          f"  (the model's band on the ratio: {(hi_i - band) / hi_o:.2f}x"
          f" .. {hi_i / (hi_o - band):.2f}x)")
    print(f"  2.1 at Tw = {MB_TW}, Ts = {1 + SHOT_FLOOR}:"
          f" {interchanged(MB_TW, 1 + SHOT_FLOOR, TD) / shipped(MB_TW, 1 + SHOT_FLOOR, TD):.2f}x"
          f" folded, {interchanged(MB_TW, 1 + SHOT_FLOOR, TD, restore=True) / shipped(MB_TW, 1 + SHOT_FLOOR, TD):.2f}x"
          f" unfolded as the core is built")
    print(f"  6.2's EO-res row said 7.2x, which assumed Ts = 1; C2 measured the"
          f" shot's floor and MB measured the orders")

    # 4d. The same points on the SoC (C4(c), measured).
    section("4d. Section 6.2's points on the SoC, at the SoC's own shape (C4(c))")
    snt = -(-SOC_N // NUM_COLS)
    skt = -(-SOC_K // NUM_ROWS)
    print(f"  shape M={SOC_M} N={SOC_N} K={SOC_K} (Nt={snt} Kt={skt}) -- this SoC's"
          f" NPU, not 6.2's; the two are not comparable")
    print(f"  {'point':<9}{'measured':>10}{'model':>9}{'outside':>9}{'share':>7}"
          f"   {'wmove':>10}{'want':>10}")
    for name, (tw, ts, resident, got, wmove) in MEASURED_C4C.items():
        if resident:
            want_c = core_resident(ts, m=SOC_M, n=SOC_N, k=SOC_K)
            # The select's cycle a program, plus the restore.
            want_w = snt * skt * MB_TW + SOC_M * (skt - 1) * SOC_N
        else:
            want_c = core_cycles(tw, ts, m=SOC_M, n=SOC_N, k=SOC_K)
            want_w = SOC_N * SOC_K + snt * skt * (tw + (tw & 1)) \
                     + SOC_M * (skt - 1) * SOC_N
        # The model owns the weight movement exactly: S_WLOAD and S_ACCLD have no
        # feed dependence, so this is an equality and not a bound.
        assert wmove == want_w, (name, wmove, want_w)
        # What it does not own: the A-row wait, above all.  Reported, not fitted.
        outside = got - want_c
        assert outside > 0, (name, got, want_c)
        print(f"  {name:<9}{got:>10,}{want_c:>9,}{outside:>9,}"
              f"{100.0 * outside / got:>6.0f}%   {wmove:>10,}{want_w:>10,}")
    print("  the weight movement is the model's to the cycle at every point;"
          " the rest is the feed")
    print("  the operands are the same size at every point, so what moves is the"
          " SHARE, not the term")
    print("  arow_stall_cnt has no CSR, so the attribution is a subtraction and"
          " not a measurement (track F, F0)")

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
