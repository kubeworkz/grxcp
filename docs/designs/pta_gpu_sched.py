"""
G1: the three weight-set policies of pta_gpu_integration.md section 3, priced.

The GPU has a scheduling problem the CPU path does not.  A weight-stationary
analog tile says only work whose weight set is programmed may run, and changing it
costs Tw; a warp scheduler that round-robins across CTAs with different weight
sets thrashes the mesh.  Section 3 gives three answers and says the interesting
work is choosing between them with numbers.  This is that pricing, as the first
half of G1 -- the second half is a SimX run, and the proposal to grxgpu says what
it has to do.  Nothing here patches grxgpu (AGENTS.md section 2); it reads its
configuration and its KMU and prices what they already do.

THE POLICIES (pta_gpu_integration.md section 3):

  1. affinity    the CTA dispatcher binds CTAs sharing a B tile to one cluster
                 and issues them contiguously
  2. cache       W weight banks per tile, LRU, so W weight sets are hot.  The
                 c930's two-bank structure is W = 2
  3. declared    the kernel names its weight slot in the launch descriptor and a
                 mismatch serialises on the tile's own lock

Policies 1 and 3 are orders; policy 2 is a capacity.  They are not three
alternatives on one axis, which is why they are swept against each other here
rather than ranked.

WHAT THE G100 ALREADY HAS, read out of grxgpu rather than assumed.
`sim/simx/kmu/kmu.cpp` walks the grid as a nested cluster tiling: `cluster_dim` is
a 3-D tile of the grid, filled by `intra_offset` before `group_origin` advances,
and it is settable at run time through VX_DCR_KMU_CLUSTER_DIM_X/Y/Z.  So policy 1
needs no hardware -- it is `cluster_dim` chosen so a cluster's CTA tile shares its
B tile.  Section 3 calls policy 1 "cheapest in hardware, requires the runtime to
expose the weight set as a dispatch attribute"; the attribute exists, and what is
missing is the runtime choosing it.

THE MODEL.  A tile holds one R x C weight set, so a GEMM's weight sets are the
(kr, nc) blocks of B and there are (K/R) * (N/C) of them.  A shot pushes one A row
through and costs PTA_TS + SHOT_FLOOR, the floor the c930's C2 gate measured.  The
three policies differ ONLY in the order the tile sees those weight sets, so the
shot term is identical across them and what is priced is the weight movement.
That is why this can be a model rather than a simulation: the quantity in question
is a property of the reference stream.

TWO COSTS, NOT ONE.  A reference to a weight set the tile is already driving is
free.  A reference to one of the other W-1 resident banks is a SELECT and costs
the settle alone -- the c930 measured that at one cycle (MB_TW).  A reference to a
set that is not resident is a LOAD and costs the scan as well: R * C beats at one
element a beat, which is a DXA K-major transfer and the same serial discipline the
c930's S_PRELOAD has.  Counting these separately is what makes W worth sweeping;
counting them together makes the banks look useless.

Measured anchors, reused rather than restated (pta_tw_sweep.py):
    the shot floor PTA_TS + 2, from grx930 `make core_c2`
    the resident bank select Tw = 1, from grx930 `make core_mb`
    the four section 6.2 points and their settles

Assumed, and stated because it is not measured: that a DXA K-major transfer
delivers one weight element per beat (the GPU document's section 2 reading of
`dest_kmajor`), and that a cluster owns one tile (its section 2 scope).

Standard library only.  Run:  python3 docs/designs/pta_gpu_sched.py
"""
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_tw_sweep as sweep

# ---- the G100's configuration, from grxgpu/VX_config.toml -----------------
NUM_CLUSTERS = 8

# ---- the tile, and the shot ------------------------------------------------
# The emulated geometry first, then wider ones pta_chiplet_link.py also carries.
# R is the operand depth a shot consumes, C the outputs it produces.
GEOMETRIES = ((8, 8), (32, 32), (128, 64))

SHOT_FLOOR = sweep.SHOT_FLOOR          # 2: register in, register out
MB_TW = sweep.MB_TW                    # 1: the resident bank select's own cycle

# ---- the Tw axis: section 6.2's points -------------------------------------
# (name, settle cycles, PTA_TS).  The settle is paid on a select and on a load;
# the scan only on a load.
POINTS = (
    ("TO-1ms",  100_000, 5),
    ("TO-10us",   1_000, 5),
    ("EO-scan",       0, 1),
    ("EO-res",    MB_TW, 1),
)

# ---- a real kernel's tiling ------------------------------------------------
# A CUTLASS-class SGEMM: 4096^3 with a 128x128 output tile and a 32-deep K step.
# BK is what sets the period of the cycle the tile sees, so it is the number that
# matters most below.
KERNEL = dict(M=4096, N=4096, K=4096, BM=128, BN=128, BK=32)


def tile_sets(kernel, R, C):
    """(kr, nc) counts, and the per-CTA fan-out of one k step."""
    return (kernel['K'] // R, kernel['N'] // C,
            kernel['BK'] // R, kernel['BN'] // C)


def stream_k_inner(kernel, R, C, cluster):
    """One cluster's weight-set stream in the natural GPU order, as a generator.

    The grid is (n-blocks, m-blocks) and a CTA loops k inside, so the weight set
    changes WITHIN a CTA -- the fact the policies have to contend with, and easy
    to miss when thinking of a CTA as owning one weight set.  The KMU's default
    walk has cluster_dim = 1, so CTAs go round robin over clusters in grid order
    with x (the n block) fastest.
    """
    mt = kernel['M'] // kernel['BM']
    nt = kernel['N'] // kernel['BN']
    kt = kernel['K'] // kernel['BK']
    _, ncs, kr_per, nc_per = tile_sets(kernel, R, C)
    cta = 0
    for mb in range(mt):
        for nb in range(nt):
            mine = (cta % NUM_CLUSTERS) == cluster
            cta += 1
            if not mine:
                continue
            for k in range(kt):
                for i in range(kr_per):
                    kr = k * kr_per + i
                    base = kr * ncs + nb * nc_per
                    for j in range(nc_per):
                        yield base + j


def stream_affinity(kernel, R, C, cluster):
    """Policy 1: cluster_dim grouping the CTAs that share a B tile.

    A CTA's B tile is fixed by its n block, so the grouping is over m -- a cluster
    takes a column of the grid, cluster_dim = (1, mt) in the KMU's terms, which is
    a DCR write and no hardware.  The k loop still runs inside each CTA, which is
    the whole question.
    """
    mt = kernel['M'] // kernel['BM']
    nt = kernel['N'] // kernel['BN']
    kt = kernel['K'] // kernel['BK']
    _, ncs, kr_per, nc_per = tile_sets(kernel, R, C)
    for nb in range(nt):
        if (nb % NUM_CLUSTERS) != cluster:
            continue
        for mb in range(mt):                      # every row block, same B tile
            for k in range(kt):
                for i in range(kr_per):
                    kr = k * kr_per + i
                    base = kr * ncs + nb * nc_per
                    for j in range(nc_per):
                        yield base + j


def stream_declared(kernel, R, C, cluster):
    """Policy 3: the kernel names its slot, so it can be written k-OUTER.

    This is the policy's real content and section 3's wording does not contain it.
    Making the slot a launch attribute lets the k loop be hoisted ABOVE the grid --
    one launch per weight set, every A row streamed through it -- so the tile sees
    each weight set once instead of once per CTA.  The serialisation section 3
    describes is then the cost of getting the declaration wrong, not the cost of
    the policy.
    """
    krs, ncs, _, _ = tile_sets(kernel, R, C)
    i = 0
    for kr in range(krs):
        for nc in range(ncs):
            if (i % NUM_CLUSTERS) == cluster:
                yield kr * ncs + nc
            i += 1


def walk(stream, W):
    """Selects and loads on one tile's stream with W banks, LRU.

    Free when the set is already driving the tile; a SELECT when it is in another
    resident bank; a LOAD when it is not resident at all.  W = 1 is no cache, so
    every change is a load.
    """
    banks = OrderedDict()
    cur = None
    refs = selects = loads = 0
    for wid in stream:
        refs += 1
        if wid == cur:
            continue
        if wid in banks:
            banks.move_to_end(wid)
            selects += 1
        else:
            loads += 1
            banks[wid] = True
            if len(banks) > W:
                banks.popitem(last=False)
        cur = wid
    return refs, selects, loads


def movement_cycles(selects, loads, settle, R, C):
    """The weight-movement term: a select pays the settle, a load the scan too."""
    return selects * settle + loads * (R * C + settle)


def shot_cycles(kernel, R, C, pta_ts):
    """One cluster's shots.  Identical across the three policies, by construction."""
    shots = (kernel['M'] * (kernel['K'] // R) * (kernel['N'] // C)) // NUM_CLUSTERS
    return shots * (pta_ts + SHOT_FLOOR)


POLICIES = (("natural", stream_k_inner),
            ("affinity", stream_affinity),
            ("declared", stream_declared))


def main():
    k = KERNEL
    mt, nt, kt = k['M'] // k['BM'], k['N'] // k['BN'], k['K'] // k['BK']
    print("G1, part one: the three weight-set policies priced on a reference")
    print("stream.  The second half is SimX; this says what it should find.")
    print()
    print(f"kernel: C[{k['M']}x{k['N']}] = A[{k['M']}x{k['K']}] . "
          f"B[{k['K']}x{k['N']}], CTA tile {k['BM']}x{k['BN']}, BK={k['BK']}")
    print(f"  grid {nt} x {mt} = {nt * mt} CTAs over {NUM_CLUSTERS} clusters,"
          f" each CTA looping {kt} k steps")
    print()

    R0, C0 = GEOMETRIES[0]
    _, _, kr0, nc0 = tile_sets(k, R0, C0)
    print(f"At the emulated {R0}x{C0} tile a CTA's k step is {kr0} weight sets"
          f" down K and {nc0} across N, so one CTA touches"
          f" {kt * kr0 * nc0} in sequence.")
    print("  The weight set changes INSIDE a CTA, not between CTAs, and the")
    print("  stream a tile sees is cyclic with that period.")
    print()

    for R, C in GEOMETRIES:
        if k['BK'] % R or k['BN'] % C:
            print(f"tile {R}x{C}: skipped -- BK and BN must divide by R and C"
                  f" for a CTA's B tile to tile exactly")
            print()
            continue
        krs, ncs, kr_per, nc_per = tile_sets(k, R, C)
        period = kr_per * nc_per * kt
        print(f"tile {R}x{C}: {krs * ncs:,} weight sets in the GEMM,"
              f" {R * C} beats to scan one, a cluster's stream cycles with"
              f" period {period}")
        print(f"  {'W':>6}  {'natural sel/load':>22} {'affinity sel/load':>22}"
              f" {'declared sel/load':>22}")
        for W in (1, 2, 4, period // 2, period, 2 * period):
            if W < 1:
                continue
            cells = []
            for _, fn in POLICIES:
                refs, sel, ld = walk(fn(k, R, C, 0), W)
                cells.append(f"{sel:>10,} /{ld:>10,}")
            print(f"  {W:>6}  {cells[0]:>22} {cells[1]:>22} {cells[2]:>22}")
        refs_n, _, _ = walk(stream_k_inner(k, R, C, 0), 1)
        refs_d, _, _ = walk(stream_declared(k, R, C, 0), 1)
        print(f"  references in a cluster's stream: natural {refs_n:,},"
              f" declared {refs_d:,}")
        print()

    # ---- the same thing in cycles, at the four points ---------------------
    R, C = GEOMETRIES[0]
    _, _, kr_per, nc_per = tile_sets(k, R, C)
    period = kr_per * nc_per * kt
    sc = {name: shot_cycles(k, R, C, ts) for name, _, ts in POINTS}
    print(f"Cycles for one cluster at the {R}x{C} tile.  W = 2 is the c930's"
          f" structure; W = {period} is the stream's whole period.")
    print(f"  {'point':<9} {'W':>5} {'policy':<9} {'weight cyc':>16}"
          f" {'shot cyc':>16} {'weight share':>13}")
    for name, settle, pta_ts in POINTS:
        for W in (2, period):
            for label, fn in POLICIES:
                _, sel, ld = walk(fn(k, R, C, 0), W)
                wc = movement_cycles(sel, ld, settle, R, C)
                share = wc / (wc + sc[name]) if (wc + sc[name]) else 0.0
                print(f"  {name:<9} {W:>5} {label:<9} {wc:>16,}"
                      f" {sc[name]:>16,} {share:>12.1%}")
        print()

    findings()
    checks()


def checks():
    """Every claim the proposal makes, as an assert.

    The first version of this file asserted that affinity never helps.  It does,
    and the assert caught it: what is true is narrower and more useful.
    """
    k = KERNEL
    R, C = GEOMETRIES[0]
    kt = k['K'] // k['BK']
    _, _, kr_per, nc_per = tile_sets(k, R, C)
    period = kr_per * nc_per * kt
    mt = k['M'] // k['BM']

    def w(fn, W):
        return walk(fn(k, R, C, 0), W)

    # 1. THE NULL RESULT, properly bounded.  At any W below the stream's period
    #    affinity is bit-identical to the natural order -- not close, identical --
    #    because the cycling is intra-CTA and re-ordering CTAs cannot touch it.
    #    W below the period is every buildable W: the period here is 8192 banks.
    for W in (1, 2, 4, 8, 16, period // 2, period - 1):
        assert w(stream_k_inner, W) == w(stream_affinity, W), (W,)

    # 2. At the period, and not before, affinity collapses the loads by the row
    #    blocks that share a B tile.
    refs, sel2, ld2 = w(stream_k_inner, 2)
    _, sel_a, ld_a = w(stream_affinity, period)
    assert ld_a * mt == ld2, (ld_a, mt, ld2)

    # 3. The natural order gets nothing even at twice the period: its cluster
    #    sees CTAs from several n blocks, so its working set is a multiple of the
    #    cycle however many banks there are.
    assert w(stream_k_inner, 2 * period) == (refs, 0, refs)

    # 4. The declared policy reaches affinity's best load count with ONE bank.
    refs_d, sel_d, ld_d = w(stream_declared, 1)
    assert ld_d == ld_a, (ld_d, ld_a)
    assert sel_d == 0

    # 5. And it strictly dominates, because affinity pays a select on every
    #    reference that is not a load while the declared stream has none.
    assert sel_a == refs - ld_a, (sel_a, refs, ld_a)
    for name, settle, _ in POINTS:
        a = movement_cycles(sel_a, ld_a, settle, R, C)
        d = movement_cycles(sel_d, ld_d, settle, R, C)
        assert d <= a, (name, d, a)

    # 6. The anchors are the measured ones.
    assert SHOT_FLOOR == 2 and MB_TW == 1
    assert movement_cycles(1, 0, MB_TW, R, C) == 1          # a resident select
    assert movement_cycles(0, 1, 0, R, C) == R * C          # EO-scan is the scan

    print("checks: all pass.")


def findings():
    """What the numbers above say, with the speedups that decide it."""
    k = KERNEL
    R, C = GEOMETRIES[0]
    kt = k['K'] // k['BK']
    _, _, kr_per, nc_per = tile_sets(k, R, C)
    period = kr_per * nc_per * kt
    mt = k['M'] // k['BM']

    print("G1's answer, four readings.")
    print()
    print(f"1. THE NULL RESULT IS REAL, AND NARROWER THAN EXPECTED. At any W")
    print(f"   below the stream's period, affinity is BIT-IDENTICAL to the")
    print(f"   natural order -- same selects, same loads, at W = 1, 2, 4, 16,")
    print(f"   {period // 2}. The period is {period} banks at the {R}x{C} tile")
    print(f"   ({kr_per} x {nc_per} weight sets per k step, {kt} k steps), so")
    print(f"   every buildable W is in that regime. The c930's W = 2 is.")
    print()
    print(f"2. AFFINITY DOES HELP, BUT ONLY AT THAT CAPACITY. At W = {period}")
    print(f"   its loads fall by {mt}x, the row blocks sharing a B tile. Below")
    print(f"   it, nothing. So affinity and the cache are not alternatives on")
    print(f"   one axis: each is worthless without the other. The natural order")
    print(f"   gets nothing even at 2x the period, because a cluster sees CTAs")
    print(f"   from several n blocks and its working set is a multiple of the")
    print(f"   cycle however many banks there are.")
    print()
    print(f"3. THE DECLARED POLICY REACHES AFFINITY'S BEST LOAD COUNT WITH ONE")
    print(f"   BANK, and pays no selects at all. Naming the slot is what lets")
    print(f"   the k loop be hoisted ABOVE the grid, and that is where the {mt}x")
    print(f"   comes from -- not from the scheduler. It is also the only one of")
    print(f"   the three that needs no hardware and no dispatcher change.")
    print()

    # The decision number: total cycles, best policy against the natural order.
    print("4. WHAT IT IS WORTH, end to end, one cluster:")
    print(f"   {'point':<9} {'natural W=2':>16} {'declared W=1':>16} {'speedup':>9}")
    for name, settle, pta_ts in POINTS:
        sc = shot_cycles(k, R, C, pta_ts)
        _, sn, ln = walk(stream_k_inner(k, R, C, 0), 2)
        _, sd, ld = walk(stream_declared(k, R, C, 0), 1)
        tot_n = movement_cycles(sn, ln, settle, R, C) + sc
        tot_d = movement_cycles(sd, ld, settle, R, C) + sc
        print(f"   {name:<9} {tot_n:>16,} {tot_d:>16,} {tot_n / tot_d:>8.2f}x")
    print()
    print("   The 32x is the LOAD reduction; end to end the policy choice is")
    print("   worth 25x at TO-1ms, 2.1x at TO-10us and 1.16x at both Pockels")
    print("   points, because the shot term it cannot touch takes over. That is")
    print("   the same ordering the CPU path's 2.1 cost model gives and for the")
    print("   same reason: once Tw is small the weight movement stops being the")
    print("   term that matters. The GPU's scheduling problem is a thermo-optic")
    print("   problem, and on a Pockels tile it is worth 16%.")
    print()
    print("WHAT THIS MODEL ASSUMES, and what SimX has to settle:")
    print(f"   - that a cluster sees every {NUM_CLUSTERS}th CTA in grid order."
          f" The KMU is ONE per")
    print("     processor and cores pull from it on demand, so the real split is")
    print("     whatever 16 cores' progress makes it, not a round robin.")
    print("   - that affinity can route a whole n column to one cluster. The")
    print("     mechanism is cluster_dim, but cta_dispatcher.cpp bounds a CTA")
    print("     cluster by LMEM co-residency and warp slots (usable_slots), so a")
    print("     column of 32 row blocks may not fit one. Affinity is modelled at")
    print("     its theoretical best here and still loses, so the bound only")
    print("     strengthens reading 3.")
    print("   - that LMEM port contention does not reorder the stream. SimX")
    print("     models that arbiter and this does not.")
    print()


if __name__ == '__main__':
    main()
