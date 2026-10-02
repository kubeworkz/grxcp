"""
Section 8 item 3: which weight set occupies which bank, when the sets outnumber
the banks.

The item has been deferred since C1 with the condition "not before the Tw-to-Ts
ratio sweep says so".  The sweep has reported (C2, MB, C4(c), SoC-B), so this
settles the item with numbers instead of leaving it open.  It is a model, like
pta_feed_model.py and pta_gpu_sched.py, and it reuses their measured anchors.

WHAT THE QUESTION ACTUALLY IS.  Within one GEMM there is no policy question at
all: the interchanged nest section 2.1 ships visits each (N tile, K tile) once and
streams every A row through it, so each weight set is needed exactly once and any
replacement policy loads it exactly once.  The banks are not a within-GEMM cache.
What they buy is RESIDENCY ACROSS GEMMs -- MB's resident mode with WSKIP, where
the scan is paid once ever rather than once per GEMM, and C4(c) measured the
difference: at the SoC's shape a scanned point pays N*K = 192 beats plus a settle
per set, a resident one pays the select's single cycle.

So the policy question is: over a SEQUENCE of GEMMs, which sets stay.

THE WORKLOAD IS THE PROGRAM'S OWN.  D3 fixed the network at 784-100-10 with ReLU
so that C1, C3 and A3 report on the same one.  Inference over a batch runs layer 1
then layer 2 per input, so the tile's weight-set reference stream is CYCLIC with
the period of one forward pass -- and a cyclic stream is where LRU is not merely
unhelpful but worst-possible, which is the same structure G1 found on the GPU for
a different reason.

Measured anchors, reused rather than restated (pta_tw_sweep.py):
    the resident bank select, Tw = 1, from grx930 `make core_mb`
    the shot floor PTA_TS + 2, from grx930 `make core_c2`
    the four section 6.2 points and their settles

Assumed, and stated because it is not measured: that a weight scan moves one
element per beat (the c930's S_PRELOAD, and the DMA's K-major transfer), and that
a bank holds one R x C weight set.

Standard library only.  Run:  python3 docs/designs/pta_bank_policy.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_tw_sweep as sweep

MB_TW = sweep.MB_TW              # 1: the resident select's own cycle
SHOT_FLOOR = sweep.SHOT_FLOOR    # 2: register in, register out

# D3's network, fixed once for C1, C3 and A3.
LAYERS = (("fc1", 784, 100), ("fc2", 100, 10))

# The emulated tile, then wider ones pta_chiplet_link.py also carries.
GEOMETRIES = ((8, 8), (32, 32), (128, 64))

# Section 6.2's points: (name, settle cycles, PTA_TS).
POINTS = (("TO-1ms", 100_000, 5), ("TO-10us", 1_000, 5),
          ("EO-scan", 0, 1), ("EO-res", MB_TW, 1))

BATCH = 8          # forward passes, so a cyclic stream has somewhere to repeat


def ceil_div(a, b):
    return -(-a // b)


def layer_sets(kin, nout, R, C):
    """The (kr, nc) weight sets one layer's B needs at an R x C tile."""
    return ceil_div(kin, R) * ceil_div(nout, C)


def stream(R, C, batch=BATCH):
    """The tile's weight-set reference stream for `batch` forward passes.

    Ids are global across layers, so a bank holding one layer's set cannot be
    confused with another's.  Within a layer the order is the core's: N tile
    outer, K tile inner, which is the interchanged nest's order.
    """
    out = []
    base = 0
    bases = []
    for _, kin, nout in LAYERS:
        bases.append(base)
        base += layer_sets(kin, nout, R, C)
    for _ in range(batch):
        for li, (_, kin, nout) in enumerate(LAYERS):
            for nc in range(ceil_div(nout, C)):
                for kr in range(ceil_div(kin, R)):
                    out.append(bases[li] + nc * ceil_div(kin, R) + kr)
    return out


def stream_batched(R, C):
    """The same work with the batch INSIDE the weight set instead of outside.

    Batch-serial inference reloads every set once per input: the stream above is
    `batch` repeats of the whole network.  Batched inference visits each set once
    and streams all `batch` rows through it, which is exactly what section 2.1's
    interchange does for M within one GEMM, one level further out.  Same shots,
    same arithmetic, `batch` times fewer references.
    """
    base = 0
    out = []
    for _, kin, nout in LAYERS:
        for nc in range(ceil_div(nout, C)):
            for kr in range(ceil_div(kin, R)):
                out.append(base + nc * ceil_div(kin, R) + kr)
        base += layer_sets(kin, nout, R, C)
    return out


# ---- the policies ---------------------------------------------------------
# Each returns the number of MISSES: references whose set was not resident and
# so had to be scanned in.  A resident reference costs the select instead.

def misses_lru(refs, W):
    banks = []                       # least recently used first
    m = 0
    for r in refs:
        if r in banks:
            banks.remove(r)
            banks.append(r)
            continue
        m += 1
        banks.append(r)
        if len(banks) > W:
            banks.pop(0)
    return m


def misses_fifo(refs, W):
    banks = []
    held = set()
    m = 0
    for r in refs:
        if r in held:
            continue
        m += 1
        banks.append(r)
        held.add(r)
        if len(banks) > W:
            held.discard(banks.pop(0))
    return m


def misses_min(refs, W):
    """Belady: evict whatever is needed farthest in the future.  Not
    implementable -- it is the bound any policy is measured against."""
    nxt = {}                         # id -> list of future positions
    for i, r in enumerate(refs):
        nxt.setdefault(r, []).append(i)
    pos = {k: 0 for k in nxt}
    held = set()
    m = 0
    for i, r in enumerate(refs):
        pos[r] += 1
        if r in held:
            continue
        m += 1
        if len(held) < W:
            held.add(r)
            continue

        def next_use(x):
            lst, p = nxt[x], pos[x]
            return lst[p] if p < len(lst) else float('inf')

        victim = max(held, key=next_use)
        if next_use(victim) > i:
            held.discard(victim)
            held.add(r)
    return m


def misses_pin_smallest(refs, W, R, C):
    """Pin the smallest layer's sets, cycle the rest.

    The policy a glance at the workload suggests, and the one the hardware can
    actually do: a set of pinned banks plus one victim.  No history, no
    comparators -- which is why it is worth knowing whether it is enough.
    """
    sizes = [(layer_sets(kin, nout, R, C), i) for i, (_, kin, nout) in enumerate(LAYERS)]
    sizes.sort()
    pinned = set()
    base = 0
    bases = []
    for _, kin, nout in LAYERS:
        bases.append(base)
        base += layer_sets(kin, nout, R, C)
    budget = W
    for n, li in sizes:
        if n <= budget:
            pinned |= set(range(bases[li], bases[li] + n))
            budget -= n
    # Whatever is left cycles through one victim slot.
    held = set()
    seen_pinned = set()
    m = 0
    victim = None
    for r in refs:
        if r in pinned:
            if r in seen_pinned:
                continue
            m += 1                   # its one-off scan, then resident for ever
            seen_pinned.add(r)
            continue
        if r == victim:
            continue
        m += 1
        victim = r
    return m


def main():
    print("Section 8 item 3, settled: which weight set occupies which bank.")
    print()
    print("Within one GEMM there is no question -- the interchanged nest needs")
    print("each set once, so every policy loads it once.  The banks buy residency")
    print("ACROSS GEMMs, so the question is which sets stay over a sequence.")
    print()
    print(f"The workload is D3's network, {LAYERS[0][1]}-{LAYERS[0][2]}"
          f"-{LAYERS[1][2]}, run over a batch of {BATCH}:")
    for R, C in GEOMETRIES:
        tot = sum(layer_sets(kin, nout, R, C) for _, kin, nout in LAYERS)
        per = [(nm, layer_sets(kin, nout, R, C)) for nm, kin, nout in LAYERS]
        print(f"  tile {R:>3}x{C:<3}  " +
              ", ".join(f"{nm} {n:,}" for nm, n in per) +
              f"   total {tot:,} weight sets, {R * C} beats to scan one")
    print()

    for R, C in GEOMETRIES:
        refs = stream(R, C)
        total_sets = sum(layer_sets(kin, nout, R, C) for _, kin, nout in LAYERS)
        print(f"tile {R}x{C}: {len(refs):,} references over a period of"
              f" {total_sets:,}")
        print(f"  {'W':>6} {'LRU':>9} {'FIFO':>9} {'MIN':>9} {'pin':>9}"
              f"   {'best vs LRU':>12} {'of the period':>14}")
        ws = [2, 8, 32, 128, total_sets // 4, total_sets // 2, total_sets]
        for W in sorted(set(w for w in ws if w >= 1)):
            lru = misses_lru(refs, W)
            fifo = misses_fifo(refs, W)
            mn = misses_min(refs, W)
            pin = misses_pin_smallest(refs, W, R, C)
            best = min(lru, fifo, mn, pin)
            print(f"  {W:>6} {lru:>9,} {fifo:>9,} {mn:>9,} {pin:>9,}"
                  f"   {lru / best if best else 0:>11.2f}x {W / total_sets:>13.1%}")
        print()

    # ---- what it is worth in cycles -----------------------------------------
    R, C = GEOMETRIES[0]
    refs = stream(R, C)
    total_sets = sum(layer_sets(kin, nout, R, C) for _, kin, nout in LAYERS)
    # The shots are the same whatever the policy: one per A row per set.
    shots = len(refs) * 1            # one row per reference at this granularity
    print(f"Cycles at the {R}x{C} tile, W = 32 (MB's measured configuration)."
          f"  A miss costs the scan ({R * C} beats) plus the settle; a hit costs"
          f" the select ({MB_TW}).")
    print(f"  {'point':<9} {'policy':<6} {'misses':>9} {'weight cyc':>13}"
          f" {'shot cyc':>12} {'weight share':>13}")
    W = 32
    rows = (("LRU", misses_lru(refs, W)),
            ("MIN", misses_min(refs, W)),
            ("pin", misses_pin_smallest(refs, W, R, C)))
    for name, settle, pta_ts in POINTS:
        sc = shots * (pta_ts + SHOT_FLOOR)
        for pol, miss in rows:
            hits = len(refs) - miss
            wc = miss * (R * C + settle) + hits * MB_TW
            share = wc / (wc + sc) if (wc + sc) else 0.0
            print(f"  {name:<9} {pol:<6} {miss:>9,} {wc:>13,} {sc:>12,}"
                  f" {share:>12.1%}")
        print()

    # ---- the lever the table above points at --------------------------------
    batched = stream_batched(R, C)
    print(f"The same work with the batch INSIDE the weight set. Batch-serial")
    print(f"inference reloads every set once per input ({len(refs):,} references);")
    print(f"batched visits each once and streams {BATCH} rows through it"
          f" ({len(batched):,}).")
    print(f"  {'point':<9} {'order':<14} {'misses':>8} {'weight cyc':>13}"
          f" {'shot cyc':>12} {'total':>14} {'vs serial':>10}")
    for name, settle, pta_ts in POINTS:
        serial_m = misses_lru(refs, W)
        serial_w = serial_m * (R * C + settle) + (len(refs) - serial_m) * MB_TW
        serial_s = len(refs) * (pta_ts + SHOT_FLOOR)
        batch_m = misses_lru(batched, W)
        batch_w = batch_m * (R * C + settle) + (len(batched) - batch_m) * MB_TW
        batch_s = len(refs) * (pta_ts + SHOT_FLOOR)     # the same shots, exactly
        st, bt = serial_w + serial_s, batch_w + batch_s
        print(f"  {name:<9} {'batch-serial':<14} {serial_m:>8,} {serial_w:>13,}"
              f" {serial_s:>12,} {st:>14,} {'':>10}")
        print(f"  {name:<9} {'batched':<14} {batch_m:>8,} {batch_w:>13,}"
              f" {batch_s:>12,} {bt:>14,} {st / bt:>9.2f}x")
    print()

    findings(R, C, refs, total_sets)
    checks()


def findings(R, C, refs, total_sets):
    W = 32
    lru = misses_lru(refs, W)
    mn = misses_min(refs, W)
    pin = misses_pin_smallest(refs, W, R, C)
    print("What this says, four readings.")
    print()
    small = min(layer_sets(kin, nout, R, C) for _, kin, nout in LAYERS)
    print(f"1. AT THE BANK COUNTS THAT EXIST, ALMOST NO POLICY MATTERS. The")
    print(f"   working set is {total_sets:,} weight sets at the {R}x{C} tile and MB")
    print(f"   measured the tile at 32 banks, which is {32 / total_sets:.1%} of it. The")
    print(f"   best policy saves {100.0 * (lru - min(mn, pin)) / lru:.1f}% of the"
          f" scans against LRU ({lru:,} down to")
    print(f"   {min(mn, pin):,}), because there is almost nothing to keep.")
    print()
    print(f"   There is one threshold worth knowing. Pinning can do NOTHING until")
    print(f"   the banks hold the smallest layer whole, which for this network at")
    print(f"   this tile is {small} sets -- below that it is bit-identical to LRU. MB's")
    print(f"   32 banks, chosen for section 6.2's Nt*Kt, is the first count past it.")
    print()
    print("2. LRU IS NEVERTHELESS THE WORST CHOICE, and for a structural reason")
    print("   rather than a tuning one. A batch of forward passes is a CYCLIC")
    print("   reference stream, and on a cyclic stream LRU evicts precisely the")
    print("   set needed next. That is the same shape G1 found on the GPU, where")
    print("   affinity was bit-identical to doing nothing below the stream's")
    print("   period. Two tracks, two mechanisms, one conclusion: the tile's")
    print("   reference streams are cyclic, and caching cyclic streams does not")
    print("   work.")
    print()
    print("3. THE LEVER IS THE LOOP ORDER, NOT THE POLICY. The cycles table")
    print("   above is 95% to 100% weight movement at every point, because a")
    print(f"   {R * C}-beat scan is paid to do ONE shot. No replacement policy")
    print("   touches that -- the best buys 2.2%. Putting the batch inside the")
    print(f"   weight set instead of outside it cuts the scans {BATCH}x, which is")
    print("   section 2.1's interchange argument one level further out: hold the")
    print("   set, stream the rows. Same shots, same arithmetic.")
    print()
    print("   And the other axis is the tile's width, not its bank count. The")
    print(f"   same network needs {sum(layer_sets(kin, nout, 8, 8) for _, kin, nout in LAYERS):,}"
          f" weight sets at 8x8 and"
          f" {sum(layer_sets(kin, nout, 128, 64) for _, kin, nout in LAYERS)} at 128x64,")
    print("   where 32 banks hold the whole network and every policy collapses to")
    print("   one scan per set.")
    print()
    print("4. SO THE ITEM STAYS DEFERRED, WITH A CONDITION INSTEAD OF A FEELING.")
    print("   A policy is worth building when the banks approach the reference")
    print("   stream's period -- measured above, the crossover is around a quarter")
    print("   of it. Until then the c930's refusal to run a GEMM needing more")
    print("   tiles than it has banks is the right behaviour, because aliasing")
    print("   them would buy nothing measurable.")
    print()


def checks():
    """The claims, as asserts."""
    R, C = GEOMETRIES[0]
    refs = stream(R, C)
    total = sum(layer_sets(kin, nout, R, C) for _, kin, nout in LAYERS)

    # 1. Within one GEMM every policy is identical, because each set is needed
    #    once.  One forward pass IS one GEMM per layer, so a batch of 1 has no
    #    reuse at all and every policy must agree.
    one = stream(R, C, batch=1)
    for W in (2, 8, 32):
        a = misses_lru(one, W)
        assert a == len(one), (W, a, len(one))
        assert misses_fifo(one, W) == len(one)
        assert misses_min(one, W) == len(one)

    # 2. On the cyclic stream LRU takes no hits at all below the period.
    for W in (2, 8, 32, 128):
        assert misses_lru(refs, W) == len(refs), (W,)

    # 3. MIN beats LRU as soon as there is a batch, so the stream really is
    #    reusable and claim 2 is about the policy and not the workload.
    for W in (2, 8, 32, 128):
        assert misses_min(refs, W) < misses_lru(refs, W), (W,)

    # 4. pin has a THRESHOLD: it keeps the smallest layer and cycles the rest, so
    #    below that layer's size it pins nothing and is bit-identical to LRU.  The
    #    first version of this check asserted it always helps, and this caught it.
    small = min(layer_sets(kin, nout, R, C) for _, kin, nout in LAYERS)
    assert small == 26, small          # fc2 at the 8x8 tile
    for W in (2, 8, small - 1):
        assert misses_pin_smallest(refs, W, R, C) == misses_lru(refs, W), (W,)
    for W in (small, 32, 128):
        assert misses_pin_smallest(refs, W, R, C) < misses_lru(refs, W), (W,)

    # 5. And at the full period everything collapses to one scan per set: the
    #    bound, which says the model's accounting is right.
    assert misses_min(refs, total) == total, (misses_min(refs, total), total)

    # 6. Batching reduces the misses by exactly the batch, and nothing else:
    #    the arithmetic is identical, so this is a loop order and not a trade.
    batched = stream_batched(R, C)
    assert len(refs) == BATCH * len(batched), (len(refs), BATCH, len(batched))
    for W in (2, 8, 32):
        assert misses_lru(batched, W) * BATCH == misses_lru(refs, W), (W,)

    # 7. The anchors are the measured ones.
    assert MB_TW == 1 and SHOT_FLOOR == 2

    print("checks: all pass --")
    print("  with no batch every policy is identical: each set is needed once")
    print("  on the cyclic stream LRU takes no hits below the period")
    print("  MIN beats it at every W, so the stream is reusable")
    print(f"  pin is identical to LRU below {small} banks and better at or above")
    print("  at the full period every policy reaches one scan per set")
    print(f"  batching cuts the misses by exactly {BATCH}x, the batch itself")


if __name__ == '__main__':
    main()
