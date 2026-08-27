#!/usr/bin/env python3
"""Does grxBLAS pick the right sgemm kernel for the block, IN THE BLOCK?

tests/bench/sgemm_sweep.cpp answers the same question for GEMMs measured alone.
That is not the same question, and for most of a year the two appeared to give
opposite answers: the sweep said blocking wins at attention's 8x8x8 batch-2
shape by 1.39x, and the block profile said shipping that rule cost 3766 cycles.
The rule that loses in isolation was kept on the strength of the block number.

The block number was wrong. Attention is four launches sharing one probe
buffer, MCYCLE restarts at zero at every launch, and the "cost of attention"
was a maximum over four unrelated clocks -- see include/grx/grx_cycles.h. The
conflict was never real.

WHAT THIS DOES INSTEAD, and why it is not just the sweep again. It runs the
whole transformer block, flips EXACTLY ONE of its sgemm calls to a different
kernel, and runs it again -- once per call per alternative, at every shape the
bench measures. Nothing else moves. That is the only way to price one stage's
kernel choice where the stage actually lives, next to the launches it shares a
machine with.

Every flip is CONFIRMED against the trace rather than assumed. Asking for a
kernel is not the same as getting one: a floor in the decision, a module without
that entry point, and a misspelled variable all look identical from out here,
and all three would show up as "this flip changed nothing" -- which reads like
evidence about the kernel and is evidence about the harness.

ONLY CALLS INSIDE A MEASURED STAGE ARE PRICED, and finding out why took a wrong
turn worth recording. The block bench makes one sgemm before any probe is
attached -- an availability check, 1x1x1 -- and flipping THAT call moved the
block total by 659 cycles. It is in no stage; its only route into a stage-sum is
the machine state it leaves for the first stage that runs. The first attempt at
a fix was to warm both kernel paths before measuring, on the theory that a
profiler should not measure its own warm-up. It did not work: the warm-up calls
simply became the last thing to run before stage one, and flipping THEM moved
the total instead. The artifact is structural -- whatever runs last before the
first measured stage perturbs it -- so the honest response is to price only what
is measured. The library reports which calls carried a probe; those are the
ones.

Calls are selected by INDEX, not by shape, because shape does not tell the
block's stages apart: at S=8 attention's two GEMMs are both 8x8x8, and at S=16
the qkv projection and attention's output GEMM are both 8x16x16. See the
measurement-hook note in src/libs/grxblas/grxblas.cpp.

THE GATE: no single flip may make the block faster. A flip that helps is the
rule picking the slower kernel for that call, in place, with everything else
held still -- which is the only kind of wrongness that can make a program
slower than it was before the tuned kernel existed. Declining a win is
reported, not gated, for the same reason the isolated sweep does that.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def totals_of(doc):
    """Block total per shape, from the same valid stages the bench reports."""
    return [sum(st["span"] for st in sh["stages"] if st["valid"])
            for sh in doc["shapes"]]


def score(baseline, flips):
    """Which flips made the block FASTER, ranked. Pure, so it can be self-tested.

    `flips` is a list of dicts with 'call', 'shape', 'to' and 'totals'. A flip
    is scored per shape, because a call belongs to one shape and leaves the
    others untouched -- summing across shapes would dilute a real loss by the
    zeros of every shape the call is not in.
    """
    helped, hurt = [], []
    for f in flips:
        best = None
        for i, (b, t) in enumerate(zip(baseline, f["totals"])):
            d = t - b
            if d == 0:
                continue
            if best is None or abs(d) > abs(best[1]):
                best = (i, d)
        if best is None:
            continue
        rec = dict(f, shape_index=best[0], delta=best[1])
        (helped if best[1] < 0 else hurt).append(rec)
    helped.sort(key=lambda r: r["delta"])
    hurt.sort(key=lambda r: -r["delta"])
    return helped, hurt


KERNELS = ("naive", "rb", "2d", "4x2", "4x4")

# The measurement hook that forces each kernel. Not API -- see the note in
# src/libs/grxblas/grxblas.cpp. Each takes a list of call indices.
ENV_FOR = {
    "naive": "GRXBLAS_SGEMM_NAIVE",
    "rb": "GRXBLAS_SGEMM_RB",
    "2d": "GRXBLAS_SGEMM_2D",
    "4x2": "GRXBLAS_SGEMM_4X2",
    "4x4": "GRXBLAS_SGEMM_4X4",
}


def run_block(binary, env_extra, out_json, trace=None):
    env = dict(os.environ)
    for k in list(ENV_FOR.values()) + ["GRXBLAS_SGEMM_TRACE"]:
        env.pop(k, None)
    env.update(env_extra)
    if trace:
        env["GRXBLAS_SGEMM_TRACE"] = trace
    # --sweep: the bench's differential control is about the block as it ships,
    # and every run here has one of its GEMMs forced onto another kernel. See
    # the flag's note in tests/bench/block_cycles.cpp. Nothing else is
    # suppressed and the JSON is the same.
    r = subprocess.run([binary, "--out", out_json, "--sweep"], env=env,
                       capture_output=True, text=True)
    if r.returncode == 77:
        return None, 77
    if r.returncode != 0:
        sys.stdout.write(r.stdout[-2000:])
        return None, r.returncode
    with open(out_json) as f:
        return json.load(f), 0


def read_trace(path):
    """call index -> the row the library wrote for it."""
    with open(path) as f:
        return {int(r["call"]): r for r in csv.DictReader(f)}


def shape_of(row):
    return (f'{row["m"]}x{row["n"]}x{row["k"]}'
            + (f'b{row["batch"]}' if row["batch"] != "1" else "")
            + ("T" if row["transa"] == "1" else ""))


def sweep(binary, out_path=None):
    tmp = tempfile.mkdtemp(prefix="blocksweep.")
    trace = os.path.join(tmp, "census.csv")
    base, rc = run_block(binary, {}, os.path.join(tmp, "base.json"), trace)
    if rc == 77:
        print("  no device or no kernels; skipping")
        return 77
    if base is None:
        print("  the block bench failed before any flip was tried")
        return rc or 1
    baseline = totals_of(base)
    census = read_trace(trace)
    seqs = [sh["seq"] for sh in base["shapes"]]

    print(f"  the block issues {len(census)} sgemm calls; "
          f"baseline totals {baseline}")
    print(f"  {'call':>4}  {'shape':<14} {'ran':>5}  {'instead':>7} "
          f"{'d(block)':>9}  where")

    flips, unreachable, outside = [], 0, 0
    for call in sorted(census):
        row = census[call]
        ran = row["kernel"]
        if row.get("probed") != "1":
            # Outside every measured stage. Named rather than dropped: a sweep
            # that silently skips calls reads as coverage it does not have.
            print(f"  {call:>4}  {shape_of(row):<14} {ran:>5}  {'--':>7} "
                  f"{'--':>9}  no probe: in no measured stage")
            outside += 1
            continue
        for alt in KERNELS:
            if alt == ran:
                continue
            ftrace = os.path.join(tmp, f"t{call}_{alt}.csv")
            doc, rc = run_block(binary, {ENV_FOR[alt]: f"#{call}"},
                                os.path.join(tmp, f"f{call}_{alt}.json"), ftrace)
            if doc is None:
                print(f"  {call:>4}  flip to {alt} failed (rc={rc})")
                return rc or 1
            got = read_trace(ftrace).get(call, {})
            if got.get("kernel") != alt:
                # Not a result about the kernel. Printed anyway, because a
                # silently skipped alternative is how a sweep comes to claim
                # coverage it does not have.
                unreachable += 1
                print(f"  {call:>4}  {shape_of(row):<14} {ran:>5}  {alt:>7} "
                      f"{'--':>9}  not reachable: {got.get('why', '?')}")
                continue
            t = totals_of(doc)
            d = [t[i] - baseline[i] for i in range(len(baseline))]
            worst = max(range(len(d)), key=lambda i: abs(d[i]))
            flips.append(dict(call=call, shape=shape_of(row), frm=ran, to=alt,
                              totals=t))
            where = f"S={seqs[worst]}" if d[worst] else "no shape moved"
            print(f"  {call:>4}  {shape_of(row):<14} {ran:>5}  {alt:>7} "
                  f"{d[worst]:>+9d}  {where}")

    helped, hurt = score(baseline, flips)
    print()
    print(f"  {len(flips)} flips measured, one call at a time; "
          f"{unreachable} alternatives were not reachable; "
          f"{outside} calls sat outside every measured stage.")
    print(f"  {len(hurt)} made the block slower (the rule was right there).")
    print(f"  {len(helped)} made it FASTER (the rule picked a slower kernel).")
    for h in helped:
        print(f"      call {h['call']} {h['shape']}: {h['frm']} -> "
              f"{h['to']} saves {-h['delta']} cycles")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(dict(baseline=baseline, flips=flips), f, indent=1)
        print(f"  wrote {out_path}")

    ok = not helped
    print()
    print(f"  {'ok  ' if ok else 'FAIL'}  no call's kernel choice can be "
          f"improved by changing it in place")
    return 0 if ok else 1


def self_test():
    """The scoring, without a device. Planted answers, checked both ways."""
    fails = 0

    def check(what, got, want):
        nonlocal fails
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
        if not ok:
            print(f"          got {got!r}, wanted {want!r}")
            fails += 1

    base = [1000, 2000]
    # A flip that costs on shape 0 and does nothing on shape 1: the rule was
    # right, and the untouched shape must not dilute it.
    helped, hurt = score(base, [dict(call=1, shape="a", to="naive",
                                     totals=[1100, 2000])])
    check("a flip that costs is scored as the rule being right",
          (len(helped), len(hurt)), (0, 1))
    check("and it is priced on the shape it moved", hurt[0]["delta"], 100)

    # A flip that helps must be caught even when another shape moved further
    # in the other direction -- summing would hide it.
    helped, _ = score(base, [dict(call=2, shape="b", to="rb",
                                  totals=[900, 2000])])
    check("a flip that helps is caught", len(helped), 1)
    check("and named with the saving", helped[0]["delta"], -100)

    # Ranking: the biggest saving first, not the first one seen.
    helped, _ = score(base, [
        dict(call=3, shape="c", to="rb", totals=[990, 2000]),
        dict(call=4, shape="d", to="rb", totals=[500, 2000]),
    ])
    check("savings are ranked by size", [h["call"] for h in helped], [4, 3])

    # A flip that moves nothing at all is not evidence either way.
    helped, hurt = score(base, [dict(call=5, shape="e", to="rb",
                                     totals=[1000, 2000])])
    check("a flip that moves nothing is not counted",
          (len(helped), len(hurt)), (0, 0))
    return 1 if fails else 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", help="path to the block_cycles bench")
    ap.add_argument("--json", help="write the measured flips here")
    ap.add_argument("--self-test", action="store_true",
                    help="check the scoring without a device")
    a = ap.parse_args(argv[1:])
    if a.self_test:
        return self_test()
    if not a.bin:
        print("error: --bin <block_cycles> is required", file=sys.stderr)
        return 2
    return sweep(a.bin, a.json)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
