#!/usr/bin/env python3
"""Compare measured device cycles against the golden baselines in ci/perf/.

AGENTS.md section 4 has said since the first commit that perf baselines are
golden data and that a moved number means real cycles moved. There were no
baselines. Nothing compared any cycle count to any stored number: the benches
printed, a human read the print, and a regression was caught only if someone
happened to remember last week's figure. That is what this closes.

EXACT, NOT A BAND -- and that is a measurement, not a preference.

  Three consecutive runs of tests/bench/block_cycles and tests/bench/gemm_cycles
  on simx produced byte-identical output, every stage, every shape. Rebuilding
  src/libs/kernels_all.cpp produced a bit-identical .vxbin. So the tolerance is
  ZERO by default: any change at all is a real change in what the model does,
  and a band would only hide small ones.

  A metric may carry a non-zero tolerance in its baseline's "tolerance" map, and
  the report says so wherever one is used, because a silent band is exactly the
  thing this file exists to prevent. None is set today.

  WHAT EXACT DOES NOT MEAN. It does not mean one stage's number is independent
  of the others. Planting a wasted loop in dnn_gelu and nothing else moved gelu
  +42.3% AND moved all ten other stages, between -6.0% and +8.7%, with their
  source untouched: the .vxbin relinks and every kernel's addresses move.
  Reverting restored all 152 metrics to exact, and the planted build was itself
  reproducible run to run, so that spread is layout and not noise.

  The consequence is a workflow, not a tolerance. Touch a device kernel and
  expect the whole file to move; the report ranks movers by magnitude so the
  one you caused is the first line, and the regenerated diff is what a reviewer
  reads. A band wide enough to absorb 8.7% of layout would also swallow a
  genuine 8% regression, which is a worse trade than asking a reviewer to look.

RAW INTEGERS ONLY. Spans and warp counts are stored; shares, cycles per element
and speedups are DERIVED here and printed. A baseline holding "4.6%" drifts
against its own rounding and needs a tolerance to survive it. One holding 13834
does not, and it is also the number a person can go and reproduce.

WHAT A RED GATE MEANS. Not "make it green." Either root-cause the delta, or
regenerate with --regenerate as a reviewed step, so the diff of the baseline
file is in the change alongside the code that moved it. That is the whole point
of storing them in git.

This compares files; it does not run anything. ci/run_real.sh's own bench steps
write them with --out, so the numbers gated here are the numbers that were
printed, and no bench runs twice to be checked.

  ./ci/check_perf.py --results <dir> [--regenerate]
  ./ci/check_perf.py --self-test
"""

import contextlib
import io
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINES = ROOT / "ci/perf/baselines"

# results file written by run_real.sh  ->  baseline file under ci/perf/baselines
EXPECTED = [
    ("perf_block_rb.json", "block_cycles.register-blocked.json"),
    ("perf_block_naive.json", "block_cycles.naive.json"),
    ("perf_gemm.json", "gemm_cycles.json"),
    ("perf_gemv.json", "gemv_cycles.json"),
]


def flatten(doc):
    """The comparable surface of a result document: metric path -> integer.

    Structure -- shape list, stage names and order -- is deliberately part of
    the KEY, not the value. A renamed or reordered stage then shows up as a
    structural difference rather than as a set of numbers that moved, which is
    a different problem with a different fix.
    """
    out = {}
    if doc.get("bench") == "block_cycles":
        for sh in doc["shapes"]:
            tag = f"S={sh['seq']}"
            for i, st in enumerate(sh["stages"]):
                key = f"{tag}[{i}] {st['name']}"
                out[key + " .span"] = st["span"]
                out[key + " .warps"] = st["warps"]
                # The most warps live at once. Pinned because it is what proves
                # a span came from ONE launch: MCYCLE restarts at zero at every
                # launch, so a stage that quietly starts spanning two moves its
                # cycles and its maxLive together, and pinning only the cycles
                # is how the last one went unnoticed for three commits.
                if "maxLive" in st:
                    out[key + " .maxLive"] = st["maxLive"]
                out[key + " .valid"] = 1 if st["valid"] else 0
    elif doc.get("bench") == "gemm_cycles":
        for sh in doc["shapes"]:
            tag = f"{sh['m']}x{sh['n']}x{sh['k']}"
            for field in ("sgemm_span", "tensor_span", "nslots", "busy_median"):
                out[f"{tag} .{field}"] = sh[field]
    elif doc.get("bench") == "gemv_cycles":
        # The transpose is part of the key because sgemv is two traversals
        # wearing one name: OP_N is one thread per output row, OP_T is one warp
        # per output column. A change can help one and hurt the other, and a
        # key that dropped the letter would average them.
        for p in doc["points"]:
            tag = f"{p['trans']} {p['m']}x{p['n']}"
            for field in ("span", "busy_median", "warps"):
                out[f"{tag} .{field}"] = p[field]
    else:
        raise ValueError(f"unknown bench: {doc.get('bench')!r}")
    return out


def block_totals(doc):
    """Per shape: (seq, total cycles, attention's share as a fraction)."""
    rows = []
    for sh in doc["shapes"]:
        total = sum(st["span"] for st in sh["stages"] if st["valid"])
        attn = sum(
            st["span"]
            for st in sh["stages"]
            if st["valid"] and st["name"].startswith("attention")
        )
        rows.append((sh["seq"], total, attn / total if total else 0.0))
    return rows


def compare(name, measured, baseline):
    """Returns (verdict, lines). Verdict is 'ok', 'moved', 'structure' or 'device'."""
    lines = []

    # The device record first, and it is a REFUSAL rather than a failure. Cycle
    # counts from a different core width, a different SM count or a different
    # simulator are not a regression and not a pass -- they are numbers about
    # another machine, and reporting them either way would be a lie.
    if measured.get("device") != baseline.get("device"):
        lines.append("        measured on:  " + json.dumps(measured.get("device")))
        lines.append("        baseline for: " + json.dumps(baseline.get("device")))
        return "device", lines

    if measured.get("config") != baseline.get("config"):
        lines.append("        measured config:  " + json.dumps(measured.get("config")))
        lines.append("        baseline config:  " + json.dumps(baseline.get("config")))
        return "device", lines

    m, b = flatten(measured), flatten(baseline)
    only_m, only_b = sorted(set(m) - set(b)), sorted(set(b) - set(m))
    if only_m or only_b:
        for k in only_b[:8]:
            lines.append(f"        gone from this build: {k}")
        for k in only_m[:8]:
            lines.append(f"        new in this build:    {k}")
        extra = len(only_m) + len(only_b) - len(only_b[:8]) - len(only_m[:8])
        if extra:
            lines.append(f"        (+{extra} more)")
        lines.append("        the SHAPE of the measurement changed, so no number here")
        lines.append("        is comparable. Regenerate if this was intended.")
        return "structure", lines

    tol = baseline.get("tolerance", {})
    moved = []
    for key in sorted(m):
        want, got = b[key], m[key]
        band = float(tol.get(key, 0.0))
        if want and abs(got - want) <= abs(want) * band:
            if band and got != want:
                lines.append(
                    f"        within tolerance {band:g}: {key}  {want} -> {got}"
                )
            continue
        if got != want:
            moved.append((key, want, got))

    if moved:
        # Ranked by magnitude, largest first, and the largest is named on its
        # own line -- because a device-source change moves EVERY stage, not the
        # one it touched.
        #
        # Measured: planting a wasted 32-iteration loop in dnn_gelu and nothing
        # else moved gelu +42.3%, and moved all ten other stages too, between
        # -6.0% and +8.7%, with their source untouched. The .vxbin is relinked
        # and every kernel's addresses move with it. Reverting restored all 152
        # metrics to exact, and the planted build was itself reproducible run to
        # run, so this is layout, not noise.
        #
        # So RANK is the signal when the image changed, and the exact value is
        # the signal when it did not. Both are true at once and the report has
        # to make the first one readable, or the real mover hides among ten
        # incidental ones.
        def pct_of(item):
            key, want, got = item
            return 100.0 * (got - want) / want if want else float("inf")

        moved.sort(key=lambda it: -abs(pct_of(it)))
        top_key, top_want, top_got = moved[0]
        lines.append(
            f"        largest mover: {top_key}  {top_want} -> {top_got}"
            f"   {pct_of(moved[0]):+.1f}%   ({len(moved)} of {len(m)} metrics moved)"
        )
        for item in moved:
            key, want, got = item
            lines.append(f"        {key}:  {want} -> {got}   {pct_of(item):+.1f}%")
        return "moved", lines

    lines.append(f"        {len(m)} metrics, all exact")
    return "ok", lines


def derived_report(docs):
    """The numbers a person reads, computed from the raw ones that are gated."""
    out = []
    gem = docs.get("perf_gemm.json")
    if gem:
        out.append("  gemm_cycles, derived:")
        for sh in gem["shapes"]:
            elems = sh["m"] * sh["n"]
            per = sh["sgemm_span"] / elems
            tper = sh["tensor_span"] / elems if sh["tensor_span"] else 0.0
            sp = f"{per / tper:6.2f}x" if tper else "     -"
            out.append(
                f"        {sh['m']:3d}x{sh['n']:3d}x{sh['k']:3d}   sgemm {per:8.1f}"
                f"   GemmEx {tper:7.1f}   {sp}"
            )

    rb, naive = docs.get("perf_block_rb.json"), docs.get("perf_block_naive.json")
    if rb:
        out.append("  block_cycles, derived:")
        rows = {s: (t, a) for s, t, a in block_totals(rb)}
        nrows = {s: (t, a) for s, t, a in block_totals(naive)} if naive else {}
        for seq in sorted(rows):
            total, attn = rows[seq]
            line = (
                f"        S={seq:<3d} total {total:8d}   attention {100 * attn:4.1f}%"
            )
            if seq in nrows:
                ntotal, nattn = nrows[seq]
                line += (
                    f"   |  naive {ntotal:8d} ({100 * nattn:4.1f}%)"
                    f"   blocking {ntotal / total:.2f}x"
                )
            out.append(line)
    return out


def controls(docs):
    """Checks between documents, which no single baseline can express."""
    lines, ok = [], True
    rb, naive = docs.get("perf_block_rb.json"), docs.get("perf_block_naive.json")
    if rb and naive:
        # THE CONTROL THAT MAKES THE LABELS MEAN ANYTHING. The bench writes
        # "register-blocked" because GRXBLAS_SGEMM_NAIVE was unset, not because
        # it watched the kernel get chosen -- the blocked kernels are optional
        # symbol lookups and fall back silently. If the two configurations
        # produce the same cycles, the selection did nothing and BOTH baselines
        # are records of the same kernel under two names.
        #
        # The file name still says "register-blocked" while the rule now picks
        # the 2D micro-tile. Kept: the baseline pair is "the rule's choice
        # against the reference", and renaming it would break every stored
        # metric key for a label. What it holds is checked by the metrics, not
        # by the name.
        rmap = {s: t for s, t, _ in block_totals(rb)}
        nmap = {s: t for s, t, _ in block_totals(naive)}
        same = [s for s in rmap if s in nmap and rmap[s] == nmap[s]]
        if same:
            lines.append(
                "  FAIL  naive and the rule's choice measured IDENTICAL cycles at "
                f"S={same}."
            )
            lines.append(
                "        The kernel selection did not take effect, so one of these"
            )
            lines.append("        two baselines is not what its label says.")
            ok = False
        else:
            worst = min(nmap[s] / rmap[s] for s in rmap if s in nmap)
            lines.append(
                "  ok    the two sgemm kernels are distinguishable; register "
                f"blocking is worth at least {worst:.2f}x on the whole block"
            )
    return ok, lines


def run(results_dir, regenerate, baselines=None):
    # baselines is a parameter, not the constant, so the self-test can point it
    # at an empty directory. The first version read the constant: once real
    # baselines existed on disk, the self-test's missing-baseline case stopped
    # reaching the missing-baseline path and started failing. A test whose
    # verdict depends on the repository around it is not testing the code.
    baselines = baselines or BASELINES
    docs, rc = {}, 0

    for result_name, baseline_name in EXPECTED:
        rpath = results_dir / result_name
        bpath = baselines / baseline_name
        if not rpath.exists():
            print(f"  FAIL  {result_name} was not produced by this run")
            rc = 1
            continue
        measured = json.loads(rpath.read_text())
        docs[result_name] = measured

        if regenerate:
            bpath.parent.mkdir(parents=True, exist_ok=True)
            bpath.write_text(json.dumps(measured, indent=2) + "\n")
            try:
                shown = bpath.relative_to(ROOT)
            except ValueError:
                shown = bpath
            print(f"  wrote {shown}")
            continue

        if not bpath.exists():
            try:
                shown = bpath.relative_to(ROOT)
            except ValueError:
                shown = bpath
            print(f"  FAIL  no baseline: {shown}")
            print("        run with --regenerate, and review the file it writes")
            rc = 1
            continue

        baseline = json.loads(bpath.read_text())
        verdict, lines = compare(result_name, measured, baseline)
        label = baseline_name[: -len(".json")]
        if verdict == "ok":
            print(f"  ok    {label}")
        elif verdict == "device":
            print(f"  REFUSED  {label}: this is a different machine or configuration")
            rc = 1
        elif verdict == "structure":
            print(f"  FAIL  {label}: the measurement changed shape")
            rc = 1
        else:
            print(f"  FAIL  {label}: cycles moved")
            rc = 1
        for line in lines:
            print(line)

    for line in derived_report(docs):
        print(line)

    if not regenerate:
        ok, lines = controls(docs)
        for line in lines:
            print(line)
        if not ok:
            rc = 1

    if regenerate:
        print("  baselines rewritten. REVIEW THE DIFF -- that is the whole")
        print("  mechanism: a moved number is meant to be visible in the change.")
    elif rc == 0:
        print("  ok    every gated cycle count matches its baseline exactly")
    return rc


# --------------------------------------------------------------------------
# Self-test. Runs with no device and no results, so tier 1 can check the
# comparator's logic even though only tier 2 can produce real numbers.

def _doc(total=1000, attn=100, name="attention (2 GEMMs+mask+softmax)",
         device=None, config=None):
    return {
        "bench": "block_cycles",
        "config": config or {"sgemm": "register-blocked"},
        "device": device or {"name": "X", "sms": 1, "warp": 4, "mhz": 400},
        "shapes": [
            {
                "seq": 8, "dim": 16, "heads": 2, "ff": 64,
                "stages": [
                    {"name": "layernorm 1", "span": total - attn, "warps": 8,
                     "valid": True},
                    {"name": name, "span": attn, "warps": 96, "valid": True},
                ],
            }
        ],
    }


def self_test():
    ok = True

    def check(what, got, want):
        nonlocal ok
        if got != want:
            print(f"  FAIL  self-test: {what}: got {got!r}, want {want!r}")
            ok = False

    base = _doc()
    check("identical documents pass", compare("t", _doc(), base)[0], "ok")
    check("one cycle of drift fails", compare("t", _doc(total=1001), base)[0], "moved")
    check(
        "a different device is refused, not compared",
        compare("t", _doc(device={"name": "Y", "sms": 8, "warp": 32, "mhz": 400}),
                base)[0],
        "device",
    )
    check(
        "a different config is refused, not compared",
        compare("t", _doc(config={"sgemm": "naive"}), base)[0],
        "device",
    )
    check(
        "a renamed stage is a structural change, not a regression",
        compare("t", _doc(name="attention v2"), base)[0],
        "structure",
    )

    # A tolerance, if one is ever set, must be honoured AND announced.
    with_tol = dict(base)
    with_tol["tolerance"] = {"S=8[0] layernorm 1 .span": 0.05}
    verdict, lines = compare("t", _doc(total=1020), with_tol)
    check("a declared tolerance absorbs a small move", verdict, "ok")
    check(
        "and says so",
        any("within tolerance" in x for x in lines),
        True,
    )
    check(
        "but not a large one",
        compare("t", _doc(total=1200), with_tol)[0],
        "moved",
    )

    # The cross-document control: identical naive and rb numbers mean the
    # kernel selection did nothing, whatever the labels say.
    rb = _doc()
    naive = _doc(config={"sgemm": "naive"})
    good, _ = controls({"perf_block_rb.json": rb,
                        "perf_block_naive.json": _doc(total=1500,
                                                      config={"sgemm": "naive"})})
    check("distinguishable kernels pass the control", good, True)
    bad, _ = controls({"perf_block_rb.json": rb, "perf_block_naive.json": naive})
    check("identical kernels under two labels fail the control", bad, False)

    # And the whole path, over files, including the missing-baseline case.
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        (d / "perf_block_rb.json").write_text(json.dumps(_doc()))
        empty = d / "no-baselines-here"
        empty.mkdir()
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            rc = run(d, regenerate=False, baselines=empty)
        check("a run with no baselines on disk cannot pass", rc != 0, True)
        check(
            "and says which file is missing",
            "no baseline" in captured.getvalue(),
            True,
        )

    if ok:
        print("  ok    self-test: the comparator refuses, fails and passes as stated")
    return ok


def main(argv):
    if "--self-test" in argv:
        return 0 if self_test() else 1
    if not self_test():
        return 1

    regenerate = "--regenerate" in argv
    results = None
    for i, a in enumerate(argv):
        if a == "--results" and i + 1 < len(argv):
            results = pathlib.Path(argv[i + 1])
    if results is None:
        print("usage: check_perf.py --results <dir> [--regenerate]")
        return 2
    if not results.is_dir():
        print(f"  FAIL  no results directory: {results}")
        return 1
    return run(results, regenerate)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
