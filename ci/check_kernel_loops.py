#!/usr/bin/env python3
"""What the shipped kernels' hot loops are made of, gated against a baseline.

EVERY LARGE FINDING IN THE TUNING WORK WAS VISIBLE HERE FIRST, and none of them
was found by looking. They were found by measuring a speedup, disbelieving it,
and disassembling:

  * sgemm's k loop re-decided a loop-invariant transpose every iteration --
    64 instructions for 8 multiply-adds, of which 18 were conditional selects
    and 12 were index widenings. Worth 3x once hoisted.
  * sgemm_4x4 spilled seven slots into its k loop, which is why a wider tile
    with a better load count did not win. The address arithmetic was what would
    not fit, not the accumulators.
  * dnn_gelu carried twelve vx_split instructions on an elementwise kernel,
    because float selects compile to branches on this toolchain while integer
    ones become conditional moves.

All three are one static property of the shipped image: the shape of the
innermost loop that does the work. This gate pins that shape so a regression is
loud, and names the kernels that still spill or diverge so they do not become
invisible.

WHAT "THE HOT LOOP" MEANS HERE. Every backward branch inside a kernel bounds a
loop; the one picked is the INNERMOST loop that does floating-point work --
innermost meaning it encloses no other backward branch -- with the most float
work winning ties. That is a heuristic and it is stated as one: for a kernel
whose work is not floating-point (dnn_causal_mask) it falls back to the
innermost loop of any kind, and the baseline records what it picked rather than
pretending the number means more than it does.

The definition was "the loop with the most float operations" for exactly one
commit. Hoisting two loop-invariant null tests out of dnn_layernorm's element
loop split it into four sibling loops, and that rule then picked the ROW loop
enclosing all four: more instructions, more spills, more divergence reported for
a change that made the kernel 9% faster. A heuristic that reads a genuine
improvement as a regression is not measuring what it names.

WHAT IS GATED. Exact integers against ci/perf/baselines/kernel_loops.json, no
tolerance, for the same reason the cycle baselines have none: the disassembly of
a deterministic build is deterministic. A moved number means the compiler
produced different code, which is worth a human look whether it got better or
worse.

WHAT IS REPORTED AND NOT GATED. Which kernels still carry stack traffic or warp
divergence in that loop. Those are defects, they are named every run, and they
are not failures -- turning a known defect into a red gate on the day it is
discovered stops the suite being usable, and hiding it stops it being honest.

THE FIRST CENSUS OVER-REPORTED, and the correction is on the record because the
number was published. With the old "most float operations" rule this named five
kernels as spilling or diverging, dnn_layernorm worst at seven stack accesses
and four vx_split. Those instructions exist, but they are in the ROW loop -- once
per row -- not in the element loop the report implied. Picking the innermost
float loop instead, the only divergence left in the whole image is one vx_split
in each of the two tensor kernels, in a loop with no float work in it at all.

A per-row branch and a per-element branch differ by the row width. Saying so is
the difference between a census and a scare.

WHAT ins/fp DOES NOT TELL YOU, MEASURED. It is a rate, and it was read as a
worklist twice.

  * It does not say where the time is. dnn_add_bias headed this ranking at
    13.00 and is 3.4% of a transformer block; dnn_gelu sits at the BOTTOM at
    1.44 and is 16%. sgemv was second at 15.00 and does not appear in the block
    at all.
  * It does not say what the loop is waiting for. dnn_add_bias moves three
    words per fadd.s -- two loads and a store -- and its address arithmetic
    hides underneath them. Hoisting it took the loop from 13 instructions to
    11, exactly as predicted, and the block did not move: 155347 -> 155134 at
    S=8 and 323119 -> 323232 at S=16, the second of those slightly WORSE. In
    the same build the two bias stages moved in OPPOSITE directions, +3.5% and
    -9.3%, while untouched kernels moved +4.1% and +3.1%. That is the relink,
    not the change. It was reverted.

So the ranking below is printed with memory traffic beside it. A loop moving two
or more words per float op has somewhere for its arithmetic to hide, and the
only thing that settles whether a saving is real is a cycle count.

AND IT COUNTS ONE LOOP, NOT THE KERNEL. dnn_softmax made three passes over each
row and computed its exponential TWICE per element -- once to build the sum,
once to write the result. Keeping the exponential in the output row instead
halved the number of dev_exp calls and made softmax 1.39x to 1.74x faster over
S = 8 to 64, with output bit-identical to the last digit. This census reported
it as a REGRESSION: 34 instructions to 41, ins/fp 2.00 to 2.41, because the loop
it looks at absorbed the store that the deleted third pass used to do. The
kernel got smaller (259 bytes to 227) and its frame shrank (112 to 80) in the
same change.

Nothing is wrong with the number. It is the answer to "what is the innermost
float loop made of", and that question stops tracking cost the moment work moves
BETWEEN loops. Compare ins across a structural change and it will mislead; the
bench is what settles it, every time.

THE FLAG IS NOT A VERDICT, and sgemv is why. It carries the same marker at 2.0
words per float op, and hoisting ITS address arithmetic was worth 1.12x to 1.94x
on the span -- because it removed eight instructions of fifteen, not two of
thirteen. Memory traffic says the arithmetic CAN hide, not that it does. What
separated the two cases was the size of the saving, and neither the rate nor the
flag predicted it; the bench did.
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASELINE = os.path.join(HERE, "perf", "baselines", "kernel_loops.json")

# Runtime and libc symbols that come with the image and are nobody's kernel.
NOT_KERNELS = {"_start", "__init_tls", "__libc_init_array", "memcpy", "memset",
               "sqrtf"}

# Float ops for the purpose of "what is this loop for". fmv and the load/store
# forms are excluded: moving a float is not work on it.
NOT_REALLY_FP = {"flw", "fsw", "fmv.s", "fmv.w.x", "fmv.x.w"}

# WHAT EACH KERNEL IS FOR, because ins/fp is a cost and this census was read as
# a worklist.
#
# The ranking put sgemm at 24.00 instructions per float op and sgemm_rb at
# 13.25, ahead of every kernel a program actually reaches. Both numbers are
# correct and neither is a defect: sgemm is the ORACLE that every tuned kernel
# is checked against bit for bit, and sgemm_rb is the 4x1 rung of the tile
# ladder that the sweeps price the others against. Making either of them faster
# would destroy the instrument. sgemm_4x2 and sgemm_4x4 are staged: they exist,
# they are reachable by asking, and nothing has measured them into the rule.
#
# THE CENSUS CANNOT SEE THIS. Reachability is a property of the host library's
# decision function (decide_sgemm_kernel in src/libs/grxblas/grxblas.cpp), and
# this script reads a disassembly. So the roles are DECLARED here and PROVED
# elsewhere: tests/libs/test_grxblas_rb.cpp runs the shipping rule across the
# threshold and asserts the set of kernels it can pick. A label here that the
# rule contradicts is a failure there, which is the only arrangement in which
# writing it down is honest.
SHIPS, ORACLE, CONTROL, STAGED = "ships", "oracle", "control", "staged"
ROLES = {
    "sgemm":     (ORACLE,  "the reference; every tuned kernel is checked against it"),
    "sgemm_rb":  (CONTROL, "the 4x1 rung; the rule reaches it only if sgemm_2d is absent"),
    "sgemm_4x2": (STAGED,  "reachable by asking; not measured into the rule"),
    "sgemm_4x4": (STAGED,  "reachable by asking; spills, and the ladder says why"),
}


def role_of(name, roles=None):
    """(role, why). Anything not named in the table is on the shipping path."""
    return (ROLES if roles is None else roles).get(name, (SHIPS, ""))


def disassemble(elf, objdump):
    r = subprocess.run([objdump, "-d", elf], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"objdump failed on {elf}:\n{r.stderr[-800:]}")
    return r.stdout


def parse(text):
    """(address -> mnemonic+operands) and the symbol table, from objdump -d."""
    rows, syms = [], []
    for line in text.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+\S+\s+(.*)", line)
        if m:
            rows.append((int(m.group(1), 16), m.group(2).strip()))
            continue
        m = re.match(r"([0-9a-f]+)\s+<(\w+)>:", line)
        if m:
            syms.append((int(m.group(1), 16), m.group(2)))
    rows.sort()
    syms.sort()
    return rows, syms


def loop_stats(loop):
    ops = [t.split()[0] for t in loop]
    fp = sum(1 for o in ops
             if o.startswith("f") and o not in NOT_REALLY_FP)
    return {
        "ins": len(loop),
        "fp": fp,
        "loads": ops.count("flw"),
        "stores": ops.count("fsw"),
        "stack": sum(1 for t in loop if "(sp)" in t),
        "split": ops.count("vx_split") + ops.count("vx_split_n"),
        "join": ops.count("vx_join"),
    }


def back_edges(body, lo):
    """(start, end) of every backward branch's span, innermost first."""
    out = []
    for addr, text in body:
        if not re.match(r"(b[a-z]+|j)\s", text):
            continue
        m = re.search(r"(0x[0-9a-f]+)\s*<", text)
        if not m:
            continue
        target = int(m.group(1), 16)
        if lo <= target < addr:
            out.append((target, addr))
    out.sort(key=lambda e: e[1] - e[0])
    return out


def hot_loop(body, lo, hi):
    """The INNERMOST loop that does float work; most float work wins ties.

    "Innermost" means it encloses no other backward branch. That definition
    replaced "the loop with the most float operations", and the reason is worth
    keeping: hoisting two loop-invariant null tests out of dnn_layernorm's
    element loop split it into four sibling loops, and the old rule then picked
    the ROW loop that encloses all four -- reporting more instructions, more
    spills and more divergence for a change that made the kernel 9% faster. A
    heuristic that reads a genuine improvement as a regression is not measuring
    the thing it names.

    A kernel with no float work anywhere -- dnn_causal_mask -- falls back to the
    innermost loop of any kind, and the baseline records what that picked rather
    than pretending the number means more than it does.
    """
    edges = back_edges(body, lo)
    if not edges:
        return None

    def encloses_another(start, end):
        return any(start <= s and e <= end and (s, e) != (start, end)
                   for s, e in edges)

    best = None
    for start, end in edges:
        if encloses_another(start, end):
            continue
        loop = [t for a, t in body if start <= a <= end]
        st = loop_stats(loop)
        if st["fp"] == 0:
            continue
        key = (st["fp"], -(end - start))
        if best is None or key > best[0]:
            best = (key, st)
    if best:
        return best[1]
    # No float work anywhere: the innermost loop of any kind.
    start, end = edges[0]
    return loop_stats([t for a, t in body if start <= a <= end])


def census(text):
    rows, syms = parse(text)
    out = {}
    for i, (addr, name) in enumerate(syms):
        if name in NOT_KERNELS:
            continue
        end = syms[i + 1][0] if i + 1 < len(syms) else addr + 0x10000
        body = [(a, t) for a, t in rows if addr <= a < end]
        if not body:
            continue
        frame = 0
        for _, t in body[:3]:
            m = re.search(r"addi\s+sp, sp, -0x([0-9a-f]+)", t)
            if m:
                frame = int(m.group(1), 16)
        st = hot_loop(body, addr, end)
        entry = {"size": len(body), "frame": frame}
        entry.update(st if st else {"ins": 0, "fp": 0, "loads": 0,
                                    "stores": 0, "stack": 0, "split": 0,
                                    "join": 0})
        out[name] = entry
    return out


def report(measured, baseline, quiet=False, roles=None):
    """Print the census, compare, and return (ok, lines_of_concern).

    `roles` is the reachability table to apply, ROLES by default. It is a
    parameter so the self-test can run against its two-kernel sample image
    without the real table -- which names four kernels that image does not
    contain -- failing every unrelated check.
    """
    if roles is None:
        roles = ROLES
    if not quiet:
        print(f"  {'kernel':18s} {'ins':>5s} {'fp':>4s} {'ld':>4s} {'st':>4s} "
              f"{'stk':>4s} {'spl':>4s} {'join':>5s}   ins/fp")
    concerns = []
    for name in sorted(measured):
        m = measured[name]
        ratio = f"{m['ins'] / m['fp']:6.2f}" if m["fp"] else "     -"
        role, _ = role_of(name, roles)
        note = ""
        if m["stack"]:
            note += "  SPILLS"
        if m["split"]:
            note += "  DIVERGES"
        if not quiet:
            print(f"  {name:18s} {m['ins']:5d} {m['fp']:4d} {m['loads']:4d} "
                  f"{m.get('stores', 0):4d} {m['stack']:4d} {m['split']:4d} "
                  f"{m['join']:5d} {ratio}"
                  f"  {'' if role == SHIPS else role}{note}")
        if note:
            concerns.append((name, m["stack"], m["split"]))

    # The cost ranking, over the kernels a program can actually reach. Printed
    # separately from the table because the table is sorted by name and this is
    # the line someone reads to decide what to work on next -- and reading it
    # off the full table is how sgemm's 24.00 and sgemm_rb's 13.25 came to head
    # a list of things to fix. Both are instruments. Neither is a defect.
    if not quiet:
        ranked = sorted(
            ((m["ins"] / m["fp"], n) for n, m in measured.items()
             if m["fp"] and role_of(n, roles)[0] == SHIPS),
            reverse=True)
        if ranked:
            print()
            print("  cost per float op, SHIPPING kernels only"
                  " (the instruments are excluded and named above):")
            for r, n in ranked:
                m = measured[n]
                mem = (m["loads"] + m.get("stores", 0)) / m["fp"]
                # Two or more words moved per float op: the arithmetic has
                # somewhere to hide, so a small instruction saving may buy
                # nothing and only cycles can say. NOT "instructions do not
                # matter here" -- sgemv carries the same flag and its hoist was
                # worth 1.9x, because it removed EIGHT instructions of thirteen
                # rather than two. See the header for both results.
                flag = "   memory traffic dominates: price it in cycles" \
                       if mem >= 2.0 else ""
                print(f"      {n:18s} {r:6.2f}   {mem:4.1f} words/fp{flag}")

    if baseline is None:
        return True, concerns

    moved, added, gone = [], [], []
    for name, m in sorted(measured.items()):
        b = baseline.get(name)
        if b is None:
            added.append(name)
            continue
        for field in ("size", "frame", "ins", "fp", "loads", "stores",
                      "stack", "split", "join"):
            if m[field] != b.get(field):
                moved.append((name, field, b.get(field), m[field]))
    for name in sorted(baseline):
        if name not in measured:
            gone.append(name)

    # A role for a kernel that is not in the image is a claim about something
    # that no longer exists, and it is the way this table goes stale: the
    # kernel is deleted, the label survives, and the ranking silently starts
    # excluding a name nothing produces.
    stale_roles = sorted(n for n in roles if n not in measured)

    ok = not (moved or added or gone or stale_roles)
    if not quiet:
        if added:
            print(f"\n  new in this build: {', '.join(added)}")
        if gone:
            print(f"  gone from this build: {', '.join(gone)}")
        if stale_roles:
            print(f"  ROLES names kernels this image does not contain: "
                  f"{', '.join(stale_roles)}")
        for name, field, was, now in moved:
            print(f"  {name}.{field}: {was} -> {now}")
    return ok, concerns


def run(vxbin, objdump, regenerate):
    elf = vxbin[:-len(".vxbin")] + ".elf" if vxbin.endswith(".vxbin") else vxbin
    if not os.path.exists(elf):
        print(f"  no {elf}; the kernel build leaves one beside every .vxbin")
        return 77
    measured = census(disassemble(elf, objdump))
    if not measured:
        print("  no kernel symbols in the image")
        return 1

    baseline = None
    if os.path.exists(BASELINE) and not regenerate:
        with open(BASELINE) as f:
            baseline = json.load(f).get("kernels")

    ok, concerns = report(measured, baseline)

    if concerns:
        print()
        print("  standing defects in these hot loops, reported and not gated:")
        for name, stack, split in concerns:
            what = []
            if stack:
                what.append(f"{stack} stack access{'es' if stack > 1 else ''}")
            if split:
                what.append(f"{split} vx_split")
            print(f"      {name}: {', '.join(what)}")
        print("  A spill or a branch inside the loop that does the work is a"
              " defect.")
        print("  It is named every run so it cannot go quiet, and it is not a"
              " failure:")
        print("  a gate that turns red the day a defect is FOUND stops the"
              " suite being usable.")

    if regenerate:
        with open(BASELINE, "w") as f:
            json.dump({"note": "hot-loop census of the shipped kernel image;"
                               " see ci/check_kernel_loops.py",
                       "kernels": measured}, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"\n  wrote {BASELINE}. REVIEW THE DIFF -- a moved number means"
              " the compiler\n  produced different code, which is worth a look"
              " whichever way it went.")
        return 0

    print()
    if baseline is None:
        print("  ok    census taken; no baseline stored yet")
        return 0
    print(f"  {'ok  ' if ok else 'FAIL'}  every kernel's hot loop is the shape"
          f" the baseline records")
    return 0 if ok else 1


# --- the checker's own logic, without a toolchain -------------------------

SAMPLE = """
0000000180000000 <kernel_a>:
180000000: 00000013  addi\tsp, sp, -0x40
180000004: 00000013  flw\tft0, 0x0(a0)
180000008: 00000013  fmadd.s\tft1, ft0, ft0, ft1
18000000c: 00000013  addi\ta0, a0, 0x4
180000010: 00000013  bnez\ta1, 0x180000004 <kernel_a+0x4>
180000014: 00000013  ret
0000000180000018 <kernel_b>:
180000018: 00000013  addi\tsp, sp, -0x80
18000001c: 00000013  vx_split\ta0
180000020: 00000013  flw\tft0, 0x8(sp)
180000024: 00000013  fadd.s\tft1, ft0, ft1
180000028: 00000013  vx_join\ta0
18000002c: 00000013  bnez\ta1, 0x18000001c <kernel_b+0x4>
"""


def self_test():
    fails = 0

    def check(what, got, want):
        nonlocal fails
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
        if not ok:
            print(f"          got {got!r}, wanted {want!r}")
            fails += 1

    c = census(SAMPLE)
    check("both kernels are found", sorted(c), ["kernel_a", "kernel_b"])
    check("the frame size is read from the prologue", c["kernel_a"]["frame"], 0x40)
    check("the hot loop is bounded by the backward branch",
          c["kernel_a"]["ins"], 4)
    check("float work is counted", c["kernel_a"]["fp"], 1)
    check("and a load is not counted as float work", c["kernel_a"]["loads"], 1)
    check("a spill inside the loop is seen", c["kernel_b"]["stack"], 1)
    check("and so is warp divergence",
          (c["kernel_b"]["split"], c["kernel_b"]["join"]), (1, 1))

    # The comparison has to be able to FAIL, and to say which field moved.
    base = json.loads(json.dumps(c))
    base["kernel_a"]["ins"] += 1
    ok, _ = report(c, base, quiet=True, roles={})
    check("a moved number is caught", ok, False)
    ok, _ = report(c, json.loads(json.dumps(c)), quiet=True, roles={})
    check("and an unmoved one is not", ok, True)
    ok, _ = report(c, {"kernel_a": base["kernel_a"]}, quiet=True, roles={})
    check("a kernel the baseline does not know is caught", ok, False)

    # A role naming a kernel that is not in the image has to be a failure, or
    # the table rots quietly. The sample image contains neither of the real
    # ones, so ROLES is temporarily pointed at a name it does have and then at
    # one it does not.
    ok, _ = report(c, json.loads(json.dumps(c)), quiet=True,
                   roles={"kernel_a": (CONTROL, "present")})
    check("a role for a kernel that IS in the image is fine", ok, True)
    ok, _ = report(c, json.loads(json.dumps(c)), quiet=True,
                   roles={"gone_kernel": (CONTROL, "not in this image")})
    check("a role for a kernel that is NOT is caught", ok, False)

    check("a kernel with no declared role ships", role_of("kernel_a")[0], SHIPS)
    check("and a declared one does not", role_of("sgemm_rb")[0], CONTROL)

    _, concerns = report(c, None, quiet=True, roles={})
    check("the spilling kernel is named as a concern",
          [n for n, _, _ in concerns], ["kernel_b"])
    return 1 if fails else 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vxbin", help="the shipped kernel image (its .elf is read)")
    ap.add_argument("--objdump", default="llvm-objdump")
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv[1:])
    if a.self_test:
        return self_test()
    if not a.vxbin:
        print("error: --vxbin <image> is required", file=sys.stderr)
        return 2
    return run(a.vxbin, a.objdump, a.regenerate)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
