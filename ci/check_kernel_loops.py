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
loop; the one picked is the one with the most floating-point operations in it,
tie-broken toward the shortest. That is a heuristic and it is stated as one:
for a kernel whose work is not floating-point -- dnn_causal_mask -- it picks
something arbitrary, and the baseline records whatever it picked rather than
pretending the number means more than it does.

WHAT IS GATED. Exact integers against ci/perf/baselines/kernel_loops.json, no
tolerance, for the same reason the cycle baselines have none: the disassembly of
a deterministic build is deterministic. A moved number means the compiler
produced different code, which is worth a human look whether it got better or
worse.

WHAT IS REPORTED AND NOT GATED. Which kernels still carry stack traffic or warp
divergence in that loop. Those are defects, they are named every run, and they
are not failures -- turning a known defect into a red gate on the day it is
discovered stops the suite being usable, and hiding it stops it being honest.
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
        "stack": sum(1 for t in loop if "(sp)" in t),
        "split": ops.count("vx_split") + ops.count("vx_split_n"),
        "join": ops.count("vx_join"),
    }


def hot_loop(body, lo, hi):
    """The backward-branch loop with the most float work; shortest wins ties."""
    best = None
    for addr, text in body:
        if not re.match(r"(b[a-z]+|j)\s", text):
            continue
        m = re.search(r"(0x[0-9a-f]+)\s*<", text)
        if not m:
            continue
        target = int(m.group(1), 16)
        if not (lo <= target < addr):
            continue
        loop = [t for a, t in body if target <= a <= addr]
        st = loop_stats(loop)
        key = (st["fp"], -(addr - target))
        if best is None or key > best[0]:
            best = (key, st)
    return best[1] if best else None


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
                                    "stack": 0, "split": 0, "join": 0})
        out[name] = entry
    return out


def report(measured, baseline, quiet=False):
    """Print the census, compare, and return (ok, lines_of_concern)."""
    if not quiet:
        print(f"  {'kernel':18s} {'ins':>5s} {'fp':>4s} {'ld':>4s} {'stk':>4s} "
              f"{'spl':>4s} {'join':>5s}   ins/fp")
    concerns = []
    for name in sorted(measured):
        m = measured[name]
        ratio = f"{m['ins'] / m['fp']:6.2f}" if m["fp"] else "     -"
        note = ""
        if m["stack"]:
            note += "  SPILLS"
        if m["split"]:
            note += "  DIVERGES"
        if not quiet:
            print(f"  {name:18s} {m['ins']:5d} {m['fp']:4d} {m['loads']:4d} "
                  f"{m['stack']:4d} {m['split']:4d} {m['join']:5d} {ratio}{note}")
        if note:
            concerns.append((name, m["stack"], m["split"]))

    if baseline is None:
        return True, concerns

    moved, added, gone = [], [], []
    for name, m in sorted(measured.items()):
        b = baseline.get(name)
        if b is None:
            added.append(name)
            continue
        for field in ("size", "frame", "ins", "fp", "loads", "stack", "split",
                      "join"):
            if m[field] != b.get(field):
                moved.append((name, field, b.get(field), m[field]))
    for name in sorted(baseline):
        if name not in measured:
            gone.append(name)

    ok = not (moved or added or gone)
    if not quiet:
        if added:
            print(f"\n  new in this build: {', '.join(added)}")
        if gone:
            print(f"  gone from this build: {', '.join(gone)}")
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
    ok, _ = report(c, base, quiet=True)
    check("a moved number is caught", ok, False)
    ok, _ = report(c, json.loads(json.dumps(c)), quiet=True)
    check("and an unmoved one is not", ok, True)
    ok, _ = report(c, {"kernel_a": base["kernel_a"]}, quiet=True)
    check("a kernel the baseline does not know is caught", ok, False)

    _, concerns = report(c, None, quiet=True)
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
