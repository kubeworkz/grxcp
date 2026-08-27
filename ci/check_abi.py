#!/usr/bin/env python3
"""Every target that touches a shared ABI struct must lay it out identically.

The kernel argument structs are WRITTEN BY THE HOST and READ BY THE DEVICE. A
disagreement about one field's offset is not an error anywhere: the kernel casts
the blob and reads whatever is at that offset, so it computes a wrong answer and
reports success. grxblas_sgemm_args carries an abi_version in its first field
precisely because that failure mode was anticipated -- but a version check
catches a struct that CHANGED, not two compilers that lay the same definition
out differently.

Nothing had ever compared them. The structs use fixed-width types and explicit
padding throughout, which is why nothing has broken, and until the GRX930 the
host was x86-64 in every configuration anyone built. Phase 7 makes the host RV64
for the first time: a new ABI on the writing side of a layout nobody had checked.

HOW, without running anything. Two of the four targets cannot execute here --
the device images are bare-metal RISC-V for a core this container does not have.
So each fact becomes an array whose LENGTH is the number, and the sizes come out
of the object file with `nm --print-size`, whether or not the target could ever
run. Offsets carry a +1 because a zero-length array is not portable C; it is
subtracted back here.

THE PROBE IS GENERATED FROM THE HEADERS, every field of every struct, not a
hand-picked list. The first version WAS a hand-picked list, and ablation caught
it: planting a `size_t` at the front of grxdnn_gelu_args -- a struct the list
covered by SIZE ONLY -- changed nothing the gate looked at. LP64 grew the struct
by 8; ILP32 grew it by 4 and then padded 4 to realign the uint64_t that follows,
so both sizes moved by 8 and stayed equal while every field behind it sat at a
different offset on the device. Sizes alone do not catch a layout change. Offsets
do, and a list of them written by hand is one new field away from not covering
the field that breaks.

TARGETS
  x86_64            the host CI runs on
  riscv64 (host)    the GRX930's own host half, via the cross compiler
  device rv64       what the kernels are actually compiled as
  device rv32       the other device width the toolchain supports

The two device targets are skipped, loudly, when the VOLT toolchain is absent.
They are the important half -- the reading side -- so a run without them says
so rather than reporting a clean comparison of the two hosts alone.

WHERE THE DISCRIMINATING POWER ACTUALLY IS, measured rather than assumed. The
two hosts are both LP64 and agree on every layout these headers can express;
planting a `size_t` in grxblas_sgemm_args left x86_64 and riscv64-host
identical and moved EIGHT facts on device-rv32, including the struct's size
from 120 to 112. So this gate's live edge is host against device-rv32, and the
riscv64 host leg earns its keep by EXECUTING (alignment, argument passing,
struct return) rather than by disagreeing about offsets. Both are worth having;
only one of them is what this file catches.

  ./ci/check_abi.py [--tooldir <dir>] [--verbose]
"""

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
INCLUDES = ["-I" + str(ROOT / "include"), "-I" + str(ROOT / "src/libs")]

# The headers shared across the host/device boundary, and how the probe must
# include them.
# crosses=True means a DEVICE kernel includes this header, so its structs are
# written by the host and read by the device and every target must agree.
# crosses=False means host-only: grxcc writes it and the runtime reads it, both
# compiled for the same host, so a native pointer in there is legitimate.
#
# The distinction is not cosmetic. grx_abi.h's grx_kernel_desc holds a
# `const char*` and a `const grx_kernel_param*`, so it is 40 bytes on LP64 and
# 32 on ILP32 -- and the first version of this gate reported that as nine
# failures. It is not a defect. The device never sees the struct.
ABI_HEADERS = [
    ("grx/grx_abi.h", ROOT / "include/grx/grx_abi.h", False),
    # Read by include/grx/device/grx_tex.h, which every sampling kernel
    # includes -- so the host writes this and the device reads it.
    ("grx/grx_texture.h", ROOT / "include/grx/grx_texture.h", True),
    # grxCycleSlot is written by the DEVICE (include/grx/device/grx_cycles.h
    # includes this) and read by the host summariser, so its layout has to
    # agree across the boundary like any other argument block. It was missing
    # from this list until a field was added to the header beside it.
    ("grx/grx_cycles.h", ROOT / "include/grx/grx_cycles.h", True),
    ("grxblas/sgemm_abi.h", ROOT / "src/libs/grxblas/sgemm_abi.h", True),
    ("grxblas/blas12_abi.h", ROOT / "src/libs/grxblas/blas12_abi.h", True),
    ("grxblas/hgemm_abi.h", ROOT / "src/libs/grxblas/hgemm_abi.h", True),
    ("grxdnn/dnn_abi.h", ROOT / "src/libs/grxdnn/dnn_abi.h", True),
]

# Where device code lives. The crosses= flags above are a claim about these
# files, and claims get checked.
DEVICE_SOURCES = ["src/libs/*/kernels/*.cpp", "src/libs/*/kernels/*.h",
                  "tests/kernels/*/kernel.cpp", "src/libs/kernels_all.cpp"]


def audit_boundary():
    """A host-only header must not be included by device code. Returns problems."""
    host_only = {inc for inc, _, crosses in ABI_HEADERS if not crosses}
    bad = []
    for pattern in DEVICE_SOURCES:
        for src in ROOT.glob(pattern):
            text = src.read_text(errors="replace")
            for inc in host_only:
                stem = inc.split("/")[-1]
                if stem in text:
                    bad.append(f"{src.relative_to(ROOT)} includes {stem}, "
                               "which this gate treats as host-only")
    return bad

COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
STRUCT = re.compile(r"(?:typedef\s+)?struct\s+(\w+)?\s*\{(.*?)\}\s*(\w*)\s*;", re.S)
# A declarator may carry an array bound -- `uint8_t reserved[3]` -- which
# offsetof does not care about but the name must be stripped of.
FIELD = re.compile(
    r"^([A-Za-z_][\w \t\*]*?)\s*"
    r"([A-Za-z_]\w*(?:\s*\[[^\]]*\])*(?:\s*,\s*[A-Za-z_]\w*(?:\s*\[[^\]]*\])*)*)$")
ARRAY = re.compile(r"\s*\[[^\]]*\]")


def structs_in(path):
    """(c_name, [field names]) for every struct in one ABI header.

    Deliberately simple, because these headers are deliberately simple: POD
    only, no functions, no C++ (AGENTS.md section 2). Anything this cannot
    parse is REPORTED rather than skipped -- a struct that silently drops out
    of the probe is a struct nobody is checking.
    """
    text = COMMENT.sub(" ", path.read_text())
    out, unparsed = [], []
    for m in STRUCT.finditer(text):
        tag, body, typedef_name = m.group(1), m.group(2), m.group(3)
        name = f"struct {tag}" if tag else typedef_name
        if not tag and not typedef_name:
            unparsed.append("anonymous struct")
            continue
        if tag and typedef_name:
            name = typedef_name          # typedef struct X {...} Y;
        fields = []
        for decl in body.split(";"):
            decl = " ".join(decl.split())
            if not decl:
                continue
            fm = FIELD.match(decl)
            if not fm:
                unparsed.append(f"{name}: {decl}")
                continue
            for n in fm.group(2).split(","):
                fields.append(ARRAY.sub("", n).strip().lstrip("*"))
        if fields:
            out.append((name, fields))
    return out, unparsed


def generate_probe(dest, want_crosses):
    """Write the probe, and return (struct count, field count, complaints)."""
    lines = ["#include <stddef.h>", "#include <stdint.h>", ""]
    for inc, _, crosses in ABI_HEADERS:
        if crosses == want_crosses:
            lines.append(f'#include "{inc}"')
    lines.append("")
    nstruct = nfield = 0
    complaints = []
    seen = set()
    for _, path, crosses in ABI_HEADERS:
        if crosses != want_crosses:
            continue
        found, bad = structs_in(path)
        complaints += [f"{path.name}: {b}" for b in bad]
        for name, fields in found:
            tag = name.replace("struct ", "").strip()
            if tag in seen:
                continue
            seen.add(tag)
            nstruct += 1
            lines.append(f"char grxabi_size__{tag}[sizeof({name})];")
            lines.append(f"char grxabi_algn__{tag}[_Alignof({name})];")
            for f in fields:
                nfield += 1
                lines.append(
                    f"char grxabi_off__{tag}__{f}[offsetof({name}, {f}) + 1];")
            lines.append("")
    dest.write_text("\n".join(lines) + "\n")
    return nstruct, nfield, complaints


def host_targets():
    out = [("x86_64", ["gcc", "-std=c11"])]
    cross = shutil.which("riscv64-linux-gnu-gcc")
    if cross:
        out.append(("riscv64-host", [cross, "-std=c11"]))
    return out


def device_targets(tooldir):
    clang = pathlib.Path(tooldir) / "llvm-vortex/bin/clang++"
    if not clang.exists():
        return []
    return [
        ("device-rv64", [str(clang), "--target=riscv64-unknown-elf",
                         "-march=rv64imafd", "-mabi=lp64d", "-std=c11"]),
        ("device-rv32", [str(clang), "--target=riscv32-unknown-elf",
                         "-march=rv32imaf", "-mabi=ilp32f", "-std=c11"]),
    ]


def layout_of(name, argv, probe, tmp):
    """Compile the probe for one target and read the numbers back out."""
    obj = tmp / (name + ".o")
    cmd = argv + ["-c", "-x", "c", str(probe)] + INCLUDES + ["-o", str(obj)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAIL  {name}: the ABI probe does not compile")
        for line in (r.stderr or "").splitlines()[:6]:
            print("        " + line)
        return None

    nm = shutil.which("nm") or "nm"
    r = subprocess.run([nm, "--print-size", "--defined-only", str(obj)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAIL  {name}: nm could not read {obj.name}")
        return None

    facts = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        # address size type name -- a symbol with no size prints three fields.
        if len(parts) != 4:
            continue
        _, size, _, sym = parts
        if not sym.startswith("grxabi_"):
            continue
        value = int(size, 16)
        if sym.startswith("grxabi_off__"):
            value -= 1          # the +1 the probe added to keep the array legal
        facts[sym] = value
    return facts


def main(argv):
    verbose = "--verbose" in argv
    tooldir = "/home/claude/tools"
    for i, a in enumerate(argv):
        if a == "--tooldir" and i + 1 < len(argv):
            tooldir = argv[i + 1]

    hosts = host_targets()
    devices = device_targets(tooldir)

    problems = audit_boundary()
    if problems:
        print(f"  FAIL  {len(problems)} host-only ABI headers reached device code:")
        for b in problems:
            print("        " + b)
        print("        Either that header now crosses the boundary and its")
        print("        crosses= flag is wrong, or the include is a mistake.")
        return 1

    # Two passes. The cross-boundary headers are compared against every target,
    # because the host writes those structs and the device reads them. The
    # host-only headers are compared against the hosts alone -- they hold native
    # pointers by design, and asking rv32 about them would report a fact that is
    # true and means nothing.
    passes = [("host<->device", True, hosts + devices),
              ("host only", False, hosts)]

    nstruct = nfield = 0
    failed = False
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        for label, crosses, targets in passes:
            probe = tmp / f"abi_probe_{int(crosses)}.c"
            ns, nf, complaints = generate_probe(probe, crosses)
            if verbose:
                print(probe.read_text())
            if complaints:
                # Not skipped quietly. A declaration this cannot read is a
                # declaration nobody is checking, and the whole point is that
                # the coverage is not a hand-kept list.
                print(f"  FAIL  {len(complaints)} declarations could not be parsed:")
                for c in complaints[:8]:
                    print("        " + c)
                print("        Either simplify the declaration or teach the parser.")
                return 1
            if ns == 0:
                print(f"  FAIL  no structs found for the {label} pass")
                return 1
            nstruct += ns
            nfield += nf

            layouts = {}
            for name, argvv in targets:
                facts = layout_of(name, argvv, probe, tmp)
                if facts is None:
                    return 1
                layouts[name] = facts

            names = list(layouts)
            keys = sorted(layouts[names[0]])
            bad = []
            for key in keys:
                values = {n: layouts[n].get(key) for n in names}
                if len(set(values.values())) != 1:
                    bad.append((key, values))

            print(f"  {ns} structs, {nf} fields, {len(keys)} facts  [{label}]  "
                  + ", ".join(names))
            if verbose:
                for key in keys:
                    row = "  ".join(f"{n}={layouts[n].get(key)}" for n in names)
                    print(f"        {key[len('grxabi_'):]}: {row}")
            if bad:
                failed = True
                print(f"  FAIL  {len(bad)} layout facts differ [{label}]:")
                for key, values in bad:
                    row = "  ".join(f"{n}={v}" for n, v in values.items())
                    print(f"        {key[len('grxabi_'):]}: {row}")

    if not devices:
        print("  note  the DEVICE targets were not checked: no VOLT toolchain "
              f"under {tooldir}.")
        print("        The device is the side that reads these structs, so this "
              "run compared")
        print("        the writing side against itself.")

    if failed:
        print("        A host and a device that disagree here do not fail. The")
        print("        kernel reads the blob at the offset it was compiled for")
        print("        and returns a wrong answer.")
        return 1

    print("  ok    every shared ABI struct has the same layout on every target")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
