#!/usr/bin/env python3
"""Keep the CUDA API table and the compatibility header from drifting apart.

Three files have to agree or a port breaks in a confusing way:

  tools/common/cuda_api_table.inc   what GRXCP claims to support
  include/grx/grx_cuda_compat.h     what a translated source can actually name
  include/grx/grx_runtime.h         what the runtime actually declares

An entry marked supported in the table but missing from the compat header is
the worst of the three failure modes: grx-conform reports it as covered, grxify
translates a program that uses it, and the program then fails to compile. This
check exists because that exact drift already happened once, for
cudaPointerAttributes, and was caught by an end-to-end test rather than by
review.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TABLE = ROOT / "tools/common/cuda_api_table.inc"
COMPAT = ROOT / "include/grx/grx_cuda_compat.h"
HEADERS = [ROOT / "include/grx/grx_runtime.h", ROOT / "include/grx/grx_types.h"]

SUPPORTED = ("MAPPED", "PARTIAL", "UNSUPPORTED")


def main() -> int:
    # Skip the macro definitions and the comments that document them: both
    # `#define GRX_CUDA_API(cuda, grx, ...)` and `// Columns: GRX_CUDA_API(...)`
    # look exactly like entries to a naive regex.
    table = "\n".join(
        line for line in TABLE.read_text().splitlines()
        if not line.lstrip().startswith(("#define", "//"))
    )
    compat = COMPAT.read_text()
    headers = "\n".join(h.read_text() for h in HEADERS)

    compat_defines = set(re.findall(r"#define\s+(cuda\w+)", compat))
    runtime_decls = set(re.findall(r"\b(grx\w+)\s*\(", headers))

    problems = []

    entries = [
        (m.group(1), m.group(2), m.group(3), "function")
        for m in re.finditer(r"GRX_CUDA_API\(\s*(\w+)\s*,\s*(\w*)\s*,\s*(\w+)", table)
    ] + [
        (m.group(1), m.group(2), m.group(3), "type")
        for m in re.finditer(r"GRX_CUDA_TYPE\(\s*(\w+)\s*,\s*(\w*)\s*,\s*(\w+)", table)
    ]

    for cuda, grx, status, kind in entries:
        if status == "ABSENT":
            if grx:
                problems.append(f"{cuda}: marked ABSENT but names a GRX symbol '{grx}'")
            if cuda in compat_defines:
                problems.append(
                    f"{cuda}: marked ABSENT but the compat header defines it, so a "
                    f"port using it compiles and should not"
                )
            continue

        if not grx:
            problems.append(f"{cuda}: marked {status} but names no GRX symbol")
            continue

        # cu*-prefixed driver API entries are reachable directly, not through
        # the compat header, so they are exempt from the #define requirement.
        if cuda.startswith("cuda") and cuda not in compat_defines:
            problems.append(
                f"{cuda}: marked {status} but grx_cuda_compat.h does not define it"
            )

        if kind == "function" and grx not in runtime_decls:
            problems.append(
                f"{grx}: mapped from {cuda} but no public header declares it"
            )

    if problems:
        print("compat table check FAILED:")
        for p in problems:
            print(f"  {p}")
        return 1

    print(f"compat table check ok ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
