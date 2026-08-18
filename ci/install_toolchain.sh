#!/usr/bin/env bash
# Install the device toolchain needed to compile a kernel into a .vxbin.
#
# Four pieces, all prebuilt by the GRX-G100 project:
#   riscv64-gnu-toolchain   assembler, linker, binutils
#   llvm-vortex (VOLT)      the SIMT compiler; this is what understands
#                           __global__, the thread mask and the Vortex ISA
#                           extensions
#   libc64                  the C library kernels link against
#   libcrt64                compiler-rt builtins for baremetal
#
# About 580 MB downloaded, ~1.5 GB installed. Only needed for kernels: the
# runtime, the SimX backend and everything ci/run_real.sh does short of
# launching all build without it (see ci/build_sysroot.sh).
#
#   ./ci/install_toolchain.sh --tooldir $HOME/tools [--grxgpu <path>]
#
# ONE GOTCHA WORTH THE COMMENT. grxgpu's own installer fetches from
# github.com/<org>/<repo>/raw/<rev>/..., which some corporate proxies and
# sandboxes reject with 403 while allowing raw.githubusercontent.com. This
# script uses raw.githubusercontent.com directly for that reason. If both are
# blocked, download the tarballs by hand and point --tooldir at the result.

set -euo pipefail

TOOLDIR="${TOOLDIR:-$HOME/tools}"
GRXGPU=""
REV=""
OSVERSION="ubuntu/focal"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tooldir)   TOOLDIR="$2"; shift 2 ;;
    --grxgpu)    GRXGPU="$2"; shift 2 ;;
    --rev)       REV="$2"; shift 2 ;;
    --osversion) OSVERSION="$2"; shift 2 ;;
    -h|--help)   sed -n '2,26p' "$0" | sed 's|^# \{0,1\}||'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Prefer the revision the checkout pins over anything hardcoded here.
if [[ -z "$REV" && -n "$GRXGPU" && -f "$GRXGPU/VERSION" ]]; then
  REV="$(grep -E '^TOOLCHAIN_REV=' "$GRXGPU/VERSION" | cut -d= -f2 | tr -d '\r')"
fi
REV="${REV:-v3.0}"

BASE="https://raw.githubusercontent.com/vortexgpgpu/vortex-toolchain-prebuilt/$REV"
mkdir -p "$TOOLDIR"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

# The big toolchains are split into 50 MB parts because of GitHub's file size
# limit; they are concatenated back before extraction.
fetch_split() {
  local name="$1" path="$2" parts="$3"
  if [[ -d "$TOOLDIR/$name" ]]; then
    echo "==> $name already installed, skipping"
    return
  fi
  echo "==> downloading $name ($parts parts)"
  ( cd "$SCRATCH"
    for x in $(eval echo "{a..$parts}"); do
      curl -sfL -O "$BASE/$path/$name.tar.bz2.parta$x" \
        || { echo "failed: $name part $x" >&2; exit 1; }
    done
    cat "$name".tar.bz2.parta* > "$name.tar.bz2"
    tar -xf "$name.tar.bz2" -C "$TOOLDIR"
    rm -f "$name".tar.bz2* )
}

fetch_single() {
  local name="$1"
  if [[ -d "$TOOLDIR/$name" ]]; then
    echo "==> $name already installed, skipping"
    return
  fi
  echo "==> downloading $name"
  ( cd "$SCRATCH"
    curl -sfL -O "$BASE/$name/$name.tar.bz2"
    tar -xf "$name.tar.bz2" -C "$TOOLDIR"
    rm -f "$name.tar.bz2" )
}

fetch_split riscv64-gnu-toolchain "riscv64-gnu-toolchain/$OSVERSION" j
fetch_split llvm-vortex           "llvm-vortex/$OSVERSION"           c
fetch_single libc64
fetch_single libcrt64

echo
echo "Toolchain installed under $TOOLDIR:"
for d in riscv64-gnu-toolchain llvm-vortex libc64 libcrt64; do
  printf "  %-24s %s\n" "$d" "$([[ -d "$TOOLDIR/$d" ]] && echo ok || echo MISSING)"
done
echo
echo "Reconfigure the GRX-G100 build with --tooldir=$TOOLDIR so the kernel"
echo "library builds, then use ci/build_kernel.sh."
