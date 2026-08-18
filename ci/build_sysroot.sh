#!/usr/bin/env bash
# Build a GRX-G100 sysroot from a grxgpu checkout, far enough for GRXCP.
#
# GRXCP links against the INSTALLED sysroot and never against a source tree
# (AGENTS.md section 2). This script produces that sysroot. It builds only what
# the runtime needs -- the SimX backend and the driver -- which deliberately
# avoids the RISC-V toolchain, LLVM and Verilator. Building a kernel needs
# those; enumerating a device, allocating memory and moving data does not.
#
#   ./ci/build_sysroot.sh --grxgpu <path> [--xlen 64] [--jobs N]
#                         [--tooldir <path>]
#                         [--configs "-DVX_CFG_EXT_TCU_ENABLE ..."]
#
# On success it prints the VORTEX_PATH to export. Roughly ten minutes cold,
# almost all of it ramulator.
#
# --configs selects the hardware configuration, exactly as grxgpu's own CONFIGS
# variable does: the default VX_config.toml is the small FPGA baseline with the
# tensor unit and the DMA engine OFF. A device with tensor cores needs
#
#   --configs "-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE"
#
# and the string is RECORDED in the installed sysroot so ci/build_kernel.sh
# compiles device code for the same machine the runtime was built for. Without
# that record the two disagree silently: the runtime reports a tensor unit, the
# kernel is compiled as though it has none, and a tensor test passes having
# tested nothing. See ci/README.md, "configuration provenance".
#
# THREE THINGS THAT WILL BITE YOU, all found the hard way:
#
#   1. CRLF line endings. A checkout made on Windows leaves \r on every line,
#      and grxgpu's own configure script then dies with
#      "$'\r': command not found". This script detects it and stops rather than
#      rewriting your working copy behind your back; pass --fix-line-endings to
#      normalize a COPY.
#
#   2. Submodules. third_party/{softfloat,ramulator} are empty in a
#      non-recursive clone. This script fetches them at the exact commits the
#      superproject pins.
#
#   3. ramulator2 HEAD does not work. Upstream reorganized its headers
#      (src/base -> src/ramulator/base) and replaced spdlog with fmt, so a
#      fresh clone of main fails to compile against grxgpu's include paths. The
#      pinned commit is not optional.

set -euo pipefail

GRXGPU=""
XLEN=64
JOBS="$(nproc 2>/dev/null || echo 4)"
FIX_LINE_ENDINGS=0
WORKDIR=""
CONFIGS="${CONFIGS:-}"
TOOLDIR="${TOOLDIR:-$HOME/tools}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --grxgpu)            GRXGPU="$2"; shift 2 ;;
    --xlen)              XLEN="$2"; shift 2 ;;
    --jobs)              JOBS="$2"; shift 2 ;;
    --workdir)           WORKDIR="$2"; shift 2 ;;
    --tooldir)           TOOLDIR="$2"; shift 2 ;;
    --configs)           CONFIGS="$2"; shift 2 ;;
    --fix-line-endings)  FIX_LINE_ENDINGS=1; shift ;;
    -h|--help)
      sed -n '2,43p' "$0" | sed 's|^# \{0,1\}||'
      exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$GRXGPU" || ! -f "$GRXGPU/VX_config.toml" ]]; then
  echo "error: --grxgpu must point at a grxgpu checkout (containing VX_config.toml)" >&2
  exit 2
fi

GRXGPU="$(cd "$GRXGPU" && pwd)"
WORKDIR="${WORKDIR:-$GRXGPU}"

for tool in cmake g++ make python3 git; do
  command -v "$tool" >/dev/null || { echo "error: $tool not found" >&2; exit 1; }
done

# --- 1. line endings -------------------------------------------------------
if grep -qU $'\r' "$GRXGPU/configure" 2>/dev/null; then
  if [[ "$FIX_LINE_ENDINGS" == "1" ]]; then
    echo "==> normalizing CRLF line endings in $WORKDIR"
    if [[ "$WORKDIR" != "$GRXGPU" ]]; then
      mkdir -p "$WORKDIR"
      cp -r "$GRXGPU"/. "$WORKDIR"/
    fi
    find "$WORKDIR" -path "$WORKDIR/.git" -prune -o -type f \
      \( -name '*.sh' -o -name '*.in' -o -name 'Makefile*' -o -name '*.mk' \
      -o -name '*.py' -o -name '*.toml' -o -name 'configure' -o -name '*.h' \
      -o -name '*.cpp' -o -name '*.hpp' -o -name '*.c' -o -name '*.S' \
      -o -name '*.ld' -o -name '*.sv' -o -name '*.vh' -o -name '*.yaml' \) \
      -print0 | xargs -0 sed -i 's/\r$//'
    sed -i 's/\r$//' "$WORKDIR/configure"
  else
    cat >&2 <<'MSG'
error: this checkout has CRLF line endings, and grxgpu's configure script will
       fail with "$'\r': command not found".

       Fix it in the checkout (recommended, it is a real defect there):
           git config core.autocrlf input
           git rm --cached -r . && git reset --hard

       Or re-run this script with --fix-line-endings --workdir <scratch dir>
       to normalize a copy and leave your working tree alone.
MSG
    exit 1
  fi
fi

cd "$WORKDIR"

# --- 2. pinned dependencies ------------------------------------------------
# Fetch at the superproject's recorded commit rather than at a branch tip: for
# ramulator that is a correctness requirement, not tidiness (see the header).
fetch_pinned() {
  local path="$1" url="$2" sha="$3"
  if [[ -n "$(ls -A "$path" 2>/dev/null)" ]]; then
    echo "==> $path already populated, leaving it alone"
    return
  fi
  echo "==> fetching $path at ${sha:0:12}"
  rm -rf "$path"; mkdir -p "$path"
  ( cd "$path"
    git init -q .
    git remote add origin "$url"
    git fetch -q --depth=1 origin "$sha"
    git checkout -q FETCH_HEAD )
}

# The `|| true` is load-bearing. In a checkout with no git metadata -- a
# tarball, an export, a copy without .git -- ls-tree exits 128, and with
# `set -o pipefail` that aborts the whole script at the assignment below,
# before the documented default SHAs can be applied. Empty here means "fall
# back to the pins recorded in this file", which is what the next lines do.
pinned_sha() {
  git -C "$GRXGPU" ls-tree HEAD "third_party/$1" 2>/dev/null | awk '{print $3}' || true
}

SOFTFLOAT_SHA="$(pinned_sha softfloat)"
RAMULATOR_SHA="$(pinned_sha ramulator)"
: "${SOFTFLOAT_SHA:=b51ef8f3201669b2288104c28546fc72532a1ea4}"
: "${RAMULATOR_SHA:=e62c84a6f0e06566ba6e182d308434b4532068a5}"

fetch_pinned third_party/softfloat \
  https://github.com/ucb-bar/berkeley-softfloat-3.git "$SOFTFLOAT_SHA"
fetch_pinned third_party/ramulator \
  https://github.com/CMU-SAFARI/ramulator2.git "$RAMULATOR_SHA"

echo "==> building third-party dependencies (ramulator takes a while)"
make -C third_party softfloat ramulator -j"$JOBS"

# --- 3. configure and build ------------------------------------------------
[[ -f VERSION ]] || echo "VERSION=3.0" > VERSION

mkdir -p build && cd build
echo "==> configure --xlen=$XLEN"
bash ../configure --xlen="$XLEN" --tooldir="$TOOLDIR"

echo "==> building the driver and the SimX backend"
echo "    CONFIGS=${CONFIGS:-<VX_config.toml defaults>}"
# Only simx: rtlsim needs Verilator and the hw/dpi sources, and nothing in
# GRXCP's current gates requires it.
export CONFIGS
make -C sw/runtime/simx -j"$JOBS"
make -C sw/runtime/stub  -j"$JOBS" 2>/dev/null || true

# The CTA runtime (libvortex2.a) is device code, so it needs the RISC-V
# toolchain the rest of this script deliberately avoids. Build it here when the
# toolchain is present -- with the SAME CONFIGS, which is the whole point of
# this script taking them -- because ci/build_kernel.sh links every kernel
# against it, and a CTA runtime built for a different machine than the kernel
# is the same silent disagreement in a second place.
if [[ -d "$TOOLDIR/llvm-vortex" ]]; then
  echo "==> building the device-side CTA runtime (libvortex2.a)"
  make -C sw/kernel -j"$JOBS"
else
  echo "==> skipping the device-side CTA runtime: no toolchain in $TOOLDIR"
  echo "    ci/build_kernel.sh needs it. Install with ci/install_toolchain.sh,"
  echo "    then re-run this script."
fi

echo "==> installing the sysroot"
make install

VORTEX_PATH="$(pwd)/install"

# --- 4. configuration provenance -------------------------------------------
# The installed sysroot carries no record of the configuration it was built
# with -- no generated VX_config.h, and vortex-kernel.pc's Cflags do not carry
# the defines. So a consumer compiling device code has no way to ask the
# sysroot what machine it is for, and defaults to the repo's baseline toml,
# which is not necessarily what is installed here.
#
# Until grxgpu records this itself (it should, and the file below is shaped so
# that switching to an upstream mechanism is a one-line change), GRXCP writes
# the string it used into a clearly GRXCP-owned path. ci/build_kernel.sh reads
# it. This is provenance, not configuration: nothing reads it to DECIDE
# anything, only to stay consistent with a decision already made here.
mkdir -p "$VORTEX_PATH/share/grxcp"
printf '%s\n' "$CONFIGS" > "$VORTEX_PATH/share/grxcp/device_configs"
cat <<EOF

Sysroot ready.

  export VORTEX_PATH=$VORTEX_PATH
  export PKG_CONFIG_PATH=\$VORTEX_PATH/lib/pkgconfig:\$PKG_CONFIG_PATH
  export LD_LIBRARY_PATH=\$VORTEX_PATH/runtime/lib:\$LD_LIBRARY_PATH
  export VORTEX_DRIVER=simx

Then run GRXCP's tier-2 gates:

  ./ci/run_real.sh
EOF
