#!/usr/bin/env bash
# Build and smoke-test GRXCP against the mock driver -- no Vortex sysroot, no
# simulator, no FPGA. This is what CI runs on every commit so a broken runtime
# is caught in seconds rather than in a simulator run.
#
# It proves the code compiles, links, and reports a self-consistent device
# record. It proves NOTHING about real hardware: the Phase 0 exit gate is
# grx-smi on simx and rtlsim, which needs the real sysroot (see ci/README.md).
#
#   ./ci/build_mock.sh [--vortex-include <dir>]
#
# The mock still needs the real vortex2.h so it implements the actual driver
# ABI rather than a paraphrase of it. Point --vortex-include at the GRX-G100
# installed sysroot's include directory, or set VORTEX_PATH.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build-mock"

VORTEX_INCLUDE="${VORTEX_PATH:+$VORTEX_PATH/include}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --vortex-include) VORTEX_INCLUDE="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${VORTEX_INCLUDE:-}" || ! -f "$VORTEX_INCLUDE/vortex2.h" ]]; then
  echo "error: vortex2.h not found." >&2
  echo "  Set VORTEX_PATH to the installed GRX-G100 sysroot, or pass" >&2
  echo "  --vortex-include <dir containing vortex2.h>." >&2
  exit 1
fi

CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -Wall -Wextra -O1 -g -I$ROOT/include -I$VORTEX_INCLUDE"

mkdir -p "$BUILD"

echo "==> compiling runtime"
RT_OBJS=()
for src in "$ROOT"/src/runtime/*.cpp; do
  obj="$BUILD/$(basename "${src%.cpp}").o"
  $CXX $CXXFLAGS -c "$src" -o "$obj"
  RT_OBJS+=("$obj")
done

echo "==> compiling mock driver (test fixture, never installed)"
$CXX $CXXFLAGS -c "$ROOT/tests/mock/vortex_mock.cpp" -o "$BUILD/vortex_mock.o"
RT_OBJS+=("$BUILD/vortex_mock.o")

echo "==> linking tools"
for tool in grx-smi grx-conform; do
  $CXX $CXXFLAGS -c "$ROOT/tools/$tool/main.cpp" -o "$BUILD/$tool.o"
  $CXX "${RT_OBJS[@]}" "$BUILD/$tool.o" -o "$BUILD/$tool"
done
# grxify links nothing from the runtime on purpose: a source translator must
# work on a machine with no device stack installed.
$CXX $CXXFLAGS -c "$ROOT/tools/grxify/main.cpp" -o "$BUILD/grxify.o"
$CXX "$BUILD/grxify.o" -o "$BUILD/grxify"

echo "==> linking unit tests"
TESTS=()
for src in "$ROOT"/tests/unit/*.cpp; do
  name="$(basename "${src%.cpp}")"
  $CXX $CXXFLAGS -I"$ROOT/tests/unit" -c "$src" -o "$BUILD/$name.o"
  $CXX "${RT_OBJS[@]}" "$BUILD/$name.o" -o "$BUILD/$name"
  TESTS+=("$name")
done

echo
echo "==> unit tests (default config)"
for t in "${TESTS[@]}"; do
  printf -- "--- %s\n" "$t"
  "$BUILD/$t"
done

echo
echo "==> unit tests (FPGA backend: managed memory must be gated off)"
for t in "${TESTS[@]}"; do
  VORTEX_DRIVER=xrt "$BUILD/$t" > /dev/null || { echo "FAILED on xrt: $t"; exit 1; }
done
echo "all tests pass on the FPGA backend selection too"

echo
echo "==> API table / compat header drift check"
python3 "$ROOT/ci/check_compat_table.py"

echo
echo "==> conformance report"
"$BUILD/grx-conform"

echo
echo "==> end-to-end: translate a CUDA source, compile it, link it, run it"
"$BUILD/grxify" --check "$ROOT/tests/conformance/portable_port.cu"
"$BUILD/grxify" "$ROOT/tests/conformance/portable_port.cu" \
  -o "$BUILD/portable_port.grx.cpp" 2>/dev/null
$CXX $CXXFLAGS -c "$BUILD/portable_port.grx.cpp" -o "$BUILD/portable_port.o"
$CXX "${RT_OBJS[@]}" "$BUILD/portable_port.o" -o "$BUILD/portable_port"
"$BUILD/portable_port"

echo
echo "==> grxify reports the unmappable calls in an awkward port"
if "$BUILD/grxify" --check "$ROOT/tests/conformance/sample_port.cu" 2>&1; then
  echo "FAILED: grxify should have reported unmappable calls"; exit 1
fi
echo "(the diagnostics above are the expected result)"

echo
echo "==> grx-smi (default config)"
"$BUILD/grx-smi"

echo "==> grx-smi (flagship G100 preset)"
VORTEX_DRIVER=rtlsim \
GRXMOCK_NUM_THREADS=32 GRXMOCK_NUM_WARPS=64 \
GRXMOCK_NUM_CORES=16 GRXMOCK_NUM_CLUSTERS=8 \
GRXMOCK_SOCKET_SIZE=4 GRXMOCK_ISSUE_WIDTH=4 \
GRXMOCK_LOCAL_MEM=262144 GRXMOCK_GLOBAL_MEM=137438953472 \
GRXMOCK_MEM_BANKS=8 GRXMOCK_MEM_BANK_SIZE=17179869184 \
GRXMOCK_CLOCK_MHZ=2000 GRXMOCK_PEAK_BW_MBS=6400000 \
  "$BUILD/grx-smi"

echo "all mock checks passed"
