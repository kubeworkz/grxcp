#!/usr/bin/env bash
# Tier 2: build GRXCP against a real GRX-G100 sysroot and run the gates that
# the mock driver cannot answer.
#
# What this proves that ci/build_mock.sh does not: the runtime links against
# the actual driver with no source changes, the device it reports is the one
# the simulator actually models, and data moves through the real command
# processor rather than through a std::memcpy in a test fixture.
#
# What it still does not prove: kernel execution. Launching a kernel needs a
# device binary, which needs the RISC-V toolchain and VOLT. That is the last
# piece of the phase 1 exit gate.
#
#   export VORTEX_PATH=<sysroot>          # see ci/build_sysroot.sh
#   ./ci/run_real.sh [--driver simx]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build-real"
DRIVER="${VORTEX_DRIVER:-simx}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --driver) DRIVER="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${VORTEX_PATH:-}" || ! -f "$VORTEX_PATH/lib/pkgconfig/vortex-runtime.pc" ]]; then
  echo "error: VORTEX_PATH must point at an installed GRX-G100 sysroot." >&2
  echo "       Build one with: ./ci/build_sysroot.sh --grxgpu <path>" >&2
  exit 1
fi

export PKG_CONFIG_PATH="$VORTEX_PATH/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$VORTEX_PATH/runtime/lib:${LD_LIBRARY_PATH:-}"
export VORTEX_DRIVER="$DRIVER"

CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -Wall -Wextra -O1 -g -I$ROOT/include $(pkg-config --cflags vortex-runtime)"
LIBS="$(pkg-config --libs vortex-runtime)"

mkdir -p "$BUILD"

echo "==> compiling the runtime against the real driver"
OBJS=()
for src in "$ROOT"/src/runtime/*.cpp; do
  obj="$BUILD/$(basename "${src%.cpp}").o"
  $CXX $CXXFLAGS -c "$src" -o "$obj"
  OBJS+=("$obj")
done

echo "==> building tools"
for tool in grx-smi grx-conform; do
  $CXX $CXXFLAGS -c "$ROOT/tools/$tool/main.cpp" -o "$BUILD/$tool.o"
  $CXX "${OBJS[@]}" "$BUILD/$tool.o" $LIBS -o "$BUILD/$tool"
done
$CXX $CXXFLAGS -c "$ROOT/tools/grxify/main.cpp" -o "$BUILD/grxify.o"
$CXX "$BUILD/grxify.o" -o "$BUILD/grxify"

echo
echo "==> PHASE 0 GATE: grx-smi on a real $DRIVER device"
"$BUILD/grx-smi"

echo "==> unit tests on a real $DRIVER device"
# test_launch is excluded on purpose: it builds modules in the mock driver's
# own image format, which the real loader has no reason to accept. Testing
# launch for real needs a .vxbin, which needs the device toolchain.
for t in test_device_props test_memory test_stream_event; do
  $CXX $CXXFLAGS -I"$ROOT/tests/unit" -c "$ROOT/tests/unit/$t.cpp" -o "$BUILD/$t.o"
  $CXX "${OBJS[@]}" "$BUILD/$t.o" $LIBS -o "$BUILD/$t"
  printf -- "--- %s\n" "$t"
  "$BUILD/$t" | tail -3
done

echo
echo "==> end-to-end: a CUDA source, translated, compiled, linked, run on $DRIVER"
"$BUILD/grxify" "$ROOT/tests/conformance/portable_port.cu" \
  -o "$BUILD/portable_port.grx.cpp" 2>/dev/null
$CXX $CXXFLAGS -c "$BUILD/portable_port.grx.cpp" -o "$BUILD/portable_port.o"
$CXX "${OBJS[@]}" "$BUILD/portable_port.o" $LIBS -o "$BUILD/portable_port"
"$BUILD/portable_port"

echo
echo "==> conformance report against the real device"
"$BUILD/grx-conform" | tail -14

echo
echo "tier 2 passed on $DRIVER"
