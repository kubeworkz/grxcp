#!/usr/bin/env bash
# Tier 2: build GRXCP against a real GRX-G100 sysroot and run the gates that
# the mock driver cannot answer.
#
# What this proves that ci/build_mock.sh does not: the runtime links against
# the actual driver with no source changes, the device it reports is the one
# the simulator actually models, and data moves through the real command
# processor rather than through a std::memcpy in a test fixture.
#
# It also runs real kernels when the device toolchain is available: vecadd,
# which is the phase 1 exit gate, and the grxBLAS sgemm gate. Both compute on
# the device and are checked on the host.
#
#   export VORTEX_PATH=<sysroot>          # see ci/build_sysroot.sh
#   ./ci/run_real.sh [--driver simx] [--grxgpu <path>] [--tooldir <path>]
#
# Without --grxgpu and a toolchain the kernel gate is skipped and said so
# explicitly, rather than quietly reporting a pass over work that never ran.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build-real"
DRIVER="${VORTEX_DRIVER:-simx}"
GRXGPU="${GRXGPU:-}"
TOOLDIR="${TOOLDIR:-$HOME/tools}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --driver)  DRIVER="$2"; shift 2 ;;
    --grxgpu)  GRXGPU="$2"; shift 2 ;;
    --tooldir) TOOLDIR="$2"; shift 2 ;;
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
echo "==> PHASE 1 GATE: compile a kernel, run it, check the arithmetic"
if [[ -n "$GRXGPU" && -d "$TOOLDIR/llvm-vortex" ]]; then
  "$ROOT/ci/build_kernel.sh" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
    "$ROOT/tests/kernels/vecadd/kernel.cpp" -o "$BUILD/vecadd.vxbin"
  $CXX $CXXFLAGS -I"$ROOT/tests/kernels/vecadd" \
    -c "$ROOT/tests/kernels/vecadd/main.cpp" -o "$BUILD/vecadd_main.o"
  $CXX "${OBJS[@]}" "$BUILD/vecadd_main.o" $LIBS -o "$BUILD/vecadd"
  # Sizes chosen to cover the partial-warp path: 70 and 255 are not multiples
  # of the warp width, and 1 leaves all but one lane masked off.
  for n in 64 70 255 1; do
    "$BUILD/vecadd" "$BUILD/vecadd.vxbin" "$n" | tail -1
  done
else
  echo "SKIPPED: needs --grxgpu <path> and a device toolchain in $TOOLDIR."
  echo "         Install it with ci/install_toolchain.sh, then rebuild the"
  echo "         GRX-G100 kernel library: make -C <grxgpu>/build/sw/kernel"
fi

echo
echo "==> TENSOR GATE: one WMMA tile, checked exactly against a CPU reference"
# Whether to run this is the runtime's call, not a guess from the config file:
# grx-smi reports what the device says it has, and grx_wmma.h refuses to compile
# for a device without a tensor unit, so attempting the build on one would be a
# hard error rather than a skip.
if ! "$BUILD/grx-smi" 2>/dev/null | grep -q 'capabilities.*tensor'; then
  echo "SKIPPED: this device reports no tensor unit."
  echo "         Rebuild the sysroot with ci/build_sysroot.sh --configs"
  echo "         \"-DVX_CFG_EXT_TCU_ENABLE\" to exercise it."
elif [[ -z "$GRXGPU" || ! -d "$TOOLDIR/llvm-vortex" ]]; then
  echo "SKIPPED: needs --grxgpu <path> and a device toolchain in $TOOLDIR."
else
  "$ROOT/ci/build_kernel.sh" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
    "$ROOT/tests/kernels/wmma/kernel.cpp" -o "$BUILD/wmma.vxbin" >/dev/null
  $CXX $CXXFLAGS -I"$ROOT/tests/kernels/wmma" \
    -c "$ROOT/tests/kernels/wmma/main.cpp" -o "$BUILD/wmma_main.o"
  $CXX "${OBJS[@]}" "$BUILD/wmma_main.o" $LIBS -o "$BUILD/wmma_gate"
  if ! "$BUILD/wmma_gate" "$BUILD/wmma.vxbin"; then
    rc=$?
    [[ $rc -eq 77 ]] || exit $rc
  fi
fi

echo
echo "==> DXA GATE: an asynchronous tile copy, checked element for element"
if ! "$BUILD/grx-smi" 2>/dev/null | grep -q 'capabilities.*async-copy'; then
  echo "SKIPPED: this device reports no DMA engine."
  echo "         Rebuild the sysroot with ci/build_sysroot.sh --configs"
  echo "         \"-DVX_CFG_EXT_DXA_ENABLE\" to exercise it."
elif [[ -z "$GRXGPU" || ! -d "$TOOLDIR/llvm-vortex" ]]; then
  echo "SKIPPED: needs --grxgpu <path> and a device toolchain in $TOOLDIR."
else
  "$ROOT/ci/build_kernel.sh" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
    "$ROOT/tests/kernels/dxa/kernel.cpp" -o "$BUILD/dxa.vxbin" >/dev/null
  $CXX $CXXFLAGS -I"$ROOT/tests/kernels/dxa" \
    -c "$ROOT/tests/kernels/dxa/main.cpp" -o "$BUILD/dxa_main.o"
  $CXX "${OBJS[@]}" "$BUILD/dxa_main.o" $LIBS -o "$BUILD/dxa_gate"
  if ! "$BUILD/dxa_gate" "$BUILD/dxa.vxbin"; then
    rc=$?
    [[ $rc -eq 77 ]] || exit $rc
  fi
fi

echo
echo "==> grxBLAS: library builds"
$CXX $CXXFLAGS -c "$ROOT/src/libs/grxblas/grxblas.cpp" -o "$BUILD/grxblas.o"

echo "==> GRXBLAS GATE: sgemm against a CPU reference"
if [[ -n "$GRXGPU" && -d "$TOOLDIR/llvm-vortex" ]]; then
  "$ROOT/ci/build_kernel.sh" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
    "$ROOT/src/libs/grxblas/kernels/sgemm.cpp" -o "$BUILD/grxblas_sgemm.vxbin" \
    >/dev/null
  $CXX $CXXFLAGS -I"$ROOT/tests/unit" -c "$ROOT/tests/libs/test_grxblas.cpp" \
    -o "$BUILD/test_grxblas.o"
  $CXX "${OBJS[@]}" "$BUILD/grxblas.o" "$BUILD/test_grxblas.o" $LIBS \
    -o "$BUILD/test_grxblas"
  "$BUILD/test_grxblas"
else
  echo "SKIPPED: needs --grxgpu <path> and a device toolchain in $TOOLDIR."
fi

echo
echo "==> conformance report against the real device"
"$BUILD/grx-conform" | tail -14

echo
echo "tier 2 passed on $DRIVER"
