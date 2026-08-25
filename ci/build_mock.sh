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

VORTEX_INCLUDE=""
VORTEX_CFLAGS=""
HOST_TRIPLE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --vortex-include) VORTEX_INCLUDE="$2"; shift 2 ;;
    --host)           HOST_TRIPLE="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Ask pkg-config before guessing at directory layouts: the installed sysroot
# does not put its headers where an --prefix=/usr/local intuition expects
# (they land in $VORTEX_PATH/runtime/include), and a script that hardcodes one
# layout fails with "not found" while the header is sitting right there.
if [[ -z "$VORTEX_INCLUDE" && -n "${VORTEX_PATH:-}" ]]; then
  if PKG_CONFIG_PATH="$VORTEX_PATH/lib/pkgconfig:${PKG_CONFIG_PATH:-}" \
     pkg-config --exists vortex-runtime 2>/dev/null; then
    # Take EVERY include directory pkg-config names, not just the one holding
    # vortex2.h: the runtime also includes VX_types.h, which the sysroot keeps
    # in the kernel include directory. Picking one directory compiles most of
    # the runtime and then fails on whichever file needs the other.
    VORTEX_CFLAGS="$(PKG_CONFIG_PATH="$VORTEX_PATH/lib/pkgconfig:${PKG_CONFIG_PATH:-}" \
                     pkg-config --cflags-only-I vortex-runtime)"
    for flag in $VORTEX_CFLAGS; do
      [[ -f "${flag#-I}/vortex2.h" ]] && { VORTEX_INCLUDE="${flag#-I}"; break; }
    done
  fi
  for d in "$VORTEX_PATH/runtime/include" "$VORTEX_PATH/include"; do
    [[ -n "$VORTEX_INCLUDE" ]] && break
    [[ -f "$d/vortex2.h" ]] && VORTEX_INCLUDE="$d"
  done
fi

if [[ -z "$VORTEX_INCLUDE" || ! -f "$VORTEX_INCLUDE/vortex2.h" ]]; then
  echo "error: vortex2.h not found${VORTEX_PATH:+ under $VORTEX_PATH}." >&2
  echo "  Set VORTEX_PATH to the installed GRX-G100 sysroot, or pass" >&2
  echo "  --vortex-include <dir containing vortex2.h>." >&2
  exit 1
fi

# --- host target ------------------------------------------------------------
#
# GRXCP's host half has to run on the machine the GPU is attached to, and for
# the GRX930 that machine is a RISC-V64 SoC rather than an x86 box. A runtime
# that has only ever been compiled for x86_64 is a runtime nobody has checked
# for x86-isms, so this script can build and RUN the whole mock stack for
# another host through a cross compiler and an emulator.
#
#   ./ci/build_mock.sh --host riscv64-linux-gnu
#
# Running matters more than compiling. A cross compile catches inline asm and
# __x86_64__ ifdefs; only an execution catches a struct laid out differently, a
# signed char assumption, or an alignment fault. qemu-user runs the result
# straight from this container.
RUN=()
if [[ -n "$HOST_TRIPLE" ]]; then
  CXX="$HOST_TRIPLE-g++"
  BUILD="$ROOT/build-mock-${HOST_TRIPLE%%-*}"
  if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "SKIPPED: no $CXX. Install it (Debian/Ubuntu: g++-$HOST_TRIPLE) to"
    echo "         check that the runtime is not x86-specific."
    exit 0
  fi
  # Native builds need no emulator; anything else does, and saying so beats
  # producing binaries nobody ran.
  if [[ "$HOST_TRIPLE" != "$(uname -m)"* ]]; then
    emu="qemu-${HOST_TRIPLE%%-*}-static"
    if ! command -v "$emu" >/dev/null 2>&1; then
      echo "SKIPPED: $CXX is present but $emu is not, so the result could be"
      echo "         built and not run. Install qemu-user-static."
      exit 0
    fi
    RUN=("$emu" -L "/usr/$HOST_TRIPLE")
  fi
  echo "==> host target: $HOST_TRIPLE  (CXX=$CXX${RUN:+, run under ${RUN[0]}})"
fi

CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -Wall -Wextra -O1 -g -I$ROOT/include -I$VORTEX_INCLUDE $VORTEX_CFLAGS"

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
  "${RUN[@]}" "$BUILD/$t"
done

echo
echo "==> unit tests (FPGA backend: managed memory must be gated off)"
for t in "${TESTS[@]}"; do
  VORTEX_DRIVER=xrt "${RUN[@]}" "$BUILD/$t" > /dev/null || { echo "FAILED on xrt: $t"; exit 1; }
done
echo "all tests pass on the FPGA backend selection too"

echo
echo "==> API table / compat header drift check"
python3 "$ROOT/ci/check_compat_table.py"

echo
echo "==> conformance report"
"${RUN[@]}" "$BUILD/grx-conform"

echo
echo "==> end-to-end: translate a CUDA source, compile it, link it, run it"
"${RUN[@]}" "$BUILD/grxify" --check "$ROOT/tests/conformance/portable_port.cu"
"${RUN[@]}" "$BUILD/grxify" "$ROOT/tests/conformance/portable_port.cu" \
  -o "$BUILD/portable_port.grx.cpp" 2>/dev/null
$CXX $CXXFLAGS -c "$BUILD/portable_port.grx.cpp" -o "$BUILD/portable_port.o"
$CXX "${RT_OBJS[@]}" "$BUILD/portable_port.o" -o "$BUILD/portable_port"
"${RUN[@]}" "$BUILD/portable_port"

echo
echo "==> grx-prof against the mock: no counters, and no invented ones"
if [[ ${#RUN[@]} -gt 0 ]]; then
  echo "SKIPPED on a cross host: grx-prof runs its subject in a CHILD process,"
  echo "        and qemu-user does not emulate a binary it did not start."
  echo "        The counter-unavailable path is covered by the native run."
else
# The mock refuses vx_device_mpm_query, because it models a control plane and
# has no pipeline to count. This is the only place in CI that exercises the
# runtime's counter-unavailable path, and the claim being checked is precise:
# the trace still has its kernel slices, and NONE of them carries a device
# counter. A zero here would be a number nobody measured.
$CXX $CXXFLAGS -c "$ROOT/tools/grx-prof/main.cpp" -o "$BUILD/grx-prof.o"
$CXX "$BUILD/grx-prof.o" -o "$BUILD/grx-prof"
"$BUILD/grx-prof" --out "$BUILD/prof_mock.json" --quiet -- \
  "$BUILD/test_launch" > "$BUILD/prof_mock.log" 2>&1
python3 - "$BUILD/prof_mock.json" <<'PY'
import json, sys
trace = json.load(open(sys.argv[1]))
launches = [e for e in trace["traceEvents"]
            if e.get("ph") == "X" and e.get("cat") == "launch"]
if not launches:
    print("FAILED: the trace has no kernel slice"); sys.exit(1)
invented = [k for k in launches
            if any(a.startswith("device.") for a in k.get("args", {}))]
if invented:
    print("FAILED: a device counter appeared on a backend that has none:")
    print(" ", invented[0]["args"]); sys.exit(1)
print(f"  ok    {len(launches)} kernel slices, no fabricated device counters")
PY
fi

echo
echo "==> grxify reports the unmappable calls in an awkward port"
if "${RUN[@]}" "$BUILD/grxify" --check "$ROOT/tests/conformance/sample_port.cu" 2>&1; then
  echo "FAILED: grxify should have reported unmappable calls"; exit 1
fi
echo "(the diagnostics above are the expected result)"

echo
echo "==> grx-smi (default config)"
"${RUN[@]}" "$BUILD/grx-smi"

echo "==> grx-smi (flagship G100 preset)"
VORTEX_DRIVER=rtlsim \
GRXMOCK_NUM_THREADS=32 GRXMOCK_NUM_WARPS=64 \
GRXMOCK_NUM_CORES=16 GRXMOCK_NUM_CLUSTERS=8 \
GRXMOCK_SOCKET_SIZE=4 GRXMOCK_ISSUE_WIDTH=4 \
GRXMOCK_LOCAL_MEM=262144 GRXMOCK_GLOBAL_MEM=137438953472 \
GRXMOCK_MEM_BANKS=8 GRXMOCK_MEM_BANK_SIZE=17179869184 \
GRXMOCK_CLOCK_MHZ=2000 GRXMOCK_PEAK_BW_MBS=6400000 \
  "${RUN[@]}" "$BUILD/grx-smi"

echo
echo "==> NPU BACKEND GATE: what it decides, against a register model"
# The c930 backend has no Vortex dependency and needs no sysroot, so this runs
# in tier 1 on any machine with a compiler.
#
# test_npu_c930.cc covers the offline half -- register offsets, validation,
# INT8 packing, a GEMM reference -- and says out loud that it cannot reach the
# rest: "[SKIP] Mock mode requires mmap infrastructure". Both bugs were in the
# part it skipped. src/backends/npu_c930/test_npu_c930_model.cc drives the
# backend through four register models instead, two of which are the states a
# machine WITHOUT a c930 produces.
#
# A model is not hardware. Passing here says the backend's logic is right; it
# says nothing about the c930, and no result from it may be reported as the NPU
# working.
NPU_DIR="$ROOT/src/backends/npu_c930"
if [[ -f "$NPU_DIR/test_npu_c930_model.cc" ]]; then
  $CXX -std=c++17 -Wall -Wextra -O1 -I"$NPU_DIR" \
    "$NPU_DIR/test_npu_c930_model.cc" "$NPU_DIR/npu_c930.cpp" \
    -o "$BUILD/test_npu_c930_model"
  "${RUN[@]}" "$BUILD/test_npu_c930_model"

  # The offline suite too, since nothing else built it: src/CMakeLists.txt globs
  # this directory's *.cpp into libgrxrt and never adds the subdirectory, so the
  # test_npu_c930 target it defines was never configured.
  $CXX -std=c++17 -O1 -I"$NPU_DIR" \
    "$NPU_DIR/test_npu_c930.cc" "$NPU_DIR/npu_c930.cpp" \
    -o "$BUILD/test_npu_c930"
  "${RUN[@]}" "$BUILD/test_npu_c930" | tail -3
else
  echo "SKIPPED: no NPU backend in this tree."
fi

echo
echo "==> NPU GROUNDWORK: a device with no pipeline refuses launches"
# Phase 7 begins here, before there is an NPU to talk to. The c930 NPU is a
# systolic array with no SIMT pipeline, and grxcp_architecture.md section 6
# fixes the contract: grxLaunchKernel on it returns grxErrorNotSupported and
# does NOT silently fall back to the GPU. A fallback would give the right answer
# on the wrong engine, which is invisible until someone measures.
#
# Run TWICE. The refusal on its own would pass against a runtime that refused
# every launch on every device, so the default configuration is the control.
echo "--- a device reporting zero warps"
GRXMOCK_NUM_WARPS=0 "${RUN[@]}" "$BUILD/test_no_kernel_launch"
echo "--- control: the same binary on the default device, which CAN launch"
"${RUN[@]}" "$BUILD/test_no_kernel_launch"

echo
echo "==> CMAKE GATE: the project's own build system configures and builds"
# Everything above this line compiles GRXCP by hand, because that is what a
# mock build needs. Nothing checked that CMakeLists.txt worked -- and it did
# not, from the first commit until the GRX930 team asked whether
# `cmake -DGRXCP_ENABLE_NPU=ON` builds. The top-level file had
# `if(X) add_subdirectory(y) endif()` on one line, which CMake parses as an
# error and not as a terse spelling; src/CMakeLists.txt and cmake/grxrt.pc.in
# did not exist at all. A build system nobody runs is a build system that does
# not work, and this project ships one to its users.
#
# The NPU flag is checked HERE rather than left to be discovered, and it is
# checked for REFUSING: there is no src/backends/npu_c930/ in this tree, so a
# configure that succeeded with the flag on would be a GPU build wearing an NPU
# flag. See docs/designs/grxcp_roadmap.md phase 7.
if [[ -n "$HOST_TRIPLE" ]]; then
  echo "SKIPPED: cross build; the cmake gate runs on the native host."
elif ! command -v cmake >/dev/null 2>&1; then
  echo "SKIPPED: no cmake on this machine."
else
  CMBUILD="$BUILD/cmake"
  rm -rf "$CMBUILD"
  # The same PKG_CONFIG_PATH the rest of this script uses, so cmake finds the
  # sysroot's .pc files rather than reporting the project broken when the only
  # thing missing is a search path.
  export PKG_CONFIG_PATH="${VORTEX_PATH:+$VORTEX_PATH/lib/pkgconfig:}${PKG_CONFIG_PATH:-}"
  cmake -S "$ROOT" -B "$CMBUILD" -DCMAKE_BUILD_TYPE=Release \
    -DGRXCP_USE_MOCK_DRIVER=ON \
    -DGRXCP_VORTEX_INCLUDE_DIR="$VORTEX_INCLUDE" \
    > "$CMBUILD.log" 2>&1 || {
      echo "FAILED: cmake could not configure this project."
      tail -20 "$CMBUILD.log" | sed 's/^/        /'; exit 1; }
  cmake --build "$CMBUILD" -j"$(nproc 2>/dev/null || echo 4)" \
    >> "$CMBUILD.log" 2>&1 || {
      echo "FAILED: cmake configured but could not build."
      grep -E "error:" "$CMBUILD.log" | head -10 | sed 's/^/        /'; exit 1; }
  echo "  ok    configured and built"

  # Every library and tool the file claims to produce, produced.
  missing=0
  for f in libgrxrt.so src/libs/libgrxblas.so src/libs/libgrxdnn.so \
           tools/grx-smi tools/grxcc tools/grxify; do
    [[ -e "$CMBUILD/$f" ]] || { echo "  FAIL  cmake did not build $f"; missing=1; }
  done
  [[ $missing -eq 0 ]] || exit 1
  echo "  ok    grxrt, grxblas, grxdnn and the tools are all present"

  # The NPU flag, which is the thing the GRX930 team asked about. Both
  # configurations are built, because "the GPU path is unchanged when the flag
  # is off" is a claim and not an assumption.
  if ! cmake -S "$ROOT" -B "$BUILD/cmake-npu" -DGRXCP_ENABLE_NPU=ON \
       -DGRXCP_USE_MOCK_DRIVER=ON \
       -DGRXCP_VORTEX_INCLUDE_DIR="$VORTEX_INCLUDE" \
       > "$BUILD/cmake-npu.log" 2>&1 ||
     ! cmake --build "$BUILD/cmake-npu" -j"$(nproc 2>/dev/null || echo 4)" \
       >> "$BUILD/cmake-npu.log" 2>&1; then
    echo "  FAIL  -DGRXCP_ENABLE_NPU=ON does not build."
    grep -E "error:|Error|CMake Error" "$BUILD/cmake-npu.log" | head -8 |
      sed 's/^/        /'
    exit 1
  fi
  echo "  ok    -DGRXCP_ENABLE_NPU=ON configures and builds"

  # The flag has to reach the COMPILER, not just the source glob. It did not:
  # npu_c930.cpp was compiled into libgrxrt while every #ifdef GRXCP_ENABLE_NPU
  # block in context.cpp compiled to nothing, so the build contained the backend
  # and could not enumerate the device it drives. The test targets next to the
  # backend were never configured either.
  for t in test_npu_c930 test_npu_c930_model; do
    if [[ ! -x "$BUILD/cmake-npu/src/backends/npu_c930/$t" ]]; then
      echo "  FAIL  the NPU build did not produce $t"
      exit 1
    fi
  done
  echo "  ok    the NPU backend's own test targets were configured and built"

  # AND THE NPU MUST NOT BE ENUMERATED HERE. There is no c930 in CI. A build
  # flag says what code exists, not what hardware is attached, and a device that
  # appears in grx-smi on a machine that has none is the failure this whole
  # section exists to prevent.
  npu_seen="$("$BUILD/cmake-npu/tools/grx-smi" 2>/dev/null | grep -c 'NPU' || true)"
  if [[ "$npu_seen" != "0" ]]; then
    echo "  FAIL  a GRX930 NPU was enumerated on a machine that has none:"
    "$BUILD/cmake-npu/tools/grx-smi" 2>&1 | grep -i npu | sed 's/^/        /'
    exit 1
  fi
  echo "  ok    and no NPU is enumerated on this machine, which has none"
fi

echo "all mock checks passed"
