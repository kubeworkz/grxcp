#!/usr/bin/env bash
# Tier 2: build GRXCP against a real GRX-G100 sysroot and run the gates that
# the mock driver cannot answer.
#
# What this proves that ci/build_mock.sh does not: the runtime links against
# the actual driver with no source changes, the device it reports is the one
# the simulator actually models, and data moves through the real command
# processor rather than through a std::memcpy in a test fixture.
#
# It also runs the kernel gates when the device toolchain is available --
# vecadd (the phase 1 exit gate), the WMMA tile, the DXA async copy, the cycle
# probe, and grxBLAS sgemm -- each computing on the device and checked on the
# host. The last section is a REPORT rather than a gate: sgemm v0's cost in
# device cycles, which is the baseline the tuned kernel has to beat.
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
# grx-sanitize and grx-prof run a program and read its report; neither opens a
# device, so they link no more than grxify does.
for tool in grx-sanitize grx-prof; do
  $CXX $CXXFLAGS -c "$ROOT/tools/$tool/main.cpp" -o "$BUILD/$tool.o"
  $CXX "$BUILD/$tool.o" -o "$BUILD/$tool"
done

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
echo "==> WARP GATE: shuffle and vote against CUDA semantics"
if [[ -z "$GRXGPU" || ! -d "$TOOLDIR/llvm-vortex" ]]; then
  echo "SKIPPED: needs --grxgpu <path> and a device toolchain in $TOOLDIR."
else
  "$ROOT/ci/build_kernel.sh" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
    "$ROOT/tests/kernels/warp/kernel.cpp" -o "$BUILD/warp.vxbin" >/dev/null
  $CXX $CXXFLAGS -I"$ROOT/tests/kernels/warp" \
    -c "$ROOT/tests/kernels/warp/main.cpp" -o "$BUILD/warp_main.o"
  $CXX "${OBJS[@]}" "$BUILD/warp_main.o" $LIBS -o "$BUILD/warp_gate"
  if ! "$BUILD/warp_gate" "$BUILD/warp.vxbin"; then
    rc=$?
    [[ $rc -eq 77 ]] || exit $rc
  fi
fi

echo
echo "==> PROF GATE: a Perfetto trace, and counters that respond to the work"
# The phase 2 exit gate asks that grx-prof "produces a Perfetto trace a human
# can read". Readable is checked three ways: the file parses as a trace, the
# kernel slice carries the device cycle count, and the report states which of
# its numbers are host-clock -- a timeline whose axis lies about what it
# measures is not readable, it is misleading.
#
# The fourth check is the one that matters most. A profiler that emits numbers
# nobody has watched respond to their input is not measuring anything, so the
# same kernel runs at three sizes and the device cycle count has to climb. It
# is the same discipline the cycle gate applies to grx::cycle_probe.
if [[ -z "$GRXGPU" || ! -d "$TOOLDIR/llvm-vortex" ]]; then
  echo "SKIPPED: needs --grxgpu <path> and a device toolchain in $TOOLDIR."
else
  [[ -f "$BUILD/vecadd.vxbin" ]] || \
    "$ROOT/ci/build_kernel.sh" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
      "$ROOT/tests/kernels/vecadd/kernel.cpp" -o "$BUILD/vecadd.vxbin" >/dev/null
  [[ -f "$BUILD/vecadd" ]] || {
    $CXX $CXXFLAGS -I"$ROOT/tests/kernels/vecadd" \
      -c "$ROOT/tests/kernels/vecadd/main.cpp" -o "$BUILD/vecadd_main.o"
    $CXX "${OBJS[@]}" "$BUILD/vecadd_main.o" $LIBS -o "$BUILD/vecadd"; }

  for n in 64 256 1024; do
    "$BUILD/grx-prof" --out "$BUILD/prof_$n.json" -- \
      "$BUILD/vecadd" "$BUILD/vecadd.vxbin" "$n" > "$BUILD/prof_$n.log" 2>&1 || {
        echo "  FAIL  grx-prof failed at n=$n"; cat "$BUILD/prof_$n.log"; exit 1; }
  done

  python3 - "$BUILD" <<'PY' || exit 1
import json, sys, os
build = sys.argv[1]
cycles = {}
for n in (64, 256, 1024):
    with open(os.path.join(build, f"prof_{n}.json")) as f:
        trace = json.load(f)
    events = trace["traceEvents"]
    slices = [e for e in events if e.get("ph") == "X"]
    launches = [e for e in slices if e.get("cat") == "launch"]
    if not launches:
        print(f"  FAIL  n={n}: the trace has no kernel slice"); sys.exit(1)
    k = launches[0]
    if k.get("dur", 0) <= 0:
        print(f"  FAIL  n={n}: the kernel slice has no duration"); sys.exit(1)
    c = k.get("args", {}).get("device.cycles")
    if not c:
        print(f"  FAIL  n={n}: the kernel slice carries no device cycle count")
        sys.exit(1)
    if not any(e.get("ph") == "M" and e.get("name") == "process_name"
               for e in events):
        print(f"  FAIL  n={n}: the trace names no process"); sys.exit(1)
    cycles[n] = c
print("  ok    trace parses, kernel slice carries device cycles  " +
      " ".join(f"n={n}:{c}" for n, c in cycles.items()))
if not (cycles[64] < cycles[256] < cycles[1024]):
    print("  FAIL  device cycles do not climb with the work"); sys.exit(1)
ratio = cycles[1024] / cycles[256]
if not (2.0 <= ratio <= 5.0):
    print(f"  FAIL  4x the work moved cycles by {ratio:.2f}x, expected 2-5x")
    sys.exit(1)
print(f"  ok    4x the work costs {ratio:.2f}x the cycles")
PY

  for want in "host clock" "where the cycles went" "Occupancy the dispatcher"; do
    grep -q "$want" "$BUILD/prof_1024.log" || {
      echo "  FAIL  the report never says \"$want\""; exit 1; }
  done
  echo "  ok    the report separates device cycles from the host clock"

  # A program that never reaches the GRXCP runtime must be reported as
  # unprofiled, not as a program that did nothing interesting.
  if "$BUILD/grx-prof" --no-trace -- /bin/true > "$BUILD/prof_none.log" 2>&1; then
    echo "  FAIL  reported success for a run it never profiled"
    cat "$BUILD/prof_none.log"; exit 1
  fi
  grep -q "not in profiling mode" "$BUILD/prof_none.log" || {
    echo "  FAIL  no warning that nothing was profiled"
    cat "$BUILD/prof_none.log"; exit 1; }
  echo "  ok    an unprofiled run is reported as unprofiled"
fi

echo
echo "==> SANITIZE GATE: planted memory bugs, found and located to the line"
# The phase 2 exit gate asks that grx-sanitize "detects a deliberately planted
# out-of-bounds write and reports the source line". Four planted bugs, each
# checked for the exact line it lives on -- and, because a detector that never
# fires would pass a test that only looks for failures, two controls: the same
# kernel with the bug removed must come back clean, and the same bug in an
# UNINSTRUMENTED build must be reported as unchecked rather than as clean.
if [[ -z "$GRXGPU" || ! -d "$TOOLDIR/llvm-vortex" ]]; then
  echo "SKIPPED: needs --grxgpu <path> and a device toolchain in $TOOLDIR."
else
  SAN_SRC="$ROOT/tests/kernels/sanitize/kernel.cpp"
  "$ROOT/ci/build_kernel.sh" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" --sanitize \
    "$SAN_SRC" -o "$BUILD/sanitize.vxbin" >/dev/null
  "$ROOT/ci/build_kernel.sh" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
    "$SAN_SRC" -o "$BUILD/sanitize_plain.vxbin" >/dev/null
  $CXX $CXXFLAGS -I"$ROOT/tests/kernels/sanitize" \
    -c "$ROOT/tests/kernels/sanitize/main.cpp" -o "$BUILD/sanitize_main.o"
  $CXX "${OBJS[@]}" "$BUILD/sanitize_main.o" $LIBS -o "$BUILD/test_sanitize"
  export TOOLDIR   # how grx-sanitize finds llvm-symbolizer

  # Line numbers come from the source, not from this script: a marker comment
  # is a thing an editor moves along with the code, and a hard-coded 31 is not.
  san_line() { grep -n "PLANTED: $1" "$SAN_SRC" | head -1 | cut -d: -f1; }

  san_expect() {   # scenario, marker text
    local sc="$1" want; want="$(san_line "$2")"
    local log="$BUILD/san_$sc.log"
    if "$BUILD/grx-sanitize" -- "$BUILD/test_sanitize" \
         "$BUILD/sanitize.vxbin" "$sc" >"$log" 2>&1; then
      echo "  FAIL  $sc: grx-sanitize found nothing"; cat "$log"; exit 1
    fi
    if ! grep -q "kernel.cpp:$want:" "$log"; then
      echo "  FAIL  $sc: no finding at kernel.cpp:$want"; cat "$log"; exit 1
    fi
    printf '  ok    %-16s located at kernel.cpp:%s\n' "$sc" "$want"
  }

  san_expect oob-write      "one past the end"
  san_expect oob-read       "before the start"
  san_expect use-after-free "buffer already freed"
  san_expect oob-shared     "past sharedMem"

  # Control 1: the same program, the same allocator, no planted bug.
  if ! "$BUILD/grx-sanitize" -- "$BUILD/test_sanitize" \
         "$BUILD/sanitize.vxbin" clean >"$BUILD/san_clean.log" 2>&1; then
    echo "  FAIL  clean: findings in a correct kernel"
    cat "$BUILD/san_clean.log"; exit 1
  fi
  echo "  ok    clean            no findings"

  # Control 2: instrumentation actually matters. Without it the tool must say
  # the run was unchecked -- silence here would mean every unsanitized build
  # passes this gate forever.
  if "$BUILD/grx-sanitize" -- "$BUILD/test_sanitize" \
       "$BUILD/sanitize_plain.vxbin" oob-write >"$BUILD/san_plain.log" 2>&1; then
    echo "  FAIL  uninstrumented: reported success on an unchecked run"
    cat "$BUILD/san_plain.log"; exit 1
  fi
  if ! grep -q "no instrumentation" "$BUILD/san_plain.log"; then
    echo "  FAIL  uninstrumented: no warning that nothing was checked"
    cat "$BUILD/san_plain.log"; exit 1
  fi
  echo "  ok    uninstrumented   reported as unchecked, not as clean"
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
echo "==> WATCH: is the tensor unit still deadlocking on a second CTA"
# A watch, not a gate: the defect is upstream, the workaround is in place, and
# what CI has to do is notice the day it is fixed. It runs the launch in a
# child process under a timeout, so a deadlock costs a minute rather than the
# whole run.
if "$BUILD/grx-smi" 2>/dev/null | grep -q 'capabilities.*tensor' &&
   [[ -n "$GRXGPU" && -d "$TOOLDIR/llvm-vortex" ]]; then
  "$ROOT/ci/build_kernel.sh" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
    "$ROOT/tests/repro/tcu_multi_cta/kernel.cpp" -o "$BUILD/tcu_repro.vxbin" \
    >/dev/null
  $CXX $CXXFLAGS -c "$ROOT/tests/repro/tcu_multi_cta/main.cpp" \
    -o "$BUILD/tcu_repro_main.o"
  $CXX "${OBJS[@]}" "$BUILD/tcu_repro_main.o" $LIBS -o "$BUILD/tcu_repro"
  "$BUILD/tcu_repro" "$BUILD/tcu_repro.vxbin" || true
else
  echo "SKIPPED: no tensor unit or no device toolchain."
fi

echo
echo "==> CYCLE GATE: does the device cycle probe measure work"
if [[ -z "$GRXGPU" || ! -d "$TOOLDIR/llvm-vortex" ]]; then
  echo "SKIPPED: needs --grxgpu <path> and a device toolchain in $TOOLDIR."
else
  "$ROOT/ci/build_kernel.sh" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
    "$ROOT/tests/kernels/cycles/kernel.cpp" -o "$BUILD/cycles.vxbin" >/dev/null
  $CXX $CXXFLAGS -I"$ROOT/tests/kernels/cycles" \
    -c "$ROOT/tests/kernels/cycles/main.cpp" -o "$BUILD/cycles_main.o"
  $CXX "${OBJS[@]}" "$BUILD/cycles_main.o" $LIBS -o "$BUILD/cycles_gate"
  if ! "$BUILD/cycles_gate" "$BUILD/cycles.vxbin"; then
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

echo "==> GRXBLAS GATES: sgemm and GemmEx against a CPU reference"
if [[ -n "$GRXGPU" && -d "$TOOLDIR/llvm-vortex" ]]; then
  # One module with every kernel. Each .vxbin links at the same load address,
  # so two of them cannot be resident at once and a library that needs both
  # sgemm and GemmEx has to ship them together. See kernels/all.cpp.
  if "$BUILD/grx-smi" 2>/dev/null | grep -q 'capabilities.*tensor'; then
    KSRC="$ROOT/src/libs/grxblas/kernels/all.cpp"
  else
    echo "    (no tensor unit: building the scalar-only module)"
    KSRC="$ROOT/src/libs/grxblas/kernels/sgemm.cpp"
  fi
  "$ROOT/ci/build_kernel.sh" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
    "$KSRC" -o "$BUILD/grxblas_kernels.vxbin" >/dev/null

  for t in test_grxblas test_grxblas_ex; do
    $CXX $CXXFLAGS -I"$ROOT/tests/unit" -c "$ROOT/tests/libs/$t.cpp" \
      -o "$BUILD/$t.o"
    $CXX "${OBJS[@]}" "$BUILD/grxblas.o" "$BUILD/$t.o" $LIBS -o "$BUILD/$t"
    printf -- "--- %s\n" "$t"
    if ! "$BUILD/$t"; then
      rc=$?
      [[ $rc -eq 77 ]] || exit $rc
    fi
  done
else
  echo "SKIPPED: needs --grxgpu <path> and a device toolchain in $TOOLDIR."
fi

echo
echo "==> PHASE 3 EXIT GATE: scalar against tensor, in device cycles"
if [[ -n "$GRXGPU" && -d "$TOOLDIR/llvm-vortex" ]]; then
  $CXX $CXXFLAGS -c "$ROOT/tests/bench/gemm_cycles.cpp" -o "$BUILD/gemm_cycles.o"
  $CXX "${OBJS[@]}" "$BUILD/grxblas.o" "$BUILD/gemm_cycles.o" $LIBS \
    -o "$BUILD/sgemm_bench"
  if ! "$BUILD/sgemm_bench"; then
    rc=$?
    [[ $rc -eq 77 ]] || exit $rc
  fi
else
  echo "SKIPPED: needs --grxgpu <path> and a device toolchain in $TOOLDIR."
fi

echo
echo "==> conformance report against the real device"
"$BUILD/grx-conform" | tail -14

echo
echo "tier 2 passed on $DRIVER"
