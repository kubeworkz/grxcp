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
# device, so they link no more than grxify does. grxcc is the same shape: it
# rewrites text and shells out to a compiler, so a machine that only COMPILES
# for GRX-G100 needs no driver installed to build it.
for tool in grx-sanitize grx-prof grxcc; do
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
echo "==> CG GATE: cooperative groups, up to and including the grid barrier"
# thread_block, thread_block_tile at two widths, coalesced_group inside a
# divergent branch, the cluster, and this_grid().sync() through a cooperative
# launch. Every reference is computed independently on the host.
#
# The grid barrier carries its own control: the same kernel with the barrier
# removed must get the answer WRONG. Block 0 stalls before publishing, so a
# block that does not wait reads the sentinel -- without that, both blocks
# would publish long before either read and the test would pass with or
# without a barrier.
if [[ -z "$GRXGPU" || ! -d "$TOOLDIR/llvm-vortex" ]]; then
  echo "SKIPPED: needs --grxgpu <path> and a device toolchain in $TOOLDIR."
else
  "$ROOT/ci/build_kernel.sh" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
    "$ROOT/tests/kernels/cg/kernel.cpp" -o "$BUILD/cg.vxbin" >/dev/null
  $CXX $CXXFLAGS -I"$ROOT/tests/kernels/cg" \
    -c "$ROOT/tests/kernels/cg/main.cpp" -o "$BUILD/cg_main.o"
  $CXX "${OBJS[@]}" "$BUILD/cg_main.o" $LIBS -o "$BUILD/cg_gate"
  if ! "$BUILD/cg_gate" "$BUILD/cg.vxbin"; then
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

  for n in 256 1024 4096; do
    "$BUILD/grx-prof" --out "$BUILD/prof_$n.json" -- \
      "$BUILD/vecadd" "$BUILD/vecadd.vxbin" "$n" > "$BUILD/prof_$n.log" 2>&1 || {
        echo "  FAIL  grx-prof failed at n=$n"; cat "$BUILD/prof_$n.log"; exit 1; }
  done

  python3 - "$BUILD" <<'PY' || exit 1
import json, sys, os
build = sys.argv[1]
cycles = {}
for n in (256, 1024, 4096):
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
if not (cycles[256] < cycles[1024] < cycles[4096]):
    print("  FAIL  device cycles do not climb with the work"); sys.exit(1)

# THE SLOPE, NOT THE RATIO.
#
# This used to require cycles[1024]/cycles[256] to land in 2-5x. That band was
# calibrated to a 4-warp core and stopped being true the moment the device
# configuration changed: on a 16-warp core the same measurement reads 1.52x,
# because a launch costs thousands of cycles whatever the grid is and the extra
# work is absorbed by parallelism. Nothing was wrong -- the check was
# configuration-specific and said so only by failing.
#
# What has to hold on ANY configuration is that the MARGINAL cost of an element
# is positive and roughly stable. A fixed launch overhead cancels out of a
# difference, so this is the same claim -- the counter responds to the work --
# without a constant tuned to one machine.
m1 = (cycles[1024] - cycles[256]) / (1024 - 256)
m2 = (cycles[4096] - cycles[1024]) / (4096 - 1024)
if m1 <= 0 or m2 <= 0:
    print(f"  FAIL  marginal cycles per element is not positive "
          f"({m1:.2f}, {m2:.2f})")
    sys.exit(1)
if not (0.33 <= m2 / m1 <= 3.0):
    print(f"  FAIL  marginal cycles per element moved {m2 / m1:.2f}x between "
          f"the two intervals ({m1:.2f} then {m2:.2f}); an unblocked vecadd "
          f"should cost about the same per element at every size")
    sys.exit(1)
print(f"  ok    marginal cost per element is stable: "
      f"{m1:.2f} then {m2:.2f} cycles")
PY

  for want in "host clock" "where the cycles went" "Occupancy the dispatcher"; do
    grep -q "$want" "$BUILD/prof_4096.log" || {
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
echo "==> BARRIER GATE: is __syncthreads() still surviving divergence"
# Half gate, half watch. guarded_good is GRXCP's convergent __syncthreads() and
# MUST pass -- a failure there is our regression. guarded_bad is upstream's bare
# vx_barrier and is expected to deadlock; it runs in a child under a timeout so
# CI notices the day the toolchain stops duplicating it.
if [[ -n "$GRXGPU" && -d "$TOOLDIR/llvm-vortex" ]]; then
  "$ROOT/ci/build_kernel.sh" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
    -I "$ROOT/include" \
    "$ROOT/tests/repro/barrier_duplication/kernel.cpp" \
    -o "$BUILD/barrier_repro.vxbin" >/dev/null
  $CXX $CXXFLAGS -c "$ROOT/tests/repro/barrier_duplication/main.cpp" \
    -o "$BUILD/barrier_repro_main.o"
  $CXX "${OBJS[@]}" "$BUILD/barrier_repro_main.o" $LIBS -o "$BUILD/barrier_repro"
  if ! "$BUILD/barrier_repro" "$BUILD/barrier_repro.vxbin"; then
    rc=$?
    [[ $rc -eq 77 ]] || exit $rc
  fi
else
  echo "SKIPPED: needs --grxgpu <path> and a device toolchain in $TOOLDIR."
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

echo "==> GRXBLAS GATES: level 1, level 2, sgemm and GemmEx vs a CPU reference"
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

  for t in test_grxblas test_grxblas_l12 test_grxblas_ex; do
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
echo "==> PHASE 4 GATE: grxcc compiles one file with <<<>>> and it runs"
# The claim under test is narrow and specific: a source file in the shape a
# CUDA programmer writes -- __global__ kernels and <<<>>> launches in the same
# translation unit, no module load, no .vxbin named anywhere -- goes in, and a
# working program comes out. The sample checks VALUES rather than return codes,
# because a mispacked argument blob is a wrong answer and not an error.
if [[ -n "$GRXGPU" && -d "$TOOLDIR/llvm-vortex" ]]; then
  "$BUILD/grxcc" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
    --build-kernel "$ROOT/ci/build_kernel.sh" -I "$ROOT/include" \
    "$ROOT/tests/grxcc/vecadd.grx.cpp" \
    "${OBJS[@]}" $LIBS -o "$BUILD/grxcc_vecadd" >/dev/null
  if ! "$BUILD/grxcc_vecadd"; then
    rc=$?
    [[ $rc -eq 77 ]] || exit $rc
  fi

  # The parser, separately from the arithmetic. A mis-lexed file fails by
  # generating mangled source rather than by computing the wrong number, so this
  # sample puts kernels at three scopes and surrounds them with decoys -- a
  # __global__ in a comment, a <<< in a string and in a raw string, a namespace
  # alias -- and then checks values so a kernel that never ran cannot pass.
  "$BUILD/grxcc" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
    --build-kernel "$ROOT/ci/build_kernel.sh" -I "$ROOT/include" \
    "$ROOT/tests/grxcc/scopes.grx.cpp" \
    "${OBJS[@]}" $LIBS -o "$BUILD/grxcc_scopes" >/dev/null
  if ! "$BUILD/grxcc_scopes"; then
    rc=$?
    [[ $rc -eq 77 ]] || exit $rc
  fi

  # __launch_bounds__ and per-kernel register metadata, each against a control:
  # an unbounded twin that must accept the launch the bounded one refuses, and a
  # register-hungry kernel that must read higher than a trivial one. The .vxbin
  # argument lets it also confirm the -1 sentinel still appears where nothing
  # was measured.
  "$BUILD/grxcc" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
    --build-kernel "$ROOT/ci/build_kernel.sh" -I "$ROOT/include" \
    "$ROOT/tests/grxcc/attributes.grx.cpp" \
    "${OBJS[@]}" $LIBS -o "$BUILD/grxcc_attributes" >/dev/null
  if ! "$BUILD/grxcc_attributes" "$BUILD/vecadd.vxbin"; then
    rc=$?
    [[ $rc -eq 77 ]] || exit $rc
  fi

  # Negative controls: the driver must REJECT what it cannot compile correctly
  # rather than emitting something that silently does nothing. Each of these is
  # a construct grxcc's own documentation says it does not support, and a
  # documented limit that is not enforced is just a bug with a paragraph.
  negative_case() {
    local what="$1" body="$2"
    printf '%s\n' "$body" > "$BUILD/grxcc_negative.grx.cpp"
    if "$BUILD/grxcc" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
         --build-kernel "$ROOT/ci/build_kernel.sh" -I "$ROOT/include" \
         --emit-only "$BUILD/grxcc_negative.grx.cpp" \
         -o "$BUILD/grxcc_negative" >/dev/null 2>&1; then
      echo "FAILED: grxcc accepted $what."
      exit 1
    fi
    echo "ok    grxcc rejects $what"
  }

  negative_case "a launch of a name that is not a __global__" \
'#include <grx/grx.h>
void not_a_kernel(int);
int main() { not_a_kernel<<<1, 1>>>(0); return 0; }'

  negative_case "a templated kernel" \
'#include <grx/grx.h>
template <typename T> __global__ void k(T* p) { p[0] = T{}; }
int main() { return 0; }'

  negative_case "a kernel that is not at namespace scope" \
'#include <grx/grx.h>
struct S { __global__ static void k(float* p) { p[0] = 1.0f; } };
int main() { return 0; }'

  negative_case "two kernels sharing an unqualified name" \
'#include <grx/grx.h>
namespace a { __global__ void run(float* p) { p[0] = 1.0f; } }
namespace b { __global__ void run(float* p) { p[0] = 2.0f; } }
int main() { return 0; }'
else
  echo "SKIPPED: needs --grxgpu <path> and a device toolchain in $TOOLDIR."
fi

echo
echo "==> HOST MATRIX GATE: grxcc targets a host that is not this one"
# GRX-G100 hangs off a GRX930, which is a RISC-V64 SoC -- so the host half of a
# GRXCP program has to compile for riscv64-linux-gnu, not only for the x86 box
# the simulator happens to run on. grxcc shells its host pass out to $CXX, so
# targeting another host is a matter of setting it; this gate is what says that
# is true rather than assumed.
#
# COMPILE, not link and not run. The driver in this container is built for
# x86_64, so there is nothing for a riscv64 object to link against here. What
# is proved is that the code grxcc GENERATES carries no x86 dependency.
# ci/build_mock.sh --host riscv64-linux-gnu proves the stronger thing about the
# runtime itself -- it cross-builds the whole mock stack and RUNS it under
# qemu-user, and a planted __builtin_ia32_rdtsc was watched failing there.
HOST_TRIPLE_ALT="${HOST_TRIPLE_ALT:-riscv64-linux-gnu}"
if [[ -z "$GRXGPU" || ! -d "$TOOLDIR/llvm-vortex" ]]; then
  echo "SKIPPED: needs --grxgpu <path> and a device toolchain in $TOOLDIR."
elif ! command -v "$HOST_TRIPLE_ALT-g++" >/dev/null 2>&1; then
  echo "SKIPPED: no $HOST_TRIPLE_ALT-g++ (Debian/Ubuntu: g++-$HOST_TRIPLE_ALT)."
else
  CXX="$HOST_TRIPLE_ALT-g++" "$BUILD/grxcc" \
    --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
    --build-kernel "$ROOT/ci/build_kernel.sh" -I "$ROOT/include" -c \
    "$ROOT/tests/grxcc/vecadd.grx.cpp" -o "$BUILD/vecadd_$HOST_TRIPLE_ALT.o" \
    > "$BUILD/hostmatrix.log" 2>&1 || {
      echo "FAIL  the host pass did not compile for $HOST_TRIPLE_ALT:"
      grep -E "error:" "$BUILD/hostmatrix.log" | head -3 | sed 's/^/        /'
      exit 1; }

  # And it has to BE that machine. A cross compiler that silently fell back to
  # the native target would pass a check that only looked at the exit code.
  machine="$("$HOST_TRIPLE_ALT-readelf" -h "$BUILD/vecadd_$HOST_TRIPLE_ALT.o" \
             2>/dev/null | sed -n 's/^ *Machine: *//p')"
  case "$machine" in
    *RISC-V*|*"$HOST_TRIPLE_ALT"*)
      echo "  ok    the host pass compiles for $HOST_TRIPLE_ALT ($machine)" ;;
    *)
      echo "FAIL  the object claims machine '$machine', not $HOST_TRIPLE_ALT"
      exit 1 ;;
  esac
fi

echo
echo "==> CUDA SAMPLES GATE: eleven CUDA programs, compiled unmodified"
# The phase 4 exit gate's second claim. Every file in tests/cuda_samples is
# ordinary CUDA whose only concession to GRXCP is including grx_cuda_compat.h
# instead of cuda_runtime.h -- no grx* name appears in any of them, and none
# includes a grx/device/ header, because a CUDA file does not either.
#
# The first pass over these failed eleven times out of eleven. What that found
# is in tests/cuda_samples/README.md; the rule was that the platform changed and
# the samples did not.
#
# 11_histogram_atomics is the exception and is checked the other way round: on a
# build with no A extension it MUST refuse to compile, with a message naming the
# reason, because the alternative is an AMO the simulator aborts on silently.
if [[ -n "$GRXGPU" && -d "$TOOLDIR/llvm-vortex" ]]; then
  samples_failed=0
  for src in "$ROOT"/tests/cuda_samples/*.cu; do
    name="$(basename "$src" .cu)"
    if [[ "$name" == "11_histogram_atomics" ]]; then continue; fi
    if ! "$BUILD/grxcc" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
         --build-kernel "$ROOT/ci/build_kernel.sh" -I "$ROOT/include" \
         "$src" "${OBJS[@]}" $LIBS -o "$BUILD/sample_$name" \
         > "$BUILD/sample_$name.build.log" 2>&1; then
      echo "FAIL  $name did not compile:"
      grep -E "error:" "$BUILD/sample_$name.build.log" | head -3 | sed 's/^/        /'
      samples_failed=$((samples_failed + 1))
      continue
    fi
    if out="$("$BUILD/sample_$name" 2>&1)"; then
      echo "$out" | tail -1 | sed 's/^/  /'
    else
      rc=$?
      if [[ $rc -eq 77 ]]; then
        echo "  SKIPPED $name (no device)"
      else
        echo "FAIL  $name compiled but did not pass:"
        echo "$out" | tail -3 | sed 's/^/        /'
        samples_failed=$((samples_failed + 1))
      fi
    fi
  done

  # The atomics sample, checked for a refusal that names the reason. A build
  # WITH the extension is the other case and is expected to compile and run --
  # so this reads the device rather than assuming the configuration.
  atomics_log="$BUILD/sample_11_histogram_atomics.build.log"
  if "$BUILD/grxcc" --grxgpu "$GRXGPU" --tooldir "$TOOLDIR" \
       --build-kernel "$ROOT/ci/build_kernel.sh" -I "$ROOT/include" \
       "$ROOT/tests/cuda_samples/11_histogram_atomics.cu" "${OBJS[@]}" $LIBS \
       -o "$BUILD/sample_11" > "$atomics_log" 2>&1; then
    if "$BUILD/grx-smi" 2>/dev/null | grep -q 'capabilities.*atomics'; then
      "$BUILD/sample_11" | tail -1 | sed 's/^/  /'
    else
      echo "FAIL  atomicAdd compiled on a device that reports no atomics."
      echo "      An AMO here aborts the simulator with no message; see"
      echo "      docs/designs/cuda_mapping.md section 7.16."
      samples_failed=$((samples_failed + 1))
    fi
  elif grep -q "VX_CFG_EXT_A_ENABLE off" "$atomics_log"; then
    echo "  ok    atomicAdd refused by name on a build with no A extension"
  else
    echo "FAIL  the atomics sample failed for a reason that is not the atomics:"
    grep -E "error:" "$atomics_log" | head -3 | sed 's/^/        /'
    samples_failed=$((samples_failed + 1))
  fi

  if [[ $samples_failed -ne 0 ]]; then
    echo "FAILED: $samples_failed CUDA sample(s)"
    exit 1
  fi
else
  echo "SKIPPED: needs --grxgpu <path> and a device toolchain in $TOOLDIR."
fi

echo
echo "==> conformance report against the real device"
"$BUILD/grx-conform" | tail -14

echo
echo "tier 2 passed on $DRIVER"
