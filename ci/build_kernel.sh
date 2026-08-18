#!/usr/bin/env bash
# Compile a GRXCP device kernel into a .vxbin.
#
# This is what grxcc will do internally in phase 4. Having it as a script first
# means the device side is exercised and understood before a compiler driver
# tries to automate it, and it gives library kernels a way to build today.
#
#   ./ci/build_kernel.sh --grxgpu <path> --tooldir <path> \
#                        tests/kernels/vecadd/kernel.cpp -o vecadd.vxbin
#
# The device configuration comes from the INSTALLED sysroot ($VORTEX_PATH), not
# from grxgpu's VX_config.toml defaults -- see the comment at XCONFIGS below,
# and ci/README.md under "configuration provenance". --configs overrides it.
#
# The kernel compiles with VOLT (clang with +xvortex), links against the CTA
# runtime libvortex2.a, and is converted to a .vxbin by grxgpu's vxbin.py --
# which appends the VXSYMTAB footer that lets grxModuleGetFunction resolve an
# entry by name.

set -euo pipefail

GRXGPU=""
TOOLDIR="${TOOLDIR:-$HOME/tools}"
XLEN=64
OUT=""
SRC=""
CONFIGS=""
CONFIGS_SET=0
EXTRA_INCLUDES=()

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --grxgpu)  GRXGPU="$2"; shift 2 ;;
    --tooldir) TOOLDIR="$2"; shift 2 ;;
    --xlen)    XLEN="$2"; shift 2 ;;
    --configs) CONFIGS="$2"; CONFIGS_SET=1; shift 2 ;;
    -I)        EXTRA_INCLUDES+=("-I$2"); shift 2 ;;
    -o)        OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0" | sed 's|^# \{0,1\}||'; exit 0 ;;
    *)         SRC="$1"; shift ;;
  esac
done

[[ -n "$SRC" && -f "$SRC" ]] || { echo "error: no kernel source given" >&2; exit 2; }
[[ -n "$GRXGPU" && -f "$GRXGPU/VX_config.toml" ]] || {
  echo "error: --grxgpu must point at a grxgpu checkout" >&2; exit 2; }
OUT="${OUT:-${SRC%.*}.vxbin}"

GRXGPU="$(cd "$GRXGPU" && pwd)"
BUILD="$GRXGPU/build"
LLVM="$TOOLDIR/llvm-vortex"
RVT="$TOOLDIR/riscv${XLEN}-gnu-toolchain"
LIBC="$TOOLDIR/libc${XLEN}"
LIBCRT="$TOOLDIR/libcrt${XLEN}"

for d in "$LLVM" "$RVT" "$LIBC" "$LIBCRT"; do
  [[ -d "$d" ]] || { echo "error: missing $d (run ci/install_toolchain.sh)" >&2; exit 1; }
done
[[ -f "$BUILD/sw/kernel/libvortex2.a" ]] || {
  echo "error: $BUILD/sw/kernel/libvortex2.a not found." >&2
  echo "       Configure grxgpu with --tooldir=$TOOLDIR and run: make -C build/sw/kernel" >&2
  exit 1; }

# The device build must see exactly the configuration the runtime was built
# with; gen_config.py is the same generator grxgpu uses for its own kernels.
#
# "Exactly" is the whole point, and it does not happen by itself. VX_config.toml
# is the small baseline with the tensor unit off; a sysroot built with
# CONFIGS="-DVX_CFG_EXT_TCU_ENABLE ..." is a different machine, and compiling a
# kernel from the toml alone produces device code for hardware that is not the
# hardware the runtime will run it on. ci/build_sysroot.sh records the string it
# used; read it back here.
if [[ "$CONFIGS_SET" == "0" && -n "${VORTEX_PATH:-}" ]]; then
  if [[ -f "$VORTEX_PATH/share/grxcp/device_configs" ]]; then
    CONFIGS="$(cat "$VORTEX_PATH/share/grxcp/device_configs")"
  else
    echo "warning: $VORTEX_PATH has no recorded configuration." >&2
    echo "         Building from VX_config.toml defaults, which may not be the" >&2
    echo "         configuration the runtime was built with. Rebuild the sysroot" >&2
    echo "         with ci/build_sysroot.sh, or pass --configs explicitly." >&2
  fi
fi

XCONFIGS="$(cd "$GRXGPU" && python3 ci/gen_config.py \
  --config="$GRXGPU/VX_config.toml" --cflags="-DVX_CFG_XLEN=$XLEN $CONFIGS")"

# Print the parts a kernel author actually depends on. A silent build is how a
# tensor-core test ends up passing without a tensor core.
echo "==> device configuration"
for feature in EXT_TCU_ENABLED EXT_DXA_ENABLED TCU_FP16_ENABLED TCU_WGMMA_ENABLED \
               NUM_THREADS NUM_WARPS; do
  value="$(printf '%s\n' $XCONFIGS | sed -n "s/^-DVX_CFG_${feature}=//p" | tail -1)"
  printf '    %-20s %s\n' "$feature" "${value:-<unset>}"
done

if [[ "$XLEN" == "64" ]]; then
  ARCH=(-march=rv64imafd -mabi=lp64d); STARTUP_ADDR=0x180000000
else
  ARCH=(-march=rv32imaf  -mabi=ilp32f); STARTUP_ADDR=0x80000000
fi

echo "==> compiling $(basename "$SRC") with VOLT"
"$LLVM/bin/clang++" \
  --target="riscv${XLEN}-unknown-elf" \
  --sysroot="$RVT/riscv${XLEN}-unknown-elf" \
  --gcc-toolchain="$RVT" \
  -Xclang -target-feature -Xclang +xvortex \
  -Xclang -target-feature -Xclang +zicond \
  -mllvm -disable-loop-idiom-all -Wno-unused-command-line-argument \
  "${ARCH[@]}" -O3 -mcmodel=medany -fno-rtti -fno-exceptions \
  -nostartfiles -nostdlib -fdata-sections -ffunction-sections \
  -I"$ROOT/include" -I"$(dirname "$SRC")" \
  -I"$GRXGPU/sw/kernel/include" -I"$BUILD/sw" -I"$BUILD/hw" -I"$GRXGPU/sw/common" \
  "${EXTRA_INCLUDES[@]+"${EXTRA_INCLUDES[@]}"}" \
  -DNDEBUG -D__VORTEX__ -DKMU_ENABLE $XCONFIGS \
  "$SRC" \
  -Wl,-Bstatic,--gc-sections,-T,"$GRXGPU/sw/kernel/scripts/link${XLEN}.ld",--defsym=STARTUP_ADDR=$STARTUP_ADDR \
  "$BUILD/sw/kernel/libvortex2.a" \
  -L"$LIBC/lib" -lm -lc \
  "$LIBCRT/lib/baremetal/libclang_rt.builtins-riscv${XLEN}.a" \
  -o "${OUT%.vxbin}.elf"

echo "==> packaging .vxbin"
OBJCOPY="$LLVM/bin/llvm-objcopy" \
  python3 "$GRXGPU/sw/kernel/scripts/vxbin.py" "${OUT%.vxbin}.elf" "$OUT"

echo "==> entry points"
"$LLVM/bin/llvm-nm" "${OUT%.vxbin}.elf" | grep '__vx_kentry_' | sed 's/^/    /'
echo "$OUT"
