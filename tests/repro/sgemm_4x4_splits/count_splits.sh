#!/usr/bin/env bash
# Count vx_split / vx_join per kernel in a device ELF.
#
# vx_split: opcode 0x0b (CUSTOM0), funct3 = 2   (vx_intrinsics.h)
# vx_join : opcode 0x0b (CUSTOM0), funct3 = 3
#
# A kernel with divergent branches and zero splits is the 7.27 signature.
set -euo pipefail
ELF="${1:?usage: count_splits.sh <device.elf> [objdump] [nm]}"
OBJD="${2:-${TOOLDIR:-$HOME/tools}/riscv64-gnu-toolchain/bin/riscv64-unknown-elf-objdump}"
NM="${3:-${TOOLDIR:-$HOME/tools}/riscv64-gnu-toolchain/bin/riscv64-unknown-elf-nm}"

mapfile -t SYMS < <("$NM" -n "$ELF" | grep " T __vx_kentry_" | awk '{print $1" "$3}')
for i in "${!SYMS[@]}"; do
  addr="${SYMS[$i]%% *}"; name="${SYMS[$i]##* }"
  if [ $((i+1)) -lt ${#SYMS[@]} ]; then next="${SYMS[$((i+1))]%% *}"
  else next=$(printf '%x' $((0x$addr + 0x100000))); fi
  printf '%-32s ' "${name#__vx_kentry_}"
  "$OBJD" -d --start-address="0x$addr" --stop-address="0x$next" "$ELF" 2>/dev/null \
    | grep -oE '^[[:space:]]+[0-9a-f]+:[[:space:]]+[0-9a-f]{8}' | awk '{print $2}' \
    | python3 -c "
import sys
sp=jn=0
for l in sys.stdin:
    w=int(l.strip(),16)
    if (w & 0x7f) != 0x0b: continue
    f3=(w>>12)&7
    if   f3==2: sp+=1
    elif f3==3: jn+=1
print('vx_split=%-5d vx_join=%-5d %s' % (sp, jn, '*** NO RECONVERGENCE ***' if sp==0 else ''))
"
done
