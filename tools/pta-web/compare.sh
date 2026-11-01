#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gridworkz
# compare.sh -- run canary settings natively and as WebAssembly; demand the same bytes.
#
# The rows are the canary's own (pta_mnist.sh, `canary`): seed 1, draw 0, the
# 128 x 64 tile, the gate's first thousand images, the scale from a thousand
# held-out ones. A browser demo is only worth building if its numbers are the
# harness's numbers, so any difference fails this, and the table says how long
# each took both ways.
#
# Usage: tools/pta-web/compare.sh OUT_DIR    (after build.sh; needs node)
set -u
OUT=$1
NET="$(cd "$(dirname "$0")/../.." && pwd)/docs/designs/canary/canary_seed1.net"
g="--rows 128 --cols 64"
q="--abits 6 --adcbits 8"
r="$q --thermal8 0.25 --photons8 30 --prog 1 --xtalk 0.02"
t="--host 2 --probe 1 --gate 1 --gatescale 1000"
src="--src 0.01 --srcline 0.05 --srcflat 0.2 --buses 2"
fix="--levelprobe 16 --levelfix weights8"
ROWS=(
  "format_only|$g --impair quant $q --images 1000 $t"
  "as_budgeted|$g --impair quant,thermal,shot,prog,xtalk $r --images 1000 $t"
  "lit_fitted|$g --impair quant,thermal,shot,prog,xtalk $r --images 1000 $src $fix --gatefit 1 $t"
)
SUM="$OUT/summary.md"; mkdir -p "$OUT/runs"
{ echo "| row | native s | wasm (node) s | ratio | identical |"; echo "|---|---|---|---|---|"; } > "$SUM"
bad=0
for row in "${ROWS[@]}"; do
  name=${row%%|*}; args=${row#*|}
  t0=$(date +%s.%N); "$OUT/pta_mnist" eval --data "$OUT/data" --net "$NET" --seed 1 $args > "$OUT/runs/$name.native.txt" 2>&1; rn=$?
  t1=$(date +%s.%N); node "$OUT/pta_mnist_node.js" eval --data "$OUT/data" --net "$NET" --seed 1 $args > "$OUT/runs/$name.wasm.txt" 2>&1; rw=$?
  t2=$(date +%s.%N)
  dn=$(awk "BEGIN{print $t1 - $t0}"); dw=$(awk "BEGIN{print $t2 - $t1}")
  if [ $rn -eq 0 ] && [ $rw -eq 0 ] && cmp -s "$OUT/runs/$name.native.txt" "$OUT/runs/$name.wasm.txt"; then same=yes; else same="**no** (exit $rn/$rw)"; bad=1; fi
  printf '| %s | %.2f | %.2f | %.2f | %s |\n' "$name" "$dn" "$dw" "$(awk "BEGIN{print $dw / $dn}")" "$same" >> "$SUM"
  echo "== $name: native ${dn}s, wasm ${dw}s, identical: $same"
  echo "--- native output:"; head -c 1500 "$OUT/runs/$name.native.txt"; echo
  [ "$same" = yes ] || { echo "--- diff:"; diff "$OUT/runs/$name.native.txt" "$OUT/runs/$name.wasm.txt" | head -40; }
done
cat "$SUM"
exit $bad
