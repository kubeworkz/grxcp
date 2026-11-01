#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gridworkz
# compare.sh -- run canary settings natively, as the node build and as the
# browser module itself; demand the same bytes from all three.
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
HERE=$(cd "$(dirname "$0")" && pwd)
NET="$(cd "$HERE/../.." && pwd)/docs/designs/canary/canary_seed1.net"
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
{ echo "| row | native s | wasm, node build s | wasm, browser module s | identical |"; echo "|---|---|---|---|---|"; } > "$SUM"
bad=0
for row in "${ROWS[@]}"; do
  name=${row%%|*}; args=${row#*|}
  t0=$(date +%s.%N); "$OUT/pta_mnist" eval --data "$OUT/data" --net "$NET" --seed 1 $args > "$OUT/runs/$name.native.txt" 2>&1; rn=$?
  t1=$(date +%s.%N); node "$OUT/pta_mnist_node.js" eval --data "$OUT/data" --net "$NET" --seed 1 $args > "$OUT/runs/$name.wasm.txt" 2>&1; rw=$?
  t2=$(date +%s.%N); node "$HERE/run-web.mjs" "$OUT/web" "$OUT/data" eval --data @DATA --net @NET --seed 1 $args > "$OUT/runs/$name.web.txt" 2>&1; rb=$?
  t3=$(date +%s.%N)
  dn=$(awk "BEGIN{print $t1 - $t0}"); dw=$(awk "BEGIN{print $t2 - $t1}"); db=$(awk "BEGIN{print $t3 - $t2}")
  # The browser module reads /canary.net where the others read the repo's path;
  # the harness does not print the path, so the bytes must still agree.
  if [ $rn -eq 0 ] && [ $rw -eq 0 ] && [ $rb -eq 0 ] && cmp -s "$OUT/runs/$name.native.txt" "$OUT/runs/$name.wasm.txt" \
     && cmp -s "$OUT/runs/$name.native.txt" "$OUT/runs/$name.web.txt"; then same=yes; else same="**no** (exit $rn/$rw/$rb)"; bad=1; fi
  printf '| %s | %.2f | %.2f | %.2f | %s |\n' "$name" "$dn" "$dw" "$db" "$same" >> "$SUM"
  echo "== $name: native ${dn}s, node build ${dw}s, browser module ${db}s, identical: $same"
  echo "--- native output:"; head -c 1500 "$OUT/runs/$name.native.txt"; echo
  [ "$same" = yes ] || { echo "--- diff, node build:"; diff "$OUT/runs/$name.native.txt" "$OUT/runs/$name.wasm.txt" | head -20;
                          echo "--- diff, browser module:"; diff "$OUT/runs/$name.native.txt" "$OUT/runs/$name.web.txt" | head -20; }
done
cat "$SUM"

# rows.json: each row's arguments and the line the harness printed for it, so a
# page can offer the canary's own settings and show their published result.
python3 - "$OUT" "${ROWS[@]}" <<'PY'
import json, sys
out, rows = sys.argv[1], sys.argv[2:]
doc = []
for r in rows:
    name, args = r.split('|', 1)
    line = open(f'{out}/runs/{name}.native.txt').read().strip()
    doc.append(dict(name=name, args=args.split(), output=line))
json.dump(doc, open(f'{out}/web/rows.json', 'w'), indent=2)
PY
exit $bad
