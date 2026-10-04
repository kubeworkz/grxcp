#!/usr/bin/env bash
# grx930's accuracy budget, every evaluation of it, run again through the twin
# and held to the line the harness wrote when it ran the budget itself.
#
#   pta_mnist_budget_via_twin.sh WORK TWIN_BINARY [JOBS]
#
# WORK is the harness's work directory as `pta_mnist.sh budget` leaves it:
# data/, nets/, out/budget_settings.txt and out/budget.d.  TWIN_BINARY is the
# harness built with pta_mnist_via_twin.c -- README.md has the two commands.
#
# Each network runs on its own seed, as the harness's script runs them, and a
# pass is a byte-for-byte match of what the program prints.  Anything else is
# named: a line that differs, a setting with no record to hold it to, a run
# that stopped.  The last line is the count of each.
set -u
WORK=$1
BIN=$2
JOBS=${3:-5}
OUT=$WORK/out/budget_twin.d
rm -rf "$OUT"
mkdir -p "$OUT"

one() {
    local net=$1 seed=$2
    shift 2
    local name
    name="$(basename "$net" .net)_$(echo "--seed $seed $*" | tr -c 'A-Za-z0-9.,' '_').txt"
    # shellcheck disable=SC2068
    "$BIN" eval --data "$WORK/data" --net "$WORK/nets/$net" --seed "$seed" $@ \
        > "$OUT/$name" 2> "$OUT/$name.err"
    echo "$? $name"
}
export -f one
export WORK BIN OUT

while read -r line; do
    for s in 1 2 3 4 5; do
        echo "d8_b6_s$s.net $s $line"
    done
done < "$WORK/out/budget_settings.txt" |
    xargs -P "$JOBS" -I{} bash -c 'one {}' > "$OUT/_status.txt"

ran=0; same=0; differ=0; missing=0; failed=0
while read -r rc name; do
    ran=$((ran + 1))
    if [ "$rc" != 0 ]; then
        failed=$((failed + 1)); echo "FAILED $rc $name"; continue
    fi
    if [ ! -s "$WORK/out/budget.d/$name" ]; then
        missing=$((missing + 1)); echo "NO RECORD $name"; continue
    fi
    if cmp -s "$OUT/$name" "$WORK/out/budget.d/$name"; then
        same=$((same + 1))
    else
        differ=$((differ + 1)); echo "DIFFERENT $name"
    fi
done < "$OUT/_status.txt"
gemms=$(cat "$OUT"/*.err | grep -o '^pta_mnist_via_twin: [0-9]*' | awk '{s += $2} END {print s}')
aged=$(cat "$OUT"/*.err | grep -o '[0-9]* drift steps aged' | awk '$1 > 0 {n += 1} END {print n + 0}')
echo "ran=$ran identical=$same different=$differ no_record=$missing failed=$failed gemms=$gemms aged_runs=$aged"
[ "$ran" -gt 0 ] && [ "$same" = "$ran" ]
