#!/usr/bin/env bash
# grx930's accuracy budget on the chiplet's tile, run again through the twin
# and held to the line the harness wrote when it ran the same thing itself.
#
#   pta_mnist_geometry_via_twin.sh WORK TWIN_BINARY [JOBS] [TILE ...]
#
# WORK is the harness's work directory as `pta_mnist.sh geometry` leaves it:
# data/, nets/, out/geometry_settings.txt, out/geometry_tiles.txt and
# out/geometry.d.  TWIN_BINARY is the harness built with pta_mnist_via_twin.c --
# README.md has the two commands.  TILE is a label from geometry_tiles.txt,
# "256x64" unless told otherwise; "all" is every tile in that file.
#
# A tile that is not the core's gets a layer as one GEMM, so here the twin is
# given what a chiplet would be: one command a layer.
#
# The settings that calibrate are left out, and counted: the harness calibrates
# through a host's trim write and the map has no register for one, so the twin
# cannot run them (README.md).  Each network runs on its own seed, and a pass is
# a byte-for-byte match of what the program prints.
set -u
WORK=$1
BIN=$2
JOBS=${3:-5}
shift 3 2>/dev/null || shift $#
TILES=${*:-256x64}
OUT=$WORK/out/geometry_twin.d
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

skipped=0
while IFS='|' read -r label gopt; do
    case " $TILES " in *" all "*|*" $label "*) ;; *) continue ;; esac
    while IFS='|' read -r name setting; do
        case "$setting" in *--calibrate*) skipped=$((skipped + 5)); continue ;; esac
        for s in 1 2 3 4 5; do
            # shellcheck disable=SC2086
            set -- $gopt $setting
            echo "d8_b6_s$s.net $s $*"
        done
    done < "$WORK/out/geometry_settings.txt"
done < "$WORK/out/geometry_tiles.txt" > "$OUT/_runs.txt"
xargs -P "$JOBS" -I{} bash -c 'one {}' < "$OUT/_runs.txt" > "$OUT/_status.txt"

ran=0; same=0; differ=0; missing=0; failed=0
while read -r rc name; do
    ran=$((ran + 1))
    if [ "$rc" != 0 ]; then
        failed=$((failed + 1)); echo "FAILED $rc $name"; continue
    fi
    if [ ! -s "$WORK/out/geometry.d/$name" ]; then
        missing=$((missing + 1)); echo "NO RECORD $name"; continue
    fi
    if cmp -s "$OUT/$name" "$WORK/out/geometry.d/$name"; then
        same=$((same + 1))
    else
        differ=$((differ + 1)); echo "DIFFERENT $name"
    fi
done < "$OUT/_status.txt"
gemms=$(cat "$OUT"/*.err | grep -o '^pta_mnist_via_twin: [0-9]*' | awk '{s += $2} END {print s + 0}')
echo "tiles=\"$TILES\" ran=$ran identical=$same different=$differ no_record=$missing failed=$failed" \
     "gemms=$gemms left_out_for_calibrating=$skipped"
[ "$ran" -gt 0 ] && [ "$same" = "$ran" ]
