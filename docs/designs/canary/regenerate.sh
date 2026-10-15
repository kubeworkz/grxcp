#!/usr/bin/env bash
# regenerate.sh MNIST_GZ_DIR WORK_DIR
#
# The first board's canary made again from nothing but this folder and MNIST's
# four archives, and held to this folder, byte for byte: the five networks of
# its recipe trained, seed 1's the canary, and its golden tensors written, by
# the harness kept in harness/.  MNIST_GZ_DIR holds train-images-idx3-ubyte.gz,
# train-labels-idx1-ubyte.gz, t10k-images-idx3-ubyte.gz and
# t10k-labels-idx1-ubyte.gz.  WORK_DIR is a new, empty directory.  Linux or
# WSL, with bash, gcc, gzip, sha256sum and cmp; environment.lock is the
# environment it was made in.  About ten minutes.
#
# Exits 0 only if the harness and the archives are the ones pinned here and
# every tensor, network and id list comes out this folder's own.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
[ $# -eq 2 ] || { sed -n '2,15p' "$0"; exit 2; }
mnist=$(cd "$1" && pwd)
work=$2
[ -d "$work" ] && [ -z "$(ls -A "$work")" ] || { echo "regenerate: $work is not a new, empty directory" >&2; exit 2; }
work=$(cd "$work" && pwd)
fail() { echo "regenerate: $*" >&2; exit 1; }

# 1. the harness, as kept here
(cd "$here/harness" && sha256sum -c --quiet) <<'EOF' || fail "harness/ is not the harness this folder was made with"
495e370989f7275ba05fc276e8b9a894b03fa5b463274923b7ef2b564b3a728d  pta_mnist.c
66d50632de0b6f66a3fa2191f933aa59b4f46bae57ca966fd60e34ae58682a03  pta_mnist.sh
a9d008d8ee6ed367a364c22cb2f7a978dbe566b3a0fa625e563ed536add385bd  pta_tile_model.c
dd5255ccf8f6fa02f67f9f621d8efe065cfcf32b953f14aa24b288a8cb70b186  pta_tile_model.h
EOF
echo "harness: the four sources pinned here"

# 2. MNIST's archives, by what they hold
while read -r want name; do
    got=$(gzip -dc "$mnist/$name.gz" | sha256sum | cut -c1-64)
    [ "$got" = "$want" ] || fail "$name.gz does not hold MNIST's $name"
done <<'EOF'
ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db train-images-idx3-ubyte
65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5 train-labels-idx1-ubyte
0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7 t10k-images-idx3-ubyte
ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2 t10k-labels-idx1-ubyte
EOF
echo "mnist: the four archives hold MNIST's files"

# 3. what it runs on, for the record
echo "env: $(gcc --version | head -1); $(ldd --version | head -1); $(uname -srm)"

# 4. the recipe's five networks, then the canary's golden tensors
mkdir -p "$work/src" "$work/run"
cp "$here/harness/"* "$work/src/"
chmod +x "$work/src/pta_mnist.sh"
start=$(date +%s)
SHAPE=chain PIXELS=as-is "$work/src/pta_mnist.sh" "$mnist" "$work/run" canary > "$work/canary.log" 2>&1 ||
    fail "the canary mode failed: $work/canary.log"
SHAPE=chain PIXELS=as-is "$work/src/pta_mnist.sh" "$mnist" "$work/run" golden > "$work/golden.log" 2>&1 ||
    fail "the golden mode failed: $work/golden.log"
echo "ran: canary and golden, $(( $(date +%s) - start )) s"
grep '^== golden: ' "$work/golden.log"

# 5. held to this folder, byte for byte
printf 'mnist-t10k-%05d\n' $(seq 0 999) > "$work/ids.txt"
printf 'mnist-train-%05d\n' $(seq 54000 54999) > "$work/cal_ids.txt"
same=0
differ=0
check() {   # check THIS_FOLDER'S_FILE MADE_FILE
    if cmp -s "$here/$1" "$2"; then same=$((same + 1)); else differ=$((differ + 1)); echo "DIFFERS: $1"; fi
}
for f in "$here"/*.bin; do
    check "$(basename "$f")" "$work/run/out/golden.a/$(basename "$f")"
done
check canary_seed1.net "$work/run/nets/h2_d8_b6_host2_s1_n0.1_e8.net"
check start_seed1_8bit.net "$work/run/nets/h2_d8_b8_s1.net"
check ids.txt "$work/ids.txt"
check cal_ids.txt "$work/cal_ids.txt"
echo "held to this folder: $same the same, byte for byte; $differ differ"
[ "$differ" -eq 0 ] && [ "$same" -eq 23 ] || fail "the canary did not come out this folder's own"
echo "regenerate: every tensor, network and id list is this folder's own"
