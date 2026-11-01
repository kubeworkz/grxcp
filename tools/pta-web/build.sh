#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gridworkz
# build.sh -- the canary's pinned harness, natively and as WebAssembly.
#
# Whether a browser can run the PTA error model on the frozen canary and print
# exactly what the harness prints. This builds the
# harness pinned in docs/designs/canary/harness (the sources regenerate.sh
# checks) twice from the same files and flags -- gcc, and emcc -- and prepares
# the data both read: MNIST checked against regenerate.sh's digests, pooled to
# 8 x 8 by the harness's own `pool`.
#
# Usage: tools/pta-web/build.sh OUT_DIR      (needs gcc, emcc, curl, gzip)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
SRC="$ROOT/docs/designs/canary/harness"
OUT=$1
mkdir -p "$OUT/mnist" "$OUT/data" "$OUT/web"

# The flags pta_mnist.sh uses for SHAPE=chain, the canary's shape.
CFLAGS="-O2 -std=c99 -Wall -Wextra -pedantic -I$SRC -DN_SIDE=8 -DN_HID=64 -DN_OUT=16"

echo "== MNIST, checked against docs/designs/canary/regenerate.sh"
while read -r want name; do
  [ -s "$OUT/mnist/$name.gz" ] || curl -sSfL --retry 3 -o "$OUT/mnist/$name.gz" \
    "https://ossci-datasets.s3.amazonaws.com/mnist/$name.gz"
  got=$(gzip -dc "$OUT/mnist/$name.gz" | sha256sum | cut -c1-64)
  [ "$got" = "$want" ] || { echo "$name.gz does not hold MNIST's $name" >&2; exit 1; }
done <<'DIGESTS'
ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db train-images-idx3-ubyte
65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5 train-labels-idx1-ubyte
0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7 t10k-images-idx3-ubyte
ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2 t10k-labels-idx1-ubyte
DIGESTS

echo "== native build"
gcc --version | head -1
gcc $CFLAGS -o "$OUT/pta_mnist" "$SRC/pta_mnist.c" "$SRC/pta_tile_model.c" -lm
"$OUT/pta_mnist" selftest

echo "== data, pooled to 8 x 8 by the harness itself"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  gzip -dc "$OUT/mnist/$f.gz" > "$OUT/data/$f"
  case $f in *images*) "$OUT/pta_mnist" pool "$OUT/data/$f" "$OUT/data/$f" > /dev/null ;; esac
done
echo chain > "$OUT/data/shape"; echo as-is > "$OUT/data/pixels"

echo "== WebAssembly builds"
emcc --version | head -1
# For node: the same command line, with the real filesystem, so it reads exactly what gcc's does.
emcc $CFLAGS -sALLOW_MEMORY_GROWTH=1 -sNODERAWFS=1 -sEXIT_RUNTIME=1 \
  -o "$OUT/pta_mnist_node.js" "$SRC/pta_mnist.c" "$SRC/pta_tile_model.c" -lm
# For the browser: a module the page runs in a worker, data in its in-memory filesystem.
emcc $CFLAGS -O3 -sALLOW_MEMORY_GROWTH=1 -sMODULARIZE=1 -sEXPORT_NAME=PtaMnist \
  -sENVIRONMENT=web,worker -sINVOKE_RUN=0 -sEXIT_RUNTIME=0 \
  -sEXPORTED_RUNTIME_METHODS=callMain,FS \
  -o "$OUT/web/pta_mnist.js" "$SRC/pta_mnist.c" "$SRC/pta_tile_model.c" -lm
cp "$ROOT/docs/designs/canary/canary_seed1.net" "$OUT/web/"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  gzip -9c "$OUT/data/$f" > "$OUT/web/$f.gz"
done

echo "== sizes"
( cd "$OUT/web" && for f in *; do printf '%-34s %9d bytes   gzip %9d\n' "$f" "$(wc -c < "$f")" "$(gzip -9c "$f" | wc -c)"; done )
