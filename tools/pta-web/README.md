# tools/pta-web

The canary's PTA harness (`docs/designs/canary/harness`) built as a WebAssembly
module a browser can run, and checked against the native harness byte for byte.

- `build.sh OUT` builds the pinned harness with gcc and with emcc (a node build,
  and the browser module), checks MNIST against `regenerate.sh`'s digests, and
  pools it to 8 x 8 with the harness's own `pool`.
- `compare.sh OUT` runs the canary's rows (seed 1, 128 x 64 tile, 1,000 images:
  format only; as budgeted; as budgeted, lit, fitted) natively, as the node
  build and as the browser module (`run-web.mjs`), and fails unless all three
  print the same bytes. It writes `rows.json` and a timing table.
- `run-web.mjs` runs the browser module the way a page does: data in its
  in-memory filesystem, `callMain` with the harness's arguments.

`.github/workflows/pta-web.yml` runs both on changes to the canary or these
tools, and on `main` publishes a passing build to the `pta-web` branch, with a
`manifest.json` of digests. A build that differs from the harness is never
published.

The module takes the harness's own command line. `eval --data /data --net
/canary.net` plus the flags in `rows.json` reproduces those rows; the same
flags at other values (`--abits`, `--adcbits`, `--thermal8`, `--photons8`,
`--prog`, `--xtalk`, `--src`...) run the canary at other settings.
