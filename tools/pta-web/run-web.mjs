// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Gridworkz
// run-web.mjs -- run the browser module (web/pta_mnist.js) the way a page does:
// data copied into its in-memory filesystem, callMain with the harness's own
// arguments, stdout collected. Usage: node run-web.mjs WEB_DIR DATA_DIR ARGS...
import { readFileSync, readdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join } from 'node:path';

const [webDir, dataDir, ...args] = process.argv.slice(2);
const require = createRequire(import.meta.url);
const PtaMnist = require(join(webDir, 'pta_mnist.js'));
const out = [];
const mod = await PtaMnist({
  print: (t) => out.push(t),
  printErr: (t) => out.push(t),
  locateFile: (f) => join(webDir, f),
});
mod.FS.mkdir('/data');
for (const f of readdirSync(dataDir)) mod.FS.writeFile(`/data/${f}`, readFileSync(join(dataDir, f)));
mod.FS.writeFile('/canary.net', readFileSync(join(webDir, 'canary_seed1.net')));
let code = 0;
try { code = mod.callMain(args.map((a) => a.replace('@DATA', '/data').replace('@NET', '/canary.net'))); }
catch (e) { if (e && typeof e.status === 'number') code = e.status; else throw e; }
process.stdout.write(out.join('\n') + (out.length ? '\n' : ''));
process.exit(code ?? 0);
