# The launch preamble scales with the grid — and what it IS is still open

The block bench reports a per-launch **preamble** — the interval between the
launch (which zeroes MCYCLE) and the first warp reaching its cycle probe. At
S=16 it measured 9418 cycles per launch, 39.7% of a transformer block, and
looked near-constant across twelve different kernels. That constancy was a
coincidence of the shapes those kernels launch. **It is not a fixed cost.**

## What it actually is

A kernel that does nothing but timestamp itself, launched at varying grid
sizes. Every block records its own entry time, so the question "does work
overlap dispatch" is answerable rather than assumed.

rtlsim, 4 cores, 16 threads per block:

| blocks | first entry | last entry | spread |
|---|---|---|---|
| 1 | 2910 | 2910 | 0 |
| 2 | 4462 | 4480 | 18 |
| 4 | 7198 | 13453 | 6255 |
| 8 | 21819 | 29328 | 7509 |
| 16 | 45221 | 57615 | 12394 |
| 32 | 68872 | 90206 | 21334 |
| 64 | 100774 | 124372 | 23598 |

**The EARLIEST block's entry time grows with the grid.** Not just the last.

**Do not call this hardware CTA dispatch.** The first version of this file did,
and that was a label attached to a measurement rather than a finding. Block
distribution is a SOFTWARE loop in grxgpu's CTA runtime
(`sw/kernel/src/vx_spawn.c`, `process_thread_groups`): each warp group computes
`start_group`/`group_stride` and iterates `callback(arg)` over its blocks in
sequence. A warp running four blocks back to back produces later entry times
with no hardware dispatcher involved. What that does NOT explain is why the
*earliest* entry moves, which is the part worth asking grxgpu about.

## What it is not

Three candidates were on the table and the measurement kills two:

| candidate | measured |
|---|---|
| the instrument — `vx_rdcycle_sync`'s fence at entry | **27 cycles.** Not it. |
| the first memory access, cold dcache | **18 cycles.** Not it. |
| fixed device bring-up | **1828 cycles at one block** — real, but a floor, not the story |

The kernel takes an unserialized `csrr` as its very first instruction, so the
first number has nothing of ours in front of it.

## The two backends disagree, which is a fidelity-contract row

simx, same experiment:

| blocks | first entry | last entry |
|---|---|---|
| 1 | 3030 | 3030 |
| 4 | 6863 | 7672 |
| 16 | 26408 | 29406 |
| 32 | **26408** | 33157 |
| 64 | **26416** | 34756 |

simx **plateaus at 16 blocks** — which is exactly residency here, 4 cores x 16
warps = 64 warps, and 16 blocks x 4 warps = 64. Past that, blocks queue behind
retiring ones and the first entry stops moving. That is the behaviour you would
want.

rtlsim does not plateau: 45221, 68872, 100774. Either its dispatcher genuinely
serialises the whole grid, or the shim does something simx does not. Unresolved,
and worth resolving before any grid-sizing decision is made on simx numbers.

## Why it matters more than the number it replaces

**It reframes the 51%.** `developer_interface.md` section 3 prices per-launch
fixed cost at 51.4% of a block from a 9418-cycle preamble. That figure is real
for the shapes measured and is *not a constant* — it is roughly the dispatch
cost of the grids those kernels happen to launch. At production sequence
lengths the grids are far larger and this term grows with them.

**It is a multi-SM scaling wall.** A 128-SM part wants thousands of CTAs in
flight. At ~1500 cycles per CTA, serial, before any work begins, a
thousand-CTA launch spends over a million cycles dispatching. That is the
mechanism behind phase 8's result that a 1.97x span speedup is 1.27x end to
end, and behind the per-launch preamble *growing* with core count (9418 →
10899 → 11383 at 1, 2 and 4 SMs): more cores, bigger grids, more dispatch.

**And it narrows what fusion buys.** Fusing two kernels removes one dispatch of
that grid, which is the full per-launch cost — so fusion still pays. But it does
not reduce the CTA count of the work that remains, so it cannot address the term
that grows with the machine. Something has to make dispatch cheaper or overlap
it with execution.

## Reproducing

```
./ci/build_kernel.sh --grxgpu <grxgpu> tests/repro/launch_preamble/kernel.cpp \
    -o build-real/preamble.vxbin
# host: launch at 1,2,4,...,64 blocks; read each block's own entry timestamp
```

The kernel writes `{t_raw, t_sync, t_arg, core+1}` per block. `t_raw` is a plain
`csrr` on the first instruction; `t_sync` the serialized read the real probe
uses; `t_arg` after one load from the argument blob.
