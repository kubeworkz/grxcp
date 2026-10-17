# Handoff to the grx930 team: PTA register-map items needing the tile owner

From the grxcp Fotonix baseline (docs/designs/pta_fotonix_baseline.md
§7, v0.3). Three items, each with its acceptance check. Nothing here
changes tile behavior — it wires an already-frozen contract into the
twin and the CSR block.

## 1. Is PTA_CAPS0.DIN_W the datapath width? (closes the DIN_W gap)

PTA_CAL_CFG's probe amplitude is bounded by the datapath width, which
today's map expects firmware to discover by search (observable via
CAL_ERR, cleared by MODEL_RST). If CAPS0's DIN_W field is that width,
reply so; firmware then computes the legal amplitude range from CAPS0
and the search is retired with zero map change. If it is not (quantiser
width vs datapath width), say so and the fallback lands instead:
width-relative amplitude encoding (DIN_W - 2 - n), computed by the
tile, no new register. Either answer retires the search; a non-answer
keeps it.

Acceptance: firmware sets amplitude from a register read on first try,
no CAL_ERR, on both the bench tile and the SoC tile.

## 2. Wire PTA_ID and PTA_CAPS2 (baseline v0.3 values)

- PTA_ID (0x000) reset: 0x50540100 — magic 0x5054, map version 0x01.
- PTA_CAPS2 (0x00C): bit 31 TWIN (twin = 1), bit 30 calibration engine
  present, bit 29 activation stage present, bits [15:0] shot rate in
  MHz. The twin sets TWIN; grx-smi checks the ID word.

Acceptance: twin reports TWIN with the modelled rate; grx-smi's ID
check passes against both the twin and the CSR block.

## 3. Pin the harness commit behind gemm_seed()

The baseline's seed derivation (splitmix64 over (base<<32)|index) was
read from c930/sim/pta_mnist.c on branch npu/shim-tile-model, which
has uncommitted local mods. Pin the commit hash whose text is
normative, so the EIC seed block has a fixed target to diff against.

Acceptance: a commit hash recorded in the baseline's provenance flag;
baseline updated to v0.4 citing it.
