# Photonic flow skeleton (open-source, pre-PDK)

Pre-PDK layout and verification flow for the PTA first-testchip vehicle
defined in docs/designs/pta_fotonix_baseline.md §3. Runs end to end on
placeholder rules today; the foundry layer map and rule values drop in on
PDK access without restructuring.

## Toolchain

- GDSFactory: parametric cells, chip layout, GDS export (see
  requirements.txt; not vendored here).
- KLayout: layout viewing/editing and DRC execution. The rule deck in
  drc/ is a placeholder stub, not a foundry deck.
- SAX: S-parameter circuit exploration from designer-supplied models.
- Lumerical (licensed): MODE/FDTD/CHARGE/HEAT reference flows per the
  Fotonix certification; example-driven modeling only until PDK models land.

## Placeholder doctrine

Every layer number, rule value, and device dimension in pdk/ and drc/ is
a placeholder until it carries a PDK source. A "clean" run against these
placeholders means clean against placeholders only — never GF Fotonix
DRC-clean. No open-source PDK targets the GF45CLO/45SPCLO stack; the
licensed kit comes through GF Connect (account created 2026-10-16,
verification pending) or an MPW intermediary.

## Layout

- pdk/layermap.py — placeholder layer map as data, with validate().
- drc/rules.py — placeholder rule table as data, with validate().
- check.py — stdlib-only self-check runner over the above. Run:
  python photonic/check.py from the repo root.

## Status

Skeleton only. No vehicle cells yet; the first cell set (waveguide,
splitter, MZM coupon, Ge-PD coupon, coupler test structures) is the next
addition once this runner is green.
