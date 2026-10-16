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

Toolchain verified 2026-10-16: klayout 0.30.12, gdsfactory 9.51.0,
sax 0.18.2 installed; smoke.py passes (gdsfactory -> GDS -> KLayout
round-trip on placeholder layer (1, 0)). First vehicle cell set is in
cells.py (waveguide, 1x2 splitter, MZM, Ge-PD, coupler array) with
widths checked against the placeholder minima; vehicle.gds reloads all
5 coupons in KLayout.
