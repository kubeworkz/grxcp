"""End-to-end smoke test: gdsfactory -> GDS -> KLayout round-trip.

Builds a straight waveguide on the placeholder SI_CORE layer (1, 0),
exports GDS, reloads it with KLayout, and asserts the polygon survives.
Requires gdsfactory and klayout (see requirements.txt). Exit 0 on pass.

Run from the repo root:  python photonic/smoke.py
"""

import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "build", "smoke.gds")


def main():
    import gdsfactory as gf

    try:
        gf.gpdk.PDK.activate()  # generic PDK: geometry only, no foundry data
    except AttributeError:
        pass

    try:
        wg = gf.components.straight(length=10.0, layer=(1, 0))
    except (AttributeError, TypeError):
        wg = gf.Component("smoke_straight")
        wg.add_polygon([(0, 0), (10, 0), (10, 0.5), (0, 0.5)], layer=(1, 0))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    wg.write_gds(OUT)

    import klayout.db as db

    layout = db.Layout()
    layout.read(OUT)
    layer_ix = layout.layer(1, 0)
    count = sum(shapes for shapes in
                (cell.shapes(layer_ix).size() for cell in layout.each_cell()))
    if count < 1:
        print(f"smoke: FAIL, no polygons on (1, 0) in {OUT}")
        return 1
    print(f"smoke: PASS, {count} shape(s) on (1, 0) via "
          f"gdsfactory {gf.__version__} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
