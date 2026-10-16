"""First vehicle coupon cell set (placeholder geometry, GF-native devices).

Cells, all on the placeholder layers of pdk/layermap.py:
  wg_straight   silicon straight waveguide + taper pair
  mmi_1x2       1x2 splitter coupon (body + tapers)
  mzm_coupon    Mach-Zehnder coupon: two splitters, two arms, heater, pads
  pd_coupon     germanium photodiode coupon: Ge rect + contact pads
  coupler_array edge-coupler test row: tapers + fiber-trench marks

Dimensions are placeholders sized above the drc/rules.py minima, not
foundry device geometries. validate() checks every cell is non-empty and
every silicon feature meets si_min_width.

Run from the repo root:  python photonic/cells.py  (writes build/vehicle.gds)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdk.layermap import LAYERS
from drc.rules import RULES

SI = LAYERS["SI_CORE"]
GE = LAYERS["GE_PD"]
HEATER = LAYERS["HEATER"]
PAD = LAYERS["METAL_PAD"]
TRENCH = LAYERS["FIBER_TRENCH"]

WG_W = 0.45        # nominal waveguide width (rule min 0.12)
TAPER_L = 20.0     # edge-coupler taper length
TAPER_W0 = 0.12    # taper tip width
MMI_W = 2.4        # splitter body width
MMI_L = 12.0       # splitter body length
ARM_GAP = 3.0      # MZM arm spacing
ARM_L = 100.0      # MZM arm length
GE_W = 3.0         # photodiode width
GE_L = 10.0        # photodiode length
PAD_S = 60.0       # bond pad edge
PAD_PITCH = 100.0  # bond pad pitch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "build", "vehicle.gds")


def _activate():
    import gdsfactory as gf
    try:
        gf.gpdk.PDK.activate()
    except AttributeError:
        pass
    return gf


def _rect(comp, x0, y0, x1, y1, layer):
    comp.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer=layer)


def wg_straight(gf, length=50.0):
    c = gf.Component("wg_straight")
    _rect(c, 0, -WG_W / 2, length, WG_W / 2, SI)
    return c


def taper(gf, name="taper"):
    c = gf.Component(name)
    c.add_polygon([(0, -TAPER_W0 / 2), (TAPER_L, -WG_W / 2),
                   (TAPER_L, WG_W / 2), (0, TAPER_W0 / 2)], layer=SI)
    return c


def mmi_1x2(gf):
    c = gf.Component("mmi_1x2")
    _rect(c, 0, -MMI_W / 2, MMI_L, MMI_W / 2, SI)
    _rect(c, -10, -WG_W / 2, 0, WG_W / 2, SI)
    _rect(c, MMI_L, -WG_W / 2 - 1.0, MMI_L + 10, -1.0 + WG_W / 2, SI)
    _rect(c, MMI_L, 1.0 - WG_W / 2, MMI_L + 10, 1.0 + WG_W / 2, SI)
    return c


def _splitter_geom(c, x0):
    _rect(c, x0, -MMI_W / 2, x0 + MMI_L, MMI_W / 2, SI)
    _rect(c, x0 - 10, -WG_W / 2, x0, WG_W / 2, SI)
    _rect(c, x0 + MMI_L, -WG_W / 2 - 1.0,
          x0 + MMI_L + 10, -1.0 + WG_W / 2, SI)
    _rect(c, x0 + MMI_L, 1.0 - WG_W / 2,
          x0 + MMI_L + 10, 1.0 + WG_W / 2, SI)


def mzm_coupon(gf):
    c = gf.Component("mzm_coupon")
    total = MMI_L + ARM_L + MMI_L
    for y in (-ARM_GAP / 2 - WG_W / 2, ARM_GAP / 2 - WG_W / 2):
        _rect(c, MMI_L, y, MMI_L + ARM_L, y + WG_W, SI)
    _rect(c, MMI_L, -ARM_GAP / 2 - WG_W / 2,
          MMI_L + ARM_L, -ARM_GAP / 2 - WG_W / 2 + 2.0, HEATER)
    for x0 in (0, MMI_L + ARM_L):
        _splitter_geom(c, x0)
    for i, y in enumerate((-PAD_PITCH, PAD_PITCH)):
        _rect(c, total / 2 - PAD_S / 2 + i * 0, y - PAD_S / 2,
              total / 2 + PAD_S / 2, y + PAD_S / 2, PAD)
    return c


def pd_coupon(gf):
    c = gf.Component("pd_coupon")
    _rect(c, 0, -GE_W / 2, GE_L, GE_W / 2, GE)
    _rect(c, -5, -WG_W / 2, 0, WG_W / 2, SI)
    for y in (-PAD_PITCH / 2, PAD_PITCH / 2):
        _rect(c, GE_L / 2 - PAD_S / 2, y - PAD_S / 2,
              GE_L / 2 + PAD_S / 2, y + PAD_S / 2, PAD)
    return c


def coupler_array(gf, n=4):
    c = gf.Component("coupler_array")
    for i in range(n):
        ref = c.add_ref(taper(gf, name=f"taper_{i}"))
        ref.dymin = i * PAD_PITCH / 4
        _rect(c, -30, ref.dymin - 5, -10, ref.dymin + 5, TRENCH)
    return c


def build_top(gf):
    top = gf.Component("vehicle_coupons")
    y = 0.0
    for builder in (wg_straight, mmi_1x2, mzm_coupon, pd_coupon,
                    coupler_array):
        ref = top.add_ref(builder(gf))
        ref.dymin = y
        y = ref.dymax + 20.0
    return top


def validate(top):
    """Non-empty cells + silicon widths vs placeholder minima."""
    errors = []
    min_w = RULES["si_min_width"][0]
    for ref in top.insts:
        cell = ref.cell
        try:
            empty = cell.bbox().empty()
        except AttributeError:
            empty = cell.dbbox().empty()
        if empty:
            errors.append(f"{cell.name}: empty")
    if WG_W < min_w or TAPER_W0 < min_w:
        errors.append("waveguide width below si_min_width placeholder")
    return errors


def main():
    gf = _activate()
    top = build_top(gf)
    errors = validate(top)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    top.write_gds(OUT)
    if errors:
        print("cells: FAIL")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"cells: PASS, 5 coupons -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
