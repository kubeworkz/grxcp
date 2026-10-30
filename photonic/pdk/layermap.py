"""Placeholder layer map for the PTA first-testchip vehicle.

Every (layer, datatype) below is a PLACEHOLDER until replaced by the
licensed GF Fotonix PDK layer map. Nothing here may be used for signoff.
"""

PLACEHOLDER = True  # False only when every entry below cites a PDK source.

# name -> (layer, datatype). Values are arbitrary and unique by construction.
LAYERS = {
    "SI_CORE": (1, 0),      # silicon waveguide core
    "SIN_CORE": (2, 0),     # silicon-nitride waveguide core
    "GE_PD": (3, 0),        # germanium photodiode region
    "DOPING_N": (4, 0),     # n implant / modulator junction
    "DOPING_P": (5, 0),     # p implant / modulator junction
    "HEATER": (6, 0),       # thermal tuner metal (vehicle test structures)
    "METAL_PAD": (7, 0),    # electrical bond pads
    "FIBER_TRENCH": (8, 0),  # edge-coupler trench / fiber attach
    "MARKER": (9, 0),       # exclusion / region-of-interest markers
}


def validate():
    """Check the map is internally consistent. Returns a list of errors."""
    errors = []
    if PLACEHOLDER is not True:
        errors.append("PLACEHOLDER flag tampered: must stay True pre-PDK")
    seen = {}
    for name, ld in LAYERS.items():
        if (not isinstance(ld, tuple) or len(ld) != 2
                or not all(isinstance(v, int) for v in ld)):
            errors.append(f"{name}: not an (int, int) pair: {ld!r}")
            continue
        layer, datatype = ld
        if not (0 <= layer <= 255 and 0 <= datatype <= 255):
            errors.append(f"{name}: out of GDSII uint8 range: {ld!r}")
        if ld in seen:
            errors.append(f"{name}: collides with {seen[ld]} on {ld!r}")
        seen[ld] = name
    return errors
