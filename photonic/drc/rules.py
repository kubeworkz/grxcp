"""Placeholder DRC rule table for the PTA first-testchip vehicle.

Every value below is a PLACEHOLDER until replaced by the licensed GF
Fotonix DRC deck. A clean run against these rules means clean against
placeholders only — never foundry signoff.
"""

PLACEHOLDER = True  # False only when every value below cites a PDK source.

# rule -> (value_um, description). Values are illustrative starting points.
RULES = {
    "si_min_width": (0.120, "minimum silicon waveguide width"),
    "si_min_space": (0.150, "minimum silicon-to-silicon spacing"),
    "sin_min_width": (0.300, "minimum SiN waveguide width"),
    "bend_min_radius_si": (5.0, "minimum silicon bend radius"),
    "heater_min_width": (1.000, "minimum heater metal width"),
    "pad_min_size": (50.0, "minimum bond pad edge"),
    "pad_min_pitch": (100.0, "minimum bond pad pitch"),
}


def validate():
    """Check the table is internally consistent. Returns a list of errors."""
    errors = []
    if PLACEHOLDER is not True:
        errors.append("PLACEHOLDER flag tampered: must stay True pre-PDK")
    for name, entry in RULES.items():
        if (not isinstance(entry, tuple) or len(entry) != 2
                or not isinstance(entry[0], (int, float))
                or not isinstance(entry[1], str)):
            errors.append(f"{name}: not a (number, description) pair")
            continue
        if entry[0] <= 0:
            errors.append(f"{name}: non-positive value {entry[0]!r}")
    return errors
