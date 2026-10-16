"""Stdlib-only self-check for the photonic flow skeleton.

Validates the placeholder PDK layer map and DRC rule table for internal
consistency, and asserts the placeholder doctrine still holds (nothing may
claim foundry validity pre-PDK). Exit 0 on pass, 1 on any failure.

Run from the repo root:  python photonic/check.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdk import layermap
from drc import rules


def main():
    errors = []

    errors.extend(f"layermap: {e}" for e in layermap.validate())
    errors.extend(f"drc rules: {e}" for e in rules.validate())

    if not layermap.PLACEHOLDER or not rules.PLACEHOLDER:
        errors.append("placeholder doctrine violated: a PLACEHOLDER flag "
                      "was cleared without a licensed PDK")

    if "SI_CORE" not in layermap.LAYERS or "GE_PD" not in layermap.LAYERS:
        errors.append("layer map missing vehicle-critical layers "
                      "(SI_CORE, GE_PD)")

    if errors:
        print("photonic skeleton check: FAIL")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("photonic skeleton check: PASS "
          f"({len(layermap.LAYERS)} layers, {len(rules.RULES)} rules, "
          "all placeholders declared)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
