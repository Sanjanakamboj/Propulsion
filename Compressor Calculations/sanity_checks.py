"""PASS/FAIL sanity checks for compressor mean-line stage designs, in the
spirit of the design_sanity_check in the reference Turbine Stage Design
notebook -- de Haller number, Mach numbers, hub-to-tip ratio, and AN^2 --
but with configurable limits rather than hardcoded to one design point,
since "good" limits depend on the specific engine class. The shared
SanityCheck record type and report formatter live in Utils/sanity.py.
"""

import sys
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent.parent / "Utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from sanity import SanityCheck, check, format_sanity_report

__all__ = ["SanityCheck", "format_sanity_report", "DEFAULT_COMPRESSOR_LIMITS", "compressor_stage_sanity_check"]

DEFAULT_COMPRESSOR_LIMITS = dict(
    de_haller=(0.72, None),  # (low, high); de Haller should not go below 0.72
    Mw1=(None, 0.85),
    M1=(None, 0.85),
    hub_to_tip_ratio=(0.4, None),
    an2=(None, 3.0e7),
)


def compressor_stage_sanity_check(result, limits: dict = None):
    lim = dict(DEFAULT_COMPRESSOR_LIMITS)
    if limits:
        lim.update(limits)

    hub_to_tip = (result.annulus_1.hub_to_tip_ratio + result.annulus_2.hub_to_tip_ratio) / 2.0

    return [
        check("de Haller (W2/W1)", result.de_haller, lim["de_haller"], "-"),
        check("Relative Mach Mw1", result.Mw1, lim["Mw1"], "-"),
        check("Absolute Mach M1", result.M1, lim["M1"], "-"),
        check("Hub-to-tip ratio", hub_to_tip, lim["hub_to_tip_ratio"], "-"),
        check("AN^2", result.an2, lim["an2"], "m^2 rpm^2"),
    ]
