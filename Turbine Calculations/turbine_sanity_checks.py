"""PASS/FAIL sanity checks for turbine mean-line stage designs, in the
spirit of the design_sanity_check in the reference Turbine Stage Design
notebook -- Mach numbers, flow angles, total turning, exit swirl, and AN^2 --
but with configurable limits rather than hardcoded to one design point,
since "good" limits depend on the specific engine class. The shared
SanityCheck record type and report formatter live in Utils/sanity.py.

Named turbine_sanity_checks.py (not sanity_checks.py) because Compressor
Calculations has its own same-named module, and design_engine.py puts both
folders on sys.path simultaneously.
"""

import sys
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent.parent / "Utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from sanity import SanityCheck, check, format_sanity_report

__all__ = ["SanityCheck", "format_sanity_report", "DEFAULT_TURBINE_LIMITS", "turbine_stage_sanity_check"]

DEFAULT_TURBINE_LIMITS = dict(
    M2=(0.70, 0.85),
    beta2=(None, 47.5),
    Mw2=(None, 0.50),
    Mw3=(0.65, 0.80),
    beta3=(None, 75.0),
    delta_beta=(110.0, 120.0),
    alpha3=(None, 35.0),
    an2=(None, 3.0e7),
)


def turbine_stage_sanity_check(result, limits: dict = None):
    lim = dict(DEFAULT_TURBINE_LIMITS)
    if limits:
        lim.update(limits)

    delta_beta = result.beta2_deg + result.beta3_deg

    return [
        check("Absolute Mach M2", result.M2, lim["M2"], "-"),
        check("Relative flow angle beta2", result.beta2_deg, lim["beta2"], "deg"),
        check("Relative Mach Mw2", result.Mw2, lim["Mw2"], "-"),
        check("Relative Mach Mw3", result.Mw3, lim["Mw3"], "-"),
        check("Relative flow angle beta3", result.beta3_deg, lim["beta3"], "deg"),
        check("Total turning (beta2+beta3)", delta_beta, lim["delta_beta"], "deg"),
        check("Exit swirl alpha3", abs(result.alpha3_deg), lim["alpha3"], "deg"),
        check("AN^2", result.an2, lim["an2"], "m^2 rpm^2"),
    ]
