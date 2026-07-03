"""Blade sweep -- an axial-direction offset of the stacking axis as a
function of span, used by stacking.py. Same linear-angle treatment as
lean.py, just applied to the axial rather than tangential direction.
"""

import math


def linear_sweep_offset(span_fraction: float, span_height: float, sweep_angle_deg: float) -> float:
    """Axial offset at a given span_fraction (0=hub, 1=tip), for a constant
    sweep angle applied over the full span_height."""
    if not (0.0 <= span_fraction <= 1.0):
        raise ValueError("span_fraction must be in [0, 1]")
    if span_height < 0.0:
        raise ValueError("span_height must be >= 0")
    return span_fraction * span_height * math.tan(math.radians(sweep_angle_deg))


def make_linear_sweep_offset(span_height: float, sweep_angle_deg: float):
    """Returns a span_fraction -> offset callable, ready to pass to
    stacking.stack_sections as sweep_offset."""
    return lambda span_fraction: linear_sweep_offset(span_fraction, span_height, sweep_angle_deg)
