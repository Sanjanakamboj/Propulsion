"""Blade lean -- a lateral (tangential-direction) offset of the stacking
axis as a function of span, used by stacking.py. A linear lean applies a
constant lean ANGLE (offset grows linearly with span height); rotor_blade_
design.py already reports a lean_angle_deg derived from the blade height
change hub-to-tip (a byproduct of chord/stagger sizing) -- this module lets
that same angle (or any other chosen one) actually be APPLIED to the 3D
stack, rather than just reported.
"""

import math


def linear_lean_offset(span_fraction: float, span_height: float, lean_angle_deg: float) -> float:
    """Lateral offset at a given span_fraction (0=hub, 1=tip), for a
    constant lean angle applied over the full span_height."""
    if not (0.0 <= span_fraction <= 1.0):
        raise ValueError("span_fraction must be in [0, 1]")
    if span_height < 0.0:
        raise ValueError("span_height must be >= 0")
    return span_fraction * span_height * math.tan(math.radians(lean_angle_deg))


def make_linear_lean_offset(span_height: float, lean_angle_deg: float):
    """Returns a span_fraction -> offset callable, ready to pass to
    stacking.stack_sections as lean_offset."""
    return lambda span_fraction: linear_lean_offset(span_fraction, span_height, lean_angle_deg)
