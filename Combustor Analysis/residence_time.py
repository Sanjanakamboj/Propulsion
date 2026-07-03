"""Combustor residence time.

tau = V_combustor / Q_volumetric, where Q_volumetric = mdot / rho -- an
exact relationship. combustor_volume is a plain design input here (this
0D cycle model has no combustor liner geometry of its own), the same way
blade_speed_limit or T_blade_limit are design inputs elsewhere in this
project, not something derived from nothing.

Typical real combustor residence times run a few milliseconds -- that's a
rough order-of-magnitude memory, not a precise citation, so treat any
target you pick as a starting point to refine, not a hard spec.
"""


def residence_time(combustor_volume: float, mdot: float, rho: float) -> float:
    """tau = combustor_volume * rho / mdot, in seconds."""
    if combustor_volume <= 0.0:
        raise ValueError("combustor_volume must be > 0")
    if mdot <= 0.0:
        raise ValueError("mdot must be > 0")
    if rho <= 0.0:
        raise ValueError("rho must be > 0")
    return combustor_volume * rho / mdot


def required_volume(target_residence_time: float, mdot: float, rho: float) -> float:
    """Inverse of residence_time: the combustor volume needed to hit a
    target residence time at the given mass flow/density."""
    if target_residence_time <= 0.0:
        raise ValueError("target_residence_time must be > 0")
    if mdot <= 0.0:
        raise ValueError("mdot must be > 0")
    if rho <= 0.0:
        raise ValueError("rho must be > 0")
    return target_residence_time * mdot / rho
