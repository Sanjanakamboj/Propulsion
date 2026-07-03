"""Free-vortex radial variation of the velocity triangle -- the physics
behind blade twist. Standard axial turbomachinery design method (Dixon &
Hall; any axial compressor/turbine text): for constant axial velocity
across span, radial equilibrium is satisfied by conserving angular
momentum at each axial station independently:

    r * Vt(r) = const          (free vortex)
    U(r) = Omega * r            (rigid-body blade speed)

so Vt falls off with radius while U rises with it -- the relative flow
angle beta(r) = atan2(Wt(r), Vx) therefore varies from hub to tip, which
IS blade twist.

This project's stage solvers use two different sign conventions for
Wt = Vt -+ U depending on station (see turbine.py's docstring) -- rather
than have the caller track which applies, _infer_relative_sign figures it
out from the already-known mean-line beta/Vt/U, then applies the same
convention at every other radius.
"""

import math
from dataclasses import dataclass


def free_vortex_tangential_velocity(Vt_mean: float, mean_radius: float, r: float) -> float:
    if r <= 0.0 or mean_radius <= 0.0:
        raise ValueError("r and mean_radius must be > 0")
    return Vt_mean * mean_radius / r


def blade_speed_at_radius(U_mean: float, mean_radius: float, r: float) -> float:
    if mean_radius <= 0.0:
        raise ValueError("mean_radius must be > 0")
    return U_mean * r / mean_radius


def infer_relative_sign(Vt_mean: float, U_mean: float, Vx: float, beta_mean_deg: float) -> float:
    """Returns -1.0 if this station's beta implies Wt = Vt - U, or +1.0 if
    it implies Wt = Vt + U -- whichever matches Vx*tan(beta_mean) more
    closely."""
    Wt_expected = Vx * math.tan(math.radians(beta_mean_deg))
    if abs((Vt_mean - U_mean) - Wt_expected) <= abs((Vt_mean + U_mean) - Wt_expected):
        return -1.0
    return 1.0


@dataclass(frozen=True)
class TwistedSection:
    radius: float
    span_fraction: float  # 0 at hub, 1 at tip
    beta_in_deg: float
    beta_out_deg: float
    U: float


def local_flow_angle_deg(Vt_mean: float, U_mean: float, Vx: float, beta_mean_deg: float, mean_radius: float, r: float) -> float:
    sign = infer_relative_sign(Vt_mean, U_mean, Vx, beta_mean_deg)
    Vt_r = free_vortex_tangential_velocity(Vt_mean, mean_radius, r)
    U_r = blade_speed_at_radius(U_mean, mean_radius, r)
    Wt_r = Vt_r + sign * U_r
    return math.degrees(math.atan2(Wt_r, Vx))


def twisted_section_at_radius(
    Vt_in_mean: float, Vt_out_mean: float, U_mean: float, Vx: float,
    beta_in_mean_deg: float, beta_out_mean_deg: float,
    mean_radius: float, hub_radius: float, tip_radius: float, r: float,
) -> TwistedSection:
    if not (hub_radius <= r <= tip_radius):
        raise ValueError(f"r ({r}) must be within [hub_radius, tip_radius] ([{hub_radius}, {tip_radius}])")
    span_fraction = (r - hub_radius) / (tip_radius - hub_radius) if tip_radius > hub_radius else 0.0
    return TwistedSection(
        radius=r,
        span_fraction=span_fraction,
        beta_in_deg=local_flow_angle_deg(Vt_in_mean, U_mean, Vx, beta_in_mean_deg, mean_radius, r),
        beta_out_deg=local_flow_angle_deg(Vt_out_mean, U_mean, Vx, beta_out_mean_deg, mean_radius, r),
        U=blade_speed_at_radius(U_mean, mean_radius, r),
    )
