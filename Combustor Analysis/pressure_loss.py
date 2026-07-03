"""Combustor stagnation pressure loss models.

Two models, both exact once their one free parameter is known:

- Fixed fraction: P0_out = P0_in * (1 - loss_frac) -- the model
  Brayton Cycle Analysis's Combustor stage (stages.py) already uses.
  loss_frac is a lumped design choice (typically ~0.03-0.06 for a real
  combustor), not derived from flow physics.
- Dynamic-pressure loss coefficient: dP0 = K * 0.5*rho*V^2 -- the standard
  cascade/duct loss-coefficient definition, letting a loss be expressed
  relative to the flow's own dynamic head rather than a flat fraction of
  the inlet stagnation pressure.

Deliberately NOT included: a fundamental heat-addition (Rayleigh flow)
pressure-loss estimate. The low-Mach approximation for that is a real,
citable result (Lefebvre, "Gas Turbine Combustion"), but the exact
coefficient isn't something to reproduce from memory with confidence --
better to flag it as unimplemented than guess at a formula that looks
right but might carry a wrong constant.
"""


def pressure_drop_fixed_fraction(P0_in: float, loss_frac: float) -> float:
    if not (0.0 <= loss_frac < 1.0):
        raise ValueError("loss_frac must be in [0, 1)")
    return P0_in * (1.0 - loss_frac)


def required_loss_fraction(P0_in: float, P0_out: float) -> float:
    if not (0.0 < P0_out <= P0_in):
        raise ValueError("P0_out must be in (0, P0_in]")
    return 1.0 - P0_out / P0_in


def dynamic_pressure_loss(P0_in: float, rho: float, V: float, K: float) -> float:
    """P0_out for a loss coefficient K applied to the flow's dynamic
    pressure (0.5*rho*V^2)."""
    if rho <= 0.0:
        raise ValueError("rho must be > 0")
    if K < 0.0:
        raise ValueError("K must be >= 0")
    dP0 = K * 0.5 * rho * V**2
    if dP0 >= P0_in:
        raise ValueError("computed pressure loss exceeds P0_in")
    return P0_in - dP0


def required_loss_coefficient(P0_in: float, P0_out: float, rho: float, V: float) -> float:
    """Inverse of dynamic_pressure_loss: the K that produces a known
    P0_in -> P0_out drop at the given flow condition."""
    if not (0.0 < P0_out <= P0_in):
        raise ValueError("P0_out must be in (0, P0_in]")
    if rho <= 0.0 or V <= 0.0:
        raise ValueError("rho and V must be > 0")
    dP0 = P0_in - P0_out
    return dP0 / (0.5 * rho * V**2)
