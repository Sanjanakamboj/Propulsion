"""Fuel-air ratio <-> exit temperature, both directions.

Brayton Cycle Analysis's Combustor stage (stages.py) only solves one
direction -- given a target exit temperature (the design TIT), it backs
out the fuel-air ratio needed to reach it. This module makes that energy
balance a standalone, reusable pair of exact functions, and adds the
missing inverse: given a chosen fuel-air ratio, what exit temperature does
it produce. Both come from the same adiabatic energy balance:

    heat_added = cp_hot * (T_exit - T_in) = combustion_efficiency * FAR * LHV
"""


def fuel_air_ratio(T_in: float, T_exit: float, cp_hot: float, combustion_efficiency: float, fuel_lhv: float) -> float:
    """FAR needed to raise the flow from T_in to T_exit."""
    if not (0.0 < combustion_efficiency <= 1.0):
        raise ValueError("combustion_efficiency must be in (0, 1]")
    if fuel_lhv <= 0.0:
        raise ValueError("fuel_lhv must be > 0")
    if T_exit <= T_in:
        raise ValueError("T_exit must be > T_in")
    heat_added = cp_hot * (T_exit - T_in)
    return heat_added / (combustion_efficiency * fuel_lhv)


def exit_temperature(T_in: float, far: float, cp_hot: float, combustion_efficiency: float, fuel_lhv: float) -> float:
    """Inverse of fuel_air_ratio: the exit temperature a chosen FAR produces."""
    if not (0.0 < combustion_efficiency <= 1.0):
        raise ValueError("combustion_efficiency must be in (0, 1]")
    if fuel_lhv <= 0.0:
        raise ValueError("fuel_lhv must be > 0")
    if far <= 0.0:
        raise ValueError("far must be > 0")
    heat_added = combustion_efficiency * far * fuel_lhv
    return T_in + heat_added / cp_hot


def fuel_flow(mdot_air: float, far: float) -> float:
    """Fuel mass flow rate for a given air mass flow and FAR."""
    if mdot_air <= 0.0:
        raise ValueError("mdot_air must be > 0")
    if far <= 0.0:
        raise ValueError("far must be > 0")
    return mdot_air * far
