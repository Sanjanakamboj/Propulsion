"""Thrust breakdown -- the same bookkeeping engine.py (Brayton Cycle
Analysis) already does inline, extracted standalone: momentum thrust +
pressure thrust (nonzero only when the nozzle is underexpanded) = gross
thrust; gross thrust minus ram drag = net thrust; net thrust normalized by
air mass flow = specific thrust; fuel flow over net thrust = TSFC.
"""

from dataclasses import dataclass


def nozzle_exit_area(mdot_gas: float, rho_exit: float, V_exit: float) -> float:
    if rho_exit <= 0.0 or V_exit <= 0.0:
        raise ValueError("rho_exit and V_exit must be > 0")
    return mdot_gas / (rho_exit * V_exit)


def momentum_thrust(mdot_gas: float, V_exit: float) -> float:
    return mdot_gas * V_exit


def pressure_thrust(P_exit: float, P_ambient: float, area_exit: float) -> float:
    """Nonzero only for an underexpanded nozzle (P_exit > P_ambient) --
    zero for a fully-expanded (design-matched C-D, or unchoked converging)
    exit where P_exit == P_ambient."""
    return (P_exit - P_ambient) * area_exit


def gross_thrust(mom_thrust: float, press_thrust: float) -> float:
    return mom_thrust + press_thrust


def ram_drag(mdot_air: float, V0: float) -> float:
    if mdot_air <= 0.0:
        raise ValueError("mdot_air must be > 0")
    return mdot_air * V0


def net_thrust(gross: float, drag: float) -> float:
    return gross - drag


def specific_thrust(net: float, mdot_air: float) -> float:
    if mdot_air <= 0.0:
        raise ValueError("mdot_air must be > 0")
    return net / mdot_air


def tsfc(fuel_flow: float, net: float) -> float:
    if net <= 0.0:
        raise ValueError("net thrust must be > 0")
    return fuel_flow / net


@dataclass(frozen=True)
class ThrustBreakdown:
    momentum_thrust: float
    pressure_thrust: float
    gross_thrust: float
    ram_drag: float
    net_thrust: float
    specific_thrust: float
    tsfc: float


def compute_thrust(
    mdot_gas: float,
    V_exit: float,
    P_exit: float,
    P_ambient: float,
    rho_exit: float,
    mdot_air: float,
    V0: float,
    fuel_flow: float,
) -> ThrustBreakdown:
    """Convenience wrapper computing every quantity above from a nozzle
    exit state (mdot_gas/V_exit/P_exit/rho_exit -- from converging.py or
    converging_diverging.py) plus flight/fuel conditions."""
    area_exit = nozzle_exit_area(mdot_gas, rho_exit, V_exit)
    mom = momentum_thrust(mdot_gas, V_exit)
    press = pressure_thrust(P_exit, P_ambient, area_exit)
    gross = gross_thrust(mom, press)
    drag = ram_drag(mdot_air, V0)
    net = net_thrust(gross, drag)
    return ThrustBreakdown(
        momentum_thrust=mom,
        pressure_thrust=press,
        gross_thrust=gross,
        ram_drag=drag,
        net_thrust=net,
        specific_thrust=specific_thrust(net, mdot_air),
        tsfc=tsfc(fuel_flow, net),
    )
