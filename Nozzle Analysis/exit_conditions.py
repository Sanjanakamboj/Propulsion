"""Shared isentropic-flow building blocks used by both converging.py and
converging_diverging.py -- exact compressible-flow relations, not fitted
correlations.
"""

import math
from dataclasses import dataclass


def critical_pressure_ratio(gamma: float) -> float:
    """P0/P_throat at which a converging nozzle chokes (M=1 at the throat)."""
    return ((gamma + 1.0) / 2.0) ** (gamma / (gamma - 1.0))


def is_choked(P0_in: float, P_ambient: float, gamma: float) -> bool:
    return (P0_in / P_ambient) > critical_pressure_ratio(gamma)


def isentropic_exit_temperature(T0_in: float, P0_in: float, P_exit: float, gamma: float) -> float:
    if not (0.0 < P_exit <= P0_in):
        raise ValueError("P_exit must be in (0, P0_in]")
    return T0_in * (P_exit / P0_in) ** ((gamma - 1.0) / gamma)


def actual_exit_temperature(T0_in: float, T_exit_ideal: float, isentropic_efficiency: float) -> float:
    if not (0.0 < isentropic_efficiency <= 1.0):
        raise ValueError("isentropic_efficiency must be in (0, 1]")
    return T0_in - isentropic_efficiency * (T0_in - T_exit_ideal)


def exit_velocity_subsonic(T0_in: float, T_exit: float, cp: float) -> float:
    return math.sqrt(max(0.0, 2.0 * cp * (T0_in - T_exit)))


def sonic_velocity(T: float, gamma: float, R: float) -> float:
    if T <= 0.0:
        raise ValueError("T must be > 0")
    return math.sqrt(gamma * R * T)


def exit_mach(V: float, T_exit: float, gamma: float, R: float) -> float:
    return V / sonic_velocity(T_exit, gamma, R)


def exit_mach_from_pressure_ratio(P0_in: float, P_exit: float, gamma: float) -> float:
    """Direct (no iteration needed) solve of the isentropic total-to-static
    relation P0/P = (1 + (gamma-1)/2 * M^2)^(gamma/(gamma-1)) for M -- valid
    for both subsonic and supersonic M, since it's just algebraic inversion
    of a monotonic relation."""
    if not (0.0 < P_exit <= P0_in):
        raise ValueError("P_exit must be in (0, P0_in]")
    ratio = P0_in / P_exit
    return math.sqrt((2.0 / (gamma - 1.0)) * (ratio ** ((gamma - 1.0) / gamma) - 1.0))


@dataclass(frozen=True)
class NozzleExitState:
    choked: bool
    P_exit: float
    T_exit: float
    V_exit: float
    M_exit: float
