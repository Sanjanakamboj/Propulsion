"""Design-matched converging-diverging (C-D) nozzle exit-state solve.

Only relevant when the nozzle pressure ratio exceeds critical (a
converging-only nozzle would be choked and underexpanded there, see
converging.py) -- a C-D nozzle adds a diverging section sized so the flow
fully expands to ambient pressure, recovering the extra thrust an
underexpanded converging nozzle leaves on the table as pressure thrust.

This models the DESIGN-matched case (exit static pressure = ambient) via
the isentropic total-to-static relation solved directly for exit Mach
(exit_conditions.exit_mach_from_pressure_ratio -- exact, no iteration
needed) plus the standard isentropic area-Mach relation for the resulting
throat-to-exit area ratio. It does NOT model a fixed-geometry nozzle
operating off-design (over/under-expanded at a NPR other than its design
point) -- that needs the area ratio as a further input and is a real, but
separate, extension.
"""

from dataclasses import dataclass

from exit_conditions import (
    actual_exit_temperature,
    critical_pressure_ratio,
    exit_mach_from_pressure_ratio,
)


def area_ratio_from_mach(M: float, gamma: float) -> float:
    """Standard isentropic area-Mach relation A/A* (A* = throat/sonic
    reference area). Equals exactly 1.0 at M=1 by construction."""
    if M <= 0.0:
        raise ValueError("M must be > 0")
    return (1.0 / M) * ((2.0 + (gamma - 1.0) * M**2) / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


@dataclass(frozen=True)
class CDNozzleExitState:
    M_exit: float
    P_exit: float  # = P_ambient, by the design-matched assumption
    T_exit: float
    V_exit: float
    area_ratio: float  # A_exit / A_throat


def solve_cd_nozzle_design_matched(
    T0_in: float,
    P0_in: float,
    P_ambient: float,
    gamma: float,
    R: float,
    isentropic_efficiency: float = 0.98,
) -> CDNozzleExitState:
    critical = critical_pressure_ratio(gamma)
    if (P0_in / P_ambient) <= critical:
        raise ValueError(
            f"nozzle pressure ratio ({P0_in / P_ambient:.3f}) is at or below critical "
            f"({critical:.3f}) -- a converging-only nozzle already fully expands here, "
            "no diverging section is needed (see converging.py)"
        )

    M_exit = exit_mach_from_pressure_ratio(P0_in, P_ambient, gamma)
    T_exit_ideal = T0_in / (1.0 + 0.5 * (gamma - 1.0) * M_exit**2)
    T_exit = actual_exit_temperature(T0_in, T_exit_ideal, isentropic_efficiency)
    V_exit = M_exit * (gamma * R * T_exit) ** 0.5
    area_ratio = area_ratio_from_mach(M_exit, gamma)

    return CDNozzleExitState(M_exit=M_exit, P_exit=P_ambient, T_exit=T_exit, V_exit=V_exit, area_ratio=area_ratio)
