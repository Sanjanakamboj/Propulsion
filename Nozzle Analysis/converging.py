"""Full converging-nozzle exit-state solve -- the same logic as Brayton
Cycle Analysis's Nozzle stage (stages.py), built from exit_conditions.py's
shared building blocks and extracted standalone. If the nozzle pressure
ratio exceeds critical, the nozzle chokes (M=1 at the throat, exit static
pressure stays above ambient -- underexpanded); otherwise it expands fully
to ambient pressure.
"""

from exit_conditions import (
    NozzleExitState,
    actual_exit_temperature,
    critical_pressure_ratio,
    exit_mach,
    exit_velocity_subsonic,
    is_choked,
    isentropic_exit_temperature,
    sonic_velocity,
)


def solve_converging_nozzle(
    T0_in: float,
    P0_in: float,
    P_ambient: float,
    cp: float,
    gamma: float,
    R: float,
    isentropic_efficiency: float = 0.98,
) -> NozzleExitState:
    choked = is_choked(P0_in, P_ambient, gamma)

    if choked:
        P_exit = P0_in / critical_pressure_ratio(gamma)
        T_exit_ideal = T0_in * (2.0 / (gamma + 1.0))
    else:
        P_exit = P_ambient
        T_exit_ideal = isentropic_exit_temperature(T0_in, P0_in, P_exit, gamma)

    T_exit = actual_exit_temperature(T0_in, T_exit_ideal, isentropic_efficiency)

    if choked:
        V_exit = sonic_velocity(T_exit, gamma, R)
    else:
        V_exit = exit_velocity_subsonic(T0_in, T_exit, cp)

    M_exit = exit_mach(V_exit, T_exit, gamma, R)

    return NozzleExitState(choked=choked, P_exit=P_exit, T_exit=T_exit, V_exit=V_exit, M_exit=M_exit)
