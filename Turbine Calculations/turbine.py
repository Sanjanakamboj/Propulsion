"""Axial turbine stage mean-line design.

Methodology adapted from a validated single-stage HP turbine mean-line
notebook (stator -> rotor, pressure-based degree of reaction, iterative exit
static pressure solve to hit a required specific work). Given a stage inlet
stagnation state and the specific work that stage must extract (from the
0D cycle in Brayton Cycle Analysis), this solves the velocity triangles at
the stator exit / rotor inlet (station 2) and rotor exit (station 3), then
sizes the annulus.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent.parent / "Utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from mean_line_common import AnnulusGeometry, an2, annulus_from_mass_flow, blade_speed_from_loading, mean_radius

try:
    from scipy.optimize import brentq
except ImportError as exc:  # pragma: no cover
    raise ImportError("turbine.py requires scipy (pip install scipy)") from exc


@dataclass(frozen=True)
class TurbineStageDesignInputs:
    stage_loading: float  # target psi = specific_work / U^2
    flow_coefficient: float  # phi = Vx / U
    degree_of_reaction: float  # pressure-based DOR: P2 = DOR*(P1 - P3) + P3
    blade_speed_limit: float  # m/s, mechanical limit at mean radius
    rotational_speed_rpm: float
    stator_efficiency: float = 0.90
    rotor_efficiency: float = 0.90
    inlet_mach_number: float = 0.15  # combustor-exit Mach feeding the stage
    inlet_flow_angle_deg: float = 0.0  # axial inlet swirl angle

    def __post_init__(self):
        if self.stage_loading <= 0.0:
            raise ValueError("stage_loading must be > 0")
        if self.flow_coefficient <= 0.0:
            raise ValueError("flow_coefficient must be > 0")
        if not (0.0 <= self.degree_of_reaction < 1.0):
            raise ValueError("degree_of_reaction must be in [0, 1)")
        if self.blade_speed_limit <= 0.0:
            raise ValueError("blade_speed_limit must be > 0")
        if self.rotational_speed_rpm <= 0.0:
            raise ValueError("rotational_speed_rpm must be > 0")
        for name in ("stator_efficiency", "rotor_efficiency"):
            value = getattr(self, name)
            if not (0.0 < value <= 1.0):
                raise ValueError(f"{name} must be in (0, 1], got {value}")
        if not (0.0 < self.inlet_mach_number < 1.0):
            raise ValueError("inlet_mach_number must be in (0, 1)")


@dataclass(frozen=True)
class TurbineStageResult:
    U: float
    achieved_stage_loading: float
    flow_coefficient: float
    degree_of_reaction: float
    Vx: float
    T1: float
    P1: float
    V1: float
    T2: float
    T2s: float  # static isentropic stator-exit temperature (h-s diagram ideal marker)
    T02_rel: float  # rotor-relative stagnation temperature at station 2 (h-s ladder)
    P2: float
    T02: float
    P02: float
    V2: float
    alpha2_deg: float
    M2: float
    Vt2: float
    W2: float
    beta2_deg: float
    Mw2: float
    Wt2: float
    T3: float
    T3s: float  # static isentropic rotor-exit temperature (h-s diagram ideal marker)
    P3: float
    T03: float
    P03: float
    pressure_ratio: float  # P01 / P03 (expansion ratio, > 1)
    V3: float
    alpha3_deg: float
    M3: float
    Vt3: float
    W3: float
    beta3_deg: float
    Mw3: float
    Wt3: float
    specific_work: float  # J/kg, actual (should match the requested value)
    annulus_2: AnnulusGeometry
    annulus_3: AnnulusGeometry
    an2: float


def _stage_state(P3, P1, T01, P01, Vx, U, degree_of_reaction, stator_efficiency, rotor_efficiency, cp, gamma, R):
    P2 = degree_of_reaction * (P1 - P3) + P3

    V2s = math.sqrt(2.0 * cp * T01 * (1.0 - (P2 / P1) ** ((gamma - 1.0) / gamma)))
    V2 = math.sqrt(stator_efficiency) * V2s
    T02 = T01  # adiabatic stator, no work done
    T2 = T02 - V2**2 / (2.0 * cp)
    T2s = T02 - V2s**2 / (2.0 * cp)  # static isentropic stator-exit temperature, for h-s diagrams

    rho2 = P2 / (R * T2)
    a2 = math.sqrt(gamma * R * T2)
    M2 = V2 / a2

    alpha2 = math.degrees(math.acos(min(max(Vx / V2, -1.0), 1.0)))
    Vt2 = math.sqrt(max(V2**2 - Vx**2, 0.0))
    Wt2 = Vt2 - U
    beta2 = math.degrees(math.atan2(Wt2, Vx))
    W2 = math.sqrt(Wt2**2 + Vx**2)
    Mw2 = W2 / a2
    T02_rel = T2 + W2**2 / (2.0 * cp)
    P02_rel = P2 * (T02_rel / T2) ** (gamma / (gamma - 1.0))

    W3s = math.sqrt(2.0 * cp * T02_rel * (1.0 - (P3 / P02_rel) ** ((gamma - 1.0) / gamma)))
    W3 = math.sqrt(rotor_efficiency) * W3s
    T3 = T02_rel - W3**2 / (2.0 * cp)
    T3s = T02_rel - W3s**2 / (2.0 * cp)  # static isentropic rotor-exit temperature, for h-s diagrams
    rho3 = P3 / (R * T3)
    a3 = math.sqrt(gamma * R * T3)
    Mw3 = W3 / a3

    Vx3 = Vx
    beta3 = math.degrees(math.acos(min(max(Vx3 / W3, -1.0), 1.0)))
    Wt3 = W3 * math.sin(math.radians(beta3))
    Vt3 = Wt3 - U
    alpha3 = math.degrees(math.atan2(Vt3, Vx3))
    V3 = math.sqrt(Vt3**2 + Vx3**2)
    T03 = T3 + V3**2 / (2.0 * cp)
    P03 = P3 * (T03 / T3) ** (gamma / (gamma - 1.0))

    specific_work = cp * (T01 - T03)

    state = dict(
        P2=P2, V2=V2, T02=T02, T2=T2, T2s=T2s, T02_rel=T02_rel, rho2=rho2, M2=M2, alpha2=alpha2,
        W2=W2, beta2=beta2, Mw2=Mw2, Vt2=Vt2, Wt2=Wt2,
        W3=W3, T3=T3, T3s=T3s, rho3=rho3, Mw3=Mw3, beta3=beta3,
        Vx3=Vx3, Vt3=Vt3, Wt3=Wt3, alpha3=alpha3, V3=V3, T03=T03, P03=P03,
    )
    return specific_work, state


def solve_turbine_stage(
    T01: float,
    P01: float,
    specific_work_required: float,
    mass_flow: float,
    cp: float,
    gamma: float,
    design: TurbineStageDesignInputs,
) -> TurbineStageResult:
    R = cp * (gamma - 1.0) / gamma

    M1 = design.inlet_mach_number
    ram_factor = 1.0 + 0.5 * (gamma - 1.0) * M1**2
    T1 = T01 / ram_factor
    P1 = P01 / ram_factor ** (gamma / (gamma - 1.0))
    a1 = math.sqrt(gamma * R * T1)
    V1 = M1 * a1

    U, achieved_psi = blade_speed_from_loading(specific_work_required, design.stage_loading, design.blade_speed_limit)
    Vx = design.flow_coefficient * U

    def residual(P3):
        work, _ = _stage_state(P3, P1, T01, P01, Vx, U, design.degree_of_reaction, design.stator_efficiency, design.rotor_efficiency, cp, gamma, R)
        return work - specific_work_required

    P3 = brentq(residual, 0.05 * P1, 0.999 * P1, xtol=1e-3)
    specific_work, state = _stage_state(P3, P1, T01, P01, Vx, U, design.degree_of_reaction, design.stator_efficiency, design.rotor_efficiency, cp, gamma, R)

    mean_diameter = 2.0 * mean_radius(U, design.rotational_speed_rpm)
    annulus_2 = annulus_from_mass_flow(mass_flow, state["rho2"], Vx, mean_diameter)
    annulus_3 = annulus_from_mass_flow(mass_flow, state["rho3"], state["Vx3"], mean_diameter)
    an2_value = an2((annulus_2.area + annulus_3.area) / 2.0, design.rotational_speed_rpm)

    return TurbineStageResult(
        U=U,
        achieved_stage_loading=achieved_psi,
        flow_coefficient=design.flow_coefficient,
        degree_of_reaction=design.degree_of_reaction,
        Vx=Vx,
        T1=T1,
        P1=P1,
        V1=V1,
        T2=state["T2"],
        T2s=state["T2s"],
        T02_rel=state["T02_rel"],
        P2=state["P2"],
        T02=state["T02"],
        P02=state["P2"] * (state["T02"] / state["T2"]) ** (gamma / (gamma - 1.0)),
        V2=state["V2"],
        alpha2_deg=state["alpha2"],
        M2=state["M2"],
        Vt2=state["Vt2"],
        W2=state["W2"],
        beta2_deg=state["beta2"],
        Mw2=state["Mw2"],
        Wt2=state["Wt2"],
        T3=state["T3"],
        T3s=state["T3s"],
        P3=P3,
        T03=state["T03"],
        P03=state["P03"],
        pressure_ratio=P01 / state["P03"],
        V3=state["V3"],
        alpha3_deg=state["alpha3"],
        M3=state["V3"] / math.sqrt(gamma * R * state["T3"]),
        Vt3=state["Vt3"],
        W3=state["W3"],
        beta3_deg=state["beta3"],
        Mw3=state["Mw3"],
        Wt3=state["Wt3"],
        specific_work=specific_work,
        annulus_2=annulus_2,
        annulus_3=annulus_3,
        an2=an2_value,
    )


def design_turbine_stages(
    T01: float,
    P01: float,
    total_specific_work: float,
    mass_flow: float,
    cp: float,
    gamma: float,
    design: TurbineStageDesignInputs,
    max_stages: int = 8,
):
    """Split the total required specific work evenly across as many
    repeating stages as needed to keep each stage's blade speed at/under the
    limit, then solve each stage in sequence (each stage's exit stagnation
    state feeds the next stage's inlet)."""
    for n_stages in range(1, max_stages + 1):
        per_stage_work = total_specific_work / n_stages
        U_needed = math.sqrt(per_stage_work / design.stage_loading)
        if U_needed <= design.blade_speed_limit:
            break
    else:
        raise ValueError(f"Could not fit within {max_stages} stages at this blade speed limit; raise blade_speed_limit or stage_loading")

    stages = []
    stage_T01, stage_P01 = T01, P01
    for _ in range(n_stages):
        result = solve_turbine_stage(stage_T01, stage_P01, per_stage_work, mass_flow, cp, gamma, design)
        stages.append(result)
        stage_T01, stage_P01 = result.T03, result.P03

    return stages
