"""Axial compressor stage mean-line design.

Standard repeating-stage method: given the stage loading (psi), flow
coefficient (phi), and enthalpy-based degree of reaction (Lambda), the
rotor/stator relative and absolute flow angles are fully determined
algebraically (no iteration needed, unlike the turbine's pressure-matching
problem) via:

    psi = phi * (tan(beta1) - tan(beta2))
    Lambda = (phi / 2) * (tan(beta1) + tan(beta2))

solved as  tan(beta1) = (psi + 2*Lambda) / (2*phi),  tan(beta2) = (2*Lambda - psi) / (2*phi).
Inlet/exit swirl (alpha1, alpha2) then follow from triangle closure,
tan(alpha) = U/Cx - tan(beta) -- for a REPEATING stage, alpha1 is therefore
a consequence of (psi, phi, Lambda), not an independent choice (e.g. at
Lambda=0.5 the classic mirror-symmetry beta1=alpha2, beta2=alpha1 falls out
of this automatically).

The stator is assumed loss-free at the stagnation-pressure level (all stage
inefficiency is folded into one stage isentropic efficiency applied to the
stage's overall stagnation temperature rise); the stage repeats, so exit
swirl/velocity (station 3) equals the inlet condition (station 1).
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent.parent / "Utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from mean_line_common import AnnulusGeometry, an2, annulus_from_mass_flow, blade_speed_from_loading, mean_radius


@dataclass(frozen=True)
class CompressorStageDesignInputs:
    stage_loading: float  # target psi = specific_work / U^2
    flow_coefficient: float  # phi = Cx / U
    degree_of_reaction: float  # enthalpy-based Lambda, in (0, 1)
    blade_speed_limit: float  # m/s, mechanical limit at mean radius
    rotational_speed_rpm: float
    stage_efficiency: float = 0.90

    def __post_init__(self):
        if self.stage_loading <= 0.0:
            raise ValueError("stage_loading must be > 0")
        if self.flow_coefficient <= 0.0:
            raise ValueError("flow_coefficient must be > 0")
        if not (0.0 < self.degree_of_reaction < 1.0):
            raise ValueError("degree_of_reaction must be in (0, 1)")
        if self.blade_speed_limit <= 0.0:
            raise ValueError("blade_speed_limit must be > 0")
        if self.rotational_speed_rpm <= 0.0:
            raise ValueError("rotational_speed_rpm must be > 0")
        if not (0.0 < self.stage_efficiency <= 1.0):
            raise ValueError("stage_efficiency must be in (0, 1]")


@dataclass(frozen=True)
class CompressorStageResult:
    U: float
    achieved_stage_loading: float
    flow_coefficient: float
    degree_of_reaction: float
    Cx: float
    T1: float
    P1: float
    V1: float
    alpha1_deg: float
    M1: float
    Vt1: float
    W1: float
    beta1_deg: float
    Mw1: float
    Wt1: float
    T2: float
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
    P3: float
    T03: float
    T03s: float  # isentropic stagnation exit temperature (h-s diagram ideal marker)
    P03: float
    pressure_ratio: float  # P03 / P01
    de_haller: float  # W2 / W1 -- should stay above ~0.72 to avoid rotor stall
    specific_work: float
    annulus_1: AnnulusGeometry
    annulus_2: AnnulusGeometry
    an2: float


def solve_compressor_stage(
    T01: float,
    P01: float,
    specific_work_required: float,
    mass_flow: float,
    cp: float,
    gamma: float,
    design: CompressorStageDesignInputs,
) -> CompressorStageResult:
    R = cp * (gamma - 1.0) / gamma

    U, achieved_psi = blade_speed_from_loading(specific_work_required, design.stage_loading, design.blade_speed_limit)
    Cx = design.flow_coefficient * U
    phi = Cx / U

    tan_beta1 = (achieved_psi + 2.0 * design.degree_of_reaction) / (2.0 * phi)
    tan_beta2 = (2.0 * design.degree_of_reaction - achieved_psi) / (2.0 * phi)
    beta1 = math.atan(tan_beta1)
    beta2 = math.atan(tan_beta2)

    # Repeating stage: alpha1/alpha2 follow from triangle closure given
    # (psi, phi, Lambda) -- they are not independent choices (see module docstring).
    tan_alpha1 = (U / Cx) - tan_beta1
    tan_alpha2 = (U / Cx) - tan_beta2

    Cw1 = Cx * tan_alpha1
    Ww1 = Cx * tan_beta1
    V1 = math.hypot(Cx, Cw1)
    W1 = math.hypot(Cx, Ww1)

    Cw2 = Cx * tan_alpha2
    Ww2 = Cx * tan_beta2
    V2 = math.hypot(Cx, Cw2)
    W2 = math.hypot(Cx, Ww2)

    specific_work = achieved_psi * U**2

    T02 = T01 + specific_work / cp
    T03 = T02  # stator does no work
    T03s = T01 + design.stage_efficiency * (T03 - T01)
    P03 = P01 * (T03s / T01) ** (gamma / (gamma - 1.0))
    P02 = P03  # stator assumed loss-free at the stagnation-pressure level

    T1 = T01 - V1**2 / (2.0 * cp)
    P1 = P01 * (T1 / T01) ** (gamma / (gamma - 1.0))
    a1 = math.sqrt(gamma * R * T1)

    T2 = T02 - V2**2 / (2.0 * cp)
    P2 = P02 * (T2 / T02) ** (gamma / (gamma - 1.0))
    a2 = math.sqrt(gamma * R * T2)

    T3 = T03 - V1**2 / (2.0 * cp)  # repeating stage: C3 = C1
    P3 = P03 * (T3 / T03) ** (gamma / (gamma - 1.0))

    mean_diameter = 2.0 * mean_radius(U, design.rotational_speed_rpm)
    rho1 = P1 / (R * T1)
    rho2 = P2 / (R * T2)
    annulus_1 = annulus_from_mass_flow(mass_flow, rho1, Cx, mean_diameter)
    annulus_2 = annulus_from_mass_flow(mass_flow, rho2, Cx, mean_diameter)
    an2_value = an2((annulus_1.area + annulus_2.area) / 2.0, design.rotational_speed_rpm)

    return CompressorStageResult(
        U=U,
        achieved_stage_loading=achieved_psi,
        flow_coefficient=phi,
        degree_of_reaction=design.degree_of_reaction,
        Cx=Cx,
        T1=T1, P1=P1, V1=V1, alpha1_deg=math.degrees(math.atan(tan_alpha1)), M1=V1 / a1, Vt1=Cw1,
        W1=W1, beta1_deg=math.degrees(beta1), Mw1=W1 / a1, Wt1=Ww1,
        T2=T2, P2=P2, T02=T02, P02=P02, V2=V2, alpha2_deg=math.degrees(math.atan(tan_alpha2)), M2=V2 / a2, Vt2=Cw2,
        W2=W2, beta2_deg=math.degrees(beta2), Mw2=W2 / a2, Wt2=Ww2,
        T3=T3, P3=P3, T03=T03, T03s=T03s, P03=P03,
        pressure_ratio=P03 / P01,
        de_haller=W2 / W1,
        specific_work=specific_work,
        annulus_1=annulus_1,
        annulus_2=annulus_2,
        an2=an2_value,
    )


def design_compressor_stages(
    T01: float,
    P01: float,
    total_specific_work: float,
    mass_flow: float,
    cp: float,
    gamma: float,
    design: CompressorStageDesignInputs,
    max_stages: int = 20,
):
    """Split the total required specific work evenly across as many
    repeating stages as needed to keep each stage's blade speed at/under the
    limit, then solve each stage in sequence."""
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
        result = solve_compressor_stage(stage_T01, stage_P01, per_stage_work, mass_flow, cp, gamma, design)
        stages.append(result)
        stage_T01, stage_P01 = result.T03, result.P03

    return stages
