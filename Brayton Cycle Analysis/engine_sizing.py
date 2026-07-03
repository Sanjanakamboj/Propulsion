"""Stage 3: Engine sizing -- turn a required thrust into a mass flow and
physical annulus dimensions, given an already-closed cycle design point.

Net specific thrust (thrust per unit air mass flow) does not depend on the
absolute scale of mdot_air in this 0D model -- gross thrust, ram drag, and
fuel flow all scale linearly with it. So the required mass flow follows
directly from a single reference-scale cycle run, with no iteration needed.
"""

import dataclasses
import math

from engine import TurbojetDesignInputs, run_turbojet
from stations import GasProperties


def size_mass_flow_for_thrust(design: TurbojetDesignInputs, required_thrust_N: float) -> float:
    reference = dataclasses.replace(design, mdot_air=1.0)
    results, _ = run_turbojet(reference)
    return required_thrust_N / results.specific_thrust


def compressor_face_area(mdot_air: float, T0: float, P0: float, gas: GasProperties, axial_mach: float) -> float:
    """Annulus flow area at the compressor face for a chosen axial Mach number."""
    T_static = T0 / (1.0 + 0.5 * (gas.gamma - 1.0) * axial_mach**2)
    P_static = P0 * (T_static / T0) ** (gas.gamma / (gas.gamma - 1.0))
    rho = P_static / (gas.R * T_static)
    velocity = axial_mach * math.sqrt(gas.gamma * gas.R * T_static)
    return mdot_air / (rho * velocity)


def annulus_tip_diameter(area: float, hub_to_tip_ratio: float) -> float:
    """Outer (tip) diameter of an annulus with the given flow area and hub/tip ratio."""
    if not (0.0 <= hub_to_tip_ratio < 1.0):
        raise ValueError("hub_to_tip_ratio must be in [0, 1)")
    r_tip = math.sqrt(area / (math.pi * (1.0 - hub_to_tip_ratio**2)))
    return 2.0 * r_tip


@dataclasses.dataclass(frozen=True)
class EngineSizingResults:
    mdot_air: float  # kg/s
    compressor_face_area: float  # m^2
    compressor_face_diameter: float  # m
    nozzle_exit_area: float  # m^2


def size_engine(
    design: TurbojetDesignInputs,
    required_thrust_N: float,
    axial_mach: float = 0.5,
    hub_to_tip_ratio: float = 0.5,
):
    """Solve mass flow for the required thrust, then re-run the cycle at that
    scale to get the physical annulus dimensions. Returns
    (EngineSizingResults, sized_design, TurbojetResults, stage_records)."""
    mdot_air = size_mass_flow_for_thrust(design, required_thrust_N)
    sized_design = dataclasses.replace(design, mdot_air=mdot_air)
    results, stage_records = run_turbojet(sized_design)

    station2 = results.stations["2"]
    area2 = compressor_face_area(mdot_air, station2.T0, station2.P0, station2.gas, axial_mach)
    diameter2 = annulus_tip_diameter(area2, hub_to_tip_ratio)

    sizing = EngineSizingResults(
        mdot_air=mdot_air,
        compressor_face_area=area2,
        compressor_face_diameter=diameter2,
        nozzle_exit_area=results.nozzle_exit_area,
    )
    return sizing, sized_design, results, stage_records
