"""Off-design performance: run a FIXED engine design (same compressor
pressure ratio, TIT, and all component efficiencies as the design point)
across a swept flight envelope.

A fully rigorous off-design solve needs compressor/turbine performance
maps (how PR and efficiency shift with corrected speed away from the
design point) to solve the component-matching problem -- those don't
exist in this toolkit (see Compressor Calculations' maps.py note and this
folder's engine_maps.py). What this module implements instead is the
standard first-order approximation taught for exactly this situation
(Mattingly, "Elements of Gas Turbine Propulsion"; Cohen/Rogers/
Saravanamuttoo, "Gas Turbine Theory", Ch. 8): assume the engine keeps
operating at the SAME non-dimensional point (same PR, same TIT, same
efficiencies) at every flight condition, and let only the DIMENSIONAL mass
flow scale with the corrected-flow parameter at the compressor face:

    mdot * sqrt(T02) / P02 = const

This is exact given that assumption (not a fitted correlation), reuses
Brayton Cycle Analysis's already-validated run_turbojet for every other
computation, and captures the dominant real effect (thrust falls sharply
with altitude as air density drops). It does NOT capture how compressor/
turbine efficiency or surge margin actually shift off their design point.
"""

import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path

_BRAYTON_DIR = Path(__file__).resolve().parent.parent / "Brayton Cycle Analysis"
if str(_BRAYTON_DIR) not in sys.path:
    sys.path.insert(0, str(_BRAYTON_DIR))

from engine import TurbojetDesignInputs, run_turbojet  # noqa: E402
from stages import Inlet  # noqa: E402

from atmosphere import isa_atmosphere  # noqa: E402


def corrected_mass_flow_ratio(P_new: float, T_new: float, P_design: float, T_design: float) -> float:
    if P_design <= 0.0 or T_design <= 0.0:
        raise ValueError("P_design and T_design must be > 0")
    return (P_new / P_design) / (T_new / T_design) ** 0.5


def scaled_mass_flow(mdot_design: float, P_new: float, T_new: float, P_design: float, T_design: float) -> float:
    if mdot_design <= 0.0:
        raise ValueError("mdot_design must be > 0")
    return mdot_design * corrected_mass_flow_ratio(P_new, T_new, P_design, T_design)


def compressor_face_conditions(design: TurbojetDesignInputs, ambient_T: float, ambient_P: float, mach: float):
    """(T02, P02) at the compressor face for the given flight condition,
    via the same Inlet stage engine.py itself uses -- ensures the ram-
    recovery convention matches exactly."""
    inlet = Inlet(mach, design.inlet_pressure_recovery, design.cold_gas)
    station2, _ = inlet.run(ambient_T, ambient_P, design.mdot_air)
    return station2.T0, station2.P0


@dataclass(frozen=True)
class OffDesignPoint:
    mach: float
    altitude_m: float
    ambient_T: float
    ambient_P: float
    mdot_air: float
    results: object  # engine.TurbojetResults


def run_off_design_point(design: TurbojetDesignInputs, T02_design: float, P02_design: float, mach: float, altitude_m: float) -> OffDesignPoint:
    """Runs the fixed design at one new (mach, altitude_m) flight
    condition. T02_design/P02_design are the DESIGN point's own compressor-
    face conditions (the corrected-flow reference)."""
    ambient_T, ambient_P = isa_atmosphere(altitude_m)
    T02_new, P02_new = compressor_face_conditions(design, ambient_T, ambient_P, mach)
    mdot_new = scaled_mass_flow(design.mdot_air, P02_new, T02_new, P02_design, T02_design)

    new_inputs = dataclasses.replace(design, ambient_T=ambient_T, ambient_P=ambient_P, flight_mach=mach, mdot_air=mdot_new)
    results, _ = run_turbojet(new_inputs)

    return OffDesignPoint(mach=mach, altitude_m=altitude_m, ambient_T=ambient_T, ambient_P=ambient_P, mdot_air=mdot_new, results=results)


def sweep_off_design(design: TurbojetDesignInputs, envelope_points) -> list:
    """Runs the fixed design across every point in envelope_points (a list
    of mission.FlightEnvelopePoint or anything with .mach/.altitude_m).
    The design's OWN (ambient_T, ambient_P, flight_mach) are used to
    compute the design-point compressor-face reference conditions."""
    T02_design, P02_design = compressor_face_conditions(design, design.ambient_T, design.ambient_P, design.flight_mach)
    return [run_off_design_point(design, T02_design, P02_design, p.mach, p.altitude_m) for p in envelope_points]
