"""Per-unit-mass thermodynamic states and process helpers for the simple
(closed, shaft-power) Brayton cycle variants -- ideal, real, regenerative,
intercooled, reheated. Distinct from stations.py's Station: these cycles
work per unit mass (no mdot) and are closed (state 1 = compressor inlet is
also where heat is rejected back to, unlike the open turbojet path in
engine.py/stages.py).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

_TOOLKIT_ROOT = Path(__file__).resolve().parent.parent
if str(_TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT_ROOT))

from config import AIR, COMBUSTION_GAS, GasProperties  # noqa: F401


@dataclass(frozen=True)
class CycleState:
    """A single thermodynamic point (label, T, P) in a Brayton cycle."""

    label: str
    T: float  # K
    P: float  # Pa


def isentropic_temp_ratio(gas: GasProperties, pressure_ratio: float) -> float:
    """T_b / T_a for an isentropic process with the given pressure ratio P_b / P_a."""
    return pressure_ratio ** ((gas.gamma - 1.0) / gas.gamma)


def isentropic_compression(state_in: CycleState, pressure_ratio: float, gas: GasProperties, label: str, isentropic_efficiency: float = 1.0) -> CycleState:
    """Compress state_in through the given pressure ratio. isentropic_efficiency=1.0
    gives the ideal (isentropic) exit state; <1.0 applies the standard
    eta_c = (T_out_ideal - T_in) / (T_out_actual - T_in) correction."""
    if not (0.0 < isentropic_efficiency <= 1.0):
        raise ValueError("isentropic_efficiency must be in (0, 1]")
    P_out = state_in.P * pressure_ratio
    T_out_ideal = state_in.T * isentropic_temp_ratio(gas, pressure_ratio)
    T_out = state_in.T + (T_out_ideal - state_in.T) / isentropic_efficiency
    return CycleState(label, T_out, P_out)


def isentropic_expansion(state_in: CycleState, pressure_ratio: float, gas: GasProperties, label: str, isentropic_efficiency: float = 1.0) -> CycleState:
    """Expand state_in through the given pressure ratio (< 1, i.e. P_out/P_in).
    isentropic_efficiency=1.0 gives the ideal (isentropic) exit state; <1.0
    applies the standard eta_t = (T_in - T_out_actual) / (T_in - T_out_ideal)
    correction."""
    if not (0.0 < isentropic_efficiency <= 1.0):
        raise ValueError("isentropic_efficiency must be in (0, 1]")
    P_out = state_in.P * pressure_ratio
    T_out_ideal = state_in.T * isentropic_temp_ratio(gas, pressure_ratio)
    T_out = state_in.T - isentropic_efficiency * (state_in.T - T_out_ideal)
    return CycleState(label, T_out, P_out)


def constant_pressure_heat_addition(state_in: CycleState, T_out: float, label: str) -> CycleState:
    """Heat addition (combustor) at constant pressure up to T_out."""
    return CycleState(label, T_out, state_in.P)


def constant_pressure_heat_rejection(state_in: CycleState, T_out: float, label: str) -> CycleState:
    """Heat rejection (closing the cycle back toward state 1) at constant pressure."""
    return CycleState(label, T_out, state_in.P)
