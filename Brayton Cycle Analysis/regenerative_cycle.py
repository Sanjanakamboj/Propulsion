"""Regenerative Brayton cycle: a recuperator uses turbine-exhaust heat to
preheat the compressor-exit air before it reaches the combustor, cutting
the fuel heat that must be added for the same turbine inlet temperature
(work output is unchanged -- regeneration only recovers otherwise-wasted
exhaust heat).

5-state layout: 1 (compressor inlet) -> 2 (compressor exit) -> 3
(regenerator cold-side exit / combustor inlet, preheated) -> 4 (combustor
exit / turbine inlet) -> 5 (turbine exit / regenerator hot-side inlet).
No pressure losses are modeled in the regenerator or combustor (P3=P2,
P4=P3), consistent with ideal_cycle.py/real_cycle.py's level of idealization.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

_TOOLKIT_ROOT = Path(__file__).resolve().parent.parent
if str(_TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT_ROOT))

from config import GasProperties
from efficiency import thermal_efficiency as _thermal_efficiency
from state_properties import CycleState, constant_pressure_heat_addition, isentropic_compression, isentropic_expansion


@dataclass(frozen=True)
class RegenerativeCycleInputs:
    T1: float  # K, compressor inlet temperature
    P1: float  # Pa, compressor inlet pressure
    pressure_ratio: float  # P2 / P1
    T4: float  # K, turbine inlet temperature (peak cycle temperature)
    gas: GasProperties
    regenerator_effectiveness: float = 0.75  # epsilon = (T3-T2)/(T5-T2), 0=no regen, 1=ideal
    compressor_efficiency: float = 0.85
    turbine_efficiency: float = 0.90

    def __post_init__(self):
        if self.T1 <= 0.0:
            raise ValueError("T1 must be > 0")
        if self.P1 <= 0.0:
            raise ValueError("P1 must be > 0")
        if self.pressure_ratio <= 1.0:
            raise ValueError("pressure_ratio must be > 1")
        if self.T4 <= self.T1:
            raise ValueError("T4 must be > T1")
        if not (0.0 <= self.regenerator_effectiveness <= 1.0):
            raise ValueError("regenerator_effectiveness must be in [0, 1]")
        if not (0.0 < self.compressor_efficiency <= 1.0):
            raise ValueError("compressor_efficiency must be in (0, 1]")
        if not (0.0 < self.turbine_efficiency <= 1.0):
            raise ValueError("turbine_efficiency must be in (0, 1]")


@dataclass(frozen=True)
class RegenerativeCycleResult:
    states: list  # [state1, state2, state3, state4, state5]
    compressor_work: float  # J/kg
    turbine_work: float  # J/kg
    net_work: float  # J/kg, turbine_work - compressor_work (regeneration doesn't change work)
    heat_added: float  # J/kg, combustor heat input (reduced vs non-regenerative for the same T4)
    heat_rejected: float  # J/kg, heat rejected to ambient by the regenerator's hot-side exit
    thermal_efficiency: float  # net_work / heat_added


def run_regenerative_cycle(inputs: RegenerativeCycleInputs) -> RegenerativeCycleResult:
    gas = inputs.gas
    state1 = CycleState("1", inputs.T1, inputs.P1)
    state2 = isentropic_compression(state1, inputs.pressure_ratio, gas, label="2", isentropic_efficiency=inputs.compressor_efficiency)
    state4 = CycleState("4", inputs.T4, state2.P)  # combustor exit / turbine inlet, no pressure loss
    state5 = isentropic_expansion(state4, 1.0 / inputs.pressure_ratio, gas, label="5", isentropic_efficiency=inputs.turbine_efficiency)

    T3 = state2.T + inputs.regenerator_effectiveness * (state5.T - state2.T)
    state3 = constant_pressure_heat_addition(state2, T3, label="3")  # regenerator cold-side exit / combustor inlet

    compressor_work = gas.cp * (state2.T - state1.T)
    turbine_work = gas.cp * (state4.T - state5.T)
    net_work = turbine_work - compressor_work
    heat_added = gas.cp * (state4.T - state3.T)

    # Regenerator hot-side exit temperature, from the energy balance that the
    # heat the hot stream gives up must equal what the cold stream gains
    # (equal mass flow, constant cp on both sides).
    hot_side_exit_T = state5.T - (state3.T - state2.T)
    heat_rejected = gas.cp * (hot_side_exit_T - state1.T)

    return RegenerativeCycleResult(
        states=[state1, state2, state3, state4, state5],
        compressor_work=compressor_work,
        turbine_work=turbine_work,
        net_work=net_work,
        heat_added=heat_added,
        heat_rejected=heat_rejected,
        thermal_efficiency=_thermal_efficiency(net_work, heat_added),
    )
