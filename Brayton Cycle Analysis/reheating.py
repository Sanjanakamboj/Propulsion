"""Reheat Brayton cycle: expansion is split into HP and LP turbine stages
with a reheat combustor between them that reheats the gas back up at
constant pressure, increasing total turbine work for the same overall
pressure ratio (at the cost of extra fuel burned in the reheat combustor).

6-state layout: 1 (compressor inlet) -> 2 (compressor exit / combustor
inlet) -> 3 (combustor exit / HP turbine inlet) -> 4 (HP turbine exit /
reheat combustor inlet) -> 5 (reheat combustor exit / LP turbine inlet)
-> 6 (LP turbine exit). No pressure loss modeled in either combustor
(P3=P2, P5=P4). The HP turbine's expansion ratio is a free parameter; the
LP turbine's expansion ratio is derived so the overall pressure ratio
(compressor PR) closes back to P1 at state 6.
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
class ReheatingCycleInputs:
    T1: float  # K, compressor inlet temperature
    P1: float  # Pa, compressor inlet pressure
    pressure_ratio: float  # overall P2/P1, closes back to P1 at state 6
    T3: float  # K, HP turbine inlet temperature (combustor exit)
    pressure_ratio_hp_turbine: float  # P3 / P4, HP turbine's (partial) expansion ratio
    T5: float  # K, LP turbine inlet temperature (reheat combustor exit)
    gas: GasProperties
    compressor_efficiency: float = 0.85
    turbine_efficiency: float = 0.90  # applied to both HP and LP turbine stages

    def __post_init__(self):
        if self.T1 <= 0.0:
            raise ValueError("T1 must be > 0")
        if self.P1 <= 0.0:
            raise ValueError("P1 must be > 0")
        if self.pressure_ratio <= 1.0:
            raise ValueError("pressure_ratio must be > 1")
        if self.T3 <= self.T1:
            raise ValueError("T3 must be > T1")
        if not (1.0 < self.pressure_ratio_hp_turbine < self.pressure_ratio):
            raise ValueError("pressure_ratio_hp_turbine must be in (1, pressure_ratio) so the LP turbine still has a pressure ratio > 1 to expand through")
        if self.T5 <= 0.0:
            raise ValueError("T5 must be > 0")
        if not (0.0 < self.compressor_efficiency <= 1.0):
            raise ValueError("compressor_efficiency must be in (0, 1]")
        if not (0.0 < self.turbine_efficiency <= 1.0):
            raise ValueError("turbine_efficiency must be in (0, 1]")


@dataclass(frozen=True)
class ReheatingCycleResult:
    states: list  # [state1, state2, state3, state4, state5, state6]
    compressor_work: float  # J/kg
    turbine_work: float  # J/kg, total across both stages
    net_work: float  # J/kg
    heat_added: float  # J/kg, combustor + reheat combustor heat input
    heat_rejected: float  # J/kg, final exhaust heat rejected
    thermal_efficiency: float  # net_work / heat_added


def run_reheating_cycle(inputs: ReheatingCycleInputs) -> ReheatingCycleResult:
    gas = inputs.gas
    state1 = CycleState("1", inputs.T1, inputs.P1)
    state2 = isentropic_compression(state1, inputs.pressure_ratio, gas, label="2", isentropic_efficiency=inputs.compressor_efficiency)
    state3 = constant_pressure_heat_addition(state2, inputs.T3, label="3")
    state4 = isentropic_expansion(state3, 1.0 / inputs.pressure_ratio_hp_turbine, gas, label="4", isentropic_efficiency=inputs.turbine_efficiency)
    state5 = constant_pressure_heat_addition(state4, inputs.T5, label="5")
    lp_turbine_pressure_ratio = inputs.pressure_ratio / inputs.pressure_ratio_hp_turbine
    state6 = isentropic_expansion(state5, 1.0 / lp_turbine_pressure_ratio, gas, label="6", isentropic_efficiency=inputs.turbine_efficiency)

    compressor_work = gas.cp * (state2.T - state1.T)
    turbine_work = gas.cp * (state3.T - state4.T) + gas.cp * (state5.T - state6.T)
    net_work = turbine_work - compressor_work
    heat_added = gas.cp * (state3.T - state2.T) + gas.cp * (state5.T - state4.T)
    heat_rejected = gas.cp * (state6.T - state1.T)

    return ReheatingCycleResult(
        states=[state1, state2, state3, state4, state5, state6],
        compressor_work=compressor_work,
        turbine_work=turbine_work,
        net_work=net_work,
        heat_added=heat_added,
        heat_rejected=heat_rejected,
        thermal_efficiency=_thermal_efficiency(net_work, heat_added),
    )
