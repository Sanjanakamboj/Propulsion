"""Intercooled Brayton cycle: compression is split into LP and HP stages
with an intercooler between them that cools the air back down at constant
pressure, reducing total compressor work for the same overall pressure
ratio (at the cost of an extra heat exchanger).

6-state layout: 1 (LP compressor inlet) -> 2 (LP compressor exit /
intercooler inlet) -> 3 (intercooler exit / HP compressor inlet) -> 4 (HP
compressor exit / combustor inlet) -> 5 (combustor exit / turbine inlet)
-> 6 (turbine exit). No pressure loss modeled in the intercooler or
combustor (P3=P2, P5=P4).
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
class IntercoolingCycleInputs:
    T1: float  # K, LP compressor inlet temperature
    P1: float  # Pa, LP compressor inlet pressure
    pressure_ratio_lp: float  # P2 / P1
    pressure_ratio_hp: float  # P4 / P3
    T_intercool_exit: float  # K, temperature after the intercooler (often ~= T1)
    T5: float  # K, turbine inlet temperature (peak cycle temperature)
    gas: GasProperties
    compressor_efficiency: float = 0.85  # applied to both LP and HP stages
    turbine_efficiency: float = 0.90

    def __post_init__(self):
        if self.T1 <= 0.0:
            raise ValueError("T1 must be > 0")
        if self.P1 <= 0.0:
            raise ValueError("P1 must be > 0")
        if self.pressure_ratio_lp <= 1.0:
            raise ValueError("pressure_ratio_lp must be > 1")
        if self.pressure_ratio_hp <= 1.0:
            raise ValueError("pressure_ratio_hp must be > 1")
        if self.T_intercool_exit <= 0.0:
            raise ValueError("T_intercool_exit must be > 0")
        if self.T5 <= self.T_intercool_exit:
            raise ValueError("T5 must be > T_intercool_exit")
        if not (0.0 < self.compressor_efficiency <= 1.0):
            raise ValueError("compressor_efficiency must be in (0, 1]")
        if not (0.0 < self.turbine_efficiency <= 1.0):
            raise ValueError("turbine_efficiency must be in (0, 1]")


@dataclass(frozen=True)
class IntercoolingCycleResult:
    states: list  # [state1, state2, state3, state4, state5, state6]
    compressor_work: float  # J/kg, total across both stages
    turbine_work: float  # J/kg
    net_work: float  # J/kg
    heat_added: float  # J/kg, combustor heat input
    heat_rejected: float  # J/kg, intercooler + final exhaust heat rejected
    thermal_efficiency: float  # net_work / heat_added


def run_intercooling_cycle(inputs: IntercoolingCycleInputs) -> IntercoolingCycleResult:
    gas = inputs.gas
    state1 = CycleState("1", inputs.T1, inputs.P1)
    state2 = isentropic_compression(state1, inputs.pressure_ratio_lp, gas, label="2", isentropic_efficiency=inputs.compressor_efficiency)
    state3 = CycleState("3", inputs.T_intercool_exit, state2.P)  # intercooler exit, constant pressure
    state4 = isentropic_compression(state3, inputs.pressure_ratio_hp, gas, label="4", isentropic_efficiency=inputs.compressor_efficiency)
    state5 = constant_pressure_heat_addition(state4, inputs.T5, label="5")
    overall_pressure_ratio = inputs.pressure_ratio_lp * inputs.pressure_ratio_hp
    state6 = isentropic_expansion(state5, 1.0 / overall_pressure_ratio, gas, label="6", isentropic_efficiency=inputs.turbine_efficiency)

    compressor_work = gas.cp * (state2.T - state1.T) + gas.cp * (state4.T - state3.T)
    turbine_work = gas.cp * (state5.T - state6.T)
    net_work = turbine_work - compressor_work
    heat_added = gas.cp * (state5.T - state4.T)
    heat_rejected = gas.cp * (state2.T - state3.T) + gas.cp * (state6.T - state1.T)

    return IntercoolingCycleResult(
        states=[state1, state2, state3, state4, state5, state6],
        compressor_work=compressor_work,
        turbine_work=turbine_work,
        net_work=net_work,
        heat_added=heat_added,
        heat_rejected=heat_rejected,
        thermal_efficiency=_thermal_efficiency(net_work, heat_added),
    )
