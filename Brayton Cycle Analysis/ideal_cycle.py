"""Basic ideal (closed, no-loss) Brayton cycle: 1->2 isentropic compression,
2->3 constant-pressure heat addition, 3->4 isentropic expansion, 4->1
constant-pressure heat rejection back to the compressor inlet state --
the standard 4-state textbook power-cycle model (no combustion losses,
no component inefficiencies, no pressure drop in the heat exchangers).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

_TOOLKIT_ROOT = Path(__file__).resolve().parent.parent
if str(_TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT_ROOT))

from config import GasProperties
from efficiency import thermal_efficiency as _thermal_efficiency
from state_properties import (
    CycleState,
    constant_pressure_heat_addition,
    isentropic_compression,
    isentropic_expansion,
)


@dataclass(frozen=True)
class IdealCycleInputs:
    T1: float  # K, compressor inlet temperature
    P1: float  # Pa, compressor inlet pressure
    pressure_ratio: float  # P2 / P1
    T3: float  # K, turbine inlet temperature (peak cycle temperature)
    gas: GasProperties

    def __post_init__(self):
        if self.T1 <= 0.0:
            raise ValueError("T1 must be > 0")
        if self.P1 <= 0.0:
            raise ValueError("P1 must be > 0")
        if self.pressure_ratio <= 1.0:
            raise ValueError("pressure_ratio must be > 1")
        if self.T3 <= self.T1:
            raise ValueError("T3 must be > T1")


@dataclass(frozen=True)
class IdealCycleResult:
    states: list  # [state1, state2, state3, state4]
    compressor_work: float  # J/kg, work IN to the compressor
    turbine_work: float  # J/kg, work OUT of the turbine
    net_work: float  # J/kg, turbine_work - compressor_work
    heat_added: float  # J/kg, combustor heat input
    heat_rejected: float  # J/kg, heat rejected 4->1
    thermal_efficiency: float  # net_work / heat_added


def run_ideal_cycle(inputs: IdealCycleInputs) -> IdealCycleResult:
    gas = inputs.gas
    state1 = CycleState("1", inputs.T1, inputs.P1)
    state2 = isentropic_compression(state1, inputs.pressure_ratio, gas, label="2")
    state3 = constant_pressure_heat_addition(state2, inputs.T3, label="3")
    state4 = isentropic_expansion(state3, 1.0 / inputs.pressure_ratio, gas, label="4")

    compressor_work = gas.cp * (state2.T - state1.T)
    turbine_work = gas.cp * (state3.T - state4.T)
    net_work = turbine_work - compressor_work
    heat_added = gas.cp * (state3.T - state2.T)
    heat_rejected = gas.cp * (state4.T - state1.T)

    return IdealCycleResult(
        states=[state1, state2, state3, state4],
        compressor_work=compressor_work,
        turbine_work=turbine_work,
        net_work=net_work,
        heat_added=heat_added,
        heat_rejected=heat_rejected,
        thermal_efficiency=_thermal_efficiency(net_work, heat_added),
    )
