"""Gas properties and stagnation-state stations shared across engine stages.

All thermodynamic states in the engine model are stagnation (total) states
(T0, P0) except where a stage explicitly needs a static condition (the
nozzle exit, where velocity is extracted). Working in stagnation properties
throughout is the standard convention for 0-D gas turbine cycle analysis.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

_TOOLKIT_ROOT = Path(__file__).resolve().parent.parent
if str(_TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT_ROOT))

from config import AIR, COMBUSTION_GAS, GasProperties  # noqa: F401


@dataclass(frozen=True)
class Station:
    """A stagnation thermodynamic state at one point (station) in the engine."""

    label: str
    T0: float  # K
    P0: float  # Pa
    gas: GasProperties
    mdot: float  # kg/s


def isentropic_temp_ratio(gas: GasProperties, pressure_ratio: float) -> float:
    """T_b / T_a for an isentropic process with the given pressure ratio P_b / P_a."""
    return pressure_ratio ** ((gas.gamma - 1.0) / gas.gamma)
