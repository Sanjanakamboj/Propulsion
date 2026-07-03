"""Shared physical/gas constants used across the toolkit's modules --
gas turbine cycle analysis conventionally works in two gas mixtures (cold
air upstream of combustion, hot combustion gas downstream), each
characterized by (cp, gamma). Modules take cp/gamma as plain floats rather
than importing AIR/COMBUSTION_GAS directly, so a module is free to use a
different gas model -- these are just the toolkit's default reference
values, defined once here rather than per-module.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GasProperties:
    cp: float  # J/(kg*K)
    gamma: float

    @property
    def R(self) -> float:
        return self.cp * (self.gamma - 1.0) / self.gamma


AIR = GasProperties(cp=1005.0, gamma=1.4)
COMBUSTION_GAS = GasProperties(cp=1148.0, gamma=1.333)
