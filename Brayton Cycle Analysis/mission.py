"""Stage 1: Mission requirements and the standard atmosphere.

Everything in the design pipeline starts here: cruise Mach, altitude, and
required thrust define the ambient conditions and the target the cycle
design (engine.py) and sizing (engine_sizing.py) stages must hit.
"""

import math
from dataclasses import dataclass

_T0_SL = 288.15  # K, ISA sea-level temperature
_P0_SL = 101_325.0  # Pa, ISA sea-level pressure
_LAPSE_RATE = -0.0065  # K/m, troposphere lapse rate
_TROPOPAUSE_ALT = 11_000.0  # m
_TROPOPAUSE_T = _T0_SL + _LAPSE_RATE * _TROPOPAUSE_ALT  # 216.65 K
_G0 = 9.80665  # m/s^2
_R_AIR = 287.05287  # J/(kg*K)
_MAX_ALTITUDE = 20_000.0  # m, upper limit of this simplified ISA model


def isa_atmosphere(altitude_m: float) -> tuple[float, float]:
    """International Standard Atmosphere: (T [K], P [Pa]) at altitude_m,
    covering the troposphere and isothermal lower stratosphere (0-20 km)."""
    if not (0.0 <= altitude_m <= _MAX_ALTITUDE):
        raise ValueError(f"altitude_m must be in [0, {_MAX_ALTITUDE}] m for this ISA model")

    if altitude_m <= _TROPOPAUSE_ALT:
        T = _T0_SL + _LAPSE_RATE * altitude_m
        P = _P0_SL * (T / _T0_SL) ** (-_G0 / (_LAPSE_RATE * _R_AIR))
    else:
        T = _TROPOPAUSE_T
        P_tropopause = _P0_SL * (_TROPOPAUSE_T / _T0_SL) ** (-_G0 / (_LAPSE_RATE * _R_AIR))
        P = P_tropopause * math.exp(-_G0 * (altitude_m - _TROPOPAUSE_ALT) / (_R_AIR * _TROPOPAUSE_T))
    return T, P


@dataclass(frozen=True)
class MissionRequirements:
    cruise_mach: float
    cruise_altitude_m: float
    required_thrust_N: float  # net thrust required at the cruise design point

    def __post_init__(self):
        if self.cruise_mach < 0.0:
            raise ValueError("cruise_mach must be >= 0")
        if not (0.0 <= self.cruise_altitude_m <= _MAX_ALTITUDE):
            raise ValueError(f"cruise_altitude_m must be in [0, {_MAX_ALTITUDE}] m")
        if self.required_thrust_N <= 0.0:
            raise ValueError("required_thrust_N must be > 0")

    @property
    def ambient_conditions(self) -> tuple[float, float]:
        """(ambient_T, ambient_P) at the cruise altitude."""
        return isa_atmosphere(self.cruise_altitude_m)
