"""Flight envelope grid generation for off-design performance sweeps --
distinct from Brayton Cycle Analysis's MissionRequirements, which
describes ONE design point (cruise Mach/altitude/thrust target). This
describes the SET of (Mach, altitude) points to evaluate a fixed engine
design across.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FlightEnvelopePoint:
    mach: float
    altitude_m: float

    def __post_init__(self):
        if self.mach < 0.0:
            raise ValueError("mach must be >= 0")
        if self.altitude_m < 0.0:
            raise ValueError("altitude_m must be >= 0")


def generate_flight_envelope(mach_values, altitude_values_m) -> list:
    """Every combination of the given Mach and altitude values, as a flat
    list of FlightEnvelopePoint (altitude-major order: all Mach values at
    the first altitude, then all Mach values at the next, ...)."""
    return [FlightEnvelopePoint(mach=m, altitude_m=a) for a in altitude_values_m for m in mach_values]
