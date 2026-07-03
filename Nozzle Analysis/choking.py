"""Choking margin/status for a converging nozzle.

Like a turbine nozzle throat, a propulsion nozzle choking is a normal,
often-designed-for operating regime (it's what lets a converging-only
nozzle produce useful thrust at all when NPR exceeds critical) -- not a
fault the way compressor choking is. This reports margin/status, not a
pass/fail verdict.
"""

from dataclasses import dataclass

from exit_conditions import critical_pressure_ratio


def nozzle_pressure_ratio(P0_in: float, P_ambient: float) -> float:
    if P_ambient <= 0.0:
        raise ValueError("P_ambient must be > 0")
    return P0_in / P_ambient


def choke_margin_pct(P0_in: float, P_ambient: float, gamma: float) -> float:
    """Fractional margin of the nozzle pressure ratio below the critical
    (choking) ratio, as a percentage. Positive = subsonic/unchoked;
    negative = choked (NPR past critical)."""
    npr = nozzle_pressure_ratio(P0_in, P_ambient)
    critical = critical_pressure_ratio(gamma)
    return (critical - npr) / critical * 100.0


def choke_status(P0_in: float, P_ambient: float, gamma: float, tolerance: float = 0.02) -> str:
    """'unchoked', 'choked' (within tolerance of the critical ratio), or
    'supersonic_at_throat' (NPR well past critical -- a converging-only
    nozzle can't actually reach this, it just stays choked at M=1; this
    status signals a converging-diverging nozzle would deliver more thrust)."""
    npr = nozzle_pressure_ratio(P0_in, P_ambient)
    critical = critical_pressure_ratio(gamma)
    if npr < critical * (1.0 - tolerance):
        return "unchoked"
    if npr <= critical * (1.0 + tolerance):
        return "choked"
    return "supersonic_at_throat"


@dataclass(frozen=True)
class NozzleChokeAssessment:
    nozzle_pressure_ratio: float
    critical_pressure_ratio: float
    margin_pct: float
    status: str


def assess_choking(P0_in: float, P_ambient: float, gamma: float) -> NozzleChokeAssessment:
    return NozzleChokeAssessment(
        nozzle_pressure_ratio=nozzle_pressure_ratio(P0_in, P_ambient),
        critical_pressure_ratio=critical_pressure_ratio(gamma),
        margin_pct=choke_margin_pct(P0_in, P_ambient, gamma),
        status=choke_status(P0_in, P_ambient, gamma),
    )
