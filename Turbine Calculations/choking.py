"""Choking checks for a turbine stage's nozzle (stator) and rotor throats.

Unlike a compressor, where approaching sonic flow (choking) is always
undesirable, a turbine NOZZLE is very often deliberately choked at its
throat (M2 ~ 1) in real HP turbine design -- once choked, the nozzle's
mass flow is capped and stops responding to further downstream pressure
drop, which is a normal, expected operating regime, not a fault. What IS a
genuine problem is M > 1 downstream of a purely converging passage (this
mean-line model has no converging-diverging throat geometry), since that's
unphysical without one. So this module reports margin/status, not a
pass/fail verdict the way surge_choke.py does for the compressor.
"""

from dataclasses import dataclass


def choke_margin_mach(mach: float, limit: float = 1.0) -> float:
    """Fractional margin below sonic (or another chosen Mach limit), as a
    percentage. Positive = subsonic; zero = choked; negative = supersonic
    (unphysical here, since this model has no diverging passage section)."""
    if limit <= 0.0:
        raise ValueError("limit must be > 0")
    return (limit - mach) / limit * 100.0


def choke_status(mach: float, choked_tolerance: float = 0.02) -> str:
    """'subsonic', 'choked' (within choked_tolerance of M=1), or
    'supersonic' (M > 1 + choked_tolerance -- physically invalid for a
    purely converging passage)."""
    if mach > 1.0 + choked_tolerance:
        return "supersonic"
    if mach >= 1.0 - choked_tolerance:
        return "choked"
    return "subsonic"


@dataclass(frozen=True)
class TurbineChokeAssessment:
    nozzle_margin_pct: float  # from M2
    nozzle_status: str
    rotor_margin_pct: float  # from Mw3 (rotor-relative)
    rotor_status: str

    @property
    def is_physically_valid(self) -> bool:
        """False only if either row is genuinely supersonic -- a choked
        nozzle (status == 'choked') is a valid, common operating point."""
        return self.nozzle_status != "supersonic" and self.rotor_status != "supersonic"


def assess_stage(result) -> TurbineChokeAssessment:
    """Convenience wrapper: check both rows' Mach numbers against sonic
    directly from a solved TurbineStageResult (M2 for the nozzle, Mw3 for
    the rotor-relative exit)."""
    return TurbineChokeAssessment(
        nozzle_margin_pct=choke_margin_mach(result.M2),
        nozzle_status=choke_status(result.M2),
        rotor_margin_pct=choke_margin_mach(result.Mw3),
        rotor_status=choke_status(result.Mw3),
    )
