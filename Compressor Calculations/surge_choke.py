"""Surge and choke margin estimates for a single compressor stage.

A true surge/choke assessment needs a full compressor performance map
(pressure ratio vs. corrected mass flow across speed lines) built from an
off-design model -- that doesn't exist in this mean-line, design-point-only
toolkit (component matching / map generation was scoped to a later stage
of the project). What this module provides instead are the standard
DESIGN-POINT proxies for stall and choke risk, each expressed as a
fractional margin to its conventional limit:

- de Haller margin: how far the rotor's W2/W1 sits above the classic 0.72
  stall-margin threshold.
- diffusion factor margin: how far the diffusion factor (see
  diffusion_factor.py) sits below the conventional 0.6 stall limit.
- choke margin: how far the rotor inlet relative Mach number Mw1 sits
  below sonic (Mw1 -> 1 means the passage is approaching choked flow).
"""

from dataclasses import dataclass


def surge_margin_de_haller(de_haller: float, limit: float = 0.72) -> float:
    """Fractional margin above the de Haller stall limit, as a percentage.
    Positive = safe (above the limit); negative = already violating it."""
    if limit <= 0.0:
        raise ValueError("limit must be > 0")
    return (de_haller - limit) / limit * 100.0


def surge_margin_diffusion_factor(diffusion_factor_value: float, limit: float = 0.6) -> float:
    """Fractional margin below the diffusion factor stall limit, as a
    percentage. Positive = safe (below the limit); negative = exceeding it."""
    if limit <= 0.0:
        raise ValueError("limit must be > 0")
    return (limit - diffusion_factor_value) / limit * 100.0


def choke_margin_mach(mach: float, limit: float = 1.0) -> float:
    """Fractional margin below sonic (or another chosen Mach limit), as a
    percentage. Positive = safe (below the limit); negative = choked."""
    if limit <= 0.0:
        raise ValueError("limit must be > 0")
    return (limit - mach) / limit * 100.0


@dataclass(frozen=True)
class SurgeChokeAssessment:
    de_haller_margin_pct: float
    diffusion_factor_margin_pct: float
    choke_margin_pct: float

    @property
    def is_safe(self) -> bool:
        return self.de_haller_margin_pct >= 0.0 and self.diffusion_factor_margin_pct >= 0.0 and self.choke_margin_pct >= 0.0


def assess_stage(result, diffusion_factor_value: float, de_haller_limit: float = 0.72, diffusion_factor_limit: float = 0.6, choke_mach_limit: float = 1.0) -> SurgeChokeAssessment:
    """Convenience wrapper: run all three margin checks against a solved
    CompressorStageResult. diffusion_factor_value is passed in rather than
    recomputed here, since it needs a chosen solidity (see
    diffusion_factor.py) that this stage result alone doesn't carry."""
    return SurgeChokeAssessment(
        de_haller_margin_pct=surge_margin_de_haller(result.de_haller, de_haller_limit),
        diffusion_factor_margin_pct=surge_margin_diffusion_factor(diffusion_factor_value, diffusion_factor_limit),
        choke_margin_pct=choke_margin_mach(result.Mw1, choke_mach_limit),
    )
