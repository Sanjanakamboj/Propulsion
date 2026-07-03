"""Equivalence ratio and air/fuel mass-flow bookkeeping.

Complements fuel_air_ratio.py's energy-balance FAR with the standard way
combustion is actually characterized: relative to the STOICHIOMETRIC fuel-
air ratio (the ratio that consumes all the oxygen with no excess fuel).
Gas turbine combustors run very lean overall (phi well below 1, typically
~0.3-0.5 for the mean FAR quoted here -- most of the compressor air bypasses
the primary combustion zone for dilution/cooling, which this 0D model
doesn't resolve).

STOICHIOMETRIC_FAR_KEROSENE is a commonly-cited approximate reference value
for kerosene/Jet-A (Saravanamuttoo et al., "Gas Turbine Theory") -- treat it
as a default to override, not an exact constant for your specific fuel.
"""

from dataclasses import dataclass

STOICHIOMETRIC_FAR_KEROSENE = 0.068  # approximate reference value for kerosene/Jet-A


def equivalence_ratio(far: float, far_stoichiometric: float = STOICHIOMETRIC_FAR_KEROSENE) -> float:
    if far_stoichiometric <= 0.0:
        raise ValueError("far_stoichiometric must be > 0")
    if far < 0.0:
        raise ValueError("far must be >= 0")
    return far / far_stoichiometric


def far_from_equivalence_ratio(phi: float, far_stoichiometric: float = STOICHIOMETRIC_FAR_KEROSENE) -> float:
    if far_stoichiometric <= 0.0:
        raise ValueError("far_stoichiometric must be > 0")
    if phi < 0.0:
        raise ValueError("phi must be >= 0")
    return phi * far_stoichiometric


def combustion_regime(phi: float, tolerance: float = 0.02) -> str:
    """'lean' (phi < 1), 'stoichiometric' (within tolerance of 1), or 'rich'
    (phi > 1)."""
    if phi < 1.0 - tolerance:
        return "lean"
    if phi > 1.0 + tolerance:
        return "rich"
    return "stoichiometric"


def air_mass_flow_from_fuel(mdot_fuel: float, far: float) -> float:
    if mdot_fuel <= 0.0:
        raise ValueError("mdot_fuel must be > 0")
    if far <= 0.0:
        raise ValueError("far must be > 0")
    return mdot_fuel / far


@dataclass(frozen=True)
class CombustionState:
    far: float
    equivalence_ratio: float
    regime: str


def assess_combustion(far: float, far_stoichiometric: float = STOICHIOMETRIC_FAR_KEROSENE) -> CombustionState:
    phi = equivalence_ratio(far, far_stoichiometric)
    return CombustionState(far=far, equivalence_ratio=phi, regime=combustion_regime(phi))
