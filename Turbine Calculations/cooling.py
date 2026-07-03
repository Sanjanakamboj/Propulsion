"""Blade cooling requirement estimate for a turbine stage.

This computes how much cooling effectiveness is needed to keep a blade at
or below a given allowable metal temperature -- it does NOT select a
material, model coolant flow rate/blowing ratio, or predict film-cooling
performance from geometry (those need empirical correlations/material data
this project has deliberately kept out of scope, see Materials/). The
allowable blade temperature is a plain design input here, the same way
blade_speed_limit is elsewhere in this project -- not a materials lookup.

Uses the standard, exact cooling effectiveness definition (Han, Dutta &
Ekkad, "Gas Turbine Heat Transfer and Cooling Technology"; any turbine
cooling text):

    eta_c = (T_gas - T_blade) / (T_gas - T_coolant)

where T_gas is the gas temperature the blade surface actually sees (the
ROTOR-RELATIVE stagnation temperature T02_rel for a rotor blade, or the
absolute stagnation temperature T01 for a stator vane), and T_coolant is
the coolant supply temperature (typically compressor bleed air).
"""

from dataclasses import dataclass


def cooling_effectiveness(T_gas: float, T_blade: float, T_coolant: float) -> float:
    if T_gas <= T_coolant:
        raise ValueError("T_gas must be > T_coolant")
    return (T_gas - T_blade) / (T_gas - T_coolant)


def blade_temperature_from_effectiveness(T_gas: float, eta_c: float, T_coolant: float) -> float:
    if T_gas <= T_coolant:
        raise ValueError("T_gas must be > T_coolant")
    return T_gas - eta_c * (T_gas - T_coolant)


def required_cooling_effectiveness(T_gas: float, T_blade_limit: float, T_coolant: float) -> float:
    """Effectiveness needed to hold the blade at T_blade_limit. Returns 0.0
    (no cooling needed) if T_gas is already at or below T_blade_limit."""
    if T_gas <= T_coolant:
        raise ValueError("T_gas must be > T_coolant")
    if T_blade_limit <= T_coolant:
        raise ValueError("T_blade_limit must be > T_coolant -- otherwise no achievable effectiveness (<=1) can reach it")
    if T_gas <= T_blade_limit:
        return 0.0
    return (T_gas - T_blade_limit) / (T_gas - T_coolant)


@dataclass(frozen=True)
class CoolingRequirement:
    T_gas: float
    T_blade_limit: float
    required_effectiveness: float

    @property
    def cooling_needed(self) -> bool:
        return self.required_effectiveness > 0.0

    @property
    def achievable(self) -> bool:
        """False if even eta_c=1 (coolant-temperature blade) can't reach
        T_blade_limit -- i.e. the limit is at or below the coolant supply
        temperature itself."""
        return self.required_effectiveness <= 1.0


@dataclass(frozen=True)
class TurbineStageCoolingAssessment:
    nozzle: CoolingRequirement
    rotor: CoolingRequirement


def assess_stage_cooling(result, T01: float, T_coolant: float, T_blade_limit: float) -> TurbineStageCoolingAssessment:
    """Convenience wrapper: cooling requirement for both rows of a solved
    TurbineStageResult. The nozzle (stator) sees the absolute stagnation
    temperature T01 (a solve input, not stored on the result); the rotor
    sees its own relative stagnation temperature T02_rel."""
    nozzle_req = required_cooling_effectiveness(T01, T_blade_limit, T_coolant) if T01 > T_coolant else 0.0
    rotor_req = required_cooling_effectiveness(result.T02_rel, T_blade_limit, T_coolant) if result.T02_rel > T_coolant else 0.0
    return TurbineStageCoolingAssessment(
        nozzle=CoolingRequirement(T_gas=T01, T_blade_limit=T_blade_limit, required_effectiveness=nozzle_req),
        rotor=CoolingRequirement(T_gas=result.T02_rel, T_blade_limit=T_blade_limit, required_effectiveness=rotor_req),
    )
