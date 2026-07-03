"""Loss-coefficient <-> isentropic-efficiency conversions for a compressor
stage.

compressor.py's stage solve uses a single lumped isentropic (adiabatic)
stage efficiency (T03s vs T03) rather than tracking rotor/stator loss
separately -- there's no independent pressure-drop bookkeeping to derive a
stagnation-pressure loss coefficient from. What this module DOES provide is
the standard, exact enthalpy-loss-coefficient definition used throughout
cascade literature (Dixon & Hall, Cohen/Rogers/Saravanamuttoo):

    zeta = cp * (T_actual - T_ideal) / (0.5 * V_ref^2)

applied at the stage level with the rotor inlet relative velocity W1 as the
reference dynamic head (the standard convention for a compressor rotor loss
coefficient). This lets a solved design be reported/compared in loss-
coefficient terms, and lets a target loss coefficient (e.g. from a future
predictive correlation) be converted back into the stage_efficiency this
model's CompressorStageDesignInputs actually takes.

This is NOT a predictive empirical loss correlation (e.g. Lieblein
momentum-thickness data, Ainley-Mathieson, Soderberg) -- building one of
those needs curve-fit constants from the original correlating charts, which
isn't something to guess at from memory. This module only converts between
two exact, equivalent ways of expressing the same already-known loss.
"""

from dataclasses import dataclass


def enthalpy_loss_coefficient(T_actual: float, T_ideal: float, cp: float, V_ref: float) -> float:
    """zeta = cp*(T_actual - T_ideal) / (0.5*V_ref^2). T_actual/T_ideal are
    stagnation temperatures at the same station (e.g. T03 vs T03s)."""
    if V_ref <= 0.0:
        raise ValueError("V_ref must be > 0")
    return cp * (T_actual - T_ideal) / (0.5 * V_ref**2)


def isentropic_efficiency_from_loss_coefficient(zeta: float, delta_T_actual: float, cp: float, V_ref: float) -> float:
    """Inverse of enthalpy_loss_coefficient: given a target zeta and the
    stage's actual stagnation temperature rise (T03 - T01), returns the
    equivalent isentropic efficiency to plug into
    CompressorStageDesignInputs.stage_efficiency.

    Derivation: eta = (T03s-T01)/(T03-T01) = 1 - (T03-T03s)/(T03-T01), and
    zeta = cp*(T03-T03s)/(0.5*V_ref^2) => T03-T03s = zeta*0.5*V_ref^2/cp.
    """
    if delta_T_actual <= 0.0:
        raise ValueError("delta_T_actual (T03 - T01) must be > 0")
    if V_ref <= 0.0:
        raise ValueError("V_ref must be > 0")
    eta = 1.0 - (zeta * 0.5 * V_ref**2 / cp) / delta_T_actual
    if not (0.0 < eta <= 1.0):
        raise ValueError(f"resulting isentropic efficiency ({eta:.3f}) is outside (0, 1] -- zeta is too large for this delta_T_actual/V_ref")
    return eta


@dataclass(frozen=True)
class CompressorStageLoss:
    zeta: float  # enthalpy loss coefficient, referenced to W1
    isentropic_efficiency: float


def compute_stage_loss(result, cp: float) -> CompressorStageLoss:
    """Convenience wrapper: derive the stage's equivalent enthalpy loss
    coefficient and isentropic efficiency directly from a solved
    CompressorStageResult (cp isn't stored on the result, so it's passed
    in -- the same value used to solve the stage)."""
    T01 = result.T1 + result.V1**2 / (2.0 * cp)
    zeta = enthalpy_loss_coefficient(result.T03, result.T03s, cp=cp, V_ref=result.W1)
    eta = (result.T03s - T01) / (result.T03 - T01)
    return CompressorStageLoss(zeta=zeta, isentropic_efficiency=eta)
