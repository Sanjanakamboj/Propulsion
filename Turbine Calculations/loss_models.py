"""Per-row loss coefficients for a turbine stage.

Unlike the compressor (which lumps rotor+stator loss into one stage
isentropic efficiency), turbine.py already solves the stator and rotor
rows separately and keeps each row's own isentropic exit state (T2s, T3s)
-- so this module computes the standard, exact per-row loss coefficients
directly (Dixon & Hall; Cohen/Rogers/Saravanamuttoo):

    zeta_N = (T2 - T2s) / (V2^2 / (2*cp))   -- nozzle (stator) loss coefficient
    zeta_R = (T3 - T3s) / (W3^2 / (2*cp))   -- rotor loss coefficient

Stage-level efficiency (which folds these row losses together with the
reheat effect between rows) lives in efficiency.py.
"""

from dataclasses import dataclass


def nozzle_loss_coefficient(T2: float, T2s: float, cp: float, V2: float) -> float:
    if V2 <= 0.0:
        raise ValueError("V2 must be > 0")
    return (T2 - T2s) / (V2**2 / (2.0 * cp))


def rotor_loss_coefficient(T3: float, T3s: float, cp: float, W3: float) -> float:
    if W3 <= 0.0:
        raise ValueError("W3 must be > 0")
    return (T3 - T3s) / (W3**2 / (2.0 * cp))


@dataclass(frozen=True)
class TurbineRowLoss:
    zeta_N: float
    zeta_R: float


def compute_row_losses(result, cp: float) -> TurbineRowLoss:
    """Convenience wrapper: derive both row loss coefficients directly from
    a solved TurbineStageResult (cp isn't stored on the result, so it's
    passed in -- the same value used to call solve_turbine_stage)."""
    return TurbineRowLoss(
        zeta_N=nozzle_loss_coefficient(result.T2, result.T2s, cp, result.V2),
        zeta_R=rotor_loss_coefficient(result.T3, result.T3s, cp, result.W3),
    )
