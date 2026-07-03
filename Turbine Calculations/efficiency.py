"""Stage-level efficiency definitions for a turbine stage.

The stator and rotor are solved as separate rows in turbine.py (each with
its own isentropic efficiency), but the overall STAGE efficiency needs its
own definition -- comparing the actual work extracted to the work a single
isentropic expansion from the stage inlet stagnation state straight to the
stage exit pressure would have delivered:

    eta_tt = (T01 - T03) / (T01 - T03ss)   -- total-to-total efficiency
    eta_ts = (T01 - T03) / (T01 - T3ss)    -- total-to-static efficiency

T03ss/T3ss are reached by ONE isentropic expansion from the stage inlet
straight to P03/P3 -- not the same as the per-row isentropic states T2s/T3s
in loss_models.py, which are each row's own local ideal exit condition.
Since P3 <= P03 always, T3ss <= T03ss always, so eta_ts <= eta_tt always:
total-to-static never credits the exit kinetic energy back, so it's always
the more conservative (or equal) number.

These are exact definitions from the stage's own already-solved state, not
fitted empirical correlations.
"""

from dataclasses import dataclass


def total_to_total_efficiency(T01: float, T03: float, P01: float, P03: float, gamma: float) -> float:
    if not (0.0 < P03 < P01):
        raise ValueError("P03 must be in (0, P01) for an expansion")
    T03ss = T01 * (P03 / P01) ** ((gamma - 1.0) / gamma)
    if T01 <= T03ss:
        raise ValueError("T01 must be > T03ss (no ideal work available)")
    return (T01 - T03) / (T01 - T03ss)


def total_to_static_efficiency(T01: float, T03: float, P01: float, P3: float, gamma: float) -> float:
    if not (0.0 < P3 < P01):
        raise ValueError("P3 must be in (0, P01) for an expansion")
    T3ss = T01 * (P3 / P01) ** ((gamma - 1.0) / gamma)
    if T01 <= T3ss:
        raise ValueError("T01 must be > T3ss (no ideal work available)")
    return (T01 - T03) / (T01 - T3ss)


@dataclass(frozen=True)
class TurbineStageEfficiency:
    eta_tt: float
    eta_ts: float


def compute_stage_efficiency(result, T01: float, P01: float, gamma: float) -> TurbineStageEfficiency:
    """Convenience wrapper: derive both stage efficiency definitions
    directly from a solved TurbineStageResult (T01/P01/gamma are the same
    values used to call solve_turbine_stage, not stored on the result)."""
    return TurbineStageEfficiency(
        eta_tt=total_to_total_efficiency(T01, result.T03, P01, result.P03, gamma),
        eta_ts=total_to_static_efficiency(T01, result.T03, P01, result.P3, gamma),
    )
