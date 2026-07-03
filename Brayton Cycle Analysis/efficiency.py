"""Shared efficiency/performance-metric formulas used by every Brayton
cycle variant (ideal, real, regenerative, intercooled, reheated), so the
division is defined once rather than duplicated per-module.
"""


def thermal_efficiency(net_work: float, heat_added: float) -> float:
    return net_work / heat_added


def back_work_ratio(compressor_work: float, turbine_work: float) -> float:
    return compressor_work / turbine_work


def ideal_brayton_efficiency(pressure_ratio: float, gamma: float) -> float:
    """Closed-form thermal efficiency of the ideal (no-loss) Brayton cycle,
    eta = 1 - 1/PR^((gamma-1)/gamma) -- independent of T1/T3, useful as a
    quick reference/upper bound when checking a numeric cycle result."""
    return 1.0 - 1.0 / (pressure_ratio ** ((gamma - 1.0) / gamma))
