"""Thrust extraction and lapse-ratio calculation from an off-design sweep
(off_design.sweep_off_design's list of OffDesignPoint). The underlying
thrust physics lives in Brayton Cycle Analysis's engine.py (design-point)
and Nozzle Analysis's thrust.py (nozzle-level breakdown) -- this module
just tabulates/derives standard performance-analysis metrics from a swept
set of already-computed results, rather than recomputing thrust itself.
"""


def extract_net_thrust(off_design_points) -> list:
    return [p.results.net_thrust for p in off_design_points]


def extract_gross_thrust(off_design_points) -> list:
    return [p.results.gross_thrust for p in off_design_points]


def thrust_lapse_ratio(off_design_points, reference_net_thrust: float) -> list:
    """Net thrust at each point, normalized to a reference (typically
    sea-level-static or the design-point thrust) -- the standard way
    thrust lapse is reported (e.g. Mattingly)."""
    if reference_net_thrust <= 0.0:
        raise ValueError("reference_net_thrust must be > 0")
    return [p.results.net_thrust / reference_net_thrust for p in off_design_points]
