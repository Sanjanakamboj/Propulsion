"""Efficiency extraction from an off-design sweep (off_design.sweep_off_
design's list of OffDesignPoint). Thermal/propulsive/overall efficiency
are computed in Brayton Cycle Analysis's engine.py -- this just tabulates
them across a swept set of results.
"""


def extract_thermal_efficiency(off_design_points) -> list:
    return [p.results.thermal_efficiency for p in off_design_points]


def extract_propulsive_efficiency(off_design_points) -> list:
    return [p.results.propulsive_efficiency for p in off_design_points]


def extract_overall_efficiency(off_design_points) -> list:
    return [p.results.overall_efficiency for p in off_design_points]
