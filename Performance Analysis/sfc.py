"""TSFC extraction from an off-design sweep (off_design.sweep_off_design's
list of OffDesignPoint). TSFC itself is computed in Brayton Cycle
Analysis's engine.py -- this just tabulates it across a swept set of
results, in either kg/(N*s) (SI, as stored) or kg/(N*hr) (the
conventionally reported unit).
"""


def extract_tsfc(off_design_points) -> list:
    """kg/(N*s), as stored on TurbojetResults."""
    return [p.results.tsfc for p in off_design_points]


def extract_tsfc_per_hour(off_design_points) -> list:
    """kg/(N*hr) -- the unit TSFC is conventionally reported in."""
    return [p.results.tsfc * 3600.0 for p in off_design_points]
