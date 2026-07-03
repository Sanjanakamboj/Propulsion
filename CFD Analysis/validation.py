"""Compare CFD-predicted exit conditions against the mean-line design
targets that fed the CFD case in the first place (e.g. Turbine
Calculations' Mw3) -- a sanity check that the two levels of fidelity
roughly agree, using the same PASS/FAIL SanityCheck framework as
Compressor/Turbine Calculations' sanity_checks.py (Utils/sanity.py).

sample_near_plane is a simple point-based proxy for a plane average (mean
of all mesh points within `tolerance` of x=x_target) -- not a flux-weighted
area integral, which would be more rigorous but needs a plane-cut/
integration step this module doesn't attempt. Good enough to catch a
mean-line design that's grossly inconsistent with its own CFD case;
not a substitute for a proper mixed-out average.
"""

import sys
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent.parent / "Utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from sanity import check, format_sanity_report  # noqa: F401,E402

__all__ = ["sample_near_plane", "validate_against_mean_line", "format_sanity_report"]


def sample_near_plane(flow_vtu_path: str, field: str, x_target: float, tolerance: float) -> float:
    import pyvista as pv

    mesh = pv.read(flow_vtu_path)
    if field not in mesh.point_data:
        raise ValueError(f"field '{field}' not found in {flow_vtu_path} -- available: {list(mesh.point_data.keys())}")

    xs = mesh.points[:, 0]
    mask = (xs > x_target - tolerance) & (xs < x_target + tolerance)
    if not mask.any():
        raise ValueError(f"no mesh points found within {tolerance} of x={x_target}")
    return float(mesh.point_data[field][mask].mean())


def validate_against_mean_line(flow_vtu_path: str, x_target: float, tolerance: float, mean_line_targets: dict):
    """mean_line_targets: {field_name: (target_value, relative_tolerance)}.
    Returns a list of SanityCheck comparing each field's CFD-sampled value
    against target * (1 +- relative_tolerance)."""
    checks = []
    for field, (target, rel_tol) in mean_line_targets.items():
        cfd_value = sample_near_plane(flow_vtu_path, field, x_target, tolerance)
        low, high = target * (1.0 - rel_tol), target * (1.0 + rel_tol)
        checks.append(check(f"CFD vs mean-line: {field}", cfd_value, (low, high)))
    return checks
