"""2D turbomachinery blade section (camberline + thickness parametrization),
generated via the third-party ParaBlade library bundled in
parablade-master/ -- adapted from Turbine Stage Design.ipynb's blade
generation cell.
"""

import os
import sys
from dataclasses import dataclass, field

_PARABLADE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parablade-master", "parablade")
if _PARABLADE_DIR not in sys.path:
    sys.path.insert(0, _PARABLADE_DIR)

try:
    import numpy as np
    from blade_2D_camber_thickness import Blade2DCamberThickness
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        f"blade_section.py requires numpy and the bundled ParaBlade library at {_PARABLADE_DIR}"
    ) from exc

DEFAULT_THICKNESS_UPPER = (0.15, 0.20, 0.13, 0.07, 0.03, 0.02)
DEFAULT_THICKNESS_LOWER = (0.15, 0.20, 0.13, 0.07, 0.03, 0.02)


@dataclass(frozen=True)
class BladeSectionInputs:
    stagger_angle_deg: float
    beta_in_deg: float  # inlet metal angle (theta_in)
    beta_out_deg: float  # exit metal angle magnitude; ParaBlade wants it signed opposite theta_in
    le_radius_over_cx: float
    te_radius_over_cx: float
    thickness_upper: tuple = field(default_factory=lambda: DEFAULT_THICKNESS_UPPER)
    thickness_lower: tuple = field(default_factory=lambda: DEFAULT_THICKNESS_LOWER)

    def __post_init__(self):
        if not (0.0 <= self.stagger_angle_deg < 90.0):
            raise ValueError("stagger_angle_deg must be in [0, 90)")
        if self.le_radius_over_cx <= 0.0 or self.te_radius_over_cx <= 0.0:
            raise ValueError("le_radius_over_cx and te_radius_over_cx must be > 0")
        if len(self.thickness_upper) != 6 or len(self.thickness_lower) != 6:
            raise ValueError("thickness_upper/thickness_lower must each have 6 control values")


def build_blade_section(design: BladeSectionInputs, n_samples: int = 1000) -> Blade2DCamberThickness:
    """Returns a Blade2DCamberThickness with its section already sampled,
    ready for .get_upper_side_coordinates / .plot_blade_section / etc."""
    flow_dir_x = float(np.cos(np.deg2rad(design.beta_in_deg)))
    flow_dir_y = float(np.sin(np.deg2rad(design.beta_in_deg)))

    section_variables = dict(
        stagger=-design.stagger_angle_deg,
        theta_in=design.beta_in_deg,
        theta_out=-design.beta_out_deg,
        radius_in=design.le_radius_over_cx,
        radius_out=design.te_radius_over_cx,
        dist_in=flow_dir_x,
        dist_out=flow_dir_y,
        thickness_upper_1=design.thickness_upper[0],
        thickness_upper_2=design.thickness_upper[1],
        thickness_upper_3=design.thickness_upper[2],
        thickness_upper_4=design.thickness_upper[3],
        thickness_upper_5=design.thickness_upper[4],
        thickness_upper_6=design.thickness_upper[5],
        thickness_lower_1=design.thickness_lower[0],
        thickness_lower_2=design.thickness_lower[1],
        thickness_lower_3=design.thickness_lower[2],
        thickness_lower_4=design.thickness_lower[3],
        thickness_lower_5=design.thickness_lower[4],
        thickness_lower_6=design.thickness_lower[5],
    )
    section_variables = {key: np.asarray(value) for key, value in section_variables.items()}

    u = np.linspace(0.0, 1.0, n_samples)
    blade = Blade2DCamberThickness(section_variables)
    blade.get_section_coordinates(u)
    blade.check_analytic_curvature()
    return blade


def upper_surface_coordinates(blade: Blade2DCamberThickness, axial_chord: float, n: int = 600):
    """Dimensional (x, y) suction-surface coordinates, scaled by axial_chord."""
    u = np.linspace(0.0, 1.0, n)
    xy = np.real(blade.get_upper_side_coordinates(u))
    return xy[0, :] * axial_chord, xy[1, :] * axial_chord


def lower_surface_coordinates(blade: Blade2DCamberThickness, axial_chord: float, n: int = 600):
    """Dimensional (x, y) pressure-surface coordinates, scaled by axial_chord."""
    u = np.linspace(0.0, 1.0, n)
    xy = np.real(blade.get_lower_side_coordinates(u))
    return xy[0, :] * axial_chord, xy[1, :] * axial_chord


def camberline_coordinates(blade: Blade2DCamberThickness, axial_chord: float, n: int = 600):
    """Dimensional (x, y) camberline coordinates, scaled by axial_chord."""
    u = np.linspace(0.0, 1.0, n)
    xy = np.real(blade.get_camberline_coordinates(u))
    return xy[0, :] * axial_chord, xy[1, :] * axial_chord
