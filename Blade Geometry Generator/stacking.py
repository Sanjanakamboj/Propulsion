"""Stack a list of 2D spanwise sections (blade_sections.py) into a 3D
blade point cloud.

Each section's 2D surface coordinates (in its own local axial/tangential
frame) become one (x, y, z) "slice" of the 3D blade, with z = radius (the
span direction) and optional per-span x/y offsets from sweep.py/lean.py.
This treats each spanwise cut as a flat planar section stacked along the
span -- a standard simplified representation for preliminary blade
geometry, not a true cylindrical-surface unwrap (which would additionally
need the section's tangential coordinate divided by its own local radius
to get an angular position). That refinement matters for a
manufacturing-ready model; it's not needed for the mean-line-driven shape
this toolkit produces.
"""

from dataclasses import dataclass

import numpy as np

from blade_section import camberline_coordinates, lower_surface_coordinates, upper_surface_coordinates


@dataclass(frozen=True)
class StackedSection:
    radius: float
    span_fraction: float
    upper: tuple  # (x, y, z) arrays
    lower: tuple
    camberline: tuple


@dataclass(frozen=True)
class StackedBladeGeometry:
    sections: list  # list[StackedSection], hub to tip


def stack_sections(spanwise_sections, n_points_per_surface: int = 200, lean_offset=None, sweep_offset=None) -> StackedBladeGeometry:
    """spanwise_sections: output of blade_sections.generate_spanwise_sections.
    lean_offset/sweep_offset: callables span_fraction -> offset (m); default
    to zero (no lean/sweep, a radially-stacked blade)."""
    if lean_offset is None:
        lean_offset = lambda f: 0.0  # noqa: E731
    if sweep_offset is None:
        sweep_offset = lambda f: 0.0  # noqa: E731

    stacked = []
    for section in spanwise_sections:
        dx = sweep_offset(section.span_fraction)
        dy = lean_offset(section.span_fraction)
        z_value = section.radius

        def to_3d(x, y):
            z = np.full_like(x, z_value)
            return (x + dx, y + dy, z)

        x_u, y_u = upper_surface_coordinates(section.blade, section.axial_chord, n=n_points_per_surface)
        x_l, y_l = lower_surface_coordinates(section.blade, section.axial_chord, n=n_points_per_surface)
        x_c, y_c = camberline_coordinates(section.blade, section.axial_chord, n=n_points_per_surface)

        stacked.append(
            StackedSection(
                radius=section.radius,
                span_fraction=section.span_fraction,
                upper=to_3d(x_u, y_u),
                lower=to_3d(x_l, y_l),
                camberline=to_3d(x_c, y_c),
            )
        )
    return StackedBladeGeometry(sections=stacked)
