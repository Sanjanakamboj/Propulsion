"""Write a stacked 3D blade's full point cloud (stacking.py's
StackedBladeGeometry) to a flat CSV -- one row per point, tagged with which
section/surface it belongs to.
"""

import csv


def write_blade_csv(geometry, path: str) -> None:
    """geometry: a stacking.StackedBladeGeometry."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section_index", "span_fraction", "radius", "surface", "point_index", "x", "y", "z"])
        for section_index, section in enumerate(geometry.sections):
            for surface_name, (x, y, z) in (("upper", section.upper), ("lower", section.lower), ("camberline", section.camberline)):
                for point_index in range(len(x)):
                    writer.writerow([section_index, section.span_fraction, section.radius, surface_name, point_index, x[point_index], y[point_index], z[point_index]])
