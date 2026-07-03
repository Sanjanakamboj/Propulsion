"""Write a stacked 3D blade (stacking.py's StackedBladeGeometry) to an
ASCII STL file -- triangulating quad strips between corresponding points
on adjacent sections' upper/lower surfaces, plus a fan-triangulated end
cap at the hub and tip connecting upper to lower surface.

STL is a plain triangle-soup format (no CAD kernel needed, unlike STEP --
see this folder's other notes on why export_step.py isn't built). This is
NOT guaranteed perfectly watertight/manifold at the leading/trailing edges
(the upper and lower surface curves are separate parametrizations that may
not share exact endpoints) -- fine for visualization or as a starting
point for CAD import, not a substitute for manufacturing-grade geometry
cleanup.
"""

import numpy as np


def _triangle_normal(v1, v2, v3):
    n = np.cross(v2 - v1, v3 - v1)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0])
    return n / norm


def _surface_triangles(sections, surface_name: str):
    """Quad-strip triangulation between corresponding points on adjacent
    sections' given surface (upper or lower)."""
    triangles = []
    for i in range(len(sections) - 1):
        x0, y0, z0 = getattr(sections[i], surface_name)
        x1, y1, z1 = getattr(sections[i + 1], surface_name)
        n = len(x0)
        for j in range(n - 1):
            p00 = np.array([x0[j], y0[j], z0[j]])
            p01 = np.array([x0[j + 1], y0[j + 1], z0[j + 1]])
            p10 = np.array([x1[j], y1[j], z1[j]])
            p11 = np.array([x1[j + 1], y1[j + 1], z1[j + 1]])
            triangles.append((p00, p01, p10))
            triangles.append((p01, p11, p10))
    return triangles


def _end_cap_triangles(section, reverse: bool):
    """Fan triangulation connecting upper and lower surface at one section
    (hub or tip), closing that end of the blade. reverse flips winding so
    hub/tip caps face outward consistently."""
    x_u, y_u, z_u = section.upper
    x_l, y_l, z_l = section.lower
    n = len(x_u)
    triangles = []
    for j in range(n - 1):
        p_u0 = np.array([x_u[j], y_u[j], z_u[j]])
        p_u1 = np.array([x_u[j + 1], y_u[j + 1], z_u[j + 1]])
        p_l0 = np.array([x_l[j], y_l[j], z_l[j]])
        p_l1 = np.array([x_l[j + 1], y_l[j + 1], z_l[j + 1]])
        if reverse:
            triangles.append((p_u0, p_l0, p_u1))
            triangles.append((p_l0, p_l1, p_u1))
        else:
            triangles.append((p_u0, p_u1, p_l0))
            triangles.append((p_u1, p_l1, p_l0))
    return triangles


def blade_triangles(geometry):
    """All triangles (surface + end caps) for a StackedBladeGeometry, as a
    list of (v1, v2, v3) numpy-array 3-tuples."""
    if len(geometry.sections) < 2:
        raise ValueError("geometry needs at least 2 sections to triangulate")
    triangles = []
    triangles += _surface_triangles(geometry.sections, "upper")
    triangles += _surface_triangles(geometry.sections, "lower")
    triangles += _end_cap_triangles(geometry.sections[0], reverse=True)
    triangles += _end_cap_triangles(geometry.sections[-1], reverse=False)
    return triangles


def write_blade_stl(geometry, path: str, name: str = "blade") -> int:
    """Writes geometry to an ASCII STL file at path. Returns the triangle count."""
    triangles = blade_triangles(geometry)
    with open(path, "w") as f:
        f.write(f"solid {name}\n")
        for v1, v2, v3 in triangles:
            normal = _triangle_normal(v1, v2, v3)
            f.write(f"facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write("  outer loop\n")
            for v in (v1, v2, v3):
                f.write(f"    vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            f.write("  endloop\nendfacet\n")
        f.write(f"endsolid {name}\n")
    return len(triangles)
