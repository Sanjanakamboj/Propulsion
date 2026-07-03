"""3-blade cascade domain and mesh generation for CFD analysis -- adapted
from Turbine Stage Design.ipynb, Part 5a (geometry pre-check, no meshing
tool needed) and Part 5b (GMSH mesh generation).

The blade row is modelled as three blades in a row (no periodicity):
    WALL_MID = the middle blade (the one actually measured)
    WALL_UP  = pressure side of the blade above, offset by +pitch
    WALL_LO  = suction side of the blade below, offset by -pitch
    INLET/OUTLET = planes extended upstream/downstream along the flow angles
"""

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CascadeDomain:
    xs_suction: np.ndarray
    ys_suction: np.ndarray
    xs_pressure: np.ndarray
    ys_pressure: np.ndarray
    wall_upper: np.ndarray  # (N,2), pressure side of the blade above, +pitch, extended to inlet/outlet
    wall_lower: np.ndarray  # (N,2), suction side of the blade below, -pitch, extended to inlet/outlet
    camberline: np.ndarray  # (N,2), middle-blade camberline, extended to inlet/outlet
    le: tuple  # (x, y)
    te: tuple  # (x, y)
    x_in: float
    x_out: float
    pitch: float
    axial_chord: float


def build_cascade_domain(
    blade, axial_chord: float, pitch: float, beta_in_deg: float, beta_out_deg: float,
    n_surface: int = 600, extension_chords: float = 0.5,
) -> CascadeDomain:
    """blade: a Blade2DCamberThickness from Blade Geometry Generator's
    blade_section.py. beta_in_deg/beta_out_deg are the flow angles the inlet
    and outlet planes are extended along (e.g. beta2/beta3 for a turbine rotor)."""
    u = np.linspace(0.0, 1.0, n_surface)
    us = np.real(blade.get_upper_side_coordinates(u))
    ls = np.real(blade.get_lower_side_coordinates(u))
    xs_us, ys_us = us[0, :] * axial_chord, us[1, :] * axial_chord
    xs_ls, ys_ls = ls[0, :] * axial_chord, ls[1, :] * axial_chord
    le = (float(xs_us[-1]), float(ys_us[-1]))
    te = (float(xs_us[0]), float(ys_us[0]))

    wall_upper = np.column_stack([xs_ls, ys_ls + pitch])
    wall_lower = np.column_stack([xs_us[::-1], ys_us[::-1] - pitch])

    xg = np.linspace(le[0], te[0], 400)
    order_us, order_ls = np.argsort(xs_us), np.argsort(xs_ls)
    y_suc = np.interp(xg, xs_us[order_us], ys_us[order_us])
    y_pres = np.interp(xg, xs_ls[order_ls], ys_ls[order_ls])
    camberline = np.column_stack([xg, 0.5 * (y_suc + y_pres)])

    ext = extension_chords * axial_chord
    x_in, x_out = le[0] - ext, te[0] + ext
    slope_in = np.tan(np.radians(beta_in_deg))
    slope_out = -np.tan(np.radians(beta_out_deg))  # flow leaves in the opposite tangential sense

    def extend_upstream(p):
        return np.array([x_in, p[1] + slope_in * (x_in - p[0])])

    def extend_downstream(p):
        return np.array([x_out, p[1] + slope_out * (x_out - p[0])])

    wall_upper_ext = np.vstack([extend_upstream(wall_upper[0]), wall_upper, extend_downstream(wall_upper[-1])])
    wall_lower_ext = np.vstack([extend_upstream(wall_lower[0]), wall_lower, extend_downstream(wall_lower[-1])])
    camberline_ext = np.vstack([extend_upstream(camberline[0]), camberline, extend_downstream(camberline[-1])])

    return CascadeDomain(
        xs_suction=xs_us, ys_suction=ys_us, xs_pressure=xs_ls, ys_pressure=ys_ls,
        wall_upper=wall_upper_ext, wall_lower=wall_lower_ext, camberline=camberline_ext,
        le=le, te=te, x_in=x_in, x_out=x_out, pitch=pitch, axial_chord=axial_chord,
    )


def plot_cascade_domain(domain: CascadeDomain, ax=None):
    """Pure-matplotlib domain preview -- no meshing tool required, run this
    before generate_cascade_mesh to sanity-check the domain."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(domain.xs_suction, domain.ys_suction, "b-", lw=2, label="Middle blade -- suction")
    ax.plot(domain.xs_pressure, domain.ys_pressure, "r-", lw=2, label="Middle blade -- pressure")
    ax.plot(domain.wall_upper[:, 0], domain.wall_upper[:, 1], "r--", lw=1.5, label="WALL_UP (upper blade pressure side)")
    ax.plot(domain.wall_lower[:, 0], domain.wall_lower[:, 1], "b--", lw=1.5, label="WALL_LO (lower blade suction side)")
    ax.plot(domain.camberline[:, 0], domain.camberline[:, 1], "k:", lw=1, label="Camberline (extended)")
    ax.axvline(domain.x_in, color="gray", ls=":", lw=1)
    ax.axvline(domain.x_out, color="gray", ls=":", lw=1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("3-Blade Cascade Domain (pre-mesh check)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def generate_cascade_mesh(
    domain: CascadeDomain, output_dir: str, mesh_name: str = "blade_mesh",
    size_min_factor: float = 1.0 / 110, size_max_factor: float = 1.0 / 7,
    dist_min_factor: float = 0.03, dist_max_factor: float = 0.35, stride: int = 4,
):
    """GMSH 2D mesh of the cascade domain. Writes <output_dir>/<mesh_name>.su2
    and .msh. Returns (su2_path, msh_path, n_nodes, n_elements)."""
    import gmsh

    try:
        gmsh.finalize()
    except Exception:
        pass

    gmsh.initialize()
    gmsh.model.add("cascade_3blade")
    geo = gmsh.model.geo
    lc = domain.axial_chord / 8

    def add_points(arr):
        return [geo.addPoint(float(p[0]), float(p[1]), 0, lc) for p in arr]

    lw_f, uw_f = domain.wall_lower, domain.wall_upper
    le_x, le_y = domain.le
    te_x, te_y = domain.te

    p_li = geo.addPoint(float(lw_f[0, 0]), float(lw_f[0, 1]), 0, lc)
    p_lo = geo.addPoint(float(lw_f[-1, 0]), float(lw_f[-1, 1]), 0, lc)
    p_ui = geo.addPoint(float(uw_f[0, 0]), float(uw_f[0, 1]), 0, lc)
    p_uo = geo.addPoint(float(uw_f[-1, 0]), float(uw_f[-1, 1]), 0, lc)
    p_le = geo.addPoint(le_x, le_y, 0, lc)
    p_te = geo.addPoint(te_x, te_y, 0, lc)

    sp_lw = geo.addBSpline([p_li] + add_points(lw_f[1:-1:stride]) + [p_lo])
    sp_uw = geo.addBSpline([p_ui] + add_points(uw_f[1:-1:stride]) + [p_uo])

    n_surf = len(domain.xs_suction)
    sp_suc = geo.addBSpline(
        [p_te] + add_points(np.column_stack([domain.xs_suction, domain.ys_suction])[stride : n_surf - stride : stride]) + [p_le]
    )
    sp_pre = geo.addBSpline(
        [p_le] + add_points(np.column_stack([domain.xs_pressure, domain.ys_pressure])[stride : n_surf - stride : stride]) + [p_te]
    )

    l_inlet = geo.addLine(p_li, p_ui)
    l_outlet = geo.addLine(p_uo, p_lo)

    outer_cl = geo.addCurveLoop([sp_lw, -l_outlet, -sp_uw, -l_inlet])
    blade_cl = geo.addCurveLoop([-sp_suc, -sp_pre])
    surf = geo.addPlaneSurface([outer_cl, blade_cl])
    geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [l_inlet], name="INLET")
    gmsh.model.addPhysicalGroup(1, [l_outlet], name="OUTLET")
    gmsh.model.addPhysicalGroup(1, [sp_uw], name="WALL_UP")
    gmsh.model.addPhysicalGroup(1, [sp_lw], name="WALL_LO")
    gmsh.model.addPhysicalGroup(1, [sp_suc, sp_pre], name="WALL_MID")
    gmsh.model.addPhysicalGroup(2, [surf], name="FLUID")

    dist_f = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_f, "CurvesList", [sp_suc, sp_pre, sp_uw, sp_lw])
    gmsh.model.mesh.field.setNumber(dist_f, "Sampling", 400)
    thr_f = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thr_f, "InField", dist_f)
    gmsh.model.mesh.field.setNumber(thr_f, "SizeMin", domain.axial_chord * size_min_factor)
    gmsh.model.mesh.field.setNumber(thr_f, "SizeMax", domain.axial_chord * size_max_factor)
    gmsh.model.mesh.field.setNumber(thr_f, "DistMin", domain.axial_chord * dist_min_factor)
    gmsh.model.mesh.field.setNumber(thr_f, "DistMax", domain.axial_chord * dist_max_factor)
    gmsh.model.mesh.field.setAsBackgroundMesh(thr_f)

    gmsh.model.mesh.generate(2)
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    _, el_tags, _ = gmsh.model.mesh.getElements(dim=2)
    n_nodes = len(node_tags)
    n_elements = sum(len(t) for t in el_tags)

    os.makedirs(output_dir, exist_ok=True)
    su2_path = os.path.join(output_dir, f"{mesh_name}.su2")
    msh_path = os.path.join(output_dir, f"{mesh_name}.msh")
    gmsh.write(su2_path)
    gmsh.write(msh_path)
    gmsh.finalize()

    return su2_path, msh_path, n_nodes, n_elements
