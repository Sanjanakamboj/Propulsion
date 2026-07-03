"""Field contour plots (pressure, pressure coefficient, Mach, etc.) from a
solved flow.vtu -- same read/triangulate approach as SU2.py's
plot_mesh_quality, generalized to any scalar field SU2 writes to the
solution (Pressure, Pressure_Coefficient, Mach, Temperature, ...).
"""


def plot_field_contours(flow_vtu_path: str, field: str = "Pressure", ax=None, n_levels: int = 20, cmap: str = "viridis"):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import pyvista as pv

    mesh = pv.read(flow_vtu_path)
    mesh_tri = mesh.triangulate()
    if field not in mesh_tri.point_data:
        raise ValueError(f"field '{field}' not found in {flow_vtu_path} -- available: {list(mesh_tri.point_data.keys())}")

    cells = mesh_tri.cells.reshape(-1, 4)
    tris = cells[:, 1:]
    x, y = mesh_tri.points[:, 0], mesh_tri.points[:, 1]
    triang = mtri.Triangulation(x, y, tris)
    values = mesh_tri.point_data[field]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))

    contour = ax.tricontourf(triang, values, levels=n_levels, cmap=cmap)
    ax.figure.colorbar(contour, ax=ax, label=field)
    ax.set_aspect("equal")
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [m]")
    ax.set_title(f"{field} Contours", fontsize=13, fontweight="bold")
    return ax, contour


def plot_pressure_contours(flow_vtu_path: str, ax=None, n_levels: int = 20):
    return plot_field_contours(flow_vtu_path, field="Pressure", ax=ax, n_levels=n_levels, cmap="viridis")


def plot_cp_contours(flow_vtu_path: str, ax=None, n_levels: int = 20):
    """Pressure coefficient (Cp) -- SU2 already computes this field directly
    (Pressure_Coefficient), no extra post-processing needed."""
    return plot_field_contours(flow_vtu_path, field="Pressure_Coefficient", ax=ax, n_levels=n_levels, cmap="coolwarm")
