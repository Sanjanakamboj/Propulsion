"""3D visualization of a stacked blade (stacking.py's StackedBladeGeometry)
-- a wireframe of the upper/lower surfaces across all spanwise sections,
showing the resulting twist/lean/sweep shape.
"""


def plot_stacked_blade(geometry, ax=None, n_camberline_lines: int = 0):
    """geometry: a stacking.StackedBladeGeometry. Draws each section's
    upper/lower surface as a ring, plus lines connecting corresponding
    points hub-to-tip (showing the stacking/twist shape). Set
    n_camberline_lines > 0 to also draw evenly-spaced camberline profiles."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    for section in geometry.sections:
        x_u, y_u, z_u = section.upper
        x_l, y_l, z_l = section.lower
        ax.plot(x_u, y_u, z_u, color="#0571b0", lw=1.2)
        ax.plot(x_l, y_l, z_l, color="#c1272d", lw=1.2)

    # Hub-to-tip lines at a handful of chordwise stations, to show the twist.
    n_points = len(geometry.sections[0].upper[0])
    sample_indices = [0, n_points // 4, n_points // 2, 3 * n_points // 4, n_points - 1]
    for idx in sample_indices:
        xs = [s.upper[0][idx] for s in geometry.sections]
        ys = [s.upper[1][idx] for s in geometry.sections]
        zs = [s.upper[2][idx] for s in geometry.sections]
        ax.plot(xs, ys, zs, color="#9aa0a6", lw=0.8, ls="--")

    ax.set_xlabel("Axial  [m]")
    ax.set_ylabel("Tangential  [m]")
    ax.set_zlabel("Radius  [m]")
    ax.set_title("Stacked Blade Geometry", fontsize=13, fontweight="bold")
    return ax
