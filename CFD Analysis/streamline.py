"""Streamline extraction and plotting from a solved flow.vtu -- the
cascade case is planar (2D, z=0 everywhere), so this uses pyvista's
evenly-spaced-2D streamline algorithm rather than the general 3D one.
"""


def compute_streamlines(flow_vtu_path: str, vectors: str = "Velocity", start_position=(0.0, 0.0, 0.0), separating_distance: float = 3.0, separating_distance_ratio: float = 0.5):
    """Returns the pyvista PolyData of evenly-spaced 2D streamlines."""
    import pyvista as pv

    mesh = pv.read(flow_vtu_path)
    if vectors not in mesh.point_data:
        raise ValueError(f"vector field '{vectors}' not found in {flow_vtu_path} -- available: {list(mesh.point_data.keys())}")

    return mesh.streamlines_evenly_spaced_2D(
        vectors=vectors, start_position=start_position,
        separating_distance=separating_distance, separating_distance_ratio=separating_distance_ratio,
    )


def iter_polylines(lines_array):
    """PolyData.lines is VTK's flat connectivity format: [n_points, id0,
    id1, ..., idN, n_points, id0, ...] -- one variable-length polyline at a
    time, not fixed-size segments. Yields each polyline's point-id array."""
    i = 0
    while i < len(lines_array):
        n = int(lines_array[i])
        yield lines_array[i + 1 : i + 1 + n]
        i += 1 + n


def plot_streamlines(flow_vtu_path: str, vectors: str = "Velocity", ax=None, color_by: str = None, **streamline_kwargs):
    """color_by: an optional scalar field name to color the streamlines by
    (e.g. 'Mach'); defaults to a single color if not given."""
    import matplotlib.pyplot as plt
    import numpy as np

    streamlines = compute_streamlines(flow_vtu_path, vectors=vectors, **streamline_kwargs)
    points = streamlines.points

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))

    if color_by is not None and color_by in streamlines.point_data:
        values = streamlines.point_data[color_by]
        segments, seg_values = [], []
        for ids in iter_polylines(streamlines.lines):
            for j in range(len(ids) - 1):
                p0, p1 = ids[j], ids[j + 1]
                segments.append([points[p0][:2], points[p1][:2]])
                seg_values.append(0.5 * (values[p0] + values[p1]))
        from matplotlib.collections import LineCollection

        lc = LineCollection(np.array(segments), cmap="viridis", array=np.array(seg_values), linewidths=1.2)
        ax.add_collection(lc)
        ax.figure.colorbar(lc, ax=ax, label=color_by)
        ax.autoscale()
    else:
        for ids in iter_polylines(streamlines.lines):
            ax.plot(points[ids, 0], points[ids, 1], color="#0571b0", lw=1.0)

    ax.set_aspect("equal")
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [m]")
    ax.set_title("Streamlines", fontsize=13, fontweight="bold")
    return ax
