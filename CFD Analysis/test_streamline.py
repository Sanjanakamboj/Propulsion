import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import pyvista as pv

from streamline import compute_streamlines, iter_polylines, plot_streamlines


@pytest.fixture
def flow_vtu_path(tmp_path):
    # Synthetic uniform +x flow field -- streamlines should come out as
    # straight horizontal lines, an easy correctness check.
    plane = pv.Plane(i_size=1.0, j_size=1.0, i_resolution=20, j_resolution=20).triangulate().cast_to_unstructured_grid()
    n = plane.n_points
    velocity = np.zeros((n, 3))
    velocity[:, 0] = 1.0
    plane.point_data["Velocity"] = velocity
    plane.point_data["Mach"] = np.full(n, 0.5)
    path = str(tmp_path / "flow.vtu")
    plane.save(path)
    return path


def test_iter_polylines_parses_vtk_flat_connectivity_format():
    # [n_points, id0, id1, n_points, id0, id1, id2]
    lines_array = np.array([2, 0, 1, 3, 2, 3, 4])
    polylines = list(iter_polylines(lines_array))
    assert len(polylines) == 2
    assert list(polylines[0]) == [0, 1]
    assert list(polylines[1]) == [2, 3, 4]


def test_compute_streamlines_produces_lines_covering_all_points(flow_vtu_path):
    streamlines = compute_streamlines(flow_vtu_path, start_position=(-0.4, 0.0, 0.0))
    assert streamlines.n_lines > 0
    total_points_in_lines = sum(len(ids) for ids in iter_polylines(streamlines.lines))
    assert total_points_in_lines == streamlines.n_points


def test_uniform_flow_produces_horizontal_streamlines(flow_vtu_path):
    streamlines = compute_streamlines(flow_vtu_path, start_position=(-0.4, 0.0, 0.0))
    points = streamlines.points
    for ids in iter_polylines(streamlines.lines):
        ys = points[ids, 1]
        assert ys.max() - ys.min() < 1e-6  # straight horizontal line for uniform +x flow


def test_plot_streamlines_runs_without_error(flow_vtu_path):
    ax = plot_streamlines(flow_vtu_path, start_position=(-0.4, 0.0, 0.0))
    assert ax.get_title() == "Streamlines"


def test_plot_streamlines_with_color_by_runs_without_error(flow_vtu_path):
    ax = plot_streamlines(flow_vtu_path, start_position=(-0.4, 0.0, 0.0), color_by="Mach")
    assert ax.get_title() == "Streamlines"


def test_compute_streamlines_rejects_unknown_vector_field(flow_vtu_path):
    with pytest.raises(ValueError):
        compute_streamlines(flow_vtu_path, vectors="Nonexistent_Field")
