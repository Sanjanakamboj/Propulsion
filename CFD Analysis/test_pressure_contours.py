import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import pyvista as pv

from pressure_contours import plot_cp_contours, plot_field_contours, plot_pressure_contours


@pytest.fixture
def flow_vtu_path(tmp_path):
    # Synthetic stand-in for a solved flow.vtu -- exercises the read/plot
    # logic without depending on the external SU2 binary in the test suite
    # (same pattern SU2.py's own tests already use).
    plane = pv.Plane(i_size=0.3, j_size=0.3, i_resolution=20, j_resolution=20).triangulate().cast_to_unstructured_grid()
    n = plane.n_points
    plane.point_data["Pressure"] = np.linspace(100000.0, 150000.0, n)
    plane.point_data["Pressure_Coefficient"] = np.linspace(-1.0, 1.0, n)
    path = str(tmp_path / "flow.vtu")
    plane.save(path)
    return path


def test_plot_field_contours_runs_without_error(flow_vtu_path):
    ax, contour = plot_field_contours(flow_vtu_path, field="Pressure")
    assert "Pressure" in ax.get_title()


def test_plot_pressure_contours_runs_without_error(flow_vtu_path):
    ax, contour = plot_pressure_contours(flow_vtu_path)
    assert ax.get_title() == "Pressure Contours"


def test_plot_cp_contours_runs_without_error(flow_vtu_path):
    ax, contour = plot_cp_contours(flow_vtu_path)
    assert "Pressure_Coefficient" in ax.get_title()


def test_plot_field_contours_rejects_unknown_field(flow_vtu_path):
    with pytest.raises(ValueError):
        plot_field_contours(flow_vtu_path, field="Nonexistent_Field")
