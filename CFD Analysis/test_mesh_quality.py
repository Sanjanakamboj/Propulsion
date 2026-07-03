import matplotlib

matplotlib.use("Agg")

import pytest
import pyvista as pv

from mesh_quality import compute_quality_report, plot_quality_map


@pytest.fixture
def good_mesh_path(tmp_path):
    # A regular triangulated plane -- should score well on both measures.
    plane = pv.Plane(i_size=1.0, j_size=1.0, i_resolution=20, j_resolution=20).triangulate().cast_to_unstructured_grid()
    path = str(tmp_path / "mesh.vtu")
    plane.save(path)
    return path


def test_compute_quality_report_scaled_jacobian_runs_and_reports_reasonable_values(good_mesh_path):
    report = compute_quality_report(good_mesh_path, measure="scaled_jacobian")
    assert report.n_cells > 0
    assert 0.0 <= report.min <= report.mean <= report.max <= 1.0 + 1e-9


def test_compute_quality_report_aspect_ratio_runs_and_reports_reasonable_values(good_mesh_path):
    report = compute_quality_report(good_mesh_path, measure="aspect_ratio")
    assert report.n_cells > 0
    assert report.min >= 1.0  # aspect_ratio is bounded below by 1 (equilateral)


def test_regular_plane_mesh_has_no_poor_cells_under_default_limits(good_mesh_path):
    report = compute_quality_report(good_mesh_path, measure="scaled_jacobian")
    assert report.n_cells_poor == 0
    assert report.fraction_poor == pytest.approx(0.0)


def test_custom_limits_override_defaults(good_mesh_path):
    # An absurdly strict limit should flag every cell as poor.
    report = compute_quality_report(good_mesh_path, measure="scaled_jacobian", limits={"scaled_jacobian": (1.1, None)})
    assert report.n_cells_poor == report.n_cells
    assert report.fraction_poor == pytest.approx(1.0)


def test_fraction_poor_is_zero_when_no_cells():
    from mesh_quality import MeshQualityReport

    report = MeshQualityReport(measure="scaled_jacobian", min=0, max=0, mean=0, n_cells=0, n_cells_poor=0)
    assert report.fraction_poor == 0.0


def test_plot_quality_map_runs_without_error(good_mesh_path):
    ax, tpc = plot_quality_map(good_mesh_path, measure="scaled_jacobian")
    assert "scaled_jacobian" in ax.get_title()
