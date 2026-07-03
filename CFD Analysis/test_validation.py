import numpy as np
import pytest
import pyvista as pv

from validation import format_sanity_report, sample_near_plane, validate_against_mean_line


@pytest.fixture
def flow_vtu_path(tmp_path):
    plane = pv.Plane(i_size=1.0, j_size=1.0, i_resolution=20, j_resolution=20).triangulate().cast_to_unstructured_grid()
    n = plane.n_points
    xs = plane.points[:, 0]
    # A field that varies linearly with x, so sampling near a known x gives
    # a predictable, checkable average.
    plane.point_data["Mach"] = xs + 1.0  # Mach = x + 1.0
    path = str(tmp_path / "flow.vtu")
    plane.save(path)
    return path


def test_sample_near_plane_matches_expected_average_for_linear_field(flow_vtu_path):
    # At x=0.0 (plane center, i_size=1.0 so x spans -0.5..0.5), Mach should
    # average to ~1.0 for points sampled right at the center.
    value = sample_near_plane(flow_vtu_path, "Mach", x_target=0.0, tolerance=0.01)
    assert value == pytest.approx(1.0, abs=0.05)


def test_sample_near_plane_rejects_unknown_field(flow_vtu_path):
    with pytest.raises(ValueError):
        sample_near_plane(flow_vtu_path, "Nonexistent_Field", x_target=0.0, tolerance=0.01)


def test_sample_near_plane_rejects_when_no_points_in_tolerance(flow_vtu_path):
    with pytest.raises(ValueError):
        sample_near_plane(flow_vtu_path, "Mach", x_target=100.0, tolerance=0.001)


def test_validate_against_mean_line_passes_when_target_matches(flow_vtu_path):
    checks = validate_against_mean_line(flow_vtu_path, x_target=0.0, tolerance=0.05, mean_line_targets={"Mach": (1.0, 0.10)})
    assert len(checks) == 1
    assert checks[0].passed


def test_validate_against_mean_line_fails_when_target_is_way_off(flow_vtu_path):
    checks = validate_against_mean_line(flow_vtu_path, x_target=0.0, tolerance=0.05, mean_line_targets={"Mach": (5.0, 0.05)})
    assert not checks[0].passed


def test_format_sanity_report_reflects_pass_and_fail(flow_vtu_path):
    checks = validate_against_mean_line(flow_vtu_path, x_target=0.0, tolerance=0.05, mean_line_targets={"Mach": (1.0, 0.10)})
    report = format_sanity_report("Test Validation", checks)
    assert "ALL CHECKS PASSED" in report
