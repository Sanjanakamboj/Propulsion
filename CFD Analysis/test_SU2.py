import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Blade Geometry Generator"))

from blade_section import BladeSectionInputs, build_blade_section  # noqa: E402

from SU2 import SU2CascadeInputs, plot_mesh_quality, write_su2_config

# Values matching the validated Turbine Stage Design.ipynb example (station 2/3).
NOTEBOOK_INPUTS = dict(
    gamma=1.33,
    gas_constant=287.0,
    relative_inlet_mach=0.3081,
    static_temperature=1553.73,
    static_pressure=1428997.01,
    inlet_total_temperature=1578.0609,
    inlet_total_pressure=1521338.1450,
    inlet_flow_angle_deg=42.6413,
    outlet_static_pressure=1107264.6278,
    reference_length=0.1493073856861406,
)


def test_write_su2_config_matches_notebook_marker_inlet_direction(tmp_path):
    inputs = SU2CascadeInputs(**NOTEBOOK_INPUTS)
    config_path = str(tmp_path / "blade_su2.cfg")
    write_su2_config(inputs, mesh_filename="blade_mesh.su2", config_path=config_path)

    text = (tmp_path / "blade_su2.cfg").read_text()
    assert "MACH_NUMBER= 0.3081" in text
    assert "FREESTREAM_TEMPERATURE= 1553.73" in text
    assert "FREESTREAM_PRESSURE= 1428997.01" in text
    assert "MARKER_OUTLET= (OUTLET, 1107264.6278)" in text
    assert "MESH_FILENAME= blade_mesh.su2" in text

    # Direction cosines should match cos(beta2)/sin(beta2) to the notebook's
    # own printed values (0.7356093167, 0.6774060327).
    dir_x = math.cos(math.radians(42.6413))
    dir_y = math.sin(math.radians(42.6413))
    assert dir_x == pytest.approx(0.7356093167, rel=1e-6)
    assert dir_y == pytest.approx(0.6774060327, rel=1e-6)
    assert f"{dir_x:.10f}" in text
    assert f"{dir_y:.10f}" in text


def test_config_contains_expected_boundary_markers(tmp_path):
    inputs = SU2CascadeInputs(**NOTEBOOK_INPUTS)
    config_path = str(tmp_path / "blade_su2.cfg")
    write_su2_config(inputs, mesh_filename="blade_mesh.su2", config_path=config_path)
    text = (tmp_path / "blade_su2.cfg").read_text()
    assert "MARKER_HEATFLUX= (WALL_MID, 0.0)" in text
    assert "MARKER_EULER= (WALL_UP, WALL_LO)" in text
    assert "SOLVER= RANS" in text
    assert "KIND_TURB_MODEL= SA" in text


def test_inner_iterations_default_and_override(tmp_path):
    inputs = SU2CascadeInputs(**NOTEBOOK_INPUTS)
    assert inputs.inner_iterations == 4000

    fast_inputs = SU2CascadeInputs(**{**NOTEBOOK_INPUTS, "inner_iterations": 50})
    config_path = str(tmp_path / "blade_su2.cfg")
    write_su2_config(fast_inputs, mesh_filename="blade_mesh.su2", config_path=config_path)
    text = (tmp_path / "blade_su2.cfg").read_text()
    assert "INNER_ITER= 50" in text


@pytest.mark.parametrize(
    "overrides",
    [
        dict(gamma=1.0),
        dict(gas_constant=0.0),
        dict(relative_inlet_mach=0.0),
        dict(static_temperature=0.0),
        dict(outlet_static_pressure=-1.0),
        dict(reference_length=0.0),
        dict(inner_iterations=0),
    ],
)
def test_invalid_inputs_raise(overrides):
    kwargs = dict(NOTEBOOK_INPUTS)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        SU2CascadeInputs(**kwargs)


def test_run_su2_raises_clear_error_when_binary_missing(tmp_path):
    from SU2 import run_su2

    with pytest.raises(FileNotFoundError):
        run_su2(str(tmp_path / "cfg.cfg"), str(tmp_path), su2_binary="/nonexistent/SU2_CFD")


def test_plot_mesh_quality_reads_a_solved_mesh(tmp_path):
    # Synthetic stand-in for a solved flow.vtu -- exercises the read/plot
    # logic without depending on the external SU2 binary in the test suite
    # (the real end-to-end SU2 run was verified manually).
    import pyvista as pv

    plane = pv.Plane(i_size=0.3, j_size=0.3, i_resolution=20, j_resolution=20).triangulate().cast_to_unstructured_grid()
    vtu_path = tmp_path / "flow.vtu"
    plane.save(str(vtu_path))

    design = BladeSectionInputs(
        stagger_angle_deg=39.0, beta_in_deg=42.6413, beta_out_deg=69.5316,
        le_radius_over_cx=0.05 * 0.1696 / 0.1493, te_radius_over_cx=0.0007 / 0.1493,
    )
    blade = build_blade_section(design)

    fig, axes, info = plot_mesh_quality(str(vtu_path), blade, axial_chord=0.1493, pitch=0.1696)
    assert info["n_points"] == plane.n_points
    assert info["n_cells"] > 0
    assert len(axes) == 2
