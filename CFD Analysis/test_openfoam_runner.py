import math
import os

import pytest

from boundary_conditions import INLET, OUTLET, WALL_LO, WALL_MID, WALL_UP
from mesh import build_cascade_domain
from openfoam_runner import (
    OpenFOAMCascadeInputs,
    _docker_exec,
    _dynamic_viscosity,
    _molecular_weight,
    build_extruded_cascade_mesh,
    patch_boundary_types,
    set_patch_types,
    write_openfoam_case,
)

GAMMA, R = 1.333, 287.0


@pytest.fixture
def inputs():
    return OpenFOAMCascadeInputs(
        gamma=GAMMA, gas_constant=R, relative_inlet_mach=0.2977,
        static_temperature=1553.73, static_pressure=410160.7,
        inlet_flow_angle_deg=42.9725, outlet_static_pressure=350000.0,
        reference_length=0.1493, reynolds_number=1.0e6, n_iterations=200,
    )


def test_inlet_velocity_matches_manual_formula(inputs):
    a = math.sqrt(GAMMA * R * 1553.73)
    V = 0.2977 * a
    Ux, Uy, Uz = inputs.inlet_velocity
    assert Ux == pytest.approx(V * math.cos(math.radians(42.9725)))
    assert Uy == pytest.approx(V * math.sin(math.radians(42.9725)))
    assert Uz == pytest.approx(0.0)


def test_molecular_weight_matches_manual_formula():
    assert _molecular_weight(287.0) == pytest.approx(8314.5 / 287.0)


def test_dynamic_viscosity_gives_the_target_reynolds_number(inputs):
    mu = _dynamic_viscosity(inputs)
    rho = inputs.static_pressure / (inputs.gas_constant * inputs.static_temperature)
    V = math.hypot(*inputs.inlet_velocity[:2])
    re = rho * V * inputs.reference_length / mu
    assert re == pytest.approx(inputs.reynolds_number, rel=1e-6)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(gamma=1.0),
        dict(gas_constant=0.0),
        dict(relative_inlet_mach=0.0),
        dict(static_temperature=0.0),
        dict(static_pressure=0.0),
        dict(reference_length=0.0),
        dict(n_iterations=0),
    ],
)
def test_openfoam_cascade_inputs_rejects_invalid_values(overrides):
    base = dict(
        gamma=GAMMA, gas_constant=R, relative_inlet_mach=0.2977,
        static_temperature=1553.73, static_pressure=410160.7,
        inlet_flow_angle_deg=42.9725, outlet_static_pressure=350000.0,
        reference_length=0.1493,
    )
    base.update(overrides)
    with pytest.raises(ValueError):
        OpenFOAMCascadeInputs(**base)


def test_write_openfoam_case_creates_all_expected_files(inputs, tmp_path):
    case_dir = str(tmp_path / "case")
    write_openfoam_case(inputs, case_dir)

    for rel_path in [
        "constant/thermophysicalProperties", "constant/turbulenceProperties",
        "system/controlDict", "system/fvSchemes", "system/fvSolution",
        "0/U", "0/p", "0/T", "0/k", "0/omega", "0/nut", "0/alphat",
    ]:
        assert os.path.isfile(os.path.join(case_dir, rel_path)), rel_path


def test_write_openfoam_case_U_file_contains_inlet_velocity(inputs, tmp_path):
    case_dir = str(tmp_path / "case")
    write_openfoam_case(inputs, case_dir)
    content = open(os.path.join(case_dir, "0", "U")).read()
    Ux, Uy, Uz = inputs.inlet_velocity
    assert f"{Ux:.6f}" in content
    assert INLET in content
    assert OUTLET in content
    assert WALL_MID in content


def test_write_openfoam_case_boundary_patches_use_shared_marker_names(inputs, tmp_path):
    case_dir = str(tmp_path / "case")
    write_openfoam_case(inputs, case_dir)
    p_content = open(os.path.join(case_dir, "0", "p")).read()
    assert INLET in p_content
    assert OUTLET in p_content
    assert WALL_UP in p_content
    assert WALL_LO in p_content


SAMPLE_BOUNDARY_FILE = """FoamFile
{
    version     2.0;
}
7
(
    BACK
    {
        type            patch;
        physicalType    patch;
        nFaces          100;
        startFace       0;
    }
    WALL_MID
    {
        type            patch;
        physicalType    patch;
        nFaces          50;
        startFace       100;
    }
    FRONT
    {
        type            patch;
        physicalType    patch;
        nFaces          100;
        startFace       150;
    }
)
"""


def test_set_patch_types_rewrites_type_and_drops_physical_type(tmp_path):
    path = tmp_path / "boundary"
    path.write_text(SAMPLE_BOUNDARY_FILE)
    set_patch_types(str(path), {"BACK": "empty"})
    content = path.read_text()
    assert "type            empty;" in content
    assert "BACK" in content
    # WALL_MID and FRONT blocks untouched
    assert content.count("type            patch;") == 2


def test_set_patch_types_rejects_missing_patch_name(tmp_path):
    path = tmp_path / "boundary"
    path.write_text(SAMPLE_BOUNDARY_FILE)
    with pytest.raises(ValueError):
        set_patch_types(str(path), {"NONEXISTENT": "empty"})


def test_patch_boundary_types_applies_empty_and_wall(tmp_path):
    path = tmp_path / "boundary"
    path.write_text(SAMPLE_BOUNDARY_FILE)
    patch_boundary_types(str(path))
    content = path.read_text()
    # BACK -> empty, WALL_MID -> wall
    back_block = content.split("BACK")[1].split("WALL_MID")[0]
    wall_block = content.split("WALL_MID")[1]
    assert "type            empty;" in back_block
    assert "type            wall;" in wall_block


def test_docker_exec_builds_expected_command(monkeypatch):
    captured = {}

    class FakeCompletedProcess:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd
        return FakeCompletedProcess()

    monkeypatch.setattr("subprocess.run", fake_run)
    rc, out = _docker_exec("openfoam", "checkMesh", work_dir="/data/case")
    assert rc == 0
    assert out == "ok"
    assert captured["cmd"][:3] == ["docker", "exec", "openfoam"]
    assert "cd /data/case && checkMesh" in captured["cmd"][-1]


@pytest.fixture(scope="module")
def turbine_domain():
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "Blade Geometry Generator"))
    from blade_section import BladeSectionInputs, build_blade_section

    design = BladeSectionInputs(
        stagger_angle_deg=39.0, beta_in_deg=42.6413, beta_out_deg=69.5316,
        le_radius_over_cx=0.05 * 0.1696 / 0.1493, te_radius_over_cx=0.0007 / 0.1493,
    )
    blade = build_blade_section(design)
    return build_cascade_domain(blade, axial_chord=0.1493, pitch=0.1696, beta_in_deg=42.6413, beta_out_deg=69.5316)


def test_build_extruded_cascade_mesh_writes_a_msh_file(turbine_domain, tmp_path):
    msh_path = build_extruded_cascade_mesh(turbine_domain, str(tmp_path))
    assert os.path.isfile(msh_path)
    assert os.path.getsize(msh_path) > 0


def test_build_extruded_cascade_mesh_contains_all_expected_physical_groups(turbine_domain, tmp_path):
    msh_path = build_extruded_cascade_mesh(turbine_domain, str(tmp_path))
    content = open(msh_path).read()
    for name in ("INLET", "OUTLET", "WALL_UP", "WALL_LO", "WALL_MID", "FRONT", "BACK", "FLUID"):
        assert f'"{name}"' in content
