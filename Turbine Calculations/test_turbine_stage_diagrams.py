import math

import matplotlib

matplotlib.use("Agg")

import pytest

from turbine import TurbineStageDesignInputs, solve_turbine_stage
from turbine_stage_diagrams import (
    plot_parameter_table,
    plot_velocity_triangles,
    turbine_stage_diagrams,
    turbine_stage_hs_ladder,
    turbine_stage_parameter_sections,
)

TURBINE_GAMMA, TURBINE_R = 1.33, 287.0
TURBINE_CP = TURBINE_GAMMA * TURBINE_R / (TURBINE_GAMMA - 1.0)


@pytest.fixture
def turbine_result():
    # Matches the validated notebook example.
    design = TurbineStageDesignInputs(
        stage_loading=1.8, flow_coefficient=0.5, degree_of_reaction=0.35,
        blade_speed_limit=350.0, rotational_speed_rpm=3000.0,
    )
    return solve_turbine_stage(
        T01=1679.21, P01=2_056_992.49, specific_work_required=219_270.0,
        mass_flow=750.0, cp=TURBINE_CP, gamma=TURBINE_GAMMA, design=design,
    )


def test_velocity_triangle_vector_lengths_match_stored_magnitudes(turbine_result):
    for Wt, Vt, Vx, W, V in [
        (turbine_result.Wt2, turbine_result.Vt2, turbine_result.Vx, turbine_result.W2, turbine_result.V2),
        (turbine_result.Wt3, turbine_result.Vt3, turbine_result.Vx, turbine_result.W3, turbine_result.V3),
    ]:
        assert math.hypot(Wt, Vx) == pytest.approx(W, rel=1e-9)
        assert math.hypot(Vt, Vx) == pytest.approx(V, rel=1e-9)


def test_plot_velocity_triangles_runs_without_error(turbine_result):
    ax = plot_velocity_triangles(
        turbine_result.Wt2, turbine_result.Vt2, turbine_result.Wt3, turbine_result.Vt3,
        turbine_result.Vx, turbine_result.U, label_in="2", label_out="3",
    )
    assert ax.get_xlabel().startswith("Tangential")


def test_turbine_hs_ladder_stator_and_rotor_generate_entropy(turbine_result):
    ax = turbine_stage_hs_ladder(turbine_result, TURBINE_CP, TURBINE_GAMMA)
    assert ax.get_ylabel().startswith("Enthalpy")

    # Recompute the entropy values the ladder plots, to check monotonicity
    # (real stator/rotor losses must generate entropy from station to station).
    R = TURBINE_CP * (TURBINE_GAMMA - 1.0) / TURBINE_GAMMA
    T1, P1 = turbine_result.T1, turbine_result.P1

    def entropy(T, P):
        return TURBINE_CP * math.log(T / T1) - R * math.log(P / P1)

    s1 = entropy(T1, P1)
    s2 = entropy(turbine_result.T2, turbine_result.P2)
    s3 = entropy(turbine_result.T3, turbine_result.P3)
    assert s1 == pytest.approx(0.0, abs=1e-9)
    assert s2 > s1
    assert s3 > s2


def test_turbine_parameter_sections_cover_expected_stations(turbine_result):
    sections = turbine_stage_parameter_sections(turbine_result)
    headers = [header for header, _ in sections]
    assert any("STATION 1" in h for h in headers)
    assert any("STATION 2" in h for h in headers)
    assert any("STATION 3" in h for h in headers)
    assert any("SUMMARY" in h for h in headers)
    for _, entries in sections:
        for parameter, symbol, unit, value in entries:
            assert isinstance(parameter, str) and parameter
            assert isinstance(value, str) and value


def test_plot_parameter_table_runs_without_error(turbine_result):
    fig, ax = plot_parameter_table(turbine_stage_parameter_sections(turbine_result), "Test Table")
    assert ax.get_title() == "Test Table"


def test_turbine_stage_diagrams_saves_three_files(turbine_result, tmp_path):
    prefix = str(tmp_path / "turbine")
    fig_vt, fig_hs, fig_table = turbine_stage_diagrams(turbine_result, TURBINE_CP, TURBINE_GAMMA, save_prefix=prefix)
    assert (tmp_path / "turbine_velocity_triangles.png").exists()
    assert (tmp_path / "turbine_hs_diagram.png").exists()
    assert (tmp_path / "turbine_table.png").exists()
