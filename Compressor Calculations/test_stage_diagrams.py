import math

import matplotlib

matplotlib.use("Agg")

import pytest

from compressor import CompressorStageDesignInputs, design_compressor_stages, solve_compressor_stage
from stage_diagrams import (
    compressor_multistage_hs_diagram,
    compressor_stage_diagrams,
    compressor_stage_hs_ladder,
    compressor_stage_parameter_sections,
    plot_parameter_table,
    plot_velocity_triangles,
)

COMPRESSOR_GAMMA, COMPRESSOR_R = 1.4, 287.0
COMPRESSOR_CP = COMPRESSOR_GAMMA * COMPRESSOR_R / (COMPRESSOR_GAMMA - 1.0)


@pytest.fixture
def compressor_result():
    design = CompressorStageDesignInputs(
        stage_loading=0.35, flow_coefficient=0.5, degree_of_reaction=0.5,
        blade_speed_limit=350.0, rotational_speed_rpm=8000.0,
    )
    return solve_compressor_stage(
        T01=288.15, P01=101_325.0, specific_work_required=30_000.0,
        mass_flow=50.0, cp=COMPRESSOR_CP, gamma=COMPRESSOR_GAMMA, design=design,
    )


def test_velocity_triangle_vector_lengths_match_stored_magnitudes(compressor_result):
    for Wt, Vt, Vx, W, V in [
        (compressor_result.Wt1, compressor_result.Vt1, compressor_result.Cx, compressor_result.W1, compressor_result.V1),
        (compressor_result.Wt2, compressor_result.Vt2, compressor_result.Cx, compressor_result.W2, compressor_result.V2),
    ]:
        assert math.hypot(Wt, Vx) == pytest.approx(W, rel=1e-9)
        assert math.hypot(Vt, Vx) == pytest.approx(V, rel=1e-9)


def test_plot_velocity_triangles_runs_without_error(compressor_result):
    ax = plot_velocity_triangles(
        compressor_result.Wt1, compressor_result.Vt1, compressor_result.Wt2, compressor_result.Vt2,
        compressor_result.Cx, compressor_result.U, label_in="1", label_out="2",
    )
    assert ax.get_xlabel().startswith("Tangential")


def test_compressor_hs_ladder_ideal_branch_matches_station_1_entropy(compressor_result):
    ax = compressor_stage_hs_ladder(compressor_result, COMPRESSOR_CP, COMPRESSOR_GAMMA)
    assert ax.get_ylabel().startswith("Enthalpy")

    R = COMPRESSOR_CP * (COMPRESSOR_GAMMA - 1.0) / COMPRESSOR_GAMMA
    T1, P1 = compressor_result.T1, compressor_result.P1
    T01 = T1 + compressor_result.V1**2 / (2.0 * COMPRESSOR_CP)
    P01 = P1 * (T01 / T1) ** (COMPRESSOR_GAMMA / (COMPRESSOR_GAMMA - 1.0))

    def entropy(T, P):
        return COMPRESSOR_CP * math.log(T / T1) - R * math.log(P / P1)

    s_ideal = entropy(compressor_result.T03s, compressor_result.P03)
    # The isentropic branch shares station 1's entropy exactly (0), and the
    # real stage needs more enthalpy rise than ideal to reach the same P03.
    assert s_ideal == pytest.approx(0.0, abs=1e-6)
    assert compressor_result.T03 > compressor_result.T03s


def test_compressor_parameter_sections_cover_expected_stations(compressor_result):
    sections = compressor_stage_parameter_sections(compressor_result)
    headers = [header for header, _ in sections]
    assert any("STATION 1" in h for h in headers)
    assert any("STATION 2" in h for h in headers)
    assert any("STATION 3" in h for h in headers)
    assert any("SUMMARY" in h for h in headers)


def test_plot_parameter_table_runs_without_error(compressor_result):
    fig, ax = plot_parameter_table(compressor_stage_parameter_sections(compressor_result), "Test Table")
    assert ax.get_title() == "Test Table"


def test_compressor_stage_diagrams_saves_three_files(compressor_result, tmp_path):
    prefix = str(tmp_path / "compressor")
    fig_vt, fig_hs, fig_table = compressor_stage_diagrams(compressor_result, COMPRESSOR_CP, COMPRESSOR_GAMMA, save_prefix=prefix)
    assert (tmp_path / "compressor_velocity_triangles.png").exists()
    assert (tmp_path / "compressor_hs_diagram.png").exists()
    assert (tmp_path / "compressor_table.png").exists()


@pytest.fixture
def multistage_compressor():
    design = CompressorStageDesignInputs(
        stage_loading=0.35, flow_coefficient=0.5, degree_of_reaction=0.5,
        blade_speed_limit=350.0, rotational_speed_rpm=8000.0,
    )
    return design_compressor_stages(
        T01=288.15, P01=101_325.0, total_specific_work=300_000.0,
        mass_flow=50.0, cp=COMPRESSOR_CP, gamma=COMPRESSOR_GAMMA, design=design,
    )


def test_compressor_multistage_hs_diagram_real_path_matches_stage_chain(multistage_compressor):
    ax, result = compressor_multistage_hs_diagram(288.15, 101_325.0, multistage_compressor, COMPRESSOR_CP, COMPRESSOR_GAMMA)
    assert len(result.real_points) == len(multistage_compressor) + 1
    assert result.real_points[0].h == pytest.approx(COMPRESSOR_CP * 288.15)
    for i, stage in enumerate(multistage_compressor, start=1):
        assert result.real_points[i].h == pytest.approx(COMPRESSOR_CP * stage.T03, rel=1e-9)


def test_compressor_multistage_hs_diagram_shows_reheat_factor_above_one_for_a_real_compressor(multistage_compressor):
    # A real (lossy, stage_efficiency < 1) multi-stage compressor should
    # always show a genuine reheat penalty -- more total local-ideal work
    # is needed than the single overall isentropic path would require.
    ax, result = compressor_multistage_hs_diagram(288.15, 101_325.0, multistage_compressor, COMPRESSOR_CP, COMPRESSOR_GAMMA)
    assert result.reheat_factor > 1.0
    assert ax.get_xlabel().startswith("Entropy")
    assert "reheat factor" in ax.get_title().lower()
