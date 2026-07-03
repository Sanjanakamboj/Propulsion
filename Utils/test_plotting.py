import math

import matplotlib

matplotlib.use("Agg")

import pytest

from plotting import _bowtie_apex, multistage_hs_diagram, plot_parameter_table, plot_velocity_triangles


# Real values from the validated turbine mean-line example (station 2 uses
# the standard Vt=Wt+U convention, station 3 the reversed Vt=Wt-U one).
@pytest.mark.parametrize(
    "Wt, Vt, Vx, U",
    [
        (160.71, 509.73, 174.51, 349.02),
        (467.54, 118.52, 174.51, 349.02),
    ],
)
def test_bowtie_apex_satisfies_both_distance_constraints(Wt, Vt, Vx, U):
    apex = _bowtie_apex(Wt, Vt, Vx, U)
    W, V = math.hypot(Wt, Vx), math.hypot(Vt, Vx)
    assert math.hypot(apex[0], apex[1]) == pytest.approx(W, rel=1e-6)
    assert math.hypot(apex[0] - U, apex[1]) == pytest.approx(V, rel=1e-6)


def test_bowtie_apex_raises_if_no_consistent_sign():
    with pytest.raises(ValueError):
        _bowtie_apex(Wt=100.0, Vt=999999.0, Vx=175.0, U=349.0)


def test_plot_velocity_triangles_runs_without_error():
    ax = plot_velocity_triangles(Wt_in=160.71, Vt_in=509.73, Wt_out=467.54, Vt_out=118.52, Vx=174.51, U=349.02, label_in="2", label_out="3")
    assert ax.get_xlabel().startswith("Tangential")


def test_plot_parameter_table_runs_without_error():
    sections = [
        ("STATION 1", [("Static Pressure", "$P_1$", "Pa", "101325.00")]),
        ("SUMMARY", [("Blade Speed", "$U$", "m/s", "349.02")]),
    ]
    fig, ax = plot_parameter_table(sections, "Test Table")
    assert ax.get_title() == "Test Table"


# Hand-verified synthetic 2-stage compressor: T0=288.15K, P0=101325 Pa,
# stage pressure ratio 2.0 each, stage isentropic efficiency 0.85.
CP, GAMMA = 1005.0, 1.4
T0, P0 = 288.15, 101325.0
T1, P1 = 362.39562877531716, 202650.0
T2, P2 = 455.77161809980043, 405300.0
STAGES = [(T0, P0), (T1, P1), (T2, P2)]


def test_multistage_hs_diagram_real_points_match_input_states():
    ax, result = multistage_hs_diagram(STAGES, CP, GAMMA)
    assert [p.h for p in result.real_points] == pytest.approx([CP * T0, CP * T1, CP * T2])
    assert result.real_points[0].s == pytest.approx(0.0)
    assert result.real_points[1].s == pytest.approx(31.369349160317313, rel=1e-6)
    assert result.real_points[2].s == pytest.approx(62.73869832063451, rel=1e-6)


def test_multistage_hs_diagram_stage1_local_ideal_matches_hand_calc():
    ax, result = multistage_hs_diagram(STAGES, CP, GAMMA)
    stage1_ideal = result.local_ideal_points[0]
    assert stage1_ideal.label == "2s"
    assert stage1_ideal.s == pytest.approx(0.0)  # stage 1's local ideal shares state 1's entropy
    assert stage1_ideal.h == pytest.approx(CP * 351.2587844590196, rel=1e-6)


def test_multistage_hs_diagram_stage2_local_ideal_shares_entropy_with_real_state_2():
    ax, result = multistage_hs_diagram(STAGES, CP, GAMMA)
    stage2_ideal = result.local_ideal_points[1]
    assert stage2_ideal.label == "3s"
    assert stage2_ideal.s == pytest.approx(result.real_points[1].s, rel=1e-9)  # same entropy as real state 2
    assert stage2_ideal.h == pytest.approx(CP * 441.76521970112793, rel=1e-6)


def test_multistage_hs_diagram_overall_ideal_shares_entropy_with_state_1():
    ax, result = multistage_hs_diagram(STAGES, CP, GAMMA)
    overall_ideal = result.overall_ideal_points[0]
    assert overall_ideal.label == "3ss"
    assert overall_ideal.s == pytest.approx(0.0)  # same entropy column as state 1 and "2s"
    assert overall_ideal.h == pytest.approx(CP * 428.1892544148116, rel=1e-6)


def test_multistage_hs_diagram_reheat_factor_matches_hand_calc():
    ax, result = multistage_hs_diagram(STAGES, CP, GAMMA)
    assert result.reheat_factor == pytest.approx(1.017417408998721, rel=1e-6)


def test_multistage_hs_diagram_reheat_factor_is_one_for_a_single_stage():
    # With only one stage, local ideal == overall ideal by construction --
    # no reheat penalty is possible yet.
    ax, result = multistage_hs_diagram([(T0, P0), (T1, P1)], CP, GAMMA)
    assert result.reheat_factor == pytest.approx(1.0, rel=1e-9)
    assert result.overall_ideal_points == []


def test_multistage_hs_diagram_rejects_fewer_than_two_states():
    with pytest.raises(ValueError):
        multistage_hs_diagram([(T0, P0)], CP, GAMMA)


def test_multistage_hs_diagram_runs_without_error_for_many_stages():
    # A longer chain (e.g. a 10-stage compressor) shouldn't error out. Each
    # stage applies a real pressure ratio and isentropic efficiency < 1
    # (the same physically-consistent construction as the 2-stage case
    # above), not arbitrary multipliers -- otherwise the "real" state
    # isn't actually a valid lossy-compression outcome of the previous one.
    stage_pr, eta = 1.3, 0.85
    states = [(T0, P0)]
    T_prev, P_prev = T0, P0
    for _ in range(10):
        T_ideal = T_prev * stage_pr ** ((GAMMA - 1.0) / GAMMA)
        T_real = T_prev + (T_ideal - T_prev) / eta
        P_real = P_prev * stage_pr
        states.append((T_real, P_real))
        T_prev, P_prev = T_real, P_real
    ax, result = multistage_hs_diagram(states, CP, GAMMA)
    assert ax.get_xlabel().startswith("Entropy")
    assert len(result.real_points) == 11
    assert result.reheat_factor > 1.0
