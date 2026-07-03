import matplotlib

matplotlib.use("Agg")

import pytest

from choking import assess_choking
from converging import solve_converging_nozzle
from converging_diverging import solve_cd_nozzle_design_matched
from plots import nozzle_diagrams, nozzle_parameter_sections, plot_thrust_breakdown
from thrust import compute_thrust

GAMMA = 1.333
CP = 1148.0
R = CP * (GAMMA - 1.0) / GAMMA
T0_IN, P0_IN, P_AMBIENT = 1340.4660176240825, 271982.75895383774, 22632.040095007793
ETA = 0.98
MDOT_AIR, V0, FUEL_FLOW = 1.0, 241.99490680590773, 0.02794540232049333


@pytest.fixture
def exit_state():
    return solve_converging_nozzle(T0_IN, P0_IN, P_AMBIENT, CP, GAMMA, R, ETA)


@pytest.fixture
def choke_assessment():
    return assess_choking(P0_IN, P_AMBIENT, GAMMA)


@pytest.fixture
def thrust_breakdown(exit_state):
    mdot_gas = MDOT_AIR + FUEL_FLOW
    rho_exit = exit_state.P_exit / (R * exit_state.T_exit)
    return compute_thrust(mdot_gas, exit_state.V_exit, exit_state.P_exit, P_AMBIENT, rho_exit, MDOT_AIR, V0, FUEL_FLOW)


def test_nozzle_parameter_sections_cover_expected_headers(exit_state, choke_assessment, thrust_breakdown):
    sections = nozzle_parameter_sections(exit_state, choke_assessment, thrust_breakdown)
    headers = [header for header, _ in sections]
    assert any("EXIT CONDITIONS" in h for h in headers)
    assert any("CHOKING" in h for h in headers)
    assert any("THRUST" in h for h in headers)
    assert not any("C-D NOZZLE" in h for h in headers)


def test_nozzle_parameter_sections_includes_area_ratio_when_given(exit_state, choke_assessment, thrust_breakdown):
    cd_result = solve_cd_nozzle_design_matched(T0_IN, P0_IN, P_AMBIENT, GAMMA, R, ETA)
    sections = nozzle_parameter_sections(exit_state, choke_assessment, thrust_breakdown, area_ratio=cd_result.area_ratio)
    headers = [header for header, _ in sections]
    assert any("C-D NOZZLE" in h for h in headers)


def test_plot_thrust_breakdown_runs_without_error(thrust_breakdown):
    ax = plot_thrust_breakdown(thrust_breakdown)
    assert ax.get_title() == "Thrust Breakdown"


def test_nozzle_diagrams_saves_two_files(exit_state, choke_assessment, thrust_breakdown, tmp_path):
    prefix = str(tmp_path / "nozzle")
    fig_table, fig_thrust = nozzle_diagrams(exit_state, choke_assessment, thrust_breakdown, save_prefix=prefix)
    assert (tmp_path / "nozzle_table.png").exists()
    assert (tmp_path / "nozzle_thrust_breakdown.png").exists()
