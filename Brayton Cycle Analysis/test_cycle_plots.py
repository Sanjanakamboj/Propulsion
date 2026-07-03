import math

import matplotlib

matplotlib.use("Agg")

import pytest

from config import AIR
from cycle_plots import build_cycle_path, plot_cycle_diagrams, plot_pv_diagram, plot_ts_diagram
from ideal_cycle import IdealCycleInputs, run_ideal_cycle
from intercooling import IntercoolingCycleInputs, run_intercooling_cycle
from real_cycle import RealCycleInputs, run_real_cycle
from regenerative_cycle import RegenerativeCycleInputs, run_regenerative_cycle
from reheating import ReheatingCycleInputs, run_reheating_cycle


@pytest.fixture
def ideal_states():
    return run_ideal_cycle(IdealCycleInputs(T1=288.15, P1=101_325.0, pressure_ratio=8.0, T3=1400.0, gas=AIR)).states


def test_build_cycle_path_covers_every_state_label(ideal_states):
    _, _, _, _, markers = build_cycle_path(ideal_states, AIR)
    for state in ideal_states:
        assert state.label in markers
        assert markers[state.label][0] == pytest.approx(state.T)
        assert markers[state.label][1] == pytest.approx(state.P)


def test_closed_cycle_returns_to_starting_entropy(ideal_states):
    # A genuinely closed cycle must return to the same (T, P) at state 1,
    # so its entropy (a state function) must return to exactly 0.
    T, _, s, _, _ = build_cycle_path(ideal_states, AIR)
    assert s[-1] == pytest.approx(0.0, abs=1e-9)
    assert T[-1] == pytest.approx(ideal_states[0].T)


def test_plot_ts_diagram_runs_without_error(ideal_states):
    ax = plot_ts_diagram(ideal_states, AIR)
    assert ax.get_ylabel().startswith("Temperature")


def test_plot_pv_diagram_runs_without_error(ideal_states):
    ax = plot_pv_diagram(ideal_states, AIR)
    assert ax.get_ylabel().startswith("Pressure")


def test_plot_cycle_diagrams_saves_file(ideal_states, tmp_path):
    save_path = str(tmp_path / "cycle.png")
    plot_cycle_diagrams(ideal_states, AIR, save_path=save_path)
    assert (tmp_path / "cycle.png").exists()


@pytest.mark.parametrize(
    "states",
    [
        run_real_cycle(RealCycleInputs(T1=288.15, P1=101_325.0, pressure_ratio=8.0, T3=1400.0, gas=AIR)).states,
        run_regenerative_cycle(RegenerativeCycleInputs(T1=288.15, P1=101_325.0, pressure_ratio=8.0, T4=1400.0, gas=AIR)).states,
        run_intercooling_cycle(IntercoolingCycleInputs(T1=288.15, P1=101_325.0, pressure_ratio_lp=3.0, pressure_ratio_hp=3.0, T_intercool_exit=288.15, T5=1400.0, gas=AIR)).states,
        run_reheating_cycle(ReheatingCycleInputs(T1=288.15, P1=101_325.0, pressure_ratio=9.0, T3=1400.0, pressure_ratio_hp_turbine=3.0, T5=1400.0, gas=AIR)).states,
    ],
)
def test_every_variant_closes_and_plots_without_error(states):
    _, _, s, _, markers = build_cycle_path(states, AIR)
    assert s[-1] == pytest.approx(0.0, abs=1e-9)
    assert len(markers) == len(states)
    ax = plot_ts_diagram(states, AIR)
    assert ax is not None
