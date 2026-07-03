import matplotlib

matplotlib.use("Agg")

import pytest

from diagrams import build_engine_path, plot_cycle_diagrams
from engine import TurbojetDesignInputs, run_turbojet


@pytest.fixture
def cycle():
    design = TurbojetDesignInputs(
        ambient_T=288.15,
        ambient_P=101_325.0,
        flight_mach=0.0,
        mdot_air=50.0,
        compressor_pressure_ratio=15.0,
        turbine_inlet_temperature=1500.0,
    )
    return run_turbojet(design)


def test_build_engine_path_returns_equal_length_arrays(cycle):
    results, records = cycle
    T, P, s, v, stage_tags, markers = build_engine_path(records)
    assert len(T) == len(P) == len(s) == len(v) == len(stage_tags)
    assert len(T) > 0


def test_markers_cover_all_named_stations(cycle):
    results, records = cycle
    _, _, _, _, _, markers = build_engine_path(records)
    assert set(markers.keys()) == {"0", "2", "3", "4", "5", "9"}


def test_entropy_never_decreases_along_the_path(cycle):
    results, records = cycle
    _, _, s, _, _, _ = build_engine_path(records)
    assert all(b >= a - 1e-9 for a, b in zip(s, s[1:]))


def test_stage_tags_appear_in_engine_order(cycle):
    results, records = cycle
    _, _, _, _, stage_tags, _ = build_engine_path(records)
    seen_order = []
    for tag in stage_tags:
        if not seen_order or seen_order[-1] != tag:
            seen_order.append(tag)
    assert seen_order == ["Inlet", "Compressor", "Combustor", "Turbine", "Nozzle"]


def test_plot_cycle_diagrams_runs_without_error(cycle, tmp_path):
    results, records = cycle
    save_path = tmp_path / "diagrams.png"
    fig, (ax_ts, ax_pv) = plot_cycle_diagrams(records, save_path=str(save_path))
    assert save_path.exists()
    assert ax_ts.get_xlabel().startswith("Specific entropy")
    assert ax_pv.get_xlabel().startswith("Specific volume")
