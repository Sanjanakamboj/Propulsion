import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from mission import FlightEnvelopePoint
from off_design import sweep_off_design
from plots import plot_efficiencies_vs, plot_thrust_vs, plot_tsfc_vs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Brayton Cycle Analysis"))
from engine import TurbojetDesignInputs  # noqa: E402


@pytest.fixture(scope="module")
def swept_points():
    design = TurbojetDesignInputs(
        ambient_T=216.65, ambient_P=22632.04, flight_mach=0.82, mdot_air=137.4,
        compressor_pressure_ratio=24.0, compressor_efficiency=0.87,
        turbine_inlet_temperature=1700.0, turbine_efficiency=0.90,
        combustor_pressure_loss_frac=0.04, combustor_efficiency=0.99, nozzle_efficiency=0.98,
    )
    envelope = [FlightEnvelopePoint(mach=0.82, altitude_m=a) for a in (0.0, 5000.0, 11000.0)]
    return sweep_off_design(design, envelope)


def test_plot_thrust_vs_altitude_runs_without_error(swept_points):
    ax = plot_thrust_vs(swept_points, x_attr="altitude_m")
    assert ax.get_title() == "Thrust Lapse"


def test_plot_tsfc_vs_altitude_runs_without_error(swept_points):
    ax = plot_tsfc_vs(swept_points, x_attr="altitude_m")
    assert "TSFC" in ax.get_title()


def test_plot_efficiencies_vs_altitude_runs_without_error(swept_points):
    ax = plot_efficiencies_vs(swept_points, x_attr="altitude_m")
    assert "Efficiency" in ax.get_title()
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert set(legend_labels) == {"Thermal", "Propulsive", "Overall"}


def test_plot_thrust_vs_rejects_invalid_x_attr(swept_points):
    with pytest.raises(ValueError):
        plot_thrust_vs(swept_points, x_attr="nonsense")
