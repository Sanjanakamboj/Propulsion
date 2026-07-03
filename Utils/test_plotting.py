import math

import matplotlib

matplotlib.use("Agg")

import pytest

from plotting import _bowtie_apex, plot_parameter_table, plot_velocity_triangles


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
