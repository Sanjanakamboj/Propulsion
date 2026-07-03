import pytest

from mission import FlightEnvelopePoint
from off_design import (
    compressor_face_conditions,
    corrected_mass_flow_ratio,
    run_off_design_point,
    scaled_mass_flow,
    sweep_off_design,
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Brayton Cycle Analysis"))
from engine import TurbojetDesignInputs  # noqa: E402


@pytest.fixture
def design():
    return TurbojetDesignInputs(
        ambient_T=216.65, ambient_P=22632.04, flight_mach=0.82, mdot_air=137.4,
        compressor_pressure_ratio=24.0, compressor_efficiency=0.87,
        turbine_inlet_temperature=1700.0, turbine_efficiency=0.90,
        combustor_pressure_loss_frac=0.04, combustor_efficiency=0.99, nozzle_efficiency=0.98,
    )


def test_corrected_mass_flow_ratio_is_one_at_the_reference_condition():
    assert corrected_mass_flow_ratio(P_new=100000.0, T_new=288.15, P_design=100000.0, T_design=288.15) == pytest.approx(1.0)


def test_corrected_mass_flow_ratio_matches_manual_formula():
    ratio = corrected_mass_flow_ratio(P_new=150000.0, T_new=320.0, P_design=100000.0, T_design=288.15)
    assert ratio == pytest.approx((150000.0 / 100000.0) / (320.0 / 288.15) ** 0.5)


def test_scaled_mass_flow_rejects_non_positive_design_mdot():
    with pytest.raises(ValueError):
        scaled_mass_flow(0.0, 100000.0, 288.15, 100000.0, 288.15)


def test_compressor_face_conditions_reasonable_for_design_point(design):
    T02, P02 = compressor_face_conditions(design, design.ambient_T, design.ambient_P, design.flight_mach)
    assert T02 > design.ambient_T  # ram heating
    assert P02 > design.ambient_P  # ram pressure recovery


def test_sea_level_static_gives_much_higher_thrust_than_altitude_design_point(design):
    # This is the dominant, well-known real-world off-design behavior:
    # denser sea-level air gives far more thrust at the same throttle
    # setting (same PR/TIT/efficiencies) than at the 11 km design altitude.
    T02_design, P02_design = compressor_face_conditions(design, design.ambient_T, design.ambient_P, design.flight_mach)
    sea_level_point = run_off_design_point(design, T02_design, P02_design, mach=design.flight_mach, altitude_m=0.0)
    assert sea_level_point.mdot_air > design.mdot_air * 2.0
    assert sea_level_point.results.net_thrust > 120_000.0 * 2.0


def test_off_design_point_at_the_design_condition_reproduces_design_mdot(design):
    T02_design, P02_design = compressor_face_conditions(design, design.ambient_T, design.ambient_P, design.flight_mach)
    point = run_off_design_point(design, T02_design, P02_design, mach=design.flight_mach, altitude_m=11000.0)
    assert point.mdot_air == pytest.approx(design.mdot_air, rel=1e-6)
    assert point.results.net_thrust == pytest.approx(120_019.48, rel=1e-3)


def test_sweep_off_design_covers_every_envelope_point(design):
    envelope = [FlightEnvelopePoint(mach=0.82, altitude_m=11000.0), FlightEnvelopePoint(mach=0.82, altitude_m=0.0)]
    points = sweep_off_design(design, envelope)
    assert len(points) == 2
    assert points[0].altitude_m == 11000.0
    assert points[1].altitude_m == 0.0
    assert points[1].results.net_thrust > points[0].results.net_thrust


def test_thrust_decreases_monotonically_with_altitude_at_fixed_mach(design):
    envelope = [FlightEnvelopePoint(mach=0.82, altitude_m=a) for a in (0.0, 5000.0, 11000.0)]
    points = sweep_off_design(design, envelope)
    thrusts = [p.results.net_thrust for p in points]
    assert thrusts[0] > thrusts[1] > thrusts[2]
