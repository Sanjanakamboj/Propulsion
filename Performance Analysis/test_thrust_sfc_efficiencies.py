import sys
from pathlib import Path

import pytest

from efficiencies import extract_overall_efficiency, extract_propulsive_efficiency, extract_thermal_efficiency
from mission import FlightEnvelopePoint
from off_design import sweep_off_design
from sfc import extract_tsfc, extract_tsfc_per_hour
from thrust import extract_gross_thrust, extract_net_thrust, thrust_lapse_ratio

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


def test_extract_net_thrust_matches_the_underlying_results(swept_points):
    values = extract_net_thrust(swept_points)
    assert values == [p.results.net_thrust for p in swept_points]
    assert len(values) == 3


def test_extract_gross_thrust_matches_the_underlying_results(swept_points):
    values = extract_gross_thrust(swept_points)
    assert values == [p.results.gross_thrust for p in swept_points]


def test_thrust_lapse_ratio_is_one_at_the_reference_point(swept_points):
    ref = swept_points[0].results.net_thrust
    ratios = thrust_lapse_ratio(swept_points, ref)
    assert ratios[0] == pytest.approx(1.0)
    assert all(r <= 1.0 + 1e-9 for r in ratios)  # sea level (index 0) is the highest-thrust point


def test_thrust_lapse_ratio_rejects_non_positive_reference(swept_points):
    with pytest.raises(ValueError):
        thrust_lapse_ratio(swept_points, 0.0)


def test_extract_tsfc_matches_the_underlying_results(swept_points):
    values = extract_tsfc(swept_points)
    assert values == [p.results.tsfc for p in swept_points]


def test_extract_tsfc_per_hour_is_tsfc_times_3600(swept_points):
    tsfc = extract_tsfc(swept_points)
    tsfc_hr = extract_tsfc_per_hour(swept_points)
    assert tsfc_hr == pytest.approx([t * 3600.0 for t in tsfc])


def test_extract_efficiencies_match_the_underlying_results(swept_points):
    assert extract_thermal_efficiency(swept_points) == [p.results.thermal_efficiency for p in swept_points]
    assert extract_propulsive_efficiency(swept_points) == [p.results.propulsive_efficiency for p in swept_points]
    assert extract_overall_efficiency(swept_points) == [p.results.overall_efficiency for p in swept_points]
    for eff in extract_overall_efficiency(swept_points):
        assert 0.0 < eff < 1.0
