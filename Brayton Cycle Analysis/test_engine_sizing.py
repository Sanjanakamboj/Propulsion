import dataclasses

import pytest

from engine import TurbojetDesignInputs, run_turbojet
from engine_sizing import annulus_tip_diameter, compressor_face_area, size_engine, size_mass_flow_for_thrust
from mission import MissionRequirements
from stations import AIR


@pytest.fixture
def design():
    mission = MissionRequirements(cruise_mach=0.82, cruise_altitude_m=11_000.0, required_thrust_N=120_000.0)
    ambient_T, ambient_P = mission.ambient_conditions
    return mission, TurbojetDesignInputs(
        ambient_T=ambient_T,
        ambient_P=ambient_P,
        flight_mach=mission.cruise_mach,
        compressor_pressure_ratio=24.0,
        turbine_inlet_temperature=1650.0,
    )


def test_specific_thrust_is_independent_of_mass_flow_scale(design):
    _, base = design
    r1 = run_turbojet(dataclasses.replace(base, mdot_air=1.0))[0]
    r2 = run_turbojet(dataclasses.replace(base, mdot_air=250.0))[0]
    assert r1.specific_thrust == pytest.approx(r2.specific_thrust, rel=1e-9)


def test_sized_mass_flow_achieves_required_thrust(design):
    mission, base = design
    mdot = size_mass_flow_for_thrust(base, mission.required_thrust_N)
    sized = dataclasses.replace(base, mdot_air=mdot)
    results, _ = run_turbojet(sized)
    assert results.net_thrust == pytest.approx(mission.required_thrust_N, rel=1e-6)


def test_size_engine_end_to_end(design):
    mission, base = design
    sizing, sized_design, results, records = size_engine(base, mission.required_thrust_N)

    assert sizing.mdot_air > 0.0
    assert sized_design.mdot_air == pytest.approx(sizing.mdot_air)
    assert results.net_thrust == pytest.approx(mission.required_thrust_N, rel=1e-6)
    assert sizing.compressor_face_area > 0.0
    assert sizing.compressor_face_diameter > 0.0
    assert sizing.nozzle_exit_area > 0.0


def test_compressor_face_area_matches_hand_calc_for_zero_mach_limit():
    # As axial_mach -> 0, static conditions approach stagnation conditions.
    T0, P0 = 288.15, 101_325.0
    area = compressor_face_area(mdot_air=100.0, T0=T0, P0=P0, gas=AIR, axial_mach=1e-4)
    rho0 = P0 / (AIR.R * T0)
    velocity0 = 1e-4 * (AIR.gamma * AIR.R * T0) ** 0.5
    expected_area = 100.0 / (rho0 * velocity0)
    assert area == pytest.approx(expected_area, rel=1e-3)


def test_annulus_tip_diameter_zero_hub_to_tip_matches_full_disk():
    area = 1.0
    diameter = annulus_tip_diameter(area, hub_to_tip_ratio=0.0)
    import math

    expected = 2.0 * math.sqrt(area / math.pi)
    assert diameter == pytest.approx(expected)


def test_annulus_tip_diameter_invalid_ratio_raises():
    with pytest.raises(ValueError):
        annulus_tip_diameter(1.0, hub_to_tip_ratio=1.0)
    with pytest.raises(ValueError):
        annulus_tip_diameter(1.0, hub_to_tip_ratio=-0.1)
