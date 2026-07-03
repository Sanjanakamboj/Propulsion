import pytest

from mission import MissionRequirements, isa_atmosphere


def test_sea_level_matches_isa_reference():
    T, P = isa_atmosphere(0.0)
    assert T == pytest.approx(288.15)
    assert P == pytest.approx(101_325.0)


def test_tropopause_matches_known_isa_table_values():
    T, P = isa_atmosphere(11_000.0)
    assert T == pytest.approx(216.65, abs=0.01)
    assert P == pytest.approx(22_632.0, rel=5e-3)  # ISA table value


def test_stratosphere_matches_known_isa_table_value():
    T, P = isa_atmosphere(20_000.0)
    assert T == pytest.approx(216.65, abs=0.01)  # isothermal stratosphere
    assert P == pytest.approx(5_474.9, rel=5e-3)  # ISA table value


def test_temperature_and_pressure_decrease_with_altitude():
    T1, P1 = isa_atmosphere(0.0)
    T2, P2 = isa_atmosphere(5_000.0)
    T3, P3 = isa_atmosphere(11_000.0)
    assert T1 > T2 > T3
    assert P1 > P2 > P3


def test_altitude_out_of_range_raises():
    with pytest.raises(ValueError):
        isa_atmosphere(-1.0)
    with pytest.raises(ValueError):
        isa_atmosphere(25_000.0)


def test_mission_requirements_exposes_ambient_conditions():
    mission = MissionRequirements(cruise_mach=0.82, cruise_altitude_m=11_000.0, required_thrust_N=120_000.0)
    T, P = mission.ambient_conditions
    assert T == pytest.approx(216.65, abs=0.01)
    assert P == pytest.approx(22_632.0, rel=5e-3)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(cruise_mach=-0.1),
        dict(cruise_altitude_m=-1.0),
        dict(cruise_altitude_m=25_000.0),
        dict(required_thrust_N=0.0),
        dict(required_thrust_N=-100.0),
    ],
)
def test_invalid_mission_requirements_raise(kwargs):
    base = dict(cruise_mach=0.82, cruise_altitude_m=11_000.0, required_thrust_N=120_000.0)
    base.update(kwargs)
    with pytest.raises(ValueError):
        MissionRequirements(**base)
