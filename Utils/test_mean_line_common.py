import math

import pytest

from mean_line_common import an2, annulus_from_mass_flow, blade_speed_from_loading, mean_radius


def test_blade_speed_from_loading_below_limit():
    U, psi = blade_speed_from_loading(specific_work=200_000.0, stage_loading=1.8, blade_speed_limit=400.0)
    assert U == pytest.approx(math.sqrt(200_000.0 / 1.8))
    assert psi == pytest.approx(1.8)


def test_blade_speed_from_loading_capped_by_limit():
    U, psi = blade_speed_from_loading(specific_work=200_000.0, stage_loading=0.1, blade_speed_limit=300.0)
    assert U == pytest.approx(300.0)
    assert psi == pytest.approx(200_000.0 / 300.0**2)
    assert psi > 0.1  # loading rises above target since U was capped


@pytest.mark.parametrize("kwargs", [dict(specific_work=0.0), dict(stage_loading=0.0), dict(blade_speed_limit=0.0)])
def test_blade_speed_from_loading_invalid_raises(kwargs):
    base = dict(specific_work=200_000.0, stage_loading=1.8, blade_speed_limit=400.0)
    base.update(kwargs)
    with pytest.raises(ValueError):
        blade_speed_from_loading(**base)


def test_mean_radius_matches_omega_relation():
    U = 300.0
    rpm = 3000.0
    r = mean_radius(U, rpm)
    omega = 2.0 * math.pi * rpm / 60.0
    assert r * omega == pytest.approx(U)


def test_annulus_from_mass_flow_matches_continuity():
    geom = annulus_from_mass_flow(mass_flow=50.0, density=1.2, axial_velocity=150.0, mean_diameter=0.5)
    assert geom.area == pytest.approx(50.0 / (1.2 * 150.0))
    assert geom.mean_diameter == pytest.approx(0.5)
    assert geom.tip_radius - geom.hub_radius == pytest.approx(geom.blade_height)
    assert (geom.tip_radius + geom.hub_radius) / 2.0 == pytest.approx(0.25)


def test_annulus_from_mass_flow_infeasible_raises():
    with pytest.raises(ValueError):
        annulus_from_mass_flow(mass_flow=1000.0, density=0.1, axial_velocity=10.0, mean_diameter=0.2)


def test_an2_scales_with_area_and_rpm_squared():
    assert an2(area=1.0, rotational_speed_rpm=1000.0) == pytest.approx(1.0e6)
    assert an2(area=2.0, rotational_speed_rpm=1000.0) == pytest.approx(2.0e6)
