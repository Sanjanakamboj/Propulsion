import pytest

from twist import (
    blade_speed_at_radius,
    free_vortex_tangential_velocity,
    infer_relative_sign,
    local_flow_angle_deg,
    twisted_section_at_radius,
)

# Validated turbine stage mean-line + annulus_2 geometry.
VT2, VT3, U_MEAN, VX = 509.7264849326719, 118.51391491378473, 349.0224443594805, 174.51122217974026
BETA2, BETA3 = 42.64137733692157, 69.5315587853695
MEAN_R = 1.1109729453965465
HUB_R, TIP_R = 1.0149115920014755, 1.2070342987916174


def test_free_vortex_reproduces_mean_at_mean_radius():
    assert free_vortex_tangential_velocity(VT2, MEAN_R, MEAN_R) == pytest.approx(VT2)


def test_free_vortex_scales_inversely_with_radius():
    assert free_vortex_tangential_velocity(VT2, MEAN_R, 2.0 * MEAN_R) == pytest.approx(VT2 / 2.0)


def test_blade_speed_reproduces_mean_at_mean_radius():
    assert blade_speed_at_radius(U_MEAN, MEAN_R, MEAN_R) == pytest.approx(U_MEAN)


def test_blade_speed_scales_linearly_with_radius():
    assert blade_speed_at_radius(U_MEAN, MEAN_R, 2.0 * MEAN_R) == pytest.approx(2.0 * U_MEAN)


def test_infer_relative_sign_matches_known_station_conventions():
    # Station 2 (rotor inlet) uses Wt = Vt - U; station 3 (rotor exit) uses Wt = Vt + U.
    assert infer_relative_sign(VT2, U_MEAN, VX, BETA2) == pytest.approx(-1.0)
    assert infer_relative_sign(VT3, U_MEAN, VX, BETA3) == pytest.approx(1.0)


def test_local_flow_angle_reproduces_mean_beta_at_mean_radius():
    assert local_flow_angle_deg(VT2, U_MEAN, VX, BETA2, MEAN_R, MEAN_R) == pytest.approx(BETA2, rel=1e-9)
    assert local_flow_angle_deg(VT3, U_MEAN, VX, BETA3, MEAN_R, MEAN_R) == pytest.approx(BETA3, rel=1e-9)


def test_local_flow_angle_matches_hand_verified_hub_and_tip():
    assert local_flow_angle_deg(VT2, U_MEAN, VX, BETA2, MEAN_R, HUB_R) == pytest.approx(53.878741776697254, rel=1e-6)
    assert local_flow_angle_deg(VT2, U_MEAN, VX, BETA2, MEAN_R, TIP_R) == pytest.approx(27.27073131194068, rel=1e-6)
    assert local_flow_angle_deg(VT3, U_MEAN, VX, BETA3, MEAN_R, HUB_R) == pytest.approx(68.74224578943708, rel=1e-6)
    assert local_flow_angle_deg(VT3, U_MEAN, VX, BETA3, MEAN_R, TIP_R) == pytest.approx(70.33322868586804, rel=1e-6)


def test_twisted_section_at_mean_radius_reproduces_mean_line_exactly():
    section = twisted_section_at_radius(VT2, VT3, U_MEAN, VX, BETA2, BETA3, MEAN_R, HUB_R, TIP_R, MEAN_R)
    assert section.beta_in_deg == pytest.approx(BETA2, rel=1e-9)
    assert section.beta_out_deg == pytest.approx(BETA3, rel=1e-9)
    assert section.U == pytest.approx(U_MEAN, rel=1e-9)
    assert section.span_fraction == pytest.approx(0.5, rel=1e-9)


def test_twisted_section_span_fraction_is_zero_at_hub_and_one_at_tip():
    hub_section = twisted_section_at_radius(VT2, VT3, U_MEAN, VX, BETA2, BETA3, MEAN_R, HUB_R, TIP_R, HUB_R)
    tip_section = twisted_section_at_radius(VT2, VT3, U_MEAN, VX, BETA2, BETA3, MEAN_R, HUB_R, TIP_R, TIP_R)
    assert hub_section.span_fraction == pytest.approx(0.0)
    assert tip_section.span_fraction == pytest.approx(1.0)


def test_twisted_section_shows_genuine_twist_hub_to_tip():
    hub_section = twisted_section_at_radius(VT2, VT3, U_MEAN, VX, BETA2, BETA3, MEAN_R, HUB_R, TIP_R, HUB_R)
    tip_section = twisted_section_at_radius(VT2, VT3, U_MEAN, VX, BETA2, BETA3, MEAN_R, HUB_R, TIP_R, TIP_R)
    assert hub_section.beta_in_deg != pytest.approx(tip_section.beta_in_deg)
    assert hub_section.beta_out_deg != pytest.approx(tip_section.beta_out_deg)


def test_twisted_section_rejects_radius_outside_hub_tip_range():
    with pytest.raises(ValueError):
        twisted_section_at_radius(VT2, VT3, U_MEAN, VX, BETA2, BETA3, MEAN_R, HUB_R, TIP_R, HUB_R - 0.1)
    with pytest.raises(ValueError):
        twisted_section_at_radius(VT2, VT3, U_MEAN, VX, BETA2, BETA3, MEAN_R, HUB_R, TIP_R, TIP_R + 0.1)


def test_free_vortex_rejects_non_positive_radius():
    with pytest.raises(ValueError):
        free_vortex_tangential_velocity(VT2, MEAN_R, 0.0)
