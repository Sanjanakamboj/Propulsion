import pytest

from atmosphere import isa_atmosphere, isa_atmosphere_with_offset


def test_isa_atmosphere_with_offset_zero_offset_matches_standard_day():
    T_std, P_std = isa_atmosphere(5000.0)
    T_off, P_off = isa_atmosphere_with_offset(5000.0, delta_T=0.0)
    assert T_off == pytest.approx(T_std)
    assert P_off == pytest.approx(P_std)


def test_isa_atmosphere_with_offset_hot_day_raises_temperature_only():
    T_std, P_std = isa_atmosphere(0.0)
    T_hot, P_hot = isa_atmosphere_with_offset(0.0, delta_T=15.0)
    assert T_hot == pytest.approx(T_std + 15.0)
    assert P_hot == pytest.approx(P_std)


def test_isa_atmosphere_with_offset_cold_day_lowers_temperature_only():
    T_std, P_std = isa_atmosphere(0.0)
    T_cold, P_cold = isa_atmosphere_with_offset(0.0, delta_T=-20.0)
    assert T_cold == pytest.approx(T_std - 20.0)
    assert P_cold == pytest.approx(P_std)
