import pytest

from pressure_loss import (
    dynamic_pressure_loss,
    pressure_drop_fixed_fraction,
    required_loss_coefficient,
    required_loss_fraction,
)

# Validated Brayton Cycle Analysis design point: P3 = 827846.80 Pa,
# combustor_pressure_loss_frac = 0.04 -> P4 = 794732.93 Pa.
P3, LOSS_FRAC, P4 = 827846.8017565166, 0.04, 794732.9296862559


def test_pressure_drop_fixed_fraction_matches_validated_brayton_cycle_result():
    p_out = pressure_drop_fixed_fraction(P3, LOSS_FRAC)
    assert p_out == pytest.approx(P4, rel=1e-6)


def test_required_loss_fraction_is_the_correct_inverse():
    frac = required_loss_fraction(P3, P4)
    assert frac == pytest.approx(LOSS_FRAC, rel=1e-6)


def test_pressure_drop_fixed_fraction_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        pressure_drop_fixed_fraction(P3, -0.1)
    with pytest.raises(ValueError):
        pressure_drop_fixed_fraction(P3, 1.0)


def test_required_loss_fraction_rejects_P0_out_above_P0_in():
    with pytest.raises(ValueError):
        required_loss_fraction(P3, P3 * 1.1)


def test_dynamic_pressure_loss_matches_manual_formula():
    P0_out = dynamic_pressure_loss(P0_in=827846.8, rho=4.35, V=20.0, K=0.5)
    expected = 827846.8 - 0.5 * 0.5 * 4.35 * 20.0**2
    assert P0_out == pytest.approx(expected)


def test_required_loss_coefficient_is_the_correct_inverse():
    P0_out = dynamic_pressure_loss(P0_in=827846.8, rho=4.35, V=20.0, K=0.5)
    K = required_loss_coefficient(827846.8, P0_out, rho=4.35, V=20.0)
    assert K == pytest.approx(0.5, rel=1e-9)


def test_dynamic_pressure_loss_rejects_loss_exceeding_inlet_pressure():
    with pytest.raises(ValueError):
        dynamic_pressure_loss(P0_in=1000.0, rho=4.35, V=200.0, K=5.0)


def test_dynamic_pressure_loss_rejects_non_positive_rho_or_negative_K():
    with pytest.raises(ValueError):
        dynamic_pressure_loss(P0_in=827846.8, rho=0.0, V=20.0, K=0.5)
    with pytest.raises(ValueError):
        dynamic_pressure_loss(P0_in=827846.8, rho=4.35, V=20.0, K=-0.1)


def test_required_loss_coefficient_rejects_non_positive_rho_or_V():
    with pytest.raises(ValueError):
        required_loss_coefficient(827846.8, 800000.0, rho=0.0, V=20.0)
    with pytest.raises(ValueError):
        required_loss_coefficient(827846.8, 800000.0, rho=4.35, V=0.0)
