import pytest

from residence_time import required_volume, residence_time


def test_residence_time_matches_manual_formula():
    tau = residence_time(combustor_volume=0.05, mdot=140.0, rho=1.6)
    assert tau == pytest.approx(0.05 * 1.6 / 140.0)


def test_residence_time_is_in_a_plausible_millisecond_range():
    # Order-of-magnitude sanity check, not a precise spec.
    tau = residence_time(combustor_volume=0.05, mdot=140.0, rho=1.6)
    assert 0.0001 < tau < 0.05


def test_required_volume_is_the_correct_inverse():
    tau = residence_time(combustor_volume=0.05, mdot=140.0, rho=1.6)
    recovered_volume = required_volume(tau, mdot=140.0, rho=1.6)
    assert recovered_volume == pytest.approx(0.05, rel=1e-9)


def test_larger_volume_gives_longer_residence_time():
    tau_small = residence_time(combustor_volume=0.02, mdot=140.0, rho=1.6)
    tau_large = residence_time(combustor_volume=0.08, mdot=140.0, rho=1.6)
    assert tau_large > tau_small


def test_higher_mass_flow_gives_shorter_residence_time():
    tau_low_mdot = residence_time(combustor_volume=0.05, mdot=100.0, rho=1.6)
    tau_high_mdot = residence_time(combustor_volume=0.05, mdot=200.0, rho=1.6)
    assert tau_high_mdot < tau_low_mdot


@pytest.mark.parametrize("bad_volume", [0.0, -0.01])
def test_residence_time_rejects_non_positive_volume(bad_volume):
    with pytest.raises(ValueError):
        residence_time(bad_volume, mdot=140.0, rho=1.6)


@pytest.mark.parametrize("bad_mdot", [0.0, -1.0])
def test_residence_time_rejects_non_positive_mdot(bad_mdot):
    with pytest.raises(ValueError):
        residence_time(0.05, mdot=bad_mdot, rho=1.6)


def test_required_volume_rejects_non_positive_target():
    with pytest.raises(ValueError):
        required_volume(0.0, mdot=140.0, rho=1.6)
