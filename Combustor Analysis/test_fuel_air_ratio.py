import pytest

from fuel_air_ratio import exit_temperature, fuel_air_ratio, fuel_flow

# Real numbers from the validated Brayton Cycle Analysis turbojet design
# point: T3 (combustor inlet) = 663.73 K, T4 (TIT) = 1700 K,
# cp_hot = 1148.0 (COMBUSTION_GAS), combustion_efficiency = 0.99, LHV = 43e6.
T_IN, T_EXIT = 663.7319017566193, 1700.0
CP_HOT = 1148.0
ETA_COMB = 0.99
LHV = 43e6
EXPECTED_FAR = 0.02794540232049333


def test_fuel_air_ratio_matches_validated_brayton_cycle_result():
    far = fuel_air_ratio(T_IN, T_EXIT, CP_HOT, ETA_COMB, LHV)
    assert far == pytest.approx(EXPECTED_FAR, rel=1e-4)


def test_exit_temperature_is_the_correct_inverse():
    far = fuel_air_ratio(T_IN, T_EXIT, CP_HOT, ETA_COMB, LHV)
    recovered_T_exit = exit_temperature(T_IN, far, CP_HOT, ETA_COMB, LHV)
    assert recovered_T_exit == pytest.approx(T_EXIT, rel=1e-9)


def test_higher_target_temperature_needs_more_fuel():
    far_low = fuel_air_ratio(T_IN, 1400.0, CP_HOT, ETA_COMB, LHV)
    far_high = fuel_air_ratio(T_IN, 1700.0, CP_HOT, ETA_COMB, LHV)
    assert far_high > far_low


def test_lower_combustion_efficiency_needs_more_fuel_for_same_temperature_rise():
    far_efficient = fuel_air_ratio(T_IN, T_EXIT, CP_HOT, 0.99, LHV)
    far_inefficient = fuel_air_ratio(T_IN, T_EXIT, CP_HOT, 0.90, LHV)
    assert far_inefficient > far_efficient


def test_fuel_flow_matches_manual_calc():
    far = fuel_air_ratio(T_IN, T_EXIT, CP_HOT, ETA_COMB, LHV)
    assert fuel_flow(mdot_air=1.0, far=far) == pytest.approx(far)
    assert fuel_flow(mdot_air=50.0, far=far) == pytest.approx(50.0 * far)


def test_fuel_air_ratio_rejects_exit_temperature_not_above_inlet():
    with pytest.raises(ValueError):
        fuel_air_ratio(T_IN, T_IN, CP_HOT, ETA_COMB, LHV)
    with pytest.raises(ValueError):
        fuel_air_ratio(T_IN, T_IN - 10.0, CP_HOT, ETA_COMB, LHV)


@pytest.mark.parametrize("bad_eta", [0.0, -0.1, 1.1])
def test_fuel_air_ratio_rejects_invalid_combustion_efficiency(bad_eta):
    with pytest.raises(ValueError):
        fuel_air_ratio(T_IN, T_EXIT, CP_HOT, bad_eta, LHV)


def test_fuel_air_ratio_rejects_non_positive_lhv():
    with pytest.raises(ValueError):
        fuel_air_ratio(T_IN, T_EXIT, CP_HOT, ETA_COMB, 0.0)


def test_fuel_flow_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        fuel_flow(mdot_air=0.0, far=0.02)
    with pytest.raises(ValueError):
        fuel_flow(mdot_air=50.0, far=0.0)
