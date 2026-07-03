import pytest

from ideal_cycle import IdealCycleInputs, run_ideal_cycle
from config import AIR


@pytest.fixture
def result():
    inputs = IdealCycleInputs(T1=288.15, P1=101_325.0, pressure_ratio=8.0, T3=1400.0, gas=AIR)
    return run_ideal_cycle(inputs)


def test_states_cover_all_four_points_with_correct_pressures(result):
    s1, s2, s3, s4 = result.states
    assert s1.P == pytest.approx(101_325.0)
    assert s2.P == pytest.approx(101_325.0 * 8.0)
    assert s3.P == pytest.approx(s2.P)
    assert s4.P == pytest.approx(s1.P)


def test_peak_temperature_is_T3(result):
    s1, s2, s3, s4 = result.states
    assert s3.T == pytest.approx(1400.0)
    assert s3.T == max(s.T for s in result.states)


def test_net_work_equals_turbine_minus_compressor_work(result):
    assert result.net_work == pytest.approx(result.turbine_work - result.compressor_work)


def test_energy_balance_heat_added_minus_rejected_equals_net_work(result):
    # Ideal closed cycle: net work out must equal net heat in (1st law over the loop).
    assert result.heat_added - result.heat_rejected == pytest.approx(result.net_work, rel=1e-9)


def test_thermal_efficiency_matches_ideal_brayton_formula(result):
    # eta_ideal = 1 - 1/PR^((gamma-1)/gamma), the textbook closed-form result.
    expected = 1.0 - 1.0 / (8.0 ** ((AIR.gamma - 1.0) / AIR.gamma))
    assert result.thermal_efficiency == pytest.approx(expected, rel=1e-9)


def test_higher_pressure_ratio_gives_higher_ideal_efficiency():
    low = run_ideal_cycle(IdealCycleInputs(T1=288.15, P1=101_325.0, pressure_ratio=4.0, T3=1400.0, gas=AIR))
    high = run_ideal_cycle(IdealCycleInputs(T1=288.15, P1=101_325.0, pressure_ratio=12.0, T3=1400.0, gas=AIR))
    assert high.thermal_efficiency > low.thermal_efficiency


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(T1=0.0, P1=101_325.0, pressure_ratio=8.0, T3=1400.0),
        dict(T1=288.15, P1=0.0, pressure_ratio=8.0, T3=1400.0),
        dict(T1=288.15, P1=101_325.0, pressure_ratio=1.0, T3=1400.0),
        dict(T1=288.15, P1=101_325.0, pressure_ratio=8.0, T3=200.0),
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        IdealCycleInputs(gas=AIR, **kwargs)
