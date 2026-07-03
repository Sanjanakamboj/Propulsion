import pytest

from ideal_cycle import IdealCycleInputs, run_ideal_cycle
from real_cycle import RealCycleInputs, run_real_cycle
from config import AIR


@pytest.fixture
def result():
    inputs = RealCycleInputs(
        T1=288.15, P1=101_325.0, pressure_ratio=8.0, T3=1400.0, gas=AIR,
        compressor_efficiency=0.85, turbine_efficiency=0.90,
    )
    return run_real_cycle(inputs)


def test_states_cover_all_four_points_with_correct_pressures(result):
    s1, s2, s3, s4 = result.states
    assert s1.P == pytest.approx(101_325.0)
    assert s2.P == pytest.approx(101_325.0 * 8.0)
    assert s3.P == pytest.approx(s2.P)
    assert s4.P == pytest.approx(s1.P)


def test_real_cycle_has_lower_efficiency_than_ideal_at_same_conditions():
    ideal = run_ideal_cycle(IdealCycleInputs(T1=288.15, P1=101_325.0, pressure_ratio=8.0, T3=1400.0, gas=AIR))
    real = run_real_cycle(RealCycleInputs(T1=288.15, P1=101_325.0, pressure_ratio=8.0, T3=1400.0, gas=AIR, compressor_efficiency=0.85, turbine_efficiency=0.90))
    assert real.thermal_efficiency < ideal.thermal_efficiency
    assert real.net_work < ideal.net_work


def test_perfect_efficiencies_reduce_to_ideal_cycle():
    ideal = run_ideal_cycle(IdealCycleInputs(T1=288.15, P1=101_325.0, pressure_ratio=8.0, T3=1400.0, gas=AIR))
    real = run_real_cycle(RealCycleInputs(T1=288.15, P1=101_325.0, pressure_ratio=8.0, T3=1400.0, gas=AIR, compressor_efficiency=1.0, turbine_efficiency=1.0))
    assert real.thermal_efficiency == pytest.approx(ideal.thermal_efficiency, rel=1e-9)
    assert real.net_work == pytest.approx(ideal.net_work, rel=1e-9)


def test_back_work_ratio_matches_compressor_over_turbine_work(result):
    assert result.back_work_ratio == pytest.approx(result.compressor_work / result.turbine_work)


def test_energy_balance_heat_added_minus_rejected_equals_net_work(result):
    assert result.heat_added - result.heat_rejected == pytest.approx(result.net_work, rel=1e-9)


@pytest.mark.parametrize("efficiency_kwarg", ["compressor_efficiency", "turbine_efficiency"])
@pytest.mark.parametrize("bad_value", [0.0, -0.1, 1.1])
def test_invalid_efficiency_raises(efficiency_kwarg, bad_value):
    kwargs = dict(T1=288.15, P1=101_325.0, pressure_ratio=8.0, T3=1400.0, gas=AIR)
    kwargs[efficiency_kwarg] = bad_value
    with pytest.raises(ValueError):
        RealCycleInputs(**kwargs)
