import pytest

from real_cycle import RealCycleInputs, run_real_cycle
from regenerative_cycle import RegenerativeCycleInputs, run_regenerative_cycle
from config import AIR

COMMON = dict(T1=288.15, P1=101_325.0, pressure_ratio=8.0, compressor_efficiency=0.85, turbine_efficiency=0.90)


@pytest.fixture
def result():
    inputs = RegenerativeCycleInputs(T4=1400.0, gas=AIR, regenerator_effectiveness=0.75, **COMMON)
    return run_regenerative_cycle(inputs)


def test_states_cover_all_five_points_with_correct_pressures(result):
    s1, s2, s3, s4, s5 = result.states
    assert s1.P == pytest.approx(101_325.0)
    assert s2.P == pytest.approx(101_325.0 * 8.0)
    assert s3.P == pytest.approx(s2.P)  # no regenerator pressure loss modeled
    assert s4.P == pytest.approx(s2.P)  # no combustor pressure loss modeled
    assert s5.P == pytest.approx(s1.P)


def test_preheated_temperature_is_between_compressor_and_turbine_exit(result):
    s1, s2, s3, s4, s5 = result.states
    assert s5.T > s2.T  # turbine exit hotter than compressor exit -- makes regen meaningful
    assert s2.T < s3.T < s5.T


def test_zero_effectiveness_matches_non_regenerative_real_cycle():
    regen = run_regenerative_cycle(RegenerativeCycleInputs(T4=1400.0, gas=AIR, regenerator_effectiveness=0.0, **COMMON))
    real = run_real_cycle(RealCycleInputs(T3=1400.0, gas=AIR, **COMMON))
    assert regen.heat_added == pytest.approx(real.heat_added, rel=1e-9)
    assert regen.thermal_efficiency == pytest.approx(real.thermal_efficiency, rel=1e-9)


def test_net_work_unchanged_by_regeneration():
    # Regeneration recovers exhaust heat -- it doesn't change compressor/turbine work.
    no_regen = run_regenerative_cycle(RegenerativeCycleInputs(T4=1400.0, gas=AIR, regenerator_effectiveness=0.0, **COMMON))
    with_regen = run_regenerative_cycle(RegenerativeCycleInputs(T4=1400.0, gas=AIR, regenerator_effectiveness=0.75, **COMMON))
    assert with_regen.net_work == pytest.approx(no_regen.net_work, rel=1e-9)
    assert with_regen.compressor_work == pytest.approx(no_regen.compressor_work, rel=1e-9)
    assert with_regen.turbine_work == pytest.approx(no_regen.turbine_work, rel=1e-9)


def test_higher_effectiveness_reduces_heat_added_and_raises_efficiency():
    low = run_regenerative_cycle(RegenerativeCycleInputs(T4=1400.0, gas=AIR, regenerator_effectiveness=0.3, **COMMON))
    high = run_regenerative_cycle(RegenerativeCycleInputs(T4=1400.0, gas=AIR, regenerator_effectiveness=0.9, **COMMON))
    assert high.heat_added < low.heat_added
    assert high.thermal_efficiency > low.thermal_efficiency


def test_energy_balance_heat_added_minus_rejected_equals_net_work(result):
    assert result.heat_added - result.heat_rejected == pytest.approx(result.net_work, rel=1e-9)


@pytest.mark.parametrize("bad_effectiveness", [-0.1, 1.1])
def test_invalid_effectiveness_raises(bad_effectiveness):
    with pytest.raises(ValueError):
        RegenerativeCycleInputs(T4=1400.0, gas=AIR, regenerator_effectiveness=bad_effectiveness, **COMMON)
