import pytest

from real_cycle import RealCycleInputs, run_real_cycle
from reheating import ReheatingCycleInputs, run_reheating_cycle
from config import AIR


@pytest.fixture
def result():
    inputs = ReheatingCycleInputs(
        T1=288.15, P1=101_325.0, pressure_ratio=9.0, T3=1400.0,
        pressure_ratio_hp_turbine=3.0, T5=1400.0, gas=AIR,
        compressor_efficiency=0.85, turbine_efficiency=0.90,
    )
    return run_reheating_cycle(inputs)


def test_states_cover_all_six_points_with_correct_pressures(result):
    s1, s2, s3, s4, s5, s6 = result.states
    assert s1.P == pytest.approx(101_325.0)
    assert s2.P == pytest.approx(101_325.0 * 9.0)
    assert s3.P == pytest.approx(s2.P)  # combustor, no pressure loss modeled
    assert s4.P == pytest.approx(s3.P / 3.0)
    assert s5.P == pytest.approx(s4.P)  # reheat combustor, no pressure loss modeled
    assert s6.P == pytest.approx(s1.P, rel=1e-9)  # overall expansion closes back to P1


def test_reheat_produces_more_turbine_work_than_single_stage_expansion():
    # Classic result: for the same overall pressure ratio and peak temperature,
    # splitting expansion with reheat (back up to T3) produces more total
    # turbine work than one single-stage turbine.
    single_stage = run_real_cycle(RealCycleInputs(
        T1=288.15, P1=101_325.0, pressure_ratio=9.0, T3=1400.0, gas=AIR,
        compressor_efficiency=0.85, turbine_efficiency=0.90,
    ))
    result = run_reheating_cycle(ReheatingCycleInputs(
        T1=288.15, P1=101_325.0, pressure_ratio=9.0, T3=1400.0,
        pressure_ratio_hp_turbine=3.0, T5=1400.0, gas=AIR,
        compressor_efficiency=0.85, turbine_efficiency=0.90,
    ))
    assert result.turbine_work > single_stage.turbine_work


def test_energy_balance_heat_added_minus_rejected_equals_net_work(result):
    assert result.heat_added - result.heat_rejected == pytest.approx(result.net_work, rel=1e-9)


def test_net_work_equals_turbine_minus_compressor_work(result):
    assert result.net_work == pytest.approx(result.turbine_work - result.compressor_work)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(pressure_ratio_hp_turbine=1.0),  # must be > 1
        dict(pressure_ratio_hp_turbine=9.0),  # must be < overall pressure_ratio
        dict(pressure_ratio_hp_turbine=10.0),  # must be < overall pressure_ratio
        dict(compressor_efficiency=0.0),
        dict(turbine_efficiency=1.1),
    ],
)
def test_invalid_inputs_raise(kwargs):
    base = dict(
        T1=288.15, P1=101_325.0, pressure_ratio=9.0, T3=1400.0,
        pressure_ratio_hp_turbine=3.0, T5=1400.0, gas=AIR,
        compressor_efficiency=0.85, turbine_efficiency=0.90,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        ReheatingCycleInputs(**base)
