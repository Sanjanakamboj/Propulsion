import pytest

from intercooling import IntercoolingCycleInputs, run_intercooling_cycle
from real_cycle import RealCycleInputs, run_real_cycle
from config import AIR


@pytest.fixture
def result():
    inputs = IntercoolingCycleInputs(
        T1=288.15, P1=101_325.0, pressure_ratio_lp=3.0, pressure_ratio_hp=3.0,
        T_intercool_exit=288.15, T5=1400.0, gas=AIR,
        compressor_efficiency=0.85, turbine_efficiency=0.90,
    )
    return run_intercooling_cycle(inputs)


def test_states_cover_all_six_points_with_correct_pressures(result):
    s1, s2, s3, s4, s5, s6 = result.states
    assert s1.P == pytest.approx(101_325.0)
    assert s2.P == pytest.approx(101_325.0 * 3.0)
    assert s3.P == pytest.approx(s2.P)  # intercooler, no pressure loss modeled
    assert s4.P == pytest.approx(s2.P * 3.0)
    assert s5.P == pytest.approx(s4.P)  # combustor, no pressure loss modeled
    assert s6.P == pytest.approx(s1.P)


def test_intercooler_exit_matches_requested_temperature(result):
    s3 = result.states[2]
    assert s3.T == pytest.approx(288.15)


def test_two_stage_intercooled_compression_uses_less_work_than_single_stage(result):
    # Classic result: for the same overall pressure ratio and inlet temperature,
    # splitting compression with full intercooling (back to T1) uses less total
    # compressor work than one single-stage compressor.
    single_stage = run_real_cycle(RealCycleInputs(
        T1=288.15, P1=101_325.0, pressure_ratio=9.0, T3=1400.0, gas=AIR,
        compressor_efficiency=0.85, turbine_efficiency=0.90,
    ))
    assert result.compressor_work < single_stage.compressor_work


def test_energy_balance_heat_added_minus_rejected_equals_net_work(result):
    assert result.heat_added - result.heat_rejected == pytest.approx(result.net_work, rel=1e-9)


def test_net_work_equals_turbine_minus_compressor_work(result):
    assert result.net_work == pytest.approx(result.turbine_work - result.compressor_work)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(pressure_ratio_lp=1.0),
        dict(pressure_ratio_hp=1.0),
        dict(T_intercool_exit=1500.0),  # >= T5, invalid
        dict(compressor_efficiency=1.1),
        dict(turbine_efficiency=0.0),
    ],
)
def test_invalid_inputs_raise(kwargs):
    base = dict(
        T1=288.15, P1=101_325.0, pressure_ratio_lp=3.0, pressure_ratio_hp=3.0,
        T_intercool_exit=288.15, T5=1400.0, gas=AIR,
        compressor_efficiency=0.85, turbine_efficiency=0.90,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        IntercoolingCycleInputs(**base)
