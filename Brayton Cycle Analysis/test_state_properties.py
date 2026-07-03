import pytest

from state_properties import (
    CycleState,
    constant_pressure_heat_addition,
    constant_pressure_heat_rejection,
    isentropic_compression,
    isentropic_expansion,
    isentropic_temp_ratio,
)
from config import AIR


def test_isentropic_temp_ratio_matches_definition():
    ratio = isentropic_temp_ratio(AIR, pressure_ratio=8.0)
    assert ratio == pytest.approx(8.0 ** ((AIR.gamma - 1.0) / AIR.gamma))


def test_isentropic_compression_ideal_matches_isentropic_relation():
    state1 = CycleState("1", T=288.15, P=101_325.0)
    state2 = isentropic_compression(state1, pressure_ratio=8.0, gas=AIR, label="2")
    assert state2.P == pytest.approx(state1.P * 8.0)
    assert state2.T == pytest.approx(state1.T * isentropic_temp_ratio(AIR, 8.0))


def test_isentropic_compression_with_inefficiency_requires_more_temperature_rise():
    state1 = CycleState("1", T=288.15, P=101_325.0)
    ideal = isentropic_compression(state1, pressure_ratio=8.0, gas=AIR, label="2")
    real = isentropic_compression(state1, pressure_ratio=8.0, gas=AIR, label="2", isentropic_efficiency=0.85)
    assert real.T > ideal.T
    assert real.P == pytest.approx(ideal.P)


def test_isentropic_expansion_ideal_matches_isentropic_relation():
    state3 = CycleState("3", T=1400.0, P=810_600.0)
    state4 = isentropic_expansion(state3, pressure_ratio=1.0 / 8.0, gas=AIR, label="4")
    assert state4.P == pytest.approx(state3.P / 8.0)
    assert state4.T == pytest.approx(state3.T * isentropic_temp_ratio(AIR, 1.0 / 8.0))


def test_isentropic_expansion_with_inefficiency_extracts_less_temperature_drop():
    state3 = CycleState("3", T=1400.0, P=810_600.0)
    ideal = isentropic_expansion(state3, pressure_ratio=1.0 / 8.0, gas=AIR, label="4")
    real = isentropic_expansion(state3, pressure_ratio=1.0 / 8.0, gas=AIR, label="4", isentropic_efficiency=0.85)
    assert real.T > ideal.T


@pytest.mark.parametrize("efficiency", [0.0, -0.1, 1.1])
def test_isentropic_compression_rejects_invalid_efficiency(efficiency):
    state1 = CycleState("1", T=288.15, P=101_325.0)
    with pytest.raises(ValueError):
        isentropic_compression(state1, pressure_ratio=8.0, gas=AIR, label="2", isentropic_efficiency=efficiency)


def test_constant_pressure_heat_addition_holds_pressure_sets_temperature():
    state2 = CycleState("2", T=600.0, P=810_600.0)
    state3 = constant_pressure_heat_addition(state2, T_out=1400.0, label="3")
    assert state3.P == pytest.approx(state2.P)
    assert state3.T == pytest.approx(1400.0)


def test_constant_pressure_heat_rejection_holds_pressure_sets_temperature():
    state4 = CycleState("4", T=750.0, P=101_325.0)
    state1 = constant_pressure_heat_rejection(state4, T_out=288.15, label="1")
    assert state1.P == pytest.approx(state4.P)
    assert state1.T == pytest.approx(288.15)
