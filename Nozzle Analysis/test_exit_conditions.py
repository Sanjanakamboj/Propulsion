import math

import pytest

from exit_conditions import (
    actual_exit_temperature,
    critical_pressure_ratio,
    exit_mach,
    exit_mach_from_pressure_ratio,
    exit_velocity_subsonic,
    is_choked,
    isentropic_exit_temperature,
    sonic_velocity,
)

GAMMA = 1.333
CP = 1148.0
R = CP * (GAMMA - 1.0) / GAMMA

# Validated Brayton Cycle Analysis cruise design point, nozzle inlet (station 5):
T0_IN, P0_IN, P_AMBIENT = 1340.4660176240825, 271982.75895383774, 22632.040095007793
ETA = 0.98
EXPECTED_P_EXIT_CHOKED = 146825.5198845399
EXPECTED_T_EXIT_CHOKED = 1152.9616540615266
EXPECTED_V_EXIT_CHOKED = 663.896673407283


def test_critical_pressure_ratio_matches_manual_formula():
    ratio = critical_pressure_ratio(GAMMA)
    assert ratio == pytest.approx(((GAMMA + 1.0) / 2.0) ** (GAMMA / (GAMMA - 1.0)))


def test_is_choked_true_for_the_validated_high_pressure_ratio_design_point():
    assert is_choked(P0_IN, P_AMBIENT, GAMMA)


def test_is_choked_false_for_a_low_pressure_ratio():
    assert not is_choked(P0_in=30000.0, P_ambient=22632.0, gamma=GAMMA)


def test_choked_branch_matches_validated_brayton_cycle_result():
    critical_ratio = critical_pressure_ratio(GAMMA)
    P_exit = P0_IN / critical_ratio
    T_exit_ideal = T0_IN * (2.0 / (GAMMA + 1.0))
    T_exit = actual_exit_temperature(T0_IN, T_exit_ideal, ETA)
    V_exit = sonic_velocity(T_exit, GAMMA, R)

    assert P_exit == pytest.approx(EXPECTED_P_EXIT_CHOKED, rel=1e-6)
    assert T_exit == pytest.approx(EXPECTED_T_EXIT_CHOKED, rel=1e-6)
    assert V_exit == pytest.approx(EXPECTED_V_EXIT_CHOKED, rel=1e-6)


def test_exit_mach_is_approximately_one_at_the_choked_throat():
    M = exit_mach(EXPECTED_V_EXIT_CHOKED, EXPECTED_T_EXIT_CHOKED, GAMMA, R)
    assert M == pytest.approx(1.0, rel=1e-6)


def test_isentropic_exit_temperature_matches_manual_formula():
    T_exit_ideal = isentropic_exit_temperature(T0_IN, P0_IN, P_AMBIENT, GAMMA)
    assert T_exit_ideal == pytest.approx(T0_IN * (P_AMBIENT / P0_IN) ** ((GAMMA - 1.0) / GAMMA))


def test_exit_mach_from_pressure_ratio_matches_hand_verified_supersonic_case():
    M = exit_mach_from_pressure_ratio(P0_IN, P_AMBIENT, GAMMA)
    assert M == pytest.approx(2.274052208657084, rel=1e-6)


def test_exit_mach_from_pressure_ratio_gives_subsonic_M_for_a_low_pressure_ratio():
    M = exit_mach_from_pressure_ratio(P0_in=30000.0, P_exit=22632.0, gamma=GAMMA)
    assert 0.0 < M < 1.0


def test_exit_velocity_subsonic_matches_manual_formula():
    V = exit_velocity_subsonic(T0_in=500.0, T_exit=450.0, cp=1005.0)
    assert V == pytest.approx(math.sqrt(2.0 * 1005.0 * 50.0))


def test_actual_exit_temperature_rejects_invalid_efficiency():
    with pytest.raises(ValueError):
        actual_exit_temperature(500.0, 450.0, isentropic_efficiency=0.0)
    with pytest.raises(ValueError):
        actual_exit_temperature(500.0, 450.0, isentropic_efficiency=1.1)


def test_isentropic_exit_temperature_rejects_P_exit_above_P0_in():
    with pytest.raises(ValueError):
        isentropic_exit_temperature(500.0, 100000.0, 200000.0, GAMMA)


def test_exit_mach_from_pressure_ratio_rejects_P_exit_above_P0_in():
    with pytest.raises(ValueError):
        exit_mach_from_pressure_ratio(100000.0, 200000.0, GAMMA)


def test_sonic_velocity_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        sonic_velocity(0.0, GAMMA, R)
