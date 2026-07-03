import pytest

from converging import solve_converging_nozzle

GAMMA = 1.333
CP = 1148.0
R = CP * (GAMMA - 1.0) / GAMMA

# Validated Brayton Cycle Analysis cruise design point (station 5 -> 9).
T0_IN, P0_IN, P_AMBIENT = 1340.4660176240825, 271982.75895383774, 22632.040095007793
ETA = 0.98
EXPECTED_P_EXIT = 146825.5198845399
EXPECTED_T_EXIT = 1152.9616540615266
EXPECTED_V_EXIT = 663.896673407283


def test_solve_converging_nozzle_matches_validated_brayton_cycle_result():
    result = solve_converging_nozzle(T0_IN, P0_IN, P_AMBIENT, CP, GAMMA, R, ETA)
    assert result.choked
    assert result.P_exit == pytest.approx(EXPECTED_P_EXIT, rel=1e-6)
    assert result.T_exit == pytest.approx(EXPECTED_T_EXIT, rel=1e-6)
    assert result.V_exit == pytest.approx(EXPECTED_V_EXIT, rel=1e-6)
    assert result.M_exit == pytest.approx(1.0, rel=1e-6)


def test_solve_converging_nozzle_unchoked_expands_fully_to_ambient():
    result = solve_converging_nozzle(T0_in=800.0, P0_in=30000.0, P_ambient=22632.0, cp=1005.0, gamma=1.4, R=287.0, isentropic_efficiency=0.98)
    assert not result.choked
    assert result.P_exit == pytest.approx(22632.0)
    assert result.M_exit < 1.0


def test_solve_converging_nozzle_underexpanded_exit_pressure_stays_above_ambient():
    result = solve_converging_nozzle(T0_IN, P0_IN, P_AMBIENT, CP, GAMMA, R, ETA)
    assert result.P_exit > P_AMBIENT


def test_higher_isentropic_efficiency_gives_lower_exit_velocity_when_choked():
    # Counterintuitive but correct: when choked, V_exit is just the LOCAL
    # sonic velocity at the throat. Higher efficiency cools the throat gas
    # closer to its ideal (lower) temperature, which *lowers* the local
    # speed of sound -- thrust still rises with efficiency via higher exit
    # density/mass flow through the same throat area, not via V_exit.
    low_eta = solve_converging_nozzle(T0_IN, P0_IN, P_AMBIENT, CP, GAMMA, R, isentropic_efficiency=0.85)
    high_eta = solve_converging_nozzle(T0_IN, P0_IN, P_AMBIENT, CP, GAMMA, R, isentropic_efficiency=0.98)
    assert low_eta.choked and high_eta.choked
    assert high_eta.V_exit < low_eta.V_exit


def test_higher_isentropic_efficiency_gives_higher_exit_velocity_when_unchoked():
    # Unchoked: V_exit = sqrt(2*cp*(T0_in - T_exit)) -- higher efficiency
    # means T_exit lands closer to its (lower) ideal value, widening the
    # temperature drop and so raising V_exit, the more familiar direction.
    kwargs = dict(T0_in=800.0, P0_in=30000.0, P_ambient=22632.0, cp=1005.0, gamma=1.4, R=287.0)
    low_eta = solve_converging_nozzle(isentropic_efficiency=0.85, **kwargs)
    high_eta = solve_converging_nozzle(isentropic_efficiency=0.98, **kwargs)
    assert not low_eta.choked and not high_eta.choked
    assert high_eta.V_exit > low_eta.V_exit
