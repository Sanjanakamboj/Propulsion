import pytest

from converging import solve_converging_nozzle
from converging_diverging import area_ratio_from_mach, solve_cd_nozzle_design_matched

GAMMA = 1.333
CP = 1148.0
R = CP * (GAMMA - 1.0) / GAMMA

# Validated Brayton Cycle Analysis cruise design point (station 5 -> 9, choked).
T0_IN, P0_IN, P_AMBIENT = 1340.4660176240825, 271982.75895383774, 22632.040095007793
ETA = 0.98

# Hand-verified reference values for this design point's fully-expanded C-D case.
EXPECTED_M_EXIT = 2.274052208657084
EXPECTED_T_EXIT = 732.6878672010785
EXPECTED_V_EXIT = 1203.5191138182827
EXPECTED_AREA_RATIO = 2.2586234515453643


def test_area_ratio_equals_one_at_sonic():
    assert area_ratio_from_mach(1.0, GAMMA) == pytest.approx(1.0, rel=1e-9)


def test_area_ratio_increases_with_mach_above_sonic():
    ratio_low = area_ratio_from_mach(1.5, GAMMA)
    ratio_high = area_ratio_from_mach(3.0, GAMMA)
    assert ratio_high > ratio_low > 1.0


def test_solve_cd_nozzle_matches_hand_verified_design_point():
    result = solve_cd_nozzle_design_matched(T0_IN, P0_IN, P_AMBIENT, GAMMA, R, ETA)
    assert result.M_exit == pytest.approx(EXPECTED_M_EXIT, rel=1e-6)
    assert result.P_exit == pytest.approx(P_AMBIENT)
    assert result.T_exit == pytest.approx(EXPECTED_T_EXIT, rel=1e-6)
    assert result.V_exit == pytest.approx(EXPECTED_V_EXIT, rel=1e-6)
    assert result.area_ratio == pytest.approx(EXPECTED_AREA_RATIO, rel=1e-6)
    assert result.M_exit > 1.0  # supersonic, fully expanded


def test_cd_nozzle_delivers_more_exit_velocity_than_the_underexpanded_converging_case():
    # The whole point of the diverging section: recover the pressure-thrust
    # potential an underexpanded converging nozzle leaves as static P_exit
    # above ambient, converting it into extra velocity instead.
    converging_result = solve_converging_nozzle(T0_IN, P0_IN, P_AMBIENT, CP, GAMMA, R, ETA)
    cd_result = solve_cd_nozzle_design_matched(T0_IN, P0_IN, P_AMBIENT, GAMMA, R, ETA)
    assert converging_result.choked
    assert cd_result.V_exit > converging_result.V_exit
    assert cd_result.P_exit < converging_result.P_exit  # fully expanded vs underexpanded


def test_solve_cd_nozzle_rejects_pressure_ratio_at_or_below_critical():
    with pytest.raises(ValueError):
        solve_cd_nozzle_design_matched(T0_in=800.0, P0_in=30000.0, P_ambient=22632.0, gamma=1.4, R=287.0)


def test_area_ratio_from_mach_rejects_non_positive_mach():
    with pytest.raises(ValueError):
        area_ratio_from_mach(0.0, GAMMA)
