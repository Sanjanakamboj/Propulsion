import pytest

from thrust import (
    compute_thrust,
    gross_thrust,
    momentum_thrust,
    net_thrust,
    nozzle_exit_area,
    pressure_thrust,
    ram_drag,
    specific_thrust,
    tsfc,
)

# Validated Brayton Cycle Analysis cruise design point.
MDOT_GAS = 1.0279454023204933
V_EXIT = 663.896673407283
P_EXIT = 146825.5198845399
P_AMBIENT = 22632.040095007793
RHO_EXIT = 0.4440488111221769
MDOT_AIR = 1.0
V0 = 241.99490680590773
FUEL_FLOW = 0.02794540232049333

EXPECTED_AREA = 0.0034868947950705666
EXPECTED_PRESSURE_THRUST = 433.04959825982115
EXPECTED_GROSS = 1115.4991313047078
EXPECTED_NET = 873.5042244988001
EXPECTED_TSFC = 3.199229212260292e-05


def test_nozzle_exit_area_matches_validated_result():
    area = nozzle_exit_area(MDOT_GAS, RHO_EXIT, V_EXIT)
    assert area == pytest.approx(EXPECTED_AREA, rel=1e-6)


def test_pressure_thrust_matches_validated_result():
    area = nozzle_exit_area(MDOT_GAS, RHO_EXIT, V_EXIT)
    press = pressure_thrust(P_EXIT, P_AMBIENT, area)
    assert press == pytest.approx(EXPECTED_PRESSURE_THRUST, rel=1e-6)


def test_pressure_thrust_is_zero_when_fully_expanded():
    assert pressure_thrust(P_exit=22632.0, P_ambient=22632.0, area_exit=0.05) == pytest.approx(0.0)


def test_gross_and_net_thrust_match_validated_result():
    area = nozzle_exit_area(MDOT_GAS, RHO_EXIT, V_EXIT)
    mom = momentum_thrust(MDOT_GAS, V_EXIT)
    press = pressure_thrust(P_EXIT, P_AMBIENT, area)
    gross = gross_thrust(mom, press)
    drag = ram_drag(MDOT_AIR, V0)
    net = net_thrust(gross, drag)
    assert gross == pytest.approx(EXPECTED_GROSS, rel=1e-6)
    assert net == pytest.approx(EXPECTED_NET, rel=1e-6)


def test_specific_thrust_and_tsfc_match_validated_result():
    spec = specific_thrust(EXPECTED_NET, MDOT_AIR)
    fuel_burn = tsfc(FUEL_FLOW, EXPECTED_NET)
    assert spec == pytest.approx(EXPECTED_NET)  # mdot_air = 1.0 here
    assert fuel_burn == pytest.approx(EXPECTED_TSFC, rel=1e-6)


def test_compute_thrust_wrapper_matches_validated_result():
    breakdown = compute_thrust(MDOT_GAS, V_EXIT, P_EXIT, P_AMBIENT, RHO_EXIT, MDOT_AIR, V0, FUEL_FLOW)
    assert breakdown.gross_thrust == pytest.approx(EXPECTED_GROSS, rel=1e-6)
    assert breakdown.net_thrust == pytest.approx(EXPECTED_NET, rel=1e-6)
    assert breakdown.tsfc == pytest.approx(EXPECTED_TSFC, rel=1e-6)
    assert breakdown.pressure_thrust > 0.0  # underexpanded (choked) exit


def test_nozzle_exit_area_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        nozzle_exit_area(MDOT_GAS, rho_exit=0.0, V_exit=V_EXIT)
    with pytest.raises(ValueError):
        nozzle_exit_area(MDOT_GAS, rho_exit=RHO_EXIT, V_exit=0.0)


def test_ram_drag_rejects_non_positive_mdot_air():
    with pytest.raises(ValueError):
        ram_drag(mdot_air=0.0, V0=V0)


def test_tsfc_rejects_non_positive_net_thrust():
    with pytest.raises(ValueError):
        tsfc(FUEL_FLOW, net=0.0)
