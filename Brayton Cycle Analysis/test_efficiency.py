import pytest

from config import AIR
from efficiency import back_work_ratio, ideal_brayton_efficiency, thermal_efficiency


def test_thermal_efficiency_is_net_work_over_heat_added():
    assert thermal_efficiency(net_work=200_000.0, heat_added=800_000.0) == pytest.approx(0.25)


def test_back_work_ratio_is_compressor_over_turbine_work():
    assert back_work_ratio(compressor_work=150_000.0, turbine_work=500_000.0) == pytest.approx(0.3)


def test_ideal_brayton_efficiency_matches_closed_form():
    expected = 1.0 - 1.0 / (8.0 ** ((AIR.gamma - 1.0) / AIR.gamma))
    assert ideal_brayton_efficiency(8.0, AIR.gamma) == pytest.approx(expected)


def test_ideal_brayton_efficiency_increases_with_pressure_ratio():
    assert ideal_brayton_efficiency(12.0, AIR.gamma) > ideal_brayton_efficiency(4.0, AIR.gamma)
