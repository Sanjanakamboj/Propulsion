import matplotlib

matplotlib.use("Agg")

import pytest

from postprocess import (
    blade_surface_cp_profile,
    blade_surface_pressure_profile,
    plot_blade_cp_profile,
    pressure_coefficient,
    read_surface_flow_csv,
    static_pressure_from_conservative,
)

GAMMA = 1.333

SURFACE_CSV = '''"PointID","x","y","Density","Momentum_x","Momentum_y","Energy","Nu_Tilde"
1, 0.0, 0.0, 1.3, 100.0, 10.0, 900000.0, 0.0
2, 0.05, 0.01, 1.31, 105.0, 8.0, 905000.0, 0.0
3, 0.10, 0.02, 1.32, 110.0, 6.0, 910000.0, 0.0
'''


@pytest.fixture
def surface_csv_path(tmp_path):
    path = tmp_path / "surface_flow.csv"
    path.write_text(SURFACE_CSV)
    return str(path)


def test_static_pressure_from_conservative_matches_ideal_gas_relation():
    P = static_pressure_from_conservative(density=1.3, momentum_x=100.0, momentum_y=10.0, energy=900000.0, gamma=GAMMA)
    expected = (GAMMA - 1.0) * (900000.0 - 0.5 * (100.0**2 + 10.0**2) / 1.3)
    assert P == pytest.approx(expected)


def test_static_pressure_from_conservative_rejects_non_positive_density():
    with pytest.raises(ValueError):
        static_pressure_from_conservative(density=0.0, momentum_x=1.0, momentum_y=1.0, energy=1000.0, gamma=GAMMA)


def test_pressure_coefficient_is_zero_when_P_equals_P_ref():
    assert pressure_coefficient(P=101325.0, P_ref=101325.0, rho_ref=1.2, V_ref=50.0) == pytest.approx(0.0)


def test_pressure_coefficient_matches_manual_formula():
    cp = pressure_coefficient(P=105000.0, P_ref=101325.0, rho_ref=1.2, V_ref=50.0)
    assert cp == pytest.approx((105000.0 - 101325.0) / (0.5 * 1.2 * 50.0**2))


def test_pressure_coefficient_rejects_non_positive_rho_or_V():
    with pytest.raises(ValueError):
        pressure_coefficient(P=105000.0, P_ref=101325.0, rho_ref=0.0, V_ref=50.0)
    with pytest.raises(ValueError):
        pressure_coefficient(P=105000.0, P_ref=101325.0, rho_ref=1.2, V_ref=0.0)


def test_read_surface_flow_csv_parses_all_rows(surface_csv_path):
    data = read_surface_flow_csv(surface_csv_path)
    assert len(data.x) == 3
    assert data.density == pytest.approx([1.3, 1.31, 1.32])


def test_blade_surface_pressure_profile_matches_manual_calc(surface_csv_path):
    x, y, P = blade_surface_pressure_profile(surface_csv_path, gamma=GAMMA)
    assert len(P) == 3
    expected_first = static_pressure_from_conservative(1.3, 100.0, 10.0, 900000.0, GAMMA)
    assert P[0] == pytest.approx(expected_first)


def test_blade_surface_cp_profile_matches_manual_calc(surface_csv_path):
    x, y, cp = blade_surface_cp_profile(surface_csv_path, gamma=GAMMA, P_ref=200000.0, rho_ref=1.2, V_ref=50.0)
    _, _, P = blade_surface_pressure_profile(surface_csv_path, gamma=GAMMA)
    expected_first = pressure_coefficient(P[0], 200000.0, 1.2, 50.0)
    assert cp[0] == pytest.approx(expected_first)


def test_plot_blade_cp_profile_runs_without_error(surface_csv_path):
    ax = plot_blade_cp_profile(surface_csv_path, gamma=GAMMA, P_ref=200000.0, rho_ref=1.2, V_ref=50.0)
    assert "Pressure Coefficient" in ax.get_title()
