import pytest

from engine import TurbojetDesignInputs, run_turbojet
from stations import AIR


@pytest.fixture
def design():
    return TurbojetDesignInputs(
        ambient_T=288.15,
        ambient_P=101_325.0,
        flight_mach=0.0,
        mdot_air=50.0,
        compressor_pressure_ratio=12.0,
        compressor_efficiency=0.87,
        turbine_inlet_temperature=1400.0,
        turbine_efficiency=0.90,
        mechanical_efficiency=0.99,
        combustor_pressure_loss_frac=0.04,
        combustor_efficiency=0.99,
        nozzle_efficiency=0.98,
    )


def test_turbine_power_balances_compressor_power(design):
    results, records = run_turbojet(design)
    compressor_power = results.compressor_specific_work * design.mdot_air
    turbine_power = results.turbine_specific_work * results.stations["4"].mdot
    assert turbine_power == pytest.approx(compressor_power / design.mechanical_efficiency, rel=1e-9)


def test_station_temperatures_progress_sensibly(design):
    results, records = run_turbojet(design)
    s = results.stations
    assert s["2"].T0 == pytest.approx(design.ambient_T, rel=1e-9)  # static run, no ram heating
    assert s["3"].T0 > s["2"].T0  # compressor raises temperature
    assert s["4"].T0 == pytest.approx(design.turbine_inlet_temperature)  # design TIT
    assert s["5"].T0 < s["4"].T0  # turbine extracts energy
    assert s["9"].T0 < s["5"].T0  # nozzle continues expansion


def test_station_pressures_progress_sensibly(design):
    results, records = run_turbojet(design)
    s = results.stations
    assert s["3"].P0 == pytest.approx(s["2"].P0 * design.compressor_pressure_ratio)
    assert s["4"].P0 < s["3"].P0  # combustor pressure loss
    assert s["5"].P0 < s["4"].P0  # turbine expansion
    assert s["9"].P0 <= s["5"].P0


def test_fuel_air_ratio_and_flow_positive(design):
    results, records = run_turbojet(design)
    assert results.fuel_air_ratio > 0.0
    assert results.fuel_flow == pytest.approx(results.fuel_air_ratio * design.mdot_air)


def test_thrust_and_tsfc_positive_for_static_run(design):
    results, records = run_turbojet(design)
    assert results.net_thrust > 0.0
    assert results.specific_thrust > 0.0
    assert results.tsfc > 0.0
    assert results.nozzle_exit_velocity > 0.0


def test_static_run_has_zero_propulsive_and_overall_efficiency(design):
    # Flight Mach = 0 -> zero flight velocity -> no propulsive power delivered.
    results, records = run_turbojet(design)
    assert results.flight_velocity == pytest.approx(0.0)
    assert results.propulsive_efficiency == pytest.approx(0.0)
    assert results.overall_efficiency == pytest.approx(0.0)


def test_flight_mach_increases_flight_velocity_and_enables_propulsive_efficiency():
    design = TurbojetDesignInputs(
        ambient_T=250.0,
        ambient_P=40_000.0,
        flight_mach=0.8,
        mdot_air=50.0,
        compressor_pressure_ratio=15.0,
        turbine_inlet_temperature=1450.0,
    )
    results, records = run_turbojet(design)
    expected_V0 = 0.8 * (AIR.gamma * AIR.R * 250.0) ** 0.5
    assert results.flight_velocity == pytest.approx(expected_V0, rel=1e-6)
    assert results.propulsive_efficiency > 0.0
    assert results.overall_efficiency > 0.0


def test_higher_pressure_ratio_reduces_fuel_air_ratio_for_fixed_tit(design):
    import dataclasses

    low_rp = run_turbojet(dataclasses.replace(design, compressor_pressure_ratio=8.0))[0]
    high_rp = run_turbojet(dataclasses.replace(design, compressor_pressure_ratio=20.0))[0]
    # Higher pressure ratio raises T3 (compressor exit), so less heat (and fuel) is
    # needed to reach the same fixed turbine inlet temperature.
    assert high_rp.fuel_air_ratio < low_rp.fuel_air_ratio


def test_invalid_inputs_raise():
    base = dict(ambient_T=288.15, ambient_P=101_325.0)
    with pytest.raises(ValueError):
        TurbojetDesignInputs(**base, compressor_pressure_ratio=1.0)
    with pytest.raises(ValueError):
        TurbojetDesignInputs(**base, turbine_inlet_temperature=100.0)
    with pytest.raises(ValueError):
        TurbojetDesignInputs(**base, compressor_efficiency=1.2)
    with pytest.raises(ValueError):
        TurbojetDesignInputs(**base, flight_mach=-0.1)
