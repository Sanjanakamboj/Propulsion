import math

import pytest

from stages import Combustor, Compressor, Inlet, Nozzle, Turbine
from stations import AIR, COMBUSTION_GAS, Station


def _entropy(gas, T, P, T_ref, P_ref):
    return gas.cp * math.log(T / T_ref) - gas.R * math.log(P / P_ref)


def test_compressor_isentropic_leg_has_zero_entropy_change():
    inlet = Station("2", 288.15, 101_325.0, AIR, 50.0)
    compressor = Compressor(pressure_ratio=12.0, isentropic_efficiency=0.87)
    exit_station, extra = compressor.run(inlet)

    T_ideal = extra["T0_ideal"]
    s_inlet = _entropy(AIR, inlet.T0, inlet.P0, inlet.T0, inlet.P0)
    s_ideal = _entropy(AIR, T_ideal, exit_station.P0, inlet.T0, inlet.P0)
    assert s_ideal == pytest.approx(s_inlet, abs=1e-9)


def test_compressor_real_exit_has_higher_entropy_than_ideal():
    inlet = Station("2", 288.15, 101_325.0, AIR, 50.0)
    compressor = Compressor(pressure_ratio=12.0, isentropic_efficiency=0.87)
    exit_station, extra = compressor.run(inlet)

    T_ideal = extra["T0_ideal"]
    s_ideal = _entropy(AIR, T_ideal, exit_station.P0, inlet.T0, inlet.P0)
    s_actual = _entropy(AIR, exit_station.T0, exit_station.P0, inlet.T0, inlet.P0)
    assert s_actual > s_ideal
    assert exit_station.T0 > T_ideal  # real compressor exit is hotter than ideal


def test_compressor_perfect_efficiency_matches_ideal():
    inlet = Station("2", 288.15, 101_325.0, AIR, 50.0)
    compressor = Compressor(pressure_ratio=12.0, isentropic_efficiency=1.0)
    exit_station, extra = compressor.run(inlet)
    assert exit_station.T0 == pytest.approx(extra["T0_ideal"], rel=1e-9)


def test_turbine_work_matches_requested_power():
    inlet = Station("4", 1400.0, 1_200_000.0, COMBUSTION_GAS, 51.0)
    turbine = Turbine(isentropic_efficiency=0.90, mechanical_efficiency=0.99)
    required_power = 20e6  # W
    exit_station, extra = turbine.run(inlet, compressor_power=required_power * turbine.mechanical_efficiency)
    assert extra["power"] == pytest.approx(required_power, rel=1e-9)
    assert extra["specific_work"] * inlet.mdot == pytest.approx(required_power, rel=1e-9)
    assert exit_station.T0 < inlet.T0


def test_turbine_isentropic_leg_has_zero_entropy_change():
    inlet = Station("4", 1400.0, 1_200_000.0, COMBUSTION_GAS, 51.0)
    turbine = Turbine(isentropic_efficiency=0.90, mechanical_efficiency=0.99)
    exit_station, extra = turbine.run(inlet, compressor_power=15e6)

    T_ideal = extra["T0_ideal"]
    s_inlet = _entropy(COMBUSTION_GAS, inlet.T0, inlet.P0, inlet.T0, inlet.P0)
    s_ideal = _entropy(COMBUSTION_GAS, T_ideal, exit_station.P0, inlet.T0, inlet.P0)
    assert s_ideal == pytest.approx(s_inlet, abs=1e-9)


def test_combustor_solves_fuel_air_ratio_for_target_tit():
    inlet = Station("3", 700.0, 1_200_000.0, AIR, 50.0)
    combustor = Combustor(
        exit_temperature=1400.0, pressure_loss_frac=0.04, combustion_efficiency=0.99, fuel_lhv=43e6
    )
    exit_station, extra = combustor.run(inlet)
    assert exit_station.T0 == pytest.approx(1400.0)
    assert extra["fuel_air_ratio"] > 0.0
    assert exit_station.mdot == pytest.approx(inlet.mdot + extra["fuel_flow"])
    assert exit_station.P0 == pytest.approx(inlet.P0 * 0.96)
    assert exit_station.gas is combustor.combustion_gas


def test_inlet_static_case_has_no_ram_rise():
    inlet = Inlet(flight_mach=0.0, pressure_recovery=0.99)
    exit_station, extra = inlet.run(ambient_T=288.15, ambient_P=101_325.0, mdot=50.0)
    assert exit_station.T0 == pytest.approx(288.15)
    assert exit_station.P0 == pytest.approx(101_325.0 * 0.99)


def test_inlet_flight_mach_raises_stagnation_temperature_and_pressure():
    inlet = Inlet(flight_mach=0.8, pressure_recovery=0.98)
    exit_station, extra = inlet.run(ambient_T=250.0, ambient_P=40_000.0, mdot=50.0)
    assert exit_station.T0 > 250.0
    assert exit_station.P0 > 40_000.0


def test_nozzle_unchoked_expands_to_ambient_pressure():
    inlet = Station("5", 900.0, 150_000.0, COMBUSTION_GAS, 51.0)
    nozzle = Nozzle(isentropic_efficiency=0.98)
    exit_station, extra = nozzle.run(inlet, ambient_P=101_325.0)
    assert not extra["choked"]
    assert exit_station.P0 == pytest.approx(101_325.0)
    assert extra["velocity"] > 0.0


def test_nozzle_choked_when_pressure_ratio_exceeds_critical():
    inlet = Station("5", 900.0, 400_000.0, COMBUSTION_GAS, 51.0)
    nozzle = Nozzle(isentropic_efficiency=0.98)
    exit_station, extra = nozzle.run(inlet, ambient_P=101_325.0)
    assert extra["choked"]
    assert exit_station.P0 > 101_325.0  # underexpanded, exit static P above ambient
    assert extra["velocity"] > 0.0
