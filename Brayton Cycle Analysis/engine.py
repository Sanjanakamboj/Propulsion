"""Single-spool turbojet cycle: chains Inlet -> Compressor -> Combustor ->
Turbine -> Nozzle, with the turbine solved to balance compressor shaft power.
"""

import math
from dataclasses import dataclass, field

from stages import Combustor, Compressor, Inlet, Nozzle, Turbine
from stations import AIR, COMBUSTION_GAS, GasProperties, Station


@dataclass(frozen=True)
class TurbojetDesignInputs:
    ambient_T: float  # K, static ambient temperature
    ambient_P: float  # Pa, static ambient pressure
    flight_mach: float = 0.0
    mdot_air: float = 1.0  # kg/s, core air mass flow (scales power/thrust)
    inlet_pressure_recovery: float = 0.98
    compressor_pressure_ratio: float = 12.0
    compressor_efficiency: float = 0.87
    turbine_inlet_temperature: float = 1400.0  # K, design TIT
    turbine_efficiency: float = 0.90
    mechanical_efficiency: float = 0.99
    combustor_pressure_loss_frac: float = 0.04
    combustor_efficiency: float = 0.99
    fuel_lhv: float = 43e6  # J/kg
    nozzle_efficiency: float = 0.98
    cold_gas: GasProperties = field(default_factory=lambda: AIR)
    hot_gas: GasProperties = field(default_factory=lambda: COMBUSTION_GAS)

    def __post_init__(self):
        if self.compressor_pressure_ratio <= 1.0:
            raise ValueError("compressor_pressure_ratio must be > 1")
        if self.turbine_inlet_temperature <= self.ambient_T:
            raise ValueError("turbine_inlet_temperature must exceed ambient_T")
        for name in (
            "compressor_efficiency",
            "turbine_efficiency",
            "mechanical_efficiency",
            "combustor_efficiency",
            "nozzle_efficiency",
            "inlet_pressure_recovery",
        ):
            value = getattr(self, name)
            if not (0.0 < value <= 1.0):
                raise ValueError(f"{name} must be in (0, 1], got {value}")
        if not (0.0 <= self.combustor_pressure_loss_frac < 1.0):
            raise ValueError("combustor_pressure_loss_frac must be in [0, 1)")
        if self.flight_mach < 0.0:
            raise ValueError("flight_mach must be >= 0")
        if self.mdot_air <= 0.0:
            raise ValueError("mdot_air must be > 0")


@dataclass(frozen=True)
class TurbojetResults:
    stations: dict  # station label -> Station
    compressor_specific_work: float  # J/kg
    turbine_specific_work: float  # J/kg
    fuel_air_ratio: float
    fuel_flow: float  # kg/s
    nozzle_exit_velocity: float  # m/s, actual physical exit velocity
    effective_jet_velocity: float  # m/s, equivalent fully-expanded velocity (see run_turbojet)
    nozzle_exit_area: float  # m^2
    nozzle_choked: bool
    flight_velocity: float  # m/s
    gross_thrust: float  # N
    net_thrust: float  # N
    specific_thrust: float  # N per kg/s of air
    tsfc: float  # kg fuel / (N*s)
    thermal_efficiency: float
    propulsive_efficiency: float
    overall_efficiency: float


def run_turbojet(inputs: TurbojetDesignInputs):
    """Run the full engine cycle. Returns (TurbojetResults, stage_records),
    where stage_records is an ordered list of dicts describing each stage's
    inlet/exit stations and T-s/P-v path segments, for use by diagrams.py."""
    cold, hot = inputs.cold_gas, inputs.hot_gas

    ambient_station = Station("0", inputs.ambient_T, inputs.ambient_P, cold, inputs.mdot_air)

    inlet = Inlet(inputs.flight_mach, inputs.inlet_pressure_recovery, cold)
    station2, inlet_extra = inlet.run(inputs.ambient_T, inputs.ambient_P, inputs.mdot_air)

    compressor = Compressor(inputs.compressor_pressure_ratio, inputs.compressor_efficiency)
    station3, comp_extra = compressor.run(station2)
    compressor_power = comp_extra["power"]

    combustor = Combustor(
        inputs.turbine_inlet_temperature,
        inputs.combustor_pressure_loss_frac,
        inputs.combustor_efficiency,
        inputs.fuel_lhv,
        hot,
    )
    station4, comb_extra = combustor.run(station3)

    turbine = Turbine(inputs.turbine_efficiency, inputs.mechanical_efficiency)
    station5, turb_extra = turbine.run(station4, compressor_power)

    nozzle = Nozzle(inputs.nozzle_efficiency)
    station9, noz_extra = nozzle.run(station5, inputs.ambient_P)

    stage_records = [
        {"name": "Inlet", "inlet": ambient_station, "exit": station2, "extra": inlet_extra, "stage": inlet},
        {"name": "Compressor", "inlet": station2, "exit": station3, "extra": comp_extra, "stage": compressor},
        {"name": "Combustor", "inlet": station3, "exit": station4, "extra": comb_extra, "stage": combustor},
        {"name": "Turbine", "inlet": station4, "exit": station5, "extra": turb_extra, "stage": turbine},
        {"name": "Nozzle", "inlet": station5, "exit": station9, "extra": noz_extra, "stage": nozzle},
    ]

    V0 = inputs.flight_mach * math.sqrt(cold.gamma * cold.R * inputs.ambient_T)
    V9 = noz_extra["velocity"]
    mdot_gas = station4.mdot

    rho9 = station9.P0 / (hot.R * station9.T0)
    area9 = mdot_gas / (rho9 * V9)
    pressure_thrust = (station9.P0 - inputs.ambient_P) * area9

    gross_thrust = mdot_gas * V9 + pressure_thrust
    ram_drag = inputs.mdot_air * V0
    net_thrust = gross_thrust - ram_drag
    specific_thrust = net_thrust / inputs.mdot_air
    tsfc = comb_extra["fuel_flow"] / net_thrust

    # Effective/equivalent jet velocity: the velocity an ideal, fully-expanded
    # nozzle would need to produce the same gross thrust with the same mass
    # flow. Needed because this nozzle model is convergent-only, so a choked,
    # underexpanded exit leaves real thrust in the pressure term rather than
    # in V9 -- using raw V9 in the efficiency formulas below would understate
    # thermal/propulsive efficiency whenever the nozzle is underexpanded.
    V9_effective = gross_thrust / mdot_gas

    ke_out = 0.5 * mdot_gas * V9_effective**2
    ke_in = 0.5 * inputs.mdot_air * V0**2
    thermal_efficiency = (ke_out - ke_in) / (comb_extra["fuel_flow"] * inputs.fuel_lhv)
    propulsive_efficiency = 2.0 * V0 / (V9_effective + V0) if (V9_effective + V0) > 0 else 0.0
    overall_efficiency = thermal_efficiency * propulsive_efficiency

    results = TurbojetResults(
        stations={"0": ambient_station, "2": station2, "3": station3, "4": station4, "5": station5, "9": station9},
        compressor_specific_work=comp_extra["specific_work"],
        turbine_specific_work=turb_extra["specific_work"],
        fuel_air_ratio=comb_extra["fuel_air_ratio"],
        fuel_flow=comb_extra["fuel_flow"],
        nozzle_exit_velocity=V9,
        effective_jet_velocity=V9_effective,
        nozzle_exit_area=area9,
        nozzle_choked=noz_extra["choked"],
        flight_velocity=V0,
        gross_thrust=gross_thrust,
        net_thrust=net_thrust,
        specific_thrust=specific_thrust,
        tsfc=tsfc,
        thermal_efficiency=thermal_efficiency,
        propulsive_efficiency=propulsive_efficiency,
        overall_efficiency=overall_efficiency,
    )
    return results, stage_records
