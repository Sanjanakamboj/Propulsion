"""Engine component stages for a single-spool turbojet cycle.

Each stage's `run(...)` takes an inlet Station (plus any stage-specific
extra arguments) and returns `(exit_station, extra)`, where `extra` carries
whatever intermediate values (ideal/isentropic sub-state, specific work,
...) the diagram needs. Each stage's `path_segments(...)` returns a list of
`(leg_label, gas, T_a, P_a, T_b, P_b)` tuples describing the T-s/P-v path
through that stage, split into an isentropic piece and an isobaric loss
piece wherever the stage is not ideal -- this is what makes entropy
generation visible on the diagrams.
"""

import math

from stations import GasProperties, Station, isentropic_temp_ratio


class Inlet:
    """Ambient -> compressor face: ram compression (adiabatic, so T0 is
    always conserved from the freestream energy) plus a stagnation pressure
    recovery factor capturing inlet losses (shocks, friction, boundary layer)."""

    def __init__(self, flight_mach: float = 0.0, pressure_recovery: float = 0.98, gas: GasProperties = None):
        from stations import AIR

        self.flight_mach = flight_mach
        self.pressure_recovery = pressure_recovery
        self.gas = gas or AIR

    def run(self, ambient_T: float, ambient_P: float, mdot: float):
        gas = self.gas
        M = self.flight_mach
        ram_factor = 1.0 + 0.5 * (gas.gamma - 1.0) * M**2
        T0 = ambient_T * ram_factor
        P0_ideal = ambient_P * ram_factor ** (gas.gamma / (gas.gamma - 1.0))
        P0 = self.pressure_recovery * P0_ideal
        exit_station = Station("2", T0, P0, gas, mdot)
        extra = {"ambient_T": ambient_T, "ambient_P": ambient_P, "P0_ideal": P0_ideal}
        return exit_station, extra

    def path_segments(self, inlet: Station, exit: Station, extra: dict):
        gas = self.gas
        return [
            ("Inlet (ram compression)", gas, inlet.T0, inlet.P0, exit.T0, extra["P0_ideal"]),
            ("Inlet (pressure loss)", gas, exit.T0, extra["P0_ideal"], exit.T0, exit.P0),
        ]


class Compressor:
    def __init__(self, pressure_ratio: float, isentropic_efficiency: float):
        self.pressure_ratio = pressure_ratio
        self.isentropic_efficiency = isentropic_efficiency

    def run(self, inlet: Station):
        gas = inlet.gas
        P_exit = inlet.P0 * self.pressure_ratio
        T_exit_ideal = inlet.T0 * isentropic_temp_ratio(gas, self.pressure_ratio)
        specific_work = gas.cp * (T_exit_ideal - inlet.T0) / self.isentropic_efficiency
        T_exit = inlet.T0 + specific_work / gas.cp
        exit_station = Station("3", T_exit, P_exit, gas, inlet.mdot)
        extra = {"specific_work": specific_work, "T0_ideal": T_exit_ideal, "power": specific_work * inlet.mdot}
        return exit_station, extra

    def path_segments(self, inlet: Station, exit: Station, extra: dict):
        gas = inlet.gas
        T_ideal = extra["T0_ideal"]
        return [
            ("Compressor (isentropic)", gas, inlet.T0, inlet.P0, T_ideal, exit.P0),
            ("Compressor (loss)", gas, T_ideal, exit.P0, exit.T0, exit.P0),
        ]


class Combustor:
    """Burns fuel to reach a specified turbine inlet temperature (the design
    TIT), solving for the fuel-air ratio needed rather than taking it as an
    input. This is where the working fluid switches from cold air to hot
    combustion gas properties."""

    def __init__(
        self,
        exit_temperature: float,
        pressure_loss_frac: float,
        combustion_efficiency: float,
        fuel_lhv: float,
        combustion_gas: GasProperties = None,
    ):
        from stations import COMBUSTION_GAS

        self.exit_temperature = exit_temperature
        self.pressure_loss_frac = pressure_loss_frac
        self.combustion_efficiency = combustion_efficiency
        self.fuel_lhv = fuel_lhv
        self.combustion_gas = combustion_gas or COMBUSTION_GAS

    def run(self, inlet: Station):
        hot_gas = self.combustion_gas
        T_exit = self.exit_temperature
        P_exit = inlet.P0 * (1.0 - self.pressure_loss_frac)
        heat_added = hot_gas.cp * (T_exit - inlet.T0)
        fuel_air_ratio = heat_added / (self.combustion_efficiency * self.fuel_lhv)
        fuel_flow = inlet.mdot * fuel_air_ratio
        exit_station = Station("4", T_exit, P_exit, hot_gas, inlet.mdot + fuel_flow)
        extra = {"fuel_air_ratio": fuel_air_ratio, "fuel_flow": fuel_flow, "heat_added": heat_added}
        return exit_station, extra

    def path_segments(self, inlet: Station, exit: Station, extra: dict):
        return [
            ("Combustor (pressure loss)", inlet.gas, inlet.T0, inlet.P0, inlet.T0, exit.P0),
            ("Combustor (heat addition)", exit.gas, inlet.T0, exit.P0, exit.T0, exit.P0),
        ]


class Turbine:
    """Sized to exactly balance compressor shaft power (single-spool
    turbojet) -- the turbine pressure ratio is a *result*, not a free input."""

    def __init__(self, isentropic_efficiency: float, mechanical_efficiency: float = 0.99):
        self.isentropic_efficiency = isentropic_efficiency
        self.mechanical_efficiency = mechanical_efficiency

    def run(self, inlet: Station, compressor_power: float):
        gas = inlet.gas
        required_power = compressor_power / self.mechanical_efficiency
        specific_work = required_power / inlet.mdot
        T_exit_ideal = inlet.T0 - specific_work / (self.isentropic_efficiency * gas.cp)
        T_exit = inlet.T0 - specific_work / gas.cp
        # Solve P_exit from the isentropic relation T_exit_ideal/T_inlet = (P_exit/P_inlet)**((gamma-1)/gamma)
        P_exit = inlet.P0 * (T_exit_ideal / inlet.T0) ** (gas.gamma / (gas.gamma - 1.0))
        exit_station = Station("5", T_exit, P_exit, gas, inlet.mdot)
        extra = {"specific_work": specific_work, "T0_ideal": T_exit_ideal, "power": required_power}
        return exit_station, extra

    def path_segments(self, inlet: Station, exit: Station, extra: dict):
        gas = inlet.gas
        T_ideal = extra["T0_ideal"]
        return [
            ("Turbine (isentropic)", gas, inlet.T0, inlet.P0, T_ideal, exit.P0),
            ("Turbine (loss)", gas, T_ideal, exit.P0, exit.T0, exit.P0),
        ]


class Nozzle:
    """Expands turbine-exit gas to ambient static pressure (or to the sonic
    condition if the pressure ratio exceeds the critical/choking ratio),
    producing the exhaust velocity used for thrust."""

    def __init__(self, isentropic_efficiency: float = 0.98):
        self.isentropic_efficiency = isentropic_efficiency

    def run(self, inlet: Station, ambient_P: float):
        gas = inlet.gas
        critical_ratio = ((gas.gamma + 1.0) / 2.0) ** (gas.gamma / (gas.gamma - 1.0))
        choked = (inlet.P0 / ambient_P) > critical_ratio

        if choked:
            P_exit = inlet.P0 / critical_ratio
            T_exit_ideal = inlet.T0 * (2.0 / (gas.gamma + 1.0))
        else:
            P_exit = ambient_P
            T_exit_ideal = inlet.T0 * (P_exit / inlet.P0) ** ((gas.gamma - 1.0) / gas.gamma)

        T_exit = inlet.T0 - self.isentropic_efficiency * (inlet.T0 - T_exit_ideal)

        if choked:
            velocity = math.sqrt(gas.gamma * gas.R * T_exit)  # sonic velocity at the throat
        else:
            velocity = math.sqrt(max(0.0, 2.0 * gas.cp * (inlet.T0 - T_exit)))

        exit_station = Station("9", T_exit, P_exit, gas, inlet.mdot)
        extra = {"velocity": velocity, "choked": choked, "T0_ideal": T_exit_ideal}
        return exit_station, extra

    def path_segments(self, inlet: Station, exit: Station, extra: dict):
        gas = inlet.gas
        T_ideal = extra["T0_ideal"]
        return [
            ("Nozzle (isentropic)", gas, inlet.T0, inlet.P0, T_ideal, exit.P0),
            ("Nozzle (loss)", gas, T_ideal, exit.P0, exit.T0, exit.P0),
        ]
