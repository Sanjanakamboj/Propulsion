"""End-to-end demo: Stage 1 (Mission) -> Stage 2 (Cycle Design) -> Stage 3
(Engine Sizing) for a single-spool turbojet cruise design point."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diagrams import plot_cycle_diagrams
from engine import TurbojetDesignInputs
from engine_sizing import size_engine
from mission import MissionRequirements

# --- Stage 1: Mission requirements ---
mission = MissionRequirements(
    cruise_mach=0.82,
    cruise_altitude_m=11_000.0,
    required_thrust_N=120_000.0,
)
ambient_T, ambient_P = mission.ambient_conditions
print(f"Ambient conditions @ {mission.cruise_altitude_m / 1000:.1f} km: T = {ambient_T:.2f} K, P = {ambient_P / 1e3:.2f} kPa")

# --- Stage 2: Cycle design (design-point choices) ---
design = TurbojetDesignInputs(
    ambient_T=ambient_T,
    ambient_P=ambient_P,
    flight_mach=mission.cruise_mach,
    compressor_pressure_ratio=24.0,
    compressor_efficiency=0.87,
    turbine_inlet_temperature=1700.0,
    turbine_efficiency=0.90,
    combustor_pressure_loss_frac=0.04,
    combustor_efficiency=0.99,
    nozzle_efficiency=0.98,
)

# --- Stage 3: Engine sizing (mass flow + annulus dimensions for the required thrust) ---
sizing, sized_design, results, stage_records = size_engine(design, mission.required_thrust_N)

print("\n--- Cycle design point ---")
print(f"Pressure ratio           = {design.compressor_pressure_ratio:.1f}")
print(f"Turbine inlet temp (TIT) = {design.turbine_inlet_temperature:.0f} K")
print(f"Mass flow                = {sizing.mdot_air:.1f} kg/s")
print(f"Specific thrust          = {results.specific_thrust:.1f} N/(kg/s)")
print(f"TSFC                     = {results.tsfc * 3600:.4f} kg/(N*hr)  ({results.tsfc * 3600 * 9.80665:.4f} lb/(lbf*hr))")
print(f"Thermal efficiency       = {results.thermal_efficiency * 100:.1f} %")
print(f"Propulsive efficiency    = {results.propulsive_efficiency * 100:.1f} %")
print(f"Overall efficiency       = {results.overall_efficiency * 100:.1f} %")
print(f"Net thrust               = {results.net_thrust / 1e3:.1f} kN (target: {mission.required_thrust_N / 1e3:.1f} kN)")

print("\n--- Engine sizing ---")
print(f"Compressor face area     = {sizing.compressor_face_area:.3f} m^2")
print(f"Compressor face diameter = {sizing.compressor_face_diameter:.2f} m")
print(f"Nozzle exit area         = {sizing.nozzle_exit_area:.3f} m^2")
print(f"Nozzle choked            = {results.nozzle_choked}")

plot_cycle_diagrams(stage_records, save_path="turbojet_cycle_diagrams.png")
print("\nSaved per-stage T-s and P-v diagrams to turbojet_cycle_diagrams.png")
