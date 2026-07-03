"""Demo: intercooled Brayton cycle -- two-stage compression with an
intercooler, compared against an equivalent single-stage compressor at the
same overall pressure ratio."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import AIR
from cycle_plots import plot_cycle_diagrams
from intercooling import IntercoolingCycleInputs, run_intercooling_cycle
from real_cycle import RealCycleInputs, run_real_cycle

single_stage = run_real_cycle(RealCycleInputs(
    T1=288.15, P1=101_325.0, pressure_ratio=9.0, T3=1400.0, gas=AIR,
    compressor_efficiency=0.85, turbine_efficiency=0.90,
))
result = run_intercooling_cycle(IntercoolingCycleInputs(
    T1=288.15, P1=101_325.0, pressure_ratio_lp=3.0, pressure_ratio_hp=3.0,
    T_intercool_exit=288.15, T5=1400.0, gas=AIR,
    compressor_efficiency=0.85, turbine_efficiency=0.90,
))

print("--- Intercooled Brayton Cycle (overall PR = 9.0, split 3.0 x 3.0) ---")
for state in result.states:
    print(f"  State {state.label}: T = {state.T:7.2f} K, P = {state.P / 1e3:9.2f} kPa")
print(f"Compressor work (2-stage) = {result.compressor_work / 1e3:.2f} kJ/kg  (single-stage: {single_stage.compressor_work / 1e3:.2f} kJ/kg)")
print(f"Net work                  = {result.net_work / 1e3:.2f} kJ/kg")
print(f"Thermal efficiency        = {result.thermal_efficiency * 100:.1f} %")

plot_cycle_diagrams(result.states, AIR, title_prefix="Intercooled Brayton Cycle", save_path="intercooling_cycle_diagrams.png")
print("\nSaved T-s and P-v diagrams to intercooling_cycle_diagrams.png")
