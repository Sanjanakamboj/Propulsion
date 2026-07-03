"""Demo: basic ideal (no-loss) Brayton cycle."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import AIR
from cycle_plots import plot_cycle_diagrams
from ideal_cycle import IdealCycleInputs, run_ideal_cycle

inputs = IdealCycleInputs(T1=288.15, P1=101_325.0, pressure_ratio=8.0, T3=1400.0, gas=AIR)
result = run_ideal_cycle(inputs)

print("--- Ideal Brayton Cycle ---")
print(f"Pressure ratio        = {inputs.pressure_ratio:.1f}")
for state in result.states:
    print(f"  State {state.label}: T = {state.T:7.2f} K, P = {state.P / 1e3:9.2f} kPa")
print(f"Compressor work        = {result.compressor_work / 1e3:.2f} kJ/kg")
print(f"Turbine work           = {result.turbine_work / 1e3:.2f} kJ/kg")
print(f"Net work                = {result.net_work / 1e3:.2f} kJ/kg")
print(f"Heat added              = {result.heat_added / 1e3:.2f} kJ/kg")
print(f"Thermal efficiency      = {result.thermal_efficiency * 100:.1f} %")

plot_cycle_diagrams(result.states, AIR, title_prefix="Ideal Brayton Cycle", save_path="ideal_cycle_diagrams.png")
print("\nSaved T-s and P-v diagrams to ideal_cycle_diagrams.png")
