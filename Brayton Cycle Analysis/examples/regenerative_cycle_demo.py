"""Demo: regenerative Brayton cycle -- recuperating turbine-exhaust heat to
preheat the compressor-exit air, compared against the same design point
with no regeneration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import AIR
from cycle_plots import plot_cycle_diagrams
from regenerative_cycle import RegenerativeCycleInputs, run_regenerative_cycle

no_regen = run_regenerative_cycle(RegenerativeCycleInputs(
    T1=288.15, P1=101_325.0, pressure_ratio=8.0, T4=1400.0, gas=AIR,
    regenerator_effectiveness=0.0, compressor_efficiency=0.85, turbine_efficiency=0.90,
))
result = run_regenerative_cycle(RegenerativeCycleInputs(
    T1=288.15, P1=101_325.0, pressure_ratio=8.0, T4=1400.0, gas=AIR,
    regenerator_effectiveness=0.75, compressor_efficiency=0.85, turbine_efficiency=0.90,
))

print("--- Regenerative Brayton Cycle (effectiveness = 0.75) ---")
for state in result.states:
    print(f"  State {state.label}: T = {state.T:7.2f} K, P = {state.P / 1e3:9.2f} kPa")
print(f"Net work                 = {result.net_work / 1e3:.2f} kJ/kg  (unchanged by regeneration)")
print(f"Heat added                = {result.heat_added / 1e3:.2f} kJ/kg  (no regen: {no_regen.heat_added / 1e3:.2f} kJ/kg)")
print(f"Thermal efficiency        = {result.thermal_efficiency * 100:.1f} %  (no regen: {no_regen.thermal_efficiency * 100:.1f} %)")

plot_cycle_diagrams(result.states, AIR, title_prefix="Regenerative Brayton Cycle", save_path="regenerative_cycle_diagrams.png")
print("\nSaved T-s and P-v diagrams to regenerative_cycle_diagrams.png")
