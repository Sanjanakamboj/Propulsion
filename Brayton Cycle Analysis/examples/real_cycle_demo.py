"""Demo: real (non-regenerative) Brayton cycle with compressor/turbine
isentropic efficiencies, compared against the ideal cycle at the same
pressure ratio and temperatures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import AIR
from cycle_plots import plot_cycle_diagrams
from ideal_cycle import IdealCycleInputs, run_ideal_cycle
from real_cycle import RealCycleInputs, run_real_cycle

ideal = run_ideal_cycle(IdealCycleInputs(T1=288.15, P1=101_325.0, pressure_ratio=8.0, T3=1400.0, gas=AIR))
real = run_real_cycle(RealCycleInputs(
    T1=288.15, P1=101_325.0, pressure_ratio=8.0, T3=1400.0, gas=AIR,
    compressor_efficiency=0.85, turbine_efficiency=0.90,
))

print("--- Real Brayton Cycle ---")
for state in real.states:
    print(f"  State {state.label}: T = {state.T:7.2f} K, P = {state.P / 1e3:9.2f} kPa")
print(f"Compressor work         = {real.compressor_work / 1e3:.2f} kJ/kg")
print(f"Turbine work            = {real.turbine_work / 1e3:.2f} kJ/kg")
print(f"Net work                = {real.net_work / 1e3:.2f} kJ/kg")
print(f"Heat added               = {real.heat_added / 1e3:.2f} kJ/kg")
print(f"Back work ratio          = {real.back_work_ratio:.3f}")
print(f"Thermal efficiency       = {real.thermal_efficiency * 100:.1f} %  (ideal: {ideal.thermal_efficiency * 100:.1f} %)")

plot_cycle_diagrams(real.states, AIR, title_prefix="Real Brayton Cycle", save_path="real_cycle_diagrams.png")
print("\nSaved T-s and P-v diagrams to real_cycle_diagrams.png")
