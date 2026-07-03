"""Off-design performance plots: thrust, TSFC, and efficiency vs. altitude
or Mach, from an off_design.sweep_off_design result list. Assumes the
sweep varies ONE of (mach, altitude_m) at a fixed value of the other --
for a full carpet plot across both, call these once per fixed value of the
other variable (e.g. once per Mach number, each producing one line on a
shared axes).
"""

from thrust import extract_net_thrust
from sfc import extract_tsfc_per_hour
from efficiencies import extract_overall_efficiency, extract_propulsive_efficiency, extract_thermal_efficiency


def _x_values(off_design_points, x_attr: str):
    if x_attr not in ("altitude_m", "mach"):
        raise ValueError("x_attr must be 'altitude_m' or 'mach'")
    return [getattr(p, x_attr) for p in off_design_points]


def plot_thrust_vs(off_design_points, x_attr: str = "altitude_m", ax=None, label: str = None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(_x_values(off_design_points, x_attr), extract_net_thrust(off_design_points), marker="o", label=label)
    ax.set_xlabel("Altitude  [m]" if x_attr == "altitude_m" else "Mach")
    ax.set_ylabel("Net Thrust  [N]")
    ax.set_title("Thrust Lapse", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_tsfc_vs(off_design_points, x_attr: str = "altitude_m", ax=None, label: str = None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(_x_values(off_design_points, x_attr), extract_tsfc_per_hour(off_design_points), marker="o", label=label)
    ax.set_xlabel("Altitude  [m]" if x_attr == "altitude_m" else "Mach")
    ax.set_ylabel("TSFC  [kg/(N*hr)]")
    ax.set_title("TSFC vs. Flight Condition", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_efficiencies_vs(off_design_points, x_attr: str = "altitude_m", ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = _x_values(off_design_points, x_attr)
    ax.plot(x, extract_thermal_efficiency(off_design_points), marker="o", label="Thermal")
    ax.plot(x, extract_propulsive_efficiency(off_design_points), marker="s", label="Propulsive")
    ax.plot(x, extract_overall_efficiency(off_design_points), marker="^", label="Overall")
    ax.set_xlabel("Altitude  [m]" if x_attr == "altitude_m" else "Mach")
    ax.set_ylabel("Efficiency")
    ax.set_title("Efficiency vs. Flight Condition", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax
