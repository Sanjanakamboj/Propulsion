"""Nozzle result plots -- a styled parameter table (matching the same
visual style used elsewhere in this project, via Utils/plotting.py) and a
thrust breakdown bar chart (momentum vs pressure thrust, ram drag, net).
"""

import sys
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent.parent / "Utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from plotting import plot_parameter_table  # noqa: E402

__all__ = ["plot_parameter_table", "nozzle_parameter_sections", "plot_thrust_breakdown", "nozzle_diagrams"]


def nozzle_parameter_sections(exit_state, choke_assessment, thrust_breakdown, area_ratio=None):
    sections = [
        (
            "EXIT CONDITIONS",
            [
                ("Exit Mach", "$M_{exit}$", "-", f"{exit_state.M_exit:.3f}"),
                ("Exit Pressure", "$P_{exit}$", "Pa", f"{exit_state.P_exit:.2f}"),
                ("Exit Temperature", "$T_{exit}$", "K", f"{exit_state.T_exit:.2f}"),
                ("Exit Velocity", "$V_{exit}$", "m/s", f"{exit_state.V_exit:.2f}"),
            ],
        ),
        (
            "CHOKING",
            [
                ("Nozzle Pressure Ratio", "$NPR$", "-", f"{choke_assessment.nozzle_pressure_ratio:.3f}"),
                ("Critical Pressure Ratio", "$NPR_{crit}$", "-", f"{choke_assessment.critical_pressure_ratio:.3f}"),
                ("Status", "-", "-", choke_assessment.status),
            ],
        ),
        (
            "THRUST",
            [
                ("Momentum Thrust", "$F_{mom}$", "N", f"{thrust_breakdown.momentum_thrust:.2f}"),
                ("Pressure Thrust", "$F_{press}$", "N", f"{thrust_breakdown.pressure_thrust:.2f}"),
                ("Gross Thrust", "$F_{gross}$", "N", f"{thrust_breakdown.gross_thrust:.2f}"),
                ("Ram Drag", "$D_{ram}$", "N", f"{thrust_breakdown.ram_drag:.2f}"),
                ("Net Thrust", "$F_{net}$", "N", f"{thrust_breakdown.net_thrust:.2f}"),
                ("Specific Thrust", "$F_s$", "N/(kg/s)", f"{thrust_breakdown.specific_thrust:.2f}"),
                ("TSFC", "-", "kg/(N*s)", f"{thrust_breakdown.tsfc:.3e}"),
            ],
        ),
    ]
    if area_ratio is not None:
        sections.append(("C-D NOZZLE GEOMETRY", [("Area Ratio", "$A_{exit}/A_{throat}$", "-", f"{area_ratio:.3f}")]))
    return sections


def plot_thrust_breakdown(thrust_breakdown, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    labels = ["Momentum\nThrust", "Pressure\nThrust", "Gross\nThrust", "Ram\nDrag", "Net\nThrust"]
    values = [
        thrust_breakdown.momentum_thrust,
        thrust_breakdown.pressure_thrust,
        thrust_breakdown.gross_thrust,
        -thrust_breakdown.ram_drag,
        thrust_breakdown.net_thrust,
    ]
    colors = ["#0571b0", "#2f9e44", "#444444", "#c1272d", "#e8893a"]

    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="black", lw=0.8)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.0f} N", xy=(bar.get_x() + bar.get_width() / 2, value), xytext=(0, 4 if value >= 0 else -12), textcoords="offset points", ha="center", fontsize=9)

    ax.set_ylabel("Force  [N]")
    ax.set_title("Thrust Breakdown", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", color="#e8eaed", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def nozzle_diagrams(exit_state, choke_assessment, thrust_breakdown, area_ratio=None, save_prefix: str | None = None):
    """Returns (fig_table, fig_thrust); saves two PNGs
    (<save_prefix>_table.png / _thrust_breakdown.png) if save_prefix is given."""
    import matplotlib.pyplot as plt

    fig_table, _ = plot_parameter_table(
        nozzle_parameter_sections(exit_state, choke_assessment, thrust_breakdown, area_ratio),
        "Nozzle — Exit Conditions & Thrust",
    )

    fig_thrust, ax_thrust = plt.subplots(figsize=(8, 5))
    plot_thrust_breakdown(thrust_breakdown, ax=ax_thrust)
    fig_thrust.tight_layout()

    if save_prefix is not None:
        fig_table.savefig(f"{save_prefix}_table.png", dpi=150, bbox_inches="tight")
        fig_thrust.savefig(f"{save_prefix}_thrust_breakdown.png", dpi=150, bbox_inches="tight")

    return fig_table, fig_thrust
