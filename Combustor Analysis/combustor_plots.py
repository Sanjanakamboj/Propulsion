"""Combustor result plots -- a styled parameter table (matching the same
visual style used for compressor/turbine stage tables, via Utils/plotting.py)
and an equivalence-ratio gauge showing where the design point sits relative
to the lean/stoichiometric/rich zones.
"""

import sys
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent.parent / "Utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from plotting import plot_parameter_table  # noqa: E402

__all__ = ["plot_parameter_table", "combustor_parameter_sections", "plot_equivalence_ratio_gauge", "combustor_diagrams"]


def combustor_parameter_sections(T_in, T_exit, P_in, P_out, far, combustion_state, tau=None):
    """combustion_state: a combustion.CombustionState (far, equivalence_ratio, regime)."""
    sections = [
        (
            "COMBUSTOR INLET / EXIT",
            [
                ("Inlet Temperature", "$T_{in}$", "K", f"{T_in:.2f}"),
                ("Exit Temperature", "$T_{exit}$", "K", f"{T_exit:.2f}"),
                ("Inlet Pressure", "$P_{in}$", "Pa", f"{P_in:.2f}"),
                ("Exit Pressure", "$P_{out}$", "Pa", f"{P_out:.2f}"),
                ("Pressure Loss", r"$\Delta P/P_{in}$", "-", f"{(P_in - P_out) / P_in:.4f}"),
            ],
        ),
        (
            "COMBUSTION",
            [
                ("Fuel-Air Ratio", "$FAR$", "-", f"{far:.5f}"),
                ("Equivalence Ratio", r"$\phi$", "-", f"{combustion_state.equivalence_ratio:.3f}"),
                ("Regime", "-", "-", combustion_state.regime),
            ],
        ),
    ]
    if tau is not None:
        sections.append(("RESIDENCE TIME", [("Residence Time", r"$\tau$", "ms", f"{tau * 1e3:.2f}")]))
    return sections


def plot_equivalence_ratio_gauge(phi, ax=None, phi_max=2.0):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.2))

    ax.axvspan(0.0, 1.0 - 0.02, color="#a8d5a2", alpha=0.6, label="lean")
    ax.axvspan(1.0 - 0.02, 1.0 + 0.02, color="#f2d675", alpha=0.7, label="stoichiometric")
    ax.axvspan(1.0 + 0.02, phi_max, color="#e88a8a", alpha=0.6, label="rich")

    ax.axvline(phi, color="black", lw=2.5, zorder=5)
    ax.plot(phi, 0.5, "v", color="black", ms=12, zorder=6)
    ax.annotate(f"$\\phi$ = {phi:.3f}", xy=(phi, 0.85), ha="center", fontsize=11, fontweight="bold")

    ax.set_xlim(0.0, phi_max)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Equivalence Ratio  $\\phi$")
    ax.set_title("Combustor Operating Point", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    return ax


def combustor_diagrams(T_in, T_exit, P_in, P_out, far, combustion_state, tau=None, save_prefix: str | None = None):
    """Returns (fig_table, fig_gauge); saves two PNGs
    (<save_prefix>_table.png / _equivalence_ratio.png) if save_prefix is given."""
    import matplotlib.pyplot as plt

    fig_table, _ = plot_parameter_table(
        combustor_parameter_sections(T_in, T_exit, P_in, P_out, far, combustion_state, tau),
        "Combustor — Thermodynamic Quantities",
    )

    fig_gauge, ax_gauge = plt.subplots(figsize=(10, 2.2))
    plot_equivalence_ratio_gauge(combustion_state.equivalence_ratio, ax=ax_gauge)
    fig_gauge.tight_layout()

    if save_prefix is not None:
        fig_table.savefig(f"{save_prefix}_table.png", dpi=150, bbox_inches="tight")
        fig_gauge.savefig(f"{save_prefix}_equivalence_ratio.png", dpi=150, bbox_inches="tight")

    return fig_table, fig_gauge
