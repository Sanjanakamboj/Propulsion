"""Per-stage h-s ladder diagram and parameter table for a single turbine
mean-line stage -- styled to match the reference Turbine Stage Design
notebook's plots. Velocity-triangle plotting and the parameter table
renderer are shared with Compressor Calculations and live in
Utils/plotting.py.

Named turbine_stage_diagrams.py (not stage_diagrams.py) because
Compressor Calculations has its own same-named module, and design_engine.py
puts both folders on sys.path simultaneously.
"""

import math
import sys
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent.parent / "Utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from plotting import hline_label, ke_arrow, plot_parameter_table, plot_velocity_triangles

__all__ = [
    "plot_velocity_triangles",
    "plot_parameter_table",
    "turbine_stage_hs_ladder",
    "turbine_stage_parameter_sections",
    "turbine_stage_diagrams",
]


# ============================================================
# H-S LADDER DIAGRAM
# ============================================================


def turbine_stage_hs_ladder(result, cp, gamma, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    R = cp * (gamma - 1.0) / gamma
    T1, P1 = result.T1, result.P1
    T01 = T1 + result.V1**2 / (2.0 * cp)
    P01 = P1 * (T01 / T1) ** (gamma / (gamma - 1.0))

    def entropy(T, P):
        return cp * math.log(T / T1) - R * math.log(P / P1)

    T3ss = T01 * (result.P3 / P01) ** ((gamma - 1.0) / gamma)
    h3ss = cp * T3ss
    H03ss = h3ss + result.V3**2 / 2.0
    H03_R = cp * result.T3 + result.W3**2 / 2.0

    h1, H01 = cp * T1, cp * T01
    h2, h2s, H02, H02_R = cp * result.T2, cp * result.T2s, cp * T01, cp * result.T02_rel
    h3, h3s, H03 = cp * result.T3, cp * result.T3s, cp * result.T03

    s1 = entropy(T1, P1)
    s2 = entropy(result.T2, result.P2)
    s3 = entropy(result.T3, result.P3)

    ax.plot([s1, s2], [H01, H02], linewidth=2, color="tab:blue")
    ax.scatter([s1, s2], [H01, H02], s=40, color="tab:blue", zorder=5)
    ax.text(s1, H01, "  01")
    ax.text(s2, H02, "  02")

    ax.plot([s1, s2, s3], [h1, h2, h3], linewidth=2, color="tab:orange")
    ax.scatter([s1, s2, s3], [h1, h2, h3], s=40, color="tab:orange", zorder=5)
    ax.text(s1, h1, "  1")
    ax.text(s2, h2, "  2")
    ax.text(s3, h3, "  3")

    ax.plot([s1] * 5, [H01, h1, h2s, H03ss, h3ss], linewidth=2, color="tab:green")
    ax.scatter([s1] * 5, [H01, h1, h2s, H03ss, h3ss], s=40, color="tab:green", zorder=5)

    ax.plot([s2] * 4, [H02, H02_R, h2, h3s], linewidth=2, color="tab:red")
    ax.scatter([s2] * 4, [H02, H02_R, h2, h3s], s=40, color="tab:red", zorder=5)

    ax.plot([s3] * 3, [H02_R, H03, h3], linewidth=2, color="tab:purple")
    ax.scatter([s3] * 3, [H02_R, H03, h3], s=40, color="tab:purple", zorder=5)

    s_label = min(s1, s2, s3, 0.0) - 5.0
    ax.set_xlim(s_label - 6, max(s1, s2, s3) + 10)

    hline_label(ax, H01, s1, s_label, "dashed", r"H$_{01}$ = H$_{02}$")
    hline_label(ax, h1, s1, s_label, "dotted", r"h$_{1}$")
    hline_label(ax, h2, s2, s_label, "dotted", r"h$_{2}$")
    hline_label(ax, h2s, s1, s_label, "dotted", r"h$_{2s}$")
    hline_label(ax, H02_R, s3, s_label, "-.", r"H$_{02_R}$")
    hline_label(ax, H03, s3, s_label, "dashed", r"H$_{03}$")
    hline_label(ax, h3, s3, s_label, "dotted", r"h$_{3}$")
    hline_label(ax, H03ss, s1, s_label, "dotted", r"H$_{03ss}$")
    hline_label(ax, h3s, s2, s_label, "dotted", r"h$_{3s}$")

    ds = max(2.0, 0.03 * abs(s3 - s1))
    ke_arrow(ax, s1 + ds, h1, H01, r"$\frac{V_1^2}{2}$")
    ke_arrow(ax, s2 + ds, h2, H02, r"$\frac{V_2^2}{2}$")
    ke_arrow(ax, s2 - ds, h2, H02_R, r"$\frac{W_2^2}{2}$", ha="right")
    ke_arrow(ax, s3 + ds, h3, H03, r"$\frac{V_3^2}{2}$")
    ke_arrow(ax, s3 - ds, h3, H03_R, r"$\frac{W_3^2}{2}$", ha="right")

    ax.set_xlabel("Entropy  s  (J/kg·K)")
    ax.set_ylabel("Enthalpy  h, H  (J/kg)")
    ax.set_title("h-s Diagram for Turbine Stage", fontsize=16, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.2)
    return ax


# ============================================================
# PARAMETER SECTIONS
# ============================================================


def turbine_stage_parameter_sections(result):
    return [
        (
            "STATION 1 — Stage Inlet",
            [
                ("Static Pressure", "$P_1$", "Pa", f"{result.P1:.2f}"),
                ("Static Temperature", "$T_1$", "K", f"{result.T1:.2f}"),
                ("Absolute Velocity", "$V_1$", "m/s", f"{result.V1:.2f}"),
            ],
        ),
        (
            "STATION 2 — Stator Exit / Rotor Inlet",
            [
                ("Static Pressure", "$P_2$", "Pa", f"{result.P2:.2f}"),
                ("Static Temperature", "$T_2$", "K", f"{result.T2:.2f}"),
                ("Absolute Velocity", "$V_2$", "m/s", f"{result.V2:.2f}"),
                ("Flow Angle (abs)", r"$\alpha_2$", "deg", f"{result.alpha2_deg:.2f}"),
                ("Mach Number (abs)", "$M_2$", "-", f"{result.M2:.2f}"),
                ("Relative Velocity", "$W_2$", "m/s", f"{result.W2:.2f}"),
                ("Flow Angle (rel)", r"$\beta_2$", "deg", f"{result.beta2_deg:.2f}"),
                ("Mach Number (rel)", "$M_{w2}$", "-", f"{result.Mw2:.2f}"),
                ("Channel Span Height", "$h_2$", "m", f"{result.annulus_2.blade_height:.3f}"),
                ("Hub Radius", "$R_{H2}$", "m", f"{result.annulus_2.hub_radius:.3f}"),
                ("Tip Radius", "$R_{T2}$", "m", f"{result.annulus_2.tip_radius:.3f}"),
                ("Channel Area", "$A_2$", "m^2", f"{result.annulus_2.area:.3f}"),
            ],
        ),
        (
            "STATION 3 — Rotor Exit",
            [
                ("Total Pressure", "$P_{03}$", "Pa", f"{result.P03:.2f}"),
                ("Static Pressure", "$P_3$", "Pa", f"{result.P3:.2f}"),
                ("Static Temperature", "$T_3$", "K", f"{result.T3:.2f}"),
                ("Absolute Velocity", "$V_3$", "m/s", f"{result.V3:.2f}"),
                ("Flow Angle (abs)", r"$\alpha_3$", "deg", f"{result.alpha3_deg:.2f}"),
                ("Relative Velocity", "$W_3$", "m/s", f"{result.W3:.2f}"),
                ("Flow Angle (rel)", r"$\beta_3$", "deg", f"{result.beta3_deg:.2f}"),
                ("Mach Number (rel)", "$M_{w3}$", "-", f"{result.Mw3:.2f}"),
                ("Channel Span Height", "$h_3$", "m", f"{result.annulus_3.blade_height:.3f}"),
                ("Hub Radius", "$R_{H3}$", "m", f"{result.annulus_3.hub_radius:.3f}"),
                ("Tip Radius", "$R_{T3}$", "m", f"{result.annulus_3.tip_radius:.3f}"),
                ("Channel Area", "$A_3$", "m^2", f"{result.annulus_3.area:.3f}"),
            ],
        ),
        (
            "STAGE SUMMARY",
            [
                ("Blade Speed", "$U$", "m/s", f"{result.U:.2f}"),
                ("Stage Loading", r"$\psi$", "-", f"{result.achieved_stage_loading:.3f}"),
                ("Flow Coefficient", r"$\phi$", "-", f"{result.flow_coefficient:.3f}"),
                ("Degree of Reaction", r"$\Lambda$", "-", f"{result.degree_of_reaction:.2f}"),
                ("Pressure Ratio", r"$\Pi$", "-", f"{result.pressure_ratio:.3f}"),
                ("Specific Work", r"$\Delta h_0$", "kJ/kg", f"{result.specific_work / 1e3:.2f}"),
                ("AN^2", "$AN^2$", "m^2 rpm^2", f"{result.an2:.3e}"),
            ],
        ),
    ]


# ============================================================
# TOP-LEVEL WRAPPER
# ============================================================


def turbine_stage_diagrams(result, cp, gamma, save_prefix: str | None = None):
    """Returns (fig_velocity_triangles, fig_hs, fig_table); saves three PNGs
    (<save_prefix>_velocity_triangles.png / _hs_diagram.png / _table.png) if
    save_prefix is given."""
    import matplotlib.pyplot as plt

    fig_vt, ax_vt = plt.subplots(figsize=(11, 7.4))
    plot_velocity_triangles(result.Wt2, result.Vt2, result.Wt3, result.Vt3, result.Vx, result.U, label_in="2", label_out="3", ax=ax_vt)
    fig_vt.tight_layout()

    fig_hs, ax_hs = plt.subplots(figsize=(12, 6))
    turbine_stage_hs_ladder(result, cp, gamma, ax=ax_hs)
    fig_hs.tight_layout()

    fig_table, _ = plot_parameter_table(turbine_stage_parameter_sections(result), "Turbine Stage — Thermodynamic Quantities")

    if save_prefix is not None:
        fig_vt.savefig(f"{save_prefix}_velocity_triangles.png", dpi=150, bbox_inches="tight")
        fig_hs.savefig(f"{save_prefix}_hs_diagram.png", dpi=150, bbox_inches="tight")
        fig_table.savefig(f"{save_prefix}_table.png", dpi=150, bbox_inches="tight")

    return fig_vt, fig_hs, fig_table
