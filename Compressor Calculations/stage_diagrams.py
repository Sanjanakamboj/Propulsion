"""Per-stage h-s ladder diagram and parameter table for a single compressor
mean-line stage -- styled to match the reference Turbine Stage Design
notebook's plots. Velocity-triangle plotting and the parameter table
renderer are shared with Turbine Calculations and live in Utils/plotting.py.
"""

import math
import sys
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent.parent / "Utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from plotting import hline_label, ke_arrow, multistage_hs_diagram, plot_parameter_table, plot_velocity_triangles

__all__ = [
    "plot_velocity_triangles",
    "plot_parameter_table",
    "compressor_stage_hs_ladder",
    "compressor_stage_parameter_sections",
    "compressor_stage_diagrams",
    "compressor_multistage_hs_diagram",
]


# ============================================================
# H-S LADDER DIAGRAM
# ============================================================


def compressor_stage_hs_ladder(result, cp, gamma, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    R = cp * (gamma - 1.0) / gamma
    T1, P1 = result.T1, result.P1
    T01 = T1 + result.V1**2 / (2.0 * cp)

    def entropy(T, P):
        return cp * math.log(T / T1) - R * math.log(P / P1)

    h1, H01 = cp * T1, cp * T01
    h2, H02 = cp * result.T2, cp * result.T02
    h3, H03 = cp * result.T3, cp * result.T03
    H03s = cp * result.T03s

    s1 = entropy(T1, P1)
    s2 = entropy(result.T2, result.P2)
    s3 = entropy(result.T3, result.P3)

    ax.plot([s1, s2, s3], [h1, h2, h3], linewidth=2, color="tab:orange")
    ax.scatter([s1, s2, s3], [h1, h2, h3], s=40, color="tab:orange", zorder=5)
    ax.text(s1, h1, "  1")
    ax.text(s2, h2, "  2")
    ax.text(s3, h3, "  3")

    ax.plot([s1, s2, s3], [H01, H02, H03], linewidth=2, color="tab:blue")
    ax.scatter([s1, s2, s3], [H01, H02, H03], s=40, color="tab:blue", zorder=5)
    ax.text(s1, H01, "  01")
    ax.text(s3, H03, "  03")

    # Ideal (isentropic) branch off station 1 -- single lumped stage
    # efficiency, so there's one ideal marker, not a per-row/stator split.
    ax.plot([s1, s1], [H01, H03s], linewidth=2, color="tab:green")
    ax.scatter([s1, s1], [H01, H03s], s=40, color="tab:green", zorder=5)

    s_label = min(s1, s2, s3, 0.0) - 5.0
    ax.set_xlim(s_label - 6, max(s1, s2, s3) + 10)

    hline_label(ax, H01, s1, s_label, "dashed", r"H$_{01}$")
    hline_label(ax, h1, s1, s_label, "dotted", r"h$_{1}$")
    hline_label(ax, h2, s2, s_label, "dotted", r"h$_{2}$")
    hline_label(ax, H02, s2, s_label, "dashed", r"H$_{02}$")
    hline_label(ax, h3, s3, s_label, "dotted", r"h$_{3}$")
    hline_label(ax, H03, s3, s_label, "dashed", r"H$_{03}$")
    hline_label(ax, H03s, s1, s_label, "dotted", r"H$_{03s}$")

    ds = max(2.0, 0.05 * abs(s2 - s1))
    ke_arrow(ax, s1 + ds, h1, H01, r"$\frac{V_1^2}{2}$")
    ke_arrow(ax, s2 + ds, h2, H02, r"$\frac{V_2^2}{2}$")
    ke_arrow(ax, s3 + ds, h3, H03, r"$\frac{V_3^2}{2}$")

    ax.set_xlabel("Entropy  s  (J/kg·K)")
    ax.set_ylabel("Enthalpy  h, H  (J/kg)")
    ax.set_title("h-s Diagram for Compressor Stage", fontsize=16, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.2)
    return ax


# ============================================================
# MULTI-STAGE REHEAT-FACTOR H-S DIAGRAM
# ============================================================


def compressor_multistage_hs_diagram(T01_first, P01_first, compressor_stages, cp, gamma, ax=None):
    """Multi-stage reheat-factor h-s diagram (Utils/plotting.py) across ALL
    of a compressor's stages -- distinct from compressor_stage_hs_ladder
    above, which shows one representative stage's own rotor+stator path.
    T01_first/P01_first are the compressor's own inlet stagnation state
    (not stored on any individual CompressorStageResult); each subsequent
    stage's inlet is the previous stage's T03/P03 exit.

    Returns (ax, MultistageHsResult)."""
    stagnation_states = [(T01_first, P01_first)] + [(s.T03, s.P03) for s in compressor_stages]
    return multistage_hs_diagram(stagnation_states, cp, gamma, ax=ax)


# ============================================================
# PARAMETER SECTIONS
# ============================================================


def compressor_stage_parameter_sections(result):
    return [
        (
            "STATION 1 — Rotor Inlet",
            [
                ("Static Pressure", "$P_1$", "Pa", f"{result.P1:.2f}"),
                ("Static Temperature", "$T_1$", "K", f"{result.T1:.2f}"),
                ("Absolute Velocity", "$V_1$", "m/s", f"{result.V1:.2f}"),
                ("Flow Angle (abs)", r"$\alpha_1$", "deg", f"{result.alpha1_deg:.2f}"),
                ("Mach Number (abs)", "$M_1$", "-", f"{result.M1:.2f}"),
                ("Relative Velocity", "$W_1$", "m/s", f"{result.W1:.2f}"),
                ("Flow Angle (rel)", r"$\beta_1$", "deg", f"{result.beta1_deg:.2f}"),
                ("Mach Number (rel)", "$M_{w1}$", "-", f"{result.Mw1:.2f}"),
                ("Channel Span Height", "$h_1$", "m", f"{result.annulus_1.blade_height:.3f}"),
                ("Hub Radius", "$R_{H1}$", "m", f"{result.annulus_1.hub_radius:.3f}"),
                ("Tip Radius", "$R_{T1}$", "m", f"{result.annulus_1.tip_radius:.3f}"),
                ("Channel Area", "$A_1$", "m^2", f"{result.annulus_1.area:.3f}"),
            ],
        ),
        (
            "STATION 2 — Rotor Exit / Stator Inlet",
            [
                ("Static Pressure", "$P_2$", "Pa", f"{result.P2:.2f}"),
                ("Static Temperature", "$T_2$", "K", f"{result.T2:.2f}"),
                ("Absolute Velocity", "$V_2$", "m/s", f"{result.V2:.2f}"),
                ("Flow Angle (abs)", r"$\alpha_2$", "deg", f"{result.alpha2_deg:.2f}"),
                ("Relative Velocity", "$W_2$", "m/s", f"{result.W2:.2f}"),
                ("Flow Angle (rel)", r"$\beta_2$", "deg", f"{result.beta2_deg:.2f}"),
                ("Channel Span Height", "$h_2$", "m", f"{result.annulus_2.blade_height:.3f}"),
                ("Hub Radius", "$R_{H2}$", "m", f"{result.annulus_2.hub_radius:.3f}"),
                ("Tip Radius", "$R_{T2}$", "m", f"{result.annulus_2.tip_radius:.3f}"),
                ("Channel Area", "$A_2$", "m^2", f"{result.annulus_2.area:.3f}"),
            ],
        ),
        (
            "STATION 3 — Stator Exit",
            [
                ("Total Pressure", "$P_{03}$", "Pa", f"{result.P03:.2f}"),
                ("Static Pressure", "$P_3$", "Pa", f"{result.P3:.2f}"),
                ("Static Temperature", "$T_3$", "K", f"{result.T3:.2f}"),
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
                ("de Haller Number", "$W_2/W_1$", "-", f"{result.de_haller:.3f}"),
                ("Specific Work", r"$\Delta h_0$", "kJ/kg", f"{result.specific_work / 1e3:.2f}"),
                ("AN^2", "$AN^2$", "m^2 rpm^2", f"{result.an2:.3e}"),
            ],
        ),
    ]


# ============================================================
# TOP-LEVEL WRAPPER
# ============================================================


def compressor_stage_diagrams(result, cp, gamma, save_prefix: str | None = None):
    """Returns (fig_velocity_triangles, fig_hs, fig_table); saves three PNGs
    (<save_prefix>_velocity_triangles.png / _hs_diagram.png / _table.png) if
    save_prefix is given."""
    import matplotlib.pyplot as plt

    fig_vt, ax_vt = plt.subplots(figsize=(11, 7.4))
    plot_velocity_triangles(result.Wt1, result.Vt1, result.Wt2, result.Vt2, result.Cx, result.U, label_in="1", label_out="2", ax=ax_vt)
    fig_vt.tight_layout()

    fig_hs, ax_hs = plt.subplots(figsize=(12, 6))
    compressor_stage_hs_ladder(result, cp, gamma, ax=ax_hs)
    fig_hs.tight_layout()

    fig_table, _ = plot_parameter_table(compressor_stage_parameter_sections(result), "Compressor Stage — Thermodynamic Quantities")

    if save_prefix is not None:
        fig_vt.savefig(f"{save_prefix}_velocity_triangles.png", dpi=150, bbox_inches="tight")
        fig_hs.savefig(f"{save_prefix}_hs_diagram.png", dpi=150, bbox_inches="tight")
        fig_table.savefig(f"{save_prefix}_table.png", dpi=150, bbox_inches="tight")

    return fig_vt, fig_hs, fig_table
