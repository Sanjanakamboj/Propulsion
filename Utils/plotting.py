"""Shared, machine-agnostic plotting primitives used by both Compressor
Calculations and Turbine Calculations: the combined "bowtie" velocity
triangle, h-s ladder building blocks, and the styled parameter table --
factored out to avoid duplicating identical plotting code between the two
mean-line stage modules.
"""

import math
from dataclasses import dataclass


# ============================================================
# VELOCITY TRIANGLES (combined "bowtie" style, shared U baseline)
# ============================================================


def _bowtie_apex(Wt, Vt, Vx, U):
    """Position of a station's apex on the shared A=(0,0)/C=(U,0) baseline,
    choosing the sign (+-Wt) that satisfies both |apex-A|=W and |apex-C|=V --
    the correct sign flips between stations (see turbine.py/compressor.py
    docstrings on the Vt = Wt +- U convention difference)."""
    V = math.hypot(Vt, Vx)
    for sign in (1.0, -1.0):
        apex = (sign * Wt, Vx)
        if math.isclose(math.hypot(apex[0] - U, apex[1]), V, rel_tol=1e-6, abs_tol=1e-3):
            return apex
    raise ValueError("could not construct a consistent velocity triangle apex")


def plot_velocity_triangles(Wt_in, Vt_in, Wt_out, Vt_out, Vx, U, label_in="2", label_out="3", ax=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, FancyArrowPatch

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 7.4))

    A, C = (0.0, 0.0), (U, 0.0)
    B = _bowtie_apex(Wt_in, Vt_in, Vx, U)
    D = _bowtie_apex(Wt_out, Vt_out, Vx, U)
    W_in, V_in = math.hypot(Wt_in, Vx), math.hypot(Vt_in, Vx)
    W_out, V_out = math.hypot(Wt_out, Vx), math.hypot(Vt_out, Vx)

    # Normalize left-to-right reading order (inlet on the left) -- which
    # sign _bowtie_apex picks can put either station on either side.
    if B[0] > D[0]:
        A, C = (U, 0.0), (0.0, 0.0)
        B, D = (U - B[0], B[1]), (U - D[0], D[1])

    c_u, c_w_in, c_v_in, c_w_out, c_v_out = "#2b2b2b", "#c1272d", "#0571b0", "#e8893a", "#2f9e44"

    ax.plot([B[0], D[0]], [Vx, Vx], ls=(0, (4, 4)), color="#9aa0a6", lw=1.2, zorder=1)
    ax.annotate(
        f"$V_x = {Vx:.0f}$ m/s", xy=((B[0] + D[0]) / 2, Vx), xytext=(0, 22),
        textcoords="offset points", ha="center", color="#5f6368", fontsize=11,
    )

    def vec(p0, p1, color, label, mag, xytext):
        ax.add_patch(
            FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=22, lw=2.6, color=color, shrinkA=0, shrinkB=0, zorder=4)
        )
        mid = (0.52 * p0[0] + 0.48 * p1[0], 0.52 * p0[1] + 0.48 * p1[1])
        ax.annotate(
            f"{label}\n{mag:.0f} m/s", xy=mid, xytext=xytext, textcoords="offset points",
            color=color, fontsize=12.5, ha="center", va="center", fontweight="bold",
        )

    def arc_to(apex, target, color, label, R, lr=1.32):
        d = (target[0] - apex[0], target[1] - apex[1])
        th = math.degrees(math.atan2(d[1], d[0])) % 360
        t1, t2 = min(th, 270.0), max(th, 270.0)
        ax.add_patch(Arc(apex, 2 * R, 2 * R, theta1=t1, theta2=t2, color=color, lw=1.8, zorder=5))
        mid = math.radians((t1 + t2) / 2)
        ax.annotate(
            label, xy=(apex[0] + lr * R * math.cos(mid), apex[1] + lr * R * math.sin(mid)),
            color=color, fontsize=13, ha="center", va="center", fontweight="bold",
        )

    u_left, u_right = (A, C) if A[0] <= C[0] else (C, A)
    vec(u_left, u_right, c_u, r"$\vec{U}$", U, (0, -20))
    vec(B, A, c_w_in, rf"$\vec{{W}}_{{{label_in}}}$", W_in, (-26, 6))
    vec(B, C, c_v_in, rf"$\vec{{V}}_{{{label_in}}}$", V_in, (20, 14))
    vec(D, A, c_w_out, rf"$\vec{{W}}_{{{label_out}}}$", W_out, (28, 6))
    vec(D, C, c_v_out, rf"$\vec{{V}}_{{{label_out}}}$", V_out, (26, 6))

    arc_to(B, C, c_v_in, rf"$\alpha_{{{label_in}}}$", 58)
    arc_to(B, A, c_w_in, rf"$\beta_{{{label_in}}}$", 34)
    arc_to(D, C, c_v_out, rf"$\alpha_{{{label_out}}}$", 58)
    arc_to(D, A, c_w_out, rf"$\beta_{{{label_out}}}$", 34)

    ax.plot(*B, "o", color="#444", ms=5, zorder=6)
    ax.plot(*D, "o", color="#444", ms=5, zorder=6)
    ax.annotate(f"Inlet ({label_in})", xy=B, xytext=(8, 12), textcoords="offset points", ha="left", fontsize=10.5, color="#444", style="italic")
    ax.annotate(f"Exit ({label_out})", xy=D, xytext=(-8, 12), textcoords="offset points", ha="right", fontsize=10.5, color="#444", style="italic")

    xs, ys = [A[0], C[0], B[0], D[0]], [A[1], C[1], B[1], D[1]]
    ax.set_xlim(min(xs) - 90, max(xs) + 90)
    ax.set_ylim(min(ys) - 70, max(ys) + 60)
    ax.set_xlabel("Tangential direction  [m/s]")
    ax.set_ylabel("Axial direction  [m/s]")
    ax.set_title(f"Velocity Triangles — Inlet ({label_in}) and Exit ({label_out})", fontsize=15, fontweight="bold", pad=14)
    ax.grid(True, color="#e8eaed", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#bbb")
    ax.spines["bottom"].set_color("#bbb")
    return ax


# ============================================================
# H-S LADDER DIAGRAM BUILDING BLOCKS
# ============================================================


def hline_label(ax, y, x_end, s_label, ls, text):
    ax.hlines(y, s_label, x_end, linestyles=ls, linewidth=1, color="black")
    ax.text(s_label, y, text, va="center", ha="right", color="black", fontstyle="italic")


def ke_arrow(ax, s_pos, y_lo, y_hi, label, ha="left"):
    ax.annotate("", xy=(s_pos, y_hi), xytext=(s_pos, y_lo), arrowprops=dict(arrowstyle="<|-|>", linewidth=1.0, color="black", shrinkA=0, shrinkB=0))
    ax.text(s_pos + (0.02 if ha == "left" else -0.02), 0.5 * (y_lo + y_hi), label, va="center", ha=ha, fontsize=11)


# ============================================================
# MULTI-STAGE REHEAT-FACTOR H-S DIAGRAM
# ============================================================
#
# The single-stage h-s ladders above (in stage_diagrams.py/turbine_stage_
# diagrams.py) show ONE stage's stator+rotor path. This is a different,
# classic diagram (Cohen/Rogers/Saravanamuttoo, "Gas Turbine Theory";
# Dixon & Hall) showing how SUCCESSIVE STAGES stack up: the real path
# 1->2->...->N, each stage's own LOCAL isentropic point (computed from that
# stage's own real, already-heated inlet -- e.g. "3s" from real state 2),
# and the OVERALL isentropic point at each stage's exit pressure,
# referenced back to the very first state's entropy (e.g. "3ss", sharing
# entropy with state 1 and "2s"). The gap between "3s" and "3ss" is the
# reheat effect: a real upstream stage leaves the working fluid hotter
# (higher entropy) than an ideal one would, so the next stage's own local
# ideal work is bigger than its share of the single overall ideal path --
# summed across all stages this is the reheat factor, RF >= 1 for a
# compression process.


@dataclass(frozen=True)
class HsPoint:
    label: str
    s: float
    h: float


@dataclass(frozen=True)
class MultistageHsResult:
    reheat_factor: float
    real_points: list  # HsPoint, index 0 = inlet, index i = stage i's real exit
    local_ideal_points: list  # HsPoint, index i-1 = stage i's own local isentropic point
    overall_ideal_points: list  # HsPoint, index i-2 = stage i's overall isentropic point (i>=2; stage 1's overall == its local)


def multistage_hs_diagram(stagnation_states, cp, gamma, ax=None):
    """stagnation_states: [(T0_0, P0_0), (T0_1, P0_1), ..., (T0_N, P0_N)] --
    state 0 is the inlet to stage 1, state i (i=1..N) is stage i's REAL
    exit condition. cp/gamma are assumed constant across all stages (valid
    for a single compressor's cold air, or a single turbine's hot gas).

    Returns (ax, MultistageHsResult)."""
    import matplotlib.pyplot as plt
    import numpy as np

    if len(stagnation_states) < 2:
        raise ValueError("stagnation_states needs at least 2 entries (inlet + >=1 stage exit)")

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 7.5))

    T0_ref, P0_ref = stagnation_states[0]
    R = cp * (gamma - 1.0) / gamma

    def entropy(T, P):
        return cp * math.log(T / T0_ref) - R * math.log(P / P0_ref)

    def enthalpy(T):
        return cp * T

    n_stages = len(stagnation_states) - 1

    real_points = [HsPoint("1", 0.0, enthalpy(T0_ref))]
    local_ideal_points = []
    overall_ideal_points = []
    total_local_ideal_work = 0.0

    T_prev, P_prev, s_prev = T0_ref, P0_ref, 0.0
    for i in range(1, n_stages + 1):
        T_i, P_i = stagnation_states[i]
        s_i = entropy(T_i, P_i)
        real_points.append(HsPoint(str(i + 1), s_i, enthalpy(T_i)))

        T_local_s = T_prev * (P_i / P_prev) ** ((gamma - 1.0) / gamma)
        local_ideal_points.append(HsPoint(f"{i + 1}s", s_prev, enthalpy(T_local_s)))
        total_local_ideal_work += enthalpy(T_local_s) - enthalpy(T_prev)

        if i >= 2:
            T_overall_s = T0_ref * (P_i / P0_ref) ** ((gamma - 1.0) / gamma)
            overall_ideal_points.append(HsPoint(f"{i + 1}ss", 0.0, enthalpy(T_overall_s)))

        T_prev, P_prev, s_prev = T_i, P_i, s_i

    h_N_ss = overall_ideal_points[-1].h if overall_ideal_points else local_ideal_points[-1].h
    overall_ideal_work = h_N_ss - real_points[0].h
    reheat_factor = total_local_ideal_work / overall_ideal_work

    # Isobar curves: for a perfect gas, h(s) at fixed P is exact given this
    # project's entropy convention (s = cp*ln(T/Tref) - R*ln(P/Pref)), so
    # these are the real thermodynamic isobars, not a visual approximation.
    all_s = [p.s for p in real_points + local_ideal_points + overall_ideal_points]
    s_span = max(all_s) - min(all_s) if max(all_s) > min(all_s) else 1.0
    for i, (T_i, P_i) in enumerate(stagnation_states):
        s_lo, s_hi = min(all_s) - 0.05 * s_span, max(all_s) + 0.05 * s_span
        s_samples = np.linspace(s_lo, s_hi, 60)
        T_samples = T0_ref * np.exp((s_samples + R * math.log(P_i / P0_ref)) / cp)
        ax.plot(s_samples, cp * T_samples, color="#9aa0a6", lw=1.0, ls="-", zorder=1)
        ax.annotate(f"$p_{{{i + 1}}}$", xy=(s_hi, cp * T_samples[-1]), xytext=(4, 0), textcoords="offset points", color="#5f6368", fontsize=10, va="center")

    ax.plot([p.s for p in real_points], [p.h for p in real_points], color="black", lw=2.5, zorder=5)
    ax.scatter([p.s for p in real_points], [p.h for p in real_points], color="black", s=45, zorder=6)
    for p in real_points:
        ax.annotate(p.label, xy=(p.s, p.h), xytext=(6, 6), textcoords="offset points", fontsize=12, fontweight="bold")

    for p in local_ideal_points + overall_ideal_points:
        ax.scatter([p.s], [p.h], color="#666666", s=35, zorder=6)
        ax.annotate(p.label, xy=(p.s, p.h), xytext=(6, -12), textcoords="offset points", fontsize=10, color="#444444")

    ax.set_xlabel("Entropy  s  (J/kg·K)")
    ax.set_ylabel("Enthalpy  h  (J/kg)")
    ax.set_title(f"Multi-Stage h-s Diagram (reheat factor = {reheat_factor:.4f})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)

    return ax, MultistageHsResult(
        reheat_factor=reheat_factor,
        real_points=real_points,
        local_ideal_points=local_ideal_points,
        overall_ideal_points=overall_ideal_points,
    )


# ============================================================
# PARAMETER TABLE
# ============================================================


def plot_parameter_table(sections, title, figsize=(14, 10)):
    """sections: [(header_text, [(parameter, symbol, unit, value_str), ...]), ...]"""
    import matplotlib.pyplot as plt

    rows = []
    header_rows = []
    sno = 1
    for header, entries in sections:
        rows.append(["", header, "", "", ""])
        header_rows.append(len(rows))
        for parameter, symbol, unit, value in entries:
            rows.append([sno, parameter, symbol, unit, value])
            sno += 1

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    columns = ["S.No", "Parameter", "Symbol", "Unit", "Value"]
    # Explicit bbox (rather than table.scale(), which can grow the table past
    # the axes bounds and collide with the title) reserves the top margin.
    table = ax.table(
        cellText=rows, colLabels=columns, cellLoc="left",
        colWidths=[0.06, 0.26, 0.1, 0.08, 0.16], bbox=[0, 0, 1, 0.93],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    for c in range(5):
        table[(0, c)].set_facecolor("#d0d0d0")
        table[(0, c)].set_text_props(weight="bold", ha="center", va="center")

    for r in range(1, len(rows) + 1):
        table[(r, 0)].get_text().set_ha("center")

    for r in header_rows:
        for c in range(5):
            cell = table[(r, c)]
            cell.set_facecolor("#303030")
            cell.set_edgecolor("#303030")
            cell.set_text_props(color="white", weight="bold", ha="center", va="center")
            if c != 1:
                cell.get_text().set_text("")

    for r in range(1, len(rows) + 1):
        if r not in header_rows and r % 2 == 0:
            for c in range(5):
                table[(r, c)].set_facecolor("#f2f2f2")

    ax.set_title(title, fontsize=18, fontweight="bold", style="italic", pad=20)
    fig.tight_layout()
    return fig, ax
