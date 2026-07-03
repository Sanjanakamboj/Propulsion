"""T-s and P-v diagrams for the staged turbojet engine model.

Each engine stage (Inlet, Compressor, Combustor, Turbine, Nozzle) contributes
one or two legs to the path -- an isentropic piece plus an isobaric loss
piece wherever the stage isn't ideal -- using that leg's own gas properties
(cold air up to the combustor, hot combustion gas from the combustor on).
Entropy is accumulated leg-by-leg from s = 0 at the ambient state; this is a
real (open) engine, so the path runs ambient -> ... -> nozzle exit and is not
closed into a loop.
"""

import math

STAGE_COLORS = {
    "Inlet": "tab:gray",
    "Compressor": "tab:blue",
    "Combustor": "tab:orange",
    "Turbine": "tab:green",
    "Nozzle": "tab:purple",
}

# For each stage, the (gas, T_ideal, P_ideal) of its loss-free companion
# state, given (inlet, exit, extra). Compressor/Turbine/Nozzle compare
# against the isentropic state at the SAME exit pressure (the classic
# isentropic-efficiency picture); Inlet compares at the same temperature
# (ram heating is loss-free, only P0 recovery is lossy); Combustor compares
# against a no-pressure-loss combustion reaching the same TIT.
_IDEAL_STAGE_STATE = {
    "Inlet": lambda inlet, exit, extra: (exit.gas, exit.T0, extra["P0_ideal"]),
    "Compressor": lambda inlet, exit, extra: (inlet.gas, extra["T0_ideal"], exit.P0),
    "Combustor": lambda inlet, exit, extra: (exit.gas, exit.T0, inlet.P0),
    "Turbine": lambda inlet, exit, extra: (inlet.gas, extra["T0_ideal"], exit.P0),
    "Nozzle": lambda inlet, exit, extra: (inlet.gas, extra["T0_ideal"], exit.P0),
}


def _segment(gas, T_a, P_a, T_b, P_b, s_a, n=60):
    """Sample T, P, s, v along one leg, continuing entropy from s_a."""
    R = gas.R
    same_P = math.isclose(P_a, P_b, rel_tol=1e-9)
    same_T = math.isclose(T_a, T_b, rel_tol=1e-9)
    T_vals = [T_a + (T_b - T_a) * i / (n - 1) for i in range(n)]

    if same_P:
        P_vals = [P_a] * n
    elif same_T:
        P_vals = [P_a + (P_b - P_a) * i / (n - 1) for i in range(n)]
    else:
        k = math.log(P_b / P_a) / math.log(T_b / T_a)
        P_vals = [P_a * (T / T_a) ** k for T in T_vals]

    s_vals = [s_a + gas.cp * math.log(T / T_a) - R * math.log(P / P_a) for T, P in zip(T_vals, P_vals)]
    v_vals = [R * T / P for T, P in zip(T_vals, P_vals)]
    return T_vals, P_vals, s_vals, v_vals


def build_engine_path(stage_records):
    """Concatenate every stage's path segments into full T, P, s, v arrays,
    tagged per-sample with the owning stage name, plus station markers
    (label -> (T, P, s, v)) at each stage boundary."""
    T, P, s, v, stage_tags = [], [], [], [], []
    markers = {}

    first_inlet = stage_records[0]["inlet"]
    markers[first_inlet.label] = (first_inlet.T0, first_inlet.P0, 0.0, first_inlet.gas.R * first_inlet.T0 / first_inlet.P0)

    for rec in stage_records:
        stage, inlet, exit_station, extra = rec["stage"], rec["inlet"], rec["exit"], rec["extra"]
        for leg_label, gas, T_a, P_a, T_b, P_b in stage.path_segments(inlet, exit_station, extra):
            s_a = s[-1] if s else 0.0
            T_seg, P_seg, s_seg, v_seg = _segment(gas, T_a, P_a, T_b, P_b, s_a)
            T.extend(T_seg)
            P.extend(P_seg)
            s.extend(s_seg)
            v.extend(v_seg)
            stage_tags.extend([rec["name"]] * len(T_seg))
        markers[exit_station.label] = (T[-1], P[-1], s[-1], v[-1])

    return T, P, s, v, stage_tags, markers


def ideal_state_points(stage_records, markers):
    """Loss-free companion state for each stage's exit, keyed by
    '<exit_label>s' (e.g. '3s'), used to visualize where each stage's
    inefficiency generates entropy against the real exit marker."""
    points = {}
    for rec in stage_records:
        config = _IDEAL_STAGE_STATE.get(rec["name"])
        if config is None:
            continue
        inlet = rec["inlet"]
        gas, T_ideal, P_ideal = config(inlet, rec["exit"], rec["extra"])
        s_inlet = markers[inlet.label][2]
        s_ideal = s_inlet + gas.cp * math.log(T_ideal / inlet.T0) - gas.R * math.log(P_ideal / inlet.P0)
        v_ideal = gas.R * T_ideal / P_ideal
        points[f"{rec['exit'].label}s"] = (T_ideal, P_ideal, s_ideal, v_ideal, rec["exit"].label)
    return points


def _plot_ideal_points(ax, markers, ideal_points, x_index, y_index, y_scale=1.0):
    for ideal_label, point in ideal_points.items():
        *coords, real_label = point
        x_ideal, y_ideal = coords[x_index], coords[y_index] * y_scale
        x_real, y_real = markers[real_label][x_index], markers[real_label][y_index] * y_scale
        ax.plot([x_real, x_ideal], [y_real, y_ideal], "--", color="gray", linewidth=1, zorder=4)
        ax.plot(x_ideal, y_ideal, "o", markerfacecolor="none", markeredgecolor="black", markersize=6, zorder=6)
        ax.annotate(
            ideal_label, (x_ideal, y_ideal), textcoords="offset points", xytext=(-8, -10), fontsize=8, color="dimgray"
        )


def _plot_by_stage(ax, x, y, stage_tags):
    start = 0
    plotted_labels = set()
    for i in range(1, len(stage_tags) + 1):
        if i == len(stage_tags) or stage_tags[i] != stage_tags[start]:
            name = stage_tags[start]
            color = STAGE_COLORS.get(name, "black")
            label = name if name not in plotted_labels else None
            ax.plot(x[start:i], y[start:i], "-", color=color, linewidth=2, label=label)
            plotted_labels.add(name)
            start = i


def _closing_reference_line(ax, markers, x_index, y_index, y_scale, stage_records):
    """Thin dashed guide between ambient (station '0') and the nozzle exit,
    showing whether the nozzle fully expanded back to ambient pressure."""
    first_label = stage_records[0]["inlet"].label
    last_label = stage_records[-1]["exit"].label
    x0, y0 = markers[first_label][x_index], markers[first_label][y_index] * y_scale
    x9, y9 = markers[last_label][x_index], markers[last_label][y_index] * y_scale
    ax.plot([x9, x0], [y9, y0], ":", color="silver", linewidth=1, zorder=1)


def plot_ts_diagram(stage_records, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    T, _, s, _, stage_tags, markers = build_engine_path(stage_records)
    _plot_by_stage(ax, s, T, stage_tags)
    _closing_reference_line(ax, markers, x_index=2, y_index=0, y_scale=1.0, stage_records=stage_records)

    ideal_points = ideal_state_points(stage_records, markers)
    _plot_ideal_points(ax, markers, ideal_points, x_index=2, y_index=0)

    for label, (Tp, Pp, sp, _) in markers.items():
        ax.plot(sp, Tp, "o", color="black", zorder=5)
        ax.annotate(f"{label}\n{Pp / 1e3:.0f} kPa", (sp, Tp), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Specific entropy, s [J/(kg·K)] (relative to ambient)")
    ax.set_ylabel("Stagnation temperature, T0 [K]")
    ax.set_title("Turbojet Cycle: T-s Diagram")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_pv_diagram(stage_records, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    _, P, _, v, stage_tags, markers = build_engine_path(stage_records)
    P_kpa = [p / 1e3 for p in P]
    _plot_by_stage(ax, v, P_kpa, stage_tags)
    _closing_reference_line(ax, markers, x_index=3, y_index=1, y_scale=1e-3, stage_records=stage_records)

    ideal_points = ideal_state_points(stage_records, markers)
    _plot_ideal_points(ax, markers, ideal_points, x_index=3, y_index=1, y_scale=1e-3)

    for label, (_, Pp, _, vp) in markers.items():
        ax.plot(vp, Pp / 1e3, "o", color="black", zorder=5)
        ax.annotate(label, (vp, Pp / 1e3), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Specific volume, v [m³/kg]")
    ax.set_ylabel("Stagnation pressure, P0 [kPa]")
    ax.set_title("Turbojet Cycle: P-v Diagram")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_cycle_diagrams(stage_records, save_path: str | None = None):
    import matplotlib.pyplot as plt

    fig, (ax_ts, ax_pv) = plt.subplots(1, 2, figsize=(13, 5))
    plot_ts_diagram(stage_records, ax=ax_ts)
    plot_pv_diagram(stage_records, ax=ax_pv)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig, (ax_ts, ax_pv)
