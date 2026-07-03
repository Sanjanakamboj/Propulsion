"""T-s and P-v diagrams for the simple (closed) Brayton cycle variants --
ideal_cycle.py, real_cycle.py, regenerative_cycle.py, intercooling.py,
reheating.py -- all of which return a plain ordered list of CycleState
under one constant gas. Unlike diagrams.py (the open turbojet path), these
ARE closed cycles: the last state connects back to the first.
"""

import math


def _segment(gas, T_a, P_a, T_b, P_b, s_a, n=60):
    """Sample T, P, s, v along one leg, continuing entropy from s_a. Same
    interpolation strategy as diagrams.py's helper of the same name: an
    isobaric leg stays at constant P, otherwise T and P are connected by a
    polytropic-looking curve through the two given endpoints (this is a
    visualization aid, not a claim that the real process is polytropic --
    for isentropic legs it reduces exactly to the isentrope since s_a=s_b)."""
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


def build_cycle_path(states, gas):
    """Concatenate segments between consecutive CycleState points, closing
    the loop from the last state back to the first. Returns (T, P, s, v,
    markers) where markers maps each state's label to its (T, P, s, v)."""
    T, P, s, v = [], [], [], []
    markers = {states[0].label: (states[0].T, states[0].P, 0.0, gas.R * states[0].T / states[0].P)}

    loop_states = list(states) + [states[0]]
    for a, b in zip(loop_states[:-1], loop_states[1:]):
        s_a = s[-1] if s else 0.0
        T_seg, P_seg, s_seg, v_seg = _segment(gas, a.T, a.P, b.T, b.P, s_a)
        T.extend(T_seg)
        P.extend(P_seg)
        s.extend(s_seg)
        v.extend(v_seg)
        markers[b.label] = (T[-1], P[-1], s[-1], v[-1])

    return T, P, s, v, markers


def plot_ts_diagram(states, gas, title="Brayton Cycle: T-s Diagram", ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    T, _, s, _, markers = build_cycle_path(states, gas)
    ax.plot(s, T, "-", color="tab:blue", linewidth=2)

    for label, (Tp, Pp, sp, _) in markers.items():
        ax.plot(sp, Tp, "o", color="black", zorder=5)
        ax.annotate(f"{label}\n{Pp / 1e3:.0f} kPa", (sp, Tp), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Specific entropy, s [J/(kg·K)] (relative to state 1)")
    ax.set_ylabel("Temperature, T [K]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_pv_diagram(states, gas, title="Brayton Cycle: P-v Diagram", ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    _, P, _, v, markers = build_cycle_path(states, gas)
    P_kpa = [p / 1e3 for p in P]
    ax.plot(v, P_kpa, "-", color="tab:blue", linewidth=2)

    for label, (_, Pp, _, vp) in markers.items():
        ax.plot(vp, Pp / 1e3, "o", color="black", zorder=5)
        ax.annotate(label, (vp, Pp / 1e3), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Specific volume, v [m³/kg]")
    ax.set_ylabel("Pressure, P [kPa]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_cycle_diagrams(states, gas, title_prefix="Brayton Cycle", save_path: str | None = None):
    import matplotlib.pyplot as plt

    fig, (ax_ts, ax_pv) = plt.subplots(1, 2, figsize=(13, 5))
    plot_ts_diagram(states, gas, title=f"{title_prefix}: T-s Diagram", ax=ax_ts)
    plot_pv_diagram(states, gas, title=f"{title_prefix}: P-v Diagram", ax=ax_pv)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig, (ax_ts, ax_pv)
