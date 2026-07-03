"""Blade-to-blade passage geometry (geometric throat, passage-width
variation along the axial chord) from a generated 2D blade section --
adapted from Turbine Stage Design.ipynb, Part 4 ("BLADE PASSAGE GEOMETRY").
"""

import os
import sys
from dataclasses import dataclass

_PARABLADE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parablade-master", "parablade")
if _PARABLADE_DIR not in sys.path:
    sys.path.insert(0, _PARABLADE_DIR)

import numpy as np
from scipy.interpolate import interp1d


@dataclass(frozen=True)
class PassageGeometryResult:
    throat_over_axial_chord: float  # o / Cx
    throat_over_pitch: float  # o / s (throat-to-pitch ratio)
    throat_axial_location_over_cx: float  # x / Cx where the minimum gap occurs
    throat: float  # m, dimensional
    x: np.ndarray  # m, axial stations
    suction_surface_y: np.ndarray  # m, blade 1 suction surface
    pressure_surface_y: np.ndarray  # m, blade 2 (pitched) pressure surface
    gap: np.ndarray  # m, local passage width


def compute_passage_geometry(blade, axial_chord: float, pitch: float, n: int = 1200) -> PassageGeometryResult:
    u_fine = np.linspace(0.0, 1.0, 2000)
    us = np.real(blade.get_upper_side_coordinates(u_fine))  # (2, N) suction surface
    ls = np.real(blade.get_lower_side_coordinates(u_fine))  # (2, N) pressure surface

    s_over_cx = pitch / axial_chord

    us_s = us[:, np.argsort(us[0, :])]
    ls_s = ls[:, np.argsort(ls[0, :])]

    x_lo = max(us_s[0, 0], ls_s[0, 0])
    x_hi = min(us_s[0, -1], ls_s[0, -1])
    x_c = np.linspace(x_lo, x_hi, n)

    y_suc = interp1d(us_s[0, :], us_s[1, :], kind="linear")(x_c)  # suction surface
    y_pres = interp1d(ls_s[0, :], ls_s[1, :], kind="linear")(x_c) + s_over_cx  # adjacent blade's pressure surface

    gap = y_pres - y_suc
    i_thr = int(np.argmin(gap))

    return PassageGeometryResult(
        throat_over_axial_chord=float(gap[i_thr]),
        throat_over_pitch=float(gap[i_thr] / s_over_cx),
        throat_axial_location_over_cx=float(x_c[i_thr]),
        throat=float(gap[i_thr] * axial_chord),
        x=x_c * axial_chord,
        suction_surface_y=y_suc * axial_chord,
        pressure_surface_y=y_pres * axial_chord,
        gap=gap * axial_chord,
    )


def plot_blade_passage(blade, axial_chord: float, pitch: float, passage: PassageGeometryResult, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 9))

    s_over_cx = pitch / axial_chord
    u_fine = np.linspace(0.0, 1.0, 600)
    us = np.real(blade.get_upper_side_coordinates(u_fine))
    ls = np.real(blade.get_lower_side_coordinates(u_fine))
    cl = np.real(blade.get_camberline_coordinates(u_fine))
    us_s = us[:, np.argsort(us[0, :])]
    ls_s = ls[:, np.argsort(ls[0, :])]

    ax.plot(us[0, :], us[1, :], color="steelblue", linewidth=2.2, label="Blade 1 -- suction surface")
    ax.fill_between(us_s[0, :], us_s[1, :] - 0.35, us_s[1, :], color="steelblue", alpha=0.18)

    ax.plot(ls[0, :], ls[1, :] + s_over_cx, color="crimson", linewidth=2.2, label="Blade 2 -- pressure surface")
    ax.fill_between(ls_s[0, :], ls_s[1, :] + s_over_cx, ls_s[1, :] + s_over_cx + 0.35, color="crimson", alpha=0.18)

    x_c, y_suc, y_pres = passage.x / axial_chord, passage.suction_surface_y / axial_chord, passage.pressure_surface_y / axial_chord
    ax.fill_between(x_c, y_suc, y_pres, alpha=0.14, color="limegreen", label="Passage channel")
    ax.plot(cl[0, :], cl[1, :] + s_over_cx / 2.0, linestyle="--", color="dimgray", linewidth=1.2, label="Camberline (midpassage)")

    x_thr = passage.throat_axial_location_over_cx
    i_thr = int(np.argmin(np.abs(x_c - x_thr)))
    ax.annotate("", xy=(x_thr, y_pres[i_thr]), xytext=(x_thr, y_suc[i_thr]), arrowprops=dict(arrowstyle="<|-|>", color="darkgreen", lw=1.8, mutation_scale=14))
    ax.text(x_thr + 0.03, 0.5 * (y_suc[i_thr] + y_pres[i_thr]), f"Throat\n$o/C_x$ = {passage.throat_over_axial_chord:.3f}", color="darkgreen", fontsize=10, va="center")

    ax.annotate("", xy=(1.10, s_over_cx), xytext=(1.10, 0.0), arrowprops=dict(arrowstyle="<|-|>", color="black", lw=1.4, mutation_scale=14))
    ax.text(1.12, s_over_cx / 2.0, f"$s/C_x$\n= {s_over_cx:.3f}", fontsize=10, va="center")

    for xv, lbl in [(0.0, "LE"), (1.0, "TE")]:
        ax.axvline(xv, color="k", linestyle=":", linewidth=0.9, alpha=0.5)
        ax.text(xv + 0.01, s_over_cx + 0.06, lbl, fontsize=9, color="black")

    ax.set_xlim(-0.08, 1.22)
    ax.set_xlabel(r"$x / C_x$  (axial direction)", fontsize=13)
    ax.set_ylabel(r"$y / C_x$  (pitchwise direction)", fontsize=13)
    ax.set_title("Rotor Blade Passage Geometry -- Mean-line Section", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    return ax


def plot_passage_width(passage: PassageGeometryResult, axial_chord: float, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    x_c = passage.x / axial_chord
    gap = passage.gap / axial_chord

    ax.plot(x_c, gap, color="royalblue", linewidth=2.2, label=r"Gap  $g/C_x$")
    ax.axhline(passage.throat_over_axial_chord, color="red", linestyle="--", linewidth=1.4, label=f"Throat  $o/C_x$ = {passage.throat_over_axial_chord:.4f}")
    ax.axvline(passage.throat_axial_location_over_cx, color="darkgreen", linestyle=":", linewidth=1.2, label=f"Throat at  $x/C_x$ = {passage.throat_axial_location_over_cx:.3f}")
    ax.fill_between(x_c, 0, gap, alpha=0.10, color="royalblue")

    ax.set_xlabel(r"$x / C_x$", fontsize=12)
    ax.set_ylabel(r"Passage gap  $g / C_x$", fontsize=12)
    ax.set_title("Passage Width Variation along Axial Chord", fontsize=13, fontweight="bold")
    ax.legend(loc="lower center", fontsize=10)
    ax.grid(True, alpha=0.3)
    return ax
