"""General post-processing helpers for a solved SU2 cascade case --
recovering static pressure from surface_flow.csv's raw conservative
variables (density, momentum, energy -- SU2 doesn't write a derived
Pressure column to that file, unlike flow.vtu) and computing the blade
surface Cp profile from it.

static_pressure_from_conservative is the standard ideal-gas relation
P = (gamma-1)*(rhoE - 0.5*rho*|V|^2) -- exact, not an empirical fit.
Verified against a real solved case: recovers SU2's own volume-mesh
Pressure field to 6+ significant figures at a matching point.
"""

import csv
from dataclasses import dataclass


def static_pressure_from_conservative(density: float, momentum_x: float, momentum_y: float, energy: float, gamma: float) -> float:
    if density <= 0.0:
        raise ValueError("density must be > 0")
    return (gamma - 1.0) * (energy - 0.5 * (momentum_x**2 + momentum_y**2) / density)


def pressure_coefficient(P: float, P_ref: float, rho_ref: float, V_ref: float) -> float:
    if rho_ref <= 0.0 or V_ref <= 0.0:
        raise ValueError("rho_ref and V_ref must be > 0")
    return (P - P_ref) / (0.5 * rho_ref * V_ref**2)


@dataclass(frozen=True)
class SurfaceFlowData:
    x: list
    y: list
    density: list
    momentum_x: list
    momentum_y: list
    energy: list


def read_surface_flow_csv(path: str) -> SurfaceFlowData:
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return SurfaceFlowData(
        x=[float(r["x"]) for r in rows],
        y=[float(r["y"]) for r in rows],
        density=[float(r["Density"]) for r in rows],
        momentum_x=[float(r["Momentum_x"]) for r in rows],
        momentum_y=[float(r["Momentum_y"]) for r in rows],
        energy=[float(r["Energy"]) for r in rows],
    )


def blade_surface_pressure_profile(surface_csv_path: str, gamma: float):
    """Returns (x, y, P) lists for every point on the surface_flow.csv
    (WALL_MID) extract."""
    data = read_surface_flow_csv(surface_csv_path)
    P = [
        static_pressure_from_conservative(rho, mx, my, e, gamma)
        for rho, mx, my, e in zip(data.density, data.momentum_x, data.momentum_y, data.energy)
    ]
    return data.x, data.y, P


def blade_surface_cp_profile(surface_csv_path: str, gamma: float, P_ref: float, rho_ref: float, V_ref: float):
    """Returns (x, y, Cp) lists for the blade surface, referenced to the
    given freestream/inlet conditions."""
    x, y, P = blade_surface_pressure_profile(surface_csv_path, gamma)
    cp = [pressure_coefficient(p, P_ref, rho_ref, V_ref) for p in P]
    return x, y, cp


def plot_blade_cp_profile(surface_csv_path: str, gamma: float, P_ref: float, rho_ref: float, V_ref: float, ax=None):
    import matplotlib.pyplot as plt

    x, y, cp = blade_surface_cp_profile(surface_csv_path, gamma, P_ref, rho_ref, V_ref)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(x, cp, s=6, color="#0571b0")
    ax.invert_yaxis()  # convention: Cp axis inverted (suction peak reads as a high point)
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("$C_p$")
    ax.set_title("Blade Surface Pressure Coefficient", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    return ax
