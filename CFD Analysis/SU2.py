"""SU2 RANS configuration, execution, and post-processing for the 3-blade
cascade CFD case -- adapted from Turbine Stage Design.ipynb, Part 5c (config
+ run) and the mesh-quality-check cell (post-processing a solved flow.vtu).

Solves the blade row in the RELATIVE (rotating) frame as a fixed 3-blade
cascade: inlet = relative stagnation conditions, outlet = static exit
pressure, WALL_MID = viscous no-slip (the measured middle blade), WALL_UP/
WALL_LO = inviscid slip (neighbour walls standing in for periodicity).
"""

import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class SU2CascadeInputs:
    gamma: float
    gas_constant: float
    relative_inlet_mach: float  # e.g. Mw2 for a turbine rotor
    static_temperature: float  # e.g. T2, used for FREESTREAM_TEMPERATURE
    static_pressure: float  # e.g. P2, used for FREESTREAM_PRESSURE
    inlet_total_temperature: float  # relative total temp, e.g. T02_rel
    inlet_total_pressure: float  # relative total pressure, e.g. P02_rel
    inlet_flow_angle_deg: float  # relative inlet flow angle, e.g. beta2_deg
    outlet_static_pressure: float  # e.g. P3
    reference_length: float  # axial chord, m
    reynolds_number: float = 1.0e6
    inner_iterations: int = 4000

    def __post_init__(self):
        if self.gamma <= 1.0:
            raise ValueError("gamma must be > 1")
        if self.gas_constant <= 0.0:
            raise ValueError("gas_constant must be > 0")
        if self.relative_inlet_mach <= 0.0:
            raise ValueError("relative_inlet_mach must be > 0")
        if self.static_temperature <= 0.0 or self.inlet_total_temperature <= 0.0:
            raise ValueError("temperatures must be > 0")
        if self.static_pressure <= 0.0 or self.inlet_total_pressure <= 0.0 or self.outlet_static_pressure <= 0.0:
            raise ValueError("pressures must be > 0")
        if self.reference_length <= 0.0:
            raise ValueError("reference_length must be > 0")
        if self.inner_iterations <= 0:
            raise ValueError("inner_iterations must be > 0")


def write_su2_config(inputs: SU2CascadeInputs, mesh_filename: str, config_path: str) -> str:
    import math

    dir_x = math.cos(math.radians(inputs.inlet_flow_angle_deg))
    dir_y = math.sin(math.radians(inputs.inlet_flow_angle_deg))

    cfg = f"""%%% SU2 configuration -- rotor blade 3-blade cascade (relative frame)
SOLVER= RANS
KIND_TURB_MODEL= SA
MATH_PROBLEM= DIRECT
RESTART_SOL= NO

%%% FLUID PROPERTIES
FLUID_MODEL= IDEAL_GAS
GAMMA_VALUE= {inputs.gamma:.4f}
GAS_CONSTANT= {inputs.gas_constant:.4f}

%%% FREE-STREAM (relative-frame rotor inlet, static reference conditions)
MACH_NUMBER= {inputs.relative_inlet_mach:.4f}
FREESTREAM_TEMPERATURE= {inputs.static_temperature:.2f}
FREESTREAM_PRESSURE= {inputs.static_pressure:.2f}
AOA= 0.0
SIDESLIP_ANGLE= 0.0
INIT_OPTION= TD_CONDITIONS
REYNOLDS_NUMBER= {inputs.reynolds_number:.1E}
KIND_TRANS_MODEL= NONE

%%% REFERENCE VALUES
REF_LENGTH= {inputs.reference_length:.8f}
REF_AREA= 0.0
REF_DIMENSIONALIZATION= DIMENSIONAL

%%% BOUNDARY CONDITIONS (no periodicity -- neighbour surfaces are walls)
INLET_TYPE= TOTAL_CONDITIONS
MARKER_INLET= (INLET, {inputs.inlet_total_temperature:.4f}, {inputs.inlet_total_pressure:.4f}, {dir_x:.10f}, {dir_y:.10f}, 0.0)
MARKER_OUTLET= (OUTLET, {inputs.outlet_static_pressure:.4f})
MARKER_HEATFLUX= (WALL_MID, 0.0)
MARKER_EULER= (WALL_UP, WALL_LO)
MARKER_PLOTTING= (WALL_MID)
MARKER_MONITORING= (WALL_MID)

%%% NUMERICS (CFL ramp from 2 -> 50 for a robust start)
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 2.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= (0.1, 1.2, 0.5, 50.0)
CONV_NUM_METHOD_FLOW= JST
MUSCL_FLOW= NO
JST_SENSOR_COEFF= (0.5, 0.02)
TIME_DISCRE_FLOW= EULER_IMPLICIT
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO
TIME_DISCRE_TURB= EULER_IMPLICIT
CFL_REDUCTION_TURB= 0.5

%%% LINEAR SOLVER
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-4
LINEAR_SOLVER_ITER= 15

%%% CONVERGENCE
INNER_ITER= {inputs.inner_iterations}
CONV_RESIDUAL_MINVAL= -9
CONV_STARTITER= 50
CONV_FIELD= RMS_DENSITY

%%% I/O
MESH_FILENAME= {mesh_filename}
MESH_FORMAT= SU2
OUTPUT_FILES= (PARAVIEW, SURFACE_CSV, RESTART)
RESTART_FILENAME= restart_flow.dat
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
CONV_FILENAME= history
SCREEN_WRT_FREQ_INNER= 250
OUTPUT_WRT_FREQ= 1000
"""
    with open(config_path, "w") as f:
        f.write(cfg)
    return config_path


def run_su2(config_path: str, work_dir: str, su2_binary: str, stream_output: bool = True):
    """Runs SU2_CFD as a subprocess (blocking). Returns (returncode, output_lines)."""
    if not os.path.isfile(su2_binary):
        raise FileNotFoundError(f"SU2_CFD binary not found at {su2_binary}")

    proc = subprocess.Popen(
        [su2_binary, os.path.basename(config_path)],
        cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    lines = []
    for line in proc.stdout:
        if stream_output:
            print(line, end="")
        lines.append(line)
    proc.wait()
    return proc.returncode, lines


def plot_mesh_quality(flow_vtu_path: str, blade, axial_chord: float, pitch: float, n_surface: int = 600):
    """Post-processing check of the solved mesh (Part 5, mesh quality check):
    full-domain triangulation plus the middle blade with its neighbour
    walls, so meshing/domain problems show up before looking at any results."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import numpy as np
    import pyvista as pv

    mesh = pv.read(flow_vtu_path)
    mesh_tri = mesh.triangulate()
    cells = mesh_tri.cells.reshape(-1, 4)
    tris = cells[:, 1:]
    xm, ym = mesh_tri.points[:, 0], mesh_tri.points[:, 1]
    triang = mtri.Triangulation(xm, ym, tris)

    u = np.linspace(0.0, 1.0, n_surface)
    us = np.real(blade.get_upper_side_coordinates(u))
    ls = np.real(blade.get_lower_side_coordinates(u))
    xs_us, ys_us = us[0, :] * axial_chord, us[1, :] * axial_chord
    xs_ls, ys_ls = ls[0, :] * axial_chord, ls[1, :] * axial_chord
    le_x, le_y = float(xs_us[-1]), float(ys_us[-1])
    te_x, te_y = float(xs_us[0]), float(ys_us[0])

    fig, axes = plt.subplots(1, 2, figsize=(9, 9))

    axes[0].triplot(triang, "k-", lw=0.2, alpha=0.6)
    axes[0].set_aspect("equal")
    axes[0].set_title("Full Domain Mesh (3-blade cascade)", fontsize=13)
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")

    axes[1].triplot(triang, "k-", lw=0.3, alpha=0.7)
    axes[1].plot(xs_us, ys_us, "b-", lw=2, label="Middle blade -- suction")
    axes[1].plot(xs_ls, ys_ls, "r-", lw=2, label="Middle blade -- pressure")
    axes[1].plot(xs_ls, ys_ls + pitch, "r--", lw=1.8, alpha=0.85, label="WALL_UP = upper blade pressure side")
    axes[1].plot(xs_us, ys_us - pitch, "b--", lw=1.8, alpha=0.85, label="WALL_LO = lower blade suction side")

    x_lo, x_hi = le_x - 0.10 * axial_chord, te_x + 0.10 * axial_chord
    y_lo = (ys_us - pitch).min() - 0.02 * axial_chord
    y_hi = (ys_ls + pitch).max() + 0.02 * axial_chord
    axes[1].set_xlim([x_lo, x_hi])
    axes[1].set_ylim([y_lo, y_hi])
    axes[1].set_aspect("equal")
    axes[1].set_title("Middle blade & neighbour walls", fontsize=13)
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[1].legend(fontsize=9, loc="lower left")

    fig.suptitle("Mesh Quality Check -- 3-Blade Cascade", fontsize=14, fontweight="bold")
    fig.tight_layout()

    return fig, axes, {"n_points": mesh.n_points, "n_cells": mesh.n_cells, "x_range": (float(xm.min()), float(xm.max())), "y_range": (float(ym.min()), float(ym.max()))}
