"""Solver-agnostic description of the 3-blade cascade's boundary
conditions -- the same physical setup (relative-frame total-conditions
inlet, static-pressure outlet, a viscous middle blade with inviscid
neighbour walls standing in for periodicity) that SU2.py's
write_su2_config already writes into an SU2 .cfg file. Factoring the flow-
direction math and BC values out standalone lets su2_runner and (once
built) an OpenFOAM-equivalent runner share one physical description, even
though each solver's file format for expressing it differs.
"""

import math
from dataclasses import dataclass

# Shared marker/patch names, used consistently across mesh.py's GMSH
# physical groups, SU2.py's .cfg markers, and (once built) OpenFOAM's
# boundary patches.
INLET = "INLET"
OUTLET = "OUTLET"
WALL_MID = "WALL_MID"  # the measured blade -- viscous no-slip
WALL_UP = "WALL_UP"  # neighbour blade's pressure side -- inviscid slip
WALL_LO = "WALL_LO"  # neighbour blade's suction side -- inviscid slip


def flow_direction_vector(angle_deg: float) -> tuple:
    """Unit direction vector (dir_x, dir_y, dir_z) for a flow angle
    measured from the axial (x) direction."""
    return (math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg)), 0.0)


@dataclass(frozen=True)
class CascadeBoundaryConditions:
    """Physical description of the cascade's boundary conditions,
    independent of solver file format."""

    inlet_total_temperature: float
    inlet_total_pressure: float
    inlet_flow_angle_deg: float
    outlet_static_pressure: float

    def __post_init__(self):
        if self.inlet_total_temperature <= 0.0:
            raise ValueError("inlet_total_temperature must be > 0")
        if self.inlet_total_pressure <= 0.0:
            raise ValueError("inlet_total_pressure must be > 0")
        if self.outlet_static_pressure <= 0.0:
            raise ValueError("outlet_static_pressure must be > 0")

    @property
    def inlet_direction(self) -> tuple:
        return flow_direction_vector(self.inlet_flow_angle_deg)
