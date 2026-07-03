"""Rotor blade chord, pitch, and blade-count sizing (Zweifel criterion) from
mean-line stage results -- adapted from Turbine Stage Design.ipynb, Part 3
(the "ROTOR BLADE DESIGN" cell).

Takes the mean-line stage's own outputs (blade height at rotor inlet/exit,
mean diameter, inlet/exit relative flow angles) as plain floats rather than
importing Compressor Calculations' result types directly, so this folder
stays self-contained -- the caller (e.g. design_engine.py) wires the two
together.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BladeSizingInputs:
    aspect_ratio: float = 1.0  # AR = blade_height_in / axial_chord
    zweifel_coefficient: float = 1.0  # Z, tangential-loading criterion
    stagger_angle_deg: float = 39.0
    thickness_to_chord: float = 0.25  # t_max / axial_chord
    te_radius: float = 0.0007  # m
    le_radius_fraction_of_pitch: float = 0.05

    def __post_init__(self):
        if self.aspect_ratio <= 0.0:
            raise ValueError("aspect_ratio must be > 0")
        if self.zweifel_coefficient <= 0.0:
            raise ValueError("zweifel_coefficient must be > 0")
        if not (0.0 <= self.stagger_angle_deg < 90.0):
            raise ValueError("stagger_angle_deg must be in [0, 90)")
        if not (0.0 < self.thickness_to_chord < 1.0):
            raise ValueError("thickness_to_chord must be in (0, 1)")
        if self.te_radius <= 0.0:
            raise ValueError("te_radius must be > 0")
        if not (0.0 < self.le_radius_fraction_of_pitch < 1.0):
            raise ValueError("le_radius_fraction_of_pitch must be in (0, 1)")


@dataclass(frozen=True)
class BladeSizingResult:
    chord: float  # m, true (mean) chord
    axial_chord: float  # m, Cx
    pitch: float  # m, blade-to-blade spacing at the mean radius
    pitch_to_chord: float  # s / c
    pitch_to_axial_chord: float  # s / Cx
    num_blades: int
    lean_angle_deg: float  # epsilon, from span-wise (hub-to-tip) height change
    le_radius: float  # m
    te_radius: float  # m
    max_thickness: float  # m


def size_rotor_blade(
    blade_height_in: float,
    blade_height_out: float,
    mean_diameter: float,
    beta_in_deg: float,
    beta_out_deg: float,
    design: BladeSizingInputs = BladeSizingInputs(),
) -> BladeSizingResult:
    """beta_in_deg/beta_out_deg are the relative (or absolute, for a stator)
    flow angles at rotor inlet/exit -- whichever pair the blade row actually
    turns the flow between."""
    chord = blade_height_in / design.aspect_ratio
    axial_chord = chord * math.cos(math.radians(design.stagger_angle_deg))
    lean_angle_deg = math.degrees(math.atan((blade_height_out - blade_height_in) / axial_chord))

    beta_in_r = math.radians(beta_in_deg)
    beta_out_r = math.radians(beta_out_deg)
    pitch = (design.zweifel_coefficient * axial_chord) / (
        2.0 * math.cos(beta_out_r) ** 2 * (math.tan(beta_out_r) + math.tan(beta_in_r))
    )
    num_blades = math.ceil((math.pi * mean_diameter) / pitch)

    le_radius = design.le_radius_fraction_of_pitch * pitch
    max_thickness = design.thickness_to_chord * axial_chord

    return BladeSizingResult(
        chord=chord,
        axial_chord=axial_chord,
        pitch=pitch,
        pitch_to_chord=pitch / chord,
        pitch_to_axial_chord=pitch / axial_chord,
        num_blades=num_blades,
        lean_angle_deg=lean_angle_deg,
        le_radius=le_radius,
        te_radius=design.te_radius,
        max_thickness=max_thickness,
    )
