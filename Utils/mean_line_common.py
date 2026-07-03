"""Shared axial turbomachinery mean-line helpers, used by both compressor.py
and turbine.py -- blade speed sizing from stage loading, mean-radius/annulus
sizing, and the AN^2 mechanical design parameter.
"""

import math
from dataclasses import dataclass


def blade_speed_from_loading(specific_work: float, stage_loading: float, blade_speed_limit: float):
    """Solve U from the stage loading coefficient psi = specific_work / U^2,
    capped at a mechanical blade-speed limit. Returns (U, achieved_psi)."""
    if specific_work <= 0.0:
        raise ValueError("specific_work must be > 0")
    if stage_loading <= 0.0:
        raise ValueError("stage_loading must be > 0")
    if blade_speed_limit <= 0.0:
        raise ValueError("blade_speed_limit must be > 0")

    U_from_psi = math.sqrt(specific_work / stage_loading)
    U = min(U_from_psi, blade_speed_limit)
    achieved_psi = specific_work / U**2
    return U, achieved_psi


def mean_radius(U: float, rotational_speed_rpm: float) -> float:
    omega = 2.0 * math.pi * rotational_speed_rpm / 60.0
    return U / omega


@dataclass(frozen=True)
class AnnulusGeometry:
    area: float  # m^2
    mean_diameter: float  # m
    blade_height: float  # m
    hub_radius: float  # m
    tip_radius: float  # m

    @property
    def hub_to_tip_ratio(self) -> float:
        return self.hub_radius / self.tip_radius


def annulus_from_mass_flow(mass_flow: float, density: float, axial_velocity: float, mean_diameter: float) -> AnnulusGeometry:
    if mass_flow <= 0.0 or density <= 0.0 or axial_velocity <= 0.0 or mean_diameter <= 0.0:
        raise ValueError("mass_flow, density, axial_velocity, and mean_diameter must all be > 0")

    area = mass_flow / (density * axial_velocity)
    height = area / (math.pi * mean_diameter)
    r_mean = mean_diameter / 2.0
    if height >= mean_diameter:
        raise ValueError("blade height exceeds mean diameter -- infeasible annulus, revisit design inputs")

    return AnnulusGeometry(
        area=area,
        mean_diameter=mean_diameter,
        blade_height=height,
        hub_radius=r_mean - 0.5 * height,
        tip_radius=r_mean + 0.5 * height,
    )


def an2(area: float, rotational_speed_rpm: float) -> float:
    """AN^2 mechanical design parameter [m^2 * rpm^2], a standard proxy for
    rotor disk stress -- typically kept below ~3-4e7 for uncooled disks."""
    return area * rotational_speed_rpm**2
