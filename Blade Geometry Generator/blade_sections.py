"""Generate 2D blade sections at multiple span stations (hub, mean, tip, or
any number of intermediate stations), applying twist.py's free-vortex flow-
angle variation at each -- the multi-radius counterpart to
blade_section.py's single mean-line 2D section (airfoil_generator.py).

Axial chord is kept constant across span (an untapered blade) -- the same
simplification rotor_blade_design.py already makes by sizing one chord for
the whole blade, not a per-station value.
"""

from dataclasses import dataclass

from blade_section import BladeSectionInputs, build_blade_section
from twist import twisted_section_at_radius


@dataclass(frozen=True)
class SpanwiseSection:
    radius: float
    span_fraction: float
    beta_in_deg: float
    beta_out_deg: float
    axial_chord: float
    blade: object  # Blade2DCamberThickness, from build_blade_section


def generate_spanwise_sections(
    Vt_in_mean: float,
    Vt_out_mean: float,
    U_mean: float,
    Vx: float,
    beta_in_mean_deg: float,
    beta_out_mean_deg: float,
    mean_radius: float,
    hub_radius: float,
    tip_radius: float,
    stagger_angle_deg: float,
    axial_chord: float,
    le_radius_over_cx: float,
    te_radius_over_cx: float,
    n_sections: int = 3,
) -> list:
    if n_sections < 2:
        raise ValueError("n_sections must be >= 2")

    sections = []
    for i in range(n_sections):
        span_fraction = i / (n_sections - 1)
        r = hub_radius + span_fraction * (tip_radius - hub_radius)
        twisted = twisted_section_at_radius(
            Vt_in_mean, Vt_out_mean, U_mean, Vx,
            beta_in_mean_deg, beta_out_mean_deg,
            mean_radius, hub_radius, tip_radius, r,
        )
        section_inputs = BladeSectionInputs(
            stagger_angle_deg=stagger_angle_deg,
            beta_in_deg=twisted.beta_in_deg,
            beta_out_deg=twisted.beta_out_deg,
            le_radius_over_cx=le_radius_over_cx,
            te_radius_over_cx=te_radius_over_cx,
        )
        blade = build_blade_section(section_inputs)
        sections.append(
            SpanwiseSection(
                radius=r,
                span_fraction=span_fraction,
                beta_in_deg=twisted.beta_in_deg,
                beta_out_deg=twisted.beta_out_deg,
                axial_chord=axial_chord,
                blade=blade,
            )
        )
    return sections
