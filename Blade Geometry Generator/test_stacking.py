import numpy as np
import pytest

from blade_sections import generate_spanwise_sections
from lean import make_linear_lean_offset
from stacking import stack_sections
from sweep import make_linear_sweep_offset

VT2, VT3, U_MEAN, VX = 509.7264849326719, 118.51391491378473, 349.0224443594805, 174.51122217974026
BETA2, BETA3 = 42.64137733692157, 69.5315587853695
MEAN_R = 1.1109729453965465
HUB_R, TIP_R = 1.0149115920014755, 1.2070342987916174
AXIAL_CHORD = 0.1493073856861406


@pytest.fixture(scope="module")
def spanwise_sections():
    return generate_spanwise_sections(
        VT2, VT3, U_MEAN, VX, BETA2, BETA3, MEAN_R, HUB_R, TIP_R,
        stagger_angle_deg=39.0, axial_chord=AXIAL_CHORD,
        le_radius_over_cx=0.05, te_radius_over_cx=0.0047,
        n_sections=3,
    )


def test_stack_sections_returns_one_stacked_section_per_input(spanwise_sections):
    geometry = stack_sections(spanwise_sections)
    assert len(geometry.sections) == 3


def test_stacked_section_z_matches_its_radius(spanwise_sections):
    geometry = stack_sections(spanwise_sections)
    for stacked, source in zip(geometry.sections, spanwise_sections):
        assert stacked.radius == pytest.approx(source.radius)
        for x, y, z in (stacked.upper, stacked.lower, stacked.camberline):
            assert np.allclose(z, source.radius)


def test_stack_sections_with_no_offsets_leaves_x_y_unchanged_in_span(spanwise_sections):
    geometry_plain = stack_sections(spanwise_sections)
    geometry_offset = stack_sections(
        spanwise_sections,
        lean_offset=make_linear_lean_offset(span_height=1.0, lean_angle_deg=20.0),
    )
    hub_plain = geometry_plain.sections[0]
    hub_offset = geometry_offset.sections[0]
    # At the hub (span_fraction=0), lean/sweep offsets are zero -- both should match.
    assert np.allclose(hub_plain.upper[1], hub_offset.upper[1])


def test_lean_offset_shifts_tip_section_tangentially(spanwise_sections):
    geometry_plain = stack_sections(spanwise_sections)
    geometry_leaned = stack_sections(
        spanwise_sections,
        lean_offset=make_linear_lean_offset(span_height=TIP_R - HUB_R, lean_angle_deg=20.0),
    )
    tip_plain_y = geometry_plain.sections[-1].upper[1]
    tip_leaned_y = geometry_leaned.sections[-1].upper[1]
    assert not np.allclose(tip_plain_y, tip_leaned_y)


def test_sweep_offset_shifts_tip_section_axially(spanwise_sections):
    geometry_plain = stack_sections(spanwise_sections)
    geometry_swept = stack_sections(
        spanwise_sections,
        sweep_offset=make_linear_sweep_offset(span_height=TIP_R - HUB_R, sweep_angle_deg=15.0),
    )
    tip_plain_x = geometry_plain.sections[-1].upper[0]
    tip_swept_x = geometry_swept.sections[-1].upper[0]
    assert not np.allclose(tip_plain_x, tip_swept_x)
