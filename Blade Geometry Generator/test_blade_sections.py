import pytest

from blade_section import upper_surface_coordinates
from blade_sections import generate_spanwise_sections

# Validated turbine stage mean-line + annulus_2 geometry.
VT2, VT3, U_MEAN, VX = 509.7264849326719, 118.51391491378473, 349.0224443594805, 174.51122217974026
BETA2, BETA3 = 42.64137733692157, 69.5315587853695
MEAN_R = 1.1109729453965465
HUB_R, TIP_R = 1.0149115920014755, 1.2070342987916174
AXIAL_CHORD = 0.1493073856861406


def _generate(n_sections=3):
    return generate_spanwise_sections(
        VT2, VT3, U_MEAN, VX, BETA2, BETA3, MEAN_R, HUB_R, TIP_R,
        stagger_angle_deg=39.0, axial_chord=AXIAL_CHORD,
        le_radius_over_cx=0.05, te_radius_over_cx=0.0047,
        n_sections=n_sections,
    )


def test_generate_spanwise_sections_returns_requested_count():
    sections = _generate(n_sections=3)
    assert len(sections) == 3


def test_first_and_last_sections_are_hub_and_tip():
    sections = _generate(n_sections=3)
    assert sections[0].radius == pytest.approx(HUB_R)
    assert sections[0].span_fraction == pytest.approx(0.0)
    assert sections[-1].radius == pytest.approx(TIP_R)
    assert sections[-1].span_fraction == pytest.approx(1.0)


def test_middle_section_is_at_mean_radius_and_reproduces_mean_line_angles():
    sections = _generate(n_sections=3)
    mean_section = sections[1]
    assert mean_section.radius == pytest.approx(MEAN_R, rel=1e-6)
    assert mean_section.beta_in_deg == pytest.approx(BETA2, rel=1e-9)
    assert mean_section.beta_out_deg == pytest.approx(BETA3, rel=1e-9)


def test_sections_show_genuine_twist_across_span():
    sections = _generate(n_sections=3)
    betas_in = [s.beta_in_deg for s in sections]
    assert betas_in[0] != pytest.approx(betas_in[1])
    assert betas_in[1] != pytest.approx(betas_in[2])


def test_axial_chord_is_constant_across_span_untapered():
    sections = _generate(n_sections=3)
    assert all(s.axial_chord == pytest.approx(AXIAL_CHORD) for s in sections)


def test_each_section_produces_a_usable_2d_blade_shape():
    sections = _generate(n_sections=3)
    for section in sections:
        x, y = upper_surface_coordinates(section.blade, section.axial_chord)
        assert len(x) > 0
        assert x.max() - x.min() == pytest.approx(section.axial_chord, rel=0.05)


def test_generate_spanwise_sections_rejects_too_few_sections():
    with pytest.raises(ValueError):
        _generate(n_sections=1)
