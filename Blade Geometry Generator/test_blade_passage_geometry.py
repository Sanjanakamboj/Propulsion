import matplotlib

matplotlib.use("Agg")

import pytest

from blade_passage_geometry import compute_passage_geometry, plot_blade_passage, plot_passage_width
from blade_section import BladeSectionInputs, build_blade_section

AXIAL_CHORD = 0.1493073856861406
PITCH = 0.16958165896178573


@pytest.fixture
def blade():
    design = BladeSectionInputs(
        stagger_angle_deg=39.0,
        beta_in_deg=42.6413,
        beta_out_deg=69.5316,
        le_radius_over_cx=0.05 * PITCH / AXIAL_CHORD,
        te_radius_over_cx=0.0007 / AXIAL_CHORD,
    )
    return build_blade_section(design)


def test_matches_validated_notebook_example(blade):
    passage = compute_passage_geometry(blade, AXIAL_CHORD, PITCH)
    assert passage.throat_over_axial_chord == pytest.approx(0.8052, rel=5e-3)
    assert passage.throat_axial_location_over_cx == pytest.approx(0.2836, rel=5e-3)
    assert passage.throat_over_pitch == pytest.approx(0.7090, rel=5e-3)
    assert passage.throat == pytest.approx(0.1202, rel=5e-3)


def test_throat_is_the_minimum_of_the_gap_array(blade):
    passage = compute_passage_geometry(blade, AXIAL_CHORD, PITCH)
    assert passage.throat == pytest.approx(min(passage.gap), rel=1e-9)


def test_gap_arrays_are_consistent_length(blade):
    passage = compute_passage_geometry(blade, AXIAL_CHORD, PITCH, n=500)
    assert len(passage.x) == len(passage.gap) == len(passage.suction_surface_y) == len(passage.pressure_surface_y) == 500


def test_plot_blade_passage_runs_without_error(blade):
    passage = compute_passage_geometry(blade, AXIAL_CHORD, PITCH)
    ax = plot_blade_passage(blade, AXIAL_CHORD, PITCH, passage)
    assert ax.get_title().startswith("Rotor Blade Passage")


def test_plot_passage_width_runs_without_error(blade):
    passage = compute_passage_geometry(blade, AXIAL_CHORD, PITCH)
    ax = plot_passage_width(passage, AXIAL_CHORD)
    assert ax.get_title().startswith("Passage Width")
