import math

import matplotlib

matplotlib.use("Agg")

import pytest

from blade_section import (
    BladeSectionInputs,
    build_blade_section,
    camberline_coordinates,
    lower_surface_coordinates,
    upper_surface_coordinates,
)

# Same validated example as test_rotor_blade_design.py.
AXIAL_CHORD = 0.1493073856861406
PITCH = 0.16958165896178573


@pytest.fixture
def design():
    return BladeSectionInputs(
        stagger_angle_deg=39.0,
        beta_in_deg=42.6413,
        beta_out_deg=69.5316,
        le_radius_over_cx=0.05 * PITCH / AXIAL_CHORD,
        te_radius_over_cx=0.0007 / AXIAL_CHORD,
    )


def test_build_blade_section_runs_without_error(design):
    blade = build_blade_section(design)
    assert blade is not None


def test_surface_coordinates_span_the_axial_chord(design):
    blade = build_blade_section(design)
    xu, yu = upper_surface_coordinates(blade, AXIAL_CHORD)
    xl, yl = lower_surface_coordinates(blade, AXIAL_CHORD)
    # Leading/trailing edge should land close to x=0 and x=Cx (dimensional);
    # a small overshoot past 0 is expected since the LE radius rounds over.
    assert min(xu.min(), xl.min()) == pytest.approx(0.0, abs=0.03 * AXIAL_CHORD)
    assert max(xu.max(), xl.max()) == pytest.approx(AXIAL_CHORD, rel=0.02)


def test_camberline_lies_between_surfaces_at_midchord(design):
    blade = build_blade_section(design)
    xu, yu = upper_surface_coordinates(blade, AXIAL_CHORD, n=200)
    xl, yl = lower_surface_coordinates(blade, AXIAL_CHORD, n=200)
    xc, yc = camberline_coordinates(blade, AXIAL_CHORD, n=200)
    # At the sample nearest mid-chord, the camberline y should sit between
    # the two surfaces' y values (order may be either way depending on
    # ParaBlade's internal parametrization direction).
    i_u = min(range(len(xu)), key=lambda i: abs(xu[i] - AXIAL_CHORD / 2))
    i_l = min(range(len(xl)), key=lambda i: abs(xl[i] - AXIAL_CHORD / 2))
    i_c = min(range(len(xc)), key=lambda i: abs(xc[i] - AXIAL_CHORD / 2))
    lo, hi = sorted([yu[i_u], yl[i_l]])
    assert lo - 1e-3 <= yc[i_c] <= hi + 1e-3


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(stagger_angle_deg=90.0),
        dict(le_radius_over_cx=0.0),
        dict(te_radius_over_cx=-1.0),
        dict(thickness_upper=(0.1,) * 5),
    ],
)
def test_invalid_design_inputs_raise(kwargs):
    base = dict(stagger_angle_deg=39.0, beta_in_deg=42.64, beta_out_deg=69.53, le_radius_over_cx=0.03, te_radius_over_cx=0.005)
    base.update(kwargs)
    with pytest.raises(ValueError):
        BladeSectionInputs(**base)
