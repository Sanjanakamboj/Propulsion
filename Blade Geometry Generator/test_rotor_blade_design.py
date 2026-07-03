import pytest

from rotor_blade_design import BladeSizingInputs, size_rotor_blade

# Regression values from the validated Turbine Stage Design.ipynb worked
# example (750 kg/s, N=3000 rpm HP turbine stage): height_2=0.1921 m,
# mean_diameter=2.2219 m, beta2=42.6413 deg, beta3=69.5316 deg,
# AR=1.0, Z=1.0, stagger=39 deg -> Cx=0.1493 m, pitch=0.1696 m, N_blades=42.


@pytest.fixture
def notebook_design():
    return BladeSizingInputs(aspect_ratio=1.0, zweifel_coefficient=1.0, stagger_angle_deg=39.0)


def test_matches_validated_notebook_example(notebook_design):
    result = size_rotor_blade(
        blade_height_in=0.19212270679014198,
        blade_height_out=0.23464801304027977,
        mean_diameter=2.221945890793093,
        beta_in_deg=42.6413,
        beta_out_deg=69.5316,
        design=notebook_design,
    )
    assert result.chord == pytest.approx(0.1921, rel=1e-3)
    assert result.axial_chord == pytest.approx(0.1493, rel=1e-3)
    assert result.pitch == pytest.approx(0.1696, rel=1e-3)
    assert result.pitch_to_axial_chord == pytest.approx(1.1358, rel=1e-3)
    assert result.num_blades == 42


def test_higher_aspect_ratio_gives_shorter_chord(notebook_design):
    import dataclasses

    base = size_rotor_blade(0.192, 0.235, 2.222, 42.64, 69.53, design=notebook_design)
    tall = size_rotor_blade(0.192, 0.235, 2.222, 42.64, 69.53, design=dataclasses.replace(notebook_design, aspect_ratio=2.0))
    assert tall.chord < base.chord


def test_lean_angle_zero_when_span_constant(notebook_design):
    result = size_rotor_blade(
        blade_height_in=0.2, blade_height_out=0.2, mean_diameter=2.0,
        beta_in_deg=42.0, beta_out_deg=65.0, design=notebook_design,
    )
    assert result.lean_angle_deg == pytest.approx(0.0, abs=1e-9)


def test_num_blades_scales_with_mean_diameter(notebook_design):
    small = size_rotor_blade(0.192, 0.235, 1.0, 42.64, 69.53, design=notebook_design)
    large = size_rotor_blade(0.192, 0.235, 4.0, 42.64, 69.53, design=notebook_design)
    assert large.num_blades > small.num_blades


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(aspect_ratio=0.0),
        dict(zweifel_coefficient=-1.0),
        dict(stagger_angle_deg=90.0),
        dict(thickness_to_chord=1.0),
        dict(te_radius=0.0),
        dict(le_radius_fraction_of_pitch=1.0),
    ],
)
def test_invalid_design_inputs_raise(kwargs):
    base = dict(aspect_ratio=1.0, zweifel_coefficient=1.0, stagger_angle_deg=39.0)
    base.update(kwargs)
    with pytest.raises(ValueError):
        BladeSizingInputs(**base)
