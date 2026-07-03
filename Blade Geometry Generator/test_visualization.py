import matplotlib

matplotlib.use("Agg")

import pytest

from blade_sections import generate_spanwise_sections
from stacking import stack_sections
from visualization import plot_stacked_blade

VT2, VT3, U_MEAN, VX = 509.7264849326719, 118.51391491378473, 349.0224443594805, 174.51122217974026
BETA2, BETA3 = 42.64137733692157, 69.5315587853695
MEAN_R = 1.1109729453965465
HUB_R, TIP_R = 1.0149115920014755, 1.2070342987916174
AXIAL_CHORD = 0.1493073856861406


@pytest.fixture(scope="module")
def geometry():
    sections = generate_spanwise_sections(
        VT2, VT3, U_MEAN, VX, BETA2, BETA3, MEAN_R, HUB_R, TIP_R,
        stagger_angle_deg=39.0, axial_chord=AXIAL_CHORD,
        le_radius_over_cx=0.05, te_radius_over_cx=0.0047,
        n_sections=3,
    )
    return stack_sections(sections, n_points_per_surface=50)


def test_plot_stacked_blade_runs_without_error(geometry):
    ax = plot_stacked_blade(geometry)
    assert ax.get_title() == "Stacked Blade Geometry"


def test_plot_stacked_blade_accepts_an_existing_axes(geometry):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    returned_ax = plot_stacked_blade(geometry, ax=ax)
    assert returned_ax is ax
