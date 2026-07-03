import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Blade Geometry Generator"))

from blade_section import BladeSectionInputs, build_blade_section  # noqa: E402

from mesh import build_cascade_domain, generate_cascade_mesh, plot_cascade_domain

# Validated Turbine Stage Design.ipynb example.
AXIAL_CHORD = 0.1493073856861406
PITCH = 0.16958165896178573
BETA_IN, BETA_OUT = 42.6413, 69.5316


@pytest.fixture
def blade():
    design = BladeSectionInputs(
        stagger_angle_deg=39.0, beta_in_deg=BETA_IN, beta_out_deg=BETA_OUT,
        le_radius_over_cx=0.05 * PITCH / AXIAL_CHORD, te_radius_over_cx=0.0007 / AXIAL_CHORD,
    )
    return build_blade_section(design)


@pytest.fixture
def domain(blade):
    return build_cascade_domain(blade, AXIAL_CHORD, PITCH, beta_in_deg=BETA_IN, beta_out_deg=BETA_OUT)


def test_inlet_and_outlet_height_equal_two_pitches(domain):
    inlet_height = domain.wall_upper[0, 1] - domain.wall_lower[0, 1]
    outlet_height = domain.wall_upper[-1, 1] - domain.wall_lower[-1, 1]
    assert inlet_height == pytest.approx(2.0 * PITCH, rel=1e-6)
    assert outlet_height == pytest.approx(2.0 * PITCH, rel=1e-6)


def test_wall_upper_interior_is_pressure_surface_shifted_by_pitch(domain):
    # wall_upper's un-extended (interior) portion is literally the pressure
    # surface shifted by +pitch -- check against the actual x/y arrays.
    n = len(domain.xs_pressure)
    interior = domain.wall_upper[1 : 1 + n]
    assert interior[:, 0] == pytest.approx(domain.xs_pressure, rel=1e-9)
    assert interior[:, 1] == pytest.approx(domain.ys_pressure + PITCH, rel=1e-9)


def test_domain_extends_beyond_le_and_te(domain):
    assert domain.x_in < domain.le[0]
    assert domain.x_out > domain.te[0]


def test_plot_cascade_domain_runs_without_error(domain):
    ax = plot_cascade_domain(domain)
    assert ax.get_title().startswith("3-Blade Cascade Domain")


def test_generate_cascade_mesh_matches_notebook_scale(domain, tmp_path):
    su2_path, msh_path, n_nodes, n_elements = generate_cascade_mesh(domain, str(tmp_path))
    assert Path(su2_path).exists()
    assert Path(msh_path).exists()
    # Notebook's validated run: ~17889 nodes, ~34875 cells -- allow generous
    # tolerance since GMSH's frontal-Delaunay algorithm isn't bit-for-bit
    # deterministic across versions, but should be the same order of magnitude.
    assert 5_000 < n_nodes < 60_000
    assert 5_000 < n_elements < 120_000
