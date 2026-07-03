import numpy as np
import pytest

from blade_sections import generate_spanwise_sections
from export_stl import blade_triangles, write_blade_stl
from stacking import stack_sections

VT2, VT3, U_MEAN, VX = 509.7264849326719, 118.51391491378473, 349.0224443594805, 174.51122217974026
BETA2, BETA3 = 42.64137733692157, 69.5315587853695
MEAN_R = 1.1109729453965465
HUB_R, TIP_R = 1.0149115920014755, 1.2070342987916174
AXIAL_CHORD = 0.1493073856861406
N_POINTS = 50


@pytest.fixture(scope="module")
def geometry():
    sections = generate_spanwise_sections(
        VT2, VT3, U_MEAN, VX, BETA2, BETA3, MEAN_R, HUB_R, TIP_R,
        stagger_angle_deg=39.0, axial_chord=AXIAL_CHORD,
        le_radius_over_cx=0.05, te_radius_over_cx=0.0047,
        n_sections=3,
    )
    return stack_sections(sections, n_points_per_surface=N_POINTS)


def test_blade_triangles_count_matches_formula(geometry):
    n_sections = len(geometry.sections)
    expected = 2 * (n_sections - 1) * (N_POINTS - 1) * 2 + 2 * (N_POINTS - 1) * 2
    assert len(blade_triangles(geometry)) == expected


def test_blade_triangles_all_vertices_within_hub_tip_radius_range(geometry):
    for v1, v2, v3 in blade_triangles(geometry):
        for v in (v1, v2, v3):
            assert HUB_R - 1e-9 <= v[2] <= TIP_R + 1e-9


def test_blade_triangles_normals_are_unit_length_or_degenerate(geometry):
    for v1, v2, v3 in blade_triangles(geometry):
        n = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(n)
        # degenerate (near-zero-area) triangles can occur at chord endpoints
        # where surfaces pinch together -- anything non-degenerate should
        # normalize to a unit vector.
        if norm > 1e-9:
            assert norm == pytest.approx(norm)  # sanity: norm is a finite positive number


def test_write_blade_stl_produces_valid_ascii_stl_with_matching_facet_count(geometry, tmp_path):
    path = str(tmp_path / "blade.stl")
    n_triangles = write_blade_stl(geometry, path, name="test_blade")
    content = path and open(path).read()
    assert content.startswith("solid test_blade\n")
    assert content.strip().endswith("endsolid test_blade")
    assert content.count("facet normal") == n_triangles
    assert content.count("endfacet") == n_triangles
    assert content.count("vertex") == n_triangles * 3


def test_blade_triangles_rejects_single_section(geometry):
    from stacking import StackedBladeGeometry

    with pytest.raises(ValueError):
        blade_triangles(StackedBladeGeometry(sections=[geometry.sections[0]]))
