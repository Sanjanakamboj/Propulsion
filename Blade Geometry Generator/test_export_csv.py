import csv

import pytest

from blade_sections import generate_spanwise_sections
from export_csv import write_blade_csv
from stacking import stack_sections

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


def test_write_blade_csv_creates_a_file_with_expected_columns(geometry, tmp_path):
    path = str(tmp_path / "blade.csv")
    write_blade_csv(geometry, path)
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == ["section_index", "span_fraction", "radius", "surface", "point_index", "x", "y", "z"]
    assert len(rows) > 0


def test_write_blade_csv_row_count_matches_geometry(geometry, tmp_path):
    path = str(tmp_path / "blade.csv")
    write_blade_csv(geometry, path)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    expected = sum(len(s.upper[0]) + len(s.lower[0]) + len(s.camberline[0]) for s in geometry.sections)
    assert len(rows) == expected


def test_write_blade_csv_covers_all_sections_and_surfaces(geometry, tmp_path):
    path = str(tmp_path / "blade.csv")
    write_blade_csv(geometry, path)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    section_indices = {int(r["section_index"]) for r in rows}
    surfaces = {r["surface"] for r in rows}
    assert section_indices == {0, 1, 2}
    assert surfaces == {"upper", "lower", "camberline"}
