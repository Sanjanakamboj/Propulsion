import matplotlib

matplotlib.use("Agg")

import pytest

from combustion import assess_combustion
from plots import combustor_diagrams, combustor_parameter_sections, plot_equivalence_ratio_gauge

T_IN, T_EXIT = 663.7319017566193, 1700.0
P_IN, P_OUT = 827846.8017565166, 794732.9296862559
FAR = 0.02794540232049333


def test_combustor_parameter_sections_cover_expected_headers():
    state = assess_combustion(FAR)
    sections = combustor_parameter_sections(T_IN, T_EXIT, P_IN, P_OUT, FAR, state, tau=0.003)
    headers = [header for header, _ in sections]
    assert any("INLET" in h for h in headers)
    assert any("COMBUSTION" in h for h in headers)
    assert any("RESIDENCE" in h for h in headers)


def test_combustor_parameter_sections_omits_residence_time_section_when_not_given():
    state = assess_combustion(FAR)
    sections = combustor_parameter_sections(T_IN, T_EXIT, P_IN, P_OUT, FAR, state)
    headers = [header for header, _ in sections]
    assert not any("RESIDENCE" in h for h in headers)


def test_plot_equivalence_ratio_gauge_runs_without_error():
    state = assess_combustion(FAR)
    ax = plot_equivalence_ratio_gauge(state.equivalence_ratio)
    assert ax.get_xlabel().startswith("Equivalence")


def test_combustor_diagrams_saves_two_files(tmp_path):
    state = assess_combustion(FAR)
    prefix = str(tmp_path / "combustor")
    fig_table, fig_gauge = combustor_diagrams(T_IN, T_EXIT, P_IN, P_OUT, FAR, state, tau=0.003, save_prefix=prefix)
    assert (tmp_path / "combustor_table.png").exists()
    assert (tmp_path / "combustor_equivalence_ratio.png").exists()
