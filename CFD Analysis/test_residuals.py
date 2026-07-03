import matplotlib

matplotlib.use("Agg")

import pytest

from residuals import is_converged, plot_residuals, read_residual_history

SU2_HISTORY_CSV = '''"Time_Iter","Outer_Iter","Inner_Iter",    "rms[Rho]"    ,    "rms[RhoU]"   ,    "rms[RhoV]"   ,    "rms[RhoE]"   ,     "rms[nu]"
          0,           0,           0,     -0.6280528581,       1.624028424,        1.62350512,       5.631775931,      -4.783790232
          0,           0,           1,     -0.8127104826,       1.751840108,       1.629702813,       5.442815285,      -4.841613292
          0,           0,           2,     -2.500000000,        -3.500000000,      -4.500000000,      -9.100000000,     -9.500000000
'''


@pytest.fixture
def history_csv_path(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text(SU2_HISTORY_CSV)
    return str(path)


def test_read_residual_history_parses_su2_format(history_csv_path):
    history = read_residual_history(history_csv_path)
    assert history.iterations == [0, 1, 2]
    assert set(history.residuals.keys()) == {"rms[Rho]", "rms[RhoU]", "rms[RhoV]", "rms[RhoE]", "rms[nu]"}
    assert history.residuals["rms[Rho]"] == pytest.approx([-0.6280528581, -0.8127104826, -2.5])


def test_is_converged_false_when_final_residuals_above_threshold(history_csv_path):
    history = read_residual_history(history_csv_path)
    assert not is_converged(history, threshold=-9.0)


def test_is_converged_true_when_all_final_residuals_below_threshold(history_csv_path):
    history = read_residual_history(history_csv_path)
    # Last row: Rho=-2.5, RhoU=-3.5, RhoV=-4.5, RhoE=-9.1, nu=-9.5
    assert is_converged(history, threshold=-2.0)
    assert not is_converged(history, threshold=-5.0)


def test_plot_residuals_runs_without_error(history_csv_path):
    history = read_residual_history(history_csv_path)
    ax = plot_residuals(history)
    assert "Residual" in ax.get_title()


def test_read_residual_history_rejects_file_without_inner_iter_column(tmp_path):
    bad_path = tmp_path / "bad.csv"
    bad_path.write_text('"Foo","Bar"\n1,2\n')
    with pytest.raises(ValueError):
        read_residual_history(str(bad_path))


def test_read_residual_history_rejects_file_without_residual_columns(tmp_path):
    bad_path = tmp_path / "bad.csv"
    bad_path.write_text('"Inner_Iter","Foo"\n0,1\n')
    with pytest.raises(ValueError):
        read_residual_history(str(bad_path))
