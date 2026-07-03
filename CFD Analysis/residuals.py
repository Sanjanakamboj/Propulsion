"""Parse and plot SU2's residual history (history.csv, written per
write_su2_config's CONV_FILENAME=history) -- one RMS residual column per
conserved variable (density, momentum, energy, turbulence variable), so
convergence can be checked visually instead of by scrolling solver output.
"""

import csv
from dataclasses import dataclass


@dataclass(frozen=True)
class ResidualHistory:
    iterations: list  # Inner_Iter values
    residuals: dict  # column name (stripped) -> list of values


def read_residual_history(history_csv_path: str) -> ResidualHistory:
    with open(history_csv_path) as f:
        reader = csv.reader(f)
        header = [h.strip().strip('"') for h in next(reader)]
        rows = [row for row in reader]

    if "Inner_Iter" not in header:
        raise ValueError(f"'Inner_Iter' column not found in {history_csv_path} -- is this an SU2 history file?")

    iter_idx = header.index("Inner_Iter")
    residual_cols = [(i, name) for i, name in enumerate(header) if name.startswith("rms[")]
    if not residual_cols:
        raise ValueError(f"no 'rms[...]' residual columns found in {history_csv_path}")

    iterations = [int(row[iter_idx]) for row in rows]
    residuals = {name: [float(row[i]) for row in rows] for i, name in residual_cols}
    return ResidualHistory(iterations=iterations, residuals=residuals)


def is_converged(history: ResidualHistory, threshold: float = -9.0) -> bool:
    """True if every residual's final value is at or below threshold (SU2's
    RMS residuals are already log10-scaled, matching CONV_RESIDUAL_MINVAL)."""
    if not history.residuals:
        return False
    return all(values[-1] <= threshold for values in history.residuals.values())


def plot_residuals(history: ResidualHistory, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    for name, values in history.residuals.items():
        ax.plot(history.iterations, values, label=name.replace("rms[", "").replace("]", ""))

    ax.set_xlabel("Inner Iteration")
    ax.set_ylabel(r"log$_{10}$(RMS residual)")
    ax.set_title("SU2 Residual Convergence History", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    return ax
