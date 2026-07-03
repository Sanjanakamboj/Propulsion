"""Quantitative geometric mesh-quality metrics via pyvista's cell_quality
filter (Verdict library measures) -- distinct from SU2.py's
plot_mesh_quality, which is a VISUAL sanity check (does the domain/blade
look right) rather than a numeric quality metric.

scaled_jacobian close to 1 = well-shaped cell, near/below 0 = degenerate;
aspect_ratio close to 1 = equilateral-like, higher = stretched. The
default limits below are common CFD meshing guidelines, not hard physical
constants -- override per your own solver's tolerance.
"""

from dataclasses import dataclass

DEFAULT_QUALITY_LIMITS = dict(
    aspect_ratio=(None, 5.0),
    scaled_jacobian=(0.2, None),
)


@dataclass(frozen=True)
class MeshQualityReport:
    measure: str
    min: float
    max: float
    mean: float
    n_cells: int
    n_cells_poor: int

    @property
    def fraction_poor(self) -> float:
        return self.n_cells_poor / self.n_cells if self.n_cells else 0.0


def compute_quality_report(mesh_path: str, measure: str = "scaled_jacobian", limits: dict = None) -> MeshQualityReport:
    import pyvista as pv

    lim = dict(DEFAULT_QUALITY_LIMITS)
    if limits:
        lim.update(limits)
    low, high = lim.get(measure, (None, None))

    mesh = pv.read(mesh_path).triangulate()
    result = mesh.cell_quality(quality_measure=measure)
    values = result.cell_data[measure]

    if low is not None:
        n_poor = int((values < low).sum())
    elif high is not None:
        n_poor = int((values > high).sum())
    else:
        n_poor = 0

    return MeshQualityReport(measure=measure, min=float(values.min()), max=float(values.max()), mean=float(values.mean()), n_cells=len(values), n_cells_poor=n_poor)


def plot_quality_map(mesh_path: str, measure: str = "scaled_jacobian", ax=None, cmap: str = "RdYlGn"):
    """Colors each cell by its quality value -- low-quality (skewed/
    degenerate) regions stand out immediately."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import pyvista as pv

    mesh = pv.read(mesh_path).triangulate()
    result = mesh.cell_quality(quality_measure=measure)
    values = result.cell_data[measure]

    cells = mesh.cells.reshape(-1, 4)
    tris = cells[:, 1:]
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    triang = mtri.Triangulation(x, y, tris)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))

    tpc = ax.tripcolor(triang, facecolors=values, cmap=cmap)
    ax.figure.colorbar(tpc, ax=ax, label=measure)
    ax.set_aspect("equal")
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [m]")
    ax.set_title(f"Mesh Quality -- {measure}", fontsize=13, fontweight="bold")
    return ax, tpc
