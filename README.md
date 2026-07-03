# Gas Turbine Design Toolkit

A preliminary single-spool turbojet design toolkit, built module by module to mirror the real industry design chronology: mission requirements → 0D cycle design → engine sizing → component mean-line design → blade geometry → CFD verification → off-design performance.

Given a mission (cruise Mach, altitude, required thrust) and a handful of design-point choices (pressure ratio, turbine inlet temperature, stage loadings, blade geometry parameters), the toolkit sizes a complete engine, designs its compressor and turbine stages, generates 3D blade geometry, runs cascade CFD, and reports off-design performance across the flight envelope — all from one editable input file.

## Quickstart

```bash
git clone --recurse-submodules https://github.com/Sanjanakamboj/Propulsion.git
cd Propulsion
pip install -r requirements.txt
python3 design_engine.py
```

`Blade Geometry Generator/parablade-master` is a git submodule pointing to [NAnand-TUD/parablade](https://github.com/NAnand-TUD/parablade) rather than vendored source — if you already cloned without `--recurse-submodules`, fetch it with:

```bash
git submodule update --init
```

Edit [`engine_inputs.txt`](engine_inputs.txt) (plain INI format, comments explain every field) and re-run `design_engine.py` to design a new engine. Each run writes to `Results/<run_id>/` (cycle diagrams, per-stage tables/diagrams, blade geometry plots, sanity-check report) and appends one row to `design_log.xlsx` (never overwritten).

## Project layout

Each folder is a self-contained design stage; `design_engine.py` at the root wires them together into one pipeline.

| Folder | Design stage |
|---|---|
| [`Brayton Cycle Analysis`](Brayton%20Cycle%20Analysis) | Mission/ISA atmosphere, 0D single-spool turbojet cycle (Inlet→Compressor→Combustor→Turbine→Nozzle), engine sizing, T-s/P-v diagrams. Also includes standalone classic Brayton power-cycle variants (ideal, real, regenerative, intercooled, reheated). |
| [`Compressor Calculations`](Compressor%20Calculations) | Axial compressor mean-line design: velocity triangles, stage loading, de Haller number, diffusion factor, loss models, surge/choke margins. |
| [`Turbine Calculations`](Turbine%20Calculations) | Axial turbine mean-line design: velocity triangles solved to match required specific work, loss coefficients, stage efficiency, choking, blade cooling requirement. |
| [`Utils`](Utils) | Shared mean-line geometry, plotting, and sanity-check machinery used by both Compressor and Turbine Calculations. |
| [`Blade Geometry Generator`](Blade%20Geometry%20Generator) | 2D airfoil sections (ParaBlade-based) sized via Zweifel's criterion, extended to full 3D blade geometry: free-vortex radial twist, multi-span-station sections, 3D stacking with lean/sweep, CSV/STL export, 3D visualization. |
| [`CFD Analysis`](CFD%20Analysis) | 3-blade cascade CFD of the turbine rotor blade: GMSH meshing, SU2 (RANS) automation, and an OpenFOAM (`rhoSimpleFoam`) automation path via Docker. Mesh quality metrics, residual/streamline/pressure-contour post-processing, CFD-vs-mean-line validation. |
| [`Combustor Analysis`](Combustor%20Analysis) | Fuel-air ratio, equivalence ratio, pressure loss, residence time. |
| [`Nozzle Analysis`](Nozzle%20Analysis) | Converging and converging-diverging nozzle exit conditions, choking, thrust breakdown (momentum + pressure thrust). |
| [`Performance Analysis`](Performance%20Analysis) | Off-design performance sweep across the flight envelope (thrust lapse, TSFC, efficiencies vs. Mach/altitude) for a fixed engine design. |
| `Materials`, `Data`, `Documentation`, `Validation` | Scaffolded for future use; currently out of scope (see below). |

Root files: `design_engine.py` (master orchestrator), `engine_inputs.txt` (user-editable design point), `config.py` (shared gas properties), `requirements.txt`.

## Scope

This is a **preliminary design** toolkit — 0D/1D mean-line methods plus verification-level CFD, not a detailed/manufacturing-ready design tool. Deliberately out of scope, and why:

- **Mechanical/structural design** (rotor stress, disk sizing, FEA) and **materials selection** — not attempted; `Materials/` is scaffolded but empty.
- **STEP/CAD export** — no CAD kernel (e.g. OpenCASCADE) is bundled or assumed available; blade geometry exports to CSV and STL instead.
- **Emissions prediction** — NOx/CO estimation needs empirical kinetics correlations that aren't included, to avoid presenting unverified numbers as if they were validated.
- **Compressor/turbine performance maps** and full off-design component matching — the off-design model here uses the standard fixed-non-dimensional-operating-point approximation (exact given that assumption), not real map data.

Compressor blade 2D section generation is currently skipped in the default pipeline: Zweifel's criterion (used for blade pitch sizing) is an accelerating-cascade (turbine) criterion and produces a self-intersecting section for the compressor's much smaller flow turning. `Compressor Calculations/diffusion_factor.py` implements the correct compressor-specific criterion (Lieblein diffusion factor) but isn't yet wired into the blade generation step.

## Testing

Each folder's test suite is run independently:

```bash
cd "Brayton Cycle Analysis" && python3 -m pytest -q
```

503 tests pass across the toolkit as of this writing. Tests are grounded in either exact analytical relations (verified by hand-derivation) or real solver/tool output (e.g. an actual SU2/OpenFOAM CFD run, an actual GMSH mesh) rather than only mocked data, wherever that's practical.

## Optional external tools

- **SU2** (`CFD Analysis/SU2.py`) — set `su2_binary` in `engine_inputs.txt`'s `[cfd]` section to your `SU2_CFD` binary path, and `run_cfd = true` to enable a full cascade CFD run (a few minutes).
- **OpenFOAM** (`CFD Analysis/openfoam_runner.py`) — drives an OpenFOAM installation running in a Docker container (tested against `opencfd/openfoam-default`) via `docker exec`; requires Docker running and that image available.

Neither is required for the default `design_engine.py` run — CFD is opt-in.
