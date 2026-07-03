# Engineering Methods Reference

The analytical methods implemented in this toolkit, by folder, with the textbook/source convention each is based on. Intended as a quick reference for what a given module actually computes and which assumption it rests on — see each folder's module docstrings for the full derivation.

## Brayton Cycle Analysis
- Single-spool turbojet 0D cycle (Inlet→Compressor→Combustor→Turbine→Nozzle), turbine sized to balance compressor shaft power rather than taking pressure ratio as a free input.
- International Standard Atmosphere (troposphere lapse rate + isothermal lower stratosphere), plus ISA+ΔT hot/cold-day offset.
- Mass-flow-invariant specific thrust used to size `mdot_air` directly from a required thrust (no iteration needed).
- Classic closed-cycle Brayton variants (ideal, real/lossy, regenerative, intercooled, reheated) as a separate, simpler layer from the open-cycle turbojet model.
- Effective/equivalent jet velocity correction for underexpanded (choked, convergent-only) nozzles, so thermal/propulsive efficiency formulas aren't understated.

## Compressor Calculations / Turbine Calculations
- Axial mean-line design: stage loading (ψ), flow coefficient (φ), degree of reaction (Λ), Euler work equation, repeating-stage assumption.
- Turbine stage solved via iterative root-finding (`scipy.optimize.brentq`) on exit static pressure to hit a required specific work; compressor solved algebraically (no iteration needed) via closed-form flow-angle relations.
- de Haller number and Lieblein diffusion factor (compressor stall-margin criteria); Zweifel's tangential-loading criterion (turbine blade pitch sizing — confirmed not to generalize to compressor cascades).
- AN² mechanical-stress proxy, hub-to-tip ratio, configurable PASS/FAIL sanity-check framework.
- Loss-coefficient ↔ isentropic-efficiency conversions (exact algebraic relations, not empirical curve fits); turbine total-to-total/total-to-static stage efficiency definitions.
- Blade cooling requirement via the standard cooling-effectiveness definition, η_c = (T_gas − T_blade)/(T_gas − T_coolant).

## Blade Geometry Generator
- ParaBlade `Blade2DCamberThickness` parametrization (camberline + thickness distribution) for 2D airfoil sections.
- Free-vortex radial equilibrium (r·Vt = const, U = Ωr) for blade twist — the standard method for extending a single mean-line design across span.
- 3D blade stacking with linear lean/sweep offsets; hand-rolled STL triangulation (no CAD-kernel dependency).

## CFD Analysis
- GMSH 2D cascade meshing (distance-based mesh size fields) with a 3-blade domain (measured blade + two neighbour walls standing in for periodicity).
- SU2 RANS (Spalart-Allmaras) automation for the rotor-relative frame cascade solve.
- OpenFOAM (`rhoSimpleFoam`, steady compressible RANS, kOmegaSST) automation via Docker — including GMSH-mesh extrusion into OpenFOAM's one-cell-thick-3D-slab convention for a nominally-2D case.
- Mesh quality via the Verdict library's cell-quality measures (aspect ratio, scaled Jacobian); pyvista-based streamline/pressure-contour post-processing; CFD-vs-mean-line design-point validation.

## Combustor Analysis
- Adiabatic energy-balance fuel-air ratio solve (both directions: FAR from target exit temperature, and exit temperature from a chosen FAR).
- Equivalence ratio and lean/stoichiometric/rich classification.
- Fixed-fraction and dynamic-pressure-loss-coefficient combustor pressure-loss models.

## Nozzle Analysis
- Converging-nozzle exit state (choked/unchoked, underexpanded when choked) and converging-diverging nozzle exit state (design-matched, fully expanded), via the isentropic total-to-static Mach relation solved directly (no iteration needed).
- Standard isentropic area-Mach relation for C-D nozzle area ratio.
- Momentum + pressure thrust breakdown, specific thrust, TSFC.

## Performance Analysis
- Off-design performance sweep via the fixed-non-dimensional-operating-point approximation: hold compressor pressure ratio, TIT, and all component efficiencies at their design values, and scale mass flow via the corrected-flow parameter (ṁ√T₀₂/P₀₂ = const) as flight condition changes.

## Testing philosophy
Wherever practical, tests are grounded in real, independently-checkable numbers rather than only mocked/synthetic data: hand-derived closed-form results, values pulled from a validated reference notebook, or the actual output of a real solver run (SU2, OpenFOAM, GMSH). Where a genuinely new physical relationship is implemented, it's verified numerically against a hand calculation before being trusted in test assertions — this caught at least one wrong assumption during development (that higher nozzle efficiency always raises exit velocity — false for a choked convergent nozzle, where it's the opposite).
