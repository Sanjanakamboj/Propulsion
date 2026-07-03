"""OpenFOAM case generation and execution for the 3-blade cascade CFD case
-- the OpenFOAM analogue to SU2.py, using rhoSimpleFoam (steady
compressible RANS) as the closest OpenFOAM match to SU2's compressible
RANS solve.

OpenFOAM has no native OpenFOAM binary in this development environment --
this module drives an OpenFOAM installation running in a Docker container
(image opencfd/openfoam-default) via `docker exec`, exchanging files
through a bind-mounted host directory. All case-file syntax (thermo-
physicalProperties, turbulenceProperties, fvSchemes, fvSolution, boundary
condition dictionaries) was verified against this exact OpenFOAM version's
own bundled tutorials (compressible/rhoSimpleFoam/aerofoilNACA0012 and
angledDuctExplicitFixedCoeff), not written from memory.

Key difference from SU2's mesh: SU2 accepts a pure 2D triangulated mesh
directly. OpenFOAM is inherently 3D -- a "2D" case must be a one-cell-thick
3D slab, with the two end-cap patches marked boundary type "empty". This
module extrudes mesh.py's 2D cascade domain by one layer in GMSH to
produce that slab, then verifies (via gmsh.model.getBoundary, not assumed
ordering) which extruded lateral surface corresponds to which original
INLET/OUTLET/WALL_MID/WALL_UP/WALL_LO curve before assigning physical
groups -- extrusion does NOT preserve positional ordering for a surface
with an interior hole (confirmed: the blade's suction/pressure curves come
out in swapped order relative to naive positional assumption).
"""

import math
import os
import re
import subprocess
from dataclasses import dataclass

from boundary_conditions import INLET, OUTLET, WALL_LO, WALL_MID, WALL_UP

_FOAM_FILE_HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


@dataclass(frozen=True)
class OpenFOAMCascadeInputs:
    gamma: float
    gas_constant: float
    relative_inlet_mach: float
    static_temperature: float
    static_pressure: float
    inlet_flow_angle_deg: float
    outlet_static_pressure: float
    reference_length: float
    reynolds_number: float = 1.0e6
    n_iterations: int = 1000

    def __post_init__(self):
        if self.gamma <= 1.0:
            raise ValueError("gamma must be > 1")
        if self.gas_constant <= 0.0:
            raise ValueError("gas_constant must be > 0")
        if self.relative_inlet_mach <= 0.0:
            raise ValueError("relative_inlet_mach must be > 0")
        if self.static_temperature <= 0.0:
            raise ValueError("static_temperature must be > 0")
        if self.static_pressure <= 0.0 or self.outlet_static_pressure <= 0.0:
            raise ValueError("pressures must be > 0")
        if self.reference_length <= 0.0:
            raise ValueError("reference_length must be > 0")
        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be > 0")

    @property
    def inlet_velocity(self) -> tuple:
        """(Ux, Uy, 0) at the inlet, from the relative inlet Mach number,
        static temperature, and flow angle -- the same static conditions
        SU2CascadeInputs uses for its TOTAL_CONDITIONS inlet, expressed
        here as a direct fixedValue velocity instead (OpenFOAM's more
        standard internal-flow inlet BC, verified against
        angledDuctExplicitFixedCoeff)."""
        a = math.sqrt(self.gamma * self.gas_constant * self.static_temperature)
        V = self.relative_inlet_mach * a
        angle = math.radians(self.inlet_flow_angle_deg)
        return (V * math.cos(angle), V * math.sin(angle), 0.0)


def _molecular_weight(gas_constant: float) -> float:
    """OpenFOAM's specie dict wants molWeight (g/mol); R_universal =
    8314.5 J/(kmol*K) / molWeight = gas_constant (specific, J/(kg*K))."""
    return 8314.5 / gas_constant


def _dynamic_viscosity(inputs: OpenFOAMCascadeInputs) -> float:
    """mu from the target Reynolds number: Re = rho*V*L/mu, with rho from
    the ideal gas law at static conditions and V from inlet_velocity."""
    rho = inputs.static_pressure / (inputs.gas_constant * inputs.static_temperature)
    V = math.hypot(*inputs.inlet_velocity[:2])
    return rho * V * inputs.reference_length / inputs.reynolds_number


def write_openfoam_case(inputs: OpenFOAMCascadeInputs, case_dir: str) -> None:
    """Writes constant/, system/, and 0/ dictionary files (everything
    except the mesh, which comes from write_and_convert_mesh) for a
    rhoSimpleFoam cascade case."""
    os.makedirs(os.path.join(case_dir, "constant"), exist_ok=True)
    os.makedirs(os.path.join(case_dir, "system"), exist_ok=True)
    os.makedirs(os.path.join(case_dir, "0"), exist_ok=True)

    mol_weight = _molecular_weight(inputs.gas_constant)
    mu = _dynamic_viscosity(inputs)
    Ux, Uy, Uz = inputs.inlet_velocity

    _write(os.path.join(case_dir, "constant", "thermophysicalProperties"), _FOAM_FILE_HEADER.format(cls="dictionary", obj="thermophysicalProperties") + f"""
thermoType
{{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}}

mixture
{{
    specie
    {{
        molWeight   {mol_weight:.6f};
    }}
    thermodynamics
    {{
        Cp          {inputs.gamma * inputs.gas_constant / (inputs.gamma - 1.0):.4f};
        Hf          0;
    }}
    transport
    {{
        mu          {mu:.6e};
        Pr          0.71;
    }}
}}
""")

    _write(os.path.join(case_dir, "constant", "turbulenceProperties"), _FOAM_FILE_HEADER.format(cls="dictionary", obj="turbulenceProperties") + """
simulationType          RAS;

RAS
{
    RASModel            kOmegaSST;
    turbulence          on;
    printCoeffs         on;
}
""")

    _write(os.path.join(case_dir, "system", "controlDict"), _FOAM_FILE_HEADER.format(cls="dictionary", obj="controlDict") + f"""
application     rhoSimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {inputs.n_iterations};
deltaT          1;
writeControl    timeStep;
writeInterval   {max(inputs.n_iterations // 4, 1)};
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
""")

    _write(os.path.join(case_dir, "system", "fvSchemes"), _FOAM_FILE_HEADER.format(cls="dictionary", obj="fvSchemes") + """
ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
    limited         cellLimited Gauss linear 1;
    grad(U)         $limited;
    grad(k)         $limited;
    grad(omega)     $limited;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind limited;
    energy          bounded Gauss linearUpwind limited;
    div(phi,e)      $energy;
    div(phi,K)      $energy;
    div(phi,Ekp)    $energy;
    turbulence      bounded Gauss upwind;
    div(phi,k)      $turbulence;
    div(phi,omega)  $turbulence;
    div(phid,p)     Gauss upwind;
    div((phi|interpolate(rho)),p)  bounded Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U)))))    Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method          meshWave;
}
""")

    _write(os.path.join(case_dir, "system", "fvSolution"), _FOAM_FILE_HEADER.format(cls="dictionary", obj="fvSolution") + """
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-6;
        relTol          0.01;
    }
    "(U|k|omega|e)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0.1;
    }
}

SIMPLE
{
    residualControl
    {
        p               1e-4;
        U               1e-4;
        "(k|omega|e)"   1e-4;
    }
    nNonOrthogonalCorrectors 0;
    pMinFactor      0.1;
    pMaxFactor      2;
}

relaxationFactors
{
    fields
    {
        p               0.7;
        rho             0.01;
    }
    equations
    {
        U               0.3;
        e               0.7;
        "(k|omega)"     0.7;
    }
}
""")

    _write(os.path.join(case_dir, "0", "U"), _FOAM_FILE_HEADER.format(cls="volVectorField", obj="U") + f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({Ux:.6f} {Uy:.6f} {Uz:.6f});

boundaryField
{{
    {INLET}
    {{
        type            fixedValue;
        value           uniform ({Ux:.6f} {Uy:.6f} {Uz:.6f});
    }}
    {OUTLET}
    {{
        type            inletOutlet;
        inletValue      uniform (0 0 0);
        value           uniform ({Ux:.6f} {Uy:.6f} {Uz:.6f});
    }}
    {WALL_MID}
    {{
        type            noSlip;
    }}
    "({WALL_UP}|{WALL_LO})"
    {{
        type            slip;
    }}
    "(FRONT|BACK)"
    {{
        type            empty;
    }}
}}
""")

    _write(os.path.join(case_dir, "0", "p"), _FOAM_FILE_HEADER.format(cls="volScalarField", obj="p") + f"""
dimensions      [1 -1 -2 0 0 0 0];
internalField   uniform {inputs.outlet_static_pressure:.4f};

boundaryField
{{
    {INLET}
    {{
        type            zeroGradient;
    }}
    {OUTLET}
    {{
        type            fixedValue;
        value           uniform {inputs.outlet_static_pressure:.4f};
    }}
    {WALL_MID}
    {{
        type            zeroGradient;
    }}
    "({WALL_UP}|{WALL_LO})"
    {{
        type            zeroGradient;
    }}
    "(FRONT|BACK)"
    {{
        type            empty;
    }}
}}
""")

    _write(os.path.join(case_dir, "0", "T"), _FOAM_FILE_HEADER.format(cls="volScalarField", obj="T") + f"""
dimensions      [0 0 0 1 0 0 0];
internalField   uniform {inputs.static_temperature:.4f};

boundaryField
{{
    {INLET}
    {{
        type            fixedValue;
        value           uniform {inputs.static_temperature:.4f};
    }}
    {OUTLET}
    {{
        type            inletOutlet;
        inletValue      uniform {inputs.static_temperature:.4f};
        value           uniform {inputs.static_temperature:.4f};
    }}
    {WALL_MID}
    {{
        type            zeroGradient;
    }}
    "({WALL_UP}|{WALL_LO})"
    {{
        type            zeroGradient;
    }}
    "(FRONT|BACK)"
    {{
        type            empty;
    }}
}}
""")

    turbulence_intensity = 0.05
    k_inlet = 1.5 * (turbulence_intensity * math.hypot(Ux, Uy)) ** 2
    omega_inlet = math.sqrt(k_inlet) / (0.07 * inputs.reference_length)

    _write(os.path.join(case_dir, "0", "k"), _FOAM_FILE_HEADER.format(cls="volScalarField", obj="k") + f"""
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {k_inlet:.6e};

boundaryField
{{
    {INLET}
    {{
        type            fixedValue;
        value           uniform {k_inlet:.6e};
    }}
    {OUTLET}
    {{
        type            inletOutlet;
        inletValue      uniform {k_inlet:.6e};
        value           uniform {k_inlet:.6e};
    }}
    {WALL_MID}
    {{
        type            kqRWallFunction;
        value           uniform {k_inlet:.6e};
    }}
    "({WALL_UP}|{WALL_LO})"
    {{
        type            slip;
    }}
    "(FRONT|BACK)"
    {{
        type            empty;
    }}
}}
""")

    _write(os.path.join(case_dir, "0", "omega"), _FOAM_FILE_HEADER.format(cls="volScalarField", obj="omega") + f"""
dimensions      [0 0 -1 0 0 0 0];
internalField   uniform {omega_inlet:.6e};

boundaryField
{{
    {INLET}
    {{
        type            fixedValue;
        value           uniform {omega_inlet:.6e};
    }}
    {OUTLET}
    {{
        type            inletOutlet;
        inletValue      uniform {omega_inlet:.6e};
        value           uniform {omega_inlet:.6e};
    }}
    {WALL_MID}
    {{
        type            omegaWallFunction;
        value           uniform {omega_inlet:.6e};
    }}
    "({WALL_UP}|{WALL_LO})"
    {{
        type            slip;
    }}
    "(FRONT|BACK)"
    {{
        type            empty;
    }}
}}
""")

    _write(os.path.join(case_dir, "0", "nut"), _FOAM_FILE_HEADER.format(cls="volScalarField", obj="nut") + f"""
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;

boundaryField
{{
    {INLET}
    {{
        type            calculated;
        value           uniform 0;
    }}
    {OUTLET}
    {{
        type            calculated;
        value           uniform 0;
    }}
    {WALL_MID}
    {{
        type            nutkWallFunction;
        value           uniform 0;
    }}
    "({WALL_UP}|{WALL_LO})"
    {{
        type            slip;
    }}
    "(FRONT|BACK)"
    {{
        type            empty;
    }}
}}
""")

    _write(os.path.join(case_dir, "0", "alphat"), _FOAM_FILE_HEADER.format(cls="volScalarField", obj="alphat") + f"""
dimensions      [1 -1 -1 0 0 0 0];
internalField   uniform 0;

boundaryField
{{
    {INLET}
    {{
        type            calculated;
        value           uniform 0;
    }}
    {OUTLET}
    {{
        type            calculated;
        value           uniform 0;
    }}
    {WALL_MID}
    {{
        type            compressible::alphatWallFunction;
        value           uniform 0;
    }}
    "({WALL_UP}|{WALL_LO})"
    {{
        type            slip;
    }}
    "(FRONT|BACK)"
    {{
        type            empty;
    }}
}}
""")


def _write(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)


def build_extruded_cascade_mesh(domain, output_dir: str, mesh_name: str = "blade_mesh_3d", stride: int = 4, extrude_thickness_factor: float = 0.125):
    """Builds the same 2D cascade domain geometry as mesh.py's
    generate_cascade_mesh, then extrudes it by one layer in Z (OpenFOAM's
    "2D as a one-cell-thick 3D slab" convention -- pure 2D triangle meshes
    aren't accepted by gmshToFoam). The two end-cap patches are named
    FRONT/BACK. Lateral surface -> physical group assignment is verified
    via gmsh.model.getBoundary against each original curve tag, NOT
    assumed from extrusion output ordering (confirmed live: the blade
    hole's two curves come out swapped relative to positional assumption).

    Writes <output_dir>/<mesh_name>.msh. Returns the .msh path."""
    import gmsh
    import numpy as np

    try:
        gmsh.finalize()
    except Exception:
        pass

    gmsh.initialize()
    gmsh.model.add("cascade_3blade_extruded")
    geo = gmsh.model.geo
    lc = domain.axial_chord / 8

    def add_points(arr):
        return [geo.addPoint(float(p[0]), float(p[1]), 0, lc) for p in arr]

    lw_f, uw_f = domain.wall_lower, domain.wall_upper
    le_x, le_y = domain.le
    te_x, te_y = domain.te

    p_li = geo.addPoint(float(lw_f[0, 0]), float(lw_f[0, 1]), 0, lc)
    p_lo = geo.addPoint(float(lw_f[-1, 0]), float(lw_f[-1, 1]), 0, lc)
    p_ui = geo.addPoint(float(uw_f[0, 0]), float(uw_f[0, 1]), 0, lc)
    p_uo = geo.addPoint(float(uw_f[-1, 0]), float(uw_f[-1, 1]), 0, lc)
    p_le = geo.addPoint(le_x, le_y, 0, lc)
    p_te = geo.addPoint(te_x, te_y, 0, lc)

    sp_lw = geo.addBSpline([p_li] + add_points(lw_f[1:-1:stride]) + [p_lo])
    sp_uw = geo.addBSpline([p_ui] + add_points(uw_f[1:-1:stride]) + [p_uo])

    n_surf = len(domain.xs_suction)
    sp_suc = geo.addBSpline(
        [p_te] + add_points(np.column_stack([domain.xs_suction, domain.ys_suction])[stride : n_surf - stride : stride]) + [p_le]
    )
    sp_pre = geo.addBSpline(
        [p_le] + add_points(np.column_stack([domain.xs_pressure, domain.ys_pressure])[stride : n_surf - stride : stride]) + [p_te]
    )

    l_inlet = geo.addLine(p_li, p_ui)
    l_outlet = geo.addLine(p_uo, p_lo)

    outer_cl = geo.addCurveLoop([sp_lw, -l_outlet, -sp_uw, -l_inlet])
    blade_cl = geo.addCurveLoop([-sp_suc, -sp_pre])
    surf = geo.addPlaneSurface([outer_cl, blade_cl])
    geo.synchronize()

    # Mesh the 2D surface FIRST (respecting the size field), THEN extrude
    # the already-meshed surface -- extruding before meshing lets GMSH's
    # background field conflict with a structured sweep, which silently
    # falls back to subdividing each prism into tets (confirmed live: this
    # produced ~47% severely non-orthogonal faces). Meshing first gives a
    # clean 1-layer PRISM sweep instead (confirmed: element type 6 only,
    # one prism per original triangle, no "Subdividing"/"Swapping" in the
    # GMSH log).
    dist_f = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_f, "CurvesList", [sp_suc, sp_pre, sp_uw, sp_lw])
    gmsh.model.mesh.field.setNumber(dist_f, "Sampling", 400)
    thr_f = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thr_f, "InField", dist_f)
    gmsh.model.mesh.field.setNumber(thr_f, "SizeMin", domain.axial_chord / 110)
    gmsh.model.mesh.field.setNumber(thr_f, "SizeMax", domain.axial_chord / 7)
    gmsh.model.mesh.field.setNumber(thr_f, "DistMin", domain.axial_chord * 0.03)
    gmsh.model.mesh.field.setNumber(thr_f, "DistMax", domain.axial_chord * 0.35)
    gmsh.model.mesh.field.setAsBackgroundMesh(thr_f)

    gmsh.model.mesh.generate(2)

    thickness = domain.axial_chord * extrude_thickness_factor
    out = geo.extrude([(2, surf)], 0, 0, thickness, numElements=[1], recombine=True)
    geo.synchronize()

    far_cap_tag = out[0][1]
    lateral_tags = [tag for dim, tag in out[2:]]

    curve_to_group = {sp_lw: WALL_LO, l_outlet: OUTLET, sp_uw: WALL_UP, l_inlet: INLET, sp_suc: WALL_MID, sp_pre: WALL_MID}
    group_to_surfaces = {}
    for curve_tag, group_name in curve_to_group.items():
        match = next((s for s in lateral_tags if curve_tag in [abs(t) for _, t in gmsh.model.getBoundary([(2, s)], combined=False, oriented=False)]), None)
        if match is None:
            raise RuntimeError(f"could not find the extruded lateral surface for curve {curve_tag} ({group_name}) -- extrusion topology assumption failed")
        group_to_surfaces.setdefault(group_name, []).append(match)

    for group_name, surfaces in group_to_surfaces.items():
        gmsh.model.addPhysicalGroup(2, surfaces, name=group_name)
    gmsh.model.addPhysicalGroup(2, [surf], name="BACK")
    gmsh.model.addPhysicalGroup(2, [far_cap_tag], name="FRONT")
    gmsh.model.addPhysicalGroup(3, [out[1][1]], name="FLUID")

    gmsh.model.mesh.generate(3)

    os.makedirs(output_dir, exist_ok=True)
    msh_path = os.path.join(output_dir, f"{mesh_name}.msh")
    gmsh.write(msh_path)
    gmsh.finalize()

    return msh_path


def _docker_exec(container: str, command: str, work_dir: str = None) -> tuple:
    """Runs a command inside the given running Docker container. work_dir
    is a path INSIDE the container (e.g. /data/<case>). Returns
    (returncode, combined_output)."""
    cd = f"cd {work_dir} && " if work_dir else ""
    proc = subprocess.run(
        ["docker", "exec", container, "bash", "-lc", f"{cd}{command}"],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def convert_mesh_to_foam(container: str, container_case_dir: str, msh_filename: str):
    """Runs gmshToFoam inside the container, then patches
    constant/polyMesh/boundary so the FRONT/BACK patches are type empty
    (gmshToFoam writes them as a generic patch/wall type by default)."""
    returncode, output = _docker_exec(container, f"gmshToFoam {msh_filename}", work_dir=container_case_dir)
    if returncode != 0:
        raise RuntimeError(f"gmshToFoam failed:\n{output}")
    return returncode, output


def set_patch_types(boundary_file_path: str, patch_type_map: dict) -> None:
    """Rewrites the named patches' `type` (and drops any `physicalType`) in
    an OpenFOAM constant/polyMesh/boundary file, in place. gmshToFoam
    writes every patch as the generic type `patch` -- OpenFOAM needs FRONT/
    BACK set to `empty` (the "2D-as-a-1-cell-thick-3D-slab" convention) and
    WALL_MID set to `wall` (wall-function boundary conditions, e.g.
    nutkWallFunction, require the underlying patch type to be `wall`, not
    just a field-level BC choice -- confirmed live: rhoSimpleFoam refuses
    to start otherwise)."""
    with open(boundary_file_path) as f:
        content = f.read()

    for name, patch_type in patch_type_map.items():
        pattern = re.compile(rf"({re.escape(name)}\s*\{{)(.*?)(\n\s*\}})", re.DOTALL)

        def _replace(match, patch_type=patch_type):
            block = match.group(2)
            block = re.sub(r"type\s+\w+;", f"type            {patch_type};", block)
            block = re.sub(r"\n\s*physicalType\s+\w+;", "", block)
            return match.group(1) + block + match.group(3)

        content, n = pattern.subn(_replace, content)
        if n == 0:
            raise ValueError(f"patch '{name}' not found in {boundary_file_path}")

    with open(boundary_file_path, "w") as f:
        f.write(content)


def patch_boundary_types(boundary_file_path: str) -> None:
    """Applies this case's full set of required patch-type fixes: FRONT/
    BACK -> empty, WALL_MID -> wall."""
    set_patch_types(boundary_file_path, {"FRONT": "empty", "BACK": "empty", WALL_MID: "wall"})


def run_check_mesh(container: str, container_case_dir: str):
    return _docker_exec(container, "checkMesh", work_dir=container_case_dir)


def run_rho_simple_foam(container: str, container_case_dir: str):
    return _docker_exec(container, "rhoSimpleFoam", work_dir=container_case_dir)
