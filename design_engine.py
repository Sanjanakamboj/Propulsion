"""Master engine design driver -- reads engine_inputs.txt, runs the design
pipeline across the toolkit's modules, saves diagrams, and logs the run.

Usage:
    python3 "design_engine.py"

Reads `engine_inputs.txt` and writes to `Results/<run_id>/` and
`design_log.xlsx`, all next to this script.

As other toolkit folders are built out, they plug in here after compressor/
turbine mean-line design, consuming the per-stage results:
    - Blade Geometry Generator: blade profiles from the mean-line output
    - CFD Analysis: aerodynamic verification of the generated blades
    - Performance Analysis: off-design performance across the flight envelope
"""

import configparser
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

TOOLKIT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLKIT_ROOT / "Brayton Cycle Analysis"))
sys.path.insert(0, str(TOOLKIT_ROOT / "Compressor Calculations"))
sys.path.insert(0, str(TOOLKIT_ROOT / "Turbine Calculations"))
sys.path.insert(0, str(TOOLKIT_ROOT / "Blade Geometry Generator"))
sys.path.insert(0, str(TOOLKIT_ROOT / "CFD Analysis"))
sys.path.insert(0, str(TOOLKIT_ROOT / "Combustor Analysis"))
sys.path.insert(0, str(TOOLKIT_ROOT / "Nozzle Analysis"))
sys.path.insert(0, str(TOOLKIT_ROOT / "Utils"))

from blade_passage_geometry import compute_passage_geometry, plot_blade_passage, plot_passage_width  # noqa: E402
from blade_section import BladeSectionInputs, build_blade_section  # noqa: E402
from blade_sections import generate_spanwise_sections  # noqa: E402
from combustion import assess_combustion  # noqa: E402
from compressor import CompressorStageDesignInputs, design_compressor_stages  # noqa: E402
from converging import solve_converging_nozzle  # noqa: E402
from choking import assess_choking as assess_nozzle_choking  # noqa: E402
from diagrams import plot_cycle_diagrams  # noqa: E402
from engine import TurbojetDesignInputs  # noqa: E402
from engine_sizing import size_engine  # noqa: E402
from export_csv import write_blade_csv  # noqa: E402
from export_stl import write_blade_stl  # noqa: E402
from mesh import build_cascade_domain, generate_cascade_mesh, plot_cascade_domain  # noqa: E402
from mission import MissionRequirements  # noqa: E402
from residence_time import residence_time as compute_residence_time  # noqa: E402
from rotor_blade_design import BladeSizingInputs, size_rotor_blade  # noqa: E402
from sanity_checks import compressor_stage_sanity_check, format_sanity_report  # noqa: E402
from stacking import stack_sections  # noqa: E402
from stage_diagrams import compressor_multistage_hs_diagram, compressor_stage_diagrams  # noqa: E402
from SU2 import SU2CascadeInputs, plot_mesh_quality, run_su2, write_su2_config  # noqa: E402
from thrust import compute_thrust as compute_nozzle_thrust  # noqa: E402
from turbine import TurbineStageDesignInputs, design_turbine_stages  # noqa: E402
from turbine_sanity_checks import turbine_stage_sanity_check  # noqa: E402
from turbine_stage_diagrams import turbine_stage_diagrams  # noqa: E402
from visualization import plot_stacked_blade  # noqa: E402

from combustor_plots import combustor_diagrams  # noqa: E402
from nozzle_plots import nozzle_diagrams  # noqa: E402

LOG_COLUMNS = [
    "run_id",
    "timestamp",
    "cruise_mach",
    "cruise_altitude_m",
    "required_thrust_N",
    "compressor_pressure_ratio",
    "turbine_inlet_temperature_K",
    "mdot_air_kg_s",
    "net_thrust_N",
    "specific_thrust_N_per_kg_s",
    "tsfc_kg_per_N_hr",
    "thermal_efficiency",
    "propulsive_efficiency",
    "overall_efficiency",
    "compressor_face_diameter_m",
    "nozzle_exit_area_m2",
    "nozzle_choked",
    "compressor_n_stages",
    "compressor_mean_diameter_m",
    "compressor_max_an2",
    "turbine_n_stages",
    "turbine_mean_diameter_m",
    "turbine_max_an2",
    "compressor_sanity_all_passed",
    "turbine_sanity_all_passed",
    "compressor_num_blades",
    "compressor_chord_m",
    "compressor_throat_m",
    "turbine_num_blades",
    "turbine_chord_m",
    "turbine_throat_m",
    "compressor_reheat_factor",
    "combustor_equivalence_ratio",
    "combustor_residence_time_ms",
    "nozzle_choke_status",
    "output_dir",
]


def load_inputs(path: Path):
    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if not config.read(path):
        raise FileNotFoundError(f"Could not read input file: {path}")

    mission_cfg = config["mission"]
    mission = MissionRequirements(
        cruise_mach=mission_cfg.getfloat("cruise_mach"),
        cruise_altitude_m=mission_cfg.getfloat("cruise_altitude_m"),
        required_thrust_N=mission_cfg.getfloat("required_thrust_N"),
    )

    cycle_cfg = config["cycle"]
    ambient_T, ambient_P = mission.ambient_conditions
    design = TurbojetDesignInputs(
        ambient_T=ambient_T,
        ambient_P=ambient_P,
        flight_mach=mission.cruise_mach,
        compressor_pressure_ratio=cycle_cfg.getfloat("compressor_pressure_ratio"),
        compressor_efficiency=cycle_cfg.getfloat("compressor_efficiency"),
        turbine_inlet_temperature=cycle_cfg.getfloat("turbine_inlet_temperature"),
        turbine_efficiency=cycle_cfg.getfloat("turbine_efficiency"),
        mechanical_efficiency=cycle_cfg.getfloat("mechanical_efficiency"),
        combustor_pressure_loss_frac=cycle_cfg.getfloat("combustor_pressure_loss_frac"),
        combustor_efficiency=cycle_cfg.getfloat("combustor_efficiency"),
        fuel_lhv=cycle_cfg.getfloat("fuel_lhv"),
        nozzle_efficiency=cycle_cfg.getfloat("nozzle_efficiency"),
        inlet_pressure_recovery=cycle_cfg.getfloat("inlet_pressure_recovery"),
    )

    sizing_cfg = config["sizing"]
    sizing_params = dict(
        axial_mach=sizing_cfg.getfloat("axial_mach"),
        hub_to_tip_ratio=sizing_cfg.getfloat("hub_to_tip_ratio"),
    )

    rotational_speed_rpm = config["shaft"].getfloat("rotational_speed_rpm")

    compressor_cfg = config["compressor_stage"]
    compressor_design = CompressorStageDesignInputs(
        stage_loading=compressor_cfg.getfloat("stage_loading"),
        flow_coefficient=compressor_cfg.getfloat("flow_coefficient"),
        degree_of_reaction=compressor_cfg.getfloat("degree_of_reaction"),
        blade_speed_limit=compressor_cfg.getfloat("blade_speed_limit"),
        rotational_speed_rpm=rotational_speed_rpm,
        stage_efficiency=compressor_cfg.getfloat("stage_efficiency"),
    )
    compressor_max_stages = compressor_cfg.getint("max_stages")

    turbine_cfg = config["turbine_stage"]
    turbine_design = TurbineStageDesignInputs(
        stage_loading=turbine_cfg.getfloat("stage_loading"),
        flow_coefficient=turbine_cfg.getfloat("flow_coefficient"),
        degree_of_reaction=turbine_cfg.getfloat("degree_of_reaction"),
        blade_speed_limit=turbine_cfg.getfloat("blade_speed_limit"),
        rotational_speed_rpm=rotational_speed_rpm,
        stator_efficiency=turbine_cfg.getfloat("stator_efficiency"),
        rotor_efficiency=turbine_cfg.getfloat("rotor_efficiency"),
        inlet_mach_number=turbine_cfg.getfloat("inlet_mach_number"),
    )
    turbine_max_stages = turbine_cfg.getint("max_stages")

    blade_cfg = config["blade_geometry"]
    blade_design = BladeSizingInputs(
        aspect_ratio=blade_cfg.getfloat("aspect_ratio"),
        zweifel_coefficient=blade_cfg.getfloat("zweifel_coefficient"),
        stagger_angle_deg=blade_cfg.getfloat("stagger_angle_deg"),
        thickness_to_chord=blade_cfg.getfloat("thickness_to_chord"),
        te_radius=blade_cfg.getfloat("te_radius_m"),
        le_radius_fraction_of_pitch=blade_cfg.getfloat("le_radius_fraction_of_pitch"),
    )

    cfd_cfg = config["cfd"]
    cfd_params = dict(
        run_cfd=cfd_cfg.getboolean("run_cfd"),
        su2_binary=cfd_cfg.get("su2_binary"),
        inner_iterations=cfd_cfg.getint("inner_iterations"),
        reynolds_number=cfd_cfg.getfloat("reynolds_number"),
    )

    combustor_volume_m3 = config["combustor"].getfloat("combustor_volume_m3")

    return (
        mission, design, sizing_params,
        compressor_design, compressor_max_stages,
        turbine_design, turbine_max_stages,
        blade_design, cfd_params, combustor_volume_m3,
    )


def _generate_blade_geometry(
    blade_height_in: float, blade_height_out: float, mean_diameter: float,
    beta_in_deg: float, beta_out_deg: float, blade_design: BladeSizingInputs, save_prefix: str,
    generate_section: bool = True,
):
    """Chord/pitch/Zweifel sizing (always) -> 2D blade section -> passage
    geometry (only if generate_section), for one representative (repeating)
    stage. Saves the two passage-geometry plots and returns
    (sizing, passage, blade) -- passage and blade are None when
    generate_section=False.

    generate_section defaults to True but should be False for a COMPRESSOR
    rotor: the bundled ParaBlade parametrization (and the Zweifel-pitch
    formula's sign convention) assumes an ACCELERATING cascade
    (beta_in < beta_out, i.e. a turbine rotor/nozzle). A compressor rotor
    DECELERATES the relative flow (beta_in > beta_out, a diffusing cascade),
    which was confirmed to produce a self-intersecting blade (negative
    throat width) -- compressor cascades need a different parametrization
    (e.g. NACA-65 / circular-arc camberline, Lieblein diffusion factor)
    that hasn't been built yet.
    """
    sizing = size_rotor_blade(
        blade_height_in=blade_height_in, blade_height_out=blade_height_out,
        mean_diameter=mean_diameter, beta_in_deg=beta_in_deg, beta_out_deg=beta_out_deg,
        design=blade_design,
    )
    if not generate_section:
        return sizing, None, None

    section = BladeSectionInputs(
        stagger_angle_deg=blade_design.stagger_angle_deg,
        beta_in_deg=beta_in_deg,
        beta_out_deg=beta_out_deg,
        le_radius_over_cx=sizing.le_radius / sizing.axial_chord,
        te_radius_over_cx=sizing.te_radius / sizing.axial_chord,
    )
    blade = build_blade_section(section)
    passage = compute_passage_geometry(blade, sizing.axial_chord, sizing.pitch)

    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots(figsize=(13, 9))
    plot_blade_passage(blade, sizing.axial_chord, sizing.pitch, passage, ax=ax1)
    fig1.tight_layout()
    fig1.savefig(f"{save_prefix}_passage_geometry.png", dpi=150, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    plot_passage_width(passage, sizing.axial_chord, ax=ax2)
    fig2.tight_layout()
    fig2.savefig(f"{save_prefix}_passage_width.png", dpi=150, bbox_inches="tight")

    return sizing, passage, blade


def _write_stage_csv(path: Path, stages, fields):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage"] + fields)
        for i, stage in enumerate(stages, start=1):
            writer.writerow([i] + [getattr(stage, field) for field in fields])


def log_run(log_path: Path, row_values: dict):
    import openpyxl

    if log_path.exists():
        workbook = openpyxl.load_workbook(log_path)
        sheet = workbook.active
    else:
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Design Runs"
        sheet.append(LOG_COLUMNS)

    sheet.append([row_values[col] for col in LOG_COLUMNS])
    workbook.save(log_path)


def main():
    (
        mission,
        design,
        sizing_params,
        compressor_design,
        compressor_max_stages,
        turbine_design,
        turbine_max_stages,
        blade_design,
        cfd_params,
        combustor_volume_m3,
    ) = load_inputs(TOOLKIT_ROOT / "engine_inputs.txt")

    sizing, sized_design, results, stage_records = size_engine(design, mission.required_thrust_N, **sizing_params)

    station2 = results.stations["2"]
    compressor_stages = design_compressor_stages(
        T01=station2.T0,
        P01=station2.P0,
        total_specific_work=results.compressor_specific_work,
        mass_flow=sizing.mdot_air,
        cp=sized_design.cold_gas.cp,
        gamma=sized_design.cold_gas.gamma,
        design=compressor_design,
        max_stages=compressor_max_stages,
    )

    station4 = results.stations["4"]
    turbine_stages = design_turbine_stages(
        T01=station4.T0,
        P01=station4.P0,
        total_specific_work=results.turbine_specific_work,
        mass_flow=station4.mdot,
        cp=sized_design.hot_gas.cp,
        gamma=sized_design.hot_gas.gamma,
        design=turbine_design,
        max_stages=turbine_max_stages,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = TOOLKIT_ROOT / "Results" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    diagram_path = output_dir / "cycle_diagrams.png"
    plot_cycle_diagrams(stage_records, save_path=str(diagram_path))

    _write_stage_csv(
        output_dir / "compressor_stages.csv", compressor_stages,
        ["U", "achieved_stage_loading", "flow_coefficient", "T03", "P03", "de_haller", "an2"],
    )
    _write_stage_csv(
        output_dir / "turbine_stages.csv", turbine_stages,
        ["U", "achieved_stage_loading", "flow_coefficient", "T03", "P03", "M2", "Mw3", "an2"],
    )

    # Every stage is a repeating stage (same stage_loading/flow_coefficient/
    # degree_of_reaction target), so the velocity triangle and h-s shape are
    # identical stage-to-stage -- one representative diagram each is enough
    # (see compressor_stages.csv / turbine_stages.csv for the full per-stage
    # thermodynamic state, which does vary).
    compressor_stage_diagrams(
        compressor_stages[0], sized_design.cold_gas.cp, sized_design.cold_gas.gamma,
        save_prefix=str(output_dir / "compressor_stage"),
    )
    turbine_stage_diagrams(
        turbine_stages[0], sized_design.hot_gas.cp, sized_design.hot_gas.gamma,
        save_prefix=str(output_dir / "turbine_stage"),
    )

    # Multi-stage reheat-factor h-s diagram across ALL compressor stages --
    # distinct from compressor_stage_diagrams above (which shows one
    # representative stage's own rotor+stator path). Turbine only has 2
    # stages in this design point, where the single-stage ladder above
    # already covers what matters; the compressor's many stages are where
    # the inter-stage reheat effect actually shows up.
    multistage_hs_ax, multistage_hs_result = compressor_multistage_hs_diagram(
        station2.T0, station2.P0, compressor_stages, sized_design.cold_gas.cp, sized_design.cold_gas.gamma,
    )
    multistage_hs_ax.figure.tight_layout()
    multistage_hs_ax.figure.savefig(str(output_dir / "compressor_multistage_hs_diagram.png"), dpi=150, bbox_inches="tight")

    # Compressor: chord/blade-count sizing only. Zweifel's criterion (used
    # for the pitch) is a turbine (accelerating-cascade) loading criterion --
    # applied to a compressor's much smaller flow turning it predicts a pitch
    # too tight for the blade's own thickness, so the 2D section/passage
    # geometry (which needs a proper diffusion-factor-based pitch) is
    # skipped until that's built. See sizing.pitch for the (Zweifel, not
    # directly usable) reference value.
    compressor_blade_sizing, compressor_passage, _ = _generate_blade_geometry(
        blade_height_in=compressor_stages[0].annulus_1.blade_height,
        blade_height_out=compressor_stages[0].annulus_2.blade_height,
        mean_diameter=compressor_stages[0].annulus_1.mean_diameter,
        beta_in_deg=compressor_stages[0].beta1_deg,
        beta_out_deg=compressor_stages[0].beta2_deg,
        blade_design=blade_design,
        save_prefix=str(output_dir / "compressor_blade"),
        generate_section=False,
    )
    turbine_blade_sizing, turbine_passage, turbine_blade = _generate_blade_geometry(
        blade_height_in=turbine_stages[0].annulus_2.blade_height,
        blade_height_out=turbine_stages[0].annulus_3.blade_height,
        mean_diameter=turbine_stages[0].annulus_2.mean_diameter,
        beta_in_deg=turbine_stages[0].beta2_deg,
        beta_out_deg=turbine_stages[0].beta3_deg,
        blade_design=blade_design,
        save_prefix=str(output_dir / "turbine_blade"),
    )

    hot_gas = sized_design.hot_gas
    turbine_stage = turbine_stages[0]

    # Full 3D turbine blade (Blade Geometry Generator): free-vortex twist at
    # hub/mean/tip, stacked into one 3D geometry, exported to CSV/STL plus a
    # 3D visualization. Turbine only, for the same reason the compressor's
    # 2D section is skipped above. hub/tip radius averaged between
    # annulus_2 (LE) and annulus_3 (TE) since size_rotor_blade already
    # collapses to one mean_diameter for the whole blade.
    hub_radius_avg = (turbine_stage.annulus_2.hub_radius + turbine_stage.annulus_3.hub_radius) / 2.0
    tip_radius_avg = (turbine_stage.annulus_2.tip_radius + turbine_stage.annulus_3.tip_radius) / 2.0
    spanwise_sections = generate_spanwise_sections(
        Vt_in_mean=turbine_stage.Vt2, Vt_out_mean=turbine_stage.Vt3, U_mean=turbine_stage.U, Vx=turbine_stage.Vx,
        beta_in_mean_deg=turbine_stage.beta2_deg, beta_out_mean_deg=turbine_stage.beta3_deg,
        mean_radius=turbine_stage.annulus_2.mean_diameter / 2.0,
        hub_radius=hub_radius_avg, tip_radius=tip_radius_avg,
        stagger_angle_deg=blade_design.stagger_angle_deg, axial_chord=turbine_blade_sizing.axial_chord,
        le_radius_over_cx=turbine_blade_sizing.le_radius / turbine_blade_sizing.axial_chord,
        te_radius_over_cx=turbine_blade_sizing.te_radius / turbine_blade_sizing.axial_chord,
        n_sections=3,
    )
    turbine_blade_3d = stack_sections(spanwise_sections)
    blade_3d_ax = plot_stacked_blade(turbine_blade_3d)
    blade_3d_ax.figure.savefig(str(output_dir / "turbine_blade_3d.png"), dpi=150, bbox_inches="tight")
    write_blade_csv(turbine_blade_3d, str(output_dir / "turbine_blade_3d.csv"))
    n_blade_triangles = write_blade_stl(turbine_blade_3d, str(output_dir / "turbine_blade_3d.stl"))

    # Combustor Analysis: equivalence ratio, pressure loss, residence time --
    # reports the SAME fuel_air_ratio the 0D cycle (stages.py's Combustor)
    # already solved, in this folder's own terms, plus the residence-time
    # check the 0D cycle doesn't do.
    station3 = results.stations["3"]
    combustion_state = assess_combustion(results.fuel_air_ratio)
    rho_combustor_exit = station4.P0 / (hot_gas.R * station4.T0)
    combustor_tau = compute_residence_time(combustor_volume_m3, station4.mdot, rho_combustor_exit)
    combustor_diagrams(
        T_in=station3.T0, T_exit=station4.T0, P_in=station3.P0, P_out=station4.P0,
        far=results.fuel_air_ratio, combustion_state=combustion_state, tau=combustor_tau,
        save_prefix=str(output_dir / "combustor"),
    )

    # Nozzle Analysis: independent converging-nozzle solve + choking status +
    # thrust breakdown, fed from the same station5 the 0D cycle already
    # solved (stages.py's Nozzle) -- numerically reproduces that same exit
    # state (validated separately in Nozzle Analysis/test_converging.py),
    # and additionally breaks thrust into its momentum/pressure components.
    station5 = results.stations["5"]
    R_hot = hot_gas.cp * (hot_gas.gamma - 1.0) / hot_gas.gamma
    nozzle_exit_state = solve_converging_nozzle(
        T0_in=station5.T0, P0_in=station5.P0, P_ambient=design.ambient_P,
        cp=hot_gas.cp, gamma=hot_gas.gamma, R=R_hot, isentropic_efficiency=design.nozzle_efficiency,
    )
    nozzle_choke_assessment = assess_nozzle_choking(station5.P0, design.ambient_P, hot_gas.gamma)
    rho_nozzle_exit = nozzle_exit_state.P_exit / (R_hot * nozzle_exit_state.T_exit)
    nozzle_thrust_breakdown = compute_nozzle_thrust(
        mdot_gas=station5.mdot, V_exit=nozzle_exit_state.V_exit, P_exit=nozzle_exit_state.P_exit,
        P_ambient=design.ambient_P, rho_exit=rho_nozzle_exit, mdot_air=sizing.mdot_air,
        V0=results.flight_velocity, fuel_flow=results.fuel_flow,
    )
    nozzle_diagrams(
        nozzle_exit_state, nozzle_choke_assessment, nozzle_thrust_breakdown,
        save_prefix=str(output_dir / "nozzle"),
    )

    # CFD: 3-blade cascade of the turbine rotor blade (compressor blade
    # section is skipped for the same reason noted above, so there's no
    # compressor blade to mesh yet). The fast geometry pre-check always
    # runs; the real GMSH mesh + SU2 solve only run if cfd_params.run_cfd.
    domain = build_cascade_domain(
        turbine_blade, turbine_blade_sizing.axial_chord, turbine_blade_sizing.pitch,
        beta_in_deg=turbine_stage.beta2_deg, beta_out_deg=turbine_stage.beta3_deg,
    )
    domain_ax = plot_cascade_domain(domain)
    domain_ax.figure.tight_layout()
    domain_ax.figure.savefig(str(output_dir / "turbine_cascade_domain.png"), dpi=150, bbox_inches="tight")

    if cfd_params["run_cfd"]:
        su2_path, msh_path, n_nodes, n_elements = generate_cascade_mesh(domain, str(output_dir))
        print(f"  CFD mesh generated    = {n_nodes} nodes, {n_elements} elements")

        P02_rel = turbine_stage.P2 * (turbine_stage.T02_rel / turbine_stage.T2) ** (hot_gas.gamma / (hot_gas.gamma - 1.0))
        su2_inputs = SU2CascadeInputs(
            gamma=hot_gas.gamma,
            gas_constant=hot_gas.R,
            relative_inlet_mach=turbine_stage.Mw2,
            static_temperature=turbine_stage.T2,
            static_pressure=turbine_stage.P2,
            inlet_total_temperature=turbine_stage.T02_rel,
            inlet_total_pressure=P02_rel,
            inlet_flow_angle_deg=turbine_stage.beta2_deg,
            outlet_static_pressure=turbine_stage.P3,
            reference_length=turbine_blade_sizing.axial_chord,
            reynolds_number=cfd_params["reynolds_number"],
            inner_iterations=cfd_params["inner_iterations"],
        )
        config_path = str(output_dir / "blade_su2.cfg")
        write_su2_config(su2_inputs, mesh_filename=os.path.basename(su2_path), config_path=config_path)

        print("  Running SU2_CFD ...")
        returncode, _ = run_su2(config_path, work_dir=str(output_dir), su2_binary=cfd_params["su2_binary"])
        if returncode == 0:
            flow_vtu = str(output_dir / "flow.vtu")
            fig_mq, _, mesh_info = plot_mesh_quality(flow_vtu, turbine_blade, turbine_blade_sizing.axial_chord, turbine_blade_sizing.pitch)
            fig_mq.savefig(str(output_dir / "turbine_mesh_quality.png"), dpi=150, bbox_inches="tight")
            print(f"  SU2 finished          = {mesh_info['n_points']} points, {mesh_info['n_cells']} cells -> flow.vtu")
        else:
            print(f"  SU2 exited with code {returncode} -- see {output_dir / 'blade_su2.cfg'} and rerun manually to inspect")

    compressor_checks = compressor_stage_sanity_check(compressor_stages[0])
    turbine_checks = turbine_stage_sanity_check(turbine_stages[0])
    sanity_report = format_sanity_report("Compressor Stage (representative)", compressor_checks)
    sanity_report += "\n" + format_sanity_report("Turbine Stage (representative)", turbine_checks)
    (output_dir / "sanity_checks.txt").write_text(sanity_report)

    compressor_diameters = [s.annulus_1.mean_diameter for s in compressor_stages]
    turbine_diameters = [s.annulus_2.mean_diameter for s in turbine_stages]

    log_run(
        TOOLKIT_ROOT / "design_log.xlsx",
        {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "cruise_mach": mission.cruise_mach,
            "cruise_altitude_m": mission.cruise_altitude_m,
            "required_thrust_N": mission.required_thrust_N,
            "compressor_pressure_ratio": design.compressor_pressure_ratio,
            "turbine_inlet_temperature_K": design.turbine_inlet_temperature,
            "mdot_air_kg_s": sizing.mdot_air,
            "net_thrust_N": results.net_thrust,
            "specific_thrust_N_per_kg_s": results.specific_thrust,
            "tsfc_kg_per_N_hr": results.tsfc * 3600.0,
            "thermal_efficiency": results.thermal_efficiency,
            "propulsive_efficiency": results.propulsive_efficiency,
            "overall_efficiency": results.overall_efficiency,
            "compressor_face_diameter_m": sizing.compressor_face_diameter,
            "nozzle_exit_area_m2": sizing.nozzle_exit_area,
            "nozzle_choked": results.nozzle_choked,
            "compressor_n_stages": len(compressor_stages),
            "compressor_mean_diameter_m": sum(compressor_diameters) / len(compressor_diameters),
            "compressor_max_an2": max(s.an2 for s in compressor_stages),
            "turbine_n_stages": len(turbine_stages),
            "turbine_mean_diameter_m": sum(turbine_diameters) / len(turbine_diameters),
            "turbine_max_an2": max(s.an2 for s in turbine_stages),
            "compressor_sanity_all_passed": all(c.passed for c in compressor_checks),
            "turbine_sanity_all_passed": all(c.passed for c in turbine_checks),
            "compressor_num_blades": compressor_blade_sizing.num_blades,
            "compressor_chord_m": compressor_blade_sizing.chord,
            "compressor_throat_m": compressor_passage.throat if compressor_passage else None,
            "turbine_num_blades": turbine_blade_sizing.num_blades,
            "turbine_chord_m": turbine_blade_sizing.chord,
            "turbine_throat_m": turbine_passage.throat,
            "compressor_reheat_factor": multistage_hs_result.reheat_factor,
            "combustor_equivalence_ratio": combustion_state.equivalence_ratio,
            "combustor_residence_time_ms": combustor_tau * 1e3,
            "nozzle_choke_status": nozzle_choke_assessment.status,
            "output_dir": str(output_dir),
        },
    )

    print(f"Run {run_id}")
    print(f"  Mass flow             = {sizing.mdot_air:.1f} kg/s")
    print(f"  Net thrust            = {results.net_thrust / 1e3:.1f} kN (target {mission.required_thrust_N / 1e3:.1f} kN)")
    print(f"  TSFC                  = {results.tsfc * 3600:.4f} kg/(N*hr)")
    print(f"  Thermal efficiency    = {results.thermal_efficiency * 100:.1f} %")
    print(f"  Propulsive efficiency = {results.propulsive_efficiency * 100:.1f} %")
    print(f"  Overall efficiency    = {results.overall_efficiency * 100:.1f} %")
    print(f"  Compressor diameter   = {sizing.compressor_face_diameter:.2f} m")
    print(f"  Compressor stages     = {len(compressor_stages)} (mean diameter {sum(compressor_diameters)/len(compressor_diameters):.2f} m)")
    print(f"  Turbine stages        = {len(turbine_stages)} (mean diameter {sum(turbine_diameters)/len(turbine_diameters):.2f} m)")
    print(f"  Compressor blade      = chord {compressor_blade_sizing.chord*1e3:.1f} mm, {compressor_blade_sizing.num_blades} blades (2D section skipped -- see note above on Zweifel vs diffusion factor)")
    print(f"  Turbine blade         = chord {turbine_blade_sizing.chord*1e3:.1f} mm, {turbine_blade_sizing.num_blades} blades, throat {turbine_passage.throat*1e3:.2f} mm")
    print(f"  Turbine blade 3D      = {len(spanwise_sections)} spanwise sections, {n_blade_triangles} STL triangles")
    print(f"  Compressor reheat RF  = {multistage_hs_result.reheat_factor:.4f} (across {len(compressor_stages)} stages)")
    print(f"  Combustor             = phi {combustion_state.equivalence_ratio:.3f} ({combustion_state.regime}), residence time {combustor_tau*1e3:.2f} ms")
    print(f"  Nozzle                = {nozzle_choke_assessment.status}, V_exit {nozzle_exit_state.V_exit:.1f} m/s")
    print(f"  Cycle diagrams saved  {diagram_path}")
    print(f"  Stage tables/diagrams {output_dir}")
    print(f"  Logged to             {TOOLKIT_ROOT / 'design_log.xlsx'}")
    print(sanity_report)


if __name__ == "__main__":
    main()
