import math

import pytest

from compressor import CompressorStageDesignInputs, design_compressor_stages, solve_compressor_stage

GAMMA = 1.4
R = 287.0
CP = GAMMA * R / (GAMMA - 1.0)


def _design(**overrides):
    base = dict(
        stage_loading=0.35,
        flow_coefficient=0.5,
        degree_of_reaction=0.5,
        blade_speed_limit=350.0,
        rotational_speed_rpm=8000.0,
        stage_efficiency=0.90,
    )
    base.update(overrides)
    return CompressorStageDesignInputs(**base)


def test_fifty_percent_reaction_gives_mirror_symmetric_triangle():
    # Classic Lambda=0.5 identity: beta1 == alpha2 and beta2 == alpha1.
    result = solve_compressor_stage(
        T01=288.15, P01=101_325.0, specific_work_required=30_000.0,
        mass_flow=50.0, cp=CP, gamma=GAMMA, design=_design(degree_of_reaction=0.5),
    )
    assert result.beta1_deg == pytest.approx(result.alpha2_deg, rel=1e-9)
    assert result.beta2_deg == pytest.approx(result.alpha1_deg, rel=1e-9)


def test_euler_work_matches_velocity_triangle():
    result = solve_compressor_stage(
        T01=288.15, P01=101_325.0, specific_work_required=30_000.0,
        mass_flow=50.0, cp=CP, gamma=GAMMA, design=_design(degree_of_reaction=0.35),
    )
    Cw1 = result.Cx * math.tan(math.radians(result.alpha1_deg))
    Cw2 = result.Cx * math.tan(math.radians(result.alpha2_deg))
    euler_work = result.U * (Cw2 - Cw1)
    assert euler_work == pytest.approx(result.specific_work, rel=1e-6)


def test_stagnation_temperature_and_pressure_rise():
    result = solve_compressor_stage(
        T01=288.15, P01=101_325.0, specific_work_required=30_000.0,
        mass_flow=50.0, cp=CP, gamma=GAMMA, design=_design(),
    )
    assert result.T03 > 288.15
    assert result.P03 > 101_325.0
    assert result.T03 == pytest.approx(288.15 + result.specific_work / CP, rel=1e-9)


def test_repeating_stage_static_temp_rise_matches_stagnation_rise():
    # Station 3 (stator exit) shares station 1's absolute velocity in a
    # repeating stage, so the static temperature rise across the whole
    # stage (T3 - T1) must equal the stagnation rise (T03 - T01) exactly.
    result = solve_compressor_stage(
        T01=288.15, P01=101_325.0, specific_work_required=30_000.0,
        mass_flow=50.0, cp=CP, gamma=GAMMA, design=_design(),
    )
    assert (result.T3 - result.T1) == pytest.approx(result.T03 - 288.15, rel=1e-9)


def test_isentropic_efficiency_reduces_pressure_rise():
    ideal = solve_compressor_stage(
        T01=288.15, P01=101_325.0, specific_work_required=30_000.0,
        mass_flow=50.0, cp=CP, gamma=GAMMA, design=_design(stage_efficiency=1.0),
    )
    lossy = solve_compressor_stage(
        T01=288.15, P01=101_325.0, specific_work_required=30_000.0,
        mass_flow=50.0, cp=CP, gamma=GAMMA, design=_design(stage_efficiency=0.85),
    )
    assert lossy.P03 < ideal.P03
    assert lossy.T03 == pytest.approx(ideal.T03)  # same actual work in either case


def test_blade_speed_capped_at_mechanical_limit():
    design = _design(stage_loading=0.05, blade_speed_limit=250.0)  # low loading -> would need high U
    result = solve_compressor_stage(
        T01=288.15, P01=101_325.0, specific_work_required=30_000.0,
        mass_flow=50.0, cp=CP, gamma=GAMMA, design=design,
    )
    assert result.U == pytest.approx(250.0)
    assert result.achieved_stage_loading > 0.05


def test_design_compressor_stages_splits_and_chains():
    design = _design(blade_speed_limit=300.0)
    total_work = 250_000.0
    stages = design_compressor_stages(
        T01=288.15, P01=101_325.0, total_specific_work=total_work,
        mass_flow=50.0, cp=CP, gamma=GAMMA, design=design,
    )
    assert len(stages) > 1
    total_delivered = sum(s.specific_work for s in stages)
    assert total_delivered == pytest.approx(total_work, rel=1e-6)
    for stage in stages:
        assert stage.U <= 300.0 + 1e-6
    # Chained: each stage's inlet stagnation temp is the previous stage's exit.
    assert stages[1].T1 > stages[0].T1


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(stage_loading=0.0),
        dict(flow_coefficient=-1.0),
        dict(degree_of_reaction=1.0),
        dict(degree_of_reaction=0.0),
        dict(blade_speed_limit=0.0),
        dict(rotational_speed_rpm=0.0),
        dict(stage_efficiency=1.2),
    ],
)
def test_invalid_design_inputs_raise(kwargs):
    base = dict(stage_loading=0.35, flow_coefficient=0.5, degree_of_reaction=0.5, blade_speed_limit=350.0, rotational_speed_rpm=8000.0)
    base.update(kwargs)
    with pytest.raises(ValueError):
        CompressorStageDesignInputs(**base)
