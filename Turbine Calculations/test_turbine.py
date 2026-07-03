import pytest

from turbine import TurbineStageDesignInputs, design_turbine_stages, solve_turbine_stage


@pytest.fixture
def notebook_design():
    # Matches the validated single-stage HP turbine example in
    # Turbine Stage Design.ipynb (750 kg/s, N=3000 rpm, psi target 1.8,
    # phi=0.5, DOR=0.35).
    return TurbineStageDesignInputs(
        stage_loading=1.8,
        flow_coefficient=0.5,
        degree_of_reaction=0.35,
        blade_speed_limit=350.0,
        rotational_speed_rpm=3000.0,
        stator_efficiency=0.90,
        rotor_efficiency=0.90,
        inlet_mach_number=0.15,
        inlet_flow_angle_deg=0.0,
    )


@pytest.fixture
def notebook_gas():
    gamma = 1.33
    R = 287.0
    cp = gamma * R / (gamma - 1.0)
    return cp, gamma


def test_matches_validated_notebook_example(notebook_design, notebook_gas):
    cp, gamma = notebook_gas
    result = solve_turbine_stage(
        T01=1679.21,
        P01=2_056_992.4936606083,
        specific_work_required=219_270.0,
        mass_flow=750.0,
        cp=cp,
        gamma=gamma,
        design=notebook_design,
    )

    assert result.U == pytest.approx(349.026, rel=1e-3)
    assert result.achieved_stage_loading == pytest.approx(1.800, rel=1e-3)
    assert result.P3 == pytest.approx(1_107_264.63, rel=1e-3)
    assert result.M2 == pytest.approx(0.6996, rel=1e-3)
    assert result.beta2_deg == pytest.approx(42.6413, rel=1e-3)
    assert result.Mw2 == pytest.approx(0.3081, rel=1e-3)
    assert result.Mw3 == pytest.approx(0.6661, rel=1e-3)
    assert result.beta3_deg == pytest.approx(69.5316, rel=1e-3)
    assert result.alpha3_deg == pytest.approx(34.1813, rel=1e-3)
    assert result.an2 == pytest.approx(13_405_704.38, rel=1e-3)
    assert result.specific_work == pytest.approx(219_270.0, rel=1e-6)


def test_stage_delivers_requested_specific_work(notebook_design, notebook_gas):
    cp, gamma = notebook_gas
    result = solve_turbine_stage(
        T01=1650.0, P01=1_900_000.0, specific_work_required=180_000.0,
        mass_flow=500.0, cp=cp, gamma=gamma, design=notebook_design,
    )
    assert result.specific_work == pytest.approx(180_000.0, rel=1e-6)


def test_temperatures_drop_through_the_stage(notebook_design, notebook_gas):
    cp, gamma = notebook_gas
    result = solve_turbine_stage(
        T01=1650.0, P01=1_900_000.0, specific_work_required=180_000.0,
        mass_flow=500.0, cp=cp, gamma=gamma, design=notebook_design,
    )
    assert result.T2 < 1650.0
    assert result.T3 < result.T2


def test_blade_speed_capped_at_mechanical_limit(notebook_gas):
    cp, gamma = notebook_gas
    design = TurbineStageDesignInputs(
        stage_loading=0.5,  # low loading -> would demand a very high U
        flow_coefficient=0.5,
        degree_of_reaction=0.35,
        blade_speed_limit=300.0,
        rotational_speed_rpm=3000.0,
    )
    result = solve_turbine_stage(
        T01=1650.0, P01=1_900_000.0, specific_work_required=180_000.0,
        mass_flow=500.0, cp=cp, gamma=gamma, design=design,
    )
    assert result.U == pytest.approx(300.0)
    assert result.achieved_stage_loading > 0.5  # loading rises since U was capped


def test_design_turbine_stages_splits_work_to_respect_blade_speed(notebook_design, notebook_gas):
    cp, gamma = notebook_gas
    # Total work matching the notebook's actual 4-stage turbine (4 x 219.27 kJ/kg per
    # stage, each within the 350 m/s blade speed limit) -- should resolve to 4 stages.
    total_work = 4 * 219_270.0
    stages = design_turbine_stages(
        T01=1679.21, P01=2_056_992.49, total_specific_work=total_work,
        mass_flow=750.0, cp=cp, gamma=gamma, design=notebook_design,
    )
    assert len(stages) == 4
    total_delivered = sum(s.specific_work for s in stages)
    assert total_delivered == pytest.approx(total_work, rel=1e-6)
    for stage in stages:
        assert stage.U <= 350.0 + 1e-6


def test_design_turbine_stages_chains_stagnation_state(notebook_design, notebook_gas):
    cp, gamma = notebook_gas
    stages = design_turbine_stages(
        T01=1679.21, P01=2_056_992.49, total_specific_work=219_270.0 * 2,
        mass_flow=750.0, cp=cp, gamma=gamma, design=notebook_design,
    )
    assert len(stages) == 2
    assert stages[1].T2 < stages[0].T2  # second stage inlet is cooler than first


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(stage_loading=0.0),
        dict(flow_coefficient=-1.0),
        dict(degree_of_reaction=1.0),
        dict(blade_speed_limit=0.0),
        dict(rotational_speed_rpm=0.0),
        dict(stator_efficiency=1.2),
        dict(inlet_mach_number=1.5),
    ],
)
def test_invalid_design_inputs_raise(kwargs):
    base = dict(stage_loading=1.8, flow_coefficient=0.5, degree_of_reaction=0.35, blade_speed_limit=350.0, rotational_speed_rpm=3000.0)
    base.update(kwargs)
    with pytest.raises(ValueError):
        TurbineStageDesignInputs(**base)
