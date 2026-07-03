import pytest

from cooling import (
    assess_stage_cooling,
    blade_temperature_from_effectiveness,
    cooling_effectiveness,
    required_cooling_effectiveness,
)
from turbine import TurbineStageDesignInputs, solve_turbine_stage

TURBINE_GAMMA, TURBINE_R = 1.33, 287.0
TURBINE_CP = TURBINE_GAMMA * TURBINE_R / (TURBINE_GAMMA - 1.0)

T01 = 1679.21


@pytest.fixture
def stage_result():
    design = TurbineStageDesignInputs(
        stage_loading=1.8, flow_coefficient=0.5, degree_of_reaction=0.35,
        blade_speed_limit=350.0, rotational_speed_rpm=3000.0,
    )
    return solve_turbine_stage(
        T01=T01, P01=2_056_992.49, specific_work_required=219_270.0,
        mass_flow=750.0, cp=TURBINE_CP, gamma=TURBINE_GAMMA, design=design,
    )


def test_cooling_effectiveness_matches_manual_formula():
    eta_c = cooling_effectiveness(T_gas=1578.06, T_blade=1250.0, T_coolant=800.0)
    assert eta_c == pytest.approx((1578.06 - 1250.0) / (1578.06 - 800.0))


def test_blade_temperature_is_the_correct_inverse():
    eta_c = cooling_effectiveness(T_gas=1578.06, T_blade=1250.0, T_coolant=800.0)
    T_blade = blade_temperature_from_effectiveness(T_gas=1578.06, eta_c=eta_c, T_coolant=800.0)
    assert T_blade == pytest.approx(1250.0)


def test_required_cooling_effectiveness_matches_manual_formula():
    eta_c = required_cooling_effectiveness(T_gas=1578.06, T_blade_limit=1250.0, T_coolant=800.0)
    assert eta_c == pytest.approx((1578.06 - 1250.0) / (1578.06 - 800.0))
    assert 0.0 < eta_c < 1.0  # a realistic, achievable film-cooling effectiveness


def test_required_cooling_effectiveness_is_zero_when_gas_already_below_limit():
    assert required_cooling_effectiveness(T_gas=1200.0, T_blade_limit=1250.0, T_coolant=800.0) == pytest.approx(0.0)


def test_required_cooling_effectiveness_rejects_limit_at_or_below_coolant_temp():
    with pytest.raises(ValueError):
        required_cooling_effectiveness(T_gas=1578.06, T_blade_limit=750.0, T_coolant=800.0)


def test_cooling_effectiveness_rejects_gas_not_hotter_than_coolant():
    with pytest.raises(ValueError):
        cooling_effectiveness(T_gas=800.0, T_blade=750.0, T_coolant=800.0)


def test_assess_stage_cooling_reports_higher_requirement_for_hotter_row(stage_result):
    # T01 (nozzle's gas temp) > T02_rel (rotor's relative gas temp) for this
    # stage, since the rotor sees a cooler relative frame -- the nozzle
    # should need MORE cooling effectiveness for the same blade limit.
    assessment = assess_stage_cooling(stage_result, T01=T01, T_coolant=800.0, T_blade_limit=1250.0)
    assert T01 > stage_result.T02_rel
    assert assessment.nozzle.required_effectiveness > assessment.rotor.required_effectiveness
    assert assessment.nozzle.cooling_needed
    assert assessment.nozzle.achievable


def test_cooling_requirement_not_achievable_when_limit_requires_over_unity_effectiveness():
    from cooling import CoolingRequirement

    req = CoolingRequirement(T_gas=1578.06, T_blade_limit=1250.0, required_effectiveness=1.2)
    assert not req.achievable
