import pytest

from efficiency import compute_stage_efficiency, total_to_static_efficiency, total_to_total_efficiency
from turbine import TurbineStageDesignInputs, solve_turbine_stage

TURBINE_GAMMA, TURBINE_R = 1.33, 287.0
TURBINE_CP = TURBINE_GAMMA * TURBINE_R / (TURBINE_GAMMA - 1.0)

T01, P01 = 1679.21, 2_056_992.49


@pytest.fixture
def stage_result():
    design = TurbineStageDesignInputs(
        stage_loading=1.8, flow_coefficient=0.5, degree_of_reaction=0.35,
        blade_speed_limit=350.0, rotational_speed_rpm=3000.0,
    )
    return solve_turbine_stage(
        T01=T01, P01=P01, specific_work_required=219_270.0,
        mass_flow=750.0, cp=TURBINE_CP, gamma=TURBINE_GAMMA, design=design,
    )


def test_total_to_static_efficiency_never_exceeds_total_to_total(stage_result):
    # Exit KE is credited in eta_tt (P03) but wasted in eta_ts (static P3 <
    # P03 always), so eta_ts <= eta_tt is a guaranteed relationship,
    # regardless of the specific design point.
    eta_tt = total_to_total_efficiency(T01, stage_result.T03, P01, stage_result.P03, TURBINE_GAMMA)
    eta_ts = total_to_static_efficiency(T01, stage_result.T03, P01, stage_result.P3, TURBINE_GAMMA)
    assert eta_ts <= eta_tt
    assert 0.0 < eta_ts < eta_tt < 1.0


def test_compute_stage_efficiency_wrapper_matches_individual_calls(stage_result):
    eff = compute_stage_efficiency(stage_result, T01, P01, TURBINE_GAMMA)
    assert eff.eta_tt == pytest.approx(total_to_total_efficiency(T01, stage_result.T03, P01, stage_result.P03, TURBINE_GAMMA))
    assert eff.eta_ts == pytest.approx(total_to_static_efficiency(T01, stage_result.T03, P01, stage_result.P3, TURBINE_GAMMA))


@pytest.mark.parametrize("bad_P03", [0.0, 2_100_000.0])  # <=0 or >= P01
def test_total_to_total_efficiency_rejects_invalid_pressure_ratio(bad_P03):
    with pytest.raises(ValueError):
        total_to_total_efficiency(T01, 1500.0, P01, bad_P03, TURBINE_GAMMA)


@pytest.mark.parametrize("bad_P3", [0.0, 2_100_000.0])
def test_total_to_static_efficiency_rejects_invalid_pressure_ratio(bad_P3):
    with pytest.raises(ValueError):
        total_to_static_efficiency(T01, 1500.0, P01, bad_P3, TURBINE_GAMMA)
