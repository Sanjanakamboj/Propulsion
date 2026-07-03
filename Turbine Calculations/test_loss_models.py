import pytest

from loss_models import compute_row_losses, nozzle_loss_coefficient, rotor_loss_coefficient
from turbine import TurbineStageDesignInputs, solve_turbine_stage

TURBINE_GAMMA, TURBINE_R = 1.33, 287.0
TURBINE_CP = TURBINE_GAMMA * TURBINE_R / (TURBINE_GAMMA - 1.0)


@pytest.fixture
def stage_result():
    design = TurbineStageDesignInputs(
        stage_loading=1.8, flow_coefficient=0.5, degree_of_reaction=0.35,
        blade_speed_limit=350.0, rotational_speed_rpm=3000.0,
    )
    return solve_turbine_stage(
        T01=1679.21, P01=2_056_992.49, specific_work_required=219_270.0,
        mass_flow=750.0, cp=TURBINE_CP, gamma=TURBINE_GAMMA, design=design,
    )


def test_nozzle_loss_coefficient_matches_manual_formula(stage_result):
    zeta = nozzle_loss_coefficient(stage_result.T2, stage_result.T2s, TURBINE_CP, stage_result.V2)
    expected = (stage_result.T2 - stage_result.T2s) / (stage_result.V2**2 / (2.0 * TURBINE_CP))
    assert zeta == pytest.approx(expected)
    assert zeta > 0.0  # real stator: T2 > T2s


def test_rotor_loss_coefficient_matches_manual_formula(stage_result):
    zeta = rotor_loss_coefficient(stage_result.T3, stage_result.T3s, TURBINE_CP, stage_result.W3)
    expected = (stage_result.T3 - stage_result.T3s) / (stage_result.W3**2 / (2.0 * TURBINE_CP))
    assert zeta == pytest.approx(expected)
    assert zeta > 0.0  # real rotor: T3 > T3s


def test_loss_coefficients_are_zero_for_a_perfect_row():
    assert nozzle_loss_coefficient(T2=500.0, T2s=500.0, cp=1156.7, V2=300.0) == pytest.approx(0.0)
    assert rotor_loss_coefficient(T3=500.0, T3s=500.0, cp=1156.7, W3=300.0) == pytest.approx(0.0)


def test_compute_row_losses_wrapper_matches_individual_calls(stage_result):
    losses = compute_row_losses(stage_result, TURBINE_CP)
    assert losses.zeta_N == pytest.approx(nozzle_loss_coefficient(stage_result.T2, stage_result.T2s, TURBINE_CP, stage_result.V2))
    assert losses.zeta_R == pytest.approx(rotor_loss_coefficient(stage_result.T3, stage_result.T3s, TURBINE_CP, stage_result.W3))


def test_nozzle_loss_coefficient_rejects_non_positive_V2():
    with pytest.raises(ValueError):
        nozzle_loss_coefficient(T2=500.0, T2s=490.0, cp=1156.7, V2=0.0)


def test_rotor_loss_coefficient_rejects_non_positive_W3():
    with pytest.raises(ValueError):
        rotor_loss_coefficient(T3=500.0, T3s=490.0, cp=1156.7, W3=0.0)
