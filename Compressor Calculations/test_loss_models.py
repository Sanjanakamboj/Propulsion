import pytest

from compressor import CompressorStageDesignInputs, solve_compressor_stage
from loss_models import compute_stage_loss, enthalpy_loss_coefficient, isentropic_efficiency_from_loss_coefficient

COMPRESSOR_GAMMA, COMPRESSOR_R = 1.4, 287.0
COMPRESSOR_CP = COMPRESSOR_GAMMA * COMPRESSOR_R / (COMPRESSOR_GAMMA - 1.0)


@pytest.fixture
def stage_result():
    design = CompressorStageDesignInputs(
        stage_loading=0.35, flow_coefficient=0.5, degree_of_reaction=0.5,
        blade_speed_limit=350.0, rotational_speed_rpm=8000.0, stage_efficiency=0.90,
    )
    return solve_compressor_stage(
        T01=288.15, P01=101_325.0, specific_work_required=30_000.0,
        mass_flow=50.0, cp=COMPRESSOR_CP, gamma=COMPRESSOR_GAMMA, design=design,
    )


def test_enthalpy_loss_coefficient_matches_manual_formula(stage_result):
    zeta = enthalpy_loss_coefficient(stage_result.T03, stage_result.T03s, COMPRESSOR_CP, stage_result.W1)
    expected = COMPRESSOR_CP * (stage_result.T03 - stage_result.T03s) / (0.5 * stage_result.W1**2)
    assert zeta == pytest.approx(expected)


def test_enthalpy_loss_coefficient_is_zero_for_perfect_efficiency():
    zeta = enthalpy_loss_coefficient(T_actual=300.0, T_ideal=300.0, cp=1005.0, V_ref=200.0)
    assert zeta == pytest.approx(0.0)


def test_compute_stage_loss_recovers_the_design_isentropic_efficiency(stage_result):
    loss = compute_stage_loss(stage_result, cp=COMPRESSOR_CP)
    assert loss.isentropic_efficiency == pytest.approx(0.90, rel=1e-6)
    assert loss.zeta > 0.0  # real stage: T03 > T03s


def test_efficiency_conversion_is_the_correct_inverse(stage_result):
    T01 = stage_result.T1 + stage_result.V1**2 / (2.0 * COMPRESSOR_CP)
    delta_T_actual = stage_result.T03 - T01
    zeta = enthalpy_loss_coefficient(stage_result.T03, stage_result.T03s, COMPRESSOR_CP, stage_result.W1)
    eta = isentropic_efficiency_from_loss_coefficient(zeta, delta_T_actual, COMPRESSOR_CP, stage_result.W1)
    assert eta == pytest.approx(0.90, rel=1e-6)


def test_higher_loss_coefficient_gives_lower_recovered_efficiency():
    low_zeta_eta = isentropic_efficiency_from_loss_coefficient(zeta=0.05, delta_T_actual=30.0, cp=1005.0, V_ref=245.0)
    high_zeta_eta = isentropic_efficiency_from_loss_coefficient(zeta=0.20, delta_T_actual=30.0, cp=1005.0, V_ref=245.0)
    assert high_zeta_eta < low_zeta_eta


def test_enthalpy_loss_coefficient_rejects_non_positive_V_ref():
    with pytest.raises(ValueError):
        enthalpy_loss_coefficient(T_actual=300.0, T_ideal=295.0, cp=1005.0, V_ref=0.0)


def test_isentropic_efficiency_from_loss_coefficient_rejects_non_positive_delta_T():
    with pytest.raises(ValueError):
        isentropic_efficiency_from_loss_coefficient(zeta=0.1, delta_T_actual=0.0, cp=1005.0, V_ref=200.0)


def test_isentropic_efficiency_from_loss_coefficient_rejects_unachievable_zeta():
    # A huge zeta implies more loss than the entire actual temperature rise -- eta <= 0.
    with pytest.raises(ValueError):
        isentropic_efficiency_from_loss_coefficient(zeta=1000.0, delta_T_actual=30.0, cp=1005.0, V_ref=245.0)
