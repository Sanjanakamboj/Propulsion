import pytest

from compressor import CompressorStageDesignInputs, solve_compressor_stage
from diffusion_factor import required_solidity_for_diffusion_factor, diffusion_factor
from surge_choke import assess_stage, choke_margin_mach, surge_margin_de_haller, surge_margin_diffusion_factor

COMPRESSOR_GAMMA, COMPRESSOR_R = 1.4, 287.0
COMPRESSOR_CP = COMPRESSOR_GAMMA * COMPRESSOR_R / (COMPRESSOR_GAMMA - 1.0)


@pytest.fixture
def stage_result():
    design = CompressorStageDesignInputs(
        stage_loading=0.35, flow_coefficient=0.5, degree_of_reaction=0.5,
        blade_speed_limit=350.0, rotational_speed_rpm=8000.0,
    )
    return solve_compressor_stage(
        T01=288.15, P01=101_325.0, specific_work_required=30_000.0,
        mass_flow=50.0, cp=COMPRESSOR_CP, gamma=COMPRESSOR_GAMMA, design=design,
    )


def test_surge_margin_de_haller_positive_when_above_limit():
    assert surge_margin_de_haller(0.80, limit=0.72) > 0.0


def test_surge_margin_de_haller_negative_when_below_limit():
    assert surge_margin_de_haller(0.65, limit=0.72) < 0.0


def test_surge_margin_de_haller_matches_manual_calc():
    assert surge_margin_de_haller(0.80, limit=0.72) == pytest.approx((0.80 - 0.72) / 0.72 * 100.0)


def test_surge_margin_diffusion_factor_positive_when_below_limit():
    assert surge_margin_diffusion_factor(0.45, limit=0.6) > 0.0


def test_surge_margin_diffusion_factor_negative_when_above_limit():
    assert surge_margin_diffusion_factor(0.70, limit=0.6) < 0.0


def test_choke_margin_positive_when_subsonic():
    assert choke_margin_mach(0.85, limit=1.0) > 0.0


def test_choke_margin_negative_when_supersonic():
    assert choke_margin_mach(1.1, limit=1.0) < 0.0


def test_choke_margin_matches_manual_calc():
    assert choke_margin_mach(0.85, limit=1.0) == pytest.approx((1.0 - 0.85) / 1.0 * 100.0)


def test_assess_stage_reports_this_projects_known_borderline_de_haller(stage_result):
    # This validated design point's de Haller (~0.71) sits just under 0.72 --
    # already known to FAIL as a sanity check, so margin should be negative.
    df = diffusion_factor(stage_result.W1, stage_result.W2, stage_result.Wt1, stage_result.Wt2, solidity=1.3)
    assessment = assess_stage(stage_result, diffusion_factor_value=df)
    assert stage_result.de_haller < 0.72
    assert assessment.de_haller_margin_pct < 0.0
    assert not assessment.is_safe


def test_assess_stage_is_safe_when_all_three_margins_are_positive(stage_result):
    sigma = required_solidity_for_diffusion_factor(stage_result.W1, stage_result.W2, stage_result.Wt1, stage_result.Wt2, target_df=0.45)
    df = diffusion_factor(stage_result.W1, stage_result.W2, stage_result.Wt1, stage_result.Wt2, sigma)
    # Force a safe de Haller/choke scenario to isolate is_safe's AND logic.
    assessment = assess_stage(stage_result, diffusion_factor_value=df, de_haller_limit=0.60, choke_mach_limit=1.0)
    assert assessment.diffusion_factor_margin_pct > 0.0
    assert assessment.is_safe


@pytest.mark.parametrize("bad_limit", [0.0, -0.1])
def test_margin_functions_reject_non_positive_limit(bad_limit):
    with pytest.raises(ValueError):
        surge_margin_de_haller(0.8, limit=bad_limit)
    with pytest.raises(ValueError):
        surge_margin_diffusion_factor(0.5, limit=bad_limit)
    with pytest.raises(ValueError):
        choke_margin_mach(0.8, limit=bad_limit)
