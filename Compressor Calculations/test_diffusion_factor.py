import pytest

from compressor import CompressorStageDesignInputs, solve_compressor_stage
from diffusion_factor import diffusion_factor, required_solidity_for_diffusion_factor

# The project's validated representative compressor stage (see compressor.py's
# own test fixtures): W1=245.93, W2=174.59, Wt1=197.62, Wt2=95.15 m/s.
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


def test_diffusion_factor_matches_manual_formula():
    # DF = 1 - W2/W1 + |Wt1-Wt2| / (2*sigma*W1)
    df = diffusion_factor(W1=245.93, W2=174.59, Wt1=197.62, Wt2=95.15, solidity=1.3)
    expected = 1.0 - 174.59 / 245.93 + abs(197.62 - 95.15) / (2.0 * 1.3 * 245.93)
    assert df == pytest.approx(expected)


def test_required_solidity_is_the_correct_inverse():
    sigma = required_solidity_for_diffusion_factor(W1=245.93, W2=174.59, Wt1=197.62, Wt2=95.15, target_df=0.45)
    assert diffusion_factor(245.93, 174.59, 197.62, 95.15, sigma) == pytest.approx(0.45, rel=1e-9)


def test_lower_solidity_gives_higher_diffusion_factor():
    # Fewer/thinner blades (lower solidity) load each blade more.
    df_loose = diffusion_factor(W1=245.93, W2=174.59, Wt1=197.62, Wt2=95.15, solidity=1.0)
    df_tight = diffusion_factor(W1=245.93, W2=174.59, Wt1=197.62, Wt2=95.15, solidity=2.0)
    assert df_loose > df_tight


def test_required_solidity_for_validated_stage_gives_sane_axial_compressor_range(stage_result):
    # This is the actual root-cause fix: Zweifel's criterion gave solidity ~3.6
    # (pitch/chord ~0.28, self-intersecting blade) for this exact stage.
    # Lieblein's diffusion factor at a typical DF=0.45 design target should
    # land in the normal axial-compressor solidity range (~0.8-1.5).
    sigma = required_solidity_for_diffusion_factor(
        stage_result.W1, stage_result.W2, stage_result.Wt1, stage_result.Wt2, target_df=0.45,
    )
    assert 0.8 < sigma < 1.5


def test_target_df_at_or_below_floor_raises():
    # floor = 1 - W2/W1 = 1 - 174.59/245.93 ~ 0.290
    with pytest.raises(ValueError):
        required_solidity_for_diffusion_factor(W1=245.93, W2=174.59, Wt1=197.62, Wt2=95.15, target_df=0.29)


@pytest.mark.parametrize("bad_W1", [0.0, -10.0])
def test_diffusion_factor_rejects_non_positive_W1(bad_W1):
    with pytest.raises(ValueError):
        diffusion_factor(W1=bad_W1, W2=100.0, Wt1=50.0, Wt2=20.0, solidity=1.0)


@pytest.mark.parametrize("bad_solidity", [0.0, -1.0])
def test_diffusion_factor_rejects_non_positive_solidity(bad_solidity):
    with pytest.raises(ValueError):
        diffusion_factor(W1=200.0, W2=150.0, Wt1=100.0, Wt2=50.0, solidity=bad_solidity)
