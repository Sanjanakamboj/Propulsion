import pytest

from choking import assess_stage, choke_margin_mach, choke_status
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


def test_choke_margin_matches_manual_calc():
    assert choke_margin_mach(0.70, limit=1.0) == pytest.approx((1.0 - 0.70) / 1.0 * 100.0)


def test_choke_margin_zero_at_exactly_sonic():
    assert choke_margin_mach(1.0, limit=1.0) == pytest.approx(0.0)


def test_choke_margin_negative_when_supersonic():
    assert choke_margin_mach(1.1, limit=1.0) < 0.0


@pytest.mark.parametrize(
    "mach, expected",
    [
        (0.70, "subsonic"),
        (0.99, "choked"),
        (1.0, "choked"),
        (1.01, "choked"),
        (1.10, "supersonic"),
    ],
)
def test_choke_status_thresholds(mach, expected):
    assert choke_status(mach, choked_tolerance=0.02) == expected


def test_assess_stage_reports_subsonic_for_this_validated_design_point(stage_result):
    # M2 ~ 0.70, Mw3 ~ 0.67 for this stage -- both comfortably subsonic.
    assessment = assess_stage(stage_result)
    assert assessment.nozzle_status == "subsonic"
    assert assessment.rotor_status == "subsonic"
    assert assessment.is_physically_valid


def test_is_physically_valid_false_only_when_supersonic():
    from choking import TurbineChokeAssessment
    valid = TurbineChokeAssessment(nozzle_margin_pct=5.0, nozzle_status="choked", rotor_margin_pct=10.0, rotor_status="subsonic")
    invalid = TurbineChokeAssessment(nozzle_margin_pct=-5.0, nozzle_status="supersonic", rotor_margin_pct=10.0, rotor_status="subsonic")
    assert valid.is_physically_valid
    assert not invalid.is_physically_valid


def test_choke_margin_rejects_non_positive_limit():
    with pytest.raises(ValueError):
        choke_margin_mach(0.8, limit=0.0)
