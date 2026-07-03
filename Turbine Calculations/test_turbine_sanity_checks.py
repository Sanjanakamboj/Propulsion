import pytest

from turbine import TurbineStageDesignInputs, solve_turbine_stage
from turbine_sanity_checks import format_sanity_report, turbine_stage_sanity_check

TURBINE_GAMMA, TURBINE_R = 1.33, 287.0
TURBINE_CP = TURBINE_GAMMA * TURBINE_R / (TURBINE_GAMMA - 1.0)


@pytest.fixture
def turbine_result():
    # Matches the validated notebook example -- known to fail M2 (0.6996 < 0.70)
    # and pass everything else, per the notebook's own printed sanity check.
    design = TurbineStageDesignInputs(
        stage_loading=1.8, flow_coefficient=0.5, degree_of_reaction=0.35,
        blade_speed_limit=350.0, rotational_speed_rpm=3000.0,
    )
    return solve_turbine_stage(
        T01=1679.21, P01=2_056_992.49, specific_work_required=219_270.0,
        mass_flow=750.0, cp=TURBINE_CP, gamma=TURBINE_GAMMA, design=design,
    )


def test_turbine_sanity_check_matches_notebook_pass_fail(turbine_result):
    checks = turbine_stage_sanity_check(turbine_result)
    by_name = {c.name: c for c in checks}

    # Notebook's own printed sanity check reported M2 = 0.6996 as FAIL (LOW)
    # against a 0.70-0.85 limit, and every other row as PASS.
    assert not by_name["Absolute Mach M2"].passed
    assert by_name["Relative flow angle beta2"].passed
    assert by_name["Relative Mach Mw2"].passed
    assert by_name["Relative Mach Mw3"].passed
    assert by_name["Relative flow angle beta3"].passed
    assert by_name["Total turning (beta2+beta3)"].passed
    assert by_name["Exit swirl alpha3"].passed
    assert by_name["AN^2"].passed


def test_turbine_sanity_check_values_match_result_fields(turbine_result):
    checks = turbine_stage_sanity_check(turbine_result)
    by_name = {c.name: c for c in checks}
    assert by_name["Absolute Mach M2"].value == pytest.approx(turbine_result.M2)
    assert by_name["AN^2"].value == pytest.approx(turbine_result.an2)
    assert by_name["Total turning (beta2+beta3)"].value == pytest.approx(
        turbine_result.beta2_deg + turbine_result.beta3_deg
    )


def test_format_sanity_report_contains_fail_marker_when_a_check_fails(turbine_result):
    checks = turbine_stage_sanity_check(turbine_result)
    report = format_sanity_report("Turbine Stage 1", checks)
    assert "FAIL" in report
    assert "SOME CHECKS FAILED" in report
