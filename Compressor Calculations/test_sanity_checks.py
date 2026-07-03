import pytest

from compressor import CompressorStageDesignInputs, solve_compressor_stage
from sanity_checks import compressor_stage_sanity_check, format_sanity_report

COMPRESSOR_GAMMA, COMPRESSOR_R = 1.4, 287.0
COMPRESSOR_CP = COMPRESSOR_GAMMA * COMPRESSOR_R / (COMPRESSOR_GAMMA - 1.0)


@pytest.fixture
def compressor_result():
    design = CompressorStageDesignInputs(
        stage_loading=0.35, flow_coefficient=0.5, degree_of_reaction=0.5,
        blade_speed_limit=350.0, rotational_speed_rpm=8000.0,
    )
    return solve_compressor_stage(
        T01=288.15, P01=101_325.0, specific_work_required=30_000.0,
        mass_flow=50.0, cp=COMPRESSOR_CP, gamma=COMPRESSOR_GAMMA, design=design,
    )


def test_compressor_sanity_check_de_haller_borderline_fails(compressor_result):
    # This design point's de Haller (~0.71) sits just under the classic 0.72
    # stall-margin threshold -- should FAIL by default.
    checks = compressor_stage_sanity_check(compressor_result)
    by_name = {c.name: c for c in checks}
    assert by_name["de Haller (W2/W1)"].value < 0.72
    assert not by_name["de Haller (W2/W1)"].passed


def test_compressor_sanity_check_custom_limits_override_defaults(compressor_result):
    # Loosen the de Haller limit and confirm the same design now passes.
    checks = compressor_stage_sanity_check(compressor_result, limits={"de_haller": (0.60, None)})
    by_name = {c.name: c for c in checks}
    assert by_name["de Haller (W2/W1)"].passed


def test_format_sanity_report_all_pass_when_limits_loosened(compressor_result):
    checks = compressor_stage_sanity_check(compressor_result, limits={"de_haller": (0.0, None)})
    report = format_sanity_report("Compressor Stage 1", checks)
    assert "ALL CHECKS PASSED" in report
