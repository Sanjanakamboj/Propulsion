import pytest

from sanity import check, format_sanity_report


def test_check_passes_within_bounds():
    c = check("Test Metric", 0.5, (0.0, 1.0), unit="-")
    assert c.passed


def test_check_fails_below_low_bound():
    c = check("Test Metric", -0.1, (0.0, 1.0), unit="-")
    assert not c.passed


def test_check_fails_above_high_bound():
    c = check("Test Metric", 1.1, (0.0, 1.0), unit="-")
    assert not c.passed


def test_check_handles_one_sided_bounds():
    assert check("Metric", 5.0, (None, 10.0)).passed
    assert not check("Metric", 15.0, (None, 10.0)).passed
    assert check("Metric", 5.0, (1.0, None)).passed
    assert not check("Metric", 0.5, (1.0, None)).passed


def test_format_sanity_report_all_pass():
    checks = [check("A", 0.5, (0.0, 1.0)), check("B", 5.0, (None, 10.0))]
    report = format_sanity_report("Title", checks)
    assert "ALL CHECKS PASSED" in report
    assert "FAIL" not in report


def test_format_sanity_report_some_fail():
    checks = [check("A", 1.5, (0.0, 1.0)), check("B", 5.0, (None, 10.0))]
    report = format_sanity_report("Title", checks)
    assert "SOME CHECKS FAILED" in report
    assert "FAIL" in report
    assert "PASS" in report
