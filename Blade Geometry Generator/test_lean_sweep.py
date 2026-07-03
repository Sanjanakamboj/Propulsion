import math

import pytest

from lean import linear_lean_offset, make_linear_lean_offset
from sweep import linear_sweep_offset, make_linear_sweep_offset


def test_linear_lean_offset_is_zero_at_hub():
    assert linear_lean_offset(0.0, span_height=0.2, lean_angle_deg=15.0) == pytest.approx(0.0)


def test_linear_lean_offset_matches_manual_formula_at_tip():
    offset = linear_lean_offset(1.0, span_height=0.2, lean_angle_deg=15.0)
    assert offset == pytest.approx(0.2 * math.tan(math.radians(15.0)))


def test_linear_lean_offset_scales_with_span_fraction():
    half = linear_lean_offset(0.5, span_height=0.2, lean_angle_deg=15.0)
    full = linear_lean_offset(1.0, span_height=0.2, lean_angle_deg=15.0)
    assert half == pytest.approx(full / 2.0)


def test_make_linear_lean_offset_returns_a_working_callable():
    offset_fn = make_linear_lean_offset(span_height=0.2, lean_angle_deg=15.0)
    assert offset_fn(0.0) == pytest.approx(0.0)
    assert offset_fn(1.0) == pytest.approx(linear_lean_offset(1.0, 0.2, 15.0))


def test_linear_lean_offset_rejects_span_fraction_outside_zero_one():
    with pytest.raises(ValueError):
        linear_lean_offset(-0.1, span_height=0.2, lean_angle_deg=15.0)
    with pytest.raises(ValueError):
        linear_lean_offset(1.1, span_height=0.2, lean_angle_deg=15.0)


def test_linear_sweep_offset_is_zero_at_hub():
    assert linear_sweep_offset(0.0, span_height=0.2, sweep_angle_deg=10.0) == pytest.approx(0.0)


def test_linear_sweep_offset_matches_manual_formula_at_tip():
    offset = linear_sweep_offset(1.0, span_height=0.2, sweep_angle_deg=10.0)
    assert offset == pytest.approx(0.2 * math.tan(math.radians(10.0)))


def test_make_linear_sweep_offset_returns_a_working_callable():
    offset_fn = make_linear_sweep_offset(span_height=0.2, sweep_angle_deg=10.0)
    assert offset_fn(1.0) == pytest.approx(linear_sweep_offset(1.0, 0.2, 10.0))


def test_linear_sweep_offset_rejects_negative_span_height():
    with pytest.raises(ValueError):
        linear_sweep_offset(0.5, span_height=-0.1, sweep_angle_deg=10.0)
