import math

import pytest

from boundary_conditions import (
    INLET,
    OUTLET,
    WALL_LO,
    WALL_MID,
    WALL_UP,
    CascadeBoundaryConditions,
    flow_direction_vector,
)


def test_flow_direction_vector_matches_manual_formula():
    d = flow_direction_vector(42.6413)
    assert d[0] == pytest.approx(math.cos(math.radians(42.6413)))
    assert d[1] == pytest.approx(math.sin(math.radians(42.6413)))
    assert d[2] == pytest.approx(0.0)


def test_flow_direction_vector_is_a_unit_vector():
    d = flow_direction_vector(30.0)
    assert math.hypot(d[0], d[1], d[2]) == pytest.approx(1.0)


def test_flow_direction_vector_at_zero_degrees_is_pure_axial():
    d = flow_direction_vector(0.0)
    assert d == pytest.approx((1.0, 0.0, 0.0))


def test_cascade_boundary_conditions_inlet_direction_matches_flow_direction_vector():
    bc = CascadeBoundaryConditions(
        inlet_total_temperature=1578.06, inlet_total_pressure=500000.0,
        inlet_flow_angle_deg=42.6413, outlet_static_pressure=300000.0,
    )
    assert bc.inlet_direction == flow_direction_vector(42.6413)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(inlet_total_temperature=0.0),
        dict(inlet_total_pressure=0.0),
        dict(outlet_static_pressure=0.0),
        dict(inlet_total_temperature=-10.0),
    ],
)
def test_cascade_boundary_conditions_rejects_invalid_inputs(overrides):
    base = dict(inlet_total_temperature=1578.06, inlet_total_pressure=500000.0, inlet_flow_angle_deg=42.6413, outlet_static_pressure=300000.0)
    base.update(overrides)
    with pytest.raises(ValueError):
        CascadeBoundaryConditions(**base)


def test_marker_names_are_distinct_strings():
    names = {INLET, OUTLET, WALL_MID, WALL_UP, WALL_LO}
    assert len(names) == 5
