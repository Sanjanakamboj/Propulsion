import pytest

from mission import FlightEnvelopePoint, generate_flight_envelope


def test_flight_envelope_point_rejects_negative_mach():
    with pytest.raises(ValueError):
        FlightEnvelopePoint(mach=-0.1, altitude_m=1000.0)


def test_flight_envelope_point_rejects_negative_altitude():
    with pytest.raises(ValueError):
        FlightEnvelopePoint(mach=0.5, altitude_m=-1.0)


def test_generate_flight_envelope_covers_every_combination():
    points = generate_flight_envelope([0.5, 0.8], [0.0, 5000.0, 11000.0])
    assert len(points) == 6
    combos = {(p.mach, p.altitude_m) for p in points}
    assert combos == {(0.5, 0.0), (0.8, 0.0), (0.5, 5000.0), (0.8, 5000.0), (0.5, 11000.0), (0.8, 11000.0)}


def test_generate_flight_envelope_is_altitude_major_order():
    points = generate_flight_envelope([0.5, 0.8], [0.0, 5000.0])
    assert [(p.mach, p.altitude_m) for p in points] == [(0.5, 0.0), (0.8, 0.0), (0.5, 5000.0), (0.8, 5000.0)]


def test_generate_flight_envelope_handles_single_values():
    points = generate_flight_envelope([0.8], [11000.0])
    assert len(points) == 1
    assert points[0] == FlightEnvelopePoint(mach=0.8, altitude_m=11000.0)
