import pytest

from choking import assess_choking, choke_margin_pct, choke_status, nozzle_pressure_ratio

GAMMA = 1.333
# Validated Brayton Cycle Analysis cruise design point (choked).
P0_IN, P_AMBIENT = 271982.75895383774, 22632.040095007793


def test_nozzle_pressure_ratio_matches_manual_calc():
    assert nozzle_pressure_ratio(P0_IN, P_AMBIENT) == pytest.approx(P0_IN / P_AMBIENT)


def test_choke_margin_negative_for_the_validated_choked_design_point():
    assert choke_margin_pct(P0_IN, P_AMBIENT, GAMMA) < 0.0


def test_choke_margin_positive_for_a_low_pressure_ratio():
    assert choke_margin_pct(P0_in=30000.0, P_ambient=22632.0, gamma=GAMMA) > 0.0


@pytest.mark.parametrize(
    "npr_factor, expected",
    [
        (0.5, "unchoked"),
        (1.0, "choked"),
        (1.01, "choked"),
        (5.0, "supersonic_at_throat"),
    ],
)
def test_choke_status_thresholds(npr_factor, expected):
    from exit_conditions import critical_pressure_ratio

    critical = critical_pressure_ratio(GAMMA)
    P0_in = critical * npr_factor * 22632.0
    assert choke_status(P0_in, P_ambient=22632.0, gamma=GAMMA, tolerance=0.02) == expected


def test_assess_choking_bundles_consistent_fields():
    assessment = assess_choking(P0_IN, P_AMBIENT, GAMMA)
    assert assessment.nozzle_pressure_ratio == pytest.approx(P0_IN / P_AMBIENT)
    assert assessment.status == "supersonic_at_throat"
    assert assessment.margin_pct < 0.0


def test_nozzle_pressure_ratio_rejects_non_positive_ambient_pressure():
    with pytest.raises(ValueError):
        nozzle_pressure_ratio(P0_IN, 0.0)
