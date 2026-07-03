import pytest

from combustion import (
    STOICHIOMETRIC_FAR_KEROSENE,
    air_mass_flow_from_fuel,
    assess_combustion,
    combustion_regime,
    equivalence_ratio,
    far_from_equivalence_ratio,
)

# The validated Brayton Cycle Analysis design point's FAR.
FAR = 0.02794540232049333


def test_equivalence_ratio_matches_manual_calc():
    phi = equivalence_ratio(FAR, STOICHIOMETRIC_FAR_KEROSENE)
    assert phi == pytest.approx(FAR / STOICHIOMETRIC_FAR_KEROSENE)


def test_far_from_equivalence_ratio_is_the_correct_inverse():
    phi = equivalence_ratio(FAR, STOICHIOMETRIC_FAR_KEROSENE)
    recovered_far = far_from_equivalence_ratio(phi, STOICHIOMETRIC_FAR_KEROSENE)
    assert recovered_far == pytest.approx(FAR)


def test_real_turbojet_design_point_runs_lean():
    # ~0.41 phi for this cruise design point -- well below 1, consistent
    # with a gas turbine's heavily-diluted overall combustion.
    phi = equivalence_ratio(FAR, STOICHIOMETRIC_FAR_KEROSENE)
    assert phi < 1.0
    assert combustion_regime(phi) == "lean"


@pytest.mark.parametrize(
    "phi, expected",
    [
        (0.5, "lean"),
        (0.99, "stoichiometric"),
        (1.0, "stoichiometric"),
        (1.01, "stoichiometric"),
        (1.5, "rich"),
    ],
)
def test_combustion_regime_thresholds(phi, expected):
    assert combustion_regime(phi, tolerance=0.02) == expected


def test_air_mass_flow_from_fuel_matches_manual_calc():
    mdot_air = air_mass_flow_from_fuel(mdot_fuel=1.5, far=FAR)
    assert mdot_air == pytest.approx(1.5 / FAR)


def test_assess_combustion_bundles_consistent_fields():
    state = assess_combustion(FAR)
    assert state.far == pytest.approx(FAR)
    assert state.equivalence_ratio == pytest.approx(FAR / STOICHIOMETRIC_FAR_KEROSENE)
    assert state.regime == "lean"


def test_equivalence_ratio_rejects_non_positive_stoichiometric_far():
    with pytest.raises(ValueError):
        equivalence_ratio(FAR, far_stoichiometric=0.0)


def test_air_mass_flow_from_fuel_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        air_mass_flow_from_fuel(mdot_fuel=0.0, far=FAR)
    with pytest.raises(ValueError):
        air_mass_flow_from_fuel(mdot_fuel=1.0, far=0.0)
