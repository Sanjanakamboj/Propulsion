"""Lieblein diffusion factor -- the compressor-specific blade-loading
criterion, replacing Zweifel's tangential-loading coefficient for
decelerating (compressor) cascades.

Zweifel's criterion (used for turbine rotors in Blade Geometry Generator's
rotor_blade_design.py) is derived for an ACCELERATING cascade and, applied
to a compressor's much smaller flow turning, predicts a pitch far too tight
for the blade's own thickness -- confirmed to produce a self-intersecting
(negative-throat) 2D blade section when tried. The diffusion factor is the
standard compressor design criterion instead:

    DF = 1 - W2/W1 + |Wt1 - Wt2| / (2 * sigma * W1)

where sigma = chord / pitch (solidity), W1/W2 are the relative velocities
at rotor inlet/exit, and Wt1/Wt2 their tangential components. Keeping
DF below ~0.6 (a long-standing empirical stall-margin limit from
Lieblein's original cascade data) sets the minimum solidity a blade row
needs for a given amount of diffusion.
"""

DEFAULT_MAX_DIFFUSION_FACTOR = 0.6  # conventional stall-margin limit (Lieblein)


def diffusion_factor(W1: float, W2: float, Wt1: float, Wt2: float, solidity: float) -> float:
    """Forward calculation: DF for a given solidity (chord / pitch)."""
    if W1 <= 0.0:
        raise ValueError("W1 must be > 0")
    if solidity <= 0.0:
        raise ValueError("solidity must be > 0")
    delta_Wt = abs(Wt1 - Wt2)
    return 1.0 - W2 / W1 + delta_Wt / (2.0 * solidity * W1)


def required_solidity_for_diffusion_factor(W1: float, W2: float, Wt1: float, Wt2: float, target_df: float = 0.45) -> float:
    """Inverse calculation: the solidity (chord / pitch) needed to hit
    target_df, given the stage's relative velocities. target_df must be
    strictly greater than (1 - W2/W1) -- that's the DF's floor as
    solidity -> infinity, so anything at or below it is unachievable at
    any finite solidity."""
    if W1 <= 0.0:
        raise ValueError("W1 must be > 0")
    floor = 1.0 - W2 / W1
    if target_df <= floor:
        raise ValueError(f"target_df ({target_df}) must be > 1 - W2/W1 ({floor:.4f}), the DF floor as solidity -> infinity")
    delta_Wt = abs(Wt1 - Wt2)
    return delta_Wt / (2.0 * W1 * (target_df - floor))
