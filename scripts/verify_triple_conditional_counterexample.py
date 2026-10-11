#!/usr/bin/env python3
"""Exact strict-metric obstruction to conditioning the triple saving on B.

There are 39 sites and 21 labelled clients at every site.  Distances within
a site are 1/100 and all other nonzero distances (including depot distances)
are one.  Every client has probability 1/40.  We condition on a guide set B
containing exactly one client at every site.

For this B the rootful and rootless projected cores coincide at the site
level, and neither can improve the tour--facility objective.  Nevertheless
the realization-wise nearest-radius deficit charges are too small to prove
the desired factor-2.9 inequality conditionally on B.
"""

from fractions import Fraction


SITES = 39
LABELS_PER_SITE = 21
P = Fraction(1, 40)
EPSILON = Fraction(1, 100)


def objective_for_retained_sites(retained: int) -> Fraction:
    """Tour--facility objective for the projected core of a fixed guide B."""
    site_mass = LABELS_PER_SITE * P
    noncenter_mass = (LABELS_PER_SITE - 1) * P
    if retained == 0:
        return 2 * SITES * site_mass
    tour = retained + 1
    attachment = (
        retained * noncenter_mass * EPSILON
        + (SITES - retained) * site_mass
    )
    return tour + 2 * attachment


def main() -> None:
    guide_tour = Fraction(SITES + 1)
    guide_attachment = (
        SITES * (LABELS_PER_SITE - 1) * P * EPSILON
    )
    guide_objective = guide_tour + 2 * guide_attachment

    savings = [
        guide_objective - objective_for_retained_sites(retained)
        for retained in range(SITES + 1)
    ]
    assert all(value <= 0 for value in savings)
    assert savings[0] == -Fraction(14, 25)
    assert all(
        savings[retained] == -Fraction(SITES - retained, 25)
        for retained in range(1, SITES + 1)
    )

    # Every guide's nearest other center (or the depot) is at distance one.
    nearest_sum = Fraction(SITES)
    realized_deficit = guide_tour - nearest_sum
    high_probability_charge = SITES * P
    conditional_lhs = 2 * realized_deficit + 2 * high_probability_charge
    conditional_rhs = guide_tour / 10

    assert realized_deficit == 1
    assert high_probability_charge == Fraction(39, 40)
    assert conditional_lhs == Fraction(79, 20)
    assert conditional_rhs == 4
    assert conditional_lhs - conditional_rhs == -Fraction(1, 20)

    print("max projected-core saving =", max(Fraction(), *savings))
    print("2 conditional deficit charges =", conditional_lhs)
    print("tsp(B)/10 =", conditional_rhs)
    print("conditional gap =", conditional_lhs - conditional_rhs)


if __name__ == "__main__":
    main()
