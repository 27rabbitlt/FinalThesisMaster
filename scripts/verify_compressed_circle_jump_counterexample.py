#!/usr/bin/env python3
"""Certify a structural counterexample to two compressed-core sublemmas.

The counterexample is an asymptotic sequence of finite rational instances.
Put Bernoulli-p vertices on a long discrete circle, add one always-present
depot, and let x be circular distance measured in units of p.  For

    h = 8 log(2)

use, for distinct points, the metric

    d_eps(x) = 1 + eps*x + 1{x > h}.

Taking p -> 0, circle length -> infinity, and eps -> 0 produces two
independent rate-one Poisson processes B,C.  Nearest-guide cells are the
ordinary one-dimensional Voronoi cells.  Their occupied nuclei form T.

This script performs the exact rational part of the proof.  It expands a
finite list of disjoint local Poisson/Voronoi events and proves

    M_1 - N_T > D
    M_1 - L_T > 2D

in the limit.  Thus both proposed integrated residual inequalities are
false.  The direct strong core-drop conjecture is *not* refuted by this
family; a certified coarse bound leaves it substantial slack.  The same
family does refute two tempting routes to that conjecture: a pointwise
threshold version and a lower certificate based on first-order TSP deletion
influences.

No floating-point value is used by an assertion.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from math import comb, factorial


Polynomial = dict[int, Fraction]


def add(left: Polynomial, right: Polynomial) -> Polynomial:
    answer = dict(left)
    for degree, coefficient in right.items():
        answer[degree] = answer.get(degree, Fraction(0)) + coefficient
    return {degree: value for degree, value in answer.items() if value}


def scale(poly: Polynomial, factor: Fraction) -> Polynomial:
    return {
        degree: factor * coefficient
        for degree, coefficient in poly.items()
        if factor * coefficient
    }


def multiply_truncated(
    left: list[Fraction], right: list[Fraction], degree: int
) -> list[Fraction]:
    answer = [Fraction(0) for _ in range(degree + 1)]
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            if i + j <= degree:
                answer[i + j] += x * y
    return answer


def tail_at_eight_log_two(rates: list[Fraction]) -> Polynomial:
    """Return the polynomial in ell=log(2) for a hypoexponential tail.

    If independent coordinates have unnormalised densities exp(-r_i x),
    this returns

      integral_{sum x_i > 8 ell} product_i exp(-r_i x_i) dx.

    Every rate here is a half-integer, so exp(-8 r_i ell)=2^(-8 r_i)
    is rational.  Partial fractions are computed exactly.
    """

    multiplicities = Counter(rates)
    answer: Polynomial = {}
    for rate, multiplicity in multiplicities.items():
        # Taylor coefficients of
        #   product_{q != rate}(s+q)^(-multiplicity(q))
        # at s=-rate, through degree multiplicity-1.
        series = [Fraction(1)] + [
            Fraction(0) for _ in range(multiplicity - 1)
        ]
        for other, other_multiplicity in multiplicities.items():
            if other == rate:
                continue
            difference = other - rate
            factor = [
                Fraction(
                    (-1) ** degree
                    * comb(other_multiplicity + degree - 1, degree),
                    1,
                )
                / difference ** (other_multiplicity + degree)
                for degree in range(multiplicity)
            ]
            series = multiply_truncated(
                series, factor, multiplicity - 1
            )

        exponential = Fraction(2) ** (-8 * rate)
        assert (-8 * rate).denominator == 1
        for taylor_degree, partial_fraction_coefficient in enumerate(
            series
        ):
            pole_order = multiplicity - taylor_degree
            # The tail of
            # A/(s+r)^k is
            # A exp(-rh) sum_{j=0}^{k-1} h^j/(j! r^(k-j)).
            for power in range(pole_order):
                coefficient = (
                    partial_fraction_coefficient
                    * exponential
                    * Fraction(8) ** power
                    / (
                        factorial(power)
                        * rate ** (pole_order - power)
                    )
                )
                answer[power] = answer.get(power, Fraction(0)) + coefficient
    return {degree: value for degree, value in answer.items() if value}


def gap_category(a: int, require_short_successor: bool) -> Polynomial:
    """Probability polynomial for one retained-gap category.

    Index the Poisson guide nuclei by the integers.  The previous retained
    nucleus is -a-1, nuclei -a,...,-1 have empty second-sample cells, and
    nucleus 0 is retained.  Their total intervening guide gap must exceed h.

    If require_short_successor is true, nucleus 1 must also be retained and
    its guide gap from 0 must be at most h.  This makes nucleus 0 the start
    of a nontrivial short component.
    """

    empty_cells = set(range(-a, 0))
    endpoint_cells = [-a - 1, 0]
    if require_short_successor:
        endpoint_cells.append(1)
    number_of_endpoint_terms = 1 << len(endpoint_cells)
    answer: Polynomial = {}

    for mask in range(number_of_endpoint_terms):
        # Expand each occupied-cell factor 1-exp(-cell length).
        exponential_cells = set(empty_cells)
        for index, cell in enumerate(endpoint_cells):
            if mask & (1 << index):
                exponential_cells.add(cell)
        sign = Fraction((-1) ** mask.bit_count())

        last_gap = 1 if require_short_successor else 0
        rates = {
            gap: Fraction(1)
            + Fraction(1, 2)
            * sum(
                adjacent_cell in exponential_cells
                for adjacent_cell in (gap, gap + 1)
            )
            for gap in range(-a - 2, last_gap + 1)
        }

        # The two gaps outside the retained endpoints are unrestricted.
        outside_factor = (
            Fraction(1, 1)
            / rates[-a - 2]
            / rates[last_gap]
        )
        left_tail = tail_at_eight_log_two(
            [rates[gap] for gap in range(-a - 1, 0)]
        )
        term = scale(left_tail, sign * outside_factor)

        if require_short_successor:
            # Integrate the next guide gap over [0,h].
            right_rate = rates[0]
            right_factor = (
                1 - Fraction(2) ** (-8 * right_rate)
            ) / right_rate
            term = scale(term, right_factor)
        answer = add(answer, term)
    return answer


def evaluate(poly: Polynomial, value: Fraction) -> Fraction:
    return sum(
        coefficient * value**degree
        for degree, coefficient in poly.items()
    )


def main() -> None:
    # Seven consecutive-retained-gap categories give enough strict slack.
    long_gap_lower: Polynomial = {}
    nontrivial_component_lower: Polynomial = {}
    for a in range(7):
        long_gap_lower = add(long_gap_lower, gap_category(a, False))
        nontrivial_component_lower = add(
            nontrivial_component_lower, gap_category(a, True)
        )

    expected_long_polynomial = {
        6: Fraction(1, 810),
        5: Fraction(1, 144),
        4: Fraction(127, 6912),
        3: Fraction(277, 9216),
        2: Fraction(4735, 147456),
        1: Fraction(13511, 131072),
        0: -Fraction(197689, 2097152),
    }
    expected_nontrivial_polynomial = {
        6: Fraction(4367, 7077888),
        5: Fraction(68231, 18874368),
        4: Fraction(2950235, 301989888),
        3: Fraction(19521515, 1207959552),
        2: Fraction(37320665, 2147483648),
        1: Fraction(974110105, 17179869184),
        0: -Fraction(127009181215, 2473901162496),
    }
    assert long_gap_lower == expected_long_polynomial
    assert nontrivial_component_lower == expected_nontrivial_polynomial

    # log(2)=2 sum_{k>=0} 1/((2k+1)3^(2k+1)).  Six positive terms certify
    # the rational lower bound below.
    log_two_lower = Fraction(693147, 10**6)
    atanh_partial = sum(
        Fraction(2, (2 * k + 1) * 3 ** (2 * k + 1))
        for k in range(6)
    )
    assert atanh_partial > log_two_lower

    long_gap_bound = evaluate(long_gap_lower, log_two_lower)
    nontrivial_bound = evaluate(
        nontrivial_component_lower, log_two_lower
    )

    # For h=8 log 2, a=exp(-h)=1/256.  The original nontrivial-component
    # density is a(1-a).  Original isolated guides have density a^2.
    a = Fraction(1, 256)
    original_residual_density = a * (1 - a)
    original_isolated_density = a**2

    # M-N has limiting density at least nontrivial_bound.
    assert nontrivial_bound > original_residual_density

    # M-L_T has density at least long_gap_bound-a^2.  This beats 2D.
    assert (
        long_gap_bound - original_isolated_density
        > 2 * original_residual_density
    )

    # The family does not refute the direct core-drop conjecture.  For
    # a>=7, discard endpoint-occupancy and long-gap requirements.  The
    # chance that a specified run of a cells is empty is
    # (4/9)2^(-(a-1)); its summed tail is 1/72.
    tail_upper = Fraction(1, 72)
    log_two_upper = Fraction(7, 10)
    # e^(7/10)>1+7/10+(7/10)^2/2+(7/10)^3/6>2.
    assert (
        1
        + log_two_upper
        + log_two_upper**2 / 2
        + log_two_upper**3 / 6
        > 2
    )
    assert evaluate(long_gap_lower, log_two_upper) < Fraction(1, 100)
    retained_intensity = Fraction(5, 9)
    core_long_gap_upper = Fraction(1, 100) + tail_upper
    core_drop_lower = (
        1 + a - retained_intensity - core_long_gap_upper
    )
    # Since e>8/3 and 1-a+2a^2<1, the conjectured right side is <3/8.
    assert 1 - a + 2 * a**2 < 1
    assert core_drop_lower > Fraction(3, 8)

    # First-order deletion influence is not a sufficient certificate for
    # the desired right side.  If the guide intensity is lambda, the
    # original-tour density is
    #
    #   lambda (1 + exp(-lambda h)).
    #
    # Hence the limiting sum of add/delete marginals at lambda=1 is
    # I_del = 1 + a(1-h).  Meanwhile 2R-P = 1-a+2a^2.  Their difference is
    # a(h-2+2a), which is strictly positive at h=8 log 2.
    influence_gap_lower = a * (8 * log_two_lower - 2 + 2 * a)
    assert influence_gap_lower > 0

    # Nor can the strong core drop be proved separately at every distance
    # threshold.  At the jump threshold, the original bad-edge, radius,
    # and compressed bad-edge densities are respectively a, a^2, and at
    # least long_gap_bound.  Therefore the candidate
    #
    #   k(B)-k(T) >= (2 r-k(B))/e
    #
    # fails.  Use 1/e<3/8 to certify the strict comparison rationally.
    scale_core_violation_lower = (
        long_gap_bound
        - a
        - Fraction(3, 8) * (a - 2 * a**2)
    )
    assert scale_core_violation_lower > 0

    # The repaired compression-versus-influence bridge itself is harmless
    # here: I_del<1, while the certified core drop is >3/8>1/e.
    assert 1 + a * (1 - 8 * log_two_lower) < 1

    print("circle-jump asymptotic counterexample certified")
    print(
        "long-core-gap lower =",
        float(long_gap_bound),
        "(needed >",
        float(2 * a - a**2),
        ")",
    )
    print(
        "nontrivial-core-component lower =",
        float(nontrivial_bound),
        "(original =",
        float(original_residual_density),
        ")",
    )
    print(
        "liminf (M_1-L_T-2D) density >",
        float(
            long_gap_bound
            - original_isolated_density
            - 2 * original_residual_density
        ),
    )
    print(
        "liminf (M_1-N_T-D) density >",
        float(nontrivial_bound - original_residual_density),
    )
    print(
        "certified core-drop lower >",
        float(core_drop_lower),
        "> 3/8, so the strong core-drop target survives this family",
    )
    print(
        "lim [(2R-P)-I_del] density >",
        float(influence_gap_lower),
    )
    print(
        "scale-wise strong-core violation density >",
        float(scale_core_violation_lower),
    )


if __name__ == "__main__":
    main()
