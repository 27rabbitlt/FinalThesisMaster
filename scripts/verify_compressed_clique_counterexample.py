#!/usr/bin/env python3
"""Exact checks for a hard-regime counterexample to A1 <= P.

There are ``g`` disjoint client cliques of size ``h``.  Distinct clients in
one clique have distance one; all other distinct pairs, including every
client--depot pair, have distance two.  All clients activate with probability
``p``.  In the first-guide projection, a client whose clique contains no
guide is assigned to the depot; otherwise a non-guide is assigned to the
least-index guide in its clique.

The script verifies the closed forms by exhaustive enumeration on a small
instance, then checks a finite counterexample using exact ``Fraction``
arithmetic only.
"""

from __future__ import annotations

from fractions import Fraction


def subset_probability(mask: int, n: int, p: Fraction) -> Fraction:
    return p ** mask.bit_count() * (1 - p) ** (n - mask.bit_count())


def clique_tsp(mask: int, g: int, h: int) -> int:
    if mask == 0:
        return 0
    occupied = sum(
        bool(mask & (((1 << h) - 1) << (i * h))) for i in range(g)
    )
    return mask.bit_count() + occupied + 2


def projected_mask(guide: int, sample: int, g: int, h: int) -> int:
    retained = 0
    block_mask = (1 << h) - 1
    for i in range(g):
        shift = i * h
        local_guide = (guide >> shift) & block_mask
        local_sample = (sample >> shift) & block_mask
        if local_guide == 0:
            continue
        least = local_guide & -local_guide
        for j in range(h):
            bit = 1 << j
            if not local_sample & bit:
                continue
            retained |= (bit if local_guide & bit else least) << shift
    return retained


def formulas(
    g: int, h: int, p: Fraction
) -> tuple[Fraction, Fraction, Fraction, Fraction, Fraction]:
    z = 1 - p
    q = 1 - z**h
    # Expected number of retained guide points in one clique.
    mu = (
        q
        - z * (1 - p + p * p) ** h
        + z ** (2 * h + 1)
        + h * p * p
        - p * q
    )
    posterior = g * h * p + g * q + 2 * (1 - z ** (g * h))
    nearest = g * h * p * (z + z**h)
    nearest_other = g * h * p * (1 + z ** (h - 1))
    attachment = g * p * (2 * h - h * q * q - mu)
    core_tour = g * mu + g * q * q + 2 * (1 - (1 - q * q) ** g)
    return posterior, nearest, nearest_other, attachment, core_tour


def brute_force(
    g: int, h: int, p: Fraction
) -> tuple[Fraction, Fraction]:
    n = g * h
    core_tour = Fraction()
    attachment = Fraction()
    block_mask = (1 << h) - 1
    for guide in range(1 << n):
        guide_probability = subset_probability(guide, n, p)
        for sample in range(1 << n):
            weight = guide_probability * subset_probability(sample, n, p)
            retained = projected_mask(guide, sample, g, h)
            core_tour += weight * clique_tsp(retained, g, h)
            total_distance = 0
            for v in range(n):
                if retained & (1 << v):
                    continue
                block = (retained >> ((v // h) * h)) & block_mask
                total_distance += 1 if block else 2
            attachment += weight * p * total_distance
    return attachment, core_tour


def main() -> None:
    small = (2, 3, Fraction(1, 5))
    closed = formulas(*small)
    brute_attachment, brute_core = brute_force(*small)
    assert closed[3] == brute_attachment
    assert closed[4] == brute_core

    # 10,000 clients, but the symmetry-reduced formulas stay tiny and exact.
    g, h, p = 1000, 10, Fraction(1, 100)
    posterior, nearest, nearest_other, attachment, core_tour = formulas(g, h, p)
    deficit = posterior - nearest
    gamma = (posterior - core_tour) + 2 * (posterior - attachment)

    assert attachment > posterior
    assert attachment - nearest_other > posterior - nearest_other
    assert deficit < posterior / 20
    assert core_tour < 9 * posterior / 10
    assert gamma > posterior / 10

    # The displayed metric has ties.  The following explicit rational
    # perturbation makes every first-guide nearest point unique while keeping
    # the same projection cells.  Give local ranks a=1,...,h and set
    #   d(u,v)=1+eps(a_u+a_v)  in one clique,
    #   d(u,v)=2                in different cliques,
    #   d(u,r)=2-3eps.
    # It is a metric for eps<1/(2h), and differs from the displayed metric by
    # at most delta=2h*eps.  A tour objective moves by at most
    # (E|B|+Pr[B nonempty])*delta, and an assignment objective by at most
    # E|B|*delta.  These exact conservative bounds preserve all strict tests.
    epsilon = Fraction(1, 10**8)
    assert epsilon < Fraction(1, 2 * h)
    delta = 2 * h * epsilon
    expected_size = g * h * p
    tour_error = (expected_size + 1) * delta
    assignment_error = expected_size * delta
    assert attachment - posterior > assignment_error + tour_error
    assert (
        posterior / 20 - deficit
        > assignment_error + Fraction(19, 20) * tour_error
    )
    assert (
        9 * posterior / 10 - core_tour
        > tour_error + Fraction(9, 10) * tour_error
    )

    print("small exhaustive enumeration agrees with every closed form")
    print(f"A1/P       = {float(attachment / posterior):.15f}")
    print(f"D/P        = {float(deficit / posterior):.15f}")
    print(f"L/P        = {float(nearest_other / posterior):.15f}")
    print(f"M1/P       = {float(core_tour / posterior):.15f}")
    print(f"Gamma1/P   = {float(gamma / posterior):.15f}")
    print(f"(M1+2A1)/P= {float((core_tour + 2 * attachment) / posterior):.15f}")
    print("the explicit epsilon=10^-8 perturbation gives unique nearest points")
    print("all finite comparisons passed in exact Fraction arithmetic")


if __name__ == "__main__":
    main()
