#!/usr/bin/env python3
"""Exact counterexample to the rootless-compression Gamma layer inequality.

When the guide set B is nonempty, every C vertex is projected to its nearest
client in B (the depot is excluded); for B empty the core T is empty.  This
checks, in Fraction arithmetic, a four-client threshold metric for which

    E[k_t(T)] + 2 a_t - 3 E[k_t(A)] > 0.

It therefore also refutes the proposed inequality after adding r_t/e.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations


N = 4
ROOT = N
P = Fraction(99, 100)
THRESHOLD = Fraction(3, 2)
DISTANCE = (
    (0, 1, 2, 1, 1),
    (1, 0, 1, 1, 1),
    (2, 1, 0, 1, 1),
    (1, 1, 1, 0, 2),
    (1, 1, 1, 2, 0),
)


def bits(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def probability(mask: int) -> Fraction:
    return P ** mask.bit_count() * (1 - P) ** (N - mask.bit_count())


def bad_tour_edges(mask: int) -> int:
    vertices = tuple(bits(mask))
    if not vertices:
        return 0
    return min(
        sum(
            DISTANCE[u][v] > THRESHOLD
            for u, v in zip((ROOT,) + order, order + (ROOT,))
        )
        for order in permutations(vertices)
    )


def rootless_core(guide: int, sample: int) -> int:
    centers = tuple(bits(guide))
    if not centers:
        return 0
    retained = 0
    for vertex in bits(sample):
        image = min(centers, key=lambda center: (DISTANCE[vertex][center], center))
        retained |= 1 << image
    return retained


def uncovered(vertex: int, target: int) -> int:
    if DISTANCE[vertex][ROOT] <= THRESHOLD:
        return 0
    return int(
        all(DISTANCE[vertex][center] > THRESHOLD for center in bits(target))
    )


def main() -> None:
    weights = tuple(probability(mask) for mask in range(1 << N))
    tours = tuple(bad_tour_edges(mask) for mask in range(1 << N))
    expected_tour = sum(
        (weights[mask] * tours[mask] for mask in range(1 << N)),
        Fraction(),
    )
    expected_core_tour = Fraction()
    attachment = Fraction()
    for guide in range(1 << N):
        for sample in range(1 << N):
            weight = weights[guide] * weights[sample]
            core = rootless_core(guide, sample)
            expected_core_tour += weight * tours[core]
            attachment += weight * sum(
                P * uncovered(vertex, core) for vertex in range(N)
            )
    radius = sum(
        weights[guide]
        * sum(P * uncovered(vertex, guide) for vertex in range(N))
        for guide in range(1 << N)
    )
    rational_gap = expected_core_tour + 2 * attachment - 3 * expected_tour

    assert attachment == P * (2 * (1 - P) ** N - (1 - P) ** (2 * N))
    assert radius == P * (1 - P) ** N
    assert rational_gap > 0
    print("E k_t(A) =", expected_tour, float(expected_tour))
    print("E k_t(T*) =", expected_core_tour, float(expected_core_tour))
    print("a*_t =", attachment, float(attachment))
    print("r_t =", radius, float(radius))
    print("E k_t(T*) + 2a*_t - 3E k_t(A) =", rational_gap, float(rational_gap))
    print("Adding r_t/e only increases this strict violation.")


if __name__ == "__main__":
    main()
