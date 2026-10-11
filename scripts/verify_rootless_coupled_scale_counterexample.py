#!/usr/bin/env python3
"""Exact counterexample to a coupled rootless-compression scale inequality.

Take the five-vertex unit-edge path

    x -- a -- r -- b -- y

with the middle vertex r as depot, the other four vertices as clients, common
activation probability 9/10, and threshold 7/2.  Only the leaf pair x,y is
farther apart than the threshold; every client is threshold-near the depot.
Consequently both uncovered terms vanish.  Nevertheless rootless nearest-B
compression makes a long-edge leaf-pair core more than three times as often
as it occurs in an activation sample.  Thus

    3 E k_t(A) - E k_t(T*) - 2 a*_t - c r_t < 0

for every coefficient c, including c=1/e.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations


N = 4
ROOT = 4
PROBABILITY = Fraction(9, 10)
THRESHOLD = Fraction(7, 2)

# Client order is a,b,y,x, so the unit path is x-a-r-b-y.
DISTANCE = (
    (0, 2, 3, 1, 1),
    (2, 0, 1, 3, 1),
    (3, 1, 0, 4, 2),
    (1, 3, 4, 0, 2),
    (1, 1, 2, 2, 0),
)


def mask_probability(mask: int) -> Fraction:
    return PROBABILITY ** mask.bit_count() * (1 - PROBABILITY) ** (N - mask.bit_count())


def rootless_projection(guide: int, sample: int) -> int:
    if guide == 0:
        return 0
    centers = [v for v in range(N) if guide & (1 << v)]
    target = 0
    for v in range(N):
        if sample & (1 << v):
            image = min(centers, key=lambda b: (DISTANCE[v][b], b))
            target |= 1 << image
    return target


def long_tour_edges(mask: int) -> int:
    vertices = [v for v in range(N) if mask & (1 << v)]
    if not vertices:
        return 0
    return min(
        sum(
            DISTANCE[u][v] > THRESHOLD
            for u, v in zip((ROOT,) + order, order + (ROOT,))
        )
        for order in permutations(vertices)
    )


def main() -> None:
    weights = [mask_probability(mask) for mask in range(1 << N)]
    defects = [long_tour_edges(mask) for mask in range(1 << N)]
    posterior_defect = sum(weights[mask] * defects[mask] for mask in range(1 << N))
    compressed_defect = sum(
        weights[guide] * weights[sample] * defects[rootless_projection(guide, sample)]
        for guide in range(1 << N)
        for sample in range(1 << N)
    )

    # Every client is within threshold of the depot, so both uncovered terms
    # are identically zero for every realization.
    attachment = Fraction(0)
    radius = Fraction(0)
    coupled = 3 * posterior_defect - compressed_defect - 2 * attachment

    assert posterior_defect == Fraction(81, 10000)
    assert compressed_defect == Fraction(6561, 250000)
    assert coupled == -Fraction(243, 125000)
    assert coupled < 0

    print("E[k_t(A)]               =", posterior_defect, float(posterior_defect))
    print("E[k_t(T*)]              =", compressed_defect, float(compressed_defect))
    print("a*_t = r_t              =", 0)
    print("coupled left-hand side  =", coupled, float(coupled))


if __name__ == "__main__":
    main()
