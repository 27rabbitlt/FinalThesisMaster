#!/usr/bin/env python3
"""Exact obstruction to a scale-by-scale proof of M_1-N_T <= D.

The integrated inequality remains a live conjecture.  What fails is its
most direct threshold strengthening

    E[k_t(T)-i_t(T)]
      <= E[k_t(B)-i_t(B)] + E sum_{b in I_t(B)} p_b.

Here k_t(S) is the minimum number of tour edges longer than t and i_t(S)
is the number of nondepot vertices isolated in the threshold graph.
The witness is the unit path on six clients and the depot, with p=1/2
and 3<t<4.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations


CLIENTS = 6
ROOT = CLIENTS
P = Fraction(1, 2)
SHORT_DISTANCE = 3  # Any threshold strictly between 3 and 4.


def distance(u: int, v: int) -> int:
    return abs(u - v)


def long_edge(u: int, v: int) -> int:
    return int(distance(u, v) > SHORT_DISTANCE)


def min_long_tour_edges(mask: int) -> int:
    vertices = [v for v in range(CLIENTS) if mask & (1 << v)]
    if not vertices:
        return 0
    return min(
        long_edge(ROOT, order[0])
        + sum(long_edge(u, v) for u, v in zip(order, order[1:]))
        + long_edge(order[-1], ROOT)
        for order in permutations(vertices)
    )


def isolated(mask: int) -> set[int]:
    vertices = [v for v in range(CLIENTS) if mask & (1 << v)]
    return {
        v
        for v in vertices
        if distance(v, ROOT) > SHORT_DISTANCE
        and all(
            u == v or distance(u, v) > SHORT_DISTANCE
            for u in vertices
        )
    }


def main() -> None:
    k = [min_long_tour_edges(mask) for mask in range(1 << CLIENTS)]

    original_residual = Fraction(0)
    self_deficit_layer = Fraction(0)
    compressed_residual = Fraction(0)
    retained_original_isolates = Fraction(0)
    compressed_isolates = Fraction(0)

    # Every guide set has probability 2^-6.
    guide_probability = Fraction(1, 2**CLIENTS)
    for guide in range(1 << CLIENTS):
        centers = [v for v in range(CLIENTS) if guide & (1 << v)]
        available = [*centers, ROOT]
        image = [
            v
            if guide & (1 << v)
            else min(available, key=lambda u: (distance(v, u), u))
            for v in range(CLIENTS)
        ]
        original_isolates = isolated(guide)
        original_residual += guide_probability * (
            k[guide] - len(original_isolates)
        )
        self_deficit_layer += guide_probability * P * len(
            original_isolates
        )

        occupancy = {
            b: 1
            - (1 - P)
            ** sum(image[v] == b for v in range(CLIENTS))
            for b in centers
        }
        retained = guide
        while True:
            conditional_probability = guide_probability
            for b in centers:
                conditional_probability *= (
                    occupancy[b]
                    if retained & (1 << b)
                    else 1 - occupancy[b]
                )

            retained_isolates = isolated(retained)
            compressed_residual += conditional_probability * (
                k[retained] - len(retained_isolates)
            )
            compressed_isolates += (
                conditional_probability * len(retained_isolates)
            )
            retained_original_isolates += conditional_probability * sum(
                b in original_isolates
                for b in centers
                if retained & (1 << b)
            )

            if retained == 0:
                break
            retained = (retained - 1) & guide

    obstruction = (
        compressed_residual
        - original_residual
        - self_deficit_layer
    )
    assert original_residual == Fraction(21, 32)
    assert self_deficit_layer == Fraction(7, 128)
    assert compressed_residual == Fraction(3235, 4096)
    assert obstruction == Fraction(323, 4096) > 0

    # The analogous layer target for N_T-L_T is not refuted here.
    new_isolates = compressed_isolates - retained_original_isolates
    inflation_slack = (
        new_isolates - original_residual - self_deficit_layer
    )
    assert compressed_isolates == Fraction(485, 2048)
    assert retained_original_isolates == Fraction(205, 2048)
    assert inflation_slack == -Fraction(147, 256)

    print("exact compressed parity-layer obstruction verified")
    print(f"E[k_t(B)-i_t(B)]={original_residual}")
    print(f"J_t={self_deficit_layer}")
    print(f"E[k_t(T)-i_t(T)]={compressed_residual}")
    print(f"violating gap={obstruction}={float(obstruction):.12f}")
    print(f"N_T-L_T layer slack={inflation_slack}")


if __name__ == "__main__":
    main()
