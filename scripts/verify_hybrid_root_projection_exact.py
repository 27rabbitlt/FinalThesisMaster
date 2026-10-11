#!/usr/bin/env python3
"""Exact four-client audit of the pathwise rooted/rootless hybrid.

For each auxiliary outcome (B,C), the construction chooses the cheaper of
the depot-including and client-only projected cores under

    phi(S) = tsp(S) + 2 sum_v p_v d(v,S union {r}).

Every arithmetic comparison below uses ``Fraction``.  Passing this bounded
audit is falsification evidence, not a universal proof.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations

from verify_threshold_leave_one_out import (
    N,
    ROOT,
    graph_metric,
    mask_probability,
    projected_mask,
)


PROBABILITIES = (
    *((Fraction(x, 100),) * N for x in (1, 5, 20, 50, 80, 95, 99)),
    tuple(Fraction(x, 7) for x in (1, 2, 4, 6)),
    tuple(Fraction(x, 20) for x in (1, 6, 14, 19)),
    tuple(Fraction(x, 100) for x in (1, 17, 83, 99)),
)


def tsp_cost(mask: int, distance: list[list[int]]) -> int:
    vertices = tuple(v for v in range(N) if mask & (1 << v))
    if not vertices:
        return 0
    return min(
        sum(distance[u][v] for u, v in zip((ROOT,) + order, order + (ROOT,)))
        for order in permutations(vertices)
    )


def audit(distance: list[list[int]], probabilities: tuple[Fraction, ...]):
    weights = [mask_probability(mask, probabilities) for mask in range(1 << N)]
    costs = [tsp_cost(mask, distance) for mask in range(1 << N)]
    attachments = [
        sum(
            probabilities[v]
            * min(
                [distance[v][ROOT]]
                + [distance[v][u] for u in range(N) if mask & (1 << u)]
            )
            for v in range(N)
        )
        for mask in range(1 << N)
    ]
    phi = [Fraction(costs[mask]) + 2 * attachments[mask] for mask in range(1 << N)]
    posterior = sum(weights[mask] * costs[mask] for mask in range(1 << N))
    radius = sum(
        weights[guide]
        * sum(
            probabilities[v]
            * min(
                [distance[v][ROOT]]
                + [distance[v][u] for u in range(N) if guide & (1 << u)]
            )
            for v in range(N)
        )
        for guide in range(1 << N)
    )

    rooted = Fraction()
    rootless = Fraction()
    hybrid = Fraction()
    three_way = Fraction()
    for guide in range(1 << N):
        for second in range(1 << N):
            weight = weights[guide] * weights[second]
            rooted_value = phi[projected_mask(guide, second, distance, False)]
            rootless_value = phi[projected_mask(guide, second, distance, True)]
            rooted += weight * rooted_value
            rootless += weight * rootless_value
            hybrid += weight * min(rooted_value, rootless_value)
            three_way += weight * min(phi[guide], rooted_value, rootless_value)

    assert hybrid <= min(rooted, rootless)
    assert 19 * hybrid + 2 * radius <= 57 * posterior
    assert 19 * (rooted + rootless) + 4 * radius <= 114 * posterior
    # Coarse P-scale target: Gamma_hyb >= P/5 - 2(P-R).
    assert 5 * hybrid + 10 * radius <= 24 * posterior
    assert 5 * (rooted + rootless) + 20 * radius <= 48 * posterior
    assert 10 * three_way <= 29 * posterior
    return posterior, radius, rooted, rootless, hybrid, three_way


def main() -> None:
    connected = audits = 0
    worst_hybrid = None
    worst_average_gamma = None
    largest_pathwise_gain = None
    worst_coarse_gamma = None
    weakest_three_way_potential = None
    for graph_mask in range(1 << 10):
        distance = graph_metric(graph_mask)
        if distance is None:
            continue
        connected += 1
        for probabilities in PROBABILITIES:
            posterior, radius, rooted, rootless, hybrid, three_way = audit(
                distance, probabilities
            )
            hybrid_factor = hybrid / posterior
            average_gamma_gap = (
                (rooted + rootless) / 2 + 2 * radius / 19 - 3 * posterior
            ) / posterior
            pathwise_gain = (min(rooted, rootless) - hybrid) / posterior
            coarse_gamma_gap = (5 * hybrid + 10 * radius - 24 * posterior) / (
                5 * posterior
            )
            three_way_potential = 3 - three_way / posterior
            context = (graph_mask, probabilities, distance)
            if worst_hybrid is None or hybrid_factor > worst_hybrid[0]:
                worst_hybrid = (hybrid_factor, context)
            if (
                worst_average_gamma is None
                or average_gamma_gap > worst_average_gamma[0]
            ):
                worst_average_gamma = (average_gamma_gap, context)
            if (
                largest_pathwise_gain is None
                or pathwise_gain > largest_pathwise_gain[0]
            ):
                largest_pathwise_gain = (pathwise_gain, context)
            if worst_coarse_gamma is None or coarse_gamma_gap > worst_coarse_gamma[0]:
                worst_coarse_gamma = (coarse_gamma_gap, context)
            if (
                weakest_three_way_potential is None
                or three_way_potential < weakest_three_way_potential[0]
            ):
                weakest_three_way_potential = (three_way_potential, context)
            audits += 1
    print(f"verified {audits} exact hybrid audits on {connected} graph metrics")
    print("worst hybrid factor", worst_hybrid)
    print("worst averaged 2R/19 gap", worst_average_gamma)
    print("largest pathwise gain", largest_pathwise_gain)
    print("worst Gamma_hyb >= P/5-2D gap", worst_coarse_gamma)
    print("weakest three-way potential", weakest_three_way_potential)


if __name__ == "__main__":
    main()
