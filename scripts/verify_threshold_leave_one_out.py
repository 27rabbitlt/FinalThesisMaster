#!/usr/bin/env python3
"""Exact audit of the threshold leave-one-out identity.

For every connected labelled graph on a depot and four clients, use its
shortest-path metric, every nontrivial half-integral threshold, and several
rational probability vectors.  The script checks, in Fraction arithmetic,

    a_t = r_t + I_t,
    E[i_t(B)] = r_t + J_t,
    a_t <= E[i_t(B) + ell_t^star(B)],
    a_t + r_t <= 2 E[k_t(B)].

The last inequality is only an exhaustive finite check, not a proof.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import combinations, permutations


N = 4
ROOT = N
PAIRS = tuple(combinations(range(N + 1), 2))
PROBABILITY_VECTORS = (
    (Fraction(1, 5),) * N,
    tuple(Fraction(x, 7) for x in (1, 2, 4, 6)),
    tuple(Fraction(x, 20) for x in (1, 6, 14, 19)),
)


def mask_probability(mask: int, probabilities: tuple[Fraction, ...]) -> Fraction:
    answer = Fraction(1)
    for vertex, probability in enumerate(probabilities):
        answer *= probability if mask & (1 << vertex) else 1 - probability
    return answer


def graph_metric(graph_mask: int) -> list[list[int]] | None:
    infinity = 100
    distance = [
        [0 if u == v else infinity for v in range(N + 1)]
        for u in range(N + 1)
    ]
    for index, (u, v) in enumerate(PAIRS):
        if graph_mask & (1 << index):
            distance[u][v] = distance[v][u] = 1
    for middle in range(N + 1):
        for u in range(N + 1):
            for v in range(N + 1):
                distance[u][v] = min(
                    distance[u][v], distance[u][middle] + distance[middle][v]
                )
    if any(distance[ROOT][v] >= infinity for v in range(N)):
        return None
    return distance


def projected_mask(
    guide: int,
    sample: int,
    distance: list[list[int]],
    client_only: bool = False,
) -> int:
    guide_centers = [v for v in range(N) if guide & (1 << v)]
    if client_only and not guide_centers:
        return 0
    centers = guide_centers if client_only else guide_centers + [ROOT]
    retained = 0
    for vertex in range(N):
        if sample & (1 << vertex):
            image = min(centers, key=lambda center: (distance[vertex][center], center))
            if image != ROOT:
                retained |= 1 << image
    return retained


def uncovered(vertex: int, target: int, distance: list[list[int]], threshold: Fraction) -> int:
    if distance[vertex][ROOT] <= threshold:
        return 0
    return int(
        all(
            not target & (1 << center) or distance[vertex][center] > threshold
            for center in range(N)
        )
    )


def bad_tour_edges(mask: int, distance: list[list[int]], threshold: Fraction) -> int:
    vertices = [v for v in range(N) if mask & (1 << v)]
    if not vertices:
        return 0
    best = 2 * (N + 1)
    for order in permutations(vertices):
        walk = (ROOT,) + order + (ROOT,)
        bad = sum(distance[u][v] > threshold for u, v in zip(walk, walk[1:]))
        best = min(best, bad)
    return best


def isolated(mask: int, distance: list[list[int]], threshold: Fraction) -> int:
    return sum(
        uncovered(vertex, mask ^ (1 << vertex), distance, threshold)
        for vertex in range(N)
        if mask & (1 << vertex)
    )


def tour_relevant_leaves(mask: int, distance: list[list[int]], threshold: Fraction) -> int:
    """Threshold-degree-one clients, suppressing a sole root-near client."""
    active_count = mask.bit_count()
    return sum(
        (
            int(distance[vertex][ROOT] <= threshold)
            + sum(
            bool(mask & (1 << other)) and distance[vertex][other] <= threshold
            for other in range(N)
            if other != vertex
        )
        == 1
        )
        and not (active_count == 1 and distance[vertex][ROOT] <= threshold)
        for vertex in range(N)
        if mask & (1 << vertex)
    )


def audit(
    distance: list[list[int]],
    threshold: Fraction,
    probabilities: tuple[Fraction, ...],
    client_only_projection: bool = False,
) -> None:
    weights = [mask_probability(mask, probabilities) for mask in range(1 << N)]
    projections = [
        [
            projected_mask(guide, sample, distance, client_only_projection)
            for sample in range(1 << N)
        ]
        for guide in range(1 << N)
    ]

    attachment = sum(
        weights[guide]
        * weights[sample]
        * sum(
            probabilities[v] * uncovered(v, projections[guide][sample], distance, threshold)
            for v in range(N)
        )
        for guide in range(1 << N)
        for sample in range(1 << N)
    )
    radius = sum(
        weights[guide]
        * sum(probabilities[v] * uncovered(v, guide, distance, threshold) for v in range(N))
        for guide in range(1 << N)
    )

    pivotal = Fraction(0)
    for v in range(N):
        other_bits = [u for u in range(N) if u != v]
        for guide in range(1 << N):
            for compact_sample in range(1 << (N - 1)):
                sample = sum(
                    (1 << u)
                    for index, u in enumerate(other_bits)
                    if compact_sample & (1 << index)
                )
                sample_probability = Fraction(1)
                for u in other_bits:
                    sample_probability *= (
                        probabilities[u] if sample & (1 << u) else 1 - probabilities[u]
                    )
                if (
                    uncovered(v, guide, distance, threshold) == 0
                    and uncovered(v, projections[guide][sample], distance, threshold) == 1
                ):
                    pivotal += (
                        probabilities[v]
                        * (1 - probabilities[v])
                        * weights[guide]
                        * sample_probability
                    )

    expected_isolated = sum(
        weights[mask] * isolated(mask, distance, threshold)
        for mask in range(1 << N)
    )
    expected_tour_relevant_leaves = sum(
        weights[mask] * tour_relevant_leaves(mask, distance, threshold)
        for mask in range(1 << N)
    )
    correction = Fraction(0)
    for v in range(N):
        no_other_neighbor = sum(
            weights[mask]
            for mask in range(1 << N)
            if not mask & (1 << v)
            and uncovered(v, mask, distance, threshold)
        ) / (1 - probabilities[v])
        correction += probabilities[v] ** 2 * no_other_neighbor

    expected_bad_edges = sum(
        weights[mask] * bad_tour_edges(mask, distance, threshold)
        for mask in range(1 << N)
    )

    assert attachment == radius + pivotal
    assert expected_isolated == radius + correction
    assert attachment <= expected_isolated + expected_tour_relevant_leaves
    assert attachment + radius <= 2 * expected_bad_edges
    if client_only_projection:
        # Bounded-size diagnostic only.  The analogous universal statement is
        # false on many-component sparse families even though every instance
        # in this four-client audit satisfies it.
        assert attachment <= expected_bad_edges
    for mask in range(1 << N):
        isolated_count = isolated(mask, distance, threshold)
        leaf_count = tour_relevant_leaves(mask, distance, threshold)
        bad_edges = bad_tour_edges(mask, distance, threshold)
        assert isolated_count + leaf_count <= 2 * bad_edges - isolated_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-only-projection", action="store_true")
    args = parser.parse_args()
    connected = 0
    audits = 0
    for graph_mask in range(1 << len(PAIRS)):
        distance = graph_metric(graph_mask)
        if distance is None:
            continue
        connected += 1
        diameter = max(max(row) for row in distance)
        for level in range(diameter):
            threshold = Fraction(2 * level + 1, 2)
            for probabilities in PROBABILITY_VECTORS:
                audit(
                    distance,
                    threshold,
                    probabilities,
                    client_only_projection=args.client_only_projection,
                )
                audits += 1
    mode = "client-only" if args.client_only_projection else "root-including"
    print(
        f"verified {audits} exact audits on {connected} connected labelled graphs "
        f"({mode})"
    )


if __name__ == "__main__":
    main()
