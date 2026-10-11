#!/usr/bin/env python3
"""Exact small-graph audit of rootless two-sample compression.

For a nonempty guide set B, every vertex of the second sample C is sent to
its nearest *client* in B; the depot is excluded as a possible image.  For
B empty the compressed set is empty.  We test the scale inequality

    E u_t(A,T(B,C)) + E u_t(A,B) <= 2 E k_t(A),

where u_t(X,Y) is the probability-weighted number of vertices of X with no
threshold-neighbour in Y union the depot, and k_t(A) is the minimum number
of threshold-long edge occurrences in a depot tour through A.  All
probabilities and expectations are evaluated with Fraction arithmetic.

The graph metrics are exhaustive (all connected labelled unweighted graphs
on a depot and at most four clients).  The finite list of probability vectors
is a stress test, not an exhaustive proof over probabilities.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product


def mask_probability(mask: int, probabilities: tuple[Fraction, ...]) -> Fraction:
    answer = Fraction(1)
    for vertex, probability in enumerate(probabilities):
        answer *= probability if mask & (1 << vertex) else 1 - probability
    return answer


def graph_metric(n: int, graph_mask: int) -> list[list[int]] | None:
    root = n
    pairs = tuple(combinations(range(n + 1), 2))
    infinity = n + 2
    distance = [
        [0 if u == v else infinity for v in range(n + 1)]
        for u in range(n + 1)
    ]
    for index, (u, v) in enumerate(pairs):
        if graph_mask & (1 << index):
            distance[u][v] = distance[v][u] = 1
    for middle in range(n + 1):
        for u in range(n + 1):
            for v in range(n + 1):
                distance[u][v] = min(
                    distance[u][v], distance[u][middle] + distance[middle][v]
                )
    if any(distance[root][v] >= infinity for v in range(n)):
        return None
    return distance


def projected_mask_rootless(
    n: int, guide: int, sample: int, distance: list[list[int]]
) -> int:
    if guide == 0:
        return 0
    centers = [v for v in range(n) if guide & (1 << v)]
    retained = 0
    for vertex in range(n):
        if sample & (1 << vertex):
            image = min(centers, key=lambda center: (distance[vertex][center], center))
            retained |= 1 << image
    return retained


def uncovered(
    n: int,
    vertex: int,
    target: int,
    distance: list[list[int]],
    threshold: Fraction,
) -> int:
    root = n
    if distance[vertex][root] <= threshold:
        return 0
    return int(
        all(
            not target & (1 << center) or distance[vertex][center] > threshold
            for center in range(n)
        )
    )


def all_bad_tour_edges(
    n: int, distance: list[list[int]], threshold: Fraction
) -> list[int]:
    """Held--Karp minimum threshold-long edge multiplicity for every subset."""
    root = n
    infinity = 2 * (n + 1)
    answer = [0] * (1 << n)
    for mask in range(1, 1 << n):
        dp: dict[tuple[int, int], int] = {}
        for last in range(n):
            if mask & (1 << last):
                dp[(1 << last, last)] = int(distance[root][last] > threshold)
        for submask in range(1, 1 << n):
            if submask & ~mask or not submask & (submask - 1):
                continue
            for last in range(n):
                if not submask & (1 << last):
                    continue
                previous = submask ^ (1 << last)
                dp[(submask, last)] = min(
                    dp.get((previous, before), infinity)
                    + int(distance[before][last] > threshold)
                    for before in range(n)
                    if previous & (1 << before)
                )
        answer[mask] = min(
            dp[(mask, last)] + int(distance[last][root] > threshold)
            for last in range(n)
            if mask & (1 << last)
        )
    return answer


def probability_vectors(n: int) -> tuple[tuple[Fraction, ...], ...]:
    base = {
        (Fraction(1, 10),) * n,
        (Fraction(1, 2),) * n,
        (Fraction(9, 10),) * n,
        tuple(Fraction(1 + 2 * i, 2 * n + 2) for i in range(n)),
        tuple(Fraction(2 * n + 1 - 2 * i, 2 * n + 2) for i in range(n)),
    }
    # For up to three clients, also exhaust a heterogeneous three-point grid.
    if n <= 3:
        base.update(product((Fraction(1, 5), Fraction(1, 2), Fraction(4, 5)), repeat=n))
    return tuple(sorted(base))


def audit(
    n: int,
    graph_mask: int,
    distance: list[list[int]],
    threshold: Fraction,
    probabilities: tuple[Fraction, ...],
) -> tuple[Fraction, Fraction, Fraction]:
    weights = [mask_probability(mask, probabilities) for mask in range(1 << n)]
    bad = all_bad_tour_edges(n, distance, threshold)
    radius = sum(
        weights[guide]
        * sum(
            probabilities[v] * uncovered(n, v, guide, distance, threshold)
            for v in range(n)
        )
        for guide in range(1 << n)
    )
    attachment = Fraction(0)
    for guide in range(1 << n):
        for sample in range(1 << n):
            target = projected_mask_rootless(n, guide, sample, distance)
            attachment += weights[guide] * weights[sample] * sum(
                probabilities[v] * uncovered(n, v, target, distance, threshold)
                for v in range(n)
            )
    expected_bad = sum(weights[mask] * bad[mask] for mask in range(1 << n))
    if attachment + radius > 2 * expected_bad:
        raise AssertionError(
            "rootless scale inequality fails: "
            f"n={n}, graph_mask={graph_mask}, threshold={threshold}, "
            f"p={probabilities}, a={attachment}, r={radius}, Ek={expected_bad}, "
            f"gap={attachment + radius - 2 * expected_bad}, d={distance}"
        )
    return attachment, radius, expected_bad


def main() -> None:
    audits = 0
    graph_count = 0
    for n in range(1, 5):
        pair_count = (n + 1) * n // 2
        for graph_mask in range(1 << pair_count):
            distance = graph_metric(n, graph_mask)
            if distance is None:
                continue
            graph_count += 1
            diameter = max(max(row) for row in distance)
            for level in range(diameter):
                threshold = Fraction(2 * level + 1, 2)
                for probabilities in probability_vectors(n):
                    audit(n, graph_mask, distance, threshold, probabilities)
                    audits += 1
    print(
        f"verified {audits} exact rootless audits on {graph_count} connected "
        "labelled graph metrics (up to four clients)"
    )


if __name__ == "__main__":
    main()
