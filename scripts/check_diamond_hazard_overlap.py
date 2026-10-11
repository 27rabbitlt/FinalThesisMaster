#!/usr/bin/env python3
"""Exact rational check of a first overlapping-cycle hazard stress test.

The graph starts from one unit edge and replaces every edge twice by a
diamond (two internally disjoint two-edge paths, with each new edge half as
long).  One endpoint is the depot and the other eleven vertices are clients.
Every client is active with probability 7/10.  Both the posterior TSP value
and the optimal rooted hazard-tree value are computed by exact subset DPs.

This is a finite diagnostic certificate, not a claim for all diamond graphs.
"""

from __future__ import annotations

from fractions import Fraction


Q = Fraction


def diamond_metric(depth: int) -> list[list[Q]]:
    edges: list[tuple[int, int, Q]] = [(0, 1, Q(1))]
    vertex_count = 2
    for _ in range(depth):
        replacement: list[tuple[int, int, Q]] = []
        for u, v, length in edges:
            a, b = vertex_count, vertex_count + 1
            vertex_count += 2
            replacement.extend(
                [
                    (u, a, length / 2),
                    (a, v, length / 2),
                    (u, b, length / 2),
                    (b, v, length / 2),
                ]
            )
        edges = replacement

    # The exact DPs use the final index as the depot.
    client_count = vertex_count - 1
    relabel = {0: client_count}
    relabel.update({v: v - 1 for v in range(1, vertex_count)})
    distance: list[list[Q | None]] = [
        [Q(0) if i == j else None for j in range(vertex_count)]
        for i in range(vertex_count)
    ]
    for old_u, old_v, length in edges:
        u, v = relabel[old_u], relabel[old_v]
        distance[u][v] = distance[v][u] = length

    for k in range(vertex_count):
        for i in range(vertex_count):
            if distance[i][k] is None:
                continue
            for j in range(vertex_count):
                if distance[k][j] is None:
                    continue
                candidate = distance[i][k] + distance[k][j]
                if distance[i][j] is None or candidate < distance[i][j]:
                    distance[i][j] = candidate
    assert all(entry is not None for row in distance for entry in row)
    return [[entry for entry in row] for row in distance]  # type: ignore[misc]


def subset_tsp_costs(distance: list[list[Q]]) -> list[Q]:
    client_count = len(distance) - 1
    depot = client_count
    size = 1 << client_count
    path: list[list[Q | None]] = [[None] * client_count for _ in range(size)]
    tours = [Q(0)] * size
    for v in range(client_count):
        path[1 << v][v] = distance[depot][v]
    for mask in range(1, size):
        for v in range(client_count):
            if not mask & (1 << v):
                continue
            previous = mask ^ (1 << v)
            if previous:
                path[mask][v] = min(
                    path[previous][u] + distance[u][v]  # type: ignore[operator]
                    for u in range(client_count)
                    if previous & (1 << u)
                )
        tours[mask] = min(
            path[mask][v] + distance[v][depot]  # type: ignore[operator]
            for v in range(client_count)
            if mask & (1 << v)
        )
    return tours


def posterior_value(distance: list[list[Q]], probability: Q) -> Q:
    client_count = len(distance) - 1
    answer = Q(0)
    for mask, cost in enumerate(subset_tsp_costs(distance)):
        active = mask.bit_count()
        answer += (
            probability**active
            * (1 - probability) ** (client_count - active)
            * cost
        )
    return answer


def optimal_hazard(distance: list[list[Q]], probability: Q) -> Q:
    """Exact rooted-tree partition recurrence from the structural memo."""
    client_count = len(distance) - 1
    depot = client_count
    size = 1 << client_count
    occupancy = [Q(0)] * size
    for mask in range(1, size):
        occupancy[mask] = 1 - (1 - probability) ** mask.bit_count()

    value: list[list[Q | None]] = [
        [None] * size for _ in range(client_count + 1)
    ]
    for root in range(client_count + 1):
        value[root][0] = Q(0)

    for cardinality in range(1, client_count + 1):
        for mask in range(1, size):
            if mask.bit_count() != cardinality:
                continue
            anchor = mask & -mask
            remainder = mask ^ anchor
            for root in range(client_count + 1):
                if root < client_count and mask & (1 << root):
                    continue
                best: Q | None = None
                submask = remainder
                while True:
                    branch = submask | anchor
                    rest = mask ^ branch
                    vertices = branch
                    while vertices:
                        bit = vertices & -vertices
                        child = bit.bit_length() - 1
                        candidate = (
                            occupancy[branch] * distance[root][child]
                            + value[child][branch ^ bit]  # type: ignore[operator]
                            + value[root][rest]  # type: ignore[operator]
                        )
                        if best is None or candidate < best:
                            best = candidate
                        vertices ^= bit
                    if submask == 0:
                        break
                    submask = (submask - 1) & remainder
                value[root][mask] = best
    result = value[depot][size - 1]
    assert result is not None
    return result


def main() -> None:
    distance = diamond_metric(depth=2)
    probability = Q(7, 10)
    posterior = posterior_value(distance, probability)
    hazard = optimal_hazard(distance, probability)

    assert len(distance) == 12
    assert posterior == Q(29_456_713_153, 10_000_000_000)
    assert hazard == Q(906_437, 400_000)
    assert hazard < posterior

    print(f"posterior = {posterior} = {float(posterior):.12f}")
    print(f"hazard = {hazard} = {float(hazard):.12f}")
    print(f"ratio = {float(hazard / posterior):.12f}")


if __name__ == "__main__":
    main()
