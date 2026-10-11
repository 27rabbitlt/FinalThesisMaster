#!/usr/bin/env python3
"""Exact small-instance search for a closest-active-ancestor tree certificate.

For a rooted spanning tree T and active set A, every active client is linked
to its closest active ancestor in T.  A DFS order of T costs at most twice
this adjusted tree.  Therefore a universal bound

    min_T E[CA_T(A)] <= 1.45 E[TSP(A)]

would imply the desired 2.9 theorem.  This script enumerates every labelled
tree through Prüfer codes and searches for finite counterexamples.  It is an
exploratory falsifier, not a proof.
"""

from __future__ import annotations

import argparse
import itertools
import random
from functools import lru_cache


def prufer_edges(code: tuple[int, ...], count: int) -> list[tuple[int, int]]:
    degree = [1] * count
    for vertex in code:
        degree[vertex] += 1
    edges: list[tuple[int, int]] = []
    for vertex in code:
        leaf = next(index for index, value in enumerate(degree) if value == 1)
        edges.append((leaf, vertex))
        degree[leaf] -= 1
        degree[vertex] -= 1
    leaves = [index for index, value in enumerate(degree) if value == 1]
    assert len(leaves) == 2
    edges.append((leaves[0], leaves[1]))
    return edges


def rooted_parents(edges: list[tuple[int, int]], count: int) -> tuple[int, ...]:
    adjacency = [[] for _ in range(count)]
    for first, second in edges:
        adjacency[first].append(second)
        adjacency[second].append(first)
    parent = [-1] * count
    parent[0] = 0
    stack = [0]
    while stack:
        vertex = stack.pop()
        for neighbor in adjacency[vertex]:
            if parent[neighbor] == -1:
                parent[neighbor] = vertex
                stack.append(neighbor)
    assert all(value >= 0 for value in parent)
    return tuple(parent)


@lru_cache(maxsize=None)
def all_rooted_trees(count: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        rooted_parents(prufer_edges(code, count), count)
        for code in itertools.product(range(count), repeat=count - 2)
    )


def floyd(weights: list[list[int]]) -> list[list[int]]:
    distance = [row[:] for row in weights]
    for pivot in range(len(distance)):
        for first in range(len(distance)):
            for second in range(len(distance)):
                distance[first][second] = min(
                    distance[first][second],
                    distance[first][pivot] + distance[pivot][second],
                )
    return distance


def random_graph_metric(client_count: int, rng: random.Random) -> list[list[int]]:
    count = client_count + 1
    infinity = 10**9
    weights = [
        [0 if first == second else infinity for second in range(count)]
        for first in range(count)
    ]
    for vertex in range(1, count):
        parent = rng.randrange(vertex)
        length = rng.randint(1, 30)
        weights[parent][vertex] = weights[vertex][parent] = length
    for first in range(count):
        for second in range(first + 1, count):
            if rng.random() < 0.45:
                length = rng.randint(1, 30)
                weights[first][second] = weights[second][first] = min(
                    weights[first][second], length
                )
    return floyd(weights)


def probabilities(probability: list[float]) -> list[float]:
    result = []
    for mask in range(1 << len(probability)):
        value = 1.0
        for index, p_value in enumerate(probability):
            value *= p_value if mask & (1 << index) else 1.0 - p_value
        result.append(value)
    return result


def subset_tsp(distance: list[list[int]]) -> list[int]:
    client_count = len(distance) - 1

    @lru_cache(maxsize=None)
    def path(mask: int, last: int) -> int:
        bit = 1 << (last - 1)
        if mask == bit:
            return distance[0][last]
        previous_mask = mask ^ bit
        return min(
            path(previous_mask, previous) + distance[previous][last]
            for previous in range(1, client_count + 1)
            if previous_mask & (1 << (previous - 1))
        )

    result = [0] * (1 << client_count)
    for mask in range(1, 1 << client_count):
        result[mask] = min(
            path(mask, last) + distance[last][0]
            for last in range(1, client_count + 1)
            if mask & (1 << (last - 1))
        )
    return result


def closest_ancestor_scenario(
    parent: tuple[int, ...], mask: int, distance: list[list[int]]
) -> int:
    value = 0
    for vertex in range(1, len(parent)):
        if not mask & (1 << (vertex - 1)):
            continue
        ancestor = parent[vertex]
        while ancestor and not mask & (1 << (ancestor - 1)):
            ancestor = parent[ancestor]
        value += distance[vertex][ancestor]
    return value


def evaluate(distance: list[list[int]], p_value: list[float]) -> tuple[float, float, tuple[int, ...]]:
    scenario_probability = probabilities(p_value)
    posterior = sum(
        weight * cost
        for weight, cost in zip(scenario_probability, subset_tsp(distance))
    )
    best = float("inf")
    best_parent: tuple[int, ...] | None = None
    for parent in all_rooted_trees(len(distance)):
        value = sum(
            weight * closest_ancestor_scenario(parent, mask, distance)
            for mask, weight in enumerate(scenario_probability)
        )
        if value < best:
            best = value
            best_parent = parent
    assert best_parent is not None
    return posterior, best, best_parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260907)
    arguments = parser.parse_args()
    rng = random.Random(arguments.seed)
    alphabet = (0.01, 0.04, 0.1, 0.25, 0.5, 0.9, 0.99)
    worst = None
    for trial in range(arguments.trials):
        distance = random_graph_metric(arguments.n, rng)
        profiles = [[value] * arguments.n for value in alphabet]
        profiles.extend(
            [rng.choice(alphabet) for _ in range(arguments.n)] for _ in range(5)
        )
        for p_value in profiles:
            posterior, closest_ancestor, parent = evaluate(distance, p_value)
            ratio = closest_ancestor / posterior
            if worst is None or ratio > worst[0]:
                worst = (ratio, trial, p_value, distance, parent)
                print("worst", ratio, "p", p_value, "parent", parent, flush=True)
            if ratio > 1.45 + 1.0e-10:
                print("COUNTEREXAMPLE", worst)
                return
    print("FINAL", worst)


if __name__ == "__main__":
    main()
