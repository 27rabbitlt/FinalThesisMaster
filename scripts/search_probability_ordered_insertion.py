#!/usr/bin/env python3
"""Falsify probability-ordered cheapest insertion for symmetric stochastic TSP.

The candidate is deliberately parameter-free.  Vertices are processed in
nonincreasing activation probability.  Among the currently highest-probability
vertices, choose the vertex and tour gap with minimum metric insertion excess.
The final rooted cyclic order is evaluated by the exact stopped-product
formula.  We compare the candidate together with the classical P+2R bound to
2.9 P.

This is an exploratory checker, not a proof.
"""

from __future__ import annotations

import argparse
import math
import random
from functools import lru_cache


def floyd(weights: list[list[float]]) -> list[list[float]]:
    distance = [row[:] for row in weights]
    for pivot in range(len(distance)):
        for first in range(len(distance)):
            via = distance[first][pivot]
            for second in range(len(distance)):
                distance[first][second] = min(
                    distance[first][second], via + distance[pivot][second]
                )
    return distance


def random_graph_metric(client_count: int, rng: random.Random) -> list[list[float]]:
    count = client_count + 1
    infinity = 1.0e100
    weights = [
        [0.0 if first == second else infinity for second in range(count)]
        for first in range(count)
    ]
    for vertex in range(1, count):
        parent = rng.randrange(vertex)
        length = rng.randint(1, 40)
        weights[parent][vertex] = weights[vertex][parent] = length
    for first in range(count):
        for second in range(first + 1, count):
            if rng.random() < 0.45:
                length = rng.randint(1, 40)
                weights[first][second] = weights[second][first] = min(
                    weights[first][second], length
                )
    return floyd(weights)


def subset_probabilities(probabilities: list[float]) -> list[float]:
    values = [1.0]
    for probability in probabilities:
        values = [
            old * (1.0 - probability) for old in values
        ] + [old * probability for old in values]
    # The iterative layout above is not binary-mask order.  Recompute directly;
    # n is deliberately small in this falsifier.
    result = []
    for mask in range(1 << len(probabilities)):
        value = 1.0
        for index, probability in enumerate(probabilities):
            value *= probability if mask & (1 << index) else 1.0 - probability
        result.append(value)
    return result


def subset_tsp_values(distance: list[list[float]]) -> list[float]:
    client_count = len(distance) - 1

    @lru_cache(maxsize=None)
    def path(mask: int, last: int) -> float:
        bit = 1 << (last - 1)
        if mask == bit:
            return distance[0][last]
        previous_mask = mask ^ bit
        return min(
            path(previous_mask, previous) + distance[previous][last]
            for previous in range(1, client_count + 1)
            if previous_mask & (1 << (previous - 1))
        )

    values = [0.0] * (1 << client_count)
    for mask in range(1, 1 << client_count):
        values[mask] = min(
            path(mask, last) + distance[last][0]
            for last in range(1, client_count + 1)
            if mask & (1 << (last - 1))
        )
    return values


def stopped_cost(
    order: tuple[int, ...], probabilities: list[float], distance: list[list[float]]
) -> float:
    value = 0.0
    for first_position, first in enumerate(order):
        first_probability = probabilities[first - 1]
        prefix_empty = math.prod(
            1.0 - probabilities[order[index] - 1]
            for index in range(first_position)
        )
        suffix_empty = math.prod(
            1.0 - probabilities[order[index] - 1]
            for index in range(first_position + 1, len(order))
        )
        value += first_probability * (prefix_empty + suffix_empty) * distance[0][first]
        gap_empty = 1.0
        for second in order[first_position + 1 :]:
            value += (
                first_probability
                * probabilities[second - 1]
                * gap_empty
                * distance[first][second]
            )
            gap_empty *= 1.0 - probabilities[second - 1]
    return value


def probability_ordered_insertion(
    probabilities: list[float], distance: list[list[float]]
) -> tuple[int, ...]:
    remaining = set(range(1, len(probabilities) + 1))
    order: list[int] = []
    tolerance = 1.0e-12
    while remaining:
        level = max(probabilities[vertex - 1] for vertex in remaining)
        eligible = [
            vertex
            for vertex in remaining
            if abs(probabilities[vertex - 1] - level) <= tolerance
        ]
        best: tuple[float, int, int] | None = None
        for vertex in eligible:
            rooted = [0, *order, 0]
            for gap in range(len(order) + 1):
                left, right = rooted[gap], rooted[gap + 1]
                excess = (
                    distance[left][vertex]
                    + distance[vertex][right]
                    - distance[left][right]
                )
                candidate = (excess, vertex, gap)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        _, vertex, gap = best
        order.insert(gap, vertex)
        remaining.remove(vertex)
    return tuple(order)


def evaluate(distance: list[list[float]], probabilities: list[float]) -> dict[str, float]:
    subset_probability = subset_probabilities(probabilities)
    subset_optimum = subset_tsp_values(distance)
    posterior = sum(
        probability * optimum
        for probability, optimum in zip(subset_probability, subset_optimum)
    )
    radius = 0.0
    for mask, probability in enumerate(subset_probability):
        centers = [0] + [
            vertex
            for vertex in range(1, len(probabilities) + 1)
            if mask & (1 << (vertex - 1))
        ]
        radius += probability * sum(
            probabilities[vertex - 1]
            * min(distance[vertex][center] for center in centers)
            for vertex in range(1, len(probabilities) + 1)
        )
    order = probability_ordered_insertion(probabilities, distance)
    insertion = stopped_cost(order, probabilities, distance)
    baseline = posterior + 2.0 * radius
    return {
        "P": posterior,
        "R": radius,
        "D/P": (posterior - radius) / posterior,
        "W/P": insertion / posterior,
        "combined/P": min(insertion, baseline) / posterior,
        "order": order,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260907)
    arguments = parser.parse_args()
    rng = random.Random(arguments.seed)
    alphabet = (0.002, 0.01, 0.04, 0.1, 0.25, 0.5, 0.9, 0.999)
    worst: tuple[float, object] | None = None
    hard_worst: tuple[float, object] | None = None
    for trial in range(arguments.trials):
        distance = random_graph_metric(arguments.n, rng)
        profiles = [[probability] * arguments.n for probability in alphabet]
        profiles.extend(
            [rng.choice(alphabet) for _ in range(arguments.n)] for _ in range(8)
        )
        for probabilities in profiles:
            result = evaluate(distance, probabilities)
            record = (trial, probabilities, distance, result)
            if worst is None or result["combined/P"] > worst[0]:
                worst = (result["combined/P"], record)
                print("combined worst", result, "p", probabilities, flush=True)
            if result["D/P"] <= 0.05 and (
                hard_worst is None or result["W/P"] > hard_worst[0]
            ):
                hard_worst = (result["W/P"], record)
                print("hard worst", result, "p", probabilities, flush=True)
            if result["combined/P"] > 2.9 + 1.0e-9:
                print("COUNTEREXAMPLE", result, "p", probabilities)
                for row in distance:
                    print(row)
                return
    print("FINAL combined worst", worst)
    print("FINAL hard worst", hard_worst)


if __name__ == "__main__":
    main()
