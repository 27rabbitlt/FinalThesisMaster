#!/usr/bin/env python3
"""Exact small-instance search for the rooted open-path reduction.

For a full client order pi, Q(pi) is the expected length after shortcutting
inactive clients, starting at the depot but without returning to it.  Closing
the route adds at most E[max_{v in A} d(r,v)] <= OPT_post/2.  Consequently

    min_pi Q(pi) <= 12/5 OPT_post

would imply OPT_prior <= 29/10 OPT_post.  This script searches for finite
counterexamples by exhaustive order enumeration.  It is not a proof.
"""

from __future__ import annotations

import argparse
import itertools
import random
from functools import lru_cache


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
    result = []
    for mask in range(1 << len(probabilities)):
        value = 1.0
        for index, probability in enumerate(probabilities):
            value *= probability if mask & (1 << index) else 1.0 - probability
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

    values = [0] * (1 << client_count)
    for mask in range(1, 1 << client_count):
        values[mask] = min(
            path(mask, last) + distance[last][0]
            for last in range(1, client_count + 1)
            if mask & (1 << (last - 1))
        )
    return values


def open_cost(
    order: tuple[int, ...], probabilities: list[float], distance: list[list[int]]
) -> float:
    value = 0.0
    prefix_empty = 1.0
    for position, vertex in enumerate(order):
        probability = probabilities[vertex - 1]
        value += probability * prefix_empty * distance[0][vertex]
        gap_empty = 1.0
        for later in order[position + 1 :]:
            value += (
                probability
                * probabilities[later - 1]
                * gap_empty
                * distance[vertex][later]
            )
            gap_empty *= 1.0 - probabilities[later - 1]
        prefix_empty *= 1.0 - probability
    return value


def evaluate(distance: list[list[int]], probabilities: list[float]):
    scenario_probability = subset_probabilities(probabilities)
    posterior = sum(
        weight * cost
        for weight, cost in zip(scenario_probability, subset_tsp(distance))
    )
    best_value = float("inf")
    best_order = None
    for order in itertools.permutations(range(1, len(probabilities) + 1)):
        value = open_cost(order, probabilities, distance)
        if value < best_value:
            best_value = value
            best_order = order
    assert best_order is not None
    farthest = 0.0
    for mask, weight in enumerate(scenario_probability):
        farthest += weight * max(
            [0]
            + [
                distance[0][vertex]
                for vertex in range(1, len(probabilities) + 1)
                if mask & (1 << (vertex - 1))
            ]
        )
    return posterior, best_value, farthest, best_order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260907)
    arguments = parser.parse_args()
    rng = random.Random(arguments.seed)
    alphabet = (0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 0.9, 0.99)
    worst = None
    for trial in range(arguments.trials):
        distance = random_graph_metric(arguments.n, rng)
        profiles = [[value] * arguments.n for value in alphabet]
        profiles.extend(
            [rng.choice(alphabet) for _ in range(arguments.n)] for _ in range(5)
        )
        for probabilities in profiles:
            posterior, path, farthest, order = evaluate(distance, probabilities)
            ratio = path / posterior
            record = (ratio, trial, probabilities, distance, path, posterior, farthest, order)
            if worst is None or ratio > worst[0]:
                worst = record
                print(
                    "worst",
                    ratio,
                    "close/P",
                    farthest / posterior,
                    "p",
                    probabilities,
                    "order",
                    order,
                    flush=True,
                )
            if ratio > 2.4 + 1.0e-10:
                print("COUNTEREXAMPLE", record)
                return
    print("FINAL", worst)


if __name__ == "__main__":
    main()
