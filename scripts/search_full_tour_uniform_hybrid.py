#!/usr/bin/env python3
"""Falsify a two-geometry master-order dichotomy.

The candidate construction takes the cheaper of

1. a shortest full-support metric TSP tour, evaluated after activation
   shortcutting (with favorable tie breaking), and
2. the average activation-shortcut cost of a uniformly random client order.

The first choice preserves line/cycle structure; the second ignores rare
vertices that can force a bad full-support topology.  This program compares
their minimum with the exact posterior optimum on small instances.  It is a
floating-point falsifier, not a proof.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random

from search_open_path_gap import random_graph_metric, subset_probabilities, subset_tsp
from search_open_residual_rounding import (
    euclidean_metric,
    line_metric,
    probability_profiles,
    star_metric,
)


def closed_cost(
    order: tuple[int, ...],
    probabilities: list[float],
    distance: list[list[float]],
) -> float:
    value = 0.0
    prefix_empty = 1.0
    suffix_empty = 1.0
    for position, vertex in enumerate(order):
        probability = probabilities[vertex - 1]
        value += probability * prefix_empty * distance[0][vertex]
        prefix_empty *= 1.0 - probability
        reverse_vertex = order[-1 - position]
        reverse_probability = probabilities[reverse_vertex - 1]
        value += reverse_probability * suffix_empty * distance[reverse_vertex][0]
        suffix_empty *= 1.0 - reverse_probability
        gap_empty = 1.0
        for later in order[position + 1 :]:
            value += (
                probability
                * probabilities[later - 1]
                * gap_empty
                * distance[vertex][later]
            )
            gap_empty *= 1.0 - probabilities[later - 1]
    return value


def full_tour_length(
    order: tuple[int, ...], distance: list[list[float]]
) -> float:
    return (
        distance[0][order[0]]
        + sum(distance[first][second] for first, second in zip(order, order[1:]))
        + distance[order[-1]][0]
    )


def evaluate(distance: list[list[float]], probabilities: list[float]):
    scenario_probability = subset_probabilities(probabilities)
    posterior = sum(
        weight * value
        for weight, value in zip(scenario_probability, subset_tsp(distance))
    )
    best_length = math.inf
    best_stopped = math.inf
    uniform_total = 0.0
    count = 0
    for order in itertools.permutations(range(1, len(probabilities) + 1)):
        stopped = closed_cost(order, probabilities, distance)
        uniform_total += stopped
        count += 1
        length = full_tour_length(order, distance)
        if length < best_length - 1.0e-10:
            best_length = length
            best_stopped = stopped
        elif abs(length - best_length) <= 1.0e-10:
            best_stopped = min(best_stopped, stopped)
    uniform = uniform_total / count
    hybrid = min(best_stopped, uniform)
    return hybrid / posterior, best_stopped / posterior, uniform / posterior, posterior


def clustered_metric(client_count: int, rng: random.Random):
    root_distance = 10.0 ** rng.uniform(0.0, 2.0)
    within = root_distance * 10.0 ** rng.uniform(-4.0, -1.0)
    return [
        [
            0.0
            if first == second
            else root_distance
            if first == 0 or second == 0
            else within
            for second in range(client_count + 1)
        ]
        for first in range(client_count + 1)
    ]


def binary_metric(client_count: int, rng: random.Random):
    result = [[0.0] * (client_count + 1) for _ in range(client_count + 1)]
    for first in range(client_count + 1):
        for second in range(first + 1, client_count + 1):
            result[first][second] = result[second][first] = (
                1.0 if rng.random() < 0.25 else 2.0
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    builders = (
        ("graph", random_graph_metric),
        ("line", line_metric),
        ("euclidean", euclidean_metric),
        ("star", star_metric),
        ("cluster", clustered_metric),
        ("binary", binary_metric),
    )
    worst = None
    for trial in range(args.trials):
        name, builder = builders[trial % len(builders)]
        distance = builder(args.n, rng)
        for probabilities in probability_profiles(args.n, rng):
            result = evaluate(distance, probabilities)
            record = (result[0], name, trial, probabilities, distance, result)
            if worst is None or record[0] > worst[0]:
                worst = record
                print(
                    "worst hybrid",
                    result[0],
                    "full",
                    result[1],
                    "uniform",
                    result[2],
                    "metric",
                    name,
                    "trial",
                    trial,
                    flush=True,
                )
            if result[0] > 2.9 + 1.0e-8:
                print("COUNTEREXAMPLE", record)
                return
    print("FINAL", worst)


if __name__ == "__main__":
    main()
