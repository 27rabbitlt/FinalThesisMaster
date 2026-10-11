#!/usr/bin/env python3
"""Falsify the natural vertex-deletion induction for rooted open orders.

Let Q(p) be the minimum expected depot-starting open shortcut path and P(p)
the expected posterior TSP.  A minimal-counterexample induction at constant c
would follow if every nonzero instance had a vertex v with

    Q(p) - Q(p with p_v = 0)
        <= c (P(p) - P(p with p_v = 0)).

This program checks that local statement exactly up to floating-point
evaluation on small graph metrics.  It is a falsifier, not a proof.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random

from search_open_path_gap import (
    open_cost,
    random_graph_metric,
    subset_probabilities,
    subset_tsp,
)


def posterior(distance: list[list[int]], probabilities: list[float]) -> float:
    return sum(
        weight * cost
        for weight, cost in zip(
            subset_probabilities(probabilities), subset_tsp(distance)
        )
    )


def optimum_open(distance: list[list[int]], probabilities: list[float]) -> float:
    support = tuple(
        vertex
        for vertex, probability in enumerate(probabilities, start=1)
        if probability > 0.0
    )
    if not support:
        return 0.0
    return min(
        open_cost(order, probabilities, distance)
        for order in itertools.permutations(support)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    alphabet = (0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 0.9, 0.99)
    worst = None
    for trial in range(args.trials):
        distance = random_graph_metric(args.n, rng)
        profiles = [[value] * args.n for value in alphabet]
        profiles.extend(
            [rng.choice(alphabet) for _ in range(args.n)] for _ in range(5)
        )
        for probabilities in profiles:
            post = posterior(distance, probabilities)
            path = optimum_open(distance, probabilities)
            ratios = []
            for index, probability in enumerate(probabilities):
                if probability == 0.0:
                    continue
                deleted = probabilities[:]
                deleted[index] = 0.0
                delta_post = post - posterior(distance, deleted)
                delta_path = path - optimum_open(distance, deleted)
                if delta_post > 1.0e-12:
                    ratios.append((delta_path / delta_post, index + 1))
            minimum = min(ratios, default=(math.inf, -1))
            record = (
                minimum[0],
                trial,
                probabilities,
                path / post,
                minimum[1],
                distance,
            )
            if worst is None or record[0] > worst[0]:
                worst = record
                print("worst minimum", *record[:5], flush=True)
            if minimum[0] > 2.4 + 1.0e-9:
                print("COUNTEREXAMPLE", record)
                return
    print("FINAL", worst)


if __name__ == "__main__":
    main()
