#!/usr/bin/env python3
"""Test a one-sample Voronoi order for the mean-radius residual.

Let mu_v be the expected nearest-other distance and

    g_uv = max(d_uv-mu_u-mu_v, 0).

For a guide set B, orient an optimal d-tour, assign every missing client to
its nearest guide center, and concatenate the resulting Voronoi cells in the
guide-tour order.  This creates a full order before an independent activation
set is drawn.  The program exactly averages its stopped open g-cost on small
instances and compares it with P-L, where L=sum p_v mu_v.

The calculation is a falsifier, not a proof.  It deliberately chooses the
best d-optimal guide orientation but uses a fixed label order inside cells.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random

from search_open_path_gap import open_cost, random_graph_metric, subset_probabilities, subset_tsp
from search_open_residual_rounding import (
    euclidean_metric,
    line_metric,
    nearest_other_means,
    probability_profiles,
    star_metric,
)


def optimal_orders(distance: list[list[float]]):
    n = len(distance) - 1
    result: list[list[tuple[int, ...]]] = [[tuple()] for _ in range(1 << n)]
    for mask in range(1, 1 << n):
        vertices = tuple(v for v in range(1, n + 1) if mask & (1 << (v - 1)))
        best = math.inf
        orders: list[tuple[int, ...]] = []
        for order in itertools.permutations(vertices):
            value = (
                distance[0][order[0]]
                + sum(distance[u][v] for u, v in zip(order, order[1:]))
                + distance[order[-1]][0]
            )
            if value < best - 1.0e-10:
                best = value
                orders = [order]
            elif abs(value - best) <= 1.0e-10:
                orders.append(order)
        result[mask] = orders
    return result


def full_voronoi_order(
    distance: list[list[float]],
    guide_mask: int,
    center_order: tuple[int, ...],
) -> tuple[int, ...]:
    n = len(distance) - 1
    centers = (0, *center_order)
    cells = {center: [] for center in centers}
    for vertex in range(1, n + 1):
        if guide_mask & (1 << (vertex - 1)):
            continue
        center = min(centers, key=lambda item: (distance[vertex][item], item))
        cells[center].append(vertex)

    order: list[int] = []
    order.extend(sorted(cells[0]))
    for center in center_order:
        order.append(center)
        order.extend(sorted(cells[center]))
    assert sorted(order) == list(range(1, n + 1))
    return tuple(order)


def evaluate(distance: list[list[float]], probabilities: list[float]):
    n = len(probabilities)
    scenario_probability = subset_probabilities(probabilities)
    posterior = sum(
        weight * value
        for weight, value in zip(scenario_probability, subset_tsp(distance))
    )
    means = nearest_other_means(distance, probabilities)
    mean_radii = [0.0] + means
    residual = [
        [
            max(
                distance[first][second]
                - mean_radii[first]
                - mean_radii[second],
                0.0,
            )
            for second in range(n + 1)
        ]
        for first in range(n + 1)
    ]
    second_budget = posterior - sum(
        probability * mean
        for probability, mean in zip(probabilities, means)
    )

    guide_orders = optimal_orders(distance)
    expected = 0.0
    worst_guide = None
    for guide_mask, weight in enumerate(scenario_probability):
        candidate = min(
            (
                open_cost(
                    full_voronoi_order(distance, guide_mask, center_order),
                    probabilities,
                    residual,
                ),
                center_order,
            )
            for center_order in guide_orders[guide_mask]
        )
        expected += weight * candidate[0]
        if worst_guide is None or candidate[0] > worst_guide[0]:
            worst_guide = (candidate[0], guide_mask, candidate[1])
    return expected / second_budget, expected, second_budget, worst_guide


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    builders = (
        ("graph", random_graph_metric),
        ("line", line_metric),
        ("euclidean", euclidean_metric),
        ("star", star_metric),
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
                    "worst mean-residual Voronoi ratio",
                    result[0],
                    "metric",
                    name,
                    "trial",
                    trial,
                    flush=True,
                )
            if result[0] > 7.0 + 1.0e-8:
                print("COUNTEREXAMPLE", record)
                return
    print("FINAL", worst)


if __name__ == "__main__":
    main()
