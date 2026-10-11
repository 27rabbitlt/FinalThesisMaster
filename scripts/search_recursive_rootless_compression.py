#!/usr/bin/env python3
"""Exact-state search for a recursive rootless Voronoi hierarchy.

Start with B_0 drawn from the product distribution.  Given nonempty B_j,
draw a fresh product sample and retain every center whose rootless Voronoi
cell is hit; B_{j+1} is the resulting subset.  The route uses the pathwise
best tour--facility objective J(B_j) over a fixed number of levels.  State
transition probabilities are exact products (floating-point arithmetic).
"""

from __future__ import annotations

import argparse
import functools
import math
import random

from search_compressed_two_guide import mask_probability
from search_hazard_tree_ratio import named_metric
from search_rootless_integrated_random_graphs import Instance, random_probability
from search_rootless_threshold_random_graphs import random_connected_metric


def recursive_value(instance: Instance, p, levels: int):
    n = instance.n
    weights = [mask_probability(mask, p) for mask in range(1 << n)]
    assignment = [
        sum(p[v] * instance.nearest[mask][v] for v in range(n))
        for mask in range(1 << n)
    ]
    objective = [
        instance.costs[mask] + 2.0 * assignment[mask]
        for mask in range(1 << n)
    ]
    transition = []
    for state in range(1 << n):
        occupancy = {}
        for center, cell in instance.rootless_cells[state]:
            empty = 1.0
            for v in range(n):
                if cell & (1 << v):
                    empty *= 1.0 - p[v]
            occupancy[center] = 1.0 - empty
        rows = []
        sub = state
        while True:
            probability = 1.0
            for center, _ in instance.rootless_cells[state]:
                probability *= (
                    occupancy[center]
                    if sub & (1 << center)
                    else 1.0 - occupancy[center]
                )
            rows.append((sub, probability))
            if sub == 0:
                break
            sub = (sub - 1) & state
        transition.append(rows)

    @functools.lru_cache(None)
    def future(state: int, best: int, remaining: int) -> float:
        if objective[state] < objective[best]:
            best = state
        if remaining == 0 or state == 0:
            return objective[best]
        return sum(
            probability * future(next_state, best, remaining - 1)
            for next_state, probability in transition[state]
        )

    expected_best = 0.0
    for initial in range(1 << n):
        expected_best += weights[initial] * future(initial, initial, levels)

    distributions = [weights]
    expected_level = [sum(weights[s] * objective[s] for s in range(1 << n))]
    for _ in range(levels):
        next_distribution = [0.0] * (1 << n)
        for state, state_probability in enumerate(distributions[-1]):
            for next_state, probability in transition[state]:
                next_distribution[next_state] += state_probability * probability
        distributions.append(next_distribution)
        expected_level.append(
            sum(next_distribution[s] * objective[s] for s in range(1 << n))
        )
    return expected_best, expected_level, objective, transition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=7)
    parser.add_argument("--metrics", type=int, default=200)
    parser.add_argument("--probabilities", type=int, default=100)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--named", action="store_true")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    families = (
        "unit", "path", "cycle", "star", "binary", "random-tree",
        "graph12", "graph123", "client-graph-19", "euclidean",
        "multiscale-graph", "multiscale-complete",
    )
    worst = (-math.inf, None)
    smallest_margin = (math.inf, None)
    for metric_index in range(args.metrics):
        if args.named:
            family = families[metric_index % len(families)]
            d = named_metric(family, args.clients, rng)
            label = family
        else:
            d, edges = random_connected_metric(args.clients, rng)
            label = edges
        instance = Instance(d, label)
        candidates = [
            (value,) * args.clients
            for value in (0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95)
        ]
        candidates.extend(
            tuple(random_probability(rng) for _ in range(args.clients))
            for _ in range(args.probabilities)
        )
        for p in candidates:
            posterior, radius, *_ = instance.terms(p, include_rootful=False)
            route, level_values, _, _ = recursive_value(instance, p, args.levels)
            ratio = route / posterior if posterior else 0.0
            gamma = 3.0 * posterior - route
            margin = gamma - (posterior / 5.0 - 2.0 * (posterior - radius))
            record = (label, p, posterior, radius, route, level_values, d)
            if ratio > worst[0]:
                worst = (ratio, record)
            if margin < smallest_margin[0]:
                smallest_margin = (margin, record)
            if margin < -1e-9:
                print("COARSE COUNTEREXAMPLE", margin, record)
                return
        if metric_index % 20 == 0:
            print("progress", metric_index, "worst", worst[0], "margin", smallest_margin[0], flush=True)
    print("worst recursive route/P", worst)
    print("smallest coarse-target margin", smallest_margin)


if __name__ == "__main__":
    main()
