#!/usr/bin/env python3
"""Annealed sparse-graph stress test for recursive rootless compression.

This is a structural large-instance model of (RG1), not an exact checker for
a finite stochastic-TSP instance.  Condition on ``k`` initial guide centers.
Their induced short-edge graph (including the depot) is G(k+1,d/k).  A core
of size ``s`` is thinned independently with survival probability
``1-exp(-k/s)``, the balanced-cell limit of the rootless hierarchy.  Outside
assignment has annealed value k*(1+(1-d/k)**(s+1)).

The script precomputes every subset TSP cost in the associated 1/2 metric and
then samples complete hierarchy paths.  Its purpose is to probe the only
known asymptotic hard regime near D/P=0.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics

from search_hazard_tree_ratio import metric_closure, subset_tsp_costs


def random_two_distance_metric(k: int, d: float, rng: random.Random):
    theta = min(1.0, d / k)
    size = k + 1
    weights = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            value = 1.0 if rng.random() < theta else 2.0
            weights[i][j] = weights[j][i] = value
    return metric_closure(weights)


def conditional_deleted_count(s: int, deletion: float, rng: random.Random) -> int:
    """Sample Binomial(s,deletion) conditioned to be positive."""
    probabilities = []
    total = 1.0 - (1.0 - deletion) ** s
    running = 0.0
    for deleted in range(1, s + 1):
        probability = (
            math.comb(s, deleted)
            * deletion**deleted
            * (1.0 - deletion) ** (s - deleted)
            / total
        )
        running += probability
        probabilities.append(running)
    draw = rng.random()
    for deleted, cumulative in enumerate(probabilities, 1):
        if draw <= cumulative:
            return deleted
    return s


def one_path(costs, k: int, d: float, rng: random.Random) -> float:
    theta = min(1.0, d / k)
    state = (1 << k) - 1
    best = math.inf
    while True:
        s = state.bit_count()
        assignment = k * (1.0 + (1.0 - theta) ** (s + 1))
        best = min(best, costs[state] + 2.0 * assignment)
        if s == 0:
            return best
        deletion = math.exp(-k / s)
        deleted_count = conditional_deleted_count(s, deletion, rng)
        vertices = [v for v in range(k) if state & (1 << v)]
        for v in rng.sample(vertices, deleted_count):
            state ^= 1 << v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=14)
    parser.add_argument("--d", type=float, default=3.0)
    parser.add_argument("--graphs", type=int, default=100)
    parser.add_argument("--paths", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=8675309)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    posteriors = []
    routes = []
    for graph_index in range(args.graphs):
        metric = random_two_distance_metric(args.k, args.d, rng)
        costs = subset_tsp_costs(metric)
        posteriors.append(costs[-1])
        routes.extend(
            one_path(costs, args.k, args.d, rng)
            for _ in range(args.paths)
        )
        if graph_index % 10 == 0:
            print("progress", graph_index, flush=True)

    posterior = statistics.mean(posteriors)
    route = statistics.mean(routes)
    theta = min(1.0, args.d / args.k)
    radius = args.k * (1.0 + (1.0 - theta) ** (args.k + 1))
    deficit = posterior - radius
    saving = posterior + 2.0 * radius - route
    rh10_rhs = posterior / 5.0 - 4.0 * deficit
    print("P", posterior)
    print("R", radius)
    print("D/P", deficit / posterior)
    print("U/P", route / posterior)
    print("recursive saving/P", saving / posterior)
    print("RH10 rhs/P", rh10_rhs / posterior)
    print("RH10 margin/P", (saving - rh10_rhs) / posterior)


if __name__ == "__main__":
    main()
