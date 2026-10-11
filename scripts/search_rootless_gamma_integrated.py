#!/usr/bin/env python3
"""Search the integrated Gamma inequality for rootless two-guide compression.

For nonempty B, map every C vertex to its nearest client of B; for empty B
use the empty core.  With M=E[tsp(T*)], A the expected attachment, P the
posterior benchmark, and R the ordinary guide radius, test

    M + 2 A + R/e <= 3 P.

It also compares against the original depot-including projection and tests
the best-of-two 2.9 factor.  This is an exploratory search, not a proof.
"""

from __future__ import annotations

import argparse
import math
import random
from itertools import combinations

from search_compressed_two_guide import evaluate, mask_probability
from search_hazard_tree_ratio import (
    metric_closure,
    named_metric,
    probability_vectors,
    subset_tsp_costs,
)


def radius(distance, probabilities):
    n = len(probabilities)
    root = n
    return sum(
        mask_probability(guide, probabilities)
        * sum(
            probabilities[v]
            * min(
                [distance[v][root]]
                + [distance[v][b] for b in range(n) if guide & (1 << b)]
            )
            for v in range(n)
        )
        for guide in range(1 << n)
    )


def check(distance, probabilities, costs=None):
    if costs is None:
        costs = subset_tsp_costs(distance)
    posterior, rootless_bound, rootless_backbone, rootless_attachment = evaluate(
        distance,
        probabilities,
        costs,
        client_only_projection=True,
    )
    _, rooted_bound, rooted_backbone, rooted_attachment = evaluate(
        distance,
        probabilities,
        costs,
        client_only_projection=False,
    )
    guide_radius = radius(distance, probabilities)
    gamma_e_gap = rootless_bound + guide_radius / math.e - 3.0 * posterior
    gamma_19_gap = rootless_bound + 2.0 * guide_radius / 19.0 - 3.0 * posterior
    best_bound = min(rooted_bound, rootless_bound)
    best_29_gap = best_bound - 2.9 * posterior
    average_29_gap = (rooted_bound + rootless_bound) / 2.0 - 2.9 * posterior
    average_gamma_19_gap = (
        (rooted_bound + rootless_bound) / 2.0
        + 2.0 * guide_radius / 19.0
        - 3.0 * posterior
    )
    scale = posterior if posterior else 1.0
    return (
        gamma_e_gap,
        gamma_19_gap,
        best_29_gap,
        average_29_gap,
        posterior,
        rooted_bound,
        rootless_bound,
        rooted_backbone,
        rootless_backbone,
        rooted_attachment,
        rootless_attachment,
        guide_radius,
        gamma_e_gap / scale,
        gamma_19_gap / scale,
        best_bound / scale,
        (rooted_bound + rootless_bound) / (2.0 * scale),
        (posterior - guide_radius) / scale,
        average_gamma_19_gap / scale,
        rooted_bound / scale,
        rootless_bound / scale,
        (rootless_bound - rooted_bound) / scale,
    )


def connected_graph_metrics(clients):
    size = clients + 1
    root = clients
    pairs = tuple(combinations(range(size), 2))
    infinity = size + 1
    for graph_mask in range(1 << len(pairs)):
        distance = [
            [0 if u == v else infinity for v in range(size)]
            for u in range(size)
        ]
        for index, (u, v) in enumerate(pairs):
            if graph_mask & (1 << index):
                distance[u][v] = distance[v][u] = 1
        for middle in range(size):
            for u in range(size):
                for v in range(size):
                    distance[u][v] = min(
                        distance[u][v],
                        distance[u][middle] + distance[middle][v],
                    )
        if all(distance[root][v] <= clients for v in range(clients)):
            yield graph_mask, distance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=7)
    parser.add_argument("--metrics", type=int, default=1000)
    parser.add_argument("--probabilities", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--exhaustive-graphs", action="store_true")
    parser.add_argument(
        "--evolve",
        choices=(
            "gamma-e",
            "gamma-19",
            "best",
            "average",
            "average-gamma-19",
            "rooted",
            "rootless",
            "rootless-minus-rooted",
            "rooted-minus-rootless",
        ),
        help="evolve a weighted shortest-path metric and heterogeneous p",
    )
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--popsize", type=int, default=15)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    worst_gamma_e = None
    worst_gamma_19 = None
    worst_best_29 = None
    worst_average_29 = None
    worst_hard_best_29 = None
    worst_hard_gamma_e = None
    worst_average_gamma_19 = None

    def update_worst(result, context):
        nonlocal worst_gamma_e, worst_gamma_19, worst_best_29, worst_average_29
        nonlocal worst_hard_best_29, worst_hard_gamma_e, worst_average_gamma_19
        record = (*result, *context)
        if worst_gamma_e is None or result[12] > worst_gamma_e[12]:
            worst_gamma_e = record
        if worst_gamma_19 is None or result[13] > worst_gamma_19[13]:
            worst_gamma_19 = record
        if worst_best_29 is None or result[14] > worst_best_29[14]:
            worst_best_29 = record
        if worst_average_29 is None or result[15] > worst_average_29[15]:
            worst_average_29 = record
        if result[16] < 0.05:
            if worst_hard_best_29 is None or result[14] > worst_hard_best_29[14]:
                worst_hard_best_29 = record
            if worst_hard_gamma_e is None or result[12] > worst_hard_gamma_e[12]:
                worst_hard_gamma_e = record
        if worst_average_gamma_19 is None or result[17] > worst_average_gamma_19[17]:
            worst_average_gamma_19 = record
        return record

    if args.evolve:
        from scipy.optimize import differential_evolution

        n = args.clients
        total = n + 1
        pairs = [(u, v) for u in range(total) for v in range(u)]
        objective_index = {
            "gamma-e": 12,
            "gamma-19": 13,
            "best": 14,
            "average": 15,
            "average-gamma-19": 17,
            "rooted": 18,
            "rootless": 19,
            "rootless-minus-rooted": 20,
        }[args.evolve]
        calls = 0
        incumbent = None

        def decode(vector):
            weights = [
                [0.0 if u == v else float("inf") for v in range(total)]
                for u in range(total)
            ]
            for log_length, (u, v) in zip(vector, pairs):
                weights[u][v] = weights[v][u] = math.exp(log_length)
            distance = metric_closure(weights)
            probabilities = tuple(
                1.0 / (1.0 + math.exp(-logit))
                for logit in vector[len(pairs) :]
            )
            return distance, probabilities

        def objective(vector):
            nonlocal calls, incumbent
            distance, probabilities = decode(vector)
            result = check(distance, probabilities)
            calls += 1
            score = result[objective_index]
            if args.evolve == "rooted-minus-rootless":
                score = -result[20]
            if incumbent is None or score > incumbent[0]:
                incumbent = (score, result, probabilities, distance)
                print(
                    f"evolve {args.evolve} score={score:.12g} calls={calls}",
                    flush=True,
                )
            return -score

        dimension = len(pairs) + n
        differential_evolution(
            objective,
            [(-2.5, 2.5)] * len(pairs) + [(-8.0, 8.0)] * n,
            seed=args.seed,
            maxiter=args.maxiter,
            popsize=args.popsize,
            polish=True,
            updating="immediate",
            workers=1,
            tol=1e-9,
        )
        print("evolutionary incumbent", incumbent)
        return

    if args.exhaustive_graphs:
        grid = (0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99)
        for graph_mask, distance in connected_graph_metrics(args.clients):
            costs = subset_tsp_costs(distance)
            candidates = [(p,) * args.clients for p in grid]
            candidates.extend(
                tuple(rng.choice(grid) for _ in range(args.clients))
                for _ in range(args.probabilities)
            )
            for probabilities in candidates:
                result = check(distance, probabilities, costs)
                update_worst(
                    result,
                    (graph_mask, tuple(probabilities), distance),
                )
        print("worst Gamma >= R/e gap", worst_gamma_e)
        print("worst Gamma >= 2R/19 gap", worst_gamma_19)
        print("worst best-of-two 2.9 gap", worst_best_29)
        print("worst average-of-two 2.9 gap", worst_average_29)
        print("worst average Gamma >= 2R/19 gap", worst_average_gamma_19)
        print("worst hard best-of-two", worst_hard_best_29)
        print("worst hard Gamma >= R/e", worst_hard_gamma_e)
        return

    families = (
        "unit",
        "path",
        "cycle",
        "star",
        "binary",
        "random-tree",
        "graph12",
        "graph123",
        "client-graph-19",
        "euclidean",
        "multiscale-graph",
        "multiscale-complete",
    )
    for metric_index in range(args.metrics):
        family = families[metric_index % len(families)]
        distance = named_metric(family, args.clients, rng)
        costs = subset_tsp_costs(distance)
        for probabilities in probability_vectors(
            args.clients,
            rng,
            args.probabilities,
        ):
            result = check(distance, probabilities, costs)
            previous = worst_best_29
            record = update_worst(
                result,
                (family, tuple(probabilities), distance),
            )
            if previous is None or result[14] > previous[14]:
                print("worst best-of-two", record[:17], flush=True)
    print("worst Gamma >= R/e gap", worst_gamma_e)
    print("worst Gamma >= 2R/19 gap", worst_gamma_19)
    print("worst best-of-two 2.9 gap", worst_best_29)
    print("worst average-of-two 2.9 gap", worst_average_29)
    print("worst average Gamma >= 2R/19 gap", worst_average_gamma_19)
    print("worst hard best-of-two", worst_hard_best_29)
    print("worst hard Gamma >= R/e", worst_hard_gamma_e)


if __name__ == "__main__":
    main()
