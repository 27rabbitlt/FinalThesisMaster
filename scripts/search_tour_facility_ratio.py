#!/usr/bin/env python3
"""Falsification search for a deterministic tour-core master route.

For a metric and independent activation probabilities p, compute

    J* = min_S { tsp(S union {r}) + 2 sum_v p_v d(v,S union {r}) }

and compare it with P = E[tsp(A union {r})].  Every S gives a valid
a-priori master route.  The search is evidence only, never a proof.
"""

from __future__ import annotations

import argparse
import random

from search_hazard_tree_ratio import (
    expected_posterior,
    named_metric,
    probability_vectors,
    subset_tsp_costs,
)


def evaluate(distance, probabilities, tours=None):
    n = len(probabilities)
    root = n
    if tours is None:
        tours = subset_tsp_costs(distance)
    posterior = expected_posterior(probabilities, tours)
    best = float("inf")
    best_mask = 0
    for mask, tour in enumerate(tours):
        centers = [root] + [v for v in range(n) if mask & (1 << v)]
        attachment = sum(
            probabilities[v] * min(distance[v][u] for u in centers)
            for v in range(n)
        )
        value = tour + 2.0 * attachment
        if value < best:
            best = value
            best_mask = mask
    return posterior, best, best_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=9)
    parser.add_argument("--metrics", type=int, default=500)
    parser.add_argument("--probabilities", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    rng = random.Random(args.seed)
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
    worst = None
    for metric_index in range(args.metrics):
        family = families[metric_index % len(families)]
        distance = named_metric(family, args.clients, rng)
        tours = subset_tsp_costs(distance)
        for probabilities in probability_vectors(
            args.clients, rng, args.probabilities
        ):
            posterior, objective, mask = evaluate(distance, probabilities, tours)
            ratio = objective / posterior if posterior else 0.0
            if worst is None or ratio > worst[0]:
                worst = (
                    ratio,
                    family,
                    tuple(probabilities),
                    posterior,
                    objective,
                    mask,
                    distance,
                )
                print("worst", worst[:6], flush=True)
            if ratio > 2.9 + 1e-9:
                print("COUNTEREXAMPLE", worst)
                return
    print("no 2.9 violation; worst", worst)


if __name__ == "__main__":
    main()
