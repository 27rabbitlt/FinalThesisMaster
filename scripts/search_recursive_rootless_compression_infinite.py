#!/usr/bin/env python3
"""Adversarial audit of the infinite recursive rootless hierarchy.

This extends ``search_recursive_rootless_compression.py`` by solving the
absorbing subset chain exactly (up to floating-point arithmetic), rather than
truncating it at a prescribed level.  It searches both for

    U_rec / P > 2.9

and for failure of the stronger RH10 sufficient inequality

    3 P - U_rec >= P/5 - 2(P-R).

The search is exploratory.  Any reported witness must be rationalized and
checked separately with exact arithmetic before it is used mathematically.
"""

from __future__ import annotations

import argparse
import functools
import math
import random

from scipy.optimize import differential_evolution

from search_compressed_two_guide import mask_probability
from search_hazard_tree_ratio import named_metric
from search_rootless_integrated_random_graphs import Instance, random_probability
from search_rootless_threshold_random_graphs import random_connected_metric


def recursive_value_infinite(instance: Instance, p):
    """Return infinite-horizon E[min_j J(B_j)] and useful audit data."""

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
            if probability > 0.0:
                rows.append((sub, probability))
            if sub == 0:
                break
            sub = (sub - 1) & state
        transition.append(rows)

    @functools.lru_cache(None)
    def future(state: int, best: int) -> float:
        if objective[state] < objective[best]:
            best = state
        rows = transition[state]
        self_probability = sum(prob for sub, prob in rows if sub == state)
        if self_probability >= 1.0 - 1e-14:
            return objective[best]
        strict_value = sum(
            prob * future(sub, best)
            for sub, prob in rows
            if sub != state
        )
        return strict_value / (1.0 - self_probability)

    expected_best = sum(
        weights[initial] * future(initial, initial)
        for initial in range(1 << n)
        if weights[initial] > 0.0
    )
    return expected_best, objective, transition


def audit(instance: Instance, p):
    posterior, radius, *_ = instance.terms(p, include_rootful=False)
    route, _, _ = recursive_value_infinite(instance, p)
    deficit = posterior - radius
    gamma = 3.0 * posterior - route
    rh10_margin = gamma - (posterior / 5.0 - 2.0 * deficit)
    ratio = route / posterior if posterior > 1e-15 else 0.0
    normalized_margin = (
        rh10_margin / posterior if posterior > 1e-15 else 1.0
    )
    return ratio, rh10_margin, normalized_margin, posterior, radius, route


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--metrics", type=int, default=200)
    parser.add_argument("--probabilities", type=int, default=100)
    parser.add_argument("--optimize", type=int, default=10)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--named", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    families = (
        "unit", "path", "cycle", "star", "binary", "random-tree",
        "graph12", "graph123", "client-graph-19", "euclidean",
        "multiscale-graph", "multiscale-complete",
    )
    worst_ratio = (-math.inf, None)
    worst_margin = (math.inf, None)
    candidates = []

    for metric_index in range(args.metrics):
        if args.named:
            family = families[metric_index % len(families)]
            d = named_metric(family, args.clients, rng)
            label = family
        else:
            d, label = random_connected_metric(args.clients, rng)
        instance = Instance(d, label)
        local_margin = (math.inf, None)
        probability_vectors = [
            (value,) * args.clients
            for value in (0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95)
        ]
        for deterministic in range(1, args.clients):
            for value in (0.001, 0.01, 0.05, 0.1, 0.25, 0.5):
                probability_vectors.append(
                    (1.0,) * deterministic
                    + (value,) * (args.clients - deterministic)
                )
        probability_vectors.extend(
            tuple(random_probability(rng) for _ in range(args.clients))
            for _ in range(args.probabilities)
        )
        for p in probability_vectors:
            result = audit(instance, p)
            ratio, margin, normalized_margin, posterior, radius, route = result
            record = (label, tuple(p), posterior, radius, route, d)
            if ratio > worst_ratio[0]:
                worst_ratio = (ratio, record)
            if normalized_margin < worst_margin[0]:
                worst_margin = (normalized_margin, record)
            if normalized_margin < local_margin[0]:
                local_margin = (normalized_margin, tuple(p))
            if ratio > 2.9 + 1e-9 or margin < -1e-9:
                print("COUNTEREXAMPLE", ratio, margin, record, flush=True)
                return
        candidates.append((local_margin[0], instance, local_margin[1]))
        if metric_index % 20 == 0:
            print(
                "progress", metric_index,
                "ratio", worst_ratio[0],
                "margin", worst_margin[0],
                flush=True,
            )

    candidates.sort(key=lambda row: row[0])
    for _, instance, _ in candidates[: args.optimize]:
        result = differential_evolution(
            lambda values: audit(instance, values)[2],
            [(1e-5, 1.0 - 1e-5)] * args.clients,
            seed=rng.randrange(1 << 30),
            maxiter=160,
            popsize=10,
            polish=True,
            updating="immediate",
        )
        checked = audit(instance, result.x)
        ratio, margin, normalized_margin, posterior, radius, route = checked
        record = (
            instance.edges, tuple(result.x), posterior, radius, route,
            instance.d,
        )
        if ratio > worst_ratio[0]:
            worst_ratio = (ratio, record)
        if normalized_margin < worst_margin[0]:
            worst_margin = (normalized_margin, record)
        print("optimized", ratio, margin, normalized_margin, flush=True)
        if ratio > 2.9 + 1e-8 or margin < -1e-8:
            print("OPTIMIZED COUNTEREXAMPLE", ratio, margin, record)
            return

    print("worst U_rec/P", worst_ratio)
    print("smallest RH10 margin", worst_margin)


if __name__ == "__main__":
    main()
