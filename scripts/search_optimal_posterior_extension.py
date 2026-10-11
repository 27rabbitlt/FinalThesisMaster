#!/usr/bin/env python3
"""Search the optimal posterior-order extension certificate.

Draw B from the product activation law.  Choose a shortest rooted tour order
on B, with tie-breaking that minimizes the exact p-stopped cost of the best
full permutation having that order as a subsequence.  Average this best
extension cost over B and compare it with the posterior value.

For n <= 8 the script enumerates all full permutations once and tabulates the
best cost for every restricted order.  Floating-point arithmetic makes this an
exploratory search, not a proof certificate.
"""

from __future__ import annotations

import argparse
from itertools import permutations
import math
import random

from search_recursive_posterior_buckets import stopped_cost_fast
from search_recursive_self_thinning import (
    posterior,
    random_graph_metric,
    subset_probability,
    tour_cost,
)


def subset_tsp_and_orders(
    d: list[list[float]],
) -> tuple[list[float], list[list[tuple[int, ...]]]]:
    n = len(d) - 1
    costs = [0.0] * (1 << n)
    orders: list[list[tuple[int, ...]]] = [[] for _ in range(1 << n)]
    orders[0] = [tuple()]
    for mask in range(1, 1 << n):
        vertices = tuple(v + 1 for v in range(n) if mask >> v & 1)
        best = math.inf
        for order in permutations(vertices):
            value = tour_cost(order, d)
            if value < best - 1e-10:
                best = value
                orders[mask] = [order]
            elif abs(value - best) <= 1e-10:
                orders[mask].append(order)
        costs[mask] = best
    return costs, orders


def best_extension_table(
    d: list[list[float]], p: list[float]
) -> dict[tuple[int, ...], float]:
    n = len(p)
    answer: dict[tuple[int, ...], float] = {}
    for full_order in permutations(range(1, n + 1)):
        value = stopped_cost_fast(full_order, p, d)
        for mask in range(1 << n):
            restricted = tuple(v for v in full_order if mask >> (v - 1) & 1)
            previous = answer.get(restricted)
            if previous is None or value < previous:
                answer[restricted] = value
    return answer


def certificate_ratio(
    d: list[list[float]], p: list[float]
) -> tuple[float, float, float, tuple[int, tuple[int, ...], float] | None]:
    n = len(p)
    costs, optimum_orders = subset_tsp_and_orders(d)
    post = posterior(costs, p)
    extensions = best_extension_table(d, p)
    value = 0.0
    worst_scenario = None
    for mask in range(1 << n):
        probability = subset_probability(mask, p)
        best_order = min(optimum_orders[mask], key=lambda order: extensions[order])
        extension = extensions[best_order]
        value += probability * extension
        excess = extension - costs[mask]
        if worst_scenario is None or excess > worst_scenario[2]:
            worst_scenario = (mask, best_order, excess)
    return value / post if post else 1.0, value, post, worst_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2029)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    alphabet = (0.01, 0.03, 0.08, 0.15, 0.3, 0.5, 0.75, 0.95)
    worst = (0.0, None)
    for trial in range(args.trials):
        d = random_graph_metric(args.n, rng)
        profiles = [[x] * args.n for x in alphabet]
        profiles.extend(
            [rng.choice(alphabet) for _ in range(args.n)] for _ in range(8)
        )
        for p in profiles:
            record = certificate_ratio(d, p)
            if record[0] > worst[0]:
                worst = (record[0], (trial, p, record, d))
                print("worst", worst, flush=True)
            if record[0] > 2.9 + 1e-8:
                print("COUNTEREXAMPLE", worst)
                return
    print("final worst", worst)


if __name__ == "__main__":
    main()
