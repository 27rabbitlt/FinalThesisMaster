#!/usr/bin/env python3
"""Explore a parameter-free recursive posterior-bucket order.

For a current client set S, repeatedly draw a product-p subset until it is
nonempty, put a shortest posterior path on that subset first, and recurse on
the complement.  Among shortest paths, choose the orientation/tie-break that
minimizes the exact expected stopped cost after concatenation with the
already-defined residual-order distribution.  This script evaluates the
resulting distribution over full orders exactly (up to floating arithmetic)
for small instances.  It is exploratory, not a proof certificate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
from itertools import permutations
import math
import random

from search_recursive_self_thinning import (
    all_subset_optima,
    floyd,
    posterior,
    random_graph_metric,
    tour_cost,
)


def stopped_cost_fast(
    order: tuple[int, ...], probabilities: list[float], d: list[list[float]]
) -> float:
    """Exact stopped-product formula in O(len(order)^2) arithmetic."""
    value = 0.0
    prefix_empty = 1.0
    for vertex in order:
        probability = probabilities[vertex - 1]
        value += prefix_empty * probability * d[0][vertex]
        prefix_empty *= 1.0 - probability
    suffix_empty = 1.0
    for vertex in reversed(order):
        probability = probabilities[vertex - 1]
        value += probability * suffix_empty * d[vertex][0]
        suffix_empty *= 1.0 - probability
    for i, u in enumerate(order):
        skipped = 1.0
        for v in order[i + 1 :]:
            value += (
                probabilities[u - 1]
                * probabilities[v - 1]
                * skipped
                * d[u][v]
            )
            skipped *= 1.0 - probabilities[v - 1]
    return value


def shortest_orders(mask: int, d: list[list[float]]) -> list[tuple[int, ...]]:
    vertices = tuple(v + 1 for v in range(len(d) - 1) if mask >> v & 1)
    if not vertices:
        return [tuple()]
    best = math.inf
    answer: list[tuple[int, ...]] = []
    for order in permutations(vertices):
        value = tour_cost(order, d)
        if value < best - 1e-10:
            best = value
            answer = [order]
        elif abs(value - best) <= 1e-10:
            answer.append(order)
    return answer


def recursive_distribution(
    d: list[list[float]], p: list[float]
) -> dict[tuple[int, ...], float]:
    n = len(p)
    shortest = [shortest_orders(mask, d) for mask in range(1 << n)]

    @lru_cache(None)
    def eval_order(order: tuple[int, ...]) -> float:
        return stopped_cost_fast(order, p, d)

    @lru_cache(None)
    def distribution(state: int) -> tuple[tuple[tuple[int, ...], float], ...]:
        if state == 0:
            return ((tuple(), 1.0),)
        empty = 1.0
        for v in range(n):
            if state >> v & 1:
                empty *= 1.0 - p[v]
        normalizer = 1.0 - empty
        assert normalizer > 0.0
        answer: defaultdict[tuple[int, ...], float] = defaultdict(float)
        block = state
        while block:
            block_mass = 1.0
            for v in range(n):
                if state >> v & 1:
                    block_mass *= p[v] if block >> v & 1 else 1.0 - p[v]
            block_mass /= normalizer
            residual = state ^ block
            residual_dist = distribution(residual)
            best_value = math.inf
            best_prefix: tuple[int, ...] | None = None
            for prefix in shortest[block]:
                value = sum(
                    mass * eval_order(prefix + suffix)
                    for suffix, mass in residual_dist
                )
                if value < best_value - 1e-10 or (
                    abs(value - best_value) <= 1e-10
                    and (best_prefix is None or prefix < best_prefix)
                ):
                    best_value = value
                    best_prefix = prefix
            assert best_prefix is not None
            for suffix, mass in residual_dist:
                answer[best_prefix + suffix] += block_mass * mass
            block = (block - 1) & state
        assert abs(sum(answer.values()) - 1.0) < 1e-8
        return tuple(answer.items())

    return dict(distribution((1 << n) - 1))


def ratio(d: list[list[float]], p: list[float]) -> tuple[float, float, float]:
    costs, _ = all_subset_optima(d)
    post = posterior(costs, p)
    order_dist = recursive_distribution(d, p)
    value = sum(mass * stopped_cost_fast(order, p, d) for order, mass in order_dist.items())
    return value / post if post else 1.0, value, post


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=991)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    alphabet = (0.03, 0.08, 0.15, 0.3, 0.5, 0.75, 0.95)
    worst = (0.0, None)
    for trial in range(args.trials):
        d = random_graph_metric(args.n, rng)
        profiles = [[x] * args.n for x in alphabet]
        profiles.extend(
            [rng.choice(alphabet) for _ in range(args.n)] for _ in range(6)
        )
        for p in profiles:
            value, route, post = ratio(d, p)
            if value > worst[0]:
                worst = (value, (trial, p, route, post, d))
                print("worst", worst, flush=True)
            if value > 2.9 + 1e-8:
                print("COUNTEREXAMPLE", worst)
                return
    print("final worst", worst)


if __name__ == "__main__":
    main()
