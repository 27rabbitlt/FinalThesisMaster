#!/usr/bin/env python3
"""Fast state-summary search for recursive posterior buckets.

This is the same geometric-bucket construction as
``search_recursive_posterior_buckets.py``, but it propagates only expected
cost and first/last-active endpoint masses.  The concatenation identity makes
those summaries exact, avoiding an explicit distribution over permutations.
Floating-point arithmetic is used; this is an exploratory stress test.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
import math
import random

from search_recursive_posterior_buckets import stopped_cost_fast
from search_recursive_self_thinning import (
    all_subset_optima,
    posterior,
    random_graph_metric,
    tour_cost,
)


@dataclass(frozen=True)
class Summary:
    cost: float
    first: tuple[float, ...]
    last: tuple[float, ...]


def endpoint_masses(
    order: tuple[int, ...], p: list[float], n: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    first = [0.0] * n
    empty = 1.0
    for vertex in order:
        first[vertex - 1] = empty * p[vertex - 1]
        empty *= 1.0 - p[vertex - 1]
    last = [0.0] * n
    empty = 1.0
    for vertex in reversed(order):
        last[vertex - 1] = p[vertex - 1] * empty
        empty *= 1.0 - p[vertex - 1]
    return tuple(first), tuple(last)


def empty_probability(mask: int, p: list[float]) -> float:
    answer = 1.0
    for v, probability in enumerate(p):
        if mask >> v & 1:
            answer *= 1.0 - probability
    return answer


def recursive_summary(d: list[list[float]], p: list[float]) -> Summary:
    n = len(p)
    full = (1 << n) - 1
    spoke = [d[0][v + 1] for v in range(n)]
    saving = [
        [
            0.0
            if u == v
            else spoke[u] + spoke[v] - d[u + 1][v + 1]
            for v in range(n)
        ]
        for u in range(n)
    ]

    shortest: list[tuple[tuple[int, ...], ...]] = []
    for mask in range(1 << n):
        vertices = tuple(v + 1 for v in range(n) if mask >> v & 1)
        if not vertices:
            shortest.append((tuple(),))
            continue
        best = math.inf
        orders: list[tuple[int, ...]] = []
        for order in permutations(vertices):
            value = tour_cost(order, d)
            if value < best - 1e-10:
                best = value
                orders = [order]
            elif abs(value - best) <= 1e-10:
                orders.append(order)
        shortest.append(tuple(orders))

    @lru_cache(None)
    def order_data(order: tuple[int, ...]):
        first, last = endpoint_masses(order, p, n)
        return stopped_cost_fast(order, p, d), first, last

    @lru_cache(None)
    def solve(state: int) -> Summary:
        if state == 0:
            zeros = (0.0,) * n
            return Summary(0.0, zeros, zeros)
        empty_state = empty_probability(state, p)
        normalizer = 1.0 - empty_state
        total_cost = 0.0
        total_first = [0.0] * n
        total_last = [0.0] * n
        block = state
        while block:
            mass = 1.0
            for v, probability in enumerate(p):
                if state >> v & 1:
                    mass *= probability if block >> v & 1 else 1.0 - probability
            mass /= normalizer
            residual_mask = state ^ block
            residual = solve(residual_mask)
            empty_block = empty_probability(block, p)
            empty_residual = empty_probability(residual_mask, p)

            chosen = None
            chosen_value = math.inf
            for order in shortest[block]:
                local_cost, first, last = order_data(order)
                cross = sum(
                    last[u] * residual.first[v] * saving[u][v]
                    for u in range(n)
                    for v in range(n)
                )
                value = local_cost + residual.cost - cross
                if value < chosen_value - 1e-10 or (
                    abs(value - chosen_value) <= 1e-10
                    and (chosen is None or order < chosen[0])
                ):
                    chosen_value = value
                    chosen = (order, first, last)
            assert chosen is not None
            _, first, last = chosen
            total_cost += mass * chosen_value
            for v in range(n):
                total_first[v] += mass * (
                    first[v] + empty_block * residual.first[v]
                )
                total_last[v] += mass * (
                    residual.last[v] + empty_residual * last[v]
                )
            block = (block - 1) & state
        return Summary(total_cost, tuple(total_first), tuple(total_last))

    return solve(full)


def ratio(d: list[list[float]], p: list[float]) -> tuple[float, float, float]:
    costs, _ = all_subset_optima(d)
    post = posterior(costs, p)
    route = recursive_summary(d, p).cost
    return route / post if post else 1.0, route, post


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1009)
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
