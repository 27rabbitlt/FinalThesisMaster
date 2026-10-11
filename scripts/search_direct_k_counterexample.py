#!/usr/bin/env python3
"""Search small instances against direct unvisited-row posterior rounding.

Exploration helper: calculations use floating point.  A final witness must be
rechecked separately in exact arithmetic.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations
import argparse
import random


def floyd(weight: list[list[float]]) -> list[list[float]]:
    d = [row[:] for row in weight]
    n = len(d)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    return d


def selected_posterior(d: list[list[float]], p: list[float]):
    n = len(p)
    y = [[0.0] * (n + 1) for _ in range(n + 1)]
    posterior = 0.0
    for mask in range(1 << n):
        probability = 1.0
        active = []
        for i in range(n):
            if mask >> i & 1:
                probability *= p[i]
                active.append(i + 1)
            else:
                probability *= 1.0 - p[i]
        if not active:
            continue
        best_cost = float("inf")
        best_order = None
        for order in permutations(active):
            cost = d[0][order[0]] + sum(d[u][v] for u, v in zip(order, order[1:])) + d[order[-1]][0]
            if cost < best_cost - 1e-10 or (abs(cost - best_cost) <= 1e-10 and (best_order is None or order < best_order)):
                best_cost = cost
                best_order = order
        posterior += probability * best_cost
        cycle = (0, *best_order, 0)
        for u, v in zip(cycle, cycle[1:]):
            y[u][v] += probability
            y[v][u] += probability
    return posterior, y


def direct_value(d: list[list[float]], p: list[float], y: list[list[float]]) -> float:
    n = len(p)
    full = (1 << n) - 1

    @lru_cache(None)
    def rec(remaining: int, current: int, last_active: int) -> float:
        if not remaining:
            return d[last_active][0]
        choices = [i for i in range(n) if remaining >> i & 1]
        denominator = sum(y[current][i + 1] for i in choices)
        if denominator <= 1e-200:
            raise ArithmeticError("zero remaining row")
        ans = 0.0
        for i in choices:
            v = i + 1
            choose = y[current][v] / denominator
            rest = remaining ^ (1 << i)
            ans += choose * (
                (1.0 - p[i]) * rec(rest, v, last_active)
                + p[i] * (d[last_active][v] + rec(rest, v, v))
            )
        return ans

    return rec(full, 0, 0)


def random_graph_metric(n: int, rng: random.Random) -> list[list[float]]:
    inf = 1e30
    w = [[0.0 if i == j else inf for j in range(n + 1)] for i in range(n + 1)]
    # First a random tree, then extra edges.
    for v in range(1, n + 1):
        u = rng.randrange(v)
        weight = rng.randint(1, 20)
        w[u][v] = w[v][u] = weight
    for u in range(n + 1):
        for v in range(u + 1, n + 1):
            if rng.random() < 0.25:
                weight = rng.randint(1, 20)
                w[u][v] = w[v][u] = min(w[u][v], weight)
    return floyd(w)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    profiles = [
        [x] * args.n for x in (0.02, 0.05, 0.1, 0.2, 0.5, 0.8)
    ]
    profiles.extend(
        [[rng.choice((0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.98)) for _ in range(args.n)] for _ in range(4)]
    )
    best = (0.0, None)
    for trial in range(args.trials):
        d = random_graph_metric(args.n, rng)
        for p in profiles:
            posterior, y = selected_posterior(d, p)
            value = direct_value(d, p, y)
            ratio = value / posterior
            if ratio > best[0]:
                best = (ratio, (trial, d, p, posterior, value))
                print("best", ratio, "trial", trial, "p", p, "P", posterior, "W", value, flush=True)
                if ratio > 2.9:
                    print("distance")
                    for row in d:
                        print(row)
                    return


if __name__ == "__main__":
    main()
