#!/usr/bin/env python3
"""Search the recursive posterior-block inequality on small metric instances.

This is an exploratory floating-point search, not a proof checker.  For a
probability vector p, let P(p) be the expected optimal posterior tour cost and
let Q(p) be the expected p-stopped cost of a shortest tour on an independent
p-sample B.  The recursive block construction would yield a c-approximation
if

    Q(p) <= c * (P(p) - P(p * (1-p))).

For every sampled set B we minimize stopped cost among all shortest tours on
B, since this tie-breaking is available to the construction.
"""

from __future__ import annotations

from itertools import permutations
import argparse
import math
import random


def floyd(weights: list[list[float]]) -> list[list[float]]:
    d = [row[:] for row in weights]
    for k in range(len(d)):
        for i in range(len(d)):
            for j in range(len(d)):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    return d


def subset_probability(mask: int, probabilities: list[float]) -> float:
    value = 1.0
    for i, p in enumerate(probabilities):
        value *= p if mask >> i & 1 else 1.0 - p
    return value


def tour_cost(order: tuple[int, ...], d: list[list[float]]) -> float:
    if not order:
        return 0.0
    return (
        d[0][order[0]]
        + sum(d[u][v] for u, v in zip(order, order[1:]))
        + d[order[-1]][0]
    )


def stopped_cost(
    order: tuple[int, ...], probabilities: list[float], d: list[list[float]]
) -> float:
    value = 0.0
    for active_mask in range(1 << len(order)):
        probability = 1.0
        active: list[int] = []
        for j, vertex in enumerate(order):
            p = probabilities[vertex - 1]
            if active_mask >> j & 1:
                probability *= p
                active.append(vertex)
            else:
                probability *= 1.0 - p
        value += probability * tour_cost(tuple(active), d)
    return value


def all_subset_optima(
    d: list[list[float]], tie_probabilities: list[float] | None = None
) -> tuple[list[float], list[tuple[int, ...]]]:
    n = len(d) - 1
    costs = [0.0] * (1 << n)
    orders: list[tuple[int, ...]] = [tuple()] * (1 << n)
    for mask in range(1, 1 << n):
        vertices = tuple(i + 1 for i in range(n) if mask >> i & 1)
        best_cost = math.inf
        best_stopped = math.inf
        best_order: tuple[int, ...] | None = None
        for order in permutations(vertices):
            raw = tour_cost(order, d)
            if raw > best_cost + 1e-10:
                continue
            stopped = (
                stopped_cost(order, tie_probabilities, d)
                if tie_probabilities is not None
                else 0.0
            )
            if (
                raw < best_cost - 1e-10
                or stopped < best_stopped - 1e-10
                or (
                    abs(raw - best_cost) <= 1e-10
                    and abs(stopped - best_stopped) <= 1e-10
                    and (best_order is None or order < best_order)
                )
            ):
                best_cost = raw
                best_stopped = stopped
                best_order = order
        assert best_order is not None
        costs[mask] = best_cost
        orders[mask] = best_order
    return costs, orders


def posterior(costs: list[float], probabilities: list[float]) -> float:
    return sum(
        subset_probability(mask, probabilities) * cost
        for mask, cost in enumerate(costs)
    )


def ratio(d: list[list[float]], p: list[float]) -> tuple[float, float, float, float]:
    costs, best_orders = all_subset_optima(d, p)
    p_now = posterior(costs, p)
    p_residual = posterior(costs, [x * (1.0 - x) for x in p])
    q = sum(
        subset_probability(mask, p) * stopped_cost(best_orders[mask], p, d)
        for mask in range(1 << len(p))
    )
    drop = p_now - p_residual
    return q / drop, q, drop, p_now


def random_graph_metric(n: int, rng: random.Random) -> list[list[float]]:
    inf = 1e100
    w = [[0.0 if i == j else inf for j in range(n + 1)] for i in range(n + 1)]
    for v in range(1, n + 1):
        u = rng.randrange(v)
        length = rng.randint(1, 30)
        w[u][v] = w[v][u] = length
    for u in range(n + 1):
        for v in range(u + 1, n + 1):
            if rng.random() < 0.35:
                length = rng.randint(1, 30)
                w[u][v] = w[v][u] = min(w[u][v], length)
    return floyd(w)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    alphabet = (0.01, 0.03, 0.08, 0.2, 0.5, 0.8, 0.97)
    best = 0.0
    for trial in range(args.trials):
        d = random_graph_metric(args.n, rng)
        profiles = [[x] * args.n for x in alphabet]
        profiles.extend(
            [rng.choice(alphabet) for _ in range(args.n)] for _ in range(5)
        )
        for p in profiles:
            value, q, drop, post = ratio(d, p)
            if value > best:
                best = value
                print(
                    "best", value, "trial", trial, "p", p,
                    "Q", q, "drop", drop, "P", post,
                    flush=True,
                )
            if value > 2.9 + 1e-8:
                print("distance")
                for row in d:
                    print(row)
                return


if __name__ == "__main__":
    main()
