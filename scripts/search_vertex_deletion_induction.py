#!/usr/bin/env python3
"""Explore a vertex-deletion induction for the symmetric stochastic TSP.

For each small metric instance, compute exact brute-force posterior and
a-priori values (in floating point) and inspect

    (F(p)-F(p with p_v=0)) / (P(p)-P(p with p_v=0)).

If some vertex always has ratio at most 2.9, minimal-counterexample induction
would prove the desired theorem.  This program only searches for obstructions.
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


def tour_cost(order: tuple[int, ...], d: list[list[float]]) -> float:
    if not order:
        return 0.0
    return d[0][order[0]] + sum(
        d[u][v] for u, v in zip(order, order[1:])
    ) + d[order[-1]][0]


def subset_prob(mask: int, p: list[float]) -> float:
    value = 1.0
    for i, probability in enumerate(p):
        value *= probability if mask >> i & 1 else 1.0 - probability
    return value


def posterior(d: list[list[float]], p: list[float]) -> float:
    n = len(p)
    value = 0.0
    for mask in range(1, 1 << n):
        active = tuple(i + 1 for i in range(n) if mask >> i & 1)
        optimum = min(tour_cost(order, d) for order in permutations(active))
        value += subset_prob(mask, p) * optimum
    return value


def stopped_cost(order: tuple[int, ...], d: list[list[float]], p: list[float]) -> float:
    n = len(order)
    value = 0.0
    for mask in range(1, 1 << n):
        active = tuple(order[j] for j in range(n) if mask >> j & 1)
        probability = 1.0
        for j, vertex in enumerate(order):
            probability *= p[vertex - 1] if mask >> j & 1 else 1.0 - p[vertex - 1]
        value += probability * tour_cost(active, d)
    return value


def priori(d: list[list[float]], p: list[float]) -> float:
    return min(
        stopped_cost(order, d, p)
        for order in permutations(range(1, len(p) + 1))
    )


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
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    alphabet = (0.03, 0.1, 0.3, 0.5, 0.8, 0.97)
    worst_min = 0.0
    for trial in range(args.trials):
        d = random_graph_metric(args.n, rng)
        profiles = [[x] * args.n for x in alphabet]
        profiles.extend(
            [rng.choice(alphabet) for _ in range(args.n)] for _ in range(3)
        )
        for p in profiles:
            post = posterior(d, p)
            prior = priori(d, p)
            deletion_ratios = []
            for v in range(args.n):
                p_minus = p[:]
                p_minus[v] = 0.0
                post_minus = posterior(d, p_minus)
                prior_minus = priori(d, p_minus)
                denominator = post - post_minus
                if denominator > 1e-10:
                    deletion_ratios.append((prior - prior_minus) / denominator)
            minimum = min(deletion_ratios, default=-math.inf)
            if minimum > worst_min:
                worst_min = minimum
                print(
                    "worst minimum", minimum, "F/P", prior / post,
                    "trial", trial, "p", p, flush=True,
                )
            if minimum > 2.9 + 1e-8:
                print("distance")
                for row in d:
                    print(row)
                return


if __name__ == "__main__":
    main()
