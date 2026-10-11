#!/usr/bin/env python3
"""Exploratory search for posterior-selector inherited-portal hierarchies.

For each realization, a deterministic optimal posterior tour is computed and
then oriented by a fair coin.  For every nonempty client subset C, ``mu_C``
is the conditional marginal of the first active member of C in that global
oriented order.  These marginals obey portal inheritance.  A subset DP finds
the cheapest *binary* laminar hierarchy, using exact edgewise transportation
LPs between every child and its parent.

This is an exploratory floating-point search, not a proof certificate.
"""

from __future__ import annotations

import argparse
import itertools
import random
from functools import lru_cache

import numpy as np
from scipy.optimize import linprog

from search_hazard_tree_ratio import metric_closure


def tours_with_orders(d: list[list[float]]) -> tuple[list[float], list[tuple[int, ...]]]:
    n = len(d) - 1
    root = n
    size = 1 << n
    costs = [0.0] * size
    orders: list[tuple[int, ...]] = [tuple() for _ in range(size)]
    for mask in range(1, size):
        verts = tuple(v for v in range(n) if mask & (1 << v))
        best_cost = float("inf")
        best_order: tuple[int, ...] | None = None
        for order in itertools.permutations(verts):
            cost = d[root][order[0]] + d[order[-1]][root]
            cost += sum(d[a][b] for a, b in zip(order, order[1:]))
            if cost < best_cost - 1e-12:
                best_cost = cost
                best_order = order
        costs[mask] = best_cost
        orders[mask] = best_order or tuple()
    return costs, orders


def evaluate(d: list[list[float]], p: list[float]) -> tuple[float, float, float]:
    n = len(p)
    full = (1 << n) - 1
    tours, orders = tours_with_orders(d)
    scenario_prob = [0.0] * (1 << n)
    posterior = 0.0
    for mask in range(1 << n):
        prob = 1.0
        for v, pv in enumerate(p):
            prob *= pv if mask & (1 << v) else 1.0 - pv
        scenario_prob[mask] = prob
        posterior += prob * tours[mask]

    q = [0.0] * (1 << n)
    mu: list[list[float]] = [[0.0] * n for _ in range(1 << n)]
    for c in range(1, 1 << n):
        q[c] = 1.0
        for v, pv in enumerate(p):
            if c & (1 << v):
                q[c] *= 1.0 - pv
        q[c] = 1.0 - q[c]
        h = [0.0] * n
        for a in range(1, 1 << n):
            inter = a & c
            if not inter:
                continue
            restricted = [v for v in orders[a] if inter & (1 << v)]
            h[restricted[0]] += 0.5 * scenario_prob[a]
            h[restricted[-1]] += 0.5 * scenario_prob[a]
        mu[c] = [x / q[c] for x in h]

    @lru_cache(maxsize=None)
    def transport(child: int, parent: int) -> float:
        outside = [u for u in range(n) if parent & (1 << u) and not child & (1 << u)]
        inside = [v for v in range(n) if child & (1 << v)]
        rows = [mu[parent][u] for u in outside]
        cols = [mu[child][v] - mu[parent][v] for v in inside]
        rows = [max(0.0, x) for x in rows]
        cols = [max(0.0, x) for x in cols]
        mass = sum(rows)
        if mass < 1e-12:
            return 0.0
        # Repair only roundoff in the last column.
        cols[-1] += mass - sum(cols)
        cvec = np.array([d[u][v] for u in outside for v in inside])
        aeq = []
        beq = []
        for i in range(len(outside)):
            row = np.zeros(len(cvec))
            row[i * len(inside):(i + 1) * len(inside)] = 1.0
            aeq.append(row)
            beq.append(rows[i])
        for j in range(len(inside)):
            row = np.zeros(len(cvec))
            row[j::len(inside)] = 1.0
            aeq.append(row)
            beq.append(cols[j])
        result = linprog(cvec, A_eq=np.array(aeq), b_eq=np.array(beq),
                         bounds=(0.0, None), method="highs")
        if not result.success:
            raise RuntimeError(result.message)
        return float(result.fun)

    dp = [0.0] * (1 << n)
    for mask in sorted(range(1, 1 << n), key=int.bit_count):
        if mask.bit_count() == 1:
            continue
        anchor = mask & -mask
        best = float("inf")
        a = (mask - 1) & mask
        while a:
            if a & anchor:
                b = mask ^ a
                if b:
                    value = (
                        dp[a] + q[a] * transport(a, mask)
                        + dp[b] + q[b] * transport(b, mask)
                    )
                    best = min(best, value)
            a = (a - 1) & mask
        dp[mask] = best

    root = n
    root_cost = q[full] * sum(mu[full][v] * d[root][v] for v in range(n))
    portal = root_cost + dp[full]
    return posterior, portal, portal / posterior


def random_metric(n: int, rng: random.Random) -> list[list[float]]:
    total = n + 1
    weights = [[0.0] * total for _ in range(total)]
    for i in range(total):
        for j in range(i + 1, total):
            weights[i][j] = weights[j][i] = rng.uniform(0.05, 2.0)
    return metric_closure(weights)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    best = (0.0, None)
    for trial in range(args.trials):
        d = random_metric(args.n, rng)
        p = [rng.uniform(0.03, 0.97) for _ in range(args.n)]
        result = evaluate(d, p)
        if result[2] > best[0]:
            best = (result[2], (trial, p, result, d))
            print("best", best[0], "trial", trial, "P,portal", result[:2])
    print("final", best)


if __name__ == "__main__":
    main()
