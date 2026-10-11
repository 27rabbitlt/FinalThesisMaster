#!/usr/bin/env python3
"""Explore the sampled-posterior-order optimal-extension construction.

For a guide set B, let Ext(B) be the minimum exact p-stopped cost of a full
client order whose restriction to B is an optimal rooted TSP order on B.  We
compute

    Eext = E_B Ext(B)

exactly up to floating-point arithmetic, along with P, R, and the improvement
over the classical P+2R sampled-master-route certificate.  This is a
falsification/search script, not a proof checker.
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


def subset_probability(mask: int, p: list[float]) -> float:
    probability = 1.0
    for i, pi in enumerate(p):
        probability *= pi if mask >> i & 1 else 1.0 - pi
    return probability


def tour_cost(order: tuple[int, ...], d: list[list[float]]) -> float:
    if not order:
        return 0.0
    return (
        d[0][order[0]]
        + sum(d[u][v] for u, v in zip(order, order[1:]))
        + d[order[-1]][0]
    )


def stopped_cost(order: tuple[int, ...], p: list[float], d: list[list[float]]) -> float:
    """Exact expected shortcut cost, using its consecutive-active-pair form."""
    n = len(order)
    value = 0.0
    prefix_q = [1.0]
    for v in order:
        prefix_q.append(prefix_q[-1] * (1.0 - p[v - 1]))
    suffix_q = [1.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_q[i] = suffix_q[i + 1] * (1.0 - p[order[i] - 1])
    for i, u in enumerate(order):
        pu = p[u - 1]
        value += pu * (prefix_q[i] + suffix_q[i + 1]) * d[0][u]
        gap_q = 1.0
        for j in range(i + 1, n):
            v = order[j]
            value += pu * p[v - 1] * gap_q * d[u][v]
            gap_q *= 1.0 - p[v - 1]
    return value


def all_subset_optima(d: list[list[float]]) -> list[float]:
    n = len(d) - 1
    opt = [0.0] * (1 << n)
    for mask in range(1, 1 << n):
        vertices = tuple(i + 1 for i in range(n) if mask >> i & 1)
        opt[mask] = min(tour_cost(order, d) for order in permutations(vertices))
    return opt


def restriction(order: tuple[int, ...], mask: int) -> tuple[int, ...]:
    return tuple(v for v in order if mask >> (v - 1) & 1)


def values(d: list[list[float]], p: list[float]) -> dict[str, float]:
    n = len(p)
    opt = all_subset_optima(d)
    q = [subset_probability(mask, p) for mask in range(1 << n)]
    post = sum(q[mask] * opt[mask] for mask in range(1 << n))

    ext = [math.inf] * (1 << n)
    for order in permutations(range(1, n + 1)):
        stopped = stopped_cost(order, p, d)
        for mask in range(1 << n):
            suborder = restriction(order, mask)
            if abs(tour_cost(suborder, d) - opt[mask]) <= 1e-9:
                ext[mask] = min(ext[mask], stopped)
    assert all(math.isfinite(x) for x in ext)
    expected_ext = sum(q[mask] * ext[mask] for mask in range(1 << n))

    radius = 0.0
    for mask in range(1 << n):
        centers = [0] + [i + 1 for i in range(n) if mask >> i & 1]
        radius += q[mask] * sum(
            p[v - 1] * min(d[v][b] for b in centers)
            for v in range(1, n + 1)
        )
    deficit = post - radius
    baseline = post + 2.0 * radius
    required_gain = max(0.0, 0.1 * post - 2.0 * deficit)
    actual_gain = baseline - expected_ext
    return {
        "P": post,
        "R": radius,
        "D": deficit,
        "E": expected_ext,
        "E/P": expected_ext / post,
        "gain/P": actual_gain / post,
        "required_gain/P": required_gain / post,
        "slack/P": (2.9 * post - expected_ext) / post,
    }


def random_graph_metric(n: int, rng: random.Random) -> list[list[float]]:
    inf = 1e100
    w = [[0.0 if i == j else inf for j in range(n + 1)] for i in range(n + 1)]
    for v in range(1, n + 1):
        u = rng.randrange(v)
        w[u][v] = w[v][u] = rng.randint(1, 30)
    for u in range(n + 1):
        for v in range(u + 1, n + 1):
            if rng.random() < 0.4:
                w[u][v] = w[v][u] = min(w[u][v], rng.randint(1, 30))
    return floyd(w)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    alphabet = (0.02, 0.05, 0.1, 0.25, 0.5, 0.8, 0.97)
    worst = None
    tight_gain = None
    for trial in range(args.trials):
        d = random_graph_metric(args.n, rng)
        profiles = [[x] * args.n for x in alphabet]
        profiles.extend(
            [rng.choice(alphabet) for _ in range(args.n)] for _ in range(3)
        )
        for p in profiles:
            result = values(d, p)
            if worst is None or result["E/P"] > worst[0]:
                worst = (result["E/P"], trial, p, d, result)
                print("worst ratio", worst[:3], result, flush=True)
            gain_slack = result["gain/P"] - result["required_gain/P"]
            if result["D"] / result["P"] < 0.05 and (
                tight_gain is None or gain_slack < tight_gain[0]
            ):
                tight_gain = (gain_slack, trial, p, d, result)
                print("tight hard gain", tight_gain[:3], result, flush=True)
            if result["E/P"] > 2.9 + 1e-8:
                print("COUNTEREXAMPLE", result, "p", p)
                for row in d:
                    print(row)
                return
    print("FINAL worst", worst)
    print("FINAL tight hard gain", tight_gain)


if __name__ == "__main__":
    main()
