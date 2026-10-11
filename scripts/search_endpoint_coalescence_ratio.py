#!/usr/bin/env python3
"""Exact finite-state search for the binary endpoint-coalescence objective.

For a metric on one depot (index 0) and n clients (indices 1,...,n), this
script computes

    P = E[ optimal depot tour through the active clients ]

by Held--Karp, and enumerates every unordered full binary hierarchy on the
clients to compute the minimum value of equations (11)--(13) in
``coalescence_route_notes.md``.  Arithmetic in the evaluator is floating
point; a finite witness found here should be recertified separately with
Fractions before it is used as a counterexample.

The default mode performs reproducible random stress tests on shortest-path
metrics.  ``--self-test`` checks the recurrences against direct enumeration
of child swaps and active sets on tiny instances.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
from functools import lru_cache
from typing import Iterable, Sequence


Tree = int | tuple["Tree", "Tree"]


def bits(mask: int) -> Iterable[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


@lru_cache(maxsize=None)
def hierarchies(mask: int) -> tuple[Tree, ...]:
    """All non-plane full binary trees on the leaves in ``mask``."""
    if mask & (mask - 1) == 0:
        return (mask.bit_length() - 1,)
    anchor = mask & -mask
    out: list[Tree] = []
    sub = (mask - 1) & mask
    while sub:
        other = mask ^ sub
        if sub & anchor and other:
            for left in hierarchies(sub):
                for right in hierarchies(other):
                    out.append((left, right))
        sub = (sub - 1) & mask
    return tuple(out)


def floyd_warshall(d: list[list[float]]) -> list[list[float]]:
    d = [row[:] for row in d]
    for k in range(len(d)):
        for i in range(len(d)):
            for j in range(len(d)):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    return d


def subset_probabilities(p: Sequence[float]) -> list[float]:
    n = len(p)
    out = [0.0] * (1 << n)
    for mask in range(1 << n):
        value = 1.0
        for i in range(n):
            value *= p[i] if mask >> i & 1 else 1.0 - p[i]
        out[mask] = value
    return out


def posterior_tour_costs(d: Sequence[Sequence[float]], n: int) -> list[float]:
    """Held--Karp optimum for every subset of n clients."""
    inf = float("inf")
    dp = [[inf] * n for _ in range(1 << n)]
    for j in range(n):
        dp[1 << j][j] = d[0][j + 1]
    for mask in range(1, 1 << n):
        if mask & (mask - 1) == 0:
            continue
        for j in bits(mask):
            prev = mask ^ (1 << j)
            dp[mask][j] = min(
                dp[prev][i] + d[i + 1][j + 1] for i in bits(prev)
            )
    ans = [0.0] * (1 << n)
    for mask in range(1, 1 << n):
        ans[mask] = min(dp[mask][j] + d[j + 1][0] for j in bits(mask))
    return ans


def posterior_tours(
    d: Sequence[Sequence[float]], n: int
) -> tuple[list[float], list[tuple[int, ...]]]:
    """Held--Karp costs and one deterministically tie-broken client order."""
    inf = float("inf")
    dp = [[inf] * n for _ in range(1 << n)]
    paths: list[list[tuple[int, ...] | None]] = [
        [None] * n for _ in range(1 << n)
    ]
    for j in range(n):
        dp[1 << j][j] = d[0][j + 1]
        paths[1 << j][j] = (j,)
    for mask in range(1, 1 << n):
        if mask & (mask - 1) == 0:
            continue
        for j in bits(mask):
            prev = mask ^ (1 << j)
            candidates = [
                (dp[prev][i] + d[i + 1][j + 1], paths[prev][i] + (j,))
                for i in bits(prev)
                if paths[prev][i] is not None
            ]
            dp[mask][j], paths[mask][j] = min(candidates)
    costs = [0.0] * (1 << n)
    tours: list[tuple[int, ...]] = [tuple() for _ in range(1 << n)]
    for mask in range(1, 1 << n):
        candidates = [
            (dp[mask][j] + d[j + 1][0], paths[mask][j]) for j in bits(mask)
        ]
        costs[mask], tours[mask] = min(candidates)
    return costs, tours


def sampled_voronoi_bound(
    d: Sequence[Sequence[float]], p: Sequence[float]
) -> tuple[float, float, float]:
    """Return (Q+2R,Q,R) for the sampled posterior-tour Voronoi construction.

    A guide realization B is sampled.  Every client is assigned to its nearest
    point of B union the depot, and occupied cells are shortcut in the chosen
    optimal tour of B.  Q is that expected center-tour cost and R is the usual
    expected nearest-guide attachment cost.
    """
    n = len(p)
    probs = subset_probabilities(p)
    _, tours = posterior_tours(d, n)
    qtotal = 0.0
    rtotal = 0.0
    for guide, pg in enumerate(probs):
        if pg == 0.0:
            continue
        centers = (-1,) + tuple(bits(guide))  # -1 denotes the depot.
        assigned = []
        for v in range(n):
            center = min(centers, key=lambda u: (d[v + 1][u + 1], u))
            assigned.append(center)
            rtotal += pg * p[v] * d[v + 1][center + 1]
        order = tours[guide]
        for actual, pa in enumerate(probs):
            if pa == 0.0:
                continue
            occupied = {assigned[v] for v in bits(actual)}
            kept = [v for v in order if v in occupied]
            if not kept:
                continue
            cost = d[0][kept[0] + 1] + d[kept[-1] + 1][0]
            cost += sum(d[u + 1][v + 1] for u, v in zip(kept, kept[1:]))
            qtotal += pg * pa * cost
    return qtotal + 2.0 * rtotal, qtotal, rtotal


def expected_posterior(d: Sequence[Sequence[float]], p: Sequence[float]) -> float:
    probs = subset_probabilities(p)
    costs = posterior_tour_costs(d, len(p))
    return sum(a * b for a, b in zip(probs, costs))


def occupancy(mask: int, p: Sequence[float]) -> float:
    empty = 1.0
    for i in bits(mask):
        empty *= 1.0 - p[i]
    return 1.0 - empty


def hierarchy_value(
    tree: Tree, d: Sequence[Sequence[float]], p: Sequence[float]
) -> tuple[int, float, tuple[float, ...], float]:
    """Return (leaf mask, occupancy, conditional endpoint law, open cost)."""
    n = len(p)
    if isinstance(tree, int):
        nu = [0.0] * n
        nu[tree] = 1.0
        return 1 << tree, p[tree], tuple(nu), 0.0
    ma, qa, nua, ella = hierarchy_value(tree[0], d, p)
    mb, qb, nub, ellb = hierarchy_value(tree[1], d, p)
    qu = qa + qb - qa * qb
    aa = qa * (2.0 - qb) / (2.0 * qu)
    bb = qb * (2.0 - qa) / (2.0 * qu)
    nu = tuple(aa * nua[i] + bb * nub[i] for i in range(n))
    cross = sum(
        nua[i] * nub[j] * d[i + 1][j + 1]
        for i in range(n)
        for j in range(n)
    )
    return ma | mb, qu, nu, ella + ellb + qa * qb * cross


def coalescence_value(
    tree: Tree, d: Sequence[Sequence[float]], p: Sequence[float]
) -> float:
    _, q, nu, ell = hierarchy_value(tree, d, p)
    depot = sum(nu[i] * d[0][i + 1] for i in range(len(p)))
    return ell + 2.0 * q * depot


def min_coalescence(
    d: Sequence[Sequence[float]], p: Sequence[float]
) -> tuple[float, Tree]:
    hs = hierarchies((1 << len(p)) - 1)
    best_tree = hs[0]
    best = coalescence_value(best_tree, d, p)
    for tree in hs[1:]:
        value = coalescence_value(tree, d, p)
        if value < best:
            best, best_tree = value, tree
    return best, best_tree


def all_orders(tree: Tree) -> tuple[tuple[int, ...], ...]:
    if isinstance(tree, int):
        return ((tree,),)
    left, right = all_orders(tree[0]), all_orders(tree[1])
    out = []
    for a in left:
        for b in right:
            out.extend((a + b, b + a))
    return tuple(out)


def direct_expected_cost(
    tree: Tree, d: Sequence[Sequence[float]], p: Sequence[float]
) -> float:
    probs = subset_probabilities(p)
    total = 0.0
    orders = all_orders(tree)
    for order in orders:
        for mask, prob in enumerate(probs):
            active = [i for i in order if mask >> i & 1]
            if not active:
                continue
            cost = d[0][active[0] + 1] + d[active[-1] + 1][0]
            cost += sum(d[i + 1][j + 1] for i, j in zip(active, active[1:]))
            total += prob * cost / len(orders)
    return total


def random_graph_metric(rng: random.Random, n: int) -> list[list[float]]:
    size = n + 1
    d = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            # Log-uniform edges create both clusters and expensive separators.
            value = math.exp(rng.uniform(-2.0, 2.0))
            d[i][j] = d[j][i] = value
    return floyd_warshall(d)


def self_test() -> None:
    rng = random.Random(731921)
    for n in range(2, 6):
        d = random_graph_metric(rng, n)
        p = [rng.uniform(0.05, 0.95) for _ in range(n)]
        for tree in hierarchies((1 << n) - 1):
            recursive = coalescence_value(tree, d, p)
            direct = direct_expected_cost(tree, d, p)
            if abs(recursive - direct) > 2e-10 * max(1.0, direct):
                raise AssertionError((n, tree, recursive, direct))
    # Unit metric: every hierarchy has exact posterior value.
    for n in range(2, 8):
        d = [[float(i != j) for j in range(n + 1)] for i in range(n + 1)]
        p = [rng.uniform(0.01, 0.99) for _ in range(n)]
        post = expected_posterior(d, p)
        for tree in hierarchies((1 << n) - 1):
            value = coalescence_value(tree, d, p)
            if abs(value - post) > 2e-10 * max(1.0, post):
                raise AssertionError(("unit", n, tree, value, post))
    print("self-test passed")


def search(seed: int, n: int, trials: int) -> None:
    rng = random.Random(seed)
    record = (0.0, None, None, None, None)
    for trial in range(1, trials + 1):
        d = random_graph_metric(rng, n)
        # Logit-uniform probabilities emphasize heterogeneous occupancy scales.
        p = [1.0 / (1.0 + math.exp(-rng.uniform(-5.0, 5.0))) for _ in range(n)]
        post = expected_posterior(d, p)
        coal, tree = min_coalescence(d, p)
        ratio = coal / post if post else 1.0
        if ratio > record[0]:
            record = (ratio, d, p, post, (coal, tree))
            print(
                f"trial={trial} ratio={ratio:.12f} P={post:.12g} "
                f"C={coal:.12g} tree={tree}",
                flush=True,
            )
    ratio, d, p, post, result = record
    print("record ratio", ratio)
    print("p", p)
    print("d")
    for row in d:
        print(row)
    print("P", post, "C/tree", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--trials", type=int, default=1000)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        search(args.seed, args.clients, args.trials)


if __name__ == "__main__":
    main()
