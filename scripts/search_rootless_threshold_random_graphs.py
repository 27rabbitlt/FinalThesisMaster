#!/usr/bin/env python3
"""Fast adversarial search for the rootless compression scale inequality.

This is exploratory.  It samples connected unweighted graphs, uses their
shortest-path metrics at threshold 3/2, and searches heterogeneous activation
probabilities for a positive value of a_t+r_t-2 E k_t.  Conditional attachment
is evaluated directly from the rootless Voronoi-cell partition, in O(n 2^n)
time rather than enumerating the retained center set.
"""

from __future__ import annotations

import argparse
import math
import random

from scipy.optimize import differential_evolution


def metric_from_edges(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    size = n + 1
    inf = size + 1
    d = [[0 if i == j else inf for j in range(size)] for i in range(size)]
    for i, j in edges:
        d[i][j] = d[j][i] = 1
    for k in range(size):
        for i in range(size):
            for j in range(size):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    return d


def random_connected_metric(n: int, rng: random.Random) -> tuple[list[list[int]], list[tuple[int, int]]]:
    size = n + 1
    order = list(range(size))
    rng.shuffle(order)
    edges = []
    for index in range(1, size):
        edges.append((order[index], order[rng.randrange(index)]))
    density = rng.uniform(0.05, 0.75)
    present = {tuple(sorted(edge)) for edge in edges}
    for i in range(size):
        for j in range(i + 1, size):
            if (i, j) not in present and rng.random() < density:
                edges.append((i, j))
                present.add((i, j))
    return metric_from_edges(n, edges), sorted(present)


def bad_tours(n: int, d: list[list[int]]) -> list[int]:
    root = n
    answer = [0] * (1 << n)
    for mask in range(1, 1 << n):
        dp: dict[tuple[int, int], int] = {}
        for last in range(n):
            if mask & (1 << last):
                dp[1 << last, last] = int(d[root][last] > 1)
        for sub in range(1, 1 << n):
            if sub & ~mask or not sub & (sub - 1):
                continue
            for last in range(n):
                if sub & (1 << last):
                    before_mask = sub ^ (1 << last)
                    dp[sub, last] = min(
                        dp[before_mask, before] + int(d[before][last] > 1)
                        for before in range(n)
                        if before_mask & (1 << before)
                    )
        answer[mask] = min(
            dp[mask, last] + int(d[last][root] > 1)
            for last in range(n)
            if mask & (1 << last)
        )
    return answer


class Instance:
    def __init__(self, n: int, d: list[list[int]], edges: list[tuple[int, int]]):
        self.n = n
        self.d = d
        self.edges = edges
        self.bad = bad_tours(n, d)
        self.root_far = [d[v][n] > 1 for v in range(n)]
        self.neighborhood = [
            sum(1 << u for u in range(n) if d[v][u] <= 1)
            for v in range(n)
        ]
        # cover[B][v] is the union of rootless B-Voronoi cells whose centers
        # threshold-cover v.  T misses v iff C avoids precisely this site set.
        self.cover = [[0] * n for _ in range(1 << n)]
        for guide in range(1, 1 << n):
            centers = [b for b in range(n) if guide & (1 << b)]
            cell = {b: 0 for b in centers}
            for x in range(n):
                b = min(centers, key=lambda center: (d[x][center], center))
                cell[b] |= 1 << x
            for v in range(n):
                self.cover[guide][v] = sum(
                    cell[b] for b in centers if self.neighborhood[v] & (1 << b)
                )

    def terms(self, p: tuple[float, ...] | list[float]) -> tuple[float, float, float]:
        n = self.n
        weights = [1.0] * (1 << n)
        for mask in range(1 << n):
            value = 1.0
            for v in range(n):
                value *= p[v] if mask & (1 << v) else 1.0 - p[v]
            weights[mask] = value
        expected_bad = sum(weights[mask] * self.bad[mask] for mask in range(1 << n))
        radius = 0.0
        for v in range(n):
            if not self.root_far[v]:
                continue
            miss = 1.0
            for x in range(n):
                if self.neighborhood[v] & (1 << x):
                    miss *= 1.0 - p[x]
            radius += p[v] * miss
        attachment = 0.0
        for guide in range(1 << n):
            conditional = 0.0
            for v in range(n):
                if not self.root_far[v]:
                    continue
                miss = 1.0
                cover = self.cover[guide][v]
                for x in range(n):
                    if cover & (1 << x):
                        miss *= 1.0 - p[x]
                conditional += p[v] * miss
            attachment += weights[guide] * conditional
        return attachment, radius, expected_bad

    def gap(self, p: tuple[float, ...] | list[float]) -> float:
        a, r, k = self.terms(p)
        return a + r - 2.0 * k


def random_probability(rng: random.Random) -> float:
    mode = rng.randrange(4)
    if mode == 0:
        return rng.uniform(0.001, 0.999)
    if mode == 1:
        return 1.0 / (1.0 + math.exp(-rng.uniform(-7.0, 7.0)))
    if mode == 2:
        return rng.betavariate(0.3, 0.3)
    return rng.choice((0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.99))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--graphs", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--optimize", type=int, default=20)
    parser.add_argument("--seed", type=int, default=9173)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    candidates: list[tuple[float, Instance, tuple[float, ...]]] = []
    best = (-math.inf, None, None)
    best_attachment = (-math.inf, None, None)
    for graph_index in range(args.graphs):
        d, edges = random_connected_metric(args.clients, rng)
        instance = Instance(args.clients, d, edges)
        if not any(instance.root_far):
            continue
        local = (-math.inf, None)
        for _ in range(args.samples):
            p = tuple(random_probability(rng) for _ in range(args.clients))
            gap = instance.gap(p)
            if gap > local[0]:
                local = (gap, p)
            if gap > best[0]:
                best = (gap, instance, p)
            a, r, k = instance.terms(p)
            if a - k > best_attachment[0]:
                best_attachment = (a - k, instance, p)
        candidates.append((local[0], instance, local[1]))
        if local[0] > 1e-9:
            a, r, k = instance.terms(local[1])
            print("COUNTEREXAMPLE", local[0], edges, local[1], a, r, k, d)
            return
        if graph_index % 100 == 0:
            print("progress", graph_index, "best", best[0], flush=True)
    candidates.sort(key=lambda row: row[0], reverse=True)
    for _, instance, initial in candidates[: args.optimize]:
        result = differential_evolution(
            lambda values: -instance.gap(values),
            [(1e-5, 1 - 1e-5)] * args.clients,
            seed=rng.randrange(1 << 30),
            maxiter=250,
            popsize=12,
            polish=True,
            updating="immediate",
        )
        gap = -result.fun
        if gap > best[0]:
            best = (gap, instance, tuple(result.x))
        if gap > 1e-9:
            a, r, k = instance.terms(result.x)
            print(
                "OPTIMIZED COUNTEREXAMPLE",
                gap,
                instance.edges,
                tuple(result.x),
                a,
                r,
                k,
                instance.d,
            )
            return
    gap, instance, p = best
    print("no violation; best", gap, instance.edges, p, instance.terms(p), instance.d)
    gap, instance, p = best_attachment
    print("best a-Ek", gap, instance.edges, p, instance.terms(p), instance.d)


if __name__ == "__main__":
    main()
