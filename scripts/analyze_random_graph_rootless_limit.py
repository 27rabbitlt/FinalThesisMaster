#!/usr/bin/env python3
"""Estimate the sparse-{1,2}-metric limit for rootless compression.

For c<1, G(n,c/n) has only o(n) vertices in cyclic components.  On every
tree component the Hamilton-completion/path-cover defect is n minus the
maximum number of edges in a degree-at-most-two subforest.  A two-state tree
DP computes this exactly.  The resulting defect density rho(c) predicts

  F(c) = [alpha (1 + rho(alpha c)) + 2 (1 + exp(-alpha c))]
         / [1 + rho(c)]

for the integrated compressed bound in the rare-client random-graph metric.
The calculation is a structural stress test, not a proof of a finite bound.
"""

from __future__ import annotations

import argparse
import math
import random

import networkx as nx


def tree_path_cover_defect_density(n: int, c: float, seed: int) -> tuple[float, float]:
    graph = nx.fast_gnp_random_graph(n, c / n, seed=seed)
    defect = 0
    cyclic_vertices = 0
    seen: set[int] = set()
    for start in graph:
        if start in seen:
            continue
        stack = [(start, -1, False)]
        order: list[tuple[int, int]] = []
        vertices = 0
        edge_twice = 0
        while stack:
            vertex, parent, exiting = stack.pop()
            if exiting:
                order.append((vertex, parent))
                continue
            if vertex in seen:
                continue
            seen.add(vertex)
            vertices += 1
            edge_twice += graph.degree(vertex)
            stack.append((vertex, parent, True))
            for child in graph[vertex]:
                if child != parent and child not in seen:
                    stack.append((child, vertex, False))
        edges = edge_twice // 2
        if edges != vertices - 1:
            cyclic_vertices += vertices
            # Cyclic components are o(n) for c<1.  Counting every vertex as
            # one path keeps the reported tree-DP estimate conservative.
            defect += vertices
            continue
        dp0: dict[int, int] = {}
        dp1: dict[int, int] = {}
        for vertex, parent in order:
            base = 0
            gains = []
            for child in graph[vertex]:
                if child == parent:
                    continue
                base += dp0[child]
                gains.append(1 + dp1[child] - dp0[child])
            gains.sort(reverse=True)
            dp0[vertex] = base + sum(gain for gain in gains[:2] if gain > 0)
            dp1[vertex] = base + sum(gain for gain in gains[:1] if gain > 0)
        defect += vertices - dp0[start]
    return defect / n, cyclic_vertices / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertices", type=int, default=200000)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--stop", type=float, default=0.98)
    parser.add_argument("--step", type=float, default=0.025)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    values: dict[float, float] = {}
    cyclic: dict[float, float] = {}
    c = args.start
    while c <= args.stop + 1e-12:
        key = round(c, 12)
        estimates = [
            tree_path_cover_defect_density(args.vertices, key, rng.randrange(2**32))
            for _ in range(args.trials)
        ]
        values[key] = sum(value for value, _ in estimates) / args.trials
        cyclic[key] = sum(value for _, value in estimates) / args.trials
        print(
            f"c={key:.6f} rho={values[key]:.9f} "
            f"cyclic_vertices={cyclic[key]:.3g}",
            flush=True,
        )
        c += args.step

    alpha = 1.0 - 1.0 / math.e
    best = None
    keys = sorted(values)
    for c in keys:
        target = alpha * c
        nearest = min(keys, key=lambda x: abs(x - target))
        factor = (
            alpha * (1.0 + values[nearest])
            + 2.0 * (1.0 + math.exp(-alpha * c))
        ) / (1.0 + values[c])
        record = (factor, c, nearest, values[c], values[nearest])
        if best is None or factor > best[0]:
            best = record
        print(f"factor c={c:.6f}: {factor:.9f} (rho(alpha*c) at {nearest:.6f})")
    print("best", best)


if __name__ == "__main__":
    main()
