#!/usr/bin/env python3
"""Exhaust graph metrics for integrated rootless compression (float search).

All connected labelled unweighted graphs on a depot and n clients are used;
the probability search is the displayed common grid plus deterministic
heterogeneous degree-based vectors.  This is an adversarial search rather
than an exact symbolic proof over the continuum of probabilities.
"""

from __future__ import annotations

import argparse
import itertools
import math

from search_rootless_integrated_random_graphs import Instance
from search_rootless_threshold_random_graphs import metric_from_edges


def connected(n: int, edges: list[tuple[int, int]]) -> bool:
    seen = {n}
    while True:
        larger = seen | {
            v
            for u, v in edges
            if u in seen
        } | {
            u
            for u, v in edges
            if v in seen
        }
        if larger == seen:
            return len(seen) == n + 1
        seen = larger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=5)
    args = parser.parse_args()
    n = args.clients
    pairs = tuple(itertools.combinations(range(n + 1), 2))
    common = (0.001, 0.01, 0.03, 0.07, 0.15, 0.3, 0.5, 0.7, 0.9, 0.97, 0.99)
    graph_count = 0
    audit_count = 0
    minimum_weak = (math.inf, None)
    minimum_ratio = (math.inf, None)
    maximum_route = (-math.inf, None)
    for graph_mask in range(1 << len(pairs)):
        edges = [edge for index, edge in enumerate(pairs) if graph_mask & (1 << index)]
        if not connected(n, edges):
            continue
        graph_count += 1
        d = metric_from_edges(n, edges)
        instance = Instance(d, edges)
        degrees = [sum(v in edge for edge in edges) for v in range(n)]
        candidates = [(value,) * n for value in common]
        for low, high in ((0.02, 0.8), (0.1, 0.95), (0.4, 0.99)):
            order = sorted(range(n), key=lambda v: (degrees[v], v))
            vector = [low] * n
            for rank, v in enumerate(order):
                vector[v] = low + (high - low) * rank / max(1, n - 1)
            candidates.append(tuple(vector))
            candidates.append(tuple(reversed(vector)))
        for p in candidates:
            terms = instance.terms(p, include_rootful=False)
            posterior, radius, m_star, a_star, gamma_star, _, _ = terms
            weak = gamma_star - 2 * radius / 19
            ratio = gamma_star / radius if radius > 1e-15 else math.inf
            route = (m_star + 2 * a_star) / posterior if posterior else 0
            record = (graph_mask, p, terms, edges, d)
            if weak < minimum_weak[0]:
                minimum_weak = (weak, record)
            if ratio < minimum_ratio[0]:
                minimum_ratio = (ratio, record)
            if route > maximum_route[0]:
                maximum_route = (route, record)
            if weak < -1e-9:
                print("COUNTEREXAMPLE", weak, record)
                return
            audit_count += 1
        if graph_count % 1000 == 0:
            print(
                "progress",
                graph_count,
                "weak",
                minimum_weak[0],
                "ratio",
                minimum_ratio[0],
                "route",
                maximum_route[0],
                flush=True,
            )
    print("exhausted", graph_count, "connected graphs and", audit_count, "audits")
    print("minimum weak margin", minimum_weak)
    print("minimum Gamma*/R", minimum_ratio)
    print("maximum rootless route/P", maximum_route)


if __name__ == "__main__":
    main()
