#!/usr/bin/env python3
"""Stress-test a scale-wise route to A1 + R <= 2 P.

At threshold t, let k_t(S) be the minimum number of tour edges longer than t
in a cycle through S and the depot.  The candidate inequality is

    a_t + r_t <= 2 E[k_t(A)],

where r_t is the layer-cake integrand of the one-guide radius R and a_t is
the integrand of the compressed-core attachment A1.  If true at every t,
integration would imply A1+R <= 2P.  This file is exploratory: it searches
for finite counterexamples and prints the full first violation.
"""

from __future__ import annotations

import argparse
import random
from itertools import combinations

from search_compressed_two_guide import mask_probability
from search_hazard_tree_ratio import (
    named_metric,
    probability_vectors,
    subset_tsp_costs,
)


def threshold_terms(d, p, threshold, client_only_projection=False):
    n = len(p)
    root = n
    binary = [
        [0.0 if d[i][j] <= threshold else 1.0 for j in range(n + 1)]
        for i in range(n + 1)
    ]
    binary_tours = subset_tsp_costs(binary)
    expected_long_edges = sum(
        mask_probability(mask, p) * binary_tours[mask]
        for mask in range(1 << n)
    )
    radius = 0.0
    nearest_other = 0.0
    for v in range(n):
        if d[v][root] <= threshold:
            continue
        empty_other_neighborhood = 1.0
        for u in range(n):
            if u != v and d[v][u] <= threshold:
                empty_other_neighborhood *= 1.0 - p[u]
        nearest_other += p[v] * empty_other_neighborhood
    attachment = 0.0
    compressed_long_edges = 0.0
    for guide in range(1 << n):
        guide_probability = mask_probability(guide, p)
        centers = [v for v in range(n) if guide & (1 << v)]
        available = (
            tuple(centers)
            if client_only_projection and centers
            else (*centers, root)
        )
        image = [
            v if guide & (1 << v) else min(available, key=lambda u: (d[v][u], u))
            for v in range(n)
        ]
        radius_centers = (*centers, root)
        radius += guide_probability * sum(
            p[v]
            * (min(d[v][center] for center in radius_centers) > threshold)
            for v in range(n)
        )
        q = {}
        for center in centers:
            empty = 1.0
            for v in range(n):
                if image[v] == center:
                    empty *= 1.0 - p[v]
            q[center] = 1.0 - empty
        retained = guide
        while True:
            probability = guide_probability
            for center in centers:
                probability *= q[center] if retained & (1 << center) else 1.0 - q[center]
            selected = [root] + [v for v in centers if retained & (1 << v)]
            compressed_long_edges += probability * binary_tours[retained]
            attachment += probability * sum(
                p[v]
                * (min(d[v][u] for u in selected) > threshold)
                for v in range(n)
            )
            if retained == 0:
                break
            retained = (retained - 1) & guide
    return (
        attachment,
        radius,
        nearest_other,
        expected_long_edges,
        compressed_long_edges,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--metrics", type=int, default=100)
    parser.add_argument("--probabilities", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--exhaustive-graphs", action="store_true")
    parser.add_argument("--client-only-projection", action="store_true")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    if args.exhaustive_graphs:
        exhaustive_graphs(args.clients, args.client_only_projection)
        return
    families = (
        "unit",
        "path",
        "cycle",
        "star",
        "binary",
        "random-tree",
        "graph12",
        "graph123",
        "client-graph-19",
        "euclidean",
        "multiscale-graph",
        "multiscale-complete",
    )
    worst = None
    worst_isolated = None
    worst_core_drop = None
    worst_deficit_core_drop = None
    worst_rootless_gamma = None
    for metric_index in range(args.metrics):
        family = families[metric_index % len(families)]
        d = named_metric(family, args.clients, rng)
        thresholds = sorted({d[i][j] for i in range(args.clients + 1) for j in range(i)})
        threshold_intervals = [0.5 * (x + y) for x, y in zip([0.0] + thresholds, thresholds)]
        for p in probability_vectors(args.clients, rng, args.probabilities):
            for threshold in threshold_intervals:
                (
                    attachment,
                    radius,
                    nearest_other,
                    long_edges,
                    compressed_long_edges,
                ) = threshold_terms(
                    d,
                    p,
                    threshold,
                    client_only_projection=args.client_only_projection,
                )
                violation = attachment + radius - 2.0 * long_edges
                isolated_violation = attachment + radius - 2.0 * nearest_other
                if worst_isolated is None or isolated_violation > worst_isolated[0]:
                    worst_isolated = (isolated_violation, family, threshold, tuple(p), attachment, radius, nearest_other, long_edges, d)
                if worst is None or violation > worst[0]:
                    worst = (violation, isolated_violation, family, threshold, tuple(p), attachment, radius, nearest_other, long_edges, d)
                    print("worst =", worst, flush=True)
                if violation > 1e-8:
                    print("COUNTEREXAMPLE", worst)
                    return
                if args.client_only_projection:
                    rootless_gamma_violation = (
                        compressed_long_edges
                        + 2.0 * attachment
                        + radius / 2.718281828459045
                        - 3.0 * long_edges
                    )
                    record = (
                        rootless_gamma_violation,
                        family,
                        threshold,
                        tuple(p),
                        attachment,
                        radius,
                        long_edges,
                        compressed_long_edges,
                        d,
                    )
                    if (
                        worst_rootless_gamma is None
                        or rootless_gamma_violation > worst_rootless_gamma[0]
                    ):
                        worst_rootless_gamma = record
                    if rootless_gamma_violation > 1e-8:
                        print("ROOTLESS-GAMMA COUNTEREXAMPLE", record)
                        return
                if not args.client_only_projection:
                    core_drop_violation = (
                        (2.0 * radius - long_edges) / 2.718281828459045
                        - (long_edges - compressed_long_edges)
                    )
                    if (
                        worst_core_drop is None
                        or core_drop_violation > worst_core_drop[0]
                    ):
                        worst_core_drop = (
                            core_drop_violation,
                            family,
                            threshold,
                            tuple(p),
                            radius,
                            long_edges,
                            compressed_long_edges,
                            d,
                        )
                    alpha = 1.0 - 1.0 / 2.718281828459045
                    deficit_core_violation = (
                        compressed_long_edges
                        - alpha * long_edges
                        - 2.0 * (long_edges - radius)
                    )
                    if (
                        worst_deficit_core_drop is None
                        or deficit_core_violation > worst_deficit_core_drop[0]
                    ):
                        worst_deficit_core_drop = (
                            deficit_core_violation,
                            family,
                            threshold,
                            tuple(p),
                            radius,
                            long_edges,
                            compressed_long_edges,
                            d,
                        )
    print("no violation; worst =", worst)
    print("worst against isolated lower bound =", worst_isolated)
    print("worst scale-wise core-drop violation =", worst_core_drop)
    print(
        "worst deficit-sensitive scale core violation =",
        worst_deficit_core_drop,
    )
    print("worst rootless Gamma violation =", worst_rootless_gamma)


def exhaustive_graphs(n, client_only_projection=False):
    """Exhaust connected unweighted graphs and common-probability values."""
    size = n + 1
    pairs = list(combinations(range(size), 2))
    worst = None
    worst_rootless_gamma = None
    connected = 0
    for graph_mask in range(1 << len(pairs)):
        infinity = size + 1.0
        d = [[0.0 if i == j else infinity for j in range(size)] for i in range(size)]
        for index, (u, v) in enumerate(pairs):
            if graph_mask & (1 << index):
                d[u][v] = d[v][u] = 1.0
        for k in range(size):
            for i in range(size):
                for j in range(size):
                    d[i][j] = min(d[i][j], d[i][k] + d[k][j])
        if any(d[0][v] > size for v in range(size)):
            continue
        connected += 1
        candidates = [(probability,) * n for probability in (0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 0.99)]
        graph_rng = random.Random(104729 * graph_mask + n)
        candidates.extend(
            tuple(graph_rng.choice((0.01, 0.05, 0.2, 0.5, 0.8, 0.95)) for _ in range(n))
            for _ in range(40)
        )
        for p in candidates:
            (
                attachment,
                radius,
                nearest_other,
                long_edges,
                compressed_long_edges,
            ) = threshold_terms(
                d,
                p,
                1.5,
                client_only_projection=client_only_projection,
            )
            violation = attachment + radius - 2.0 * long_edges
            if worst is None or violation > worst[0]:
                worst = (violation, attachment + radius - 2.0 * nearest_other, graph_mask, p, attachment, radius, nearest_other, long_edges, d)
            if violation > 1e-9:
                print("COUNTEREXAMPLE", worst)
                return
            if client_only_projection:
                rootless_gamma_violation = (
                    compressed_long_edges
                    + 2.0 * attachment
                    + radius / 2.718281828459045
                    - 3.0 * long_edges
                )
                record = (
                    rootless_gamma_violation,
                    graph_mask,
                    p,
                    attachment,
                    radius,
                    long_edges,
                    compressed_long_edges,
                    d,
                )
                if (
                    worst_rootless_gamma is None
                    or rootless_gamma_violation > worst_rootless_gamma[0]
                ):
                    worst_rootless_gamma = record
                if rootless_gamma_violation > 1e-9:
                    print("ROOTLESS-GAMMA COUNTEREXAMPLE", record)
                    return
            if not client_only_projection:
                core_drop_violation = (
                    (2.0 * radius - long_edges) / 2.718281828459045
                    - (long_edges - compressed_long_edges)
                )
                if core_drop_violation > 1e-9:
                    print(
                        "CORE-DROP COUNTEREXAMPLE",
                        core_drop_violation,
                        graph_mask,
                        p,
                        radius,
                        long_edges,
                        compressed_long_edges,
                        d,
                    )
                    return
    print(f"exhausted {connected} connected graphs; worst = {worst}")
    print("worst rootless Gamma violation =", worst_rootless_gamma)


if __name__ == "__main__":
    main()
