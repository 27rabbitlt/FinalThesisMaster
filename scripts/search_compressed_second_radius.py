#!/usr/bin/env python3
"""Search the per-vertex second-radius bound for the compressed core.

For B,C iid product samples and T=f_B(C), the candidate is

    E[d(v,T union {r})] <= E[Y_v(B)]

for every client v, where Y_v(B) is the second order statistic of distances
from v to B plus two copies of the depot.  The weighted sum of this statement
would imply A1-R <= 2D through the independently proved second-radius budget.
"""

from __future__ import annotations

import argparse
import random

from search_compressed_two_guide import mask_probability
from search_hazard_tree_ratio import named_metric, probability_vectors


def evaluate(d, p):
    n = len(p)
    root = n
    attachment = [0.0] * n
    second = [0.0] * n
    first = [0.0] * n
    nearest_other = [0.0] * n
    for guide in range(1 << n):
        guide_probability = mask_probability(guide, p)
        centers = [u for u in range(n) if guide & (1 << u)]
        available = (*centers, root)
        image = [
            v if guide & (1 << v) else min(available, key=lambda u: (d[v][u], u))
            for v in range(n)
        ]
        q = {}
        for center in centers:
            empty = 1.0
            for u in range(n):
                if image[u] == center:
                    empty *= 1.0 - p[u]
            q[center] = 1.0 - empty
        for v in range(n):
            ordered = sorted([d[v][u] for u in centers] + [d[v][root], d[v][root]])
            first[v] += guide_probability * ordered[0]
            second[v] += guide_probability * ordered[1]
            other_selected = [root] + [u for u in centers if u != v]
            nearest_other[v] += guide_probability * min(
                d[v][u] for u in other_selected
            )
        retained = guide
        while True:
            probability = guide_probability
            for center in centers:
                probability *= q[center] if retained & (1 << center) else 1.0 - q[center]
            selected = [root] + [u for u in centers if retained & (1 << u)]
            for v in range(n):
                attachment[v] += probability * min(d[v][u] for u in selected)
            if retained == 0:
                break
            retained = (retained - 1) & guide
    return attachment, first, second, nearest_other


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=7)
    parser.add_argument("--metrics", type=int, default=100)
    parser.add_argument("--probabilities", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    families = (
        "unit", "path", "cycle", "star", "binary", "random-tree",
        "graph12", "graph123", "client-graph-19", "euclidean",
        "multiscale-graph", "multiscale-complete",
    )
    worst = None
    worst_with_j = None
    for metric_index in range(args.metrics):
        family = families[metric_index % len(families)]
        d = named_metric(family, args.clients, rng)
        for p in probability_vectors(args.clients, rng, args.probabilities):
            attachment, first, second, nearest_other = evaluate(d, p)
            for v in range(args.clients):
                gap = attachment[v] - second[v]
                if worst is None or gap > worst[0]:
                    worst = (gap, family, v, tuple(p), attachment[v], first[v], second[v], d)
                    print("worst", worst, flush=True)
                corrected_gap = attachment[v] - second[v] - p[v] * nearest_other[v]
                if worst_with_j is None or corrected_gap > worst_with_j[0]:
                    worst_with_j = (
                        corrected_gap,
                        family,
                        v,
                        tuple(p),
                        attachment[v],
                        second[v],
                        nearest_other[v],
                        d,
                    )
    print("no violation; worst", worst)
    print("worst [attachment-Y-p_v mu_v]", worst_with_j)


if __name__ == "__main__":
    main()
