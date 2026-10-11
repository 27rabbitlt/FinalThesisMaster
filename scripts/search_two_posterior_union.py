#!/usr/bin/env python3
"""Search the two-posterior-union master-route bound.

For independent product samples B,C, put U=B union C.  An optimal rooted
tour of U, with doubled nearest-core excursions for every client, gives the
valid fixed-order upper bound

    U_2 = E[tsp(U)] + 2 sum_v p_v E[d(v,U union {r})].

This script evaluates P, the usual one-guide radius R, and U_2 exactly by
subset enumeration (up to floating-point arithmetic), then compares the
better of U_2 and P+2R with 2.9 P.
"""

from __future__ import annotations

import argparse
import random
from typing import Sequence

from search_hazard_tree_ratio import (
    expected_posterior,
    named_metric,
    probability_vectors,
    subset_tsp_costs,
)


def mask_probability(mask: int, p: Sequence[float]) -> float:
    value = 1.0
    for v, pv in enumerate(p):
        value *= pv if mask & (1 << v) else 1.0 - pv
    return value


def assignment_radius(d: Sequence[Sequence[float]], weights: Sequence[float],
                      core_p: Sequence[float]) -> float:
    """E_S sum_v weights[v] d(v,S union root), S~core_p."""
    n = len(weights)
    root = n
    value = 0.0
    for mask in range(1 << n):
        probability = mask_probability(mask, core_p)
        selected = [root] + [v for v in range(n) if mask & (1 << v)]
        value += probability * sum(
            weights[v] * min(d[v][u] for u in selected) for v in range(n)
        )
    return value


def evaluate(d: Sequence[Sequence[float]], p: Sequence[float]):
    costs = subset_tsp_costs(d)
    union_p = [1.0 - (1.0 - pv) ** 2 for pv in p]
    posterior = expected_posterior(p, costs)
    radius = assignment_radius(d, p, p)
    union_tour = expected_posterior(union_p, costs)
    union_radius = assignment_radius(d, p, union_p)
    baseline = posterior + 2.0 * radius
    union_bound = union_tour + 2.0 * union_radius
    return posterior, radius, union_tour, union_radius, baseline, union_bound


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--metrics", type=int, default=100)
    parser.add_argument("--probabilities", type=int, default=100)
    parser.add_argument("--seed", type=int, default=29002)
    parser.add_argument(
        "--families",
        default="unit,path,cycle,star,binary,random-tree,graph12,graph123,"
        "client-graph-19,euclidean,multiscale-graph,multiscale-complete",
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)
    worst = None
    worst_hard = None
    worst_union = None
    for family in args.families.split(","):
        count = args.metrics if family.startswith(("random", "graph", "client", "euclidean", "multiscale")) else 1
        for _ in range(count):
            d = named_metric(family, args.clients, rng)
            for p in probability_vectors(args.clients, rng, args.probabilities):
                values = evaluate(d, p)
                posterior, radius, union_tour, union_radius, baseline, union_bound = values
                if posterior <= 1e-14:
                    continue
                ratio = min(baseline, union_bound) / posterior
                item = (ratio, family, tuple(p), values)
                if worst is None or item[0] > worst[0]:
                    worst = item
                union_item = (union_bound / posterior, family, tuple(p), values)
                if worst_union is None or union_item[0] > worst_union[0]:
                    worst_union = union_item
                if (posterior - radius) / posterior < 0.05:
                    if worst_hard is None or item[0] > worst_hard[0]:
                        worst_hard = item
    assert worst is not None
    ratio, family, p, values = worst
    posterior, radius, union_tour, union_radius, baseline, union_bound = values
    print(f"worst ratio={ratio:.12f} family={family} p={p}")
    print(f"P={posterior:.12f} R={radius:.12f} D/P={(posterior-radius)/posterior:.12f}")
    print(f"P2={union_tour:.12f} R2={union_radius:.12f}")
    print(f"baseline/P={baseline/posterior:.12f} union/P={union_bound/posterior:.12f}")
    if worst_hard is None:
        print("no instance with D/P < 0.05")
    else:
        ratio, family, p, values = worst_hard
        posterior, radius, union_tour, union_radius, baseline, union_bound = values
        print(f"hard ratio={ratio:.12f} family={family} p={p}")
        print(f"hard D/P={(posterior-radius)/posterior:.12f} union/P={union_bound/posterior:.12f}")
    assert worst_union is not None
    ratio, family, p, values = worst_union
    posterior, radius, union_tour, union_radius, baseline, union_bound = values
    print(f"worst union ratio={ratio:.12f} family={family} p={p}")
    print(f"union-case D/P={(posterior-radius)/posterior:.12f} baseline/P={baseline/posterior:.12f}")


if __name__ == "__main__":
    main()
