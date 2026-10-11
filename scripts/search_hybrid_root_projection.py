#!/usr/bin/env python3
"""Audit a per-(B,C) hybrid of rooted and rootless compressed cores.

For each two-guide outcome, form both projections and choose the core S that
minimizes

    phi(S) = tsp(S) + 2 sum_v p_v d(v,S union {root}).

The choice is made before the evaluation realization, so averaging these
master orders is valid.  This script uses exact subset enumeration apart from
floating-point arithmetic and is exploratory, not a proof.
"""

from __future__ import annotations

import argparse
import random
from typing import Sequence

from search_compressed_two_guide import (
    compressed_radius_terms,
    mask_probability,
    sampled_nearest_assignment,
)
from search_hazard_tree_ratio import (
    expected_posterior,
    named_metric,
    probability_vectors,
    subset_tsp_costs,
)


def attachment(
    mask: int, distance: Sequence[Sequence[float]], probabilities: Sequence[float]
) -> float:
    n = len(probabilities)
    selected = [n] + [v for v in range(n) if mask & (1 << v)]
    return sum(
        probabilities[v] * min(distance[v][u] for u in selected)
        for v in range(n)
    )


def projected_mask(
    guide: int,
    second: int,
    distance: Sequence[Sequence[float]],
    rootless: bool,
) -> int:
    n = len(distance) - 1
    root = n
    centers = [v for v in range(n) if guide & (1 << v)]
    available = centers if rootless and centers else centers + [root]
    retained = 0
    for v in range(n):
        if not second & (1 << v):
            continue
        image = v if guide & (1 << v) else min(
            available, key=lambda u: (distance[v][u], u)
        )
        if image != root:
            retained |= 1 << image
    return retained


def evaluate_hybrid(
    distance: Sequence[Sequence[float]],
    probabilities: Sequence[float],
    costs: Sequence[float] | None = None,
) -> tuple[float, float, float, float, float]:
    """Return P and rooted, rootless, two-core hybrid, three-way hybrid."""
    n = len(probabilities)
    if costs is None:
        costs = subset_tsp_costs(distance)
    posterior = expected_posterior(probabilities, costs)
    weights = [mask_probability(mask, probabilities) for mask in range(1 << n)]
    attach = [attachment(mask, distance, probabilities) for mask in range(1 << n)]
    rooted = 0.0
    rootless = 0.0
    hybrid = 0.0
    tri_hybrid = 0.0
    for guide in range(1 << n):
        for second in range(1 << n):
            weight = weights[guide] * weights[second]
            if weight == 0.0:
                continue
            rooted_mask = projected_mask(guide, second, distance, False)
            rootless_mask = projected_mask(guide, second, distance, True)
            rooted_value = costs[rooted_mask] + 2 * attach[rooted_mask]
            rootless_value = costs[rootless_mask] + 2 * attach[rootless_mask]
            guide_value = costs[guide] + 2 * attach[guide]
            rooted += weight * rooted_value
            rootless += weight * rootless_value
            hybrid += weight * min(rooted_value, rootless_value)
            tri_hybrid += weight * min(
                guide_value, rooted_value, rootless_value
            )
    return posterior, rooted, rootless, hybrid, tri_hybrid


def main() -> None:
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
    worst_factor = None
    worst_gamma = None
    worst_hard = None
    worst_linear = None
    worst_nearest_split = None
    worst_tri_linear = None
    largest_baseline_clip = None
    largest_gain = None
    for metric_index in range(args.metrics):
        family = families[metric_index % len(families)]
        distance = named_metric(family, args.clients, rng)
        costs = subset_tsp_costs(distance)
        for probabilities in probability_vectors(
            args.clients, rng, args.probabilities
        ):
            p, rooted, rootless, hybrid, tri_hybrid = evaluate_hybrid(
                distance, probabilities, costs
            )
            if p <= 1e-14:
                continue
            r = sampled_nearest_assignment(distance, probabilities)
            nearest_other, _, _, _ = compressed_radius_terms(
                distance, probabilities
            )
            d = p - r
            factor = hybrid / p
            gamma_gap = (2 * r / 19 - (3 * p - hybrid)) / p
            linear_gap = (hybrid - (2.8 * p + 2 * d)) / p
            nearest_split_gap = (
                hybrid - 2 * r
                - (0.8 * nearest_other + 4 * (p - nearest_other))
            ) / p
            tri_linear_gap = (tri_hybrid - (2.8 * p + 2 * d)) / p
            baseline_clip = (hybrid - tri_hybrid) / p
            gain = (min(rooted, rootless) - hybrid) / p
            record = (
                family, tuple(probabilities), p, r, rooted, rootless, hybrid,
                tri_hybrid, d / p, factor, gamma_gap, gain,
            )
            if worst_factor is None or factor > worst_factor[0]:
                worst_factor = (factor, *record)
            if worst_gamma is None or gamma_gap > worst_gamma[0]:
                worst_gamma = (gamma_gap, *record)
            if worst_linear is None or linear_gap > worst_linear[0]:
                worst_linear = (linear_gap, *record)
            if (
                worst_nearest_split is None
                or nearest_split_gap > worst_nearest_split[0]
            ):
                worst_nearest_split = (nearest_split_gap, *record)
            if worst_tri_linear is None or tri_linear_gap > worst_tri_linear[0]:
                worst_tri_linear = (tri_linear_gap, *record)
            if (
                largest_baseline_clip is None
                or baseline_clip > largest_baseline_clip[0]
            ):
                largest_baseline_clip = (baseline_clip, *record)
            if d < p / 20 and (worst_hard is None or gamma_gap > worst_hard[0]):
                worst_hard = (gamma_gap, *record)
            if largest_gain is None or gain > largest_gain[0]:
                largest_gain = (gain, *record)
    print("worst hybrid factor", worst_factor)
    print("worst 2R/19 violation", worst_gamma)
    print("worst U_hyb <= 2.8P+2D violation", worst_linear)
    print("worst Phi_hyb <= .8L+4(P-L) violation", worst_nearest_split)
    print("worst tri-hybrid <= 2.8P+2D violation", worst_tri_linear)
    print("largest gain from including B pathwise", largest_baseline_clip)
    print("worst hard violation", worst_hard)
    print("largest pathwise-over-global gain", largest_gain)


if __name__ == "__main__":
    main()
