#!/usr/bin/env python3
"""Focused audit of two sufficient inequalities for rootless compression.

The proposed split is

    A* + R <= 2 P,
    P - M* >= 4 R / 19.

In the hard regime D=P-R<P/20 these imply
Gamma*=(P-M*)+2(P-A*) >= 2R/19.  This script is only an
exact-enumeration/floating-point stress test, not a proof.
"""

from __future__ import annotations

import argparse
import random

from search_compressed_two_guide import (
    compressed_radius_terms,
    evaluate,
    sampled_nearest_assignment,
)
from search_hazard_tree_ratio import named_metric, probability_vectors, subset_tsp_costs


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
    worst_attachment = None
    worst_core = None
    worst_joint = None
    worst_hard_core = None
    worst_hard_joint = None
    worst_vs_sampled = None
    worst_hard_vs_sampled = None
    worst_four_fifths_radius = None
    for metric_index in range(args.metrics):
        family = families[metric_index % len(families)]
        distance = named_metric(family, args.clients, rng)
        costs = subset_tsp_costs(distance)
        for probabilities in probability_vectors(
            args.clients, rng, args.probabilities
        ):
            p, _, m_star, a_star = evaluate(
                distance, probabilities, costs, client_only_projection=True
            )
            r = sampled_nearest_assignment(distance, probabilities)
            nearest_other, self_deficit, retained_radius, _ = compressed_radius_terms(
                distance, probabilities
            )
            if p <= 1e-14 or r <= 1e-14:
                continue
            d = p - r
            attachment_gap = (a_star + r - 2 * p) / p
            core_gap = (4 * r / 19 - (p - m_star)) / p
            gamma = (p - m_star) + 2 * (p - a_star)
            joint_gap = (2 * r / 19 - gamma) / p
            sampled_gap = (m_star + 2 * a_star - p - 2 * r) / p
            four_fifths_radius_gap = (
                retained_radius - 0.8 * nearest_other - self_deficit
            ) / p
            record = (
                family, tuple(probabilities), p, r, m_star, a_star,
                d / p, (p - m_star) / r, gamma / r,
            )
            if worst_attachment is None or attachment_gap > worst_attachment[0]:
                worst_attachment = (attachment_gap, *record)
            if worst_core is None or core_gap > worst_core[0]:
                worst_core = (core_gap, *record)
            if worst_joint is None or joint_gap > worst_joint[0]:
                worst_joint = (joint_gap, *record)
            if worst_vs_sampled is None or sampled_gap > worst_vs_sampled[0]:
                worst_vs_sampled = (sampled_gap, *record)
            if (
                worst_four_fifths_radius is None
                or four_fifths_radius_gap > worst_four_fifths_radius[0]
            ):
                worst_four_fifths_radius = (four_fifths_radius_gap, *record)
            if d < p / 20:
                if worst_hard_core is None or core_gap > worst_hard_core[0]:
                    worst_hard_core = (core_gap, *record)
                if worst_hard_joint is None or joint_gap > worst_hard_joint[0]:
                    worst_hard_joint = (joint_gap, *record)
                if (
                    worst_hard_vs_sampled is None
                    or sampled_gap > worst_hard_vs_sampled[0]
                ):
                    worst_hard_vs_sampled = (sampled_gap, *record)
    print("max (A*+R-2P)/P", worst_attachment)
    print("max (4R/19-(P-M*))/P", worst_core)
    print("max (2R/19-Gamma*)/P", worst_joint)
    print("hard max core gap", worst_hard_core)
    print("hard max joint gap", worst_hard_joint)
    print("max compressed-minus-sampled", worst_vs_sampled)
    print("hard max compressed-minus-sampled", worst_hard_vs_sampled)
    print("max L_T-.8L-J", worst_four_fifths_radius)


if __name__ == "__main__":
    main()
