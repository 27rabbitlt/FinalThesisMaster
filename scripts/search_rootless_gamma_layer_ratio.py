#!/usr/bin/env python3
"""Stress the deficit-corrected rootless-compression layer inequality.

For each metric threshold, compute

  V = E k_t(T*) + 2 a*_t + r_t/e - 3 E k_t(A),
  Delta = E k_t(A) - r_t,

and report the largest positive ratio V/Delta.  A uniform bound below five
would still be strong enough for the intended D<P/20 hard-regime argument.
"""

from __future__ import annotations

import argparse
import math
import random

from search_compressed_threshold_inequality import threshold_terms
from search_hazard_tree_ratio import named_metric, probability_vectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=7)
    parser.add_argument("--metrics", type=int, default=1000)
    parser.add_argument("--probabilities", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    rng = random.Random(args.seed)
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
    for metric_index in range(args.metrics):
        family = families[metric_index % len(families)]
        distance = named_metric(family, args.clients, rng)
        levels = sorted(
            {
                distance[u][v]
                for u in range(args.clients + 1)
                for v in range(u)
            }
        )
        thresholds = [
            (low + high) / 2.0
            for low, high in zip([0.0] + levels, levels)
        ]
        for probabilities in probability_vectors(
            args.clients,
            rng,
            args.probabilities,
        ):
            for threshold in thresholds:
                attachment, radius, _, tour, core_tour = threshold_terms(
                    distance,
                    probabilities,
                    threshold,
                    client_only_projection=True,
                )
                violation = (
                    core_tour
                    + 2.0 * attachment
                    + radius / math.e
                    - 3.0 * tour
                )
                deficit = tour - radius
                ratio = violation / deficit if deficit > 1e-14 else -math.inf
                record = (
                    ratio,
                    violation,
                    deficit,
                    family,
                    threshold,
                    tuple(probabilities),
                    attachment,
                    radius,
                    tour,
                    core_tour,
                    distance,
                )
                if worst is None or ratio > worst[0]:
                    worst = record
                    print("worst", worst[:10], flush=True)
    print("no cutoff; worst", worst)


if __name__ == "__main__":
    main()
