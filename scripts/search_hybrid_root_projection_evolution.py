#!/usr/bin/env python3
"""Evolutionary falsification audit of the pathwise hybrid P-scale target."""

from __future__ import annotations

import argparse
import math

from scipy.optimize import differential_evolution

from search_compressed_two_guide import sampled_nearest_assignment
from search_hazard_tree_ratio import metric_closure
from search_hybrid_root_projection import evaluate_hybrid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    n = args.clients
    total = n + 1
    pairs = [(u, v) for u in range(total) for v in range(u)]
    calls = 0
    incumbent = None

    def decode(vector):
        weights = [
            [0.0 if u == v else float("inf") for v in range(total)]
            for u in range(total)
        ]
        for log_length, (u, v) in zip(vector, pairs):
            weights[u][v] = weights[v][u] = math.exp(log_length)
        distance = metric_closure(weights)
        probabilities = tuple(
            1.0 / (1.0 + math.exp(-logit))
            for logit in vector[len(pairs) :]
        )
        return distance, probabilities

    def objective(vector):
        nonlocal calls, incumbent
        distance, probabilities = decode(vector)
        posterior, rooted, rootless, hybrid, tri_hybrid = evaluate_hybrid(
            distance, probabilities
        )
        radius = sampled_nearest_assignment(distance, probabilities)
        deficit = posterior - radius
        coarse_gap = (hybrid - 2.8 * posterior - 2.0 * deficit) / posterior
        factor = hybrid / posterior
        calls += 1
        if incumbent is None or coarse_gap > incumbent[0]:
            incumbent = (
                coarse_gap,
                factor,
                deficit / posterior,
                posterior,
                rooted,
                rootless,
                hybrid,
                tri_hybrid,
                probabilities,
                distance,
            )
            print(
                f"gap={coarse_gap:.12g} factor={factor:.12g} "
                f"D/P={deficit/posterior:.12g} calls={calls}",
                flush=True,
            )
        return -coarse_gap

    differential_evolution(
        objective,
        [(-2.5, 2.5)] * len(pairs) + [(-8.0, 8.0)] * n,
        seed=args.seed,
        maxiter=args.maxiter,
        popsize=args.popsize,
        polish=True,
        updating="immediate",
        workers=1,
        tol=1e-9,
    )
    print("incumbent", incumbent)


if __name__ == "__main__":
    main()
