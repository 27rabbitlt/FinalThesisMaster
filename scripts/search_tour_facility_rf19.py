#!/usr/bin/env python3
"""Falsification search for the deterministic tour--facility RF19 target.

For a symmetric metric d and independent probabilities p this computes

    P = E[tsp(A)],
    R = sum_v p_v E[d(v, B union {r})],
    J* = min_S {tsp(S) + 2 sum_v p_v d(v, S union {r})},

and searches for J* > 14 P / 5 + 2(P-R).  Metrics are represented directly
by complete edge lengths in [1,2], which automatically satisfy the triangle
inequality.  Differential evolution is evidence only, not a proof.
"""

from __future__ import annotations

import argparse
from functools import lru_cache

import numpy as np
from scipy.optimize import differential_evolution


def subset_tours(distance: np.ndarray) -> np.ndarray:
    n = distance.shape[0] - 1
    root = n

    @lru_cache(None)
    def path(mask: int, last: int) -> float:
        if mask == (1 << last):
            return float(distance[root, last])
        prev_mask = mask ^ (1 << last)
        return min(
            path(prev_mask, prev) + float(distance[prev, last])
            for prev in range(n)
            if prev_mask & (1 << prev)
        )

    tours = np.zeros(1 << n)
    for mask in range(1, 1 << n):
        tours[mask] = min(
            path(mask, last) + float(distance[last, root])
            for last in range(n)
            if mask & (1 << last)
        )
    return tours


def quantities(distance: np.ndarray, probabilities: np.ndarray):
    n = len(probabilities)
    root = n
    tours = subset_tours(distance)
    subset_prob = np.ones(1 << n)
    for mask in range(1 << n):
        value = 1.0
        for v, p_v in enumerate(probabilities):
            value *= p_v if mask & (1 << v) else 1.0 - p_v
        subset_prob[mask] = value

    posterior = float(np.dot(subset_prob, tours))
    assignment_by_mask = np.zeros(1 << n)
    for mask in range(1 << n):
        centers = [root] + [v for v in range(n) if mask & (1 << v)]
        assignment_by_mask[mask] = sum(
            probabilities[v] * min(distance[v, u] for u in centers)
            for v in range(n)
        )
    nearest_sample = float(np.dot(subset_prob, assignment_by_mask))
    objectives = tours + 2.0 * assignment_by_mask
    best_mask = int(np.argmin(objectives))
    best = float(objectives[best_mask])
    deficit = posterior - nearest_sample
    target = 2.8 * posterior + 2.0 * deficit
    return posterior, nearest_sample, deficit, best, best_mask, target


def unpack(vector: np.ndarray, n: int):
    distance = np.zeros((n + 1, n + 1))
    cursor = 0
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            distance[i, j] = distance[j, i] = vector[cursor]
            cursor += 1
    probabilities = vector[cursor:]
    return distance, probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--popsize", type=int, default=15)
    args = parser.parse_args()

    n = args.clients
    edge_count = n * (n + 1) // 2
    bounds = [(1.0, 2.0)] * edge_count + [(1.0e-4, 0.9999)] * n

    best_seen = None

    def objective(vector: np.ndarray) -> float:
        nonlocal best_seen
        distance, probabilities = unpack(vector, n)
        values = quantities(distance, probabilities)
        posterior, nearest_sample, deficit, best, mask, target = values
        normalized_margin = (target - best) / posterior
        record = (normalized_margin, values, distance.copy(), probabilities.copy())
        if best_seen is None or normalized_margin < best_seen[0]:
            best_seen = record
            print(
                "best margin",
                normalized_margin,
                "J/P",
                best / posterior,
                "D/P",
                deficit / posterior,
                "mask",
                mask,
                flush=True,
            )
        return normalized_margin

    for seed in range(args.seeds):
        differential_evolution(
            objective,
            bounds,
            seed=seed,
            maxiter=args.maxiter,
            popsize=args.popsize,
            polish=True,
            updating="immediate",
            workers=1,
            tol=1.0e-9,
        )

    assert best_seen is not None
    margin, values, distance, probabilities = best_seen
    print("FINAL normalized target slack", margin)
    print("P, R, D, J*, mask, target", values)
    print("probabilities", probabilities.tolist())
    print("distance")
    for row in distance:
        print(row.tolist())


if __name__ == "__main__":
    main()
