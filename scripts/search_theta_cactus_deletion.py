#!/usr/bin/env python3
"""Audit the structural route that deletes one theta edge to get a cactus.

For a weighted theta graph G and product terminal law p, deleting any one
edge leaves a connected cactus.  The cactus theorem gives a hazard tree of
cost at most P(G-e).  This script tests whether some deletion always has
P(G-e) <= (29/20) P(G).  Exact Fraction arithmetic is used throughout;
the script is a falsification aid rather than a proof.
"""

from __future__ import annotations

import argparse
import random
from fractions import Fraction

from scipy.optimize import linprog

from search_theta_weighted_ust import metric_closure, theta_graph


def subset_tsp_costs(vertex_count, distance):
    client_count = vertex_count - 1
    size = 1 << client_count
    infinity = None
    paths = [[infinity] * client_count for _ in range(size)]
    tours = [Fraction(0)] * size
    for v in range(1, vertex_count):
        paths[1 << (v - 1)][v - 1] = distance[0][v]
    for mask in range(1, size):
        for v in range(1, vertex_count):
            bit = 1 << (v - 1)
            if not mask & bit:
                continue
            previous = mask ^ bit
            if previous:
                paths[mask][v - 1] = min(
                    paths[previous][u - 1] + distance[u][v]
                    for u in range(1, vertex_count)
                    if previous & (1 << (u - 1))
                )
        tours[mask] = min(
            paths[mask][v - 1] + distance[v][0]
            for v in range(1, vertex_count)
            if mask & (1 << (v - 1))
        )
    return tours


def expected_value(costs, probabilities):
    answer = Fraction(0)
    for mask, cost in enumerate(costs):
        probability = Fraction(1)
        for v, p in enumerate(probabilities):
            probability *= p if mask & (1 << v) else 1 - p
        answer += probability * cost
    return answer


def audit(path_lengths, probabilities):
    vertex_count, edges, _ = theta_graph(path_lengths)
    original = expected_value(
        subset_tsp_costs(vertex_count, metric_closure(vertex_count, edges)),
        probabilities,
    )
    deleted = []
    for omitted in range(len(edges)):
        remaining = [edge for index, edge in enumerate(edges) if index != omitted]
        value = expected_value(
            subset_tsp_costs(vertex_count, metric_closure(vertex_count, remaining)),
            probabilities,
        )
        deleted.append(value)
    best = min(deleted)
    ratio = best / original if original else Fraction(0)
    total_length = sum(length for _, _, length in edges)
    weighted_average = sum(
        edges[index][2] * value for index, value in enumerate(deleted)
    ) / total_length
    return original, best, ratio, deleted.index(best), weighted_average / original if original else 0


def optimal_cactus_pair_stretch(path_lengths):
    vertex_count, edges, _ = theta_graph(path_lengths)
    original = metric_closure(vertex_count, edges)
    deleted_metrics = [
        metric_closure(
            vertex_count,
            [edge for index, edge in enumerate(edges) if index != omitted],
        )
        for omitted in range(len(edges))
    ]
    rows = []
    for u in range(vertex_count):
        for v in range(u):
            rows.append(
                [
                    *[float(distance[u][v]) for distance in deleted_metrics],
                    -float(original[u][v]),
                ]
            )
    result = linprog(
        [0.0] * len(edges) + [1.0],
        A_ub=rows,
        b_ub=[0.0] * len(rows),
        A_eq=[[1.0] * len(edges) + [0.0]],
        b_eq=[1.0],
        bounds=[(0.0, None)] * (len(edges) + 1),
        method="highs",
    )
    assert result.success
    return result.fun, tuple(result.x[:-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--max-edges-per-path", type=int, default=2)
    parser.add_argument("--threshold", type=Fraction, default=Fraction(29, 20))
    parser.add_argument("--pair-stretch", action="store_true")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    worst = None
    worst_weighted = None
    for _ in range(args.trials):
        counts = tuple(rng.randint(1, args.max_edges_per_path) for _ in range(3))
        lengths = tuple(
            tuple(Fraction(rng.randint(1, 20), rng.randint(1, 10)) for _ in range(k))
            for k in counts
        )
        vertex_count, _, _ = theta_graph(lengths)
        probabilities = tuple(
            Fraction(rng.randint(0, 10), 10) for _ in range(vertex_count - 1)
        )
        if args.pair_stretch:
            ratio, distribution = optimal_cactus_pair_stretch(lengths)
            if ratio > float(args.threshold) + 1e-8:
                print("PAIR-STRETCH COUNTEREXAMPLE", ratio, lengths, distribution)
                return
            continue
        original, best, ratio, edge, weighted_ratio = audit(lengths, probabilities)
        if original and (worst is None or ratio > worst[0]):
            worst = (ratio, lengths, probabilities, original, best, edge, weighted_ratio)
        if original and (worst_weighted is None or weighted_ratio > worst_weighted[0]):
            worst_weighted = (weighted_ratio, lengths, probabilities, original)
        if original and ratio > args.threshold:
            print("COUNTEREXAMPLE", worst)
            return
    print("no violation; worst", worst)
    print("worst length-weighted single deletion", worst_weighted)


if __name__ == "__main__":
    main()
