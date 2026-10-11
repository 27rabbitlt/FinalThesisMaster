#!/usr/bin/env python3
"""Falsify the proposed closing-cost deficit inequality.

For a symmetric stochastic TSP instance let

    P = E[TSP(A)],
    R = sum_v p_v E[d(v,B)],
    D = P-R,
    M = E[max_{v in A} d(r,v)].

The conjectured structural lemma is M <= D.  This program searches the
ratio M/D on small [1,2]-metrics (which are automatically metric) with
heterogeneous activation probabilities.  It is a numerical falsifier only.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

from scipy.optimize import differential_evolution

from search_open_path_gap import subset_probabilities, subset_tsp
from search_open_residual_rounding import nearest_other_means


def decode(values: Sequence[float], client_count: int):
    vertex_count = client_count + 1
    edge_count = vertex_count * (vertex_count - 1) // 2
    distance = [[0.0] * vertex_count for _ in range(vertex_count)]
    position = 0
    for first in range(vertex_count):
        for second in range(first + 1, vertex_count):
            distance[first][second] = distance[second][first] = values[position]
            position += 1
    probabilities = [
        1.0 / (1.0 + math.exp(-value))
        for value in values[edge_count:]
    ]
    return distance, probabilities


def evaluate(distance: list[list[float]], probabilities: list[float]):
    scenario_probability = subset_probabilities(probabilities)
    posterior = sum(
        weight * value
        for weight, value in zip(scenario_probability, subset_tsp(distance))
    )
    means = nearest_other_means(distance, probabilities)
    radial = sum(
        probability * (1.0 - probability) * mean
        for probability, mean in zip(probabilities, means)
    )
    farthest = sum(
        weight
        * max(
            [0.0]
            + [
                distance[0][vertex]
                for vertex in range(1, len(distance))
                if mask & (1 << (vertex - 1))
            ]
        )
        for mask, weight in enumerate(scenario_probability)
    )
    deficit = posterior - radial
    return farthest / deficit, posterior, radial, farthest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()

    edge_count = (args.n + 1) * args.n // 2
    bounds = [(1.0, 2.0)] * edge_count + [(-9.0, 9.0)] * args.n
    incumbent = None
    calls = 0

    def objective(values: Sequence[float]) -> float:
        nonlocal incumbent, calls
        calls += 1
        distance, probabilities = decode(values, args.n)
        result = evaluate(distance, probabilities)
        if incumbent is None or result[0] > incumbent[0][0]:
            incumbent = (result, distance, probabilities)
            print(
                f"call={calls} M/D={result[0]:.15f} "
                f"P={result[1]:.12g} R={result[2]:.12g} M={result[3]:.12g}",
                flush=True,
            )
        return -result[0]

    outcome = differential_evolution(
        objective,
        bounds,
        seed=args.seed,
        maxiter=args.generations,
        popsize=args.population,
        polish=False,
        updating="immediate",
        workers=1,
        tol=0.0,
        atol=0.0,
    )
    assert incumbent is not None
    result, distance, probabilities = incumbent
    print("optimizer:", outcome.message)
    print("M/D =", format(result[0], ".17g"))
    print("P R M =", *(format(value, ".17g") for value in result[1:]))
    print("p =", " ".join(format(value, ".17g") for value in probabilities))
    print("d =")
    for row in distance:
        print(" ".join(format(value, ".17g") for value in row))


if __name__ == "__main__":
    main()
