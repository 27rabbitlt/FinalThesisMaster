#!/usr/bin/env python3
"""Search for obstructions to the open residual-order route.

For a symmetric metric d, independent client probabilities p, and an
always-active depot 0, define

    mu_v  = E[d(v, B \\ {v})],
    rho_v = (1-p_v) mu_v,
    R     = sum_v p_v rho_v,
    D     = OPT_post(d) - R,
    e_uv  = max(d_uv-rho_u-rho_v, 0).

The structural route under investigation would follow from

    min_pi E[e-cost of the active subsequence of (0, pi)] <= 9 D.       (A)

This script exhausts all client orders and all activation sets on small
instances.  It also reports the ratio to the posterior optimum open e-path;
a ratio above 3 disproves the stronger generic residual-rounding lemma but
does not by itself disprove (A).

This is a floating-point falsifier, not a proof.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
from functools import lru_cache

from search_open_path_gap import (
    floyd,
    open_cost,
    random_graph_metric,
    subset_probabilities,
    subset_tsp,
)


def nearest_other_means(
    distance: list[list[float]], probabilities: list[float]
) -> list[float]:
    """Return E distance from each client to depot or another active client."""
    scenario_probability = subset_probabilities(probabilities)
    means = [0.0] * len(probabilities)
    for mask, weight in enumerate(scenario_probability):
        for index in range(len(probabilities)):
            candidates = [distance[index + 1][0]]
            candidates.extend(
                distance[index + 1][other + 1]
                for other in range(len(probabilities))
                if other != index and mask & (1 << other)
            )
            means[index] += weight * min(candidates)
    return means


def residual_metric(
    distance: list[list[float]], probabilities: list[float]
) -> tuple[list[float], list[list[float]]]:
    means = nearest_other_means(distance, probabilities)
    radii = [0.0] + [
        (1.0 - probability) * mean
        for probability, mean in zip(probabilities, means)
    ]
    residual = [
        [
            max(distance[first][second] - radii[first] - radii[second], 0.0)
            for second in range(len(distance))
        ]
        for first in range(len(distance))
    ]
    return radii, residual


def posterior_open_costs(weights: list[list[float]]) -> list[float]:
    """Minimum rooted open Hamilton path cost for every client subset."""
    client_count = len(weights) - 1

    @lru_cache(maxsize=None)
    def path(mask: int, last: int) -> float:
        bit = 1 << (last - 1)
        if mask == bit:
            return weights[0][last]
        previous_mask = mask ^ bit
        return min(
            path(previous_mask, previous) + weights[previous][last]
            for previous in range(1, client_count + 1)
            if previous_mask & (1 << (previous - 1))
        )

    values = [0.0] * (1 << client_count)
    for mask in range(1, 1 << client_count):
        values[mask] = min(
            path(mask, last)
            for last in range(1, client_count + 1)
            if mask & (1 << (last - 1))
        )
    return values


def evaluate(
    distance: list[list[float]], probabilities: list[float]
) -> dict[str, object]:
    scenario_probability = subset_probabilities(probabilities)
    tsp_by_mask = subset_tsp(distance)
    posterior_d = sum(
        weight * value
        for weight, value in zip(scenario_probability, tsp_by_mask)
    )
    farthest = sum(
        weight
        * max(
            [0.0]
            + [
                distance[0][vertex]
                for vertex in range(1, len(probabilities) + 1)
                if mask & (1 << (vertex - 1))
            ]
        )
        for mask, weight in enumerate(scenario_probability)
    )
    means = nearest_other_means(distance, probabilities)
    radii = [0.0] + [
        (1.0 - probability) * mean
        for probability, mean in zip(probabilities, means)
    ]
    residual = [
        [
            max(distance[first][second] - radii[first] - radii[second], 0.0)
            for second in range(len(distance))
        ]
        for first in range(len(distance))
    ]
    mean_radii = [0.0] + means
    deep_residual = [
        [
            max(
                distance[first][second]
                - mean_radii[first]
                - mean_radii[second],
                0.0,
            )
            for second in range(len(distance))
        ]
        for first in range(len(distance))
    ]
    radial_mass = sum(
        probability * radii[index + 1]
        for index, probability in enumerate(probabilities)
    )
    deficit = posterior_d - radial_mass
    nearest_sum_by_mask = []
    overflow = 0.0
    for mask, weight in enumerate(scenario_probability):
        nearest_sum = 0.0
        for index in range(len(probabilities)):
            if not (mask & (1 << index)):
                continue
            candidates = [distance[index + 1][0]]
            candidates.extend(
                distance[index + 1][other + 1]
                for other in range(len(probabilities))
                if other != index and mask & (1 << other)
            )
            nearest_sum += min(candidates)
        nearest_sum_by_mask.append(nearest_sum)
        farthest_mask = max(
            [0.0]
            + [
                distance[0][vertex]
                for vertex in range(1, len(probabilities) + 1)
                if mask & (1 << (vertex - 1))
            ]
        )
        overflow += weight * max(
            nearest_sum + farthest_mask - tsp_by_mask[mask], 0.0
        )

    best_fixed = math.inf
    best_order: tuple[int, ...] | None = None
    best_full_e = math.inf
    best_full_e_order: tuple[int, ...] | None = None
    best_full_d = math.inf
    best_full_d_order: tuple[int, ...] | None = None
    best_fixed_g = math.inf
    best_fixed_g_order: tuple[int, ...] | None = None
    best_full_g = math.inf
    best_full_g_order: tuple[int, ...] | None = None
    total_fixed_g = 0.0
    order_count = 0
    for order in itertools.permutations(range(1, len(probabilities) + 1)):
        value = open_cost(order, probabilities, residual)
        if value < best_fixed:
            best_fixed = value
            best_order = order
        full_e = residual[0][order[0]] + sum(
            residual[first][second] for first, second in zip(order, order[1:])
        )
        if full_e < best_full_e:
            best_full_e = full_e
            best_full_e_order = order
        full_d = distance[0][order[0]] + sum(
            distance[first][second] for first, second in zip(order, order[1:])
        )
        if full_d < best_full_d:
            best_full_d = full_d
            best_full_d_order = order
        fixed_g = open_cost(order, probabilities, deep_residual)
        total_fixed_g += fixed_g
        order_count += 1
        if fixed_g < best_fixed_g:
            best_fixed_g = fixed_g
            best_fixed_g_order = order
        full_g = deep_residual[0][order[0]] + sum(
            deep_residual[first][second]
            for first, second in zip(order, order[1:])
        )
        if full_g < best_full_g:
            best_full_g = full_g
            best_full_g_order = order

    posterior_e = sum(
        weight * value
        for weight, value in zip(
            scenario_probability, posterior_open_costs(residual)
        )
    )
    pair_upper = sum(
        probabilities[index] * residual[0][index + 1]
        for index in range(len(probabilities))
    ) + sum(
        probabilities[first - 1]
        * probabilities[second - 1]
        * residual[first][second]
        for first in range(1, len(probabilities) + 1)
        for second in range(first + 1, len(probabilities) + 1)
    )
    root_residual_mass = sum(
        probabilities[index] * residual[0][index + 1]
        for index in range(len(probabilities))
    )
    return {
        "P": posterior_d,
        "R": radial_mass,
        "D": deficit,
        "M": farthest,
        "L": sum(
            probability * mean
            for probability, mean in zip(probabilities, means)
        ),
        "J": sum(
            probability * probability * mean
            for probability, mean in zip(probabilities, means)
        ),
        "overflow": overflow,
        "mu": means,
        "rho": radii,
        "e": residual,
        "fixed_e": best_fixed,
        "post_e": posterior_e,
        "order": best_order,
        "fixed_over_D": best_fixed / deficit if deficit > 1.0e-14 else math.inf,
        "fixed_over_post_e": (
            best_fixed / posterior_e if posterior_e > 1.0e-14 else math.inf
        ),
        "full_e_order": best_full_e_order,
        "full_e_stopped": open_cost(best_full_e_order, probabilities, residual),
        "full_d_order": best_full_d_order,
        "full_d_stopped_e": open_cost(best_full_d_order, probabilities, residual),
        "rho_up_stopped_e": open_cost(
            tuple(sorted(range(1, len(probabilities) + 1), key=radii.__getitem__)),
            probabilities,
            residual,
        ),
        "rho_down_stopped_e": open_cost(
            tuple(
                sorted(
                    range(1, len(probabilities) + 1),
                    key=radii.__getitem__,
                    reverse=True,
                )
            ),
            probabilities,
            residual,
        ),
        "pair_upper": pair_upper,
        "root_residual_mass": root_residual_mass,
        "fixed_g": best_fixed_g,
        "fixed_g_order": best_fixed_g_order,
        "uniform_g": total_fixed_g / order_count,
        "full_g_stopped": open_cost(
            best_full_g_order, probabilities, deep_residual
        ),
    }


def line_metric(client_count: int, rng: random.Random) -> list[list[float]]:
    points = [0.0]
    points.extend(sorted(rng.uniform(-50.0, 50.0) for _ in range(client_count)))
    return [[abs(first - second) for second in points] for first in points]


def euclidean_metric(client_count: int, rng: random.Random) -> list[list[float]]:
    points = [(0.0, 0.0)]
    points.extend(
        (rng.uniform(-30.0, 30.0), rng.uniform(-30.0, 30.0))
        for _ in range(client_count)
    )
    return [
        [math.dist(first, second) for second in points]
        for first in points
    ]


def star_metric(client_count: int, rng: random.Random) -> list[list[float]]:
    spokes = [0.0] + [10.0 ** rng.uniform(-1.0, 1.0) for _ in range(client_count)]
    return [
        [0.0 if first == second else spokes[first] + spokes[second]
         for second in range(client_count + 1)]
        for first in range(client_count + 1)
    ]


def probability_profiles(client_count: int, rng: random.Random):
    alphabet = (0.001, 0.005, 0.02, 0.1, 0.3, 0.5, 0.8, 0.98, 0.999)
    for value in alphabet:
        yield [value] * client_count
    for _ in range(10):
        yield [rng.choice(alphabet) for _ in range(client_count)]
    for _ in range(10):
        yield [10.0 ** rng.uniform(-3.0, -0.0005) for _ in range(client_count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    worst_direct = None
    worst_rounding = None
    worst_candidates = {
        "full_e_stopped": None,
        "full_d_stopped_e": None,
        "rho_up_stopped_e": None,
        "rho_down_stopped_e": None,
        "pair_upper": None,
        "root_residual_mass": None,
        "fixed_g": None,
        "uniform_g": None,
        "full_g_stopped": None,
        "M": None,
        "overflow": None,
    }
    metric_builders = (
        ("graph", random_graph_metric),
        ("line", line_metric),
        ("euclidean", euclidean_metric),
        ("star", star_metric),
    )
    for trial in range(args.trials):
        name, builder = metric_builders[trial % len(metric_builders)]
        distance = builder(args.n, rng)
        for probabilities in probability_profiles(args.n, rng):
            result = evaluate(distance, probabilities)
            direct_record = (
                result["fixed_over_D"], name, trial, probabilities, distance, result
            )
            rounding_record = (
                result["fixed_over_post_e"],
                name,
                trial,
                probabilities,
                distance,
                result,
            )
            if worst_direct is None or direct_record[0] > worst_direct[0]:
                worst_direct = direct_record
                print(
                    "worst fixed_e/D",
                    direct_record[0],
                    "metric",
                    name,
                    "trial",
                    trial,
                    "fixed_e/post_e",
                    result["fixed_over_post_e"],
                    flush=True,
                )
            if worst_rounding is None or rounding_record[0] > worst_rounding[0]:
                worst_rounding = rounding_record
                print(
                    "worst fixed_e/post_e",
                    rounding_record[0],
                    "metric",
                    name,
                    "trial",
                    trial,
                    "fixed_e/D",
                    result["fixed_over_D"],
                    flush=True,
                )
            for key in worst_candidates:
                denominator = (
                    result["P"] - result["L"]
                    if key in {"fixed_g", "uniform_g", "full_g_stopped"}
                    else result["D"]
                )
                if key == "overflow":
                    denominator = result["J"]
                ratio = result[key] / denominator if denominator > 1.0e-14 else math.inf
                old = worst_candidates[key]
                if old is None or ratio > old[0]:
                    worst_candidates[key] = (
                        ratio, name, trial, probabilities, distance, result
                    )
            if result["fixed_over_D"] > 9.0 + 1.0e-8:
                print("COUNTEREXAMPLE TO 9D", direct_record)
                return
    print("FINAL DIRECT", worst_direct)
    print("FINAL ROUNDING", worst_rounding)
    for key, record in worst_candidates.items():
        print("FINAL CANDIDATE", key, record)


if __name__ == "__main__":
    main()
