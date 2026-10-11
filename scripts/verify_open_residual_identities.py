#!/usr/bin/env python3
"""Exact rational audit of the proved open-residual identities.

This checks the algebra in ``thesis/symmetric_gap_open_residual_notes.md`` on
several finite metrics.  It is not a certificate for either conjectural
universal inequality Q_e <= 9D or M <= D; those ratios are printed only as
diagnostics.  All asserted identities and inequalities use Fraction arithmetic.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations


Q = Fraction


def subset_probability(mask: int, probabilities: tuple[Q, ...]) -> Q:
    value = Q(1)
    for index, probability in enumerate(probabilities):
        value *= probability if mask & (1 << index) else 1 - probability
    return value


def active_clients(mask: int, count: int) -> tuple[int, ...]:
    return tuple(index + 1 for index in range(count) if mask & (1 << index))


def cycle_cost(order: tuple[int, ...], distance) -> Q:
    if not order:
        return Q(0)
    return (
        distance[0][order[0]]
        + sum((distance[u][v] for u, v in zip(order, order[1:])), Q(0))
        + distance[order[-1]][0]
    )


def open_path_cost(order: tuple[int, ...], weights) -> Q:
    if not order:
        return Q(0)
    return weights[0][order[0]] + sum(
        (weights[u][v] for u, v in zip(order, order[1:])), Q(0)
    )


def subset_optimal_orders(distance, count: int):
    costs = [Q(0)] * (1 << count)
    orders = [tuple()] * (1 << count)
    for mask in range(1, 1 << count):
        clients = active_clients(mask, count)
        best_order = min(
            permutations(clients), key=lambda order: (cycle_cost(order, distance), order)
        )
        costs[mask] = cycle_cost(best_order, distance)
        orders[mask] = best_order
    return costs, orders


def nearest_distances(mask: int, distance, count: int):
    nearest = []
    second_gap = []
    for client in range(1, count + 1):
        candidates = [distance[client][0], distance[client][0]]
        candidates.extend(
            distance[client][other]
            for other in active_clients(mask, count)
            if other != client
        )
        candidates.sort()
        nearest.append(candidates[0])
        second_gap.append(candidates[1] - candidates[0])
    return tuple(nearest), tuple(second_gap)


def stopped_cost(full_order: tuple[int, ...], mask: int, weights) -> Q:
    order = tuple(
        vertex for vertex in full_order if mask & (1 << (vertex - 1))
    )
    return open_path_cost(order, weights)


def stopped_cycle_cost(full_order: tuple[int, ...], mask: int, weights) -> Q:
    order = tuple(
        vertex for vertex in full_order if mask & (1 << (vertex - 1))
    )
    return cycle_cost(order, weights)


def audit(name: str, raw_distance, probabilities: tuple[Q, ...]) -> None:
    count = len(probabilities)
    distance = tuple(tuple(Q(value) for value in row) for row in raw_distance)
    for first in range(count + 1):
        assert distance[first][first] == 0
        for second in range(count + 1):
            assert distance[first][second] == distance[second][first]
            for pivot in range(count + 1):
                assert (
                    distance[first][second]
                    <= distance[first][pivot] + distance[pivot][second]
                )
    scenario_probability = tuple(
        subset_probability(mask, probabilities) for mask in range(1 << count)
    )
    tsp, posterior_orders = subset_optimal_orders(distance, count)
    nearest_by_mask = []
    second_gap_by_mask = []
    for mask in range(1 << count):
        nearest, second_gap = nearest_distances(mask, distance, count)
        nearest_by_mask.append(nearest)
        second_gap_by_mask.append(second_gap)

    means = tuple(
        sum(
            (scenario_probability[mask] * nearest_by_mask[mask][index]
             for mask in range(1 << count)),
            Q(0),
        )
        for index in range(count)
    )
    radii = (Q(0),) + tuple(
        (1 - probabilities[index]) * means[index] for index in range(count)
    )
    mean_radii = (Q(0), *means)
    residual = tuple(
        tuple(
            max(distance[u][v] - radii[u] - radii[v], Q(0))
            for v in range(count + 1)
        )
        for u in range(count + 1)
    )
    deep_residual = tuple(
        tuple(
            max(distance[u][v] - mean_radii[u] - mean_radii[v], Q(0))
            for v in range(count + 1)
        )
        for u in range(count + 1)
    )

    posterior = sum(
        (scenario_probability[mask] * tsp[mask] for mask in range(1 << count)),
        Q(0),
    )
    linear_nearest = sum(
        (probabilities[index] * means[index] for index in range(count)), Q(0)
    )
    quadratic_nearest = sum(
        (
            probabilities[index] ** 2 * means[index]
            for index in range(count)
        ),
        Q(0),
    )
    radial = sum(
        (probabilities[index] * radii[index + 1] for index in range(count)),
        Q(0),
    )
    deficit = posterior - radial
    assert radial == linear_nearest - quadratic_nearest
    assert posterior >= linear_nearest
    assert Q(0) <= quadratic_nearest <= deficit

    b = tuple(
        sum(
            (
                scenario_probability[mask] * second_gap_by_mask[mask][index]
                for mask in range(1 << count)
            ),
            Q(0),
        )
        for index in range(count)
    )
    assert posterior >= linear_nearest + sum(
        (probabilities[index] * b[index] for index in range(count)), Q(0)
    ) / 2

    abs_rho = sum(
        (
            probabilities[index]
            * sum(
                (
                    scenario_probability[mask]
                    * abs(nearest_by_mask[mask][index] - radii[index + 1])
                    for mask in range(1 << count)
                ),
                Q(0),
            )
            for index in range(count)
        ),
        Q(0),
    )
    assert abs_rho <= 4 * deficit - quadratic_nearest
    positive_rho = sum(
        (
            probabilities[index]
            * sum(
                (
                    scenario_probability[mask]
                    * max(
                        nearest_by_mask[mask][index] - radii[index + 1], Q(0)
                    )
                    for mask in range(1 << count)
                ),
                Q(0),
            )
            for index in range(count)
        ),
        Q(0),
    )
    assert 2 * positive_rho == abs_rho + quadratic_nearest
    assert positive_rho <= 2 * deficit

    abs_mu = sum(
        (
            probabilities[index]
            * sum(
                (
                    scenario_probability[mask]
                    * abs(nearest_by_mask[mask][index] - means[index])
                    for mask in range(1 << count)
                ),
                Q(0),
            )
            for index in range(count)
        ),
        Q(0),
    )
    assert abs_mu <= 2 * sum(
        (probabilities[index] * b[index] for index in range(count)), Q(0)
    )
    assert abs_mu / 2 <= 2 * (posterior - linear_nearest)

    posterior_residual_path = sum(
        (
            scenario_probability[mask]
            * open_path_cost(posterior_orders[mask], residual)
            for mask in range(1 << count)
        ),
        Q(0),
    )
    posterior_deep_path = sum(
        (
            scenario_probability[mask]
            * open_path_cost(posterior_orders[mask], deep_residual)
            for mask in range(1 << count)
        ),
        Q(0),
    )
    assert posterior_residual_path <= 3 * deficit - quadratic_nearest
    assert posterior_deep_path <= 3 * (posterior - linear_nearest)

    farthest = sum(
        (
            scenario_probability[mask]
            * max(
                (distance[0][vertex] for vertex in active_clients(mask, count)),
                default=Q(0),
            )
            for mask in range(1 << count)
        ),
        Q(0),
    )
    assert 2 * farthest <= posterior

    best_residual = None
    best_deep = None
    for full_order in permutations(range(1, count + 1)):
        open_d = sum(
            (
                scenario_probability[mask]
                * stopped_cost(full_order, mask, distance)
                for mask in range(1 << count)
            ),
            Q(0),
        )
        open_e = sum(
            (
                scenario_probability[mask]
                * stopped_cost(full_order, mask, residual)
                for mask in range(1 << count)
            ),
            Q(0),
        )
        open_g = sum(
            (
                scenario_probability[mask]
                * stopped_cost(full_order, mask, deep_residual)
                for mask in range(1 << count)
            ),
            Q(0),
        )
        closed_d = sum(
            (
                scenario_probability[mask]
                * stopped_cycle_cost(full_order, mask, distance)
                for mask in range(1 << count)
            ),
            Q(0),
        )
        assert open_d <= 2 * radial + open_e
        assert closed_d <= open_d + farthest
        assert open_e <= 2 * quadratic_nearest + open_g
        best_residual = open_e if best_residual is None else min(best_residual, open_e)
        best_deep = open_g if best_deep is None else min(best_deep, open_g)

    assert best_residual is not None and best_deep is not None
    overflow = sum(
        (
            scenario_probability[mask]
            * max(
                sum(
                    (
                        nearest_by_mask[mask][vertex - 1]
                        for vertex in active_clients(mask, count)
                    ),
                    Q(0),
                )
                + max(
                    (
                        distance[0][vertex]
                        for vertex in active_clients(mask, count)
                    ),
                    default=Q(0),
                )
                - tsp[mask],
                Q(0),
            )
            for mask in range(1 << count)
        ),
        Q(0),
    )

    print(name)
    print(f"  P={posterior}, R={radial}, D={deficit}, L={linear_nearest}, J={quadratic_nearest}")
    print(f"  posterior e-path={posterior_residual_path}, bound={3*deficit-quadratic_nearest}")
    print(f"  posterior g-path={posterior_deep_path}, bound={3*(posterior-linear_nearest)}")
    print(f"  best Qe/D={float(best_residual/deficit):.12f}")
    print(f"  best Qg/(P-L)={float(best_deep/(posterior-linear_nearest)):.12f}")
    print(f"  M/D={float(farthest/deficit):.12f}, overflow/J={float(overflow/quadratic_nearest):.12f}")


def main() -> None:
    audit(
        "heterogeneous five-client graph metric",
        (
            (0, 8, 11, 5, 13, 9),
            (8, 0, 3, 7, 5, 10),
            (11, 3, 0, 6, 8, 7),
            (5, 7, 6, 0, 9, 4),
            (13, 5, 8, 9, 0, 6),
            (9, 10, 7, 4, 6, 0),
        ),
        (Q(1, 5), Q(2, 3), Q(1, 20), Q(4, 5), Q(1, 2)),
    )
    audit(
        "six-client self-thinning witness",
        (
            (0, 18, 20, 32, 27, 16, 24),
            (18, 0, 2, 14, 9, 7, 6),
            (20, 2, 0, 12, 9, 5, 6),
            (32, 14, 12, 0, 6, 17, 9),
            (27, 9, 9, 6, 0, 14, 3),
            (16, 7, 5, 17, 14, 0, 11),
            (24, 6, 6, 9, 3, 11, 0),
        ),
        (Q(1, 2),) * 6,
    )
    print("all exact proved-identity audits passed")


if __name__ == "__main__":
    main()
