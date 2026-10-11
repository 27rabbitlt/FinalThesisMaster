#!/usr/bin/env python3
"""Exact checks for the posterior CFL point and exponential-core identities.

All arithmetic is rational.  The script checks:

* the scenario-wise CFL cut constraints for a small nontrivial metric;
* the averaged identities B=P/2 and C=R when sampled vertices self-assign;
* the geometric nested-core layer-cake identity; and
* the fixed-center obstruction to replacing a load-aware backbone by a raw
  metric MST.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from fractions import Fraction


Q = Fraction


def probability(mask: int, p: tuple[Fraction, ...]) -> Fraction:
    value = Q(1)
    for i, p_i in enumerate(p):
        value *= p_i if mask & (1 << i) else 1 - p_i
    return value


def best_tour(
    d: tuple[tuple[Fraction, ...], ...], mask: int
) -> tuple[Fraction, tuple[int, ...]]:
    n = len(d) - 1
    root = n
    active = tuple(i for i in range(n) if mask & (1 << i))
    if not active:
        return Q(0), tuple()
    candidates = []
    for order in itertools.permutations(active):
        cost = d[root][order[0]] + d[order[-1]][root]
        cost += sum(d[u][v] for u, v in zip(order, order[1:]))
        candidates.append((cost, order))
    return min(candidates)


def half_incidence(order: tuple[int, ...], root: int) -> dict[tuple[int, int], Fraction]:
    incidence: dict[tuple[int, int], int] = defaultdict(int)
    if order:
        walk = (root, *order, root)
        for u, v in zip(walk, walk[1:]):
            incidence[tuple(sorted((u, v)))] += 1
    return {edge: Q(mult, 2) for edge, mult in incidence.items()}


def check_fractional_cfl() -> None:
    # Three clients followed by the depot.  This is a strict positive metric
    # with enough asymmetry to exercise nearest-center and tour tie breaking.
    d = (
        (Q(0), Q(1), Q(3, 2), Q(7, 4)),
        (Q(1), Q(0), Q(5, 4), Q(3, 2)),
        (Q(3, 2), Q(5, 4), Q(0), Q(1)),
        (Q(7, 4), Q(3, 2), Q(1), Q(0)),
    )
    p = (Q(1, 3), Q(2, 5), Q(3, 7))
    n = len(p)
    root = n
    z_bar: dict[tuple[int, int], Fraction] = defaultdict(Fraction)
    x_bar: dict[tuple[int, int], Fraction] = defaultdict(Fraction)
    posterior = Q(0)
    radius = Q(0)

    for mask in range(1 << n):
        prob = probability(mask, p)
        tour_cost, order = best_tour(d, mask)
        posterior += prob * tour_cost
        z_scenario = half_incidence(order, root)

        images = []
        active = tuple(i for i in range(n) if mask & (1 << i))
        for i in range(n):
            if mask & (1 << i):
                image = i
            else:
                image = min((*active, root), key=lambda j: (d[i][j], j))
            images.append(image)
            x_bar[i, image] += prob
            radius += p[i] * prob * d[i][image]
        for edge, value in z_scenario.items():
            z_bar[edge] += prob * value

        # Audit every scenario-wise cut constraint in (82).
        for client in range(n):
            for set_mask in range(1 << n):
                rhs = Q(int(bool(set_mask & (1 << images[client]))))
                lhs = Q(0)
                for (u, v), value in z_scenario.items():
                    u_inside = u < n and bool(set_mask & (1 << u))
                    v_inside = v < n and bool(set_mask & (1 << v))
                    if u_inside != v_inside:
                        lhs += value
                assert lhs >= rhs

    for i in range(n):
        assert sum(x_bar[i, j] for j in range(n + 1)) == 1
    backbone_cost = sum(d[u][v] * value for (u, v), value in z_bar.items())
    assignment_cost = sum(
        p[i] * d[i][j] * x_bar[i, j]
        for i in range(n)
        for j in range(n + 1)
    )
    assert backbone_cost == posterior / 2
    assert assignment_cost == radius
    assert all(x_bar[i, i] == p[i] for i in range(n))


def check_nested_core_identity() -> None:
    epsilon = Q(2, 11)
    lengths_and_loads = ((Q(3, 5), 1), (Q(7, 6), 2), (Q(5, 4), 5))
    direct = sum(
        length * (1 - (1 - epsilon) ** load)
        for length, load in lengths_and_loads
    )
    layers = sum(
        epsilon
        * (1 - epsilon) ** (j - 1)
        * sum(length for length, load in lengths_and_loads if load >= j)
        for j in range(1, 1 + max(load for _, load in lengths_and_loads))
    )
    assert direct == layers


def check_raw_mst_core_obstruction() -> None:
    t = 5
    epsilon = Q(1, t**3)
    m = t**4
    n = t**2
    q_far = 1 - (1 - epsilon) ** (m + 1)
    raw_mst_lower_bound = n * q_far
    non_mst_upper_bound = 2 * q_far + 2 * (n - 1) * epsilon
    assert raw_mst_lower_bound > 10 * non_mst_upper_bound


def main() -> None:
    check_fractional_cfl()
    check_nested_core_identity()
    check_raw_mst_core_obstruction()
    print("exact posterior-CFL and exponential nested-core checks passed")


if __name__ == "__main__":
    main()
