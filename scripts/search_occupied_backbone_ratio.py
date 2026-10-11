#!/usr/bin/env python3
"""Falsification search for the correlated occupied-backbone gateway.

For each small independent-activation instance this computes, by exhaustive
enumeration, the quantities in equations (49)--(52) of the structural memo:

    Q = E_{A,B}[ cost(tau_B[S_B(A)]) ],
    R = sum_v p_v E_B[d(v,B)],
    D = P-R.

The proposed structural gateway is Q <= 0.9 P + 2 D.  This executable is only
a counterexample search; a passing run is not evidence of a proof.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
from typing import Sequence

from search_hazard_tree_ratio import metric_closure, named_metric, probability_vectors


def mask_probability(mask: int, p: Sequence[float]) -> float:
    value = 1.0
    for v, pv in enumerate(p):
        value *= pv if mask & (1 << v) else 1.0 - pv
    return value


def optimal_tours(
    d: Sequence[Sequence[float]],
) -> tuple[list[float], list[tuple[tuple[int, ...], ...]]]:
    """Return exact subset TSP costs and every optimal oriented order."""
    n = len(d) - 1
    root = n
    costs = [0.0] * (1 << n)
    orders: list[tuple[tuple[int, ...], ...]] = [(tuple(),) for _ in range(1 << n)]
    for mask in range(1, 1 << n):
        active = tuple(v for v in range(n) if mask & (1 << v))
        best_cost = math.inf
        best_orders: list[tuple[int, ...]] = []
        for order in itertools.permutations(active):
            value = d[root][order[0]] + d[order[-1]][root]
            value += sum(d[u][v] for u, v in zip(order, order[1:]))
            if value < best_cost - 1e-12:
                best_cost = value
                best_orders = [order]
            elif abs(value - best_cost) <= 1e-12:
                best_orders.append(order)
        costs[mask] = best_cost
        assert best_orders
        orders[mask] = tuple(best_orders)
    return costs, orders


def expected_shortcut(
    d: Sequence[Sequence[float]], order: Sequence[int], q: Sequence[float]
) -> float:
    """Expected root-closed shortcut cost for independent center occupancies."""
    root = len(d) - 1
    value = 0.0
    for mask in range(1 << len(order)):
        probability = 1.0
        occupied: list[int] = []
        for i, qi in enumerate(q):
            if mask & (1 << i):
                probability *= qi
                occupied.append(order[i])
            else:
                probability *= 1.0 - qi
        if not occupied:
            continue
        cost = d[root][occupied[0]] + d[occupied[-1]][root]
        cost += sum(d[u][v] for u, v in zip(occupied, occupied[1:]))
        value += probability * cost
    return value


def evaluate(
    d: Sequence[Sequence[float]],
    p: Sequence[float],
    tours: tuple[list[float], list[tuple[tuple[int, ...], ...]]] | None = None,
) -> tuple[float, float, float, float]:
    n = len(p)
    root = n
    costs, orders = tours if tours is not None else optimal_tours(d)
    posterior = sum(mask_probability(mask, p) * costs[mask] for mask in range(1 << n))
    radius = 0.0
    occupied_backbone = 0.0

    for guide_mask in range(1 << n):
        guide_probability = mask_probability(guide_mask, p)
        centers = tuple(v for v in range(n) if guide_mask & (1 << v))
        available = (*centers, root)

        # Every guide client is fixed to itself.  Other clients use a
        # deterministic nearest-center tie break.
        image: list[int] = []
        for v in range(n):
            if guide_mask & (1 << v):
                image.append(v)
            else:
                image.append(min(available, key=lambda b: (d[v][b], b)))
            radius += guide_probability * p[v] * d[v][image[-1]]

        guide_orders = orders[guide_mask]
        # The guide's optimal-tour tie breaking is ours to choose before the
        # independent evaluation sample.  Use the best such fixed order; a
        # counterexample must survive this freedom.
        representative_order = guide_orders[0]
        cell_occupancies: list[float] = []
        for center in representative_order:
            inactive = 1.0
            for v in range(n):
                if image[v] == center:
                    inactive *= 1.0 - p[v]
            cell_occupancies.append(1.0 - inactive)
        occupancy_by_center = dict(zip(representative_order, cell_occupancies))
        occupied_backbone += guide_probability * min(
            expected_shortcut(
                d,
                order,
                [occupancy_by_center[center] for center in order],
            )
            for order in guide_orders
        )

    deficit = posterior - radius
    slack = 0.9 * posterior + 2.0 * deficit - occupied_backbone
    return posterior, radius, occupied_backbone, slack


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--metrics", type=int, default=200)
    parser.add_argument("--probabilities", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument(
        "--families",
        default="unit,path,cycle,star,binary,random-tree,graph12,graph123,client-graph-19,euclidean,multiscale-graph,multiscale-complete",
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)
    families = args.families.split(",")
    worst: tuple[float, str, tuple[float, ...], tuple[float, ...], list[list[float]]] | None = None
    for metric_index in range(args.metrics):
        family = families[metric_index % len(families)]
        d = named_metric(family, args.clients, rng)
        tours = optimal_tours(d)
        for p in probability_vectors(args.clients, rng, args.probabilities):
            result = evaluate(d, p, tours)
            normalized_slack = result[-1] / result[0] if result[0] else math.inf
            if worst is None or normalized_slack < worst[0]:
                worst = (normalized_slack, family, tuple(p), result, d)
                print(
                    f"worst slack/P={normalized_slack:.12f} family={family} "
                    f"P={result[0]:.12f} R={result[1]:.12f} "
                    f"Q={result[2]:.12f}",
                    flush=True,
                )
                if normalized_slack < -1e-9:
                    print("COUNTEREXAMPLE p =", " ".join(format(x, ".17g") for x in p))
                    print("d =")
                    for row in d:
                        print(" ".join(format(x, ".17g") for x in row))
                    return
    print("final worst =", worst)


if __name__ == "__main__":
    main()
