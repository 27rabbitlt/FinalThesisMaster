#!/usr/bin/env python3
"""Falsification search for a sampled-cycle hazard-tree construction.

Draw an independent guide realization B, take an optimal depot-rooted cycle
on B, attach every nonguide client to a nearest guide (or the depot), and
delete one guide-cycle edge with probability proportional to its length.
For an independent evaluation realization A, the expected pruned-backbone
cost is L - sum_i g_i^2/L, where the g_i are the gaps between occupied
Voronoi cells.  Thus the averaged hazard of this randomized tree is

    H = P + R - S,

where P is the posterior optimum, R is the expected attachment cost, and S
is the expected squared-gap deletion credit.  This program searches for
violations of H <= 3P/2 and H <= 29P/20 on tiny instances.  A passing run is
only evidence, not a proof.
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Sequence

from search_hazard_tree_ratio import named_metric, probability_vectors
from search_occupied_backbone_ratio import mask_probability, optimal_tours


def squared_gap_credit(
    d: Sequence[Sequence[float]],
    order: Sequence[int],
    q: Sequence[float],
) -> float:
    """Expected sum of squared occupied-cell gaps divided by cycle length."""
    root = len(d) - 1
    broken = (root, *order, root)
    edge_lengths = [d[u][v] for u, v in zip(broken, broken[1:])]
    cycle_length = sum(edge_lengths)
    if cycle_length == 0.0:
        return 0.0

    answer = 0.0
    for mask in range(1 << len(order)):
        probability = 1.0
        marked_positions = [0]
        for i, qi in enumerate(q):
            if mask & (1 << i):
                probability *= qi
                marked_positions.append(i + 1)
            else:
                probability *= 1.0 - qi
        marked_positions.append(len(order) + 1)

        # The two root occurrences are one cyclic mark.  Consecutive entries
        # in this root-broken representation are exactly the cyclic gaps.
        square_sum = 0.0
        for left, right in zip(marked_positions, marked_positions[1:]):
            gap = sum(edge_lengths[left:right])
            square_sum += gap * gap
        answer += probability * square_sum / cycle_length
    return answer


def evaluate(
    d: Sequence[Sequence[float]],
    p: Sequence[float],
    tours: tuple[list[float], list[tuple[tuple[int, ...], ...]]] | None = None,
) -> tuple[float, float, float, float]:
    """Return (P,R,S,H) for the best optimal-tour tie break for each B."""
    n = len(p)
    root = n
    costs, orders = tours if tours is not None else optimal_tours(d)
    posterior = sum(mask_probability(mask, p) * costs[mask] for mask in range(1 << n))
    radius = 0.0
    saving = 0.0

    for guide_mask in range(1 << n):
        guide_probability = mask_probability(guide_mask, p)
        centers = tuple(v for v in range(n) if guide_mask & (1 << v))
        available = (*centers, root)

        image: list[int] = []
        for v in range(n):
            if guide_mask & (1 << v):
                image.append(v)
            else:
                image.append(min(available, key=lambda b: (d[v][b], b)))
            radius += guide_probability * p[v] * d[v][image[-1]]

        guide_orders = orders[guide_mask]
        representative_order = guide_orders[0]
        occupancy_by_center: dict[int, float] = {}
        for center in representative_order:
            inactive = 1.0
            for v in range(n):
                if image[v] == center:
                    inactive *= 1.0 - p[v]
            occupancy_by_center[center] = 1.0 - inactive

        # Cell occupancies depend on the centers, not on their cyclic order.
        # Choose the optimal posterior-tour tie break with the largest credit.
        saving += guide_probability * max(
            squared_gap_credit(
                d,
                order,
                [occupancy_by_center[center] for center in order],
            )
            for order in guide_orders
        )

    hazard = posterior + radius - saving
    return posterior, radius, saving, hazard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--metrics", type=int, default=200)
    parser.add_argument("--probabilities", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--families",
        default=(
            "unit,path,cycle,star,binary,random-tree,graph12,graph123,"
            "client-graph-19,euclidean,multiscale-graph,multiscale-complete"
        ),
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
            ratio = result[-1] / result[0] if result[0] else 0.0
            if worst is None or ratio > worst[0]:
                worst = (ratio, family, tuple(p), result, d)
                print(
                    f"worst H/P={ratio:.12f} family={family} "
                    f"P={result[0]:.12f} R={result[1]:.12f} "
                    f"S={result[2]:.12f}",
                    flush=True,
                )
                if ratio > 1.5 + 1e-9:
                    print("THREE-HALVES COUNTEREXAMPLE")
                    print("p =", " ".join(format(x, ".17g") for x in p))
                    print("d =")
                    for row in d:
                        print(" ".join(format(x, ".17g") for x in row))
                    return
    print("final worst =", worst)


if __name__ == "__main__":
    main()
