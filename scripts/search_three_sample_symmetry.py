#!/usr/bin/env python3
"""Search deterministic three-sample inequalities for compressed cores.

For sets X,Y,Z, write T(Y,Z)=f_Y(Z) and

    H(X;Y,Z) = tsp(T(Y,Z)) + 2 d(X,T(Y,Z)).

The main candidate checked here is the cyclic inequality

    sum_cyc H(A;B,C)
      <= sum_cyc [tsp(C) + 2 d(C,B)].

Its expectation would say that the compressed-core bound never exceeds the
ordinary one-guide bound.  No activation probabilities enter the pointwise
test, so exhaustive subset enumeration is possible.
"""

from __future__ import annotations

import argparse
import random
from fractions import Fraction
from typing import Sequence

from search_hazard_tree_ratio import named_metric, subset_tsp_costs


def bits(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def set_distance(
    source: int, target: int, distance: Sequence[Sequence[float]]
) -> float:
    n = len(distance) - 1
    selected = (n, *bits(target))
    return sum(min(distance[v][u] for u in selected) for v in bits(source))


def projected_mask(
    guide: int,
    sample: int,
    distance: Sequence[Sequence[float]],
    client_only: bool = False,
) -> int:
    n = len(distance) - 1
    guide_centers = tuple(bits(guide))
    if client_only and not guide_centers:
        return 0
    available = guide_centers if client_only else (*guide_centers, n)
    result = 0
    for v in bits(sample):
        image = min(available, key=lambda u: (distance[v][u], u))
        if image != n:
            result |= 1 << image
    return result


def master_cost(
    evaluation: int,
    guide: int,
    sample: int,
    distance: Sequence[Sequence[float]],
    tours: Sequence[float],
) -> float:
    core = projected_mask(guide, sample, distance)
    return tours[core] + 2.0 * set_distance(evaluation, core, distance)


def cyclic_gap(
    a: int,
    b: int,
    c: int,
    distance: Sequence[Sequence[float]],
    tours: Sequence[float],
) -> float:
    left = (
        master_cost(a, b, c, distance, tours)
        + master_cost(b, c, a, distance, tours)
        + master_cost(c, a, b, distance, tours)
    )
    right = (
        tours[a]
        + tours[b]
        + tours[c]
        + 2.0
        * (
            set_distance(c, b, distance)
            + set_distance(a, c, distance)
            + set_distance(b, a, distance)
        )
    )
    return left - right


def triple_statistics(
    a: int,
    b: int,
    c: int,
    distance: Sequence[Sequence[float]],
    tours: Sequence[float],
) -> tuple[float, float, float]:
    """Return cyclic master sum, posterior-tour sum, and all cross radii."""
    master = (
        master_cost(a, b, c, distance, tours)
        + master_cost(b, c, a, distance, tours)
        + master_cost(c, a, b, distance, tours)
    )
    posterior = tours[a] + tours[b] + tours[c]
    cross = sum(
        set_distance(x, y, distance)
        for x, y in (
            (a, b),
            (a, c),
            (b, a),
            (b, c),
            (c, a),
            (c, b),
        )
    )
    return master, posterior, cross


def search_metric(name: str, distance: Sequence[Sequence[float]]) -> tuple:
    n = len(distance) - 1
    tours = subset_tsp_costs(distance)
    worst = (-float("inf"), 0, 0, 0)
    constraints = []
    for a in range(1 << n):
        for b in range(1 << n):
            for c in range(1 << n):
                gap = cyclic_gap(a, b, c, distance, tours)
                constraints.append((*triple_statistics(a, b, c, distance, tours), a, b, c))
                if gap > worst[0]:
                    worst = (gap, a, b, c)
    print(f"{name}: max(left-right)={worst[0]:.12g}, masks={worst[1:]}")
    # Best factor c+2 alpha obtainable from a nonnegative inequality
    # master <= c * posterior + alpha * (all six directed cross radii).
    best = None
    for step in range(2001):
        alpha = 2.0 * step / 2000.0
        required_c = 0.0
        witness = None
        feasible = True
        for master, posterior, cross, a, b, c in constraints:
            if posterior == 0.0:
                if master > alpha * cross + 1e-9:
                    feasible = False
                    break
                continue
            candidate = (master - alpha * cross) / posterior
            if candidate > required_c:
                required_c = candidate
                witness = (a, b, c)
        if feasible:
            factor = max(required_c, 0.0) + 2.0 * alpha
            if best is None or factor < best[0]:
                best = (factor, max(required_c, 0.0), alpha, witness)
    print(f"{name}: best c+2alpha grid={best}")
    return (*worst, best)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--metrics", type=int, default=20)
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
    global_worst = None
    for i in range(args.metrics):
        family = families[i % len(families)]
        distance = named_metric(family, args.clients, rng)
        result = search_metric(f"{i}:{family}", distance)
        if global_worst is None or result[0] > global_worst[0]:
            global_worst = (result[0], family, result[1:], distance)
    print("global worst =", global_worst)


if __name__ == "__main__":
    main()
