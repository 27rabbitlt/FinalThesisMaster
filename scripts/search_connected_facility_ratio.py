#!/usr/bin/env python3
"""Exact experiments for the connected-facility structural bound.

For a symmetric stochastic TSP instance, compute

    G* = min_{S contains r} mst(S) + sum_v p_v d(v,S)

and compare it with the exact expected posterior tour value P.  Attaching every
client as a leaf to a nearest point of S shows that the optimal rooted hazard
tree has value at most G*.  Consequently G* <= 1.45 P would by itself imply a
2.9 a-priori/posterior bound through a doubled-tree master order.

This is a discovery/stress-test tool, not a numerical proof of that inequality.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Sequence

from search_hazard_tree_ratio import (
    expected_posterior,
    metric_closure,
    named_metric,
    probability_vectors,
    subset_tsp_costs,
)


@dataclass(frozen=True)
class FacilityEvaluation:
    posterior: float
    facility: float
    ratio: float
    centers_mask: int


def mst_cost(d: Sequence[Sequence[float]], vertices: Sequence[int]) -> float:
    """Prim's algorithm on the complete metric induced by ``vertices``."""
    if len(vertices) <= 1:
        return 0.0
    used = {vertices[0]}
    best = {v: d[vertices[0]][v] for v in vertices[1:]}
    total = 0.0
    while len(used) < len(vertices):
        v = min((u for u in vertices if u not in used), key=best.__getitem__)
        total += best[v]
        used.add(v)
        for u in vertices:
            if u not in used:
                best[u] = min(best[u], d[v][u])
    return total


def best_connected_facility(
    d: Sequence[Sequence[float]], p: Sequence[float]
) -> tuple[float, int]:
    """Return G* and the client-center mask attaining it."""
    n = len(p)
    root = n
    best_value = float("inf")
    best_mask = 0
    for mask in range(1 << n):
        centers = [root] + [v for v in range(n) if mask & (1 << v)]
        value = mst_cost(d, centers)
        for v, pv in enumerate(p):
            value += pv * min(d[v][s] for s in centers)
        if value < best_value - 1e-14:
            best_value = value
            best_mask = mask
    return best_value, best_mask


def evaluate(
    d: Sequence[Sequence[float]],
    p: Sequence[float],
    subset_costs: Sequence[float] | None = None,
) -> FacilityEvaluation:
    if subset_costs is None:
        subset_costs = subset_tsp_costs(d)
    posterior = expected_posterior(p, subset_costs)
    facility, centers = best_connected_facility(d, p)
    return FacilityEvaluation(
        posterior=posterior,
        facility=facility,
        ratio=facility / posterior if posterior else 0.0,
        centers_mask=centers,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--metrics", type=int, default=100)
    parser.add_argument("--probabilities", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--evolve", action="store_true")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument(
        "--multiscale",
        action="store_true",
        help="evolve a log-scale weighted graph and take its metric closure",
    )
    parser.add_argument(
        "--families",
        default="unit,path,cycle,star,binary,random-tree,graph12,graph123,"
        "client-graph-19,euclidean",
    )
    args = parser.parse_args()

    if args.evolve:
        evolutionary_search(args)
        return

    rng = random.Random(args.seed)
    incumbent = None
    families = args.families.split(",")
    for metric_index in range(args.metrics):
        family = families[metric_index % len(families)]
        d = named_metric(family, args.clients, rng)
        costs = subset_tsp_costs(d)
        for p in probability_vectors(args.clients, rng, args.probabilities):
            result = evaluate(d, p, costs)
            if incumbent is None or result.ratio > incumbent[0].ratio:
                incumbent = (result, family, tuple(p), d)
                print(
                    f"best ratio={result.ratio:.12f} family={family} "
                    f"P={result.posterior:.12f} G={result.facility:.12f} "
                    f"centers=0x{result.centers_mask:x}",
                    flush=True,
                )
                print("p =", " ".join(f"{x:.12g}" for x in p), flush=True)

    assert incumbent is not None
    result, family, p, d = incumbent
    print("final family =", family)
    print("ratio =", format(result.ratio, ".17g"))
    print("posterior =", format(result.posterior, ".17g"))
    print("facility =", format(result.facility, ".17g"))
    print("centers_mask =", result.centers_mask)
    print("p =", " ".join(format(x, ".17g") for x in p))
    print("d =")
    for row in d:
        print(" ".join(format(x, ".17g") for x in row))


def evolutionary_search(args: argparse.Namespace) -> None:
    """Jointly optimize a [1,2]-metric and heterogeneous probabilities."""
    from scipy.optimize import differential_evolution

    n = args.clients
    total = n + 1
    pairs = [(i, j) for i in range(total) for j in range(i)]
    calls = 0
    incumbent = None

    def decode(x: Sequence[float]) -> tuple[list[list[float]], tuple[float, ...]]:
        d = [[0.0] * total for _ in range(total)]
        for value, (i, j) in zip(x, pairs):
            length = math.exp(value) if args.multiscale else value
            d[i][j] = d[j][i] = length
        if args.multiscale:
            d = metric_closure(d)
        p = tuple(
            1.0 / (1.0 + math.exp(-z)) for z in x[len(pairs) :]
        )
        return d, p

    def objective(x: Sequence[float]) -> float:
        nonlocal calls, incumbent
        calls += 1
        d, p = decode(x)
        result = evaluate(d, p)
        if incumbent is None or result.ratio > incumbent[0].ratio:
            incumbent = (result, d, p)
            print(
                f"call={calls} best ratio={result.ratio:.12f} "
                f"P={result.posterior:.12f} G={result.facility:.12f} "
                f"centers=0x{result.centers_mask:x}",
                flush=True,
            )
        return -result.ratio

    edge_bounds = [(-4.0, 4.0)] if args.multiscale else [(1.0, 2.0)]
    bounds = edge_bounds * len(pairs) + [(-7.0, 7.0)] * n
    result = differential_evolution(
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
    evaluation, d, p = incumbent
    print("optimizer:", result.message)
    print("ratio =", format(evaluation.ratio, ".17g"))
    print("posterior =", format(evaluation.posterior, ".17g"))
    print("facility =", format(evaluation.facility, ".17g"))
    print("centers_mask =", evaluation.centers_mask)
    print("p =", " ".join(format(x, ".17g") for x in p))
    print("d =")
    for row in d:
        print(" ".join(format(x, ".17g") for x in row))


if __name__ == "__main__":
    main()
