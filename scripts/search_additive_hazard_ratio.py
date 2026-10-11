#!/usr/bin/env python3
"""Exact small-instance search for deficit-sensitive hazard-tree bounds.

For a finite metric and independent activation probabilities, compute exactly
(up to floating-point arithmetic)

    P     = E[tsp(B)],
    R     = sum_v p_v E[d(v,B)],
    D     = P-R,
    X     = 2(P-L+E_rho),
    Phi   = min_T Phi_p(T),

using subset enumeration and the hazard-tree DP from
``search_hazard_tree_ratio.py``.  The two diagnostic ratios are

    (Phi-R)/D   and   (Phi-R)/X.

Thus a value above 10 refutes ``Phi <= R+10D``, while a value above 1
refutes the stronger ``Phi <= R+X``.  This is a structural counterexample
search, not a tuning search for the predecessor proof's scalar parameters.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Sequence

from search_hazard_tree_ratio import (
    best_hazard_tree,
    expected_posterior,
    metric_closure,
    named_metric,
    probability_vectors,
    subset_tsp_costs,
)


@dataclass(frozen=True)
class AdditiveEvaluation:
    posterior: float
    radius_mass: float
    deficit: float
    nearest_neighbor_mass: float
    radius_deviation: float
    edge_slack: float
    hazard: float
    deficit_ratio: float
    slack_ratio: float
    parents: tuple[int, ...]


def subset_probabilities(p: Sequence[float]) -> list[float]:
    values = [1.0]
    for pv in p:
        values = [x * (1.0 - pv) for x in values] + [x * pv for x in values]
    return values


def evaluate_additive(
    d: Sequence[Sequence[float]],
    p: Sequence[float],
    subset_costs: Sequence[float] | None = None,
) -> AdditiveEvaluation:
    """Compute P,R,D,X and the exact best rooted hazard-tree objective."""
    n = len(p)
    root = n
    if subset_costs is None:
        subset_costs = subset_tsp_costs(d)
    probabilities = subset_probabilities(p)
    posterior = sum(a * b for a, b in zip(probabilities, subset_costs))

    rho = [0.0] * n
    mu = [0.0] * n
    e_rho = 0.0
    for mask, probability in enumerate(probabilities):
        active = [u for u in range(n) if mask & (1 << u)]
        for v in range(n):
            if mask & (1 << v):
                x_v = 0.0
            else:
                x_v = min([d[v][root]] + [d[v][u] for u in active])
            rho[v] += probability * x_v

            # D_v=d(v,B\{v}); unlike X_v, it ignores v even when v is active.
            others = [u for u in active if u != v]
            d_v = min([d[v][root]] + [d[v][u] for u in others])
            mu[v] += probability * d_v

    # A second pass is needed because E_rho uses the already averaged rho_v.
    for mask, probability in enumerate(probabilities):
        active = [u for u in range(n) if mask & (1 << u)]
        for v in active:
            others = [u for u in active if u != v]
            d_v = min([d[v][root]] + [d[v][u] for u in others])
            e_rho += probability * abs(d_v - rho[v])

    radius_mass = sum(pv * rv for pv, rv in zip(p, rho))
    deficit = posterior - radius_mass
    ell = sum(pv * mv for pv, mv in zip(p, mu))
    edge_slack = 2.0 * (posterior - ell + e_rho)
    scale = max(1.0, posterior, radius_mass, edge_slack)
    assert all(
        abs(rv - (1.0 - pv) * mv) <= 1e-10 * scale
        for pv, rv, mv in zip(p, rho, mu)
    )
    # Jensen gives E_rho >= sum_v p_v^2 mu_v, hence X >= 2D;
    # the verified edge-charge lemma gives X <= 10D.
    assert edge_slack + 1e-10 * scale >= 2.0 * deficit
    assert edge_slack <= 10.0 * deficit + 1e-10 * scale
    hazard, parents = best_hazard_tree(d, p)
    excess = hazard - radius_mass
    deficit_ratio = excess / deficit if deficit > 1e-14 else -math.inf
    slack_ratio = excess / edge_slack if edge_slack > 1e-14 else -math.inf
    return AdditiveEvaluation(
        posterior,
        radius_mass,
        deficit,
        ell,
        e_rho,
        edge_slack,
        hazard,
        deficit_ratio,
        slack_ratio,
        parents,
    )


def random_search(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    families = args.families.split(",")
    best_d = None
    best_s = None
    for index in range(args.metrics):
        family = families[index % len(families)]
        d = named_metric(family, args.clients, rng)
        costs = subset_tsp_costs(d)
        for p in probability_vectors(args.clients, rng, args.probabilities):
            result = evaluate_additive(d, p, costs)
            record = (result.deficit_ratio, family, d, p, result)
            if best_d is None or record[0] > best_d[0]:
                best_d = record
                print(
                    f"best (Phi-R)/D={record[0]:.12f} family={family} "
                    f"(Phi-R)/X={result.slack_ratio:.12f}",
                    flush=True,
                )
            record_s = (result.slack_ratio, family, d, p, result)
            if best_s is None or record_s[0] > best_s[0]:
                best_s = record_s
                print(
                    f"best (Phi-R)/X={record_s[0]:.12f} family={family} "
                    f"(Phi-R)/D={result.deficit_ratio:.12f}",
                    flush=True,
                )
    for label, record in (("deficit", best_d), ("slack", best_s)):
        assert record is not None
        _, family, d, p, result = record
        print(f"\nBEST {label}: family={family}")
        print_result(d, p, result)


def evolutionary_search(args: argparse.Namespace) -> None:
    from scipy.optimize import differential_evolution

    n = args.clients
    total = n + 1
    pairs = [(i, j) for i in range(total) for j in range(i)]
    incumbent = None
    calls = 0

    def decode(x: Sequence[float]):
        d = [[0.0] * total for _ in range(total)]
        for value, (i, j) in zip(x, pairs):
            length = math.exp(value) if args.evolve_closure else value
            d[i][j] = d[j][i] = length
        if args.evolve_closure:
            d = metric_closure(d)
        p = tuple(
            1.0 / (1.0 + math.exp(-z)) for z in x[len(pairs) :]
        )
        return d, p

    def objective(x: Sequence[float]) -> float:
        nonlocal incumbent, calls
        calls += 1
        d, p = decode(x)
        result = evaluate_additive(d, p)
        score = (
            result.deficit_ratio
            if args.target == "deficit"
            else result.slack_ratio
        )
        if math.isfinite(score) and (incumbent is None or score > incumbent[0]):
            incumbent = (score, d, p, result)
            print(
                f"call={calls} best {args.target}={score:.12f} "
                f"other={result.slack_ratio if args.target == 'deficit' else result.deficit_ratio:.12f}",
                flush=True,
            )
        return -score if math.isfinite(score) else 1e20

    metric_bounds = [(-4.0, 4.0)] if args.evolve_closure else [(1.0, 2.0)]
    bounds = metric_bounds * len(pairs) + [(-8.0, 8.0)] * n
    differential_evolution(
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
    _, d, p, result = incumbent
    print_result(d, p, result)


def print_result(d, p, result: AdditiveEvaluation) -> None:
    print(f"P = {result.posterior:.17g}")
    print(f"R = {result.radius_mass:.17g}")
    print(f"D = {result.deficit:.17g}")
    print(f"L = {result.nearest_neighbor_mass:.17g}")
    print(f"E_rho = {result.radius_deviation:.17g}")
    print(f"X = {result.edge_slack:.17g}")
    print(f"Phi = {result.hazard:.17g}")
    print(f"(Phi-R)/D = {result.deficit_ratio:.17g}")
    print(f"(Phi-R)/X = {result.slack_ratio:.17g}")
    print("p =", " ".join(f"{x:.17g}" for x in p))
    print("parents =", result.parents)
    print("d =")
    for row in d:
        print(" ".join(f"{x:.17g}" for x in row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--metrics", type=int, default=100)
    parser.add_argument("--probabilities", type=int, default=20)
    parser.add_argument(
        "--families",
        default="graph12,graph123,euclidean,multiscale-graph,multiscale-complete,theta",
    )
    parser.add_argument("--evolve", action="store_true")
    parser.add_argument("--evolve-closure", action="store_true")
    parser.add_argument("--target", choices=("deficit", "slack"), default="deficit")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population", type=int, default=8)
    args = parser.parse_args()
    if args.evolve:
        evolutionary_search(args)
    else:
        random_search(args)


if __name__ == "__main__":
    main()
