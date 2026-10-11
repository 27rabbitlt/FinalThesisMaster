#!/usr/bin/env python3
"""Search the star-versus-full-MST hazard hybrid against posterior TSP.

This is an exploratory floating-point search.  For metrics with tied MSTs,
the checker enumerates all spanning trees only in explicit verification code;
the random search below uses continuous edge lengths and hence has a unique
MST almost surely.
"""

from __future__ import annotations

import argparse
import itertools
import random

from search_hazard_tree_ratio import (
    _tree_hazard_from_edges,
    expected_posterior,
    metric_closure,
    named_metric,
    probability_vectors,
    subset_tsp_costs,
)


def mst_edges(d: list[list[float]]) -> list[tuple[int, int]]:
    """Kruskal MST with deterministic tie breaking."""
    n = len(d)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edges = sorted((d[i][j], i, j) for i in range(n) for j in range(i))
    answer = []
    for _, i, j in edges:
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
            answer.append((i, j))
            if len(answer) == n - 1:
                break
    return answer


def mst_edges_on_nodes(
    d: list[list[float]], nodes: list[int]
) -> list[tuple[int, int]]:
    local = [[d[u][v] for v in nodes] for u in nodes]
    return [(nodes[i], nodes[j]) for i, j in mst_edges(local)]


def best_mst_backbone_spider(
    d: list[list[float]], p: list[float]
) -> tuple[float, tuple[int, ...], tuple[int, ...]]:
    """Exhaust the small-instance MST-backbone/leaf-assignment family.

    Pick client centers, join them and the depot by their metric MST, and
    attach every noncenter as a leaf either to the depot or to one center.
    Centers are required to belong to their own clusters.
    """
    n = len(p)
    root = n
    best = float("inf")
    best_centers: tuple[int, ...] = tuple()
    best_assignment: tuple[int, ...] = tuple([root] * n)
    for center_mask in range(1 << n):
        centers = tuple(v for v in range(n) if center_mask & (1 << v))
        noncenters = tuple(v for v in range(n) if not center_mask & (1 << v))
        backbone = mst_edges_on_nodes(d, [root, *centers]) if centers else []
        choices = (root, *centers)
        for images in itertools.product(choices, repeat=len(noncenters)):
            assignment = list(range(n))
            for v, image in zip(noncenters, images):
                assignment[v] = image
            edges = list(backbone)
            edges.extend((v, assignment[v]) for v in noncenters)
            value = _tree_hazard_from_edges(d, p, edges)
            if value < best - 1e-12:
                best = value
                best_centers = centers
                best_assignment = tuple(assignment)
    return best, best_centers, best_assignment


def best_assignment_for_centers(
    d: list[list[float]], p: list[float], centers: tuple[int, ...]
) -> tuple[float, tuple[int, ...]]:
    """Minimize the exact hazard over leaf assignments for fixed centers."""
    n = len(p)
    root = n
    noncenters = tuple(v for v in range(n) if v not in centers)
    backbone = mst_edges_on_nodes(d, [root, *centers]) if centers else []
    best = float("inf")
    best_assignment: tuple[int, ...] = tuple([root] * n)
    for images in itertools.product((root, *centers), repeat=len(noncenters)):
        assignment = list(range(n))
        for v, image in zip(noncenters, images):
            assignment[v] = image
        edges = [*backbone, *((v, assignment[v]) for v in noncenters)]
        value = _tree_hazard_from_edges(d, p, edges)
        if value < best - 1e-12:
            best = value
            best_assignment = tuple(assignment)
    return best, best_assignment


def sampled_center_average(d: list[list[float]], p: list[float]) -> float:
    """Average best fixed-center hazard when centers are an iid p-sample."""
    n = len(p)
    answer = 0.0
    for center_mask in range(1 << n):
        probability = 1.0
        centers = []
        for v, pv in enumerate(p):
            if center_mask & (1 << v):
                probability *= pv
                centers.append(v)
            else:
                probability *= 1.0 - pv
        value, _ = best_assignment_for_centers(d, p, tuple(centers))
        answer += probability * value
    return answer


def nearest_sample_radius(d: list[list[float]], p: list[float]) -> float:
    """Return R=sum_i p_i E[d(i,B union {root})] by subset enumeration."""
    n = len(p)
    root = n
    answer = 0.0
    for mask in range(1 << n):
        probability = 1.0
        active = []
        for v, pv in enumerate(p):
            if mask & (1 << v):
                probability *= pv
                active.append(v)
            else:
                probability *= 1.0 - pv
        for v, pv in enumerate(p):
            answer += probability * pv * min(d[v][u] for u in (*active, root))
    return answer


def evaluate(d: list[list[float]], p: list[float]) -> tuple[float, float, float, float]:
    posterior = expected_posterior(p, subset_tsp_costs(d))
    root = len(p)
    star = sum(p[v] * d[root][v] for v in range(len(p)))
    mst_hazard = _tree_hazard_from_edges(d, p, mst_edges(d))
    return posterior, star, mst_hazard, min(star, mst_hazard) / posterior


def random_metric(n: int, rng: random.Random) -> list[list[float]]:
    total = n + 1
    weights = [[0.0] * total for _ in range(total)]
    for i in range(total):
        for j in range(i):
            weights[i][j] = weights[j][i] = 10.0 ** rng.uniform(-2.0, 2.0)
    return metric_closure(weights)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=7)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument(
        "--full-family",
        action="store_true",
        help="exhaust the MST-backbone/leaf family instead of the two-topology hybrid",
    )
    parser.add_argument(
        "--families",
        default="unit,path,cycle,star,binary,random-tree,graph12,graph123,client-graph-19,euclidean,multiscale-graph,multiscale-complete",
    )
    parser.add_argument(
        "--sampled-centers",
        action="store_true",
        help="also evaluate iid-p sampled centers with optimal leaf assignments",
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)
    best = None
    if args.full_family:
        families = args.families.split(",")
        worst_sampled = None
        worst_additive = None
        worst_hard_additive = None
        for trial in range(args.trials):
            family = families[trial % len(families)]
            d = named_metric(family, args.clients, rng)
            costs = subset_tsp_costs(d)
            for p in probability_vectors(args.clients, rng, 1):
                posterior = expected_posterior(p, costs)
                value, centers, assignment = best_mst_backbone_spider(d, list(p))
                ratio = value / posterior if posterior else 0.0
                radius = nearest_sample_radius(d, list(p))
                additive_ratio = (value - radius) / posterior if posterior else 0.0
                sampled = (
                    sampled_center_average(d, list(p)) / posterior
                    if args.sampled_centers and posterior
                    else float("nan")
                )
                if best is None or ratio > best[0]:
                    best = (
                        ratio,
                        trial,
                        family,
                        posterior,
                        value,
                        p,
                        d,
                        centers,
                        assignment,
                        sampled,
                    )
                    print(
                        "best",
                        ratio,
                        "trial",
                        trial,
                        "family",
                        family,
                        "P/Psi",
                        (posterior, value),
                        "centers",
                        centers,
                        "sampled/P",
                        sampled,
                        flush=True,
                    )
                if args.sampled_centers and (
                    worst_sampled is None or sampled > worst_sampled[0]
                ):
                    worst_sampled = (sampled, trial, family, posterior, p, d)
                    print(
                        "worst sampled/P",
                        sampled,
                        "trial",
                        trial,
                        "family",
                        family,
                        flush=True,
                    )
                if worst_additive is None or additive_ratio > worst_additive[0]:
                    worst_additive = (
                        additive_ratio,
                        trial,
                        family,
                        posterior,
                        radius,
                        value,
                        p,
                        d,
                    )
                if radius >= 0.95 * posterior and (
                    worst_hard_additive is None
                    or additive_ratio > worst_hard_additive[0]
                ):
                    worst_hard_additive = (
                        additive_ratio,
                        trial,
                        family,
                        posterior,
                        radius,
                        value,
                        p,
                        d,
                    )
                    print(
                        "worst (Psi-R)/P",
                        additive_ratio,
                        "trial",
                        trial,
                        "family",
                        family,
                        flush=True,
                    )
        print("final", best)
        print("final additive", worst_additive)
        print("final hard additive", worst_hard_additive)
        if args.sampled_centers:
            print("final sampled", worst_sampled)
        return
    for trial in range(args.trials):
        d = random_metric(args.clients, rng)
        p = [10.0 ** rng.uniform(-3.0, -0.0005) for _ in range(args.clients)]
        result = evaluate(d, p)
        score = (
            sampled_center_average(d, p) / result[0]
            if args.sampled_centers and result[0]
            else result[-1]
        )
        if best is None or score > best[0]:
            best = (score, trial, result, p, d, mst_edges(d))
            print(
                "best",
                best[0],
                "trial",
                trial,
                "P/star/mst",
                result[:3],
                flush=True,
            )
    print("final", best)


if __name__ == "__main__":
    main()
