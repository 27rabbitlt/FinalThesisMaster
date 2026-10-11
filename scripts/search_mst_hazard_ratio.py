#!/usr/bin/env python3
"""Stress-test whether some full-metric MST always has hazard at most P.

This is a conjecture-falsification utility, not a proof.  It enumerates every
labelled spanning tree through Prufer codes, keeps all minimum-cost trees, and
compares the least hazard among them with the exact expected posterior TSP.
"""

from __future__ import annotations

import argparse
import itertools
import random

from search_hazard_tree_ratio import (
    expected_posterior,
    named_metric,
    probability_vectors,
    subset_tsp_costs,
)


def prufer_edges(code: tuple[int, ...], order: int) -> list[tuple[int, int]]:
    degree = [1] * order
    for vertex in code:
        degree[vertex] += 1
    edges: list[tuple[int, int]] = []
    for vertex in code:
        leaf = next(i for i, value in enumerate(degree) if value == 1)
        edges.append((leaf, vertex))
        degree[leaf] -= 1
        degree[vertex] -= 1
    last = [i for i, value in enumerate(degree) if value == 1]
    edges.append((last[0], last[1]))
    return edges


def rooted_masks(edges: list[tuple[int, int]], root: int) -> tuple[int, ...]:
    adjacency = [[] for _ in range(root + 1)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    parent = [-1] * (root + 1)
    order = [root]
    for u in order:
        for v in adjacency[u]:
            if v != parent[u]:
                parent[v] = u
                order.append(v)
    masks = [0] * (root + 1)
    for v in reversed(order[1:]):
        masks[v] |= 1 << v
        masks[parent[v]] |= masks[v]
    return tuple(masks[v] for v in range(root))


def all_mst_profiles(d: list[list[float]]) -> list[tuple[tuple[int, ...], tuple[float, ...]]]:
    order = len(d)
    root = order - 1
    best = float("inf")
    profiles: list[tuple[tuple[int, ...], tuple[float, ...]]] = []
    for code in itertools.product(range(order), repeat=order - 2):
        edges = prufer_edges(code, order)
        cost = sum(d[u][v] for u, v in edges)
        if cost > best + 1e-12:
            continue
        masks = rooted_masks(edges, root)
        # Recover each parent-edge length as the unique tree edge crossing the
        # corresponding descendant cut.
        lengths = tuple(
            next(
                d[u][v]
                for u, v in edges
                if bool(masks[client] & (1 << u)) != bool(masks[client] & (1 << v))
            )
            for client in range(root)
        )
        if cost < best - 1e-12:
            best = cost
            profiles = []
        profiles.append((masks, lengths))
    return profiles


def profile_hazard(
    profile: tuple[tuple[int, ...], tuple[float, ...]], p: tuple[float, ...]
) -> float:
    masks, lengths = profile
    value = 0.0
    for mask, length in zip(masks, lengths):
        inactive = 1.0
        for v, pv in enumerate(p):
            if mask & (1 << v):
                inactive *= 1.0 - pv
        value += length * (1.0 - inactive)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--metrics", type=int, default=100)
    parser.add_argument("--probabilities", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--families",
        default="unit,path,cycle,star,binary,random-tree,graph12,graph123,client-graph-19,euclidean,multiscale-graph,multiscale-complete",
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)
    best = (0.0, "", (), 0.0, 0.0)
    best_hybrid = (0.0, "", (), 0.0, 0.0, 0.0)
    families = args.families.split(",")
    for index in range(args.metrics):
        family = families[index % len(families)]
        d = named_metric(family, args.clients, rng)
        posterior_costs = subset_tsp_costs(d)
        profiles = all_mst_profiles(d)
        for p in probability_vectors(args.clients, rng, args.probabilities):
            posterior = expected_posterior(p, posterior_costs)
            hazard = min(profile_hazard(profile, p) for profile in profiles)
            star = sum(p[v] * d[v][args.clients] for v in range(args.clients))
            hybrid = min(hazard, star)
            ratio = hazard / posterior if posterior else 0.0
            hybrid_ratio = hybrid / posterior if posterior else 0.0
            if ratio > best[0]:
                best = (ratio, family, tuple(p), posterior, hazard)
                print(
                    f"best ratio={ratio:.12f} family={family} "
                    f"P={posterior:.12f} mst_hazard={hazard:.12f} "
                    f"mst_count={len(profiles)}",
                    flush=True,
                )
            if hybrid_ratio > best_hybrid[0]:
                best_hybrid = (
                    hybrid_ratio,
                    family,
                    tuple(p),
                    posterior,
                    hazard,
                    star,
                )
                print(
                    f"best hybrid ratio={hybrid_ratio:.12f} family={family} "
                    f"P={posterior:.12f} mst_hazard={hazard:.12f} "
                    f"star_hazard={star:.12f}",
                    flush=True,
                )
    print("final MST =", best)
    print("final hybrid =", best_hybrid)


if __name__ == "__main__":
    main()
