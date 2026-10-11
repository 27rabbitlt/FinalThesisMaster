#!/usr/bin/env python3
"""Exact audits for the deterministic tour--facility core route.

This checker has two independent parts.

1.  On rooted weighted tree metrics whose tree nodes are precisely the depot
    and the admissible client/connector labels, it checks the heavy-subtree
    construction from ``facility_core_duality.md``.  All arithmetic is
    rational and posterior expectations are also enumerated directly.

2.  It checks a finite strict grouped metric on which J*/P > 5/2, refuting
    the tempting universal strengthening J* <= 2P.  The large witness is
    evaluated by proved closed formulas; a small instance is exhaustively
    enumerated to audit those formulas.

3.  It gives an exact six-client audit of the omitted LCA-to-guide segment
    in the sampled-guide argument (P41.2), including E[J(B)] > 2P.

The finite audits support the accompanying symbolic proofs; they do not turn
the tested grids into a universal proof.
"""

from __future__ import annotations

from fractions import Fraction as F
from math import prod
import random


def mask_probability(mask: int, probabilities: list[F]) -> F:
    value = F(1)
    for v, p_v in enumerate(probabilities):
        value *= p_v if mask & (1 << v) else 1 - p_v
    return value


def tree_data(parent: list[int], lengths: list[F], probabilities: list[F]):
    n = len(probabilities)
    root = n
    children = [[] for _ in range(n + 1)]
    for v, par in enumerate(parent):
        children[par].append(v)

    descendants = [0] * n

    def visit(v: int) -> int:
        mask = 0 if v == root else 1 << v
        for child in children[v]:
            mask |= visit(child)
        if v != root:
            descendants[v] = mask
        return mask

    visit(root)

    distance = [[F(0) for _ in range(n + 1)] for _ in range(n + 1)]
    ancestor_paths = []
    for start in range(n + 1):
        path = {start: F(0)}
        cur = start
        total = F(0)
        while cur != root:
            total += lengths[cur]
            cur = parent[cur]
            path[cur] = total
        ancestor_paths.append(path)
    for u in range(n + 1):
        for v in range(n + 1):
            common = set(ancestor_paths[u]).intersection(ancestor_paths[v])
            distance[u][v] = min(
                ancestor_paths[u][a] + ancestor_paths[v][a] for a in common
            )
    return descendants, distance


def audit_tree_instance(
    parent: list[int], lengths: list[F], probabilities: list[F]
) -> None:
    n = len(probabilities)
    root = n
    descendants, distance = tree_data(parent, lengths, probabilities)

    subtree_mass = [
        sum(probabilities[v] for v in range(n) if descendants[e] & (1 << v))
        for e in range(n)
    ]
    subtree_occupancy = [
        1
        - prod(
            (
                1 - probabilities[v]
                for v in range(n)
                if descendants[e] & (1 << v)
            ),
            start=F(1),
        )
        for e in range(n)
    ]

    posterior_edge_formula = 2 * sum(
        lengths[e] * subtree_occupancy[e] for e in range(n)
    )
    tours = []
    posterior_enumerated = F(0)
    for mask in range(1 << n):
        subtree_length = sum(
            lengths[e] for e in range(n) if descendants[e] & mask
        )
        tour = 2 * subtree_length
        tours.append(tour)
        posterior_enumerated += mask_probability(mask, probabilities) * tour
    assert posterior_edge_formula == posterior_enumerated

    heavy_mask = sum(
        1 << e for e in range(n) if subtree_mass[e] >= 1
    )
    ideal_half = sum(
        lengths[e] * min(F(1), subtree_mass[e]) for e in range(n)
    )

    objectives = []
    for mask in range(1 << n):
        centers = [root] + [v for v in range(n) if mask & (1 << v)]
        attachment = sum(
            probabilities[v] * min(distance[v][s] for s in centers)
            for v in range(n)
        )
        objectives.append(tours[mask] + 2 * attachment)
    constructed = objectives[heavy_mask]
    optimum = min(objectives)

    assert optimum <= constructed
    assert constructed <= 2 * ideal_half
    # Rational certificate for the analytic optimal factor
    # 1/(1-exp(-1)) = 1.58197... < 8/5.
    assert 5 * constructed <= 8 * posterior_edge_formula


def grouped_formula(g: int, m: int, delta: F):
    p = F(1, 2 * m)
    empty_group = (1 - p) ** m
    occupied_group = 1 - empty_group
    any_group = 1 - empty_group**g
    posterior = (
        g * occupied_group
        + any_group
        + delta * g * (F(1, 2) - occupied_group)
    )
    nearest_sample = F(g, 2) * (
        empty_group + delta * (1 - p - empty_group)
    )
    optimum = F(g)
    return posterior, nearest_sample, optimum


def held_karp_tours(distance: list[list[F]]) -> list[F]:
    n = len(distance) - 1
    root = n
    path: dict[tuple[int, int], F] = {}
    tours = [F(0)] * (1 << n)
    for mask in range(1, 1 << n):
        for last in range(n):
            if not mask & (1 << last):
                continue
            previous = mask ^ (1 << last)
            if previous == 0:
                path[mask, last] = distance[root][last]
            else:
                path[mask, last] = min(
                    path[previous, first] + distance[first][last]
                    for first in range(n)
                    if previous & (1 << first)
                )
        tours[mask] = min(
            path[mask, last] + distance[last][root]
            for last in range(n)
            if mask & (1 << last)
        )
    return tours


def audit_tsp_lattice_obstruction() -> None:
    # Four clients followed by the depot.  All nonzero distances are in
    # {1,2}, so the triangle inequality is immediate.
    distance = [
        [F(0), F(2), F(1), F(1), F(1)],
        [F(2), F(0), F(2), F(1), F(1)],
        [F(1), F(2), F(0), F(2), F(2)],
        [F(1), F(1), F(2), F(0), F(1)],
        [F(1), F(1), F(2), F(1), F(0)],
    ]
    tours = held_karp_tours(distance)

    # Submodularity fails: 5+4 < 6+4.
    left, right = 0b0111, 0b1011
    assert tours[left] + tours[right] < tours[left | right] + tours[left & right]

    # Supermodularity fails as well: 2+4 > 4+0.
    left, right = 0b0001, 0b0100
    assert tours[left] + tours[right] > tours[left | right] + tours[left & right]


def audit_small_grouped_formula() -> None:
    g, m, delta = 2, 3, F(1, 17)
    p = F(1, 2 * m)
    probabilities = [p] * (g * m)

    def tour(mask: int) -> F:
        active_by_group = [0] * g
        for v in range(g * m):
            if mask & (1 << v):
                active_by_group[v // m] += 1
        k = sum(active_by_group)
        h = sum(value > 0 for value in active_by_group)
        return F(0) if h == 0 else h + 1 + delta * (k - h)

    posterior = sum(
        mask_probability(mask, probabilities) * tour(mask)
        for mask in range(1 << (g * m))
    )
    formula_p, formula_r, formula_j = grouped_formula(g, m, delta)
    assert posterior == formula_p

    objectives = []
    for mask in range(1 << (g * m)):
        selected_by_group = [0] * g
        for v in range(g * m):
            if mask & (1 << v):
                selected_by_group[v // m] += 1
        s = sum(selected_by_group)
        k = sum(value > 0 for value in selected_by_group)
        backbone = F(0) if s == 0 else k + 1 + delta * (s - k)
        attachment = sum(
            (F(m - selected_by_group[i]) * p * delta)
            if selected_by_group[i]
            else F(m) * p
            for i in range(g)
        )
        objectives.append(backbone + 2 * attachment)
    assert min(objectives) == formula_j

    # Directly audit R by a second subset enumeration.
    radius = F(0)
    for guide in range(1 << (g * m)):
        guide_probability = mask_probability(guide, probabilities)
        assignment = F(0)
        for v in range(g * m):
            if guide & (1 << v):
                nearest = F(0)
            elif any(
                guide & (1 << u)
                for u in range((v // m) * m, (v // m + 1) * m)
            ):
                nearest = delta
            else:
                nearest = F(1)
            assignment += p * nearest
        radius += guide_probability * assignment
    assert radius == formula_r


def audit_sampled_guide_counterexample() -> None:
    """Refute the sampled-guide estimate in P41 by exact enumeration.

    The metric on the depot and six clients is unit equilateral.  It is
    represented by a star with an unlabelled center and edge lengths 1/2.
    """
    n = 6
    p = F(1, 6)
    probabilities = [p] * n

    def tour(mask: int) -> F:
        size = mask.bit_count()
        return F(0) if size == 0 else F(size + 1)

    posterior = sum(
        mask_probability(mask, probabilities) * tour(mask)
        for mask in range(1 << n)
    )
    radius = F(0)
    expected_sampled_objective = F(0)
    for guide in range(1 << n):
        guide_probability = mask_probability(guide, probabilities)
        # In the unit metric a client pays one exactly when it is not a guide.
        attachment = sum(p for v in range(n) if not guide & (1 << v))
        radius += guide_probability * attachment
        expected_sampled_objective += guide_probability * (
            tour(guide) + 2 * attachment
        )

    assert posterior == F(77687, 46656)
    assert radius == F(5, 6)
    assert expected_sampled_objective == posterior + 2 * radius
    assert expected_sampled_objective - 2 * posterior == F(73, 46656)

    # Pointwise audit for B={0}.  P41 pays only the upward leaf half-edge
    # for each of the other five demands.  The downward LCA-to-guide
    # half-edge is missing.
    singleton_attachment = F(5, 6)
    false_p41_edge_sum = F(5, 12)
    assert singleton_attachment == 2 * false_p41_edge_sum


def main() -> None:
    rng = random.Random(290319)
    probability_grid = [F(0), F(1, 20), F(1, 5), F(1, 2), F(4, 5), F(1)]
    tree_checks = 0
    for n in range(1, 8):
        root = n
        for _ in range(80):
            # Labels are generated from the root downward, so every parent is
            # either the root or a later-indexed client and cycles are impossible.
            parent = [rng.randrange(v + 1, n + 1) for v in range(n)]
            lengths = [F(rng.randrange(1, 13), rng.randrange(1, 8)) for _ in range(n)]
            probabilities = [rng.choice(probability_grid) for _ in range(n)]
            audit_tree_instance(parent, lengths, probabilities)
            tree_checks += 1

    audit_small_grouped_formula()
    audit_sampled_guide_counterexample()
    audit_tsp_lattice_obstruction()

    posterior, radius, optimum = grouped_formula(300, 50, F(1, 10**6))
    deficit = posterior - radius
    assert 2 * posterior < optimum
    assert F(5, 2) * posterior < optimum
    assert posterior + 2 * radius > 2 * posterior
    assert optimum < F(14, 5) * posterior + 2 * deficit

    print(f"verified {tree_checks} exact node-supported tree instances")
    print("verified symbolic grouped formulas by exhaustive 6-client enumeration")
    print("verified exact six-client failure of sampled-guide P41.2--P41.3")
    print("verified exact four-client failure of both TSP sub- and supermodularity")
    print("finite strict grouped witness:")
    print("  J*/P =", float(optimum / posterior))
    print("  D/P  =", float(deficit / posterior))
    print("  E[J(B)]/(2P) =", float((posterior + 2 * radius) / (2 * posterior)))
    print("  RF19 normalized slack =", float((F(14, 5) * posterior + 2 * deficit - optimum) / posterior))


if __name__ == "__main__":
    main()
