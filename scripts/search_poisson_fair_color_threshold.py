#!/usr/bin/env python3
"""Exact repeated-site fair-color checks for the Poissonized threshold lemma.

Fix an uncolored multiset of atoms on metric sites and color every atom
independently and uniformly A, B, or C.  With T=f_B(C), this checks

    E[# A-atoms missed by T + # A-atoms missed by B]
        <= 2 E[tau_t(supp(A))] <= 2 E[k_t(supp(A))].

The default run exhausts every connected labelled graph metric on a depot
and three sites, at its edge/nonedge threshold, and every site-multiplicity
vector in {0,...,6}^3.  All comparisons use integer coloring counts.
"""

from __future__ import annotations

import argparse
import math
from itertools import combinations, permutations, product


def connected_graph_metrics(sites: int):
    size = sites + 1
    root = sites
    pairs = tuple(combinations(range(size), 2))
    infinity = size + 1
    for graph_mask in range(1 << len(pairs)):
        distance = [
            [0 if u == v else infinity for v in range(size)]
            for u in range(size)
        ]
        for index, (u, v) in enumerate(pairs):
            if graph_mask & (1 << index):
                distance[u][v] = distance[v][u] = 1
        for middle in range(size):
            for u in range(size):
                for v in range(size):
                    distance[u][v] = min(
                        distance[u][v],
                        distance[u][middle] + distance[middle][v],
                    )
        if all(distance[root][v] <= sites for v in range(sites)):
            yield graph_mask, distance


def projected_mask(
    guide: int,
    sample: int,
    distance,
    client_only: bool = False,
) -> int:
    sites = len(distance) - 1
    root = sites
    guide_centers = [v for v in range(sites) if guide & (1 << v)]
    if client_only and not guide_centers:
        return 0
    centers = guide_centers if client_only else guide_centers + [root]
    retained = 0
    for vertex in range(sites):
        if sample & (1 << vertex):
            image = min(centers, key=lambda center: (distance[vertex][center], center))
            if image != root:
                retained |= 1 << image
    return retained


def uncovered_counts(source: int, target: int, distance, threshold: float):
    sites = len(distance) - 1
    root = sites
    return tuple(
        int(
            distance[v][root] > threshold
            and all(
                not target & (1 << u) or distance[v][u] > threshold
                for u in range(sites)
            )
        )
        if source & (1 << v)
        else 0
        for v in range(sites)
    )


def bad_tour_edges(mask: int, distance, threshold: float) -> int:
    sites = len(distance) - 1
    root = sites
    vertices = [v for v in range(sites) if mask & (1 << v)]
    if not vertices:
        return 0
    return min(
        sum(
            distance[u][v] > threshold
            for u, v in zip((root,) + order, order + (root,))
        )
        for order in permutations(vertices)
    )


def component_count(vertices: set[int], distance, threshold: float) -> int:
    remaining = set(vertices)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            u = stack.pop()
            reached = {v for v in remaining if distance[u][v] <= threshold}
            remaining.difference_update(reached)
            stack.extend(reached)
    return components


def toughness_defect(mask: int, distance, threshold: float) -> int:
    sites = len(distance) - 1
    root = sites
    vertices = [root] + [v for v in range(sites) if mask & (1 << v)]
    best = 0
    for removed_bits in range(1 << len(vertices)):
        removed = {
            vertex
            for index, vertex in enumerate(vertices)
            if removed_bits & (1 << index)
        }
        components = component_count(set(vertices) - removed, distance, threshold)
        if not removed:
            candidate = components if components >= 2 else 0
        else:
            candidate = max(components - len(removed), 0)
        best = max(best, candidate)
    return best


def local_color_states(multiplicity: int):
    for a_count in range(multiplicity + 1):
        for b_count in range(multiplicity - a_count + 1):
            c_count = multiplicity - a_count - b_count
            coefficient = (
                math.factorial(multiplicity)
                // math.factorial(a_count)
                // math.factorial(b_count)
                // math.factorial(c_count)
            )
            yield a_count, b_count, c_count, coefficient


def aggregated_colorings(multiplicities: tuple[int, ...]):
    """Aggregate fair atom colorings by support masks and A-site counts."""
    sites = len(multiplicities)
    aggregate: dict[tuple[int, int, int], list] = {}
    for site_states in product(*(local_color_states(m) for m in multiplicities)):
        a_mask = b_mask = c_mask = 0
        a_counts = [0] * sites
        coefficient = 1
        for site, (a_count, b_count, c_count, local_coefficient) in enumerate(site_states):
            if a_count:
                a_mask |= 1 << site
            if b_count:
                b_mask |= 1 << site
            if c_count:
                c_mask |= 1 << site
            a_counts[site] = a_count
            coefficient *= local_coefficient
        key = (a_mask, b_mask, c_mask)
        if key not in aggregate:
            aggregate[key] = [0, [0] * sites]
        aggregate[key][0] += coefficient
        for site, count in enumerate(a_counts):
            aggregate[key][1][site] += coefficient * count
    return aggregate


def graph_tables(distance, threshold: float, client_only_projection: bool = False):
    sites = len(distance) - 1
    projections = [
        [
            projected_mask(b, c, distance, client_only_projection)
            for c in range(1 << sites)
        ]
        for b in range(1 << sites)
    ]
    bad_edges = [
        bad_tour_edges(mask, distance, threshold) for mask in range(1 << sites)
    ]
    toughness = [
        toughness_defect(mask, distance, threshold) for mask in range(1 << sites)
    ]
    missed = [
        [uncovered_counts(source, target, distance, threshold) for target in range(1 << sites)]
        for source in range(1 << sites)
    ]
    isolated = [
        sum(
            bool(mask & (1 << v))
            and distance[v][sites] > threshold
            and all(
                not mask & (1 << u) or distance[v][u] > threshold
                for u in range(sites)
                if u != v
            )
            for v in range(sites)
        )
        for mask in range(1 << sites)
    ]
    return projections, bad_edges, toughness, missed, isolated


def audit(multiplicities, colorings, tables):
    sites = len(multiplicities)
    projections, bad_edges, toughness, missed, isolated = tables
    left_core = left_guide = right_toughness = right_tour = right_isolated = 0
    for (a_mask, b_mask, c_mask), (coefficient, a_masses) in colorings.items():
        core = projections[b_mask][c_mask]
        missed_core = missed[a_mask][core]
        missed_guide = missed[a_mask][b_mask]
        left_core += sum(a_masses[v] * missed_core[v] for v in range(sites))
        left_guide += sum(a_masses[v] * missed_guide[v] for v in range(sites))
        right_toughness += coefficient * 2 * toughness[a_mask]
        right_tour += coefficient * 2 * bad_edges[a_mask]
        right_isolated += coefficient * isolated[a_mask]
    assert right_toughness <= right_tour
    return (
        left_core,
        left_guide,
        right_toughness,
        right_tour,
        right_isolated,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, default=3)
    parser.add_argument("--max-multiplicity", type=int, default=6)
    parser.add_argument("--client-only-projection", action="store_true")
    args = parser.parse_args()
    if args.sites > 4:
        raise SystemExit("exact graph exhaustion is intended for at most four sites")

    multiplicity_vectors = [
        vector
        for vector in product(range(args.max_multiplicity + 1), repeat=args.sites)
        if any(vector)
    ]
    coloring_cache = {
        vector: aggregated_colorings(vector) for vector in multiplicity_vectors
    }
    audits = 0
    worst_toughness = None
    worst_tour = None
    largest_toughness_ratio = None
    largest_tour_ratio = None
    worst_guide_vs_isolated = None
    worst_core_vs_residual = None
    worst_rootless_core_vs_tour = None
    for graph_mask, distance in connected_graph_metrics(args.sites):
        tables = graph_tables(
            distance,
            1.5,
            client_only_projection=args.client_only_projection,
        )
        for multiplicities in multiplicity_vectors:
            left_core, left_guide, right_toughness, right_tour, right_isolated = audit(
                multiplicities, coloring_cache[multiplicities], tables
            )
            left = left_core + left_guide
            total_colorings = 3 ** sum(multiplicities)
            toughness_gap = (left - right_toughness) / total_colorings
            tour_gap = (left - right_tour) / total_colorings
            witness = (graph_mask, multiplicities, left, right_toughness, right_tour)
            if worst_toughness is None or toughness_gap > worst_toughness[0]:
                worst_toughness = (toughness_gap, witness, distance)
            if worst_tour is None or tour_gap > worst_tour[0]:
                worst_tour = (tour_gap, witness, distance)
            if right_toughness > 0:
                ratio = left / right_toughness
                if largest_toughness_ratio is None or ratio > largest_toughness_ratio[0]:
                    largest_toughness_ratio = (ratio, witness, distance)
            if right_tour > 0:
                ratio = left / right_tour
                if largest_tour_ratio is None or ratio > largest_tour_ratio[0]:
                    largest_tour_ratio = (ratio, witness, distance)
            if left > right_toughness:
                raise AssertionError(("toughness counterexample", worst_toughness))
            guide_gap = (left_guide - right_isolated) / total_colorings
            core_gap = (
                left_core + right_isolated - right_toughness
            ) / total_colorings
            guide_witness = (
                graph_mask,
                multiplicities,
                left_guide,
                right_isolated,
            )
            core_witness = (
                graph_mask,
                multiplicities,
                left_core,
                right_toughness - right_isolated,
            )
            if (
                worst_guide_vs_isolated is None
                or guide_gap > worst_guide_vs_isolated[0]
            ):
                worst_guide_vs_isolated = (guide_gap, guide_witness, distance)
            if worst_core_vs_residual is None or core_gap > worst_core_vs_residual[0]:
                worst_core_vs_residual = (core_gap, core_witness, distance)
            rootless_core_gap = (2 * left_core - right_tour) / total_colorings
            rootless_core_witness = (
                graph_mask,
                multiplicities,
                left_core,
                right_tour // 2,
            )
            if (
                worst_rootless_core_vs_tour is None
                or rootless_core_gap > worst_rootless_core_vs_tour[0]
            ):
                worst_rootless_core_vs_tour = (
                    rootless_core_gap,
                    rootless_core_witness,
                    distance,
                )
            if args.client_only_projection and 2 * left_core > right_tour:
                # This assertion describes only the enumerated four-site
                # range; many-component sparse families refute its universal
                # extrapolation.
                raise AssertionError(
                    ("rootless core versus tour counterexample", worst_rootless_core_vs_tour)
                )
            if left_guide > right_isolated:
                raise AssertionError(("guide versus isolated counterexample", worst_guide_vs_isolated))
            audits += 1

    mode = "client-only" if args.client_only_projection else "root-including"
    print(f"verified {audits} exact repeated-site fair-color audits ({mode})")
    print("worst toughness gap:", worst_toughness)
    print("worst tour gap:", worst_tour)
    print("largest toughness ratio:", largest_toughness_ratio)
    print("largest tour ratio:", largest_tour_ratio)
    print("worst guide-minus-isolated gap:", worst_guide_vs_isolated)
    print("worst core-minus-residual gap:", worst_core_vs_residual)
    print("worst rootless core-minus-tour gap:", worst_rootless_core_vs_tour)


if __name__ == "__main__":
    main()
