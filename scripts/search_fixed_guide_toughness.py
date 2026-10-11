#!/usr/bin/env python3
"""Exact fixed-guide audits for the fair-color toughness reduction.

Fix the guide support B and an uncolored multiset of the remaining atoms.
Color every remaining atom independently and uniformly A or C.  With
T=f_B(C), this checks the proposed pivotal inequality

    E[# A-atoms missed by T but covered by B]
        <= 2 E[tau_t(supp(A)) - i_t(supp(A))].

It also checks the stronger cell upper bound obtained by charging a missed
atom to its own B-Voronoi cell: every atom in that cell must be A.  The
default exhausts all connected labelled graph metrics on a depot and three
sites, every guide support, and remaining multiplicities in {0,...,6}^3.
All comparisons use integer coloring counts.
"""

from __future__ import annotations

import argparse
import math
from itertools import product

from search_poisson_fair_color_threshold import (
    connected_graph_metrics,
    graph_tables,
    projected_mask,
)


def split_states(multiplicities: tuple[int, ...]):
    """Aggregate the equally likely A/C atom splits by their support masks."""
    sites = len(multiplicities)
    aggregate: dict[int, list] = {}
    for a_counts in product(*(range(m + 1) for m in multiplicities)):
        coefficient = math.prod(
            math.comb(m, a) for m, a in zip(multiplicities, a_counts)
        )
        a_mask = 0
        c_mask = 0
        for site, (m, a) in enumerate(zip(multiplicities, a_counts)):
            if a:
                a_mask |= 1 << site
            if a < m:
                c_mask |= 1 << site
        key = (a_mask, c_mask)
        if key not in aggregate:
            aggregate[key] = [0, [0] * sites]
        aggregate[key][0] += coefficient
        for site, count in enumerate(a_counts):
            aggregate[key][1][site] += coefficient * count
    return aggregate


def voronoi_images(guide: int, distance) -> tuple[int, ...]:
    sites = len(distance) - 1
    root = sites
    centers = [v for v in range(sites) if guide & (1 << v)] + [root]
    return tuple(
        min(centers, key=lambda center: (distance[v][center], center))
        for v in range(sites)
    )


def audit(guide, multiplicities, colorings, tables, distance, threshold):
    sites = len(multiplicities)
    projections, _, toughness, missed, isolated = tables
    images = voronoi_images(guide, distance)
    pivotal = cell_upper = residual = 0
    for (a_mask, c_mask), (coefficient, a_masses) in colorings.items():
        core = projections[guide][c_mask]
        missed_core = missed[a_mask][core]
        missed_guide = missed[a_mask][guide]
        pivotal += sum(
            a_masses[v] * (missed_core[v] - missed_guide[v])
            for v in range(sites)
        )

        empty_cell_centers = {
            center
            for center in range(sites)
            if guide & (1 << center)
            and all(
                not c_mask & (1 << v) or images[v] != center
                for v in range(sites)
            )
        }
        cell_upper += sum(
            a_masses[v]
            for v in range(sites)
            if images[v] in empty_cell_centers
            and distance[v][sites] > threshold
            and distance[v][images[v]] <= threshold
        )
        residual += coefficient * 2 * (toughness[a_mask] - isolated[a_mask])
    assert pivotal <= cell_upper
    return pivotal, cell_upper, residual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, default=3)
    parser.add_argument("--max-multiplicity", type=int, default=6)
    args = parser.parse_args()
    if args.sites > 4:
        raise SystemExit("exact graph exhaustion is intended for at most four sites")

    vectors = list(
        product(range(args.max_multiplicity + 1), repeat=args.sites)
    )
    coloring_cache = {vector: split_states(vector) for vector in vectors}
    worst_pivotal = None
    worst_cell = None
    audits = 0
    for graph_mask, distance in connected_graph_metrics(args.sites):
        threshold = 1.5
        tables = graph_tables(distance, threshold)
        for guide in range(1 << args.sites):
            for multiplicities in vectors:
                pivotal, cell_upper, residual = audit(
                    guide,
                    multiplicities,
                    coloring_cache[multiplicities],
                    tables,
                    distance,
                    threshold,
                )
                total = 2 ** sum(multiplicities)
                pivotal_gap = (pivotal - residual) / total
                cell_gap = (cell_upper - residual) / total
                witness = (
                    graph_mask,
                    guide,
                    multiplicities,
                    pivotal,
                    cell_upper,
                    residual,
                )
                if worst_pivotal is None or pivotal_gap > worst_pivotal[0]:
                    worst_pivotal = (pivotal_gap, witness, distance)
                if worst_cell is None or cell_gap > worst_cell[0]:
                    worst_cell = (cell_gap, witness, distance)
                if pivotal > residual:
                    raise AssertionError(("pivotal counterexample", worst_pivotal))
                audits += 1

    print(f"verified {audits} exact fixed-guide A/C-split audits")
    print("worst pivotal gap:", worst_pivotal)
    print("worst cell-upper gap:", worst_cell)


if __name__ == "__main__":
    main()
