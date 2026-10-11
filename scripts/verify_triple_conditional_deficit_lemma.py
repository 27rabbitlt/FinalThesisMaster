#!/usr/bin/env python3
"""Exact audit of a conditional lemma for the pathwise triple construction.

The global factor-2.9 target follows by averaging the proposed conditional
inequality

    E_C max(0, J(B)-J(T0), J(B)-J(T*))
      + 2 (tsp(B)-sum_{b in B} rho_b)
      + 2 sum_{b in B} p_b rho_b >= tsp(B)/10,

where rho_b is the distance from b to its nearest other guide or the depot.
This script searches for exact small counterexamples.  It is an audit, not a
proof of the inequality.
"""

from __future__ import annotations

from fractions import Fraction

from verify_hybrid_root_projection_exact import PROBABILITIES, tsp_cost
from verify_threshold_leave_one_out import (
    N,
    ROOT,
    graph_metric,
    mask_probability,
    projected_mask,
)


def audit(distance: list[list[int]], probabilities: tuple[Fraction, ...]):
    weights = [mask_probability(mask, probabilities) for mask in range(1 << N)]
    costs = [tsp_cost(mask, distance) for mask in range(1 << N)]
    attachment = [
        sum(
            probabilities[v]
            * min(
                [distance[v][ROOT]]
                + [distance[v][u] for u in range(N) if mask & (1 << u)]
            )
            for v in range(N)
        )
        for mask in range(1 << N)
    ]
    objective = [Fraction(costs[mask]) + 2 * attachment[mask] for mask in range(1 << N)]

    worst = None
    for guide in range(1 << N):
        guide_vertices = [v for v in range(N) if guide & (1 << v)]
        nearest_sum = sum(
            min(
                [distance[v][ROOT]]
                + [distance[v][u] for u in guide_vertices if u != v]
            )
            for v in guide_vertices
        )
        high_probability_charge = sum(
            probabilities[v]
            * min(
                [distance[v][ROOT]]
                + [distance[v][u] for u in guide_vertices if u != v]
            )
            for v in guide_vertices
        )
        saving = Fraction()
        for second in range(1 << N):
            rooted = projected_mask(guide, second, distance, False)
            rootless = projected_mask(guide, second, distance, True)
            saving += weights[second] * max(
                Fraction(),
                objective[guide] - objective[rooted],
                objective[guide] - objective[rootless],
            )

        lhs = saving + 2 * (costs[guide] - nearest_sum) + 2 * high_probability_charge
        gap = lhs - Fraction(costs[guide], 10)
        if worst is None or gap < worst[0]:
            worst = (gap, guide, saving, nearest_sum, high_probability_charge)
        if gap < 0:
            return False, worst
    return True, worst


def main() -> None:
    connected = audits = 0
    global_worst = None
    for graph_mask in range(1 << 10):
        distance = graph_metric(graph_mask)
        if distance is None:
            continue
        connected += 1
        for probabilities in PROBABILITIES:
            passed, witness = audit(distance, probabilities)
            audits += 1
            context = (graph_mask, probabilities, distance, witness)
            if global_worst is None or witness[0] < global_worst[0]:
                global_worst = (witness[0], context)
            if not passed:
                print("COUNTEREXAMPLE", context)
                return
    print(f"verified {audits} conditional audits on {connected} graph metrics")
    print("smallest exact conditional gap", global_worst)


if __name__ == "__main__":
    main()
