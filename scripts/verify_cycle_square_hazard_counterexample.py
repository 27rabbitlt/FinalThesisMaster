#!/usr/bin/env python3
"""Exact certificate that the coefficient-one hazard conjecture is false.

The instance is the shortest-path metric of the square of a 30-cycle.  Vertex
0 is the always-active depot and every other vertex is active independently
with probability 4/5.  The certificate does *not* refute the 29/20 target;
indeed an explicit connected-dominating-set tree is safely below 29/20.

Only rational arithmetic is used.  The two mathematical ingredients checked
by the arithmetic are proved in the accompanying hazard_tree_barrier.md:

* every hazard solution in the metric closure can be uncrossed to a tree in
  the original graph; and
* every connected dominating set in C_N^2 has at least ceil((N-3)/2)
  vertices (for N=30 this is 14).
"""

from fractions import Fraction


Q = Fraction
N = 30
P_ACT = Q(4, 5)
INACTIVE = 1 - P_ACT


def metric_length_of_cyclic_gap(gap: int) -> int:
    """Distance in C_N^2 between vertices gap base-cycle edges apart."""
    base_distance = min(gap, N - gap)
    return (base_distance + 1) // 2


def expected_cyclic_order_cost() -> Q:
    """Expected shortcut cost of order 0,1,...,N-1 in the C_N^2 metric."""
    # Arcs having the depot as one endpoint.  There are two orientations for
    # every gap g.  The client endpoint is active and the g-1 interior clients
    # are inactive.
    depot_arcs = 2 * sum(
        P_ACT
        * INACTIVE ** (gap - 1)
        * metric_length_of_cyclic_gap(gap)
        for gap in range(1, N)
    )

    # Arcs whose two endpoints are clients and whose clockwise interior does
    # not contain the depot.  There are N-1-g choices for a gap of length g.
    client_arcs = sum(
        (N - 1 - gap)
        * P_ACT**2
        * INACTIVE ** (gap - 1)
        * metric_length_of_cyclic_gap(gap)
        for gap in range(1, N - 1)
    )
    return depot_arcs + client_arcs


def main() -> None:
    # If I is the nonleaf set of a spanning tree in C_30^2, I is connected
    # and dominating.  In a connected subset, at most one cyclic gap between
    # consecutive members can exceed 2; domination makes that exceptional
    # gap at most 5.  Thus 30 <= 2|I|+3 and |I| >= 14.
    min_internal = (N - 3 + 1) // 2  # ceil((N-3)/2)
    assert 2 * (min_internal - 1) + 3 < N
    assert 2 * min_internal + 3 >= N

    q_two = 1 - INACTIVE**2
    leaves_at_most = N - min_internal

    # A nonroot leaf edge is used with probability p.  Every edge entering a
    # nonroot internal vertex has at least two clients below it and is used
    # with probability at least q_2.  If the depot itself is a leaf, the same
    # expression only gets stronger by q_2-p.
    hazard_lower = leaves_at_most * P_ACT + (min_internal - 1) * q_two

    cyclic_upper = expected_cyclic_order_cost()
    assert hazard_lower > cyclic_upper

    # Posterior optimum <= the fixed cyclic order, so the preceding strict
    # inequality proves Phi* > OPT_post without having to enumerate 2^29
    # realizations.
    strict_gap = hazard_lower - cyclic_upper

    # Conversely, take the 14 even vertices except one as a unit-edge
    # connected dominating path, and attach the other 16 vertices as unit
    # leaves.  Bounding every one of its 13 backbone hazards by one gives the
    # following explicit upper bound on Phi*.
    hazard_tree_upper = leaves_at_most * P_ACT + (min_internal - 1)

    # Every nonempty k-client metric tour has at least k+1 unit-or-longer
    # edges (including the doubled depot edge when k=1).
    posterior_lower = (N - 1) * P_ACT + (1 - INACTIVE ** (N - 1))
    assert hazard_tree_upper < Q(29, 20) * posterior_lower

    print(f"cyclic posterior upper = {cyclic_upper} = {float(cyclic_upper):.12f}")
    print(f"hazard lower = {hazard_lower} = {float(hazard_lower):.12f}")
    print(f"strict Phi*-P gap >= {strict_gap} = {float(strict_gap):.12f}")
    print(
        "certified ratio Phi*/P >="
        f" {hazard_lower / cyclic_upper} ="
        f" {float(hazard_lower / cyclic_upper):.12f}"
    )
    print(f"explicit hazard-tree upper = {hazard_tree_upper}")
    print(f"posterior cardinality lower = {posterior_lower}")
    print("29/20 target remains safe on this witness")


if __name__ == "__main__":
    main()
