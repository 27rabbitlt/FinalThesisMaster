#!/usr/bin/env python3
"""Exact rational lower bound for the hazard/posterior ratio on C_200^4.

The depot is vertex 0, every other vertex has activation probability 3/5,
and distances are shortest-path distances in the fourth power of a 200-cycle.
The proof reduces metric hazard trees to graph trees, then uses the fact that
the nonleaf vertices of a graph tree form a connected dominating set.

All comparisons are exact.  In particular, the script certifies

    min_T Phi(T) / OPT_post > 53/50.

This is a counterexample to coefficient one, but an explicit tree also shows
that this instance is safely below the target 29/20.
"""

from fractions import Fraction


Q = Fraction
N = 200
POWER = 4
P_ACT = Q(3, 5)
INACTIVE = 1 - P_ACT


def metric_length_of_cyclic_gap(gap: int) -> int:
    base_distance = min(gap, N - gap)
    return (base_distance + POWER - 1) // POWER


def expected_cyclic_order_cost() -> Q:
    depot_arcs = 2 * sum(
        P_ACT
        * INACTIVE ** (gap - 1)
        * metric_length_of_cyclic_gap(gap)
        for gap in range(1, N)
    )
    client_arcs = sum(
        (N - 1 - gap)
        * P_ACT**2
        * INACTIVE ** (gap - 1)
        * metric_length_of_cyclic_gap(gap)
        for gap in range(1, N - 1)
    )
    return depot_arcs + client_arcs


def main() -> None:
    # If I is connected in C_N^k, at most one cyclic gap between consecutive
    # vertices of I exceeds k.  If I also dominates, that exceptional gap is
    # at most 2k+1.  Hence N <= k|I|+k+1.
    min_internal = (N - POWER - 1 + POWER - 1) // POWER
    assert POWER * (min_internal - 1) + POWER + 1 < N
    assert POWER * min_internal + POWER + 1 >= N

    q_two = 1 - INACTIVE**2
    leaves_at_most = N - min_internal
    hazard_lower = (
        leaves_at_most * P_ACT + (min_internal - 1) * q_two
    )

    cyclic_upper = expected_cyclic_order_cost()
    # OPT_post is no larger than the expected cost of the fixed cyclic order.
    assert Q(50) * hazard_lower > Q(53) * cyclic_upper

    # Multiples of k, with one nonroot multiple omitted, form a unit-edge
    # connected dominating path of min_internal vertices.  Attach every
    # remaining vertex as a unit leaf.
    explicit_hazard_upper = (min_internal - 1) + leaves_at_most * P_ACT
    posterior_lower = (N - 1) * P_ACT + 1 - INACTIVE ** (N - 1)
    assert Q(20) * explicit_hazard_upper < Q(29) * posterior_lower

    # Along C_N^k as N tends to infinity, the same certificate approaches
    # (1+z/k)(1-z^k), z=1-p.  Here that is exactly 6699/6250.
    limiting_certificate = (1 + INACTIVE / POWER) * (
        1 - INACTIVE**POWER
    )
    assert limiting_certificate == Q(6699, 6250)

    print(f"N={N}, k={POWER}, p={P_ACT}")
    print(f"hazard lower = {float(hazard_lower):.12f}")
    print(f"cyclic posterior upper = {float(cyclic_upper):.12f}")
    print(f"certified ratio = {float(hazard_lower / cyclic_upper):.12f} > 53/50")
    print(f"asymptotic certificate = {limiting_certificate} = {float(limiting_certificate):.12f}")
    print(f"explicit-tree/posterior-lower ratio = {float(explicit_hazard_upper / posterior_lower):.12f}")
    print("all comparisons used Fraction arithmetic")


if __name__ == "__main__":
    main()
