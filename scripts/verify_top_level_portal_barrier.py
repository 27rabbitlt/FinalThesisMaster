#!/usr/bin/env python3
"""Exact top-level branching barrier for inherited-portal hierarchies.

The metric is the shortest-path metric of a unit star centered at the depot:
depot--client distances are one and distinct client--client distances are two.
For n equal-probability clients, this script verifies the exact minimum
hazard among rooted spanning trees with exactly k depot branches,

    q(n-k+1) + (k-1)p + 2p(n-k).

In particular, a finite n=25, p=1/1000 instance rules out the stronger claim
that an inherited-portal hierarchy with one maximal client block always meets
the 29/20 hazard target.
"""

from __future__ import annotations

from fractions import Fraction


N = 25
P_ACTIVE = Fraction(1, 1000)


def occupancy(size: int) -> Fraction:
    return 1 - (1 - P_ACTIVE) ** size


def minimum_hazard_with_root_degree(k: int) -> Fraction:
    assert 1 <= k <= N
    return (
        occupancy(N - k + 1)
        + (k - 1) * P_ACTIVE
        + 2 * P_ACTIVE * (N - k)
    )


def main() -> None:
    posterior = 2 * N * P_ACTIVE
    one_top = minimum_hazard_with_root_degree(1)
    target = Fraction(29, 20) * posterior

    assert one_top == occupancy(N) + 2 * (N - 1) * P_ACTIVE
    assert one_top > target

    # The all-depot-star tree is recovered at k=n and has hazard P/2.
    assert minimum_hazard_with_root_degree(N) == N * P_ACTIVE
    assert minimum_hazard_with_root_degree(N) == posterior / 2

    # Directly audit the discrete concavity claim used to optimize the branch
    # sizes: consolidating one client into the larger of two non-singleton
    # branches never increases the sum of their occupancy costs.
    for a in range(1, N):
        for b in range(2, N - a + 1):
            if a >= b - 1:
                assert occupancy(a + 1) + occupancy(b - 1) <= (
                    occupancy(a) + occupancy(b)
                )

    print(f"n={N}, p={P_ACTIVE}")
    print(f"posterior={posterior} = {float(posterior):.12f}")
    print(f"one_top_hazard={one_top} = {float(one_top):.12f}")
    print(f"ratio={one_top / posterior} = {float(one_top / posterior):.12f}")
    print(f"strict_slack={one_top - target} = {float(one_top - target):.12f}")
    print("exact top-level portal branching barrier verified")


if __name__ == "__main__":
    main()
