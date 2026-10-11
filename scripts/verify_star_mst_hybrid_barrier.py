#!/usr/bin/env python3
"""Exact counterexample to the depot-star versus full-MST hazard hybrid.

There are N clients in path order.  Consecutive clients have distance one,
all other distinct clients have distance two, and all depot distances are L.
The activation probabilities are common and positive.  Exact formulas verify
that both the depot star and the best-hazard full-metric MST exceed 29P/20,
whereas one client-centered spider is inexpensive.
"""

from __future__ import annotations

from fractions import Fraction


N = 50
P_ACTIVE = Fraction(2, 25)
DEPOT_DISTANCE = Fraction(10)


def occupancy(size: int) -> Fraction:
    return 1 - (1 - P_ACTIVE) ** size


def posterior_value() -> Fraction:
    # For a nonempty realization, tsp = 2L + K + C - 2, where C is the
    # number of occupied runs in the client path.  E[K]=Np and
    # E[C]=Np-(N-1)p^2.
    return (
        (2 * DEPOT_DISTANCE - 2) * occupancy(N)
        + 2 * N * P_ACTIVE
        - (N - 1) * P_ACTIVE**2
    )


def mst_hazard(root_client: int) -> Fraction:
    """Hazard of the path MST whose sole depot edge meets root_client."""
    assert 1 <= root_client <= N
    return (
        DEPOT_DISTANCE * occupancy(N)
        + sum(occupancy(k) for k in range(1, root_client))
        + sum(occupancy(k) for k in range(1, N - root_client + 1))
    )


def direct_formula_audit(test_n: int = 7) -> None:
    """Enumerate all small active sets and audit the posterior formula."""
    expected = Fraction(0)
    for mask in range(1 << test_n):
        probability = P_ACTIVE ** mask.bit_count()
        probability *= (1 - P_ACTIVE) ** (test_n - mask.bit_count())
        if not mask:
            continue
        active = [i for i in range(test_n) if mask & (1 << i)]
        runs = 1 + sum(active[j] != active[j - 1] + 1 for j in range(1, len(active)))
        cost = 2 * DEPOT_DISTANCE + len(active) + runs - 2
        expected += probability * cost
    closed_form = (
        (2 * DEPOT_DISTANCE - 2) * occupancy(test_n)
        + 2 * test_n * P_ACTIVE
        - (test_n - 1) * P_ACTIVE**2
    )
    assert expected == closed_form


def main() -> None:
    direct_formula_audit()

    posterior = posterior_value()
    depot_star = N * P_ACTIVE * DEPOT_DISTANCE
    hazards = [mst_hazard(j) for j in range(1, N + 1)]
    best_mst = min(hazards)
    best_roots = [j for j, value in enumerate(hazards, start=1) if value == best_mst]
    target = Fraction(29, 20) * posterior

    # The convex prefix sums are minimized by a central root attachment.
    assert best_roots == [N // 2, N // 2 + 1]
    assert depot_star > target
    assert best_mst > target

    # An interior client hub joined to the depot, with every other client a
    # leaf, has two length-one leaf edges and N-3 length-two leaf edges.
    hub_spider = (
        DEPOT_DISTANCE * occupancy(N)
        + P_ACTIVE * (2 + 2 * (N - 3))
    )
    assert hub_spider < target

    print(f"n={N}, p={P_ACTIVE}, L={DEPOT_DISTANCE}")
    print(f"posterior={posterior} = {float(posterior):.12f}")
    print(f"depot_star={depot_star} = {float(depot_star):.12f}")
    print(f"depot_star_ratio={float(depot_star / posterior):.12f}")
    print(f"best_mst={best_mst} = {float(best_mst):.12f}, roots={best_roots}")
    print(f"best_mst_ratio={float(best_mst / posterior):.12f}")
    print(f"hybrid_slack={min(depot_star, best_mst) - target} "
          f"= {float(min(depot_star, best_mst) - target):.12f}")
    print(f"one_hub_spider={hub_spider} = {float(hub_spider):.12f}")
    print(f"one_hub_ratio={float(hub_spider / posterior):.12f}")
    print("exact depot-star/full-MST hybrid barrier verified")


if __name__ == "__main__":
    main()
