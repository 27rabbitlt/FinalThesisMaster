#!/usr/bin/env python3
"""Exact barrier for the two-posterior union/Euler certificate.

There are N clients and one depot in the unit metric.  Every client is
independently active with probability p.  For independent guide samples
B,C, put U=B union C.  This script checks, using Fraction arithmetic, that
even the reoptimized union-core certificate

    E[tsp(U)] + 2 sum_v p E[d(v,U union {r})]

and the ordinary one-guide certificate are both strictly larger than
(29/10) OPT_post.  Reoptimizing U is at least as strong as shortcutting any
Euler traversal of the union of optimal tours of B and C.

The script also checks the exact algebraic savings identity used in the
accompanying note.  The large barrier uses closed forms; a separate
five-client test exhausts subsets to audit the stopped-product identity.
No floating-point value is used in an assertion.
"""

from fractions import Fraction


Q = Fraction
N = 2000
P_ACTIVE = Q(1, 50)
Z = 1 - P_ACTIVE


def stopped_product_identity_check() -> None:
    """Enumerate one nontrivial order and verify (TT12)--(TT15)."""
    probabilities = (Q(1, 5), Q(1, 3), Q(2, 5), Q(1, 2), Q(3, 4))
    order = (2, 0, 4, 1, 3)
    n = len(order)
    root = n
    coordinates = (1, 3, 7, 8, 13, 0)
    distance = [
        [Q(abs(coordinates[i] - coordinates[j])) for j in range(n + 1)]
        for i in range(n + 1)
    ]

    def subset_probability(mask: int) -> Q:
        value = Q(1)
        for v, pv in enumerate(probabilities):
            value *= pv if mask & (1 << v) else 1 - pv
        return value

    enumerated_cost = Q(0)
    enumerated_edges: dict[tuple[int, int], Q] = {}
    enumerated_blocks = [Q(0) for _ in range(1 << n)]
    for mask in range(1 << n):
        weight = subset_probability(mask)
        active = [v for v in order if mask & (1 << v)]
        walk = [root, *active, root] if active else []
        for u, v in zip(walk, walk[1:]):
            enumerated_cost += weight * distance[u][v]
            edge = tuple(sorted((u, v)))
            enumerated_edges[edge] = enumerated_edges.get(edge, Q(0)) + weight
        for selected in range(1 << n):
            blocks = 0
            previous_inside = False
            for v in active:
                inside = bool(selected & (1 << v))
                if inside and not previous_inside:
                    blocks += 1
                previous_inside = inside
            enumerated_blocks[selected] += weight * blocks

    stopped_edges: dict[tuple[int, int], Q] = {}
    for i, u in enumerate(order):
        left_empty = Q(1)
        for k in range(i):
            left_empty *= 1 - probabilities[order[k]]
        right_empty = Q(1)
        for k in range(i + 1, n):
            right_empty *= 1 - probabilities[order[k]]
        stopped_edges[tuple(sorted((root, u)))] = (
            probabilities[u] * (left_empty + right_empty)
        )
        for j in range(i + 1, n):
            v = order[j]
            middle_empty = Q(1)
            for k in range(i + 1, j):
                middle_empty *= 1 - probabilities[order[k]]
            stopped_edges[tuple(sorted((u, v)))] = (
                probabilities[u] * probabilities[v] * middle_empty
            )

    assert stopped_edges == enumerated_edges
    formula_cost = sum(
        (multiplicity * distance[u][v] for (u, v), multiplicity in stopped_edges.items()),
        Q(0),
    )
    assert formula_cost == enumerated_cost

    rooted_order = (root, *order, root)
    full_length = sum(
        (distance[rooted_order[k]][rooted_order[k + 1]] for k in range(n + 1)),
        Q(0),
    )
    stopped_pair_cost = Q(0)
    expected_slack = Q(0)
    for i in range(n + 1):
        for j in range(i + 1, n + 2):
            endpoint_probability = Q(1)
            if i:
                endpoint_probability *= probabilities[rooted_order[i]]
            if j < n + 1:
                endpoint_probability *= probabilities[rooted_order[j]]
            for k in range(i + 1, j):
                endpoint_probability *= 1 - probabilities[rooted_order[k]]
            segment_length = sum(
                (
                    distance[rooted_order[k]][rooted_order[k + 1]]
                    for k in range(i, j)
                ),
                Q(0),
            )
            chord = distance[rooted_order[i]][rooted_order[j]]
            stopped_pair_cost += endpoint_probability * chord
            expected_slack += endpoint_probability * (segment_length - chord)
    assert stopped_pair_cost == formula_cost
    assert formula_cost == full_length - expected_slack

    for v, pv in enumerate(probabilities):
        degree = sum(
            multiplicity
            for edge, multiplicity in stopped_edges.items()
            if v in edge
        )
        assert degree == 2 * pv
    for selected in range(1 << n):
        cut = sum(
            multiplicity
            for (u, v), multiplicity in stopped_edges.items()
            if (u < n and bool(selected & (1 << u)))
            != (v < n and bool(selected & (1 << v)))
        )
        assert cut == 2 * enumerated_blocks[selected]


def main() -> None:
    stopped_product_identity_check()

    # In the unit metric, tsp(A)=|A|+1_{A nonempty}.
    nonempty_one = 1 - Z**N
    posterior = N * P_ACTIVE + nonempty_one

    # One-guide nearest-core radius and its usual sampled-tour certificate.
    radius_one = N * P_ACTIVE * Z
    deficit = posterior - radius_one
    baseline = posterior + 2 * radius_one

    # U=B union C has independent marginal probability 1-z^2.
    nonempty_union = 1 - Z ** (2 * N)
    posterior_union = N * (1 - Z**2) + nonempty_union
    radius_union = N * P_ACTIVE * Z**2
    union_certificate = posterior_union + 2 * radius_union

    # The literal union of the two sampled tours has expected length 2P;
    # reoptimizing on U can only improve that backbone.
    raw_euler_certificate = 2 * posterior + 2 * radius_union
    assert posterior_union <= 2 * posterior
    assert union_certificate <= raw_euler_certificate

    # Exact decomposition relative to 3P.
    radius_drop = radius_one - radius_union
    tour_growth = posterior_union - posterior
    union_saving = 2 * deficit + 2 * radius_drop - tour_growth
    assert 3 * posterior - union_certificate == union_saving
    combined_saving = 2 * deficit + max(Q(0), 2 * radius_drop - tour_growth)
    assert 3 * posterior - min(baseline, union_certificate) == combined_saving

    # On this instance the union growth overwhelms its radius drop, and the
    # baseline deficit is below the P/20 threshold.  Hence neither route
    # proves a factor of 2.9.
    assert 2 * radius_drop < tour_growth
    assert 20 * deficit < posterior
    assert 10 * baseline > 29 * posterior
    assert 10 * union_certificate > 29 * posterior
    assert 10 * min(baseline, union_certificate) > 29 * posterior

    # Every deterministic client order has actual expected shortcut cost P
    # in the unit metric.  This records that the obstruction is specifically
    # to the core-plus-doubled-excursions certificate, not to direct analysis
    # of the master order generated from the two tours.
    expected_cost_of_every_order = posterior
    assert expected_cost_of_every_order < Q(29, 10) * posterior

    overlap_reoptimization = N * P_ACTIVE**2 + nonempty_one**2
    assert 2 * posterior - posterior_union == overlap_reoptimization

    print(f"N={N}, p={P_ACTIVE}")
    print(f"P={float(posterior):.12f}, D/P={float(deficit/posterior):.12f}")
    print(f"baseline/P={float(baseline/posterior):.12f}")
    print(f"reoptimized-union/P={float(union_certificate/posterior):.12f}")
    print(f"raw-Euler/P={float(raw_euler_certificate/posterior):.12f}")
    print(f"tour growth / P={float(tour_growth/posterior):.12f}")
    print(f"twice radius drop / P={float(2*radius_drop/posterior):.12f}")
    print(f"overlap reoptimization / P={float(overlap_reoptimization/posterior):.12f}")
    print("stopped-product and barrier identities passed in Fraction arithmetic")


if __name__ == "__main__":
    main()
