#!/usr/bin/env python3
"""Exact checks for the hard-regime full-support-tour counterexample.

The large certificate is verified through closed rational formulas; no
enumeration of its 825,000 metric vertices is needed.  A small instance is
built explicitly to check the unit-graph forcing and the exact formula for
same-cluster consecutive transitions.
"""

from __future__ import annotations

from collections import deque
from fractions import Fraction
from itertools import combinations


Q = Fraction


def build_unit_graph(g: int, h: int):
    n = g * h
    total = 3 * n
    adjacency = [set() for _ in range(total)]

    def add_edge(u: int, v: int) -> None:
        adjacency[u].add(v)
        adjacency[v].add(u)

    # Anchor b_i is vertex i.  Cluster membership is i modulo g.
    for c in range(g):
        cluster = [c + t * g for t in range(h)]
        for u, v in combinations(cluster, 2):
            add_edge(u, v)

    # Private vertices x_i=n+2i and y_i=n+2i+1.
    full_order = []
    for i in range(n):
        x = n + 2 * i
        y = x + 1
        add_edge(i, x)
        add_edge(x, y)
        add_edge(y, (i + 1) % n)
        full_order.extend((i, x, y))
    return adjacency, tuple(full_order)


def unweighted_distances(adjacency):
    distances = []
    for source in range(len(adjacency)):
        distance = [None] * len(adjacency)
        distance[source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if distance[v] is None:
                    distance[v] = distance[u] + 1
                    queue.append(v)
        assert all(value is not None for value in distance)
        distances.append(distance)
    return distances


def cycle_cost(order, distance) -> int:
    if len(order) <= 1:
        return 0
    return sum(
        distance[order[i]][order[(i + 1) % len(order)]]
        for i in range(len(order))
    )


def same_cluster_transitions(active, g: int) -> int:
    if len(active) <= 1:
        return 0
    return sum(
        active[i] % g == active[(i + 1) % len(active)] % g
        for i in range(len(active))
    )


def subset_probability(mask: int, count: int, q: Q) -> Q:
    selected = mask.bit_count()
    return q**selected * (1 - q) ** (count - selected)


def exact_same_transition_formula(g: int, h: int, q: Q) -> Q:
    n = g * h
    a = 1 - q
    return sum(
        (2 * q + (n - k * g - 1) * q * q) * a ** (k * g - 1)
        for k in range(1, h)
    )


def brute_small_probability_checks() -> None:
    g, h = 3, 2
    n = g * h
    q = Q(2, 5)
    adjacency, full_order = build_unit_graph(g, h)
    distance = unweighted_distances(adjacency)

    assert len(full_order) == 3 * n
    assert len(set(full_order)) == 3 * n
    assert cycle_cost(full_order, distance) == 3 * n

    # Every private subdivision vertex has degree two.  The two forced path
    # incidences at every anchor consume both Hamilton-cycle degrees.
    assert all(len(adjacency[v]) == 2 for v in range(n, 3 * n))
    forced_at_anchor = [set() for _ in range(n)]
    for v in range(n, 3 * n):
        for u in adjacency[v]:
            if u < n:
                forced_at_anchor[u].add(v)
    assert all(len(neighbors) == 2 for neighbors in forced_at_anchor)

    for i in range(n):
        for j in range(i + 1, n):
            if i % g == j % g:
                assert distance[i][j] == 1
            else:
                assert distance[i][j] >= 3

    expected_same = Q(0)
    expected_cost = Q(0)
    for mask in range(1 << (n - 1)):
        probability = subset_probability(mask, n - 1, q)
        active = [0] + [i + 1 for i in range(n - 1) if mask & (1 << i)]
        same = same_cluster_transitions(active, g)
        expected_same += probability * same
        expected_cost += probability * cycle_cost(active, distance)
        if len(active) >= 2:
            assert cycle_cost(active, distance) >= 3 * len(active) - 2 * same

    formula = exact_same_transition_formula(g, h, q)
    assert expected_same == formula
    expected_nontrivial_size = 1 + (n - 1) * q - (1 - q) ** (n - 1)
    assert expected_cost >= 3 * expected_nontrivial_size - 2 * formula


def certificate(g: int, h: int, q: Q, epsilon: Q):
    n = g * h
    a = 1 - q
    posterior_upper = (n - 1) * q + 4 * g + 1 + 8 * n * epsilon
    radius_lower = (n - 1) * q * a
    same_upper = (2 * q + n * q * q) * a ** (g - 1) / (1 - a**g)
    stopped_lower = 3 * (n - 1) * q - 2 * same_upper
    return n, posterior_upper, radius_lower, same_upper, stopped_lower


def finite_certificate_checks() -> None:
    g = 25
    h = 11_000
    q = Q(1, 25)
    epsilon = Q(1, 1_000_000)
    n, upper, radius, same, stopped = certificate(g, h, q, epsilon)

    assert n == 275_000
    assert upper == Q(277_579, 25)
    assert radius == Q(6_599_976, 625)
    assert 20 * radius - 19 * upper == Q(29_899, 125) > 0
    assert same < 259

    simpler_stopped_lower = Q(812_047, 25)
    assert stopped > simpler_stopped_lower
    assert simpler_stopped_lower / upper == Q(812_047, 277_579)
    assert 10 * simpler_stopped_lower > 29 * upper

    print("finite hard-regime certificate")
    print("  clusters / cluster size / anchors:", g, h, n)
    print("  q / epsilon:", q, epsilon)
    print("  posterior upper U:", upper, float(upper))
    print("  radius lower L:", radius, float(radius))
    print("  1-L/U:", 1 - radius / upper, float(1 - radius / upper))
    print("  same-transition upper:", float(same), "< 259")
    print("  stopped/posterior ratio lower:", Q(812_047, 277_579))
    print("  decimal ratio lower:", float(Q(812_047, 277_579)))


def asymptotic_family_checks() -> None:
    # These finite members are not a proof of the limit, but they guard the
    # substitutions g=m, h=m^2, q=1/m, epsilon=1/m^8 used in the note.
    previous_radius_ratio = Q(0)
    previous_stopped_ratio = Q(0)
    for m in (150, 200, 300):
        n, upper, radius, _, stopped = certificate(
            m, m * m, Q(1, m), Q(1, m**8)
        )
        assert n == m**3
        assert 20 * radius > 19 * upper
        assert 10 * stopped > 29 * upper
        assert radius / upper > previous_radius_ratio
        assert stopped / upper > previous_stopped_ratio
        previous_radius_ratio = radius / upper
        previous_stopped_ratio = stopped / upper


def main() -> None:
    brute_small_probability_checks()
    finite_certificate_checks()
    asymptotic_family_checks()
    print("all exact and structural checks passed")


if __name__ == "__main__":
    main()
