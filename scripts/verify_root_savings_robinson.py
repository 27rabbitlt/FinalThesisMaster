#!/usr/bin/env python3
"""Exact audits for the root-savings/Robinson-order reformulation."""

from __future__ import annotations

from fractions import Fraction as Q
from itertools import permutations
from math import comb


def floyd(weights: list[list[Q]]) -> list[list[Q]]:
    d = [row[:] for row in weights]
    for k in range(len(d)):
        for i in range(len(d)):
            for j in range(len(d)):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    return d


def test_metric() -> list[list[Q]]:
    inf = Q(10**6)
    n = 5
    w = [[Q(0) if i == j else inf for j in range(n)] for i in range(n)]
    for u, v, length in (
        (0, 1, 3),
        (0, 2, 4),
        (0, 3, 5),
        (0, 4, 4),
        (1, 2, 2),
        (2, 3, 2),
        (3, 4, 3),
        (4, 1, 2),
        (1, 3, 4),
        (2, 4, 4),
    ):
        w[u][v] = w[v][u] = Q(length)
    return floyd(w)


def path_saving(s: list[list[Q]], order: tuple[int, ...]) -> Q:
    return sum(s[u][v] for u, v in zip(order, order[1:]))


def tour_cost(d: list[list[Q]], order: tuple[int, ...], root: int = 0) -> Q:
    if not order:
        return Q(0)
    walk = (root,) + order + (root,)
    return sum(d[u][v] for u, v in zip(walk, walk[1:]))


def probability(mask: int, clients: tuple[int, ...], p: tuple[Q, ...]) -> Q:
    answer = Q(1)
    for v in clients:
        answer *= p[v] if mask >> (v - 1) & 1 else 1 - p[v]
    return answer


def expected_fixed_saving(
    s: list[list[Q]], order: tuple[int, ...], p: tuple[Q, ...]
) -> Q:
    answer = Q(0)
    for i, u in enumerate(order):
        skipped = Q(1)
        for j in range(i + 1, len(order)):
            v = order[j]
            answer += p[u] * p[v] * skipped * s[u][v]
            skipped *= 1 - p[v]
    return answer


def is_robinson(t: list[list[Q]], order: tuple[int, ...]) -> bool:
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            for k in range(j + 1, len(order)):
                if t[order[i]][order[k]] > min(
                    t[order[i]][order[j]], t[order[j]][order[k]]
                ):
                    return False
    return True


def greatest_robinson_minorant(
    s: list[list[Q]], order: tuple[int, ...]
) -> list[list[Q]]:
    t = [[Q(0) for _ in s] for _ in s]
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            value = min(
                s[order[k]][order[l]]
                for k in range(i, j)
                for l in range(k + 1, j + 1)
            )
            t[order[i]][order[j]] = t[order[j]][order[i]] = value
    return t


def path_spoke_minorant(
    d: list[list[Q]], order: tuple[int, ...], root: int = 0
) -> list[list[Q]]:
    positions = [Q(0)]
    for u, v in zip(order, order[1:]):
        positions.append(positions[-1] + d[u][v])
    t = [[Q(0) for _ in d] for _ in d]
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            u, v = order[i], order[j]
            value = max(Q(0), d[root][u] + d[root][v] - (positions[j] - positions[i]))
            t[u][v] = t[v][u] = value
    return t


def check_exact_reformulation() -> None:
    d = test_metric()
    clients = (1, 2, 3, 4)
    p = (Q(1), Q(1, 3), Q(1, 2), Q(2, 3), Q(1, 4))
    a = tuple(d[0][v] for v in range(len(d)))
    s = [
        [Q(0) if u == v else a[u] + a[v] - d[u][v] for v in range(len(d))]
        for u in range(len(d))
    ]

    # Exact characterization constraints inherited from the metric.
    for u in clients:
        for v in clients:
            if u == v:
                continue
            assert 0 <= s[u][v] <= 2 * min(a[u], a[v])
            for w in clients:
                if len({u, v, w}) == 3:
                    assert s[u][w] >= s[u][v] + s[v][w] - 2 * a[v]

    baseline = 2 * sum(p[v] * a[v] for v in clients)
    posterior = Q(0)
    posterior_saving = Q(0)
    for mask in range(1 << len(clients)):
        active = tuple(v for v in clients if mask >> (v - 1) & 1)
        mass = probability(mask, clients, p)
        costs = [tour_cost(d, order) for order in permutations(active)]
        savings = [path_saving(s, order) for order in permutations(active)]
        assert min(costs) == 2 * sum(a[v] for v in active) - max(savings)
        posterior += mass * min(costs)
        posterior_saving += mass * max(savings)
    assert posterior == baseline - posterior_saving

    fixed_savings = []
    fixed_costs = []
    for order in permutations(clients):
        formula = expected_fixed_saving(s, order, p)
        enumerated_cost = Q(0)
        enumerated_saving = Q(0)
        for mask in range(1 << len(clients)):
            active = tuple(v for v in order if mask >> (v - 1) & 1)
            mass = probability(mask, clients, p)
            enumerated_cost += mass * tour_cost(d, active)
            enumerated_saving += mass * path_saving(s, active)
        assert formula == enumerated_saving
        assert enumerated_cost == baseline - formula
        fixed_savings.append(formula)
        fixed_costs.append(enumerated_cost)
    assert min(fixed_costs) == baseline - max(fixed_savings)


def check_robinson_lemmas() -> None:
    d = test_metric()
    order = (3, 1, 4, 2)
    a = tuple(d[0][v] for v in range(len(d)))
    s = [
        [Q(0) if u == v else a[u] + a[v] - d[u][v] for v in range(len(d))]
        for u in range(len(d))
    ]
    path_minorant = path_spoke_minorant(d, order)
    greatest = greatest_robinson_minorant(s, order)
    for t in (path_minorant, greatest):
        assert is_robinson(t, order)
        assert all(t[u][v] <= s[u][v] for u in order for v in order)
        for mask in range(1 << len(order)):
            active_natural = tuple(v for i, v in enumerate(order) if mask >> i & 1)
            natural = path_saving(t, active_natural)
            assert natural == max(path_saving(t, candidate) for candidate in permutations(active_natural))
    assert all(path_minorant[u][v] <= greatest[u][v] for u in order for v in order)


def far_clique_values(n: int, p: Q, radius: int) -> tuple[Q, Q, Q, Q]:
    """Return P=F, path-spoke upper, posterior saving, adjacent-only saving."""
    q = 1 - p
    occupied = 1 - q**n
    posterior = n * p + (2 * radius - 1) * occupied
    path_spoke = 2 * radius * occupied + sum(
        (n - gap) * p * p * q ** (gap - 1) * min(gap, 2 * radius)
        for gap in range(1, n)
    )
    posterior_saving = (2 * radius - 1) * (n * p - occupied)
    adjacent_saving = (n - 1) * p * p * (2 * radius - 1)
    return posterior, path_spoke, posterior_saving, adjacent_saving


def check_long_gap_counterexamples() -> None:
    n, p, radius = 150, Q(1, 4), 3
    posterior, path_spoke, posterior_saving, adjacent_saving = far_clique_values(
        n, p, radius
    )

    # Every order is statewise optimal in the original metric, so F=P.
    assert posterior == n * p + (2 * radius - 1) * (1 - (1 - p) ** n)
    assert path_spoke == Q(510_431, 4096)
    assert posterior < Q(85, 2)
    assert path_spoke > Q(29, 10) * posterior
    assert Q(510_431, 4096) > Q(29, 10) * Q(85, 2)

    # Keeping only immediate full-order adjacencies also misses far too much.
    assert posterior_saving - adjacent_saving > Q(19, 10) * posterior

    # The true constant savings kernel is already Robinson, so its greatest
    # Robinson minorant loses nothing; the failure is specific to the path
    # surrogate, not to Robinson similarities themselves.
    q = 1 - p
    true_stopped_saving = (2 * radius - 1) * (n * p - (1 - q**n))
    assert true_stopped_saving == posterior_saving

    limiting_ratio = (1 - q ** (2 * radius)) / p
    assert limiting_ratio == Q(3367, 1024) > Q(29, 10)


def check_arbitrary_tie_breaking_counterexample() -> None:
    # All clients are active, all client-client distances are one, and all
    # root distances are M.  Every client order is posterior-optimal.
    n, radius = 20, 10
    posterior = Q(2 * radius + n - 1)

    # If the selected posterior path is uniform over all permutations, each
    # unordered client pair is used with probability 2/n.  In the path-spoke
    # metric of any fixed order, its distance is the positional gap because
    # 2M >= n-1.
    sum_positional_gaps = Q(n * (n * n - 1), 6)
    routed_value = 2 * radius + Q(2, n) * sum_positional_gaps
    assert posterior == 39
    assert routed_value == 153
    assert routed_value > Q(29, 10) * posterior


def check_same_threshold_counterexample() -> None:
    # At savings levels (2,3], the metric with root radii two, cross-part
    # distances one, and within-part distances two has G_lambda=K_{m,m}.
    m = 100
    p = Q(1, 10)
    q = 1 - p
    binomial = [Q(comb(m, value)) * p**value * q ** (m - value) for value in range(m + 1)]
    high_edges = Q(0)
    layer_deficit = Q(0)
    for left in range(m + 1):
        for right in range(m + 1):
            mass = binomial[left] * binomial[right]
            maximum = 2 * min(left, right) - int(left == right and left > 0)
            high_edges += mass * maximum
            layer_deficit += mass * (left + right - maximum)

    expected_vertices = 2 * m * p
    assert high_edges + layer_deficit == expected_vertices

    # K_{m,m} is triangle-free, so a full interval is a clique only when it
    # has at most two labels.  An alternating order attains this upper bound.
    best_clique_interval_edges = (2 * m - 1) * p * p
    assert best_clique_interval_edges == Q(199, 100)
    assert high_edges - best_clique_interval_edges > Q(19, 10) * layer_deficit

    # A short analytic audit of the same strict inequality: the layer deficit
    # is E|L-R|+Pr[L=R>0] <= sqrt(18)+1 < 21/4.
    assert Q(18) < Q(17, 4) ** 2
    assert expected_vertices - best_clique_interval_edges > Q(29, 10) * Q(21, 4)


def check_recursive_extension_bound_counterexample() -> None:
    # The recursive append-and-recurse analysis pays a separate rooted tour
    # for every guide layer.  On the far clique those root payments are fake:
    # the final concatenated order is actually optimal, but the decomposed
    # upper bound already exceeds 2.9 P after only three layers.
    n = 100
    radius = 10_000
    p0 = Q(1, 2)
    p1 = p0 * (1 - p0)
    p2 = p1 * (1 - p1)
    assert (p0, p1, p2) == (Q(1, 2), Q(1, 4), Q(3, 16))

    posterior = n * p0 + (2 * radius - 1) * (1 - (1 - p0) ** n)
    first_three_occupancies = sum(
        1 - (1 - probability * probability) ** n
        for probability in (p0, p1, p2)
    )
    decomposed_lower_bound = (2 * radius - 1) * first_three_occupancies
    assert decomposed_lower_bound > Q(29, 10) * posterior

    # Coordinatewise p_t-p_{t+1}=p_t^2, which is why all non-root baseline
    # terms telescope; only the repeatedly charged occupied-layer root terms
    # cause the obstruction.
    for probability in (p0, p1, p2):
        assert probability - probability * (1 - probability) == probability**2


def check_cross_block_savings_identity() -> None:
    d = test_metric()
    blocks = ((3, 1), (4,), (2,))
    full_order = tuple(v for block in blocks for v in block)
    a = tuple(d[0][v] for v in range(len(d)))
    s = [
        [Q(0) if u == v else a[u] + a[v] - d[u][v] for v in range(len(d))]
        for u in range(len(d))
    ]
    for mask in range(1 << len(full_order)):
        active_blocks = [
            tuple(v for v in block if mask >> (v - 1) & 1) for block in blocks
        ]
        nonempty = [block for block in active_blocks if block]
        separate = sum(tour_cost(d, block) for block in nonempty)
        cross_saving = sum(s[left[-1]][right[0]] for left, right in zip(nonempty, nonempty[1:]))
        active_full = tuple(v for v in full_order if mask >> (v - 1) & 1)
        assert tour_cost(d, active_full) == separate - cross_saving


def main() -> None:
    check_exact_reformulation()
    check_robinson_lemmas()
    check_long_gap_counterexamples()
    check_arbitrary_tie_breaking_counterexample()
    check_same_threshold_counterexample()
    check_recursive_extension_bound_counterexample()
    check_cross_block_savings_identity()
    print("root-savings and Robinson-order checks passed")


if __name__ == "__main__":
    main()
