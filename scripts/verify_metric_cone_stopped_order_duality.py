#!/usr/bin/env python3
"""Exact audits for metric-cone stopped-order duality.

The script checks:
  * the vector stopped-segment identity and its triangle decomposition;
  * an exact cut-positive, forced-degree vector outside the metric dual cone;
  * a factor-four failure of the degree-only posterior relaxation; and
  * the Petersen point satisfying degree/cut constraints but not belonging
    to the Hamilton-cycle polytope.

All mathematical assertions use Fraction or integer arithmetic.
"""

from collections import deque
from fractions import Fraction
from itertools import combinations, permutations


Q = Fraction


def edge(u: int, v: int) -> tuple[int, int]:
    assert u != v
    return (u, v) if u < v else (v, u)


def complete_edges(vertex_count: int) -> list[tuple[int, int]]:
    return list(combinations(range(vertex_count), 2))


def zero_vector(vertex_count: int) -> dict[tuple[int, int], Q]:
    return {e: Q(0) for e in complete_edges(vertex_count)}


def add_scaled(
    target: dict[tuple[int, int], Q],
    source: dict[tuple[int, int], Q],
    scale: Q,
) -> None:
    for e, value in source.items():
        target[e] += scale * value


def triangle_normal(
    vertex_count: int, long_u: int, long_v: int, pivot: int
) -> dict[tuple[int, int], Q]:
    assert len({long_u, long_v, pivot}) == 3
    ans = zero_vector(vertex_count)
    ans[edge(long_u, pivot)] += 1
    ans[edge(long_v, pivot)] += 1
    ans[edge(long_u, long_v)] -= 1
    return ans


def path_slack_triangle_decomposition(
    vertex_count: int, path: tuple[int, ...]
) -> dict[tuple[int, int], Q]:
    """Return path incidence minus its endpoint chord as triangle normals."""
    assert len(path) >= 2
    ans = zero_vector(vertex_count)
    if path[0] != path[-1]:
        for t in range(2, len(path)):
            add_scaled(
                ans,
                triangle_normal(vertex_count, path[0], path[t], path[t - 1]),
                Q(1),
            )
        return ans

    # The only use below is a simple cycle with the root repeated.  Decompose
    # the open prefix, then use 2 e_ab = a^{ak|b} + a^{bk|a}.
    assert len(path) >= 4
    prefix = path[:-1]
    ans = path_slack_triangle_decomposition(vertex_count, prefix)
    a, b = path[0], path[-2]
    k = path[1]
    assert len({a, b, k}) == 3
    add_scaled(ans, triangle_normal(vertex_count, a, k, b), Q(1))
    add_scaled(ans, triangle_normal(vertex_count, b, k, a), Q(1))
    return ans


def direct_path_slack(
    vertex_count: int, path: tuple[int, ...]
) -> dict[tuple[int, int], Q]:
    ans = zero_vector(vertex_count)
    for u, v in zip(path, path[1:]):
        ans[edge(u, v)] += 1
    if path[0] != path[-1]:
        ans[edge(path[0], path[-1])] -= 1
    return ans


def stopped_segment_identity_check() -> None:
    probabilities = (Q(1, 5), Q(1, 3), Q(2, 5), Q(1, 2))
    order = (2, 0, 3, 1)
    n = len(order)
    root = n
    vertex_count = n + 1

    x = zero_vector(vertex_count)
    for i, u in enumerate(order):
        left_empty = Q(1)
        for k in range(i):
            left_empty *= 1 - probabilities[order[k]]
        right_empty = Q(1)
        for k in range(i + 1, n):
            right_empty *= 1 - probabilities[order[k]]
        x[edge(root, u)] += probabilities[u] * (left_empty + right_empty)
        for j in range(i + 1, n):
            v = order[j]
            middle_empty = Q(1)
            for k in range(i + 1, j):
                middle_empty *= 1 - probabilities[order[k]]
            x[edge(u, v)] += probabilities[u] * probabilities[v] * middle_empty

    rooted_order = (root, *order, root)
    full = zero_vector(vertex_count)
    for u, v in zip(rooted_order, rooted_order[1:]):
        full[edge(u, v)] += 1

    segment_sum = zero_vector(vertex_count)
    triangle_sum = zero_vector(vertex_count)
    for i in range(n + 1):
        for j in range(i + 1, n + 2):
            alpha = Q(1)
            if i:
                alpha *= probabilities[rooted_order[i]]
            if j < n + 1:
                alpha *= probabilities[rooted_order[j]]
            for k in range(i + 1, j):
                alpha *= 1 - probabilities[rooted_order[k]]
            path = rooted_order[i : j + 1]
            direct = direct_path_slack(vertex_count, path)
            decomposed = path_slack_triangle_decomposition(vertex_count, path)
            assert direct == decomposed
            add_scaled(segment_sum, direct, alpha)
            add_scaled(triangle_sum, decomposed, alpha)

    assert segment_sum == triangle_sum
    assert {e: full[e] - x[e] for e in full} == segment_sum

    for v, pv in enumerate(probabilities):
        assert sum(x[e] for e in x if v in e) == 2 * pv
    nonempty_probability = 1
    for pv in probabilities:
        nonempty_probability *= 1 - pv
    nonempty_probability = 1 - nonempty_probability
    assert sum(x[e] for e in x if root in e) == 2 * nonempty_probability


def cut_value(
    vector: dict[tuple[int, int], Q], selected: set[int]
) -> Q:
    return sum(
        value
        for (u, v), value in vector.items()
        if (u in selected) != (v in selected)
    )


def forced_degree_cut_counterexample_check() -> None:
    p = Q(1, 20)
    q = 1 - (1 - p) ** 4
    c = Q(29, 10)
    a = (q + p) / 2
    assert q == Q(29679, 160000)
    assert a == Q(37679, 320000)

    z0 = {
        (0, 1): a,
        (0, 2): -p / 2,
        (0, 3): -p / 2,
        (0, 4): a,
        (1, 2): p,
        (1, 3): p,
        (1, 4): -(a + p),
        (2, 3): -p / 2,
        (2, 4): p,
        (3, 4): p,
    }
    assert set(z0) == set(complete_edges(5))
    assert sum(z0[e] for e in z0 if 0 in e) == q
    for v in range(1, 5):
        assert sum(z0[e] for e in z0 if v in e) == p

    for mask in range(1, (1 << 5) - 1):
        selected = {v for v in range(5) if mask & (1 << v)}
        assert cut_value(z0, selected) >= 0

    distance = (
        (0, 3, 4, 4, 3),
        (3, 0, 5, 1, 6),
        (4, 5, 0, 6, 1),
        (4, 1, 6, 0, 5),
        (3, 6, 1, 5, 0),
    )
    for i in range(5):
        for j in range(5):
            assert distance[i][j] == distance[j][i]
            if i != j:
                assert distance[i][j] > 0
            for k in range(5):
                assert distance[i][j] <= distance[i][k] + distance[k][j]

    metric_pairing = sum(z0[e] * distance[e[0]][e[1]] for e in z0)
    assert metric_pairing == -Q(1, 20)

    scale = 2 * (c - 1)
    z = {e: scale * value for e, value in z0.items()}
    assert scale == Q(19, 5)
    assert sum(z[e] for e in z if 0 in e) == 2 * (c - 1) * q
    for v in range(1, 5):
        assert sum(z[e] for e in z if v in e) == 2 * (c - 1) * p
    for mask in range(1, (1 << 5) - 1):
        selected = {v for v in range(5) if mask & (1 << v)}
        assert cut_value(z, selected) >= 0
    assert sum(z[e] * distance[e[0]][e[1]] for e in z) == -Q(19, 100)


def degree_only_factor_four_check() -> None:
    vertex_count = 6
    first = {0, 1, 2}
    second = {3, 4, 5}
    distance = [[0] * vertex_count for _ in range(vertex_count)]
    for i in range(vertex_count):
        for j in range(i + 1, vertex_count):
            value = 1 if (i in first) == (j in first) else 10
            distance[i][j] = distance[j][i] = value
    for i in range(vertex_count):
        for j in range(vertex_count):
            for k in range(vertex_count):
                assert distance[i][j] <= distance[i][k] + distance[k][j]

    relaxed = zero_vector(vertex_count)
    for u, v in ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)):
        relaxed[edge(u, v)] += 1
    for v in range(vertex_count):
        assert sum(relaxed[e] for e in relaxed if v in e) == 2
    relaxed_cost = sum(relaxed[e] * distance[e[0]][e[1]] for e in relaxed)
    assert relaxed_cost == 6

    root = 0
    clients = tuple(range(1, vertex_count))
    best = None
    for order in permutations(clients):
        cycle = (root, *order, root)
        value = sum(distance[u][v] for u, v in zip(cycle, cycle[1:]))
        best = value if best is None else min(best, value)
    assert best == 24
    assert Q(best, relaxed_cost) == 4
    assert Q(best, 1) > Q(29, 10) * relaxed_cost


def petersen_degree_cut_separation_check() -> None:
    vertex_count = 10
    petersen_edges = {
        edge(0, 1),
        edge(1, 2),
        edge(2, 3),
        edge(3, 4),
        edge(4, 0),
        edge(0, 5),
        edge(1, 6),
        edge(2, 7),
        edge(3, 8),
        edge(4, 9),
        edge(5, 7),
        edge(7, 9),
        edge(9, 6),
        edge(6, 8),
        edge(8, 5),
    }
    assert len(petersen_edges) == 15
    y = {
        e: Q(2, 3) if e in petersen_edges else Q(0)
        for e in complete_edges(vertex_count)
    }
    for v in range(vertex_count):
        assert sum(y[e] for e in y if v in e) == 2
    for mask in range(1, (1 << vertex_count) - 1):
        selected = {v for v in range(vertex_count) if mask & (1 << v)}
        assert cut_value(y, selected) >= 2

    adjacency = {v: set() for v in range(vertex_count)}
    for u, v in petersen_edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    def has_hamilton_cycle() -> bool:
        def extend(path: tuple[int, ...], unused: frozenset[int]) -> bool:
            if not unused:
                return path[0] in adjacency[path[-1]]
            return any(
                extend((*path, nxt), unused - {nxt})
                for nxt in adjacency[path[-1]] & unused
            )

        return extend((0,), frozenset(range(1, vertex_count)))

    assert not has_hamilton_cycle()

    distance = [[vertex_count] * vertex_count for _ in range(vertex_count)]
    for source in range(vertex_count):
        distance[source][source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if distance[source][v] == vertex_count:
                    distance[source][v] = distance[source][u] + 1
                    queue.append(v)
    for i in range(vertex_count):
        for j in range(vertex_count):
            for k in range(vertex_count):
                assert distance[i][j] <= distance[i][k] + distance[k][j]

    y_cost = sum(y[e] * distance[e[0]][e[1]] for e in y)
    assert y_cost == 10

    # Held--Karp dynamic programming for a rooted Hamilton cycle in the metric.
    clients = tuple(range(1, vertex_count))
    dp: dict[tuple[int, int], int] = {}
    for v in clients:
        dp[(1 << (v - 1), v)] = distance[0][v]
    for size in range(2, len(clients) + 1):
        for subset_tuple in combinations(clients, size):
            mask = sum(1 << (v - 1) for v in subset_tuple)
            for last in subset_tuple:
                previous_mask = mask ^ (1 << (last - 1))
                dp[(mask, last)] = min(
                    dp[(previous_mask, prev)] + distance[prev][last]
                    for prev in subset_tuple
                    if prev != last
                )
    full_mask = (1 << len(clients)) - 1
    optimum = min(dp[(full_mask, last)] + distance[last][0] for last in clients)
    assert optimum == 11
    assert Q(optimum, 1) / y_cost == Q(11, 10)


def main() -> None:
    stopped_segment_identity_check()
    forced_degree_cut_counterexample_check()
    degree_only_factor_four_check()
    petersen_degree_cut_separation_check()
    print("metric-cone stopped-order duality audits passed exactly")
    print("cut-positive forced-degree witness has metric pairing -19/100")
    print("degree-only gap = 4; Petersen degree/cut gap = 11/10")


if __name__ == "__main__":
    main()
