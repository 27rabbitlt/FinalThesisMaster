#!/usr/bin/env python3
"""Exact counterexample to the recursive posterior-block marginal target.

The six-client metric and p=1/2 refute

    Q(p) <= (29/10) * (P(p) - P(p(1-p))),

even when every guide-set tour is tie-broken to minimize its stopped cost.
Here Q is the expected p-stopped cost of a shortest tour on an independent
p-sample.  The calculation uses exact rational arithmetic and exhaustive
enumeration of every subset and every tour on that subset.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations


Q = Fraction

D = (
    (0, 18, 20, 32, 27, 16, 24),
    (18, 0, 2, 14, 9, 7, 6),
    (20, 2, 0, 12, 9, 5, 6),
    (32, 14, 12, 0, 6, 17, 9),
    (27, 9, 9, 6, 0, 14, 3),
    (16, 7, 5, 17, 14, 0, 11),
    (24, 6, 6, 9, 3, 11, 0),
)
N = 6


def check_metric() -> None:
    for i in range(N + 1):
        assert D[i][i] == 0
        for j in range(N + 1):
            assert D[i][j] == D[j][i]
            for k in range(N + 1):
                assert D[i][j] <= D[i][k] + D[k][j]


def tour_cost(order: tuple[int, ...]) -> int:
    if not order:
        return 0
    return (
        D[0][order[0]]
        + sum(D[u][v] for u, v in zip(order, order[1:]))
        + D[order[-1]][0]
    )


def stopped_cost(order: tuple[int, ...], p: Q) -> Q:
    ans = Q(0)
    for mask in range(1 << len(order)):
        active = tuple(order[j] for j in range(len(order)) if mask >> j & 1)
        probability = p ** len(active) * (1 - p) ** (len(order) - len(active))
        ans += probability * tour_cost(active)
    return ans


def subset_optima(tie_p: Q) -> tuple[list[int], list[tuple[int, ...]]]:
    costs = [0] * (1 << N)
    orders: list[tuple[int, ...]] = [tuple()] * (1 << N)
    for mask in range(1, 1 << N):
        vertices = tuple(i + 1 for i in range(N) if mask >> i & 1)
        candidates = list(permutations(vertices))
        optimum = min(tour_cost(order) for order in candidates)
        shortest = [order for order in candidates if tour_cost(order) == optimum]
        chosen = min(shortest, key=lambda order: (stopped_cost(order, tie_p), order))
        costs[mask] = optimum
        orders[mask] = chosen
    return costs, orders


def posterior(costs: list[int], p: Q) -> Q:
    ans = Q(0)
    for mask, cost in enumerate(costs):
        cardinality = mask.bit_count()
        ans += p**cardinality * (1 - p) ** (N - cardinality) * cost
    return ans


def main() -> None:
    check_metric()
    p = Q(1, 2)
    costs, orders = subset_optima(p)
    post = posterior(costs, p)
    residual = posterior(costs, p * (1 - p))
    drop = post - residual
    stopped_guide = Q(0)
    for mask, order in enumerate(orders):
        cardinality = mask.bit_count()
        probability = p**cardinality * (1 - p) ** (N - cardinality)
        stopped_guide += probability * stopped_cost(order, p)

    assert post == Q(3655, 64)
    assert residual == Q(174105, 4096)
    assert drop == Q(59815, 4096)
    assert stopped_guide == Q(174105, 4096)
    assert 10 * stopped_guide > 29 * drop
    print("P(1/2) =", post)
    print("P(1/4) =", residual)
    print("drop =", drop)
    print("Q =", stopped_guide)
    print("Q/drop =", stopped_guide / drop, float(stopped_guide / drop))
    print("recursive self-thinning counterexample verified")


if __name__ == "__main__":
    main()
