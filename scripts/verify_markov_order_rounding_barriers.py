#!/usr/bin/env python3
"""Exact audits for three structural symmetric-gap barriers.

The checks use Fraction arithmetic throughout.  They verify:

1. the exact first-visit cost formula for the posterior-marginal Markov
   kernel on a deterministic endpoint line, including a factor-2.9 failure;
2. the uniform/stationary size-biased-order formula on the same line; and
3. a forced shortest-full-tour graph metric whose stopped cost is above
   2.9 times an explicit upper bound on the posterior optimum.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import permutations


Q = Fraction


def harmonic(n: int) -> Q:
    return sum((Q(1, k) for k in range(1, n + 1)), Q(0))


def line_tour_cost(order: tuple[int, ...]) -> int:
    """Root is 0 and the deterministic clients have coordinates 1,...,n."""
    if not order:
        return 0
    return order[0] + sum(abs(u - v) for u, v in zip(order, order[1:])) + order[-1]


def frontier_order_distribution(n: int) -> dict[tuple[int, ...], Q]:
    """First-visit orders of SRW on the abstract cycle 0,1,...,n,0.

    Here 0 is the root.  After k clients have been discovered, gambler's
    ruin gives probability 1/(k+2) that the next new client is reached from
    the opposite discovery frontier.
    """
    assert n >= 2
    # State: (left prefix size, right suffix size, last side, order).
    states = {
        (1, 0, 0, (1,)): Q(1, 2),
        (0, 1, 1, (n,)): Q(1, 2),
    }
    for k in range(1, n - 1):
        switch = Q(1, k + 2)
        stay = 1 - switch
        next_states = defaultdict(Q)
        for (a, b, side, order), probability in states.items():
            left_new = a + 1
            right_new = n - b
            if side == 0:
                next_states[(a + 1, b, 0, (*order, left_new))] += probability * stay
                next_states[(a, b + 1, 1, (*order, right_new))] += probability * switch
            else:
                next_states[(a, b + 1, 1, (*order, right_new))] += probability * stay
                next_states[(a + 1, b, 0, (*order, left_new))] += probability * switch
        states = next_states

    distribution = defaultdict(Q)
    for (a, _b, _side, order), probability in states.items():
        # The sole remaining client lies between the two discovered fronts.
        distribution[(*order, a + 1)] += probability
    assert sum(distribution.values(), Q(0)) == 1
    return dict(distribution)


def markov_first_visit_line_cost(n: int) -> Q:
    return (n + 1) * harmonic(n) - Q(n, 2) + Q(1, 2)


def audit_markov_first_visit() -> tuple[Q, Q]:
    # Independent small-n enumeration checks the closed form.
    for n in range(2, 9):
        distribution = frontier_order_distribution(n)
        enumerated = sum(
            probability * line_tour_cost(order)
            for order, probability in distribution.items()
        )
        assert enumerated == markov_first_visit_line_cost(n)

    n = 299
    posterior = Q(2 * n)
    stopped = markov_first_visit_line_cost(n)
    assert 10 * stopped > 29 * posterior
    return stopped, posterior


def audit_stationary_size_biased() -> tuple[Q, Q]:
    # With p_i=1, sampling without replacement by stationary masses is a
    # uniform permutation.  Check the standard expectation formula directly
    # at small n before using the n=15 witness.
    for n in range(2, 8):
        enumerated = sum(
            (Q(line_tour_cost(order)) for order in permutations(range(1, n + 1))),
            Q(0),
        ) / Q(len(tuple(permutations(range(1, n + 1)))))
        formula = Q((n + 1) * (n + 2), 3)
        assert enumerated == formula

    n = 15
    posterior = Q(2 * n)
    stopped = Q((n + 1) * (n + 2), 3)
    assert 10 * stopped > 29 * posterior
    return stopped, posterior


def graph_metric(adjacency: list[set[int]]) -> list[list[int]]:
    n = len(adjacency)
    inf = 10**9
    distance = [[0 if i == j else (1 if j in adjacency[i] else inf) for j in range(n)] for i in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
    return distance


def audit_forced_full_tour() -> tuple[Q, Q]:
    # Seven always-active anchors form the good unit cycle.  The bad anchor
    # cycle takes step 3 modulo 7.  Each bad edge is replaced by a disjoint
    # three-edge unit path, with two rare internal clients.
    anchor_count = 7
    bad = tuple((3 * i) % anchor_count for i in range(anchor_count))
    adjacency = [set() for _ in range(anchor_count + 2 * anchor_count)]

    def add_edge(u: int, v: int) -> None:
        adjacency[u].add(v)
        adjacency[v].add(u)

    for i in range(anchor_count):
        add_edge(i, (i + 1) % anchor_count)

    paths = []
    forced_cycle = [bad[0]]
    next_vertex = anchor_count
    for i, left in enumerate(bad):
        right = bad[(i + 1) % anchor_count]
        x, y = next_vertex, next_vertex + 1
        next_vertex += 2
        add_edge(left, x)
        add_edge(x, y)
        add_edge(y, right)
        paths.append((left, x, y, right))
        forced_cycle.extend((x, y, right))
    assert forced_cycle[-1] == bad[0]
    assert len(adjacency) == 21

    distance = graph_metric(adjacency)
    for i in range(anchor_count):
        for j in range(anchor_count):
            assert distance[i][j] == min((i - j) % anchor_count, (j - i) % anchor_count)
    for left, x, y, right in paths:
        path = (left, x, y, right)
        for i in range(4):
            for j in range(4):
                assert distance[path[i]][path[j]] == abs(i - j)

    # Every metric edge is at least one, while the displayed 21-edge unit
    # Hamiltonian cycle attains 21.  Moreover every internal path client has
    # exactly two unit neighbors.  Thus any length-21 tour must use both of
    # its path edges, forcing the whole bad cycle (up to reversal).
    assert all(
        distance[i][j] >= 1
        for i in range(len(adjacency))
        for j in range(i + 1, len(adjacency))
    )
    for vertex in range(anchor_count, len(adjacency)):
        assert len(adjacency[vertex]) == 2
    full_tour = Q(sum(distance[u][v] for u, v in zip(forced_cycle, forced_cycle[1:])))
    assert full_tour == 21

    # All anchors are retained.  On each bad path, deleting either or both
    # rare clients leaves a geodesically telescoping segment of cost three.
    # Hence the stopped cost of every shortest full tour is identically 21.
    for mask in range(1 << (2 * anchor_count)):
        retained = [forced_cycle[0]]
        for vertex in forced_cycle[1:-1]:
            if vertex < anchor_count or mask & (1 << (vertex - anchor_count)):
                retained.append(vertex)
        retained.append(forced_cycle[-1])
        assert sum(distance[u][v] for u, v in zip(retained, retained[1:])) == 21
    stopped = Q(21)

    # A posterior comparison walk uses the good anchor cycle (cost seven) and
    # a two-edge excursion for each active rare client.  With epsilon=1/1000,
    # this gives P <= 7 + 2*14*epsilon = 1757/250.
    epsilon = Q(1, 1000)
    posterior_upper = Q(7) + 2 * (2 * anchor_count) * epsilon
    assert posterior_upper == Q(1757, 250)
    assert 10 * stopped > 29 * posterior_upper
    return stopped, posterior_upper


def main() -> None:
    markov, line_posterior = audit_markov_first_visit()
    stationary, stationary_posterior = audit_stationary_size_biased()
    full, posterior_upper = audit_forced_full_tour()
    print("exact Markov/order barriers passed")
    print("  first-visit K-walk line ratio:", float(markov / line_posterior))
    print("  stationary size-biased line ratio:", float(stationary / stationary_posterior))
    print("  forced full-tour ratio lower bound:", float(full / posterior_upper))


if __name__ == "__main__":
    main()
