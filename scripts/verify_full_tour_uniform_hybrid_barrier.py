#!/usr/bin/env python3
"""Exact finite barrier to a full-tour/uniform-order hybrid.

The proposed hybrid takes the cheaper of

* a shortest full-support metric TSP tour (with the best tie breaking), and
* a uniformly random full client order.

Thirteen always-active anchors form a unit cycle.  The step-six anchor cycle
is subdivided into internally disjoint six-edge unit paths, with every new
internal client active with probability 10^-6.  The subdivided step-six cycle
is a unit Hamiltonian cycle on all 78 vertices, and degree-two internal
vertices force every shortest full-support tour to use it.  Once rare clients
are shortcut, that tour still costs 78.  A uniform order costs at least the
uniform-order cost on the anchors, 91/2.  Yet the posterior optimum is at most
13 + 234/10^6.  Both ratios are therefore strictly above 29/10.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations


Q = Fraction
ANCHORS = 13
STEP = 6
EPSILON = Q(1, 1_000_000)


def graph_metric(adjacency: list[set[int]]) -> list[list[int]]:
    count = len(adjacency)
    infinity = 10**9
    distance = [
        [0 if first == second else 1 if second in adjacency[first] else infinity
         for second in range(count)]
        for first in range(count)
    ]
    for pivot in range(count):
        for first in range(count):
            for second in range(count):
                distance[first][second] = min(
                    distance[first][second],
                    distance[first][pivot] + distance[pivot][second],
                )
    return distance


def cycle_distance(first: int, second: int, count: int) -> int:
    difference = abs(first - second)
    return min(difference, count - difference)


def uniform_anchor_cost(count: int) -> Q:
    pair_sum = sum(
        cycle_distance(first, second, count)
        for first in range(count)
        for second in range(first + 1, count)
    )
    return Q(2 * pair_sum, count - 1)


def audit_small_uniform_formula() -> None:
    for count in range(3, 9):
        enumerated = Q(0)
        for order in permutations(range(1, count)):
            tour = (0, *order, 0)
            enumerated += sum(
                cycle_distance(first, second, count)
                for first, second in zip(tour, tour[1:])
            )
        enumerated /= Q(1)
        for value in range(2, count):
            enumerated /= value
        assert enumerated == uniform_anchor_cost(count)


def build_instance():
    bad_order = tuple((STEP * index) % ANCHORS for index in range(ANCHORS))
    rare_count = ANCHORS * (STEP - 1)
    adjacency = [set() for _ in range(ANCHORS + rare_count)]

    def add_edge(first: int, second: int) -> None:
        adjacency[first].add(second)
        adjacency[second].add(first)

    for anchor in range(ANCHORS):
        add_edge(anchor, (anchor + 1) % ANCHORS)

    paths: list[tuple[int, ...]] = []
    forced_cycle = [bad_order[0]]
    next_vertex = ANCHORS
    for index, left in enumerate(bad_order):
        right = bad_order[(index + 1) % ANCHORS]
        internal = tuple(range(next_vertex, next_vertex + STEP - 1))
        next_vertex += STEP - 1
        path = (left, *internal, right)
        paths.append(path)
        forced_cycle.extend((*internal, right))
        for first, second in zip(path, path[1:]):
            add_edge(first, second)

    assert next_vertex == len(adjacency)
    assert forced_cycle[0] == forced_cycle[-1] == 0
    assert len(forced_cycle) - 1 == len(adjacency)
    return adjacency, paths, tuple(forced_cycle)


def main() -> None:
    audit_small_uniform_formula()
    adjacency, paths, forced_cycle = build_instance()
    distance = graph_metric(adjacency)

    # The anchor metric remains the original thirteen-cycle metric, and every
    # subdivided bad edge is a geodesic at all of its internal subsegments.
    for first in range(ANCHORS):
        for second in range(ANCHORS):
            assert distance[first][second] == cycle_distance(
                first, second, ANCHORS
            )
    for path in paths:
        for first in range(len(path)):
            for second in range(len(path)):
                assert distance[path[first]][path[second]] == abs(first - second)

    # All nonzero distances are at least one, while the forced cycle has one
    # unit edge per vertex.  Every rare vertex has graph degree two; any
    # length-|V| full tour must therefore use both incident path edges at each
    # rare vertex, which also saturates both incidences of every anchor.
    vertex_count = len(adjacency)
    assert vertex_count == ANCHORS * STEP == 78
    assert all(
        distance[first][second] >= 1
        for first in range(vertex_count)
        for second in range(first + 1, vertex_count)
    )
    assert all(len(adjacency[vertex]) == 2 for vertex in range(ANCHORS, vertex_count))
    forced_length = sum(
        distance[first][second]
        for first, second in zip(forced_cycle, forced_cycle[1:])
    )
    assert forced_length == vertex_count

    # Shortcutting any subset of internal vertices on a geodesic path leaves
    # that path's anchor-to-anchor contribution equal to STEP.  Hence every
    # shortest full-support tour has stopped cost exactly |V|.
    stopped_full = Q(vertex_count)

    # Posterior comparison: traverse the good anchor cycle and make a doubled
    # nearest-anchor excursion for each active rare client.
    one_path_attachment = sum(
        min(position, STEP - position) for position in range(1, STEP)
    )
    assert one_path_attachment == 9
    posterior_upper = Q(ANCHORS) + 2 * EPSILON * ANCHORS * one_path_attachment
    assert posterior_upper == Q(6_500_117, 500_000)

    # Adding active rare clients to a fixed cyclic order cannot reduce cost.
    # A uniform full order induces a uniform relative order on the deterministic
    # anchors, whose exact expected cycle-metric cost is 91/2.
    uniform_lower = uniform_anchor_cost(ANCHORS)
    assert uniform_lower == Q(91, 2)

    target = Q(29, 10) * posterior_upper
    assert stopped_full > target
    assert uniform_lower > target

    print("exact full-tour/uniform hybrid barrier verified")
    print(f"vertices={vertex_count}, anchors={ANCHORS}, rare={vertex_count-ANCHORS}")
    print(f"posterior upper={posterior_upper} = {float(posterior_upper):.12f}")
    print(f"shortest-full stopped={stopped_full}, ratio>{float(stopped_full/posterior_upper):.12f}")
    print(f"uniform lower={uniform_lower}, ratio>{float(uniform_lower/posterior_upper):.12f}")


if __name__ == "__main__":
    main()
