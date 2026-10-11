#!/usr/bin/env python3
"""Exact certificate refuting the shortest-full-support-tour P45 candidate.

There are seven always-active anchors a_0,...,a_6, with a_0 the depot.
The unit-edge graph contains the ordinary anchor cycle and a step-three
anchor cycle whose seven edges are each subdivided into three unit edges.
The fourteen subdivision vertices are independently active with probability
epsilon=1/1000.  Metric distances are shortest-path distances in this graph.

Every shortest full-support metric tour has length 21 and is forced to use
the subdivided step-three cycle.  If no subdivision vertex is active, its
shortcut has cost 21, whereas the posterior anchor tour has cost 7.  The
simple exact bounds below already keep the finite-epsilon ratio above 29/10.
"""

from fractions import Fraction


Q = Fraction
ANCHORS = 7
STEP = 3
SEGMENT_LENGTH = 3
EPSILON = Q(1, 1_000)


def build_graph():
    # Vertices 0,...,6 are anchors.  Vertex 0 is the depot.  Each bad-cycle
    # segment i gets two private subdivision vertices.
    adjacency = [set() for _ in range(ANCHORS + 2 * ANCHORS)]

    def add_edge(u, v):
        adjacency[u].add(v)
        adjacency[v].add(u)

    for i in range(ANCHORS):
        add_edge(i, (i + 1) % ANCHORS)

    segment_vertices = {}
    for i in range(ANCHORS):
        x = ANCHORS + 2 * i
        y = x + 1
        j = (i + STEP) % ANCHORS
        add_edge(i, x)
        add_edge(x, y)
        add_edge(y, j)
        segment_vertices[i] = (x, y)
    bad_cycle = []
    for t in range(ANCHORS):
        i = (STEP * t) % ANCHORS
        bad_cycle.extend((i, *segment_vertices[i]))
    return adjacency, bad_cycle


def all_pairs_shortest_paths(adjacency):
    n = len(adjacency)
    infinity = n + 1
    distance = [[0 if i == j else infinity for j in range(n)] for i in range(n)]
    for u, neighbors in enumerate(adjacency):
        for v in neighbors:
            distance[u][v] = 1
    for k in range(n):
        for i in range(n):
            for j in range(n):
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
    return distance


def cycle_cost(order, distance):
    return sum(distance[order[i]][order[(i + 1) % len(order)]] for i in range(len(order)))


def restricted_cycle_cost(order, active, distance):
    retained = [v for v in order if v in active]
    return cycle_cost(retained, distance)


def main():
    adjacency, bad_cycle = build_graph()
    distance = all_pairs_shortest_paths(adjacency)
    n = len(adjacency)

    # The displayed subdivided step-three cycle is a unit Hamiltonian cycle.
    assert len(bad_cycle) == n == 21
    assert len(set(bad_cycle)) == n
    assert cycle_cost(bad_cycle, distance) == n

    # All distances between distinct vertices are at least one, so any full
    # Hamiltonian cycle costs at least n.  Equality forces every tour edge to
    # be an edge of the unit graph.  Every subdivision vertex has graph
    # degree two, forcing both incident bad-path edges.  Those forced edges
    # already give every anchor degree two, so no good-cycle edge can occur.
    rare_vertices = range(ANCHORS, n)
    assert all(len(adjacency[v]) == 2 for v in rare_vertices)
    forced_anchor_neighbors = {i: set() for i in range(ANCHORS)}
    for v in rare_vertices:
        for u in adjacency[v]:
            if u < ANCHORS:
                forced_anchor_neighbors[u].add(v)
    assert all(len(forced_anchor_neighbors[i]) == 2 for i in range(ANCHORS))

    good_anchor_order = tuple(range(ANCHORS))
    bad_anchor_order = tuple((STEP * i) % ANCHORS for i in range(ANCHORS))
    assert cycle_cost(good_anchor_order, distance) == 7
    assert cycle_cost(bad_anchor_order, distance) == 21

    # On the event that all fourteen rare vertices are inactive, every
    # shortest full-support order costs 21 after shortcutting.  A posterior
    # tour can always use the good anchor cycle and attach each active rare
    # vertex to an adjacent anchor by a doubled unit edge.
    # In fact every rare-vertex realization has stopped cost exactly 21:
    # within each length-three bad segment, shortcut distances telescope to
    # three whether zero, one, or both private internal vertices survive.
    anchor_set = set(range(ANCHORS))
    for mask in range(1 << 14):
        active = anchor_set | {
            ANCHORS + bit for bit in range(14) if mask & (1 << bit)
        }
        assert restricted_cycle_cost(bad_cycle, active, distance) == 21
    stopped_lower = Q(21)
    posterior_upper = Q(7) + Q(2 * 14) * EPSILON
    assert 10 * stopped_lower > 29 * posterior_upper

    print("shortest-full-support-tour counterexample")
    print("  total vertices:", n, "(including depot)")
    print("  always-active anchors:", ANCHORS)
    print("  rare subdivision vertices:", 14, "epsilon:", EPSILON)
    print("  forced full-tour length:", n)
    print("  no-rare stopped cost:", cycle_cost(bad_anchor_order, distance))
    print("  posterior anchor-cycle cost:", cycle_cost(good_anchor_order, distance))
    print("  exact stopped cost:", stopped_lower)
    print("  exact posterior upper bound:", posterior_upper)
    print("  certified ratio lower bound:", stopped_lower / posterior_upper)


if __name__ == "__main__":
    main()
