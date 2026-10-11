#!/usr/bin/env python3
"""Exact lower bound for reducing a theta metric to one-edge-deleted cacti.

Take three internally disjoint r--t paths of m unit edges.  Any distribution
over cacti obtained by deleting one theta edge has worst expected pair stretch
at least 1 + 2(m-1)/(3m).  For m=4 this is 3/2, already above 29/20.

The proof is the summation of the endpoint-pair constraint for every edge;
this script records the exact finite arithmetic and independently checks all
post-deletion endpoint distances in the graph metric.
"""

from fractions import Fraction

from search_theta_weighted_ust import metric_closure, theta_graph


def main() -> None:
    m = 4
    paths = tuple(tuple(Fraction(1) for _ in range(m)) for _ in range(3))
    vertex_count, edges, _ = theta_graph(paths)
    original = metric_closure(vertex_count, edges)

    for omitted, (u, v, _) in enumerate(edges):
        reduced = metric_closure(
            vertex_count,
            [edge for index, edge in enumerate(edges) if index != omitted],
        )
        assert original[u][v] == 1
        assert reduced[u][v] == 2 * m - 1

    edge_count = len(edges)
    lower_bound = 1 + Fraction(2 * (m - 1), edge_count)
    assert edge_count == 3 * m == 12
    assert lower_bound == Fraction(3, 2)
    assert lower_bound > Fraction(29, 20)

    print(f"theta edges: {edge_count}")
    print(f"forced worst expected pair stretch: at least {lower_bound}")
    print(f"target: 29/20; exact excess: {lower_bound - Fraction(29, 20)}")


if __name__ == "__main__":
    main()
