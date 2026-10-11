#!/usr/bin/env python3
"""Exact counterexample to inverse-length weighted UST cycle breaking.

On a weighted cycle, drawing a spanning tree with probability proportional
to the product of inverse edge lengths is exactly length-proportional edge
deletion, and its expected marked-subtree length is at most the metric TSP
cost pointwise.  This verifier shows that the same natural rule already
fails on a theta graph.

All arithmetic is rational.  The theta graph has terminals 0 (the depot)
and 1, and three internally disjoint two-edge paths.  The short path has
total length one; the other two have common total length L.  The marked
client set is {1, 2}, where vertex 2 is the internal vertex of the short
path.  The weighted-UST ratio tends to 3/2 as L tends to infinity, and
L=100 already exceeds 29/20 exactly.  The same instance also proves that
no spanning-tree coupling which preserves the successful length-proportional
deletion marginal on each of the two constituent cycles can reach 29/20.
"""

from fractions import Fraction

from search_theta_weighted_ust import (
    all_trees,
    metric_closure,
    subtree_cost,
    theta_graph,
    tsp,
)


PATH_LENGTHS = (
    (Fraction(1, 2), Fraction(1, 2)),
    (Fraction(50), Fraction(50)),
    (Fraction(50), Fraction(50)),
)
MARKED_MASK = 3  # clients 1 and 2
LONG_PATH_LENGTH = Fraction(100)


def main() -> None:
    vertex_count, edges, _ = theta_graph(PATH_LENGTHS)
    trees = all_trees(vertex_count, edges)
    distance = metric_closure(vertex_count, edges)

    expected_subtree = sum(
        probability * subtree_cost(MARKED_MASK, vertex_count, edges, chosen)
        for chosen, probability in trees
    )
    posterior = tsp(MARKED_MASK, vertex_count, distance)
    gap = expected_subtree - posterior

    assert len(trees) == 12
    assert sum(probability for _, probability in trees) == 1
    assert expected_subtree == Fraction(301, 102)
    assert posterior == Fraction(2)
    assert gap == Fraction(97, 102)
    assert expected_subtree / posterior == Fraction(301, 204)
    assert expected_subtree / posterior > Fraction(29, 20)

    # If each cycle consisting of the short path and one long path retains
    # its cycle-lemma marginal, it deletes from the short path with
    # probability 1/(L+1).  The two such events must be disjoint in any
    # coupling supported on theta spanning trees.  Thus the short path is
    # intact with probability 1-2/(L+1), and each long path is intact with
    # probability 1/(L+1).  For the marked set above the corresponding costs
    # are 1, L+1/2, and L+1/2, independent of cut positions.
    cycle_marginal_expected = (
        (1 - Fraction(2, LONG_PATH_LENGTH + 1))
        + Fraction(2, LONG_PATH_LENGTH + 1)
        * (LONG_PATH_LENGTH + Fraction(1, 2))
    )
    cycle_marginal_ratio = cycle_marginal_expected / posterior
    assert cycle_marginal_expected == Fraction(300, 101)
    assert cycle_marginal_ratio == Fraction(150, 101)
    assert cycle_marginal_ratio > Fraction(29, 20)

    print(f"number of spanning trees: {len(trees)}")
    print(f"expected marked-subtree length: {expected_subtree}")
    print(f"posterior metric-TSP cost: {posterior}")
    print(f"strict violation: {gap} = {float(gap):.12f}")
    print(f"ratio: {expected_subtree / posterior} = "
          f"{float(expected_subtree / posterior):.12f}")
    print(f"best possible with both cycle deletion marginals: "
          f"{cycle_marginal_expected}")
    print(f"cycle-marginal ratio: {cycle_marginal_ratio} = "
          f"{float(cycle_marginal_ratio):.12f}")


if __name__ == "__main__":
    main()
