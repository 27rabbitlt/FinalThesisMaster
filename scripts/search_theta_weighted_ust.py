#!/usr/bin/env python3
"""Test inverse-length weighted spanning trees on theta graph metrics.

For a weighted cycle, sampling a spanning tree with weight proportional to
the product of inverse edge lengths deletes an edge proportional to its
length, exactly the pointwise cycle lemma.  This script tests whether the
same coupled cycle-breaking rule satisfies

    E_T[ length of the minimal A-union-root subtree of T ] <= tsp_G(A)

for every marked set A on small weighted theta graphs.  Enumeration is exact
with Fraction arithmetic.  It is a falsification search, not a proof.
"""

from __future__ import annotations

import argparse
import random
from fractions import Fraction
from itertools import combinations, permutations

from scipy.optimize import linprog


def theta_graph(
    path_lengths: tuple[tuple[Fraction, ...], ...]
) -> tuple[int, list[tuple[int, int, Fraction]], tuple[tuple[int, ...], ...]]:
    """Return root terminal 0 and the edge list of three 0--1 paths."""
    edges: list[tuple[int, int, Fraction]] = []
    path_edges: list[tuple[int, ...]] = []
    next_vertex = 2
    for lengths in path_lengths:
        vertices = [0]
        vertices.extend(range(next_vertex, next_vertex + len(lengths) - 1))
        next_vertex += len(lengths) - 1
        vertices.append(1)
        start = len(edges)
        edges.extend(
            (u, v, length)
            for u, v, length in zip(vertices, vertices[1:], lengths)
        )
        path_edges.append(tuple(range(start, len(edges))))
    return next_vertex, edges, tuple(path_edges)


def is_tree(vertex_count: int, edges, chosen: tuple[int, ...]) -> bool:
    parent = list(range(vertex_count))

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for index in chosen:
        u, v, _ = edges[index]
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        parent[ru] = rv
    return len(chosen) == vertex_count - 1 and len({find(v) for v in range(vertex_count)}) == 1


def all_trees(vertex_count: int, edges):
    answer = []
    for chosen in combinations(range(len(edges)), vertex_count - 1):
        if is_tree(vertex_count, edges, chosen):
            weight = Fraction(1)
            for index in chosen:
                weight /= edges[index][2]
            answer.append((chosen, weight))
    total = sum(weight for _, weight in answer)
    return [(chosen, weight / total) for chosen, weight in answer]


def shortest_backbone_trees(vertex_count: int, edges, path_edges, path_lengths):
    """Keep one shortest terminal path and length-bias cuts on the others."""
    shortest = min(range(3), key=lambda i: (sum(path_lengths[i]), i))
    required = set(path_edges[shortest])
    answer = []
    for chosen in combinations(range(len(edges)), vertex_count - 1):
        if required <= set(chosen) and is_tree(vertex_count, edges, chosen):
            deleted = set(range(len(edges))) - set(chosen)
            weight = Fraction(1)
            for index in deleted:
                weight *= edges[index][2]
            answer.append((chosen, weight))
    total = sum(weight for _, weight in answer)
    return [(chosen, weight / total) for chosen, weight in answer]


def quantile_coupled_shortest_backbone_trees(
    vertex_count: int, edges, path_edges, path_lengths, reverse: bool = False
):
    """Keep a shortest path and cut both ears at one common length quantile."""
    shortest = min(range(3), key=lambda i: (sum(path_lengths[i]), i))
    ears = [i for i in range(3) if i != shortest]

    def intervals(path: int, reflected: bool):
        total = sum(path_lengths[path])
        left = Fraction(0)
        answer = []
        for index, length in zip(path_edges[path], path_lengths[path]):
            right = left + length / total
            if reflected:
                answer.append((index, 1 - right, 1 - left))
            else:
                answer.append((index, left, right))
            left = right
        return answer

    first = intervals(ears[0], False)
    second = intervals(ears[1], reverse)
    probabilities = {}
    all_indices = set(range(len(edges)))
    for edge_i, left_i, right_i in first:
        for edge_j, left_j, right_j in second:
            overlap = max(Fraction(0), min(right_i, right_j) - max(left_i, left_j))
            if overlap:
                chosen = tuple(sorted(all_indices - {edge_i, edge_j}))
                assert is_tree(vertex_count, edges, chosen)
                probabilities[chosen] = probabilities.get(chosen, Fraction(0)) + overlap
    assert sum(probabilities.values()) == 1
    return list(probabilities.items())


def metric_closure(vertex_count: int, edges):
    infinity = Fraction(10**30)
    distance = [
        [Fraction(0) if u == v else infinity for v in range(vertex_count)]
        for u in range(vertex_count)
    ]
    for u, v, length in edges:
        distance[u][v] = distance[v][u] = min(distance[u][v], length)
    for middle in range(vertex_count):
        for u in range(vertex_count):
            for v in range(vertex_count):
                distance[u][v] = min(
                    distance[u][v], distance[u][middle] + distance[middle][v]
                )
    return distance


def tsp(mask: int, vertex_count: int, distance) -> Fraction:
    clients = tuple(v for v in range(1, vertex_count) if mask & (1 << (v - 1)))
    if not clients:
        return Fraction(0)
    return min(
        sum(distance[u][v] for u, v in zip((0,) + order, order + (0,)))
        for order in permutations(clients)
    )


def subtree_cost(mask: int, vertex_count: int, edges, chosen) -> Fraction:
    adjacency = [[] for _ in range(vertex_count)]
    for index in chosen:
        u, v, length = edges[index]
        adjacency[u].append((v, length))
        adjacency[v].append((u, length))

    def visit(v: int, parent: int) -> tuple[bool, Fraction]:
        required = v != 0 and bool(mask & (1 << (v - 1)))
        value = Fraction(0)
        for child, length in adjacency[v]:
            if child == parent:
                continue
            child_required, child_value = visit(child, v)
            if child_required:
                required = True
                value += length
            value += child_value
        return required, value

    return visit(0, -1)[1]


def audit(path_lengths, shortest_backbone: bool = False) -> tuple[Fraction, int, Fraction, Fraction] | None:
    vertex_count, edges, path_edges = theta_graph(path_lengths)
    trees = (
        shortest_backbone_trees(
            vertex_count, edges, path_edges, path_lengths
        )
        if shortest_backbone
        else all_trees(vertex_count, edges)
    )
    distance = metric_closure(vertex_count, edges)
    worst = None
    for mask in range(1 << (vertex_count - 1)):
        expected = sum(
            probability * subtree_cost(mask, vertex_count, edges, chosen)
            for chosen, probability in trees
        )
        posterior = tsp(mask, vertex_count, distance)
        gap = expected - posterior
        if worst is None or gap > worst[0]:
            worst = (gap, mask, expected, posterior)
    return worst if worst and worst[0] > 0 else None


def max_pair_stretch_for_distribution(path_lengths, trees):
    vertex_count, edges, _ = theta_graph(path_lengths)
    distance = metric_closure(vertex_count, edges)
    answer = (0.0, None)
    for u in range(vertex_count):
        for v in range(u):
            expected = Fraction(0)
            for chosen, probability in trees:
                td = metric_closure(vertex_count, [edges[index] for index in chosen])
                expected += probability * td[u][v]
            ratio = float(expected / distance[u][v])
            if ratio > answer[0]:
                answer = (ratio, (u, v, expected, distance[u][v]))
    return answer


def optimal_pointwise_ratio(path_lengths, restrict_shortest: bool = False):
    """Minimize rho with E tree-subtree(A) <= rho*tsp(A) for every A."""
    vertex_count, edges, path_edges = theta_graph(path_lengths)
    raw_trees = all_trees(vertex_count, edges)
    trees = [chosen for chosen, _ in raw_trees]
    if restrict_shortest:
        shortest = min(range(3), key=lambda i: (sum(path_lengths[i]), i))
        required = set(path_edges[shortest])
        trees = [chosen for chosen in trees if required <= set(chosen)]
    distance = metric_closure(vertex_count, edges)
    rows = []
    bounds = []
    for mask in range(1, 1 << (vertex_count - 1)):
        posterior = tsp(mask, vertex_count, distance)
        rows.append(
            [
                *[
                    float(subtree_cost(mask, vertex_count, edges, chosen))
                    for chosen in trees
                ],
                -float(posterior),
            ]
        )
        bounds.append(0.0)
    objective = [0.0] * len(trees) + [1.0]
    equality = [[1.0] * len(trees) + [0.0]]
    result = linprog(
        objective,
        A_ub=rows,
        b_ub=bounds,
        A_eq=equality,
        b_eq=[1.0],
        bounds=[(0.0, None)] * len(trees) + [(0.0, None)],
        method="highs",
    )
    assert result.success
    return result.fun, tuple(result.x[:-1]), trees


def optimal_intact_path_mixture_ratio(path_lengths):
    """Optimize only the intact-path mixture; cuts are length-biased iid."""
    vertex_count, edges, path_edges = theta_graph(path_lengths)
    distance = metric_closure(vertex_count, edges)
    columns = []
    for intact in range(3):
        required = set(path_edges[intact])
        candidates = []
        for chosen, _ in all_trees(vertex_count, edges):
            if required <= set(chosen):
                deleted = set(range(len(edges))) - set(chosen)
                weight = Fraction(1)
                for index in deleted:
                    weight *= edges[index][2]
                candidates.append((chosen, weight))
        total = sum(weight for _, weight in candidates)
        columns.append([(chosen, weight / total) for chosen, weight in candidates])

    rows = []
    bounds = []
    for mask in range(1, 1 << (vertex_count - 1)):
        posterior = tsp(mask, vertex_count, distance)
        rows.append(
            [
                *[
                    float(sum(prob * subtree_cost(mask, vertex_count, edges, tree)
                              for tree, prob in distribution))
                    for distribution in columns
                ],
                -float(posterior),
            ]
        )
        bounds.append(0.0)
    result = linprog(
        [0.0, 0.0, 0.0, 1.0],
        A_ub=rows,
        b_ub=bounds,
        A_eq=[[1.0, 1.0, 1.0, 0.0]],
        b_eq=[1.0],
        bounds=[(0.0, None)] * 4,
        method="highs",
    )
    assert result.success
    return result.fun, tuple(result.x[:3])


def optimal_deterministic_pointwise_ratio(path_lengths):
    """Best worst-case marked-set ratio among graph spanning trees."""
    vertex_count, edges, _ = theta_graph(path_lengths)
    distance = metric_closure(vertex_count, edges)
    best = None
    for chosen, _ in all_trees(vertex_count, edges):
        ratio = max(
            float(subtree_cost(mask, vertex_count, edges, chosen) /
                  tsp(mask, vertex_count, distance))
            for mask in range(1, 1 << (vertex_count - 1))
        )
        if best is None or ratio < best[0]:
            best = (ratio, chosen)
    assert best is not None
    return best


def optimal_pair_stretch(path_lengths):
    """Minimize worst expected pair stretch over graph spanning trees."""
    vertex_count, edges, _ = theta_graph(path_lengths)
    trees = [chosen for chosen, _ in all_trees(vertex_count, edges)]
    distance = metric_closure(vertex_count, edges)

    tree_distances = []
    for chosen in trees:
        chosen_edges = [edges[index] for index in chosen]
        tree_distances.append(metric_closure(vertex_count, chosen_edges))

    rows = []
    for u in range(vertex_count):
        for v in range(u):
            rows.append(
                [
                    *[float(td[u][v]) for td in tree_distances],
                    -float(distance[u][v]),
                ]
            )
    result = linprog(
        [0.0] * len(trees) + [1.0],
        A_ub=rows,
        b_ub=[0.0] * len(rows),
        A_eq=[[1.0] * len(trees) + [0.0]],
        b_eq=[1.0],
        bounds=[(0.0, None)] * (len(trees) + 1),
        method="highs",
    )
    assert result.success
    return result.fun, tuple(result.x[:-1]), trees


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--max-edges-per-path", type=int, default=3)
    parser.add_argument("--shortest-backbone", action="store_true")
    parser.add_argument("--optimal-pointwise", action="store_true")
    parser.add_argument("--restrict-shortest", action="store_true")
    parser.add_argument("--intact-path-mixture", action="store_true")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--deterministic-pointwise", action="store_true")
    parser.add_argument("--pair-stretch", action="store_true")
    parser.add_argument("--quantile-coupling", action="store_true")
    parser.add_argument("--reverse-quantile", action="store_true")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    for trial in range(args.trials):
        edge_counts = tuple(
            rng.randint(1, args.max_edges_per_path) for _ in range(3)
        )
        path_lengths = tuple(
            tuple(Fraction(rng.randint(1, 20), rng.randint(1, 10)) for _ in range(count))
            for count in edge_counts
        )
        if args.optimal_pointwise:
            ratio, distribution, trees = optimal_pointwise_ratio(
                path_lengths, args.restrict_shortest
            )
            if ratio > args.threshold + 1e-8:
                support = [
                    (trees[i], value)
                    for i, value in enumerate(distribution)
                    if value > 1e-8
                ]
                print("POINTWISE RATIO ABOVE ONE", path_lengths, ratio, support)
                return
            continue
        if args.intact_path_mixture:
            ratio, mixture = optimal_intact_path_mixture_ratio(path_lengths)
            if ratio > args.threshold + 1e-8:
                print("INTACT-MIXTURE RATIO ABOVE ONE", path_lengths, ratio, mixture)
                return
            continue
        if args.deterministic_pointwise:
            ratio, tree = optimal_deterministic_pointwise_ratio(path_lengths)
            if ratio > args.threshold + 1e-8:
                print("DETERMINISTIC RATIO ABOVE THRESHOLD", path_lengths, ratio, tree)
                return
            continue
        if args.pair_stretch:
            ratio, distribution, trees = optimal_pair_stretch(path_lengths)
            if ratio > args.threshold + 1e-8:
                support = [
                    (trees[i], value)
                    for i, value in enumerate(distribution)
                    if value > 1e-8
                ]
                print("PAIR STRETCH ABOVE THRESHOLD", path_lengths, ratio, support)
                return
            continue
        if args.quantile_coupling:
            vertex_count, edges, path_edges = theta_graph(path_lengths)
            trees = quantile_coupled_shortest_backbone_trees(
                vertex_count,
                edges,
                path_edges,
                path_lengths,
                args.reverse_quantile,
            )
            ratio, pair = max_pair_stretch_for_distribution(path_lengths, trees)
            if ratio > args.threshold + 1e-8:
                print("QUANTILE PAIR STRETCH ABOVE THRESHOLD", path_lengths, ratio, pair)
                return
            continue
        witness = audit(path_lengths, args.shortest_backbone)
        if witness is not None:
            print("COUNTEREXAMPLE", path_lengths, witness)
            return
    print(f"no violation in {args.trials} exact random theta audits")


if __name__ == "__main__":
    main()
