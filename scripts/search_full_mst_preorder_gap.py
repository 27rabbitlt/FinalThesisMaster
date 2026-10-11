#!/usr/bin/env python3
"""Search the best DFS preorder of a full-support MST against OPT_post.

Rare vertices can force a Hamiltonian cycle but cannot force an MST edge unless
they also create a genuinely short metric connection.  This makes full-MST
preorders a structurally different candidate from full-support TSP orders.
For small instances we enumerate every labelled tree, retain every MST tie,
enumerate all rooted child orders, and evaluate exact stopped-tour cost.

This is an exploratory falsifier, not a proof.
"""

from __future__ import annotations

import argparse
import itertools
import random
from functools import lru_cache

from search_closest_ancestor_gap import (
    all_rooted_trees,
    floyd,
    random_graph_metric,
    subset_tsp,
)


def subset_probabilities(probabilities: list[float]) -> list[float]:
    result = []
    for mask in range(1 << len(probabilities)):
        value = 1.0
        for index, probability in enumerate(probabilities):
            value *= probability if mask & (1 << index) else 1.0 - probability
        result.append(value)
    return result


def stopped_cost(order, probabilities, distance) -> float:
    value = 0.0
    for position, vertex in enumerate(order):
        probability = probabilities[vertex - 1]
        prefix_empty = 1.0
        for earlier in order[:position]:
            prefix_empty *= 1.0 - probabilities[earlier - 1]
        suffix_empty = 1.0
        for later in order[position + 1 :]:
            suffix_empty *= 1.0 - probabilities[later - 1]
        value += probability * (prefix_empty + suffix_empty) * distance[0][vertex]
        gap_empty = 1.0
        for later in order[position + 1 :]:
            value += (
                probability
                * probabilities[later - 1]
                * gap_empty
                * distance[vertex][later]
            )
            gap_empty *= 1.0 - probabilities[later - 1]
    return value


def tree_weight(parent: tuple[int, ...], distance) -> int:
    return sum(distance[vertex][parent[vertex]] for vertex in range(1, len(parent)))


@lru_cache(maxsize=None)
def dfs_preorders(parent: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    children = [[] for _ in parent]
    for vertex in range(1, len(parent)):
        children[parent[vertex]].append(vertex)

    @lru_cache(maxsize=None)
    def words(vertex: int) -> tuple[tuple[int, ...], ...]:
        if not children[vertex]:
            return ((vertex,),) if vertex else ((),)
        result = []
        for child_order in itertools.permutations(children[vertex]):
            choices = [words(child) for child in child_order]
            for child_words in itertools.product(*choices):
                suffix = tuple(item for word in child_words for item in word)
                result.append(((vertex,) if vertex else ()) + suffix)
        return tuple(set(result))

    return words(0)


def best_mst_preorder(distance, probabilities):
    trees = all_rooted_trees(len(distance))
    minimum_weight = min(tree_weight(parent, distance) for parent in trees)
    best = float("inf")
    best_record = None
    for parent in trees:
        if tree_weight(parent, distance) != minimum_weight:
            continue
        for order in dfs_preorders(parent):
            value = stopped_cost(order, probabilities, distance)
            if value < best:
                best = value
                best_record = (parent, order)
    assert best_record is not None
    return best, minimum_weight, best_record


def evaluate(distance, probabilities):
    scenario_probability = subset_probabilities(probabilities)
    posterior = sum(
        weight * cost
        for weight, cost in zip(scenario_probability, subset_tsp(distance))
    )
    value, mst_weight, record = best_mst_preorder(distance, probabilities)
    return posterior, value, mst_weight, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260907)
    arguments = parser.parse_args()
    rng = random.Random(arguments.seed)
    alphabet = (0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 0.9, 0.99)
    worst = None
    for trial in range(arguments.trials):
        distance = random_graph_metric(arguments.n, rng)
        profiles = [[value] * arguments.n for value in alphabet]
        profiles.extend(
            [rng.choice(alphabet) for _ in range(arguments.n)] for _ in range(5)
        )
        for probabilities in profiles:
            posterior, value, mst_weight, record = evaluate(distance, probabilities)
            ratio = value / posterior
            candidate = (ratio, trial, probabilities, distance, mst_weight, record)
            if worst is None or ratio > worst[0]:
                worst = candidate
                print(
                    "worst",
                    ratio,
                    "p",
                    probabilities,
                    "mst",
                    mst_weight,
                    "tree/order",
                    record,
                    flush=True,
                )
            if ratio > 2.9 + 1.0e-10:
                print("COUNTEREXAMPLE", candidate)
                return
    print("FINAL", worst)


if __name__ == "__main__":
    main()
