#!/usr/bin/env python3
"""Exact audit of the occupancy-aggregation fragmentation obstruction.

The metric is the unit nine-cycle with vertex 0 as the depot.  The eight
clients have probability 1/5 and are split into the two alternating parity
blocks.  Both aggregate blocks have the same occupancy, at least 4/9,
although every atom has occupancy below 4/9.

For every realization the checker computes:

* the canonical posterior-optimal depot tour;
* the cheapest tour in which each aggregate block is contiguous; and
* the run-fragmentation upper bound from (OA6).

All arithmetic is exact.
"""

from fractions import Fraction
from itertools import permutations


N = 9
DEPOT = 0
CLIENTS = tuple(range(1, N))
PROBABILITY = Fraction(1, 5)
GROUPS = (
    frozenset((2, 4, 6, 8)),
    frozenset((1, 3, 5, 7)),
)


def distance(left: int, right: int) -> int:
    gap = abs(left - right)
    return min(gap, N - gap)


def tour_cost(order: tuple[int, ...]) -> int:
    word = (DEPOT,) + order + (DEPOT,)
    return sum(distance(left, right) for left, right in zip(word, word[1:]))


def optimum(active: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    if not active:
        return 0, ()
    candidates = ((tour_cost(order), order) for order in permutations(active))
    return min(candidates)


def contiguous_optimum(active: tuple[int, ...]) -> int:
    pieces = [tuple(vertex for vertex in active if vertex in group) for group in GROUPS]
    if not pieces[0] or not pieces[1]:
        return optimum(active)[0]
    best = None
    for first, second in ((0, 1), (1, 0)):
        for first_order in permutations(pieces[first]):
            for second_order in permutations(pieces[second]):
                value = tour_cost(first_order + second_order)
                best = value if best is None else min(best, value)
    assert best is not None
    return best


def run_counts(order: tuple[int, ...]) -> tuple[int, int]:
    labels = [0 if vertex in GROUPS[0] else 1 for vertex in order]
    return tuple(
        sum(
            1
            for index, label in enumerate(labels)
            if label == group and (index == 0 or labels[index - 1] != group)
        )
        for group in range(2)
    )


def realization_probability(active_count: int) -> Fraction:
    return (
        PROBABILITY**active_count
        * (1 - PROBABILITY) ** (len(CLIENTS) - active_count)
    )


def main() -> None:
    diameters = tuple(
        max(distance(left, right) for left in group for right in group)
        for group in GROUPS
    )
    posterior = Fraction(0)
    contiguous = Fraction(0)
    fragmentation_bound = Fraction(0)

    for mask in range(1 << len(CLIENTS)):
        active = tuple(
            vertex
            for index, vertex in enumerate(CLIENTS)
            if mask & (1 << index)
        )
        probability = realization_probability(len(active))
        optimal_cost, optimal_order = optimum(active)
        block_cost = contiguous_optimum(active)
        runs = run_counts(optimal_order)
        repair = 2 * sum(
            max(0, run_count - 1) * diameter
            for run_count, diameter in zip(runs, diameters)
        )
        assert block_cost <= optimal_cost + repair
        posterior += probability * optimal_cost
        contiguous += probability * block_cost
        fragmentation_bound += probability * repair

    occupancies = tuple(
        1 - (1 - PROBABILITY) ** len(group) for group in GROUPS
    )
    assert all(PROBABILITY < Fraction(4, 9) <= q for q in occupancies)
    assert occupancies[0] == occupancies[1] == Fraction(369, 625)
    assert contiguous > posterior
    assert contiguous - posterior <= fragmentation_bound

    full_active = tuple(CLIENTS)
    full_optimum = optimum(full_active)[0]
    full_contiguous = contiguous_optimum(full_active)
    assert full_optimum == 9
    assert full_contiguous == 16

    print(f"block occupancies = {occupancies}")
    print(f"posterior expectation = {posterior}")
    print(f"contiguous expectation = {contiguous}")
    print(f"strict aggregation loss = {contiguous - posterior}")
    print(f"fragmentation certificate = {fragmentation_bound}")
    print(f"full-realization costs = {full_optimum}, {full_contiguous}")
    print("all exact checks passed")


if __name__ == "__main__":
    main()
