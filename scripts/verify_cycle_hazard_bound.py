#!/usr/bin/env python3
"""Exact rational checks for the length-biased cycle-breaking lemma."""

from fractions import Fraction
from itertools import combinations


def gap_lengths(edge_lengths: list[Fraction], marks: set[int]) -> list[Fraction]:
    """Cycle vertices are 0,...,m-1; edge i joins i to i+1 mod m."""
    m = len(edge_lengths)
    ordered = sorted(marks)
    gaps: list[Fraction] = []
    for index, start in enumerate(ordered):
        stop = ordered[(index + 1) % len(ordered)]
        length = Fraction(0)
        cursor = start
        while cursor != stop or length == 0:
            length += edge_lengths[cursor]
            cursor = (cursor + 1) % m
            if cursor == stop:
                break
        gaps.append(length)
    return gaps


def check_cycle(edge_lengths: list[Fraction]) -> int:
    m = len(edge_lengths)
    total = sum(edge_lengths)
    if total == 0:
        # The induced metric is identically zero, including for the empty and
        # singleton active sets, so both sides of the lemma vanish.
        return 2 ** (m - 1)
    checked = 0
    # Vertex 0 is the depot; every subset of the remaining vertices is tested.
    for cardinality in range(m):
        for active in combinations(range(1, m), cardinality):
            gaps = gap_lengths(edge_lengths, {0, *active})
            average_subtree = total - sum(gap * gap for gap in gaps) / total
            largest_gap = max(gaps)
            posterior_tour = min(total, 2 * (total - largest_gap))
            assert average_subtree <= posterior_tour
            checked += 1
    return checked


def main() -> None:
    cycles = [
        [Fraction(1)] * 7,
        [Fraction(1, 3), Fraction(2, 5), Fraction(7, 4), Fraction(1, 9)],
        [Fraction(1, 100), Fraction(11, 7), Fraction(3, 2), Fraction(5, 6), Fraction(9, 8)],
        [Fraction(0), Fraction(2, 3), Fraction(0), Fraction(5, 4)],
        [Fraction(0)] * 3,
    ]
    checks = sum(check_cycle(cycle) for cycle in cycles)
    print(f"exact pointwise checks passed for {checks} active sets on {len(cycles)} cycles")


if __name__ == "__main__":
    main()
