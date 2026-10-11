#!/usr/bin/env python3
"""Exact audits for nonidentical two-terminal theta gap states.

For each of three independently activated weighted r--t paths, endpoints are
treated as required portals.  The script computes the exact best fixed-cut
theta-tree hazard, the connected closed posterior state, and the open
odd-parity posterior state from their gap formulas.  It searches for a
violation of a requested ratio without tuning the approximation algorithm.
"""

from __future__ import annotations

import argparse
import itertools
import random
from fractions import Fraction


def path_summary(lengths, probabilities):
    assert len(probabilities) == len(lengths) - 1
    prefix = [Fraction(0)]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    distribution = []
    cut_expectations = [Fraction(0)] * len(lengths)
    for mask in range(1 << len(probabilities)):
        probability = Fraction(1)
        marks = [0]
        for index, p in enumerate(probabilities):
            active = bool(mask & (1 << index))
            probability *= p if active else 1 - p
            if active:
                marks.append(index + 1)
        marks.append(len(lengths))
        gaps = [prefix[v] - prefix[u] for u, v in zip(marks, marks[1:])]
        maximum = max(gaps)
        distribution.append((maximum, probability))
        for cut in range(len(lengths)):
            left = max(mark for mark in marks if mark <= cut)
            right = min(mark for mark in marks if mark > cut)
            cut_expectations[cut] += probability * (prefix[right] - prefix[left])
    return sum(lengths, Fraction(0)), distribution, max(cut_expectations)


def evaluate(templates):
    summaries = [path_summary(lengths, probabilities) for lengths, probabilities in templates]
    lengths = [item[0] for item in summaries]
    total = sum(lengths, Fraction(0))
    h = [item[2] for item in summaries]
    fixed_hazard = total - sum(h, Fraction(0)) + min(h)

    closed = Fraction(0)
    opened = Fraction(0)
    for states in itertools.product(*(item[1] for item in summaries)):
        maxima = [state[0] for state in states]
        probability = states[0][1] * states[1][1] * states[2][1]
        retained = [lengths[i] - maxima[i] for i in range(3)]
        two_odd = min(
            total + lengths[k] - 2 * maxima[k] for k in range(3)
        )
        zero_odd_connected = min(
            2 * (lengths[i] + retained[j] + retained[k])
            for i in range(3)
            for j, k in [tuple(x for x in range(3) if x != i)]
        )
        closed += probability * min(two_odd, zero_odd_connected)

        one_odd = min(
            lengths[i] + 2 * retained[j] + 2 * retained[k]
            for i in range(3)
            for j, k in [tuple(x for x in range(3) if x != i)]
        )
        opened += probability * min(total, one_odd)

    return fixed_hazard, closed, opened


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--max-edges", type=int, default=4)
    parser.add_argument("--threshold", type=Fraction, default=Fraction(29, 20))
    parser.add_argument("--state", choices=("closed", "open"), default="closed")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    worst = None
    for _ in range(args.trials):
        templates = []
        for _ in range(3):
            edge_count = rng.randint(1, args.max_edges)
            lengths = tuple(
                Fraction(rng.randint(1, 20), rng.randint(1, 10))
                for _ in range(edge_count)
            )
            probabilities = tuple(
                Fraction(rng.randint(0, 10), 10) for _ in range(edge_count - 1)
            )
            templates.append((lengths, probabilities))
        hazard, closed, opened = evaluate(templates)
        benchmark = closed if args.state == "closed" else opened
        ratio = hazard / benchmark
        if worst is None or ratio > worst[0]:
            worst = (ratio, tuple(templates), hazard, benchmark)
        if ratio > args.threshold:
            print("COUNTEREXAMPLE", worst)
            return
    print("no violation; worst", worst)


if __name__ == "__main__":
    main()
