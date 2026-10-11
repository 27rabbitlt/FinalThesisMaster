#!/usr/bin/env python3
"""Search global color-orbit versions of the threshold toughness target.

The desired inequality averages over independent per-atom fair colors.  This
script checks whether it already holds on each orbit under one global color
permutation; a counterexample shows that a proof really needs local
recoloring rather than a global role swap.
"""

from __future__ import annotations

import argparse
import itertools
import random

from search_hazard_tree_ratio import named_metric
from search_three_color_threshold_symmetry import (
    toughness_defects,
    uncovered,
)
from search_three_sample_symmetry import projected_mask


def samples_from_coloring(colors):
    samples = [0, 0, 0]
    for v, color in enumerate(colors):
        samples[color] |= 1 << v
    return samples


def orbit_gap(colors, d, threshold, toughness, cyclic):
    samples = samples_from_coloring(colors)
    permutations = ((0, 1, 2), (1, 2, 0), (2, 0, 1)) if cyclic else tuple(
        itertools.permutations(range(3))
    )
    left = 0
    right = 0
    terms = []
    for alpha, beta, gamma in permutations:
        a, b, c = samples[alpha], samples[beta], samples[gamma]
        core = projected_mask(b, c, d)
        value = uncovered(a, core, d, threshold) + uncovered(a, b, d, threshold)
        left += value
        right += 2 * toughness[a]
        terms.append((alpha, beta, gamma, value, toughness[a], a, b, c, core))
    return left - right, left, right, terms


def check_metric(d):
    n = len(d) - 1
    distances = sorted({d[i][j] for i in range(n + 1) for j in range(i)})
    thresholds = [0.5 * (x + y) for x, y in zip([0.0] + distances, distances)]
    worst = {True: None, False: None}
    worst_fixed_b = None
    worst_fixed_b_pivotal = None
    for threshold in thresholds:
        toughness = toughness_defects(d, threshold)
        for colors in itertools.product(range(3), repeat=n):
            for cyclic in (True, False):
                result = orbit_gap(colors, d, threshold, toughness, cyclic)
                record = (*result[:3], colors, threshold, result[3])
                if worst[cyclic] is None or record[0] > worst[cyclic][0]:
                    worst[cyclic] = record
        full = (1 << n) - 1
        for b in range(1 << n):
            remainder = full ^ b
            submask = remainder
            left = 0
            right = 0
            pivotal_left = 0
            pivotal_right = 0
            cases = 0
            while True:
                a = submask
                c = remainder ^ a
                core = projected_mask(b, c, d)
                left += uncovered(a, core, d, threshold) + uncovered(
                    a, b, d, threshold
                )
                right += 2 * toughness[a]
                pivotal_left += uncovered(a, core, d, threshold) - uncovered(
                    a, b, d, threshold
                )
                isolated = sum(
                    1
                    for v in range(n)
                    if a & (1 << v)
                    and d[v][n] > threshold
                    and all(
                        not (a & (1 << u)) or d[v][u] > threshold
                        for u in range(n)
                        if u != v
                    )
                )
                pivotal_right += 2 * (toughness[a] - isolated)
                cases += 1
                if submask == 0:
                    break
                submask = (submask - 1) & remainder
            record = (left - right, left, right, b, threshold, cases)
            if worst_fixed_b is None or record[0] / cases > worst_fixed_b[0] / worst_fixed_b[-1]:
                worst_fixed_b = record
            pivotal_record = (
                pivotal_left - pivotal_right,
                pivotal_left,
                pivotal_right,
                b,
                threshold,
                cases,
            )
            if (
                worst_fixed_b_pivotal is None
                or pivotal_record[0] / cases
                > worst_fixed_b_pivotal[0] / worst_fixed_b_pivotal[-1]
            ):
                worst_fixed_b_pivotal = pivotal_record
    return worst, worst_fixed_b, worst_fixed_b_pivotal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--metrics", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    families = (
        "unit", "path", "cycle", "star", "binary", "random-tree",
        "graph12", "graph123", "client-graph-19", "euclidean",
        "multiscale-graph", "multiscale-complete",
    )
    global_worst = {True: None, False: None}
    global_worst_fixed_b = None
    global_worst_fixed_b_pivotal = None
    for index in range(args.metrics):
        family = families[index % len(families)]
        d = named_metric(family, args.clients, rng)
        worst, worst_fixed_b, worst_fixed_b_pivotal = check_metric(d)
        for cyclic in (True, False):
            record = (*worst[cyclic], family, d)
            if global_worst[cyclic] is None or record[0] > global_worst[cyclic][0]:
                global_worst[cyclic] = record
        record = (*worst_fixed_b, family, d)
        if (
            global_worst_fixed_b is None
            or record[0] / record[5] > global_worst_fixed_b[0] / global_worst_fixed_b[5]
        ):
            global_worst_fixed_b = record
        record = (*worst_fixed_b_pivotal, family, d)
        if (
            global_worst_fixed_b_pivotal is None
            or record[0] / record[5]
            > global_worst_fixed_b_pivotal[0] / global_worst_fixed_b_pivotal[5]
        ):
            global_worst_fixed_b_pivotal = record
    print("cyclic worst=", global_worst[True])
    print("six-permutation worst=", global_worst[False])
    print("fixed-B fair A/C split worst=", global_worst_fixed_b)
    print("fixed-B pivotal vs toughness residual worst=", global_worst_fixed_b_pivotal)


if __name__ == "__main__":
    main()
