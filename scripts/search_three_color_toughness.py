#!/usr/bin/env python3
"""Stress-test the count-conditioned toughness-defect inequality.

Condition on how many of three iid samples contain each client, and average
uniformly over the sample labels.  For every metric threshold this checks

  E[ u(A, f_B(C)) + u(A,B) ] <= 2 E[tau_t(A)],

where tau is the standard Hamiltonian toughness/component lower bound on the
number of threshold-long tour edges.  This is exploratory, not a proof.
"""

from __future__ import annotations

import argparse
import random
from itertools import product

from search_hazard_tree_ratio import named_metric, subset_tsp_costs
from search_three_color_threshold_symmetry import LABEL_SUBSETS, uncovered
from search_three_sample_symmetry import projected_mask


def component_count(vertices: int, adjacency_masks: list[int]) -> int:
    count = 0
    remaining = vertices
    while remaining:
        count += 1
        bit = remaining & -remaining
        remaining ^= bit
        frontier = bit
        while frontier:
            xbit = frontier & -frontier
            frontier ^= xbit
            x = xbit.bit_length() - 1
            reached = adjacency_masks[x] & remaining
            remaining ^= reached
            frontier |= reached
    return count


def toughness_defects(d, threshold):
    n = len(d) - 1
    root = n
    adjacency = [0] * (n + 1)
    for u in range(n + 1):
        for v in range(n + 1):
            if u != v and d[u][v] <= threshold:
                adjacency[u] |= 1 << v
    answer = [0] * (1 << n)
    for client_mask in range(1 << n):
        vertices = client_mask | (1 << root)
        components = component_count(vertices, adjacency)
        tau = components if components >= 2 else 0
        deletion = vertices
        while deletion:
            remaining = vertices & ~deletion
            deficit = component_count(remaining, adjacency) - deletion.bit_count()
            tau = max(tau, deficit)
            deletion = (deletion - 1) & vertices
        answer[client_mask] = tau
    return answer


def random_projection_table(d, rng):
    n = len(d) - 1
    root = n
    table = [[0] * (1 << n) for _ in range(1 << n)]
    for b in range(1 << n):
        centers = [v for v in range(n) if b >> v & 1] + [root]
        image = []
        for x in range(n):
            if b >> x & 1:
                image.append(x)
                continue
            best_distance = min(d[x][u] for u in centers)
            choices = [u for u in centers if d[x][u] == best_distance]
            image.append(rng.choice(choices))
        for c in range(1 << n):
            retained = 0
            for x in range(n):
                if c >> x & 1 and image[x] != root:
                    retained |= 1 << image[x]
            table[b][c] = retained
    return table


def check_metric(d, rng, count_trials, random_ties=False, one_two=False):
    n = len(d) - 1
    thresholds = sorted({d[i][j] for i in range(n + 1) for j in range(i)})
    intervals = [0.5 * (x + y) for x, y in zip([0.0] + thresholds, thresholds)]
    projected = (
        random_projection_table(d, rng)
        if random_ties
        else [[projected_mask(b, c, d) for c in range(1 << n)] for b in range(1 << n)]
    )
    exhaustive = n <= 4
    if one_two:
        counts_list = list(product((1, 2), repeat=n))
    elif exhaustive:
        counts_list = list(product(range(4), repeat=n))
    else:
        counts_list = [
            tuple(rng.randrange(4) for _ in range(n)) for _ in range(count_trials)
        ]
    worst = (-1e100, None)
    for threshold in intervals:
        tau = toughness_defects(d, threshold)
        binary = [[int(d[i][j] > threshold) for j in range(n + 1)] for i in range(n + 1)]
        tours = subset_tsp_costs(binary)
        assert all(tau[mask] <= tours[mask] for mask in range(1 << n))
        uncovered_table = [
            [uncovered(source, target, d, threshold) for target in range(1 << n)]
            for source in range(1 << n)
        ]
        for counts in counts_list:
            left = right = total = 0
            for labels in product(*(LABEL_SUBSETS[k] for k in counts)):
                samples = [0, 0, 0]
                for v, label_mask in enumerate(labels):
                    for label in range(3):
                        if label_mask >> label & 1:
                            samples[label] |= 1 << v
                a, b, c = samples
                left += uncovered_table[a][projected[b][c]] + uncovered_table[a][b]
                right += 2 * tau[a]
                total += 1
            gap = (left - right) / total
            if gap > worst[0]:
                worst = (gap, (counts, threshold, left, right, total, d))
            if left > right:
                return False, worst
    return True, worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--metrics", type=int, default=1000)
    parser.add_argument("--counts", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--random-ties", action="store_true")
    parser.add_argument("--one-two", action="store_true")
    parser.add_argument("--threshold-only", action="store_true")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    families = (
        "unit", "path", "cycle", "star", "binary", "random-tree",
        "graph12", "graph123", "euclidean", "multiscale-graph",
        "multiscale-complete", "threshold-graph",
    )
    best = (-1e100, None)
    for trial in range(args.metrics):
        family = "threshold-graph" if args.threshold_only else rng.choice(families)
        if family == "threshold-graph":
            density = rng.uniform(0.03, 0.97)
            d = [
                [
                    0.0 if i == j else (1.0 if rng.random() < density else 2.0)
                    for j in range(args.clients + 1)
                ]
                for i in range(args.clients + 1)
            ]
            for i in range(args.clients + 1):
                for j in range(i):
                    d[i][j] = d[j][i]
        else:
            d = named_metric(family, args.clients, rng)
        valid, worst = check_metric(d, rng, args.counts, args.random_ties, args.one_two)
        if worst[0] > best[0]:
            best = worst
            print("best", trial, family, best, flush=True)
        if not valid:
            print("COUNTEREXAMPLE", trial, family, worst)
            return
    print("no violation; best", best)


if __name__ == "__main__":
    main()
