#!/usr/bin/env python3
"""Search the count-conditioned three-color form of the corrected cap lemma.

For fixed per-vertex multiplicities among three iid samples A,B,C, labels are
uniform.  We test

    E_labels |{v in A: d(v,f_B(C) U {r}) > t}| <= E_labels cap_t(A),

where cap_t(S) counts threshold-degree-zero clients and threshold-degree-one
clients, except for a sole root-short client whose depot edge may be repeated.
This is a proof search diagnostic, not a theorem.
"""

from __future__ import annotations

import argparse
import random
from itertools import product

from search_hazard_tree_ratio import named_metric
from search_three_sample_symmetry import projected_mask
from search_three_color_threshold_symmetry import LABEL_SUBSETS, uncovered


def cap(mask, d, threshold):
    n = len(d) - 1
    root = n
    active = [v for v in range(n) if mask >> v & 1]
    answer = 0
    for v in active:
        degree = int(d[v][root] <= threshold) + sum(
            d[v][u] <= threshold for u in active if u != v
        )
        if degree == 0:
            answer += 1
        elif degree == 1 and not (len(active) == 1 and d[v][root] <= threshold):
            answer += 1
    return answer


def check(
    d,
    exhaustive=True,
    trials=1000,
    rng=None,
    client_only_projection=False,
):
    n = len(d) - 1
    thresholds = sorted({d[i][j] for i in range(n + 1) for j in range(i)})
    intervals = [0.5 * (x + y) for x, y in zip([0.0] + thresholds, thresholds)]
    projected = [
        [
            projected_mask(b, c, d, client_only_projection)
            for c in range(1 << n)
        ]
        for b in range(1 << n)
    ]
    counts_iter = product(range(4), repeat=n) if exhaustive else (
        tuple(rng.randrange(4) for _ in range(n)) for _ in range(trials)
    )
    counts_list = list(counts_iter)
    worst = (-1e100, None)
    for threshold in intervals:
        caps = [cap(mask, d, threshold) for mask in range(1 << n)]
        for counts in counts_list:
            left = right = total = 0
            for labels in product(*(LABEL_SUBSETS[k] for k in counts)):
                samples = [0, 0, 0]
                for v, label_mask in enumerate(labels):
                    for label in range(3):
                        if label_mask >> label & 1:
                            samples[label] |= 1 << v
                a, b, c = samples
                left += uncovered(a, projected[b][c], d, threshold)
                right += caps[a]
                total += 1
            normalized = (left - right) / total
            if normalized > worst[0]:
                worst = (normalized, (counts, threshold, left, right, total, d))
            if left > right:
                return False, worst
    return True, worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--metrics", type=int, default=100)
    parser.add_argument("--counts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--client-only-projection", action="store_true")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    families = ("unit", "path", "cycle", "star", "binary", "random-tree", "euclidean", "graph12")
    best = (-1e100, None)
    for trial in range(args.metrics):
        family = rng.choice(families)
        d = named_metric(family, args.clients, rng)
        exhaustive = args.clients <= 4
        valid, worst = check(
            d,
            exhaustive,
            args.counts,
            rng,
            client_only_projection=args.client_only_projection,
        )
        if worst[0] > best[0]:
            best = worst
        if not valid:
            print("VIOLATION", family, worst)
            return
        if trial and trial % 10 == 0:
            print("progress", trial, "best", best[0])
    print("no violation; best", best)


if __name__ == "__main__":
    main()
