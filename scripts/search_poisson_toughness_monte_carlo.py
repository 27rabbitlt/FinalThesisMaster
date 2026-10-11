#!/usr/bin/env python3
"""Monte Carlo stress test for the repeated-atom fair-color toughness lemma."""

from __future__ import annotations

import argparse
import math
import random

from search_toughness_all_one import random_adjacency, toughness_values


def projection_table(adjacency: list[int], n: int, rng: random.Random) -> list[list[int]]:
    root = n
    table = [[0] * (1 << n) for _ in range(1 << n)]
    for b in range(1 << n):
        centers = [v for v in range(n) if b >> v & 1]
        images = [root] * n
        for x in range(n):
            if b >> x & 1:
                images[x] = x
                continue
            close = [u for u in centers if adjacency[x] >> u & 1]
            if adjacency[x] >> root & 1:
                close.append(root)
            images[x] = rng.choice(close or centers + [root])
        for c in range(1 << n):
            retained = 0
            bits = c
            while bits:
                bit = bits & -bits
                bits ^= bit
                image = images[bit.bit_length() - 1]
                if image != root:
                    retained |= 1 << image
            table[b][c] = retained
    return table


def missed_mass(a_counts, target, adjacency, n):
    root = n
    return sum(
        count
        for v, count in enumerate(a_counts)
        if count
        and not (adjacency[v] >> root & 1)
        and not (target & (adjacency[v] | (1 << v)))
    )


def random_multiplicities(n, rng):
    family = rng.randrange(6)
    if family == 0:
        return [rng.randrange(0, 4) for _ in range(n)]
    if family == 1:
        return [rng.randrange(0, 16) for _ in range(n)]
    if family == 2:
        common = rng.randrange(1, 25)
        return [common] * n
    if family == 3:
        return [rng.choice((0, 1, 1, 1, 2, 3, 10, 30)) for _ in range(n)]
    if family == 4:
        answer = [0] * n
        for v in rng.sample(range(n), rng.randrange(1, n + 1)):
            answer[v] = rng.randrange(1, 40)
        return answer
    return [max(0, round(math.exp(rng.uniform(-1, 4)))) for _ in range(n)]


def sample_gaps(adjacency, multiplicities, projection, tau, samples, rng):
    n = len(multiplicities)
    joint_values = []
    residual_values = []
    guide_values = []
    for _ in range(samples):
        counts = [[0] * n for _ in range(3)]
        for v, multiplicity in enumerate(multiplicities):
            for _ in range(multiplicity):
                counts[rng.randrange(3)][v] += 1
        masks = [sum(1 << v for v, count in enumerate(row) if count) for row in counts]
        a, b, c = masks
        core = missed_mass(counts[0], projection[b][c], adjacency, n)
        guide = missed_mass(counts[0], b, adjacency, n)
        isolated = sum(
            1
            for v in range(n)
            if a >> v & 1
            and not (adjacency[v] >> n & 1)
            and not ((a ^ (1 << v)) & adjacency[v])
        )
        joint_values.append(core + guide - 2 * tau[a])
        residual_values.append(core + isolated - 2 * tau[a])
        guide_values.append(guide - isolated)

    def statistics(values):
        mean = sum(values) / samples
        variance = sum((x - mean) ** 2 for x in values) / max(1, samples - 1)
        return mean, math.sqrt(variance / samples)

    return statistics(joint_values), statistics(residual_values), statistics(guide_values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--instances", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=307)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    best = (-1e100, None)
    for trial in range(args.instances):
        density = rng.uniform(0.02, 0.98)
        adjacency = random_adjacency(args.clients, density, rng)
        multiplicities = random_multiplicities(args.clients, rng)
        if not any(multiplicities):
            continue
        tau = toughness_values(adjacency, args.clients)
        projection = projection_table(adjacency, args.clients, rng)
        joint, residual, guide = sample_gaps(
            adjacency, multiplicities, projection, tau, args.samples, rng
        )
        mean, stderr = residual
        if stderr < 1e-12:
            continue
        score = mean / stderr
        if score > best[0]:
            best = (score, (mean, stderr, joint, guide, density, multiplicities, adjacency))
            print("best", trial, best, flush=True)
        if mean > 6 * stderr:
            print("LIKELY COUNTEREXAMPLE", best)
            return
    print("no likely violation", best)


if __name__ == "__main__":
    main()
