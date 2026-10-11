#!/usr/bin/env python3
"""Fast random-graph search for the all-one conditional toughness lemma.

Every client appears in exactly one of A,B,C, so the conditional average is
over all 3^n colorings.  Threshold metrics have distances 1 on graph edges
and 2 otherwise.  A random valid local tie rule is chosen for every guide set.
"""

from __future__ import annotations

import argparse
import random

from search_three_color_toughness import component_count


def toughness_values(adjacency: list[int], n: int) -> list[int]:
    root = n
    out = [0] * (1 << n)
    for mask in range(1 << n):
        vertices = mask | (1 << root)
        components = component_count(vertices, adjacency)
        value = components if components >= 2 else 0
        deletion = vertices
        while deletion:
            value = max(
                value,
                component_count(vertices & ~deletion, adjacency) - deletion.bit_count(),
            )
            deletion = (deletion - 1) & vertices
        out[mask] = value
    return out


def projection_table(adjacency: list[int], n: int, rng: random.Random) -> list[list[int]]:
    root = n
    table = [[0] * (1 << n) for _ in range(1 << n)]
    for b in range(1 << n):
        images = [root] * n
        centers = [u for u in range(n) if b >> u & 1]
        for x in range(n):
            if b >> x & 1:
                images[x] = x
                continue
            distance_one = [u for u in centers if adjacency[x] >> u & 1]
            if adjacency[x] >> root & 1:
                distance_one.append(root)
            choices = distance_one or centers + [root]
            images[x] = rng.choice(choices)
        for c in range(1 << n):
            retained = 0
            bits = c
            while bits:
                xbit = bits & -bits
                bits ^= xbit
                image = images[xbit.bit_length() - 1]
                if image != root:
                    retained |= 1 << image
            table[b][c] = retained
    return table


def uncovered(source: int, target: int, adjacency: list[int], n: int) -> int:
    root = n
    total = 0
    bits = source
    while bits:
        vbit = bits & -bits
        bits ^= vbit
        v = vbit.bit_length() - 1
        if not (adjacency[v] >> root & 1) and not (target & (adjacency[v] | vbit)):
            total += 1
    return total


def gap(adjacency: list[int], n: int, rng: random.Random) -> tuple[int, int, int]:
    tau = toughness_values(adjacency, n)
    projection = projection_table(adjacency, n, rng)
    full = (1 << n) - 1
    left = right = 0
    # base-3 coloring, encoded by choosing A then B from the complement.
    for a in range(1 << n):
        rest = full ^ a
        b = rest
        while True:
            c = rest ^ b
            left += uncovered(a, projection[b][c], adjacency, n)
            left += uncovered(a, b, adjacency, n)
            right += 2 * tau[a]
            if b == 0:
                break
            b = (b - 1) & rest
    return left - right, left, right


def random_adjacency(n: int, density: float, rng: random.Random) -> list[int]:
    adjacency = [0] * (n + 1)
    for u in range(n + 1):
        for v in range(u):
            if rng.random() < density:
                adjacency[u] |= 1 << v
                adjacency[v] |= 1 << u
    return adjacency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=211)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    best = (-10**100, None)
    for trial in range(args.trials):
        density = rng.uniform(0.02, 0.98)
        adjacency = random_adjacency(args.clients, density, rng)
        result = gap(adjacency, args.clients, rng)
        if result[0] > best[0]:
            best = (result[0], (density, result, adjacency))
            print("best", trial, best, flush=True)
        if result[0] > 0:
            print("COUNTEREXAMPLE", best)
            return
    print("no violation", best)


if __name__ == "__main__":
    main()
