#!/usr/bin/env python3
"""Explore recursive minimum-insertion-gap posterior segment riffles.

For a state (S,a,b), draw a nonempty product-p guide B.  Choose a shortest
a--b path through B.  Assign every residual vertex to a path gap minimizing
its one-vertex insertion excess, recursively solve those gap states, and
splice them around the guide anchors.  Tied guide paths are chosen to minimize
the exact two-terminal expected stopped cost of the expanded segment.

The recursion propagates internal cost and first/last-active laws, which are
an exact sufficient statistic for concatenating independent segment orders.
Floating point is used: this is an exploratory search, not a certificate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
import math
import random

from search_recursive_self_thinning import all_subset_optima, posterior, random_graph_metric


@dataclass(frozen=True)
class Summary:
    internal: float
    first: tuple[float, ...]
    last: tuple[float, ...]
    empty: float


def solve_instance(d: list[list[float]], p: list[float]) -> tuple[float, float, float]:
    n = len(p)
    full = (1 << n) - 1
    zero_vector = (0.0,) * n

    empty = [1.0] * (1 << n)
    for mask in range(1, 1 << n):
        bit = mask & -mask
        v = bit.bit_length() - 1
        empty[mask] = empty[mask ^ bit] * (1.0 - p[v])

    def singleton(v: int) -> Summary:
        values = [0.0] * n
        values[v] = p[v]
        endpoint = tuple(values)
        return Summary(0.0, endpoint, endpoint, 1.0 - p[v])

    def concatenate(left: Summary, right: Summary) -> Summary:
        cross = sum(
            left.last[u] * right.first[v] * d[u + 1][v + 1]
            for u in range(n)
            for v in range(n)
        )
        first = tuple(
            left.first[v] + left.empty * right.first[v] for v in range(n)
        )
        last = tuple(
            right.last[v] + right.empty * left.last[v] for v in range(n)
        )
        return Summary(
            left.internal + right.internal + cross,
            first,
            last,
            left.empty * right.empty,
        )

    def terminal_cost(summary: Summary, a: int, b: int) -> float:
        return (
            summary.empty * d[a][b]
            + sum(summary.first[v] * d[a][v + 1] for v in range(n))
            + summary.internal
            + sum(summary.last[v] * d[v + 1][b] for v in range(n))
        )

    @lru_cache(None)
    def shortest_paths(mask: int, a: int, b: int) -> tuple[tuple[int, ...], ...]:
        vertices = tuple(v for v in range(n) if mask >> v & 1)
        best = math.inf
        answer: list[tuple[int, ...]] = []
        for order in permutations(vertices):
            walk = (a, *(v + 1 for v in order), b)
            value = sum(d[u][v] for u, v in zip(walk, walk[1:]))
            if value < best - 1e-10:
                best = value
                answer = [order]
            elif abs(value - best) <= 1e-10:
                answer.append(order)
        return tuple(answer)

    @lru_cache(None)
    def solve(state: int, a: int, b: int) -> Summary:
        if state == 0:
            return Summary(0.0, zero_vector, zero_vector, 1.0)
        normalizer = 1.0 - empty[state]
        aggregate_internal = 0.0
        aggregate_first = [0.0] * n
        aggregate_last = [0.0] * n

        guide = state
        while guide:
            mass = 1.0
            for v in range(n):
                if state >> v & 1:
                    mass *= p[v] if guide >> v & 1 else 1.0 - p[v]
            mass /= normalizer
            residual = state ^ guide

            chosen: Summary | None = None
            chosen_value = math.inf
            chosen_order: tuple[int, ...] | None = None
            for order in shortest_paths(guide, a, b):
                anchors = (a, *(v + 1 for v in order), b)
                gap_masks = [0] * (len(order) + 1)
                for v in range(n):
                    if not (residual >> v & 1):
                        continue
                    vertex = v + 1
                    excesses = [
                        d[x][vertex] + d[vertex][y] - d[x][y]
                        for x, y in zip(anchors, anchors[1:])
                    ]
                    gap = min(range(len(excesses)), key=lambda i: (excesses[i], i))
                    gap_masks[gap] |= 1 << v

                expanded = solve(gap_masks[0], anchors[0], anchors[1])
                for index, v in enumerate(order):
                    expanded = concatenate(expanded, singleton(v))
                    expanded = concatenate(
                        expanded,
                        solve(gap_masks[index + 1], anchors[index + 1], anchors[index + 2]),
                    )
                value = terminal_cost(expanded, a, b)
                if value < chosen_value - 1e-10 or (
                    abs(value - chosen_value) <= 1e-10
                    and (chosen_order is None or order < chosen_order)
                ):
                    chosen = expanded
                    chosen_value = value
                    chosen_order = order
            assert chosen is not None
            aggregate_internal += mass * chosen.internal
            for v in range(n):
                aggregate_first[v] += mass * chosen.first[v]
                aggregate_last[v] += mass * chosen.last[v]
            guide = (guide - 1) & state

        return Summary(
            aggregate_internal,
            tuple(aggregate_first),
            tuple(aggregate_last),
            empty[state],
        )

    result = solve(full, 0, 0)
    route = terminal_cost(result, 0, 0)
    costs, _ = all_subset_optima(d)
    post = posterior(costs, p)
    return route / post if post else 1.0, route, post


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1709)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    alphabet = (0.03, 0.08, 0.15, 0.3, 0.5, 0.75, 0.95)
    worst = (0.0, None)
    for trial in range(args.trials):
        d = random_graph_metric(args.n, rng)
        profiles = [[value] * args.n for value in alphabet]
        profiles.extend(
            [rng.choice(alphabet) for _ in range(args.n)] for _ in range(5)
        )
        for p in profiles:
            value = solve_instance(d, p)
            if value[0] > worst[0]:
                worst = (value[0], (trial, p, value, d))
                print("worst", worst, flush=True)
            if value[0] > 2.9 + 1e-8:
                print("COUNTEREXAMPLE", worst)
                return
    print("final", worst)


if __name__ == "__main__":
    main()
