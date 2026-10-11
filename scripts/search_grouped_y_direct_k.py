#!/usr/bin/env python3
"""Search grouped posterior-tour mixtures against direct K renormalization."""

from __future__ import annotations

from functools import lru_cache
import argparse
import random


def add_tour(y: list[list[float]], tour: tuple[int, ...], weight: float) -> None:
    cycle = (0, *tour, 0)
    for u, v in zip(cycle, cycle[1:]):
        y[u][v] += weight
        y[v][u] += weight


def direct_runs(y: list[list[float]], m: int) -> float | None:
    n = 2 * m
    full = (1 << n) - 1

    def side(v: int) -> int:
        return 0 if v <= m else 1

    @lru_cache(None)
    def rec(remaining: int, current: int) -> float:
        if not remaining:
            return 0.0
        choices = [i + 1 for i in range(n) if remaining >> i & 1]
        total = sum(y[current][v] for v in choices)
        if total < 1e-14:
            raise ZeroDivisionError
        value = 0.0
        for v in choices:
            increment = 0.0 if current == 0 or side(current) == side(v) else 1.0
            value += y[current][v] / total * (
                increment + rec(remaining ^ (1 << (v - 1)), v)
            )
        return value

    try:
        # One run plus the number of L/R switches.
        return 1.0 + rec(full, 0)
    except ZeroDivisionError:
        return None


def random_grouped_tour(m: int, rng: random.Random) -> tuple[int, ...]:
    left = list(range(1, m + 1))
    right = list(range(m + 1, 2 * m + 1))
    rng.shuffle(left)
    rng.shuffle(right)
    if rng.random() < 0.5:
        return (*left, *right)
    return (*right, *left)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--support", type=int, default=12)
    parser.add_argument("--trials", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--floor", type=float, default=0.001)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    n = 2 * args.m
    best = 0.0
    for trial in range(args.trials):
        y = [[0.0] * (n + 1) for _ in range(n + 1)]
        eta = args.floor
        for v in range(1, n + 1):
            y[0][v] = y[v][0] = eta / args.m
        for u in range(1, n + 1):
            for v in range(u + 1, n + 1):
                if (u <= args.m) == (v <= args.m):
                    y[u][v] = y[v][u] = eta * 2.0 / args.m
                else:
                    y[u][v] = y[v][u] = eta / (args.m * args.m)
        tours = [random_grouped_tour(args.m, rng) for _ in range(args.support)]
        raw = [rng.expovariate(1.0) for _ in tours]
        total = sum(raw)
        weights = [(1.0 - eta) * x / total for x in raw]
        for tour, weight in zip(tours, weights):
            cycle = (0, *tour, 0)
            for u, v in zip(cycle, cycle[1:]):
                y[u][v] += weight
                y[v][u] += weight
        runs = direct_runs(y, args.m)
        if runs is not None and runs > best:
            best = runs
            print("best", best, "ratio", best / 2, flush=True)
            if best > 5.8:
                print("weights_and_tours")
                for weight, tour in zip(weights, tours):
                    print(weight, tour)
                return


if __name__ == "__main__":
    main()
