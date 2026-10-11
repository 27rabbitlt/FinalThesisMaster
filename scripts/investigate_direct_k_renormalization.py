#!/usr/bin/env python3
"""Numerical exploration of direct unvisited-row renormalization on a line.

This is an investigation helper, not a proof checker.  Root is 0, clients are
1,...,n, and posterior tours use increasing line order.  The exact posterior
edge marginals are used to sample the next unvisited client directly from the
current row.
"""

from __future__ import annotations

import argparse
import math
import random


def line_y(n: int, p: float) -> list[list[float]]:
    z = 1.0 - p
    y = [[0.0] * (n + 1) for _ in range(n + 1)]
    for v in range(1, n + 1):
        y[0][v] = y[v][0] = p * (z ** (v - 1) + z ** (n - v))
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            y[u][v] = y[v][u] = p * p * z ** (v - u - 1)
    return y


def sample_order(y: list[list[float]], rng: random.Random) -> list[int]:
    n = len(y) - 1
    remaining = list(range(1, n + 1))
    current = 0
    order: list[int] = []
    while remaining:
        weights = [y[current][v] for v in remaining]
        total = sum(weights)
        if total <= 0:
            raise RuntimeError("zero remaining row")
        target = rng.random() * total
        acc = 0.0
        choice_index = len(remaining) - 1
        for i, weight in enumerate(weights):
            acc += weight
            if target <= acc:
                choice_index = i
                break
        current = remaining.pop(choice_index)
        order.append(current)
    return order


def stopped_order_cost(
    order: list[int], p: float, *, cycle: bool = False, two_cluster: bool = False
) -> float:
    z = 1.0 - p
    n = len(order)
    value = 0.0
    circumference = n + 1

    def distance(u: int, v: int) -> int:
        if two_cluster:
            if u == 0 or v == 0:
                return 0 if u == v else 1
            return 0 if (u <= n // 2) == (v <= n // 2) else 2
        raw = abs(u - v)
        return min(raw, circumference - raw) if cycle else raw

    for j, v in enumerate(order):
        value += p * z**j * distance(0, v)
        value += p * z ** (n - 1 - j) * distance(v, 0)
    for i in range(n):
        u = order[i]
        for j in range(i + 1, n):
            value += p * p * z ** (j - i - 1) * distance(u, order[j])
    return value


def posterior_line_cost(n: int, p: float) -> float:
    z = 1.0 - p
    return 2.0 * sum(1.0 - z**j for j in range(1, n + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--lambda", dest="lam", type=float, default=10.0)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--cycle", action="store_true")
    parser.add_argument("--two-cluster", action="store_true")
    args = parser.parse_args()
    p = args.lam / args.n
    if not 0 < p < 1:
        raise ValueError("need 0 < lambda/n < 1")
    rng = random.Random(args.seed)
    y = line_y(args.n, p)
    samples = [sample_order(y, rng) for _ in range(args.trials)]
    values = [
        stopped_order_cost(
            order, p, cycle=args.cycle, two_cluster=args.two_cluster
        )
        for order in samples
    ]
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / max(1, len(values) - 1)
    if args.two_cluster:
        posterior = stopped_order_cost(
            list(range(1, args.n + 1)), p, two_cluster=True
        )
    elif args.cycle:
        posterior = stopped_order_cost(list(range(1, args.n + 1)), p, cycle=True)
    else:
        posterior = posterior_line_cost(args.n, p)
    stderr = math.sqrt(variance / len(values)) / posterior
    print(f"n={args.n} lambda={args.lam:g} p={p:g} trials={args.trials}")
    print(f"posterior={posterior:.12g}")
    print(f"mean={mean:.12g}")
    print(f"ratio={mean / posterior:.12g} stderr={stderr:.4g}")
    if args.cycle:
        circumference = args.n + 1
        def cd(u: int, v: int) -> int:
            raw = abs(u - v)
            return min(raw, circumference - raw)
        full_lengths = [
            cd(0, order[0])
            + sum(cd(u, v) for u, v in zip(order, order[1:]))
            + cd(order[-1], 0)
            for order in samples
        ]
        print(f"mean_full_order={sum(full_lengths) / len(full_lengths):.12g}")


if __name__ == "__main__":
    main()
