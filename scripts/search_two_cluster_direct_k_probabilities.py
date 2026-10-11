#!/usr/bin/env python3
"""Search activation profiles on a two-site grouped-order metric."""

from __future__ import annotations

from functools import lru_cache
import argparse
import random


def stopped_incidence(p: list[float]) -> list[list[float]]:
    n = len(p)
    z = [1.0 - x for x in p]
    y = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        left = 1.0
        for k in range(i):
            left *= z[k]
        right = 1.0
        for k in range(i + 1, n):
            right *= z[k]
        y[0][i + 1] = y[i + 1][0] = p[i] * (left + right)
        gap = 1.0
        for j in range(i + 1, n):
            y[i + 1][j + 1] = y[j + 1][i + 1] = p[i] * p[j] * gap
            gap *= z[j]
    return y


def direct_value(p: list[float]) -> float:
    n = len(p)
    m = n // 2
    y = stopped_incidence(p)

    def d(u: int, v: int) -> int:
        if u == 0 or v == 0:
            return 0 if u == v else 1
        return 0 if (u <= m) == (v <= m) else 2

    @lru_cache(None)
    def rec(rem: int, current: int, last: int) -> float:
        if not rem:
            return d(last, 0)
        choices = [i for i in range(n) if rem >> i & 1]
        den = sum(y[current][i + 1] for i in choices)
        if den <= 1e-250:
            raise ArithmeticError
        ans = 0.0
        for i in choices:
            v = i + 1
            choose = y[current][v] / den
            rest = rem ^ (1 << i)
            ans += choose * ((1 - p[i]) * rec(rest, v, last) + p[i] * (d(last, v) + rec(rest, v, v)))
        return ans

    return rec((1 << n) - 1, 0, 0)


def posterior(p: list[float]) -> float:
    m = len(p) // 2
    ql = 1.0
    qr = 1.0
    for x in p[:m]:
        ql *= 1.0 - x
    for x in p[m:]:
        qr *= 1.0 - x
    return 2.0 * ((1.0 - ql) + (1.0 - qr))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--trials", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    alphabet = (0.002, 0.01, 0.03, 0.1, 0.3, 0.7, 0.95, 0.998)
    best = 0.0
    for trial in range(args.trials):
        p = [rng.choice(alphabet) for _ in range(2 * args.m)]
        try:
            w = direct_value(p)
        except ArithmeticError:
            continue
        post = posterior(p)
        ratio = w / post
        if ratio > best:
            best = ratio
            print("best", ratio, "p", p, "P", post, "W", w, flush=True)
        if ratio > 2.9:
            return


if __name__ == "__main__":
    main()
