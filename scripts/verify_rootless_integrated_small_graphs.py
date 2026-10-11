#!/usr/bin/env python3
"""Exact integrated audit of rootless two-sample compression on small graphs.

All connected labelled unweighted graph metrics on a depot and at most four
clients are enumerated.  For selected rational probability vectors, Fraction
arithmetic computes P, R, and the rootless/rootful (M,A,Gamma) terms.  The
script asserts the sufficient rootless target Gamma >= 2R/19 and reports the
smallest observed ratios.  The metric enumeration is exhaustive; the finite
probability list is a stress test, not a proof for arbitrary probabilities.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product

from verify_rootless_compression_threshold import graph_metric, probability_vectors


def mask_probability(mask: int, p: tuple[Fraction, ...]) -> Fraction:
    answer = Fraction(1)
    for v, pv in enumerate(p):
        answer *= pv if mask & (1 << v) else 1 - pv
    return answer


def all_tour_costs(n: int, d: list[list[int]]) -> list[Fraction]:
    root = n
    infinity = Fraction(10**9)
    answer = [Fraction(0)] * (1 << n)
    for mask in range(1, 1 << n):
        dp: dict[tuple[int, int], Fraction] = {}
        for last in range(n):
            if mask & (1 << last):
                dp[1 << last, last] = Fraction(d[root][last])
        for sub in range(1, 1 << n):
            if sub & ~mask or not sub & (sub - 1):
                continue
            for last in range(n):
                if sub & (1 << last):
                    before_mask = sub ^ (1 << last)
                    dp[sub, last] = min(
                        dp.get((before_mask, before), infinity) + d[before][last]
                        for before in range(n)
                        if before_mask & (1 << before)
                    )
        answer[mask] = min(
            dp[mask, last] + d[last][root]
            for last in range(n)
            if mask & (1 << last)
        )
    return answer


def assignment(n: int, target: int, d: list[list[int]], p: tuple[Fraction, ...]) -> Fraction:
    selected = [n] + [v for v in range(n) if target & (1 << v)]
    return sum(p[v] * min(d[v][u] for u in selected) for v in range(n))


def compressed_terms(
    n: int,
    d: list[list[int]],
    p: tuple[Fraction, ...],
    weights: list[Fraction],
    tours: list[Fraction],
    rootless: bool,
) -> tuple[Fraction, Fraction]:
    root = n
    backbone = Fraction(0)
    attachment = Fraction(0)
    for guide in range(1 << n):
        centers = [v for v in range(n) if guide & (1 << v)]
        available = centers if rootless and centers else centers + [root]
        cells = {b: [] for b in centers}
        for v in range(n):
            image = v if guide & (1 << v) else min(
                available, key=lambda b: (d[v][b], b)
            )
            if image != root:
                cells[image].append(v)
        occupancy = {
            b: 1 - product_fraction(1 - p[v] for v in cells[b])
            for b in centers
        }
        sub = guide
        while True:
            conditional = Fraction(1)
            for b in centers:
                conditional *= occupancy[b] if sub & (1 << b) else 1 - occupancy[b]
            probability = weights[guide] * conditional
            backbone += probability * tours[sub]
            attachment += probability * assignment(n, sub, d, p)
            if sub == 0:
                break
            sub = (sub - 1) & guide
    return backbone, attachment


def product_fraction(values) -> Fraction:
    answer = Fraction(1)
    for value in values:
        answer *= value
    return answer


def main() -> None:
    graph_count = 0
    audits = 0
    minimum_rootless = None
    minimum_e = None
    minimum_better = None
    for n in range(1, 5):
        pair_count = n * (n + 1) // 2
        for graph_mask in range(1 << pair_count):
            d = graph_metric(n, graph_mask)
            if d is None:
                continue
            graph_count += 1
            tours = all_tour_costs(n, d)
            for p in probability_vectors(n):
                weights = [mask_probability(mask, p) for mask in range(1 << n)]
                posterior = sum(weights[mask] * tours[mask] for mask in range(1 << n))
                radius = sum(weights[mask] * assignment(n, mask, d, p) for mask in range(1 << n))
                m_star, a_star = compressed_terms(n, d, p, weights, tours, True)
                m_root, a_root = compressed_terms(n, d, p, weights, tours, False)
                gamma_star = 3 * posterior - m_star - 2 * a_star
                gamma_root = 3 * posterior - m_root - 2 * a_root
                assert 19 * gamma_star >= 2 * radius, (
                    n, graph_mask, p, posterior, radius, m_star, a_star, gamma_star, d
                )
                if radius:
                    ratio = gamma_star / radius
                    record = (ratio, n, graph_mask, p, posterior, radius, m_star, a_star, d)
                    if minimum_rootless is None or ratio < minimum_rootless[0]:
                        minimum_rootless = record
                    e_margin = float(ratio) - 1 / 2.718281828459045
                    if minimum_e is None or e_margin < minimum_e[0]:
                        minimum_e = (e_margin,) + record[1:]
                    better_ratio = max(gamma_star, gamma_root) / radius
                    better_record = (
                        better_ratio, n, graph_mask, p, gamma_star, gamma_root, posterior, radius, d
                    )
                    if minimum_better is None or better_ratio < minimum_better[0]:
                        minimum_better = better_record
                audits += 1
    print(f"verified {audits} integrated audits on {graph_count} connected graph metrics")
    print("minimum Gamma*/R =", minimum_rootless)
    print("minimum Gamma*/R - 1/e =", minimum_e)
    print("minimum max(Gamma*,Gamma_root)/R =", minimum_better)


if __name__ == "__main__":
    main()
