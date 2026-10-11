#!/usr/bin/env python3
"""Exact certificate against direct posterior-arc row renormalization.

The witness has a root-side block L and a remote block R.  Distances within
either side are delta and distances across the cut are one.  Uniformly random
orders inside the two blocks, followed by a fair global orientation, are
posterior-optimal tours.  Their directed arc marginal is block-exchangeable.

The direct rule starts at the root and repeatedly chooses an unvisited client
v with probability proportional to the current row z[current,v].  A small
count-state dynamic program computes its activation-thinned expected number
of cut crossings exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction as F
from itertools import permutations


DOWN = Context(prec=60, rounding=ROUND_FLOOR)
UP = Context(prec=60, rounding=ROUND_CEILING)


@dataclass(frozen=True, slots=True)
class I:
    """Outward-rounded nonnegative decimal interval."""

    lo: Decimal
    hi: Decimal

    @staticmethod
    def rational(value: F) -> "I":
        numerator = Decimal(value.numerator)
        denominator = Decimal(value.denominator)
        return I(DOWN.divide(numerator, denominator), UP.divide(numerator, denominator))

    @staticmethod
    def integer(value: int) -> "I":
        exact = Decimal(value)
        return I(exact, exact)

    def __add__(self, other: "I") -> "I":
        return I(DOWN.add(self.lo, other.lo), UP.add(self.hi, other.hi))

    def __mul__(self, other: "I") -> "I":
        return I(DOWN.multiply(self.lo, other.lo), UP.multiply(self.hi, other.hi))

    def __truediv__(self, other: "I") -> "I":
        assert other.lo > 0
        return I(DOWN.divide(self.lo, other.hi), UP.divide(self.hi, other.lo))

    def scale(self, value: int) -> "I":
        exact = Decimal(value)
        return I(DOWN.multiply(self.lo, exact), UP.multiply(self.hi, exact))


def arc_weights(a: int, b: int, p: F) -> tuple[F, F, F, F, F]:
    """Return z_LL, z_RR, z_LR=z_RL, z_root,L, z_root,R per arc."""
    q = 1 - p
    ql = 1 - q**a
    qr = 1 - q**b
    wll = F(0) if a == 1 else (a * p - ql) / (a * (a - 1))
    wrr = F(0) if b == 1 else (b * p - qr) / (b * (b - 1))
    wlr = ql * qr / (2 * a * b)
    w0l = ql * (2 - qr) / (2 * a)
    w0r = qr * (2 - ql) / (2 * b)
    return wll, wrr, wlr, w0l, w0r


def direct_cut_cost(a: int, b: int, p: F) -> I:
    """Expected cut crossings of the activation-thinned direct-rule order."""
    q = 1 - p
    exact_weights = arc_weights(a, b, p)
    wll, wrr, wlr, w0l, w0r = map(I.rational, exact_weights)
    pi, qi = I.rational(p), I.rational(q)
    zero, one = I.integer(0), I.integer(1)

    # dp[l][r][current side][last active side].  Side zero contains the root.
    dp = [[[[zero, zero] for _ in range(2)] for _r in range(b + 1)]
          for _l in range(a + 1)]
    for current in range(2):
        dp[0][0][current][1] = one  # final return from R to the root

    for count in range(1, a + b + 1):
        for l in range(max(0, count - b), min(a, count) + 1):
            r = count - l
            for current in range(2):
                wl = (wll if current == 0 else wlr).scale(l)
                wr = (wlr if current == 0 else wrr).scale(r)
                total = wl + wr
                if total.hi == 0:  # unreachable count state
                    continue
                for last in range(2):
                    value = zero
                    if l:
                        jump = I.integer(last == 1)
                        value += wl / total * (
                            pi * (jump + dp[l - 1][r][0][0])
                            + qi * dp[l - 1][r][0][last]
                        )
                    if r:
                        jump = I.integer(last == 0)
                        value += wr / total * (
                            pi * (jump + dp[l][r - 1][1][1])
                            + qi * dp[l][r - 1][1][last]
                        )
                    dp[l][r][current][last] = value

    wl = w0l.scale(a)
    wr = w0r.scale(b)
    total = wl + wr
    # The last-active side is initially the root side.  An active first R
    # client contributes the initial cut crossing.
    return (
        wl / total * dp[a - 1][b][0][0]
        + wr / total * (
            pi * (one + dp[a][b - 1][1][1])
            + qi * dp[a][b - 1][1][0]
        )
    )


def enumerate_small_marginal(a: int, b: int, p: F) -> list[list[F]]:
    """Independent enumeration of the random posterior tour distribution."""
    n = a + b
    z = [[F(0) for _ in range(n + 1)] for _ in range(n + 1)]
    for mask in range(1 << n):
        active_l = [v + 1 for v in range(a) if mask & (1 << v)]
        active_r = [a + v + 1 for v in range(b) if mask & (1 << (a + v))]
        probability = F(1)
        for v in range(n):
            probability *= p if mask & (1 << v) else 1 - p
        if not active_l and not active_r:
            continue
        orders_l = list(permutations(active_l)) or [tuple()]
        orders_r = list(permutations(active_r)) or [tuple()]
        multiplicity = 2 * len(orders_l) * len(orders_r)
        for left in orders_l:
            for right in orders_r:
                forward = (*left, *right)
                for order in (forward, tuple(reversed(forward))):
                    cycle = (0, *order, 0)
                    for u, v in zip(cycle, cycle[1:]):
                        z[u][v] += probability / multiplicity
    return z


def audit_formulas() -> None:
    a, b, p = 3, 2, F(1, 3)
    z = enumerate_small_marginal(a, b, p)
    wll, wrr, wlr, w0l, w0r = arc_weights(a, b, p)
    for u in range(1, a + 1):
        assert z[0][u] == w0l
        for v in range(1, a + 1):
            if u != v:
                assert z[u][v] == wll
        for v in range(a + 1, a + b + 1):
            assert z[u][v] == wlr
    for u in range(a + 1, a + b + 1):
        assert z[0][u] == w0r
        for v in range(a + 1, a + b + 1):
            if u != v:
                assert z[u][v] == wrr
        for v in range(1, a + 1):
            assert z[u][v] == wlr
    # Every client row and column has mass p; the root row has occupancy mass.
    for u in range(1, a + b + 1):
        assert sum(z[u]) == p
        assert sum(row[u] for row in z) == p
    assert sum(z[0]) == 1 - (1 - p) ** (a + b)


def audit_witness() -> tuple[I, F, I, F]:
    a = b = 70
    p = F(3, 20)
    delta = F(1, 1000)
    q = 1 - p

    crossing_direct = direct_cut_cost(a, b, p)
    crossing_posterior = 2 * (1 - q**b)

    # Every nonempty realized rooted cycle has N+1 edge occurrences, including
    # multiplicity two for a singleton.  This expectation is order-independent.
    edge_count = (a + b) * p + 1 - q ** (a + b)
    direct = I.rational(delta * edge_count) + I.rational(1 - delta) * crossing_direct
    posterior = delta * edge_count + (1 - delta) * crossing_posterior

    assert DOWN.multiply(Decimal(10), crossing_direct.lo) > UP.multiply(
        Decimal(29), I.rational(crossing_posterior).hi
    )
    assert DOWN.multiply(Decimal(10), direct.lo) > UP.multiply(
        Decimal(29), I.rational(posterior).hi
    )
    return crossing_direct, crossing_posterior, direct, posterior


def main() -> None:
    audit_formulas()
    crossing_direct, crossing_posterior, direct, posterior = audit_witness()
    print("exact oriented-arc direct-rule obstruction passed")
    cut_denominator = I.rational(crossing_posterior)
    strict_denominator = I.rational(posterior)
    cut_ratio = crossing_direct / cut_denominator
    strict_ratio = direct / strict_denominator
    print("  cut-only ratio interval:", cut_ratio.lo, cut_ratio.hi)
    print("  strict-metric ratio interval:", strict_ratio.lo, strict_ratio.hi)


if __name__ == "__main__":
    main()
