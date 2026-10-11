#!/usr/bin/env python3
"""Exact rational certificate for the weighted independent-layer lower bound.

This verifies the constant, not the combinatorial theorem. The proof is in
thesis/symmetric_weighted_layer_lower_bound_zh.md. Only the standard library
is used; all inequality checks use fractions.Fraction.
"""

from decimal import Decimal, localcontext
from fractions import Fraction as F


def ein_bounds(x: F, terms: int = 80) -> tuple[F, F]:
    """Bounds for Ein(x), using an even alternating-series partial sum."""
    assert x >= 0 and terms % 2 == 0 and terms > x
    power_over_factorial = F(1)
    partial = F(0)
    for k in range(1, terms + 1):
        power_over_factorial *= x / k
        partial += (-1) ** (k + 1) * power_over_factorial / k
    next_term = power_over_factorial * x / (terms + 1) ** 2
    return partial, partial + next_term


def exp_negative_bounds(x: F, terms: int = 50) -> tuple[F, F]:
    """Bounds for exp(-x), using an even alternating-series partial sum."""
    assert x >= 0 and terms % 2 == 0 and terms > x
    term = F(1)
    partial = term
    for k in range(1, terms + 1):
        term *= -x / k
        partial += term
    next_magnitude = abs(term) * x / (terms + 1)
    return partial - next_magnitude, partial


def derivative_bounds(x: F, beta: F) -> tuple[F, F]:
    exp_lo, exp_hi = exp_negative_bounds(x)
    return (1 - exp_hi) / x - beta, (1 - exp_lo) / x - beta


def decimal(x: F) -> str:
    with localcontext() as ctx:
        ctx.prec = 32
        return str(Decimal(x.numerator) / Decimal(x.denominator))


def certify(c: F, medium_length: F, target: F, ceiling: F) -> None:
    beta = 2 * medium_length - 3

    # g(x) = Ein(x) - beta*x is strictly concave for x>0, and its
    # derivative decreases from 1-beta to -beta. Bracket its unique maximum.
    root_lo, root_hi = F(1), F(2)
    assert derivative_bounds(root_lo, beta)[0] > 0
    assert derivative_bounds(root_hi, beta)[1] < 0
    for _ in range(32):
        middle = (root_lo + root_hi) / 2
        deriv_lo, deriv_hi = derivative_bounds(middle, beta)
        if deriv_lo > 0:
            root_lo = middle
        elif deriv_hi < 0:
            root_hi = middle
        else:
            raise ArithmeticError("Increase the exponential series precision")

    # Monotonicity of Ein brackets J(beta)=max_x g(x) on the root interval.
    j_lo = ein_bounds(root_lo)[0] - beta * root_hi
    j_hi = ein_bounds(root_hi)[1] - beta * root_lo
    ein_lo, ein_hi = ein_bounds(c)
    denominator = c + medium_length - 1
    ratio_lo = (2 * c + ein_lo - j_hi) / denominator
    ratio_hi = (2 * c + ein_hi - j_lo) / denominator

    assert ratio_lo > target
    assert ratio_hi < ceiling

    print(f"Parameters: c={c}, medium edge length={medium_length}, beta={beta}")
    print("Maximizer x interval:", decimal(root_lo), decimal(root_hi))
    print("J(beta) interval:", decimal(j_lo), decimal(j_hi))
    print("Ratio interval:", decimal(ratio_lo), decimal(ratio_hi))
    print(f"Exact rational certificate: {decimal(target)} < F < {decimal(ceiling)}")


def main() -> None:
    certify(F(10), F(9, 5), F(210011, 100000), F(210012, 100000))
    certify(F(10), F(181, 100), F(210017, 100000), F(210018, 100000))


if __name__ == "__main__":
    main()
