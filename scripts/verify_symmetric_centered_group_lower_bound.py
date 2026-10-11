#!/usr/bin/env python3
"""Exact certificate for the centered-group lower bound strictly above 2.18.

The combinatorial proof is in thesis/symmetric_centered_group_lower_bound_zh.md.
All comparisons here are exact rational comparisons, not floating point.
"""

from fractions import Fraction as F

from verify_symmetric_weighted_layer_lower_bound import (
    decimal,
    derivative_bounds,
    ein_bounds,
)


def log_bounds(x: F, terms: int = 100) -> tuple[F, F]:
    """Positive atanh series and a geometric upper bound on its remainder."""
    assert x >= 1
    z = (x - 1) / (x + 1)
    power = z
    partial = F(0)
    for k in range(terms):
        partial += 2 * power / (2 * k + 1)
        power *= z * z
    remainder = 2 * power / ((2 * terms + 1) * (1 - z * z))
    return partial, partial + remainder


def main() -> None:
    c, lam = F(5), F(2, 3)
    ln3_lo, ln3_hi = log_bounds(F(3))
    assert log_bounds(F(3, 2))[0] > F(2, 5)

    # The maximizer of Ein(x)-lambda*x lies below ln(3), where the
    # piecewise concave function psi(x) equals 1-exp(-x).
    x_lo, x_hi = F(1, 2), F(1)
    assert derivative_bounds(x_lo, lam)[0] > 0
    assert derivative_bounds(x_hi, lam)[1] < 0
    assert x_hi < ln3_lo
    for _ in range(32):
        middle = (x_lo + x_hi) / 2
        d_lo, d_hi = derivative_bounds(middle, lam)
        if d_lo > 0:
            x_lo = middle
        elif d_hi < 0:
            x_hi = middle
        else:
            raise ArithmeticError("Increase series precision")

    j_lo = ein_bounds(x_lo)[0] - lam * x_hi
    j_hi = ein_bounds(x_hi)[1] - lam * x_lo

    # Integral_0^5 psi(v)/v dv, with
    # psi(v)=min(1-exp(-v), (4/5)*(1-exp(-v))+2/15).
    # Its breakpoint is ln(3). Monotonicity handles that irrational endpoint.
    i_lo = (
        F(4, 5) * ein_bounds(c)[0]
        + F(1, 5) * ein_bounds(ln3_lo)[0]
        + F(2, 15) * log_bounds(c / ln3_hi)[0]
    )
    i_hi = (
        F(4, 5) * ein_bounds(c)[1]
        + F(1, 5) * ein_bounds(ln3_hi)[1]
        + F(2, 15) * log_bounds(c / ln3_lo)[1]
    )
    denominator = c + F(1, 2)
    ratio_lo = (2 * c + i_lo - j_hi) / denominator
    ratio_hi = (2 * c + i_hi - j_lo) / denominator
    assert ratio_lo > F(21803, 10000)
    assert ratio_hi < F(218031, 100000)

    print("Parameters: medium length=3/2, c=5, lambda=2/3")
    print("Maximizer interval:", decimal(x_lo), decimal(x_hi))
    print("Integral interval:", decimal(i_lo), decimal(i_hi))
    print("J(2/3) interval:", decimal(j_lo), decimal(j_hi))
    print("Ratio interval:", decimal(ratio_lo), decimal(ratio_hi))
    print("Exact rational certificate: 2.1803 < lower-bound expression < 2.18031")


if __name__ == "__main__":
    main()
