#!/usr/bin/env python3
"""Exact certificate for the refined symmetric stochastic-TSP gap bound.

This checks only the scalar substitutions in the structural proof of

    OPT_prior / OPT_post <= 3 - 9 / 2_500_000_000.

Every assertion uses rational arithmetic.  The exponential upper bounds use
a Taylor partial sum plus a geometric bound on the remaining positive tail.
"""

from fractions import Fraction as Q


def exp_bounds(x: Q, degree: int) -> tuple[Q, Q]:
    """Return rational lower and upper bounds for exp(x), where x >= 0."""
    assert x >= 0
    term = Q(1)
    partial = Q(1)
    for k in range(1, degree + 1):
        term *= x / k
        partial += term

    next_term = term * x / (degree + 1)
    tail_ratio = x / (degree + 2)
    assert tail_ratio < 1
    return partial, partial + next_term / (1 - tail_ratio)


def main() -> None:
    a = Q(31, 30)
    b = Q(153, 148)
    t = Q(15, 56)
    gamma = Q(1, 17_000)
    delta_0 = Q(9, 5_000_000_000)
    eta = 2 * delta_0

    # Refined image-edge and symmetrized-isolation constants.
    K = a + b + a * b
    B0 = 2 * b * (K - a)
    c = 10 * (
        (a + 1) / (2 * (a - 1))
        + (b + 1) / (2 * (b - 1))
    )
    s = t * (1 - gamma)
    h = c * delta_0
    Z = h / ((1 - gamma) * (1 - delta_0))

    assert K == Q(13_921, 4_440)
    assert B0 == Q(475_983, 109_520)
    assert c == 606
    assert 0 < Z < 1
    assert h < 1 - delta_0

    # Uniformly for 0 < delta <= delta_0:
    #   u <= h^(15/56),
    #   v <= (2a)^(41/56) (2 delta_0)^(15/56).
    u_upper = Q(253, 10_000)
    v_upper = Q(233, 25_000)
    assert u_upper**56 > h**15
    assert v_upper**56 > (2 * a) ** 41 * (2 * delta_0) ** 15

    # Bound f_s(Z)=(1-Z^s)/(s log(1/Z)) with rational exp brackets.
    log_lower = Q(137_283, 10_000)
    log_upper = Q(68_643, 5_000)
    exponent_upper = Q(9_193, 2_500)
    z_power_lower = Q(101, 4_000)

    _, exp_log_lower_upper = exp_bounds(log_lower, 60)
    exp_log_upper_lower, _ = exp_bounds(log_upper, 60)
    _, exp_exponent_upper = exp_bounds(exponent_upper, 60)
    assert exp_log_lower_upper < 1 / Z
    assert exp_log_upper_lower > 1 / Z
    assert s * log_upper < exponent_upper
    assert exp_exponent_upper < 1 / z_power_lower

    f_upper = (1 - z_power_lower) / (s * log_lower)
    f_simple_upper = Q(2_651, 10_000)
    assert f_upper < f_simple_upper

    # The refined, normalized small-deficit inequality, relaxed uniformly by
    # r <= 1 and delta <= delta_0.
    upper = (
        2
        + 2 * a * t
        + (1 - t) * (2 * a + B0) * u_upper
        + B0 * t / (1 - gamma) * f_simple_upper
        + B0 * (t / gamma - t * t) * delta_0
        + 2 * v_upper
        + (24 * (a + 1) - 4 * t) * delta_0
    )
    target = 3 - eta
    assert upper == Q(
        262_697_692_283_343_416_931_053,
        87_575_801_779_200_000_000_000,
    )
    assert target == Q(7_499_999_991, 2_500_000_000)
    assert upper < target
    assert target - upper == Q(
        29_712_738_983_696_663_827,
        87_575_801_779_200_000_000_000,
    )

    print("eta =", eta, float(eta))
    print("factor =", target, float(target))
    print("K, B0, c =", K, B0, c)
    print("f upper =", f_upper, float(f_upper))
    print("certified upper =", upper, float(upper))
    print("rational slack =", target - upper)
    print("decimal slack =", float(target - upper))


if __name__ == "__main__":
    main()
