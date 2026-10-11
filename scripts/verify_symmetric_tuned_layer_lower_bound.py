#!/usr/bin/env python3
"""Rational certificate for the shorter-medium-edge adaptivity-gap bound.

The proof uses the same high-girth grouping and H-coverage lemma, including
MM discounts. This script certifies the resulting scalar expression via a
concave polygonal minorant. It does not certify global optimality.
"""

from fractions import Fraction as F
from math import isqrt

from verify_symmetric_weighted_layer_lower_bound import (
    decimal, ein_bounds, exp_negative_bounds,
)

PREC = 10**24


def down(x):
    return F(x.numerator * PREC // x.denominator, PREC)


def up(x):
    return -down(-x)


def log_bounds(x):
    """Range-reduced positive atanh series, with rational remainder."""
    assert x > 0
    if x < 1:
        lo, hi = log_bounds(1/x)
        return -hi, -lo

    def unit(y):
        z = (y-1)/(y+1)
        power, total = z, F(0)
        for i in range(40):
            total += 2*power/(2*i+1)
            power *= z*z
        tail = 2*power/(81*(1-z*z))
        return down(total), up(total+tail)

    exponent = 0
    while x > 2:
        x /= 2
        exponent += 1
    lo, hi = unit(x)
    l2, h2 = unit(F(2))
    return lo+exponent*l2, hi+exponent*h2


def sqrt_bounds(x):
    assert x >= 0
    integer = isqrt(x.numerator * PREC**2 // x.denominator)
    lo, hi = F(integer, PREC), F(integer+1, PREC)
    assert lo*lo <= x <= hi*hi
    return lo, hi


def phi_node_lower(v, tau, lam):
    """Certify inf_{0<z<=1} [-2b log z + gamma(1-(1-v)z²)+2tau*v*z]."""
    gamma, b = 1-2*tau, 2*tau-lam
    assert 0 < tau < F(1, 2) and b > 0
    discriminant = (tau*v)**2 - 4*gamma*(1-v)*b
    if discriminant <= 0:
        return v
    slo, shi = sqrt_bounds(discriminant)
    zlo, zhi = 2*b/(tau*v+shi), 2*b/(tau*v+slo)
    if zlo >= 1:
        return v
    cost_lower = (-2*b*log_bounds(zhi)[1]
                  + gamma*(1-(1-v)*zhi*zhi) + 2*tau*v*zlo)
    return min(v, down(cost_lower))


def polygon(tau, lam, pieces):
    """Upper concave hull: each edge remains a chord below the true phi."""
    hull = []
    for i in range(pieces+1):
        v = F(i, pieces)
        point = (v, phi_node_lower(v, tau, lam))
        hull.append(point)
        while len(hull) >= 3:
            x, y, z = hull[-3:]
            if (y[1]-x[1])*(z[0]-y[0]) >= (z[1]-y[1])*(y[0]-x[0]):
                break
            hull.pop(-2)
    assert hull[0] == (0, 0)
    lines = []
    last_slope = F(1)
    for left, right in zip(hull, hull[1:]):
        slope = (right[1]-left[1])/(right[0]-left[0])
        intercept = left[1]-slope*left[0]
        assert 0 <= slope <= last_slope and intercept >= 0
        lines.append((left[0], right[0], slope, intercept))
        last_slope = slope
    return lines


def g_bounds(x, lines):
    elo, ehi = exp_negative_bounds(x, terms=80)
    return (min(a*(1-ehi)+b for _, _, a, b in lines),
            min(a*(1-elo)+b for _, _, a, b in lines))


def ein_interval(lo, hi):
    return down(ein_bounds(lo)[0]), up(ein_bounds(hi)[1])


def integral_bounds(c, lines):
    """Integral_0^c g(1-exp(-x))/x dx, using Ein and log brackets."""
    total_lo, total_hi = F(0), F(0)
    left_lo = left_hi = F(0)
    for _, vhi, a, b in lines:
        if vhi == 1:
            right_lo = right_hi = c
        else:
            rlo, rhi = log_bounds(1/(1-vhi))
            right_lo, right_hi = min(c, rlo), min(c, rhi)
        if left_lo >= c:
            break
        el, eh = ein_interval(left_lo, left_hi)
        rl, rh = ein_interval(right_lo, right_hi)
        segment_lo, segment_hi = a*(rl-eh), a*(rh-el)
        if b:
            assert left_lo > 0
            segment_lo += b*log_bounds(right_lo/left_hi)[0]
            segment_hi += b*log_bounds(right_hi/left_lo)[1]
        total_lo += segment_lo
        total_hi += segment_hi
        left_lo, left_hi = right_lo, right_hi
        if right_lo == c:
            break
    return down(total_lo), up(total_hi)


def certify(tau=F(7, 25), c=F(16, 5), lam=F(12, 25),
            target=F(4443, 2000), pieces=256):
    lines = polygon(tau, lam, pieces)
    # The decreasing g(x)/x crosses lambda once. Bracket it rationally.
    lo, hi = F(1, 1000), c
    assert g_bounds(lo, lines)[0] > lam*lo
    assert g_bounds(hi, lines)[1] < lam*hi
    for _ in range(60):
        mid = (lo+hi)/2
        gl, gh = g_bounds(mid, lines)
        if gl > lam*mid:
            lo = mid
        elif gh < lam*mid:
            hi = mid
        else:
            raise ArithmeticError("Increase rational precision")
    il, ih = integral_bounds(c, lines)
    wl, _ = integral_bounds(lo, lines)
    _, wh = integral_bounds(hi, lines)
    penalty_lo, penalty_hi = wl-lam*hi, wh-lam*lo
    numerator_lo, numerator_hi = 2*c+il-penalty_hi, 2*c+ih-penalty_lo
    elo, ehi = exp_negative_bounds(c, terms=80)
    denominator_lo, denominator_hi = c+tau*(1-ehi), c+tau*(1-elo)
    ratio_lo = numerator_lo/denominator_hi
    ratio_hi = numerator_hi/denominator_lo
    assert ratio_lo > target
    print(f"Parameters a={1+tau}, c={c}, lambda={lam}, pieces={pieces}")
    print("Polygon segments:", len(lines))
    print("Worst residual w interval:", decimal(lo), decimal(hi))
    print("Prior coefficient interval:", decimal(numerator_lo), decimal(numerator_hi))
    print("Adaptive coefficient interval:", decimal(denominator_lo), decimal(denominator_hi))
    print("Certified polygon-bound ratio interval:", decimal(ratio_lo), decimal(ratio_hi))
    print("STRICT LOWER BOUND:", decimal(target))
    return ratio_lo, ratio_hi


if __name__ == "__main__":
    certify()
