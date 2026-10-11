#!/usr/bin/env python3
"""Numerically explore the proven two-layer bound (not a rigorous certificate)."""

import json
import math

from scipy.integrate import quad
from scipy.optimize import brentq, differential_evolution, minimize


def phi(x: float, a: float, lam: float) -> float:
    f = -math.expm1(-x)
    if a < 1.5:
        tau = a - 1
        gamma = 1 - 2 * tau
        reserve = 2 * tau - lam
        assert reserve > 0
        discriminant = (tau * f) ** 2 - 4 * gamma * (1 - f) * reserve
        if discriminant <= 0:
            return f
        z = 2 * reserve / (tau * f + math.sqrt(discriminant))
        if z >= 1:
            return f
        return min(f, -2 * reserve * math.log(z)
                   + gamma * (1 - (1 - f) * z * z) + 2 * tau * f * z)
    b = 2 * a - 3
    if b >= 1 - 1e-14:
        return f
    k = 2 * (1 - lam) / (1 - b)
    if f <= k:
        return f
    if k <= 0:
        return b * f
    return b * f + (1 - b) * k * (1 + math.log(f / k))


def bound(c: float, a: float, lam: float, exact_adapt: bool = True) -> dict:
    def slope(x):
        return phi(x, a, lam) / x - lam if x else 1 - lam

    w = c if slope(c) >= 0 else brentq(slope, 1e-12, c)
    k = (1 - lam) / (2 - a) if 1.5 <= a < 2 else 2
    breakpoint = -math.log1p(-k) if 0 < k < 1 else math.inf
    points = [breakpoint] if w < breakpoint < c else []
    integral = quad(lambda x: phi(x, a, lam) / x, w, c,
                    points=points, epsabs=2e-10, epsrel=2e-10)[0]
    numerator = 2 * c + integral + lam * w
    denominator = c + (a - 1) * (-math.expm1(-c) if exact_adapt else 1)
    return dict(c=c, a=a, lam=lam, w=w, u=w/c,
                numerator=numerator, denominator=denominator,
                ratio=numerator / denominator)


def main():
    results = {"fixed_exact_phi_loose_adapt": bound(5, 1.5, 2/3, False),
               "fixed_exact_phi_exact_adapt": bound(5, 1.5, 2/3)}
    for exact in (False, True):
        opt = differential_evolution(
            lambda v: -bound(*v, exact_adapt=exact)["ratio"],
            [(0.1, 25), (1.5, 2), (0.05, 0.999)], seed=2026,
            tol=1e-10, polish=True)
        results[f"optimized_exact_adapt_{exact}"] = bound(*opt.x, exact_adapt=exact)
    opt = differential_evolution(
        lambda v: -bound(v[0], v[1], 2*(v[1]-1)*v[2])["ratio"],
        [(0.1, 15), (1.01, 1.5), (0.01, 0.999)], seed=2026,
        tol=1e-10, polish=True)
    results["short_medium_optimized"] = bound(
        opt.x[0], opt.x[1], 2*(opt.x[1]-1)*opt.x[2])
    results["short_medium_simple"] = bound(16/5, 32/25, 12/25)
    for a in (1.5, 1.51, 1.55, 1.6, 1.7, 1.8, 2):
        opt = minimize(lambda v: -bound(v[0], a, v[1])["ratio"], [5, 2/3],
                       method="Nelder-Mead", bounds=[(.1, 25), (.05, .999)],
                       options={"xatol":1e-9, "fatol":1e-11})
        results[f"length_{a}"] = bound(opt.x[0], a, opt.x[1])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
