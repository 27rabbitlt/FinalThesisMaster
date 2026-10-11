#!/usr/bin/env python3
"""Exact counterexample to the compressed root-far low-degree certificate.

The metric is the closure of r--u_i edges of length 1 and u_i--v edges
of length 6/5.  At threshold 3/2, v is root-far, every u_i is root-near,
and distinct spokes u_i,u_j are far.  Non-guide spokes project uniquely to
the depot.  We verify in Fraction arithmetic that

    a_t > E[i_t(B) + ell_t^circ(B)],

while the tour-incidence target a_t+r_t <= 2 E[k_t(B)] survives.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb


M = 10
S = Fraction(1, 10)       # spoke activation probability
P_V = Fraction(1, 100)    # central, root-far vertex probability


def binomial_probability(k: int) -> Fraction:
    return Fraction(comb(M, k)) * S**k * (1 - S) ** (M - k)


def long_edges(v_active: bool, active_spokes: int) -> int:
    """Minimum threshold-long edges in a rooted Hamiltonian cycle."""
    if v_active:
        return abs(active_spokes - 2)
    return max(active_spokes - 1, 0)


def main() -> None:
    q = (1 - S) ** M
    one = M * S * (1 - S) ** (M - 1)
    no_diagonal_spoke = (1 - S**2) ** M

    # T misses v iff: no spoke is present in both B and C, and either C_v is
    # absent or B has no spoke.  Splitting on B_v gives the compact formula.
    attachment = P_V * (1 - P_V) * (no_diagonal_spoke + P_V * q)
    radius = P_V * (1 - P_V) * q
    low_degree = P_V * (q + one)
    expected_long_edges = sum(
        binomial_probability(k)
        * ((1 - P_V) * long_edges(False, k) + P_V * long_edges(True, k))
        for k in range(M + 1)
    )

    assert attachment > low_degree
    assert attachment + radius <= 2 * expected_long_edges

    # Closed forms for the two posterior-tour cases.
    no_v_long_edges = M * S - (1 - q)
    with_v_long_edges = M * S - 2 + 4 * q + 2 * one
    assert expected_long_edges == (1 - P_V) * no_v_long_edges + P_V * with_v_long_edges

    print("a_t                         =", attachment, float(attachment))
    print("E[i_t + ell_t^circ]         =", low_degree, float(low_degree))
    print("strict violation            =", attachment - low_degree, float(attachment - low_degree))
    print("r_t                         =", radius, float(radius))
    print("2 E[k_t]                    =", 2 * expected_long_edges, float(2 * expected_long_edges))
    print("tour-incidence slack        =", 2 * expected_long_edges - attachment - radius,
          float(2 * expected_long_edges - attachment - radius))


if __name__ == "__main__":
    main()
