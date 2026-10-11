#!/usr/bin/env python3
"""Exact audit of the 1024-client geometric-bucket line obstruction.

All proof comparisons use ``fractions.Fraction``.  Floating-point values are
printed only as a readable summary.
"""

from fractions import Fraction as Q


def expected_span(n: int, rate: Q) -> Q:
    """Expected span of an iid rate-``rate`` subset of {1,...,n}."""
    z = 1 - rate
    # A span crosses gap k iff both disjoint sides contain a sample.
    gap_sum = sum(
        ((1 - z**k) * (1 - z ** (n - k)) for k in range(1, n)),
        Q(0),
    )
    closed_form = (
        n - 1
        - 2 * z * (1 - z ** (n - 1)) / rate
        + (n - 1) * z**n
    )
    assert gap_sum == closed_form
    assert closed_form >= n + 1 - 2 / rate
    return closed_form


def main() -> None:
    n = 1024
    p = Q(1, 2)

    # A client is both evaluation-active and assigned to level ell with rate
    # p * [p(1-p)^(ell-1)] = 2^(-(ell+1)).
    first_seven = sum(
        (expected_span(n, p * p * (1 - p) ** (ell - 1)) for ell in range(1, 8)),
        Q(0),
    )
    elementary_lower_bound = Q(7 * n - 1009)
    assert first_seven >= elementary_lower_bound

    # The posterior rooted line tour costs twice its largest active point.
    posterior = 2 * (n - 1 + Q(1, 2**n))

    # Every generated stopped tour pays at least the sum of its within-bucket
    # active spans, independently of all posterior-optimal tie-breaking.
    assert elementary_lower_bound > 3 * posterior
    assert first_seven > 3 * posterior

    print("n:", n)
    print("posterior:", float(posterior))
    print("elementary seven-level lower bound:", float(elementary_lower_bound))
    print("exact seven-level span sum:", float(first_seven))
    print("certified ratio lower bound:", float(first_seven / posterior))
    print("1024-client geometric-bucket obstruction: passed")


if __name__ == "__main__":
    main()
