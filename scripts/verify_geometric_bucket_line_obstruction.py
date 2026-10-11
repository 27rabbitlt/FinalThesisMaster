#!/usr/bin/env python3
"""Exact arithmetic audit of the geometric-bucket line obstruction.

The proof itself is in ``oriented_arc_projectivity.md``.  This checker uses
the second Bonferroni inequality rather than expanding enormous binomial
powers, so the finite witness with 2**50 clients is certified instantly.
"""

from fractions import Fraction as F


def main() -> None:
    levels = 47
    n = 1 << (levels + 3)
    quarter = n // 4
    range_sum_lower = F(0)

    for level in range(1, levels + 1):
        # With guide and evaluation probabilities both 1/2, a fixed vertex
        # is evaluation-active in bucket level j with beta=2^(-(j+1)).
        support = 1 << (level + 1)
        beta = F(1, support)
        assert quarter >= support

        # Look at only support vertices in each end quarter.  The second
        # Bonferroni inequality gives P(at least one success) at least
        # M beta - C(M,2) beta^2 = 1/2 + 1/(2M).
        hit_lower = support * beta - F(support * (support - 1), 2) * beta * beta
        assert hit_lower == F(1, 2) + F(1, 2 * support)
        assert hit_lower > F(1, 2)

        # The two end quarters are independent; hitting both forces range
        # at least n/2 inside this bucket's active restriction.
        range_sum_lower += F(n, 2) * hit_lower * hit_lower

    posterior_upper = 2 * n
    assert range_sum_lower > F(levels * n, 8)
    assert 10 * range_sum_lower > 29 * posterior_upper

    print("exact geometric-bucket line obstruction passed")
    print("  clients:", n)
    print("  certified ratio lower bound:", float(range_sum_lower / posterior_upper))
    print("  simple symbolic lower bound:", F(levels, 16))


if __name__ == "__main__":
    main()
