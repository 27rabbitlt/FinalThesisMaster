#!/usr/bin/env python3
"""Exact arithmetic checks for the cyclic-cluster additive-hazard barrier.

The combinatorial part of the proof is symbolic and appears in the structural
memo.  This script verifies the numerical witness and the endpoint comparison
used when minimizing its tree lower bound.
"""

from fractions import Fraction


def main() -> None:
    h = 256
    a = h * h
    p = Fraction(1, a)
    s = 16 * a
    n = h * s

    # The exact rational base inequality implies that a whole cluster is empty
    # with probability e=((1-p)^a)^16 < 2^-16=p.  There is no need to form
    # the much larger exact rational e itself.
    one_block_empty = (Fraction(a - 1, a)) ** a
    assert one_block_empty < Fraction(1, 2)
    assert Fraction(1, 2) ** 16 == p

    # Every tree has excess at least the smaller of the two endpoint values
    # in the heavy/light-component argument.
    first_endpoint = Fraction(2 * p * n, 9)
    second_endpoint = Fraction(n, 2 * (3 * s + 1))
    coarse_tree_excess = Fraction(h, 8)
    assert first_endpoint >= coarse_tree_excess
    assert second_endpoint >= coarse_tree_excess

    # D < s/h^3 + h p + 3 = 3+17/h.  The strict inequalities come from
    # e<p, Q<=1, and dropping the negative term in the exact D formula.
    deficit_upper_bound = Fraction(3, 1) + Fraction(17, h)
    assert deficit_upper_bound < 4
    assert coarse_tree_excess > 10 * deficit_upper_bound

    print(f"h = {h}")
    print(f"s = {s}")
    print(f"number of clients = {n}")
    print(f"p = {p}")
    print(f"tree excess lower bound = {coarse_tree_excess}")
    print(f"strict deficit upper bound = {deficit_upper_bound}")
    print(
        "certified ratio lower bound >",
        float(coarse_tree_excess / deficit_upper_bound),
    )


if __name__ == "__main__":
    main()
