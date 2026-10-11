#!/usr/bin/env python3
"""Exact one-center counterexample to the conditional M+2H bound."""

from fractions import Fraction


def main() -> None:
    # One guide b at distance one from the depot.  Its cell contains two
    # labels of activation probability 3/4, so w=sum p and q=Pr(C hits cell).
    probability = Fraction(3, 4)
    weight = 2 * probability
    retention = 1 - (1 - probability) ** 2
    tour_b = Fraction(2)
    expected_core_tour = retention * tour_b
    expected_empty_attachment = weight * (1 - retention)

    assert weight * (1 - retention) <= retention
    assert expected_core_tour == Fraction(15, 8)
    assert expected_empty_attachment == Fraction(3, 32)
    assert expected_core_tour + 2 * expected_empty_attachment == Fraction(33, 16)
    assert expected_core_tour + 2 * expected_empty_attachment > tour_b

    print("q =", retention)
    print("w =", weight)
    print("M =", expected_core_tour)
    print("H =", expected_empty_attachment)
    print("M+2H =", expected_core_tour + 2 * expected_empty_attachment)
    print("tsp(B) =", tour_b)


if __name__ == "__main__":
    main()
