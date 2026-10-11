#!/usr/bin/env python3
"""Exact obstruction when monotonicity is removed from the three-gap lemma.

Two independent product bits induce three equiprobable, mutually exclusive
events.  Their indicators have identical marginals, but the maximum is one
surely, so the iid/associated largest-gap inequality fails.
"""

from fractions import Fraction


def main() -> None:
    p_u = Fraction(1, 3)
    p_v = Fraction(1, 2)
    probabilities = {
        (1, 0): p_u * (1 - p_v),
        (1, 1): p_u * p_v,
        (0, 1): (1 - p_u) * p_v,
        (0, 0): (1 - p_u) * (1 - p_v),
    }

    moments = [Fraction(0), Fraction(0), Fraction(0)]
    expected_maximum = Fraction(0)
    for (u, v), probability in probabilities.items():
        values = (int(u == 1), int(u == 0 and v == 1), int(u == 0 and v == 0))
        expected_maximum += probability * max(values)
        for index, value in enumerate(values):
            moments[index] += probability * value * value

    assert probabilities[(1, 0)] + probabilities[(1, 1)] == Fraction(1, 3)
    assert moments == [Fraction(1, 3)] * 3
    assert expected_maximum == 1
    gap = 2 * expected_maximum - (1 + 2 * moments[0])
    assert gap == Fraction(1, 3)

    print(f"E max = {expected_maximum}")
    print(f"E Xi^2 = {moments[0]} for every i")
    print(f"failure gap = {gap}")


if __name__ == "__main__":
    main()
