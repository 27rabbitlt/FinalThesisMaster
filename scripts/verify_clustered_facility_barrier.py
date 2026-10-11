#!/usr/bin/env python3
"""Exact certificate separating a facility surrogate from hazard trees.

Instance: 30 groups of 10 clients, every client active with probability 1/10;
within-group distance 1/1000, all other nonzero distances 1.  The accompanying
structural memo proves the formulas checked here.
"""

from fractions import Fraction


def main() -> None:
    groups = 30
    group_size = 10
    p = Fraction(1, group_size)
    delta = Fraction(1, 1000)
    empty_group = (1 - p) ** group_size
    occupied_group = 1 - empty_group

    posterior = (
        groups * occupied_group
        + 1
        - empty_group**groups
        + delta * groups * empty_group
    )
    connected_facility = Fraction(groups)

    # One representative per group is joined to the depot; its nine group
    # mates are leaves on length-delta edges.
    explicit_hazard_tree = groups * (
        occupied_group + delta * Fraction(group_size - 1, group_size)
    )

    assert connected_facility / posterior > Fraction(29, 20)
    assert explicit_hazard_tree < posterior

    # A compact rational comparison avoiding the 300th-power denominator.
    posterior_upper_bound = (
        31 - Fraction(2997, 100) * Fraction(9, 10) ** 10
    )
    exact_slack = Fraction(600, 29) - posterior_upper_bound
    assert posterior < posterior_upper_bound < Fraction(600, 29)
    assert exact_slack == Fraction(4046892644113, 29000000000000)

    print(f"P = {float(posterior):.15f}")
    print(f"G* = {connected_facility}")
    print(f"G*/P = {float(connected_facility / posterior):.15f} > 29/20")
    print(f"explicit Phi = {explicit_hazard_tree}")
    print(f"explicit Phi/P = {float(explicit_hazard_tree / posterior):.15f} < 1")
    print(f"compact exact slack below 600/29 = {exact_slack}")


if __name__ == "__main__":
    main()
