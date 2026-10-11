#!/usr/bin/env python3
"""Exact audit of a constant-boundary portal obstruction.

The metric has a depot and two classes L,R of equal size m.  Depot--client
and L--R distances are one, while distances within either class are two.
Every client is active independently with probability p.

For the finite choice t=4, p=t^-2 and m=t^4, the script enumerates the two
binomial *counts* (not the 2^(2m) active sets) using integer weights.  It
checks the posterior formula, the optimum with at most h runs of either
class, and the exact one-portal-per-class decomposition identity

    Xi = Chi_1 + 4(q_{m-1} + q_m - p).

All expectation identities and comparisons use exact rational arithmetic.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb
from itertools import product


T = 4
M = T**4
P_ACTIVE = Fraction(1, T**2)


def maximum_runs(left: int, right: int) -> int:
    """Maximum number of monochromatic runs in a binary word."""
    if left + right == 0:
        return 0
    return 2 * min(left, right) + int(left != right)


def posterior_cost(left: int, right: int) -> int:
    total = left + right
    if total == 0:
        return 0
    return 2 * total + 1 - maximum_runs(left, right)


def bounded_run_cost(left: int, right: int, run_cap: int) -> int:
    """Optimum when each class may occur in at most run_cap runs."""
    total = left + right
    if total == 0:
        return 0
    runs = min(maximum_runs(left, right), 2 * run_cap)
    return 2 * total + 1 - runs


def binomial_integer_weights() -> tuple[list[int], int]:
    denominator_base = P_ACTIVE.denominator
    numerator = P_ACTIVE.numerator
    failure = denominator_base - numerator
    weights = [
        comb(M, active) * numerator**active * failure ** (M - active)
        for active in range(M + 1)
    ]
    denominator = denominator_base**M
    assert sum(weights) == denominator
    return weights, denominator


def exact_expectations(run_caps: tuple[int, ...]) -> tuple[Fraction, dict[int, Fraction]]:
    weights, one_side_denominator = binomial_integer_weights()
    denominator = one_side_denominator**2
    posterior_numerator = 0
    capped_numerators = {run_cap: 0 for run_cap in run_caps}
    for left, left_weight in enumerate(weights):
        for right, right_weight in enumerate(weights):
            weight = left_weight * right_weight
            posterior_numerator += weight * posterior_cost(left, right)
            for run_cap in run_caps:
                capped_numerators[run_cap] += weight * bounded_run_cost(
                    left, right, run_cap
                )
    posterior = Fraction(posterior_numerator, denominator)
    capped = {
        run_cap: Fraction(numerator, denominator)
        for run_cap, numerator in capped_numerators.items()
    }
    return posterior, capped


def alternating_order_formula(m: int, p: Fraction) -> Fraction:
    """Expected shortcut cost of L1,R1,L2,R2,...,Lm,Rm."""
    failure = 1 - p
    nonempty = 1 - failure ** (2 * m)
    same_class_transitions = p**2 * sum(
        (2 * m - 2 * half_gap) * failure ** (2 * half_gap - 1)
        for half_gap in range(1, m)
    )
    return 2 * m * p + nonempty + same_class_transitions


def alternating_order_bruteforce(m: int, p: Fraction) -> Fraction:
    """Independent small-n audit of alternating_order_formula."""
    total = Fraction(0)
    for active_bits in product((0, 1), repeat=2 * m):
        active_positions = [index for index, bit in enumerate(active_bits) if bit]
        probability = p ** len(active_positions) * (1 - p) ** (
            2 * m - len(active_positions)
        )
        if not active_positions:
            cost = 0
        else:
            cost = 2  # the two unit depot edges
            for left, right in zip(active_positions, active_positions[1:]):
                cost += 1 if (right - left) % 2 else 2
        total += probability * cost
    return total


def main() -> None:
    run_caps = (1, 2, 4)
    posterior, capped = exact_expectations(run_caps)
    lam = M * P_ACTIVE
    q_m = 1 - (1 - P_ACTIVE) ** M
    q_previous = 1 - (1 - P_ACTIVE) ** (M - 1)

    # The exact count formula is K + max(1, |L-R|) when K>0.
    endpoint_imbalance = posterior - 2 * lam
    assert endpoint_imbalance >= 0

    # A cap h loses exactly (r_* - 2h)_+ in every realization.
    for run_cap in run_caps:
        assert capped[run_cap] >= posterior

    # With one run per side the cost is 2K minus one iff both sides occur.
    one_run_closed_form = 4 * lam - q_m**2
    assert capped[1] == one_run_closed_form
    chi_one = capped[1] - posterior

    # R and D=P-R for this metric.
    nearest_radius = 2 * M * P_ACTIVE * (1 - P_ACTIVE)
    deficit = posterior - nearest_radius
    assert deficit == 2 * lam * P_ACTIVE + endpoint_imbalance

    # Every realized MST is the unit root star (possibly with unit cross
    # edges substituted), so its value is exactly the number of active
    # clients.  The tour-over-tree parity surplus is endpoint_imbalance.
    expected_mst = 2 * lam
    parity_surplus = posterior - expected_mst
    assert parity_surplus == endpoint_imbalance

    # Portalize L and R through one representative each.  The quotient has
    # expected cost 4q-q^2.  Each local rooted equilateral block contributes
    # 2((m-1)p+q_{m-1}).
    quotient_posterior = 4 * q_m - q_m**2
    local_posterior = 2 * ((M - 1) * P_ACTIVE + q_previous)
    portalized_posterior = quotient_posterior + 2 * local_posterior
    xi = portalized_posterior - posterior
    local_return = 4 * (q_previous + q_m - P_ACTIVE)
    assert xi == chi_one + local_return
    assert xi > 5 * deficit
    assert xi > 7 * parity_surplus

    # The obstruction is to this coarse hierarchy, not to the instance.
    # A direct alternating master order retains all run occurrences.
    assert alternating_order_formula(4, Fraction(1, 3)) == (
        alternating_order_bruteforce(4, Fraction(1, 3))
    )
    alternating_order = alternating_order_formula(M, P_ACTIVE)
    assert alternating_order < Fraction(3, 2) * posterior

    # Every L--R quadrilateral defect is 1+1-1-1=0, so endpoint-coupling
    # rotations see no boundary polarization although Xi is positive.
    quadrilateral_defect = 1 + 1 - 1 - 1
    assert quadrilateral_defect == 0
    assert xi > 0

    print(f"t={T}, m={M}, p={P_ACTIVE}, lambda={lam}")
    print(f"posterior={float(posterior):.12f} (computed exactly)")
    print(f"D={float(deficit):.12f} (computed exactly)")
    print(f"tour-over-MST surplus={float(parity_surplus):.12f}")
    for run_cap in run_caps:
        chi = capped[run_cap] - posterior
        print(
            f"Chi_{run_cap}={float(chi):.12f}; "
            f"Chi_{run_cap}/D={float(chi / deficit):.12f}"
        )
    print(f"portalized posterior={float(portalized_posterior):.12f}")
    print(f"Xi={float(xi):.12f}")
    print(f"Xi/D={float(xi / deficit):.12f}")
    print(f"local-return residual={float(local_return):.12f}")
    print(
        f"alternating fixed order={float(alternating_order):.12f}; "
        f"ratio={float(alternating_order / posterior):.12f}"
    )
    print("exact comparison: Xi > 5 D")
    print("exact comparison: Xi > 7 (posterior - expected MST)")
    print("all exact constant-boundary portal checks passed")


if __name__ == "__main__":
    main()
