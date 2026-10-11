#!/usr/bin/env python3
"""Exact finite counterexample to the proposed retained-radius contraction.

Let the depot be at 0 and let the clients be the integer points 1,...,n on
the line, with their usual path metric.  Every client is active with the same
probability p.  For a first sample B, let C_b be the nearest-guide Voronoi
cell of b (ties are resolved by the smaller client index, except that a
client wins a tie against the depot), put

    rho_b = d(b, (B - {b}) union {0}),
    q_b   = 1 - product_{v in C_b} (1-p_v),

and define

    L   = E_B sum_{b in B} rho_b,
    L_T = E_B sum_{b in B} q_b rho_b,
    J   = sum_b p_b^2 E[rho_b | b in B].

The formerly proposed inequality was

    L_T <= (1 - 1/e) L + J.

For n=140 and p=1/20, this script disproves it in exact rational arithmetic.
It aggregates equal monomials in p and 1-p before evaluating them, so the
finite witness needs no floating-point computation and no enumeration of
2^140 scenarios.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, getcontext
from fractions import Fraction


def path_radius_terms(n: int, p: Fraction) -> tuple[Fraction, Fraction]:
    """Return (L,L_T) exactly for the rooted integer path.

    Conditional on b being a guide, ``left_gap`` and ``right_gap`` specify
    its nearest guide on each side.  The terminal value on the left denotes
    the depot; the terminal value on the right denotes that no guide occurs
    there.  The cell size follows immediately from the two midpoints.
    """

    z = 1 - p
    radius_coefficients: dict[tuple[int, int], int] = defaultdict(int)
    empty_radius_coefficients: dict[tuple[int, int], int] = defaultdict(int)

    # Client coordinates are 1,...,n.  The factor p for b in B is included
    # in p_power below.
    for b in range(1, n + 1):
        for left_gap in range(1, b + 1):
            if left_gap < b:
                # A client is the nearest guide on the left.  It wins a
                # midpoint tie because its index is smaller than b's.
                left_p_power = 1
                left_z_power = left_gap - 1
                cell_left = (left_gap - 1) // 2
            else:
                # All clients to the left are absent, so the depot is next.
                # At a midpoint tie b wins against the depot.
                left_p_power = 0
                left_z_power = b - 1
                cell_left = b // 2

            # The final right_gap value is a sentinel for no right guide.
            for right_gap in range(1, n - b + 2):
                no_right_guide = right_gap == n - b + 1
                if no_right_guide:
                    right_p_power = 0
                    right_z_power = n - b
                    cell_right = n - b
                    radius = min(left_gap, b)
                else:
                    # b wins a midpoint tie against a larger-index guide.
                    right_p_power = 1
                    right_z_power = right_gap - 1
                    cell_right = right_gap // 2
                    radius = min(left_gap, right_gap)

                p_power = 1 + left_p_power + right_p_power
                z_power = left_z_power + right_z_power
                key = (p_power, z_power)
                radius_coefficients[key] += radius

                cell_size = 1 + cell_left + cell_right
                empty_key = (p_power, z_power + cell_size)
                empty_radius_coefficients[empty_key] += radius

    radius = sum(
        coefficient * p**p_power * z**z_power
        for (p_power, z_power), coefficient in radius_coefficients.items()
    )
    empty_radius = sum(
        coefficient * p**p_power * z**z_power
        for (p_power, z_power), coefficient in empty_radius_coefficients.items()
    )
    return radius, radius - empty_radius


def direct_radius_terms(n: int, p: Fraction) -> tuple[Fraction, Fraction]:
    """Small-n scenario enumeration used to audit the gap formula."""

    z = 1 - p
    radius = Fraction(0)
    retained_radius = Fraction(0)
    for mask in range(1 << n):
        guide_probability = p ** mask.bit_count() * z ** (n - mask.bit_count())
        centers = [index + 1 for index in range(n) if mask & (1 << index)]
        available = [0, *centers]
        for center in centers:
            other = [point for point in available if point != center]
            local_radius = min(abs(center - point) for point in other)

            # A client beats the depot at equal distance.  Among clients,
            # the smaller coordinate wins, matching the closed-form rule.
            cell_size = 0
            for vertex in range(1, n + 1):
                image = min(
                    available,
                    key=lambda point: (
                        abs(vertex - point),
                        point == 0,
                        point,
                    ),
                )
                cell_size += image == center
            occupancy = 1 - z**cell_size
            radius += guide_probability * local_radius
            retained_radius += guide_probability * occupancy * local_radius
    return radius, retained_radius


def decimal(value: Fraction, places: int = 15) -> str:
    getcontext().prec = places + 10
    return str(Decimal(value.numerator) / Decimal(value.denominator))


def main() -> None:
    # First audit the gap calculation against literal scenario enumeration.
    audit_n = 7
    audit_p = Fraction(1, 5)
    assert path_radius_terms(audit_n, audit_p) == direct_radius_terms(
        audit_n, audit_p
    )

    n = 140
    p = Fraction(1, 20)
    radius, retained_radius = path_radius_terms(n, p)
    self_deficit = p * radius  # J=pL for a common activation probability.

    # The elementary series estimate
    #   e = sum 1/k! < 8/3 + (1/24)/(1-1/5) = 87/32
    # implies 1-1/e < 55/87.  It therefore suffices to verify the stronger
    # rational comparison below.
    rational_upper_bound = Fraction(55, 87)
    certified_gap = (
        retained_radius - rational_upper_bound * radius - self_deficit
    )
    assert certified_gap > 0

    # This particular path is far from the small-deficit regime.  Record that
    # it does not refute the repaired inequality with the full deficit D.
    z = 1 - p
    posterior = 2 * sum(1 - z**length for length in range(1, n + 1))
    sampled_radius = (1 - p) * radius
    deficit = posterior - sampled_radius
    assert retained_radius < deficit

    print("exact path-radius counterexample verified")
    print(f"n={n}, p={p}")
    print(f"L={decimal(radius)}")
    print(f"L_T={decimal(retained_radius)}")
    print(f"J/L={decimal(self_deficit / radius)}")
    print(f"L_T/L-J/L={decimal(retained_radius / radius - p)}")
    print(f"55/87={decimal(rational_upper_bound)}")
    print(f"certified rational gap={decimal(certified_gap)}")
    print(f"D={decimal(deficit)} (so this witness is not in the hard regime)")


if __name__ == "__main__":
    main()
