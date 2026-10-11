#!/usr/bin/env python3
"""Exact counterexample to the scale-wise compressed-tour inequality.

There are seven root-near vertices u_i forming a short clique and one
root-far vertex v.  Every non-guide u_i strictly prefers the depot, although
u_i is threshold-adjacent to v and to every other u_j.  Consequently the
second sample can escape to the depot while a posterior tour has no long
edge as soon as v and at least two u_i are active.

The script uses Fraction arithmetic and independently enumerates every B,C
support and computes all subset tour defects by Held--Karp dynamic programming.
"""

from __future__ import annotations

from fractions import Fraction


M = 7
V = M
ROOT = M + 1
N = M + 1
P_U = Fraction(1, 2)
P_V = Fraction(1, 1000)
THRESHOLD = Fraction(3, 2)


def distance(x: int, y: int) -> Fraction:
    if x == y:
        return Fraction(0)
    if x == ROOT or y == ROOT:
        other = y if x == ROOT else x
        return Fraction(11, 5) if other == V else Fraction(1)
    return Fraction(6, 5)


P = (P_U,) * M + (P_V,)


def mask_probability(mask: int) -> Fraction:
    answer = Fraction(1)
    for x, probability in enumerate(P):
        answer *= probability if mask >> x & 1 else 1 - probability
    return answer


def projection(guide: int, sample: int) -> int:
    centers = [x for x in range(N) if guide >> x & 1] + [ROOT]
    retained = 0
    for x in range(N):
        if not (sample >> x & 1):
            continue
        image = x if guide >> x & 1 else min(
            centers, key=lambda center: (distance(x, center), center)
        )
        if image != ROOT:
            retained |= 1 << image
    return retained


def projection_rootless(guide: int, sample: int) -> int:
    """Nearest-guide projection with the depot excluded when guide is nonempty."""
    if guide == 0:
        return 0
    centers = [x for x in range(N) if guide >> x & 1]
    retained = 0
    for x in range(N):
        if sample >> x & 1:
            image = min(centers, key=lambda center: (distance(x, center), center))
            retained |= 1 << image
    return retained


def uncovered_v(target: int) -> int:
    return int(all(distance(V, x) > THRESHOLD for x in range(N) if target >> x & 1))


def all_tour_defects() -> list[int]:
    """Minimum number of threshold-long edges for every client subset."""
    infinity = 10**9
    answer = [0] * (1 << N)
    for mask in range(1, 1 << N):
        dp: dict[tuple[int, int], int] = {}
        for last in range(N):
            if mask >> last & 1:
                dp[(1 << last, last)] = int(distance(ROOT, last) > THRESHOLD)
        submasks = [submask for submask in range(1, 1 << N) if submask & ~mask == 0]
        for submask in sorted(submasks, key=int.bit_count):
            if submask & (submask - 1):
                for last in range(N):
                    if not (submask >> last & 1):
                        continue
                    previous = submask ^ (1 << last)
                    dp[(submask, last)] = min(
                        dp[(previous, before)]
                        + int(distance(before, last) > THRESHOLD)
                        for before in range(N)
                        if previous >> before & 1
                    )
        answer[mask] = min(
            dp[(mask, last)] + int(distance(last, ROOT) > THRESHOLD)
            for last in range(N)
            if mask >> last & 1
        )
    return answer


def all_tour_costs() -> list[Fraction]:
    infinity = Fraction(10**9)
    answer = [Fraction(0)] * (1 << N)
    for mask in range(1, 1 << N):
        dp: dict[tuple[int, int], Fraction] = {}
        for last in range(N):
            if mask >> last & 1:
                dp[(1 << last, last)] = distance(ROOT, last)
        submasks = [submask for submask in range(1, 1 << N) if submask & ~mask == 0]
        for submask in sorted(submasks, key=int.bit_count):
            if submask & (submask - 1):
                for last in range(N):
                    if not (submask >> last & 1):
                        continue
                    previous = submask ^ (1 << last)
                    dp[(submask, last)] = min(
                        dp.get((previous, before), infinity) + distance(before, last)
                        for before in range(N)
                        if previous >> before & 1
                    )
        answer[mask] = min(
            dp[(mask, last)] + distance(last, ROOT)
            for last in range(N)
            if mask >> last & 1
        )
    return answer


def assignment_cost(target: int) -> Fraction:
    centers = [x for x in range(N) if target >> x & 1] + [ROOT]
    return sum(P[x] * min(distance(x, center) for center in centers) for x in range(N))


def main() -> None:
    weights = [mask_probability(mask) for mask in range(1 << N)]
    attachment = sum(
        weights[b] * weights[c] * P_V * uncovered_v(projection(b, c))
        for b in range(1 << N)
        for c in range(1 << N)
    )
    radius = sum(weights[b] * P_V * uncovered_v(b) for b in range(1 << N))
    defects = all_tour_defects()
    expected_defect = sum(weights[b] * defects[b] for b in range(1 << N))
    tour_costs = all_tour_costs()
    posterior = sum(weights[b] * tour_costs[b] for b in range(1 << N))
    radius_cost = sum(weights[b] * assignment_cost(b) for b in range(1 << N))
    compressed_tour = Fraction(0)
    compressed_attachment = Fraction(0)
    rootless_tour = Fraction(0)
    rootless_defect = Fraction(0)
    rootless_attachment = Fraction(0)
    rootless_threshold_attachment = Fraction(0)
    rootless_raw_transport = Fraction(0)
    hybrid_route = Fraction(0)
    hybrid_with_guide_route = Fraction(0)
    for b in range(1 << N):
        rootless_centers = [x for x in range(N) if b >> x & 1]
        rootless_cell_weight = {x: Fraction(0) for x in rootless_centers}
        if rootless_centers:
            for x in range(N):
                center = min(rootless_centers, key=lambda y: (distance(x, y), y))
                rootless_cell_weight[center] += P[x]
        for c in range(1 << N):
            weight = weights[b] * weights[c]
            core = projection(b, c)
            rootful_route = tour_costs[core] + 2 * assignment_cost(core)
            compressed_tour += weight * tour_costs[core]
            compressed_attachment += weight * assignment_cost(core)
            rootless_core = projection_rootless(b, c)
            rootless_route = tour_costs[rootless_core] + 2 * assignment_cost(rootless_core)
            rootless_tour += weight * tour_costs[rootless_core]
            rootless_defect += weight * defects[rootless_core]
            rootless_attachment += weight * assignment_cost(rootless_core)
            rootless_threshold_attachment += weight * P_V * uncovered_v(rootless_core)
            hybrid_route += weight * min(rootful_route, rootless_route)
            hybrid_with_guide_route += weight * min(
                rootful_route,
                rootless_route,
                tour_costs[b] + 2 * assignment_cost(b),
            )
            selected = [ROOT] + [x for x in range(N) if rootless_core >> x & 1]
            rootless_raw_transport += weight * sum(
                rootless_cell_weight[x]
                * min(distance(x, y) for y in selected)
                for x in rootless_centers
                if not rootless_core >> x & 1
            )

    q = (1 - P_U) ** M
    no_diagonal = (1 - P_U**2) ** M
    exactly_one = M * P_U * (1 - P_U) ** (M - 1)
    attachment_formula = P_V * (1 - P_V) * (no_diagonal + P_V * q)
    radius_formula = P_V * (1 - P_V) * q
    defect_formula = P_V * (2 * q + exactly_one)
    ell = Fraction(6, 5)
    diagonal_empty = no_diagonal
    diagonal_one = M * P_U**2 * (1 - P_U**2) ** (M - 1)
    posterior_formula = (
        (1 - P_V)
        * (ell * M * P_U + (2 - ell) * (1 - q))
        + P_V
        * (2 + ell * M * P_U + 2 * ell * q + ell * exactly_one)
    )
    radius_cost_formula = (
        M * P_U * (1 - P_U)
        + P_V * (1 - P_V) * (ell + q)
    )
    diagonal_tour = ell * M * P_U**2 + (2 - ell) * (1 - diagonal_empty)
    escaped_tour = ell * M * P_U**2 + (1 - q) * (2 - ell * P_U)
    diagonal_tour_with_v = (
        2
        + ell * M * P_U**2
        + 2 * ell * diagonal_empty
        + ell * diagonal_one
    )
    compressed_tour_formula = (
        (1 - P_V) * diagonal_tour
        + P_V * (1 - P_V) * escaped_tour
        + P_V**2 * diagonal_tour_with_v
    )
    compressed_attachment_formula = (
        M * P_U * (1 - P_U**2)
        - P_V
        * P_U
        * (1 - P_V)
        * (1 - q)
        * (1 - P_U)
        + P_V
        * (
            ell * (1 - P_V**2)
            + (1 - P_V) * diagonal_empty
            + P_V * (1 - P_V) * q
        )
    )
    empty_sample = (1 - P_V) * q
    rootless_threshold_formula = P_V * (2 * empty_sample - empty_sample**2)
    assert attachment == attachment_formula
    assert radius == radius_formula
    assert expected_defect == defect_formula
    assert posterior == posterior_formula
    assert radius_cost == radius_cost_formula
    assert compressed_tour == compressed_tour_formula
    assert compressed_attachment == compressed_attachment_formula, (
        compressed_attachment,
        compressed_attachment_formula,
        compressed_attachment - compressed_attachment_formula,
    )
    assert rootless_threshold_attachment == rootless_threshold_formula
    assert attachment + radius > 2 * expected_defect
    assert rootless_threshold_attachment + radius <= 2 * expected_defect

    gamma = posterior - compressed_tour + 2 * (posterior - compressed_attachment)

    print("a_t                     =", attachment, float(attachment))
    print("r_t                     =", radius, float(radius))
    print("2 E[k_t]                =", 2 * expected_defect, float(2 * expected_defect))
    print("strict violation        =", attachment + radius - 2 * expected_defect,
          float(attachment + radius - 2 * expected_defect))
    print("P                       =", posterior, float(posterior))
    print("R                       =", radius_cost, float(radius_cost))
    print("M1                      =", compressed_tour, float(compressed_tour))
    print("A1                      =", compressed_attachment, float(compressed_attachment))
    print("A1 + R - 2P             =", compressed_attachment + radius_cost - 2 * posterior,
          float(compressed_attachment + radius_cost - 2 * posterior))
    print("Gamma1                  =", gamma, float(gamma))
    print("Gamma1/P                =", float(gamma / posterior))
    print("(M1 + 2A1)/P            =", float((compressed_tour + 2 * compressed_attachment) / posterior))
    print("(P + 2R)/P              =", float((posterior + 2 * radius_cost) / posterior))
    rootless_gamma = posterior - rootless_tour + 2 * (posterior - rootless_attachment)
    print("rootless a_t            =", rootless_threshold_attachment,
          float(rootless_threshold_attachment))
    print("rootless a_t+r_t-2E[k]  =",
          rootless_threshold_attachment + radius - 2 * expected_defect,
          float(rootless_threshold_attachment + radius - 2 * expected_defect))
    print("rootless M1             =", rootless_tour, float(rootless_tour))
    print("rootless E[k(T)]        =", rootless_defect, float(rootless_defect))
    print("rootless A1             =", rootless_attachment, float(rootless_attachment))
    print("rootless raw H          =", rootless_raw_transport,
          float(rootless_raw_transport))
    print("rootless M1+2H-P        =",
          rootless_tour + 2 * rootless_raw_transport - posterior,
          float(rootless_tour + 2 * rootless_raw_transport - posterior))
    print("rootless Gamma1/P       =", float(rootless_gamma / posterior))
    print("rootless route/P        =", float((rootless_tour + 2 * rootless_attachment) / posterior))
    print("pairwise hybrid/P       =", float(hybrid_route / posterior))
    print("hybrid incl. guide/P    =", float(hybrid_with_guide_route / posterior))
    coupled_scale = (
        3 * expected_defect
        - rootless_defect
        - 2 * rootless_threshold_attachment
        - float(radius) / 2.718281828459045
    )
    print("rootless coupled scale  =", coupled_scale)


if __name__ == "__main__":
    main()
