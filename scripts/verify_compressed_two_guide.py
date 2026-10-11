#!/usr/bin/env python3
"""Exact finite checks for the compressed two-guide construction.

For B,C drawn independently from the activation law, let f_B map every
client to a nearest guide center (or the depot), and put T=f_B(C).  The
checked master-route upper bound is

    E[tsp(T)] + 2 sum_v p_v E[d(v,T union {r})].

The guide and retained-center subsets are exhaustively enumerated and every
quantity is a Fraction.
"""

from __future__ import annotations

from fractions import Fraction


N = 8
P_ACTIVE = Fraction(1, 10)


def bits(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def subset_probability(mask: int) -> Fraction:
    return P_ACTIVE ** mask.bit_count() * (1 - P_ACTIVE) ** (N - mask.bit_count())


def subset_tsp_costs(distance: list[list[Fraction]]) -> list[Fraction]:
    root = N
    infinity = None
    path: list[list[Fraction | None]] = [[infinity] * N for _ in range(1 << N)]
    tours = [Fraction() for _ in range(1 << N)]
    for v in range(N):
        path[1 << v][v] = distance[root][v]
    for mask in range(1, 1 << N):
        for v in bits(mask):
            previous = mask ^ (1 << v)
            if previous:
                candidates = [
                    path[previous][u] + distance[u][v]
                    for u in bits(previous)
                    if path[previous][u] is not None
                ]
                path[mask][v] = min(candidates)
        tours[mask] = min(
            path[mask][v] + distance[v][root]
            for v in bits(mask)
            if path[mask][v] is not None
        )
    return tours


def evaluate(
    distance: list[list[Fraction]],
) -> tuple[
    Fraction,
    Fraction,
    Fraction,
    Fraction,
    Fraction,
    Fraction,
    Fraction,
]:
    root = N
    costs = subset_tsp_costs(distance)
    posterior = sum(
        (subset_probability(mask) * costs[mask] for mask in range(1 << N)),
        Fraction(),
    )
    compressed_tour = Fraction()
    attachment = Fraction()
    first_radius = Fraction()
    empty_cell_transport = Fraction()
    triangle_slack = Fraction()
    retained_nearest = Fraction()
    for guide in range(1 << N):
        guide_probability = subset_probability(guide)
        centers = tuple(bits(guide))
        available = (*centers, root)
        image = [
            v
            if guide & (1 << v)
            else min(available, key=lambda u: (distance[v][u], u))
            for v in range(N)
        ]
        first_radius += guide_probability * sum(
            (P_ACTIVE * distance[v][image[v]] for v in range(N)), Fraction()
        )
        cell_weight = {
            center: sum(
                (P_ACTIVE for v in range(N) if image[v] == center), Fraction()
            )
            for center in centers
        }
        occupancy: dict[int, Fraction] = {}
        for center in centers:
            empty = Fraction(1)
            for v in range(N):
                if image[v] == center:
                    empty *= 1 - P_ACTIVE
            occupancy[center] = 1 - empty

        retained = guide
        while True:
            conditional_probability = Fraction(1)
            for center in centers:
                conditional_probability *= (
                    occupancy[center]
                    if retained & (1 << center)
                    else 1 - occupancy[center]
                )
            weight = guide_probability * conditional_probability
            compressed_tour += weight * costs[retained]
            selected = (root, *bits(retained))
            assignment_value = sum(
                (
                    P_ACTIVE * min(distance[v][u] for u in selected)
                    for v in range(N)
                ),
                Fraction(),
            )
            attachment += weight * assignment_value
            transport_value = sum(
                (
                    cell_weight[center]
                    * min(distance[center][u] for u in selected)
                    for center in centers
                    if not retained & (1 << center)
                ),
                Fraction(),
            )
            empty_cell_transport += weight * transport_value
            slack_value = sum(
                (
                    P_ACTIVE
                    * (
                        distance[v][image[v]]
                        + min(distance[image[v]][u] for u in selected)
                        - min(distance[v][u] for u in selected)
                    )
                    for v in range(N)
                ),
                Fraction(),
            )
            triangle_slack += weight * slack_value
            nearest_value = sum(
                (
                    min(
                        distance[center][u]
                        for u in selected
                        if u != center
                    )
                    for center in bits(retained)
                ),
                Fraction(),
            )
            retained_nearest += weight * nearest_value
            if retained == 0:
                break
            retained = (retained - 1) & guide
    assert attachment == first_radius + empty_cell_transport - triangle_slack
    assert empty_cell_transport <= retained_nearest
    assert retained_nearest <= compressed_tour
    return (
        posterior,
        compressed_tour,
        attachment,
        first_radius,
        empty_cell_transport,
        triangle_slack,
        retained_nearest,
    )


def line_metric() -> list[list[Fraction]]:
    coordinates = tuple(range(1, N + 1)) + (0,)
    return [
        [Fraction(abs(coordinates[i] - coordinates[j])) for j in range(N + 1)]
        for i in range(N + 1)
    ]


def cycle_metric() -> list[list[Fraction]]:
    size = N + 1
    return [
        [Fraction(min(abs(i - j), size - abs(i - j))) for j in range(size)]
        for i in range(size)
    ]


def depot_star_metric() -> list[list[Fraction]]:
    root = N
    return [
        [
            Fraction(0)
            if i == j
            else Fraction(1)
            if i == root or j == root
            else Fraction(2)
            for j in range(N + 1)
        ]
        for i in range(N + 1)
    ]


def main() -> None:
    target = Fraction(29, 10)
    for name, distance in (
        ("line", line_metric()),
        ("cycle", cycle_metric()),
        ("depot-star", depot_star_metric()),
    ):
        (
            posterior,
            tour,
            attachment,
            radius,
            transport,
            slack,
            retained_nearest,
        ) = evaluate(distance)
        bound = tour + 2 * attachment
        assert bound < target * posterior
        print(
            f"{name}: P={float(posterior):.12f}, "
            f"compressed={float(bound):.12f}, "
            f"ratio={float(bound / posterior):.12f}, "
            f"A=R+H-S={float(radius):.12f}+{float(transport):.12f}"
            f"-{float(slack):.12f}, nearest={float(retained_nearest):.12f}"
        )
    print("all exact Fraction checks passed")


if __name__ == "__main__":
    main()
