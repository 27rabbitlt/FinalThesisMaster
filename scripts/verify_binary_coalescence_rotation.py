#!/usr/bin/env python3
"""Exact audit of the binary-coalescence rotation identities BR1--BR29.

The checker evaluates the recursion (11)--(13) in Fraction arithmetic and
independently enumerates every fair child orientation and activation set.
It then verifies the three-block rotation formula, its endpoint displacement,
the depth-independent bound on all ancestor and depot corrections, the
endpoint-neutral fair and globally correlated quadrilateral rotations, and
the alternating transport-cycle triangulation.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import product


PROBABILITY = (
    Fraction(2, 5),
    Fraction(1, 3),
    Fraction(3, 7),
    Fraction(1, 2),
    Fraction(4, 9),
    Fraction(5, 11),
)
POSITION = (
    Fraction(1),
    Fraction(7),
    Fraction(3),
    Fraction(10),
    Fraction(5),
    Fraction(12),
)
ROOT_POSITION = Fraction(0)


def distance(i: int, j: int) -> Fraction:
    left = ROOT_POSITION if i == -1 else POSITION[i]
    right = ROOT_POSITION if j == -1 else POSITION[j]
    return abs(left - right)


def occupancy(vertices: frozenset[int]) -> Fraction:
    empty = Fraction(1)
    for vertex in vertices:
        empty *= 1 - PROBABILITY[vertex]
    return 1 - empty


def endpoint_distance(left: dict[int, Fraction], right: dict[int, Fraction]) -> Fraction:
    return sum(
        left_mass * right_mass * distance(left_vertex, right_vertex)
        for left_vertex, left_mass in left.items()
        for right_vertex, right_mass in right.items()
    )


def data(tree):
    """Return (leaf set, occupancy, conditional endpoint law, open cost)."""
    if isinstance(tree, int):
        return frozenset((tree,)), PROBABILITY[tree], {tree: Fraction(1)}, Fraction(0)
    left, right = tree
    left_set, q_left, nu_left, ell_left = data(left)
    right_set, q_right, nu_right, ell_right = data(right)
    q_union = q_left + q_right - q_left * q_right
    endpoint = {}
    for vertex in left_set | right_set:
        endpoint[vertex] = (
            q_left * (2 - q_right) * nu_left.get(vertex, 0)
            + q_right * (2 - q_left) * nu_right.get(vertex, 0)
        ) / (2 * q_union)
    ell = (
        ell_left
        + ell_right
        + q_left * q_right * endpoint_distance(nu_left, nu_right)
    )
    return left_set | right_set, q_union, endpoint, ell


def recursive_tour_value(tree) -> Fraction:
    _, q_value, endpoint, ell = data(tree)
    return ell + 2 * q_value * sum(
        mass * distance(-1, vertex) for vertex, mass in endpoint.items()
    )


def order_distribution(tree) -> dict[tuple[int, ...], Fraction]:
    if isinstance(tree, int):
        return {(tree,): Fraction(1)}
    left, right = tree
    result: defaultdict[tuple[int, ...], Fraction] = defaultdict(Fraction)
    for left_order, left_mass in order_distribution(left).items():
        for right_order, right_mass in order_distribution(right).items():
            mass = left_mass * right_mass / 2
            result[left_order + right_order] += mass
            result[right_order + left_order] += mass
    return dict(result)


def direct_tour_value(tree) -> Fraction:
    result = Fraction(0)
    for order, order_mass in order_distribution(tree).items():
        for bits in product((0, 1), repeat=len(PROBABILITY)):
            active_probability = Fraction(1)
            active = []
            for vertex, bit in enumerate(bits):
                active_probability *= (
                    PROBABILITY[vertex] if bit else 1 - PROBABILITY[vertex]
                )
                if bit:
                    active.append(vertex)
            active_set = set(active)
            word = [vertex for vertex in order if vertex in active_set]
            if not word:
                cost = Fraction(0)
            else:
                cost = (
                    distance(-1, word[0])
                    + sum(distance(x, y) for x, y in zip(word, word[1:]))
                    + distance(word[-1], -1)
                )
            result += order_mass * active_probability * cost
    return result


def signed_difference(
    left: dict[int, Fraction], right: dict[int, Fraction]
) -> dict[int, Fraction]:
    return {
        vertex: left.get(vertex, 0) - right.get(vertex, 0)
        for vertex in set(left) | set(right)
        if left.get(vertex, 0) != right.get(vertex, 0)
    }


def mixture(
    weight: Fraction,
    first: dict[int, Fraction],
    second: dict[int, Fraction],
) -> dict[int, Fraction]:
    return {
        vertex: weight * first.get(vertex, 0)
        + (1 - weight) * second.get(vertex, 0)
        for vertex in set(first) | set(second)
    }


def globally_reversed_order_data(
    order: tuple[int, ...], probabilities: dict[int, Fraction]
) -> tuple[Fraction, dict[int, Fraction], Fraction]:
    """Exact open cost and endpoint law of an order mixed with its reversal."""
    unnormalized_endpoint: defaultdict[int, Fraction] = defaultdict(Fraction)
    open_cost = Fraction(0)
    occupied_probability = Fraction(0)
    for bits in product((0, 1), repeat=len(order)):
        probability = Fraction(1)
        active = []
        for vertex, bit in zip(order, bits):
            q_vertex = probabilities[vertex]
            probability *= q_vertex if bit else 1 - q_vertex
            if bit:
                active.append(vertex)
        if not active:
            continue
        occupied_probability += probability
        unnormalized_endpoint[active[0]] += probability / 2
        unnormalized_endpoint[active[-1]] += probability / 2
        open_cost += probability * sum(
            distance(x, y) for x, y in zip(active, active[1:])
        )
    endpoint = {
        vertex: mass / occupied_probability
        for vertex, mass in unnormalized_endpoint.items()
    }
    return occupied_probability, endpoint, open_cost


def main() -> None:
    a, u, v, w_1, w_2, w_3 = range(6)
    x_1 = ((a, u), v)
    x_2 = ((a, v), u)
    hierarchy_1 = (((x_1, w_1), w_2), w_3)
    hierarchy_2 = (((x_2, w_1), w_2), w_3)

    # Independently audit the complete recursion against all orientations and
    # all activation sets.
    value_1 = recursive_tour_value(hierarchy_1)
    value_2 = recursive_tour_value(hierarchy_2)
    assert value_1 == direct_tour_value(hierarchy_1)
    assert value_2 == direct_tour_value(hierarchy_2)

    q_a, q_u, q_v = (PROBABILITY[index] for index in (a, u, v))
    _, _, _, ell_x_1 = data(x_1)
    _, _, _, ell_x_2 = data(x_2)
    base = (
        q_a
        * q_u
        * q_v
        * (distance(a, u) - distance(a, v))
        / 2
    )
    assert ell_x_1 - ell_x_2 == base

    _, q_x, nu_x_1, _ = data(x_1)
    _, _, nu_x_2, _ = data(x_2)
    endpoint_shift = signed_difference(nu_x_1, nu_x_2)
    scale = q_a * q_u * q_v / (4 * q_x)
    expected_shift = {u: -scale, v: scale}
    assert endpoint_shift == expected_shift

    outside = (value_1 - value_2) - base
    outside_bound = q_a * q_u * q_v * distance(u, v) / 2
    assert abs(outside) <= outside_bound

    siblings = (PROBABILITY[w_1], PROBABILITY[w_2], PROBABILITY[w_3])
    multiplier = Fraction(1)
    telescoping = Fraction(0)
    for sibling in siblings:
        telescoping += multiplier * sibling
        multiplier *= 1 - sibling / 2
    telescoping += 2 * multiplier
    assert telescoping == 2

    # Four-block endpoint-neutral rotation.  The two contexts a_0,a_1 may
    # have unequal occupancies; the mixing weight is chosen so that the
    # signed U_1-U_0 endpoint displacement cancels exactly after the other
    # context has been merged.
    a_0, a_1, u_0, u_1 = range(4)
    good_0 = (((a_0, u_0), u_1), a_1)
    bad_0 = (((a_0, u_1), u_0), a_1)
    good_1 = (((a_1, u_1), u_0), a_0)
    bad_1 = (((a_1, u_0), u_1), a_0)

    q_a_0, q_a_1, q_u_0, q_u_1 = (
        PROBABILITY[index] for index in (a_0, a_1, u_0, u_1)
    )
    q_context = q_a_0 + q_a_1 - q_a_0 * q_a_1
    mix_weight = q_a_1 * (1 - q_a_0 / 2) / q_context

    _, q_four, nu_good_0, ell_good_0 = data(good_0)
    _, q_four_again, nu_good_1, ell_good_1 = data(good_1)
    _, q_four_bad, nu_bad_0, ell_bad_0 = data(bad_0)
    _, q_four_bad_again, nu_bad_1, ell_bad_1 = data(bad_1)
    assert q_four == q_four_again == q_four_bad == q_four_bad_again
    mixed_good_endpoint = mixture(mix_weight, nu_good_0, nu_good_1)
    mixed_bad_endpoint = mixture(mix_weight, nu_bad_0, nu_bad_1)
    assert mixed_good_endpoint == mixed_bad_endpoint

    mixed_good_cost = mix_weight * ell_good_0 + (1 - mix_weight) * ell_good_1
    mixed_bad_cost = mix_weight * ell_bad_0 + (1 - mix_weight) * ell_bad_1
    quadrilateral_defect = (
        distance(a_0, u_0)
        + distance(a_1, u_1)
        - distance(a_0, u_1)
        - distance(a_1, u_0)
    )
    quadrilateral_coefficient = (
        q_a_0
        * q_a_1
        * q_u_0
        * q_u_1
        * (4 - q_a_0 * q_a_1)
        / (8 * q_context)
    )
    assert (
        mixed_good_cost - mixed_bad_cost
        == quadrilateral_coefficient * quadrilateral_defect
    )

    # Because the mixed endpoint laws coincide, attaching either mixture to
    # exactly the same ancestors leaves the same identity for the full tour
    # value.  One common parent is enough to audit this composability.
    outside_vertex = 4
    mixed_good_full = (
        mix_weight * recursive_tour_value((good_0, outside_vertex))
        + (1 - mix_weight) * recursive_tour_value((good_1, outside_vertex))
    )
    mixed_bad_full = (
        mix_weight * recursive_tour_value((bad_0, outside_vertex))
        + (1 - mix_weight) * recursive_tour_value((bad_1, outside_vertex))
    )
    assert (
        mixed_good_full - mixed_bad_full
        == quadrilateral_coefficient * quadrilateral_defect
    )

    # Stronger globally correlated four-block order.  Equal context
    # occupancies make the endpoint laws identical, while the full
    # quadrilateral coefficient is a*u_0*u_1.
    correlated_probabilities = {
        a_0: Fraction(2, 5),
        a_1: Fraction(2, 5),
        u_0: Fraction(3, 7),
        u_1: Fraction(1, 2),
    }
    correlated_good = (a_0, u_0, u_1, a_1)
    correlated_bad = (a_0, u_1, u_0, a_1)
    q_corr_good, nu_corr_good, ell_corr_good = globally_reversed_order_data(
        correlated_good, correlated_probabilities
    )
    q_corr_bad, nu_corr_bad, ell_corr_bad = globally_reversed_order_data(
        correlated_bad, correlated_probabilities
    )
    assert q_corr_good == q_corr_bad
    assert nu_corr_good == nu_corr_bad
    correlated_coefficient = (
        correlated_probabilities[a_0]
        * correlated_probabilities[u_0]
        * correlated_probabilities[u_1]
    )
    assert (
        ell_corr_good - ell_corr_bad
        == correlated_coefficient * quadrilateral_defect
    )

    # With unequal context occupancies, audit both the exact signed endpoint
    # displacement and the weighted local defect.
    unequal_probabilities = dict(correlated_probabilities)
    unequal_probabilities[a_1] = Fraction(1, 3)
    q_unequal_good, nu_unequal_good, ell_unequal_good = (
        globally_reversed_order_data(correlated_good, unequal_probabilities)
    )
    q_unequal_bad, nu_unequal_bad, ell_unequal_bad = (
        globally_reversed_order_data(correlated_bad, unequal_probabilities)
    )
    assert q_unequal_good == q_unequal_bad
    endpoint_mass_shift = {
        vertex: q_unequal_good
        * (nu_unequal_good.get(vertex, 0) - nu_unequal_bad.get(vertex, 0))
        for vertex in unequal_probabilities
    }
    endpoint_mass_shift = {
        vertex: mass for vertex, mass in endpoint_mass_shift.items() if mass
    }
    unequal_scale = (
        unequal_probabilities[u_0]
        * unequal_probabilities[u_1]
        * (
            unequal_probabilities[a_1]
            - unequal_probabilities[a_0]
        )
        / 2
    )
    assert endpoint_mass_shift == {u_0: unequal_scale, u_1: -unequal_scale}
    unequal_local_defect = (
        unequal_probabilities[u_0]
        * unequal_probabilities[u_1]
        * (
            unequal_probabilities[a_0]
            * (distance(a_0, u_0) - distance(a_0, u_1))
            + unequal_probabilities[a_1]
            * (distance(a_1, u_1) - distance(a_1, u_0))
        )
    )
    assert ell_unequal_good - ell_unequal_bad == unequal_local_defect

    # Alternating transport-cycle triangulation (BR28).  Diagonal and cyclic
    # permutation couplings have identical uniform marginals.
    row = (Fraction(0), Fraction(10), Fraction(20))
    column = (Fraction(0), Fraction(10), Fraction(20))
    transport_distance = lambda x, y: abs(x - y)
    cyclic_gap = sum(
        transport_distance(row[(index + 1) % 3], column[index])
        - transport_distance(row[index], column[index])
        for index in range(3)
    ) / 3
    quadrilateral_sum = sum(
        transport_distance(row[index], column[index - 1])
        + transport_distance(row[0], column[index])
        - transport_distance(row[0], column[index - 1])
        - transport_distance(row[index], column[index])
        for index in range(1, 3)
    ) / 3
    assert cyclic_gap == quadrilateral_sum
    assert max(Fraction(0), cyclic_gap) <= sum(
        abs(
            transport_distance(row[index], column[index - 1])
            + transport_distance(row[0], column[index])
            - transport_distance(row[0], column[index - 1])
            - transport_distance(row[index], column[index])
        )
        for index in range(1, 3)
    ) / 3

    print(f"C(H1) = {value_1}")
    print(f"C(H2) = {value_2}")
    print(f"local rotation term = {base}")
    print(f"all outside corrections = {outside}")
    print(f"depth-independent bound = {outside_bound}")
    print(f"quadrilateral mixing weight = {mix_weight}")
    print(f"quadrilateral coefficient = {quadrilateral_coefficient}")
    print(f"quadrilateral defect = {quadrilateral_defect}")
    print(f"correlated quadrilateral coefficient = {correlated_coefficient}")
    print(f"unequal-context endpoint scale = {unequal_scale}")
    print(f"transport-cycle gap = {cyclic_gap}")
    print("all exact checks passed")


if __name__ == "__main__":
    main()
