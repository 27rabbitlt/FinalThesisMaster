#!/usr/bin/env python3
"""Exact checks for the ear/theta distribution-state audit.

The script verifies four claims used in ``ear_theta_extension.md``:

1. the primitive square-gap inequalities for many weighted paths;
2. the exact six-state table of the two-path diamond, both directly and by
   the parallel min-plus rule;
3. the 3/2 pointwise selector obstruction on three four-edge paths; and
4. the finite rational constant-boundary obstruction at t=4.

All mathematical comparisons use integers or ``Fraction`` arithmetic.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, permutations, product

from verify_constant_boundary_portal_obstruction import (
    M,
    P_ACTIVE,
    exact_expectations,
)


def audit_path_kernel() -> int:
    checked = 0
    edge_patterns = (
        (1, 1),
        (1, 2, 3),
        (5, 1, 4, 2),
        (2, 7, 1, 3, 5),
    )
    for edges in edge_patterns:
        length = sum(edges)
        prefix = [0]
        for edge in edges:
            prefix.append(prefix[-1] + edge)
        internal_count = len(edges) - 1
        for mask in range(1 << internal_count):
            marks = [0] + [
                j + 1 for j in range(internal_count) if mask >> j & 1
            ] + [len(edges)]
            gaps = [prefix[b] - prefix[a] for a, b in zip(marks, marks[1:])]
            final_gap = gaps[-1]
            maximum = max(gaps)
            radius = length - final_gap
            square = Fraction(sum(g * g for g in gaps), length)

            assert square >= Fraction(maximum * maximum, length)
            assert square >= Fraction(final_gap * final_gap, length)
            assert length - square <= 2 * (length - maximum)
            assert length - square <= 2 * radius
            t = Fraction(final_gap, length)
            v = square / length
            assert 1 - 2 * t + v >= (1 - t) ** 2

            # Independent audit of the length-uniform cut identity: an
            # edge-sized atom samples its containing active gap.
            sampled_gap_numerator = 0
            sampled_t_empty_numerator = 0
            for edge_index, edge_length in enumerate(edges):
                left = max(mark for mark in marks if mark <= edge_index)
                right = min(mark for mark in marks if mark > edge_index)
                sampled_gap_numerator += edge_length * (
                    prefix[right] - prefix[left]
                )
                if not any(mark > edge_index and mark < len(edges) for mark in marks):
                    sampled_t_empty_numerator += edge_length
            assert Fraction(sampled_gap_numerator, length) == square
            assert Fraction(sampled_t_empty_numerator, length) == t
            checked += 1
    return checked


DIAMOND_DISTANCE = (
    (0, 2, 1, 1),  # s
    (2, 0, 1, 1),  # t
    (1, 1, 0, 2),  # u
    (1, 1, 2, 0),  # v
)


def path_cost(start: int, end: int, middle: tuple[int, ...]) -> int:
    best = None
    for order in permutations(middle):
        walk = (start,) + order + (end,)
        value = sum(DIAMOND_DISTANCE[a][b] for a, b in zip(walk, walk[1:]))
        best = value if best is None else min(best, value)
    return DIAMOND_DISTANCE[start][end] if best is None else best


def cycle_cost(root: int, required: tuple[int, ...]) -> int:
    if not required:
        return 0
    return path_cost(root, root, required)


def direct_diamond_state(active: tuple[int, ...]):
    even = path_cost(0, 0, (1,) + active)
    odd = path_cost(0, 1, active)
    raw_s = cycle_cost(0, active)
    raw_t = cycle_cost(1, active)
    separated = min(
        cycle_cost(0, tuple(x for x in active if x in left))
        + cycle_cost(1, tuple(x for x in active if x not in left))
        for size in range(len(active) + 1)
        for left in combinations(active, size)
    )
    return separated, even, odd, raw_s, raw_t, int(not active)


def parallel_two_path_state(bits: tuple[bool, bool]):
    # One length-two path, empty or with its midpoint required.
    child = [
        (0, 4, 2, 0, 0, 1) if not bit else (2, 4, 2, 2, 2, 0)
        for bit in bits
    ]
    d = child[0][0] + child[1][0]
    rs = child[0][3] + child[1][3]
    rt = child[0][4] + child[1][4]
    empty = child[0][5] * child[1][5]
    candidates = []
    for states in product((0, 1, 2), repeat=2):  # D,E,O
        if all(state == 0 for state in states):
            continue
        parity = sum(state == 2 for state in states) % 2
        value = sum(child[i][states[i]] for i in range(2))
        candidates.append((parity, value))
    even = min(value for parity, value in candidates if parity == 0)
    odd = min(value for parity, value in candidates if parity == 1)
    return d, even, odd, rs, rt, empty


def audit_diamond() -> None:
    expected = {
        (False, False): (0, 4, 2, 0, 0, 1),
        (True, False): (2, 4, 2, 2, 2, 0),
        (False, True): (2, 4, 2, 2, 2, 0),
        (True, True): (4, 4, 4, 4, 4, 0),
    }
    for bits, state in expected.items():
        active = tuple(vertex for vertex, bit in zip((2, 3), bits) if bit)
        assert direct_diamond_state(active) == state
        assert parallel_two_path_state(bits) == state
    assert len({state[2] for state in expected.values()}) == 2


def audit_pointwise_selector() -> None:
    # The three scenarios have posterior O=4.  Every action has one chosen
    # backbone: it costs 4 in that scenario and 7 in each of the other two.
    posterior_odd = (4, 4, 4)
    for backbone in range(3):
        hazards = tuple(4 if active_path == backbone else 7 for active_path in range(3))
        assert sum(hazards) == 18
        assert max(Fraction(h, o) for h, o in zip(hazards, posterior_odd)) == Fraction(7, 4)

    # For an arbitrary rational mixture over backbone choices, the average
    # of the three scenario hazards is exactly 6, so one ratio is >=3/2.
    mixtures = (
        (Fraction(1, 3),) * 3,
        (Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)),
        (Fraction(1), Fraction(0), Fraction(0)),
    )
    for mixture in mixtures:
        scenario_hazards = tuple(
            sum(mixture[b] * (4 if i == b else 7) for b in range(3))
            for i in range(3)
        )
        assert sum(scenario_hazards) == 18
        assert max(
            scenario_hazards[i] / posterior_odd[i] for i in range(3)
        ) >= Fraction(3, 2)
    assert Fraction(29, 20) < Fraction(3, 2)


def audit_constant_boundary() -> tuple[Fraction, Fraction, Fraction]:
    posterior, capped = exact_expectations((1, 4))
    lam = M * P_ACTIVE
    nearest_radius = 2 * M * P_ACTIVE * (1 - P_ACTIVE)
    deficit = posterior - nearest_radius
    parity_surplus = posterior - 2 * lam
    chi_one = capped[1] - posterior
    chi_four = capped[4] - posterior
    assert chi_one > 4 * deficit
    assert chi_four > 3 * deficit
    assert chi_one > 6 * parity_surplus
    return posterior, deficit, chi_four


def main() -> None:
    path_cases = audit_path_kernel()
    audit_diamond()
    audit_pointwise_selector()
    posterior, deficit, chi_four = audit_constant_boundary()
    print(f"path-kernel realizations checked exactly: {path_cases}")
    print("diamond state table and parallel min-plus law: exact")
    print("pointwise theta selector lower bound: 3/2 > 29/20")
    print(
        "constant-boundary t=4 witness: "
        f"P={float(posterior):.12f}, D={float(deficit):.12f}, "
        f"Chi_4/D={float(chi_four / deficit):.12f}"
    )
    print("all ear/theta extension checks passed")


if __name__ == "__main__":
    main()
