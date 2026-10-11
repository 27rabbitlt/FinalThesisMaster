#!/usr/bin/env python3
"""Exact finite audit of the repeated-scenario portal transport identity.

The global active-set order deliberately depends on the realization, and the
activation probabilities are heterogeneous.  All calculations use rational
arithmetic.  The script verifies inheritance-compatible marginals, the row
and column equations of the residual transport plan, and the two-case cost
formula from the structural note.
"""

from __future__ import annotations

from fractions import Fraction


PROB = (Fraction(1, 3), Fraction(2, 5), Fraction(1, 2), Fraction(3, 7))
POSITION = (0, 2, 5, 9)
PARENT = frozenset(range(4))
CHILD = frozenset((0, 2))


def scenario_probability(mask: int) -> Fraction:
    ans = Fraction(1)
    for v, p_v in enumerate(PROB):
        ans *= p_v if mask & (1 << v) else 1 - p_v
    return ans


def global_order(mask: int) -> tuple[int, ...]:
    # A realization-dependent global order; every node uses its restriction.
    base = (2, 0, 3, 1) if mask.bit_count() % 2 == 0 else (1, 3, 0, 2)
    return tuple(v for v in base if mask & (1 << v))


def first(mask: int, vertices: frozenset[int]) -> int | None:
    return next((v for v in global_order(mask) if v in vertices), None)


def distance(u: int, v: int) -> Fraction:
    return Fraction(abs(POSITION[u] - POSITION[v]))


def main() -> None:
    scenarios = range(1 << len(PROB))
    q_parent = sum(
        scenario_probability(mask)
        for mask in scenarios
        if first(mask, PARENT) is not None
    )
    q_child = sum(
        scenario_probability(mask)
        for mask in scenarios
        if first(mask, CHILD) is not None
    )

    h_parent = {v: Fraction(0) for v in PARENT}
    h_child = {v: Fraction(0) for v in CHILD}
    for mask in scenarios:
        probability = scenario_probability(mask)
        z_parent = first(mask, PARENT)
        z_child = first(mask, CHILD)
        if z_parent is not None:
            h_parent[z_parent] += probability
        if z_child is not None:
            h_child[z_child] += probability

    mu_parent = {v: h_parent[v] / q_parent for v in PARENT}
    mu_child = {v: h_child[v] / q_child for v in CHILD}
    assert all(mu_child[v] >= mu_parent[v] for v in CHILD)

    gamma = {
        (u, v): Fraction(0)
        for u in PARENT - CHILD
        for v in CHILD
    }
    same_scenario_cost = Fraction(0)
    miss_parent_mass = {u: Fraction(0) for u in PARENT - CHILD}
    for mask in scenarios:
        probability = scenario_probability(mask)
        z_parent = first(mask, PARENT)
        z_child = first(mask, CHILD)
        if z_parent in PARENT - CHILD and z_child is not None:
            gamma[z_parent, z_child] += probability / q_parent
            same_scenario_cost += probability * distance(z_parent, z_child)
        if z_parent in PARENT - CHILD and z_child is None:
            miss_parent_mass[z_parent] += probability

    for u in PARENT - CHILD:
        for v in CHILD:
            gamma[u, v] += miss_parent_mass[u] / q_parent * mu_child[v]

    for u in PARENT - CHILD:
        assert sum(gamma[u, v] for v in CHILD) == mu_parent[u]
    for v in CHILD:
        assert sum(gamma[u, v] for u in PARENT - CHILD) == (
            mu_child[v] - mu_parent[v]
        )

    transport_cost = sum(
        mass * distance(u, v) for (u, v), mass in gamma.items()
    )
    miss_cost = sum(
        miss_parent_mass[u] * mu_child[v] * distance(u, v)
        for u in PARENT - CHILD
        for v in CHILD
    )
    two_case_cost = (same_scenario_cost + miss_cost) / q_parent
    assert transport_cost == two_case_cost
    assert sum(miss_parent_mass.values()) == q_parent - q_child

    print(f"q_parent={q_parent}, q_child={q_child}")
    print(f"transport_cost={transport_cost} = {float(transport_cost):.12f}")
    print("exact repeated-scenario portal transport identities verified")


if __name__ == "__main__":
    main()
