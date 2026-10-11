#!/usr/bin/env python3
"""Exact metric realization of the two-endpoint coupling obstruction.

There are four clients, each active with probability 1/2.  Client--client
distances are one and depot distances are 1, 5/4, 3/2, 7/4.  Posterior tours
adaptively expose the two active clients closest to the depot as endpoints.
No activation-independent distribution over full orders (hence no correlated
orientation hierarchy) can reproduce that endpoint-incidence vector.
"""

from __future__ import annotations

import itertools
from fractions import Fraction


N = 4
P_ACT = Fraction(1, 2)
WEIGHTS = (4, 3, 2, 1)
DEPOT = tuple(Fraction(2) - Fraction(w, 4) for w in WEIGHTS)


def d(i: int, j: int) -> Fraction:
    """Indices -1 and 0,...,3 denote the depot and clients."""
    if i == j:
        return Fraction(0)
    if i == -1:
        return DEPOT[j]
    if j == -1:
        return DEPOT[i]
    return Fraction(1)


def probability(mask: int) -> Fraction:
    # All 2^4 activation sets are equiprobable.
    return Fraction(1, 2**N)


def cost_of_word(word: tuple[int, ...]) -> Fraction:
    if not word:
        return Fraction(0)
    return (
        d(-1, word[0])
        + sum(d(u, v) for u, v in zip(word, word[1:]))
        + d(word[-1], -1)
    )


def restricted(order: tuple[int, ...], mask: int) -> tuple[int, ...]:
    return tuple(v for v in order if mask >> v & 1)


def endpoint_reward(word: tuple[int, ...]) -> int:
    if not word:
        return 0
    return WEIGHTS[word[0]] + WEIGHTS[word[-1]]


def main() -> None:
    # Verify all metric inequalities exactly.
    points = (-1, 0, 1, 2, 3)
    for i in points:
        for j in points:
            for k in points:
                assert d(i, j) <= d(i, k) + d(k, j)

    orders = tuple(itertools.permutations(range(N)))
    posterior = Fraction(0)
    posterior_reward = Fraction(0)
    for mask in range(1 << N):
        active_orders = {restricted(order, mask) for order in orders}
        best_cost = min(map(cost_of_word, active_orders))
        best_reward = max(map(endpoint_reward, active_orders))
        posterior += probability(mask) * best_cost
        posterior_reward += probability(mask) * best_reward

    order_values = []
    order_rewards = []
    for order in orders:
        value = sum(
            probability(mask) * cost_of_word(restricted(order, mask))
            for mask in range(1 << N)
        )
        reward = sum(
            probability(mask) * endpoint_reward(restricted(order, mask))
            for mask in range(1 << N)
        )
        order_values.append(value)
        order_rewards.append(reward)

    prior = min(order_values)
    best_fixed_reward = max(order_rewards)
    assert posterior == Fraction(113, 32)
    assert prior == Fraction(227, 64)
    assert prior - posterior == Fraction(1, 64)
    assert posterior_reward == Fraction(82, 16)
    assert best_fixed_reward == Fraction(81, 16)

    # Every randomized order is a convex combination of these 24 values, so
    # its reward cannot exceed the best deterministic value.
    assert posterior_reward > best_fixed_reward

    print(f"posterior endpoint reward: {posterior_reward}")
    print(f"best activation-independent endpoint reward: {best_fixed_reward}")
    print(f"posterior optimum: {posterior}")
    print(f"a-priori optimum: {prior}")
    print(f"unavoidable endpoint-coupling cost: {prior - posterior}")
    print("all exact checks passed")


if __name__ == "__main__":
    main()
