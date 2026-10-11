#!/usr/bin/env python3
"""Exact checker for the deleted-interval guide-cycle identity.

This verifies equations (64)--(67) in the coalescence route notes.  For
rational metrics, rational independent block occupancies, and arbitrary
root-broken guide cycles, it compares:

* direct enumeration of every occupied-block subset;
* the consecutive-retained-pair formula; and
* full guide length minus expected deleted-interval surplus.

All arithmetic uses fractions.Fraction.
"""

from __future__ import annotations

import random
from fractions import Fraction


def metric_closure(weights: list[list[Fraction]]) -> list[list[Fraction]]:
    distance = [row[:] for row in weights]
    for k in range(len(distance)):
        for i in range(len(distance)):
            for j in range(len(distance)):
                distance[i][j] = min(
                    distance[i][j], distance[i][k] + distance[k][j]
                )
    return distance


def direct_value(
    distance: list[list[Fraction]], order: tuple[int, ...], q: tuple[Fraction, ...]
) -> Fraction:
    value = Fraction()
    for mask in range(1 << len(order)):
        probability = Fraction(1)
        retained: list[int] = []
        for i, center in enumerate(order):
            if mask & (1 << i):
                probability *= q[i]
                retained.append(center)
            else:
                probability *= 1 - q[i]
        if not retained:
            continue
        cost = distance[0][retained[0]] + distance[retained[-1]][0]
        cost += sum(
            (distance[u][v] for u, v in zip(retained, retained[1:])),
            Fraction(),
        )
        value += probability * cost
    return value


def interval_value(
    distance: list[list[Fraction]], order: tuple[int, ...], q: tuple[Fraction, ...]
) -> tuple[Fraction, Fraction, Fraction]:
    broken = (0, *order, 0)
    keep = (Fraction(1), *q, Fraction(1))
    guide_length = sum(
        (distance[u][v] for u, v in zip(broken, broken[1:])), Fraction()
    )
    shortcut = Fraction()
    surplus = Fraction()
    for i in range(len(broken) - 1):
        empty_between = Fraction(1)
        path_length = Fraction()
        for j in range(i + 1, len(broken)):
            path_length += distance[broken[j - 1]][broken[j]]
            alpha = keep[i] * keep[j] * empty_between
            chord = distance[broken[i]][broken[j]]
            shortcut += alpha * chord
            surplus += alpha * (path_length - chord)
            empty_between *= 1 - keep[j]
    return shortcut, surplus, guide_length


def check_one(
    distance: list[list[Fraction]], order: tuple[int, ...], q: tuple[Fraction, ...]
) -> None:
    direct = direct_value(distance, order, q)
    shortcut, surplus, guide_length = interval_value(distance, order, q)
    assert direct == shortcut
    assert shortcut + surplus == guide_length


def main() -> None:
    rng = random.Random(20260906)
    checked = 0
    for clients in range(1, 8):
        vertices = clients + 1
        for _ in range(40):
            weights = [[Fraction() for _ in range(vertices)] for _ in range(vertices)]
            for i in range(vertices):
                for j in range(i + 1, vertices):
                    value = Fraction(rng.randint(1, 20), rng.randint(1, 5))
                    weights[i][j] = weights[j][i] = value
            distance = metric_closure(weights)
            order_list = list(range(1, vertices))
            rng.shuffle(order_list)
            order = tuple(order_list)
            q = tuple(
                Fraction(rng.randint(1, 9), 10) for _ in range(clients)
            )
            check_one(distance, order, q)
            checked += 1
    print(f"exact rational instances checked: {checked}")
    print("direct enumeration = interval formula = guide length - surplus")
    print("all exact checks passed")


if __name__ == "__main__":
    main()
