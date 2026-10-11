#!/usr/bin/env python3
"""Exact finite audits of the circular-cut stochastic regret certificate.

All arithmetic is rational. These checks supplement the proof; they do not
establish a universal approximation ratio or an efficient order optimizer.
"""

from fractions import Fraction as F
from itertools import permutations
from math import prod
import random


def circular_weights(distance):
    size = len(distance)
    return [
        (frozenset(range(i + 1, j + 1)),
         (distance[i][j] + distance[i + 1][(j + 1) % size]
          - distance[i][(j + 1) % size] - distance[i + 1][j]) / 2)
        for i in range(size) for j in range(i + 1, size)
    ]


def tour_cost(order, distance):
    return sum(distance[u][v] for u, v in zip(order, order[1:] + order[:1]))


def scenarios(probabilities):
    for mask in range(1 << (len(probabilities) - 1)):
        active = [0] + [v for v in range(1, len(probabilities))
                        if mask & (1 << (v - 1))]
        mass = prod(p if v in active else 1 - p
                    for v, p in enumerate(probabilities) if v)
        yield active, mass


def count_distribution(vertices, probabilities):
    distribution = [F(1)]
    for v in vertices:
        updated = [F(0)] * (len(distribution) + 1)
        for count, mass in enumerate(distribution):
            updated[count] += mass * (1 - probabilities[v])
            updated[count + 1] += mass * probabilities[v]
        distribution = updated
    return distribution


def check_basis(distance):
    weights = circular_weights(distance)
    for u in range(len(distance)):
        for v in range(len(distance)):
            assert distance[u][v] == sum(
                weight for split, weight in weights
                if (u in split) != (v in split))


def audit(distance, probabilities, *, sharp=False, nonnegative=False):
    check_basis(distance)
    weights = circular_weights(distance)
    if nonnegative:
        assert all(weight >= 0 for _, weight in weights)
    order_mean = post_mean = budget_mean = F(0)
    universe = set(range(len(distance)))
    for active, mass in scenarios(probabilities):
        order = tour_cost(active, distance)
        post = min(tour_cost([0, *suffix], distance)
                   for suffix in permutations(active[1:]))
        budget = 2 * sum(
            -weight * max(min(len(set(active) & split),
                              len(set(active) - split)) - 1, 0)
            for split, weight in weights if weight < 0)
        assert 0 <= order - post <= budget
        if sharp:
            assert order - post == budget
        if nonnegative:
            assert order == post
        order_mean += mass * order
        post_mean += mass * post
        budget_mean += mass * budget

    cut_mean = 2 * sum(weight * (1 - prod(1 - probabilities[v] for v in split))
                       for split, weight in weights)
    assert cut_mean == order_mean
    tail_budget = F(0)
    for split, weight in weights:
        if weight >= 0:
            continue
        inside = count_distribution(split, probabilities)
        outside = count_distribution(universe - split, probabilities)
        zeta = sum(sum(inside[k:]) * sum(outside[k:])
                   for k in range(2, min(len(inside), len(outside))))
        coarse = sum(probabilities[u] * probabilities[v]
                     for u in split for v in split if u < v)
        coarse *= sum(probabilities[v] for v in universe - split if v)
        assert zeta <= coarse
        tail_budget += -2 * weight * zeta
    assert tail_budget == budget_mean
    return order_mean, post_mean, budget_mean


def main():
    rng = random.Random(90109)
    for size in range(1, 9):
        for _ in range(15):
            d = [[F(0) for _ in range(size)] for _ in range(size)]
            for u in range(size):
                for v in range(u):
                    d[u][v] = d[v][u] = F(rng.randrange(-8, 9), 3)
            check_basis(d)
    print('120 arbitrary symmetric matrices: exact signed decomposition')

    audit([[F(0)]], [F(1)], nonnegative=True)
    audit([[F(0), F(3)], [F(3), F(0)]], [F(1), F(1, 3)], nonnegative=True)
    for size in range(3, 8):
        for _ in range(6):
            d = [[F(0) for _ in range(size)] for _ in range(size)]
            for u in range(size):
                for v in range(u):
                    d[u][v] = d[v][u] = F(rng.choice((1, 2)))
            p = [F(1)] + [rng.choice((F(0), F(1, 4), F(1, 2), F(3, 4), F(1)))
                          for _ in range(size - 1)]
            audit(d, p)
    print('30 random metrics plus root-only/two-point cases: all certificates exact')

    for size in (4, 5, 6, 7):
        d = [[F(min(abs(u - v), size - abs(u - v)))
              for v in range(size)] for u in range(size)]
        audit(d, [F(1)] + [F(v, 8) for v in range(1, size)], nonnegative=True)
    print('4 cycle metrics: one order optimal for every realization')

    size, left = 6, {1, 2, 3}
    d = [[F(0) if u == v else F(2 - int((u in left) != (v in left)))
          for v in range(size)] for u in range(size)]
    p = [F(1)] + [F(1, 2)] * 5
    grouped = audit(d, p, sharp=True)
    assert grouped == (F(83, 16), F(35, 8), F(13, 16))
    attachment = sum(p[v] * sum(mass * min(d[v][u] for u in active)
                               for active, mass in scenarios(p))
                     for v in range(1, size))
    assert attachment == F(21, 16)

    # Matching exact primal/dual certificates for STATIC arbitrary-cut packing.
    packing = [(frozenset({4}), F(1)), (frozenset({5}), F(1)),
               (frozenset(range(1, 6)), F(1))]
    for u in range(size):
        for v in range(u):
            assert sum(w for split, w in packing
                       if (u in split) != (v in split)) <= d[u][v]
    packing_value = 2 * sum(w * (1 - prod(1 - p[v] for v in split))
                            for split, w in packing)
    static_flow = [[F(0) for _ in range(size)] for _ in range(size)]
    for u in left:
        for v in (0, 4, 5):
            static_flow[u][v] = static_flow[v][u] = F(31, 48) if v == 0 else F(1, 3)
    for active, _ in scenarios(p):
        split = set(active) - {0}
        crossing = sum(static_flow[u][v] for u in split for v in range(size)
                       if v not in split)
        assert crossing >= 2 * (1 - prod(1 - p[v] for v in split))
    dual_value = sum(d[u][v] * static_flow[u][v]
                     for u in range(size) for v in range(u))
    assert packing_value == dual_value == F(63, 16)
    prior = min(sum(mass * tour_cost([0] + [v for v in suffix if v in active], d)
                    for active, mass in scenarios(p))
                for suffix in permutations(range(1, size)))
    assert prior == F(9, 2)
    alternating = [0, 1, 4, 2, 5, 3]
    reordered = [[d[u][v] for v in alternating] for u in alternating]
    alternate = audit(reordered, p)
    assert alternate == (F(9, 2), F(35, 8), F(39, 16))
    print('Six-point example: Lambda=21/16, static packing=63/16, P=35/8, F=9/2')
    print('Grouped signed certificate attains P; all 120 prior orders enumerated')


if __name__ == '__main__':
    main()
