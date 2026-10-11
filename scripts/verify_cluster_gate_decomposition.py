#!/usr/bin/env python3
"""Exact finite audits for cut sharing, gateway decomposition, and a cycle obstruction.

These checks verify finite identities independently of the symbolic proofs;
they do not prove a universal approximation bound.
"""

from fractions import Fraction as F
from itertools import permutations
from math import prod, log, log1p, exp, expm1, ceil


def scenario_probability(mask, probabilities):
    return prod((p if mask & (1 << (v - 1)) else 1 - p)
                for v, p in enumerate(probabilities) if v)


def all_tours(distance):
    n = len(distance) - 1
    costs = [F(0)] * (1 << n)
    paths = {}
    for mask in range(1, 1 << n):
        for v in range(1, n + 1):
            bit = 1 << (v - 1)
            if not mask & bit:
                continue
            before = mask ^ bit
            paths[mask, v] = (distance[0][v] if not before else min(
                paths[before, u] + distance[u][v]
                for u in range(1, n + 1) if before & (1 << (u - 1))))
        costs[mask] = min(paths[mask, v] + distance[v][0]
                          for v in range(1, n + 1) if mask & (1 << (v - 1)))
    return costs


def expected_order(distance, probabilities, order):
    route = (0, *order, 0)
    result = F(0)
    for i in range(len(route) - 1):
        empty = F(1)
        for j in range(i + 1, len(route)):
            result += probabilities[route[i]] * probabilities[route[j]] * empty * distance[route[i]][route[j]]
            empty *= 1 - probabilities[route[j]]
    return result


def optima(distance, probabilities):
    tours = all_tours(distance)
    post = sum(scenario_probability(mask, probabilities) * c
               for mask, c in enumerate(tours))
    prior = min(expected_order(distance, probabilities, order)
                for order in permutations(range(1, len(distance))))
    return prior, post


def check_metric(distance):
    n = len(distance)
    for u in range(n):
        for v in range(n):
            assert distance[u][v] == distance[v][u]
            assert distance[u][v] >= 0
            for w in range(n):
                assert distance[u][v] <= distance[u][w] + distance[w][v]


def gate_metric(core, blocks, local_metrics):
    n = sum(map(len, blocks))
    distance = [[F(0) for _ in range(n)] for _ in range(n)]
    locations = {v: (i, j) for i, block in enumerate(blocks) for j, v in enumerate(block)}
    for u in range(n):
        i, a = locations[u]
        for v in range(n):
            j, b = locations[v]
            distance[u][v] = (local_metrics[i][a][b] if i == j else
                              local_metrics[i][a][0] + core[i][j] + local_metrics[j][0][b])
    return distance


def audit_gate(core, blocks, local_metrics, probabilities):
    distance = gate_metric(core, blocks, local_metrics)
    check_metric(distance)
    quotient_probabilities = [1 - prod(1 - probabilities[v] for v in block)
                              for block in blocks]
    actual = optima(distance, probabilities)
    quotient = optima(core, quotient_probabilities)
    local = [optima(d, [F(1)] + [probabilities[v] for v in block[1:]])
             for block, d in zip(blocks, local_metrics)]
    predicted = tuple(quotient[j] + sum(value[j] for value in local) for j in (0, 1))
    assert actual == predicted, (actual, predicted)
    print('gateway exact:', 'prior=', actual[0], 'post=', actual[1],
          'core gap=', quotient[0] - quotient[1],
          'local gap=', sum(a - b for a, b in local))


def mst_bottleneck(distance, vertices):
    reached = {0}
    remaining = set(vertices) - reached
    lengths = []
    while remaining:
        length, target = min((distance[u][v], v) for u in reached for v in remaining)
        lengths.append(length)
        reached.add(target)
        remaining.remove(target)
    return sum(lengths), max(lengths, default=F(0))


def cycle_audit(groups, size):
    n = groups * size
    distance = [[F(0) for _ in range(n + 1)] for _ in range(n + 1)]
    for u in range(n + 1):
        for v in range(n + 1):
            if u == v:
                continue
            if not u or not v:
                distance[u][v] = F(2)
            else:
                a, b = (u - 1) // size, (v - 1) // size
                distance[u][v] = F(1 if (a - b) % groups in (0, 1, groups - 1) else 2)
    check_metric(distance)
    tours = all_tours(distance)
    p = F(1, 3)
    posterior = F(0)
    attachment = F(0)
    for mask, tour in enumerate(tours):
        vertices = [0] + [v for v in range(1, n + 1) if mask & (1 << (v - 1))]
        occupied = [any(mask & (1 << (i * size + j)) for j in range(size))
                    for i in range(groups)]
        components = (1 if all(occupied) else sum(occupied[i] and not occupied[(i - 1) % groups]
                                                 for i in range(groups)))
        expected = F(mask.bit_count() + components + 2) if mask else F(0)
        assert tour == expected
        tree, bottleneck = mst_bottleneck(distance, vertices)
        assert tour == tree + bottleneck
        order_cost = sum(distance[u][v] for u, v in zip(vertices, vertices[1:] + [0]))
        assert 0 <= order_cost - tour <= 1
        nearest = sum(min(distance[u][v] for v in vertices if v != u) for u in vertices[1:])
        assert tree >= nearest
        weight = p ** mask.bit_count() * (1 - p) ** (n - mask.bit_count())
        posterior += weight * tour
        attachment += weight * sum(p * min(distance[v][u] for u in vertices) for v in range(1, n + 1))
    z = (1 - p) ** size
    mean = n * p
    assert posterior == mean + groups * z * (1 - z) + (1 - z) ** groups + 2 * (1 - z ** groups)
    assert attachment == mean * ((1 - p) + z ** 3)
    print('cycle exact:', groups, 'groups x', size, 'vertices;', 1 << n, 'scenarios')


def asymptotic_table():
    t = 0.25
    print('closed-form floating illustration, t=1/4:')
    for k in (16, 256, 4096):
        p = k ** -4
        m = ceil(log(k) / -log1p(-p))
        z = exp(m * log1p(-p))
        mean = k * m * p
        q = -expm1(t * log1p(-p))
        zt = z ** t
        posterior = mean + k * z * (1 - z) + (1 - z) ** k + 2 * (1 - z ** k)
        sparse = k * m * q + k * zt * (1 - zt) + (1 - zt) ** k + 2 * (1 - zt ** k)
        attachment = mean * (1 - p + z ** 3)
        sparse_attachment = mean * (exp(t * log1p(-p)) + z ** (3 * t))
        print(k, 'D=', round(posterior - attachment, 6),
              'sparse-tP=', round(sparse - t * posterior, 6),
              'A(t)-Lambda=', round(sparse_attachment - attachment, 6))


def main():
    reference = [[F(x) for x in row] for row in (
        (0, 14, 13, 8, 7), (14, 0, 16, 6, 10), (13, 16, 0, 10, 12),
        (8, 6, 10, 0, 11), (7, 10, 12, 11, 0))]
    single = [[F(0)]]
    edge = [[F(0), F(2)], [F(2), F(0)]]
    blocks = [[0], [1, 5], [2], [3], [4]]
    local = [single, edge, single, single, single]
    audit_gate(reference, blocks, local, [F(1)] + [F(1, 2)] * 4 + [F(1, 3)])
    audit_gate(reference, blocks, local, [F(1), F(0), F(1, 3), F(1), F(2, 3), F(1, 4)])
    audit_gate(reference, blocks, local, [F(1), F(0), F(1, 3), F(1), F(2, 3), F(0)])
    audit_gate([[F(0), F(10)], [F(10), F(0)]], [[0], [1, 2, 3, 4, 5]],
               [single, reference], [F(1), F(1, 3)] + [F(1, 2)] * 4)
    # Positive pendant demands at the global root; the quotient root remains always present.
    audit_gate(reference, [[0, 5], [1], [2], [3], [4]],
               [edge, single, single, single, single], [F(1)] + [F(1, 2)] * 5)
    for k, m in ((4, 1), (5, 1), (4, 2)):
        cycle_audit(k, m)
    asymptotic_table()


if __name__ == '__main__':
    main()
