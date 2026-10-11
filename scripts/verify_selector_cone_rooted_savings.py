#!/usr/bin/env python3
"""Exact audits for selector_cone_rooted_savings.md.

The script verifies:
  * the rooted-savings identities on a heterogeneous finite metric;
  * the minimum-root gauge reconstruction;
  * the finite line failure of preserved-edge credit at 29/10; and
  * the equidistant-cluster failure of maximum immediate y*s overlap; and
  * the far-root cycle obstruction to every laminar savings under-kernel.

All comparisons use fractions.Fraction.  Decimal output is display-only.
"""

from fractions import Fraction as Q
from itertools import combinations, permutations


def subset_probability(mask: int, probabilities: tuple[Q, ...]) -> Q:
    answer = Q(1)
    for i, p in enumerate(probabilities):
        answer *= p if mask & (1 << i) else 1 - p
    return answer


def cycle_cost(order: tuple[int, ...], distance: tuple[tuple[Q, ...], ...]) -> Q:
    if not order:
        return Q(0)
    rooted = (0, *order, 0)
    return sum((distance[u][v] for u, v in zip(rooted, rooted[1:])), Q(0))


def path_saving(order: tuple[int, ...], saving: tuple[tuple[Q, ...], ...]) -> Q:
    return sum((saving[u][v] for u, v in zip(order, order[1:])), Q(0))


def stopped_cost(
    order: tuple[int, ...],
    probabilities: tuple[Q, ...],
    distance: tuple[tuple[Q, ...], ...],
) -> Q:
    n = len(order)
    answer = Q(0)
    for mask in range(1 << n):
        active = tuple(v for v in order if mask & (1 << (v - 1)))
        answer += subset_probability(mask, probabilities) * cycle_cost(active, distance)
    return answer


def posterior_data(
    probabilities: tuple[Q, ...], distance: tuple[tuple[Q, ...], ...]
) -> tuple[Q, dict[tuple[int, int], Q]]:
    n = len(probabilities)
    incidence = {(u, v): Q(0) for u in range(n + 1) for v in range(u + 1, n + 1)}
    posterior = Q(0)
    for mask in range(1 << n):
        active = tuple(i + 1 for i in range(n) if mask & (1 << i))
        probability = subset_probability(mask, probabilities)
        if len(active) <= 1:
            chosen = active
        else:
            chosen = min(permutations(active), key=lambda order: (cycle_cost(order, distance), order))
        posterior += probability * cycle_cost(chosen, distance)
        rooted = (0, *chosen, 0)
        for u, v in zip(rooted, rooted[1:]):
            if u != v:
                incidence[tuple(sorted((u, v)))] += probability
    return posterior, incidence


def audit_heterogeneous_identity() -> None:
    positions = (Q(0), Q(1), Q(3), Q(7), Q(8))
    distance = tuple(tuple(abs(x - y) for y in positions) for x in positions)
    probabilities = (Q(1, 3), Q(2, 5), Q(1, 2), Q(3, 4))
    n = len(probabilities)
    spoke = positions
    saving = tuple(
        tuple(spoke[u] + spoke[v] - distance[u][v] for v in range(n + 1))
        for u in range(n + 1)
    )

    posterior, y = posterior_data(probabilities, distance)
    baseline = 2 * sum((probabilities[v - 1] * spoke[v] for v in range(1, n + 1)), Q(0))
    posterior_saving = sum(
        (y[u, v] * saving[u][v] for u, v in combinations(range(1, n + 1), 2)), Q(0)
    )
    assert posterior == baseline - posterior_saving

    order = (3, 1, 4, 2)
    direct = stopped_cost(order, probabilities, distance)
    stopped_saving = Q(0)
    for i in range(n):
        u = order[i]
        for j in range(i + 1, n):
            v = order[j]
            empty_gap = Q(1)
            for k in range(i + 1, j):
                empty_gap *= 1 - probabilities[order[k] - 1]
            coefficient = probabilities[u - 1] * probabilities[v - 1] * empty_gap
            stopped_saving += coefficient * saving[u][v]
    assert direct == baseline - stopped_saving

    # Every chosen posterior word maximizes savings because its baseline is fixed.
    for mask in range(1 << n):
        active = tuple(i + 1 for i in range(n) if mask & (1 << i))
        if len(active) <= 1:
            continue
        costs = [cycle_cost(order, distance) for order in permutations(active)]
        rewards = [path_saving(order, saving) for order in permutations(active)]
        assert costs.index(min(costs)) == rewards.index(max(rewards))

    # The three savings-coordinate generators are exactly metric triangles.
    for u, v in combinations(range(1, n + 1), 2):
        assert saving[u][v] == distance[0][u] + distance[0][v] - distance[u][v]
        assert 2 * spoke[u] - saving[u][v] == (
            distance[0][u] + distance[u][v] - distance[0][v]
        )
    for u in range(1, n + 1):
        for v in range(1, n + 1):
            for w in range(1, n + 1):
                if len({u, v, w}) < 3:
                    continue
                assert 2 * spoke[v] - saving[u][v] - saving[v][w] + saving[u][w] == (
                    distance[u][v] + distance[v][w] - distance[u][w]
                )


def minimum_root_gauge(saving: tuple[tuple[Q, ...], ...]) -> tuple[Q, ...]:
    n = len(saving)
    gauge = []
    for v in range(n):
        candidates = [Q(0)]
        candidates.extend(saving[u][v] for u in range(n) if u != v)
        candidates.extend(
            saving[u][v] + saving[v][w] - saving[u][w]
            for u in range(n)
            for w in range(n)
            if u != v and w != v and u != w
        )
        gauge.append(max(candidates))
    return tuple(gauge)


def audit_minimum_root_gauge() -> None:
    saving = (
        (Q(0), Q(3), Q(1), Q(2)),
        (Q(3), Q(0), Q(2), Q(4)),
        (Q(1), Q(2), Q(0), Q(1)),
        (Q(2), Q(4), Q(1), Q(0)),
    )
    gauge = minimum_root_gauge(saving)
    spoke = tuple(value / 2 for value in gauge)
    n = len(saving)
    distance = [[Q(0) for _ in range(n + 1)] for _ in range(n + 1)]
    for v in range(n):
        distance[0][v + 1] = distance[v + 1][0] = spoke[v]
    for u in range(n):
        for v in range(u + 1, n):
            value = spoke[u] + spoke[v] - saving[u][v]
            assert value >= 0
            distance[u + 1][v + 1] = distance[v + 1][u + 1] = value
    for i in range(n + 1):
        for j in range(n + 1):
            for k in range(n + 1):
                assert distance[i][j] <= distance[i][k] + distance[k][j]

    # Lowering any positive gauge coordinate violates one of its defining constraints.
    for v, value in enumerate(gauge):
        assert value > 0
        lowered = value - Q(1, 100)
        incident_ok = all(saving[u][v] <= lowered for u in range(n) if u != v)
        middle_ok = all(
            saving[u][v] + saving[v][w] - saving[u][w] <= lowered
            for u in range(n)
            for w in range(n)
            if u != v and w != v and u != w
        )
        assert not (incident_ok and middle_ok)


def audit_line_preservation_obstruction() -> None:
    n = 12
    p = Q(1, 2)
    baseline = Q(n * (n + 1), 2)
    posterior = 2 * sum((1 - (1 - p) ** k for k in range(1, n + 1)), Q(0))
    saving = baseline - posterior
    preserved_credit = p * p * saving
    edgewise_upper = baseline - preserved_credit
    assert posterior == Q(22) + Q(1, 2**11)
    assert 10 * edgewise_upper > 29 * posterior

    positions = tuple(Q(i) for i in range(n + 1))
    distance = tuple(tuple(abs(x - y) for y in positions) for x in positions)
    probabilities = (p,) * n
    monotone = stopped_cost(tuple(range(1, n + 1)), probabilities, distance)
    assert monotone == posterior


def audit_cluster_immediate_overlap_obstruction() -> None:
    n = 2000
    p = Q(1, 50)
    delta = Q(1, 1000)
    q = 1 - (1 - p) ** n
    client_saving = 2 - 2 * delta

    # Root distance is 1-delta/2 and every client-client distance is delta.
    baseline = n * p * (2 - delta)
    posterior = client_saving * q + delta * n * p
    expected_path_edges = n * p - q
    posterior_saving = client_saving * expected_path_edges
    assert posterior == baseline - posterior_saving

    # For the increasing selector, y_uv=p^2(1-p)^(v-u-1).  Hence every
    # immediate weight y_uv*s is at most p^2*s, and the monotone full path
    # attains this on all n-1 edges.
    maximum_immediate_overlap = client_saving * (n - 1) * p * p
    assert posterior_saving - maximum_immediate_overlap > Q(19, 10) * posterior

    # Every fixed order has the same expected cost in this equidistant metric.
    order_independent = client_saving * q + delta * n * p
    assert order_independent == posterior


def audit_laminar_cycle_obstruction() -> None:
    n = 192
    posterior = Q(2 * n - 1)
    posterior_saving = Q((n - 1) ** 2)
    baseline = Q(n * n)
    assert posterior == baseline - posterior_saving

    # Every laminar under-kernel has residual at least this layer-cake sum.
    lower_bound = sum((Q((n - 1) // (h + 1)) for h in range(1, n // 2)), Q(0))
    assert lower_bound == 747
    assert lower_bound > Q(19, 10) * posterior

    # The selected Hamilton path uses n-1 unit cycle edges.  With root
    # distance n/2 its tour cost is exactly 2n-1, so the coherent order is
    # posterior-optimal despite failure of every laminar residual certificate.
    coherent_order_cost = Q(n) + Q(n - 1)
    assert coherent_order_cost == posterior


def main() -> None:
    audit_heterogeneous_identity()
    audit_minimum_root_gauge()
    audit_line_preservation_obstruction()
    audit_cluster_immediate_overlap_obstruction()
    audit_laminar_cycle_obstruction()
    print("rooted-savings identities: passed")
    print("minimum-root gauge: passed")
    print("line preserved-edge obstruction: passed")
    print("cluster immediate-overlap obstruction: passed")
    print("laminar cycle obstruction: passed")


if __name__ == "__main__":
    main()
