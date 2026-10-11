#!/usr/bin/env python3
"""Exact checks for the symmetric hop/multiscale route.

The script verifies:
  * the convex weights proving the sharp geometric-tail lemma;
  * the scalar rooted-tree cut inequality;
  * the finite cyclic occupancy correction;
  * the finite L/R obstruction to a coefficient-one hop certificate;
  * the exact limiting coefficient in the sparse-random SH-B obstruction;
  * the seven-anchor/fourteen-rare full-support-tour obstruction; and
  * the conditional depot-clone gap extraction on a finite instance.

All asserted comparisons use integers or Fraction arithmetic.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, permutations
from math import comb


Q = Fraction


def sharp_tail_weights_check() -> None:
    """Check (SH6)--(SH7) exactly for a range of k."""
    for k in range(2, 41):
        p = Q(1, k)
        q = 1 - p
        c = Q(2, k + 1)
        weights = [Q(0)] * (k + 1)
        for r in range(1, k + 1):
            weights[r] = c * p * (r - 1) * (q ** (-r) - 1)
            assert weights[r] >= 0
        assert sum(weights) == 1

        for h in range(1, k + 1):
            coefficient = weights[h] * q ** (h - 1) / (1 - q**h)
            for r in range(h + 1, k + 1):
                coefficient += (
                    weights[r] * p * q ** (h - 1) / (1 - q**r)
                )
            assert coefficient == c

        # The linear sequence a_h=h attains equality in (SH4).
        infinite_geometric_sum = 1 / (1 - q) ** 2
        first_k_sum = Q(k * (k + 1), 2)
        assert infinite_geometric_sum == Q(2 * k, k + 1) * first_k_sum


def rooted_tree_scalar_check() -> None:
    """Check the scalar inequality behind the rooted-tree hop lemma."""
    for k in range(2, 41):
        for denominator in range(k, k + 6):
            p = Q(1, denominator)
            assert p <= Q(1, k)
            for size in range(1, 3 * k + 1):
                left = sum(min(hop, size) for hop in range(1, k + 1))
                occupied = 1 - (1 - p) ** size
                right = Q(k, 1) / p * occupied
                assert left <= right


def cyclic_unit_occupancy_obstruction_check() -> None:
    """Check the singleton correction required in unrooted SH9."""
    k = 20
    p = Q(1, k)
    q = 1 - p
    vertex_count = k + 1
    hop = vertex_count * k
    posterior = vertex_count * p * (1 - q ** (vertex_count - 1))
    ratio = p**2 * hop / posterior
    assert ratio == 1 / (1 - q ** (vertex_count - 1))
    assert ratio > Q(29, 20)


def posterior_lr_exact(m: int, p: Q) -> Q:
    weights = [
        Q(comb(m, active)) * p**active * (1 - p) ** (m - active)
        for active in range(m + 1)
    ]
    assert sum(weights) == 1
    value = Q(0)
    for left, left_weight in enumerate(weights):
        for right, right_weight in enumerate(weights):
            total = left + right
            scenario = 0 if total == 0 else total + max(1, abs(left - right))
            value += left_weight * right_weight * scenario
    return value


def high_run_hop_obstruction_check() -> None:
    k = 4
    m = 50
    p = Q(1, k)
    vertex_count = 2 * m + 1
    low = (k + 1) // 2
    high = k + 1 - low
    monochromatic_per_window = comb(low, 2) + comb(high, 2)
    assert monochromatic_per_window == 4

    # Each kth-power edge occurs in at most k cyclic (k+1)-windows.
    all_monochromatic = (
        vertex_count * monochromatic_per_window + k - 1
    ) // k
    client_monochromatic = all_monochromatic - 2 * k
    assert client_monochromatic == 93
    hop_lower_bound = vertex_count * k + client_monochromatic
    assert hop_lower_bound == 497

    posterior = posterior_lr_exact(m, p)
    scaled_hop_lower_bound = p**2 * hop_lower_bound
    assert scaled_hop_lower_bound == Q(497, 16)

    # E|A-B| <= sqrt(2mp(1-p)) < 35/8 gives the short exact bound.
    variance = 2 * m * p * (1 - p)
    assert variance < Q(35, 8) ** 2
    posterior_upper_bound = Q(2 * m, k) + 1 + Q(35, 8)
    assert posterior_upper_bound == Q(243, 8)
    assert posterior < posterior_upper_bound < scaled_hop_lower_bound
    assert scaled_hop_lower_bound / posterior > 1


def sparse_random_sh_b_coefficient_check() -> None:
    """Check the exact asymptotic coefficient asserted in SH-B."""
    target = Q(29, 20)
    for k in range(2, 41):
        p = Q(1, k)
        # Per vertex, SH-B has Hop >= 2k-1-o(1) and P <= p+o(1).
        limiting_ratio = p**2 * (2 * k - 1) / p
        assert limiting_ratio == 2 - Q(1, k)
        assert limiting_ratio >= Q(3, 2) > target


def floyd_warshall(unit_graph: list[list[int]]) -> list[list[int]]:
    count = len(unit_graph)
    infinity = 10**9
    distance = [[infinity] * count for _ in range(count)]
    for vertex in range(count):
        distance[vertex][vertex] = 0
        for neighbor in unit_graph[vertex]:
            distance[vertex][neighbor] = 1
    for pivot in range(count):
        for first in range(count):
            for second in range(count):
                distance[first][second] = min(
                    distance[first][second],
                    distance[first][pivot] + distance[pivot][second],
                )
    return distance


def full_support_tour_obstruction_check() -> None:
    anchor_count = 7
    anchors = list(range(anchor_count))
    bad_order = (0, 3, 6, 2, 5, 1, 4)
    rare_count = 14
    vertex_count = anchor_count + rare_count
    graph: list[list[int]] = [[] for _ in range(vertex_count)]

    def add_unit_edge(first: int, second: int) -> None:
        graph[first].append(second)
        graph[second].append(first)

    # The good anchor cycle.
    for index in range(anchor_count):
        add_unit_edge(index, (index + 1) % anchor_count)

    # Seven internally disjoint length-three bad paths.
    next_rare = anchor_count
    expanded_bad_cycle: list[int] = []
    bad_paths: list[tuple[int, int, int, int]] = []
    for index, first in enumerate(bad_order):
        second = bad_order[(index + 1) % anchor_count]
        rare_one = next_rare
        rare_two = next_rare + 1
        next_rare += 2
        expanded_bad_cycle.extend((first, rare_one, rare_two))
        bad_paths.append((first, rare_one, rare_two, second))
        add_unit_edge(first, rare_one)
        add_unit_edge(rare_one, rare_two)
        add_unit_edge(rare_two, second)
    assert next_rare == vertex_count
    assert len(expanded_bad_cycle) == vertex_count

    distance = floyd_warshall(graph)
    for index, first in enumerate(bad_order):
        second = bad_order[(index + 1) % anchor_count]
        assert distance[first][second] == 3

    # The good anchor tour is optimal: seven positive-length edges are
    # necessary, and the unit anchor cycle realizes seven.
    best_anchor_tour = min(
        distance[0][order[0]]
        + sum(distance[order[i]][order[i + 1]] for i in range(5))
        + distance[order[-1]][0]
        for order in permutations(range(1, anchor_count))
    )
    assert best_anchor_tour == 7

    # Every rare vertex has unit-graph degree two.  Its two forced edges make
    # the expanded bad cycle an all-unit full-support Hamilton cycle.
    for rare in range(anchor_count, vertex_count):
        assert len(graph[rare]) == 2
    for anchor in anchors:
        assert sum(neighbor >= anchor_count for neighbor in graph[anchor]) == 2
    assert all(
        expanded_bad_cycle[(index + 1) % vertex_count]
        in graph[expanded_bad_cycle[index]]
        for index in range(vertex_count)
    )
    assert all(
        distance[first][second] >= 1
        for first, second in combinations(range(vertex_count), 2)
    )
    full_support_cost = vertex_count
    assert full_support_cost == 21

    # Shortcutting any subset of the two rare vertices on each bad path
    # leaves that anchor-to-anchor segment at total length three.
    for first, rare_one, rare_two, second in bad_paths:
        for mask in range(4):
            retained = [first]
            if mask & 1:
                retained.append(rare_one)
            if mask & 2:
                retained.append(rare_two)
            retained.append(second)
            assert sum(
                distance[left][right]
                for left, right in zip(retained, retained[1:])
            ) == 3

    epsilon = Q(1, 1000)
    posterior_upper_bound = Q(7) + 2 * rare_count * epsilon
    prior_bad_order = Q(21)
    assert prior_bad_order / posterior_upper_bound > Q(29, 10)


def depot_clone_extraction_check() -> None:
    """Exact finite audit of the conditional-gap argument (SH24)--(SH25)."""
    p = Q(1, 3)
    clients = (1, 2, 3)
    clones = (10, 11, 12)
    cyclic_order = (1, 10, 2, 11, 3, 12)
    coordinate = {0: 0, 1: 1, 2: 3, 3: 4}

    def location(label: int) -> int:
        return 0 if label in clones else label

    def metric(first: int, second: int) -> int:
        return abs(coordinate[location(first)] - coordinate[location(second)])

    def cyclic_cost(active_order: list[int]) -> int:
        if len(active_order) <= 1:
            return 0
        return sum(
            metric(active_order[index], active_order[(index + 1) % len(active_order)])
            for index in range(len(active_order))
        )

    root_mask_probability: dict[int, Q] = {}
    chosen_gap_probability = {clone: Q(0) for clone in clones}
    event_probability = Q(0)
    for root_mask in range(1 << len(clones)):
        active_count = root_mask.bit_count()
        probability = p**active_count * (1 - p) ** (len(clones) - active_count)
        root_mask_probability[root_mask] = probability
        if root_mask:
            event_probability += probability
            chosen = next(
                clones[index]
                for index in range(len(clones))
                if root_mask & (1 << index)
            )
            chosen_gap_probability[chosen] += probability
    assert event_probability == 1 - (1 - p) ** len(clones)

    cyclic_expectation = Q(0)
    deleted_on_event = Q(0)
    for client_mask in range(1 << len(clients)):
        client_count = client_mask.bit_count()
        client_probability = p**client_count * (1 - p) ** (
            len(clients) - client_count
        )
        for root_mask, root_probability in root_mask_probability.items():
            active = [
                label
                for label in cyclic_order
                if (
                    label in clients
                    and client_mask & (1 << clients.index(label))
                )
                or (
                    label in clones
                    and root_mask & (1 << clones.index(label))
                )
            ]
            probability = client_probability * root_probability
            original_cost = cyclic_cost(active)
            cyclic_expectation += probability * original_cost
            if root_mask:
                chosen = next(
                    clones[index]
                    for index in range(len(clones))
                    if root_mask & (1 << index)
                )
                deleted = [label for label in active if label not in clones or label == chosen]
                deleted_cost = cyclic_cost(deleted)
                assert deleted_cost <= original_cost
                deleted_on_event += probability * deleted_cost

    rooted_gap_cost: dict[int, Q] = {}
    for chosen in clones:
        expectation = Q(0)
        one_root_order = [
            label for label in cyclic_order if label not in clones or label == chosen
        ]
        for client_mask in range(1 << len(clients)):
            client_count = client_mask.bit_count()
            probability = p**client_count * (1 - p) ** (
                len(clients) - client_count
            )
            active = [
                label
                for label in one_root_order
                if label == chosen
                or client_mask & (1 << clients.index(label))
            ]
            expectation += probability * cyclic_cost(active)
        rooted_gap_cost[chosen] = expectation

    mixture = sum(
        chosen_gap_probability[clone] * rooted_gap_cost[clone]
        for clone in clones
    )
    assert mixture == deleted_on_event
    assert min(rooted_gap_cost.values()) <= deleted_on_event / event_probability
    assert deleted_on_event <= cyclic_expectation


def main() -> None:
    sharp_tail_weights_check()
    rooted_tree_scalar_check()
    cyclic_unit_occupancy_obstruction_check()
    high_run_hop_obstruction_check()
    sparse_random_sh_b_coefficient_check()
    full_support_tour_obstruction_check()
    depot_clone_extraction_check()
    print("sharp geometric-tail weights checked exactly for k=2,...,40")
    print("rooted-tree scalar hop inequality checked exactly")
    print("finite cyclic singleton/occupancy correction checked exactly")
    print("L/R coefficient-one hop obstruction checked exactly")
    print("SH-B limiting coefficient 2-1/k checked exactly for k=2,...,40")
    print("seven-anchor/fourteen-rare full-support-tour trap checked exactly")
    print("depot-clone conditional-gap extraction checked exactly")


if __name__ == "__main__":
    main()
