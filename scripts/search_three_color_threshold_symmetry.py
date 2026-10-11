#!/usr/bin/env python3
"""Test conditional vertexwise three-sample symmetrization.

Condition on k_v, the number of appearances of each client v among three iid
samples A,B,C.  Conditional on these counts, the k_v sample labels are a
uniform subset of {A,B,C}, independently for different clients.  At a metric
threshold t, this script checks

  E_labels[ uncovered(A,T(B,C)) + uncovered(A,B) ]
      <= 2 E_labels[minimum number of >t edges in a tour of A].

Here T(B,C)=f_B(C).  If valid for every count vector, threshold, and metric,
then averaging the counts and integrating t proves A1+R <= 2P.
"""

from __future__ import annotations

import argparse
import random
from itertools import combinations, product

from search_hazard_tree_ratio import named_metric, subset_tsp_costs
from search_three_sample_symmetry import projected_mask


LABEL_SUBSETS = {
    k: tuple(mask for mask in range(8) if mask.bit_count() == k)
    for k in range(4)
}


def uncovered(source, target, d, threshold):
    n = len(d) - 1
    selected = [n] + [v for v in range(n) if target & (1 << v)]
    return sum(
        min(d[v][u] for u in selected) > threshold
        for v in range(n)
        if source & (1 << v)
    )


def conditional_gap(
    counts,
    d,
    threshold,
    binary_tours,
    components=None,
    projected=None,
    uncovered_table=None,
):
    left = 0
    right = 0
    component_right = 0
    total = 1
    for k in counts:
        total *= len(LABEL_SUBSETS[k])
    for labels in product(*(LABEL_SUBSETS[k] for k in counts)):
        samples = [0, 0, 0]
        for v, label_mask in enumerate(labels):
            for label in range(3):
                if label_mask & (1 << label):
                    samples[label] |= 1 << v
        a, b, c = samples
        core = projected[b][c] if projected is not None else projected_mask(b, c, d)
        if uncovered_table is None:
            left += uncovered(a, core, d, threshold) + uncovered(a, b, d, threshold)
        else:
            left += uncovered_table[a][core] + uncovered_table[a][b]
        right += 2 * binary_tours[a]
        if components is not None:
            component_right += 2 * components[a]
    return left - right, left, right, component_right, total


def conditional_separate_gaps(
    counts,
    projected,
    uncovered_table,
    certificate,
):
    """Compare each uncovered term separately with one certificate copy."""
    compressed = 0
    guide = 0
    right = 0
    total = 1
    for k in counts:
        total *= len(LABEL_SUBSETS[k])
    for labels in product(*(LABEL_SUBSETS[k] for k in counts)):
        samples = [0, 0, 0]
        for v, label_mask in enumerate(labels):
            for label in range(3):
                if label_mask & (1 << label):
                    samples[label] |= 1 << v
        a, b, c = samples
        compressed += uncovered_table[a][projected[b][c]]
        guide += uncovered_table[a][b]
        right += certificate[a]
    return compressed - right, guide - right, compressed, guide, right, total


def conditional_gaps_by_first_sample(
    counts,
    d,
    threshold,
    binary_tours,
    projected,
    uncovered_table,
):
    """Return gaps after conditioning further on the realized first sample.

    This diagnostic tests the stronger statement that the recoloring average
    over the remaining two labels is already paid by the tour defect of A.
    """
    grouped = {}
    for labels in product(*(LABEL_SUBSETS[k] for k in counts)):
        samples = [0, 0, 0]
        for v, label_mask in enumerate(labels):
            for label in range(3):
                if label_mask & (1 << label):
                    samples[label] |= 1 << v
        a, b, c = samples
        core = projected[b][c]
        left = uncovered_table[a][core] + uncovered_table[a][b]
        old_left, frequency = grouped.get(a, (0, 0))
        grouped[a] = old_left + left, frequency + 1
    return {
        a: (left - 2 * binary_tours[a] * frequency, left, frequency)
        for a, (left, frequency) in grouped.items()
    }


def component_lower_bounds(d, threshold):
    """Number of short-edge components (zero when connected) for all subsets."""
    n = len(d) - 1
    root = n
    values = [0] * (1 << n)
    for mask in range(1 << n):
        remaining = {root, *(v for v in range(n) if mask & (1 << v))}
        components = 0
        while remaining:
            components += 1
            stack = [remaining.pop()]
            while stack:
                u = stack.pop()
                reached = {v for v in remaining if d[u][v] <= threshold}
                remaining.difference_update(reached)
                stack.extend(reached)
        values[mask] = 0 if components == 1 else components
    return values


def assignment_defects(d, threshold):
    """Minimum number of long arcs in a loopless directed cycle cover.

    This is the bipartite-matching relaxation of the rooted Hamiltonian
    cycle.  The dynamic program assigns one distinct right endpoint to each
    left endpoint; loops have infinite cost, except that the empty client
    set has value zero.
    """
    n = len(d) - 1
    root = n
    values = [0] * (1 << n)
    for mask in range(1 << n):
        vertices = [root] + [v for v in range(n) if mask & (1 << v)]
        size = len(vertices)
        if size == 1:
            values[mask] = 0
            continue
        dp = {0: 0}
        for u in vertices:
            next_dp = {}
            for used, cost in dp.items():
                for index, v in enumerate(vertices):
                    if used & (1 << index) or u == v:
                        continue
                    candidate = cost + (d[u][v] > threshold)
                    new_used = used | (1 << index)
                    next_dp[new_used] = min(next_dp.get(new_used, size + 1), candidate)
            dp = next_dp
        values[mask] = dp[(1 << size) - 1]
    return values


def toughness_defects(d, threshold):
    """A partition/toughness lower bound on the number of long tour edges."""
    n = len(d) - 1
    root = n
    values = [0] * (1 << n)
    for mask in range(1 << n):
        vertices = [root] + [v for v in range(n) if mask & (1 << v)]
        best = 0
        for removed_mask in range(1 << len(vertices)):
            remaining = {
                v
                for index, v in enumerate(vertices)
                if not (removed_mask & (1 << index))
            }
            components = 0
            while remaining:
                components += 1
                stack = [remaining.pop()]
                while stack:
                    u = stack.pop()
                    reached = {v for v in remaining if d[u][v] <= threshold}
                    remaining.difference_update(reached)
                    stack.extend(reached)
            removed = removed_mask.bit_count()
            if removed == 0:
                candidate = 0 if components <= 1 else components
            else:
                candidate = max(components - removed, 0)
            best = max(best, candidate)
        values[mask] = best
    return values


def check_metric(d, exhaustive_counts=True, rng=None, random_counts=500):
    n = len(d) - 1
    thresholds = sorted({d[i][j] for i in range(n + 1) for j in range(i)})
    threshold_intervals = [0.5 * (x + y) for x, y in zip([0.0] + thresholds, thresholds)]
    counts_iterable = product(range(4), repeat=n) if exhaustive_counts else (
        tuple(rng.randrange(4) for _ in range(n)) for _ in range(random_counts)
    )
    counts_list = list(counts_iterable)
    worst = None
    worst_components = None
    worst_by_a = None
    worst_assignment = None
    worst_assignment_components = None
    worst_toughness = None
    worst_toughness_compressed = None
    worst_toughness_guide = None
    projected = [
        [projected_mask(b, c, d) for c in range(1 << n)]
        for b in range(1 << n)
    ]
    for threshold in threshold_intervals:
        binary = [
            [0.0 if d[i][j] <= threshold else 1.0 for j in range(n + 1)]
            for i in range(n + 1)
        ]
        binary_tours = subset_tsp_costs(binary)
        components = component_lower_bounds(d, threshold)
        assignments = assignment_defects(d, threshold)
        toughness = toughness_defects(d, threshold)
        uncovered_table = [
            [uncovered(source, target, d, threshold) for target in range(1 << n)]
            for source in range(1 << n)
        ]
        for counts in counts_list:
            gap, left, right, component_right, total = conditional_gap(
                counts,
                d,
                threshold,
                binary_tours,
                components,
                projected,
                uncovered_table,
            )
            component_gap = left - component_right
            assignment_right = 2 * sum(
                assignments[a]
                for labels in product(*(LABEL_SUBSETS[k] for k in counts))
                for a in [sum(
                    (1 << v)
                    for v, label_mask in enumerate(labels)
                    if label_mask & 1
                )]
            )
            assignment_gap = left - assignment_right
            assignment_component_right = assignment_right // 2 + component_right // 2
            assignment_component_gap = left - assignment_component_right
            toughness_right = 2 * sum(
                toughness[a]
                for labels in product(*(LABEL_SUBSETS[k] for k in counts))
                for a in [sum(
                    (1 << v)
                    for v, label_mask in enumerate(labels)
                    if label_mask & 1
                )]
            )
            toughness_gap = left - toughness_right
            (
                compressed_gap,
                guide_gap,
                compressed_left,
                guide_left,
                separate_right,
                _,
            ) = conditional_separate_gaps(
                counts,
                projected,
                uncovered_table,
                toughness,
            )
            if worst_components is None or component_gap / total > worst_components[0]:
                worst_components = (component_gap / total, counts, threshold, left, component_right, total)
            if worst_assignment is None or assignment_gap / total > worst_assignment[0]:
                worst_assignment = (
                    assignment_gap / total,
                    counts,
                    threshold,
                    left,
                    assignment_right,
                    total,
                )
            if (
                worst_assignment_components is None
                or assignment_component_gap / total > worst_assignment_components[0]
            ):
                worst_assignment_components = (
                    assignment_component_gap / total,
                    counts,
                    threshold,
                    left,
                    assignment_component_right,
                    total,
                )
            if worst_toughness is None or toughness_gap / total > worst_toughness[0]:
                worst_toughness = (
                    toughness_gap / total,
                    counts,
                    threshold,
                    left,
                    toughness_right,
                    total,
                )
            if (
                worst_toughness_compressed is None
                or compressed_gap / total > worst_toughness_compressed[0]
            ):
                worst_toughness_compressed = (
                    compressed_gap / total,
                    counts,
                    threshold,
                    compressed_left,
                    separate_right,
                    total,
                )
            if (
                worst_toughness_guide is None
                or guide_gap / total > worst_toughness_guide[0]
            ):
                worst_toughness_guide = (
                    guide_gap / total,
                    counts,
                    threshold,
                    guide_left,
                    separate_right,
                    total,
                )
            if worst is None or gap / total > worst[0]:
                worst = (gap / total, counts, threshold, left, right, total)
            for a, (a_gap, a_left, frequency) in conditional_gaps_by_first_sample(
                counts,
                d,
                threshold,
                binary_tours,
                projected,
                uncovered_table,
            ).items():
                normalized = a_gap / frequency
                if worst_by_a is None or normalized > worst_by_a[0]:
                    worst_by_a = (
                        normalized,
                        counts,
                        a,
                        threshold,
                        a_left,
                        2 * binary_tours[a] * frequency,
                        frequency,
                    )
            if gap > 0:
                return (
                    False,
                    worst,
                    worst_components,
                    worst_by_a,
                    worst_assignment,
                    worst_assignment_components,
                    worst_toughness,
                    worst_toughness_compressed,
                    worst_toughness_guide,
                )
    return (
        True,
        worst,
        worst_components,
        worst_by_a,
        worst_assignment,
        worst_assignment_components,
        worst_toughness,
        worst_toughness_compressed,
        worst_toughness_guide,
    )


def exhaustive_graphs(n):
    size = n + 1
    pairs = list(combinations(range(size), 2))
    connected = 0
    global_worst = None
    global_worst_components = None
    global_worst_by_a = None
    global_worst_assignment = None
    global_worst_assignment_components = None
    global_worst_toughness = None
    global_worst_toughness_compressed = None
    global_worst_toughness_guide = None
    for graph_mask in range(1 << len(pairs)):
        infinity = size + 1.0
        d = [[0.0 if i == j else infinity for j in range(size)] for i in range(size)]
        for index, (u, v) in enumerate(pairs):
            if graph_mask & (1 << index):
                d[u][v] = d[v][u] = 1.0
        for k in range(size):
            for i in range(size):
                for j in range(size):
                    d[i][j] = min(d[i][j], d[i][k] + d[k][j])
        if any(d[0][v] > size for v in range(size)):
            continue
        connected += 1
        (
            ok,
            worst,
            worst_components,
            worst_by_a,
            worst_assignment,
            worst_assignment_components,
            worst_toughness,
            worst_toughness_compressed,
            worst_toughness_guide,
        ) = check_metric(d)
        if global_worst is None or worst[0] > global_worst[0]:
            global_worst = (worst[0], graph_mask, *worst[1:], d)
        if global_worst_components is None or worst_components[0] > global_worst_components[0]:
            global_worst_components = (worst_components[0], graph_mask, *worst_components[1:], d)
        if global_worst_by_a is None or worst_by_a[0] > global_worst_by_a[0]:
            global_worst_by_a = (worst_by_a[0], graph_mask, *worst_by_a[1:], d)
        if global_worst_assignment is None or worst_assignment[0] > global_worst_assignment[0]:
            global_worst_assignment = (
                worst_assignment[0], graph_mask, *worst_assignment[1:], d
            )
        if (
            global_worst_assignment_components is None
            or worst_assignment_components[0] > global_worst_assignment_components[0]
        ):
            global_worst_assignment_components = (
                worst_assignment_components[0],
                graph_mask,
                *worst_assignment_components[1:],
                d,
            )
        if global_worst_toughness is None or worst_toughness[0] > global_worst_toughness[0]:
            global_worst_toughness = (
                worst_toughness[0], graph_mask, *worst_toughness[1:], d
            )
        if (
            global_worst_toughness_compressed is None
            or worst_toughness_compressed[0] > global_worst_toughness_compressed[0]
        ):
            global_worst_toughness_compressed = (
                worst_toughness_compressed[0],
                graph_mask,
                *worst_toughness_compressed[1:],
                d,
            )
        if (
            global_worst_toughness_guide is None
            or worst_toughness_guide[0] > global_worst_toughness_guide[0]
        ):
            global_worst_toughness_guide = (
                worst_toughness_guide[0],
                graph_mask,
                *worst_toughness_guide[1:],
                d,
            )
        if not ok:
            print("COUNTEREXAMPLE", global_worst)
            return
    print(f"exhausted {connected} connected graphs; worst={global_worst}")
    print(f"worst against component bound={global_worst_components}")
    print(f"worst after conditioning on A={global_worst_by_a}")
    print(f"worst against assignment defect={global_worst_assignment}")
    print(
        "worst against assignment plus component defects="
        f"{global_worst_assignment_components}"
    )
    print(f"worst against toughness defect={global_worst_toughness}")
    print(
        "worst compressed term against toughness="
        f"{global_worst_toughness_compressed}"
    )
    print(f"worst guide term against toughness={global_worst_toughness_guide}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--metrics", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--exhaustive-graphs", action="store_true")
    args = parser.parse_args()
    if args.exhaustive_graphs:
        exhaustive_graphs(args.clients)
        return
    rng = random.Random(args.seed)
    families = (
        "unit", "path", "cycle", "star", "binary", "random-tree",
        "graph12", "graph123", "client-graph-19", "euclidean",
        "multiscale-graph", "multiscale-complete",
    )
    global_worst = None
    global_worst_components = None
    global_worst_by_a = None
    global_worst_assignment = None
    global_worst_assignment_components = None
    global_worst_toughness = None
    global_worst_toughness_compressed = None
    global_worst_toughness_guide = None
    for i in range(args.metrics):
        family = families[i % len(families)]
        d = named_metric(family, args.clients, rng)
        (
            ok,
            worst,
            worst_components,
            worst_by_a,
            worst_assignment,
            worst_assignment_components,
            worst_toughness,
            worst_toughness_compressed,
            worst_toughness_guide,
        ) = check_metric(
            d,
            exhaustive_counts=args.clients <= 5,
            rng=rng,
            random_counts=1000,
        )
        if global_worst is None or worst[0] > global_worst[0]:
            global_worst = (worst[0], family, *worst[1:], d)
            print("worst", global_worst, flush=True)
        if global_worst_components is None or worst_components[0] > global_worst_components[0]:
            global_worst_components = (worst_components[0], family, *worst_components[1:], d)
        if global_worst_by_a is None or worst_by_a[0] > global_worst_by_a[0]:
            global_worst_by_a = (worst_by_a[0], family, *worst_by_a[1:], d)
        if global_worst_assignment is None or worst_assignment[0] > global_worst_assignment[0]:
            global_worst_assignment = (
                worst_assignment[0], family, *worst_assignment[1:], d
            )
        if (
            global_worst_assignment_components is None
            or worst_assignment_components[0] > global_worst_assignment_components[0]
        ):
            global_worst_assignment_components = (
                worst_assignment_components[0],
                family,
                *worst_assignment_components[1:],
                d,
            )
        if global_worst_toughness is None or worst_toughness[0] > global_worst_toughness[0]:
            global_worst_toughness = (
                worst_toughness[0], family, *worst_toughness[1:], d
            )
        if (
            global_worst_toughness_compressed is None
            or worst_toughness_compressed[0] > global_worst_toughness_compressed[0]
        ):
            global_worst_toughness_compressed = (
                worst_toughness_compressed[0],
                family,
                *worst_toughness_compressed[1:],
                d,
            )
        if (
            global_worst_toughness_guide is None
            or worst_toughness_guide[0] > global_worst_toughness_guide[0]
        ):
            global_worst_toughness_guide = (
                worst_toughness_guide[0],
                family,
                *worst_toughness_guide[1:],
                d,
            )
        if not ok:
            print("COUNTEREXAMPLE")
            return
    print("no violation; worst=", global_worst)
    print("worst against component bound=", global_worst_components)
    print("worst after conditioning on A=", global_worst_by_a)
    print("worst against assignment defect=", global_worst_assignment)
    print(
        "worst against assignment plus component defects=",
        global_worst_assignment_components,
    )
    print("worst against toughness defect=", global_worst_toughness)
    print(
        "worst compressed term against toughness=",
        global_worst_toughness_compressed,
    )
    print("worst guide term against toughness=", global_worst_toughness_guide)


if __name__ == "__main__":
    main()
