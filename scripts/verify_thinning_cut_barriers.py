#!/usr/bin/env python3
"""Finite audits of optimal-forest and uncapped-cycle thinning barriers.

Forest optimization uses exhaustive subset DP with floating-point objectives.
Cycle tour costs are integers; their expectations are checked with Fractions.
The asymptotic counterexamples are proved in the accompanying research note.
"""

from fractions import Fraction as F
from math import exp, log, log1p
import random

from search_hazard_tree_ratio import best_hazard_tree, subset_tsp_costs


def capped_forest_case(groups, group_size, probability, backbone):
    count = groups * group_size
    outside = [u for u in range(count) if u not in backbone]

    def distance(u, v):
        if u == v:
            return 0
        if count in (u, v):
            return 2
        return 1 if (u // group_size - v // group_size) % groups in (0, 1, groups - 1) else 2

    # Collapse only the forest roots, choosing the cheapest Q endpoint per edge.
    roots = backbone | {count}
    d = [[distance(u, v) for v in outside]
         + [min(distance(u, q) for q in roots)] for u in outside]
    d.append([min(distance(u, q) for q in roots) for u in outside] + [0])
    optimum, _ = best_hazard_tree(d, [probability] * len(outside))
    holes = sum(all((u // group_size - i) % groups not in (0, 1, groups - 1)
                    for u in backbone) for i in range(groups))
    baseline = probability * len(outside)
    lower = baseline + (1 - probability) * (1 - (1 - probability) ** group_size) * holes / 3
    upper = baseline + 2 * holes
    assert lower - 1e-9 <= optimum <= upper + 1e-9


def uncapped_cycle_case(groups, group_size, probability):
    count = groups * group_size
    root = count

    def position(u):
        return 0 if u == root else u // group_size

    def distance(u, v):
        if u == v:
            return 0
        gap = abs(position(u) - position(v))
        return 1 if not gap else min(gap, groups - gap)

    d = [[distance(u, v) for v in range(count + 1)] for u in range(count + 1)]
    for u in range(count + 1):
        for v in range(count + 1):
            for w in range(count + 1):
                assert d[u][v] <= d[u][w] + d[w][v]
    tours = subset_tsp_costs(d)
    posterior = prior = attachment = good_probability = F(0)
    for mask, optimum in enumerate(tours):
        active = [u for u in range(count) if mask & (1 << u)]
        mass = probability ** len(active) * (1 - probability) ** (count - len(active))
        route = [root] + active
        fixed = sum(d[u][v] for u, v in zip(route, route[1:] + route[:1]))
        occupied = sorted({position(u) for u in route})
        assert fixed <= len(active) + 1 - len(occupied) + groups
        max_gap = max((occupied[(i + 1) % len(occupied)] - occupied[i]) % groups
                      or groups for i in range(len(occupied)))
        if max_gap <= groups / 2:
            assert optimum >= groups
            good_probability += mass
        posterior += mass * F(optimum)
        prior += mass * fixed
        attachment += mass * probability * sum(min(d[v][u] for u in route)
                                                for v in range(count))
    mean_count = count * probability
    empty_group = (1 - probability) ** group_size
    assert mean_count * (1 - probability) <= attachment <= posterior <= prior
    assert prior <= mean_count + 1 + (groups - 1) * empty_group
    failure_bound = groups * empty_group ** (groups // 2)
    assert 1 - good_probability <= failure_bound
    assert posterior >= groups * (1 - failure_bound)
    return len(tours)


def main():
    rng = random.Random(9341)
    forest_cases = 0
    for groups, size in ((4, 1), (5, 1), (6, 1), (4, 2)):
        count = groups * size
        for probability in (.02, .2, .7, 1.):
            masks = (list(range(1 << count)) if count <= 6
                     else [rng.randrange(1 << count) for _ in range(32)])
            for mask in masks:
                backbone = {u for u in range(count) if mask & (1 << u)}
                capped_forest_case(groups, size, probability, backbone)
                forest_cases += 1
    print(f'{forest_cases} optimal-forest DP checks passed (floating-point objectives)')

    realized_cases = 0
    for groups, size in ((4, 2), (5, 1), (6, 1), (8, 1)):
        for probability in (F(1, 4), F(2, 3), F(1)):
            realized_cases += uncapped_cycle_case(groups, size, probability)
    print(f'{realized_cases} cycle scenarios: exact tour and expectation inequalities passed')

    for groups in (32, 256, 4096, 65536):
        probability = groups ** -4
        # Ceiling m gives mp <= p log(k)/[-log(1-p)] + p.
        mean_upper = groups * (probability * log(groups) / -log1p(-probability) + probability)
        t = log(groups) ** -2
        failure_upper = groups * exp(-(groups // 2) / log(groups))
        backbone_lower = groups * max(0., 1 - failure_upper)
        denominator_upper = t * (mean_upper + 2) + 2 + mean_upper * probability
        print(f'k={groups}: rigorous-formula backbone/(tP+D) lower estimate = '
              f'{backbone_lower / denominator_upper:.6f}')


if __name__ == '__main__':
    main()
