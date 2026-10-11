#!/usr/bin/env python3
"""Exact audit of the nonidentical theta open/closed 29/20 lemma.

The three r--t paths may have unrelated edge lengths and unrelated product
activation laws on their internal vertices.  Both portals are required.  The
checker verifies the exact parity-state formulas, constructs the deterministic
fixed-cut spanning tree from the square-gap credits, and checks its hazard
against both the connected closed state and the open odd-parity state.
"""

from fractions import Fraction

from search_theta_cactus_deletion import subset_tsp_costs
from search_theta_weighted_ust import metric_closure, subtree_cost, theta_graph


TEMPLATES = (
    ((Fraction(1), Fraction(2), Fraction(3)),
     (Fraction(1, 3), Fraction(3, 5))),
    ((Fraction(3, 2), Fraction(1, 2), Fraction(5, 2)),
     (Fraction(2, 5), Fraction(4, 7))),
    ((Fraction(2, 3), Fraction(7, 3), Fraction(4, 3)),
     (Fraction(3, 4), Fraction(1, 5))),
)


def path_vertices(edges, indices):
    return [edges[indices[0]][0]] + [edges[index][1] for index in indices]


def realization_probability(mask, probabilities):
    answer = Fraction(1)
    for vertex, p in enumerate(probabilities, start=1):
        answer *= p if mask & (1 << (vertex - 1)) else 1 - p
    return answer


def path_gaps(mask, vertices, lengths):
    prefix = [Fraction(0)]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    marks = [
        position
        for position, vertex in enumerate(vertices)
        if position in (0, len(vertices) - 1)
        or mask & (1 << (vertex - 1))
    ]
    return prefix, marks, [
        prefix[right] - prefix[left] for left, right in zip(marks, marks[1:])
    ]


def main() -> None:
    path_lengths = tuple(template[0] for template in TEMPLATES)
    vertex_count, edges, path_indices = theta_graph(path_lengths)
    vertices = [path_vertices(edges, indices) for indices in path_indices]
    probabilities = [Fraction(0)] * (vertex_count - 1)
    probabilities[0] = 1  # common terminal t
    for path, (_, internal_p) in enumerate(TEMPLATES):
        for vertex, p in zip(vertices[path][1:-1], internal_p):
            probabilities[vertex - 1] = p
    probabilities = tuple(probabilities)

    lengths = [sum(template[0], Fraction(0)) for template in TEMPLATES]
    total_length = sum(lengths, Fraction(0))
    tour_costs = subset_tsp_costs(
        vertex_count, metric_closure(vertex_count, edges)
    )

    closed = Fraction(0)
    opened = Fraction(0)
    square_credits = [Fraction(0)] * 3
    cut_credits = [[Fraction(0)] * len(TEMPLATES[i][0]) for i in range(3)]
    for mask, actual_tour in enumerate(tour_costs):
        probability = realization_probability(mask, probabilities)
        if not probability:
            continue
        maxima = []
        for i in range(3):
            prefix, marks, gaps = path_gaps(mask, vertices[i], TEMPLATES[i][0])
            maximum = max(gaps)
            maxima.append(maximum)
            square_credits[i] += probability * maximum**2 / lengths[i]
            for cut in range(len(TEMPLATES[i][0])):
                left = max(mark for mark in marks if mark <= cut)
                right = min(mark for mark in marks if mark > cut)
                cut_credits[i][cut] += probability * (prefix[right] - prefix[left])

        retained = [lengths[i] - maxima[i] for i in range(3)]
        two_odd = min(
            total_length + lengths[k] - 2 * maxima[k] for k in range(3)
        )
        zero_odd_connected = min(
            2 * (lengths[i] + retained[j] + retained[k])
            for i in range(3)
            for j, k in [tuple(x for x in range(3) if x != i)]
        )
        predicted_closed = min(two_odd, zero_odd_connected)
        assert actual_tour == predicted_closed
        closed += probability * predicted_closed

        one_odd = min(
            lengths[i] + 2 * retained[j] + 2 * retained[k]
            for i in range(3)
            for j, k in [tuple(x for x in range(3) if x != i)]
        )
        opened += probability * min(total_length, one_odd)

    best_cuts = [max(range(len(values)), key=values.__getitem__) for values in cut_credits]
    actual_credits = [cut_credits[i][best_cuts[i]] for i in range(3)]
    for actual, square in zip(actual_credits, square_credits):
        assert actual >= square

    normalized_second_moments = [square_credits[i] / lengths[i] for i in range(3)]
    weighted_credits = [
        lengths[i] * normalized_second_moments[i] for i in range(3)
    ]
    backbone = min(range(3), key=weighted_credits.__getitem__)
    omitted = {
        path_indices[i][best_cuts[i]] for i in range(3) if i != backbone
    }
    chosen = tuple(index for index in range(len(edges)) if index not in omitted)

    hazard = Fraction(0)
    for mask in range(1 << (vertex_count - 1)):
        hazard += (
            realization_probability(mask, probabilities)
            * subtree_cost(mask, vertex_count, edges, chosen)
        )

    delta = sum(
        lengths[i] * (1 - normalized_second_moments[i]) for i in range(3)
    )
    c = min(weighted_credits)
    ell = min(lengths)
    product = (
        normalized_second_moments[0]
        * normalized_second_moments[1]
        * normalized_second_moments[2]
    )
    assert hazard <= delta + c
    assert closed >= delta + ell * product
    assert opened >= delta + ell * product
    assert 20 * c <= 9 * delta + 29 * ell * product
    assert hazard <= Fraction(29, 20) * closed
    assert hazard <= Fraction(29, 20) * opened

    print(f"backbone path: {backbone}; cuts: {best_cuts}")
    print(f"Phi = {hazard}")
    print(f"closed = {closed}; Phi/closed = {hazard / closed}")
    print(f"open = {opened}; Phi/open = {hazard / opened}")
    print(f"scalar certificate: 20c = {20*c}")
    print(f"                    9Delta+29ellQ = {9*delta + 29*ell*product}")


if __name__ == "__main__":
    main()
