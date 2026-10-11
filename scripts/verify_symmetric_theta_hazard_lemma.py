#!/usr/bin/env python3
"""Exact finite audit of the symmetric-theta hazard lemma (23k)--(23aa).

The proof in structural_barriers_and_targets.md applies to every weighted
path template and arbitrary heterogeneous internal probabilities, copied
independently onto three r--t paths, with t always active.  This checker uses
one nonuniform rational template and verifies:

* tsp(A) = 4L - 2 max_i M_i for every realization;
* length-biased cut averaging and the deterministic best-cut choice;
* the exact hazard of the resulting theta spanning tree; and
* the closed/open/separated parallel-state gluing law;
* Phi <= P and the open-state 4/3 bound; and
* the exact midpoint warning (23z)--(23aa), all with Fraction arithmetic.
"""

from fractions import Fraction

from search_theta_cactus_deletion import subset_tsp_costs
from search_theta_weighted_ust import (
    metric_closure,
    subtree_cost,
    theta_graph,
)


EDGE_LENGTHS = (Fraction(1), Fraction(2), Fraction(3))
INTERNAL_PROBABILITIES = (Fraction(1, 3), Fraction(3, 5))


def path_vertices(edges, path_edge_indices):
    return [edges[path_edge_indices[0]][0]] + [
        edges[index][1] for index in path_edge_indices
    ]


def marked_positions(mask, vertices):
    # Both common terminals are marked; client v has mask bit v-1.
    return [
        position
        for position, vertex in enumerate(vertices)
        if position in (0, len(vertices) - 1)
        or mask & (1 << (vertex - 1))
    ]


def gaps(mask, vertices):
    marks = marked_positions(mask, vertices)
    prefix = [Fraction(0)]
    for length in EDGE_LENGTHS:
        prefix.append(prefix[-1] + length)
    return [prefix[right] - prefix[left] for left, right in zip(marks, marks[1:])]


def gap_containing_edge(mask, vertices, cut_position):
    marks = marked_positions(mask, vertices)
    left = max(position for position in marks if position <= cut_position)
    right = min(position for position in marks if position > cut_position)
    return sum(EDGE_LENGTHS[left:right], Fraction(0))


def realization_probability(mask, probabilities):
    answer = Fraction(1)
    for vertex, p in enumerate(probabilities, start=1):
        answer *= p if mask & (1 << (vertex - 1)) else 1 - p
    return answer


def parallel_states(left, right):
    """Min-plus composition (23v) for states ordered as (D,E,O)."""
    d_left, e_left, o_left = left
    d_right, e_right, o_right = right
    return (
        d_left + d_right,
        min(
            e_left + d_right,
            d_left + e_right,
            e_left + e_right,
            o_left + o_right,
        ),
        min(
            o_left + d_right,
            d_left + o_right,
            o_left + e_right,
            e_left + o_right,
        ),
    )


def main() -> None:
    paths = (EDGE_LENGTHS,) * 3
    vertex_count, edges, path_edge_indices = theta_graph(paths)
    vertices = [path_vertices(edges, indices) for indices in path_edge_indices]
    probabilities = (
        Fraction(1),  # terminal t
        *INTERNAL_PROBABILITIES,
        *INTERNAL_PROBABILITIES,
        *INTERNAL_PROBABILITIES,
    )
    assert len(probabilities) == vertex_count - 1

    distance = metric_closure(vertex_count, edges)
    tour_costs = subset_tsp_costs(vertex_count, distance)
    total_length = sum(EDGE_LENGTHS, Fraction(0))

    posterior = Fraction(0)
    expected_open = Fraction(0)
    expected_max_gap = Fraction(0)
    expected_gap_square = Fraction(0)
    expected_cut_gaps = [Fraction(0)] * len(EDGE_LENGTHS)
    for mask, actual_tour in enumerate(tour_costs):
        probability = realization_probability(mask, probabilities)
        if not probability:
            continue
        maxima = [max(gaps(mask, path)) for path in vertices]
        predicted_tour = 4 * total_length - 2 * max(maxima)
        assert actual_tour == predicted_tour
        ordered = sorted(maxima, reverse=True)
        compact_open = 3 * total_length - 2 * max(
            Fraction(0), ordered[0] + ordered[1] - total_length
        )
        retained = [total_length - maximum for maximum in maxima]
        parity_open = min(
            3 * total_length,
            *(
                total_length + 2 * retained[j] + 2 * retained[k]
                for i in range(3)
                for j, k in [tuple(x for x in range(3) if x != i)]
            ),
        )
        assert compact_open == parity_open
        path_states = [
            (2 * (total_length - maximum), 2 * total_length, total_length)
            for maximum in maxima
        ]
        theta_states = parallel_states(
            parallel_states(path_states[0], path_states[1]), path_states[2]
        )
        assert theta_states[1] == actual_tour
        assert theta_states[2] == compact_open
        posterior += probability * actual_tour
        expected_open += probability * compact_open
        expected_max_gap += probability * max(maxima)
        expected_gap_square += probability * maxima[0] ** 2
        for cut in range(len(EDGE_LENGTHS)):
            expected_cut_gaps[cut] += (
                probability * gap_containing_edge(mask, vertices[0], cut)
            )

    length_biased_gap = sum(
        length * value
        for length, value in zip(EDGE_LENGTHS, expected_cut_gaps)
    ) / total_length
    assert length_biased_gap >= expected_gap_square / total_length
    best_cut = max(range(len(EDGE_LENGTHS)), key=expected_cut_gaps.__getitem__)
    best_gap = expected_cut_gaps[best_cut]
    assert best_gap >= length_biased_gap

    chosen = tuple(
        index
        for index in range(len(edges))
        if index not in {
            path_edge_indices[1][best_cut],
            path_edge_indices[2][best_cut],
        }
    )
    hazard = Fraction(0)
    for mask in range(1 << (vertex_count - 1)):
        probability = realization_probability(mask, probabilities)
        hazard += probability * subtree_cost(mask, vertex_count, edges, chosen)

    assert hazard == 3 * total_length - 2 * best_gap
    assert posterior == 4 * total_length - 2 * expected_max_gap
    assert hazard <= posterior
    assert hazard <= Fraction(4, 3) * expected_open

    midpoint_empty_probability = Fraction(4, 5)
    midpoint_connector = (2 - midpoint_empty_probability)
    midpoint_open = (
        3
        - 3 * midpoint_empty_probability
        + midpoint_empty_probability**3
    )
    assert midpoint_connector == Fraction(6, 5)
    assert midpoint_open == Fraction(139, 125)
    assert midpoint_connector - midpoint_open == Fraction(11, 125)
    assert midpoint_connector < Fraction(4, 3) * midpoint_open

    print(f"P = {posterior} = {float(posterior):.12f}")
    print(f"chosen cut position = {best_cut}; E gap = {best_gap}")
    print(f"Phi = {hazard} = {float(hazard):.12f}")
    print(f"Phi/P = {hazard / posterior} = {float(hazard / posterior):.12f}")
    print(f"open-state expectation = {expected_open}")
    print(f"Phi/open = {hazard / expected_open} = "
          f"{float(hazard / expected_open):.12f}")
    print(
        "midpoint connector/open = "
        f"{midpoint_connector}/{midpoint_open} = "
        f"{midpoint_connector / midpoint_open}"
    )


if __name__ == "__main__":
    main()
