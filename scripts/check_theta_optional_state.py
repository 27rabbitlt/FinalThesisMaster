#!/usr/bin/env python3
"""Exact state audit for a three-path theta block.

This is deliberately a small, focused checker.  A path template consists of
edge lengths and independent probabilities for its internal vertices.  It
enumerates the finite set of fixed-tree actions (one intact backbone and one
cut on each other path), computes their optional/required connector hazards,
and computes the exact posterior D/E/O and rooted-optional states.
"""

from __future__ import annotations

from fractions import Fraction
from functools import reduce
from itertools import product
from operator import mul


C = Fraction(29, 20)


def fprod(values):
    return reduce(mul, values, Fraction(1))


def realization_probability(bits, probabilities):
    ans = Fraction(1)
    for active, p in zip(bits, probabilities):
        ans *= p if active else 1 - p
    return ans


def path_statistics(lengths, bits):
    """Return (largest portal-to-portal gap, farthest active prefix)."""
    prefix = [Fraction(0)]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    marks = [0] + [j + 1 for j, active in enumerate(bits) if active] + [len(lengths)]
    gaps = [prefix[v] - prefix[u] for u, v in zip(marks, marks[1:])]
    farthest = max((prefix[j + 1] for j, active in enumerate(bits) if active), default=Fraction(0))
    return max(gaps), farthest


def branch_hazard(lengths, probabilities, cut):
    """Expected two-root forest cost after deleting edge ``cut``."""
    total = Fraction(0)
    # Edges strictly left of the cut are rooted at r.
    for edge in range(cut):
        occupancy = 1 - fprod(1 - p for p in probabilities[edge:cut])
        total += lengths[edge] * occupancy
    # Edges strictly right of the cut are rooted at t.
    for edge in range(cut + 1, len(lengths)):
        occupancy = 1 - fprod(1 - p for p in probabilities[cut:edge])
        total += lengths[edge] * occupancy
    return total


def connector_states(templates, backbone, cuts):
    """Return exact (H0,H1,HD) for one fixed action.

    ``cuts`` specifies a deleted edge on every path.  The backbone cut is
    restored for H0/H1 and used for the exported separating state HD.
    """
    lengths = [sum(edge_lengths, Fraction(0)) for edge_lengths, _ in templates]
    branch = [
        branch_hazard(edge_lengths, probabilities, cut)
        for (edge_lengths, probabilities), cut in zip(templates, cuts)
    ]
    hd = sum(branch, Fraction(0))
    h1 = lengths[backbone] + sum(
        branch[i] for i in range(3) if i != backbone
    )

    # On a cut path, precisely the internal vertices at or to the right of
    # the cut edge lie in the t-rooted component.
    t_side_probabilities = []
    for i, ((_, probabilities), cut) in enumerate(zip(templates, cuts)):
        if i != backbone:
            t_side_probabilities.extend(probabilities[cut:])
    t_side_inactive = fprod(1 - p for p in t_side_probabilities)

    edge_lengths, probabilities = templates[backbone]
    backbone_h0 = Fraction(0)
    for edge, length in enumerate(edge_lengths):
        suffix_inactive = fprod(1 - p for p in probabilities[edge:])
        occupancy = 1 - t_side_inactive * suffix_inactive
        backbone_h0 += length * occupancy
    h0 = backbone_h0 + sum(branch[i] for i in range(3) if i != backbone)
    return h0, h1, hd


def posterior_states(templates):
    """Return expected (P0,E,O,D) by exact product enumeration."""
    path_probabilities = [probabilities for _, probabilities in templates]
    lengths = [sum(edge_lengths, Fraction(0)) for edge_lengths, _ in templates]
    p0 = even = odd = separated = Fraction(0)
    bit_ranges = [list(product((False, True), repeat=len(ps))) for ps in path_probabilities]
    for path_bits in product(*bit_ranges):
        probability = fprod(
            realization_probability(bits, ps)
            for bits, ps in zip(path_bits, path_probabilities)
        )
        if not probability:
            continue
        stats = [
            path_statistics(edge_lengths, bits)
            for (edge_lengths, _), bits in zip(templates, path_bits)
        ]
        maxima = [item[0] for item in stats]
        radii = [item[1] for item in stats]
        retained = [lengths[i] - maxima[i] for i in range(3)]
        total_length = sum(lengths, Fraction(0))

        d = 2 * sum(retained, Fraction(0))
        o = min(
            [total_length]
            + [
                lengths[i] + 2 * retained[j] + 2 * retained[k]
                for i in range(3)
                for j, k in [tuple(x for x in range(3) if x != i)]
            ]
        )
        e = min(
            [total_length + lengths[k] - 2 * maxima[k] for k in range(3)]
            + [
                2 * (lengths[i] + retained[j] + retained[k])
                for i in range(3)
                for j, k in [tuple(x for x in range(3) if x != i)]
            ]
        )
        rooted = min(2 * sum(radii, Fraction(0)), e)
        p0 += probability * rooted
        even += probability * e
        odd += probability * o
        separated += probability * d
    return p0, even, odd, separated


def actions(templates):
    cut_ranges = [range(len(lengths)) for lengths, _ in templates]
    return [
        (backbone, cuts, connector_states(templates, backbone, cuts))
        for backbone in range(3)
        for cuts in product(*cut_ranges)
    ]


def example():
    templates = (
        ((Fraction(1), Fraction(2)), (Fraction(1, 3),)),
        ((Fraction(2), Fraction(1)), (Fraction(1, 2),)),
        ((Fraction(1), Fraction(1)), (Fraction(2, 3),)),
    )
    p0, even, odd, separated = posterior_states(templates)
    print("posterior (P0,E,O,D):", p0, even, odd, separated)
    for backbone, cuts, state in actions(templates):
        h0, h1, hd = state
        if h0 <= C * p0 and h1 <= C * even and h1 <= C * odd and hd <= C * separated:
            print("coordinatewise witness:", backbone, cuts, state)
            break
    else:
        print("no deterministic coordinatewise witness")


if __name__ == "__main__":
    example()
