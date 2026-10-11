#!/usr/bin/env python3
"""Exact checks for the modular-cell recursive-compression lemma.

For one independent cell with label probabilities ``p_v``, put

    w = sum p_v,   q = 1 - product(1-p_v).

The guide contains the cell with probability q.  A fresh fixed-cell hit
retains it with another independent probability q.  With unit backbone
charge and doubled unit attachment charge, its one-step signed saving is

    s = q(1-q)(1-2w),

and its posterior deficit contribution is

    delta = q - w(1-q).

The stronger structural cell inequality is

    s + 4 delta >= 3q/4.

It implies the RH10 cell inequality

    s >= q/5 - 4 delta

with an extra margin of ``11q/20``.

All arithmetic below is exact ``Fraction`` arithmetic.  The analytic proof
that this holds for every finite product cell is in ``recursive_core.md``.
"""

from __future__ import annotations

from fractions import Fraction as F
from itertools import product


def cell_terms(probabilities: tuple[F, ...]):
    w = sum(probabilities, F(0))
    empty = F(1)
    for probability in probabilities:
        empty *= 1 - probability
    q = 1 - empty
    saving = q * (1 - q) * (1 - 2 * w)
    deficit = q - w * (1 - q)
    strong_margin = saving + 4 * deficit - 3 * q / 4
    rh10_margin = saving - q / 5 + 4 * deficit
    route = q * q + 2 * w * (1 - q * q)
    return w, q, saving, deficit, strong_margin, rh10_margin, route


def verify_cell_grid() -> None:
    grid = (F(1, 20), F(1, 10), F(1, 4), F(1, 2), F(3, 4))
    checked = 0
    smallest = None
    for length in range(1, 6):
        for probabilities in product(grid, repeat=length):
            terms = cell_terms(probabilities)
            strong_margin = terms[4]
            rh10_margin = terms[5]
            assert strong_margin >= 0
            assert rh10_margin - strong_margin == F(11, 20) * terms[1]
            assert terms[6] <= F(26, 9) * terms[1]
            record = (strong_margin, probabilities, terms)
            if smallest is None or strong_margin < smallest[0]:
                smallest = record
            checked += 1
    print("exact cells checked", checked)
    print("smallest exact strong-cell margin", smallest)


def verify_heterogeneous_aggregate() -> None:
    groups = (
        (F(3, 1), (F(1, 100),) * 7),
        (F(5, 2), (F(1, 4), F(1, 3))),
        (F(7, 3), (F(3, 4), F(2, 5), F(1, 10))),
        (F(11, 5), (F(1, 2),)),
    )
    posterior = F(0)
    radius = F(0)
    compressed = F(0)
    saving = F(0)
    for length, probabilities in groups:
        w, q, cell_saving, _, strong_margin, _, route = cell_terms(probabilities)
        assert strong_margin >= 0
        posterior += length * q
        radius += length * w * (1 - q)
        compressed += length * route
        saving += length * cell_saving
    deficit = posterior - radius
    assert posterior + 2 * radius - compressed == saving
    assert saving + 4 * deficit >= 3 * posterior / 4
    assert saving >= posterior / 5 - 4 * deficit
    assert compressed <= F(26, 9) * posterior
    print("aggregate P", posterior)
    print("aggregate R", radius)
    print("aggregate U1", compressed)
    print("aggregate U1/P", float(compressed / posterior))
    print("aggregate strong margin", saving + 4 * deficit - 3 * posterior / 4)
    print("aggregate RH10 margin", saving - posterior / 5 + 4 * deficit)


def verify_equidistant_site_root_term() -> None:
    probabilities = (
        (F(1, 10), F(1, 10)),
        (F(1, 3),),
        (F(1, 20),) * 9,
        (F(3, 4), F(1, 8)),
    )
    terms = [cell_terms(group) for group in probabilities]
    q = [row[1] for row in terms]
    w = [row[0] for row in terms]
    q_any = 1
    q2_any = 1
    for occupancy in q:
        q_any *= 1 - occupancy
        q2_any *= 1 - occupancy * occupancy
    q_any = 1 - q_any
    q2_any = 1 - q2_any

    posterior = sum(q, F(0)) + q_any
    radius = sum((mass * (1 - occupancy) for mass, occupancy in zip(w, q)), F(0))
    compressed = (
        sum((occupancy * occupancy + 2 * mass * (1 - occupancy * occupancy)
             for mass, occupancy in zip(w, q)), F(0))
        + q2_any
    )
    deficit = posterior - radius
    recursive_saving = posterior + 2 * radius - compressed
    assert recursive_saving + 4 * deficit >= 3 * posterior / 4
    assert recursive_saving >= posterior / 5 - 4 * deficit
    assert compressed <= F(26, 9) * posterior
    print("equidistant-site P", posterior)
    print("equidistant-site U1", compressed)
    print("equidistant-site U1/P", float(compressed / posterior))
    print("equidistant-site RH10 margin", recursive_saving - posterior / 5 + 4 * deficit)


if __name__ == "__main__":
    verify_cell_grid()
    verify_heterogeneous_aggregate()
    verify_equidistant_site_root_term()
