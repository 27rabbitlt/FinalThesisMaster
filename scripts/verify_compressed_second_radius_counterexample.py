#!/usr/bin/env python3
"""Exact counterexample to bounding compressed attachment by second radii.

The tree metric has a query client v and nine arms

    v --1-- u_i --9/10-- w_i --delta-- x_i,

plus a depot joined to v by length 3.  The w_i,x_i are always active,
u_i has probability 27/500, v has probability 39/500, and
delta=1/10^7.
For B,C iid activation samples and T=f_B(C), this verifies exactly that

    A1 = sum_z p_z E d(z,T)  >  sum_z p_z E Y_z(B),

where Y_z(B) is the second-smallest distance to B after adding a second
copy of the depot.  It also checks the closed forms used in the proof note.
"""

from __future__ import annotations

from fractions import Fraction


M = 9
P_U = Fraction(27, 500)
P_V = Fraction(39, 500)
ARM = Fraction(1)
OUTSIDE = Fraction(99, 100)
DELTA = Fraction(1, 10**7)
DEPOT_EDGE = Fraction(3)

V = 0
U = tuple(range(1, M + 1))
W = tuple(range(M + 1, 2 * M + 1))
X = tuple(range(2 * M + 1, 3 * M + 1))
N = 1 + 3 * M
ROOT = N
STOCHASTIC = (V, *U)
DETERMINISTIC_MASK = sum(1 << z for z in (*W, *X))
PROBABILITIES = (P_V,) + (P_U,) * M + (Fraction(1),) * (2 * M)


def tree_metric() -> list[list[Fraction]]:
    infinity = Fraction(10**9)
    d = [
        [Fraction() if i == j else infinity for j in range(N + 1)]
        for i in range(N + 1)
    ]

    def edge(a: int, b: int, length: Fraction) -> None:
        d[a][b] = d[b][a] = length

    edge(ROOT, V, DEPOT_EDGE)
    for i in range(M):
        edge(V, U[i], ARM)
        edge(U[i], W[i], OUTSIDE)
        edge(W[i], X[i], DELTA)
    for k in range(N + 1):
        for i in range(N + 1):
            for j in range(N + 1):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    return d


def realization_probability(mask: int) -> Fraction:
    result = Fraction(1)
    for position, vertex in enumerate(STOCHASTIC):
        probability = PROBABILITIES[vertex]
        result *= probability if mask & (1 << position) else 1 - probability
    return result


def full_mask(stochastic_mask: int) -> int:
    result = DETERMINISTIC_MASK
    for position, vertex in enumerate(STOCHASTIC):
        if stochastic_mask & (1 << position):
            result |= 1 << vertex
    return result


def members(mask: int):
    return tuple(v for v in range(N) if mask & (1 << v))


def evaluate():
    """Independent enumeration of the stochastic B,C bits.

    The always-active outside pairs make all needed distances depend only on
    whether v is retained and on the masks of retained u_i centers.  This
    avoids a generic 2^(2N) subset loop while still enumerating every random
    outcome exactly.
    """
    attachment = Fraction()
    radius = Fraction()
    second_radius = Fraction()
    u_probabilities = [
        P_U ** mask.bit_count() * (1 - P_U) ** (M - mask.bit_count())
        for mask in range(1 << M)
    ]
    for bv in (0, 1):
        probability_bv = P_V if bv else 1 - P_V
        for b_mask, probability_bu in enumerate(u_probabilities):
            probability_b = probability_bv * probability_bu
            b_count = b_mask.bit_count()
            v_radius = Fraction() if bv else (Fraction(1) if b_mask else 1 + OUTSIDE)
            radius += probability_b * (
                P_V * v_radius + P_U * OUTSIDE * (M - b_count)
            )
            if bv:
                v_second = Fraction(1) if b_mask else 1 + OUTSIDE
            else:
                v_second = Fraction(1) if b_count >= 2 else 1 + OUTSIDE
            second_radius += probability_b * (
                P_V * v_second
                + P_U * (M * OUTSIDE + (M - b_count) * DELTA)
                + 2 * M * DELTA
            )
            for cv in (0, 1):
                probability_cv = P_V if cv else 1 - P_V
                for c_mask, probability_cu in enumerate(u_probabilities):
                    probability = probability_b * probability_cv * probability_cu
                    retained_u = b_mask & c_mask
                    if not bv and cv and b_mask:
                        retained_u |= b_mask & -b_mask
                    retained_v = bv and cv
                    if retained_v:
                        v_attachment = Fraction()
                    elif retained_u:
                        v_attachment = Fraction(1)
                    else:
                        v_attachment = 1 + OUTSIDE
                    attachment += probability * (
                        P_V * v_attachment
                        + P_U * OUTSIDE * (M - retained_u.bit_count())
                    )
    return attachment, radius, second_radius


def closed_forms():
    z = 1 - P_U
    a = (1 - P_U**2) ** M
    b = z**M + (1 - P_V) * M * P_U * z ** (M - 1)
    empty_near_core = (1 - P_V) * (a + P_V * z**M)
    extra_retained_u = P_V * (1 - P_V) * z * (1 - z**M)
    attachment = (
        P_V * (1 - P_V**2 + OUTSIDE * empty_near_core)
        + P_U
        * OUTSIDE
        * (M * (1 - P_U**2) - extra_retained_u)
    )
    radius = (
        P_V * (1 - P_V) * (1 + OUTSIDE * z**M)
        + M * P_U * OUTSIDE * (1 - P_U)
    )
    second_radius = (
        P_V * (1 + OUTSIDE * b)
        + M * P_U * (OUTSIDE + (1 - P_U) * DELTA)
        + 2 * M * DELTA
    )
    gap = (
        P_V
        * (
            -P_V**2
            + OUTSIDE * (empty_near_core - b)
        )
        - M * P_U**3 * OUTSIDE
        - P_U * OUTSIDE * extra_retained_u
        - M * P_U * (1 - P_U) * DELTA
        - 2 * M * DELTA
    )
    posterior = 2 * (
        DEPOT_EDGE + M * (ARM + OUTSIDE + DELTA)
    )
    return attachment, radius, second_radius, gap, posterior


def main() -> None:
    d = tree_metric()
    assert all(
        d[i][j] <= d[i][k] + d[k][j]
        for i in range(N + 1)
        for j in range(N + 1)
        for k in range(N + 1)
    )
    direct = evaluate()
    attachment, radius, second_radius, gap, posterior = closed_forms()
    assert direct == (attachment, radius, second_radius)
    assert gap == attachment - second_radius
    assert gap > 0
    assert attachment + radius < 2 * posterior
    print(f"A1={float(attachment):.12f}")
    print(f"sum p E[Y]={float(second_radius):.12f}")
    print(f"strict gap={float(gap):.12f}")
    print(f"R={float(radius):.12f}, P={float(posterior):.12f}")
    print("all exact Fraction checks passed")


if __name__ == "__main__":
    main()
