#!/usr/bin/env python3
"""Exact witness against the global-first-discovery path construction.

The metric is the line {0,1,...,n}, rooted at 0.  Every client is active
with probability p.  Draw independent auxiliary scenarios forever.  In each
scenario, append all as-yet undiscovered active clients in increasing order;
empty discovery rounds are ignored.  Increasing order is the restriction of
an optimal posterior tour in every scenario.

The resulting random order pi defines the rooted path

    0, pi(1), ..., pi(n).

This program computes its expected hazard exactly by a subset recurrence and
compares it with the exact posterior value.  It also gives the construction
the strongest possible tie-breaking: after seeing a batch and the complete
history, it may choose any bitonic order of that batch, hence the restriction
of any optimal line-metric posterior tour.  Even this optimal policy violates
the hoped-for 29/20 hazard bound on the chosen finite instance.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache


N = 8
P_ACTIVE = Fraction(1, 10)


def occupancy(k: int) -> Fraction:
    return 1 - (1 - P_ACTIVE) ** k


Q = [occupancy(k) for k in range(N + 1)]


def vertices(mask: int) -> list[int]:
    """Return client coordinates (and labels) in increasing order."""
    return [i + 1 for i in range(N) if mask & (1 << i)]


@lru_cache(maxsize=None)
def expected_hazard_tail(last: int, unseen: int) -> Fraction:
    """Expected remaining hazard after ``last``, conditional on ``unseen``.

    The next nonempty batch is a Bernoulli-p subset of ``unseen`` conditioned
    to be nonempty.  Before its j-th member is appended, the path suffix has
    exactly |unseen|-j+1 clients, giving the corresponding occupancy factor.
    """
    if unseen == 0:
        return Fraction(0)

    m = unseen.bit_count()
    total = Fraction(0)
    batch = unseen
    while batch:
        ordered = vertices(batch)
        k = len(ordered)
        probability = (
            P_ACTIVE**k * (1 - P_ACTIVE) ** (m - k) / Q[m]
        )

        immediate = Q[m] * abs(last - ordered[0])
        for j in range(1, k):
            immediate += Q[m - j] * abs(ordered[j] - ordered[j - 1])

        total += probability * (
            immediate
            + expected_hazard_tail(ordered[-1], unseen ^ batch)
        )
        batch = (batch - 1) & unseen
    return total


@lru_cache(maxsize=None)
def best_bitonic_hazard_tail(last: int, unseen: int) -> Fraction:
    """Optimal append-only hazard when every batch may be ordered bitonically.

    An optimal depot-rooted tour on a line goes monotonically away from the
    depot on one side of its cycle and monotonically back on the other.  Its
    restriction to the newly discovered clients is therefore an increasing
    subsequence followed by a decreasing subsequence.  Conversely every such
    bitonic order extends to an optimal tour by assigning all other active
    clients to either monotone side.  The minimum below consequently allows
    every optimal posterior-tour tie-break, chosen after seeing the history
    and current batch.
    """
    if unseen == 0:
        return Fraction(0)

    m = unseen.bit_count()
    total = Fraction(0)
    batch = unseen
    while batch:
        ordered = vertices(batch)
        k = len(ordered)
        probability = (
            P_ACTIVE**k * (1 - P_ACTIVE) ** (m - k) / Q[m]
        )

        best = None
        for outward_mask in range(1 << k):
            sequence = [
                ordered[j] for j in range(k) if outward_mask & (1 << j)
            ]
            sequence.extend(
                ordered[j]
                for j in range(k - 1, -1, -1)
                if not outward_mask & (1 << j)
            )
            immediate = Q[m] * abs(last - sequence[0])
            for j in range(1, k):
                immediate += Q[m - j] * abs(sequence[j] - sequence[j - 1])
            value = immediate + best_bitonic_hazard_tail(
                sequence[-1], unseen ^ batch
            )
            if best is None or value < best:
                best = value

        assert best is not None
        total += probability * best
        batch = (batch - 1) & unseen
    return total


def posterior_value() -> Fraction:
    # A nonempty realization whose furthest active client is m has tour cost
    # 2m and probability p(1-p)^(N-m).
    return 2 * sum(
        Fraction(m) * P_ACTIVE * (1 - P_ACTIVE) ** (N - m)
        for m in range(1, N + 1)
    )


def main() -> None:
    hazard = expected_hazard_tail(0, (1 << N) - 1)
    best_hazard = best_bitonic_hazard_tail(0, (1 << N) - 1)
    posterior = posterior_value()
    target = Fraction(29, 20) * posterior

    assert hazard > target
    assert best_hazard > target
    assert best_hazard <= hazard
    assert expected_hazard_tail.cache_info().currsize <= (N + 1) * (1 << N)
    assert best_bitonic_hazard_tail.cache_info().currsize <= (N + 1) * (1 << N)

    print(f"n={N}, p={P_ACTIVE}")
    print(f"posterior={posterior} = {float(posterior):.12f}")
    print(f"discovery_path_hazard={hazard} = {float(hazard):.12f}")
    print(f"ratio={float(hazard / posterior):.12f}")
    print(f"best_bitonic_hazard={best_hazard} = {float(best_hazard):.12f}")
    print(f"best_bitonic_ratio={float(best_hazard / posterior):.12f}")
    print(
        f"best_bitonic_strict_slack={best_hazard - target} "
        f"= {float(best_hazard - target):.12f}"
    )
    print("exact global-first-discovery barrier verified")


if __name__ == "__main__":
    main()
