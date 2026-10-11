#!/usr/bin/env python3
"""Exact counterexample to the conditional inequality M_B+2H_B<=tsp(B).

Use the shortest-path metric of the four-cycle (r,1,2,3,r), condition on
B={1,2,3}, and give every client probability 1/2.  Every Voronoi cell is a
singleton, hence q_b=w_b=1/2 and T is an iid half-subset of B.
"""

from fractions import Fraction


def distance(u: int, v: int) -> int:
    separation = abs(u - v)
    return min(separation, 4 - separation)


def tour_cost(mask: int) -> int:
    """TSP cost on the selected subset of clients 1,2,3 plus root 0."""
    table = (0, 2, 4, 4, 2, 4, 4, 4)
    return table[mask]


def main() -> None:
    probability = Fraction(1, 2) ** 3
    backbone = sum(probability * tour_cost(mask) for mask in range(8))
    transport = Fraction(0)
    for mask in range(8):
        retained = [0] + [b for b in range(1, 4) if mask & (1 << (b - 1))]
        for b in range(1, 4):
            if not mask & (1 << (b - 1)):
                cell_mass = Fraction(1, 2)
                transport += (
                    probability
                    * cell_mass
                    * min(distance(b, center) for center in retained)
                )
    full_tour = Fraction(4)
    lhs = backbone + 2 * transport
    assert backbone == 3
    assert transport == Fraction(13, 16)
    assert lhs == Fraction(37, 8)
    assert lhs - full_tour == Fraction(5, 8) > 0
    print("E tsp(T) =", backbone)
    print("H_B =", transport)
    print("M_B + 2 H_B =", lhs)
    print("tsp(B) =", full_tour)
    print("violation =", lhs - full_tour)


if __name__ == "__main__":
    main()
