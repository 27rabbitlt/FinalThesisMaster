#!/usr/bin/env python3
"""Generate subset-lattice TSP instances and run the solver.

Graph: 2^w vertices, one per subset of W = {0, ..., w-1}.
Distance: d(A, B) = 0 if A ⊆ B, else 1.
Depot: vertex 0 = ∅.

Probability model: for each size c = 1..w, every size-c subset S has
prob(S) = 1/C(w,c), so ~1 customer is active per level.
"""

import json
import subprocess
import sys
from math import comb


def popcount(x):
    return bin(x).count('1')


def generate_instance(w):
    n = 1 << w  # 2^w vertices

    # Distance matrix: d[i][j] = 0 if i ⊆ j (i.e. i & j == i), else 1
    dist = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0 if (i & j) == i else 1)
        dist.append(row)

    # Probabilities: for each level c, prob = 1/C(w,c)
    prob = [0.0] * n
    prob[0] = 1.0  # depot always active
    for i in range(1, n):
        c = popcount(i)
        prob[i] = 1.0 / comb(w, c)

    return {
        "n": n,
        "dist": dist,
        "prob": prob,
        "_comment": (
            f"Subset lattice: w={w}. "
            f"d(A,B)=0 iff A⊆B. "
            f"prob(S)=1/C({w},|S|), ~1 active per level."
        ),
    }


def main():
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    n = 1 << w
    data = generate_instance(w)
    filename = f"examples/subset_lattice_w{w}_all.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Subset lattice: w={w}, n={n} vertices")
    print(f"d(A,B) = 0 if A ⊆ B, else 1")
    print(f"Prob per level:")
    for c in range(1, w + 1):
        nc = comb(w, c)
        print(f"  |S|={c}: {nc} subsets, each prob {1.0/nc:.6f}, E[active]={1.0:.1f}")
    print(f"Total E[active customers] = {w}")

    solver = "./build/solver"
    cmd = [solver, filename]
    if n > 12:
        cmd.append("--no-apriori")
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)


if __name__ == "__main__":
    main()
