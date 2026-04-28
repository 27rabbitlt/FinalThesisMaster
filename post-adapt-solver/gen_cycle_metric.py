#!/usr/bin/env python3
"""Generate cycle-metric graph family.

Metric: d(u,v) = 0 if u==v, 2 if adjacent on cycle, 3 otherwise.
Depot = vertex 0 (prob 1.0), all others prob 0.5.
"""

import json, sys, os

def gen_cycle_metric(n):
    dist = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            elif abs(i - j) == 1 or abs(i - j) == n - 1:
                dist[i][j] = 2
            else:
                dist[i][j] = 3
    prob = [1.0] + [0.5] * (n - 1)
    return {
        "_comment": f"Cycle metric n={n}: adjacent=2, non-adjacent=3, prob=0.5",
        "n": n,
        "dist": dist,
        "prob": prob,
    }

if __name__ == "__main__":
    sizes = list(range(3, 13)) if len(sys.argv) == 1 else [int(x) for x in sys.argv[1:]]
    os.makedirs("examples", exist_ok=True)
    for n in sizes:
        data = gen_cycle_metric(n)
        path = f"examples/cycle_metric_n{n}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Generated {path}")
