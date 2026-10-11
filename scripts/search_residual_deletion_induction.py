#!/usr/bin/env python3
"""Test a vertex-deletion induction for the residual inequality Q_e <= D.

Both the mean radii and residual weights are recomputed after deleting a
vertex (setting its activation probability to zero).  A minimal-counterexample
induction would follow if every nonzero instance had a client v with

    Q_e(p) - Q_e(p with p_v=0)
        <= D(p) - D(p with p_v=0).

This program searches for a finite obstruction on small graph metrics.  It is
a floating-point falsifier, not a proof.
"""

from __future__ import annotations

import argparse
import math
import random

from search_open_path_gap import random_graph_metric
from search_open_residual_rounding import evaluate, probability_profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    worst = None
    for trial in range(args.trials):
        distance = random_graph_metric(args.n, rng)
        for probabilities in probability_profiles(args.n, rng):
            current = evaluate(distance, probabilities)
            local = []
            for index, probability in enumerate(probabilities):
                if probability == 0.0:
                    continue
                deleted_probabilities = probabilities[:]
                deleted_probabilities[index] = 0.0
                deleted = evaluate(distance, deleted_probabilities)
                delta_d = current["D"] - deleted["D"]
                delta_q = current["fixed_e"] - deleted["fixed_e"]
                if delta_d > 1.0e-12:
                    local.append((delta_q / delta_d, index + 1, delta_q, delta_d))
            best_vertex = min(local, default=(math.inf, -1, math.nan, math.nan))
            record = (
                best_vertex[0],
                trial,
                probabilities,
                current["fixed_over_D"],
                best_vertex,
                distance,
            )
            if worst is None or record[0] > worst[0]:
                worst = record
                print(
                    "worst minimum deletion ratio",
                    record[0],
                    "global Qe/D",
                    record[3],
                    "vertex",
                    best_vertex[1],
                    flush=True,
                )
            if best_vertex[0] > 1.0 + 1.0e-8:
                print("COUNTEREXAMPLE", record)
                return
    print("FINAL", worst)


if __name__ == "__main__":
    main()
