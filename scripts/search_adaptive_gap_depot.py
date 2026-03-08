#!/usr/bin/env python3
"""
Search for rooted stochastic TSP instances where the exact a posteriori optimum is
much smaller than the exact adaptive optimum.

Model
-----
We use the rooted / depot version discussed by the user:
- vertex 0 is a depot (root), and the route starts and ends at 0;
- every other vertex i is active independently with probability p[i];
- an adaptive policy repeatedly chooses an unrevealed vertex to probe;
    * if inactive, the salesman stays where they are;
    * if active, the salesman must move to that vertex immediately as the next move.

For a realized active set A, the a posteriori benchmark is the optimal directed TSP
cycle on {root} union A.

The adaptive value is computed by a *backward DP* over the set of revealed vertices.
State:
    dp[cur][revealed_mask]
where revealed_mask is the set of non-root vertices whose status is already known.
If we next probe v not in revealed_mask, then
    inactive  -> stay at cur and move to revealed_mask U {v}
    active    -> pay dist[cur][v], move to v, and move to revealed_mask U {v}
The base case is when all non-root vertices are revealed, in which case we return to
the depot.

This file also improves the a posteriori computation: all subset-optimal TSP cycle
costs are computed in one Held-Karp-style DP in O(n^2 2^n), instead of solving a
separate TSP for every subset.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

Matrix = List[List[float]]


def popcount(x: int) -> int:
    """Compatibility helper for Python < 3.10."""
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count("1")


@dataclass
class Instance:
    dist: Matrix
    probs: List[float]
    root: int = 0

    def __post_init__(self) -> None:
        n = len(self.dist)
        if n == 0:
            raise ValueError("distance matrix must be non-empty")
        if len(self.probs) != n:
            raise ValueError("len(probs) must equal len(dist)")
        if not (0 <= self.root < n):
            raise ValueError("root out of range")
        for row in self.dist:
            if len(row) != n:
                raise ValueError("distance matrix must be square")
        if abs(self.probs[self.root] - 1.0) > 1e-12:
            raise ValueError("root must have probability 1.0")
        for p in self.probs:
            if not (0.0 <= p <= 1.0):
                raise ValueError("probabilities must lie in [0,1]")
        self._validate_metric()

    @property
    def n(self) -> int:
        return len(self.dist)

    def _validate_metric(self) -> None:
        n = self.n
        eps = 1e-9
        for i in range(n):
            if abs(self.dist[i][i]) > eps:
                raise ValueError(f"dist[{i}][{i}] must be 0")
            for j in range(n):
                if self.dist[i][j] < -eps:
                    raise ValueError("distance matrix must be non-negative")
        for i in range(n):
            for j in range(n):
                dij = self.dist[i][j]
                for k in range(n):
                    if self.dist[i][k] + self.dist[k][j] + eps < dij:
                        raise ValueError("distance matrix violates directed triangle inequality")


# ---------------------------------------------------------------------------
# Metric generators
# ---------------------------------------------------------------------------


def floyd_warshall(mat: Matrix) -> Matrix:
    n = len(mat)
    d = [row[:] for row in mat]
    for k in range(n):
        dk = d[k]
        for i in range(n):
            dik = d[i][k]
            row = d[i]
            for j in range(n):
                cand = dik + dk[j]
                if cand < row[j]:
                    row[j] = cand
    for i in range(n):
        d[i][i] = 0.0
    return d



def random_shortest_path_metric(n: int, rng: random.Random, lo: int = 1, hi: int = 100) -> Matrix:
    """Generate a directed metric as APSP distances of a random complete digraph."""
    inf = float("inf")
    raw = [[inf] * n for _ in range(n)]
    for i in range(n):
        raw[i][i] = 0.0
        for j in range(n):
            if i != j:
                raw[i][j] = float(rng.randint(lo, hi))
    return floyd_warshall(raw)



def random_probability_vector(n: int, rng: random.Random, palette: Sequence[float]) -> List[float]:
    probs = [1.0]
    probs.extend(rng.choice(palette) for _ in range(n - 1))
    return probs


# ---------------------------------------------------------------------------
# Exact a posteriori value: all subset cycle costs in one DP
# ---------------------------------------------------------------------------


def all_subset_tsp_costs(inst: Instance) -> List[float]:
    """
    Return cycle_cost[mask] for every subset mask of non-root vertices 1..n-1,
    where cycle_cost[mask] is the minimum directed Hamiltonian cycle cost on
    {root} union mask.

    Runtime: O(n^2 2^(n-1)).
    """
    n = inst.n
    root = inst.root
    m = n - 1
    full_mask = (1 << m) - 1

    # hk[mask][j] = minimum cost to start at root, visit exactly mask, and end at j,
    # where j is a non-root vertex index 1..n-1 contained in mask.
    hk = [[math.inf] * n for _ in range(1 << m)]

    for j in range(1, n):
        bit = 1 << (j - 1)
        hk[bit][j] = inst.dist[root][j]

    for mask in range(1 << m):
        for j in range(1, n):
            if not ((mask >> (j - 1)) & 1):
                continue
            cur = hk[mask][j]
            if math.isinf(cur):
                continue
            remaining = full_mask ^ mask
            add = remaining
            while add:
                bit = add & -add
                k = bit.bit_length()
                new_mask = mask | bit
                cand = cur + inst.dist[j][k]
                if cand < hk[new_mask][k]:
                    hk[new_mask][k] = cand
                add ^= bit

    cycle_cost = [0.0] * (1 << m)
    for mask in range(1, 1 << m):
        best = math.inf
        cur = mask
        while cur:
            bit = cur & -cur
            j = bit.bit_length()
            cand = hk[mask][j] + inst.dist[j][root]
            if cand < best:
                best = cand
            cur ^= bit
        cycle_cost[mask] = best
    return cycle_cost



def expected_posteriori(inst: Instance, subset_costs: Optional[List[float]] = None) -> float:
    if subset_costs is None:
        subset_costs = all_subset_tsp_costs(inst)

    exp_value = 0.0
    for mask, cost in enumerate(subset_costs):
        prob = 1.0
        for i in range(1, inst.n):
            p = inst.probs[i]
            prob *= p if ((mask >> (i - 1)) & 1) else (1.0 - p)
        exp_value += prob * cost
    return exp_value


# ---------------------------------------------------------------------------
# Exact adaptive value: backward DP over revealed set
# ---------------------------------------------------------------------------


def adaptive_optimal_value(inst: Instance) -> Tuple[float, Dict[Tuple[int, int], int]]:
    """
    Returns (value, argmin_policy).

    Backward state:
        dp[cur][revealed_mask]
    where revealed_mask ranges over non-root vertices 1..n-1 whose statuses are
    already known. The current position cur is either the root, or an active vertex
    already contained in revealed_mask.

    Bellman equation:
        dp[cur][mask] = min_{v not in mask}
            (1-p_v) * dp[cur][mask U {v}] + p_v * (d(cur,v) + dp[v][mask U {v}])

    Base case:
        dp[cur][full_mask] = d(cur, root)
    """
    n = inst.n
    root = inst.root
    m = n - 1
    full_mask = (1 << m) - 1

    dp = [[math.inf] * (1 << m) for _ in range(n)]
    decision: Dict[Tuple[int, int], int] = {}

    for cur in range(n):
        dp[cur][full_mask] = inst.dist[cur][root]

    for size in range(m - 1, -1, -1):
        for mask in range(1 << m):
            if popcount(mask) != size:
                continue
            unrevealed = full_mask ^ mask
            if unrevealed == 0:
                continue

            # Reachable current positions for this mask.
            current_vertices = [root]
            current_vertices.extend(i for i in range(1, n) if (mask >> (i - 1)) & 1)

            for cur in current_vertices:
                best = math.inf
                best_probe = -1
                rem = unrevealed
                while rem:
                    bit = rem & -rem
                    v = bit.bit_length()  # maps bit 1<<(v-1) to vertex v
                    next_mask = mask | bit
                    p = inst.probs[v]
                    cand = (1.0 - p) * dp[cur][next_mask] + p * (inst.dist[cur][v] + dp[v][next_mask])
                    if cand < best:
                        best = cand
                        best_probe = v
                    rem ^= bit
                dp[cur][mask] = best
                decision[(cur, mask)] = best_probe

    return dp[root][0], decision


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def ratio_summary(inst: Instance) -> Dict[str, object]:
    subset_costs = all_subset_tsp_costs(inst)
    post = expected_posteriori(inst, subset_costs)
    adapt, policy = adaptive_optimal_value(inst)
    ratio = math.inf if post == 0.0 else adapt / post
    return {
        "n": inst.n,
        "root": inst.root,
        "posteriori": post,
        "adaptive": adapt,
        "ratio": ratio,
        "dist": inst.dist,
        "probs": inst.probs,
        "policy": {f"({cur},{mask})": nxt for (cur, mask), nxt in sorted(policy.items())},
    }



def search_random_instances(
    n: int,
    trials: int,
    seed: int,
    metric_samples_per_trial: int = 1,
    prob_samples_per_metric: int = 4,
    palette: Sequence[float] = (0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95),
) -> Dict[str, object]:
    rng = random.Random(seed)

    best_summary: Optional[Dict[str, object]] = None
    best_ratio = -1.0

    for _ in range(trials):
        for _ in range(metric_samples_per_trial):
            dist = random_shortest_path_metric(n, rng)
            # Reuse expensive a-posteriori subset DP across probability vectors.
            base_inst = Instance(dist=dist, probs=[1.0] + [0.5] * (n - 1), root=0)
            subset_costs = all_subset_tsp_costs(base_inst)
            for _ in range(prob_samples_per_metric):
                probs = random_probability_vector(n, rng, palette)
                inst = Instance(dist=dist, probs=probs, root=0)
                post = expected_posteriori(inst, subset_costs)
                adapt, policy = adaptive_optimal_value(inst)
                ratio = math.inf if post == 0.0 else adapt / post
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_summary = {
                        "n": inst.n,
                        "root": inst.root,
                        "posteriori": post,
                        "adaptive": adapt,
                        "ratio": ratio,
                        "dist": inst.dist,
                        "probs": inst.probs,
                        "policy": {f"({cur},{mask})": nxt for (cur, mask), nxt in sorted(policy.items())},
                    }

    if best_summary is None:
        raise RuntimeError("search did not produce any instance")
    return best_summary


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def pretty_matrix(mat: Matrix) -> str:
    lines = []
    for row in mat:
        pieces = []
        for x in row:
            if abs(x - round(x)) < 1e-9:
                pieces.append(f"{int(round(x)):4d}")
            else:
                pieces.append(f"{x:8.3f}")
        lines.append("[" + " ".join(pieces) + "]")
    return "\n".join(lines)



def print_summary(summary: Dict[str, object]) -> None:
    print(f"n                = {summary['n']}")
    print(f"root             = {summary['root']}")
    print(f"a posteriori     = {summary['posteriori']:.6f}")
    print(f"adaptive         = {summary['adaptive']:.6f}")
    print(f"adaptive/post    = {summary['ratio']:.6f}")
    print(f"probs            = {summary['probs']}")
    print("distance matrix:")
    print(pretty_matrix(summary["dist"]))



def load_instance(path: str) -> Instance:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Instance(dist=data["dist"], probs=data["probs"], root=data.get("root", 0))



def save_summary(path: str, summary: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact search for adaptive-vs-a-posteriori stochastic TSP gaps")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("evaluate", help="evaluate one JSON instance exactly")
    p_eval.add_argument("instance", help="path to JSON file with keys: dist, probs, optional root")
    p_eval.add_argument("--save", help="optional JSON file to save the full summary")

    p_search = sub.add_parser("search", help="randomly search for a large adaptive/posteriori ratio")
    p_search.add_argument("--n", type=int, default=7, help="number of vertices including root (default: 7)")
    p_search.add_argument("--trials", type=int, default=1000, help="number of random metrics to try")
    p_search.add_argument("--seed", type=int, default=0, help="random seed")
    p_search.add_argument("--prob-samples", type=int, default=4, help="probability vectors tested per metric")
    p_search.add_argument("--save", help="optional JSON file to save the best found instance summary")

    p_demo = sub.add_parser("demo", help="run a small built-in demo search")
    p_demo.add_argument("--seed", type=int, default=0)

    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "evaluate":
        inst = load_instance(args.instance)
        summary = ratio_summary(inst)
        print_summary(summary)
        if args.save:
            save_summary(args.save, summary)
            print(f"saved summary to {args.save}")
        return

    if args.cmd == "search":
        summary = search_random_instances(
            n=args.n,
            trials=args.trials,
            seed=args.seed,
            prob_samples_per_metric=args.prob_samples,
        )
        print_summary(summary)
        if args.save:
            save_summary(args.save, summary)
            print(f"saved summary to {args.save}")
        return

    if args.cmd == "demo":
        summary = search_random_instances(n=7, trials=250, seed=args.seed, prob_samples_per_metric=4)
        print_summary(summary)
        return

    raise AssertionError("unreachable")


if __name__ == "__main__":
    main()
