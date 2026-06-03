#!/usr/bin/env python3
"""Theoretical analysis of adapt/post ratio on the subset lattice.

OPT_post(R) = width(R, ⊆) = max antichain size (by Dilworth's theorem).
This script:
  1. Computes E[OPT_post] = E[width] by Monte Carlo for w up to ~10
  2. Simulates a greedy adaptive strategy (upper bound on OPT_adapt)
  3. Compares with exact solver results for small w
"""

import random
import sys
from math import comb


def popcount(x):
    return bin(x).count('1')


# ---------- Max matching (bipartite, augmenting paths) ----------

def max_matching_size(adj, n):
    """Max matching in bipartite graph.
    adj[i] = list of j such that element i is a strict subset of element j.
    Both sides have n nodes (same elements, left=source, right=sink).
    """
    match_r = [-1] * n

    def dfs(u, seen):
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                if match_r[v] < 0 or dfs(match_r[v], seen):
                    match_r[v] = u
                    return True
        return False

    result = 0
    for u in range(n):
        if adj[u]:
            seen = [False] * n
            if dfs(u, seen):
                result += 1
    return result


def compute_width(active):
    """Width (max antichain) of inclusion poset on list of bitmasks."""
    k = len(active)
    if k == 0:
        return 0

    # Build adjacency: i -> j if active[i] ⊂ active[j]
    adj = [[] for _ in range(k)]
    for i in range(k):
        for j in range(k):
            if active[i] != active[j] and (active[i] & active[j]) == active[i]:
                adj[i].append(j)

    return k - max_matching_size(adj, k)


# ---------- Greedy adaptive simulation ----------

def greedy_adaptive(active_set_mask, all_customers, w):
    """Simulate greedy adaptive: probe supersets first, smallest level first.

    active_set_mask: set of bitmasks that are active
    all_customers: list of all customer bitmasks (sorted by popcount)
    Returns: cost of this strategy
    """
    pos = 0  # depot = empty set
    cost = 0
    probed = set()

    while len(probed) < len(all_customers):
        # Find next customer to probe:
        #   Priority 1: supersets of pos (free move if active)
        #   Priority 2: smaller popcount first
        best = None
        best_is_sup = False
        best_pc = w + 1

        for c in all_customers:
            if c in probed:
                continue
            is_sup = ((pos & c) == pos)  # pos ⊆ c
            pc = popcount(c)

            # Better if: (a) superset when best isn't, or
            # (b) same superset status but smaller
            if best is None:
                best, best_is_sup, best_pc = c, is_sup, pc
            elif is_sup and not best_is_sup:
                best, best_is_sup, best_pc = c, is_sup, pc
            elif is_sup == best_is_sup and pc < best_pc:
                best, best_is_sup, best_pc = c, is_sup, pc

        if best is None:
            break

        probed.add(best)
        if best in active_set_mask:
            d = 0 if (pos & best) == pos else 1
            cost += d
            pos = best

    # Return to depot
    if pos != 0:
        cost += 1
    return cost


def monte_carlo(w, num_samples=200000):
    n = 1 << w
    all_customers = sorted(range(1, n), key=lambda x: popcount(x))

    # Precompute probabilities
    prob = {}
    for c in all_customers:
        prob[c] = 1.0 / comb(w, popcount(c))

    sum_width = 0.0
    sum_adapt = 0.0
    sum_active = 0.0

    for trial in range(num_samples):
        # Sample active set
        active = []
        for c in all_customers:
            if random.random() < prob[c]:
                active.append(c)

        k = len(active)
        sum_active += k

        if k == 0:
            continue

        # E[OPT_post] = E[width]
        width = compute_width(active)
        sum_width += width

        # Greedy adaptive (upper bound on OPT_adapt)
        active_set = set(active)
        adapt_cost = greedy_adaptive(active_set, all_customers, w)
        sum_adapt += adapt_cost

    return (sum_width / num_samples,
            sum_adapt / num_samples,
            sum_active / num_samples)


def main():
    max_w = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100000

    print(f"Subset lattice: d(A,B)=0 iff A⊆B, prob(S)=1/C(w,|S|)")
    print(f"OPT_post(R) = width(R,⊆) by Dilworth's theorem")
    print(f"Samples per w: {num_samples}")
    print()
    print(f"{'w':>3}  {'n=2^w':>6}  {'E[|R|]':>7}  {'E[post]':>8}  "
          f"{'E[adapt]':>9}  {'ad/po':>7}  {'post/w':>7}")
    print("-" * 65)

    for w in range(2, max_w + 1):
        e_post, e_adapt, e_active = monte_carlo(w, num_samples)
        ratio = e_adapt / e_post if e_post > 1e-15 else float('inf')
        print(f"{w:>3}  {1<<w:>6}  {e_active:>7.3f}  {e_post:>8.4f}  "
              f"{e_adapt:>9.4f}  {ratio:>7.4f}  {e_post/w:>7.4f}")


if __name__ == "__main__":
    main()
