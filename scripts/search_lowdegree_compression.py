#!/usr/bin/env python3
"""Numerical search for a violation of the compressed low-degree inequality.

For a shortest-path metric and threshold one, this evaluates

    a = E_{B,C} sum_v p_v 1[d(v,f_B(C))>1]
    h = E_B # {v in B : deg_B(v) <= 1},

with a remote depot.  Projection ties are resolved by a supplied vertex
priority.  The conjectured inequality is a <= h.  Summing out C by Voronoi
cells makes one evaluation require only 2^n guide sets.
"""

from __future__ import annotations

import argparse
import math
import random


def all_pairs_shortest(adjacency: list[list[bool]]) -> list[list[int]]:
    n = len(adjacency)
    inf = n + 1
    d = [[0 if i == j else (1 if adjacency[i][j] else inf) for j in range(n)] for i in range(n)]
    for k in range(n):
        for i in range(n):
            dik = d[i][k]
            for j in range(n):
                if dik + d[k][j] < d[i][j]:
                    d[i][j] = dik + d[k][j]
    return d


def evaluate(
    adjacency: list[list[bool]],
    p: list[float],
    priority: list[int],
    root_neighbors: list[bool] | None = None,
    client_only_projection: bool = False,
) -> tuple[float, float, float]:
    n = len(p)
    q = [1.0 - x for x in p]
    if root_neighbors is None:
        root_neighbors = [False] * n
    extended = [row + [root_neighbors[i]] for i, row in enumerate(adjacency)]
    extended.append(root_neighbors[:] + [False])
    full_distance = all_pairs_shortest(extended)
    d = [row[:n] for row in full_distance[:n]]
    root_distance = full_distance[n]
    rank = [0] * n
    for i, v in enumerate(priority):
        rank[v] = i

    # The right side has a closed form.
    low_degree = 0.0
    for v in range(n):
        if root_distance[v] == 1:
            continue
        neighbors = [u for u in range(n) if u != v and adjacency[v][u]]
        none = math.prod(q[u] for u in neighbors)
        one = sum(p[u] * math.prod(q[w] for w in neighbors if w != u) for u in neighbors)
        low_degree += p[v] * (none + one)

    # Corrected tour-incidence certificate: degree zero clients, plus clients
    # of full threshold degree one.  A root-short singleton is excluded because
    # its unique depot edge can be repeated by the two-vertex tour.
    cap = 0.0
    mask_probabilities = []
    for mask in range(1 << n):
        probability = math.prod(p[v] if mask >> v & 1 else q[v] for v in range(n))
        mask_probabilities.append(probability)
        active_count = mask.bit_count()
        value = 0
        for v in range(n):
            if not (mask >> v & 1):
                continue
            degree = int(root_distance[v] == 1) + sum(
                bool(mask >> u & 1) and adjacency[v][u] for u in range(n) if u != v
            )
            if degree == 0:
                value += 1
            elif degree == 1 and not (active_count == 1 and root_distance[v] == 1):
                value += 1
        cap += probability * value

    attachment = 0.0
    for mask in range(1 << n):
        probability = mask_probabilities[mask]
        centers = []
        for v in range(n):
            if mask >> v & 1:
                centers.append(v)
        if probability == 0.0:
            continue

        # In client-only mode the depot is used only when there is no guide.
        cell_count = len(centers) if client_only_projection and centers else len(centers) + 1
        cells = [[] for _ in range(cell_count)]
        center_position = {v: i for i, v in enumerate(centers)}
        for x in range(n):
            if x in center_position:
                image_position = center_position[x]
            else:
                candidates = list(range(cell_count))
                image_position = min(
                    candidates,
                    key=lambda j: (
                        d[x][centers[j]] if j < len(centers) else root_distance[x],
                        rank[centers[j]] if j < len(centers) else n,
                    ),
                )
            cells[image_position].append(x)
        guide_cells = cells if client_only_projection and centers else cells[:-1]
        empty = [math.prod(q[x] for x in cell) for cell in guide_cells]
        conditional = 0.0
        for v in range(n):
            if root_distance[v] == 1:
                continue
            conditional += p[v] * math.prod(
                empty[j]
                for j, b in enumerate(centers)
                if b == v or adjacency[v][b]
            )
        attachment += probability * conditional
    return attachment, low_degree, cap


def random_graph(n: int, density: float, rng: random.Random) -> list[list[bool]]:
    a = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i):
            a[i][j] = a[j][i] = rng.random() < density
    return a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=9)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rooted", action="store_true")
    parser.add_argument("--root-probability", type=float, default=0.1)
    parser.add_argument("--client-only-projection", action="store_true")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    grid = (0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9, 0.99)
    best = (-1e100, None)
    for trial in range(args.trials):
        density = rng.uniform(0.05, 0.95)
        graph = random_graph(args.clients, density, rng)
        root_neighbors = (
            [rng.random() < args.root_probability for _ in range(args.clients)]
            if args.rooted
            else [False] * args.clients
        )
        if args.rooted and not any(root_neighbors):
            root_neighbors[rng.randrange(args.clients)] = True
        probabilities = [rng.choice(grid) for _ in range(args.clients)]
        priority = list(range(args.clients))
        rng.shuffle(priority)
        attachment, low_degree, cap = evaluate(
            graph,
            probabilities,
            priority,
            root_neighbors,
            client_only_projection=args.client_only_projection,
        )
        gap = attachment - low_degree
        cap_gap = attachment - cap
        if gap > best[0]:
            best = (gap, (attachment, low_degree, probabilities, priority, graph))
        if cap_gap > 1e-10:
            print("CAP VIOLATION", trial, cap_gap, attachment, cap, "old gap", gap)
            print("p", probabilities)
            print("priority", priority)
            print("edges", [(i, j) for i in range(args.clients) for j in range(i) if graph[i][j]])
            print("root_neighbors", [i for i, x in enumerate(root_neighbors) if x])
            return
        if trial and trial % 100 == 0:
            print("progress", trial, "best gap", best[0])
    print("no violation; best", best)


if __name__ == "__main__":
    main()
