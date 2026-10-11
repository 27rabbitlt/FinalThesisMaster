#!/usr/bin/env python3
"""Exact audit for the hazard-tree residual attachment identities.

This is a proof checker/falsification aid, not a proof of the universal
29/20 hazard gateway.  All identity checks use Fraction arithmetic.
"""

from __future__ import annotations

import random
from fractions import Fraction as F


def metric_closure(w: list[list[F]]) -> list[list[F]]:
    d = [row[:] for row in w]
    for k in range(len(d)):
        for i in range(len(d)):
            for j in range(len(d)):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    return d


def graph_metric(n: int, edges: list[tuple[int, int, int]]) -> list[list[F]]:
    inf = F(10**9)
    w = [[F(0) if i == j else inf for j in range(n)] for i in range(n)]
    for a, b, length in edges:
        w[a][b] = w[b][a] = min(w[a][b], F(length))
    return metric_closure(w)


def occupancy(mask: int, p: list[F]) -> F:
    inactive = F(1)
    for v, pv in enumerate(p):
        if mask & (1 << v):
            inactive *= 1 - pv
    return 1 - inactive


def inactivity(mask: int, p: list[F]) -> F:
    return 1 - occupancy(mask, p)


def orient(
    edges: list[tuple[int, int]], root: int, vertices: set[int]
) -> list[int]:
    adj = {v: [] for v in vertices}
    for a, b in edges:
        if a in vertices and b in vertices:
            adj[a].append(b)
            adj[b].append(a)
    parent = [-2] * (max(vertices) + 1)
    parent[root] = -1
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if parent[b] == -2:
                parent[b] = a
                stack.append(b)
    assert all(parent[v] != -2 for v in vertices)
    return parent


def subtree_masks(parent: list[int], clients: int, vertices: set[int]) -> list[int]:
    children = {v: [] for v in vertices}
    root = next(v for v in vertices if parent[v] == -1)
    for v in vertices:
        if parent[v] >= 0:
            children[parent[v]].append(v)
    sub = [0] * len(parent)

    def visit(v: int) -> int:
        ans = (1 << v) if v < clients else 0
        for z in children[v]:
            ans |= visit(z)
        sub[v] = ans
        return ans

    visit(root)
    return sub


def tree_hazard(
    edges: list[tuple[int, int]], root: int, d: list[list[F]], p: list[F]
) -> F:
    vertices = set(range(len(p) + 1))
    parent = orient(edges, root, vertices)
    sub = subtree_masks(parent, len(p), vertices)
    return sum(
        d[v][parent[v]] * occupancy(sub[v], p)
        for v in vertices
        if parent[v] >= 0
    )


def induced_hazard(
    edges: list[tuple[int, int]], portal: int, vertices: set[int],
    d: list[list[F]], p: list[F]
) -> F:
    if len(vertices) == 1:
        return F(0)
    parent = orient(edges, portal, vertices)
    sub = subtree_masks(parent, len(p), vertices)
    return sum(
        d[v][parent[v]] * occupancy(sub[v], p)
        for v in vertices
        if parent[v] >= 0
    )


def residual_attachment_check(
    name: str, d: list[list[F]], p: list[F], edges: list[tuple[int, int]],
    samples: int = 80
) -> None:
    n = len(p)
    root = n
    all_vertices = set(range(n + 1))
    parent = orient(edges, root, all_vertices)
    sub = subtree_masks(parent, n, all_vertices)
    total = tree_hazard(edges, root, d, p)
    rng = random.Random(1919 + n)
    tested = 0

    for _ in range(samples):
        u = rng.randrange(n)
        u_mask = sub[u]
        inside = {v for v in range(n) if u_mask & (1 << v)}
        outside = all_vertices - inside
        a = parent[u]
        outside_edges = [(x, y) for x, y in edges if x in outside and y in outside]
        inside_edges = [(x, y) for x, y in edges if x in inside and y in inside]
        out_parent = orient(outside_edges, root, outside)
        out_sub = subtree_masks(out_parent, n, outside)
        phi_out = induced_hazard(outside_edges, root, outside, d, p)

        b = rng.choice(sorted(outside))
        v = rng.choice(sorted(inside))
        rho = F(0)
        z = b
        while z != root:
            rho += d[z][out_parent[z]] * inactivity(out_sub[z], p)
            z = out_parent[z]

        h_inside = induced_hazard(inside_edges, v, inside, d, p)
        rebuilt = outside_edges + inside_edges + [(b, v)]
        lhs = tree_hazard(rebuilt, root, d, p)
        rhs = phi_out + h_inside + occupancy(u_mask, p) * (d[b][v] + rho)
        assert lhs == rhs, (name, "residual identity", lhs, rhs, u, a, b, v)

        # Independently check the expanded swap formula (E13), expressed in
        # the original rooted tree rather than through the detached pieces.
        expanded = occupancy(u_mask, p) * (d[b][v] - d[a][u])

        internal_path = []
        z = v
        while z != u:
            internal_path.append(z)
            z = parent[z]
        for z in reversed(internal_path):
            expanded += d[parent[z]][z] * (
                occupancy(u_mask ^ sub[z], p) - occupancy(sub[z], p)
            )

        ancestors_a = set()
        z = a
        while z != -1:
            ancestors_a.add(z)
            z = parent[z]
        b_to_lca = []
        z = b
        while z not in ancestors_a:
            b_to_lca.append(z)
            z = parent[z]
        lca = z
        for z in b_to_lca:
            expanded += d[parent[z]][z] * (
                occupancy(sub[z] | u_mask, p) - occupancy(sub[z], p)
            )
        z = a
        while z != lca:
            expanded += d[parent[z]][z] * (
                occupancy(sub[z] & ~u_mask, p) - occupancy(sub[z], p)
            )
            z = parent[z]
        assert lhs - total == expanded, (
            name, "expanded identity", lhs - total, expanded, u, a, b, v
        )
        tested += 1

        # The original attachment is one of the same decompositions.
        if b == a and v == u:
            assert lhs == total

    print(f"{name:22s} residual identities: {tested:3d} exact checks")


def path_tree(n: int) -> list[tuple[int, int]]:
    root = n
    return [(root, n - 1)] + [(i + 1, i) for i in range(n - 1)]


def stress_instances() -> list[tuple[str, list[list[F]], list[F], list[tuple[int, int]]]]:
    ans = []

    # Line, rooted at one end.
    n = 12
    d = [[F(abs(i - j)) for j in range(n + 1)] for i in range(n + 1)]
    p = [F((3 * i) % 11 + 1, 13) for i in range(n)]
    ans.append(("line", d, p, path_tree(n)))

    # Star metric and star tree.
    n = 14
    root = n
    d = [[F(0) if i == j else F(1 if root in (i, j) else 2)
          for j in range(n + 1)] for i in range(n + 1)]
    p = [F((5 * i) % 13 + 1, 15) for i in range(n)]
    ans.append(("star", d, p, [(root, i) for i in range(n)]))

    # The N=50, p=2/25, L=10 path/hub witness.  Use its inexpensive
    # one-center spider, not the known-bad full MST.
    n = 50
    root = n
    d = [[F(0) for _ in range(n + 1)] for _ in range(n + 1)]
    for i in range(n):
        d[i][root] = d[root][i] = F(10)
        for j in range(i):
            d[i][j] = d[j][i] = F(1 if i - j == 1 else 2)
    p = [F(2, 25)] * n
    hub = n // 2
    ans.append(("H1 path/hub spider", d, p,
                [(root, hub)] + [(hub, i) for i in range(n) if i != hub]))

    # Fourth power of a cycle, with a simple cyclic path tree.
    total = 32
    n = total - 1
    root = n
    k = 4
    d = [[F((min((i-j) % total, (j-i) % total) + k - 1) // k)
          for j in range(total)] for i in range(total)]
    p = [F(3, 5)] * n
    ans.append(("cycle power C32^4", d, p, path_tree(n)))

    # Complete bipartite L/R metric with depot edges one.  An alternating
    # path retains the relevant repeated-run structure.
    m = 18
    n = 2 * m
    root = n
    d = [[F(0) for _ in range(n + 1)] for _ in range(n + 1)]
    for i in range(n):
        d[i][root] = d[root][i] = F(1)
        for j in range(i):
            cross = (i < m) != (j < m)
            d[i][j] = d[j][i] = F(1 if cross else 2)
    order = [x for i in range(m) for x in (i, m + i)]
    edges = [(root, order[0])] + list(zip(order, order[1:]))
    p = [F(1, 17)] * n
    ans.append(("bipartite L/R", d, p, edges))

    # A fixed random {1,2} metric (every such complete matrix is metric).
    rng = random.Random(2900)
    n = 30
    root = n
    d = [[F(0) if i == j else F(rng.choice((1, 2)))
          for j in range(n + 1)] for i in range(n + 1)]
    for i in range(n + 1):
        for j in range(i):
            d[i][j] = d[j][i] = min(d[i][j], d[j][i])
    p = [F(1, 31)] * n
    parents = [(root, 0)] + [(rng.randrange(i + 1, n + 1), i) for i in range(1, n)]
    # The preceding random-parent expression can select the child itself only
    # at i; replace that case by the root and retain acyclicity (parent > i).
    edges = []
    for a, b in parents:
        edges.append((root if a == b else a, b))
    ans.append(("random {1,2}", d, p, edges))

    return ans


def prim_mst(vertices: list[int], d: list[list[F]]) -> tuple[F, list[tuple[int, int]]]:
    used = {vertices[0]}
    edges = []
    cost = F(0)
    while len(used) < len(vertices):
        length, a, b = min(
            (d[a][b], a, b) for a in used for b in vertices if b not in used
        )
        used.add(b)
        edges.append((a, b))
        cost += length
    return cost, edges


def anchor_subdivision_audit() -> None:
    # Eight anchors including the depot form a unit good cycle.  Consecutive
    # anchors of a high-stretch permutation are also joined by six-edge paths
    # whose five internal vertices are rare clients.
    anchors = 8
    internal = 5
    deterministic = anchors - 1
    rare = anchors * internal
    clients = deterministic + rare
    root = clients
    inf = F(10**9)
    w = [[F(0) if i == j else inf for j in range(clients + 1)]
         for i in range(clients + 1)]
    anchor_vertex = list(range(deterministic)) + [root]

    def add(a: int, b: int) -> None:
        w[a][b] = w[b][a] = min(w[a][b], F(1))

    for i in range(anchors):
        add(anchor_vertex[i], anchor_vertex[(i + 1) % anchors])
    permutation = [7, 3, 6, 2, 5, 1, 4, 0]
    nxt = deterministic
    for i in range(anchors):
        a = anchor_vertex[permutation[i]]
        b = anchor_vertex[permutation[(i + 1) % anchors]]
        path = [a] + list(range(nxt, nxt + internal)) + [b]
        nxt += internal
        for x, y in zip(path, path[1:]):
            add(x, y)
    d = metric_closure(w)
    p = [F(1)] * deterministic + [F(1, 1000)] * rare
    core = anchor_vertex
    mst_cost, edges = prim_mst(core, d)
    radius = F(0)
    for v in range(deterministic, clients):
        length, a = min((d[v][a], a) for a in core)
        radius += p[v] * length
        edges.append((a, v))
    phi = tree_hazard(edges, root, d, p)
    assert phi == mst_cost + radius
    anchor_tsp = F(anchors)  # the displayed unit good cycle
    assert mst_cost <= anchor_tsp
    assert radius <= F(9, 20) * anchor_tsp
    assert phi <= F(29, 20) * anchor_tsp
    print("subdivided anchors     exact E2; Phi <= (29/20) TSP(anchors)")
    print(f"  MST={mst_cost}, fringe={radius}, Phi={phi}, anchor tour={anchor_tsp}")


def main() -> None:
    for instance in stress_instances():
        residual_attachment_check(*instance)
    anchor_subdivision_audit()
    print("all exact residual-attachment and anchor checks passed")


if __name__ == "__main__":
    main()
