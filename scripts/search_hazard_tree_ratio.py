#!/usr/bin/env python3
"""Exact finite-instance experiments for the rooted hazard-tree conjecture.

For a symmetric metric, independent client probabilities ``p``, and an
always-active depot, this script computes

    min_T Phi_p(T) / E[tsp(A)],

where ``Phi_p(T)`` is the expected weight of the minimal subtree of a fixed
rooted spanning tree ``T`` that connects the depot to the active clients.

Both optimizations are exact for the supplied finite instance (up to ordinary
floating-point arithmetic): posterior tour costs use Held--Karp, and the best
hazard tree uses a subset/partition dynamic program rather than a heuristic.
The latter recurrence is

    f(r,S) = min_{partition (U_i) of S} sum_i min_{u in U_i}
             [q(U_i) d(r,u) + f(u,U_i - {u})],

where q(U) is the probability that U contains an active client.

The executable search is deliberately aimed at structurally varied metrics;
it is not a tuning tool for the predecessor factor-three proof.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Evaluation:
    posterior: float
    hazard_tree: float
    ratio: float
    parents: tuple[int, ...]
    connected_facility: float
    facility_ratio: float
    facilities: int


def metric_closure(weights: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return the Floyd--Warshall metric closure of a symmetric weight matrix."""
    d = [list(row) for row in weights]
    n = len(d)
    for k in range(n):
        for i in range(n):
            dik = d[i][k]
            for j in range(n):
                via = dik + d[k][j]
                if via < d[i][j]:
                    d[i][j] = via
    return d


def subset_tsp_costs(d: Sequence[Sequence[float]]) -> list[float]:
    """Optimal depot-rooted cycle cost for every active-client subset."""
    n = len(d) - 1
    root = n
    size = 1 << n
    inf = float("inf")
    path = [[inf] * n for _ in range(size)]
    tours = [0.0] * size
    for j in range(n):
        path[1 << j][j] = d[root][j]
    for mask in range(1, size):
        bits = mask
        while bits:
            bit = bits & -bits
            j = bit.bit_length() - 1
            prev = mask ^ bit
            if prev:
                best = inf
                prev_bits = prev
                while prev_bits:
                    kbit = prev_bits & -prev_bits
                    k = kbit.bit_length() - 1
                    best = min(best, path[prev][k] + d[k][j])
                    prev_bits ^= kbit
                path[mask][j] = best
            bits ^= bit
        tours[mask] = min(
            path[mask][j] + d[j][root]
            for j in range(n)
            if mask & (1 << j)
        )
    return tours


def expected_posterior(
    p: Sequence[float], subset_costs: Sequence[float]
) -> float:
    n = len(p)
    ans = 0.0
    for mask, cost in enumerate(subset_costs):
        probability = 1.0
        for v, pv in enumerate(p):
            probability *= pv if mask & (1 << v) else 1.0 - pv
        ans += probability * cost
    return ans


def best_hazard_tree(
    d: Sequence[Sequence[float]], p: Sequence[float]
) -> tuple[float, tuple[int, ...]]:
    """Return min Phi and one optimal parent vector (depot has parent -1)."""
    n = len(p)
    root = n
    size = 1 << n
    inactive = [1.0] * size
    q = [0.0] * size
    for mask in range(1, size):
        bit = mask & -mask
        v = bit.bit_length() - 1
        inactive[mask] = inactive[mask ^ bit] * (1.0 - p[v])
        q[mask] = 1.0 - inactive[mask]

    # f[r][S] is valid only if r is the depot or r is not in S.
    f = [[float("inf")] * size for _ in range(n + 1)]
    choice: list[list[tuple[int, int] | None]] = [
        [None] * size for _ in range(n + 1)
    ]
    for r in range(n + 1):
        f[r][0] = 0.0

    masks_by_size = sorted(range(1, size), key=int.bit_count)
    for mask in masks_by_size:
        anchor = mask & -mask
        for r in range(n + 1):
            if r < n and mask & (1 << r):
                continue
            # Put the lowest-index member of mask into the next root branch;
            # this enumerates every set partition once.
            remainder = mask ^ anchor
            sub = remainder
            best = float("inf")
            best_choice = None
            while True:
                branch = sub | anchor
                rest = mask ^ branch
                branch_bits = branch
                while branch_bits:
                    ubit = branch_bits & -branch_bits
                    u = ubit.bit_length() - 1
                    value = (
                        q[branch] * d[r][u]
                        + f[u][branch ^ ubit]
                        + f[r][rest]
                    )
                    if value < best - 1e-14:
                        best = value
                        best_choice = (branch, u)
                    branch_bits ^= ubit
                if sub == 0:
                    break
                sub = (sub - 1) & remainder
            f[r][mask] = best
            choice[r][mask] = best_choice

    parents = [-2] * (n + 1)
    parents[root] = -1

    def reconstruct(r: int, mask: int) -> None:
        if not mask:
            return
        branch, u = choice[r][mask]  # type: ignore[misc]
        parents[u] = r
        reconstruct(u, branch ^ (1 << u))
        reconstruct(r, mask ^ branch)

    reconstruct(root, size - 1)
    assert all(x != -2 for x in parents)
    return f[root][size - 1], tuple(parents)


def best_connected_facility(
    d: Sequence[Sequence[float]], p: Sequence[float]
) -> tuple[float, int]:
    """Compute min_S MST({depot}+S) + sum_v p_v d(v,{depot}+S)."""
    n = len(p)
    root = n
    best = float("inf")
    best_mask = 0
    for mask in range(1 << n):
        selected = [root] + [v for v in range(n) if mask & (1 << v)]

        # Dense Prim; tiny exact-search instances make this simpler and less
        # error-prone than maintaining all subset MSTs incrementally.
        used = {root}
        mst = 0.0
        while len(used) < len(selected):
            length, vertex = min(
                (d[u][v], v)
                for u in used
                for v in selected
                if v not in used
            )
            mst += length
            used.add(vertex)
        value = mst + sum(
            p[v] * min(d[v][u] for u in selected) for v in range(n)
        )
        if value < best - 1e-14:
            best = value
            best_mask = mask
    return best, best_mask


def evaluate(
    d: Sequence[Sequence[float]],
    p: Sequence[float],
    subset_costs: Sequence[float] | None = None,
    compute_facility: bool = True,
) -> Evaluation:
    if subset_costs is None:
        subset_costs = subset_tsp_costs(d)
    posterior = expected_posterior(p, subset_costs)
    hazard, parents = best_hazard_tree(d, p)
    if compute_facility:
        facility, facilities = best_connected_facility(d, p)
    else:
        facility, facilities = float("nan"), 0
    ratio = hazard / posterior if posterior else 0.0
    facility_ratio = facility / posterior if posterior else 0.0
    return Evaluation(
        posterior,
        hazard,
        ratio,
        parents,
        facility,
        facility_ratio,
        facilities,
    )


def named_metric(name: str, n: int, rng: random.Random) -> list[list[float]]:
    """Generate a metric on n clients followed by one depot."""
    total = n + 1
    inf = 1e100
    if name == "unit":
        return [[0.0 if i == j else 1.0 for j in range(total)] for i in range(total)]

    w = [[0.0 if i == j else inf for j in range(total)] for i in range(total)]

    def edge(a: int, b: int, length: float) -> None:
        w[a][b] = min(w[a][b], length)
        w[b][a] = min(w[b][a], length)

    if name == "path":
        # Depot at one end.
        for i in range(total - 1):
            edge(i, i + 1, 1.0)
    elif name == "cycle":
        for i in range(total):
            edge(i, (i + 1) % total, 1.0)
    elif name == "star":
        for i in range(n):
            edge(i, n, 1.0)
    elif name == "binary":
        for i in range(n):
            parent = n if i == 0 else (i - 1) // 2
            edge(i, parent, 1.0)
    elif name == "random-tree":
        for i in range(n):
            parent = rng.randrange(i + 1, total)
            if parent == i:
                parent = total - 1
            edge(i, parent, rng.uniform(0.2, 2.0))
    elif name in {"graph12", "graph123"}:
        levels = 2 if name == "graph12" else 3
        for i in range(total):
            for j in range(i):
                edge(i, j, float(rng.randint(1, levels)))
    elif name == "client-graph-19":
        # A {1,2}-metric on the clients with every depot edge equal to 1.9.
        # Sparse pseudorandom instances of this form are the natural finite
        # stress test for the hazard tree: a realized client graph can have a
        # cheap tour although no fixed shallow tree covers all rare clients.
        density = rng.uniform(0.12, 0.88)
        for i in range(n):
            edge(i, n, 1.9)
            for j in range(i):
                edge(i, j, 1.0 if rng.random() < density else 2.0)
    elif name == "euclidean":
        points = [(rng.random(), rng.random()) for _ in range(total)]
        return [
            [math.dist(points[i], points[j]) for j in range(total)]
            for i in range(total)
        ]
    elif name == "multiscale-graph":
        # Connected sparse graph with several separated length scales, then
        # shortest-path closure.  Begin with a random weighted tree and add
        # shortcuts, so every generated instance is finite and connected.
        for i in range(total - 1):
            parent = rng.randrange(i + 1, total)
            edge(i, parent, 10.0 ** rng.uniform(-2.0, 2.0))
        shortcut_probability = rng.uniform(0.08, 0.45)
        for i in range(total):
            for j in range(i):
                if w[i][j] >= inf / 2 and rng.random() < shortcut_probability:
                    edge(i, j, 10.0 ** rng.uniform(-2.0, 2.0))
    elif name == "multiscale-complete":
        # Closure of a complete graph with independently sampled raw lengths.
        # Unlike multiscale-graph this normally has many competing shortest
        # paths and is not close to a single underlying tree metric.
        for i in range(total):
            for j in range(i):
                edge(i, j, 10.0 ** rng.uniform(-2.5, 2.5))
    elif name == "incidence":
        # Root--hub--terminal graph metric.  Use --zero-prefix=floor(n/3)
        # so the hubs are zero-probability Steiner vertices and only the
        # terminals activate.  Random incidence systems stress whether a
        # posterior solution can choose a sharing hub that no fixed tree can.
        hubs = max(1, n // 3)
        for hub in range(hubs):
            edge(hub, n, rng.choice((0.5, 1.0, 2.0)))
        for terminal in range(hubs, n):
            incident = [hub for hub in range(hubs) if rng.random() < 0.5]
            if not incident:
                incident = [rng.randrange(hubs)]
            for hub in incident:
                edge(terminal, hub, rng.choice((0.5, 1.0, 2.0)))
    elif name == "line-graph":
        # Clients are the edges of K_m.  Two clients are close precisely when
        # the corresponding edges share an endpoint, giving heavily
        # overlapping (nonlaminar) cheap clusters.
        m = 2
        while m * (m - 1) // 2 < n:
            m += 1
        if m * (m - 1) // 2 != n:
            raise ValueError("line-graph requires n=binom(m,2)")
        labels = list(itertools.combinations(range(m), 2))
        for i in range(n):
            edge(i, n, 1.9)
            for j in range(i):
                edge(i, j, 1.0 if set(labels[i]) & set(labels[j]) else 2.0)
    elif name == "theta":
        # Three internally disjoint root--b paths.  This is the smallest
        # obstruction to cactus blockwise accounting: its cycles overlap in
        # an entire path / in both articulation endpoints.
        if (n - 1) % 3:
            raise ValueError("theta requires n=1 mod 3")
        internal_per_path = (n - 1) // 3
        b = 0
        next_vertex = 1
        for _ in range(3):
            path = [n]
            path.extend(range(next_vertex, next_vertex + internal_per_path))
            next_vertex += internal_per_path
            path.append(b)
            scale = 10.0 ** rng.uniform(-1.0, 1.0)
            for u, v in zip(path, path[1:]):
                edge(u, v, scale * 10.0 ** rng.uniform(-0.4, 0.4))
    else:
        raise ValueError(f"unknown metric family {name!r}")
    return metric_closure(w)


def probability_vectors(
    n: int, rng: random.Random, trials: int
) -> Iterable[tuple[float, ...]]:
    grids = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 0.95]
    for x in grids:
        yield (x,) * n
    for _ in range(trials):
        mode = rng.randrange(4)
        if mode == 0:
            yield tuple(rng.random() for _ in range(n))
        elif mode == 1:
            yield tuple(10.0 ** rng.uniform(-3.0, -0.01) for _ in range(n))
        elif mode == 2:
            a, b = rng.choice(grids), rng.choice(grids)
            yield tuple(a if rng.random() < 0.5 else b for _ in range(n))
        else:
            # Logit-uniform emphasizes both endpoints.
            yield tuple(
                1.0 / (1.0 + math.exp(-rng.uniform(-6.0, 6.0)))
                for _ in range(n)
            )


def search(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    best: tuple[Evaluation, str, list[list[float]], tuple[float, ...]] | None = None
    best_facility: tuple[Evaluation, str, list[list[float]], tuple[float, ...]] | None = None
    families = args.families.split(",")
    for metric_index in range(args.metrics):
        family = families[metric_index % len(families)]
        d = named_metric(family, args.clients, rng)
        costs = subset_tsp_costs(d)
        if args.uniform_values:
            candidate_probabilities: Iterable[tuple[float, ...]] = (
                (float(value),) * args.clients
                for value in args.uniform_values.split(",")
            )
        else:
            candidate_probabilities = probability_vectors(
                args.clients, rng, args.probabilities
            )
        for p in candidate_probabilities:
            if args.zero_prefix:
                p = (0.0,) * args.zero_prefix + tuple(p[args.zero_prefix :])
            result = evaluate(d, p, costs, compute_facility=not args.skip_facility)
            if not args.skip_facility and (
                best_facility is None
                or result.facility_ratio > best_facility[0].facility_ratio
            ):
                best_facility = (result, family, d, p)
                print(
                    f"best G/P={result.facility_ratio:.12f} family={family} "
                    f"P={result.posterior:.12f} G={result.connected_facility:.12f}",
                    flush=True,
                )
            if best is None or result.ratio > best[0].ratio:
                best = (result, family, d, p)
                print(
                    f"best ratio={result.ratio:.12f} family={family} "
                    f"P={result.posterior:.12f} Phi={result.hazard_tree:.12f} "
                    f"G/P={result.facility_ratio:.12f}"
                )
                print("p =", " ".join(f"{x:.12g}" for x in p))
                print("parents =", result.parents)
                print("d =")
                for row in d:
                    print(" ".join(f"{x:.12g}" for x in row))


def evolutionary_search(args: argparse.Namespace) -> None:
    """Jointly search all [1,2]-metrics and heterogeneous probabilities."""
    from scipy.optimize import differential_evolution

    n = args.clients
    total = n + 1
    pairs = [(i, j) for i in range(total) for j in range(i)]
    calls = 0
    incumbent: tuple[float, list[list[float]], tuple[float, ...], Evaluation] | None = None

    def decode(x: Sequence[float]) -> tuple[list[list[float]], tuple[float, ...]]:
        d = [[0.0] * total for _ in range(total)]
        for value, (i, j) in zip(x, pairs):
            edge_length = math.exp(value) if args.evolve_closure else value
            d[i][j] = d[j][i] = edge_length
        if args.evolve_closure:
            d = metric_closure(d)
        logits = x[len(pairs) :]
        p = tuple(1.0 / (1.0 + math.exp(-z)) for z in logits)
        return d, p

    def objective(x: Sequence[float]) -> float:
        nonlocal calls, incumbent
        calls += 1
        d, p = decode(x)
        result = evaluate(d, p)
        score = result.ratio if args.target == "hazard" else result.facility_ratio
        if incumbent is None or score > incumbent[0]:
            incumbent = (score, d, p, result)
            print(
                f"call={calls} best {args.target} ratio={score:.12f} "
                f"P={result.posterior:.12f} Phi={result.hazard_tree:.12f}",
                flush=True,
            )
        return -score

    metric_bounds = [(-4.0, 4.0)] if args.evolve_closure else [(1.0, 2.0)]
    bounds = metric_bounds * len(pairs) + [(-7.0, 7.0)] * n
    result = differential_evolution(
        objective,
        bounds,
        seed=args.seed,
        maxiter=args.generations,
        popsize=args.population,
        polish=False,
        updating="immediate",
        workers=1,
        tol=0.0,
        atol=0.0,
    )
    assert incumbent is not None
    _, d, p, evaluation = incumbent
    print("optimizer:", result.message)
    print("ratio =", format(evaluation.ratio, ".17g"))
    print("posterior =", format(evaluation.posterior, ".17g"))
    print("hazard_tree =", format(evaluation.hazard_tree, ".17g"))
    print("connected_facility =", format(evaluation.connected_facility, ".17g"))
    print("facility_ratio =", format(evaluation.facility_ratio, ".17g"))
    print("facilities mask =", evaluation.facilities)
    print("p =", " ".join(format(x, ".17g") for x in p))
    print("parents =", evaluation.parents)
    print("d =")
    for row in d:
        print(" ".join(format(x, ".17g") for x in row))


def exhaustive_client_graphs(args: argparse.Namespace) -> None:
    """Enumerate every client {1,2}-graph for small n and uniform p."""
    n = args.clients
    if n > 6:
        raise ValueError("exhaustive client-graph mode is limited to at most 6 clients")
    root = n
    pairs = [(i, j) for i in range(n) for j in range(i)]
    if args.heterogeneous_values:
        values = tuple(float(x) for x in args.heterogeneous_values.split(","))
        probability_vectors_to_check: Iterable[tuple[float, ...]] = itertools.product(
            values, repeat=n
        )
        # Materialize because the same collection is traversed for every graph.
        probability_vectors_to_check = tuple(probability_vectors_to_check)
    else:
        probabilities = tuple(
            i / args.grid_denominator for i in range(1, args.grid_denominator)
        )
        probability_vectors_to_check = tuple((pv,) * n for pv in probabilities)
    best: tuple[Evaluation, int, tuple[float, ...], list[list[float]]] | None = None
    best_facility: tuple[
        Evaluation, int, tuple[float, ...], list[list[float]]
    ] | None = None
    for graph_mask in range(1 << len(pairs)):
        d = [[0.0 if i == j else 2.0 for j in range(n + 1)] for i in range(n + 1)]
        for i in range(n):
            d[i][root] = d[root][i] = args.depot_distance
        for bit_index, (i, j) in enumerate(pairs):
            if graph_mask & (1 << bit_index):
                d[i][j] = d[j][i] = 1.0
        costs = subset_tsp_costs(d)
        for p in probability_vectors_to_check:
            result = evaluate(d, p, costs)
            if best_facility is None or result.facility_ratio > best_facility[0].facility_ratio:
                best_facility = (result, graph_mask, tuple(p), d)
            if best is None or result.ratio > best[0].ratio:
                best = (result, graph_mask, tuple(p), d)
    assert best is not None and best_facility is not None
    result, graph_mask, p, d = best
    print(f"graphs checked = {1 << len(pairs)}")
    print(f"probability vectors per graph = {len(tuple(probability_vectors_to_check))}")
    print(f"best graph mask = {graph_mask}")
    print("p =", " ".join(format(x, ".17g") for x in p))
    print(f"ratio = {result.ratio:.17g}")
    print(f"posterior = {result.posterior:.17g}")
    print(f"hazard_tree = {result.hazard_tree:.17g}")
    print(f"connected_facility = {result.connected_facility:.17g}")
    print(f"facility_ratio = {result.facility_ratio:.17g}")
    print(f"facilities mask = {result.facilities}")
    print("parents =", result.parents)
    print("d =")
    for row in d:
        print(" ".join(format(x, ".17g") for x in row))
    facility_result, facility_graph_mask, facility_p, _ = best_facility
    print(f"best facility graph mask = {facility_graph_mask}")
    print("facility p =", " ".join(format(x, ".17g") for x in facility_p))
    print(f"max facility_ratio = {facility_result.facility_ratio:.17g}")
    print(f"facility objective = {facility_result.connected_facility:.17g}")
    print(f"facility posterior = {facility_result.posterior:.17g}")
    print(f"facility mask = {facility_result.facilities}")


def _prufer_edges(sequence: Sequence[int], total: int) -> list[tuple[int, int]]:
    degree = [1] * total
    for v in sequence:
        degree[v] += 1
    edges: list[tuple[int, int]] = []
    for v in sequence:
        leaf = min(i for i in range(total) if degree[i] == 1)
        edges.append((leaf, v))
        degree[leaf] -= 1
        degree[v] -= 1
    last = [i for i in range(total) if degree[i] == 1]
    edges.append((last[0], last[1]))
    return edges


def _tree_hazard_from_edges(
    d: Sequence[Sequence[float]], p: Sequence[float], edges: Sequence[tuple[int, int]]
) -> float:
    n = len(p)
    root = n
    adjacency = [[] for _ in range(n + 1)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    parent = [-2] * (n + 1)
    parent[root] = -1
    order = [root]
    for u in order:
        for v in adjacency[u]:
            if v != parent[u]:
                parent[v] = u
                order.append(v)
    descendants = [0] * (n + 1)
    value = 0.0
    for u in reversed(order[1:]):
        descendants[u] |= 1 << u
        mask = descendants[u]
        inactive = 1.0
        for v in range(n):
            if mask & (1 << v):
                inactive *= 1.0 - p[v]
        value += d[u][parent[u]] * (1.0 - inactive)
        descendants[parent[u]] |= mask
    return value


def self_test() -> None:
    """Cross-check both exact DPs against independent brute-force enumerations."""
    rng = random.Random(20260906)
    for case in range(12):
        n = 4
        total = n + 1
        d = [[0.0] * total for _ in range(total)]
        for i in range(total):
            for j in range(i):
                d[i][j] = d[j][i] = rng.uniform(1.0, 2.0)
        p = tuple(rng.uniform(0.02, 0.98) for _ in range(n))

        dynamic_phi, _ = best_hazard_tree(d, p)
        brute_phi = min(
            _tree_hazard_from_edges(d, p, _prufer_edges(sequence, total))
            for sequence in itertools.product(range(total), repeat=total - 2)
        )
        if not math.isclose(dynamic_phi, brute_phi, rel_tol=1e-11, abs_tol=1e-11):
            raise AssertionError((case, dynamic_phi, brute_phi))

        held_karp = subset_tsp_costs(d)
        for mask in range(1 << n):
            active = [v for v in range(n) if mask & (1 << v)]
            if not active:
                brute_tsp = 0.0
            else:
                brute_tsp = min(
                    d[n][order[0]]
                    + sum(d[order[i]][order[i + 1]] for i in range(len(order) - 1))
                    + d[order[-1]][n]
                    for order in itertools.permutations(active)
                )
            if not math.isclose(held_karp[mask], brute_tsp, rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError((case, mask, held_karp[mask], brute_tsp))
    print("self-test passed: 12 metrics, 1,500 Prüfer trees, all 192 subset tours")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=7)
    parser.add_argument("--metrics", type=int, default=100)
    parser.add_argument("--probabilities", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--evolve", action="store_true")
    parser.add_argument("--evolve-closure", action="store_true")
    parser.add_argument("--exhaustive-client-graphs", action="store_true")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--target", choices=("hazard", "facility"), default="hazard")
    parser.add_argument("--grid-denominator", type=int, default=20)
    parser.add_argument("--depot-distance", type=float, default=2.0)
    parser.add_argument("--heterogeneous-values", default="")
    parser.add_argument("--zero-prefix", type=int, default=0)
    parser.add_argument("--skip-facility", action="store_true")
    parser.add_argument("--uniform-values", default="")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--families",
        default="unit,path,cycle,star,binary,random-tree,graph12,graph123,euclidean",
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.evolve:
        evolutionary_search(args)
    elif args.exhaustive_client_graphs:
        exhaustive_client_graphs(args)
    else:
        search(args)


if __name__ == "__main__":
    main()
