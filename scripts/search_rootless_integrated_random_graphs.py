#!/usr/bin/env python3
"""Adversarial search for integrated rootless two-sample guarantees.

For each sampled graph metric and probability vector this evaluates

    Gamma* = 3P - M* - 2A*,

where, for nonempty B, the second sample is projected to nearest *client*
centers in B (the depot is excluded); for B empty the core is empty.  It
searches for violations of Gamma* >= R/e and of the weaker sufficient target
Gamma* >= 2R/19, and also records the better of rootful and rootless cores.
This is exploratory floating-point code; exact witnesses must be moved to a
dedicated Fraction checker.
"""

from __future__ import annotations

import argparse
import math
import random

from scipy.optimize import differential_evolution

from search_compressed_two_guide import mask_probability
from search_hazard_tree_ratio import named_metric, subset_tsp_costs
from search_rootless_threshold_random_graphs import random_connected_metric


def random_probability(rng: random.Random) -> float:
    mode = rng.randrange(5)
    if mode == 0:
        return rng.uniform(0.001, 0.999)
    if mode == 1:
        return 1.0 / (1.0 + math.exp(-rng.uniform(-8.0, 8.0)))
    if mode == 2:
        return rng.betavariate(0.25, 0.25)
    if mode == 3:
        return rng.betavariate(0.7, 2.0)
    return rng.choice((0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.99))


class Instance:
    def __init__(self, d, edges):
        self.d = d
        self.edges = edges
        self.n = len(d) - 1
        self.costs = subset_tsp_costs(d)
        root = self.n
        self.nearest = [
            [
                min([d[v][root]] + [d[v][u] for u in range(self.n) if mask & (1 << u)])
                for v in range(self.n)
            ]
            for mask in range(1 << self.n)
        ]
        self.rootless_cells = self._all_cells(True)
        self.rootful_cells = self._all_cells(False)
        self.rootless_projection = self._all_projections(True)
        self.rootful_projection = self._all_projections(False)

    def _all_cells(self, rootless):
        root = self.n
        answer = []
        for guide in range(1 << self.n):
            centers = [v for v in range(self.n) if guide & (1 << v)]
            available = centers if rootless and centers else centers + [root]
            cells = {b: 0 for b in centers}
            for v in range(self.n):
                image = v if guide & (1 << v) else min(
                    available, key=lambda b: (self.d[v][b], b)
                )
                if image != root:
                    cells[image] |= 1 << v
            answer.append(tuple(cells.items()))
        return answer

    def _all_projections(self, rootless):
        root = self.n
        answer = [[0] * (1 << self.n) for _ in range(1 << self.n)]
        for guide in range(1 << self.n):
            centers = [v for v in range(self.n) if guide & (1 << v)]
            available = centers if rootless and centers else centers + [root]
            image = [
                v if guide & (1 << v) else min(
                    available, key=lambda b: (self.d[v][b], b)
                )
                for v in range(self.n)
            ]
            for sample in range(1 << self.n):
                target = 0
                for v in range(self.n):
                    if sample & (1 << v) and image[v] != root:
                        target |= 1 << image[v]
                answer[guide][sample] = target
        return answer

    def compressed(self, p, weights, assignment, cells):
        backbone = 0.0
        attachment = 0.0
        for guide in range(1 << self.n):
            occupancy = {}
            for center, cell in cells[guide]:
                empty = 1.0
                for v in range(self.n):
                    if cell & (1 << v):
                        empty *= 1.0 - p[v]
                occupancy[center] = 1.0 - empty
            sub = guide
            while True:
                conditional = 1.0
                for center, _ in cells[guide]:
                    conditional *= (
                        occupancy[center]
                        if sub & (1 << center)
                        else 1.0 - occupancy[center]
                    )
                weight = weights[guide] * conditional
                backbone += weight * self.costs[sub]
                attachment += weight * assignment[sub]
                if sub == 0:
                    break
                sub = (sub - 1) & guide
        return backbone, attachment

    def terms(self, p, include_rootful=True):
        weights = [mask_probability(mask, p) for mask in range(1 << self.n)]
        assignment = [
            sum(p[v] * self.nearest[mask][v] for v in range(self.n))
            for mask in range(1 << self.n)
        ]
        posterior = sum(weights[mask] * self.costs[mask] for mask in range(1 << self.n))
        radius = sum(weights[mask] * assignment[mask] for mask in range(1 << self.n))
        m_star, a_star = self.compressed(
            p, weights, assignment, self.rootless_cells
        )
        if include_rootful:
            m_root, a_root = self.compressed(
                p, weights, assignment, self.rootful_cells
            )
        else:
            m_root = a_root = float("nan")
        gamma_star = 3.0 * posterior - m_star - 2.0 * a_star
        gamma_root = 3.0 * posterior - m_root - 2.0 * a_root
        best_facility = min(
            self.costs[mask] + 2.0 * assignment[mask]
            for mask in range(1 << self.n)
        )
        return posterior, radius, m_star, a_star, gamma_star, gamma_root, best_facility

    def weak_gap(self, p):
        _, radius, _, _, gamma_star, _, _ = self.terms(p, include_rootful=False)
        return gamma_star - 2.0 * radius / 19.0

    def ratio(self, p):
        _, radius, _, _, gamma_star, _, _ = self.terms(p, include_rootful=False)
        return gamma_star / radius if radius > 1e-15 else 1e6

    def hybrid(self, p, include_guide=True):
        """Pairwise min of rootful/rootless J, optionally also the guide J."""
        weights = [mask_probability(mask, p) for mask in range(1 << self.n)]
        assignment = [
            sum(p[v] * self.nearest[mask][v] for v in range(self.n))
            for mask in range(1 << self.n)
        ]
        objective = [
            self.costs[mask] + 2.0 * assignment[mask]
            for mask in range(1 << self.n)
        ]
        value = 0.0
        for guide in range(1 << self.n):
            for sample in range(1 << self.n):
                candidates = (
                    objective[self.rootful_projection[guide][sample]],
                    objective[self.rootless_projection[guide][sample]],
                )
                if include_guide:
                    candidates += (objective[guide],)
                value += weights[guide] * weights[sample] * min(candidates)
        return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--graphs", type=int, default=500)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--optimize", type=int, default=10)
    parser.add_argument("--seed", type=int, default=71009)
    parser.add_argument("--named", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    best_weak = (math.inf, None, None, None)
    best_e = (math.inf, None, None, None)
    best_combined = (math.inf, None, None, None)
    best_ratio = (math.inf, None, None, None)
    worst_attachment = (-math.inf, None, None, None)
    worst_hard_backbone = (-math.inf, None, None, None)
    worst_hard_route = (-math.inf, None, None, None)
    worst_facility_ratio = (-math.inf, None, None, None)
    worst_hybrid_route = (-math.inf, None, None, None)
    minimum_hybrid_margin = (math.inf, None, None, None)
    candidates = []
    families = (
        "unit",
        "path",
        "cycle",
        "star",
        "binary",
        "random-tree",
        "graph12",
        "graph123",
        "client-graph-19",
        "euclidean",
        "multiscale-graph",
        "multiscale-complete",
    )
    for graph_index in range(args.graphs):
        if args.named:
            family = families[graph_index % len(families)]
            d = named_metric(family, args.clients, rng)
            edges = [(family, graph_index)]
        else:
            d, edges = random_connected_metric(args.clients, rng)
        instance = Instance(d, edges)
        local = (math.inf, None)
        probability_vectors = [
            (value,) * args.clients
            for value in (0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.99)
        ]
        probability_vectors.extend(
            tuple(random_probability(rng) for _ in range(args.clients))
            for _ in range(args.samples)
        )
        for p in probability_vectors:
            terms = instance.terms(p)
            posterior, radius, _, _, gamma_star, gamma_root, best_facility = terms
            weak = gamma_star - 2.0 * radius / 19.0
            e_gap = gamma_star - radius / math.e
            combined = max(gamma_star, gamma_root) - 2.0 * radius / 19.0
            if weak < local[0]:
                local = (weak, p)
            if weak < best_weak[0]:
                best_weak = (weak, instance, p, terms)
            if e_gap < best_e[0]:
                best_e = (e_gap, instance, p, terms)
            if combined < best_combined[0]:
                best_combined = (combined, instance, p, terms)
            if radius > 1e-15 and gamma_star / radius < best_ratio[0]:
                best_ratio = (gamma_star / radius, instance, p, terms)
            attachment_excess = terms[3] - posterior
            if attachment_excess > worst_attachment[0]:
                worst_attachment = (attachment_excess, instance, p, terms)
            deficit = posterior - radius
            if posterior > 0 and deficit < posterior / 20.0:
                backbone_ratio = terms[2] / posterior
                route_ratio = (terms[2] + 2 * terms[3]) / posterior
                if backbone_ratio > worst_hard_backbone[0]:
                    worst_hard_backbone = (backbone_ratio, instance, p, terms)
                if route_ratio > worst_hard_route[0]:
                    worst_hard_route = (route_ratio, instance, p, terms)
            facility_ratio = best_facility / posterior if posterior else 0.0
            if facility_ratio > worst_facility_ratio[0]:
                worst_facility_ratio = (facility_ratio, instance, p, terms)
            if args.hybrid:
                hybrid_route = instance.hybrid(p)
                hybrid_ratio = hybrid_route / posterior if posterior else 0.0
                hybrid_margin = 3 * posterior - hybrid_route - 2 * radius / 19
                if hybrid_ratio > worst_hybrid_route[0]:
                    worst_hybrid_route = (hybrid_ratio, instance, p, terms)
                if hybrid_margin < minimum_hybrid_margin[0]:
                    minimum_hybrid_margin = (hybrid_margin, instance, p, terms)
            if weak < -1e-8:
                print("WEAK COUNTEREXAMPLE", weak, edges, p, terms, d)
                return
        candidates.append((local[0], instance, local[1]))
        if graph_index % 50 == 0:
            print(
                "progress",
                graph_index,
                "weak",
                best_weak[0],
                "e",
                best_e[0],
                "combined",
                best_combined[0],
                "ratio",
                best_ratio[0],
                "A-P",
                worst_attachment[0],
                "hardM",
                worst_hard_backbone[0],
                "hardRoute",
                worst_hard_route[0],
                "minJ/P",
                worst_facility_ratio[0],
                "hybrid/P",
                worst_hybrid_route[0],
                flush=True,
            )

    candidates.sort(key=lambda row: row[0])
    for _, instance, initial in candidates[: args.optimize]:
        result = differential_evolution(
            lambda values: instance.ratio(values),
            [(1e-5, 1 - 1e-5)] * args.clients,
            seed=rng.randrange(1 << 30),
            maxiter=180,
            popsize=10,
            polish=True,
            updating="immediate",
        )
        terms = instance.terms(result.x)
        weak = instance.weak_gap(result.x)
        ratio = result.fun
        if weak < best_weak[0]:
            best_weak = (weak, instance, tuple(result.x), terms)
        if weak < -1e-8:
            print(
                "OPTIMIZED WEAK COUNTEREXAMPLE",
                weak,
                instance.edges,
                tuple(result.x),
                terms,
                instance.d,
            )
            return
        if ratio < best_ratio[0]:
            best_ratio = (ratio, instance, tuple(result.x), terms)
    for label, record in (
        ("weak", best_weak),
        ("e", best_e),
        ("better-rootful-rootless", best_combined),
        ("Gamma*/R", best_ratio),
        ("A*-P", worst_attachment),
        ("hard M*/P", worst_hard_backbone),
        ("hard route/P", worst_hard_route),
        ("max minJ/P", worst_facility_ratio),
        ("max hybrid/P", worst_hybrid_route),
        ("min hybrid weak margin", minimum_hybrid_margin),
    ):
        gap, instance, p, terms = record
        if instance is None:
            print(label, "no qualifying instance")
        else:
            print(label, gap, instance.edges, p, terms, instance.d)


if __name__ == "__main__":
    main()
