#!/usr/bin/env python3
"""Stress-test a compressed two-guide master-route construction.

Draw independent guide samples B and C.  Map every client to a nearest point
of B union the depot and retain the image

    T = f_B(C) minus {depot}.

Before the evaluation sample is drawn, use an optimal cycle of T as the
master backbone and attach every client to its nearest point of T union the
depot.  The usual doubled-excursion comparison proves

    OPT_prior <= E[tsp(T)] + 2 sum_v p_v E[d(v,T union {r})].

Conditional on B, memberships of centers in T are independent, with the
Voronoi-block occupancy probabilities.  This script evaluates the bound
exactly by finite subset enumeration (apart from floating-point arithmetic)
and compares it with the fixed-guide-cycle construction.
"""

from __future__ import annotations

import argparse
import random
from typing import Sequence

from search_hazard_tree_ratio import (
    expected_posterior,
    named_metric,
    probability_vectors,
    subset_tsp_costs,
)


def subset_mst_costs(d: Sequence[Sequence[float]]) -> list[float]:
    """MST costs on every client subset together with the depot."""
    n = len(d) - 1
    root = n
    answer = [0.0] * (1 << n)
    for mask in range(1, 1 << n):
        vertices = [root] + [v for v in range(n) if mask & (1 << v)]
        used = {root}
        value = 0.0
        while len(used) < len(vertices):
            weight, vertex = min(
                (d[u][v], v)
                for u in used
                for v in vertices
                if v not in used
            )
            value += weight
            used.add(vertex)
        answer[mask] = value
    return answer


def mask_probability(mask: int, p: Sequence[float]) -> float:
    value = 1.0
    for v, pv in enumerate(p):
        value *= pv if mask & (1 << v) else 1.0 - pv
    return value


def evaluate(
    d: Sequence[Sequence[float]],
    p: Sequence[float],
    costs: Sequence[float] | None = None,
    client_only_projection: bool = False,
) -> tuple[float, float, float, float]:
    """Return (posterior, compressed bound, backbone term, attachment term)."""
    n = len(p)
    root = n
    if costs is None:
        costs = subset_tsp_costs(d)
    posterior = expected_posterior(p, costs)
    backbone = 0.0
    attachment = 0.0

    for guide in range(1 << n):
        guide_probability = mask_probability(guide, p)
        if guide_probability == 0.0:
            continue
        centers = [v for v in range(n) if guide & (1 << v)]
        available = (
            tuple(centers)
            if client_only_projection and centers
            else (*centers, root)
        )
        image: list[int] = []
        for v in range(n):
            if guide & (1 << v):
                image.append(v)
            else:
                image.append(min(available, key=lambda u: (d[v][u], u)))

        q: dict[int, float] = {}
        for center in centers:
            empty = 1.0
            for v in range(n):
                if image[v] == center:
                    empty *= 1.0 - p[v]
            q[center] = 1.0 - empty

        retained = guide
        sub = retained
        while True:
            conditional_probability = 1.0
            for center in centers:
                conditional_probability *= (
                    q[center] if sub & (1 << center) else 1.0 - q[center]
                )
            weight = guide_probability * conditional_probability
            backbone += weight * costs[sub]
            selected = [root] + [v for v in centers if sub & (1 << v)]
            attachment += weight * sum(
                p[v] * min(d[v][u] for u in selected) for v in range(n)
            )
            if sub == 0:
                break
            sub = (sub - 1) & retained

    return posterior, backbone + 2.0 * attachment, backbone, attachment


def expected_compressed_mst(
    d: Sequence[Sequence[float]], p: Sequence[float]
) -> tuple[float, float]:
    """Return expected MST cost of B and of T=f_B(C)."""
    n = len(p)
    root = n
    mst = subset_mst_costs(d)
    original = sum(mask_probability(mask, p) * mst[mask] for mask in range(1 << n))
    compressed = 0.0
    for guide in range(1 << n):
        guide_probability = mask_probability(guide, p)
        centers = [v for v in range(n) if guide & (1 << v)]
        available = (*centers, root)
        image = [
            v if guide & (1 << v) else min(available, key=lambda u: (d[v][u], u))
            for v in range(n)
        ]
        q = {}
        for center in centers:
            empty = 1.0
            for v in range(n):
                if image[v] == center:
                    empty *= 1.0 - p[v]
            q[center] = 1.0 - empty
        retained = guide
        while True:
            probability = guide_probability
            for center in centers:
                probability *= q[center] if retained & (1 << center) else 1.0 - q[center]
            compressed += probability * mst[retained]
            if retained == 0:
                break
            retained = (retained - 1) & guide
    return original, compressed


def sampled_nearest_assignment(
    d: Sequence[Sequence[float]], p: Sequence[float]
) -> float:
    """Return R=sum_v p_v E_B d(v,B union {root}) by enumeration."""
    n = len(p)
    root = n
    value = 0.0
    for guide in range(1 << n):
        weight = mask_probability(guide, p)
        selected = [root] + [v for v in range(n) if guide & (1 << v)]
        value += weight * sum(
            p[v] * min(d[v][u] for u in selected) for v in range(n)
        )
    return value


def compressed_radius_terms(
    d: Sequence[Sequence[float]], p: Sequence[float]
) -> tuple[float, float, float, float]:
    """Return (L, J, L_T, N_T) for the retained-guide radius split.

    Here L is the expected sum of each active guide's distance to its nearest
    other guide point (with the depot available), J=L-R, and L_T is the same
    original-guide radius weighted by the probability that its Voronoi cell
    is hit by the second sample.  These terms stress-test the prospective
    decomposition

        L_T <= (1-1/e)L + J,       M_1 <= L_T + 2D.
    """
    n = len(p)
    root = n
    radius = sampled_nearest_assignment(d, p)
    nearest_other = 0.0
    retained_radius = 0.0
    compressed_nearest_other = 0.0
    for guide in range(1 << n):
        guide_probability = mask_probability(guide, p)
        centers = [v for v in range(n) if guide & (1 << v)]
        available = (*centers, root)
        image = [
            v if guide & (1 << v) else min(available, key=lambda u: (d[v][u], u))
            for v in range(n)
        ]
        for center in centers:
            other = [root] + [v for v in centers if v != center]
            local_radius = min(d[center][v] for v in other)
            empty = 1.0
            for v in range(n):
                if image[v] == center:
                    empty *= 1.0 - p[v]
            occupancy = 1.0 - empty
            nearest_other += guide_probability * local_radius
            retained_radius += guide_probability * occupancy * local_radius
        retained = guide
        while True:
            conditional_probability = 1.0
            for center in centers:
                empty = 1.0
                for v in range(n):
                    if image[v] == center:
                        empty *= 1.0 - p[v]
                occupancy = 1.0 - empty
                conditional_probability *= (
                    occupancy if retained & (1 << center) else 1.0 - occupancy
                )
            retained_centers = [v for v in centers if retained & (1 << v)]
            retained_radius_sum = 0.0
            for center in retained_centers:
                other = [root] + [v for v in retained_centers if v != center]
                retained_radius_sum += min(d[center][v] for v in other)
            compressed_nearest_other += (
                guide_probability * conditional_probability * retained_radius_sum
            )
            if retained == 0:
                break
            retained = (retained - 1) & guide
    return nearest_other, nearest_other - radius, retained_radius, compressed_nearest_other


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--metrics", type=int, default=100)
    parser.add_argument("--probabilities", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument(
        "--uniform-values",
        default="",
        help="comma-separated common probabilities (otherwise use mixed grid)",
    )
    parser.add_argument(
        "--families",
        default="unit,path,cycle,star,binary,random-tree,graph12,graph123,"
        "client-graph-19,euclidean,multiscale-graph,multiscale-complete",
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)
    families = args.families.split(",")
    worst = None
    worst_attachment = None
    worst_hard_backbone = None
    worst_e_constant = None
    worst_tour_drop = None
    worst_e_radius_drop = None
    worst_parity_transfer = None
    worst_attachment_deficit = None
    worst_radius_contraction = None
    worst_core_residual = None
    worst_core_parity_nn = None
    worst_radius_inflation = None
    worst_core_parity_mst = None
    worst_mst_radius = None
    worst_deletion_influence = None
    worst_influence_bridge = None
    worst_front_loading_ratio = None
    worst_joint_front_loading_ratio = None
    worst_hard_joint_front_loading_ratio = None
    worst_weak_gamma_violation = None
    worst_hard_joint_p_ratio = None
    worst_attachment_to_core_drop = None
    for metric_index in range(args.metrics):
        family = families[metric_index % len(families)]
        d = named_metric(family, args.clients, rng)
        costs = subset_tsp_costs(d)
        if args.uniform_values:
            candidates = (
                (float(value),) * args.clients
                for value in args.uniform_values.split(",")
            )
        else:
            candidates = probability_vectors(
                args.clients, rng, args.probabilities
            )
        for p in candidates:
            result = evaluate(d, p, costs)
            mst_original, mst_compressed = expected_compressed_mst(d, p)
            nearest = sampled_nearest_assignment(d, p)
            (
                nearest_other,
                self_deficit,
                retained_radius,
                compressed_nearest_other,
            ) = compressed_radius_terms(d, p)
            deficit = result[0] - nearest
            alpha = 1.0 - 1.0 / 2.718281828459045
            if deficit > 1e-12 * max(1.0, result[0]):
                front_loading_ratio = (
                    result[2] - alpha * result[0]
                ) / deficit
                joint_front_loading_ratio = (
                    result[2]
                    - alpha * result[0]
                    + 2.0 * (result[3] - result[0])
                ) / deficit
                if (
                    worst_front_loading_ratio is None
                    or front_loading_ratio > worst_front_loading_ratio[0]
                ):
                    worst_front_loading_ratio = (
                        front_loading_ratio,
                        family,
                        tuple(p),
                        result,
                        nearest,
                    )
                if (
                    worst_joint_front_loading_ratio is None
                    or joint_front_loading_ratio
                    > worst_joint_front_loading_ratio[0]
                ):
                    worst_joint_front_loading_ratio = (
                        joint_front_loading_ratio,
                        family,
                        tuple(p),
                        result,
                        nearest,
                    )
                if (
                    deficit < result[0] / 20.0
                    and (
                        worst_hard_joint_front_loading_ratio is None
                        or joint_front_loading_ratio
                        > worst_hard_joint_front_loading_ratio[0]
                    )
                ):
                    worst_hard_joint_front_loading_ratio = (
                        joint_front_loading_ratio,
                        family,
                        tuple(p),
                        result,
                        nearest,
                    )
            joint_front_loading = (
                result[2]
                - alpha * result[0]
                + 2.0 * (result[3] - result[0])
            )
            gamma = result[0] / 2.718281828459045 - joint_front_loading
            core_drop = result[0] - result[2]
            attachment_increase = result[3] - nearest
            if core_drop > 1e-12 * max(1.0, result[0]):
                attachment_to_core_drop = attachment_increase / core_drop
                if (
                    worst_attachment_to_core_drop is None
                    or attachment_to_core_drop
                    > worst_attachment_to_core_drop[0]
                ):
                    worst_attachment_to_core_drop = (
                        attachment_to_core_drop,
                        deficit / result[0] if result[0] else 0.0,
                        family,
                        tuple(p),
                        result,
                        nearest,
                    )
            weak_gamma_violation = 2.0 * nearest / 19.0 - gamma
            if (
                worst_weak_gamma_violation is None
                or weak_gamma_violation > worst_weak_gamma_violation[0]
            ):
                worst_weak_gamma_violation = (
                    weak_gamma_violation,
                    family,
                    tuple(p),
                    result,
                    nearest,
                    joint_front_loading,
                )
            if result[0] and deficit < result[0] / 20.0:
                joint_p_ratio = joint_front_loading / result[0]
                if (
                    worst_hard_joint_p_ratio is None
                    or joint_p_ratio > worst_hard_joint_p_ratio[0]
                ):
                    worst_hard_joint_p_ratio = (
                        joint_p_ratio,
                        deficit / result[0],
                        family,
                        tuple(p),
                        result,
                        nearest,
                    )
            ratio = result[1] / result[0] if result[0] else 0.0
            if worst is None or ratio > worst[0]:
                worst = (ratio, family, tuple(p), result, d)
                print(
                    f"worst ratio={ratio:.12f} family={family} "
                    f"P={result[0]:.12f} bound={result[1]:.12f} "
                    f"tour={result[2]:.12f} attach={result[3]:.12f}",
                    flush=True,
                )
            attachment_ratio = result[3] / result[0] if result[0] else 0.0
            if worst_attachment is None or attachment_ratio > worst_attachment[0]:
                worst_attachment = (
                    attachment_ratio,
                    family,
                    tuple(p),
                    result,
                    nearest,
                )
            if result[0] and deficit < result[0] / 20.0:
                backbone_ratio = result[2] / result[0]
                if (
                    worst_hard_backbone is None
                    or backbone_ratio > worst_hard_backbone[0]
                ):
                    worst_hard_backbone = (
                        backbone_ratio,
                        deficit / result[0],
                        family,
                        tuple(p),
                        result,
                        nearest,
                    )
            e_constant_ratio = (
                result[1]
                / ((3.0 - 1.0 / 2.718281828459045) * result[0])
                if result[0]
                else 0.0
            )
            if worst_e_constant is None or e_constant_ratio > worst_e_constant[0]:
                worst_e_constant = (
                    e_constant_ratio,
                    family,
                    tuple(p),
                    result,
                    nearest,
                )
            # Positive means the tentative tour-drop inequality
            # P-M1 >= (2R-P)/e is violated.
            tour_drop_violation = (
                (2.0 * nearest - result[0]) / 2.718281828459045
                - (result[0] - result[2])
            )
            if (
                worst_tour_drop is None
                or tour_drop_violation > worst_tour_drop[0]
            ):
                worst_tour_drop = (
                    tour_drop_violation,
                    family,
                    tuple(p),
                    result,
                    nearest,
                )
            # Positive means the sharper tentative inequality
            # P-M1 >= R/e is violated.
            e_radius_violation = (
                nearest / 2.718281828459045 - (result[0] - result[2])
            )
            if (
                worst_e_radius_drop is None
                or e_radius_violation > worst_e_radius_drop[0]
            ):
                worst_e_radius_drop = (
                    e_radius_violation,
                    family,
                    tuple(p),
                    result,
                    nearest,
                )
            parity_transfer_violation = (
                (result[2] - mst_compressed) - (result[0] - mst_original)
            )
            if (
                worst_parity_transfer is None
                or parity_transfer_violation > worst_parity_transfer[0]
            ):
                worst_parity_transfer = (
                    parity_transfer_violation,
                    family,
                    tuple(p),
                    result,
                    nearest,
                    mst_original,
                    mst_compressed,
                )
            attachment_deficit_violation = result[3] + nearest - 2.0 * result[0]
            if (
                worst_attachment_deficit is None
                or attachment_deficit_violation > worst_attachment_deficit[0]
            ):
                worst_attachment_deficit = (
                    attachment_deficit_violation,
                    family,
                    tuple(p),
                    result,
                    nearest,
                )
            radius_contraction_violation = (
                retained_radius - alpha * nearest_other - self_deficit
            )
            if (
                worst_radius_contraction is None
                or radius_contraction_violation > worst_radius_contraction[0]
            ):
                worst_radius_contraction = (
                    radius_contraction_violation,
                    family,
                    tuple(p),
                    result,
                    nearest,
                    nearest_other,
                    self_deficit,
                    retained_radius,
                )
            core_residual_violation = result[2] - retained_radius - 2.0 * deficit
            if worst_core_residual is None or core_residual_violation > worst_core_residual[0]:
                worst_core_residual = (
                    core_residual_violation,
                    family,
                    tuple(p),
                    result,
                    nearest,
                    nearest_other,
                    self_deficit,
                    retained_radius,
                )
            core_parity_nn_violation = (
                result[2] - compressed_nearest_other - deficit
            )
            if worst_core_parity_nn is None or core_parity_nn_violation > worst_core_parity_nn[0]:
                worst_core_parity_nn = (
                    core_parity_nn_violation,
                    family,
                    tuple(p),
                    result,
                    nearest,
                    nearest_other,
                    self_deficit,
                    retained_radius,
                    compressed_nearest_other,
                )
            radius_inflation_violation = (
                compressed_nearest_other - retained_radius - deficit
            )
            if worst_radius_inflation is None or radius_inflation_violation > worst_radius_inflation[0]:
                worst_radius_inflation = (
                    radius_inflation_violation,
                    family,
                    tuple(p),
                    result,
                    nearest,
                    nearest_other,
                    self_deficit,
                    retained_radius,
                    compressed_nearest_other,
                )
            core_parity_mst_violation = (
                (result[2] - mst_compressed)
                - (result[0] - mst_original)
                - self_deficit
            )
            if (
                worst_core_parity_mst is None
                or core_parity_mst_violation > worst_core_parity_mst[0]
            ):
                worst_core_parity_mst = (
                    core_parity_mst_violation,
                    family,
                    tuple(p),
                    result,
                    nearest,
                    nearest_other,
                    self_deficit,
                    retained_radius,
                    mst_original,
                    mst_compressed,
                )
            mst_radius_violation = mst_compressed - retained_radius - deficit
            if worst_mst_radius is None or mst_radius_violation > worst_mst_radius[0]:
                worst_mst_radius = (
                    mst_radius_violation,
                    family,
                    tuple(p),
                    result,
                    nearest,
                    nearest_other,
                    self_deficit,
                    retained_radius,
                    mst_original,
                    mst_compressed,
                )
            deletion_influence = 0.0
            for guide in range(1 << args.clients):
                guide_probability = mask_probability(guide, p)
                for vertex in range(args.clients):
                    if guide & (1 << vertex):
                        deletion_influence += (
                            guide_probability
                            * (1.0 - p[vertex])
                            * (costs[guide] - costs[guide ^ (1 << vertex)])
                        )
            # The first quantity is positive on a violation of
            # I_del >= 2R-P.  The second tests the repaired compression
            # bridge P-M1+J >= I_del/e.
            deletion_influence_violation = (
                2.0 * nearest - result[0] - deletion_influence
            )
            influence_bridge_violation = (
                deletion_influence / 2.718281828459045
                - (result[0] - result[2] + self_deficit)
            )
            if (
                worst_deletion_influence is None
                or deletion_influence_violation > worst_deletion_influence[0]
            ):
                worst_deletion_influence = (
                    deletion_influence_violation,
                    family,
                    tuple(p),
                    result,
                    nearest,
                    self_deficit,
                    deletion_influence,
                )
            if (
                worst_influence_bridge is None
                or influence_bridge_violation > worst_influence_bridge[0]
            ):
                worst_influence_bridge = (
                    influence_bridge_violation,
                    family,
                    tuple(p),
                    result,
                    nearest,
                    self_deficit,
                    deletion_influence,
                )
    print("final worst =", worst)
    print("max A1/P =", worst_attachment)
    print("max M1/P with D/P<1/20 =", worst_hard_backbone)
    print("max U1/((3-1/e)P) =", worst_e_constant)
    print("max [(2R-P)/e-(P-M1)] =", worst_tour_drop)
    print("max [R/e-(P-M1)] =", worst_e_radius_drop)
    print("max compressed parity - original parity =", worst_parity_transfer)
    print("max [A1+R-2P] =", worst_attachment_deficit)
    print("max [L_T-(1-1/e)L-J] =", worst_radius_contraction)
    print("max [M1-L_T-2D] =", worst_core_residual)
    print("max [M1-N_T-D] =", worst_core_parity_nn)
    print("max [N_T-L_T-D] =", worst_radius_inflation)
    print("max [(M1-MST_T)-(P-MST_B)-J] =", worst_core_parity_mst)
    print("max [MST_T-L_T-D] =", worst_mst_radius)
    print("max [(2R-P)-I_del] =", worst_deletion_influence)
    print("max [I_del/e-(P-M1+J)] =", worst_influence_bridge)
    print("max [(M1-(1-1/e)P)/D] =", worst_front_loading_ratio)
    print(
        "max [(M1-(1-1/e)P+2(A1-P))/D] =",
        worst_joint_front_loading_ratio,
    )
    print(
        "same max restricted to D/P<1/20 =",
        worst_hard_joint_front_loading_ratio,
    )
    print("max [2R/19-Gamma1] =", worst_weak_gamma_violation)
    print("max X/P restricted to D/P<1/20 =", worst_hard_joint_p_ratio)
    print("max [(A1-R)/(P-M1)] =", worst_attachment_to_core_drop)


if __name__ == "__main__":
    main()
