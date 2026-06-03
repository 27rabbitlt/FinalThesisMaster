#!/usr/bin/env python3
"""
Generate recursive K_{2,2} bipartite graphs for stochastic TSP.

The graph is a bipartite structure at multiple scales:

  Level 1 (base): K_{2,2} — two groups of 2 vertices, cross edges only.
    Group A = {v1, v2}, Group B = {v3, v4}
    Cross edges: v1-v3, v1-v4, v2-v3, v2-v4 at weight w_1
    d(same group) = 2*w_1,  d(cross group) = w_1

  Level 2: Each top-level group IS a level-1 structure internally.
    Group A = {v1,v2,v3,v4} with internal K_{2,2} at weight w_1
    Group B = {v5,v6,v7,v8} with internal K_{2,2} at weight w_1
    Cross edges: A x B at weight w_2
    d(same leaf-pair) = 2*w_1,  d(cross within group) = w_1,  d(cross groups) = w_2

  Level L: 2^(L+1) customers total.

Key: NO intra-leaf-pair edges. Vertices in the same leaf pair have
     distance = 2*w_1 (through the nearest cross edges).

Usage:
  python gen_recursive_k22.py --level 1 --weights 1
  python gen_recursive_k22.py --level 2 --weights 1,2
  python gen_recursive_k22.py --level 2 --weights 1,3
"""

import argparse
import json
import sys


def build_group(level, weights, counter):
    """
    Build a group of vertices with recursive bipartite structure.

    Level 1: 2 vertices, no internal edges (leaf pair).
    Level L: two level-(L-1) sub-groups cross-connected at weight weights[L-2].

    Returns: (vertex_list, edge_list)
    """
    if level == 1:
        # Leaf pair: two vertices, no edge between them
        v1 = counter[0]; counter[0] += 1
        v2 = counter[0]; counter[0] += 1
        return [v1, v2], []

    # Recursive: two sub-groups cross-connected
    left_verts, left_edges = build_group(level - 1, weights, counter)
    right_verts, right_edges = build_group(level - 1, weights, counter)

    w = weights[level - 2]  # weight for this level's cross edges
    cross_edges = [(u, v, w) for u in left_verts for v in right_verts]

    return (left_verts + right_verts,
            left_edges + right_edges + cross_edges)


def build_full_graph(level, weights, top_weight):
    """
    Build the full recursive K_{2,2} graph.

    Two top-level groups cross-connected at top_weight.
    Each group is a build_group(level) structure.

    Returns: n, edges, prob
    """
    counter = [1]  # vertex 0 = depot

    left_verts, left_edges = build_group(level, weights, counter)
    right_verts, right_edges = build_group(level, weights, counter)

    # Top-level cross edges
    cross_edges = [(u, v, top_weight) for u in left_verts for v in right_verts]

    all_verts = left_verts + right_verts
    all_edges = left_edges + right_edges + cross_edges
    n = len(all_verts) + 1  # +1 for depot

    return n, all_verts, all_edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1,
                        help="Number of bipartite nesting levels (1=K_{2,2})")
    parser.add_argument("--weights", type=str, default="1",
                        help="Comma-separated weights (innermost first). "
                             "Need (level-1) intra-group weights + 1 top-level weight = level total")
    parser.add_argument("--prob", type=float, default=0.5)
    parser.add_argument("--depot-weight", type=float, default=None,
                        help="Weight for depot edges (default: top-level weight)")
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    weights_list = [float(x) for x in args.weights.split(",")]
    if len(weights_list) != args.level:
        print(f"Error: need {args.level} weights for level {args.level}, "
              f"got {len(weights_list)}", file=sys.stderr)
        sys.exit(1)

    # Last weight is the top-level cross weight
    top_weight = weights_list[-1]
    # Remaining weights are for intra-group structure
    intra_weights = weights_list[:-1]  # empty for level 1

    n, all_verts, all_edges = build_full_graph(args.level, intra_weights + [None],
                                                top_weight)
    # Fix: for level 1, build_group returns no edges, top cross is all we need.
    # For level > 1, intra_weights are used inside build_group.
    # Re-call properly:
    # Actually, let me just handle this correctly.

    # Rebuild: for the full graph, we call build_group at the given level
    # with all weights except the last (which is the top-level cross weight).
    # BUT the top level also needs to be a bipartite cross-connection.
    # So the structure is really: build_group(level+1) where weights includes the top weight.
    # OR: just treat the whole thing as one big build with level+1 levels.

    # Simplest approach: the full graph IS build_group(level+1, all_weights)
    # where level+1 accounts for the outermost bipartite split.
    # Actually no — build_group at level 1 = 2 vertices.
    # We want level 1 to give K_{2,2} = 4 vertices = two groups of 2 cross-connected.
    # That's build_group(2, [w]) = two build_group(1) cross-connected at w.

    # So: full_graph at level L = build_group(L+1, weights)
    # where weights has L entries (one per cross-connection level).

    # Let me redo this cleanly.
    pass


def build_group_v2(level, weights, counter):
    """
    Level 0: single vertex (leaf).
    Level L: two level-(L-1) sub-groups cross-connected at weights[L-1].
    """
    if level == 0:
        v = counter[0]; counter[0] += 1
        return [v], []

    left_verts, left_edges = build_group_v2(level - 1, weights, counter)
    right_verts, right_edges = build_group_v2(level - 1, weights, counter)

    w = weights[level - 1]
    cross = [(u, v, w) for u in left_verts for v in right_verts]

    return (left_verts + right_verts,
            left_edges + right_edges + cross)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1,
                        help="1=K_{2,2} (4 customers), 2=nested (8 customers), etc.")
    parser.add_argument("--weights", type=str, default="1",
                        help="Comma-separated: one weight per bipartite level. "
                             "Level 1 needs 1 weight, level 2 needs 2 weights, etc.")
    parser.add_argument("--prob", type=float, default=0.5)
    parser.add_argument("--depot-weight", type=float, default=None,
                        help="Weight for depot-to-customer edges (default: largest weight)")
    parser.add_argument("--selective-prob", action="store_true",
                        help="Only leaf pairs are stochastic, internal vertices deterministic")
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    weights = [float(x) for x in args.weights.split(",")]
    if len(weights) != args.level:
        print(f"Error: level {args.level} needs {args.level} weights, "
              f"got {len(weights)}", file=sys.stderr)
        sys.exit(1)

    # The full graph = build_group at (level+1) using [w_leaf] + weights
    # where w_leaf is a weight for the finest-level cross connections.
    # Actually, K_{2,2} (level 1) = two pairs of vertices cross-connected at w_1.
    # A "pair" is two vertices with NO edge (leaf pair).
    # This is build_group_v2(2, [_, w_1]) where level-0 are single vertices
    # and level-1 cross is... hmm.

    # Cleaner: build_group_v2(level) where level-0 = leaf pair (2 vertices, no edge)
    #   level L = two level-(L-1) groups cross-connected at weights[L-1]
    # Level 0: 2 vertices
    # Level 1: 4 vertices (K_{2,2})
    # Level 2: 8 vertices

    counter = [1]  # depot = 0

    def build(lev):
        if lev == 0:
            # Leaf pair: 2 vertices, no edges
            v1 = counter[0]; counter[0] += 1
            v2 = counter[0]; counter[0] += 1
            return [v1, v2], []

        left_v, left_e = build(lev - 1)
        right_v, right_e = build(lev - 1)
        w = weights[lev - 1]
        cross = [(u, v, w) for u in left_v for v in right_v]
        return left_v + right_v, left_e + right_e + cross

    verts, edges = build(args.level)
    n = len(verts) + 1

    # Depot edges
    dw = args.depot_weight if args.depot_weight is not None else max(weights)
    depot_edges = [(0, v, dw) for v in verts]
    all_edges = depot_edges + edges

    # Probabilities
    prob = [1.0] + [args.prob] * len(verts)

    edge_list = [{"s": u, "t": v, "w": w} for u, v, w in all_edges]
    result = {
        "_comment": f"Recursive K_{{2,2}} level={args.level}, weights={weights}, "
                    f"n={n}, {len(verts)} customers, all p={args.prob}",
        "n": n,
        "sym": True,
        "edges": edge_list,
        "prob": prob,
    }

    out = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out + "\n")
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
