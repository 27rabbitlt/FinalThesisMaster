#!/usr/bin/env python3
"""
Visualize the optimal adaptive policy as a decision tree.

Runs the adaptive DP, extracts the optimal probing order at each state,
and generates a Graphviz decision tree showing every probe, branch, and
travel step.

Usage:
  python visualize_policy.py examples/asym-manuel.json
  python visualize_policy.py examples/asym-manuel.json -o policy.svg
  python visualize_policy.py examples/asym-manuel.json --format pdf
  python visualize_policy.py examples/asym-manuel.json --compact  # collapse deterministic chains
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import webbrowser

INF = 1e18


def load_instance(path):
    with open(path) as f:
        data = json.load(f)
    n = data["n"]
    V = n
    prob = data["prob"]
    sym = data.get("sym", False)

    dist = [[INF] * V for _ in range(V)]
    for i in range(V):
        dist[i][i] = 0

    if "dist" in data:
        for i in range(V):
            for j in range(V):
                dist[i][j] = data["dist"][i][j]
    elif "edges" in data:
        for e in data["edges"]:
            u, v, w = e["s"], e["t"], e.get("w", 1)
            dist[u][v] = min(dist[u][v], w)
            if sym:
                dist[v][u] = min(dist[v][u], w)
    else:
        raise ValueError("Need 'dist' or 'edges'")

    # Floyd-Warshall
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return n, dist, prob


def solve_adaptive_with_policy(n, dist, prob):
    """Adaptive DP returning (cost, F table, policy table).

    Convention (matching tsp_solver.h):
      nc = n - 1  (number of customers)
      pos in 0..nc: 0 = depot (vertex 0), i = customer i (vertex i) for i in 1..nc
      mask: bitmask over nc customers; bit i (0-indexed) = customer i+1
      policy[pos][mask] = customer index i (0-based) to probe next
    """
    nc = n - 1
    full = 1 << nc

    F = [[0.0] * full for _ in range(nc + 1)]
    policy = [[-1] * full for _ in range(nc + 1)]

    # Base: all probed → return to depot
    for pos in range(nc + 1):
        F[pos][full - 1] = dist[pos][0]

    for bits in range(nc - 1, -1, -1):
        for mask in range(full):
            if bin(mask).count("1") != bits:
                continue
            for pos in range(nc + 1):
                best = INF
                best_i = -1
                for i in range(nc):
                    if mask & (1 << i):
                        continue
                    nxt = mask | (1 << i)
                    val = (1.0 - prob[i + 1]) * F[pos][nxt] + prob[i + 1] * (
                        dist[pos][i + 1] + F[i + 1][nxt]
                    )
                    if val < best:
                        best = val
                        best_i = i
                F[pos][mask] = best
                policy[pos][mask] = best_i

    return F[0][0], F, policy


def build_tree(nc, dist, prob, F, policy):
    """Build the decision tree as nested dicts by following the policy."""
    node_counter = [0]

    def make_node(pos, mask):
        nid = node_counter[0]
        node_counter[0] += 1
        full = (1 << nc) - 1

        if mask == full:
            return {
                "id": nid,
                "type": "leaf",
                "pos": pos,
                "cost_to_depot": dist[pos][0],
            }

        i = policy[pos][mask]
        vertex = i + 1
        p = prob[vertex]
        nxt = mask | (1 << i)

        node = {
            "id": nid,
            "type": "decision",
            "pos": pos,
            "mask": mask,
            "probe": vertex,
            "prob_active": p,
            "expected_cost": F[pos][mask],
            "children": [],
        }

        if p > 0:
            child = make_node(vertex, nxt)
            node["children"].append(
                {
                    "outcome": "active",
                    "prob": p,
                    "travel_cost": dist[pos][vertex],
                    "child": child,
                }
            )
        if p < 1.0:
            child = make_node(pos, nxt)
            node["children"].append(
                {
                    "outcome": "inactive",
                    "prob": 1.0 - p,
                    "travel_cost": 0,
                    "child": child,
                }
            )

        return node

    return make_node(0, 0)


def compact_tree(node):
    """Collapse chains of deterministic (single-child) nodes into one node."""
    if node["type"] == "leaf":
        return node

    # Recurse first
    for branch in node["children"]:
        branch["child"] = compact_tree(branch["child"])

    # If exactly one child (deterministic), merge with child if child is also decision
    if len(node["children"]) == 1:
        branch = node["children"][0]
        child = branch["child"]
        # Build a chain label: "probe X → (travel d) → ..."
        chain = [node]
        travel_costs = [branch["travel_cost"]]
        cur = child
        while cur["type"] == "decision" and len(cur["children"]) == 1:
            chain.append(cur)
            travel_costs.append(cur["children"][0]["travel_cost"])
            cur = cur["children"][0]["child"]

        if len(chain) > 1:
            # Merge into a single compound node
            probes = [n["probe"] for n in chain]
            travels = travel_costs[:-1]  # travel between consecutive probes
            node["compound_probes"] = probes
            node["compound_travels"] = travel_costs
            if cur["type"] == "leaf":
                node["type"] = "compound_leaf"
                node["final_pos"] = cur["pos"]
                node["cost_to_depot"] = cur["cost_to_depot"]
                node["children"] = []
            else:
                node["children"] = cur["children"]
                node["pos"] = cur["pos"]
                node["mask"] = cur["mask"]
                node["expected_cost"] = node["expected_cost"]  # keep original

    return node


def fmt_num(x):
    if x == int(x):
        return str(int(x))
    return f"{x:.2f}"


def pos_label(pos):
    return "depot" if pos == 0 else str(pos)


def tree_to_dot(tree, nc, prob, title=""):
    """Convert decision tree to Graphviz DOT format."""
    lines = [
        "digraph AdaptivePolicy {",
        '  rankdir=TB;',
        '  bgcolor="white";',
        f'  label="{title}";',
        '  labelloc=t;',
        '  labeljust=c;',
        '  fontname="Helvetica";',
        '  fontsize=16;',
        '  node [fontname="Helvetica" fontsize=10 style=filled];',
        '  edge [fontname="Helvetica" fontsize=9];',
        "",
    ]

    def add_node(node):
        nid = node["id"]

        if node["type"] == "leaf":
            cost = node["cost_to_depot"]
            pos = pos_label(node["pos"])
            label = f"pos={pos}\\nReturn to depot\\ncost={fmt_num(cost)}"
            lines.append(
                f'  n{nid} [label="{label}" shape=box fillcolor="#ffe0e0" '
                f'color="#cc4444" penwidth=1.5];'
            )
            return

        if node.get("compound_probes"):
            # Compacted deterministic chain
            probes = node["compound_probes"]
            travels = node["compound_travels"]
            parts = []
            for idx, v in enumerate(probes):
                t = travels[idx] if idx < len(travels) else 0
                if t > 0:
                    parts.append(f"Probe {v} (p=1) → travel {fmt_num(t)}")
                else:
                    parts.append(f"Probe {v} (p=1)")
            label = "\\n".join(parts)
            ec = node["expected_cost"]
            pos = pos_label(node.get("pos", 0))
            header = f"E[cost]={fmt_num(ec)}"

            if node["type"] == "compound_leaf":
                final_pos = pos_label(node["final_pos"])
                cost = node["cost_to_depot"]
                label += f"\\n→ pos={final_pos}, return={fmt_num(cost)}"
                lines.append(
                    f'  n{nid} [label="{header}\\n{label}" shape=box '
                    f'fillcolor="#e0ffe0" color="#44aa44" penwidth=1.5];'
                )
                return
            else:
                lines.append(
                    f'  n{nid} [label="{header}\\n{label}" shape=box '
                    f'fillcolor="#e8f4e8" color="#44aa44" penwidth=1.5];'
                )
        else:
            pos = pos_label(node["pos"])
            v = node["probe"]
            p = node["prob_active"]
            ec = node["expected_cost"]
            label = f"pos={pos}\\nProbe {v}  (p={p:.2g})\\nE[cost]={fmt_num(ec)}"
            if p < 1.0 and p > 0.0:
                fillcolor = "#fff3cd"  # yellow for stochastic
                bordercolor = "#cc9900"
            else:
                fillcolor = "#d4e6f1"  # blue for deterministic
                bordercolor = "#2980b9"
            lines.append(
                f'  n{nid} [label="{label}" shape=ellipse '
                f'fillcolor="{fillcolor}" color="{bordercolor}" penwidth=1.5];'
            )

        for branch in node["children"]:
            child = branch["child"]
            add_node(child)
            outcome = branch["outcome"]
            p = branch["prob"]
            tc = branch["travel_cost"]
            if outcome == "active":
                if tc > 0:
                    elabel = f"active (p={p:.2g})\\ntravel={fmt_num(tc)}"
                else:
                    elabel = f"active (p={p:.2g})"
                color = "#27ae60"
                style = "solid"
            else:
                elabel = f"inactive (p={p:.2g})"
                color = "#c0392b"
                style = "dashed"
            lines.append(
                f'  n{nid} -> n{child["id"]} '
                f'[label="{elabel}" color="{color}" fontcolor="{color}" '
                f'style={style} penwidth=1.5];'
            )

    add_node(tree)
    lines.append("}")
    return "\n".join(lines)


def count_nodes(tree):
    if tree["type"] == "leaf" or tree["type"] == "compound_leaf":
        return 1
    return 1 + sum(count_nodes(b["child"]) for b in tree["children"])


def main():
    parser = argparse.ArgumentParser(
        description="Visualize optimal adaptive policy as a decision tree."
    )
    parser.add_argument("file", help="JSON input file")
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output file (default: open in browser)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="svg",
        choices=["svg", "pdf", "png", "dot"],
        help="Output format (default: svg)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Collapse deterministic (p=1) chains into single nodes",
    )

    args = parser.parse_args()

    n, dist, prob = load_instance(args.file)
    nc = n - 1
    print(f"Instance: {args.file}")
    print(f"  n={n} vertices (depot + {nc} customers)")
    stochastic = [i + 1 for i in range(nc) if 0 < prob[i + 1] < 1]
    print(f"  Stochastic customers: {stochastic}")
    print(f"  Max tree leaves: {2**len(stochastic)}")

    cost, F, policy = solve_adaptive_with_policy(n, dist, prob)
    print(f"  Adaptive expected cost: {cost:.6f}")

    tree = build_tree(nc, dist, prob, F, policy)
    if args.compact:
        tree = compact_tree(tree)

    nn = count_nodes(tree)
    print(f"  Decision tree nodes: {nn}")

    basename = os.path.splitext(os.path.basename(args.file))[0]
    title = f"Adaptive Policy: {basename}  (E[cost]={cost:.4f})"
    dot_str = tree_to_dot(tree, nc, prob, title=title)

    if args.format == "dot":
        if args.output:
            with open(args.output, "w") as f:
                f.write(dot_str)
            print(f"Saved DOT to {args.output}")
        else:
            print(dot_str)
        return

    # Render with graphviz
    try:
        result = subprocess.run(
            ["dot", f"-T{args.format}"],
            input=dot_str,
            capture_output=True,
            text=(args.format == "svg"),
            timeout=30,
        )
        if result.returncode != 0:
            print(f"Graphviz error: {result.stderr}", file=sys.stderr)
            sys.exit(1)
    except FileNotFoundError:
        print("graphviz 'dot' not found. Install it or use --format dot", file=sys.stderr)
        sys.exit(1)

    if args.output:
        mode = "w" if args.format == "svg" else "wb"
        with open(args.output, mode) as f:
            f.write(result.stdout)
        print(f"Saved to {args.output}")
    else:
        ext = args.format
        fd, tmp_path = tempfile.mkstemp(suffix=f".{ext}", prefix="policy_tree_")
        mode = "w" if args.format == "svg" else "wb"
        with os.fdopen(fd, mode) as f:
            f.write(result.stdout)
        webbrowser.open(f"file://{tmp_path}")
        print(f"Opened in browser ({tmp_path})")


if __name__ == "__main__":
    main()
