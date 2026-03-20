#!/usr/bin/env python3
"""
Generate a two-circle graph for the stochastic TSP solver.

Graph structure:
  - Circle A: vertices 0, 1, ..., k-1   (vertex 0 = depot)
  - Circle B: vertices k, k+1, ..., 2k-1  (all customers)
  - Within each circle: adjacent vertices connected at cost 1 (cyclic, undirected)
  - Cross edges: vertex i in A <-> vertex i+k in B at specified cost

  Circle A:   0 -- 1 -- 2 -- ... -- (k-1) -- 0
              |    |    |              |
  Circle B:   k -- k+1 k+2 ... -- (2k-1) -- k

Total vertices: 2k  (depot = 0, customers = 1..2k-1, so n = 2k-1)

Usage examples:
  python gen_two_circles.py --k 4 --cross-cost 3.0 --prob 0.5
  python gen_two_circles.py --k 3 --cross-cost 2.0 --prob-a 0.8 --prob-b 0.3 -o out.json
  python gen_two_circles.py --k 5 --cross-costs 1 2 3 4 5 --prob 0.5
"""

import argparse
import json
import sys


def generate(k, cross_costs, prob_a, prob_b):
    """
    k           : number of vertices per circle (>= 2)
    cross_costs : list of k floats, cross_costs[i] = cost of edge (i, i+k)
    prob_a      : activation probability for circle A customers (vertices 1..k-1)
    prob_b      : activation probability for circle B customers (vertices k..2k-1)

    Returns a dict ready to be serialised as JSON for the solver.
    """
    n = 2 * k - 1   # number of customers (vertex 0 is depot)
    V = 2 * k       # total vertices

    edges = []

    # Circle A (vertices 0..k-1, cyclic)
    for i in range(k):
        j = (i + 1) % k
        edges.append({"from": i, "to": j, "weight": 1.0})
        edges.append({"from": j, "to": i, "weight": 1.0})

    # Circle B (vertices k..2k-1, cyclic)
    for i in range(k):
        vi = k + i
        vj = k + (i + 1) % k
        edges.append({"from": vi, "to": vj, "weight": 1.0})
        edges.append({"from": vj, "to": vi, "weight": 1.0})

    # Cross edges: vertex i <-> vertex i+k
    for i in range(k):
        c = cross_costs[i]
        edges.append({"from": i,     "to": k + i, "weight": c})
        edges.append({"from": k + i, "to": i,     "weight": c})

    # Probabilities: p[0]=0 (depot never active), p[1..k-1]=prob_a, p[k..2k-1]=prob_b
    prob = [0.0] * V
    for i in range(1, k):
        prob[i] = prob_a
    for i in range(k, V):
        prob[i] = prob_b

    return {
        "n": n,
        "edges": edges,
        "prob": prob,
        # metadata (ignored by solver, useful for humans)
        "_comment": (
            f"Two-circle graph: k={k}, cross_costs={cross_costs}, "
            f"prob_a={prob_a}, prob_b={prob_b}. "
            f"Circle A = vertices 0..{k-1} (depot=0), "
            f"Circle B = vertices {k}..{2*k-1}."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a two-circle graph for the stochastic TSP solver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--k", type=int, required=True,
                        help="Number of vertices per circle (>= 2)")

    cross_group = parser.add_mutually_exclusive_group(required=True)
    cross_group.add_argument("--cross-cost", type=float, metavar="C",
                             help="Uniform cross-edge cost for all k pairs")
    cross_group.add_argument("--cross-costs", type=float, nargs="+", metavar="C",
                             help="Per-pair cross-edge costs: k values for pairs (0,k), (1,k+1), ...")

    prob_group = parser.add_mutually_exclusive_group()
    prob_group.add_argument("--prob", type=float, metavar="P",
                            help="Uniform activation probability for all customers")
    prob_group.add_argument("--prob-ab", type=float, nargs=2, metavar=("PA", "PB"),
                            help="Separate probabilities: prob_a prob_b")

    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file path (default: stdout)")

    args = parser.parse_args()

    k = args.k
    if k < 2:
        print("Error: k must be at least 2", file=sys.stderr)
        sys.exit(1)

    # Cross-edge costs
    if args.cross_cost is not None:
        cross_costs = [args.cross_cost] * k
    else:
        if len(args.cross_costs) != k:
            print(f"Error: --cross-costs requires exactly k={k} values, "
                  f"got {len(args.cross_costs)}", file=sys.stderr)
            sys.exit(1)
        cross_costs = args.cross_costs

    # Activation probabilities
    if args.prob is not None:
        prob_a = prob_b = args.prob
    elif args.prob_ab is not None:
        prob_a, prob_b = args.prob_ab
    else:
        prob_a = prob_b = 0.5   # default

    for p in [prob_a, prob_b]:
        if not (0.0 <= p <= 1.0):
            print(f"Error: probabilities must be in [0, 1]", file=sys.stderr)
            sys.exit(1)

    data = generate(k, cross_costs, prob_a, prob_b)
    out = json.dumps(data, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(out + "\n")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
