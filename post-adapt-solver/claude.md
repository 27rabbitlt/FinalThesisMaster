# CLAUDE.md — Stochastic TSP Solver

Exact solver for **a posteriori**, **adaptive**, and **a priori** stochastic TSP on small instances. The solver is already fully implemented and working. See the thesis draft `main.pdf` in the parent folder for formal definitions.

## Quick summary of the three objectives

- **A posteriori**: knows the full active set, then solves optimal TSP for that realization. Strongest benchmark.
- **Adaptive**: probes customers one by one; if active, **must move there immediately**. State = `(position, probed_mask)`.
- **A priori**: fixes a master tour before seeing any realization; skips inactive vertices (shortcutting). Brute-force over all permutations.

Ordering: `a_posteriori <= adaptive <= a priori`.

## Files

| File | Purpose |
|------|---------|
| `tsp_solver.h` | All solver logic (Held-Karp DP for a posteriori, adaptive DP, a priori brute-force) |
| `solver.cpp` | CLI: reads JSON, calls solvers, prints results |
| `search_ratio.cpp` | Random sampler to maximise a chosen ratio (e.g. `adapt/apost`) |
| `visualize.py` | Interactive HTML graph visualisation (vis.js) |
| `visualize_policy.py` | Decision tree visualisation of optimal adaptive policy (Graphviz) |
| `gen_two_circles.py` | Generates two-circle graph instances |
| `gen-asym-manual-recursion.py` | Generates recursive asymmetric 3-cycle graphs (4/3 ratio construction) |
| `examples/` | JSON instance files. All example instances go here. |

**Rule:** extend existing files before creating new ones.

## Build & run

The project uses **CMake**. All executables are defined in `CMakeLists.txt`. Do not use raw `g++`/`clang++`.

```bash
cmake -B build && cmake --build build
./build/solver input.json [--no-apriori] [--sample N]
  # --no-apriori  auto-applied when nc > 10
  # --sample N    Monte Carlo: sample N realizations (large instances only)
./build/search_ratio [options]              # see search_ratio.cpp for flags
```

## JSON input format

Two input modes, both using shortest-path (Floyd-Warshall) distances as the metric.

### Mode 1: Edge list (preferred for structured graphs)

```json
{
  "n": 8,
  "edges": [{"s": 0, "t": 1, "w": 1}, {"s": 1, "t": 2, "w": 3}],
  "prob": [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
  "sym": true
}
```

### Mode 2: Full distance matrix

```json
{
  "n": 4,
  "dist": [[0,1,2,3],[1,0,2,3],[2,2,0,1],[3,3,1,0]],
  "prob": [1.0, 0.5, 0.5, 0.5]
}
```

### Field reference

| Field | Required | Description |
|-------|----------|-------------|
| `n` | yes | Total vertices **including** depot (depot = vertex 0) |
| `edges` | one of `edges`/`dist` | Edge list. Each edge: `{"s": src, "t": dst, "w": weight}`. `"w"` defaults to 1 if omitted. |
| `dist` | one of `edges`/`dist` | Full `n x n` distance matrix |
| `prob` | yes | Activation probabilities. `prob[0]` = 1.0 (depot always active). `prob[i]` = probability customer `i` is active. |
| `sym` | no | If `true`, edges are bidirectional. Default: `false` (directed). |
| `_comment` | no | Optional human-readable description, ignored by solver. |
| `"c"` on edges | no | Optional edge color (hex string) for visualisation only. |

### Conventions

- Depot is vertex `0`, always active (`prob[0] = 1.0`).
- Customers are vertices `1..n-1`.
- Distances are the graph-induced metric (shortest-path distances). Directed graphs give asymmetric metrics; undirected give symmetric.
- Missing edges get weight `INF`; Floyd-Warshall computes all-pairs shortest paths.

## Visualising the adaptive policy

`visualize_policy.py` reimplements the adaptive DP in Python, extracts the optimal action at every state, and renders the full decision tree via Graphviz.

```bash
# Open SVG in browser (default)
python visualize_policy.py examples/asym-manuel.json

# Save to file
python visualize_policy.py examples/asym-manuel.json -o policy.svg
python visualize_policy.py examples/asym-manuel.json -o policy.pdf --format pdf

# Collapse deterministic (p=1) chains into single nodes
python visualize_policy.py examples/asym-manuel.json --compact

# Output raw DOT source
python visualize_policy.py examples/asym-manuel.json --format dot
```

Requires Graphviz (`brew install graphviz`) for rendering; `--format dot` works without it.

Node colours: **yellow** = stochastic branch (0 < p < 1), **blue** = deterministic probe (p=1), **red box** = leaf (return to depot). Edge styles: **green solid** = active, **red dashed** = inactive.

## Algorithms (for context)

- **A posteriori**: Held-Karp DP computes `OPT(mask)` for all subsets, then `E[OPT(A)]`. Time: `O(n^2 * 2^n)`.
- **Adaptive**: DP on `F(pos, probed_mask)`. Time: `O(n^2 * 2^n)`.
- **A priori**: Brute-force all `n!` permutations. Only feasible for small `n` (~10).

## Recursive asymmetric construction (4/3 ratio lower bound)

`gen-asym-manual-recursion.py` generates a family of directed graphs that aim to show the adaptive/a posteriori ratio approaches 4/3 as depth grows.

### Construction

At each depth level, we build a directed 3-cycle of vertex groups (L → R → M → L) with edge weights `2^(dep-1)`. Recursion creates the left and right subgroups; the "mid" vertex is a new stochastic node (prob=0.5) added at each level. The depot connects to/from all customer vertices at the top level.

- **Depth 0**: single deterministic vertex (leaf)
- **Depth d**: left subtree (depth d−1) + right subtree (depth d−1) + 1 mid vertex. Directed edges: L→R, R→M, M→L, all with weight `2^(d-1)`.
- Leaf vertices (from depth-0 base cases) have `prob=1` (deterministic). Mid vertices have `prob=0.5` (stochastic).

**Vertex counts**: depth d gives `2^(d+1) - 1` customers. `2^d` deterministic (leaves), `2^d - 1` stochastic (mid nodes).

| depth | n (incl depot) | customers | stochastic | deterministic | verified ratio |
|-------|----------------|-----------|------------|---------------|----------------|
| 1 | 4 | 3 | 1 | 2 | — |
| 2 | 8 | 7 | 3 | 4 | — |
| 3 | 16 | 15 | 7 | 8 | 1.2308 (=32/26) |
| 4 | 32 | 31 | 15 | 16 | (computing, ~34.5h for 70%) |

### Why it's hard to compute

For depth 4 (n=32), the a posteriori solver enumerates 2^15 = 32768 realizations. Each realization has 17–32 active vertices requiring an exact TSP solve. Held-Karp is used for nc ≤ 25 (~6.4 GB memory), B&B for larger. Benchmarked times on 8-core/16GB Mac:

- k=0..8 (nc ≤ 24, 22819 realizations): **~34.5 hours**
- k=9 (nc=25, 5005 realizations): **~37 hours additional**
- k=10+ (nc ≥ 26, 4944 realizations): B&B, unpredictable

### Running

```bash
python gen-asym-manual-recursion.py > examples/asym-manual-dep4.json
nohup ./build/solver examples/asym-manual-dep4.json --no-apriori > depth4_results.txt 2>&1 &
```

The solver processes realizations sorted by popcount (smallest first), printing partial E[OPT] after each k-group. The adaptive upper bound (probe-first policy) computes in seconds.

## Key modeling rules

1. Adaptive probing forces immediate travel to active customers — no postponing.
2. Adaptive state is `(current_position, probed_mask)` — no need to track which probed customers were active.
3. Use one Held-Karp pass for all subsets, not a separate TSP per realization.
4. All tours are closed: start at depot 0, visit customers, return to depot 0.
