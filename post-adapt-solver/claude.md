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
| `gen_two_circles.py` | Generates two-circle graph instances |
| `examples/` | JSON instance files. All example instances go here. |

**Rule:** extend existing files before creating new ones.

## Build & run

The project uses **CMake**. All executables are defined in `CMakeLists.txt`. Do not use raw `g++`/`clang++`.

```bash
cmake -B build && cmake --build build
./build/solver input.json [--no-apriori]   # --no-apriori auto-applied when nc > 10
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

## Algorithms (for context)

- **A posteriori**: Held-Karp DP computes `OPT(mask)` for all subsets, then `E[OPT(A)]`. Time: `O(n^2 * 2^n)`.
- **Adaptive**: DP on `F(pos, probed_mask)`. Time: `O(n^2 * 2^n)`.
- **A priori**: Brute-force all `n!` permutations. Only feasible for small `n` (~10).

## Key modeling rules

1. Adaptive probing forces immediate travel to active customers — no postponing.
2. Adaptive state is `(current_position, probed_mask)` — no need to track which probed customers were active.
3. Use one Held-Karp pass for all subsets, not a separate TSP per realization.
4. All tours are closed: start at depot 0, visit customers, return to depot 0.
