# post-adapt-solver

Exact solver for **a posteriori**, **adaptive**, and **a priori** stochastic TSP.

## Build

```bash
cmake -B build && cmake --build build
```

Requires C++17 and CMake ≥ 3.10.

## `solver` — solve a single instance

```
./build/solver input.json [--no-apriori]
```

- `--no-apriori` — skip a priori computation (auto-applied when number of customers > 10).

Prints a posteriori, adaptive, and a priori expected costs plus their pairwise ratios.

### JSON input formats

**Full distance matrix:**
```json
{
  "n": 4,
  "dist": [[0,1,2,3],[1,0,2,3],[2,2,0,1],[3,3,1,0]],
  "prob": [1.0, 0.5, 0.5, 0.5]
}
```

**Edge list** (Floyd–Warshall computes all-pairs shortest paths):
```json
{
  "n": 4,
  "edges": [{"s":0,"t":1,"w":2}, {"s":1,"t":2,"w":3}],
  "prob": [1.0, 0.5, 0.5, 0.5],
  "sym": true
}
```

| Field | Description |
|-------|-------------|
| `"n"` | Total vertices **including** depot (depot = 0, customers = 1..n-1) |
| `"prob"` | Length-n array; `prob[0]` = 1.0 (depot always present), `prob[i]` = activation probability of customer `i` |
| `"dist"` | n×n distance matrix (mutually exclusive with `"edges"`) |
| `"edges"` | List of `{"s", "t", "w"}` objects; `"w"` defaults to 1 if omitted |
| `"sym"` | If present, edges are treated as bidirectional |

**Run on an example:**
```bash
./build/solver examples/Kn-HamiltonianCycle-n=5.json
```

---

## `search_ratio` — random search for large-ratio instances

Randomly samples instances and tracks the one with the highest chosen ratio.

```
./build/search_ratio [options]
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--sym` | off (asymmetric) | Use symmetric distances |
| `--ratio TYPE` | `adapt/apost` | Ratio to maximise: `adapt/apost`, `apriori/adapt`, `apriori/apost` |
| `-n N` | `7` | Max total vertices including depot |
| `-d D` | `20` | Max integer edge distance |
| `-i ITERS` | `2000000` | Number of random trials |
| `-s SEED` | `42` | RNG seed |
| `--probs P` | `0.5,1.0` | Comma-separated probabilities to sample for customers |
| `-o FILE` | `examples/best_ratio_found.json` | Output file for best instance found |

If the ratio involves a priori and `-n > 9`, n is automatically capped at 9.

**Examples:**
```bash
# Symmetric distances, maximise adaptive / a_posteriori, up to 7 vertices
./build/search_ratio --sym --ratio adapt/apost -n 7 -d 10 -o examples/sym_gap.json

# Asymmetric, maximise a_priori / a_posteriori, custom probabilities, 500k trials
./build/search_ratio --ratio apriori/apost -n 5 --probs 0.3,0.5,0.7 -i 500000

# Search for a_priori / adaptive gap with a fixed seed for reproducibility
./build/search_ratio --ratio apriori/adapt -n 8 -s 123 -i 1000000
```

---

## Other tools

| File | Purpose |
|------|---------|
| `test_correctness` | Correctness tests: DP vs brute-force on small random instances |
| `visualize.py` | Generate interactive HTML graph visualisation (vis.js) |
| `gen_two_circles.py` | Generate a two-circle graph instance as JSON |
| `examples/` | Example JSON instance files |
