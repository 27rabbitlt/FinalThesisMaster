#!/usr/bin/env python3
"""
Continuous optimization to maximize adapt/apost ratio.

Parameterizes the distance matrix (upper triangle) and probabilities,
calls the C++ solver for exact evaluation, and uses scipy to maximize.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import numpy as np

SOLVER = os.path.join(os.path.dirname(__file__), "build", "solver")
INF = 1e18


def floyd_warshall(d):
    n = len(d)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]


def call_solver(n, dist, prob):
    """Call C++ solver and return (apost, adapt) costs."""
    instance = {
        "n": n,
        "dist": [[round(float(dist[i][j]), 6) for j in range(n)] for i in range(n)],
        "prob": [round(float(p), 6) for p in prob],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(instance, f)
        fname = f.name
    try:
        result = subprocess.run(
            [SOLVER, fname, "--no-apriori"],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout
        apost = adapt = None
        for line in output.splitlines():
            m = re.search(r"A posteriori expected cost:\s+([\d.]+)", line)
            if m:
                apost = float(m.group(1))
            m = re.search(r"Adaptive expected cost:\s+([\d.]+)", line)
            if m:
                adapt = float(m.group(1))
        return apost, adapt
    except Exception as e:
        return None, None
    finally:
        os.unlink(fname)


def params_to_instance(x, n, optimize_prob=True):
    """Convert parameter vector to (dist, prob)."""
    nc = n - 1
    n_pairs = n * (n - 1) // 2

    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0.0
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = max(float(x[idx]), 0.01)
            dist[i][j] = w
            dist[j][i] = w
            idx += 1

    floyd_warshall(dist)

    prob = [1.0] * n
    if optimize_prob:
        for i in range(nc):
            prob[i + 1] = float(np.clip(x[n_pairs + i], 0.01, 1.0))

    return dist, prob


def objective(x, n, optimize_prob=True):
    """Negative ratio (for minimization)."""
    dist, prob = params_to_instance(x, n, optimize_prob)
    apost, adapt = call_solver(n, dist, prob)
    if apost is None or apost < 1e-12:
        return 0.0
    return -(adapt / apost)


def run_optimization(n, d_max, n_restarts, optimize_prob=True, seed=42):
    from scipy.optimize import minimize, differential_evolution

    nc = n - 1
    n_pairs = n * (n - 1) // 2
    n_prob = nc if optimize_prob else 0
    ndim = n_pairs + n_prob

    rng = np.random.RandomState(seed)
    best_ratio = 1.0
    best_dist = None
    best_prob = None

    bounds = [(0.1, d_max)] * n_pairs
    if optimize_prob:
        bounds += [(0.01, 1.0)] * nc

    print(f"Optimizing: n={n}, {ndim} variables, {n_restarts} restarts")
    print(f"  {n_pairs} distance vars, {n_prob} probability vars")

    t0 = time.time()
    evals = [0]

    def obj_wrapper(x):
        evals[0] += 1
        return objective(x, n, optimize_prob)

    # Multi-start Nelder-Mead
    for restart in range(n_restarts):
        x0 = np.concatenate([
            rng.uniform(0.5, d_max, n_pairs),
            rng.uniform(0.1, 1.0, n_prob) if optimize_prob else np.array([])
        ])

        try:
            res = minimize(obj_wrapper, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7})
            ratio = -res.fun
        except Exception:
            continue

        if ratio > best_ratio + 1e-9:
            best_ratio = ratio
            dist, prob = params_to_instance(res.x, n, optimize_prob)
            best_dist = [row[:] for row in dist]
            best_prob = prob[:]
            elapsed = time.time() - t0
            print(f"  restart {restart}: ratio = {ratio:.6f}  ({evals[0]} evals, {elapsed:.1f}s)")

        if restart % 50 == 0 and restart > 0:
            elapsed = time.time() - t0
            print(f"  [{restart}/{n_restarts}] best = {best_ratio:.6f}  ({evals[0]} evals, {elapsed:.1f}s)")

    # Differential evolution for global search
    print(f"\nDifferential evolution (popsize=15, maxiter=200)...")
    try:
        res_de = differential_evolution(
            obj_wrapper, bounds, maxiter=200, seed=seed, tol=1e-7,
            mutation=(0.5, 1.5), recombination=0.9, popsize=15, polish=True
        )
        ratio_de = -res_de.fun
        if ratio_de > best_ratio + 1e-9:
            best_ratio = ratio_de
            dist, prob = params_to_instance(res_de.x, n, optimize_prob)
            best_dist = [row[:] for row in dist]
            best_prob = prob[:]
            print(f"  DE improved: ratio = {ratio_de:.6f}")
        else:
            print(f"  DE best = {ratio_de:.6f} (no improvement)")
    except Exception as e:
        print(f"  DE failed: {e}")

    return best_ratio, best_dist, best_prob


def main():
    parser = argparse.ArgumentParser(description="Optimize adapt/apost ratio")
    parser.add_argument("-n", type=int, default=7,
                        help="Total vertices including depot (default: 7)")
    parser.add_argument("-d", "--d-max", type=float, default=10,
                        help="Max edge distance (default: 10)")
    parser.add_argument("-r", "--restarts", type=int, default=200,
                        help="Number of Nelder-Mead restarts (default: 200)")
    parser.add_argument("--fix-prob", action="store_true",
                        help="Fix all probabilities to 0.5")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    optimize_prob = not args.fix_prob
    ratio, dist, prob = run_optimization(
        args.n, args.d_max, args.restarts, optimize_prob, args.seed)

    print(f"\n=== BEST RESULT ===")
    print(f"n = {args.n}, ratio = {ratio:.6f}")

    if dist is not None:
        apost, adapt = call_solver(args.n, dist, prob)
        print(f"  a_posteriori = {apost:.6f}")
        print(f"  adaptive     = {adapt:.6f}")
        print(f"  prob = {[f'{p:.4f}' for p in prob]}")
        print(f"\n  Distance matrix:")
        for i in range(args.n):
            print(f"    {' '.join(f'{dist[i][j]:7.3f}' for j in range(args.n))}")

        out = args.output or f"examples/optimized_n{args.n}.json"
        result = {
            "_comment": f"Optimized adapt/apost ratio = {ratio:.6f}",
            "n": args.n,
            "dist": [[round(dist[i][j], 4) for j in range(args.n)]
                     for i in range(args.n)],
            "prob": [round(p, 4) for p in prob],
        }
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
