#!/usr/bin/env python3
"""Solve the a priori TSP LP relaxation from lp.tex using SciPy + HiGHS.

Input format (JSON):
{
  "vertices": ["a0", "a1", ...],
  "p": 1.0,
  "d_from_depot": {"a0": 1.0, "a1": 1.0, ...},
  "d_between_vertices": {
    "a0": {"a0": 0.0, "a1": 0.1, ...},
    ...
  },
  "include_redundant_constraints": false
}

The depot is implicit: only d(r,v) and d(u,v) for u,v in V are needed by the LP.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class Instance:
    vertices: List[str]
    p: float
    d_from_depot: Dict[str, float]
    d_between_vertices: Dict[str, Dict[str, float]]
    include_redundant_constraints: bool = False


class LPBuilder:
    def __init__(self, instance: Instance) -> None:
        self.inst = instance
        self.vertices = instance.vertices
        self.n = len(self.vertices)
        self.pos = list(range(self.n))
        self.x_idx: Dict[Tuple[int, int], int] = {}
        self.y_idx: Dict[Tuple[int, int, int, int], int] = {}
        self.t_idx: Dict[int, int] = {}
        self.var_names: List[str] = []
        self.bounds: List[Tuple[float | None, float | None]] = []
        self._build_variable_index()

    def _add_var(self, name: str, lb: float | None, ub: float | None) -> int:
        idx = len(self.var_names)
        self.var_names.append(name)
        self.bounds.append((lb, ub))
        return idx

    def _build_variable_index(self) -> None:
        n = self.n
        for v in range(n):
            for i in range(n):
                self.x_idx[(v, i)] = self._add_var(f"x[{self.vertices[v]},{i+1}]", 0.0, 1.0)
        for v in range(n):
            for i in range(n):
                for u in range(n):
                    if u == v:
                        continue
                    for j in range(n):
                        if j == i:
                            continue
                        self.y_idx[(v, i, u, j)] = self._add_var(
                            f"y[{self.vertices[v]},{i+1},{self.vertices[u]},{j+1}]", 0.0, 1.0
                        )
        for i in range(n + 1):
            self.t_idx[i] = self._add_var(f"t[{i}]", 0.0, 0.0 if i == 0 else None)

    def d_r(self, v: int) -> float:
        return float(self.inst.d_from_depot[self.vertices[v]])

    def d_uv(self, u: int, v: int) -> float:
        return float(self.inst.d_between_vertices[self.vertices[u]][self.vertices[v]])

    def validate(self) -> None:
        n = self.n
        if n == 0:
            raise ValueError("vertices must be nonempty")
        p = self.inst.p
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must lie in [0,1], got {p}")

        V = set(self.vertices)
        if len(V) != n:
            raise ValueError("vertex names must be distinct")
        if set(self.inst.d_from_depot.keys()) != V:
            raise ValueError("d_from_depot must contain exactly the vertices")
        if set(self.inst.d_between_vertices.keys()) != V:
            raise ValueError("d_between_vertices must contain exactly the vertices")

        for v in self.vertices:
            row = self.inst.d_between_vertices[v]
            if set(row.keys()) != V:
                raise ValueError(f"row d_between_vertices[{v}] must contain exactly the vertices")
            if row[v] != 0:
                raise ValueError(f"d_between_vertices[{v}][{v}] must be 0")
            for u in self.vertices:
                if row[u] < 0:
                    raise ValueError("distances must be nonnegative")
                if not math.isclose(row[u], self.inst.d_between_vertices[u][v], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f"distance matrix must be symmetric: mismatch at ({v},{u})")
        for v in self.vertices:
            if self.inst.d_from_depot[v] < 0:
                raise ValueError("d_from_depot must be nonnegative")

    def build(self):
        self.validate()
        n = self.n
        p = self.inst.p
        include_redundant = self.inst.include_redundant_constraints

        # Count rows first.
        num_rows = 0
        num_rows += n  # assign-vertex
        num_rows += n  # assign-position
        num_rows += (n * (n - 1) * n * (n - 1)) // 2  # symmetry, only one from each pair
        num_rows += n * n * (n - 1)  # for-u
        num_rows += n * (n - 1) * n  # for-j
        if include_redundant:
            num_rows += n * (n - 1)  # pair-total-uv
            num_rows += n * (n - 1)  # pair-total-ij
        num_rows += 1  # t0
        num_rows += n  # recurrence

        A_eq = lil_matrix((num_rows, len(self.var_names)), dtype=float)
        b_eq = np.zeros(num_rows, dtype=float)
        row = 0

        # (1) sum_i x[v,i] = 1 for each v.
        for v in range(n):
            for i in range(n):
                A_eq[row, self.x_idx[(v, i)]] = 1.0
            b_eq[row] = 1.0
            row += 1

        # (2) sum_v x[v,i] = 1 for each i.
        for i in range(n):
            for v in range(n):
                A_eq[row, self.x_idx[(v, i)]] = 1.0
            b_eq[row] = 1.0
            row += 1

        # (3) y[v,i,u,j] = y[u,j,v,i], one row per unordered pair.
        for v in range(n):
            for i in range(n):
                for u in range(n):
                    if u == v:
                        continue
                    for j in range(n):
                        if j == i:
                            continue
                        left = (v, i, u, j)
                        right = (u, j, v, i)
                        if left < right:
                            A_eq[row, self.y_idx[left]] = 1.0
                            A_eq[row, self.y_idx[right]] = -1.0
                            b_eq[row] = 0.0
                            row += 1

        # (4) sum_{u != v} y[v,i,u,j] = x[v,i] for all v and i != j.
        for v in range(n):
            for i in range(n):
                for j in range(n):
                    if j == i:
                        continue
                    for u in range(n):
                        if u == v:
                            continue
                        A_eq[row, self.y_idx[(v, i, u, j)]] = 1.0
                    A_eq[row, self.x_idx[(v, i)]] = -1.0
                    b_eq[row] = 0.0
                    row += 1

        # (5) sum_{j != i} y[v,i,u,j] = x[v,i] for all v != u and i.
        for v in range(n):
            for u in range(n):
                if u == v:
                    continue
                for i in range(n):
                    for j in range(n):
                        if j == i:
                            continue
                        A_eq[row, self.y_idx[(v, i, u, j)]] = 1.0
                    A_eq[row, self.x_idx[(v, i)]] = -1.0
                    b_eq[row] = 0.0
                    row += 1

        # (6) redundant pair-total constraints, optional.
        if include_redundant:
            for i in range(n):
                for j in range(n):
                    if j == i:
                        continue
                    for v in range(n):
                        for u in range(n):
                            if u == v:
                                continue
                            A_eq[row, self.y_idx[(v, i, u, j)]] = 1.0
                    b_eq[row] = 1.0
                    row += 1
            for v in range(n):
                for u in range(n):
                    if u == v:
                        continue
                    for i in range(n):
                        for j in range(n):
                            if j == i:
                                continue
                            A_eq[row, self.y_idx[(v, i, u, j)]] = 1.0
                    b_eq[row] = 1.0
                    row += 1

        # (7) t0 = 0.
        A_eq[row, self.t_idx[0]] = 1.0
        b_eq[row] = 0.0
        row += 1

        # (8) recurrence for i=1,...,n.
        for i in range(1, n + 1):
            # t_i - t_{i-1}
            A_eq[row, self.t_idx[i]] = 1.0
            A_eq[row, self.t_idx[i - 1]] = -1.0

            # first active at position i
            coeff_x = p * ((1.0 - p) ** (i - 1))
            for v in range(n):
                A_eq[row, self.x_idx[(v, i - 1)]] -= coeff_x * self.d_r(v)

            # predecessor j < i
            for j in range(1, i):
                coeff_y = (p ** 2) * ((1.0 - p) ** (i - j - 1))
                for v in range(n):
                    for u in range(n):
                        if u == v:
                            continue
                        A_eq[row, self.y_idx[(v, i - 1, u, j - 1)]] -= coeff_y * self.d_uv(u, v)

            b_eq[row] = 0.0
            row += 1

        assert row == num_rows, (row, num_rows)

        c = np.zeros(len(self.var_names), dtype=float)
        c[self.t_idx[n]] = 1.0

        return c, A_eq.tocsr(), b_eq, self.bounds

    def unpack_solution(self, xsol: np.ndarray, threshold: float = 1e-8) -> dict:
        n = self.n
        x_out = {
            self.vertices[v]: {str(i + 1): float(xsol[self.x_idx[(v, i)]]) for i in range(n)}
            for v in range(n)
        }
        t_out = {str(i): float(xsol[self.t_idx[i]]) for i in range(n + 1)}
        y_out = {}
        for (v, i, u, j), idx in self.y_idx.items():
            val = float(xsol[idx])
            if abs(val) > threshold:
                key = f"({self.vertices[v]},{i+1};{self.vertices[u]},{j+1})"
                y_out[key] = val
        return {"x": x_out, "t": t_out, "y_nonzero": y_out}


def load_instance(path: Path) -> Instance:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Instance(
        vertices=list(data["vertices"]),
        p=float(data["p"]),
        d_from_depot=dict(data["d_from_depot"]),
        d_between_vertices=dict(data["d_between_vertices"]),
        include_redundant_constraints=bool(data.get("include_redundant_constraints", False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path, help="Path to JSON instance file")
    parser.add_argument("--method", default="highs", choices=["highs", "highs-ds", "highs-ipm"], help="HiGHS method")
    parser.add_argument("--write-solution", type=Path, default=None, help="Write full solution as JSON")
    parser.add_argument("--threshold", type=float, default=1e-8, help="Only y values above this threshold are written")
    args = parser.parse_args()

    inst = load_instance(args.instance)
    builder = LPBuilder(inst)
    c, A_eq, b_eq, bounds = builder.build()

    print(f"Variables: {len(c)}")
    print(f"Equality constraints: {A_eq.shape[0]}")
    print(f"Nonzeros in A_eq: {A_eq.nnz}")
    print(f"Method: {args.method}")

    res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=args.method)

    print("status:", res.status)
    print("message:", res.message)
    if not res.success:
        raise SystemExit(1)

    print(f"objective t_n = {res.fun:.12g}")
    unpacked = builder.unpack_solution(res.x, threshold=args.threshold)

    print("\nt values:")
    for i, val in unpacked["t"].items():
        print(f"  t[{i}] = {val:.12g}")

    print("\nx matrix:")
    for v in inst.vertices:
        row = " ".join(f"{unpacked['x'][v][str(i+1)]:.6f}" for i in range(len(inst.vertices)))
        print(f"  {v}: {row}")

    if args.write_solution is not None:
        args.write_solution.write_text(json.dumps(unpacked, indent=2), encoding="utf-8")
        print(f"\nWrote solution to {args.write_solution}")


if __name__ == "__main__":
    main()
