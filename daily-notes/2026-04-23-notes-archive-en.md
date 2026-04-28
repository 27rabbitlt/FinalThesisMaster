# Research Notes Archive (moved from thesis/main.tex Notes section)

These notes were previously in the thesis as Section "Notes", removed on 2026-04-23 to keep the thesis folder clean. The original tex files were:
- `relevant-def.tex` — Related problem definitions (stochastic probing, stochastic k-TSP)
- `stochastic-knapsack-notes.tex` — Paper notes on stochastic knapsack (DGV 2008, Barak-Talgam-Cohen 2026)
- `adaptive-decision-tree-notes.tex` — Paper notes on Gupta-Nagarajan-Ravi 2017 (Optimal Decision Trees, Adaptive TSP)
- `results-collection.tex` — Collection of all known results on stochastic TSP variants

The original `.tex` files are preserved below for reference.

---

## Related Problem Definitions

### Stochastic Probing

The stochastic probing framework (Gupta-Nagarajan 2013) provides a unifying abstraction for stochastic combinatorial optimization. An instance has a ground set E, values, activation probabilities, inner constraints (selected elements), and outer constraints (probed elements). A probing policy sequentially selects elements; active ones are irrevocably included.

Key results:
- Matroid inner + matroid outer, submodular objectives: adaptivity gap O(1)
- Matroid inner + prefix-closed outer: adaptivity gap O(log n)
- Single matroid inner, submodular: adaptivity gap = 2

Remark: It seems difficult to model TSP tour cost via probing and summing values.

### Stochastic k-TSP

Salesperson must collect reward ≥ k by visiting vertices with stochastic rewards, minimizing expected travel cost.
- Ene et al: adaptive O(log k)-approximation, adaptivity gap ≥ e
- Jiang et al: adaptivity gap O(1), non-adaptive O(1)-approximation

### Comparison table

| | A Posteriori | Adaptive TSP | Stoch. Probing | Stoch. k-TSP |
|---|---|---|---|---|
| Uncertainty | active set | active set | active set | reward/cost |
| Information | full A revealed | one-by-one | one-by-one | upon visit |
| Forced action | none | immediate visit | irrevocable select | none |
| Objective | min E[tour] | min E[tour] | max E[value] | min E[cost] |
| Stopping | visit all active | probe all | constraints | reward ≥ k |

---

## Stochastic Knapsack Notes

### DGV (2008)

Stochastic 0/1 knapsack with n items, deterministic values, random sizes, capacity 1. Adaptivity gap ≤ 4 for Non-Risky SK. Key tools: effective value, mean truncated size, martingale argument (E[μ(A)] ≤ 2), LP relaxation.

Main results:
1. 32/7-approximation nonadaptive (≈4.57)
2. 4-approximation nonadaptive (polymatroid LP)
3. (3+ε)-approximation adaptive
4. Adaptivity gap ≤ 4 for Non-Risky SK

### Barak and Talgam-Cohen (2026)

Same framework, introduces semi-adaptivity (k adaptive queries). Key results:
- Risky-SK: full adaptivity gap upper bound 8.47 (improved from 9.5), lower bound 2
- Non-Risky-SK: lower bound 1.37
- 0-1 semi-adaptivity gap = 1 + ln 2 ≈ 1.69 (exact, for Risky-SK)
- With O~(1/ε) queries: 6.44 + √ε approximation

---

## Adaptive Decision Tree Notes (GNR 2017)

Paper studies Optimal Decision Tree, Adaptive TSP (visit-to-reveal model), and Adaptive Traveling Repairman.

Results:
- ODT: O(log m) approximation, tight
- Adaptive TSP (tree metric): O(log²n · log log n)
- Hardness: Ω(log^{2-ε} n) on trees

Key techniques: Isolation problem, Group Steiner Orienteering, scenario-halving.

Their Adaptive TSP model differs from ours: activation revealed upon visiting (not calling). Our model gives strictly more information, making it no harder.

---

## Collection of Existing Results on Stochastic TSP

### Symmetric Metrics

Gap bounds:
- Obliviousness (prior/post): ≤ 3 (Shmoys-Talwar)
- On tree metrics: all gaps = 1 (Schalekamp-Shmoys)

Approximation algorithms for a priori TSP:
- Sampling + MST doubling: 4-approx (Shmoys-Talwar)
- CFL + derandomization: 8-approx deterministic
- Sampling with σ=0.663, α=1.5: <3.1 (Blauth et al)
- Master route + Karlin-Klein-OG: 5.9 deterministic (Blauth et al)
- Tree embedding: O(log n) distribution-free (Schalekamp-Shmoys)

### Asymmetric Metrics

Gap bounds:
- Obliviousness: Ω(n^{1/4}/log n) to O(√n) (Christalla 2025)
- Clairvoyance: O(√n), **open if O(1)**

Approximation:
- High/low probability split: O(√n) vs OPT_post (CPT)
- Hop-ATSP reduction: O(log⁸n) vs OPT_prior, quasi-polynomial time (CPT)

### Open Problems

1. Clairvoyance gap for asymmetric metrics: is OPT_adapt/OPT_post = O(1)?
2. Tightening asymmetric obliviousness gap
3. Symmetric gap lower bounds: any instance with prior/post > 1+ε?
4. Polynomial-time algorithm below obliviousness gap (asymmetric)
