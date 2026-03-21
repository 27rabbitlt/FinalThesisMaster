# CLAUDE.md — Background for exact `a posteriori` vs `adaptive` stochastic TSP solver

This folder is meant to assist my research on Stochastic TSP using code to generate examples and intuition. We need a **simple exact program** that takes a graph with distances and activation probabilities as input and outputs:

1. the optimal **a posteriori TSP** expected cost;
2. the optimal **adaptive TSP** expected cost.
2. the optimal **a priori TSP** expected cost.

The definitions below follow the thesis draft in `main.pdf` (this pdf is in the parent folder but you do not need to read unless you feel necessary): `a posteriori` means “after the active set is known, solve the best tour for that realization”; `adaptive` means “probe vertices one by one, and if a probed vertex is active you must move there immediately.” `a priori` means you have to give a master tour (going from a depot, iterating over all vertices and go back to a depot) before you know anything about the realization.

Note that TSP are considered on metrics, here we are only considering the graph induced metric, i.e., the metric is defined by the shortest path distance between two vertices. It can be proved that shortest path distance is a metric. And a directed graph induces a asymmetric metric, while undirected graph induces a symmetric metric. 

---

## 1. Problem setting and conventions

Use the following conventions in code.

- There is a distinguished **depot** `r`.
- There are `n` customer vertices, indexed `1..n`.
- The depot is indexed `0`.
- Only customers can be active.
- Customer `i` is active independently with probability `p[i]`.
- Distances are given by a matrix `d[u][v]` for all vertices `u, v in {0,1,...,n}`.
- The program should support **directed / asymmetric** distances and undirected / symmetric distances. This doesnt matter too much because we can treat the graph to be directed and the user just provide the distance matrix (if the matrix is symmetric then its undirected).
- The program should compute a **closed tour**: start at depot `0`, serve required customers, return to depot `0`.

### Important note on graph representation

- an all-pairs distance matrix directly; or
- an edge list

---

## 2. What “a posteriori TSP” means

Let `A ⊆ {1,...,n}` be the realized active set.

For that realized set, define `OPT(A)` as the cost of the **best deterministic TSP tour** that:
- starts at depot `0`,
- visits every customer in `A`,
- returns to depot `0`.

Then the optimal a posteriori stochastic TSP value is

\[
\mathbb{E}_A[OPT(A)]
= \sum_{A \subseteq [n]} \mu(A) \, OPT(A),
\]

where

\[
\mu(A)=\prod_{i\in A} p_i \prod_{i\notin A}(1-p_i).
\]

### Informal meaning

`a posteriori` has full hindsight: after seeing which customers are active today, it simply solves the best TSP for that realized subset.

This is the strongest benchmark among the stochastic variants discussed in the thesis.

---

## 3. What “adaptive TSP” means in this thesis

This is **not** the same as “know the active set and then choose the best route”.

Here the policy works as follows:

1. Initially you are at depot `0`.
2. At any step, you choose an **unprobed** customer `i` to probe.
3. When customer `i` is probed:
   - with probability `p[i]`, it is active, and then **you must move to `i` immediately as your next move**;
   - with probability `1-p[i]`, it is inactive, and then **you do not move**.
4. Continue until all customers have been probed.
5. Finally return to depot `0`.

So the adaptive policy is a decision tree / dynamic program that chooses the next vertex to probe using current information.

### Crucial modeling detail

If a probed customer is active, the model forces immediate travel to that customer.
You are **not** allowed to learn that it is active and postpone visiting it until later.

This is the exact modeling choice in the thesis draft and is the key difference between `adaptive` and `a posteriori`.

---

## 4. Exact a posteriori algorithm

A simple exact implementation can compute `OPT(A)` for **all** active subsets `A` using one Held–Karp dynamic program.

### State definition

Let `mask` be a subset of customers `{1,...,n}`.
Let `dp[mask][j]` be the minimum cost to:
- start at depot `0`,
- visit exactly the customers in `mask`,
- end at customer `j`, where `j` is in `mask`.

### Recurrence

Base:

- `dp[1<<(j-1)][j] = d[0][j]`

Transition:

- for `j in mask`,

```text
 dp[mask][j] = min over k in mask, k != j of
               dp[mask without j][k] + d[k][j]
```

Final deterministic TSP cost for subset `mask`:

```text
 OPT(mask) = 0                                 if mask == 0
 OPT(mask) = min over j in mask of dp[mask][j] + d[j][0]
```

### Expected a posteriori value

Once `OPT(mask)` is known for all `mask`, compute

```text
 a_posteriori = sum over masks:
                probability(mask) * OPT(mask)
```

where

```text
 probability(mask) = product over customers i:
                     p[i]     if i in mask
                     1-p[i]   if i not in mask
```

### Time complexity

This is much better than solving a separate TSP from scratch for each realized active set.

---

## 5. Exact adaptive algorithm

The key observation is that under this thesis definition, the exact adaptive state only needs:
- the **current physical position**;
- the set of **already probed customers**.

It does **not** need to remember which already probed customers were active, because:
- if a customer was active, the policy was forced to move there immediately;
- if a customer was inactive, it has no further effect.

So an exact DP exists for general graphs.

### State definition

Let `F(pos, mask)` be the optimal expected remaining cost when:
- you are currently at vertex `pos` (either depot `0` or a customer),
- exactly the customers in `mask` have already been probed.

Here `mask` ranges over subsets of `{1,...,n}`.

### Base case

If all customers have been probed, the only remaining action is to return to the depot:

```text
F(pos, full_mask) = d[pos][0]
```

### Recurrence

From state `(pos, mask)`, choose an unprobed customer `i`.

If `i` is inactive (probability `1-p[i]`):
- you stay at `pos`,
- new state is `(pos, mask U {i})`.

If `i` is active (probability `p[i]`):
- you must move to `i`, paying `d[pos][i]`,
- new state is `(i, mask U {i})`.

Therefore

```text
F(pos, mask) = min over i not in mask of
               (1-p[i]) * F(pos, mask U {i})
             + p[i] * ( d[pos][i] + F(i, mask U {i}) )
```

The answer is

```text
adaptive = F(0, empty_mask)
```

### Why this DP is exact

Because the future only depends on:
- where you currently are,
- which customers remain unprobed.

The independent activation model ensures there is no hidden correlation that would require remembering additional information.

### Time complexity

There are `(n+1) * 2^n` states and each state tries up to `n` next probes.

So:
- time: `O(n^2 2^n)`
- memory: `O((n+1) 2^n)`

This is fully feasible for small `n` in C++.

---

## 6. a priori TSP 

We have to decide a tour before we know the activation. And Once the tour is determined, we cannot change the tour. When we are going along the tour, we shortcut these inactive vertices (for example, if the tour is 1 - 2 - 3 but 2 is inactive, then we skip 2 and directly go from 1 to 3, which is shorter than 1 2 3 because of triangle inequality).

The optimal priori TSP can be basically only calculated by brute force every permutation of vertices.

## 6. Relationship between the three objectives

These three values are **different**, obviously.

And in general

```text
a_posteriori <= adaptive <= a priori
```

The thesis question is about whether these gaps can be large.

---

## 7. Recommended assumptions for the simple exact program

To keep the first program clean and correct, use these assumptions.

### Input assumptions

- `n` customers plus depot `0`
- directed distances `d[u][v]`
- independent probabilities `p[1..n]`
- `p[0] = 0` or depot omitted from the probability list

### Numerical type

Use `double` or `long double`.

### Output

Print:
- `optimal_a_posteriori_expected_cost`
- `optimal_adaptive_expected_cost`
- optionally the ratio `adaptive / a_posteriori`

### Small-instance expectation

Both exact algorithms are exponential in `n`, so this is for **small instances only**.
For a first implementation in C++, something like `n <= 18` or `n <= 20` is the natural target, depending on machine and optimization.

---

## 8. Pseudocode sketch

### A posteriori

```cpp
for (int j = 1; j <= n; ++j)
    dp[1<<(j-1)][j] = d[0][j];

for (mask from 1 to (1<<n)-1) {
    for (j in customers contained in mask) {
        if (mask has only j) continue;
        dp[mask][j] = INF;
        prev = mask ^ (1<<(j-1));
        for (k in customers contained in prev) {
            dp[mask][j] = min(dp[mask][j], dp[prev][k] + d[k][j]);
        }
    }
}

OPT[0] = 0;
for (mask from 1 to (1<<n)-1) {
    OPT[mask] = INF;
    for (j in customers contained in mask) {
        OPT[mask] = min(OPT[mask], dp[mask][j] + d[j][0]);
    }
}

ans_post = 0;
for (mask from 0 to (1<<n)-1) {
    prob = product probability of this mask;
    ans_post += prob * OPT[mask];
}
```

### Adaptive

```cpp
for (pos in 0..n)
    F[pos][full_mask] = d[pos][0];

for (mask in reverse subset order) {
    if (mask == full_mask) continue;
    for (pos in 0..n) {
        F[pos][mask] = INF;
        for (i = 1; i <= n; ++i) if (!(mask & (1<<(i-1)))) {
            next = mask | (1<<(i-1));
            val = (1-p[i]) * F[pos][next]
                + p[i] * (d[pos][i] + F[i][next]);
            F[pos][mask] = min(F[pos][mask], val);
        }
    }
}

ans_adapt = F[0][0];
```

---

## 9. Common mistakes to avoid

### Mistake 1: confusing adaptive with a posteriori

Adaptive does **not** know the whole active set in advance.
It only learns by probing.

### Mistake 2: forgetting the forced move rule

If a probed customer is active, the next move is forced to that customer.
That is the defining rule of this thesis model.

### Mistake 3: storing too much state in the adaptive DP

For this exact model, the state does **not** need the full realization history.
`(current_position, probed_mask)` is enough.

### Mistake 4: solving a separate deterministic TSP for every subset from scratch

Do one Held–Karp over **all** customer subsets once, then reuse the subset costs.

### Mistake 5: forgetting the depot return


---

## 10. Minimal specification Claude should implement

A good first version is:

### Function signature idea

```cpp
double solve_a_posteriori(const vector<vector<double>>& d,
                          const vector<double>& p);

double solve_adaptive(const vector<vector<double>>& d,
                      const vector<double>& p);
double solve_a_priori(const vector<vector<double>>& d,
                      const vector<double>& p);
```

Where:
- `d` has size `(n+1) x (n+1)` including depot `0`
- `p` has size `n+1`
- `p[0] = 0`

### Semantics

- customers are `1..n`
- depot is `0`
- output exact expected costs

---

## 12. Graph representation

Claude should design a convenient way to represent the graph. The user would provide a graph as input, so the program has to be able to provide a format to encode the graph (together with edge weight and vertex activation prob.) (say, in .json file).

## 13. Remember to add some tests and a running demo (a working input json file for example)
