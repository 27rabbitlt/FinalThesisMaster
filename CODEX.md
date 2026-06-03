# CODEX.md - Proof-Side Working Rules

This project is a master's thesis project on stochastic TSP, especially the
relationship between adaptive, a posteriori, and a priori benchmarks.

## Scope

- Work on the proof side of the thesis.
- Use `thesis/main.tex` as the primary source of definitions and current
  theorem statements.
- Do not spend effort on the solver implementation unless the user explicitly
  asks for solver work.
- Ignore the survival-local packet DP section of `thesis/main.tex` unless the
  user explicitly asks about it.
- The main goal is to prove theorems, strengthen or correct statements, and
  turn partial ideas into self-contained TeX proofs.

## Background And Notation

The stochastic TSP instance consists of a finite metric space
`({r} union V, d)`, where `r` is the depot and `V` is the set of `n` clients.
The metric satisfies the triangle inequality
`d(u,v) <= d(u,w) + d(w,v)`. A metric is symmetric if
`d(x,y) = d(y,x)` for all vertices.

A tour on a subset `S subseteq {r} union V` containing the depot is a cyclic
sequence

```tex
T=(r,u_1,u_2,\ldots,u_{|S|-1},r)
```

that visits every vertex in `S` exactly once. Its cost is

```tex
\cost(T)
:=
d(r,u_1)+d(u_1,u_2)+\cdots
+d(u_{|S|-2},u_{|S|-1})+d(u_{|S|-1},r).
```

For `A subseteq V`, `\opt(A)` denotes the minimum tour cost on `{r} union A`.

In the stochastic setting, each client `v in V` is active independently with
probability `p_v`. The active set is `A subseteq V`, drawn from the product
distribution `q`, where

```tex
q(A)=\prod_{v\in A}p_v\cdot\prod_{v\notin A}(1-p_v).
```

There are three benchmarks.

### A Posteriori TSP

An a posteriori solution is a family `\mathcal S=(T_A)_{A subseteq V}`, where
`T_A` is a tour on `{r} union A`. Its cost is

```tex
\cost(\mathcal S):=\E_{A\sim q}[\cost(T_A)].
```

Since every realization can be optimized independently,

```tex
\optpost=\E_{A\sim q}[\opt(A)].
```

This benchmark knows the whole active set before choosing the tour.

### A Priori TSP

An a priori solution is a fixed full tour

```tex
T=(r,v_1,\ldots,v_n,r)
```

chosen before the active set is known. For realization `A`, the shortcut
`T|_A` keeps only `{r} union A` in the cyclic order prescribed by `T`. Its
expected cost is

```tex
\cost(T):=\E_{A\sim q}[\cost(T|_A)].
```

The optimum is

```tex
\optprior:=\min_T \cost(T).
```

This benchmark has the least information.

### Adaptive TSP

An adaptive policy knows the probabilities but not the realization. It calls
clients one by one. If the called client is active, the salesperson must travel
to it immediately; if it is inactive, the salesperson remains in place. The
order of calls may depend on outcomes already observed.

Formally, a policy is

```tex
\pi:({r}\cup V)\times 2^V\to {r}\cup V\cup\{\bot\},
```

where `\pi(u,S)` is the next vertex to call when the salesperson is at `u` and
`S` is the set of uncalled vertices. The policy must call an uncalled vertex
while `S` is nonempty and may return to the depot only after `S` is empty.

For a realization `A`, the induced tour starts at `r` with `S=V`. Repeatedly
call `v=\pi(u,S)`; if `v in A`, travel immediately to `v` and set `u=v`; then
remove `v` from `S`. Finally return to `r`. The adaptive optimum is

```tex
\optadapt:=\min_\pi \E_{A\sim q}[\cost(T(A;\pi))].
```

The benchmark ordering is

```tex
\optpost \le \optadapt \le \optprior.
```

The gap measures are:

- Adaptivity gap: `sup_I \optprior(I)/\optadapt(I)`.
- Clairvoyance gap: `sup_I \optadapt(I)/\optpost(I)`.
- Obliviousness gap: `sup_I \optprior(I)/\optpost(I)`.

## Proof Workflow

When the user gives an input prompt for a theorem or conjecture:

1. Identify the exact statement being attempted, including whether the metric is
   symmetric or asymmetric and which benchmark ratio is involved.
2. Work directly on the theorem. Push the proof forward until either no new
   result seems reachable, or the current statement appears to be the correct
   unconditional theorem.
3. Write the resulting proof as a TeX file or TeX fragment suitable for the
   thesis.
4. Send that TeX proof to a separate agent instance for review, using available
   multi-agent tooling. The reviewing agent should give a verdict and identify
   missing arguments, hidden assumptions, or gaps.
5. Record both the progress and the review verdict in `progress.md`.
6. Continue pushing from the review feedback.
7. After three proof-review-push rounds on the same theorem, stop and ask the
   user for human review and a new input prompt.

Do not treat numerical solver evidence as proof. Solver output may be used only
as intuition or sanity checking when explicitly useful.

