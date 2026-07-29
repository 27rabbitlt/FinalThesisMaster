# A fixed DAG with independently active edge-clients has gap one

## Scope

Let \(H=(W,F)\) be any finite directed acyclic graph.  Every edge
\(e\in F\) is a client, with an arbitrary activation probability; the
argument below is realization-wise and therefore does not require
independence.

For distinct edge-clients \(e,f\), use the valid \(1/2\) trail metric
\[
 d(e,f)=
 \begin{cases}
 1,&\operatorname{head}(e)=\operatorname{tail}(f),\\
 2,&\text{otherwise}.
 \end{cases}
\]
The depot has distance one in both directions to every edge-client.  As
proved in `trail-poset-metric-audit.md`, if \(N\) edges are active and their
active-call order has \(K\) maximal composable runs, then the closed tour
cost is exactly
\[
                         N+K.                    \tag{1}
\]

The tempting multilayer construction asks for a fixed DAG in which the
posterior can cover the active edges by few trails but every causal policy
creates many more trails.  This is impossible.

## Posterior trail count is a sum of local pairings

Fix an active edge set \(A\).  For \(v\in W\), write
\[
 I_v=\bigl|\{e\in A:\operatorname{head}(e)=v\}\bigr|,
 \qquad
 O_v=\bigl|\{e\in A:\operatorname{tail}(e)=v\}\bigr|.
\]

At \(v\), at most \(\min(I_v,O_v)\) incoming active edges can be paired
with outgoing active edges as consecutive members of trails.  These local
pairings have no compatibility conflict between different vertices:
choose an arbitrary matching of size \(\min(I_v,O_v)\) between the active
incoming and outgoing edges at every \(v\).  Every active edge then has at
most one paired predecessor and at most one paired successor.  Because
\(H\) is acyclic, the resulting components are directed paths, not cycles.
They are therefore a trail cover of \(A\).

Consequently the exact posterior trail number is
\[
 \tau(A)=|A|-\sum_{v\in W}\min(I_v,O_v).          \tag{2}
\]

## A realization-wise optimal causal policy

Fix any topological order \(\prec\) of \(W\).  Use the following policy.

1. If the current active edge is \(e=(u,v)\), query the still-uncalled
   outgoing edges of \(v\), in any fixed order, until the first active one
   is found.  Continue the same run with that edge.  If all are inactive or
   already called, end the run.
2. When no run is open, choose the \(\prec\)-least vertex having an
   uncalled outgoing edge.  Query its outgoing edges until the first active
   one is found; that edge starts the next run.  If all are inactive,
   advance to the next such vertex.

Inactive answers do not move the salesperson, so Step 1 is a legal causal
continuation attempt.  The policy uses only revealed outcomes.

### Lemma

No active edge leaving \(v\) is used to start a new run before every edge
entering \(v\) has been queried.

### Proof

Every incoming edge \((u,v)\) has \(u\prec v\).  The policy starts a run
at \(v\) only after there is no uncalled outgoing edge at any earlier tail
\(u\prec v\).  A continuation may temporarily carry the execution to much
later vertices, but after that run ends Step 2 returns to the least tail
still having an uncalled edge.  Thus it cannot advance past \(u\) while
\((u,v)\) remains uncalled. \(\square\)

Now fix \(v\).  Each of its \(I_v\) active incoming edges eventually
arrives at \(v\), either inside an existing run or as the first edge of a
new run.  On each such arrival, Step 1 finds and pairs an uncalled active
outgoing edge whenever one remains.  By the lemma, no outgoing active edge
was previously consumed as a new-run start.  Hence the policy creates
exactly
\[
                         \min(I_v,O_v)
\]
composable transitions at \(v\).  Summing over \(v\), its number of runs is
\[
 K_\pi(A)
 =|A|-\sum_v\min(I_v,O_v)
 =\tau(A).                                         \tag{3}
\]

Equations (1)--(3) give, for every realization \(A\),
\[
 \operatorname{cost}_\pi(A)
 =|A|+\tau(A)
 =\operatorname{OPT}_{\rm post}(A).                \tag{4}
\]

## Consequence

Every fixed-DAG edge/trail instance has adaptive-to-posterior ratio exactly
one.  This includes arbitrarily many layers, high-girth lifts, Ramanujan
bigraph interfaces, algebraic lifts, and all choices of unequal independent
edge probabilities.

Thus the proposed transition-rate calculation cannot be instantiated with
edge-clients.  A nontrivial multilayer route must use vertex-clients (path
cover rather than trail cover), cycles, or a higher-order compatibility
structure in which the posterior choices at different interfaces are not
independent local pairings.
