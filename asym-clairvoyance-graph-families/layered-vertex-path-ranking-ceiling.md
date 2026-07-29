# A simultaneous \(1-1/e\) transition policy for every layered path metric

## 1. Model and exact posterior value

Let \(H\) be a fixed digraph with levels \(V_0,\ldots,V_h\), with every arc
going from \(V_i\) to \(V_{i+1}\).  Its vertices are clients with arbitrary
independent activation probabilities.  For distinct clients put
\[
 d(x,y)=
 \begin{cases}
 1,&xy\in E(H),\\
 2,&xy\notin E(H),
 \end{cases}
\]
and let the depot have distance one in both directions to every client.
This is a directed metric because every nonzero distance lies in
\([1,2]\).

Fix a realization \(A\), and let \(N(A)=|A|\).  If an active-call order has
\(T\) consecutive pairs that are arcs of \(H\), its closed-depot cost is
\[
                    2N(A)-T.                       \tag{1.1}
\]

For the boundary \(i\), let
\[
 \nu_i(A)=\nu\bigl(H[A\cap V_i,A\cap V_{i+1}]\bigr)
\]
be its maximum bipartite matching size.  Matchings chosen independently at
all boundaries have indegree and outdegree at most one at every active
vertex.  Since levels increase, their union is a vertex-disjoint path
forest.  Conversely, the cheap transitions of every call order restrict to
a matching at each boundary.  Hence the maximum posterior number of cheap
transitions is exactly
\[
                         T^*(A)=\sum_{i=0}^{h-1}\nu_i(A), \tag{1.2}
\]
and
\[
              \operatorname{OPT}_{\rm post}(A)
                       =2N(A)-T^*(A).               \tag{1.3}
\]

## 2. One causal policy for all boundaries

Independently assign every level \(V_i\) a uniformly random ranking
\(\sigma_i\).  Fix also an arbitrary deterministic order inside each level.
The policy maintains at most one open path.

1. If the current active client is \(x\in V_i\), query its uncalled
   out-neighbors in \(V_{i+1}\) in \(\sigma_{i+1}\)-order.  On the first
   active answer \(y\), earn the transition \(x\to y\) and immediately
   repeat this rule from \(y\).  If no active uncalled out-neighbor remains,
   close the current path.
2. When no path is open, take the least level containing an uncalled client
   and query its clients in the fixed within-level order until an active one
   starts the next path.  Continue with Step 1.

Inactive calls cause no movement, so this is a legal adaptive TSP policy.
The least-level restart rule is essential.

### Lemma 1 (no premature target starts)

No vertex of \(V_{i+1}\) is called as a new-path start before every vertex
of \(V_i\) has been called.

### Proof

Whenever a path ends, Step 2 returns to the least level containing an
uncalled client.  A continuation can make a long excursion through later
levels, but it cannot call a new-path start during that excursion.  Thus the
restart scheduler cannot advance to \(V_{i+1}\) while an uncalled client
remains in \(V_i\). \(\square\)

### Lemma 2 (rank independence)

Conditional on the activation realization and on every private random
ranking except \(\sigma_{i+1}\), the order in which active vertices of
\(V_i\) act as sources is fixed and independent of \(\sigma_{i+1}\).

### Proof

A level-\(i\) source is either reached from \(V_{i-1}\), in an order
determined by earlier levels and \(\sigma_i\), or is eventually called as a
new-path start in the fixed within-level order.  The choices made using
\(\sigma_{i+1}\) only determine which later vertex follows the current
source and how long the ensuing forward excursion lasts.  Because all arcs
increase the level, that excursion cannot call another level-\(i\) vertex.
After it ends, the least-level scheduler resumes the same still-uncalled
level-\(i\) sequence.  It may delay the next level-\(i\) source but cannot
change its identity. \(\square\)

## 3. Boundary-by-boundary RANKING reduction

Fix a boundary \(i\) and condition as in Lemma 2.  Also fix the active set
\(A\).  By Lemma 1, a target in \(A\cap V_{i+1}\) is unavailable when a
source is processed exactly when it was already taken by an earlier
transition across this boundary; no unmatched target was prematurely
consumed as a start.

At a source \(x\), inactive targets disappear from its ranked probe list,
and the first active uncalled target is precisely the
\(\sigma_{i+1}\)-highest currently unmatched neighbor of \(x\).  The
boundary process is therefore the classical `RANKING` algorithm on
\[
 H[A\cap V_i,A\cap V_{i+1}]
\]
with an arbitrary fixed online order and an independent uniform ranking of
the offline side.

The adversarial-order `RANKING` theorem gives
\[
 \mathbb E_{\sigma_{i+1}}[M_i\mid A,\text{other ranks}]
       \ge \left(1-\frac1e\right)\nu_i(A),           \tag{3.1}
\]
where \(M_i\) is the policy's number of transitions across boundary \(i\).
Sum (3.1) over all boundaries and then average all conditioning:
\[
 \mathbb E M
 \ge \left(1-\frac1e\right)\mathbb E T^*.           \tag{3.2}
\]

Although the same call sequence couples adjacent boundaries, no
independence between the \(M_i\)'s was used.  Only the fresh target ranking
at each individual boundary was needed.

If policies must be deterministic, average (3.2) over the finite choices of
all rankings.  Some fixed rank vector has at least the same expected total
transition count.

## 4. Universal ceiling for the family

Put
\[
 \bar N=\mathbb E N(A),\qquad
 \bar T=\mathbb E T^*(A),\qquad
 \alpha=1-\frac1e.
\]
Equations (1.1), (1.3), and (3.2) imply
\[
\begin{aligned}
 \operatorname{OPT}_{\rm adapt}
   &\le 2\bar N-\alpha\bar T,\\
 \operatorname{OPT}_{\rm post}
   &=2\bar N-\bar T.
\end{aligned}                                      \tag{4.1}
\]
Since a path forest on \(N\) vertices has at most \(N\) transitions,
\(\bar T\le\bar N\).  The ratio in (4.1) is increasing in
\(\bar T/\bar N\), so
\[
 \boxed{
 \frac{\operatorname{OPT}_{\rm adapt}}
      {\operatorname{OPT}_{\rm post}}
 \le 2-\alpha
 =1+\frac1e
 \approx1.367879.}                                  \tag{4.2}
\]

## 5. What remains open

The literal fixed-DAG edge/trail model has ratio one by
`dag-edge-trail-gap-one.md`.  For vertex-client path metrics, (4.2) leaves
only the narrow interval
\[
                  \frac43 < R\le1+\frac1e.
\]
A construction in this interval must make the inherited source order on
most boundaries nearly as bad as the adversarial-order RANKING constant.
Single-boundary hardness is insufficient: a fresh random source order gives
at least \(0.696\) of the posterior matching, as recorded in
`source-first-ranking-0696-obstruction.md`.

Thus the exact remaining target is a fixed multilayer graph in which
predecessor-induced source orders recursively realize a transition rate
below \(2/3\), despite the policy's control over all query orders.  External
arrival permutations and reusable chronological gates do not instantiate
that target.
