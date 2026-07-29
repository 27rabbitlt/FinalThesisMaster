# The bipartite cheap-transition model cannot fall below \(2/3\)

> **Physical-policy audit.**  A target-first abstraction would indeed be
> invalid: after discovering an active target, a policy cannot retroactively
> choose the source of its cheap transition.  The policy proved below is
> instead source-first.  It calls the permanent source \(u\), then probes
> uncalled target neighbors in rank order, and commits the first active
> answer to \(u\).  Conditional on the active target set, this legal TSP
> policy realizes `RANKING` exactly.

## Model

Let \(G=(U,V,E)\) be bipartite.  Vertices in \(U\) are permanent source
clients.  Every \(v\in V\) is independently active with an arbitrary known
probability \(p_v\).  A cheap transition is earned only when the current
active call is a source \(u\), after which the policy queries uncalled
neighbors of \(u\) until the first active target is found.  For a realized
active set \(A\subseteq V\), the clairvoyant value is
\[
       \nu(A):=\text{maximum matching size in }G[U,A].
\]

The proposed cheap-transition metric needs an instance for which every
causal policy has expected matching value less than
\((2/3)\mathbb E\nu(A)\).  No such instance exists.

## Policy

Use the following legal nonadaptive-query policy.

1. Draw a uniformly random permutation \(\pi\) of the source clients \(U\).
2. Independently draw a uniformly random ranking \(\sigma\) of the target
   clients \(V\).
3. Process sources in order \(\pi\).  At source \(u\), query its uncalled
   target neighbors in order \(\sigma\) until the first active one is
   found, if any.
4. Commit that first active target to \(u\).

This is exactly the classical `RANKING` algorithm, with a uniformly random
arrival order on the item side.

## Conditional reduction

Fix an arbitrary active set \(A\subseteq V\).  Queries to vertices in
\(V\setminus A\) cause no movement and do not change the set of available
targets.  At a source \(u\), the first active uncalled target found in
\(\sigma\)-order is exactly the highest-ranked currently unmatched neighbor
of \(u\) in \(A\).  Therefore, conditional on \(A\), the policy's matching
has exactly the same distribution as `RANKING` on the fixed bipartite graph
\[
       G_A:=G[U,A],
\]
with \(U\) as the online side in uniformly random order \(\pi\) and \(A\) as
the randomly ranked offline side.

Mahdian and Yan proved that `RANKING` in the random-arrival model satisfies,
for every fixed unweighted bipartite graph \(H\),
\[
       \mathbb E\lvert M_{\rm Ranking}(H)\rvert
       \geq0.696\,\nu(H).
\tag{1}
\]
See Mohammad Mahdian and Qiqi Yan,
*Online bipartite matching with random arrivals: an approach based on
strongly factor-revealing LPs*, STOC 2011:

- <https://research.google/pubs/online-bipartite-matching-with-random-arrivals-an-approach-based-on-strongly-factor-revealing-lps/>
- <https://arxiv.org/abs/2503.04196> (a later paper restating and revisiting
  the \(0.696\) theorem).

Applying (1) to \(H=G_A\) gives, for every realization \(A\),
\[
 \mathbb E_{\pi,\sigma}
 \bigl[\lvert M\rvert\mid A\bigr]
 \geq0.696\,\nu(A).
\tag{2}
\]
Average (2) over the activation law:
\[
\begin{aligned}
 \mathbb E_{A,\pi,\sigma}\lvert M\rvert
 &\geq0.696\,\mathbb E_A\nu(A)\\
 &>\frac23\,\mathbb E_A\nu(A).
\end{aligned}
\tag{3}
\]

The product assumption is more than is needed.  The argument works for any
distribution on \(A\) that is independent of the policy's private random
permutations.

The orientation above is important.  Merely querying a stochastic target
and then assigning it to an arbitrary predecessor would not certify a cheap
metric transition: the salesperson must already be located at that
predecessor.  Source-first ranked probing is the physically realizable
version.

If policies are required to be deterministic, average (3) over the finite
set of pairs \((\pi,\sigma)\).  At least one fixed pair satisfies
\[
       \mathbb E_A\lvert M_{\pi,\sigma}\rvert
       \geq0.696\,\mathbb E_A\nu(A).
\tag{4}
\]
Thus derandomization by existence preserves the conclusion.

## Consequence

Every bipartite cheap-transition instance admits a causal committed-matching
policy obtaining strictly more than \(2/3\) of the expected posterior
matching savings.  Hence this graph family cannot produce the matching loss
required by the proposed \(>4/3\) TSP reduction.
