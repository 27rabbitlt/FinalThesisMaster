# Source-first height-two matching has a \(0.696\) causal guarantee

## Status

This is a rigorous obstruction to using a single height-two
source--target chamber to obtain the matching-loss coefficient
\(c<2/3\) needed by the proposed layered \(\{1,2\}\)-metric calculation.

The argument corrects the physically invalid target-first interpretation
previously attached to RANKING.  The policy below is genuinely source-first:
it visits a source client before probing any target that it might match.

## Chamber

Let \(G=(U,V,E)\) be a fixed bipartite graph.

- Every \(u\in U\) is a permanent client.
- Every \(v\in V\) is independently active (the activation probabilities
  may be arbitrary).
- Calling an inactive \(v\) while located at \(u\) leaves the policy at \(u\).
- Calling an active \(v\) after \(u\) earns one cheap transition precisely
  when \(uv\in E\), and consumes \(v\).
- A source \(u\) can be called only once.

For an active set \(A\subseteq V\), let
\[
M^*(A)=\nu(G[U,A])
\]
be the posterior maximum number of cheap source--target transitions.

## A physically executable policy

Independently sample:

1. a uniformly random permutation \(\pi\) of the sources \(U\); and
2. a uniformly random ranking \(\rho\) of the targets \(V\).

Process sources in the order \(\pi\).  On source \(u\):

1. call \(u\), so the tour is physically located at \(u\);
2. among the still-uncalled neighbors of \(u\), call them in increasing
   \(\rho\)-order;
3. stop at the first active target, thereby taking the cheap transition
   \(u\to v\); if every probed neighbor is inactive, leave \(u\) unmatched.

Every probe made by this policy is legal.  In particular, an inactive target
does not move the tour away from \(u\), while the first active target is
immediately committed to \(u\).

## Realization-wise coupling to RANKING

Fix an arbitrary active set \(A\subseteq V\).  Restrict \(\rho\) to \(A\).
At the moment source \(u\) is processed, a target in \(A\) has previously
been called if and only if it was already matched: an earlier source stops
as soon as it encounters its first active target.  Hence the first active
target found by the physical probing loop is exactly the unmatched neighbor
of \(u\) having minimum restricted rank.

Consequently, conditional on \(A\), the returned matching is exactly the
standard RANKING matching on \(G[U,A]\), with the online side \(U\) arriving
in uniformly random order and the offline side \(A\) uniformly ranked.

Mahdian and Yan's random-arrival analysis therefore gives, for every fixed
\(A\),
\[
\mathbb E_{\pi,\rho}\!\left[M_{\rm causal}\mid A\right]
   \ge \beta\,M^*(A),
\qquad
\beta\ge 0.696.
\]
Taking expectation over the activation set,
\[
\mathbb E M_{\rm causal}
   \ge 0.696\,\mathbb E M^*
   > \frac23\,\mathbb E M^* .
\]
No independence assumption is actually needed for this final averaging
step; the realization-wise guarantee works for every distribution on \(A\).

## Consequence for the proposed metric ratio

The proposed repeated-boundary heuristic was
\[
\frac{\mathrm{ALG}}{\mathrm{POST}}\longrightarrow 2-c,
\qquad
c=\frac{\mathbb E M_{\rm causal}}{\mathbb E M^*}.
\]
A height-two chamber always has \(c\ge0.696\), so this calculation can give
at most
\[
2-0.696=1.304<\frac43.
\]

Thus no height-two source-first product-activation bipartite graph,
including KVV/random-suffix, high-girth, Ramanujan, or algebraic-lift
instances, can produce the required coefficient.

This does **not** settle a genuine multilayer chamber.  At an intermediate
active client, an incoming cheap transition consumes its unique visit, so
the policy must choose its outgoing continuation immediately.  Independent
RANKING matchings at consecutive boundaries cannot simply be executed one
after another.  Any remaining construction must exploit that no-revisit
coupling across at least three layers.

## Reference

M. Mahdian and Q. Yan, *Online Bipartite Matching with Random Arrivals: An
Approach Based on Strongly Factor-Revealing LPs*, STOC 2011.  Their theorem
proves that RANKING is at least \(0.696\)-competitive in the random-arrival
model.
