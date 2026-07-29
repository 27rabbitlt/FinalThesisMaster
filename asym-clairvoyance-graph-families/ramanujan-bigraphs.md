# Ramanujan bigraphs

## Outcome

Ramanujan bigraphs offer expansion, matchings, and logarithmic diameter, but
none of these automatically creates a clairvoyance gap.  The natural
edge-switch construction has two incompatible analyses:

- if edge gadgets are separated so that every local conflict is chargeable,
  the \(d\) perfect matchings are additive copies of the \(4/3\) switch;
- if they share the bigraph vertices, inactive edges need not be used, local
  charges overlap, and a fixed causal Euler sweep can query an edge selector
  exactly at the tail of its active detour.

There is also a linear flow-balance penalty in the candidate ledger that
retains the independently preferred orientation of every edge.  This is not a
lower bound on the true posterior optimum.  Consequently the
Ramanujan property presently gives no ratio above \(4/3\).

## 1. Edge-switch template

Let \(B=(L,R,E)\) be a connected \(d\)-regular bipartite graph with
\(|L|=|R|=n\).  It may be a Ramanujan bigraph and may also have high girth.
Replace every undirected edge \(e=\{u,v\}\), \(u\in L,v\in R\), by a directed
switch with stochastic client \(m_e\):
\[
 u\longrightarrow v,\qquad
 v\longrightarrow m_e\longrightarrow u,
\]
all displayed arcs having length \(w\).  Put the inverse scaffold arcs of
length \(w\) in the generating graph as needed for a bidirected connected
backbone.  The vertices of \(L\cup R\) are permanent, and the \(m_e\) are
independent Bernoulli clients.  A depot is attached to one permanent vertex,
or is identified with it.  The bidirected backbone makes the generating graph
strongly connected, so its shortest-path closure is a directed metric.

Locally, an inactive edge favors \(u\to v\), while an active edge can be
served by \(v\to m_e\to u\).  This resembles the directed triangle in
Chapter 4, but the endpoints now belong to \(d\) different switches.

Two probability regimes matter.

- For fixed \(p>0\), there are \(pnd+o(nd)\) active edge clients.  Their
  service dominates the \(2n\) permanent vertices when \(d\) grows.
- For \(p=\lambda/d\), there are \(\Theta(n)\) active selectors, comparable
  with the permanent backbone.  This sparse regime is the only plausible
  place for expansion to couple route choices without simply flooding the
  instance with active midpoints.

## 2. Posterior tours: a valid fallback and an invalid fantasy

### A tour for every realization

Double every edge of \(B\), once in each direction, and take an Euler tour of
the resulting directed multigraph.  Mark the occurrence \(v\to u\) belonging
to \(e=\{u,v\}\), and schedule the call to \(m_e\) at that point of the
cyclic order.  If \(m_e\) is active, replace the marked arc in the expanded
walk by \(v\to m_e\to u\); if it is inactive, retain \(v\to u\).

Every permanent vertex and every active midpoint is served.  Up to the depot
attachment, this gives the realization-wise bound
\[
 \operatorname{tour}(A)
 \le 2|E|w+|A|w
 =(2nd+|A|)w.
\]
Crucially, the same bound is achieved by a legal **adaptive** call order.
Call each client when its first marked occurrence is reached in the cyclic
order.  The salesperson need not voluntarily walk through already-called
vertices.  Maintain the corresponding Euler segment as a pending walk:
inactive calls leave it pending, while the next active call shortcuts the
whole pending segment in the metric.  If that active client is \(m_e\), the
pending segment ends with \(v\to m_e\); its suffix \(m_e\to u\) is then
carried into the next segment.  The final pending segment is absorbed into
the depot return.  Each generating arc of the realization-dependent expanded
walk is charged at most once.  Thus the active movement to a selector plays
the role of its planned detour rather than an extra penalty.  Any lower bound
based only on “an active edge has the opposite preferred orientation” fails
against this sweep.

### Why independently preferred orientations do not form a tour

Suppose, more optimistically, that a candidate posterior routing ledger
retains \(u\to v\) when
\(m_e\) is inactive and the contracted trace \(v\to u\) when it is active.
Let \(A_x\) be the number of active incident edge clients at a permanent
vertex \(x\).  The divergence of these preferred contracted arcs is
\[
 b(x)=
 \begin{cases}
 d-2A_x,&x\in L,\\
 2A_x-d,&x\in R.
 \end{cases}
\]
Any Eulerian augmentation must add at least
\[
 \frac12\sum_{x\in L\cup R}|b(x)|
\]
directed scaffold traversals: one added arc can reduce the total positive
divergence by at most one.  At \(p=1/2\),
\[
 \mathbb E|b(x)|
 =\mathbb E\left|d-2\,{\rm Bin}(d,1/2)\right|
 =\Theta(\sqrt d).
\]
Hence the expected correction is \(\Omega(n\sqrt d)\) arcs before one even
connects distinct Eulerian components.

This is a rigorous obstruction to the tempting posterior calculation
\[
 \tfrac12|E|w+\tfrac12(2|E|w)=\tfrac32|E|w.
\]
That calculation treats every edge as an independent open triangle and
ignores conservation of flow at the shared bigraph vertices.  Expansion can
help route the correcting flow, but cannot remove the imbalance which has to
be routed.

There is an even more basic issue: when \(m_e\) is inactive, no client forces
the tour to use edge \(e\) at all.  A tour through the \(2n\) permanent
vertices uses only a connected spanning amount of the \(nd\) available edge
structure.  Therefore it is invalid to collect an inactive-state saving from
all \(nd\) edges unless extra permanent edge modules are added.

## 3. The separated-edge version is additive

A \(d\)-regular bipartite graph decomposes into \(d\) perfect matchings.  One
can duplicate the endpoint ports by edge color, put one triangle switch on
every matching edge, and join all copies of a vertex through a compulsory
backbone.  This repairs the preceding “unused inactive edge” problem and
makes every switch a genuine module.

It also destroys the hoped-for amplification.  With port boundaries strong
enough to prove that every module contributes its local open-service value,
the exact leading contributions are
\[
 P_e=\frac32w,\qquad A_e=2w
\]
at \(p=1/2\).  The common bigraph/backbone cost \(C\ge0\) gives
\[
 \frac{C+\sum_e A_e}{C+\sum_e P_e}\le\frac43.
\]
This is precisely additive replication.  The Ramanujan spectrum is irrelevant
once the matching ports are separated.

If the color ports are identified instead, the additivity proof disappears:
one arrival at a vertex can begin or finish many edge services, and arbitrary
interleaving can share scaffold arcs.  A matching-based paired-realization
argument can charge at most one edge incident to each permanent vertex, hence
at most \(n\) of the \(nd\) switches in one round.  Charges for different
matchings are not disjoint.  No policy-uniform summation over all \(d\)
matchings is currently justified.

## 4. Sparse active-edge regime

Set \(p=\lambda/d\), so the random active-edge graph has expected degree
\(\lambda\) and \(\Theta(n)\) active clients.  A possible posterior plan is:

1. take a fixed closed scaffold walk through all permanent vertices;
2. orient or reroute portions of it to absorb active \(m_e\);
3. add a correcting circulation for the resulting imbalance.

A possible adaptive lower bound would say that, before enough incident edge
bits at a vertex have been learned, choosing an exit makes a constant fraction
of the still-hidden active edges expensive.  Expansion would then be used to
show that such unresolved boundaries occur at linearly many vertices.

What is missing is a statement robust to remote calls.  At a vertex \(v\), a
policy may call all incident \(m_e\) before choosing an exit.  Inactive calls
cost nothing; an active call moves along a service trace which the policy had
to perform anyway.  Calling incident selectors during a deterministic
scaffold sweep is therefore much cheaper than a selector-first policy that
queries all \(nd\) edges from the depot.

Moreover, logarithmic diameter is double-edged.  It makes global correction
and early selector tours possible, while shortest-path closure caps any
claimed wrong-orientation detour by an alternate expander path.  Ramanujan
spectral expansion supplies neither a directed potential nor a port charge.

No leading constants for a ratio \(>4/3\) can currently be justified in this
regime.  The optimistic edge-independent constants are still only
\(2/(\tfrac32)=4/3\), and the balance correction increases the cost of the
all-edges preferred-trace candidate rather than the adaptive numerator; it
does not lower-bound the true posterior optimum.

## 5. Failure audit

- **Remote early calls.**  The Euler-sweep policy orders the call to \(m_e\)
  at the marked tail \(v\) of \(v\to m_e\to u\); the pending-walk argument
  remains valid even when the current position has shortcut over \(v\).
  Active movement is useful service; inactive information is free.
- **Calling a separator first.**  Calling all incident selectors upon first
  reaching a vertex is legal.  An unresolved-boundary proof must charge this
  local star query, not merely a depot-first query.
- **Interleaving.**  Edge gadgets share permanent endpoints.  Local
  open-service lower bounds cannot be summed over all edges without assigning
  each scaffold arc occurrence to at most one conflict.
- **Shortest-path shortcuts.**  The expander supplies many alternate routes
  and logarithmic diameter.  Any large directed penalty needs a potential or
  quotient lower bound after metric closure.
- **Independent bits.**  Randomly preferred edge directions have
  \(\Theta(n\sqrt d)\) flow imbalance at \(p=1/2\); retaining all of them does
  not directly produce an Euler tour.
- **Inactive edges.**  An inactive midpoint does not make its edge a
  compulsory part of the tour.  Counting a saving on every inactive edge is
  invalid unless one introduces permanent edge clients, which returns to
  additive modules.

## Verdict

**No certified improvement; the natural edge-switch use is obstructed.**
Separating the switches makes the construction additive with ceiling
\(4/3\).  Sharing the bigraph vertices removes additivity, but then preferred
orientations are imbalanced, inactive edges are optional, and a causal Euler
sweep defeats the simplest orientation-conflict lower bound.  Ramanujan
expansion currently helps the posterior and adaptive policy at least as much
as it helps a lower-bound proof.

## Next lemma

The only promising version is the sparse regime \(p=\lambda/d\).  Prove or
refute the following **online star-routing lemma** for a specific directed
replacement of a Ramanujan bigraph:

> There are constants \(c_{\rm post},\delta>0\) such that every realization
> has a depot tour of cost at most
> \((c_{\rm post}+o(1))nw\), while every causal policy pays at least
> \((c_{\rm post}+\delta-o(1))nw\), even if on first reaching a permanent
> vertex it calls all still-uncalled incident selectors and even if services
> at different vertices are arbitrarily interleaved.

To imply a ratio above \(4/3\), the proved constants must satisfy
\[
 \delta>\frac13c_{\rm post}.
\]
The proof must assign every charged movement to a vertex or directed edge
with bounded congestion and must certify its length in the shortest-path
metric.  Without this lemma, the Ramanujan condition is only a routing aid,
not a clairvoyance-gap mechanism.
