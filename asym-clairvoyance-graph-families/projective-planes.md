# Projective-plane incidence graphs

## Verdict

**Complete no-go for the most natural template.**  Take the Levi graph of the
Desarguesian plane \(\mathrm{PG}(2,q)\), make all points permanent, make the
lines independent Bernoulli clients, and give every incident point-to-line
arc length \(\alpha\) and every reverse arc length \(\beta\).  Despite
\(\alpha\ne\beta\), this instance has
\[
  \operatorname{OPT}_{\rm adapt}
  =
  \operatorname{OPT}_{\rm post}
  =
  2\gamma+(q^2+q+1)(\alpha+\beta)
\]
for every activation probability and, more strongly, for every realization.
Thus its clairvoyance gap is exactly \(1\).

There are two independent reasons.

1. The apparent asymmetry is only a vertex-potential reweighting of an
   undirected incidence metric, so it has zero directed circulation around
   every cycle and disappears from every closed depot tour.
2. A Singer cycle gives a Hamilton cycle of the incidence graph.  Along that
   cycle, calling an incident line and then the next permanent point costs
   \(\alpha+\beta\) whether the line is active or inactive.  Hence inactive
   calls reveal information for free without creating any routing advantage
   for the clairvoyant benchmark.

Merely putting independent directed-triangle switches on the flags is also
not promising: if their costs decompose additively, the replication principle
shows that their ratio is at most the largest local ratio, hence at most the
existing \(4/3\).  A positive projective-plane construction would need
nonzero directed circulation and a genuinely global coupling of many local
route choices.

## 1. Geometry and the natural stochastic instance

Let
\[
  \mathcal P=\mathrm{PG}(2,q),\qquad
  \mathcal L=\{\text{projective lines}\},\qquad
  N=q^2+q+1.
\]
There are \(N\) points and \(N\) lines.  Every point is incident with \(q+1\)
lines, every line contains \(q+1\) points, two points determine a unique line,
and two lines meet in a unique point.  The incidence graph (or Levi graph) is
bipartite, has \(2N\) vertices, girth \(6\), and diameter \(3\).

Fix a point \(p_0\) and add a depot \(r\).  The generating digraph has
vertices
\[
  \{r\}\cup\mathcal P\cup\mathcal L.
\]
For every flag \(p\in\ell\), add
\[
  p\longrightarrow\ell\quad\text{of length }\alpha,
  \qquad
  \ell\longrightarrow p\quad\text{of length }\beta,
\]
where \(\alpha,\beta>0\).  Add \(r\to p_0\) and \(p_0\to r\), each of length
\(\gamma>0\).  The graph is strongly connected, and \(d\) is its directed
shortest-path metric.

Every point is a permanent client.  Every line is independently active with
probability \(p\), where \(p\in[0,1]\) is arbitrary.  The proof below does not
use equality of the line probabilities; they may in fact be arbitrary and
independent.

This is the most direct attempt to turn point/line duality into direction:
moving from a point to a line can be cheap while returning to a point is
expensive, or conversely.

## 2. The potential obstruction

Set
\[
  s=\frac{\alpha+\beta}{2},\qquad
  \phi(r)=\phi(p)=0\ (p\in\mathcal P),\qquad
  \phi(\ell)=\frac{\alpha-\beta}{2}\ (\ell\in\mathcal L).
\]
Give every undirected incidence edge base length \(s\), and give the depot
edge base length \(\gamma\).  For every generating arc,
\[
  w(u,v)=w_{\rm sym}(\{u,v\})+\phi(v)-\phi(u).
\]
Consequently every directed generating-graph walk \(W:u\leadsto v\) obeys
\[
  w(W)=w_{\rm sym}(W)+\phi(v)-\phi(u).
\]
Taking a minimum over paths gives the exact shortest-path identity
\[
  d(u,v)=d_{\rm sym}(u,v)+\phi(v)-\phi(u).
  \tag{2.1}
\]
Thus shortest-path closure cannot create an unaccounted shortcut: it is
completely described by (2.1).  For any sequence
\(r=v_0,v_1,\ldots,v_k=r\),
\[
  \sum_{i=0}^{k-1}d(v_i,v_{i+1})
  =
  \sum_{i=0}^{k-1}d_{\rm sym}(v_i,v_{i+1}),
  \tag{2.2}
\]
because the potential terms telescope.

This proves a more general no-go: for **any** activation law on the incidence
vertices, the a-posteriori and adaptive values of this directed instance are
identical to those of its symmetric companion.  The unequal values
\(\alpha,\beta\) supply no asymmetric order conflict at all.

## 3. A Hamilton cycle from the Singer action

For completeness, here is the precise property of the Desarguesian plane
used below.  The multiplicative action of a primitive element of
\(\mathbb F_{q^3}\), modulo the scalar subgroup
\(\mathbb F_q^\times\), induces a cyclic group
\[
  G=\langle g\rangle,\qquad |G|=N,
\]
acting regularly on the points and also regularly on the lines of
\(\mathrm{PG}(2,q)\).  This is the Singer action.

Choose \(p_0\), put \(p_i=g^i p_0\), and let \(\ell_0\) be the unique line
through \(p_0\) and \(p_1\).  Put \(\ell_i=g^i\ell_0\).  Regularity implies
that \(p_0,\ldots,p_{N-1}\) enumerate the points and
\(\ell_0,\ldots,\ell_{N-1}\) enumerate the lines.  Moreover
\(\ell_i\) contains both \(p_i\) and \(p_{i+1}\), with indices modulo \(N\).
Hence
\[
  p_0,\ell_0,p_1,\ell_1,\ldots,p_{N-1},\ell_{N-1},p_0
  \tag{3.1}
\]
is a Hamilton cycle of the incidence graph.

The underlying Singer theorem is J. Singer, *A theorem in finite projective
geometry and some applications to number theory*, Transactions of the AMS
43 (1938), 377--385,
[AMS article](https://www.ams.org/tran/1938-043-03/S0002-9947-1938-1501951-4/).
The short orbit argument above is all that is needed here.

## 4. Exact a-posteriori upper bound

Fix an arbitrary active line set \(A\subseteq\mathcal L\).  Follow the cyclic
order (3.1), omitting service of inactive lines.  More explicitly, start at
\(r\), go to \(p_0\), and for \(i=0,\ldots,N-2\):

- if \(\ell_i\in A\), go \(p_i\to\ell_i\to p_{i+1}\), at cost
  \(\alpha+\beta\);
- if \(\ell_i\notin A\), go directly from \(p_i\) to \(p_{i+1}\).  The
  two-arc path through \(\ell_i\) shows that this costs at most
  \(\alpha+\beta\).

At the last point \(p_{N-1}\), visit \(\ell_{N-1}\) if active and return
through \(p_0\) to \(r\); if it is inactive, return to \(p_0\) through that
line without serving it.  Either case costs at most
\(\alpha+\beta+\gamma\) from \(p_{N-1}\), with the active case including the
preceding \(p_{N-1}\to\ell_{N-1}\) move.  Including the initial depot edge,
\[
  C_{\rm post}(A)\le 2\gamma+N(\alpha+\beta).
  \tag{4.1}
\]
This is a realization-wise bound, so no exceptional event or expectation is
hidden in it.

## 5. Universal lower bound, including arbitrary interleaving

Consider any closed metric tour that visits every permanent point, and expand
each metric move into a shortest generating-graph walk.  Every departure from
the depot and final return must use the unique depot edge, giving cost at
least \(2\gamma\).  Excursions inside the incidence graph alternate between
points and lines.

If the expanded walk uses \(H\) incidence-arc occurrences, it has \(H/2\)
point-to-point steps through line vertices.  A collection of closed
\(p_0\)-to-\(p_0\) excursions visiting all \(N\) distinct point vertices
requires at least \(N\) such steps.  Equivalently, a closed alternating walk
with fewer than \(2N\) arc occurrences has fewer than \(N\) point positions
and cannot contain all permanent points.  Therefore \(H\ge 2N\).

By the potential identity, the incidence part of every closed walk costs
\(sH\), regardless of its orientation.  Hence every realization and every
tour satisfy
\[
  C(A)\ge 2\gamma+2Ns
       =2\gamma+N(\alpha+\beta).
  \tag{5.1}
\]
This argument permits repeated vertices, passages through uncalled clients,
multiple depot excursions, arbitrary interleaving of line and point service,
and every shortcut in the metric closure.

Since an adaptive execution produces such a closed walk in every realization,
(5.1) is also a universal adaptive lower bound.  It does not assume a
one-visit decomposition or a preferred policy.

Combining (4.1) and (5.1) already proves the exact a-posteriori value.

## 6. A matching legal adaptive policy

The Hamilton order gives an even stronger result.  Use the following fixed,
nonadaptive call order:

1. call \(p_0\);
2. at \(p_i\), call \(\ell_i\);
3. for \(i<N-1\), call the permanent point \(p_{i+1}\);
4. after calling \(\ell_{N-1}\), return to the depot.

If \(\ell_i\) is inactive, its call causes no movement and the subsequent
move \(p_i\to p_{i+1}\) costs at most \(\alpha+\beta\).  If it is active, the two
moves \(p_i\to\ell_i\to p_{i+1}\) have the same total cost.  At the last
line, the final return replaces the already-called \(p_0\) call and again has
the same total segment cost.  Thus the policy is legal in the remote-call
model and costs
\[
  2\gamma+N(\alpha+\beta)
\]
for every activation vector.  Together with (5.1),
\[
  \boxed{
  \operatorname{OPT}_{\rm adapt}
  =
  \operatorname{OPT}_{\rm post}
  =
  2\gamma+N(\alpha+\beta)
  }.
\]
The prospective ratio is therefore exactly \(1\), not merely at most
\(4/3\).

## 7. Failure audit

### Remote early probing and free inactive information

The matching policy does not even need a separate probing phase.  It embeds
every line query into a mandatory point-to-point segment.  An inactive answer
and an active answer have identical segment cost.  Thus the model's free
inactive calls completely neutralize the putative line selectors.

More generally, the line set has directed diameter at most
\(\alpha+\beta\): two lines meet in a point.  Calling all line selectors
first is therefore also cheap relative to any proposal that uses only a small
expected number of active lines.

### Interleaving

The lower bound counts all expanded arc occurrences and all point positions.
It is unaffected by splitting service into arbitrarily many pieces or by
interleaving lines and points.

### Metric shortcuts

Equation (2.1) exactly characterizes the metric closure.  No drawn penalty
survives unless it is also present in the symmetric base metric.  In
particular, changing only the common point-to-line and line-to-point weights
cannot create directed route incompatibility.

### Many local flag gadgets

Replacing every flag by an independent copy of the existing triangle switch
does not use the projective plane unless choices in different flags are
coupled.  Under an additive decomposition, the ratio of the sum is at most
the maximum local ratio.  Overlapping flags may destroy additivity, but then a
ports-and-extra-pieces lemma is required; expansion alone is not a lower
bound.

### A single selected line or pencil

A selector represented by one stochastic line can be called immediately.
If it is inactive there is no movement; if active, every point is within
three incidence arcs.  Thus a single line cannot secretly choose a
macroscopic route unless its early active-call distance is itself
macroscopic, in which case the shortest-path closure through the
diameter-three incidence graph must be audited.

## 8. What would have to change

The uniform bipartite orientation has zero circulation.  On a connected
underlying graph, arc weights are a symmetric weighting plus a vertex
potential exactly when every directed cycle has zero antisymmetric
circulation.  Therefore any genuinely asymmetric projective-plane proposal
must assign flag-dependent weights with nonzero circulation on some incidence
cycle.

That change creates two immediate difficulties.

1. The incidence graph has many overlapping \(6\)-cycles and diameter \(3\).
   A large penalty on one flag can be bypassed through a short alternative
   incidence path.
2. The random input must remain the independent activity of clients.  A
   random Singer shift, random line ordering, or random orientation is part of
   the fixed metric and is known to both benchmarks; it cannot be the hidden
   state.

A possible hybrid is to use a projective plane only as an outer route code
for recursive triangle blocks.  But the outer geometry must force choices in
many blocks to be globally incompatible.  If it only places the blocks on
flags, additive replication cannot exceed \(4/3\).

## 9. Status labels and next lemma

- **Complete proof:** the uniform point-to-line/line-to-point metric on
  \(\mathrm{PG}(2,q)\), with permanent points and independent stochastic
  lines, has gap exactly \(1\).
- **Complete proof:** any metric whose arc asymmetry is a vertex potential is
  equivalent, for both benchmarks, to its symmetric companion.
- **Conditional idea:** a nonpotential flag weighting could act as a global
  route code, but neither its a-posteriori tour nor an adaptive lower bound is
  currently established.
- **Speculation:** recursive triangle blocks indexed by flags may be useful
  only if projective-plane incidence enforces nonadditive compatibility
  constraints.

**Next lemma to prove or refute (nonpotential route-code lemma).**  Find
positive flag weights on \(\mathrm{PG}(2,q)\), surviving directed
shortest-path closure, and independent selector clients such that:

1. every activation vector admits a depot tour of cost \(P_q\);
2. cheap tours for many activation vectors have incompatible prefixes, not
   just different local detours;
3. after calling all selectors first, the expected movement is already at
   least \((4/3+\varepsilon)P_q\), or else the selector-first policy is
   explicitly defeated; and
4. a ports/charging argument lower-bounds every interleaved adaptive policy.

Absent such a lemma, projective-plane incidence is decorative rather than an
amplifier of the current triangle construction.
