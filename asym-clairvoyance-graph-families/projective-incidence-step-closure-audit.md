# Exact closure audit for point--line--point incidence constructions

## Scope and conclusion

Let all points of a projective plane be permanent clients and let line
vertices, possibly with several independent client copies of one geometric
line, be stochastic clients.  The intended primitive is an incidence step
\[
                         p\longrightarrow \ell
                         \longrightarrow q,
              \qquad p,q\in\ell.                         \tag{0.1}
\]

For the natural uniform flag weights, shortest-path closure has an exact
description.  The direction bias is a vertex potential, every closed route
is an undirected Levi-graph walk in disguise, and a line-to-line move may
reuse their intersection point as transit without serving it.  In
particular:

1. with one client per geometric line, a Singer-cycle policy is
   realization-wise optimal and the gap is exactly \(1\);
2. duplicating line clients does **not** create point capacity: an already
   served point can join arbitrarily many later line-to-line moves;
3. the duplicate problem is a graphical-TSP/degree-two packing problem,
   not online matching of lines to unit-capacity points; and
4. every lower bound obtained by marking line detours on one fixed backbone
   walk is defeated by a legal adaptive pending-walk sweep of the same
   length.

Thus no \(>4/3\) construction follows from the incidence-step accounting.
Nonuniform flag circulation may define another directed route-code problem,
but it needs a new global lower bound; projective-plane incidence supplies
no consumable resource by itself.

## 1. Generating graph and exact metric closure

Let \(\Pi=(\mathcal P,\mathcal L,\in)\) be a projective plane of order \(q\),
and put
\[
                         N=q^2+q+1.
\]
For every flag \(p\in\ell\), put generating arcs
\[
       p\to\ell\text{ of length }\alpha,\qquad
       \ell\to p\text{ of length }\beta,                 \tag{1.1}
\]
where \(\alpha,\beta>0\).  A client copy of a geometric line is a distinct
line-side vertex with the same incident point neighbors and the same flag
weights.  Attach the depot \(r\) to a fixed permanent point \(p_0\) in both
directions with length \(\gamma\).  Take directed shortest-path closure.

Set
\[
 s=\frac{\alpha+\beta}{2},\qquad
 \phi(p)=0,\qquad
 \phi(\ell)=\frac{\alpha-\beta}{2}.                       \tag{1.2}
\]
Every generating flag arc has the form
\[
 w(x,y)=s+\phi(y)-\phi(x).                               \tag{1.3}
\]
Therefore, for all incidence vertices,
\[
                         d(x,y)=s\,h(x,y)+\phi(y)-\phi(x),
                                                                    \tag{1.4}
\]
where \(h(x,y)\) is undirected hop distance in the incidence graph enlarged
by the duplicate line vertices.  The potential difference is path
independent, so (1.4) remains exact after minimization.

In a projective plane, two points lie on one common line and two geometric
lines meet in one point.  Hence, for distinct clients,
\[
\begin{array}{c|c}
\text{ordered types}&d\\ \hline
p\to p'&\alpha+\beta=2s,\\
\ell\to m&\alpha+\beta=2s,\\
p\to\ell,\ p\in\ell&\alpha,\\
\ell\to p,\ p\in\ell&\beta,\\
p\to\ell,\ p\notin\ell&2\alpha+\beta,\\
\ell\to p,\ p\notin\ell&\alpha+2\beta.
\end{array}                                                \tag{1.5}
\]
The line-to-line row also holds for two distinct copies of the same
geometric line: they share every point of that line.

For every closed depot call sequence, the \(\phi\)-terms telescope.  Its
cost is exactly
\[
                         2\gamma+sH,                     \tag{1.6}
\]
where \(H\) is the total undirected hop length of the corresponding
shortest Levi-graph walks.  Unequal \(\alpha,\beta\) therefore create no
closed-tour asymmetry.

## 2. Exact service-order formulation

Fix a realization with all \(N\) point clients and \(K\) active line-client
copies.  Write a cyclic order of these \(N+K\) service clients, cutting the
cycle at \(p_0\) for the depot.

Let \(I\) be the number of consecutive point--line or line--point pairs
that are incident, and let \(J\) be the number that are nonincident.
Same-side consecutive pairs always have hop distance two.  An incident
cross-pair has hop distance one, and a nonincident cross-pair has hop
distance three.  Thus the exact cost of the order is
\[
 C(\sigma)
 =2\gamma+s\bigl(2(N+K)-I(\sigma)+J(\sigma)\bigr).        \tag{2.1}
\]
Consequently
\[
 \operatorname {OPT}_{\rm post}(A)
 =2\gamma+2s(N+K)
  -s\max_\sigma\bigl(I(\sigma)-J(\sigma)\bigr).           \tag{2.2}
\]

Formula (2.2) includes arbitrary interleaving, repeated transit through
served vertices, and every shortest-path shortcut.  The optimization is a
degree-two incidence packing embedded in one cyclic service order.  It is
not a matching of active line clients to point clients:

- a point client has degree two in the service order, but
- the same geometric point may occur as an uncalled transit vertex in
  any number of other metric moves.

The elementary position bound obtained from (1.6) is
\[
 \operatorname {OPT}_{\rm post}(A)
 \ge 2\gamma+2s\max\{N,K\}.                              \tag{2.3}
\]
Equality requires a closed alternating walk with exactly
\(\max\{N,K\}\) positions on each side.  With duplicate line clients this
is a genuine route-packing condition, not an automatic consequence of
incidence regularity.

## 3. The transit-reuse loophole

Let \(\ell_1,\ell_2\) be any two line clients and let
\(p=\ell_1\cap\ell_2\).  Shortest-path closure gives
\[
 d(\ell_1,\ell_2)
 \le d(\ell_1,p)+d(p,\ell_2)
 =\beta+\alpha=2s.                                      \tag{3.1}
\]
Equality follows from (1.5).

Nothing in (3.1) depends on whether the point client \(p\)

- has not yet been called,
- was called earlier,
- is currently the last served client, or
- is used only as an internal vertex of the expanded shortest path.

For a pencil \(\ell_1,\ldots,\ell_t\) through one point \(p\), the metric
route
\[
                         \ell_1,\ell_2,\ldots,\ell_t      \tag{3.2}
\]
has movement \(2s(t-1)\), witnessed by reusing \(p\) between every
consecutive pair.  The point is not called again.  The same is true for
duplicate clients of one geometric line.

Therefore a proof may not say that serving
\(p\to\ell\to q\) consumes either \(p\) or \(q\) as a future incidence
connector.  Service consumes an obligation, not a metric vertex or its
arcs.  Any ledger that extracts a matching of active lines to
unit-capacity points is not a lower bound on the stochastic-TSP route.

For a general design with flag-dependent weights, the exact induced
line-to-line distance is
\[
 d(\ell,m)=
 \min_{\ell=p_0,\;p_1,\ell_1,\ldots,p_k,\ell_k=m}
 \sum_{i=1}^{k}
 \bigl(b_{\ell_{i-1},p_i}+a_{p_i,\ell_i}\bigr),          \tag{3.3}
\]
where the minimum ranges over alternating incidence walks.  In particular,
every common point gives the reusable upper bound
\[
 d(\ell,m)\le b_{\ell,p}+a_{p,m}.                        \tag{3.4}
\]
Equations (3.3)--(3.4), rather than a matching constraint, are what survive
shortest-path closure.

## 4. Simple projective-plane instance has gap one

Assume there is one stochastic client on every geometric line.  A Singer
cycle gives enumerations
\[
 p_0,\ell_0,p_1,\ell_1,\ldots,p_{N-1},\ell_{N-1},p_0,
 \qquad p_i,p_{i+1}\in\ell_i.                            \tag{4.1}
\]

Use its fixed call order.  At \(p_i\), call \(\ell_i\), then call
\(p_{i+1}\).

- If \(\ell_i\) is active, the segment costs
  \(\alpha+\beta=2s\).
- If it is inactive, the call causes no movement and
  \(d(p_i,p_{i+1})=2s\).

The final segment is handled cyclically through \(p_0\).  This legal causal
policy costs
\[
                         2\gamma+2sN                    \tag{4.2}
\]
for every activation vector.

Every closed expanded walk visiting all \(N\) permanent points has at least
\(N\) point positions and hence at least \(2N\) incidence hops.  Equation
(1.6) gives the reverse lower bound.  Thus
\[
 \boxed{\operatorname {OPT}_{\rm adapt}
       =\operatorname {OPT}_{\rm post}
       =2\gamma+N(\alpha+\beta).}                        \tag{4.3}
\]
This proof already permits selector-first calls, arbitrary interleaving,
and transit through inactive line vertices.

The same statement holds for any balanced incidence design whose Levi graph
has a Hamilton cycle containing every permanent point and every possible
line client once.

## 5. Fixed marked detours never create a causal lower bound

The following observation covers the usual “permanent point backbone plus
independent line switches” proposal, including repeated point visits.

### Pending-walk lemma

Let \(W\) be a deterministic closed generating-graph walk based at the
depot.  Suppose \(W\) visits every permanent client and assigns every
stochastic line client \(x\) a marked occurrence
\[
                         p_x\to x\to q_x                 \tag{5.1}
\]
with \(p_x,q_x\) incident to \(x\).  A legal adaptive policy has movement
cost at most \(w(W)\) in every realization.

### Proof

Process the marked occurrences in their order on \(W\), calling every
permanent client at its first occurrence and calling \(x\) at its marked
occurrence.

Maintain the part of \(W\) since the last active call as a pending comparison
walk.  An inactive call causes no movement and leaves that walk pending.  At
the next active call, directed triangle inequality makes the actual metric
move no longer than the pending subwalk ending at that client.  After an
active line call, carry the marked suffix \(x\to q_x\) into the next pending
subwalk.  The final pending subwalk is paid by the depot return.  Every arc
occurrence of \(W\) is charged at most once. \(\square\)

The lemma allows \(W\) to transit through a stochastic line when it is
inactive and through a point after its client was served.  Hence any
realization-wise posterior **upper bound** obtained by taking one fixed
point backbone and independently choosing whether to “activate” its marked
line detours is automatically also a causal upper bound.  It cannot certify
a clairvoyance gap.

## 6. Duplicate line clients

Duplicate clients invalidate neither the potential identity nor the transit
lemma.  What they invalidate is the simplistic exact value (4.3): several
active copies of one line may be placed on several incidence adjacencies of
an optimized point cycle, while a fixed Singer cycle supplies only one
marked slot for that geometric line.

The correct posterior benchmark remains (2.2).  For example, two copies of
one line can occur in a service segment
\[
                         p_1,\ell^{(1)},p_2,\ell^{(2)},p_3
                                                                    \tag{6.1}
\]
through three points of that line.  Both line clients receive two incident
adjacencies.  Conversely, copies can also be served consecutively at cost
\(2s\) per line-to-line move by repeatedly transiting one already served
point.

Thus duplicates turn the problem into stochastic degree-two packing of
line copies into a cyclic point order.  They do not implement the proposed
online matching problem:

1. the “resources” have route degree two, not unit capacity;
2. their geometric vertices have unlimited transit reuse;
3. an arbitrary service order may join line clients directly at their
   intersection points; and
4. splitting or interleaving incidence steps is already included in
   (2.2).

No projective-plane lower bound above \(4/3\) follows from counting lost
point--line--point assignments.  A positive duplicate construction would
need a new lower bound for the complete stochastic cyclic-packing problem
(2.2), against a policy that probes all copies incident to its current point
and then transits through served points.  Incidence expansion alone is not
such a bound.

## 7. General-design verdict

For every connected incidence design with uniform two-way flag weights:

- the directed weights are a symmetric Levi metric plus a vertex potential;
- all closed-tour direction bias telescopes;
- served and inactive incidence vertices remain reusable in shortest paths;
  and
- a fixed marked-backbone posterior walk is reproducible causally by the
  pending-walk lemma.

For projective planes with one line client per line, the exact ratio is
\(1\).  With duplicate clients, the exact post-closure objective is (2.2),
not a matching objective, and the proposed capacity lower bound has a
concrete reusable-transit counterexample (3.2).

To escape this audit one must use flag-dependent nonzero circulation and
prove that all alternating bypasses in (3.3) remain expensive.  One must
also make the posterior's cheap walk depend globally on the realization;
otherwise the pending-walk lemma makes it a legal causal sweep.  Those two
requirements define a new route-code construction, not a consequence of
projective-plane or block-design incidence.
