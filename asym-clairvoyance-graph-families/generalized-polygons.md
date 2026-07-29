# Generalized polygons

## Verdict

**No-go for the natural directed-incidence template; a positive
apartment-based construction remains speculative.**

Let the incidence graph of a finite generalized \(n\)-gon be directed by
giving every point-to-line flag arc length \(\alpha\) and every reverse arc
length \(\beta\).  Make all points permanent and make lines independent
Bernoulli clients.  Exactly as for a projective plane, this apparent
asymmetry is a vertex-potential reweighting of an undirected metric.  Hence
the directed instance has exactly the same a-posteriori and adaptive values as
its symmetric companion.  It contains no asymmetric route conflict.

For any balanced generalized polygon whose incidence graph has a Hamilton
cycle, the same activation-neutral Hamilton policy as in the projective-plane
case proves the stronger statement
\[
  \operatorname{OPT}_{\rm adapt}
  =
  \operatorname{OPT}_{\rm post}
  =
  2\gamma+N(\alpha+\beta),
\]
so the gap is exactly \(1\).  This statement is conditional only on the
Hamilton cycle and is proved below.

The most tempting alternative is to use orientations of apartments (the
girth cycles) as hidden route choices.  There is a basic limitation: by the
Feit--Higman theorem, a finite thick generalized \(n\)-gon has
\(n\in\{2,3,4,6,8\}\).  Thus, as the order grows, its girth \(2n\) is bounded
by \(16\); apartments do not supply a growing sequence of scales.  Independent
constant-size apartment switches can only replicate a local gap unless their
choices are coupled globally.  No such coupling lemma is currently proved.

## 1. Geometry

A finite generalized \(n\)-gon is a point-line incidence structure whose
incidence graph \(G=(\mathcal P\sqcup\mathcal L,E)\) is bipartite, connected,
has diameter \(n\), and has girth \(2n\).  A cycle of length \(2n\) is called
an apartment.  Write
\[
  N_P=|\mathcal P|,\qquad N_L=|\mathcal L|.
\]
The polygon is thick when every point lies on at least three lines and every
line contains at least three points.

For finite thick polygons, the Feit--Higman nonexistence theorem restricts
\(n\) to
\[
  n\in\{2,3,4,6,8\}.
\]
The \(n=3\) case is a projective plane.  The primary reference is
W. Feit and G. Higman, *The nonexistence of certain generalized polygons*,
Journal of Algebra 1 (1964), 114--131,
[DOI page](https://www.sciencedirect.com/science/article/pii/0021869364900286).

This restriction matters for the proposed application.  Generalized polygons
have high girth relative to their diameter and degree, but they are not a
family with \(n\to\infty\).  Increasing the field/order parameter increases
the number of vertices and degree while the diameter and apartment length
stay constant.

## 2. Natural directed metric and activation law

Fix a point \(p_0\), add a depot \(r\), and choose
\(\alpha,\beta,\gamma>0\).  For every flag \(p\in\ell\), add arcs
\[
  p\to\ell\text{ of length }\alpha,\qquad
  \ell\to p\text{ of length }\beta.
\]
Add \(r\to p_0\) and \(p_0\to r\), each of length \(\gamma\).  The generating
digraph is strongly connected.  Its directed shortest-path metric is \(d\).

All point vertices are permanent clients.  Every line vertex is independently
active with probability \(p\).  More generally, the arguments allow
independent line probabilities \(p_\ell\).

This is the natural rank-two analogue of orienting movement “up” and “down”
the incidence relation.  Orienting every edge away from or toward a fixed
root by its distance layer leads to the same obstruction: because every edge
joins consecutive distance layers, the directional bias is again a vertex
potential.

## 3. Complete potential-symmetry no-go

Let
\[
  s=\frac{\alpha+\beta}{2},\qquad
  \phi(r)=\phi(p)=0,\qquad
  \phi(\ell)=\frac{\alpha-\beta}{2}.
\]
Give every undirected incidence edge symmetric length \(s\), and the depot
edge symmetric length \(\gamma\).  Every generating arc satisfies
\[
  w(u,v)=w_{\rm sym}(\{u,v\})+\phi(v)-\phi(u).
\]
Therefore
\[
  d(u,v)=d_{\rm sym}(u,v)+\phi(v)-\phi(u).
  \tag{3.1}
\]
The identity is valid after shortest-path closure, because the potential
difference is the same for every \(u\)-to-\(v\) path before minimization.

For a closed depot route, all potential terms telescope.  More generally,
for every realized call sequence, its directed movement cost equals the
movement cost of the same sequence under \(d_{\rm sym}\).  It follows
policy-by-policy that
\[
  \operatorname{OPT}_{\rm post}(d)
  =\operatorname{OPT}_{\rm post}(d_{\rm sym}),\qquad
  \operatorname{OPT}_{\rm adapt}(d)
  =\operatorname{OPT}_{\rm adapt}(d_{\rm sym}).
  \tag{3.2}
\]

Equation (3.2) is a complete obstruction, not a missing lower-bound
technique.  The uniform flag orientation does not use asymmetric geometry at
all.  Any gap it has is exactly a symmetric stochastic-TSP gap.

The same proof applies to flag-dependent symmetric edge lengths
\(s_{p\ell}\) combined with any vertex potential.  To escape it, an
antisymmetric weighting must have nonzero circulation around at least one
cycle.

## 4. Explicit a-posteriori upper bound

Although (3.2) already settles the directed aspect, it is useful to record a
realization-wise tour bound.

Choose a tree \(T_0\) in the incidence graph that contains every point and a
set \(\mathcal L_0\) of \(\lambda\) line vertices.  Such a tree can be
obtained by taking a spanning tree of \(G\) and repeatedly deleting line
leaves.  It has
\[
  |E(T_0)|=N_P+\lambda-1.
\]

For a realization \(A\subseteq\mathcal L\), let
\[
  K_{\rm out}=|A\setminus\mathcal L_0|.
\]
For every active line outside \(T_0\), attach it by one incident edge to an
arbitrary point of \(T_0\).  The resulting connected subgraph contains all
permanent points and all active lines and has
\[
  N_P+\lambda-1+K_{\rm out}
\]
edges.  A depth-first closed traversal from \(p_0\) uses every tree edge once
in each direction.  Each such pair costs \(\alpha+\beta\).  Thus, for every
realization,
\[
  C_{\rm post}(A)
  \le
  2\gamma+(\alpha+\beta)
  \bigl(N_P+\lambda-1+K_{\rm out}\bigr).
  \tag{4.1}
\]
For common activation probability \(p\),
\[
  \operatorname{OPT}_{\rm post}
  \le
  2\gamma+(\alpha+\beta)
  \bigl(N_P+\lambda-1+p(N_L-\lambda)\bigr).
  \tag{4.2}
\]
No induced-subgraph connectivity is assumed: inactive line vertices may be
used as transit vertices without being served.

## 5. Attempted universal adaptive lower bound

Expand every metric move of an arbitrary adaptive execution into a shortest
generating-graph walk.  In a realization with \(K\) active lines, let \(H\)
be the total number of incidence-arc occurrences.  The expanded closed walk
alternates between point and line positions.

All \(N_P\) permanent points must occur, and all \(K\) active lines must
occur.  A closed alternating walk with \(H\) incidence arcs has \(H/2\) point
positions and \(H/2\) line positions, counted with multiplicity.  Therefore
\[
  H\ge 2\max\{N_P,K\}.
\]
Every depot tour also uses the unique depot edge in both directions.  By the
potential identity, its cost is at least
\[
  C(A)\ge
  2\gamma+(\alpha+\beta)\max\{N_P,K\}.
  \tag{5.1}
\]
This applies realization-by-realization to every causal policy, regardless of
remote calls, randomization, repeated depot visits, interleaving, or metric
shortcuts.  Consequently
\[
  \operatorname{OPT}_{\rm adapt}
  \ge
  2\gamma+(\alpha+\beta)\,
  \mathbb E[\max\{N_P,K\}].
  \tag{5.2}
\]

Combining this lower bound with the a-posteriori upper bound gives only
\[
  \frac{\operatorname{OPT}_{\rm adapt}}
       {\operatorname{OPT}_{\rm post}}
  \ge
  \frac{2\gamma+(\alpha+\beta)\mathbb E[\max\{N_P,K\}]}
       {2\gamma+(\alpha+\beta)
        (N_P+\lambda-1+p(N_L-\lambda))}.
  \tag{5.3}
\]
For a balanced polygon \(N_P=N_L\), the numerator's activation term
disappears: \(K\le N_P\), so it is just
\(2\gamma+(\alpha+\beta)N_P\).  The right side of (5.3) is then at most one
and is useless for proving a gap.

This failure is structural.  A closed alternating route already has
\(N_P\) line slots while serving the permanent points.  Up to \(N_P\) active
line clients can potentially be placed into those slots without increasing
the elementary counting lower bound.  Incidence regularity helps both
benchmarks pack terminals; it does not force the adaptive benchmark to choose
an expensive orientation.

## 6. Conditional exact value for Hamiltonian balanced polygons

Assume \(N_P=N_L=N\) and that the incidence graph has a Hamilton cycle
\[
  p_0,\ell_0,p_1,\ell_1,\ldots,p_{N-1},\ell_{N-1},p_0.
  \tag{6.1}
\]
No further property of a generalized polygon is needed.

Use the fixed adaptive call order
\[
  p_0,\ell_0,p_1,\ell_1,\ldots,p_{N-1},\ell_{N-1},
\]
followed by the final return to \(r\).  When \(\ell_i\) is active, the
segment through it costs \(\alpha+\beta\).  When it is inactive, its call
does not move the salesperson, and the direct metric move
\(p_i\to p_{i+1}\) costs at most the same two-edge incidence path.  The last
line and final return are handled identically.  Hence
\[
  C_{\rm adapt}(A)\le2\gamma+N(\alpha+\beta)
\]
for every realization.

The universal lower bound (5.1) gives the reverse inequality because all
\(N\) points are permanent.  Therefore
\[
  \boxed{
  \operatorname{OPT}_{\rm adapt}
  =
  \operatorname{OPT}_{\rm post}
  =
  2\gamma+N(\alpha+\beta)
  }
  \tag{6.2}
\]
and the gap is \(1\).

This is a **conditional lemma**, not a claim that every finite generalized
polygon is Hamiltonian.  It covers any chosen Hamiltonian balanced subclass
and explains why Hamiltonicity, usually an attractive routing feature, is
actively harmful for this lower-bound template: it produces an
activation-neutral service order.

## 7. Selector-first upper bound

The model permits all stochastic lines to be called before any permanent
point.  This gives a general audit that is independent of potential symmetry.

Let
\[
  \Delta=\max_{u,v\in\{r\}\cup\mathcal P\cup\mathcal L}d(u,v).
\]
Since the incidence graph has diameter \(n\), with
\(M=\max\{\alpha,\beta\}\),
\[
  \Delta\le \gamma+nM.
  \tag{7.1}
\]
Call every stochastic line in an arbitrary fixed order.  If \(K\) of them
are active, the movement through the active subsequence costs at most
\(K\Delta\).  The policy now knows the entire realization.

Take an optimal a-posteriori tour for that realization and call its permanent
points in their tour order, omitting the already served active lines.  If
\(z\) is the last active selector and \(x\) is the first permanent point in
the offline tour, triangle inequality gives
\[
  d(z,x)\le d(z,r)+d(r,x)\le\Delta+d(r,x).
\]
Thus no voluntary repositioning is required: the first permanent call moves
directly from \(z\) to \(x\), and the \(\Delta\) term only upper-bounds that
legal movement.  Shortcutting the already served line clients cannot
increase the rest of the offline tour.  Therefore
\[
  C_{\rm adapt}(A)
  \le C_{\rm post}(A)+(K+1)\Delta
  \tag{7.2}
\]
and
\[
  \operatorname{OPT}_{\rm adapt}
  \le
  \operatorname{OPT}_{\rm post}
  +\Delta(pN_L+1).
  \tag{7.3}
\]

Using the lower bound
\(\operatorname{OPT}_{\rm post}\ge
2\gamma+(\alpha+\beta)N_P\), this yields
\[
  \frac{\operatorname{OPT}_{\rm adapt}}
       {\operatorname{OPT}_{\rm post}}
  \le
  1+
  \frac{(\gamma+nM)(pN_L+1)}
       {2\gamma+(\alpha+\beta)N_P}.
  \tag{7.4}
\]
Because \(n\) is one of the fixed Feit--Higman values, sparse selectors
\(pN_L=o(N_P)\) give ratio \(1+o(1)\) when
\(\gamma=O(\alpha+\beta)\) and the positive edge scales have bounded ratios.
Therefore a route choice encoded by a small expected set of line selectors
cannot create a macroscopic gap.  A dense selector set avoids this asymptotic
conclusion, but its early probing cost is already on the scale of the whole
tour and must be incorporated into any lower-bound proof.

## 8. Apartments and why high girth is not enough

An apartment is a shortest cycle
\[
  p_0,\ell_0,p_1,\ell_1,\ldots,p_{n-1},\ell_{n-1},p_0.
\]
A natural proposal is to bias its clockwise and counterclockwise directions,
then let activations decide which orientation is cheap.  Four problems arise.

1. **Apartment length is constant.**  For finite thick polygons it is at most
   \(16\).  Isolated apartment gadgets are therefore constant-size switches.
2. **Additive replication cannot amplify.**  If the expected post and
   adaptive costs sum over apartments, the global ratio is at most the
   largest local ratio.  Replicating the Chapter 4 triangle switch on many
   apartments cannot exceed \(4/3\).
3. **Apartments overlap.**  If overlap is meant to create global coupling,
   arbitrary service interleaving can use one apartment to enter or leave
   another.  A proof needs ports and a charge for every extra service piece;
   expansion or girth alone supplies neither.
4. **Metric closure uses other apartments.**  A heavy clockwise arc can be
   bypassed through a different apartment in at most \(n\) incidence steps.
   Nonzero cycle circulation must survive all such paths.

The only possible advantage over the projective-plane case is that longer
apartments might support a richer local switch.  Since \(n\) is bounded, this
can improve a local constant only if one first proves a new single-apartment
gap above \(4/3\); no such calculation or universal lower bound is currently
available.  Even then, overlapping apartments would still require a global
charging lemma.

## 9. Failure audit

### Remote probing and calling a whole separator first

Equations (7.2)--(7.4) explicitly analyze the policy that calls the entire
line-selector set first.  Any smaller pencil, residue, or apartment boundary
has no larger diameter.  A separator cannot be treated as hidden until the
policy physically reaches it; calls are remote.

### Inactive calls as free information

Inactive line calls cause no movement.  In the Hamiltonian case, this makes
the active and inactive outcomes have exactly the same segment budget.  In
the general case, it makes the selector-first cost proportional to the number
of **active** selectors, not the total number queried.

### Arbitrary interleaving

The lower bound (5.1) is based on all expanded arc occurrences and survives
arbitrary interleaving.  By contrast, no apartment-local adaptive lower bound
has been proved.  One cannot assume that an apartment is served in one visit.

### Shortest-path shortcuts

For the uniform metric, (3.1) completely settles closure and proves the
potential no-go.  For a nonuniform cyclic orientation, diameter \(n\) gives
many short bypasses.  Every proposed long penalty needs a potential,
quotient, or scale-separation certificate.

### Random apartment labels

An apartment orientation, Weyl label, or chosen root is part of the fixed
metric and known to both benchmarks.  It is not legal hidden randomness.
Only independent client activations may select among deterministic routes.

## 10. Hybridization with the recursive triangle construction

The safest conceivable hybrid is to replace each flag or chamber transition
by a recursively scaled Chapter 4 block and use the generalized polygon as an
outer compatibility graph.  This does not help if the block costs remain
additive: the composition principle caps the ratio at \(4/3\).

To do real work, the outer polygon must ensure that selecting the cheap
orientation in one block constrains the orientations available in many other
blocks.  Since apartments overlap, a candidate mechanism is nonzero
circulation around every apartment subject to shared flag weights.  However,
the following all remain unproved:

- every activation vector has a globally compatible cheap a-posteriori tour;
- a causal policy must commit to apartment circulation before learning enough
  selector bits;
- selector-first probing costs at least the desired lower bound;
- interleaving through overlapping apartments cannot evade the charges; and
- directed metric closure preserves the intended scale penalties.

## 11. Status labels and next lemma

- **Complete proof:** uniform point-to-line/line-to-point asymmetry is a
  potential reweighting and gives exactly the symmetric companion instance.
- **Complete bounds:** (4.1)--(4.2), (5.1)--(5.2), and the selector-first
  upper bound (7.3) hold for every finite generalized polygon.
- **Conditional lemma:** if a balanced incidence graph has a Hamilton cycle,
  the natural stochastic instance has exact gap \(1\).
- **Complete obstruction:** finite thick generalized polygons do not have
  growing apartment length; \(n\in\{2,3,4,6,8\}\).
- **Speculation:** shared nonzero circulation on overlapping apartments might
  couple recursive triangle choices, but no adaptive lower bound currently
  supports this.

**Next lemma to prove or refute (apartment-circulation coupling lemma).**
Construct flag weights and independent selector clients on a fixed type
\(n\in\{4,6,8\}\) such that:

1. the weights have nonzero apartment circulation and a certificate that
   shortest-path closure preserves it;
2. every activation vector admits a tour of cost \(P_q\);
3. every cheap tour chooses a globally constrained system of apartment
   orientations rather than independent local detours;
4. calling all selectors first costs at least
   \((4/3+\varepsilon)P_q\); and
5. a port/extra-piece inequality proves the same lower bound for every
   interleaved causal policy.

Until such a lemma is available, generalized polygons should be regarded as a
no-go for the uniform incidence construction and only a speculative outer
coupling device for nonpotential recursive gadgets.
