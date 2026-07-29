# Finite buildings

## Verdict

**The natural gallery metrics are a no-go; apartments are a route menu but
independent activations do not select one compatible route.  Nonpotential
directions on finite affine quotients remain speculative.**

There are two substantially different objects under the heading “finite
buildings.”

1. A **finite spherical building** is genuinely a building.  Its apartments
   are finite Coxeter complexes.  Any two chambers lie in an apartment, but
   an arbitrary set of three or more chambers need not lie in one apartment.
2. A **finite quotient of an affine building** (including a Ramanujan
   complex) is a finite complex locally modeled on a building.  It is usually
   not itself a building: global apartments and building retractions need not
   descend to the quotient.

For spherical buildings, the most natural directed gallery metric—cheap
outward moves and expensive inward moves relative to a base chamber—is
exactly a vertex-potential reweighting of an undirected gallery metric.
Closed depot costs are therefore unchanged, policy by policy.  Type-dependent
versions have the same defect whenever their antisymmetric part integrates to
a well-defined chamber potential.

Apartments do give many deterministic galleries and orders, but a legal
stochastic instance must encode the chosen route by independent client bits.
Dense independent chamber activations almost never fit in one apartment.
Sparse route selectors can be called remotely before the permanent backbone
is served.  Independent “one active branch per panel” selectors fail with
constant probability at each branch.  Retractions can lower-bound the
projected length of a walk, but they collapse whole fibers of chambers and do
not charge the repeated or interleaved service that an adaptive lower bound
needs.

For finite affine quotients, a nontrivial quotient cocycle can create genuine
directed circulation, unlike a spherical radial potential.  This is the one
credible opening.  However, the algebraic label is part of the known metric,
not hidden randomness; vertex activations do not by themselves force a
homology/deck-transformation class; universal-cover retractions do not
automatically give quotient lower bounds; and short quotient paths can erase
the intended penalty.

No ratio above \(4/3\) is proved.  The complete results below are no-go and
diagnostic statements for the natural templates.

## 1. Background and notation

### 1.1 Spherical buildings

Let \(\mathcal B\) be a finite thick spherical building of Coxeter type
\((W,S)\).  Its maximal simplices are called chambers.  Two chambers are
adjacent when they share a panel; the panel has a type \(s\in S\).  The
chamber or gallery graph is denoted \(G_{\mathcal B}\).  Its distance is the
minimum gallery length.

An apartment is a subcomplex isomorphic to the Coxeter complex of \(W\).  It
has exactly \(|W|\) chambers.  The gallery diameter of an apartment, and of
the whole spherical building, is
\[
  D=\ell(w_0),
\]
where \(w_0\) is the longest element of \(W\).  Any two chambers are contained
in some common apartment.  This statement does **not** extend to an arbitrary
finite set of chambers.

For an apartment \(A\) and a chamber \(c\in A\), the building retraction
\[
  \rho_{A,c}:\mathcal B\longrightarrow A
\]
fixes \(A\), maps every apartment containing \(c\) onto \(A\), preserves Weyl
distance from \(c\), and is nonexpanding in gallery distance.

These standard facts are developed in J. Tits, *Buildings of Spherical Type
and Finite BN-Pairs*, Lecture Notes in Mathematics 386, Springer, 1974.

Rank-two spherical buildings are exactly the generalized polygons.  Type
\(A_2\) gives projective planes.  Thus the potential and bounded-apartment
obstructions in the two incidence-geometry notes are the rank-two cases of
the analysis below.

### 1.2 Finite affine quotients

An affine building is infinite.  If a discrete group \(\Gamma\) acts with a
finite quotient, then
\[
  X=\Gamma\backslash\widetilde{\mathcal B}
\]
is a finite complex locally modeled on
\(\widetilde{\mathcal B}\).  Ramanujan complexes are highly symmetric
expanding examples, commonly of type \(\widetilde A_d\).  The constructions
of Lubotzky--Samuels--Vishne are a primary source:
A. Lubotzky, B. Samuels, and U. Vishne, *Explicit constructions of Ramanujan
complexes of type \(\widetilde A_d\)*, European Journal of Combinatorics 26
(2005), 965--993,
[arXiv](https://arxiv.org/abs/math/0406217).

The quotient \(X\) generally has nontrivial topology.  A closed quotient walk
lifts to a path from a chamber \(\widetilde c\) to
\(\gamma\widetilde c\) for some \(\gamma\in\Gamma\); it need not lift to a
closed walk.  This distinction is precisely why a height gradient upstairs
can become nonzero circulation downstairs.

## 2. Two general lemmas used in both regimes

### 2.1 Potential-reweighting lemma

Let \(H\) be any connected undirected generating graph, with positive
symmetric edge lengths \(s_{\{u,v\}}\).  Suppose the directed arc lengths are
\[
  w(u,v)=s_{\{u,v\}}+\phi(v)-\phi(u)>0.
  \tag{2.1}
\]
Then shortest-path distances satisfy
\[
  d_w(u,v)=d_s(u,v)+\phi(v)-\phi(u).
  \tag{2.2}
\]
For every closed depot sequence \(r=v_0,\ldots,v_k=r\), the potential terms
cancel:
\[
  \sum_i d_w(v_i,v_{i+1})=\sum_i d_s(v_i,v_{i+1}).
  \tag{2.3}
\]

The current position and available observations of a policy depend only on
which called clients are active, not on the numerical distance paid.
Therefore (2.3) holds for every realization of every fixed adaptive policy.
It follows that
\[
  \operatorname{OPT}_{\rm post}(d_w)
  =\operatorname{OPT}_{\rm post}(d_s),\qquad
  \operatorname{OPT}_{\rm adapt}(d_w)
  =\operatorname{OPT}_{\rm adapt}(d_s).
  \tag{2.4}
\]
This is a complete asymmetric no-go, not merely a weak lower bound.

### 2.2 Selector-first lemma

Let \(S\) be all stochastic selector clients and \(B\) a nonempty permanent
backbone.  Let
\[
  \Delta=\max_{u,v\in\{r\}\cup S\cup B}d(u,v).
\]
Call all vertices of \(S\) in any fixed order.  If \(K\) are active, the
movement along the active subsequence is at most \(K\Delta\).  The policy now
knows the whole selector vector.

Let \(T_A\) be an optimal a-posteriori tour for the realization \(A\).  Remove
the already served selectors from its client order; triangle inequality does
not increase the remaining route.  If \(z\) is the last active selector and
\(b\) is the first remaining permanent client, make the legal call to \(b\)
directly.  The accounting inequality
\[
  d(z,b)\le d(z,r)+d(r,b)\le\Delta+d(r,b)
\]
shows that no voluntary repositioning is being assumed.  Consequently,
\[
  C_{\rm adapt}(A)\le C_{\rm post}(A)+(K+1)\Delta
  \tag{2.5}
\]
and
\[
  \operatorname{OPT}_{\rm adapt}
  \le
  \operatorname{OPT}_{\rm post}
  +\Delta\left(1+\sum_{v\in S}p_v\right).
  \tag{2.6}
\]

Thus a route menu indexed by selectors of small expected active cardinality
cannot be hidden merely by placing the selectors in different residues or
apartments.  Their calls are remote, and inactive selectors reveal their bits
without movement.

## 3. Spherical template I: radial directed galleries

### 3.1 Metric and activations

Fix a base chamber \(c_0\) and attach a depot \(r\) to it by two arcs of
length \(\gamma>0\).  Let
\[
  h(c)=d_G(c_0,c).
\]
Choose \(s>|\eta|\).  For every ordered pair of adjacent chambers, put
\[
  w(c,c')=s+\eta\bigl(h(c')-h(c)\bigr).
  \tag{3.1}
\]
Adjacent chambers may lie in the same distance layer because a thick panel
contains more than two chambers; those arcs have length \(s\).  Increasing
and decreasing arcs have lengths \(s+\eta\) and \(s-\eta\).

The chamber digraph is strongly connected, and its shortest-path closure is a
directed metric.  Make \(c_0\) permanent and make every other chamber
independently active with probability \(p\).  Write
\[
  M=|\operatorname{Ch}(\mathcal B)|,\qquad
  K\sim\operatorname{Bin}(M-1,p),\qquad
  \mu=p(M-1).
\]

### 3.2 Complete no-go

Equation (3.1) has the form (2.1) with \(\phi(c)=\eta h(c)\).
Consequently this directed instance is exactly equivalent, for both
benchmarks, to the symmetric chamber metric with every gallery edge of length
\(s\).  Cheap outward motion versus expensive inward motion creates no
closed-tour asymmetry.

The same conclusion holds for any “height” or “Busemann” direction that is a
globally defined real-valued function on the finite chamber set.  Type
dependent biases also fail whenever their antisymmetric increments are
consistent across the Coxeter braid relations and hence integrate to a
chamber potential.

To be genuinely directed, a weighting must have nonzero antisymmetric
circulation around some closed gallery.  A root-distance orientation does
not.

### 3.3 A-posteriori upper bound

For each active chamber \(x\), choose a shortest gallery from \(c_0\) to
\(x\).  The union of these galleries has a connected spanning subgraph with
at most \(DK\) edges.  It also has at most \(M-1\) edges after replacing it by
a subtree of a fixed full spanning tree when that is cheaper.  A depth-first
closed traversal uses every selected edge once in each direction.  The
potential cancels on each pair, whose total cost is \(2s\).  Hence for every
realization,
\[
  C_{\rm post}(A)
  \le
  2\gamma+2s\min\{DK,M-1\}.
  \tag{3.2}
\]
Taking expectations and using
\(\mathbb E[\min\{DK,M-1\}]\le\min\{D\mu,M-1\}\),
\[
  \operatorname{OPT}_{\rm post}
  \le
  2\gamma+2s\min\{D\mu,M-1\}.
  \tag{3.3}
\]
This bound is deliberately elementary; it is valid for all realizations and
does not assume that the active chambers lie in one apartment.

### 3.4 Universal adaptive lower bound

Expand an arbitrary adaptive execution into generating gallery arcs.  If
\(K>0\), a closed \(c_0\)-to-\(c_0\) walk visiting \(c_0\) and \(K\) other
distinct chambers has at least \(K+1\) gallery-arc occurrences: a closed walk
of length \(H\) has at most \(H\) distinct cyclic positions.  If \(K=0\), no
gallery arc is required.

The potential cancels on the expanded closed route, so every policy obeys
\[
  C(A)\ge
  2\gamma+s\bigl(K+\mathbf 1_{\{K>0\}}\bigr).
  \tag{3.4}
\]
Thus
\[
  \operatorname{OPT}_{\rm adapt}
  \ge
  2\gamma+s\bigl(\mu+\Pr[K>0]\bigr).
  \tag{3.5}
\]
This lower bound survives arbitrary interleaving and all shortest-path
shortcuts, but it does not separate the benchmarks.  The same inequality
holds for the a-posteriori tour.  Gallery expansion supplies a terminal-count
bound, not an information bound.

### 3.5 A fixed-order adaptive upper bound

Call \(c_0\), then all other chambers in an arbitrary fixed order.  Inactive
calls cause no movement.  Any two chambers are at gallery distance at most
\(D\), and every arc has length at most \(s+|\eta|\).  The active subsequence
and final return give
\[
  C_{\rm adapt}(A)
  \le
  2\gamma+D(s+|\eta|)
  \bigl(K+\mathbf 1_{\{K>0\}}\bigr),
  \tag{3.6}
\]
and therefore
\[
  \operatorname{OPT}_{\rm adapt}
  \le
  2\gamma+D(s+|\eta|)
  \bigl(\mu+\Pr[K>0]\bigr).
  \tag{3.7}
\]

The generic ratio information obtained from (3.5) and (3.7) is only a
diameter factor.  More importantly, (2.4) proves that any gap in this template
is a symmetric gap.  Spherical-building direction has contributed nothing
toward beating the asymmetric \(4/3\) construction.

## 4. Why an apartment menu is not automatically a route code

### 4.1 Size of one apartment

Every apartment contains \(|W|\) chambers, regardless of the thickness of the
building.  A necessary condition for all active chamber clients to lie in one
apartment is
\[
  K+\#\{\text{permanent chambers}\}\le |W|.
  \tag{4.1}
\]
For the template above, if \(\mu\ge2|W|\), the Chernoff lower-tail bound gives
\[
  \Pr[K\le |W|]
  \le \Pr[K\le\mu/2]
  \le e^{-\mu/8}.
  \tag{4.2}
\]
Thus a dense product activation law cannot be served inside one
realization-dependent apartment with high probability, even by cardinality.
When (4.1) holds, common-apartment containment still need not hold: the
building axiom guarantees a common apartment for two chambers, not for an
arbitrary active set.

### 4.2 Fixed rank versus growing rank

For fixed Coxeter type, \(|W|\) and \(D=\ell(w_0)\) are constants while the
number of chambers grows with the thickness parameter.  Hence:

- dense independent activations give \(K\gg |W|\) and cannot fit in one
  apartment;
- sparse selector activations have bounded selector diameter and are subject
  to (2.6).

Growing rank does not by itself reverse the conclusion.  For the type
\(A_r\) building of complete flags in \(\mathbb F_q^{\,r+1}\),
\[
  W=S_{r+1},\qquad
  |W|=(r+1)!,\qquad
  D=\binom{r+1}{2},
  \tag{4.3}
\]
while the number of chambers is
\[
  M=\prod_{i=1}^{r+1}\frac{q^i-1}{q-1}
   =q^{\,\binom{r+1}{2}+O(r)}
  \quad(q\text{ fixed}).
  \tag{4.4}
\]
Thus \(M/|W|\to\infty\) extremely quickly.  A single apartment occupies a
vanishing fraction of the chamber set.  If clients are deliberately
restricted to one apartment, the thickness and the rest of the building are
irrelevant; the candidate has reduced to a directed Cayley/Coxeter-complex
instance on \(W\).

### 4.3 Independent branch selectors

At a thick panel, suppose there are \(t\ge2\) alternative next chambers and
one tries to let independent Bernoulli clients select exactly one branch.
With common probability \(p\),
\[
  \Pr[\text{exactly one active branch}]
  =t p(1-p)^{t-1}
  \le \left(1-\frac1t\right)^{t-1}
  \le\frac12.
  \tag{4.5}
\]
For \(k\) branch panels whose alternative selector-client sets are pairwise
disjoint, the corresponding events are independent, so the probability that
every panel has a unique active continuation is at most \(2^{-k}\).  Zero
active branches and multiple active branches are not exceptional events; they
dominate asymptotically.

A single Bernoulli bit per step can legally choose between two deterministic
route labels by interpreting inactive/active as \(0/1\); it does not enable
or disable either branch arc.  The bit client can be called remotely before
entering that panel.  A successful construction must make the active selector
movement itself cost the target lower-bound scale; the combinatorial number
of galleries does not establish this.

### 4.4 Reduced words are known, not hidden

In a Coxeter apartment, an opposite chamber may be reached by many reduced
galleries corresponding to reduced words for \(w_0\).  The building and all
panel types are part of the fixed metric and known to both benchmarks.
Choosing a reduced word, a basis, an apartment, or a Weyl-group label at
random is not a legal source of uncertainty.  Independent client activations
must determine which route is useful, and every selector client must itself
be served when active.

## 5. Retractions: the useful lower bound and the missing charge

Fix \(A\) and \(c\in A\).  For the symmetric unit gallery metric, retraction
is nonexpanding:
\[
  d_A\bigl(\rho_{A,c}(x),\rho_{A,c}(y)\bigr)
  \le d_{\mathcal B}(x,y).
  \tag{5.1}
\]
Therefore the projection of any expanded tour is a walk in \(A\) of no
greater length.  Let \(\operatorname{TSP}_A(Q)\) use gallery edges of length
\(s\) and be based at the projected depot chamber \(\rho_{A,c}(c_0)\).
If the active chambers have projected images
\(Q\subseteq A\), then
\[
  \operatorname{cost}(T)
  \ge s\cdot\operatorname{length}(\rho_{A,c}(T))
  \ge \operatorname{TSP}_A(Q)
  \tag{5.2}
\]
for a potential-weighted metric after removing the telescoping potential.

This is a valid quotient lower bound, but it is not enough for a clairvoyance
gap:

- many distinct chambers lie in the same retraction fiber and have the same
  image;
- serving ten active chambers in one fiber may project to one apartment
  visit;
- chambers in neighboring fibers can share panels, so leaving and reentering
  a fiber may be cheap;
- an adaptive policy may interleave service from many fibers;
- choosing a different apartment or center after seeing the realization makes
  the lower-bound map itself realization-dependent.

The Chapter 4 proof succeeds because every interrupted block service is
charged at a port.  A building retraction has no analogous fiber charge for
free.  Retraction controls the coarse gallery word but erases the multiplicity
that an adaptive lower bound needs.

## 6. Spherical template II: permanent apartment plus selectors

A more route-code-oriented proposal is:

1. choose a base apartment \(A_0\);
2. make a backbone \(B\subseteq A_0\) permanent;
3. place stochastic selector chambers in roots, residues, or apartments
   meeting \(A_0\);
4. let the active selector vector choose a folded gallery or another
   apartment through \(B\).

For a realization \(A\), a putative a-posteriori tour would follow the
selected apartment and have cost \(P\).  This is only a proposal until four
points are proved.

First, independent active selectors may be mutually incompatible with a
single apartment.  Second, selectors not on the chosen route still have to be
served when active.  Third, the selector-first policy (2.5) learns the whole
route code before serving \(B\).  Fourth, a walk can switch apartments
through their overlapping residues rather than complete one apartment
service in a single visit.

If the selector set has expected active size \(\mu_S\), then
\[
  \operatorname{OPT}_{\rm adapt}
  \le \operatorname{OPT}_{\rm post}+\Delta(\mu_S+1).
  \tag{6.1}
\]
In fixed type, \(\Delta=O(1)\) in the natural gallery scale.  Hence the ratio
is \(1+o(1)\) whenever
\[
  \Delta(\mu_S+1)=o(\operatorname{OPT}_{\rm post}).
\]
A sufficient formulation using a proposed posterior scale \(P\) additionally
requires a proved lower bound
\(\operatorname{OPT}_{\rm post}=\Omega(P)\).
Dense selectors avoid this audit only when their active service is itself on
the scale of the true posterior optimum; the claim that one apartment gives
a cheap post tour must then include all active selector chambers.

No complete a-posteriori construction satisfying these obligations is known
for this template.

## 7. Finite quotients of affine buildings

### 7.1 Generic chamber-graph template and bounds

Let \(X\) be a finite quotient with \(N\) chambers and connected chamber
adjacency graph of hop diameter \(D_X\).  Attach the depot to a base chamber
\(c_0\) by symmetric length-\(\gamma\) arcs.  Assume every directed gallery
arc has length in
\[
  0<w_{\min}\le w(e)\le w_{\max}.
\]
Make \(c_0\) permanent and every other chamber independently active with
probability \(p\); again let \(K\sim\operatorname{Bin}(N-1,p)\) and
\(\mu=p(N-1)\).

Joining \(c_0\) to every active chamber by hop-shortest paths and traversing a
spanning tree in both directions gives the realization-wise post bound
\[
  C_{\rm post}(A)
  \le
  2\gamma+2w_{\max}\min\{D_XK,N-1\}.
  \tag{7.1}
\]
Every expanded closed route visiting \(c_0\) and \(K>0\) other chambers has at
least \(K+1\) arc occurrences, so both benchmarks satisfy
\[
  C(A)\ge
  2\gamma+w_{\min}
  \bigl(K+\mathbf1_{\{K>0\}}\bigr).
  \tag{7.2}
\]
A fixed call order gives
\[
  C_{\rm adapt}(A)
  \le
  2\gamma+D_Xw_{\max}
  \bigl(K+\mathbf1_{\{K>0\}}\bigr).
  \tag{7.3}
\]

These generic bounds yield only a diameter/condition-number comparison:
as \(\mu\to\infty\), they can differ by
\(D_Xw_{\max}/w_{\min}\).  Expansion or the Ramanujan property does not turn
(7.2) into an information lower bound.

For standard fixed-local-data expanding quotient families, the relevant graph
diameter is typically \(D_X=O(\log N)\).  Applying the selector-first lemma
shows that selectors can be learned at lower-order cost whenever
\[
 D_Xw_{\max}\left(1+\sum_{v\in S}p_v\right)
 =o(\operatorname{OPT}_{\rm post}).
 \tag{7.4}
\]
Equivalently, one may compare against a proposed backbone scale \(P_N\) only
after proving \(\operatorname{OPT}_{\rm post}=\Omega(P_N)\) and keeping the
edge/depot scales controlled.  Dense activations can evade (7.4), but then
they are not concentrated in one apartment and must be routed by both
benchmarks.

### 7.2 Gradient directions still fail

If the quotient admits a global height
\(\psi:\operatorname{Ch}(X)\to\mathbb R\) and
\[
  w(u,v)=s_{\{u,v\}}+\psi(v)-\psi(u),
  \tag{7.5}
\]
then the potential-reweighting no-go applies exactly.  A finite complex does
not become usefully asymmetric merely by coloring types or orienting edges
according to a globally defined height.

### 7.3 Quotient cocycles are genuinely different

Let \(b\) be a Busemann height on the universal affine building.  If \(b\)
does not descend to the quotient but its increments define a
\(\Gamma\)-invariant arc labeling, then a closed quotient loop may lift from
\(\widetilde c\) to \(\gamma\widetilde c\) and accumulate
\[
  b(\gamma\widetilde c)-b(\widetilde c)\ne0.
  \tag{7.6}
\]
This is nonzero circulation in the quotient and is not a vertex potential
there.  It is the most credible building-specific source of a genuine
directed metric.

It is not yet a stochastic lower bound:

- the cocycle and all algebraic edge labels are fixed and known to the
  adaptive policy;
- merely visiting a set of vertices does not prescribe the homology or deck
  class of the closed tour;
- a tour may combine positive and negative cycles or use a short relation in
  the quotient;
- shortest-path closure can replace a costly labeled gallery by a cheap word
  representing the same quotient displacement;
- different metric moves can lift to different apartment segments, so a
  single universal-cover apartment does not contain the execution;
- active chambers may be transited before being called, and inactive calls do
  not constrain the lifted path.

To use (7.6), activations would have to make different homology classes cheap
for the a-posteriori tour while forcing a causal policy to commit to a class
before learning enough bits.  No vertex-only construction proving this is
currently available.

### 7.4 Retractions do not automatically descend

An affine-building retraction onto an apartment is defined upstairs.  A
quotient chamber has many lifts, and changing the lift changes its apartment
coordinate by a deck transformation.  Unless the retraction is
\(\Gamma\)-equivariant—which would usually collapse the interesting
cocycle—it does not define a map \(X\to A/\Gamma\).

Projecting each individual lifted metric move is insufficient: successive
moves may choose incompatible lifts, and the endpoint displacement records a
deck element.  A valid lower bound needs a quotient-invariant norm or stable
length on this displacement plus a charge for switching lift/apartment
pieces.

## 8. Relation to projective planes and generalized polygons

The spherical rank-two cases add no new mechanism.

- Type \(A_2\) apartments are \(6\)-cycles in a projective-plane incidence
  building.  Uniform up/down weights are the same point/line potential
  already shown to have gap \(1\) in the Hamiltonian template.
- Other rank-two spherical buildings are generalized quadrangles, hexagons,
  and octagons.  Their apartment lengths are bounded, so independent local
  apartment switches cannot provide recursive depth.
- Links in affine buildings are spherical buildings, often projective planes
  or other finite buildings.  A local directed gadget placed only in a link
  inherits the same potential and selector-first obstructions.  Global
  quotient topology, not local building incidence, is the only new feature.

Thus “more apartments through every pair” should be viewed as a source of
shortcuts until a route-incompatibility lemma proves otherwise.

## 9. Hybridization with the Chapter 4 recursive triangle

One can place a recursive triangle block on every panel, chamber transition,
root, or quotient generator.  If the service costs of those blocks add, the
composition principle gives
\[
  \frac{C+\sum_i A_i}{C+\sum_i P_i}
  \le\max\left\{1,\max_i\frac{A_i}{P_i}\right\},
\]
so copies of the \(4/3\) block cannot improve on \(4/3\).

The building must therefore couple local orientations.  Possible deterministic
couplings include:

- braid relations between reduced galleries;
- consistency of roots across apartments;
- a quotient cocycle or deck-transformation constraint; and
- shared panels between many apartment routes.

Each also creates an escape route.  Braid relations give short alternative
words; overlapping apartments permit interleaving; retractions collapse
fibers; and inverse quotient cycles may cancel cocycle cost.  A hybrid proof
needs the analogue of Chapter 4's port lemma at the level of retraction fibers
or quotient homology.

## 10. Failure audit

### Remote early probing

Equation (2.6) explicitly covers the policy that calls every route selector
before serving a permanent flag or chamber backbone.  A selector is not
hidden behind a distant panel in this model.

### Calling a whole residue or separator first

A spherical residue has diameter at most \(D\), and the whole building also
has diameter \(D\).  Calling every client in a proposed separating residue is
therefore no harder than the selector-first estimate.  In an affine quotient,
replace \(D\) by \(D_X\).

### Independent activations

A random apartment, basis, Weyl chamber, reduced word, or deck element is not
a legal random realization.  Independent one-hot branch activations fail by
(4.5).  A single independent bit per branch is legal but remotely queryable.

### Inactive transit

An expanded shortest path may pass through an inactive or uncalled chamber.
Being on an apartment route does not mean that a chamber was called or
served.  Conversely, an inactive selector call gives information without
forcing the walk to enter its residue.

### Interleaving

No current argument permits an apartment, root, residue, retraction fiber, or
lift to be assumed served in one visit.  Retraction gives projected length but
does not charge extra fiber pieces.  The generic terminal count (3.4) or
(7.2) is the only universal lower bound established here.

### Shortest-path closure

Potential metrics are fully audited by (2.2) and are a no-go.  Nonpotential
apartment or cocycle weights need a stable-norm, quotient, or scale-separation
certificate.  High apartment overlap and Coxeter relations make destructive
shortcuts more likely, not less.

### Fixed rank and growing rank

Fixed rank has bounded apartment size and diameter.  Growing rank enlarges
the route menu, but the full chamber set grows much faster than one apartment,
as (4.3)--(4.4) show.  Restricting all clients to one apartment discards the
building thickness and reduces to a Coxeter/Cayley candidate.

## 11. Status and next lemma

- **Complete proof:** every globally potential-weighted gallery metric is
  exactly equivalent, for both benchmarks, to its symmetric companion.
- **Complete bounds:** (3.2)--(3.7) hold for the radial spherical template,
  and (7.1)--(7.3) hold for any bounded-weight finite quotient chamber graph.
- **Complete obstruction:** dense independent chamber activations cannot fit
  in one apartment once \(K>|W|\); (4.2) makes this overwhelmingly likely in
  the stated regime.
- **Complete obstruction:** independent one-hot panel choices succeed with
  probability at most \(2^{-k}\) across \(k\) branching panels whose
  selector-client sets are pairwise disjoint.
- **Complete selector audit:** (2.5)--(2.6) handles remote probing before the
  permanent backbone.
- **Conditional tool:** a fixed building retraction gives the projected
  lower bound (5.2), but only for projected images and without fiber
  multiplicity.
- **Speculation:** a nontrivial cocycle on a finite affine quotient might
  couple route choices through homology.  No activation construction or
  adaptive lower bound is known.

**Next lemma to prove or refute (fiber-charged retraction lemma).**  Find a
nonpotential directed weighting, an apartment \(A\), a retraction or
quotient-coordinate map \(\rho\), and a constant \(\lambda>0\) such that
every expanded service walk \(W\), after splitting its service events into
maximal pieces inside the fibers of \(\rho\), satisfies
\[
  \operatorname{cost}(W)
  \ge
  \operatorname{cost}_A(\rho W)
  +\lambda\sum_{a\in A}\bigl(N_a(W)-1\bigr)_+,
  \tag{11.1}
\]
where \(N_a(W)\) is the number of service-containing pieces in
\(\rho^{-1}(a)\).  The inequality must survive directed shortest-path closure
and, for affine quotients, changes of lift.

If (11.1) is false because adjacent residues allow zero-cost fiber switching,
then apartment retractions cannot support the required adaptive charging
argument.  If it is true for a carefully weighted quotient cocycle, it would
provide the missing analogue of the Chapter 4 port lemma; only then would it
be meaningful to search for an independent-bit route code on the apartment
or homology coordinate.
