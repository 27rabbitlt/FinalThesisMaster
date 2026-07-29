# Comparison and research recommendation

## Bottom line

A hybrid of the switching-network and design viewpoints gives a complete
construction above \(4/3\).  The four-star layered poset in
`four-star-layered-poset-construction.md` has
\[
 \mathbb E W_{\rm post}=2+o(1),\qquad
 \inf_\pi\mathbb E K_\pi\ge3-o(1),
\]
and its positive directed-metric embedding proves
\[
 \liminf
 \frac{\operatorname{OPT}_{\rm adapt}}
      {\operatorname{OPT}_{\rm post}}\ge\frac32.
\]
The finite certificate in `four-star-poset-finite-certificate.md` gives one
explicit strict instance without computation.

The key successful feature is not expansion or girth.  Two permanent gate
lanes force every two-run causal execution to make a local \(K_{2,2}\) edge
commitment.  Four independent row/column-star selectors make one exact
two-active pattern fatal for every local decision rule.  A Bellman product
supermartingale multiplies this loss across layers even under arbitrary
remote inactive probes.

## Status table

| Family | Strongest justified result | Main obstruction | Priority |
|---|---|---|---|
| Layered four-star switching/design hybrid | **Complete theorem:** positive asymmetric metric with gap liminf at least \(3/2\), plus an explicit finite \(>4/3\) witness. | No obstruction in the proved construction; the Bellman product is needed to handle interleaving. | **Solved positive direction.** |
| Projective-plane incidence graphs | Exact theorem: the uniform point/line directed template on \(\mathrm{PG}(2,q)\) has gap \(1\). | The asymmetry is a vertex potential; a Singer Hamilton cycle gives an activation-neutral order. | Stop for the uniform template. |
| Generalized polygons | Exact potential no-go; exact gap \(1\) on balanced Hamiltonian subclasses. | Uniform incidence direction is potential; finite thick apartments have bounded length by Feit--Higman. | Low. Use only if a nonpotential circulation is explicit. |
| Cayley digraphs | Exact finite Cayley-metric gadget with gap \(8/7\). | Noncommutativity separates complete words but does not defeat selector-first calls or translated local repairs. | Medium. Best candidate host for a proved route cell. |
| High-girth lifts | Exact articulation-lift additivity: a high-girth scaffold carrying independent Chapter 4 modules stays below \(4/3\). | Girth constrains local paths, not remote calls; port coupling also creates posterior circulation defects. | Medium-low until a base route cell exists. |
| Ramanujan bigraphs | Proved realization-wise Euler-sweep upper bound and a legal causal pending sweep with the same bound; exact divergence calculation for the all-preferred-traces ledger. | Shared endpoints destroy additive charges; preferred orientations are imbalanced; inactive edges need not be used. | Medium-low. Only the sparse \(p=\lambda/d\) star-routing version remains credible. |
| Linear codes, designs, LTCs | Exact legality/no-go results plus a conditional regular-design theorem with a sharp \(4/3\) threshold. | Code distance is combinatorial; the missing commitment and wrong-mode charges are directed-geometric. | **Highest.** This is the cleanest lower-bound framework. |
| Finite buildings | Exact potential no-go for radial galleries; apartment-size, selector, and retraction diagnostics. | Independent active chambers do not select one apartment; retractions collapse service multiplicity. | Medium-low. Affine quotient cocycles are the only distinctive opening. |
| Switching networks | Exact gap \(1\) for the natural butterfly/Beneš output-terminal template. | A static reusable fabric has no persistent hidden switch state; every metric traversal can choose a fresh route. | **High as a gadget question**, low as an off-the-shelf construction. |
| Algebraic lifts | Exact pointwise-permutation obstruction and exact random defect count. | Independent pointwise choices do not form a cycle cover; the \(7/4\) and \(8/7\) calculations are only a conditional retained-trace ledger, not optimum bounds. | Low for direct pointwise lifts; medium as a certificate inside another gadget. |

## Strongest complete positive calculations

The main construction proves
\[
 \liminf
 \frac{\operatorname{OPT}_{\rm adapt}}
      {\operatorname{OPT}_{\rm post}}\ge\frac32.
\]
Its smallest recorded finite certificate has \(120002\) clients and proves
\[
 3\operatorname{OPT}_{\rm adapt}
 -4\operatorname{OPT}_{\rm post}>0.24.
\]

An earlier, smaller positive calculation remains useful as a local control:
\[
 \operatorname{OPT}_{\rm post}=\frac72,\qquad
 \operatorname{OPT}_{\rm adapt}=4,\qquad
 \frac{\operatorname{OPT}_{\rm adapt}}
      {\operatorname{OPT}_{\rm post}}=\frac87
\]
for the finite Cayley word-order gadget.

The other exact calculations are no-go examples:

- projective-plane uniform incidence: ratio \(1\);
- butterfly output terminals: ratio \(1\);
- balanced Hamiltonian generalized-polygon uniform incidence: ratio \(1\).

These are useful controls: vertex transitivity does not force ratio \(1\)
(the Cayley gadget disproves that), whereas high path diversity does not
force a gap (the butterfly theorem disproves that).

## The sharp conditional route-code theorem

The code/design note gives the most useful quantitative target.  Let \(k\)
independent fair selector bits determine \(m\) parity route modes through a
regular design.  Suppose:

- every realization has a posterior tour of cost at most \(B\);
- querying a selector before route commitment costs \(a\) in expectation;
- a wrong route mode costs \(\Delta\);
- the baseline, early-query, and wrong-mode charges are disjoint after
  arbitrary metric expansion.

Then
\[
 \operatorname{OPT}_{\rm adapt}
 \ge B+\min\{ak,\Delta m/2\},
\qquad
 \frac{\operatorname{OPT}_{\rm adapt}}
      {\operatorname{OPT}_{\rm post}}
 \ge
 1+\frac{\min\{ak,\Delta m/2\}}{B}.
\]
Thus the construction beats \(4/3\) exactly when
\[
 \min\{ak,\Delta m/2\}>B/3.
\]

The single commitment cut in this theorem is deliberately strong.  An actual
construction will probably allow progressive/interleaved commitments.  The
right replacement is an LTC-style Bellman subsolution that tracks unresolved
test risk after every possible next call.

## Cross-family eliminators

These checks should be applied before developing another construction.

1. **Potential test.**  If
   \(w(u,v)=s_{\{u,v\}}+\phi(v)-\phi(u)\), the directed part telescopes on
   every closed realized call sequence.  It creates no asymmetric effect.
2. **Product-law test.**  Random codewords, lift signs, apartment choices,
   switch settings, and edge orientations are not hidden data.  Only
   independent client activations are.
3. **Inactive-transit test.**  Inactive clients and their incident arcs remain
   usable by shortest paths.  Activation is a service requirement, not edge
   availability.
4. **Selector-first test.**  The policy may remotely call every selector.
   The active subsequence must already cost the scale claimed in the lower
   bound.
5. **Additivity test.**  If post and adaptive costs decompose over modules,
   an outer graph cannot improve the largest local ratio.
6. **Flow-consistency test.**  Locally preferred directed traces need not form
   a circulation or tour.  Holes, collisions, and vertex divergence belong
   in the posterior accounting.
7. **Interleaving test.**  A block, apartment, fiber, line, or switch cell
   cannot be assumed served in one visit.  Every extra service piece needs a
   distinct or bounded-congestion charge.
8. **Metric-closure test.**  Expansion and high connectivity usually add
   repair paths.  Every claimed penalty needs a potential, quotient norm,
   scale separation, or port lemma.

## What made the successful construction work

### Local delayed route cell

The two permanent gates at every layer prevent an active future probe from
being harmless: jumping over an unserved gate leaves two incomparable gates
for the last run and certifies a third run.  This supplies the delayed
commitment that the natural butterfly and reusable-switch templates lacked.

### Second-order versus third-order scaling

The causal conflict occurs on an exact two-active event of probability
\(\Theta(q^2)\), while posterior width exceeds two only on three-active
events of probability \(\Theta(q^3)\).  Choosing \(Lq^2\to\infty\) and
\(Lq^3\to0\) separates the two objectives.

### Interleaving-safe repetition

The product proof uses the exact local Bellman success value after ordinary
queries, factor \(1-h\) for an untouched future stage, and the transition
\[
 q\cdot0+(1-q)\cdot1\le1-h
\]
for its first premature query.  The product of these factors is a
supermartingale, so cross-layer interleaving and free negative information
do not invalidate repetition.

## Overall recommendation

Use the four-star layered poset as the new baseline construction.  The most
valuable next questions are whether its \(3/2\) lower bound can be sharpened,
whether the client count in the finite witness can be reduced, and whether a
larger \(k\times k\) star interface produces more than three forced causal
runs while keeping posterior width near \(k\).
