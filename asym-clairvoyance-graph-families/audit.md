# Proof audit of the graph-family notes

## Scope and audit standard

This audit checks the completed notes against the stochastic-TSP model in
`problem-brief.md`.  In particular:

- a policy moves only when it calls an active client (apart from the final
  return to the depot);
- an inactive call causes no movement;
- a client passed through by an expanded shortest path is neither called nor
  served;
- every client is called exactly once by an adaptive policy;
- shortest-path walks may use inactive and uncalled client vertices as
  transit;
- activations are mutually independent; and
- local service bounds must survive arbitrary interleaving.

“Sound” below means that the stated result is justified at the level claimed.
“Needs repair” means that the conclusion is likely correct but the written
proof omits a model-specific step.  “Overstated” means that the displayed
calculation is valid only for an extra conditional template or a particular
candidate routing ledger, not for the stochastic-TSP optimum asserted or
suggested by the surrounding text.

## Executive findings

The following central results survive the audit:

1. the additive-replication inequality in `composition-principles.md`;
2. the exact \(8/7\) Cayley gadget in `cayley-digraphs.md`;
3. the exact ratio-one butterfly output-terminal construction in
   `switching-networks.md`;
4. the exact ratio-one projective-plane construction in
   `projective-planes.md`;
5. the potential-reweighting obstruction, the tree-tour bounds, and the
   conditional Hamiltonian ratio-one theorem in `generalized-polygons.md`;
6. the independent-codeword obstruction, Tanner potential obstruction, and
   conditional regular-design theorem in `codes-designs-ltc.md`;
7. the abstract product-choice permutation propositions in
   `algebraic-lifts.md`; and
8. the Ramanujan preferred-orientation divergence calculation, when explicitly
   restricted to an augmentation which retains every preferred edge trace;
   and
9. the four-star layered-poset construction, including its interleaved
   Bellman product lemma, positive directed-metric embedding, asymptotic
   liminf \(3/2\) lower bound, and explicit finite \(>4/3\) certificate.

There are two substantive qualifications.

- The algebraic-lift permutation theorem is correct, but the subsequent
  \(\tfrac74|\Gamma|w\) “posterior accounting” and \(8/7\) ratio are **not**
  bounds on the completed stochastic-TSP instance.  The displayed layer does
  not identify input and output ports, does not force every inactive preferred
  trace into an optimum tour, and supplies no adaptive upper bound of
  \(2|\Gamma|w\).  The only rigorous conclusion is that a particular
  all-preferred-traces circulation would have linear defects under additional
  port assumptions.
- The high-girth scaffold additivity proposition is credible and repairable,
  but its adaptive upper-bound proof must use the pending-walk/policy-gluing
  argument from Chapter 4.  A policy cannot literally follow a scaffold walk,
  voluntarily return from a local module, and then enter the next one.

The four-star layered-poset note now proves a ratio above \(4/3\).  Earlier
negative statements in this audit refer only to the original eight
off-the-shelf family templates and not to this switching/design hybrid.

## 0. Four-star layered poset

The complete proof and a \(120002\)-client witness are in
`four-star-layered-poset-construction.md`; an independently derived finite
certificate is in `four-star-poset-finite-certificate.md`.

- **Product law:** all \(4L\) selector clients have independent
  Bernoulli-\(q\) activations; the two gates per layer are permanent.
- **Posterior:** every set of at most two row/column stars has complementary
  \(K_{2,2}\) representatives, and
  \(\mathbb E W\le2+L(4q^3-2q^4)\).
- **Causal lower bound:** every deterministic local two-run strategy loses
  one exact-pair atom of mass \(h=q^2(1-q)^2\).  The product Bellman
  supermartingale, rather than a block-order assumption, proves
  \(\Pr[K\le2]\le(1-h)^L\) under arbitrary interleaving.
- **Metric:** \(d(x,y)=\varepsilon\) for \(x<y\), \(d(x,y)=1\) otherwise,
  and depot legs \(1/2\) form a positive directed metric.  A service order
  with \(N\) active clients and \(K\) increasing runs has exact cost
  \(\varepsilon N+(1-\varepsilon)K\).
- **Finite certificate:** \(q=1/100\), \(L=20000\), and
  \(\varepsilon=10^{-7}\) give
  \(3\operatorname{OPT}_{\rm adapt}
    -4\operatorname{OPT}_{\rm post}>0.24\).  The independent
  \(q=1/200,L=56000\) certificate separately gives
  \(3\operatorname{OPT}_{\rm adapt}
    -4\operatorname{OPT}_{\rm post}>0.10826\).

One wording qualification is important: the proof establishes
\(\liminf \operatorname{OPT}_{\rm adapt}/\operatorname{OPT}_{\rm post}
\ge3/2\).  Equality with \(3/2\) would require a matching adaptive upper
bound, which is not needed for the strict construction.

## 1. Shared diagnostics

### 1.1 Additive replication — sound

`composition-principles.md`, Section 1, is algebraically correct under its
explicit hypothesis that both optimum values genuinely decompose as
\(C+\sum_i P_i\) and \(C+\sum_i A_i\).  The note correctly warns that this
hypothesis needs a ports/interleaving proof.

### 1.2 Selector-first estimate — sound with one wording condition

Section 3's estimate
\[
 D\sum_{v\in S}p_v
\]
is valid if \(D\) bounds every ordered distance between the depot and selector
locations as well as between selectors.  In a fixed call order, movement
occurs only along the active subsequence; there are
\(\sum_v X_v\) such movements including the first depot-to-active movement.
The note explicitly excludes the subsequent transition and depot return.

Suggested clarification: say “the directed diameter of
\(S\cup\{r\}\)” every time this estimate is reused.  A diameter only within
\(S\) does not bound the first active call.

### 1.3 “Precisely” serving an active set — minor wording issue

Section 4 says that \(T_x\) must “visit precisely” the permanent and active
clients.  In the shortest-path model, a generating walk may pass through
inactive client locations.  The required property is:

> \(T_x\) explicitly serves every permanent and active client; its expanded
> paths may transit through any other vertex.

This does not affect the route-code diagnostic.

### 1.4 Static clients do not open or close arcs — sound and important

Section 6 is exactly consistent with the model.  Several speculative network
templates would otherwise accidentally switch to a stochastic-edge model.

## 2. Cayley digraphs

### 2.1 Exact \(8/7\) word-order gadget — sound

`cayley-digraphs.md`, Section 1, survives a complete call-order audit.

- The listed generators realize the needed length-one and length-two marked
  distances.
- Residual finiteness can simultaneously preserve marked-point distinctness
  and exclude the finite list of forbidden length-one equalities.
- The posterior lower bounds count positive metric legs between distinct
  marked clients, not service obtained by transit.
- Before the stochastic client \(s\) is called, the only observations are the
  deterministic outcomes of permanent clients, so the four placements of the
  \(s\)-call relative to \(b,h\) are exhaustive.
- If a shortest expanded path passes through an uncalled marked vertex, it
  does not serve it.  The leg-count lower bounds therefore remain valid.

The order \(h,b,s\) is a legal policy: the calls to \(h,b\) force the first two
movements, an inactive \(s\)-call leaves the policy at \(b\), and an active
\(s\)-call forces the \(b\to s\) movement before the legal final return.

### 2.2 Finite-group potential comment — harmless but should be sharpened

Section 2 says that a “homomorphism or weighted potential”
\(\phi:\Gamma\to\mathbb R\) can certify word-metric lower bounds.  A group
homomorphism from a finite group to \((\mathbb R,+)\) is necessarily trivial.
A nonconstant *vertex potential* satisfying
\(\phi(gs)-\phi(g)\le w_s\) can still be useful.

Suggested correction: remove “homomorphism” in the finite-group setting, or
say that a homomorphism may instead target an ordered quotient before passing
to a finite truncation.

No ratio or theorem depends on this sentence.

## 3. Switching networks

### 3.1 Butterfly output-terminal theorem — sound

`switching-networks.md`, Section 1, is correct.

- The only outgoing arc of an output is its return arc to \(r\).
- Every route from \(r\) to an output uses one \(\alpha\)-arc and exactly
  \(k\) layered arcs.
- Consequently every move between two distinct active output clients resets
  through \(r\) and costs \(C=\alpha+\beta+k\).
- Inactive calls do not change the current output, so an execution with \(q\)
  active calls has realized cost exactly \(qC\), independent of the call
  order.

This proof does not rely on physically traversing a chosen butterfly route
before an active call: the metric move forced by the active output call
already has the claimed exact distance.

### 3.2 Causal-prefix-tree observation — sound if tours mean client orders

Section 3's gap-one observation is correct when each \(T_x\) is interpreted as
a depot order of served clients and each realization-dependent branch calls
the relevant selector.  It should not be interpreted as permission to
voluntarily walk through nonclient switching vertices.  Metric movement
between the successive called active clients automatically shortcuts those
internal route vertices.

### 3.3 General Beneš statement — correctly conditional

The extension in lines 80–83 is valid only for terminal constructions with a
mandatory depot reset and equal depot-to-terminal cost.  The note states those
conditions, so it is not a theorem about arbitrary Beneš uses.

## 4. Projective-plane incidence graphs

### 4.1 Potential identity — sound

`projective-planes.md`, Section 2, correctly writes every directed incidence
weight as a symmetric weight plus a vertex-potential difference.  The identity
survives shortest-path minimization because every path between fixed endpoints
has the same potential difference.  It also holds policy-by-policy: a realized
adaptive call sequence begins and ends at \(r\), so all potential terms
 telescope.

### 4.2 Singer Hamilton cycle — sound

Section 3's orbit construction gives a Hamilton cycle alternating through all
points and lines.  The last translated line contains both
\(p_{N-1}\) and \(p_0\), so the cyclic closure is valid.

### 4.3 Exact posterior and adaptive values — sound

Sections 4–6 correctly handle inactive lines as transit vertices.

The adaptive policy does not voluntarily traverse
\(p_i\to\ell_i\to p_{i+1}\).  It calls \(\ell_i\), and:

- if active, the call moves to \(\ell_i\), after which the permanent call to
  \(p_{i+1}\) moves onward;
- if inactive, it remains at \(p_i\), and the permanent call to \(p_{i+1}\)
  uses a metric distance no larger than the two-arc incidence path.

At the last line, the model permits the final return to the depot.  The
expanded return may transit through the already-called \(p_0\).

The text twice says that an inactive point-to-point segment “costs”
\(\alpha+\beta\); a shortcut could make it smaller.  Replace “costs” by
“costs at most.”  The universal lower bound then forces equality for the total
tour, so the theorem is unchanged.

### 4.4 Expanded-walk lower bound — sound

A closed collection of incidence excursions based at \(p_0\) that contains all
\(N\) permanent point endpoints has at least \(N\) point-to-point steps and
therefore at least \(2N\) incidence arcs.  Passing through an uncalled point
does not itself serve it, but only strengthens the need for an explicit point
endpoint/service occurrence.  Multiple depot excursions do not help because
the unique depot interface is \(p_0\).

## 5. Generalized polygons

### 5.1 Potential obstruction — sound, but it is not by itself gap one

`generalized-polygons.md`, Section 3, correctly proves equivalence to the
symmetric companion metric.  This removes the *directed* effect; it does not
prove that an arbitrary non-Hamiltonian symmetric companion has
clairvoyance gap one.  The note respects this distinction in Sections 3 and
11.  The opening verdict should continue to keep the exact gap-one statement
conditional on Hamiltonicity.

### 5.2 Tree posterior bound — sound

Section 4's pruning leaves a tree containing every point.  Attaching each
active outside line by one incidence edge creates a connected subgraph.
Traversing every tree edge once in each direction is a valid posterior walk;
inactive line vertices may be transit vertices.

### 5.3 Universal alternating-walk lower bound — sound

Section 5 counts expanded incidence-arc occurrences.  A closed alternating
walk has equally many point and line positions.  Each permanent point and
active line must occur as a service endpoint somewhere in the expanded
execution, so
\[
 H\ge2\max\{N_P,K\}.
\]
The potential telescoping then gives (5.1)–(5.2).  Interleaving and passages
through uncalled vertices do not invalidate the count.

### 5.4 Conditional Hamiltonian theorem — sound

Section 6 uses a fixed sequence of actual client calls.  When a line is
inactive, the next permanent point call performs the point-to-point movement;
when active, the line call and point call split the same incidence segment.
The final return handles the last line.  As in the projective-plane note,
“costs” should be read as “costs at most” before applying the lower bound.

### 5.5 Selector-first bound — sound

Section 7's policy is legal.  After all active line calls have been served,
remove active lines from an optimal posterior client order and call the
remaining permanent points in the resulting order.  If \(z\) is the last
active selector and \(x\) the first permanent client, then
\[
 d(z,x)\le d(z,r)+d(r,x),
\]
so no voluntary repositioning is hidden in the proof.

The asymptotic statement after (7.4) needs the intended scale assumption made
explicit: one needs \(\gamma=O(\alpha+\beta)\) and bounded ratios among
\(\alpha,\beta,\gamma\), not merely an informal reference to “comparable”
edge scales.

## 6. High-girth lifts

### 6.1 Articulation-lift additivity — needs a policy-gluing paragraph

`high-girth-lifts.md`, Section 2, has a sound lower-bound decomposition.  Once
all metric moves are expanded, an excursion inside \(J_i-\{q_i\}\) starts and
ends at the unique articulation \(q_i\).  Concatenating all such excursions
gives a closed local service, while deleting them projects to a scaffold walk
visiting every permanent \(q_i\).

The written adaptive upper bound is incomplete.  The matching policy cannot
literally:

1. voluntarily finish a local tour by returning to \(q_i\);
2. walk along the scaffold to \(q_j\); and
3. start the next local service.

Only active calls cause movements.  The repair is the Chapter 4 policy-gluing
argument:

> Keep the local final path to \(q_i\), the next scaffold segment, and the next
> module's entry path as a pending generating walk.  Inactive calls leave it
> pending.  The next active client call shortcuts the whole pending walk in the
> metric.  Absorb the last pending suffix into the legal final depot return.

The same pending-walk argument realizes the scaffold walk despite revisits to
already-called \(q_i\)'s.  Articulation also ensures that internal module
walks cannot shorten scaffold distances.  With these sentences added, the
exact formulas
\[
 C(S_N)+NP_J,\qquad C(S_N)+NA_J
\]
are justified.

The lower-bound simulation should likewise mention that outside activations
are sampled as an independent seed and are revealed to the virtual ambient
execution only at its virtual calls.  This is the same causality argument as
Chapter 4's interrupted-services lemma.

### 6.2 Local port-coupled constants — illustrative, not proved for the
ambient lift

Section 3, lines 113–133, specifies only a triangle and vague “permanent work
behind” its ports.  The claims
\[
 P_{\rm local}=\tfrac32w,\qquad A_{\rm local}=2w
\]
are the Chapter 4 quotient constants only if:

- both port blocks are compulsory;
- entry and exit are given the same open-service relaxation as Chapter 4;
- outer lift arcs do not introduce a shorter trace; and
- extra pieces are charged at the parent boundary.

As written, these are not values of the displayed ambient lift cell.  The
surrounding text treats them as an uncoupled heuristic, but the wording
“locally” and “give exactly” should be replaced by an explicit conditional.

### 6.3 Ball packing statement — needs regularity

The assertion that radius-\(R\) balls have
\(\Theta((d-1)^R)\) vertices is valid for a \(d\)-regular lift below half the
girth.  For an arbitrary bounded-degree lift it is only an upper bound without
a minimum-degree assumption.  This does not affect the negative verdict.

## 7. Ramanujan bigraphs

### 7.1 Euler fallback tour and adaptive pending sweep — sound

`ramanujan-bigraphs.md`, Section 2, gives a valid realization-wise walk of
length
\[
 2|E|w+|A|w
\]
up to depot attachment.  It explicitly includes each active \(m_e\).

The adaptive implementation in lines 66–75 correctly uses a pending-walk
argument.  Permanent vertices are called at their first marked occurrences;
each midpoint is called at its marked reverse-edge occurrence.  The current
position need not literally be the marked tail \(v\): the metric move to the
next active client shortcuts the entire pending Euler segment.  Passing
through an uncalled client on that segment does not serve it, but that client
is still called at its own marked position later.

Suggested clarification: call this an upper bound for a fixed adaptive order,
not a statement that the policy physically executes the Euler walk.

### 7.2 Divergence correction — sound only for the retained-trace ledger

Lines 82–105 correctly calculate
\[
 b(x)=\pm(d-2A_x)
\]
for the directed multiset containing one locally preferred contracted trace
for **every** edge.  Any Eulerian augmentation which retains that entire
multiset must add at least
\(\frac12\sum_x|b(x)|\) units of directed correction, and its expectation is
\(\Omega(n\sqrt d)\) at \(p=1/2\).

This is not a lower bound on \(\operatorname{OPT}_{\rm post}\).  An optimum
tour may omit inactive edge traces, use a different orientation, or route
through other cells.  Section 2 recognizes the inactive-edge issue, but
Section 4's phrase “the balance correction increases the posterior
denominator” is too broad.

Suggested correction:

> The balance correction increases the cost of the all-edges
> independently-preferred candidate tour; it does not lower-bound the true
> posterior optimum.

### 7.3 Separated-edge accounting — explicitly conditional and therefore sound

Section 3 conditions the \(\tfrac32w,2w\) contributions on port boundaries
strong enough to prove exact additivity.  Under that hypothesis the
composition inequality gives at most \(4/3\).  It should not be cited later as
an analyzed construction until the duplicated-port backbone and its
ports/pieces proof are written.

### 7.4 “At most one matched edge per vertex” — heuristic, not a universal
charging theorem

Lines 143–149 explain why one perfect matching gives only \(n\) disjoint
edge conflicts.  They do not prove that no more sophisticated bounded-
congestion charging over several matchings exists.  The note correctly labels
the desired summation as unjustified, so no theorem needs retraction.

## 8. Algebraic lifts

### 8.1 Product-choice permutation propositions — sound

`algebraic-lifts.md`, Section 2, is correct as a statement about pointwise
choices between two permutations.

After relabeling outputs, an output \(h\) has indegree
\[
 (1-X_h)+X_{\pi^{-1}(h)}.
\]
Every nonfixed point is defective with probability \(2p(1-p)\), giving
\[
 \mathbb E D=2p(1-p)|\operatorname{supp}\pi|.
\]
On each permutation cycle, holes and collisions alternate in equal number.
Changing one input bit affects at most two defect indicators, so the stated
linear high-probability conclusion follows from bounded differences when
\(p\) is fixed and the support is linear.

These are combinatorial propositions.  They do not by themselves give a
stochastic-TSP cost.

### 8.2 Linear posterior repair claim — overstated

Section 3, especially lines 111–147, requires three assumptions not present in
the displayed voltage layer:

1. output \(O_h\) must be identified with a state/input vertex having one
   compulsory outgoing preferred trace, or an equivalent balance condition
   must be imposed;
2. every “hole” port must be compulsory for the tour, rather than merely an
   unused output of the chosen pointwise map; and
3. the optimum posterior tour must be forced to retain every locally preferred
   trace.

With distinct \(I_g,O_g\), a missing image of the pointwise map is not
automatically a divergence defect of a circulation.  Even after identifying
the layers, a stochastic-TSP optimum can omit an inactive trace unless
permanent port work and a boundary lower bound force it.

The conditional statement that a circulation *containing all preferred
traces* needs at least \(D/2\) additional unit entries is plausible once the
identified-port model is made explicit.  It is not a lower bound on
\(\operatorname{OPT}_{\rm post}\) for the template currently defined.

Suggested correction: retitle Section 3 “Conditional defect cost for a
retained-trace circulation,” define the identified port graph, and avoid the
word “posterior” until a tour-forcing lemma is supplied.

### 8.3 The \(8/7\) ratio is not a rigorous obstruction to the gap

Lines 130–147 divide an assumed adaptive leading term
\(2|\Gamma|w\) by the conditional retained-trace cost
\(\tfrac74|\Gamma|w\).  This does not bound the actual gap in either
direction:

- no adaptive upper bound of \(2|\Gamma|w\) is proved;
- an adaptive policy may pay the same repair costs;
- the posterior optimum may avoid the retained preferred traces; and
- common scaffold/depot terms are absent.

The rigorous conclusion is only:

> the naïve posterior upper bound
> \(\tfrac32|\Gamma|w\) cannot be obtained by blindly concatenating all
> independently preferred permutation traces in the identified-port ledger.

The displayed \(8/7\) should be labeled a non-rigorous bookkeeping ratio or
removed.  The Outcome and Verdict should not say that “the posterior pays” a
linear repair term without the retained-trace qualification.

### 8.4 Cheap correction-tour paragraph — incomplete construction

Lines 149–157 assert that cheap all-to-all connectors yield a tour of
\(O((D+|\Gamma|)w)\).  The connector endpoints, compulsory ports, component
joining rule, and depot attachment are not defined.  A fixed rail can provide
a coarse walk, but its adaptive implementation again needs a fixed client
order plus a pending-walk argument.  This paragraph should be treated as a
template suggestion, not a proved a-posteriori bound.

### 8.5 One global selector can use policy memory

Lines 169–177 say that one global client \(m_j\) “cannot remember” the incoming
sheet.  The graph vertex cannot remember it, but the policy can.  With fan-in
arcs \(I_g\to m_j\) and fan-out arcs \(m_j\to O_{ga_j}\), a posterior tour or
adaptive policy can choose the appropriate outgoing destination using its
known history.

The real obstruction is still serious: the same fan makes
\(I_g\to m_j\to O_h\) available for many \(g,h\), collapsing sheet distances
after shortest-path closure.  Suggested correction:

> A global selector is legal, but any fan-in/fan-out implementation must prove
> that policy memory cannot exploit the fan as a cheap arbitrary sheet
> switch.

The correlated-fiber warning in the next sentence is correct.

## 9. Linear codes, designs, and locally testable structures

### 9.1 Independent codeword proposition — sound

`codes-designs-ltc.md`, Proposition 1, correctly shows that mutually
independent nonconstant coordinates of a uniform linear-subspace word have
full-cube support.  If all \(n\) coordinates are nonconstant and independent,
the subspace is \(\mathbb F_2^n\).

The probability \(2^{k-n}\) in the independent-coordinate alternative is
correct when \(k=\dim C\).

### 9.2 Regular-design conditional theorem — sound

Theorem 2 is valid under its strong assumptions (P), (C), and (G).
Conditioned on a complete causal history at the cut, every uncalled fair
selector remains independent and fair.  If \(B_j\not\subseteq R\), its parity
is therefore unpredictable with error \(1/2\).  The regularity count
\[
 tF(R)\le\rho|R|
\]
and the endpoint minimization in \(z=\mathbb E|R|\) are correct.

At line 188, write
\(\mathbb E[M\mid\text{complete cut history}]\), not merely
\(\mathbb E[M\mid R]\).  The displayed lower bound depends only on \(R\), so
averaging yields the same theorem.  This is a precision correction, not a
gap.

The theorem does not cover progressive/interleaved commitments; the note
explicitly says so.

### 9.3 Tanner potential theorem — sound with depot-potential clarification

Section 4 correctly identifies uniform variable/check asymmetry as a vertex
potential.  If the depot is symmetrically attached to one fixed Tanner
vertex, set \(\phi(r)\) equal to that vertex's potential; the depot arcs then
also have the required symmetric-plus-potential form.  This small definition
should be stated explicitly when the fixed vertex is a check.

The theorem correctly claims equality with the symmetric companion values,
not gap one.

### 9.4 Bellman inequality has a mathematical typo and omits the boundary

Equation (5.3) is missing a plus sign.  It should read
\[
 \Phi(u,S)\le
 (1-p_v)\Phi(u,S\setminus\{v\})
 +p_v\bigl(d(u,v)+\Phi(v,S\setminus\{v\})\bigr)
\]
for every possible next call \(v\).

A full Bellman subsolution also needs the terminal condition
\[
 \Phi(u,\varnothing)\le d(u,r).
\]
Without the plus sign the displayed formula is not an inequality in the
model's dynamic program.  The surrounding discussion is speculative, so no
proved theorem is affected.

### 9.5 Cover-menu all-active argument — sound under the fair-bit context

Section 6.2 uses that the all-active realization has positive probability.
That is true under the Section 2 assumption \(p_i=1/2\).  For arbitrary
probabilities it requires every \(p_i>0\).  The fixed universal route can be
implemented by a fixed client call order; inactive calls disappear and metric
moves shortcut the corresponding route segments.

### 9.6 Cosmetic notation collision

The next lemma says “calling \(r\) systematic selectors,” while \(r\)
elsewhere denotes the depot.  Rename this count \(h\) or \(q\).

## 10. Cross-note conclusions

### Results safe to cite as complete

- Cayley word-order gadget: gap exactly \(8/7\).
- Butterfly output clients: gap exactly \(1\).
- Uniform projective-plane incidence instance: gap exactly \(1\).
- Uniform generalized-polygon incidence metric: equal to its symmetric
  companion; gap exactly \(1\) on a balanced Hamiltonian subclass.
- Uniform Tanner orientation: equal to its symmetric companion.
- Independent coordinates of a nontrivial proper linear codeword cannot be
  the activation law.
- Pointwise choices between two distinct permutations cannot form a
  permutation for every independent bit vector.
- Articulation lifts are additive after inserting the pending-walk gluing
  argument.

### Claims that should not be cited as stochastic-TSP bounds yet

- the algebraic-lift \(\tfrac74\) posterior term and \(8/7\) ratio;
- the coarse algebraic all-to-all repair tour;
- exact \(\tfrac32w,2w\) values for an ambient port-coupled high-girth cell;
- any posterior lower bound inferred from Ramanujan preferred-orientation
  imbalance; and
- any multi-matching adaptive charge without a bounded-congestion
  ports/pieces proof.

### Recommended proof hygiene for the comparison note

Every candidate should be assigned one of four labels:

1. **exact model theorem;**
2. **conditional theorem with all hypotheses displayed;**
3. **candidate routing ledger, not an optimum bound;**
4. **speculative next lemma.**

In particular, ratios formed from a candidate posterior *lower* cost and a
hoped-for adaptive contribution should never be presented as a bound on
\(\operatorname{OPT}_{\rm adapt}/\operatorname{OPT}_{\rm post}\).  A lower-gap
construction needs an adaptive lower bound divided by a posterior upper bound;
an upper obstruction needs an adaptive upper bound divided by a posterior
lower bound.

## 11. Finite buildings

### 11.1 Potential and selector-first lemmas — sound

`finite-buildings.md`, Sections 2.1–2.2, correctly handles shortest-path
closure and adaptive legality.  Potential differences telescope on every
realized closed call sequence.  In the selector-first policy, movement occurs
only along the active selector subsequence; after deleting those already
served selectors from an offline client order, the first permanent call
legally absorbs the transition from the last active selector.  When \(K=0\),
the proof should simply start the permanent order at the depot; the harmless
\(+\Delta\) slack in (2.5) still covers this case.

### 11.2 Radial spherical template — bounds are sound

Sections 3.1–3.5 correctly identify radial gallery weights as a potential.
The tree traversal in (3.2), expanded-walk count in (3.4), and fixed client
order in (3.6) respect the model:

- a depth-first walk may transit inactive chambers, but explicitly includes
  every active chamber;
- an expanded closed walk serving \(c_0\) and \(K\) distinct active chambers
  has at least \(K+1\) gallery arcs when \(K>0\); and
- the fixed-order policy does not voluntarily follow gallery edges: the next
  active chamber call shortcuts the pending gallery segment, and the legal
  final return supplies the last segment.

The last point is implicit rather than stated, but the metric-distance proof
of (3.6) is already sufficient.

### 11.3 Apartment cardinality and one-hot calculation — one independence
qualification

Section 4.1's cardinality obstruction and Chernoff bound are correct.  Section
4.3 also correctly maximizes
\[
t p(1-p)^{t-1}\le\tfrac12.
\]
The product bound \(2^{-k}\), however, requires the \(k\) panels to use
pairwise disjoint **selector-client sets**, not merely geometrically distinct
panels.  Chambers incident to different panels can overlap, so “disjoint
branch panels” should be replaced by “branch panels whose alternative
selector chambers are disjoint,” or independence should be proved separately.

The sentence that one Bernoulli bit “chooses” a continuation should also be
read only as a route label.  Activation does not enable or disable graph arcs;
an explicit service gadget must make the two outcomes favor different client
orders.

### 11.4 Retraction lower bound — valid idea, but normalize units and depot

Section 5's nonexpanding projection is sound for the symmetric gallery length
after the potential telescopes.  Equation (5.2) needs a convention:

- if \(\operatorname{TSP}_A(Q)\) uses unit gallery edges, the last term should
  be \(s\,\operatorname{TSP}_A(Q)\);
- if it already uses edge length \(s\), the displayed formula is correct.

The apartment TSP must also be based at the projected depot chamber
\(\rho(c_0)\).  These are normalization omissions, not failures of the
retraction diagnostic.  The note correctly warns that transit/service
multiplicity inside a retraction fiber is lost.

### 11.5 Permanent-apartment ratio-one conclusion — overstated unless \(P\)
is a lower scale

Equation (6.1) is valid:
\[
\operatorname{OPT}_{\rm adapt}
\le\operatorname{OPT}_{\rm post}+\Delta(\mu_S+1).
\]
Lines 450–459 then say that \(\mu_S=o(P)\) forces ratio \(1+o(1)\), where the
preceding paragraph describes a putative posterior tour “of cost \(P\).”  A
posterior **upper bound** \(P\) cannot be used in the denominator of this
ratio.  The conclusion requires
\[
\Delta(\mu_S+1)=o(\operatorname{OPT}_{\rm post}),
\]
or a separately proved lower bound
\(\operatorname{OPT}_{\rm post}=\Omega(P)\).  Suggested correction: replace
“\(\mu_S=o(P)\)” by the displayed condition involving the actual posterior
optimum, and then state any convenient sufficient lower-bound hypothesis.

### 11.6 Affine-quotient generic bounds — sound with the same scale condition

Equations (7.1)–(7.3) are valid provided every chamber adjacency has both
directed arcs with weights in
\([w_{\min},w_{\max}]\), as the gallery template intends.  Passing through an
uncalled chamber does not serve it, but cannot reduce the number of distinct
service endpoints needed for (7.2).

The selector conclusion around (7.4) additionally needs bounded
\(w_{\max}\), depot scale, and
\[
\operatorname{OPT}_{\rm post}=\Omega(P_N)
\quad\text{or directly}\quad
\left(\sum_{v\in S}p_v\right)D_Xw_{\max}
=o(\operatorname{OPT}_{\rm post}).
\]
If “post cost \(P_N\)” is only the cost of a proposed backbone tour, the
little-\(o\) claim does not follow.

### 11.7 Cocycle and hybrid sections — appropriately speculative

Sections 7.2–10 correctly distinguish a global potential from a quotient
cocycle, keep graph labels deterministic, allow inactive chambers as transit,
and refrain from claiming a causal lower bound.  The proposed fiber-charged
retraction inequality (11.1) is explicitly a next lemma, not an established
ports decomposition.

### Finite-buildings verdict

The complete potential no-go and generic spherical/quotient bounds are safe to
cite.  The two corrections needed before synthesis are:

1. qualify the \(2^{-k}\) one-hot probability by disjoint selector sets; and
2. replace the claims “\(\mu_S=o(P)\) implies ratio \(1+o(1)\)” and its affine
   analogue by conditions measured against a proved lower bound on
   \(\operatorname{OPT}_{\rm post}\).
