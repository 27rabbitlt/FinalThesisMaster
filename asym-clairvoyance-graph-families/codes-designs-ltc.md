# Linear codes, combinatorial designs, and locally testable structures

## Executive verdict

**Proof status:** no construction above \(4/3\) is proved.  There are two
rigorous no-go statements for the most direct code/Tanner uses:

1. a nontrivial codeword cannot itself be the vector of independent
   activation bits; and
2. uniformly orienting a Tanner incidence graph only adds a vertex potential,
   so its apparent direction disappears from every closed tour.

A legal and potentially useful formulation is a fixed directed metric
containing a **route code** \(\{T_x:x\in\{0,1\}^k\}\), where \(x\) consists of
independent selector activations and a deterministic encoding \(c(x)\)
specifies route modes.  Code distance separates complete routes, but does not
force a causal policy to make mistakes.  A quantitative conditional theorem
below shows that a regular design plus two geometric inequalities would give
\[
 \frac{\operatorname{OPT}_{\rm adapt}}
      {\operatorname{OPT}_{\rm post}}
 \ge
 1+\frac{\min\{ak,\Delta m/2\}}{B}.
\]
This exceeds \(4/3\) exactly when the smaller of the early-probe and wrong-mode
penalties exceeds \(B/3\).  Proving those penalties in one shortcut-robust
directed metric remains the central open step.

## 1. What is and is not a legal code construction

Let \(C\le\mathbb F_2^n\) be a binary linear code.  The tempting proposal
“sample a uniform codeword and make coordinate \(j\) active iff its codeword
bit is one” is outside the model except in the trivial full-space case.

### Proposition 1 (independent linear-code coordinates are trivial)

Let \(Y\) be uniform on a linear subspace \(C\le\mathbb F_2^n\), and discard
coordinates that are identically zero on \(C\).  If the remaining coordinates
of \(Y\) are mutually independent, then their projection of \(C\) is the
entire Boolean cube.  In particular, if all \(n\) coordinates are nonconstant,
then \(C=\mathbb F_2^n\), which has minimum distance one.

### Proof

Every nonconstant linear coordinate is an unbiased Bernoulli random variable.
If there are \(m\) such mutually independent coordinates, all \(2^m\) binary
patterns occur with positive probability.  Their support is the coordinate
projection of \(C\), which is a linear subspace, so that projection must be
all of \(\mathbb F_2^m\).  If no coordinates were discarded, this says
\(|C|=2^n\) and hence \(C=\mathbb F_2^n\). \(\square\)

There are two legal alternatives.

* **Independent message bits.**  Let
  \(X=(X_1,\ldots,X_k)\) have independent coordinates, and use
  \(c(X)=GX\) only as a deterministic index for a route.  The stochastic
  clients are the \(X_i\)'s, not the coordinates of \(GX\).
* **Independent code-coordinate clients.**  Make the \(n\) coordinates
  independently active.  Then the realization is almost never a codeword:
  for fair bits,
  \[
        \Pr[X\in C]=\frac{|C|}{2^n}=2^{k-n}.
  \]
  For any constant-rate code with rate below one this is exponentially
  small, so posterior tours proved only for codewords say essentially nothing
  about the actual activation law.

Parity symbols, test outcomes, and syndromes may be deterministic functions
used by the analysis.  They may not be introduced as additional correlated
client-presence bits.

## 2. The route-code abstraction

Fix permanent clients \(P\), stochastic selector clients
\(S=\{s_1,\ldots,s_k\}\), and a fixed directed metric.  For clarity take
\(X_i\sim\operatorname{Bernoulli}(1/2)\), independently.  A route code consists
of depot tours
\[
       \{T_x:x\in\{0,1\}^k\}
\]
such that \(T_x\) serves \(P\cup\{s_i:x_i=1\}\).  A linear map
\[
       c:\mathbb F_2^k\longrightarrow\mathbb F_2^m
\]
can label \(m\) binary route decisions made by \(T_x\).

If the code has relative distance \(\delta\), then
\[
       x\ne x'\quad\Longrightarrow\quad
       d_H(c(x),c(x'))\ge\delta m.
\]
This proves that two *complete labels* differ in many coordinates.  It does
not establish any of the following geometric claims:

1. \(T_x\) is cheap for every \(x\);
2. a route-coordinate disagreement costs anything after directed
   shortest-path closure;
3. the disagreement is committed before the responsible bits are learned;
4. different disagreement charges survive arbitrary interleaving; or
5. calling all \(s_i\)'s first is expensive.

Thus minimum distance addresses route separation, but not causal
incompatibility.  Two far codewords may have tours with a long common prefix,
or a policy may wait to choose coordinate \(j\) until it has queried the few
message bits on which \(c(x)_j\) depends.

## 3. A quantitative conditional theorem

This theorem isolates a sufficient combinatorial/geometric package.  The
single commitment cut is restrictive, but it makes the missing metric
statement precise rather than hiding it in the phrase “large code distance.”

Let \(\mathcal B=\{B_1,\ldots,B_m\}\) be a \(t\)-uniform,
\(\rho\)-regular block design on \([k]\): every \(B_j\) has size \(t\), every
point belongs to \(\rho\) blocks, and hence
\[
       mt=\rho k.
\]
Set the target route bit
\[
       c_j(x)=\bigoplus_{i\in B_j}x_i.
\]

Assume a family of directed-metric instances has the following properties.

**(P) Posterior route code.**  For every \(x\), there is a depot tour \(T_x\)
serving the realized client set, of cost at most \(B\).

**(C) Commitment cut.**  Every deterministic adaptive execution has a
well-defined cut at which all \(m\) route modes have been irreversibly chosen.
Let \(R\subseteq[k]\) be the selectors called before that cut, and let \(M\)
be the number of chosen modes unequal to \(c_j(X)\).

**(G) Disjoint geometric charges.**  There is a nonnegative early-query charge
\(Q\) and constants \(a,\Delta>0\) such that, for every policy,
\[
 \mathbb E[\operatorname{cost}]
       \ge B+\mathbb E[Q]+\Delta\mathbb E[M],
 \qquad
 \mathbb E[Q]\ge a\,\mathbb E[|R|].
 \tag{3.1}
\]
The baseline and the two kinds of charges in (3.1) must be backed by disjoint
arc occurrences, or by an equivalent Bellman certificate; they cannot count
the same movement twice.

### Theorem 2 (regular-design route-code bound)

Under (P), (C), and (G),
\[
 \operatorname{OPT}_{\rm post}\le B,\qquad
 \operatorname{OPT}_{\rm adapt}
 \ge B+\min\{ak,\Delta m/2\}.
 \tag{3.2}
\]
Consequently
\[
 \frac{\operatorname{OPT}_{\rm adapt}}
      {\operatorname{OPT}_{\rm post}}
 \ge
 1+\frac{\min\{ak,\Delta m/2\}}{B}.
 \tag{3.3}
\]
In particular, the ratio is strictly above \(4/3\) if
\[
       \min\{ak,\Delta m/2\}>B/3.
 \tag{3.4}
\]

### Proof

Fix the information at the commitment cut.  If \(B_j\not\subseteq R\), then
at least one independent fair bit in the parity \(c_j(X)\) is unqueried.
Conditional on the complete causal history, that parity is still fair.
Whatever mode the policy chose, its conditional mismatch probability is
\(1/2\).

Let \(F(R)=|\{j:B_j\subseteq R\}|\).  Counting incidences between \(R\) and
the blocks contained in \(R\) gives
\[
       tF(R)\le \rho|R|,
       \qquad
       F(R)\le \frac{\rho|R|}{t}
             =\frac{m|R|}{k}.
\]
It follows that
\[
 \mathbb E[M\mid\text{complete cut history}]
 \ge \frac12\bigl(m-F(R)\bigr)
 \ge \frac m2\left(1-\frac{|R|}{k}\right).
\]
Writing \(z=\mathbb E|R|\) and applying (3.1),
\[
\begin{aligned}
 \mathbb E[\operatorname{cost}]
 &\ge B+az+\frac{\Delta m}{2}\left(1-\frac zk\right)\\
 &\ge B+\min\{ak,\Delta m/2\},
\end{aligned}
\]
because the affine expression in \(z\in[0,k]\) is minimized at an endpoint.
This holds for every deterministic policy.  Conditioning on a private random
seed proves it for randomized policies.  Property (P) gives the posterior
bound, proving (3.2)--(3.4). \(\square\)

### Interpretation

The two endpoints in (3.2) are exactly the two dangerous strategies.

* Query all \(k\) selectors before committing: the intended lower bound is
  \(ak\).
* Query none: at least \(m/2\) parity modes are wrong in expectation, with
  intended penalty \(\Delta m/2\).

Regularity prevents a small queried set from completely determining too many
blocks.  It does **not** prove (C) or (G).  Those are directed-geometric
statements and are substantially stronger than code distance.

For biased independent selectors, an undetermined parity is not necessarily
fair.  The factor \(1/2\) must be replaced by a uniform lower bound on the
conditional error of predicting that parity; for large blocks this tends to
\(1/2\) when \(p\) is bounded away from \(0\) and \(1\), but a construction
must use the exact value.

## 4. Tanner graphs: a rigorous natural-template obstruction

Let a Tanner graph have variable vertices \(V\), check vertices \(C\), and
incidence edges \(v\sim c\).  Give every incidence the two directed arcs
\[
       v\to c\text{ of length }\alpha,\qquad
       c\to v\text{ of length }\beta,
       \qquad \alpha,\beta>0.
\]
Attach a depot by any symmetric connection to a fixed vertex.  This is the
most direct attempt to turn the variable/check distinction into asymmetric
routing.

Set
\[
 q=\frac{\alpha+\beta}{2},\qquad
 \phi(v)=0,\qquad
 \phi(c)=\frac{\alpha-\beta}{2}.
\]
Set the depot potential equal to the potential of the Tanner vertex to which
it is symmetrically attached.
For every incidence arc,
\[
       w(u,v)=q+\phi(v)-\phi(u).
\]
Therefore every directed path satisfies
\[
       w(W)=q|W|+\phi(\operatorname{end}W)
                       -\phi(\operatorname{start}W),
\]
and shortest-path closure gives
\[
       d(u,v)=d_{\rm und}(u,v)+\phi(v)-\phi(u),
 \tag{4.1}
\]
where each undirected Tanner edge has length \(q\).  Around every closed depot
tour the potential telescopes.  Hence every fixed call sequence has exactly
the same realized cost as in the symmetric companion metric.

This proves:

### Proposition 3 (uniformly directed Tanner graphs have no directed effect)

For arbitrary permanent/stochastic choices and arbitrary independent
activation probabilities, the a-posteriori and adaptive optimum values of the
uniformly oriented Tanner construction are respectively identical to those
of its symmetric Tanner metric.

The proposition does not claim that every symmetric Tanner instance has gap
one.  It says that coding incidence plus unequal variable-to-check and
check-to-variable weights creates no directed circulation and therefore no
new asymmetric order conflict.  Any gap obtained this way was already present
in the underlying symmetric metric.

There is a second obstruction.  A client's inactivity does not delete its
Tanner vertex or incident arcs.  Expanded shortest paths may pass through an
inactive or uncalled variable, and such transit neither calls nor serves it.
Thus parity checks do not constrain the realized active set, and “failed
checks block a route” is not a legal interpretation of a Tanner graph in this
model.

To escape Proposition 3 one needs nonzero circulation around Tanner cycles,
for example edge-dependent directed weights.  But then local check routes may
concatenate into shorter repairs.  A cycle-basis potential/quotient argument
or a ports-and-pieces charge is required to show that the intended violation
penalty survives metric closure.

## 5. Locally testable structures and a Bellman route

The possible value of a locally testable code (LTC) is not that its words are
far apart; an ordinary code already provides that.  Its value would be to
turn a globally inconsistent route label into many **local witnesses** that
could potentially be charged to separate directed detours.

A legal construction would use the graph code
\[
       \mathcal C=\{(x,c(x)):x\in\mathbb F_2^k\}.
\]
Only the systematic coordinates \(x_i\) are activation bits.  The route
coordinates \(c(x)_j\) are planner choices.  Suppose a bounded-query tester
has robustness
\[
 \Pr[\text{test rejects }z]
       \ge \tau\,\frac{d_H(z,\mathcal C)}{k+m}.
 \tag{5.1}
\]
To obtain a stochastic-TSP lower bound one would still need a geometric
realization with:

1. a local directed cell for every test;
2. a detour of at least \(\Delta\) whenever that test rejects;
3. charge overlap at most \(L\), so one arc occurrence pays for at most
   \(L\) rejecting tests; and
4. an early-query charge for learning the systematic coordinates before
   reaching their natural service positions.

If these held, \(R\) rejecting local witnesses would yield extra cost at least
\(\Delta R/L\).  Robustness (5.1) would then convert distance from the legal
route code into metric cost.

The policy-uniform way to prove this is a Bellman subsolution.  For the usual
adaptive value \(F(u,S)\), one seeks a potential of the form
\[
 \Phi(u,S)=B_0(u,S)
       +\lambda\cdot
        \bigl(\text{unpaid forced-test violations or unresolved-test risk}\bigr)
 \tag{5.2}
\]
satisfying, for every possible next call \(v\in S\),
\[
 \Phi(u,S)\le
 (1-p_v)\Phi(u,S\setminus\{v\})
 {}+
 p_v\bigl(d(u,v)+\Phi(v,S\setminus\{v\})\bigr).
 \tag{5.3}
\]
Together with the terminal boundary condition
\[
 \Phi(u,\varnothing)\le d(u,r),
 \tag{5.4}
\]
this would be a complete Bellman lower-bound certificate.
Local testability can estimate the second term in (5.2), but it cannot prove
(5.3): the metric must guarantee that resolving or violating a test is paid
for by the active movement \(d(u,v)\).  Equation (5.3), including calls made
far out of the intended Tanner order, is the exact missing bridge from an LTC
to a universal adaptive lower bound.

## 6. Block and projective designs as route menus

Designs have two distinct prospective uses.

### 6.1 Parity-decision designs

The blocks \(B_j\) in Theorem 2 are local dependency sets.  Regularity gives
the clean query-versus-uncertainty tradeoff, while stronger parity expansion
can make \(x\mapsto(c_j(x))_j\) an error-correcting code.  Projective and
block designs offer uniform degrees and limited pairwise overlaps, which are
useful for bounding how often one boundary arc can be charged.

Limited overlap alone is insufficient.  It does not locate a commitment cut,
make a wrong parity mode expensive, or stop the policy from remotely calling
all points of a block before using its route cell.

### 6.2 Cover menus

Another proposal associates each block \(B\) with a cheap tour serving the
selectors in \(B\), and lets the posterior choose a block or a small block
cover containing the active set.  There are two elementary obstructions.

* A realization-wise single-block guarantee for every active set forces the
  menu to contain a block covering \([k]\), since the all-active realization
  has positive probability.  That universal route is then available to the
  adaptive benchmark.
* If only typical realizations are covered, all exceptional-event costs must
  be included.  For constant activation probability, a typical active set has
  \(\Theta(k)\) points, so small blocks cannot cover it with \(o(k/t)\)
  pieces.  If these pieces have additive service cost, the
  additive-replication principle prevents ratio amplification.

Overlapping blocks could in principle couple the route pieces.  But high
overlap also gives cheap transit/repair paths after metric closure.  A
positive proof needs a boundary lemma saying that inconsistent covers incur
distinct directed crossings even under arbitrary interleaving.

## 7. Failure audit

**Selector-first probing.**  If the selector set has directed diameter \(D\),
calling it in a fixed order costs at most
\[
       D\sum_i p_i
\]
in expected movement before the now-known route is started.  In Theorem 2
this is precisely the charge \(ak\) that must exceed \(B/3\); putting message
bits in a compact systematic part defeats the proposal.

**Inactive clients as transit.**  Neither an inactive variable vertex nor a
failed parity coordinate removes any graph arc.  A shortest path may use it
without revealing or serving it.

**Interleaving.**  A policy can query variables from many checks, commit some
route modes, and return later.  The single-cut theorem does not cover this.
An LTC proof must charge every extra service piece and maintain (5.3) after
every possible call.

**Shortest-path closure.**  Dense Tanner/design incidence gives many short
alternating paths.  A drawn wrong-mode penalty is invalid unless a potential,
scale separation, or quotient count proves that no chain of neighboring
tests repairs it cheaply.

**Correlated derived symbols.**  Codeword, syndrome, and test bits derived
from \(X\) are analysis labels only.  Declaring them active with their
correlated law violates the product-activation assumption; declaring them
independently active destroys the code constraints.

**Additive local tests.**  Placing an independent Chapter 4 triangle at every
check yields at most the largest local ratio when costs decompose.  An LTC is
useful only if violations couple many route decisions while their charges do
not overlap excessively.

## Verdict

**Conditional, with rigorous natural-template no-go results.**  Code distance,
design regularity, and local testability can respectively supply complete-word
separation, query-versus-uncertainty counting, and local inconsistency
witnesses.  None supplies the required directed-geometric penalty.  Uniform
Tanner orientation is only a potential reweighting of a symmetric metric, and
correlated codeword activations are illegal.  Theorem 2 gives exact leading
ratio algebra if a shortcut-robust commitment/charging gadget is found, but
no ratio above \(4/3\) is established here.

## Exact next lemma

Prove or refute the following **interleavable LTC route-cell lemma** for one
explicit bounded-query graph code.

There should be a finite strongly connected directed metric with independent
fair systematic selector clients, permanent port blocks, and one local cell
per tester constraint such that:

1. every \(x\) has a mode-consistent posterior depot tour of cost at most
   \(B\);
2. after expanding any causal execution, rejecting tester constraints can be
   assigned directed boundary-arc occurrences with overlap at most \(L=O(1)\),
   even when cell services are split and interleaved;
3. calling \(h\) systematic selectors before their natural cells creates
   expected charge at least \(ah\), disjoint from the violation charges; and
4. the resulting Bellman subsolution proves
   \[
      \operatorname{OPT}_{\rm adapt}
      \ge B+\min\{ak,\Delta\tau m/(2L)\},
   \]
   with the minimum strictly larger than \(B/3\).

Items 2 and 3 are the genuinely geometric content.  If they cannot hold for
bounded-degree Tanner metrics because alternating paths give cheap repairs,
that failure would constitute a useful stronger no-go theorem for the whole
LTC approach.
