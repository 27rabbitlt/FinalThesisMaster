# Algebraic lifts

## Outcome

Algebraic (voltage) lifts are the most natural way to make a local orientation
choice change a global route state.  They also expose a sharp obstruction:
if independent activation bits choose pointwise between two voltage
permutations, the preferred traces fail to be a permutation/cycle cover at a
linear number of fibers.  Avoiding all such defects for every bit vector
forces the two choices to be identical.

Thus a fixed voltage assignment can couple local choices, but independent
client activations do not supply a free random algebraic lift.  In the
canonical binary switch, the candidate ledger that retains every locally
preferred trace has a linear defect count.  Turning that ledger into a bound
on the posterior optimum would require mandatory identified ports and a
tour-forcing lemma; making repairs cheap would also introduce sheet-changing
shortcuts for the adaptive policy.

## 1. Voltage-lift template

Let \(\Gamma\) be a finite group and let a base arc \(e:u\to v\) have voltage
\(\lambda_e\in\Gamma\).  The lifted arc is
\[
 (u,g)\longrightarrow(v,g\lambda_e).
\]
All voltages and arc weights are part of the known, deterministic metric.
Random voltages, random signs, or a random lift are not legal hidden inputs.
The only hidden data are independent client-activation bits.

A canonical pointwise voltage layer has, for each \(g\in\Gamma\), an input
port \(I_g\), an output port \(O_g\), a stochastic client \(m_g\), and traces
\[
 I_g\longrightarrow O_g
\quad\text{and}\quad
 I_g\longrightarrow m_g\longrightarrow O_{\pi(g)},
\]
where \(\pi\) is the permutation of sheets induced by a nontrivial voltage.
The first trace has length \(w\), the second \(2w\), and
\(X_g=\mathbf 1_{\{m_g\text{ active}\}}\) are mutually independent
Bernoulli(\(p\)) bits.  A fixed strongly connected Cayley rail or sufficiently
expensive reverse arcs complete the generating graph; distances are its
directed shortest-path distances.

The intended posterior behavior is to take the identity trace when \(X_g=0\)
and the \(\pi\)-trace when \(X_g=1\).  The next proposition identifies the
global defect in this plan.  By itself this common-input layer has no
clairvoyance gap: a policy can call \(m_g\) at \(I_g\).  The hope would be to
graft an orientation-conflict gadget onto each connection while retaining the
algebraic output coupling.  The obstruction below applies before that harder
adaptive issue is reached.

## 2. Product-choice permutation obstruction

Let \(\sigma_0,\sigma_1\) be permutations of a finite set \(\Omega\).  For
\(x\in\{0,1\}^{\Omega}\), define the pointwise choice
\[
 f_x(g)=\sigma_{x_g}(g).
\]

### Proposition 1 (all realizations)

If \(f_x\) is a permutation for every \(x\in\{0,1\}^{\Omega}\), then
\(\sigma_0=\sigma_1\).

### Proof

The all-zero choice is \(\sigma_0\).  Fix \(g\) and flip only \(x_g\).
If \(\sigma_1(g)\ne\sigma_0(g)\), then
\(\sigma_1(g)=\sigma_0(h)\) for a unique \(h\ne g\).  The modified map sends
both \(g\) and \(h\) to \(\sigma_0(h)\) and sends no point to
\(\sigma_0(g)\), contradicting bijectivity.  Hence
\(\sigma_1(g)=\sigma_0(g)\) for every \(g\). \(\square\)

Therefore nontrivial independent pointwise voltage choices cannot give a
cycle cover, let alone a single cheap posterior tour, for every realization.
This is not a spectral or girth issue; it is forced by the product activation
law.

### Proposition 2 (exact random defect count)

Relabel outputs so that \(\sigma_0\) is the identity and put
\(\pi=\sigma_0^{-1}\sigma_1\).  Let
\[
 f_X(g)=
 \begin{cases}
 g,&X_g=0,\\
 \pi(g),&X_g=1.
 \end{cases}
\]
For an output \(h\),
\[
 \deg^-_{f_X}(h)=(1-X_h)+X_{\pi^{-1}(h)}.
\]
Hence \(h\) is a collision or a hole exactly when
\[
 X_h\ne X_{\pi^{-1}(h)}.
\]
If \(s=|\operatorname{supp}(\pi)|\), then the number \(D(X)\) of defective
outputs satisfies
\[
 \mathbb E[D(X)]=2p(1-p)s.
\]
Exactly half the defects are holes and half are collisions on every cycle of
\(\pi\).  A change of one bit changes \(D\) by at most two, so for fixed
\(p\in(0,1)\) and \(s=\Theta(|\Omega|)\), bounded differences also gives
\(D(X)=\Theta(|\Omega|)\) with probability \(1-e^{-\Theta(|\Omega|)}\).

For a translation by a nonidentity group element, \(\pi\) is fixed-point
free on each translated orbit, so \(s=|\Gamma|\).  At \(p=1/2\), there are
on average \(|\Gamma|/2\) defective outputs and \(|\Gamma|/4\) holes.

## 3. Conditional defect cost for a retained-trace circulation

The following calculation uses hypotheses not supplied by the displayed
input/output layer.  Identify each output with a compulsory next-state port,
require the routing ledger to retain one preferred trace from every fiber,
and suppose every additional connector entering a hole costs at least \(w\)
in the directed shortest-path metric.  In stochastic TSP these properties
need proof: an inactive preferred trace is not automatically compulsory, and
a distinct unused output need not create flow divergence.

Under these retained-trace hypotheses, any circulation containing all
preferred traces must add at least \(D(X)/2\) connectors, because one
connector can supply an incoming arc to at most one hole.  Therefore its
expected repair ledger obeys
\[
 \mathbb E[\text{repair cost}]
 \ge p(1-p)s\,w.
\]

For the displayed binary-layer lengths \(w\) and \(2w\), and a
fixed-point-free voltage,
\[
 \mathbb E[\text{preferred trace cost}]
 =\bigl((1-p)+2p\bigr)|\Gamma|w
 =(1+p)|\Gamma|w.
\]
At \(p=1/2\), this retained-trace ledger is therefore at least
\[
 \left(\frac32+\frac14\right)|\Gamma|w
 =\frac74|\Gamma|w
\]
before connecting different circulation components or attaching the depot.
If one informally inserts the hoped-for adaptive contribution
\(2|\Gamma|w\) and ignores adaptive repair, the bookkeeping ratio is
\[
 \frac{2}{7/4}=\frac87<\frac43.
\]

This \(8/7\) is not a bound on a completed stochastic-TSP instance in either
direction: no adaptive upper bound of \(2|\Gamma|w\) is proved, the adaptive
policy may pay repairs too, and the posterior optimum may omit the retained
traces.  The rigorous conclusion is only that the naïve posterior upper
ledger \(\tfrac32|\Gamma|w\) cannot be obtained by blindly concatenating all
independently preferred traces under the mandatory identified-port
hypotheses.

Cheap all-to-all correction arcs might repair holes and join components, but
the connector endpoints, compulsory ports, depot attachment, and legal client
call order would all have to be specified.  A fixed strongly connected rail
is likewise only a coarse template until a pending-walk implementation is
written.  In either completion, cheap correction arcs would also let the
adaptive policy change sheets after learning a bit.  If repairs are made
expensive, the retained-trace candidate pays the conditional ledger above.
A positive construction has to formalize this repair-versus-shortcut tension
without treating the ledger as an optimum bound.

## 4. Why a route-code version is not immediate

One might instead use \(k\) independent bits \(X_1,\ldots,X_k\) and let a
layered algebraic network update the sheet by
\[
 g_j=g_{j-1}a_j^{X_j}.
\]
After seeing all bits, the posterior would choose the return route indexed by
the product \(a_1^{X_1}\cdots a_k^{X_k}\).

The apparent selector gadget is not a legal lift without further work.
If a selector is replicated as \(m_{j,g}\) in every state \(g\), all
\(m_{j,g}\) are separate clients with independent activations; the tour must
serve active selectors in *all* states, not only the one state followed by
the route.  If instead there is one global client \(m_j\), a single vertex
does not itself remember the incoming sheet, but the policy can remember it
and choose a history-dependent outgoing destination.  The real problem is
that fanning every input sheet into \(m_j\) and every desired continuation
out of it makes many unintended \(I_g\to m_j\to O_h\) paths available,
threatening to collapse the sheet metric after shortest-path closure.
Declaring all fiber copies of \(m_j\) to share one activation bit would be
correlation outside the model.

This is the state-dependent-selector obstruction.  A codeword or group
product may index deterministic routes, but its dependent coordinates cannot
simply be declared to be the active set.

## 5. Adaptive lower-bound attempt

For a pointwise template augmented by an orientation-conflict cell, couple two
executions differing only in \(X_g\).  If the policy commits to the identity
trace before calling \(m_g\), the active twin should have to change sheets; if
it calls \(m_g\) first, the active twin moves to \(m_g\).  A desired per-fiber
inequality is
\[
 C_g(X_g=0)+C_g(X_g=1)\ge 4w+\Delta_{\rm label},
\]
where \(\Delta_{\rm label}>0\) is a group-displacement cost not chargeable to
other fibers.

No such inequality follows merely from the word length of \(\pi(g)g^{-1}\).
The active call may itself realize the preferred shifted trace, and cheap
repair connectors can change the sheet later.  Moreover, the defect
calculation shows that the posterior also needs linearly many sheet repairs.
Port-separating the cells would make the charges disjoint, but would remove
the voltage coupling and return to additive \(4/3\).

An algebraic label can certify a distance only if every generating arc
changes a suitable quotient coordinate in a controlled way.  On a finite
group there is no globally increasing potential around a cycle; one must use
a quotient word-metric lower bound or scale-separated layers.  The proof must
also account for inverse generators, since a short group relation is a
shortest-path shortcut.

## 6. Failure audit

- **Hidden-label legality.**  Voltages, signs, and lift permutations are fixed
  metric data.  Randomizing them is not a legal activation distribution.
- **Independent versus correlated selectors.**  Sharing one switch bit across
  a fiber, or choosing a random codeword of sheet transitions, violates
  independence unless implemented by additional clients and a causal gadget.
- **Posterior consistency.**  Pointwise choices between two permutations have
  \(2p(1-p)s\) expected defects.  A collection of locally preferred traces is
  not automatically a circulation.
- **Remote early calls.**  Calling \(m_g\) before committing is legal.  If it
  is active, the forced move may already perform the desired shifted trace.
- **Interleaving.**  Algebraic fibers share ports and correction routes.  A
  local conflict cannot be summed unless each repair or re-entry is assigned
  with bounded congestion.
- **Shortest-path closure.**  Group relations, inverse generators, and cheap
  correction arcs may make a nominally long voltage displacement short.
- **Selector proliferation.**  Replicating a selector over all states creates
  \(|\Gamma|\) independent clients per layer, all of whose active copies must
  be served.
- **Small-support escape.**  Taking
  \(|\operatorname{supp}(\pi)|=o(|\Gamma|)\) makes the defect cost small, but
  then only a sublinear number of switches are genuinely coupled and no
  macroscopic amplification follows from them.

## Verdict

**The natural binary algebraic lift is obstructed, not promising as a direct
improvement.**  Nontrivial pointwise voltage choices under independent
activations create linear defects in the all-preferred-traces ledger;
eliminating those defects for every realization forces the two pointwise
permutations to coincide.  This is not yet a bound on the posterior optimum.
Cheap repair paths would also become adaptive shortcuts.  Algebraic lifts may
still be useful inside a more elaborate layered selector gadget, but the lift
labels themselves are not hidden randomness and do not amplify the Chapter 4
switch.

## Next lemma

The right target is a **legal causal route-code lemma**, not another
pointwise 2-lift.  Construct a fixed, strongly connected directed metric with
\(k\) genuinely independent selector clients and cheap tours
\(\{T_x:x\in\{0,1\}^k\}\) such that:

1. \(T_x\) serves *all* active clients, not only selectors on one state path;
2. \(\operatorname{cost}(T_x)\le P\) for every \(x\), with no permutation
   defects or with \(o(k)\) provably cheap defects;
3. after any causal history leaving \(\Omega(k)\) bits unknown, every
   affordable sheet choice is incompatible with a constant fraction of the
   remaining completions;
4. querying the remaining selectors immediately has expected movement at
   least the same claimed penalty; and
5. a quotient word metric certifies all distances after shortest-path
   closure.

The product-choice proposition proves that such a lemma cannot be realized
by independent pointwise choices between two voltage permutations.  A
successful construction must use a non-pointwise, multi-stage service gadget
whose selectors are not replicated across hidden states.
