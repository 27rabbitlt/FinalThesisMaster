# Fully-online matching hardness does not black-box embed through chronological gates

## Verdict

The arbitrary-algorithm hardness for fully-online bipartite matching is
numerically strong enough for the desired calculation: the published upper
bound \(c_{\mathrm{FOM}}\leq 0.6297<2/3\) would give
\[
                         2-c_{\mathrm{FOM}}>4/3
\]
if matching savings occupied asymptotically half of the relevant tour
baseline.

There is, however, no valid reduction of that form in the standard
stochastic-TSP model.  The obstruction is local and precedes any asymptotic
calculation.  A fully-online vertex can first be a passive matching resource
and, if it remains unmatched, later make an active matching decision at its
own deadline.  A TSP client has only one service event.  If the same client
represents both roles, the cheap outgoing transition needed at its deadline
is already available immediately after the client is hit passively.  The
tour can therefore take one cheap transition into the client and another
cheap transition out of it, matching the logical vertex twice.

Chronological gates do not make the outgoing transition conditional on how
the client was entered.  Stage-specific copies preserve the time label but
duplicate the matching capacity.  Correlating the copies so that exactly one
is live is outside independent client activation.  This is the
state-erasure/capacity dichotomy.

A second mismatch is informational.  Fully-online hardness assumes
exogenous arrivals and hides all future vertices and edges.  A
stochastic-TSP policy may call any future client.  An inactive call reveals
the bit at zero movement cost.  Thus even a successful route-order gadget
would not import the matching lower bound without a new proof against free
negative lookahead.

The conclusion is not that no directed metric can ever exceed \(4/3\).  It
is that the \(0.6297\) fully-online matching theorem cannot be used as a
black box through the natural layered/path-cover or chronological-gate
embeddings.

## 1. The attractive numerical target

In fully-online matching, vertices arrive and later reach deadlines.  At a
deadline an unmatched vertex may be matched to an unmatched neighbor that
has already arrived.  The graph induced by future vertices is not yet
visible.

Eckl, Kirschbaum, Leichter, and Schewior prove that no algorithm has
competitive ratio greater than \(0.6297\), even fractionally and on
bipartite graphs.  Their construction reveals a sequence of tree or
biclique levels: only after the previous level departs is the subset that
connects to the next level revealed.  It ends with a KVV triangle.

Suppose, only formally, that a metric reduction had a baseline \(C_0\),
posterior matching savings \(S\), and causal matching savings at most
\(cS\).  If \(S/C_0\to1/2\), then
\[
 \frac{C_0-cS}{C_0-S}\longrightarrow 2-c.
\tag{1.1}
\]
For \(c=0.6297\), the limit is \(1.3703>4/3\).  The rest of this note
explains why the hypotheses behind (1.1) are not supplied by a
chronological gate gadget.

## 2. The safe \([B/2,B]\) metric is a path-cover objective

Let every off-diagonal distance have the form
\[
                       d(x,y)=B-s(x,y),
 \qquad 0\leq s(x,y)\leq B/2,
\tag{2.1}
\]
and give the depot legs length \(B\).  Triangle inequality is automatic:
every two-leg path has length at least \(B\), while every direct
off-diagonal distance is at most \(B\).

For a realization with \(N\) clients and service order
\(x_1,\ldots,x_N\), the closed-depot cost is
\[
      B(N+1)-\sum_{j=1}^{N-1}s(x_j,x_{j+1}).
\tag{2.2}
\]
If the positive-saving arcs form a layered acyclic digraph, the posterior
savings are exactly a maximum-weight directed path cover: every client has
at most one selected incoming arc and at most one selected outgoing arc.

This safely removes the old shortest-path-transit defect.  An intermediate
vertex cannot create a new discount, because two legs cost at least \(B\).
It does **not** turn the path-cover constraints
\[
                 \deg^-_{\mathrm{sel}}(v)\leq1,\qquad
                 \deg^+_{\mathrm{sel}}(v)\leq1
\tag{2.3}
\]
into the matching constraint
\[
                 \deg^-_{\mathrm{match}}(v)
                 +\deg^+_{\mathrm{match}}(v)\leq1.
\tag{2.4}
\]
The difference between (2.3) and (2.4) is exactly the fully-online
state-erasure problem.

## 3. A fully-online vertex needs two mutually exclusive roles

Fix a logical matching vertex \(v\).  Before \(v\)'s deadline, an earlier
vertex \(u\) may match to \(v\); call this the **passive** role.  If that
does not happen, then at \(v\)'s own deadline it may match a later-deadline
neighbor \(w\); call this the **active** role.

With one TSP client \(c_v\), the two rewards require positive-saving arcs
\[
                         c_u\longrightarrow c_v
 \quad\text{and}\quad
                         c_v\longrightarrow c_w.
\tag{3.1}
\]
But then the path-cover objective admits
\[
                         c_u\longrightarrow c_v
                         \longrightarrow c_w.
\tag{3.2}
\]
It earns both matching rewards.  In the fully-online instance, (3.2) is
infeasible because \(v\) may be incident to only one matched edge.

The same defect is causal, not merely posterior.  If a call from \(c_u\)
finds \(c_v\) active, the salesperson is now physically located at
\(c_v\).  It may immediately call \(c_w\) and use the second cheap
transition.  No revisit and no shortest-path transit is involved.

### State-erasure lemma

Suppose two legal histories end by calling the same active client \(c_v\).
Conditional on the same uncalled-client set and the same revealed
activation bits, every continuation has the same feasible calls and the
same transition costs under the two histories.

Indeed, the physical state after either call is the same vertex \(c_v\),
and future distances are the fixed numbers \(d(c_v,\cdot)\).  In
particular, the metric cannot remember whether the predecessor of \(c_v\)
was a chronological gate or a passively matched neighbor.

The histories in an intended simulation will often have different served
gate prefixes, so the policy can *know* the stage.  That does not repair
(3.1): the arc \(c_v\to c_w\) is available in both histories.  Gates can
tell the policy which transition is intended; they cannot make the
unintended transition infeasible.

## 4. Why an entry gate cannot condition the exit

The natural deadline gadget has an entry gate \(g_i\) and intends
\[
                  g_i\longrightarrow c_v\longrightarrow c_w
\tag{4.1}
\]
when \(v\) survives unmatched to stage \(i\).  If \(v\) was instead matched
passively, the intended trace contains
\[
                            c_u\longrightarrow c_v
\tag{4.2}
\]
and must forbid the second arc of (4.1).

A static pairwise metric cannot impose the predicate
\[
  \text{``\(c_v\to c_w\) is cheap iff the predecessor of \(c_v\) was
  \(g_i\)''}.
\tag{4.3}
\]
The number \(d(c_v,c_w)\) is the same after (4.1) and (4.2).

Trying to deter the illegal continuation with a lost gate reward changes
the objective rather than encoding it.  There are two cases.

1. If a bypass preserves the gate reward when \(v\) was passively matched,
   then the illegal trace takes the bypass reward as well as both matching
   rewards.
2. If there is no bypass, then a passive match and an active match carry
   different nonmatching gate rewards.  Posterior TSP is no longer an
   affine copy of maximum matching, and the \(0.6297\) upper bound says
   nothing about the new weighted decision problem.

Thus a gate penalty may define an interesting new stochastic-TSP gadget,
but it requires a fresh universal policy upper bound.  It is not an
embedding of fully-online matching.

## 5. Time expansion preserves time and destroys capacity

The standard way to retain the stage label is to replace \(c_v\) by copies
\[
                        c_{v,1},c_{v,2},\ldots,c_{v,T}.
\tag{5.1}
\]
Now the outgoing arcs may depend on the index \(i\).  But each copy has its
own predecessor slot.  The path-cover polytope contains only
\[
  \sum_{e\in\delta^-(c_{v,i})}x_e\leq1
  \qquad\text{for each }i,
\tag{5.2}
\]
not the logical capacity constraint
\[
  \sum_i\sum_{e\in\delta^-(c_{v,i})}x_e\leq1.
\tag{5.3}
\]
Consequently distinct copies can be matched at distinct stages.

The possible couplings all fail in a standard independently activated
client model.

* **Permanent copies.**  Every copy is a required client, and every copy
  supplies a separate incoming/outgoing slot.  Posterior capacity is
  multiplied.
* **Independent stochastic copies.**  Several copies may be active and all
  of them must be served.  Their activation bits do not represent one
  logical vertex state.
* **Exactly one active copy.**  This restores the interpretation but is a
  correlated one-hot activation, not a product distribution.
* **A common guard client.**  Its service event occurs once, but serving it
  does not delete distances incident to any time copy.  It therefore does
  not impose (5.3).
* **Nonclient ports or switches.**  They can alter fixed shortest-path
  distances but carry no consumable service state.  After the next common
  active client is called, their mode is erased.

This gives the precise dichotomy:
\[
\boxed{
\begin{array}{c}
\text{one shared client: correct unit capacity, no stage-dependent exit;}\\
\text{time-indexed clients: stage-dependent exit, no shared unit capacity.}
\end{array}}
\tag{5.4}
\]

## 6. Bipartite orientation does not remove the deadline problem

Because the hard graphs are bipartite, one may orient every matching reward
from the left part to the right part.  This makes (2.3) equal to matching
capacity on a static height-two graph.  It also collapses the simulation to
the one-sided source--target chamber.

The problem is that the active endpoint at a fully-online deadline need not
be the tail of this fixed orientation.  To realize a left-to-right reward
when the right endpoint reaches its deadline, the tour must first be
located at the left endpoint and then call the right endpoint.  If the left
client was already called to encode its own earlier arrival or deadline,
that service event is unavailable.  If its call is postponed, the claimed
chronological revelation was not encoded.

Giving every logical vertex an arrival copy and a deadline copy is exactly
the time expansion of Section 5.  Hence a global bipartite orientation
either returns to ordinary one-sided matching—where random source order
RANKING gives at least \(0.696\) realization-wise—or reintroduces duplicated
capacity.

## 7. Product activation also permits free future lookahead

Even if the capacity issue were ignored, the information structures are
different.

At any TSP state a policy may call a client assigned to an arbitrary future
stage.  If the client is inactive, the policy learns that bit and does not
move.  Therefore a chronological gate can charge a speculative call only
on the positive outcome:
\[
 \text{future probe}=
 \begin{cases}
   \text{free revelation},&\text{client inactive},\\
   \text{forced metric move},&\text{client active}.
 \end{cases}
\tag{7.1}
\]
Fully-online matching reveals neither outcome before the arrival event.

The known \(0.6297\) construction uses information of exactly this kind:
after one level departs, it reveals which vertices continue into the next
biclique; the earlier decisions were made without that subset.  The
original tree description has a designated child/leaf whose identity is
hidden until its parent's departure.  These are exogenous categorical or
subset revelations, not independent Bernoulli service obligations.

An ordered marker bank can synthesize a categorical last-active index from
independent bits, but all earlier active markers remain genuine clients.
Moreover, the TSP policy chooses when to probe the bank.  Thus the
fully-online theorem would still need to be reproved against the stronger
probe model (7.1); it does not transfer from the published adversarial
arrival model.

The stronger biclique impossibility is even farther from a fixed product
instance: its next surviving subset is chosen/revealed after observing the
previous level's allocation.  A stochastic-TSP metric and its independent
activation probabilities are fixed before the policy runs.

## 8. What a genuine repair would have to supply

A successful embedding would need all of the following, simultaneously:

1. a single logical capacity shared by passive and active roles;
2. a cheap outgoing transition available only after the deadline-gate
   entry, not after a passive match into the same logical vertex;
3. a fixed known directed metric;
4. independently activated clients;
5. a lower bound that allows the policy to probe arbitrary future clients,
   including free inactive probes; and
6. posterior savings with enough density for the calculation (1.1).

Items 1 and 2 require a consumable state or a predecessor-dependent
transition.  Standard stochastic TSP has neither: service removes an
obligation but does not modify the metric, and pairwise distances do not
depend on the preceding vertex.

Allowing a stateful metric that deletes \(c_v\)'s outgoing arcs after a
passive hit would solve the local problem, as would a correlated one-hot
choice among time copies or a true exogenous arrival process.  All three
are outside the model being studied.

Therefore the chronological fully-online matching route is closed in its
natural layered form.  Any future use of the \(0.6297\) constant must first
provide a new independent-client conditional-capacity gadget and prove its
information lower bound directly; the matching hardness theorem alone is
insufficient.

## Reference

Alexander Eckl, Anja Kirschbaum, Marilena Leichter, and Kevin Schewior,
*A stronger impossibility for fully online matching*, Operations Research
Letters 49(5), 2021, pp. 802--808.
