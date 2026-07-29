# High-girth lifts

## Outcome

A high-girth lift is useful for keeping *walks* locally tree-like, but the
adaptive policy is not a walk-local exploration process: it may call a client
in any fiber at any time.  The most direct construction, in which a
high-girth lift is used as a scaffold carrying copies of the recursive
triangle block, is exactly additive and therefore cannot improve \(4/3\).
The more ambitious construction, in which voltage permutations couple the
ports of different switches, needs an early-query-versus-late-conflict lemma
which high girth alone does not imply.

The conclusions below concern weighted directed shortest-path metrics.  No
small-instance calculation is used.

## 1. Lift notation and a basic construction template

Let \(Q=(U,E)\) be a finite directed multigraph.  An \(N\)-lift is specified
by a permutation \(\sigma_e\in S_N\) for each arc \(e:u\to v\).  Its vertices
are \((u,i)\), and it has the lifted arcs
\[
 (u,i)\longrightarrow (v,\sigma_e(i)),\qquad i\in[N],
\]
with the weight of the base arc.  When every undirected base edge is replaced
by two inverse lifted arcs and the underlying cover is connected, the
generating digraph is strongly connected.  Its directed shortest-path closure
is consequently a directed metric.  The underlying undirected lift can be
chosen to have girth \(g_N\to\infty\).

There are two materially different ways to put stochastic-TSP structure on
this lift.

1. **Scaffold lift.**  Attach a complete stochastic module to each lifted
   vertex through a single articulation port.  The lift only says in which
   order modules can be visited.
2. **Port-coupled lift.**  Make lifted arcs enter and leave different ports of
   a switch module.  A local orientation decision then changes the sheet in
   which subsequent work starts.

The first use is completely analyzable and gives a no-amplification theorem.
The second is the only one that could plausibly improve the ratio, but it
loses the separator structure used in Chapter 4.

## 2. Exact no-amplification theorem for scaffold lifts

Let \(J\) be any stochastic directed-metric instance with depot \(q\), fixed
realization cost \(P_J(A)\), expected a-posteriori value \(P_J\), and adaptive
value \(A_J\).  Take a strongly connected weighted lift \(S_N\) on vertices
\(q_1,\ldots,q_N\), choose \(q_1\) as the global depot, and attach an
independent copy \(J_i\) of \(J\) at \(q_i\).  The only common vertex of
\(J_i\) and the rest of the generating graph is \(q_i\).  Declare every
\(q_i\) permanent.  All stochastic clients in the different copies are
independent with the same probabilities as in \(J\).

Let \(C(S_N)\) be the minimum cost of a closed walk in the scaffold, based at
\(q_1\), which visits every \(q_i\).

### Proposition (articulation-lift additivity)

For the instance just defined,
\[
 \operatorname{OPT}_{\rm post}
   =C(S_N)+N P_J,\qquad
 \operatorname{OPT}_{\rm adapt}
   =C(S_N)+N A_J .
\]

### Proof

Fix a realization and expand every metric move in the generating graph.
Every maximal excursion into \(J_i-\{q_i\}\) starts and ends at \(q_i\).
Concatenating all such excursions in their execution order gives a closed
local walk through every active client of \(J_i\), so its internal cost is at
least \(P_J(A_i)\).  Deleting the internal excursions leaves a closed
scaffold walk visiting every permanent \(q_i\), of cost at least \(C(S_N)\).
Conversely, follow an optimal scaffold walk and insert an optimal local tour
whenever \(q_i\) is first visited.  This proves the fixed-realization equality
and hence the a-posteriori formula.

For an arbitrary adaptive policy, condition on all activations outside
\(J_i\) and on any private random seed.  These data are independent of the
bits in \(J_i\).  In the induced virtual ambient execution, the pre-sampled
outside outcomes are supplied only when their clients are virtually called,
so decisions remain causal with respect to the local history.
Concatenating the \(J_i\)-excursions at their common endpoint
\(q_i\) produces a legal randomized local policy; backward induction removes
the randomization.  Its expected internal cost is at least \(A_J\).  Summing
over \(i\), and again projecting all remaining movement to the scaffold,
gives the adaptive lower bound.  The matching policy follows an optimal
scaffold walk and executes an optimal adaptive policy in each copy.  To make
that description legal in the remote-call model, keep each local terminal
path to \(q_i\), the following scaffold segment, and the next module's entry
path as one pending generating walk.  Inactive calls leave it pending; the
next active call shortcuts the whole pending walk in the metric.  The final
pending suffix is absorbed into the legal depot return.  This is the
policy-gluing argument from Chapter 4 and also handles scaffold revisits to
already-called ports.  Thus equality holds. \(\square\)

Taking \(J=I_l\) from Chapter 4 gives
\[
 P_J=(3l+4)2^{l-2},\qquad A_J=(l+1)2^l.
\]
Therefore
\[
 \frac{C(S_N)+N(l+1)2^l}
      {C(S_N)+N(3l+4)2^{l-2}}
 \le \frac{4(l+1)}{3l+4}<\frac43.
\]
If the module scale dominates the scaffold cost, the ratio approaches the
old ratio from below.  Neither the girth nor the number of sheets amplifies
it.  This is an exact version of the additive-replication diagnostic.

The whole graph contains the short cycles internal to \(J\); it is the
*outer scaffold* that has high girth.  Subdividing internal gadget arcs can
make the unweighted girth large without changing the metric, so bare
combinatorial girth of a weighted presentation would not be meaningful
evidence of a stronger gap.

## 3. Port-coupled high-girth attempt

A non-additive template starts with a high-girth lift \(S_N\) and replaces
each selected lifted vertex \(z\) by a two-port conflict cell
\[
 L_z\longrightarrow R_z\longrightarrow m_z\longrightarrow L_z,
\]
with arcs of weight \(w\), permanent work behind \(L_z,R_z\), and
\(m_z\sim{\rm Bernoulli}(1/2)\).  Lifted scaffold arcs enter one port and
leave another according to their voltage permutation.  Under the additional
Chapter 4 hypotheses that both port blocks are compulsory, the cell uses the
same open-service boundary relaxation, outer arcs introduce no shorter
trace, and extra pieces are charged at the boundary, the quotient accounting
is:

- if \(m_z\) is inactive, the cheap open trace is \(L_z\to R_z\), of top
  cost \(w\);
- if \(m_z\) is active, the cheap open trace is
  \(R_z\to m_z\to L_z\), of top cost \(2w\);
- a causal service which commits to the inactive trace before calling \(m_z\)
  has expected top cost \(2w\).

Under those hypotheses the uncoupled leading terms are
\[
 P_{\rm local}=\frac32w,\qquad A_{\rm local}=2w,
\]
and \(n\) independently chargeable cells give \(4/3\), not more.
To beat \(4/3\), the sheet reached after one choice must constrain many later
choices so that adaptive mistakes are superadditive, while the posterior can
still stitch all preferred traces cheaply.

### A lower-bound attempt

Choose switch cells whose radius-\(R\) balls are edge-disjoint.  Couple the
two executions differing only in \(X_{m_z}\).  One would like to prove that
for every policy and every chosen \(z\), one of the following is charged:

1. \(m_z\) is called before the policy crosses the exit boundary of its
   ball, and the active twin pays an **early-query toll** of at least
   \(\Delta\); or
2. it is called after that crossing, and the two twins pay a **late
   orientation-conflict toll** of at least \(\Delta\).

If the charged arc sets were disjoint, summing the paired inequalities would
give an \(\Omega(|Z|\Delta)\) policy-uniform lower bound, robust to the order
in which balls are interleaved.

The late part can sometimes be certified from girth: two distinct port traces
cannot merge inside a radius-\(R<g_N/2\) ball without using the prescribed
boundary.  The early part does **not** follow from girth.  A call is remote.
If \(m_z\) is inactive, it reveals the bit for free; if active, the movement
ends at \(m_z\) and may itself be a favorable entry into the cell.  Moreover,
the current position need not be in the ball whose bit is called.

There is also a packing loss.  In a \(d\)-regular lift below half the girth,
radius-\(R\) balls have \(\Theta((d-1)^R)\) vertices.  For a merely
bounded-degree lift this is only an upper bound.  Using \(R\) comparable to the girth leaves
too few disjoint balls to obtain a linear number of charges.  With constant
\(R\), one obtains linearly many balls but only a constant local obstruction,
which returns to additive \(4/3\) accounting.

## 4. A-posteriori routing and why girth is not enough

For a fixed realization, take the locally preferred trace in every switch
cell.  These traces generally have unmatched entries and exits in different
sheets.  A posterior tour exists with the hoped-for cost only if they can be
joined by a cheap circulation.  High girth says that two short joining walks
are locally unique; it does not say that the required global circulation is
cheap.  Random independent orientations commonly create linear boundary
imbalance, so correction can cost the same order as all local savings.

Conversely, adding sufficiently many cheap correcting arcs makes the
posterior stitching easy but creates shortest-path shortcuts for the adaptive
policy.  This is the same tension as in the algebraic-lift analysis, without
the group notation that makes the defects explicit.

Thus no controlled a-posteriori bound below the additive baseline is currently
available for the port-coupled template.  The exact scaffold construction
above *does* have a tour for every realization, but its ratio is at most
\(4/3\).

## 5. Failure audit

- **Remote early calls.**  High girth restricts paths, not call locations.
  An early active call may move directly to the selector and simultaneously
  perform the intended entry.
- **Calling a separator first.**  Calling all switch clients is expensive
  only if their active subset has a large directed tour cost.  That fact must
  be proved; it is not a consequence of undirected girth.
- **Interleaving.**  Disjoint metric balls do not prevent calls in many balls
  from being interleaved.  A valid proof needs open-service boundaries and
  must charge every re-entry, as Chapter 4 does.
- **Shortest-path closure.**  A long alternate route around one lifted cycle
  may be shortened through other fibers.  A girth claim is valid only below
  \(g_N/2\), and only in the unweighted support.  Weighted penalties need a
  quotient or potential certificate.
- **Inactive free information.**  Half of the \(p=1/2\) selectors can be
  learned with no movement.  The cost of active early selectors must carry
  the entire selector-first defense.
- **Posterior consistency.**  Independently preferred local orientations
  need not form a tour or even a balanced circulation.

## Verdict

**No improvement from the natural scaffold use; unresolved for genuinely
port-coupled lifts.**  The articulation-lift theorem rigorously shows that an
outer high-girth lift carrying independent Chapter 4 modules has ratio at most
\(4/3\).  Girth by itself does not amplify repeated orientation conflicts.
Any positive use must make lift labels couple cells and must simultaneously
solve early querying and posterior circulation.

## Next lemma

Prove or refute the following **ball dichotomy lemma** for one explicit
port-coupled lift.  There should be a linear-size set \(Z\) of switch cells and
edge-disjoint charged regions \(D_z\) such that, for every causal policy and
each \(z\in Z\), coupling \(X_{m_z}=0,1\) yields
\[
 \mathbb E\!\left[
   \operatorname{cost}(D_z;X_{m_z}=0)
  +\operatorname{cost}(D_z;X_{m_z}=1)\right]
 \ge 2P_z+\Delta,
\]
whether \(m_z\) is called before entering \(D_z\), during an interleaved
service, or after leaving it.  In parallel, prove a realization-wise
circulation bound
\[
 \operatorname{tour}_{\rm post}(x)
 \le \sum_{z\in Z}P_z(x_z)+C
\]
with \(C=o(|Z|\Delta)\).  Without both inequalities the high-girth lift
cannot beat additive replication.
