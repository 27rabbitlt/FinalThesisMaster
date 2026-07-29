# A \(K_{2,2}\) extraction bank fixes splitting but not the tour ratio

## Outcome

There is a simple directed bank in which complete edge extractions have
exactly the value of a matching and the usual two-half-extraction shortcut
does not exceed that matching.  Thus the split-client defect can be repaired
combinatorially for \(K_{2,2}\).

The repair still cannot transfer the query--commit matching separation to a
large stochastic-TSP separation.  Every active edge client remains a service
obligation.  Its unavoidable active-client term is larger than the
omniscient-versus-causal matching loss, and even an unrealistically favorable
zero-constant accounting has ratio at most \(1.104\).

## The anchored transitive bank

Let
\[
 U=\{u_1,u_2\},\qquad V=\{v_1,v_2\},
\]
and let \(z_{ij}\) be the stochastic client representing edge \(u_iv_j\).
Add two permanent anchors \(s,t\).  Fix any total order
\[
                 s<z_{11}<z_{12}<z_{21}<z_{22}<t.
\]

The cheap-transition digraph has:

1. every forward arc in the displayed total order; and
2. the extraction arcs
   \[
                    u_i\longrightarrow z_{ij}
                    \longrightarrow v_j.
   \]

There are no other cheap arcs.  Use the \(\{1,2\}\) metric: displayed cheap
arcs have length one and all other ordered pairs have length two.  This is a
directed metric because every two-step path has length at least two.

A zero/\(\varepsilon\)-reachability version must **not** be used here:
transitive closure would create false incidence arcs such as
\(u_i\to z_{ij}\to z_{i'j'}\), reinstating the split/transit loophole.

The two permanent anchors matter.  If an interval of active bank clients is
removed from the default \(s\)-to-\(t\) bank path, transitivity reconnects
the remaining bank but the removal loses exactly one bank transition.  A
bracket
\[
                u_i\to z_e\to\cdots\to z_f\to v_j
\]
gains two extraction transitions and loses one bank transition, hence has
net value one.  A one-sided extraction gains one transition and loses one,
hence has net value zero.

## Exact \(K_{2,2}\) interval lemma

Fix an active edge set \(A\).  Let \(\nu(A)\) be its maximum bipartite
matching size.

### Lemma

Relative to the anchored default bank, the largest possible number of
additional cheap transitions is exactly \(\nu(A)\).

### Proof

A cheap path that meets the bank enters it at most once and leaves it at
most once, because all bank arcs point forward, all extraction entrances
come from \(U\), and all extraction exits go to \(V\).  The bank vertices
in a cheap path cover therefore form forward intervals.  Splitting the one
default bank path into \(c\) such intervals loses \(c-1\) bank transitions.
Every interval can gain at most one entrance from \(U\) and at most one exit
to \(V\).

Since there are only two clients on either side, the net gain is at most
two.  A positive net gain uses at least one extraction arc and hence
certifies an active edge, so it is at most \(\nu(A)\) when \(\nu(A)=1\)
unless it equals two.  A net gain of two requires four extraction arcs:
both distinct \(U\)-clients must enter bank intervals and bank intervals
must exit to both distinct \(V\)-clients.  (With at most three extraction
arcs, the gain is at most \(3-(c-1)\le1\); the two anchors prevent all bank
vertices from lying in a single doubly bracketed interval plus an
independent one-sided gain.)  Thus the active edge set meets both left
vertices and both right vertices.  For a subgraph of \(K_{2,2}\), this
implies a matching of size two.  Hence a gain of two is possible only when
\(\nu(A)=2\).

Conversely, extract each edge of a matching as the singleton interval
\(u_i,z_{ij},v_j\).  Distinct matching edges use disjoint endpoints and
give one additional transition each.  Therefore the bound is attained.
\(\square\)

This proof is special to \(K_{2,2}\).  In a larger bipartite graph, distinct
left incidences and distinct right incidences do not by themselves certify a
matching of the same size.

## The unavoidable active-client term

Let \(K=|A|\).  In the \(\{1,2\}\) metric, there are six permanent clients
\(U\cup V\cup\{s,t\}\), hence \(N=6+K\) active clients in total.  With
unit depot arcs, an ordering with \(T\) cheap internal transitions costs
\[
                         2N-T.
\]
The default anchored bank has \(K+1\) cheap transitions, and the lemma adds
\(\nu(A)\).  Consequently
\[
             \operatorname{OPT}_{\rm post}(A)
                  = 11+K-\nu(A).                    \tag{1}
\]
The \(K\)-term cannot be removed by a more favorable distance assignment.
For every directed metric and active sets \(A\subseteq B\),
\[
                         \operatorname{OPT}(A)
                     \le \operatorname{OPT}(B),       \tag{2}
\]
because shortcutting the clients in \(B\setminus A\) cannot increase a tour
cost.  In particular, activating an edge client cannot unlock a cheaper
route: all of its vertices and arcs were already available as metric transit
while it was inactive.  Activity only adds a service obligation.

Thus a proposed cost proportional to \(2-\nu(A)\), which decreases when an
additional active edge completes a matching, cannot be the realization cost
of a metric stochastic-TSP instance.

## Even the optimistic variable accounting is too small

Suppose each of the four edges is active independently with probability
\(p\).  The omniscient matching value and exact query--commit value are
\[
\begin{aligned}
 \mu(p)&=4p-4p^2+4p^3-2p^4,\\
 M(p)&=4p-4p^2+3p^3-p^4.
\end{aligned}
\]
The matching loss is \(p^3(1-p)\).

Ignore all permanent-client and depot constants, and optimistically suppose
that the only tour cost were \(K\) minus the matching value.  This is more
favorable than (1).  Its causal-to-posterior ratio would still be only
\[
\begin{aligned}
 R(p)
 &=\frac{4p-M(p)}{4p-\mu(p)}\\
 &=\frac{p^2-3p+4}{2(p^2-2p+2)}.                   \tag{3}
\end{aligned}
\]
Differentiating shows that the maximum on \([0,1]\) occurs at
\[
                         p=2-\sqrt2
\]
and is approximately
\[
                         1.1036 < \frac43.          \tag{4}
\]
Every genuine depot or permanent-bank contribution only decreases the
ratio.

## Conclusion

The anchored bank is a useful exact cure for the literal split-client
loophole in \(K_{2,2}\): two partial extractions do not simulate two matched
edges.  It simultaneously shows why that cure is insufficient.  The
query--commit separation concerns a *reward* for realized edges, whereas
metric stochastic TSP is monotone in the active client set.  The service
burden needed to represent an edge dominates the \(K_{2,2}\) matching loss,
leaving a ratio far below \(4/3\).
