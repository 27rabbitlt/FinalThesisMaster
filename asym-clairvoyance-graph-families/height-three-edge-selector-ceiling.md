# A sharp \(4/3\) ceiling for height-three edge selectors

## 1. The exact product-activation poset

Let \(G=(U,V;E)\) be any bipartite graph with
\[
                         |U|=|V|=n .
\]
All vertices in \(U\cup V\) are permanent clients.  Put every \(u\in U\)
below every \(v\in V\).  For every edge \(e=uv\in E\), add one stochastic
middle client \(x_e\), independently active with an arbitrary probability
\(p_e\), and add only the relations
\[
                         u<x_e<v .
\tag{1.1}
\]
Besides the transitive relations \(U<V\), there are no further
comparabilities.  This is a fixed height-three poset containing many
overlapping \(N\)'s whenever \(G\) has overlapping edges.

For a realization \(A\subseteq E\), write
\[
             m(A):=|A|,\qquad
             \nu(A):=\nu(G[A])
\tag{1.2}
\]
for the number of active middle clients and the maximum matching size of
their underlying edges.

### Lemma 1 (exact width)

For every realization,
\[
                 \operatorname {width}(P[A])
                    =n+m(A)-\nu(A).
\tag{1.3}
\]

### Proof

Start with \(n\) chains pairing the permanent \(U\)'s and \(V\)'s; this is
possible because every \(u\) is below every \(v\).  Every active middle
client initially forms an additional singleton chain.  If
\(M\subseteq A\) is a matching, then for every \(uv\in M\) replace one
permanent pair by
\[
                         u<x_{uv}<v .
\]
Complete the unused permanent endpoints arbitrarily, using the complete
relation \(U<V\).  A maximum matching therefore gives a chain cover with
\(n+m(A)-\nu(A)\) chains.

Conversely, consider any chain cover.  A chain containing a middle client
\(x_{uv}\) can contain at most its one possible permanent predecessor \(u\)
and its one possible permanent successor \(v\).  Relative to the baseline
consisting of \(n\) permanent \(U\)-to-\(V\) chains and \(m(A)\) middle
singletons, the chain count falls by one only when a middle client is joined
at both ends as \(u,x_{uv},v\).  The middle clients joined at both ends have
distinct \(U\)- and \(V\)-endpoints, hence their edges form a matching.
There can be at most \(\nu(A)\) of them.  This proves the reverse
inequality. \(\square\)

Thus, in the \(\varepsilon\)-poset metric of
`general-poset-causal-runs-audit.md`, the expected posterior leading term is
\[
                         P=n+m-\nu ,
\qquad
 m:=\mathbb E m(A),\quad \nu:=\mathbb E\nu(A).
\tag{1.4}
\]

## 2. The causal matching projection

Suppose a causal increasing run contains
\[
                         u,\ x_{uv},\ v .
\tag{2.1}
\]
Then the policy called \(x_{uv}\) while located at its unique possible
permanent predecessor \(u\).  If \(x_{uv}\) was inactive, the call caused no
movement and another incident middle client could be tried.  If it was
active, the policy moved to \(x_{uv}\), and continuing to \(v\) consumes the
unique successor slot of the run.  Hence every saving of one chain below
the baseline \(n+m(A)\) is exactly a query--commit matching edge.

Calling an active \(x_{uv}\) as a new run start, or attaching it only to
\(u\) or only to \(v\), does not create this saving.  It merely replaces one
permanent \(U\to V\) link by one link incident with \(x_{uv}\).  Therefore,
if \(M_\pi(A)\) is the matching committed by a causal policy,
\[
                         K_\pi(A)
                    \ge n+m(A)-|M_\pi(A)|.
\tag{2.2}
\]

Conversely, every query--commit edge policy is executable.  When it chooses
an unmatched \(u\), call \(u\), probe incident unqueried middle clients in
the prescribed order, and, on the first active answer \(x_{uv}\), call the
still-unserved \(v\).  Complete all remaining permanent clients through the
complete \(U<V\) relation and serve unused active middle clients separately.
Thus (2.2) is the exact policy-dependent quantity.

## 3. One fixed-order greedy policy and two pathwise bounds

Fix an arbitrary deterministic order of \(U\), and, for every \(u\), fix an
order of its incident edges.  When \(u\) is processed, call \(u\) and probe
the middle clients \(x_{uv}\) whose \(V\)-endpoint is still unmatched.  Stop
at the first active answer, commit \(uv\), and call \(v\); if every such
answer is inactive, leave \(u\) unmatched.  This is physically executable
because all failed probes leave the salesperson at \(u\).  The resulting
matching \(Q(A)\) is maximal in the realized graph \(G[A]\): if an active
edge \(uv\) had both endpoints unmatched at the end, then \(v\) was also
unmatched when \(u\) was processed, so \(x_{uv}\) was probed and would have
matched \(u\), a contradiction.

Every maximal matching satisfies
\[
                         |Q(A)|\ge\frac12\nu(A).
\tag{3.1}
\]
It also satisfies the less standard but equally elementary bound
\[
                         |Q(A)|\ge2\nu(A)-m(A).
\tag{3.2}
\]
To prove (3.2), fix a maximum matching \(M^*\).  Every edge of
\(M^*\setminus Q\) is incident with an edge of \(Q\), by maximality.
An edge of \(Q\cap M^*\) accounts for no missing \(M^*\)-edge, while an edge
of \(Q\setminus M^*\) has two endpoints and accounts for at most two.
Consequently
\[
 \nu-|Q|
 \le |Q\setminus M^*|
 \le |E(A)\setminus M^*|
 =m-\nu ,
\]
which is (3.2).  Equivalently, this is the usual alternating-component
comparison, with the additional observation that every unit of matching
deficit needs an edge outside the chosen maximum matching.

Taking expectations, the causal policy earns a matching saving \(q\)
satisfying
\[
                    q\ge\max\left\{\frac{\nu}{2},\,2\nu-m\right\}.
\tag{3.3}
\]
Its expected run count is at most
\[
                             A=n+m-q.
\tag{3.4}
\]

## 4. Sharp \(4/3\) ceiling

We claim
\[
                 \max\left\{\frac{\nu}{2},\,2\nu-m\right\}
                    \ge\frac{4\nu-n-m}{3}.
\tag{4.1}
\]
If \(n+m\ge(5/2)\nu\), the first term proves (4.1).  Otherwise
\[
                         m<\frac52\nu-n.
\]
Since \(\nu\le n\),
\[
       m<\frac52\nu-n
         \le \nu+\frac n2,
\]
and rearranging gives
\[
                         2\nu-m
                    \ge\frac{4\nu-n-m}{3}.
\]
This proves (4.1) in the remaining case.

Combining (1.4), (3.4), and (4.1),
\[
\begin{aligned}
 3A
 &\le3(n+m)-3q\\
 &\le3(n+m)-(4\nu-n-m)\\
 &=4(n+m-\nu)=4P.
\end{aligned}
\tag{4.2}
\]
Therefore
\[
                 \boxed{\displaystyle
                 \frac{\operatorname {OPT}_{\rm adapt}}
                      {\operatorname {OPT}_{\rm post}}
                 \le\frac43}
\tag{4.3}
\]
after disjoint-copy amortization of the depot term and
\(\varepsilon\downarrow0\).  Positive \(\varepsilon\) only adds a common
nonnegative comparable-movement term and cannot turn the displayed causal
policy into a strict lower-bound construction.

The inequality is sharp at the level of the two maximal-matching bounds:
their possible crossing occurs at
\[
                         \nu=n,\qquad m=\frac32n,
\]
where both branches in (3.3) equal \(n/2\) and (4.2) is equality.

## Verdict

The height-three edge-selector poset is an exact and useful reduction to
independent-edge query--commit matching, but it cannot exceed \(4/3\).
The obstruction does not rely on a sophisticated stochastic-matching
algorithm: one fixed probing order already produces a maximal matching,
and the two pathwise bounds (3.1)--(3.2) close the entire parameter range.

In particular, importing a query--commit hardness ratio without also
checking the active-middle baseline \(m\) is invalid.  The parameter regime
where the prophet matching is dense forces the second maximal-matching bound
to become strong enough to restore the \(4/3\) ceiling.
