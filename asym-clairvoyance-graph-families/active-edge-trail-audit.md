# Independently active edge-clients: trail-cover audit

## 1. Metric and posterior trail number

Let \(H\) be a finite directed multigraph.  Every directed edge is a client,
independently active.  Give two edge-clients \(e,f\) distance

\[
 d(e,f)=
 \begin{cases}
 1,&\operatorname{head}(e)=\operatorname{tail}(f),\\
 2,&\text{otherwise},
 \end{cases}
\]

and give depot departure and return cost \(1\).  This is the shortest-path
metric obtained by adding the displayed arcs and unit depot resets.

For an active realization \(A\), let \(N=|A|\).  If an ordering of \(A\)
has \(K\) maximal directed-trail runs, its tour cost is

\[
                       N+K.
\]

Let \(A_1,\ldots,A_s\) be the weak edge-components of the active directed
multigraph.  Put

\[
 \Delta^+(A_j)=
 \sum_{v\in V(A_j)}
       \bigl(\deg^+_{A_j}(v)-\deg^-_{A_j}(v)\bigr)_+.
\]

The minimum number of directed trails covering all active edges is

\[
 \boxed{\displaystyle
 \tau(A)=\sum_{j=1}^s\max\{1,\Delta^+(A_j)\}.}             \tag{1}
\]

Indeed, every positive degree surplus requires that many trail starts.
Conversely, balance the surpluses by adding artificial arcs inside a weak
component, take an Euler tour of the resulting balanced directed
multigraph, and delete the artificial arcs.  In an Eulerian component one
break is still required to turn its Euler circuit into an open trail.
Thus the posterior cost is exactly

\[
                         N+\tau(A).                       \tag{2}
\]

The \(4/3\) target for this family is consequently equivalent to finding a
causal policy whose run count \(K\) satisfies

\[
                 3\mathbb E K
                 \leq \mathbb E N+4\mathbb E\tau.         \tag{3}
\]

## 2. Where causal loss can occur

After serving an active edge \(e=(u,v)\), a policy may probe all still
uncalled outgoing edges of \(v\).  Inactive probes leave it at \(v\); the
first active answer gives a cheap continuation.  Thus no loss occurs while
an uncalled active outgoing edge is available.

The only possible loss is temporal: an outgoing edge of \(v\) may already
have been called as the first edge of an earlier trail, before a later
active incoming edge reaches \(v\).  Posterior reorders the two trails and
pairs those endpoints.  In a DAG one can always start at a tail having no
remaining possible incoming edge, so this temporal inversion disappears.
Directed cycles are therefore necessary.

The local offline transition problem is especially simple.  At each vertex
\(v\), it is a complete bipartite matching between active incoming and
active outgoing edges, of size
\(\min\{\deg^-(v),\deg^+(v)\}\).  The difficulty is entirely that an edge
used early as a trail start is no longer available as the target of this
local matching.

## 3. A many-cycle candidate that still stays below \(4/3\)

Consider the bidirected \(k\)-petal star.  It has centre \(o\), leaves
\(1,\ldots,k\), and two edges per petal,

\[
                  a_i:o\to i,\qquad b_i:i\to o.
\]

Activate every edge independently with probability \(p\in(0,1)\).  The
cycles \(a_i b_i\) all interact through the common centre, so this is the
first natural “many overlapping cycles” candidate.

For a realization, classify a petal as

- \(A\): only \(a_i\) is active;
- \(B\): only \(b_i\) is active;
- \(C\): both are active;
- \(0\): neither is active.

Let the corresponding counts also be denoted \(A,B,C\).  If the realization
is nonempty, all active edges lie in one weak component.  Its centre
surplus is \((A-B)_+\), and every type-\(B\) leaf contributes one additional
positive surplus.  Hence (1) becomes

\[
                 \tau=\max\{1,A,B\}.                     \tag{4}
\]

In particular, by the law of large numbers,

\[
 \frac{\mathbb E N}{k}\longrightarrow 2p,\qquad
 \frac{\mathbb E\tau}{k}\longrightarrow p(1-p).          \tag{5}
\]

Now use the fixed causal policy

\[
             a_1,b_1,a_2,b_2,\ldots,a_k,b_k.
\]

Within a petal, \(a_i\) can be followed cheaply by \(b_i\).  Across petals,
an active \(b_i\) leaves the policy at the centre and hence permits a cheap
transition to the next active \(a_j\).  A type-\(A\) petal is the only state
that leaves the policy away from the centre.

Before a typical petal, the last nonempty petal is type \(A\) with
asymptotic probability

\[
 \frac{\Pr(A)}{1-\Pr(0)}
 =\frac{p(1-p)}{1-(1-p)^2}
 =\frac{1-p}{2-p}.                                      \tag{6}
\]

A type-\(B\) petal always starts one new run.  In addition, an active
\(a_i\) starts a run precisely when the policy is not at the centre.
Therefore

\[
 \frac{\mathbb E K}{k}
 \longrightarrow
 p(1-p)+p\,\frac{1-p}{2-p}
 =
 \frac{p(1-p)(3-p)}{2-p}.                                \tag{7}
\]

Combining (2), (5), and (7), the causal/posterior ratio of this explicit
policy tends to

\[
\begin{aligned}
 R(p)
 &=
 \frac{2p+\frac{p(1-p)(3-p)}{2-p}}
      {2p+p(1-p)}\\
 &=1+\frac{1-p}{(2-p)(3-p)}
 \leq \frac76.                                           \tag{8}
\end{aligned}
\]

Thus even an unbounded number of directed 2-cycles sharing a common hub
does not approach \(4/3\); it is bounded by \(7/6\) using a nonadaptive
edge order.

## 4. Remaining target

The star audit shows that merely sharing a vertex is not enough.  A
successful construction must have cycles that share continuation resources
in a genuinely non-laminar way: probing an apparent return excursion must
consume an edge needed by a different future return, while taking the other
choice must strand the first excursion.

A useful sufficient upper-bound target is a causal
\(2\)-approximation to the posterior trail count,

\[
                         \mathbb E K\leq2\mathbb E\tau.   \tag{9}
\]

Indeed, if \(2\mathbb E\tau\leq\mathbb E N\), (9) gives
\(3\mathbb EK\leq6\mathbb E\tau\leq
\mathbb EN+4\mathbb E\tau\).  If
\(2\mathbb E\tau\geq\mathbb EN\), the trivial \(K\leq N\) policy gives the
same inequality.  Choosing the better of the two policies would prove
(3), and hence the universal \(4/3\) ceiling for this metric family.

No proof of (9), and no interacting-cycle family violating (3), is supplied
here.
