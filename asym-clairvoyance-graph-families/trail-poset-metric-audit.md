# Trail-cover and poset metric audit

This note records two exact facts needed before a stochastic
edge/trail construction can be used for the asymmetric clairvoyance gap.

## 1. A genuine metric for edge clients

Let \(H\) be a directed graph and let every directed edge of \(H\) be a
client.  For distinct clients \(e,f\), put

\[
d(e,f)=
\begin{cases}
1,&\operatorname{head}(e)=\operatorname{tail}(f),\\
2,&\text{otherwise}.
\end{cases}
\]

Add a depot \(r\) with \(d(r,e)=d(e,r)=1\), and put \(d(x,x)=0\).
This is a directed metric: every nonzero distance is at least \(1\), while
every distance is at most \(2\), so every nontrivial two-arc route has
length at least the direct distance.

For a realization \(A\) of \(N=|A|\) active edge-clients and an ordering
\(e_1,\ldots,e_N\), let

\[
B=\bigl|\{i:\operatorname{head}(e_i)\ne
\operatorname{tail}(e_{i+1})\}\bigr|.
\]

The closed-depot cost of this ordering is exactly

\[
2+(N-1)+B=N+(B+1).
\]

Its \(B+1\) maximal composable runs are directed trails in \(H\).
Consequently, if \(\tau(A)\) is the minimum number of directed trails
whose edge-disjoint union is \(A\), then

\[
\operatorname{OPT}_{\rm post}(A)=N+\tau(A).
\]

For every adaptive policy, if \(K_\pi(A)\) is the number of composable
runs in the order in which its active calls occur, then

\[
\operatorname{cost}_\pi(A)=N+K_\pi(A).
\]

Thus this reduction is exact, but it has an unavoidable \(N\)-term.  A
proof of a ratio strictly above \(4/3\) must establish

\[
\mathbb E K_\pi>
\frac13\mathbb E N+\frac43\mathbb E\tau(A)
\quad\text{for every adaptive policy }\pi.
\]

Merely proving a constant-factor online/offline gap for the number of
trails is not enough.

## 2. Why the tempting zero/one version is invalid

The assignment

\[
d_0(e,f)=0
\quad\Longleftrightarrow\quad
\operatorname{head}(e)=\operatorname{tail}(f)
\]

is generally not a metric.  If \(e\) composes with \(f\) and \(f\)
composes with \(g\), the triangle inequality would force
\(d_0(e,g)=0\), although \(e\) need not compose directly with \(g\).

Taking shortest-path closure does not repair the intended reduction.
It makes the zero-distance relation the transitive closure of the line
digraph:

- inside a strongly connected component, all corresponding clients
  become mutually zero-distance;
- in an acyclic graph, zero distance becomes reachability, so the
  posterior problem is chain cover in a reachability poset rather than
  adjacency-respecting trail cover.

Any claimed trail construction using zero-cost composability must
therefore be rejected unless this closure effect is explicitly handled.

## 3. An exact no-gap poset class

Let

\[
P=A_1\oplus A_2\oplus\cdots\oplus A_h
\]

be an ordinal sum of antichains: elements in the same \(A_i\) are
incomparable, and every element of \(A_i\) is below every element of
\(A_j\) for \(i<j\).  Give its elements arbitrary independent activation
probabilities.  For a realization, write \(X_i\) for the number of active
elements of \(A_i\).

The posterior chain-cover number is

\[
\operatorname{width}(A)=\max_i X_i.
\]

There is a causal policy attaining this value realization by
realization.  Make repeated increasing sweeps through
\(A_1,\ldots,A_h\).  During a sweep, call uncalled elements of a layer
until the first active element is found, then immediately continue to
the next layer.  If the layer is exhausted, skip it.  Every sweep
contains at most one active element from each layer, in strictly
increasing layer order, hence is one chain.  Each nonempty sweep removes
one active element from every layer that still has one, so the number
of nonempty sweeps is exactly \(\max_i X_i\).

Therefore ordinal sums of antichains have clairvoyance gap exactly one.
Large layered incidence structures do not help if their cheap relation
reduces to complete comparability between successive layers.

## Consequence for the search

A viable edge/trail candidate must simultaneously:

1. use the valid \(1/2\) metric (or another fully audited metric);
2. force more than one third of all active clients to become additional
   causal trail starts beyond the posterior trail deficit; and
3. contain genuine branching incompatibility, not just an ordinal
   layering for which first-success sweeps recover an optimal chain
   cover causally.
