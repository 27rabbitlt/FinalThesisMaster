# Three-active random selector metrics: two dense ensembles fail

## Scope

This note audits the proposed probabilistic construction in the first case
where a policy branches after seeing an active selector: condition on exactly
three uniformly distributed active selectors.  Ports are free; an additive
positive port perturbation only decreases the ratios below.

Two natural dense ensembles fail:

1. independent directed \(\{1,2\}\) distances do not even reach \(4/3\)
   against one fixed call order; and
2. a quasirandom tournament has posterior cost \(2\), but a causal
   out-neighbor-first policy costs at most \(11/4+o(1)\), a ratio
   \(11/8+o(1)\).

Thus a union bound over adaptive decision trees cannot rescue either
ensemble.

## 1. Exact identity for a \(\{1,2\}\) selector metric

For a selector order \(v_1,\ldots,v_k\), put
\[
 M(v_1,\ldots,v_k)
 :=|\{i:d(v_i,v_{i+1})=1\}|.
\]
The internal open cost is exactly
\[
               2(k-1)-M(v_1,\ldots,v_k).
\tag{1.1}
\]
Consequently the posterior maximizes the number of cheap consecutive
transitions, while a causal policy constructs such transitions without
knowing the active set.  Any purported universal \(3/2\) theorem for
\(\{1,2\}\) metrics is therefore equivalent to a causal guarantee of at
least half the posterior's expected cheap transitions.

The tournament case has such a \(3/2\) policy by a uniform random priority,
as proved in `open-three-halves-ceiling.md`.  The argument does not extend
verbatim when an unordered pair can have distance \(2\) in both directions.

## 2. Independent directed cheap arcs

For every ordered pair \(u\ne v\), independently set
\[
 d(u,v)=
 \begin{cases}
 1,&\text{with probability }q,\\
 2,&\text{with probability }1-q.
 \end{cases}
\tag{2.1}
\]
Every realization of (2.1) is a directed metric, because all nonzero
distances lie in \([1,2]\).

Condition on a uniformly random active triple.  A fixed causal decision
system selects two ordered transitions.  Before optimizing the system
against the sampled metric, its expected cost is
\[
                         A_{\rm fixed}=4-2q.
\tag{2.2}
\]

The posterior cost is \(2\) if the six possible ordered arcs on the triple
contain a directed Hamilton path of two cheap arcs.  If no such path exists
but at least one cheap arc exists, the cost is \(3\); with no cheap arc it is
\(4\).

A directed graph on three labeled vertices has no directed Hamilton path
only in the following cases:
\[
\begin{array}{c|ccc}
\text{number of cheap arcs}&0&1&2\\ \hline
\text{number of arc sets}&1&6&9.
\end{array}
\tag{2.3}
\]
Every set of at least three directed arcs contains a directed Hamilton
path.  Therefore the exact expected posterior cost is
\[
\begin{aligned}
 P(q)
 &=2+2(1-q)^6+6q(1-q)^5+9q^2(1-q)^4\\
 &=2+(1-q)^4(2+2q+5q^2).
\end{aligned}
\tag{2.4}
\]

The ratio
\[
       R(q):=\frac{4-2q}
       {2+(1-q)^4(2+2q+5q^2)}
\tag{2.5}
\]
is strictly below \(4/3\) on \(0\le q\le1\).  One elementary verification
uses only exact polynomial signs.  The inequality
\(3(4-2q)<4P(q)\) is equivalent, for \(q<2/3\), to
\[
 (1-q)^4(2+2q+5q^2)-1+\frac32q>0.
\tag{2.6}
\]
Put \(t=1-q\) and multiply the left side by two.  The resulting polynomial is
\[
                  H(t)=10t^6-24t^5+18t^4-3t+1.
\tag{2.7}
\]
On \(t\in[1/3,2/3]\), the signs of a Sturm sequence for \(H\) at the two
endpoints are respectively
\[
       (+,-,-,-,+,-,-),\qquad(+,+,-,-,+,+,-).
\]
Both have three sign variations, so \(H\) has no zero in that interval;
\(H(1/3)=10/729>0\).  On \(t\in[2/3,1]\), the Bernstein coefficients of
\(H(t)\) after affine rescaling to \([0,1]\) are
\[
 \frac{199}{729},\ \frac{67}{162},\ \frac{244}{405},\
 \frac{227}{270},\ \frac{17}{15},\ \frac32,\ 2,
\]
all positive.  Hence (2.6) holds for \(q<2/3\).  For \(q\ge2/3\),
\(4-6q\le0\) while the extra term in \(P(q)\) is positive, so the desired
inequality is immediate.

For a large sampled table, the average fixed-order cost and average
posterior triple cost concentrate around (2.2) and (2.4).  Hence with high
probability the single fixed label-order policy already certifies a ratio
below \(4/3+o(1)\).  There is no reason to union-bound over adaptive policies:
optimization can only make the adaptive value smaller.

The failure has a simple cause.  Decreasing \(q\) makes causal transitions
more expensive, but it simultaneously destroys the cheap Hamilton paths
needed by the posterior.

## 3. Quasirandom tournament on an active triple

Now give every unordered pair exactly one unit direction and one length-two
reverse direction.  Every active triple has a directed Hamilton path, so
\[
                         P=2.
\tag{3.1}
\]

A fixed priority would cost \(3+o(1)\) in a quasirandom tournament, giving
the familiar \(3/2\) bound.  Once the first active identity is known, however,
the policy can do better.

Use the following causal policy after the first active selector \(u\):

1. call every uncalled out-neighbor of \(u\), in random priority order;
2. if none is active, call the in-neighbors of \(u\);
3. after the second active selector is found, finish by calling the remaining
   clients in random priority order.

Conditional on an active triple and on its first active member \(u\), the
other two vertices have, in a quasirandom tournament, the asymptotic type
distribution
\[
 \Pr[OO]=\frac14+o(1),\quad
 \Pr[OI]=\frac12+o(1),\quad
 \Pr[II]=\frac14+o(1),
\tag{3.2}
\]
where \(O\) and \(I\) denote out- and in-neighbors of \(u\).

The first transition is reverse only in type \(II\), so its expected excess
over one is \(1/4+o(1)\).  For the second transition:

* in types \(OO\) and \(II\), random priority makes either orientation
  equally likely;
* in type \(OI\), the out-neighbor is second, and quasirandomness makes its
  orientation to the in-neighbor asymptotically fair.

Thus the second transition has expected excess \(1/2+o(1)\).  The policy's
expected internal cost is at most
\[
                    2+\frac14+\frac12+o(1)
                    =\frac{11}{4}+o(1).
\tag{3.3}
\]
Combining (3.1) and (3.3),
\[
       \frac{\operatorname {OPT}_{\rm adapt}}
            {\operatorname {OPT}_{\rm post}}
       \le \frac{11}{8}+o(1).
\tag{3.4}
\]

The quasirandom properties used in (3.2)--(3.3) hold simultaneously for all
but \(o(n)\) vertices in a random tournament, and averaging absorbs the
exceptional first vertices.  The random tournament can then be fixed by the
probabilistic method.

Thus conditioning on three active selectors makes the tournament strictly
easier than the rare-pair regime: the first active identity supplies a large
bank of known cheap out-neighbors.

## 4. Counting obstruction for sparse unit arcs

Let \(G_1\) be the digraph of unit-distance ordered pairs in an arbitrary
\(\{1,2,\ldots\}\) metric, and let \(A\) be a uniformly random \(k\)-set.
If \(G_1[A]\) contains a Hamilton path with probability \(1-o(1)\), then
\[
\begin{aligned}
 \mathbb E|E(G_1[A])|
   &\ge (1-o(1))(k-1).
\end{aligned}
\tag{4.1}
\]
Writing \(q=|E(G_1)|/(n(n-1))\), the left side is
\(qk(k-1)\).  Hence
\[
                         qk\ge1-o(1).
\tag{4.2}
\]

So the unit graph cannot simultaneously have a typical unit Hamilton path
and expected active out-degree \(o(1)\).  If \(qk\) grows, a causal policy
can probe a large current out-neighbor bank; if \(qk<1\), the posterior
usually lacks enough unit arcs even before connectivity is considered.

This does not prove a universal \(3/2\) ceiling for all three-scale metrics,
but it identifies what such a construction must exploit: global contention
among many locally available unit successors, not merely sparse
out-neighborhoods or a union bound over priority systems.

## 5. Consequence

Neither dense independent \(\{1,2\}\) tables nor quasirandom tournaments on
three-active realizations can exceed \(4/3\), let alone \(3/2\).  A viable
probabilistic construction must have all three properties:

1. typical active sets admit nearly spanning cheap path systems;
2. the first active identity does not expose a large bank of usable cheap
   successors; and
3. failures move the policy by more than a factor-two transition without
   shortest two-hop paths collapsing that third scale.

The first two requirements are already in direct tension by (4.2).
